"""Z2 S3a/S3b offline: candidate pairs, threads, change detection with scripted
fake models (agree, dispute + tie-break, duplicates) and re-chaining."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.z2 import questions as Q


def entry(i, turn, value, speaker="user", members=None):
    return dict(id=f"H:{i}", subject=f"s{i}", value=value, speaker=speaker, valid_from_turn=turn,
                members=members or [f"raw{i}"], usable_for_questions=True, supersedes_raw=[],
                status="verified")


def test_pairs_are_same_speaker_older_first_above_floor():
    es = [entry(0, 3, "a"), entry(1, 9, "b"), entry(2, 12, "c", speaker="assistant"), entry(3, 20, "d")]
    v = np.array([[1, 0], [0.8, 0.6], [1, 0], [0.1, 0.995]], dtype="float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pairs = Q.candidate_pairs(es, v, floor=0.7, k=5)
    assert (0, 1) in {(i, j) for i, j, _ in pairs}                       # 0.8 similar, older first
    assert not any(j == 2 or i == 2 for i, j, _ in pairs)                # assistant has no same-speaker partner
    assert all(es[i]["valid_from_turn"] < es[j]["valid_from_turn"] for i, j, _ in pairs)


def test_threads_are_connected_groups_ordered_by_turn():
    es = [entry(0, 50, "a"), entry(1, 3, "b"), entry(2, 9, "c")]
    v = np.array([[1, 0], [0.99, 0.14], [0, 1]], dtype="float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    assert Q.threads(es, v, floor=0.9) == [[1, 0]]


class Scripted:
    def __init__(self, verdict_for):
        self.verdict_for, self.calls = verdict_for, 0

    def __call__(self, messages, **kw):
        self.calls += 1
        body = json.loads(messages[1]["content"])
        out = []
        for e in body["earlier"]:
            v = self.verdict_for(e["text"], body["newer"])
            out.append(dict(id=e["id"], same_information=v == "same", both_can_be_true=v != "replaces"))
        return NS(text=json.dumps({"answers": out}), usage=None)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    es = [entry(0, 3, "The cat is called Miso."), entry(1, 30, "The cat is now called Tofu."),
          entry(2, 5, "The user lives in Lyon."), entry(3, 40, "The user moved to Porto."),
          entry(4, 6, "The user likes tea."), entry(5, 50, "The user likes tea a lot.")]
    monkeypatch.setattr(Q, "usable", lambda h, root: es)
    d = tmp_path / "H"
    d.mkdir()
    d.joinpath("threads.json").write_text(json.dumps(dict(pairs=[
        dict(older="H:0", newer="H:1"), dict(older="H:2", newer="H:3"), dict(older="H:4", newer="H:5")])))
    return tmp_path, es


def test_changes_need_two_families_or_the_tie_breaker(setup):
    out, es = setup
    asker = Scripted(lambda old, new: "same" if "tea" in old else "replaces")
    checker = Scripted(lambda old, new: "replaces" if "Miso" in old else "both_true")
    tie = Scripted(lambda old, new: "both_true")
    models = {"deepseek": (asker, {"p": "ds"}), "mimo": (checker, {"p": "mm"}), "muse": (tie, {"p": "mu"})}
    result = Q.detect_changes("H", None, out, models=models)
    assert [(c["older"], c["newer"]) for c in result["confirmed"]] == [("H:0", "H:1")]
    assert [(d["older"], d["tie_break"]) for d in result["disputed"]] == [("H:2", "both_true")]
    assert result["duplicates"] == [dict(older="H:4", newer="H:5")]
    chained = {e["id"]: e for e in Q.with_changes(es, result)}
    assert chained["H:0"]["valid_until_turn"] == 30 and chained["H:0"]["superseded_by"] == "H:1"
    assert chained["H:2"]["valid_until_turn"] is None


def test_tie_breaker_can_confirm_and_reruns_reuse_saved_calls(setup):
    out, es = setup
    asker = Scripted(lambda old, new: "replaces")
    checker = Scripted(lambda old, new: "both_true")
    tie = Scripted(lambda old, new: "replaces" if "Lyon" in old else "both_true")
    models = {"deepseek": (asker, {"p": "ds"}), "mimo": (checker, {"p": "mm"}), "muse": (tie, {"p": "mu"})}
    result = Q.detect_changes("H", None, out, models=models)
    assert {(c["older"], c["newer"]) for c in result["confirmed"]} == {("H:2", "H:3")}
    counts = (asker.calls, checker.calls, tie.calls)
    Q.detect_changes("H", None, out, models=models)
    assert (asker.calls, checker.calls, tie.calls) == counts


def test_select_families_by_quota_spread_and_alternating_writers():
    pools = {"fact_recall": [dict(entries=[f"H:{i}"], first=10 * i) for i in range(10)],
             "update": [dict(entries=["H:a", "H:b"], first=5)], "negative": [dict(cluster=["H:0"])]}
    fams = Q.select_families(pools, {"fact_recall": 3, "update": 2, "negative": 4, "messy_variant": 1}, "H")
    facts = [f for f in fams if f["type"] == "fact_recall"]
    assert [f["first"] for f in facts] == [0, 30, 60]                 # evenly spread
    assert [f["type"] for f in fams].count("update") == 1             # never padded past supply
    assert not any(f["type"] in ("negative", "messy_variant") for f in fams)
    assert [f["writer"] for f in fams] == ["deepseek", "mimo", "deepseek", "mimo"]


def test_phrase_kept_only_when_other_family_passes_it(tmp_path):
    es = {"H:1": entry(1, 5, "The cat is called Miso.")}
    originals = [dict(turn_number=i, prompt=f"q {i}", response="r") for i in range(1, 60)]
    fams = [dict(id="H:fam000", type="fact_recall", entries=["H:1"], first=50, writer="deepseek"),
            dict(id="H:fam001", type="fact_recall", entries=["H:1"], first=51, writer="mimo")]

    def writer(messages, **kw):
        return NS(text=json.dumps({"question": "what's my cat called again"}), usage=None)

    seen = []
    def verifier(messages, **kw):
        seen.append(json.loads(messages[1]["content"])["QUESTION"])
        leaks = len(seen) == 2                                         # second verdict rejects
        return NS(text=json.dumps({"answerable": True, "unambiguous": True, "leaks_answer": leaks,
                                   "speaker_right": True}), usage=None)

    models = {"deepseek": (lambda m, **k: writer(m) if "TASK" in m[1]["content"] else verifier(m), {"p": "ds"}),
              "mimo": (lambda m, **k: writer(m) if "TASK" in m[1]["content"] else verifier(m), {"p": "mm"})}
    kept, rejected = Q.phrase_and_verify("H", fams, es, originals, tmp_path, models=models)
    assert [k["id"] for k in kept] == ["H:fam000"] and kept[0]["verifier"] == "mimo"
    assert [r["reason"] for r in rejected] == ["verifier"]
    again, _ = Q.phrase_and_verify("H", fams, es, originals, tmp_path, models=models)   # resumes from saved calls
    assert len(seen) == 2 and [k["id"] for k in again] == ["H:fam000"]


def test_duplicates_collapse_to_the_earliest_same_speaker_entry():
    es = [entry(0, 10, "The cat is Miso."), entry(1, 40, "The cat is Miso!"),
          entry(2, 50, "The cat is Miso.", speaker="assistant"), entry(3, 60, "Tea."), entry(4, 70, "Tea, really.")]
    v = np.array([[1, 0], [0.999, 0.04], [1, 0], [0, 1], [0.3, 0.95]], dtype="float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    canon = Q.duplicate_map(es, v, {"duplicates": [dict(older="H:3", newer="H:4")]})
    assert canon["H:1"] == "H:0"                       # near-identical, same speaker
    assert canon["H:2"] == "H:2"                       # other speaker never merged
    assert canon["H:4"] == "H:3"                       # change detection's "same"


def test_ask_points_follow_cutoffs_repeats_and_update_rule():
    points = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    fam = dict(id="H:f1", type="fact_recall", first=50)
    asks = Q.ask_points(fam, points)
    assert asks[:2] == [50, 200] and all(b - a >= 0 for a, b in zip(asks, asks[1:]))
    upd = dict(id="H:u1", type="update", first=50, change_turn=120)
    assert Q.ask_points(upd, points) == [50, 175]          # before the change, and after it left recent history


def test_ground_truth_is_the_state_at_the_checkpoint_and_never_the_future():
    old = dict(entry(0, 10, "The cat is Miso."), valid_until_turn=120, superseded_by="H:1",
               citations=[dict(turn=10)])
    new = dict(entry(1, 120, "The cat is Tofu."), valid_until_turn=None, citations=[dict(turn=120)])
    es = {"H:0": old, "H:1": new}
    fam = dict(id="H:u1", type="update", entries=["H:0", "H:1"], first=50, change_turn=120)
    before = Q.ground_truth(fam, es, 50)
    after = Q.ground_truth(fam, es, 175)
    assert before["required"] == ["The cat is Miso."] and before["superseded"] == []
    assert after["required"] == ["The cat is Tofu."] and after["superseded"] == ["The cat is Miso."]
    t = dict(id="H:t1", type="temporal", entries=["H:1", "H:0"], first=150)
    assert Q.ground_truth(t, es, 150)["order"] == ["H:0", "H:1"]


def test_messy_variant_keeps_base_answers_and_rejects_meaning_drift(tmp_path):
    kept = [dict(id=f"H:fam00{i}", type="fact_recall", entries=["H:1"], first=50, writer="deepseek",
                 question="what's my cat called again?") for i in range(3)]
    rewrites = iter(["whats my cat called again lol", "my cats name?", "what is my DOG called"])

    def model(messages, **kw):
        body = json.loads(messages[1]["content"])
        if "STYLE" in body:
            return NS(text=json.dumps({"question": next(rewrites)}), usage=None)
        return NS(text=json.dumps({"same_meaning": "DOG" not in body["B"]}), usage=None)

    models = {"deepseek": (model, {"p": "ds"}), "mimo": (model, {"p": "mm"})}
    variants, rejected = Q.messy_variants("H", kept, 3, tmp_path, models=models)
    assert len(variants) == 2 and len(rejected) == 1
    assert all(v["type"] == "messy_variant" and v["base_type"] == "fact_recall" and v["writer"] == "mimo"
               for v in variants)
    upd = dict(id="H:u1:messy", type="messy_variant", base_type="update", first=50, change_turn=120)
    assert Q.ask_points(upd, [25, 50, 75, 100, 125, 150, 175]) == [50, 175]


def test_control_prompts_one_per_checkpoint_kinds_cycled(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "checkpoints", lambda h: [25, 50, 75, 100, 125])
    originals = [dict(turn_number=i, prompt=f"q {i}", response=f"a {i}") for i in range(1, 130)]
    seen = []
    def model(messages, **kw):
        body = json.loads(messages[1]["content"])
        if "KIND" in body:
            seen.append(("RECENT" in body, body["KIND"]))
            return NS(text=json.dumps({"prompt": "whats the capital of peru"}), usage=None)
        return NS(text=json.dumps({"needs_no_memory": True}), usage=None)
    models = {"deepseek": (model, {"p": "ds"}), "mimo": (model, {"p": "mm"})}
    kept, rejected = Q.control_prompts("H", originals, tmp_path, models=models)
    assert [k["kind"] for k in kept] == ["general_knowledge", "self_contained_task", "small_talk",
                                         "recent_follow_up", "general_knowledge"]
    assert [has_recent for has_recent, _ in seen] == [False, False, False, True, False]
    assert all(k["type"] == "no_memory_control" for k in kept) and not rejected


def test_finalize_needs_the_review_then_seals_and_writes_z2_asks(tmp_path):
    import pytest
    d = tmp_path / "B"
    d.mkdir()
    fams = [dict(id=f"B:fam{i:03d}", type="fact_recall", question=f"q{i}") for i in range(30)]
    ctl = [dict(id="B:ctl000", type="no_memory_control", kind="small_talk", checkpoint_turn=25, prompt="hey")]
    (d / "families.json").write_text(json.dumps(dict(kept=fams, rejected=[], controls=ctl)))
    occ = [dict(family=f["id"], type="fact_recall", checkpoint_turn=25, repeat_index=0, question=f["question"],
                reference=dict(required=["x"], citations=[dict(turn=3)]), expected_abstention=False) for f in fams]
    (d / "occurrences.json").write_text(json.dumps(occ))
    Q.write_review("B", tmp_path, per_type=2)
    assert "B:fam" in (d / "review-packet.md").read_text()
    with pytest.raises(RuntimeError, match="USER-REQUIRED"):
        Q.finalize("B", tmp_path)
    decisions = json.loads((d / "review-decisions.json").read_text())
    (d / "review-decisions.json").write_text(json.dumps(dict(decisions, reviewed=True, drop=["B:fam000"])))
    Q.finalize("B", tmp_path)
    asks = json.loads((d / "z2-asks.json").read_text())
    assert not any(a["id"].startswith("B:fam000@") for a in asks)
    assert [a for a in asks if a["kind"] == "control"][0]["question"] == "hey"
