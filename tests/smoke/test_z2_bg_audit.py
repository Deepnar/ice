"""Z2 S8 background-content audit (spec D9): sampling, LATER_CHANGES from the
ledger, judging that survives errors, agreement, and fact precision/recall."""

import json
from types import SimpleNamespace as NS

from experiments.z2 import bg_audit as A


def item(kind, rid, turns=(1,), text="The user's cat is Miso."):
    return dict(id=f"{kind}:{rid}", kind=kind, text=text,
                sources=[dict(turn=t, text=f"turn {t} text") for t in turns])


class Judge:
    """A fake judge: replies by script; an Exception entry raises."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, messages, **_):
        self.prompts.append(json.loads(messages[1]["content"]))
        reply = self.replies.pop(0) if self.replies else {"faithful": {"verdict": "yes", "reason": "ok"}}
        if isinstance(reply, Exception):
            raise reply
        return NS(text=json.dumps(reply), usage={})


def models(judge):
    return {"muse": (judge, {"model": "fake-muse"}), "deepseek": (judge, {"model": "fake-ds"})}


def test_sampling_is_deterministic_capped_per_kind_and_rejudge_is_about_a_fifth():
    items = [item("codex_fact", i) for i in range(300)] + [item("cluster", i) for i in range(7)]
    s = A.sample(items, "B")
    assert s == A.sample(list(reversed(items)), "B")
    assert sum(i["kind"] == "codex_fact" for i in s) == 100 and sum(i["kind"] == "cluster" for i in s) == 7
    assert 10 <= len(A.rejudged(s, "B")) <= 35


def test_later_changes_only_after_the_items_turns_and_within_the_store():
    ledger = [dict(id="a", value="cat is Miso", citations=[dict(turn=1)], superseded_by="b", valid_until_turn=9),
              dict(id="b", value="cat is Tofu", citations=[dict(turn=9)], superseded_by=None, valid_until_turn=None)]
    assert A.later_changes(item("turn_summary", 1, (1,)), ledger, 20) == [
        dict(earlier="cat is Miso", later="cat is Tofu", at_turn=9)]
    assert A.later_changes(item("turn_summary", 1, (1,)), ledger, 8) == []      # change not in this store yet
    assert A.later_changes(item("turn_summary", 1, (9,)), ledger, 20) == []     # not cited there


def test_current_is_judged_only_when_the_ledger_has_a_change(tmp_path):
    judge = Judge([{"faithful": {"verdict": "yes"}, "attributed": {"verdict": "yes"},
                    "complete": {"verdict": "no", "reason": "drops the name"}}])
    results, pending = A.judge_all([item("turn_summary", 1)], "B", [], 10, tmp_path, models=models(judge))
    assert judge.prompts[0]["CRITERIA"] == ["faithful", "attributed", "complete"]
    r = results["turn_summary:1"]
    assert r["status"] == "judged" and r["verdicts"]["complete"] == "no" and r["current"] == "n/a" and not pending


def test_an_erroring_item_is_retried_then_recorded_unjudged_and_the_rest_continue(tmp_path):
    items = [item("codex_fact", 1, text="The cat is Miso."), item("codex_fact", 2, text="The dog is Rex.")]
    for attempt in range(A.MAX_ITEM_ERRORS):
        judge = Judge([TimeoutError("gateway"), {"faithful": {"verdict": "yes"}}])
        results, pending = A.judge_all(items, "B", [], 10, tmp_path, models=models(judge))
        assert results["codex_fact:2"]["status"] == "judged"
    assert pending == 1
    results, pending = A.judge_all(items, "B", [], 10, tmp_path, models=models(Judge([])))
    assert results["codex_fact:1"]["status"] == "unjudged" and "gateway" in results["codex_fact:1"]["reason"]
    assert pending == 0


def test_oversized_and_sourceless_items_are_recorded_not_cut_or_dropped(tmp_path):
    big = item("session_summary", 1)
    big["sources"][0]["text"] = "x" * (A.SOURCE_CHARS + 1)
    none = dict(item("procedural", 1), sources=[])
    results, _ = A.judge_all([big, none], "B", [], 10, tmp_path, models=models(Judge([])))
    assert results["session_summary:1"].get("partwise") is True       # judged in parts, never cut
    assert results["procedural:1"]["status"] == "no_sources"


def test_fact_matching_separates_verified_unconfirmed_and_none(tmp_path):
    ledger = [dict(id="v", value="cat is Miso", speaker="user", citations=[dict(turn=1)], usable_for_questions=True),
              dict(id="u", value="dog is Rex", speaker="user", citations=[dict(turn=2)], usable_for_questions=False)]
    facts = [item("codex_fact", 1, (1,)), item("codex_fact", 2, (2,)), item("codex_fact", 3, (5,))]
    judge = Judge([{"same": ["L1"]}, {"same": ["L1"]}])
    out, _ = A.match_facts(facts, ledger, "B", tmp_path, models=models(judge))
    assert [out[f"codex_fact:{i}"]["match"] for i in (1, 2, 3)] == ["verified", "unconfirmed_source", "none"]


def test_report_states_agreement_and_disagreements():
    sampled = [item("codex_fact", 1)]
    first = {"codex_fact:1": dict(status="judged", verdicts={"faithful": "yes"}, reasons={"faithful": "a"})}
    second = {"codex_fact:1": dict(status="judged", verdicts={"faithful": "no"}, reasons={"faithful": "b"})}
    r = A.report("B", {"codex_fact": 1}, {}, sampled, first, second, {}, {}, {})
    assert r["agreement"] == dict(compared=1, same=0, rate=0.0) and r["disagreements"][0]["deepseek"] == "no"
    assert r["kinds"]["codex_fact"]["criteria"]["faithful"]["yes"] == 1


def test_oversized_items_are_judged_in_parts_and_combined():
    item = dict(id="session_summary:1", kind="session_summary",
                text=json.dumps({"topics_covered": ["cats", "dogs"], "decisions_made": "Adopt Tofu. Skip the fish."}),
                sources=[dict(turn=t, text="x" * 150_000) for t in (1, 2)])
    assert A.statements(item) == ["cats", "dogs", "Adopt Tofu.", "Skip the fish."]
    assert len(A.source_parts(item["sources"])) == 2
    verdicts, _ = A.combine_parts(4, [{"1": "supported", "2": "absent", "3": "supported", "4": "absent"},
                                      {"1": "absent", "2": "supported", "3": "absent", "4": "supported"}])
    assert verdicts == dict(faithful="yes", attributed="yes")
    verdicts, _ = A.combine_parts(2, [{"1": "supported", "2": "absent"}, {"1": "contradicted", "2": "absent"}])
    assert verdicts["faithful"] == "no"
    verdicts, _ = A.combine_parts(1, [{"1": "misattributed"}])
    assert verdicts == dict(faithful="yes", attributed="no")


def test_judge_all_routes_oversized_items_to_partwise(tmp_path):
    item = dict(id="session_summary:9", kind="session_summary", text="The cat is Tofu. The dog is Rex.",
                sources=[dict(turn=1, text="y" * (A.SOURCE_CHARS + 1))])
    judge = Judge([{"1": "supported", "2": "supported"}])
    results, _ = A.judge_all([item], "C", [], 10, tmp_path, models=models(judge))
    r = results["session_summary:9"]
    assert r["partwise"] and r["verdicts"] == dict(faithful="yes", attributed="yes") and "complete" in r["not_judged"]
