"""Z2 S2 merge, offline: cross-verification, matching, tie-break, review queue
and resume — scripted fake checkers, no cloud call."""

import json
from types import SimpleNamespace as NS

import pytest

from experiments.z2 import ledger, ledger_merge

PAIRS = [dict(turn_number=i, original_pair_sha256=f"sha{i}",
              prompt="my cat is called Miso" if i == 2 else f"user line {i}",
              response="you could adopt a second cat" if i == 2 else f"assistant reply {i}")
         for i in range(1, 4)]


def raw_entry(writer, n, subject, value, quote, role="user", speaker="user", modality="asserted"):
    citation = ledger.resolve(PAIRS, 2, role, quote)
    assert citation is not None
    return dict(id=f"{writer}:X:w0.0.{n}", history="X", writer=writer, kind="fact", subject=subject,
                value=value, speaker=speaker, modality=modality, citations=[citation],
                valid_from_turn=2, supersedes=None, change=None)


def write_raw(directory, writer, entries):
    record = dict(index=0, new=[1, 2, 3], context=[], entries=entries)
    (directory / f"ledger-raw-{writer}.jsonl").write_text(json.dumps(record) + "\n")


def seen(e, **override):
    """The observation a checker makes of *e* when it agrees with every field."""
    return dict(dict(said_by=e["speaker"], stated_as=e["modality"], wrong_detail=None), **override)


class Checker:
    """Answers by real entry id from a script (found through the entry's value,
    since prompts carry short aliases); counts calls."""

    def __init__(self, key, answers, entries=None):
        self.key, self.answers, self.calls = key, answers, 0
        self.by_value = {e["value"]: e["id"] for e in entries or []}

    def __call__(self, messages, **kw):
        self.calls += 1
        sections = messages[1]["content"].split("=====")
        lines = lambda text: [json.loads(l) for l in text.splitlines() if l.startswith("{")]
        yours = {self.by_value[r["value"]]: r["id"] for r in (lines(sections[2]) if len(sections) > 2 else [])}
        out = []
        for row in lines(sections[1]):
            answer = dict(self.answers[self.by_value[row["value"]]])
            if answer.get("match"):
                answer["match"] = yours.get(answer["match"])
            out.append(dict(id=row["id"], **answer))
        return NS(text=json.dumps({self.key: out}), usage=None)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "load_history", lambda h: PAIRS)
    ds = [raw_entry("deepseek", 0, "user's cat", "The user's cat is called Miso.", "my cat is called Miso"),
          raw_entry("deepseek", 1, "second cat", "The user decided to adopt a second cat.",
                    "adopt a second cat", role="assistant", speaker="user")]
    mm = [raw_entry("mimo", 0, "cat name", "The user has a cat named Miso.", "cat is called Miso"),
          raw_entry("mimo", 1, "second cat", "The assistant suggested adopting a second cat.",
                    "you could adopt a second cat", role="assistant", speaker="assistant",
                    modality="suggested")]
    write_raw(tmp_path, "deepseek", ds)
    write_raw(tmp_path, "mimo", mm)
    # mimo checks deepseek's entries; deepseek checks mimo's.
    everyone = ds + mm
    mimo = Checker("checks", {ds[0]["id"]: seen(ds[0], match=mm[0]["id"]),
                              ds[1]["id"]: seen(ds[1], said_by="assistant", stated_as="suggested", match=None)},
                   everyone)
    deepseek = Checker("checks", {mm[0]["id"]: seen(mm[0], match=ds[0]["id"]),
                                  mm[1]["id"]: seen(mm[1], match=None)}, everyone)
    tie = Checker("decisions", {ds[1]["id"]: seen(ds[1], said_by="assistant", stated_as="suggested")},
                  everyone)
    models = {"deepseek": (deepseek, {"p": "ds"}), "mimo": (mimo, {"p": "mm"}), "tie": (tie, {"p": "t"})}
    return tmp_path, models, ds, mm, (deepseek, mimo, tie)


def test_merge_verifies_matches_and_keeps_rejected_as_unconfirmed(setup):
    directory, models, ds, mm, _ = setup
    result, queue = ledger_merge.merge("X", directory, directory / "merged", writers=("deepseek", "mimo"),
                                       tie_breaker="tie", models=models)
    entries = {tuple(e["members"]): e for e in result["entries"]}
    cat = entries[tuple(sorted([ds[0]["id"], mm[0]["id"]]))]
    assert cat["status"] == "verified" and cat["writers"] == ["deepseek", "mimo"] and len(cat["citations"]) == 2
    suggestion = entries[(mm[1]["id"],)]
    assert suggestion["verified"] and suggestion["modality"] == "suggested"   # confirmed by the other family
    wrong = entries[(ds[1]["id"],)]                                            # wrong speaker: kept, flagged
    assert wrong["status"] == "unconfirmed" and wrong["unconfirmed_reason"] == "speaker"
    assert wrong["usable_for_questions"] is False and cat["usable_for_questions"] is True
    assert [q["status"] for q in queue] == ["unconfirmed"]
    assert result["stats"] == dict(entries=4, matched=1, confirmed_unmatched=1, disputed=1, unanswered=0,
                                   retried_calls=0, verified=2, unconfirmed=1, unconfirmed_speaker=1)


def test_tie_break_keep_is_unverified_but_usable(setup):
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    tie.answers[ds[1]["id"]] = seen(ds[1])
    result, queue = ledger_merge.merge("X", directory, directory / "merged", writers=("deepseek", "mimo"),
                                       tie_breaker="tie", models=models)
    kept = next(e for e in result["entries"] if e["members"] == [ds[1]["id"]])
    assert kept["status"] == "tie_break" and not kept["verified"] and kept["usable_for_questions"]


def test_label_only_disagreement_is_relabeled_not_dropped(setup):
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    # Checker and tie-breaker both see the user say it, every detail right, but as a suggestion.
    mimo.answers[ds[1]["id"]] = seen(ds[1], stated_as="suggested", match=None)
    tie.answers[ds[1]["id"]] = seen(ds[1], stated_as="suggested")
    result, _ = ledger_merge.merge("X", directory, directory / "merged6", writers=("deepseek", "mimo"),
                                   tie_breaker="tie", models=models)
    e = next(e for e in result["entries"] if e["members"] == [ds[1]["id"]])
    assert e["status"] == "relabeled" and e["modality"] == "suggested" and e["written_modality"] == "asserted"
    assert e["usable_for_questions"]


def test_label_disagreement_without_agreement_stays_unconfirmed(setup):
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    mimo.answers[ds[1]["id"]] = seen(ds[1], stated_as="suggested", match=None)
    tie.answers[ds[1]["id"]] = seen(ds[1], stated_as="hypothetical")
    result, _ = ledger_merge.merge("X", directory, directory / "merged7", writers=("deepseek", "mimo"),
                                   tie_breaker="tie", models=models)
    e = next(e for e in result["entries"] if e["members"] == [ds[1]["id"]])
    assert e["status"] == "unconfirmed" and e["unconfirmed_reason"] == "modality" and e["modality"] == "asserted"


def test_merge_resumes_without_new_calls(setup):
    directory, models, ds, mm, checkers = setup
    first, _ = ledger_merge.merge("X", directory, directory / "merged", writers=("deepseek", "mimo"),
                                  tie_breaker="tie", models=models)
    calls = [c.calls for c in checkers]
    again, _ = ledger_merge.merge("X", directory, directory / "merged", writers=("deepseek", "mimo"),
                                  tie_breaker="tie", models=models)
    assert [c.calls for c in checkers] == calls and again["entries"] == first["entries"]


def test_unanswered_check_is_kept_as_unconfirmed(setup):
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    models["mimo"] = (lambda messages, **kw: NS(text="no json", usage=None), {"p": "mm"})
    result, queue = ledger_merge.merge("X", directory, directory / "merged2", writers=("deepseek", "mimo"),
                                       tie_breaker="tie", models=models)
    assert result["stats"]["unanswered"] == 2
    assert {q["entry"]["id"] for q in queue if q["kind"] == "unanswered_check"} == {ds[0]["id"], ds[1]["id"]}
    flagged = [e for e in result["entries"] if e["members"] in ([ds[0]["id"]], [ds[1]["id"]])]
    assert len(flagged) == 2 and all(e["unconfirmed_reason"] == "check_unanswered" for e in flagged)


def test_nothing_is_deleted(setup):
    directory, models, ds, mm, _ = setup
    result, _ = ledger_merge.merge("X", directory, directory / "merged8", writers=("deepseek", "mimo"),
                                   tie_breaker="tie", models=models)
    assert sorted(m for e in result["entries"] for m in e["members"]) == sorted(e["id"] for e in ds + mm)


def test_match_to_own_writer_is_ignored(setup):
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    mimo.answers[ds[0]["id"]] = seen(ds[0], match=ds[1]["id"])               # not a mimo entry
    result, _ = ledger_merge.merge("X", directory, directory / "merged3", writers=("deepseek", "mimo"),
                                   tie_breaker="tie", models=models)
    assert not any(set(e["members"]) == {ds[0]["id"], ds[1]["id"]} for e in result["entries"])


@pytest.mark.parametrize("override, reason", [
    (dict(said_by="assistant"), "speaker"),
    (dict(stated_as="suggested"), "modality"),
    (dict(wrong_detail="Miso"), "wrong_detail"),
    (dict(said_by="nobody"), "not_stated"),
    (dict(stated_as="not_stated"), "not_stated"),
])
def test_observation_compared_in_code(override, reason):
    e = dict(speaker="user", modality="asserted")
    assert ledger_merge.observed(e, seen(e)) == (True, None)
    assert ledger_merge.observed(e, seen(e, **override)) == (False, reason)


def test_malformed_observation_is_unanswered():
    e = dict(speaker="user", modality="asserted")
    assert ledger_merge.observed(e, dict(supported=True)) == (None, "malformed")


def test_checker_never_sees_speaker_or_modality(setup):
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    seen_prompts = []
    real = models["mimo"][0]
    def spy(messages, **kw):
        seen_prompts.append(messages[1]["content"].split("=====")[1])
        return real(messages, **kw)
    models["mimo"] = (spy, {"p": "mm"})
    ledger_merge.merge("X", directory, directory / "merged4", writers=("deepseek", "mimo"),
                       tie_breaker="tie", models=models)
    assert seen_prompts and all('"speaker"' not in p and '"modality"' not in p for p in seen_prompts)


def test_reply_for_the_wrong_list_is_retried_once(setup):
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    real, replies = models["deepseek"][0], []
    def confused_then_right(messages, **kw):
        replies.append(1)
        if len(replies) == 1:                     # answers with its own list's aliases
            return NS(text=json.dumps({"checks": [dict(id="Y01", said_by="user", stated_as="asserted",
                                                       wrong_detail=None, match=None)]}), usage=None)
        return real(messages, **kw)
    models["deepseek"] = (confused_then_right, {"p": "ds"})
    result, queue = ledger_merge.merge("X", directory, directory / "merged5", writers=("deepseek", "mimo"),
                                       tie_breaker="tie", models=models)
    assert len(replies) == 2 and result["stats"]["retried_calls"] == 1
    assert result["stats"]["unanswered"] == 0


def test_cut_tie_break_reply_is_split_in_halves(setup):
    from experiments.lme.cloud_provider import CloudCompletionError
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    # Make both deepseek entries disputed, then let the tie-breaker only answer one at a time.
    mimo.answers[ds[0]["id"]] = seen(ds[0], said_by="nobody", match=None)
    sizes = []
    real = models["tie"][0]
    def capped(messages, **kw):
        n = sum(1 for l in messages[1]["content"].split("=====")[1].splitlines() if l.startswith("{"))
        sizes.append(n)
        if n > 1:
            raise CloudCompletionError("cloud answer completion was not successful: incomplete")
        return real(messages, **kw)
    tie.answers[ds[0]["id"]] = seen(ds[0])
    models["tie"] = (capped, {"p": "t"})
    result, _ = ledger_merge.merge("X", directory, directory / "merged9", writers=("deepseek", "mimo"),
                                   tie_breaker="tie", models=models)
    assert sizes == [2, 1, 1]
    assert result["stats"]["unanswered"] == 0


def _e(i, start, members, usable=True, supersedes=()):
    return dict(id=f"X:{i}", valid_from_turn=start, members=list(members),
                usable_for_questions=usable, supersedes_raw=list(supersedes))


def test_chain_closes_superseded_entries_at_the_change():
    old = _e("old", 3, ["r1"])
    new = _e("new", 30, ["r2"], supersedes=["r1"])
    chained = {e["id"]: e for e in ledger_merge.chain([old, new])}
    assert chained["X:old"]["valid_until_turn"] == 30 and chained["X:old"]["superseded_by"] == "X:new"
    assert chained["X:new"]["valid_until_turn"] is None
    assert [e["id"] for e in ledger_merge.state_at(list(chained.values()), 29)] == ["X:old"]
    assert [e["id"] for e in ledger_merge.state_at(list(chained.values()), 30)] == ["X:new"]
    assert old.get("valid_until_turn") is None                              # input unchanged


def test_unconfirmed_or_earlier_entries_never_close_another():
    old = _e("old", 10, ["r1"])
    unconfirmed = _e("u", 20, ["r2"], usable=False, supersedes=["r1"])
    earlier = _e("e", 5, ["r3"], supersedes=["r1"])
    chained = {e["id"]: e for e in ledger_merge.chain([old, unconfirmed, earlier])}
    assert chained["X:old"]["valid_until_turn"] is None


def test_first_change_wins_when_two_replace_the_same_entry():
    old = _e("old", 3, ["r1"])
    a, b = _e("a", 40, ["r2"], supersedes=["r1"]), _e("b", 20, ["r3"], supersedes=["r1"])
    chained = {e["id"]: e for e in ledger_merge.chain([old, a, b])}
    assert chained["X:old"]["valid_until_turn"] == 20 and chained["X:old"]["superseded_by"] == "X:b"


def test_without_a_tie_breaker_a_dispute_is_unconfirmed_and_reviewed(setup):
    """FINAL (Z3 D13): Muse writes no ground truth — a dispute is never settled by a third model."""
    directory, models, ds, mm, (deepseek, mimo, tie) = setup
    tie.answers[ds[1]["id"]] = seen(ds[1])               # would keep it, if it were asked
    result, queue = ledger_merge.merge("X", directory, directory / "merged-none", writers=("deepseek", "mimo"),
                                       tie_breaker=None, models={k: v for k, v in models.items() if k != "tie"})
    disputed = next(e for e in result["entries"] if e["members"] == [ds[1]["id"]])
    assert disputed["status"] == "unconfirmed" and not disputed["usable_for_questions"]
    assert any(q["kind"] == "disputed" and q["entry"]["id"] == ds[1]["id"] for q in queue)
    assert tie.calls == 0


def test_one_writer_makes_a_second_writer_wait_then_resume(tmp_path):
    """A stage started by the main run and a lane at once (Z3 2026-11-04): the second waits for the
    first to let go, then goes on — never two writers on one folder. (flock blocks between two
    opens in one process too, so a thread stands in for the second process.)"""
    import threading
    held = ledger_merge.one_writer(tmp_path / "out")
    said = []
    second = threading.Thread(target=lambda: ledger_merge.one_writer(tmp_path / "out", poll=0.05,
                                                                     say=lambda m, **_: said.append(m)))
    second.start()
    second.join(0.5)
    assert second.is_alive()                          # waiting while the first holds it
    held.close()                                      # the first exits
    second.join(5)
    assert not second.is_alive() and "waiting" in said[0] and "resuming" in said[-1]
