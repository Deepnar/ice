"""One-history ordering, scorer isolation and source-event resume identity."""

import copy
from datetime import datetime, timedelta, timezone

import pytest

from experiments.v3_repair import chronological_sources as sources

START = datetime(2025, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def history():
    rows = [dict(turn_number=i, prompt=f"Original question {i}.", response=f"Recorded reply {i}.",
                 timestamp=START + timedelta(seconds=i), ts_provenance="original") for i in range(1, 4)]
    probes = []
    for name, cutoff in [("older", 3), ("first", 1), ("another", 3)]:
        question = "What did I select?"
        review = dict(conversation="fixture", cutoff_turn=cutoff, reviewed_through_turn=cutoff,
                      question=question, expected_answer="Private scorer key.", knowledge_scope="private_history",
                      recent_only_answerable=cutoff == 1, task_types=["episodic_lookup"],
                      question_family_id="repeated-fact")
        probes.append((cutoff, dict(probe_id=name, question=question, expected_answer=review["expected_answer"],
                                   gold_turns=[1], label_status="source_reviewed_valid", label_review=review)))
    return rows, probes


def plan(history, **kwargs):
    return sources.build("fixture", *history, clock_basis=kwargs.pop("clock_basis", "original"), **kwargs)


def test_source_order_and_repeated_questions_keep_keys_outside_runtime(history):
    rows, probes = history
    result = plan(history)
    assert [e["id"] for e in result["events"]] == [
        "source:1", "probe:first", "source:2", "source:3", "probe:another", "probe:older"]
    assert plan((rows, list(reversed(probes)))) == result
    by_id = {probe["probe_id"]: probe for _, probe in probes}
    for index, event in enumerate(result["events"]):
        arguments = dict(source=rows[event["turn"] - 1]) if event["kind"] == "source" else dict(probe=by_id[event["probe_id"]])
        rendered = sources.runtime_projection(result, index, **arguments)
        assert "Private scorer key." not in str(rendered)
        assert "gold_turns" not in rendered and "label_review" not in rendered
        if event["kind"] == "probe":
            assert set(rendered) == {"id", "kind", "history", "turn", "clock", "clock_basis", "probe_id", "question"}
    assert not result["execution_ready"]


def test_simulated_gap_keeps_original_time_and_cannot_claim_calendar_authority(history):
    rows, probes = copy.deepcopy(history)
    for row in rows:
        row["ts_provenance"] = "synthetic_raw_import"
    with pytest.raises(ValueError, match="authentic"):
        plan((rows, probes))
    times = [START + timedelta(days=d) for d in (0, 30, 60)]
    result = plan((rows, probes), clock_basis="simulated", replay_times=times)
    events = [e for e in result["events"] if e["kind"] == "source"]
    assert events[1]["clock"] == times[1].isoformat()
    assert events[1]["original_time"] == rows[1]["timestamp"].isoformat()
    assert events[1]["provenance"] == "synthetic_raw_import"
    assert all(e["clock_basis"] == "simulated" for e in result["events"])
    assert result["identity_sha256"] != plan((rows, probes), clock_basis="simulated")["identity_sha256"]


def test_quiet_probe_clocks_preserve_original_sources_and_scorer_free_projection(history):
    rows, probes = history
    times = {p["probe_id"]: rows[cutoff - 1]["timestamp"] for cutoff, p in probes}
    times["older"] = START + timedelta(days=20)
    times["another"] = START + timedelta(days=10)
    baseline = plan(history, clock_basis="simulated")
    result = plan(history, clock_basis="simulated", probe_times=times)
    assert result["identity"]["policy"] == sources.QUERY_CLOCK_POLICY
    assert [e for e in result["events"] if e["kind"] == "source"] == [
        e for e in baseline["events"] if e["kind"] == "source"]
    assert [e["id"] for e in result["events"]] == [
        "source:1", "probe:first", "source:2", "source:3", "probe:another", "probe:older"]
    index = next(i for i, e in enumerate(result["events"]) if e["id"] == "probe:older")
    probe = next(p for _, p in probes if p["probe_id"] == "older")
    runtime = sources.runtime_projection(result, index, probe=probe)
    assert runtime["clock"] == times["older"].isoformat() and runtime["turn"] == 3
    assert "Private scorer key." not in str(runtime)
    assert runtime["clock_basis"] == "simulated" and result["identity_sha256"] != baseline["identity_sha256"]
    assert plan(history, clock_basis="simulated", probe_times=None) == baseline


@pytest.mark.parametrize("fault", ["authentic", "missing", "extra", "earlier", "future", "naive", "not_map"])
def test_unbound_or_future_quiet_clock_refuses_before_dispatch(history, fault):
    rows, probes = history
    times = {p["probe_id"]: rows[cutoff - 1]["timestamp"] for cutoff, p in probes}
    basis = "simulated"
    if fault == "authentic": basis = "original"
    elif fault == "missing": del times["first"]
    elif fault == "extra": times["unknown"] = START
    elif fault == "earlier": times["first"] = START
    elif fault == "future": times["first"] = rows[1]["timestamp"]
    elif fault == "naive": times["older"] = datetime(2025, 1, 1)
    elif fault == "not_map": times = list(times.values())
    with pytest.raises(ValueError): plan(history, clock_basis=basis, probe_times=times)


@pytest.mark.parametrize("fault", ["missing_turn", "naive", "backward", "unknown", "future_gold",
                                    "duplicate", "wrong_history", "cutoff", "unreviewed", "answer", "question"])
def test_invalid_timeline_refuses_before_dispatch(history, fault):
    rows, probes = copy.deepcopy(history)
    if fault == "missing_turn": rows.pop(1)
    elif fault == "naive": rows[0]["timestamp"] = datetime(2025, 1, 1)
    elif fault == "backward": rows[1]["timestamp"] = rows[0]["timestamp"]
    elif fault == "unknown": rows[0]["ts_provenance"] = "unknown"
    elif fault == "future_gold": probes[1][1]["gold_turns"] = [2]
    elif fault == "duplicate": probes.append(probes[0])
    elif fault == "wrong_history": probes[0][1]["label_review"]["conversation"] = "other"
    elif fault == "cutoff": probes[0][1]["label_review"]["cutoff_turn"] = 2
    elif fault == "unreviewed": probes[0][1]["label_review"]["reviewed_through_turn"] = 1
    elif fault == "answer": probes[0][1]["expected_answer"] = "Different key."
    else: probes[0][1]["question"] = "Changed question."
    with pytest.raises(ValueError):
        plan((rows, probes))


@pytest.mark.parametrize("times", [[START], [START, START, START + timedelta(days=2)]])
def test_incomplete_or_nonordered_simulated_clocks_refuse(history, times):
    with pytest.raises(ValueError):
        plan(history, clock_basis="simulated", replay_times=times)


def test_checkpoint_prefix_preserves_saved_events_and_refuses_revised_future(history):
    result = plan(history)
    receipt = sources.prefix_receipt(result, 2)
    assert sources.validate_prefix(result, receipt) == 2
    assert result["events"][2]["id"] == "source:2"
    changed = copy.deepcopy(history)
    changed[0][2]["response"] = "Revised future reply."
    with pytest.raises(ValueError, match="prefix changed"):
        sources.validate_prefix(plan(changed), receipt)
    corrupt = copy.deepcopy(result)
    corrupt["events"][0]["clock"] = START.isoformat()
    with pytest.raises(ValueError, match="identity/events changed"):
        sources.validate_prefix(corrupt, receipt)


def test_projection_refuses_mutated_source_and_answer_key_even_after_plan(history):
    result = plan(history)
    rows, probes = copy.deepcopy(history)
    rows[0]["prompt"] = "A different original."
    with pytest.raises(ValueError, match="source projection changed"):
        sources.runtime_projection(result, 0, source=rows[0])
    probes[1][1]["expected_answer"] = "A different scorer key."
    with pytest.raises(ValueError, match="occurrence projection changed"):
        sources.runtime_projection(result, 1, probe=probes[1][1])
