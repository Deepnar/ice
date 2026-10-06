"""A successful-looking completion flag cannot hide missing replay events."""
from copy import deepcopy

import pytest

from scripts.z1.replay_validation import validate_complete_replay


def complete_rows():
    return [
        {"event": "run", "meta": {"extra": {
            "version": "v3", "planned_probes": [["p", "example", 2]]}}},
        {"event": "turn", "conversation": "example", "turn": 1,
         "recorded_at": "2025-01-01T00:00:00+00:00", "before_turns": 0},
        {"event": "written", "conversation": "example", "turn": 1,
         "episodic_id": "row-1", "batch_id": "batch-1"},
        {"event": "turn", "conversation": "example", "turn": 2,
         "recorded_at": "2025-01-02T00:00:00+00:00", "before_turns": 1},
        {"event": "written", "conversation": "example", "turn": 2,
         "episodic_id": "row-2", "batch_id": "batch-2"},
        {"event": "as_of_probe", "probe_id": "p", "conversation": "example",
         "split_turn": 2, "state_turns": 2, "question_time": "2025-01-02T00:00:00+00:00",
         "gold_turns": [1], "gold_sources": [{"turn": 1, "source_ids": ["row-1", "batch-1"]}]},
        {"event": "complete", "complete_selected_corpus": True,
         "turns_by_conversation": {"example": 2}, "table_counts": {"episodic_memory": 2},
         "as_of_probes": 1, "expected_as_of_probes": 1},
    ]


def test_complete_event_stream_is_verified():
    rows = complete_rows()
    assert validate_complete_replay(iter(rows), {"example": 2})["complete"] == rows[-1]
    with pytest.raises(ValueError, match="every selected historical turn"):
        validate_complete_replay(rows, {"example": 3})


@pytest.mark.parametrize("damage,reason", [
    ("missing_preflight", "matching turn preflight"),
    ("missing_write", "historical cutoff"),
    ("duplicated_probe", "duplicate probe"),
    ("omitted_probe", "planned probe"),
    ("wrong_source", "original write"),
    ("wrong_clock", "historical cutoff"),
    ("trailing_event", "trailing events"),
    ("failure", "failures"),
])
def test_completion_flag_cannot_cover_bad_events(damage, reason):
    rows = deepcopy(complete_rows())
    if damage == "missing_preflight":
        rows.pop(1)
    elif damage == "missing_write":
        rows.pop(4)
    elif damage == "duplicated_probe":
        rows.insert(-1, deepcopy(rows[-2]))
    elif damage == "omitted_probe":
        rows.pop(-2)
    elif damage == "wrong_source":
        rows[-2]["gold_sources"][0]["source_ids"] = ["other-source"]
    elif damage == "wrong_clock":
        rows[-2]["question_time"] = "2026-01-01T00:00:00+00:00"
    elif damage == "trailing_event":
        rows.append({"event": "maintenance"})
    else:
        rows.insert(-1, {"event": "maintenance_failed"})
    with pytest.raises(ValueError, match=reason):
        validate_complete_replay(rows, {"example": 2})


def test_degraded_complete_replay_requires_bound_receipts_and_health():
    from scripts.z1.worker_recovery import POLICY, processing_health
    rows = complete_rows()
    rows[0]["meta"]["extra"].update(worker_failure_policy="continue", worker_recovery_policy=POLICY)
    context = {"policy": POLICY, "conversation": "example", "turn": 1, "job": "post_flight",
               "stage": "post_flight", "batch_id": "batch-1", "attempts": 2,
               "error_type": "ExtractionOutputError"}
    faults = [{**context, "event": "worker_attempt_failed", "attempt": attempt} for attempt in (1, 2)]
    fault = {**faults[-1], "event": "worker_degraded", "original_source_retained": True}
    rows[2]["post_flight_complete"] = False
    rows[2:2] = [*faults, fault]
    rows[-2]["memory_processing"] = processing_health([fault], 2, "example", [1])
    rows[-1]["memory_processing"] = processing_health([fault], 2)
    assert validate_complete_replay(rows, {"example": 2})["complete"]["memory_processing"]["status"] == "degraded"
    for damage in ("policy", "health", "source", "attempt"):
        bad = deepcopy(rows)
        if damage == "policy":
            bad[0]["meta"]["extra"]["worker_failure_policy"] = "strict"
        elif damage == "health":
            bad[-1]["memory_processing"]["status"] = "clean"
        elif damage == "source":
            bad[5]["batch_id"] = "other"
        else:
            bad[3]["attempt"] = 3
        with pytest.raises(ValueError):
            validate_complete_replay(bad, {"example": 2})


@pytest.mark.parametrize("damage", [None, "prefix_count", "writer", "unapproved_file", "duplicate"])
def test_instrument_change_is_bound_to_completed_prefix_and_same_writers(damage, tmp_path, monkeypatch):
    rows = complete_rows()
    old = {"code_sha256": "old", "settings_sha256": "same", "models": "same"}
    marker = {"event": "instrument_continuation", "format": "ice-v3-instrument-continuation-1",
              "from_identity": old, "to_identity": {**old, "code_sha256": "new"},
              "completed": {"example": 1}, "trace_sha256": "frozen-prefix",
              "archive_manifest_sha256": "frozen-manifest", "code_changes": {
                  path: {"before": "old", "after": "new"} for path in
                  ("scripts/z1/worker_recovery.py", "scripts/z1/replay_checkpoint.py")}}
    rows.insert(3, marker)
    register_repairs(tmp_path, monkeypatch, [marker])
    if damage == "prefix_count":
        marker["completed"] = {"example": 2}
    elif damage == "writer":
        marker["to_identity"]["models"] = "different"
    elif damage == "unapproved_file":
        marker["code_changes"]["src/workers/conversation_summary.py"] = {}
    elif damage == "duplicate":
        rows.insert(4, deepcopy(marker))
    if damage:
        with pytest.raises(ValueError, match="instrument continuation"):
            validate_complete_replay(rows, {"example": 2})
    else:
        assert validate_complete_replay(rows, {"example": 2})["complete"] == rows[-1]


def register_repairs(tmp_path, monkeypatch, markers):
    import json
    from scripts.z1 import replay_checkpoint
    registry = tmp_path / "repairs.json"
    registry.write_text(json.dumps({"format": "ice-v3-reviewed-instrument-repairs-1", "repairs": [
        {"from_code": m["from_identity"]["code_sha256"], "to_code": m["to_identity"]["code_sha256"],
         "code_changes": deepcopy(m["code_changes"])} for m in markers]}))
    monkeypatch.setattr(replay_checkpoint, "REPAIR_REGISTRY", registry)


@pytest.mark.parametrize("damage", [None, "disconnected", "reverted", "duplicate", "unregistered"])
def test_ordered_instrument_chain_remains_bound_to_frozen_writers(damage, tmp_path, monkeypatch):
    rows = complete_rows()
    old = {"code_sha256": "old", "models": "same", "settings_sha256": "same"}
    middle = {**old, "code_sha256": "middle"}
    first = {"event": "instrument_continuation", "format": "ice-v3-instrument-continuation-1",
             "from_identity": old, "to_identity": middle, "completed": {"example": 1},
             "trace_sha256": "first", "archive_manifest_sha256": "first", "code_changes": {
                 "scripts/z1/worker_recovery.py": {"before": "one", "after": "two"}}}
    second = {**first, "from_identity": middle, "to_identity": {**old, "code_sha256": "latest"},
              "completed": {"example": 2}, "code_changes": {
                  "scripts/z1/seed_v3.py": {"before": "two", "after": "three"}}}
    register_repairs(tmp_path, monkeypatch, [first, second])
    if damage == "disconnected":
        second["from_identity"] = old
    elif damage == "reverted":
        second["to_identity"] = old
    elif damage == "unregistered":
        second["code_changes"] = {"scripts/z1/seed_v3.py": {"before": "two", "after": "unknown"}}
    rows.insert(3, first)
    rows.insert(6, second)
    if damage == "duplicate":
        rows.insert(7, deepcopy(second))
    if damage:
        with pytest.raises(ValueError, match="instrument continuation"):
            validate_complete_replay(rows, {"example": 2})
    else:
        assert validate_complete_replay(rows, {"example": 2})["complete"] == rows[-1]
