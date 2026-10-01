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
