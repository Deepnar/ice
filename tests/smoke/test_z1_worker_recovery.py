"""Fault tolerance must preserve truth, boundedness and restart state."""
import io
import json
from copy import deepcopy

import pytest
from sqlalchemy.exc import TimeoutError as DatabaseTimeout

from scripts.z1.worker_recovery import WorkerOutage, WorkerRecovery
from src.workers.extraction_result import ExtractionOutputError


def test_isolated_model_fault_is_bounded_recorded_and_resumable(monkeypatch):
    monkeypatch.setattr("scripts.z1.worker_recovery.time.sleep", lambda _seconds: None)
    sink = io.StringIO()
    recovery = WorkerRecovery(sink, "continue")
    calls, rollbacks = [], []
    def failed():
        calls.append(1)
        raise ExtractionOutputError("truncated synthetic output")
    assert recovery.call(failed, job="post_flight", context={"conversation": "c", "turn": 1},
                         rollback=lambda: rollbacks.append(1)) == (False, None)
    assert len(calls) == len(rollbacks) == 2
    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [r["event"] for r in rows] == ["worker_attempt_failed"] * 2 + ["worker_degraded"]
    assert recovery.health("c", [1])["degraded_gold_turns"] == [1]
    resumed = WorkerRecovery(sink, "continue", json.loads(json.dumps(recovery.state)))
    assert resumed.call(lambda: 42, job="post_flight", context={"conversation": "c", "turn": 2},
                        rollback=lambda: None) == (True, 42)
    assert resumed.state["streaks"]["post_flight"] == 0
    assert resumed.health()["status"] == "degraded"  # Success cannot erase earlier missing work.


@pytest.mark.parametrize("exc", [ValueError("unknown"), DatabaseTimeout("DB pool"), KeyboardInterrupt()])
def test_unknown_integrity_and_interrupt_do_not_become_degraded(exc):
    sink = io.StringIO()
    recovery = WorkerRecovery(sink, "continue")
    def failed():
        raise exc
    with pytest.raises(type(exc)):
        recovery.call(failed, job="post_flight", context={}, rollback=lambda: None)
    assert sink.getvalue() == "" and recovery.health()["status"] == "clean"


def test_persistent_worker_outage_pauses_without_erasing_receipts(monkeypatch):
    monkeypatch.setattr("scripts.z1.worker_recovery.time.sleep", lambda _seconds: None)
    recovery = WorkerRecovery(io.StringIO(), "continue")
    def failed():
        raise ConnectionError("provider unavailable")
    for turn in range(1, 4):
        recovery.call(failed, job="post_flight", context={"conversation": "c", "turn": turn}, rollback=lambda: None)
    restored = WorkerRecovery(io.StringIO(), "continue", deepcopy(recovery.state))
    with pytest.raises(WorkerOutage, match="post_flight"):
        restored.check_outage()
    assert restored.health()["degraded_worker_calls"] == 3 and restored.health()["failed_attempts"] == 6


def test_processing_strata_keep_degraded_errors_in_denominators():
    from scripts.z1.judge_answers import processing_outcome_strata
    rows = [{"memory_processing": {"status": "clean"}, "arm_a_grade": "correct"},
            {"memory_processing": {"status": "degraded", "degraded_gold_turns": [1]}, "winner": "ERROR"},
            {"memory_processing": {"status": "degraded"}, "arm_a_grade": "partial"}]
    result = processing_outcome_strata(rows)
    assert sum(cell["occurrences"] for cell in result.values()) == 3
    assert result["degraded_gold"]["judge_errors"] == 1
    assert result["degraded_gold"]["arm_a_grades"] == {"ungraded": 1}
