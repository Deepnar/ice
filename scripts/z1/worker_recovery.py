"""Truthful, checkpointed tolerance of isolated model-worker failures."""
from __future__ import annotations

import json
import time

from scripts.z1.campaign_recovery import LOCAL_OUTPUT, TRANSIENT

POLICY = "ice-v3-isolated-worker-failures-1"
OUTAGE_POLICY = "ice-v3-transport-outages-1"


def transport_failure(record):
    """Returned bad content is responsive execution, not a provider outage."""
    return (record.get("error_type") in TRANSIENT
            or record.get("error_status") in (408, 500, 502, 503, 504))


def model_failure(exc):
    from sqlalchemy.exc import SQLAlchemyError
    if isinstance(exc, SQLAlchemyError):
        return False
    status = getattr(exc, "status_code", None)
    if status in (401, 402, 403, 429):
        return False
    return (type(exc).__name__ in TRANSIENT | LOCAL_OUTPUT
            or status in (408, 500, 502, 503, 504))


def processing_health(failures, failed_attempts, conversation=None, gold_turns=()):
    turns = sorted({r["turn"] for r in failures
        if r["job"] == "post_flight" and r["conversation"] == conversation})
    return {"status": "degraded" if failures else "clean",
            "failed_attempts": failed_attempts, "degraded_worker_calls": len(failures),
            "degraded_post_flight_turns": turns,
            "degraded_gold_turns": sorted(set(turns) & set(gold_turns))}


class WorkerOutage(RuntimeError):
    """Repeated exhausted invocations need an operator, not a prefix replay."""


class WorkerRecovery:
    def __init__(self, sink, policy="strict", state=None):
        if policy not in {"strict", "continue"}:
            raise ValueError("invalid worker failure policy")
        self.sink, self.policy = sink, policy
        self.state = state if state is not None else {
            "degraded": [], "failed_attempts": 0, "streaks": {}, "last_attempt": {}}
        if self.state.get("outage_policy") != OUTAGE_POLICY:
            # Old streaks counted every output failure. Reclassify only their
            # known consecutive tail; a success already reset the old counter.
            for job, count in self.state["streaks"].items():
                if type(count) is not int or count < 0:
                    raise ValueError("invalid legacy worker outage streak")
                tail = [r for r in self.state["degraded"] if r["job"] == job][-count:] if count else []
                if len(tail) != count:
                    raise ValueError("invalid legacy worker outage streak")
                streak = 0
                for record in reversed(tail):
                    if not transport_failure(record):
                        break
                    streak += 1
                self.state["streaks"][job] = streak
            self.state["outage_policy"] = OUTAGE_POLICY

    def emit(self, event, **record):
        self.sink.write(json.dumps({"event": event, "policy": POLICY, **record}, default=str) + "\n")
        self.sink.flush()

    def call(self, fn, *, job, context, rollback, attempts=2):
        if self.policy == "strict":
            return True, fn()
        for attempt in range(1, attempts + 1):
            try:
                result = fn()
            except Exception as exc:
                if not model_failure(exc):
                    raise
                rollback()
                self.state["failed_attempts"] += 1
                record = {**context, "job": job, "attempt": attempt,
                          "attempts": attempts, "error_type": type(exc).__name__,
                          "error_status": getattr(exc, "status_code", None),
                          "error": str(exc)[:500]}
                self.emit("worker_attempt_failed", **record)
                print(f"WARNING: v3 {job} model attempt {attempt}/{attempts} failed "
                      f"({record['error_type']}); original source retained", flush=True)
                if attempt < attempts:
                    time.sleep(2)
                    continue
                self.state["degraded"].append(record)
                self.state["streaks"][job] = (
                    self.state["streaks"].get(job, 0) + 1 if transport_failure(record) else 0)
                self.emit("worker_degraded", **record, original_source_retained=True)
                return False, None
            else:
                self.state["streaks"][job] = 0
                return True, result

    def health(self, conversation=None, gold_turns=()):
        return processing_health(self.state["degraded"], self.state["failed_attempts"],
                                 conversation, gold_turns)

    def check_outage(self):
        jobs = sorted(job for job, streak in self.state["streaks"].items() if streak >= 3)
        if jobs:
            raise WorkerOutage("three successive transport failures: " + ", ".join(jobs))
