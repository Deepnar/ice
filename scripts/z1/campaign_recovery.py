"""Bounded retries based on fresh persisted failures, never log-text guesses."""

import json
import os
from pathlib import Path

RETRY_DELAYS = (2, 8)
POLICY = "ice-v3-persisted-failure-recovery-1"
TRANSIENT = {"APIConnectionError", "APITimeoutError", "InternalServerError",
             "ConnectionError", "ConnectError", "ConnectTimeout", "ReadTimeout",
             "TimeoutError", "Timeout", "RemoteProtocolError", "URLError",
             "RemoteDisconnected"}
LOCAL_OUTPUT = {"ExtractionOutputError", "IncompleteCompletion"}


def failure_path(root: Path, stage: str, arm: str) -> Path | None:
    if stage == "seed":
        return root / "seed.jsonl"
    if stage == "answers":
        return root / f"answers-{arm}.json"
    if stage == "judge":
        paths = [root / f"judge-full-vs-{arm}{suffix}"
                 for suffix in (".json", ".partial.json")]
        present = [p for p in paths if p.exists()]
        return max(present, key=lambda p: p.stat().st_mtime_ns) if present else paths[0]
    return None


def stamp(path):
    if path is None or not path.exists():
        return None
    s = path.stat()
    return (str(path), s.st_ino, s.st_size, s.st_mtime_ns)


def failure_evidence(root, stage, arm, before):
    path = failure_path(root, stage, arm)
    if stamp(path) == before or path is None or not path.exists():
        return {"reason": "no_fresh_structured_failure"}
    try:
        if stage == "seed":
            with path.open("rb") as source:
                source.seek(max(0, path.stat().st_size - 65536))
                lines = source.read().splitlines()
            row = json.loads(lines[-1])
            if row.get("event") != "failed":
                return {"reason": "no_terminal_failure_event"}
            return {k: row.get(k) for k in ("error_type", "stage", "conversation", "turn")}
        data = json.loads(path.read_text())
        if stage == "answers":
            row = data.get("records", [])[-1]
            if not row.get("error"):
                return {"reason": "no_failed_answer"}
            return {"error_type": row.get("error_type"), "status": row.get("error_status")}
        row = data.get("results", [])[-1]
        for order in row.get("order_verdicts", []):
            raw = order.get("raw") or {}
            if raw.get("verdict") == "ERROR":
                return {"reason": raw.get("reason"), "error_type": raw.get("error_type"),
                        "operator_required": raw.get("operator_required", False)}
        return {"reason": row.get("reason")}
    except (OSError, ValueError, IndexError, KeyError, TypeError, AttributeError):
        return {"reason": "unreadable_failure_receipt"}


def recoverable(stage, evidence):
    # Quota/authentication/identity/resource failures need operator action.
    if (evidence.get("status") in (401, 402, 403, 429)
            or evidence.get("error_type") == "ProviderAccessError"
            or evidence.get("operator_required")):
        return False
    if stage == "seed":
        # A checkpoint restore precedes every replay attempt, including DB
        # disconnections after a partially applied periodic job.
        return evidence.get("error_type") in TRANSIENT | LOCAL_OUTPUT | {"OperationalError"}
    if stage == "answers":
        return (evidence.get("error_type") in TRANSIENT
                or evidence.get("status") in (408, 500, 502, 503, 504))
    if stage == "judge":
        return (evidence.get("reason") in {"api_http_408", "api_http_500", "api_http_502",
                                           "api_http_503", "api_http_504"}
                or evidence.get("error_type") in TRANSIENT)
    return False


def attempt_receipt(root, **row):
    """Append and fsync before any retry, retaining all failed attempts."""
    with (root / "campaign-attempts.jsonl").open("a") as sink:
        sink.write(json.dumps({"policy": POLICY, **row}, default=str) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
