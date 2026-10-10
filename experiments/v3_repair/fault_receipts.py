"""Private complete exception receipts; callers retain their existing fault policy."""

import fcntl
import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.v3_repair.prepare_inputs import digest

ROOT = Path(__file__).resolve().parents[2]
CONTEXT_FIELDS = {
    "conversation", "history", "turn", "cutoff", "probe_id", "job", "stage",
    "attempt", "attempts", "source_time", "model", "profile", "window",
    "artifact_id", "coordinate", "arm", "batch_id", "request_id",
}
RECEIPT_FIELDS = {
    "method", "endpoint", "completion_status", "response_id", "returned_model",
    "incomplete_reason", "provider_error_code", "usage",
}
USAGE_FIELDS = {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"}
USAGE_DETAILS = {
    "input_tokens_details": {"cached_tokens"}, "prompt_tokens_details": {"cached_tokens"},
    "output_tokens_details": {"reasoning_tokens"},
    "completion_tokens_details": {"reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"},
}


def safe_receipt(error):
    raw = getattr(error, "receipt", None)
    if not isinstance(raw, dict): return None
    result = {k: v for k, v in raw.items() if k in RECEIPT_FIELDS - {"usage"}
              and (v is None or isinstance(v, str))}
    usage = raw.get("usage")
    if isinstance(usage, dict):
        numeric = {k: v for k, v in usage.items() if k in USAGE_FIELDS and type(v) is int and v >= 0}
        for key, fields in USAGE_DETAILS.items():
            if isinstance(usage.get(key), dict):
                numeric[key] = {k: v for k, v in usage[key].items() if k in fields and type(v) is int and v >= 0}
        result["usage"] = numeric or None
    return result


class FaultRecorder:
    def __init__(self, path, run_identity, *, private_root=ROOT / "logs", secrets=()):
        self.path = Path(path).resolve()
        if not self.path.is_relative_to(Path(private_root).resolve()) or not run_identity:
            raise ValueError("fault receipts require a private path and run identity")
        self.run_identity = run_identity
        self.secrets = [s for s in secrets if isinstance(s, str) and s]
        self._refresh_secrets()

    def _refresh_secrets(self):
        supplied = [value for name, value in os.environ.items()
                    if any(word in name.upper().split("_") for word in ("KEY", "TOKEN", "PASSWORD", "SECRET"))
                    and len(value) >= 8]
        self.secrets = sorted(set([*self.secrets, *supplied]), key=len, reverse=True)

    def redact(self, value):
        if isinstance(value, str):
            for secret in self.secrets: value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"(?i)(bearer\s+)[^\s\"'<>]+", r"\1[REDACTED]", value)
            value = re.sub(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+:[^/\s@]+@", r"\1[REDACTED]@", value)
            return re.sub(r"(?i)([\"']?(?:api[_-]?key|token|password|secret)[\"']?\s*[=:]\s*[\"']?)[^\s\"'&,<>]+",
                          r"\1[REDACTED]", value)
        if isinstance(value, dict):
            return {str(k): self.redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [self.redact(v) for v in value]
        if value is None or type(value) in (int, float, bool): return value
        raise ValueError("fault metadata must be explicit JSON values, never arbitrary objects/locals")

    def record(self, exc, context):
        if set(context) - CONTEXT_FIELDS:
            raise ValueError("undeclared fault context fields")
        # Native cloud runners may load .env after constructing the observer.
        self._refresh_secrets()
        chain, visited = [], set()
        def capture(error, relationship):
            if id(error) in visited: return
            visited.add(id(error))
            chain.append(dict(relationship=relationship, error_type=type(error).__name__,
                              error=str(error), error_status=getattr(error, "status_code", None),
                              request_id=getattr(error, "request_id", None), provider_receipt=safe_receipt(error)))
            if error.__cause__ is not None: capture(error.__cause__, "cause")
            elif error.__context__ is not None and not error.__suppress_context__:
                capture(error.__context__, "context")
            for child in getattr(error, "exceptions", ()):
                capture(child, "group_member")
        capture(exc, "raised")
        body = self.redact(dict(run_identity=self.run_identity,
            wall_time=datetime.now(timezone.utc).isoformat(), context=context,
            error_type=type(exc).__name__, error=str(exc),
            error_status=getattr(exc, "status_code", None),
            request_id=getattr(exc, "request_id", None), provider_receipt=safe_receipt(exc),
            exception_chain=chain,
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            capture="complete_exception_chain_no_locals", policy_changed=False))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8") as sink:
            fcntl.flock(sink, fcntl.LOCK_EX)
            prior, sequence = self._chain(sink)
            body.update(sequence=sequence + 1, previous_receipt_sha256=prior)
            body["receipt_sha256"] = digest(body)
            sink.write(json.dumps(body, ensure_ascii=False, allow_nan=False) + "\n")
            sink.flush(); os.fsync(sink.fileno())
        parent = os.open(self.path.parent, os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
        return body["receipt_sha256"]

    def _chain(self, source):
        prior = None; sequence = 0
        for line in source:
            if not line.endswith("\n"):
                raise ValueError("fault journal has an interrupted tail; retain prior evidence")
            row = json.loads(line)
            unsigned = {k: v for k, v in row.items() if k != "receipt_sha256"}
            if (digest(unsigned) != row.get("receipt_sha256")
                    or row.get("previous_receipt_sha256") != prior
                    or row.get("sequence") != sequence + 1
                    or row.get("run_identity") != self.run_identity):
                raise ValueError("fault receipt chain changed; retain prior evidence")
            prior = row["receipt_sha256"]; sequence += 1
        return prior, sequence

    def verify(self):
        if not self.path.exists(): return 0
        with self.path.open() as source:
            fcntl.flock(source, fcntl.LOCK_SH)
            return self._chain(source)[1]

    def call(self, fn, *, context):
        try:
            return fn()
        except BaseException as exc:
            self.record(exc, context)
            raise
