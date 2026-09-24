"""Authoritative speaker boundaries supplied by writers, never inferred by NER."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceUnit:
    role: str
    text: str
    start: int
    end: int


def digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chat_provenance(user: str, assistant: str) -> dict:
    raw = f"User: {user}\n\nAssistant: {assistant}"
    start = len("User: ")
    assistant_start = start + len(user) + len("\n\nAssistant: ")
    return {"raw_sha256": digest(raw), "segments": [
        {"role": "user", "start": start, "end": start + len(user)},
        {"role": "assistant", "start": assistant_start, "end": len(raw)},
    ]}


def single_provenance(raw: str, role: str) -> dict:
    if role not in {"user", "assistant", "document"}:
        raise ValueError("Unsupported source role")
    return {"raw_sha256": digest(raw), "segments": [
        {"role": role, "start": 0, "end": len(raw)},
    ]}


def source_units(row) -> list[SourceUnit]:
    raw = getattr(row, "raw_text", None) or ""
    record = getattr(row, "source_spans", None)
    unknown = [SourceUnit("unknown", raw, 0, len(raw))] if raw else []
    if not isinstance(record, dict) or record.get("raw_sha256") != digest(raw):
        return unknown
    segments = record.get("segments")
    if not isinstance(segments, list) or not segments:
        return unknown
    result, previous_end = [], 0
    for segment in segments:
        if not isinstance(segment, dict):
            return unknown
        role, start, end = (segment.get(k) for k in ("role", "start", "end"))
        if (role not in {"user", "assistant", "document"}
                or type(start) is not int or type(end) is not int
                or not previous_end <= start <= end <= len(raw)):
            return unknown
        result.append(SourceUnit(role, raw[start:end], start, end))
        previous_end = end
    return result
