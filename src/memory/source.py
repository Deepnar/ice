"""Authoritative speaker boundaries supplied by writers, never inferred by NER."""

import hashlib
import re
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


# Moved from the deleted workers/extraction_result.py (NuExtract3 codex extractor)
# (CODEX_V4); the read-side source checks still resolve quotes with it.
def original_source_quote(source: str, quote: str, *, allow_case_copy=False) -> str | None:
    """Resolve copying changes to original bytes, never accept a paraphrase.

    Whitespace-only matching is the default. Source selection may explicitly
    permit case copies; ambiguous originals still stay unresolved. The returned
    value always has the original spelling, including case and whitespace.
    """
    if not isinstance(quote, str) or not quote.strip():
        return None
    quote = quote.strip()
    if quote in source:
        return quote
    def resolve(case_copy):
        parts, positions = [], []
        for match in re.finditer(r"\s+|\S", source):
            part = " " if match.group().isspace() else match.group()
            part = part.casefold() if case_copy else part
            parts.append(part)
            positions.extend([(match.start(), match.end())] * len(part))
        normalized = "".join(parts)
        wanted = re.sub(r"\s+", " ", quote)
        wanted = wanted.casefold() if case_copy else wanted
        matches, offset = set(), 0
        while (start := normalized.find(wanted, offset)) >= 0:
            end = start + len(wanted)
            original = source[positions[start][0]:positions[end - 1][1]]
            spelling = re.sub(r"\s+", " ", original)
            # A match must consume complete source characters, including a
            # Unicode case-fold expansion (never match half of ß → ss).
            if (spelling.casefold() if case_copy else spelling) == wanted:
                matches.add(original)
            offset = start + 1
        return next(iter(matches)) if len(matches) == 1 else None

    original = resolve(False)
    return original if original is not None or not allow_case_copy else resolve(True)
