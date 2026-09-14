"""Source-backed Codex excerpts: exact spans, attribution and surrounding context."""

import re
from dataclasses import dataclass

from src.memory.source import digest, source_units


@dataclass(frozen=True)
class SourceExcerpt:
    text: str
    role: str
    start: int
    end: int
    raw_sha256: str


def source_excerpts(row, sentence: str) -> list[SourceExcerpt]:
    """Resolve exact matches within authoritative units; never guess a speaker.

    Keep the containing paragraph, including surrounding conditions/corrections.
    Paragraph boundaries only select context; they never classify assertion type.
    Repeated mentions retain distinct offsets. A fabricated quote yields no match.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return []
    sentence = sentence.strip()
    raw = row.raw_text or ""
    found = {}
    for unit in source_units(row):
        separators = list(re.finditer(r"\n[ \t]*\n", unit.text))
        offset = 0
        while True:
            start = unit.text.find(sentence, offset)
            if start < 0:
                break
            end = start + len(sentence)
            left = max((m.end() for m in separators if m.end() <= start), default=0)
            right = min((m.start() for m in separators if m.start() >= end), default=len(unit.text))
            while left < right and unit.text[left].isspace():
                left += 1
            while right > left and unit.text[right - 1].isspace():
                right -= 1
            absolute_start, absolute_end = unit.start + left, unit.start + right
            found[(absolute_start, absolute_end)] = SourceExcerpt(
                raw[absolute_start:absolute_end], unit.role, absolute_start,
                absolute_end, digest(raw),
            )
            offset = start + 1
    return list(found.values())


def excerpt_is_current(row, excerpt) -> bool:
    """A source edit invalidates old offsets even when the old sentence survives."""
    raw = row.raw_text or ""
    return (getattr(excerpt, "raw_sha256", None) == digest(raw)
            and type(excerpt.start) is int and type(excerpt.end) is int
            and 0 <= excerpt.start < excerpt.end <= len(raw)
            and raw[excerpt.start:excerpt.end] == excerpt.text
            and any(u.role == excerpt.role and u.start <= excerpt.start
                    and excerpt.end <= u.end for u in source_units(row)))


def claim_representation(claim) -> str:
    """Use compression only with current source/claim/model support metadata."""
    from src.api.config import settings

    v = claim.verification if isinstance(claim.verification, dict) else {}
    if (v.get('status') == 'supported'
            and v.get('source_sha256') == digest(claim.text)
            and v.get('claim_sha256') == digest(claim.sentence)
            and v.get('verifier') == f'{settings.source_support_model}@{settings.source_support_revision}'
            and isinstance(v.get('scores'), dict)
            and v['scores'].get('entailment', 0) >= settings.source_support_threshold
            and claim.sentence in claim.text):
        return claim.sentence
    return claim.text


def store_claims(db, row, sentences, *, encoder, verifier=None):
    """Persist exact source excerpts inside the caller's graph transaction."""
    from dataclasses import asdict

    import structlog
    from src.memory.models import CodexClaim
    from src.memory.support import verify_support

    log = structlog.get_logger('ice.memory.claims')
    stored = []
    for sentence in dict.fromkeys(sentences):
        excerpts = source_excerpts(row, sentence)
        if not excerpts:
            log.warning('claim_source_span_unresolved', sentence_chars=len(sentence))
        for excerpt in excerpts:
            key = dict(source_batch=row.batch_id, raw_sha256=excerpt.raw_sha256,
                       start=excerpt.start, end=excerpt.end, sentence_sha256=digest(sentence))
            existing = db.query(CodexClaim).filter_by(**key).first()
            if existing:
                stored.append(existing)
                continue
            verdict = (verifier or verify_support)(excerpt.text, sentence)
            claim = CodexClaim(**key, episodic_id=row.id, conversation_id=row.conversation_id,
                               role=excerpt.role, sentence=sentence,
                               text=excerpt.text, verification=asdict(verdict))
            # Encode the complete evidence, not just a potentially misleading
            # selected clause. Oversized evidence remains lexically searchable.
            tokens = encoder.tokenizer(excerpt.text, truncation=False)['input_ids']
            if len(tokens) <= encoder.max_seq_length:
                claim.embedding = encoder.encode(excerpt.text).tolist()
            else:
                log.warning('claim_embedding_too_long', tokens=len(tokens),
                            limit=encoder.max_seq_length)
            db.add(claim)
            db.flush()
            stored.append(claim)
    return stored
