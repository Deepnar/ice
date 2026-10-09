"""Source-bound entity accounts using the existing cited-summary contract."""

import json

from sqlalchemy import select

from src.memory.models import CodexEvent, ColdStorage, EpisodicMemory
from src.memory.source import digest
from src.memory.summary_support import (SourceNotePrompt, assertion_instruction,
    assertion_text, original_units, parse_assertions, source_windows,
    verify_assertions, window_prompt)
from src.memory.support import supported_current
from src.workers.completion_text import IncompleteCompletion

PROOF_KEY = '_description_support'
REVIEW_KEY = '_description_review'
METHOD = 'source_bound_entity_note_v1'
SOURCE_LIMIT = 8
SCHEMA = {"type": "object", "properties": {"summary": {"type": "array", "items": {
    "type": "object", "properties": {"text": {"type": "string"}, "sources": {
        "type": "array", "items": {"type": "string"}, "minItems": 1}},
    "required": ["text", "sources"], "additionalProperties": False}}},
    "required": ["summary"], "additionalProperties": False}


def note_sources(db, entity):
    """Bounded complete sources, including the original after a cold transition."""
    batches = select(CodexEvent.batch_source).where(CodexEvent.entity_id == entity.id)
    found = {}
    for model in (EpisodicMemory, ColdStorage):
        rows = (db.query(model).filter(model.batch_id.in_(batches), model.is_private.is_(False))
                .order_by(model.timestamp.desc().nullslast(), model.id).limit(SOURCE_LIMIT).all())
        for row in rows:
            found.setdefault(str(row.id), row)
    return sorted(found.values(), key=lambda r: (r.timestamp.isoformat() if r.timestamp else '',
                                                str(r.id)), reverse=True)[:SOURCE_LIMIT]


def source_text(turns):
    return json.dumps(original_units(turns), ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def generate_note(entity, turns, *, llm=None, verifier=None):
    """A failed candidate is a diagnostic, never an overwrite or source deletion."""
    from src.workers.conversation_summary import _default_llm
    if not turns:
        return None, dict(method=METHOD, status='unknown', reason='missing_original_sources', sources=0)
    try:
        units = original_units(turns)
        source = source_text(turns)
        windows = source_windows(units)
    except ValueError as exc:
        return None, dict(method=METHOD, status='unknown', reason=str(exc), sources=len(turns))
    instruction = (
        'Write a concise source-backed account of the named entity using only '
        'these complete attributed originals. Focus on facts, roles and relations '
        'about that entity, not generic discussion process. Aim for120 words. '
        'Preserve speaker, uncertainty, negation, conditions, dates and corrections. '
        'An assistant proposal or user question is not a user approval. Do not '
        'claim current truth from an older mention or use recording time as event time. '
        'If the entity cannot be described from the sources, return an empty summary. '
        + assertion_instruction())
    prompt = SourceNotePrompt(instruction, 'Entity: ' + json.dumps(entity.canonical_name) + '\n\n' + window_prompt(windows))
    try:
        generated = llm(prompt, max_tokens=900) if llm else _default_llm(prompt, max_tokens=900, schema=SCHEMA)
    except IncompleteCompletion as exc:
        return None, dict(method=METHOD, status='unknown', reason='incomplete_generation',
                          sources=len(turns), finish_reason=exc.finish_reason, **exc.details)
    try:
        parsed = parse_assertions(generated, windows)
    except (ValueError, AttributeError, TypeError):
        return None, dict(method=METHOD, status='unknown', reason='invalid_summary_assertions', sources=len(turns),
                          source_sha256=digest(source), generated_candidate=generated if isinstance(generated, str) else None)
    candidate = assertion_text(parsed['summary'])
    proof = verify_assertions(source, candidate, parsed['summary'], windows, verifier=verifier)
    review = dict(method=METHOD, status=proof['status'], reason=proof['reason'],
                  sources=len(turns), verification=proof, entity_id=str(entity.id),
                  canonical_name=entity.canonical_name)
    return (candidate if supported_current(proof, source, candidate) else None), review


def current_note_support(db, entity):
    record = (entity.properties or {}).get(PROOF_KEY)
    if (not isinstance(record, dict) or record.get('method') != METHOD
            or record.get('entity_id') != str(entity.id)
            or record.get('canonical_name') != entity.canonical_name):
        return False
    try:
        source = source_text(note_sources(db, entity))
    except ValueError:
        return False
    return supported_current(record.get('verification'), source, entity.description or '')
