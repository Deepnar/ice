"""v3 actual background callers preserve role selection and completion status."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import BatchSummary, Conversation, EpisodicMemory, IdempotencyKey
from src.workers.completion_text import IncompleteCompletion

assert (os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_') and
        make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE'])


@pytest.fixture(autouse=True)
def controlled_background_capacity(monkeypatch):
    from src.model_registry import registry, runtime_probe
    monkeypatch.setattr(registry, 'get_model_context_window', lambda *a: 32768)
    monkeypatch.setattr(runtime_probe, 'serving_window', lambda *a: 32768)


def client_for(create):
    return NS(chat=NS(completions=NS(create=create)))


def response(content, finish='stop'):
    return NS(choices=[NS(message=NS(content=content), finish_reason=finish)])


@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_actual_turn_summary_uses_background_model_and_rejects_prefix(monkeypatch, finish):
    from src.workers import post_flight as worker
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response('Atlas keeps the schema.', finish)
    monkeypatch.setattr(worker, 'bg_client', client_for(create))
    monkeypatch.setattr(worker, 'get_bg_model_name', lambda: 'local-background')
    text, _, _ = worker.generate_summary('Keep the schema.', 'Noted.', dict(entities=[], figures=[], identifiers=[]),
                                        model_used='cloud-answer-model')
    assert calls[0]['model'] == 'local-background'
    assert calls[0]['timeout'] == worker.bg_timeout(settings.turn_summary_max_tokens)
    assert bool(text) == (finish == 'stop')


def test_turn_summary_propagates_runtime_yield(monkeypatch):
    from src.workers import post_flight as worker
    from src.workers.runtime import JobYielded
    def create(**kwargs):
        raise JobYielded('foreground active')
    monkeypatch.setattr(worker, 'bg_client', client_for(create))
    monkeypatch.setattr(worker, 'get_bg_model_name', lambda: 'local-background')
    with pytest.raises(JobYielded):
        worker.generate_summary('A prompt.', 'A response.', dict(entities=[], figures=[], identifiers=[]))


def test_conversation_summary_real_client_boundary_rejects_length(monkeypatch):
    from src.workers import bg_client_factory, conversation_summary
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda:
        client_for(lambda **kw: response('Plausible partial overview.', 'length')))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'local-background')
    with pytest.raises(IncompleteCompletion):
        conversation_summary._default_llm('Original evidence.')


def seed_turns(db, cid, sid, n=6):
    db.add(Conversation(id=cid)); db.flush()
    now = datetime.now(timezone.utc) - timedelta(days=settings.batch_summary_age_days + 2)
    turns = []
    for i in range(n):
        row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(), session_id=sid,
            timestamp=now + timedelta(minutes=i), raw_text='User: Explain before changing it.\n\nAssistant: Agreed.',
            context_reliance='Long_Term_Memory', decay_score=1.0,
            lossless_flag=False, is_document=False, is_private=False,
            idempotency_key=str(uuid.uuid4()))
        db.add(row); turns.append(row)
    db.commit()
    return turns


def test_procedural_uses_background_model_and_does_not_stamp_partial_output(monkeypatch):
    from src.workers import procedural_extractor as worker
    from src.workers.idempotency import job_key
    cid, sid = uuid.uuid4(), uuid.uuid4()
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response('PATTERN: Requests explanation. | EVIDENCE: 1, 2', 'length')
    monkeypatch.setattr(worker, 'bg_client', client_for(create))
    monkeypatch.setattr(worker, 'get_bg_model_name', lambda: 'local-background')
    monkeypatch.setattr(settings, 'procedural_min_session_turns', 2)
    with SessionLocal() as db:
        turns = seed_turns(db, cid, sid)
        try:
            with pytest.raises(IncompleteCompletion):
                worker.extract_procedural(str(turns[-1].batch_id), 'cloud-answer-model')
            assert calls[0]['model'] == 'local-background'
            key = job_key('procedural', f'{sid}:{len(turns) // max(1, settings.procedural_session_step)}')
            assert db.query(IdempotencyKey).filter_by(key=key).first() is None
        finally:
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()


def test_batch_partial_output_does_not_mark_source_covered(monkeypatch):
    from src.workers import batch_summarizer as worker
    cid, sid = uuid.uuid4(), uuid.uuid4()
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response('A plausible unfinished batch summary.', 'length')
    monkeypatch.setattr(worker, 'bg_client', client_for(create))
    monkeypatch.setattr(worker, 'get_bg_model_name', lambda: 'local-background')
    with SessionLocal() as db:
        seed_turns(db, cid, sid)
        try:
            worker.batch_summarize()
            db.expire_all()
            assert calls
            assert db.query(BatchSummary).filter_by(conversation_id=cid).count() == 0
            assert all(t.batch_summary_id is None for t in
                       db.query(EpisodicMemory).filter_by(conversation_id=cid))
        finally:
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()


def test_failed_conversation_completion_preserves_previous_checkpoint(monkeypatch):
    from src.memory.models import ConversationSummary
    from src.workers import bg_client_factory, conversation_summary as worker
    cid, sid = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda:
        client_for(lambda **kw: response('Plausible incomplete replacement.', 'length')))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'local-background')
    monkeypatch.setattr(worker, 'extract_key_terms', lambda *a: dict(entities=[], figures=[], identifiers=[]))
    with SessionLocal() as db:
        turns = seed_turns(db, cid, sid)
        checkpoint = turns[0].timestamp
        db.add(ConversationSummary(conversation_id=cid, summary_text='Previous checkpoint.',
                                   covers_through=checkpoint, covers_turns=1))
        db.commit()
        try:
            with pytest.raises(IncompleteCompletion):
                worker.run_conversation_summaries(db, embedder=NS(), conversation_ids=[cid])
            db.rollback()
            row = db.query(ConversationSummary).filter_by(conversation_id=cid).one()
            assert row.summary_text == 'Previous checkpoint.'
            assert row.covers_through == checkpoint and row.covers_turns == 1
        finally:
            db.query(ConversationSummary).filter_by(conversation_id=cid).delete()
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()
