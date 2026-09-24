"""v3 batch writer and actual SQL reader share source/support freshness."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import BatchSummary, ColdStorage, Conversation, EpisodicMemory
from src.memory.source import single_provenance
from src.memory.support import verify_support
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import batch_summarizer as worker
from src.workers.completion_text import IncompleteCompletion
from src.workers.decay import apply_decay

assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
assert make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE']
VEC = [1.] + [0.] * 1023


def test_batch_manifest_migration_roundtrip():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    from src.api.db import engine
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, 'b7e5f013c284')
    command.downgrade(cfg, 'a6d4e902b173')
    assert 'source_manifest' not in {c['name'] for c in inspect(engine).get_columns('batch_summaries')}
    command.upgrade(cfg, 'b7e5f013c284')
    assert 'source_manifest' in {c['name'] for c in inspect(engine).get_columns('batch_summaries')}


def test_cold_batch_eligibility_migration_roundtrip():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    from src.api.db import engine
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, 'c742a88eeb19')
    command.downgrade(cfg, '77b3d428e591')
    assert not {'is_document', 'decay_score'} & {
        col['name'] for col in inspect(engine).get_columns('cold_storage')}
    command.upgrade(cfg, 'c742a88eeb19')
    assert {'is_document', 'decay_score'} <= {
        col['name'] for col in inspect(engine).get_columns('cold_storage')}


@pytest.fixture
def batch_context(monkeypatch):
    cid = uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        for i in range(5):
            raw = f'Atlas port is 8391. Source observation {i}.'
            db.add(EpisodicMemory(conversation_id=cid, raw_text=raw,
                source_spans=single_provenance(raw, 'user'),
                timestamp=datetime.now(timezone.utc) - timedelta(days=100, minutes=i),
                batch_id=uuid.uuid4(), idempotency_key=str(uuid.uuid4()),
                context_reliance='Long_Term_Memory', lossless_flag=False))
        db.commit()
    calls = []
    def generate(prompt, **kwargs):
        calls.append(prompt)
        return 'The user said Atlas uses port 8391.'
    monkeypatch.setattr(worker, '_batch_llm', generate)
    monkeypatch.setattr(worker, 'embedder', NS(encode=lambda *a, **k: VEC))
    monkeypatch.setattr(worker, 'verify_support', lambda p, h: verify_support(p, h,
        scorer=lambda pairs: [dict(entailment=.99, neutral=.005, contradiction=.005)]))
    try:
        yield NS(cid=cid, calls=calls)
    finally:
        with SessionLocal() as db:
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
            db.query(ColdStorage).filter_by(conversation_id=cid).delete()
            db.query(BatchSummary).filter_by(conversation_id=cid).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()


def read(cid):
    with SessionLocal() as db:
        return HybridRetrievalOrchestrator(db, NS())._batch_summary_lookup(
            VEC, str(cid), include_cross=False)


def test_supported_writer_reader_and_idempotence(batch_context):
    ctx = batch_context
    worker.batch_summarize()
    hits = read(ctx.cid)
    assert len(hits) == 1 and 'port 8391' in hits[0].text
    assert len(hits[0].origin_batch_ids) == 5
    assert 'The user said: "Atlas port is 8391.' in ctx.calls[0]
    before = len(ctx.calls)
    worker.batch_summarize()
    assert len(ctx.calls) == before


@pytest.mark.parametrize('cold_count', [3, 5])
def test_new_batch_uses_mixed_or_all_cold_originals(batch_context, cold_count):
    ctx = batch_context
    with SessionLocal() as db:
        rows = db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid).order_by(
            EpisodicMemory.timestamp).all()
        for row in rows[:cold_count]:
            row.decay_score = .001
            row.is_archived = True
        db.commit()
    apply_decay()
    with SessionLocal() as db:
        archived = db.query(ColdStorage).filter_by(conversation_id=ctx.cid).all()
        assert len(archived) == cold_count
        assert all(row.is_document is False and row.decay_score is not None
                   for row in archived)
    worker.batch_summarize()
    hits = read(ctx.cid)
    assert len(hits) == 1 and len(hits[0].origin_batch_ids) == 5
    with SessionLocal() as db:
        warm = db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid).all()
        cold = db.query(ColdStorage).filter_by(conversation_id=ctx.cid).all()
        assert {row.batch_summary_id for row in warm + cold} == {
            db.query(BatchSummary).filter_by(conversation_id=ctx.cid).one().id}
    prior_calls = len(ctx.calls)
    worker.batch_summarize()
    assert len(ctx.calls) == prior_calls


@pytest.mark.parametrize('field,value', [('is_document', None), ('is_document', True),
                                         ('lossless_flag', None), ('is_private', True)])
def test_unknown_or_ineligible_cold_source_is_not_compressed(batch_context, field, value):
    ctx = batch_context
    with SessionLocal() as db:
        for row in db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid):
            row.decay_score = .001
            row.is_archived = True
        db.commit()
    apply_decay()
    with SessionLocal() as db:
        row = db.query(ColdStorage).filter_by(conversation_id=ctx.cid).first()
        setattr(row, field, value)
        db.commit()
    worker.batch_summarize()
    assert not read(ctx.cid)
    with SessionLocal() as db:
        assert all(row.batch_summary_id is None for row in db.query(ColdStorage).filter_by(
            conversation_id=ctx.cid))


def test_cold_edit_during_generation_cannot_earn_current_coverage(batch_context, monkeypatch):
    ctx = batch_context
    with SessionLocal() as db:
        for row in db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid):
            row.decay_score = .001
            row.is_archived = True
        db.commit()
    apply_decay()
    def edit(*a, **k):
        with SessionLocal() as db:
            row = db.query(ColdStorage).filter_by(conversation_id=ctx.cid).first()
            row.raw_text += ' Concurrent cold correction.'
            db.commit()
        return 'The user said Atlas uses port 8391.'
    monkeypatch.setattr(worker, '_batch_llm', edit)
    worker.batch_summarize()
    assert not read(ctx.cid)
    with SessionLocal() as db:
        assert all(row.batch_summary_id is None for row in db.query(ColdStorage).filter_by(
            conversation_id=ctx.cid))


def test_unsupported_compression_keeps_original_evidence(batch_context, monkeypatch):
    monkeypatch.setattr(worker, '_batch_llm', lambda *a, **k: 'Atlas uses port 9999.')
    monkeypatch.setattr(worker, 'verify_support', lambda p, h: verify_support(p, h,
        scorer=lambda pairs: [dict(entailment=.001, neutral=.009, contradiction=.99)]))
    worker.batch_summarize()
    hits = read(batch_context.cid)
    assert len(hits) == 1 and '9999' not in hits[0].text
    for i in range(5):
        assert f'Source observation {i}.' in hits[0].text


@pytest.mark.parametrize('change', ['source', 'delete', 'output', 'legacy', 'policy',
                                    'private', 'document', 'lossless'])
def test_changed_evidence_invalidates_read_and_rebuilds(batch_context, change):
    ctx = batch_context
    worker.batch_summarize()
    with SessionLocal() as db:
        summary = db.query(BatchSummary).filter_by(conversation_id=ctx.cid).one()
        turn = db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid).first()
        if change == 'source': turn.raw_text += ' Later correction: use port 8392.'
        elif change == 'delete': db.delete(turn)
        elif change == 'output': summary.summary_text += ' Unsupported addition.'
        elif change == 'legacy': summary.source_manifest = None
        elif change == 'private': turn.is_private = True
        elif change == 'document': turn.is_document = True
        elif change == 'lossless': turn.lossless_flag = True
        else:
            summary.source_manifest = {**summary.source_manifest, 'representation_policy': {}}
        db.commit()
    assert not read(ctx.cid)
    worker.batch_summarize()
    # Four remaining eligible turns intentionally remain below the batch floor.
    assert bool(read(ctx.cid)) == (change not in ('delete', 'private', 'document', 'lossless'))


def test_incomplete_generation_does_not_mark_sources(batch_context, monkeypatch):
    def incomplete(*a, **k):
        raise IncompleteCompletion('controlled incomplete response')
    monkeypatch.setattr(worker, '_batch_llm', incomplete)
    worker.batch_summarize()
    assert not read(batch_context.cid)
    with SessionLocal() as db:
        assert all(t.batch_summary_id is None for t in db.query(EpisodicMemory).filter_by(
            conversation_id=batch_context.cid))


def test_edit_during_generation_cannot_be_bound_as_fresh(batch_context, monkeypatch):
    def edit(*a, **k):
        with SessionLocal() as db:
            row = db.query(EpisodicMemory).filter_by(conversation_id=batch_context.cid).first()
            row.raw_text += ' Concurrent correction.'
            db.commit()
        return 'The user said Atlas uses port 8391.'
    monkeypatch.setattr(worker, '_batch_llm', edit)
    worker.batch_summarize()
    assert not read(batch_context.cid)


def test_batch_provider_refuses_complete_request_over_capacity(monkeypatch):
    monkeypatch.setattr(settings, 'ollama_num_ctx_max', 64)
    monkeypatch.setattr(settings, 'background_model_mode', 'dedicated')
    monkeypatch.setattr(worker, 'get_bg_model_name', lambda: 'controlled-model')
    def forbidden(**kwargs):
        pytest.fail('over-capacity input reached provider')
    monkeypatch.setattr(worker, 'bg_client', NS(chat=NS(completions=NS(create=forbidden))))
    with pytest.raises(IncompleteCompletion):
        worker._batch_llm('complete original evidence ' * 100, max_tokens=32)


def test_batch_provider_receives_entire_source_and_configured_output(monkeypatch):
    monkeypatch.setattr(settings, 'ollama_num_ctx_max', 32768)
    monkeypatch.setattr(settings, 'background_model_mode', 'dedicated')
    monkeypatch.setattr(worker, 'get_bg_model_name', lambda: 'controlled-model')
    source = 'Complete original evidence. ' * 250 + 'Final correction: port 8392.'
    def provider(**kwargs):
        assert kwargs['messages'][-1]['content'] == source
        assert kwargs['max_tokens'] == 777
        return NS(choices=[NS(finish_reason='stop', message=NS(content='Supported note.'))])
    monkeypatch.setattr(worker, 'bg_client', NS(chat=NS(completions=NS(create=provider))))
    assert worker._batch_llm(source, max_tokens=777) == 'Supported note.'
