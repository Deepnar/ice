"""v3 summary provenance survives storage moves, not source changes."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.api.prompt_assembler import conversation_summary_block
from src.memory.models import BatchSummary, ColdStorage, Conversation, ConversationSummary, EpisodicMemory
from src.memory.source import single_provenance
from src.memory.support import verify_support
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import batch_summarizer as batch_worker, conversation_summary as rolling
from src.workers.decay import apply_decay

assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
assert make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE']
VEC = [1.] + [0.] * 1023


@pytest.fixture
def source_set(monkeypatch):
    cid, other = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add_all([Conversation(id=cid), Conversation(id=other)]); db.flush()
        for i in range(5):
            raw = f'Atlas port is 8391. Source {i}.'
            db.add(EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(),
                raw_text=raw, source_spans=single_provenance(raw, 'user'), embedding=VEC,
                timestamp=datetime.now(timezone.utc) - timedelta(days=100, minutes=i),
                decay_score=.001, is_archived=True, lossless_flag=False,
                context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4())))
        db.commit()
    calls = []
    def llm(prompt, **kwargs):
        calls.append(prompt)
        return 'The user said Atlas uses port 8391.'
    verifier = lambda p,h: verify_support(p,h, scorer=lambda pairs:
        [dict(entailment=.99, neutral=.005, contradiction=.005)])
    encoder = NS(encode=lambda *a, **k: VEC)
    monkeypatch.setattr(batch_worker, '_batch_llm', llm)
    monkeypatch.setattr(batch_worker, 'verify_support', verifier)
    monkeypatch.setattr(batch_worker, 'embedder', encoder)
    monkeypatch.setattr(rolling, 'estimate_recent_window_tokens', lambda *a: 0)
    def roll():
        with SessionLocal() as db:
            return rolling.run_conversation_summaries(db, llm=llm, verifier=verifier,
                embedder=encoder, conversation_ids=[cid])
    batch_worker.batch_summarize(); roll()
    try:
        yield NS(cid=cid, other=other, calls=calls, roll=roll)
    finally:
        with SessionLocal() as db:
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
            db.query(ColdStorage).filter_by(conversation_id=cid).delete()
            db.query(BatchSummary).filter_by(conversation_id=cid).delete()
            db.query(ConversationSummary).filter_by(conversation_id=cid).delete()
            db.query(Conversation).filter(Conversation.id.in_([cid,other])).delete(); db.commit()


def readable(ctx):
    with SessionLocal() as db:
        orch = HybridRetrievalOrchestrator(db, NS())
        own = orch._batch_summary_lookup(VEC, str(ctx.cid), include_cross=False)
        cross = orch._batch_summary_lookup(VEC, str(ctx.other))
        active = conversation_summary_block(db, str(ctx.cid), 5, 10000, 1)
        return own, cross, active


@pytest.mark.parametrize('restore', [False, True])
def test_archive_preserves_both_summary_readers_without_regeneration(source_set, monkeypatch, restore):
    ctx = source_set
    before = readable(ctx)
    assert all(before)
    apply_decay()
    after = readable(ctx)
    assert all(after)
    assert before[0][0].text == after[0][0].text
    assert set(before[0][0].origin_batch_ids) == set(after[0][0].origin_batch_ids)
    assert before[1][0].text == after[1][0].text and before[2] == after[2]
    if restore:
        from src.retrieval.timescope import TimeScope
        with SessionLocal() as db:
            orch = HybridRetrievalOrchestrator(db, NS())
            now = datetime.now(timezone.utc)
            orch._active_timescope = TimeScope(mode='range',
                t0=now - timedelta(days=110), t1=now - timedelta(days=90))
            hits = orch._cold_lookup({'atlas'}, str(ctx.cid), prompt_embedding=VEC)
            monkeypatch.setattr(settings, 'retrieval_strengthen_writes', True)
            orch._resurrect_cold_hits(hits)
        assert all(readable(ctx))
    n = len(ctx.calls)
    batch_worker.batch_summarize(); ctx.roll()
    assert len(ctx.calls) == n


@pytest.mark.parametrize('change', ['edit', 'delete', 'private'])
def test_cold_source_change_invalidates_summaries(source_set, change):
    ctx = source_set
    apply_decay()
    with SessionLocal() as db:
        row = db.query(ColdStorage).filter_by(conversation_id=ctx.cid).first()
        if change == 'edit': row.raw_text += ' Correction: port 8392.'
        elif change == 'delete': db.delete(row)
        else: row.is_private = True
        db.commit()
    assert not any(readable(ctx))
    if change == 'edit':
        ctx.roll()
        own, cross, active = readable(ctx)
        assert not own and cross and active
        assert 'Correction: port 8392.' in active
        batch_worker.batch_summarize()
        own, cross, active = readable(ctx)
        assert own and cross and active
        assert 'Correction: port 8392.' in own[0].text


def test_cold_membership_constraints_apply_to_summary_sources(source_set):
    ctx = source_set
    apply_decay()
    cluster = uuid.uuid4()
    with SessionLocal() as db:
        rows = db.query(ColdStorage).filter_by(conversation_id=ctx.cid).all()
        rows[0].cluster_ids = [cluster]; db.commit()
        orch = HybridRetrievalOrchestrator(db, NS())
        assert not orch._batch_summary_lookup(VEC, str(ctx.cid), include_cross=False,
                                              scope={'exclude_cluster_ids': [cluster]})
        assert not orch._batch_summary_lookup(VEC, str(ctx.other),
                                              scope={'exclude_cluster_ids': [cluster]})
        assert orch._batch_summary_lookup(VEC, str(ctx.other), scope={'cluster_ids': [cluster]})


def test_tier_coverage_migration_roundtrip():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    from src.api.db import engine
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, 'fb29355706c8')
    command.downgrade(cfg, 'ea182446f5b7')
    assert 'batch_summary_id' not in {c['name'] for c in inspect(engine).get_columns('cold_storage')}
    command.upgrade(cfg, 'fb29355706c8')


def test_conversation_note_index_migration_roundtrip():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    from src.api.db import engine
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, '77b3d428e591')
    command.downgrade(cfg, 'fb29355706c8')
    assert 'conversation_notes' not in inspect(engine).get_table_names()
    command.upgrade(cfg, '77b3d428e591')
    inspector = inspect(engine)
    assert 'conversation_notes' in inspector.get_table_names()
    assert 'idx_conversation_notes_embedding' in {
        item['name'] for item in inspector.get_indexes('conversation_notes')}
