"""v3 rolling-summary freshness through writer and both actual SQL readers."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal, engine
from src.api.prompt_assembler import conversation_summary_block
from src.memory.models import Conversation, ConversationSummary, EpisodicMemory
from src.memory.summary_snapshot import bind_snapshot, source_snapshot, snapshot_matches
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import conversation_summary as worker

assert (os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_') and
        make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE'])
VEC = [1.0] + [0.0] * 1023


def test_snapshot_migration_roundtrip():
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, 'a6d4e902b173')
    command.downgrade(cfg, 'f3c8d5e02b19')
    assert 'source_manifest' not in {c['name'] for c in inspect(engine).get_columns('conversation_summaries')}
    command.upgrade(cfg, 'a6d4e902b173')


@pytest.fixture
def context(monkeypatch):
    cid, other = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(worker, 'estimate_recent_window_tokens', lambda *a: 0)
    monkeypatch.setattr(worker, 'extract_key_terms', lambda *a:
                        dict(entities=[], figures=[], identifiers=[]))
    db = SessionLocal()
    db.add_all([Conversation(id=cid), Conversation(id=other)]); db.commit()
    base = datetime.now(timezone.utc) - timedelta(days=2)
    def add(body, minutes):
        row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(),
            timestamp=base + timedelta(minutes=minutes), raw_text=body, inject_raw=True,
            context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4()))
        db.add(row); db.commit()
        return row
    calls = []
    def llm(prompt, **kwargs):
        calls.append(prompt)
        return f'Controlled snapshot number {len(calls)}.'
    def run():
        return worker.run_conversation_summaries(db, llm=llm,
            embedder=NS(encode=lambda *a, **k: VEC), conversation_ids=[cid])
    first = add('Atlas used port 8391.', 0)
    second = add('Atlas changed the port to 8392.', 10)
    run()
    try:
        yield NS(db=db, cid=cid, other=other, add=add, calls=calls, run=run,
                 first=first, second=second)
    finally:
        db.rollback()
        db.query(ConversationSummary).filter_by(conversation_id=cid).delete()
        db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
        db.query(Conversation).filter(Conversation.id.in_([cid, other])).delete()
        db.commit(); db.close()


def assert_readable(ctx, expected):
    own = conversation_summary_block(ctx.db, str(ctx.cid), 3, 10000, 1)
    cross = HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(
        VEC, str(ctx.other))
    assert bool(own) == expected
    assert bool(cross) == expected


def test_unchanged_snapshot_skips_generation_and_both_readers_accept(context):
    ctx = context
    assert_readable(ctx, True)
    count = len(ctx.calls)
    ctx.run()
    assert len(ctx.calls) == count


@pytest.mark.parametrize('change', ['edit', 'delete', 'backfill', 'output', 'representation', 'support'])
def test_changes_block_both_readers_and_rebuild_without_old_generated_text(context, change):
    ctx = context
    old = ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one()
    previous = old.summary_text
    if change == 'edit': ctx.first.raw_text = 'Atlas used port 9000.'
    elif change == 'delete': ctx.db.delete(ctx.first)
    elif change == 'backfill': ctx.add('An earlier source imported afterward.', -10)
    elif change == 'output': old.summary_text = 'Edited output without generation.'
    elif change == 'support': ctx.first.representation_verification = {'summary': {'status': 'unknown'}}
    else: ctx.first.summary_text = 'Changed stored representation.'
    ctx.db.commit()
    assert_readable(ctx, False)
    start = len(ctx.calls)
    result = ctx.run()
    assert result['updated'] == 1
    assert all(previous not in prompt for prompt in ctx.calls[start:])
    assert_readable(ctx, True)


def test_strictly_newer_append_keeps_dated_snapshot_and_updates_incrementally(context):
    ctx = context
    previous = ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one().summary_text
    ctx.add('A later event.', 20)
    assert_readable(ctx, True)
    start = len(ctx.calls)
    assert ctx.run()['updated'] == 1
    assert any(previous in prompt for prompt in ctx.calls[start:])
    assert all('Atlas used port 8391.' not in prompt for prompt in ctx.calls[start:])


def test_legacy_manifest_is_unknown_until_regenerated(context):
    ctx = context
    row = ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one()
    row.source_manifest = None; ctx.db.commit()
    assert_readable(ctx, False)
    ctx.run()
    assert_readable(ctx, True)


def test_source_manifest_is_output_bound_and_does_not_contain_source_text(context):
    ctx = context
    current = source_snapshot(ctx.db, ctx.cid)
    record = bind_snapshot(current, 'Snapshot.')
    assert snapshot_matches(record, 'Snapshot.', current)
    assert not snapshot_matches(record, 'Different snapshot.', current)
    assert 'Atlas' not in str(current)


def test_changed_representation_policy_invalidates_cached_fold(context, monkeypatch):
    ctx = context
    monkeypatch.setattr(settings, 'source_support_threshold', .99)
    assert_readable(ctx, False)
    ctx.run()
    assert_readable(ctx, True)


def test_actual_fold_writer_receives_complete_late_correction(context):
    ctx = context
    full = 'Discuss the alternatives. ' * 800 + 'Final decision: do not deploy Redis.'
    ctx.first.raw_text = full
    ctx.first.inject_raw = False
    ctx.first.summary_text = 'Deploy Redis.'
    ctx.first.summary_coverage = 1.0
    ctx.db.commit()
    start = len(ctx.calls)
    assert ctx.run()['updated'] == 1
    assert any(full in prompt for prompt in ctx.calls[start:])
    assert all('Deploy Redis.' not in prompt for prompt in ctx.calls[start:])
