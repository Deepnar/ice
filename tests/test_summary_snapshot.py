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
from src.memory.source import single_provenance
from src.memory.support import verify_support
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
    db = SessionLocal()
    db.add_all([Conversation(id=cid), Conversation(id=other)]); db.commit()
    base = datetime.now(timezone.utc) - timedelta(days=2)
    def add(body, minutes):
        row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(),
            timestamp=base + timedelta(minutes=minutes), raw_text=body, inject_raw=True,
            source_spans=single_provenance(body, 'user'),
            context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4()))
        db.add(row); db.commit()
        return row
    calls = []
    def llm(prompt, **kwargs):
        calls.append(prompt)
        return f'Controlled snapshot number {len(calls)}.'
    def run():
        return worker.run_conversation_summaries(db, llm=llm,
            embedder=NS(encode=lambda *a, **k: VEC), conversation_ids=[cid],
            verifier=lambda p, h: verify_support(p, h, scorer=lambda pairs:
                [dict(entailment=.99, neutral=.005, contradiction=.005)]))
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
    assert all(previous not in prompt for prompt in ctx.calls[start:])
    assert previous in ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one().summary_text
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


def test_rejected_generation_reaches_both_readers_as_original_evidence(context):
    ctx = context
    raw = 'Atlas must not deploy the database. The cancellation remains final.'
    turn = ctx.add(raw, 20)
    result = worker.run_conversation_summaries(ctx.db,
        llm=lambda *a, **k: 'Atlas must deploy the database.',
        embedder=NS(encode=lambda *a, **k: VEC), conversation_ids=[ctx.cid],
        verifier=lambda p, h: verify_support(p, h, scorer=lambda pairs:
            [dict(entailment=.001, neutral=.009, contradiction=.99)]))
    assert result['updated'] == 1
    row = ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one()
    part = row.source_manifest['parts'][-1]
    assert part['mode'] == 'source' and part['source_ids'] == [str(turn.id)]
    assert raw in row.summary_text
    assert 'Atlas must deploy the database.' not in row.summary_text
    own = conversation_summary_block(ctx.db, str(ctx.cid), 3, 10000, 1)
    cross = HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(VEC, str(ctx.other))
    assert raw in own and any(raw in f.text for f in cross)


def test_later_group_failure_does_not_half_advance_or_feed_generated_note(context, monkeypatch):
    ctx = context
    row = ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one()
    previous = row.summary_text
    previous_manifest = row.source_manifest
    ctx.add('Third independent source group.', 20)
    ctx.add('Fourth independent source group.', 30)
    monkeypatch.setattr(settings, 'conversation_summary_chunk_words', 1)
    calls = []
    def llm(prompt, **kwargs):
        calls.append(prompt)
        return 'An invented intermediate note.' if len(calls) == 1 else ''
    result = worker.run_conversation_summaries(ctx.db, llm=llm,
        embedder=NS(encode=lambda *a, **k: VEC), conversation_ids=[ctx.cid],
        verifier=lambda p, h: verify_support(p, h, scorer=lambda pairs:
            [dict(entailment=.99, neutral=.005, contradiction=.005)]))
    assert result['failed'] == 1 and len(calls) == 2
    assert all('An invented intermediate note.' not in p and previous not in p for p in calls)
    ctx.db.expire_all()
    assert row.summary_text == previous and row.source_manifest == previous_manifest


def test_changed_cached_part_cannot_be_reused_or_read(context):
    ctx = context
    import copy
    row = ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one()
    manifest = copy.deepcopy(row.source_manifest)
    manifest['parts'][0]['text'] = 'A planted cached invention.'
    row.source_manifest = manifest
    ctx.db.commit()
    assert_readable(ctx, False)
    start = len(ctx.calls)
    ctx.run()
    assert all('A planted cached invention.' not in p for p in ctx.calls[start:])
    assert_readable(ctx, True)


def test_private_turn_in_public_conversation_stays_own_scope(context):
    ctx = context
    assert_readable(ctx, True)
    ctx.first.is_private = True
    ctx.db.commit()
    ctx.run()  # Current valid manifest: privacy, not staleness, must deny cross read.
    assert conversation_summary_block(ctx.db, str(ctx.cid), 3, 10000, 1)
    assert not HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(
        VEC, str(ctx.other))


@pytest.mark.parametrize('scope_kind', ['empty', 'other', 'excluded', 'allowed'])
def test_cross_summary_obeys_resolved_conversation_scope_and_has_source_credit(context, scope_kind):
    ctx = context
    scope = {'empty': {'conversation_ids': []},
             'other': {'conversation_ids': [ctx.other]},
             'excluded': {'exclude_conversation_ids': [ctx.cid]},
             'allowed': {'conversation_ids': [ctx.cid]}}[scope_kind]
    fragments = HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(
        VEC, str(ctx.other), scope=scope)
    assert bool(fragments) == (scope_kind == 'allowed')
    if fragments:
        assert fragments[0].conversation_id == str(ctx.cid)
        assert set(fragments[0].origin_batch_ids) == {str(ctx.first.batch_id), str(ctx.second.batch_id)}


def test_search_conversation_identity_is_not_active_conversation_identity(context):
    ctx = context
    orch = HybridRetrievalOrchestrator(ctx.db, NS())
    assert orch._batch_summary_lookup(VEC, str(ctx.other), search_conv_id=str(ctx.cid))
    assert not orch._batch_summary_lookup(VEC, str(ctx.other), search_conv_id=str(ctx.other))


def test_summary_cannot_disclose_one_excluded_cluster_source(context):
    from src.memory.models import ContextCluster, EpisodicClusterLink
    ctx = context
    a, b = uuid.uuid4(), uuid.uuid4()
    ctx.db.add_all([ContextCluster(id=a, name='Allowed synthetic group'),
                    ContextCluster(id=b, name='Excluded synthetic group')])
    ctx.db.flush()
    ctx.db.add_all([EpisodicClusterLink(episodic_id=ctx.first.id, cluster_id=a),
                    EpisodicClusterLink(episodic_id=ctx.second.id, cluster_id=b)])
    ctx.db.commit()
    orch = HybridRetrievalOrchestrator(ctx.db, NS())
    try:
        assert orch._batch_summary_lookup(VEC, str(ctx.other), scope={'cluster_ids': [str(a), str(b)]})
        assert not orch._batch_summary_lookup(VEC, str(ctx.other), scope={'cluster_ids': [str(a)]})
        assert not orch._batch_summary_lookup(VEC, str(ctx.other), scope={'exclude_cluster_ids': [str(b)]})
    finally:
        ctx.db.query(EpisodicClusterLink).filter(EpisodicClusterLink.cluster_id.in_([a,b])).delete()
        ctx.db.query(ContextCluster).filter(ContextCluster.id.in_([a,b])).delete()
        ctx.db.commit()


def test_own_batch_summary_obeys_scope_and_cannot_survive_without_sources(context):
    from src.memory.models import BatchSummary
    ctx = context
    batch = BatchSummary(conversation_id=ctx.cid, summary_text='Synthetic batch note.',
                         start_turn_index=0, end_turn_index=1, embedding=VEC)
    ctx.db.add(batch); ctx.db.flush()
    ctx.first.batch_summary_id = batch.id
    ctx.db.flush()
    from src.memory.summary_snapshot import compose_parts
    parts = [{'source_ids': [str(ctx.first.id)], 'mode': 'source', 'text': ctx.first.raw_text}]
    batch.summary_text = compose_parts(parts)
    batch.source_manifest = bind_snapshot(source_snapshot(ctx.db, ctx.cid,
        batch_summary_id=batch.id), batch.summary_text, parts=parts)
    ctx.db.commit()
    orch = HybridRetrievalOrchestrator(ctx.db, NS())
    try:
        fragments = orch._batch_summary_lookup(VEC, str(ctx.cid), include_cross=False)
        assert fragments and fragments[0].origin_batch_ids == (str(ctx.first.batch_id),)
        assert not orch._batch_summary_lookup(VEC, str(ctx.cid), include_cross=False,
                                              scope={'conversation_ids': []})
        assert not orch._batch_summary_lookup(VEC, str(ctx.cid), include_cross=False,
                                              scope={'exclude_conversation_ids': [ctx.cid]})
        ctx.first.batch_summary_id = None; ctx.db.commit()
        assert not orch._batch_summary_lookup(VEC, str(ctx.cid), include_cross=False)
    finally:
        ctx.first.batch_summary_id = None
        ctx.db.flush()
        ctx.db.delete(batch); ctx.db.commit()


@pytest.mark.parametrize('subset', ['all', 'partial', 'empty'])
def test_aggregate_requires_every_source_in_explicit_batch_scope(context, subset):
    ctx = context
    batches = {'all': [ctx.first.batch_id, ctx.second.batch_id],
               'partial': [ctx.first.batch_id], 'empty': []}[subset]
    result = HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(
        VEC, str(ctx.other), scope={'batch_ids': batches})
    assert bool(result) == (subset == 'all')


def test_full_retrieve_passes_resolved_scope_to_real_summary_reader(context, monkeypatch):
    ctx = context
    orch = HybridRetrievalOrchestrator(ctx.db, NS())
    for name in ('_codex_graph', '_codex_claims', '_relevant_cluster_ids',
                 '_bm25_episodic', '_vector_episodic', '_procedural_lookup', '_cold_lookup'):
        monkeypatch.setattr(orch, name, lambda *a, **kw: [])
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', False)
    monkeypatch.setattr(orch, '_apply_bonuses', lambda f, *a: f)
    classification = NS(prompt='What changed in Atlas?', context_reliance='Long_Term_Memory',
                        max_confidence=1., intent_tags=[], topic_tags=[])
    orch.max_retrieval_tokens = 10000
    allowed = orch.retrieve(classification, str(ctx.other), VEC,
                             scope={'conversation_ids': [str(ctx.cid)]})
    denied = orch.retrieve(classification, str(ctx.other), VEC,
                            scope={'conversation_ids': []})
    assert allowed and not denied
    assert set(allowed[0].origin_batch_ids) == {str(ctx.first.batch_id), str(ctx.second.batch_id)}
