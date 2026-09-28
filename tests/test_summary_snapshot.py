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
from src.api.prompt_assembler import assemble_prompt, conversation_summary_block
from src.memory.conversation_notes import render_note, select_note_options
from src.memory.models import Conversation, ConversationNote, ConversationSummary, EpisodicMemory
from src.memory.source import single_provenance
from src.memory.support import verify_support
from src.memory.tokens import count as count_tokens
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


def test_query_selects_late_note_and_credits_only_its_source_in_final_prompt(context):
    ctx = context
    new_turn = ctx.add('The port changed again: use 9010.', 20)
    def vector(value, **kwargs):
        return ([0.0, 1.0] + [0.0] * 1022 if '9010' in value else VEC)
    result = worker.run_conversation_summaries(ctx.db,
        llm=lambda *a, **k: 'The user changed the port to 9010.',
        embedder=NS(encode=vector), conversation_ids=[ctx.cid],
        verifier=lambda p, h: verify_support(p, h, scorer=lambda pairs:
            [dict(entailment=.99, neutral=.005, contradiction=.005)]))
    assert result['updated'] == 1
    notes = ctx.db.query(ConversationNote).filter_by(conversation_id=ctx.cid).order_by(
        ConversationNote.ordinal).all()
    assert len(notes) == 2
    allowance = count_tokens(render_note(notes[-1])) + 2
    own = conversation_summary_block(ctx.db, str(ctx.cid), 3, 10000, 1,
                                     [0.0, 1.0] + [0.0] * 1022, allowance)
    assert '9010' in own and 'Controlled snapshot' not in own
    cross = HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(
        [0.0, 1.0] + [0.0] * 1022, str(ctx.other))
    assert cross and '9010' in cross[0].text
    assert cross[0].origin_batch_ids == (str(new_turn.batch_id),)
    messages = assemble_prompt(memory_slots=[], retrieved_fragments=[],
        user_message='Which port now?', conversation_summary_text=own,
        max_recent_tokens=0)
    assert '9010' in messages[0]['content']
    assert 'Controlled snapshot' not in messages[0]['content']


def test_complete_source_note_is_skipped_if_it_cannot_fit(context):
    ctx = context
    ctx.add('A later small correction.', 20)
    ctx.run()
    notes = ctx.db.query(ConversationNote).filter_by(conversation_id=ctx.cid).order_by(
        ConversationNote.ordinal).all()
    notes[0].text = 'Complete original evidence. ' * 500
    notes[0].mode = 'source'
    selected = select_note_options(notes, count_tokens(render_note(notes[-1])) + 2,
                                   {notes[0].ordinal: 1.0, notes[-1].ordinal: .5})[0]
    assert notes[-1].text in selected and notes[0].text not in selected


def test_unknown_group_keeps_complete_turns_and_selects_the_small_source(context):
    ctx = context
    large = 'The user described background context. ' * 550
    correction = 'The user settled on the violet 742 protocol.'
    ctx.first.raw_text = large
    ctx.first.source_spans = single_provenance(large, 'user')
    ctx.second.raw_text = correction
    ctx.second.source_spans = single_provenance(correction, 'user')
    ctx.db.commit()
    result = worker.run_conversation_summaries(ctx.db,
        llm=lambda *a, **k: 'An unsupported short note.',
        embedder=NS(encode=lambda *a, **k: VEC), conversation_ids=[ctx.cid],
        verifier=lambda p, h: verify_support(p, h, scorer=lambda pairs:
            [dict(entailment=.01, neutral=.98, contradiction=.01)]))
    assert result['updated'] == 1
    row = ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one()
    notes = ctx.db.query(ConversationNote).filter_by(conversation_id=ctx.cid).order_by(
        ConversationNote.ordinal).all()
    assert len(notes) == 2
    assert [n.source_ids for n in notes] == [[str(ctx.first.id)], [str(ctx.second.id)]]
    assert all(n.mode == 'source' for n in notes)
    assert large in row.summary_text and correction in row.summary_text
    assert 'An unsupported short note.' not in row.summary_text
    own = conversation_summary_block(ctx.db, str(ctx.cid), 3, 10000, 1, VEC, 650)
    assert correction in own and large not in own
    cross = HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(
        VEC, str(ctx.other))
    assert any(correction in f.text and f.origin_batch_ids == (str(ctx.second.batch_id),)
               for f in cross)


def test_valid_legacy_grouped_fallback_is_rebuilt_from_originals(context):
    from src.memory.summary_snapshot import compose_parts
    ctx = context
    row = ctx.db.query(ConversationSummary).filter_by(conversation_id=ctx.cid).one()
    source = '\n\n'.join(worker._original_source(t)[0]
                         for t in (ctx.first, ctx.second))
    old_part = {'source_ids': [str(ctx.first.id), str(ctx.second.id)],
                'mode': 'source', 'text': source, 'recorded_range': 'old group'}
    row.summary_text = compose_parts([old_part])
    row.source_manifest = bind_snapshot(source_snapshot(ctx.db, ctx.cid),
                                        row.summary_text, parts=[old_part])
    ctx.db.commit()
    result = worker.run_conversation_summaries(ctx.db,
        llm=lambda *a, **k: 'An unsupported short note.',
        embedder=NS(encode=lambda *a, **k: VEC), conversation_ids=[ctx.cid],
        verifier=lambda p, h: verify_support(p, h, scorer=lambda pairs:
            [dict(entailment=.01, neutral=.98, contradiction=.01)]))
    assert result['updated'] == 1
    notes = ctx.db.query(ConversationNote).filter_by(conversation_id=ctx.cid).order_by(
        ConversationNote.ordinal).all()
    assert len(notes) == 2
    assert all(n.mode == 'source' and len(n.source_ids) == 1 for n in notes)


def test_corrupt_derived_note_index_falls_back_to_current_aggregate(context):
    ctx = context
    note = ctx.db.query(ConversationNote).filter_by(conversation_id=ctx.cid).one()
    note.text = 'Corrupt index text.'
    ctx.db.commit()
    own = conversation_summary_block(ctx.db, str(ctx.cid), 2, 10000, 1, VEC, 25)
    cross = HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(
        VEC, str(ctx.other))
    assert 'Controlled snapshot' in own
    assert cross and 'Controlled snapshot' in cross[0].text
    assert all('Corrupt index text.' not in fragment.text for fragment in cross)
    count = len(ctx.calls)
    ctx.run()
    assert len(ctx.calls) == count  # repair the index without regeneration
    assert ctx.db.query(ConversationNote).filter_by(conversation_id=ctx.cid).one().text != 'Corrupt index text.'


def test_conversation_deletion_previews_and_cascades_derived_notes(context):
    from src.services.conversations import delete_conversation
    ctx = context
    preview = delete_conversation(ctx.db, str(ctx.cid), dry_run=True)
    assert preview['deleted']['conversation_notes'] == 1
    delete_conversation(ctx.db, str(ctx.cid))
    assert not ctx.db.query(ConversationNote).filter_by(conversation_id=ctx.cid).all()


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


def test_active_original_note_dedup_restores_batch_part_after_eviction(context, monkeypatch):
    from src.workers import batch_summarizer as batch_worker
    from src.api.prompt_budget import assemble_budgeted_prompt
    from src.memory.models import BatchSummary
    ctx = context
    for i in range(3):
        ctx.add(f'Atlas observation {i}.', 20 + i)
    for row in ctx.db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid):
        row.lossless_flag = False
        row.is_document = False
        row.is_private = False
        row.decay_score = .1
    ctx.db.commit()
    bad = lambda p, h: verify_support(p, h, scorer=lambda pairs:
        [dict(entailment=.001, neutral=.009, contradiction=.99)])
    worker.run_conversation_summaries(ctx.db, llm=lambda *a, **k: 'Unsupported port 9999.',
        embedder=NS(encode=lambda *a, **k: VEC), conversation_ids=[ctx.cid], verifier=bad)
    monkeypatch.setattr(settings, 'batch_summary_age_days', 0)
    monkeypatch.setattr(batch_worker, '_batch_llm', lambda *a, **k: 'Unsupported port 9999.')
    monkeypatch.setattr(batch_worker, 'verify_support', bad)
    monkeypatch.setattr(batch_worker, 'embedder', NS(encode=lambda *a, **k: VEC))
    batch_worker.batch_summarize()
    try:
        choices = conversation_summary_block(ctx.db, str(ctx.cid), 5, 10000, 0,
            query_embedding=VEC, include_options=True, include_source_ids=True)
        assert choices
        active = choices[0][0]
        own = HybridRetrievalOrchestrator(ctx.db, NS())._batch_summary_lookup(
            VEC, str(ctx.cid), include_cross=False)
        repeated = [f for f in own if f.source_note_body
                    and f.source_note_body in active]
        assert len(repeated) == 1
        common = dict(generation_reserve=64, safety_margin=1., memory_slots=[],
            retrieved_fragments=own, user_message='What was the Atlas port?',
            db_session=ctx.db, conversation_id=str(ctx.cid), max_recent_tokens=0)
        large = assemble_budgeted_prompt(serving_window=10000,
            conversation_summary_text=active,
            conversation_summary_options=[text for text, _ in choices],
            conversation_summary_source_ids={text: ids for text, ids in choices},
            **common)
        assert repeated[0] not in large.visible_fragments
        assert sum(message['content'].count(repeated[0].source_note_body)
                   for message in large.messages) == 1
        from scripts.z1 import trace_v3_memory as trace
        from src.api import memory_preparation as mp
        monkeypatch.setattr(trace, 'get_model_context_window', lambda _name: 10000)
        monkeypatch.setattr(trace, 'serving_window', lambda *_args: 10000)
        monkeypatch.setattr(HybridRetrievalOrchestrator,
            'set_budget_from_turn_count',
            lambda self, *_a, **_k: setattr(self, 'recent_token_budget', 0))
        monkeypatch.setattr(HybridRetrievalOrchestrator, 'retrieve',
            lambda self, *_a, **_k: own)
        monkeypatch.setattr(HybridRetrievalOrchestrator, 'record_exposure',
            lambda self, *_a, **_k: None)
        monkeypatch.setattr(settings, 'memory_source_gate_enabled', True)
        monkeypatch.setattr(mp, 'judge_source_need', lambda *_a, **_k: object())
        monkeypatch.setattr(mp, 'source_action', lambda *_a, **_k: 'keep')
        pre = NS(turn_count=5, total_tokens=10000, total_budget=10000,
                 prompt_embedding=VEC, retrieve=True, model_name='fixture',
                 classification=NS(prompt='What was the Atlas port?',
                                   context_reliance='Long_Term_Memory'),
                 scope={}, conversation_id=str(ctx.cid))
        classifier = NS(embedder=NS(encode=lambda *_a, **_k: VEC))
        traced, fetched, _ = trace._prepare(ctx.db, pre, classifier, ctx.cid)
        assert traced.source_checked and traced.action == 'keep'
        assert repeated[0] in fetched
        assert repeated[0] not in traced.prepared.visible_fragments
        assert repeated[0] not in traced.fragments
        monkeypatch.setattr(mp, 'source_action', lambda *_a, **_k: 'skip')
        skipped, skipped_fetch, _ = trace._prepare(ctx.db, pre, classifier, ctx.cid)
        assert skipped.source_checked and not skipped.retrieve
        assert skipped_fetch == [] and skipped.fragments == []
        monkeypatch.setattr(mp, 'source_action', lambda *_a, **_k: 'keep')
        no_note, _, _ = trace._prepare(ctx.db, pre, classifier, ctx.cid,
                                      arm='no_summary')
        assert repeated[0] in no_note.fragments
        no_summary = assemble_budgeted_prompt(serving_window=10000,
            conversation_summary_text=None, **common)
        tight = assemble_budgeted_prompt(
            serving_window=no_summary.ledger.total() + 64,
            conversation_summary_text=active,
            conversation_summary_options=[active],
            conversation_summary_source_ids={active: choices[0][1]}, **common)
        assert 'conversation_summary' in tight.removed
        assert repeated[0] in tight.visible_fragments
    finally:
        ctx.db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid).update(
            {EpisodicMemory.batch_summary_id: None}, synchronize_session=False)
        ctx.db.query(BatchSummary).filter_by(conversation_id=ctx.cid).delete()
        ctx.db.commit()


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


def test_auto_retrieve_keeps_current_identity_without_narrowing_search(context, monkeypatch):
    ctx = context
    orch = HybridRetrievalOrchestrator(ctx.db, NS())
    for name in ('_codex_graph', '_codex_claims', '_relevant_cluster_ids',
                 '_bm25_episodic', '_vector_episodic', '_procedural_lookup', '_cold_lookup'):
        monkeypatch.setattr(orch, name, lambda *a, **kw: [])
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', False)
    seen = {}
    summary_lookup = orch._batch_summary_lookup

    def capture_summary(*args, **kwargs):
        seen['summary_current'] = args[1]
        seen['summary_search'] = kwargs['search_conv_id']
        return summary_lookup(*args, **kwargs)

    def capture_bonuses(fragments, classification, current_id, keywords):
        seen['recency_current'] = current_id
        return fragments

    monkeypatch.setattr(orch, '_batch_summary_lookup', capture_summary)
    monkeypatch.setattr(orch, '_apply_bonuses', capture_bonuses)
    classification = NS(prompt='What changed in Atlas?', context_reliance='Long_Term_Memory',
                        max_confidence=1., intent_tags=[], topic_tags=[])
    orch.max_retrieval_tokens = 10000
    orch.retrieve(classification, str(ctx.cid), VEC, scope={})
    assert seen == {'summary_current': str(ctx.cid), 'summary_search': None,
                    'recency_current': str(ctx.cid)}

    # The low-confidence path must also use current identity for recency,
    # while retaining the global auto-mode search filter.
    monkeypatch.setattr(orch, '_vector_chunks', lambda *a, **kw: [])
    scope_filter = orch._conv_scope_filter

    def capture_scope(scope, search_id):
        seen['wide_search'] = search_id
        return scope_filter(scope, search_id)

    monkeypatch.setattr(orch, '_conv_scope_filter', capture_scope)
    classification.max_confidence = 0.0
    orch.retrieve(classification, str(ctx.cid), VEC, scope={})
    assert seen['wide_search'] is None
    assert seen['recency_current'] == str(ctx.cid)
