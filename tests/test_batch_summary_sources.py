"""v3 batch writer and actual SQL reader share source/support freshness."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import (BatchNote, BatchSummary, ColdStorage, ContextCluster,
                               Conversation, EpisodicClusterLink, EpisodicMemory)
from src.memory.source import single_provenance
from src.memory.support import verify_support
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator
from src.workers import batch_summarizer as worker
from src.workers.completion_text import IncompleteCompletion
from src.workers.decay import apply_decay

assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
assert make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE']
VEC = [1.] + [0.] * 1023


def test_batch_note_migration_roundtrip():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    from src.api.db import engine
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, '80f1ac593d72')
    command.downgrade(cfg, '7e4c9d2a0b65')
    assert 'batch_notes' not in inspect(engine).get_table_names()
    command.upgrade(cfg, '80f1ac593d72')
    assert 'idx_batch_notes_embedding' in {
        index['name'] for index in inspect(engine).get_indexes('batch_notes')}


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
    monkeypatch.setattr(settings, 'retrieval_batch_summary_limit', 5)
    monkeypatch.setattr(worker, '_batch_llm', lambda *a, **k: 'Atlas uses port 9999.')
    monkeypatch.setattr(worker, 'verify_support', lambda p, h: verify_support(p, h,
        scorer=lambda pairs: [dict(entailment=.001, neutral=.009, contradiction=.99)]))
    worker.batch_summarize()
    hits = read(batch_context.cid)
    assert len(hits) == 5 and all('9999' not in hit.text for hit in hits)
    for i in range(5):
        assert any(f'Source observation {i}.' in hit.text for hit in hits)
    assert all(len(hit.origin_batch_ids) == 1 for hit in hits)
    with SessionLocal() as db:
        parts = db.query(BatchSummary).filter_by(
            conversation_id=batch_context.cid).one().source_manifest['parts']
        assert len(parts) == 5
        assert all(part['mode'] == 'source' and len(part['source_ids']) == 1
                   for part in parts)


def test_late_original_part_ranks_and_fits_when_aggregate_does_not(batch_context, monkeypatch):
    ctx = batch_context
    target = 'The recorded deployment port is 8392.'
    with SessionLocal() as db:
        rows = db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid).order_by(
            EpisodicMemory.timestamp).all()
        for i, row in enumerate(rows):
            row.raw_text = ('Local deployment observation. ' * 40
                            + (target if i == 4 else f'Unrelated observation {i}.'))
            row.source_spans = single_provenance(row.raw_text, 'user')
        target_batch = str(rows[-1].batch_id)
        db.commit()
    monkeypatch.setattr(settings, 'retrieval_batch_summary_limit', 1)
    monkeypatch.setattr(worker, '_batch_llm', lambda *a, **k: 'Unsupported port 9999.')
    monkeypatch.setattr(worker, 'verify_support', lambda p, h: verify_support(p, h,
        scorer=lambda pairs: [dict(entailment=.001, neutral=.009, contradiction=.99)]))
    monkeypatch.setattr(worker, 'embedder', NS(encode=lambda text, **k:
        VEC if target in text else [-1.] + [0.] * 1023))
    worker.batch_summarize()
    with SessionLocal() as db:
        orch = HybridRetrievalOrchestrator(db, NS())
        hits = orch._batch_summary_lookup(VEC, str(ctx.cid), include_cross=False)
        assert len(hits) == 1 and target in hits[0].text
        assert hits[0].origin_batch_ids == (target_batch,)
        assert 'source evidence' in hits[0].text
        root = db.query(BatchSummary).filter_by(conversation_id=ctx.cid).one()
        from src.memory.tokens import count
        root_fragment = ContextFragment(text=root.summary_text, source_type='batch_summary',
            score=1., token_count=count(root.summary_text), conversation_id=str(ctx.cid))
        budget = hits[0].token_count + 10
        assert root_fragment.token_count > budget
        assert not orch._enforce_token_budget([root_fragment], max_tokens=budget,
            current_conversation_id=str(ctx.cid))
        assert orch._enforce_token_budget(hits, max_tokens=budget,
            current_conversation_id=str(ctx.cid)) == hits


@pytest.mark.skipif(os.environ.get('ICE_TEST_LIVE_ENCODER') != '1',
                    reason='opt-in actual shared-encoder ranking control')
def test_actual_shared_encoder_finds_the_late_source_part(batch_context, monkeypatch):
    from src.memory.embedder import get_embedder

    ctx = batch_context
    observations = [
        'I grow tulips in a garden plot.',
        'I prepare garlic pasta for dinner.',
        'I replace bicycle brake pads each spring.',
        'I check the train timetable before a trip.',
        'The API gateway listens on port 8392 for clients.',
    ]
    with SessionLocal() as db:
        rows = db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid).order_by(
            EpisodicMemory.timestamp).all()
        for row, observation in zip(rows, observations):
            row.raw_text = observation
            row.source_spans = single_provenance(observation, 'user')
        target_batch = str(rows[-1].batch_id)
        db.commit()
    monkeypatch.setattr(settings, 'retrieval_batch_summary_limit', 1)
    monkeypatch.setattr(worker, '_batch_llm', lambda *a, **k: 'Unsupported port 9999.')
    monkeypatch.setattr(worker, 'verify_support', lambda p, h: verify_support(p, h,
        scorer=lambda pairs: [dict(entailment=.001, neutral=.009, contradiction=.99)]))
    encoder = get_embedder()
    monkeypatch.setattr(worker, 'embedder', encoder)
    worker.batch_summarize()
    question = 'Which port does the API gateway listen on for clients?'
    query = encoder.encode(question, convert_to_tensor=False).tolist()
    with SessionLocal() as db:
        orch = HybridRetrievalOrchestrator(db, NS())
        hits = orch._batch_summary_lookup(query, str(ctx.cid), include_cross=False)
        assert len(hits) == 1 and 'port 8392' in hits[0].text
        assert hits[0].origin_batch_ids == (target_batch,)
        root = db.query(BatchSummary).filter_by(conversation_id=ctx.cid).one()
        from src.memory.tokens import count
        assert hits[0].token_count < count(root.summary_text)


def test_part_scope_selects_only_visible_sources_from_a_mixed_parent(batch_context, monkeypatch):
    ctx = batch_context
    monkeypatch.setattr(settings, 'retrieval_batch_summary_limit', 5)
    monkeypatch.setattr(worker, '_batch_llm', lambda *a, **k: 'Unsupported port 9999.')
    monkeypatch.setattr(worker, 'verify_support', lambda p, h: verify_support(p, h,
        scorer=lambda pairs: [dict(entailment=.001, neutral=.009, contradiction=.99)]))
    worker.batch_summarize()
    allowed_cluster, other_cluster = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        rows = db.query(EpisodicMemory).filter_by(conversation_id=ctx.cid).order_by(
            EpisodicMemory.timestamp).all()
        allowed = rows[-1]
        db.add_all([ContextCluster(id=allowed_cluster, name='allowed'),
                    ContextCluster(id=other_cluster, name='other')])
        db.flush()
        for row in rows:
            db.add(EpisodicClusterLink(episodic_id=row.id, cluster_id=(
                allowed_cluster if row.id == allowed.id else other_cluster)))
        db.commit()
        orch = HybridRetrievalOrchestrator(db, NS())
        for scope in ({'batch_ids': [str(allowed.batch_id)]},
                      {'cluster_ids': [str(allowed_cluster)]}):
            hits = orch._batch_summary_lookup(VEC, str(ctx.cid),
                                              include_cross=False, scope=scope)
            assert len(hits) == 1
            assert hits[0].origin_batch_ids == (str(allowed.batch_id),)
            assert 'Source observation 0.' in hits[0].text
        note = db.query(BatchNote).filter_by(summary_id=rows[0].batch_summary_id).first()
        note.text = 'Invalid index part.'
        db.commit()
        for scope in ({'batch_ids': [str(allowed.batch_id)]},
                      {'cluster_ids': [str(allowed_cluster)]}):
            assert not orch._batch_summary_lookup(VEC, str(ctx.cid),
                                                  include_cross=False, scope=scope)
        db.query(EpisodicClusterLink).filter(EpisodicClusterLink.episodic_id.in_(
            [row.id for row in rows])).delete(synchronize_session=False)
        db.query(ContextCluster).filter(ContextCluster.id.in_(
            [allowed_cluster, other_cluster])).delete(synchronize_session=False)
        db.commit()


@pytest.mark.parametrize('change', ['missing', 'text', 'batch_ids'])
def test_current_batch_index_repairs_without_generation(batch_context, change):
    ctx = batch_context
    worker.batch_summarize()
    before = len(ctx.calls)
    with SessionLocal() as db:
        summary = db.query(BatchSummary).filter_by(conversation_id=ctx.cid).one()
        note = db.query(BatchNote).filter_by(summary_id=summary.id).one()
        if change == 'missing': db.delete(note)
        elif change == 'text': note.text = 'Unverified edited index port 9999.'
        else: note.batch_ids = [str(uuid.uuid4())]
        db.commit()
    # Compatibility uses the current manifest-bound root, never the edited index.
    hits = read(ctx.cid)
    assert len(hits) == 1 and '9999' not in hits[0].text
    assert len(hits[0].origin_batch_ids) == 5
    worker.batch_summarize()
    assert len(ctx.calls) == before
    hits = read(ctx.cid)
    assert len(hits) == 1 and 'batch note created:' in hits[0].text


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
