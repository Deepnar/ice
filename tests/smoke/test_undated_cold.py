"""v3: cold evidence remains searchable without guessing its recording date."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.api import memory_preparation
from src.api.config import settings
from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import ColdChunk, ColdStorage, Conversation, EpisodicChunk, EpisodicMemory
from src.memory.source import chat_provenance
from src.retrieval.configurable_orchestrator import ConfigurableOrchestrator
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.timescope import CURRENT, TimeScope, to_scope_dict
from src.workers.decay import apply_decay


VEC = [1.] + [0.] * 1023
RAW = 'User: My recovery phrase is cobalt harbor.\n\nAssistant: Noted.'
QUESTION = 'What was my recovery phrase?'


class Encoder:
    def encode(self, text, **_kwargs):
        return np.array([VEC for _ in text] if isinstance(text, list) else VEC)


def classification(weak=False):
    return ClassificationResult(prompt=QUESTION, topic_tags=[], intent_tags=['Factual_Retrieval'],
        context_reliance='Long_Term_Memory', raw_probs=[0.] * 27, max_confidence=1., p_ltm=1.,
        head_confidences={'topic': .01 if weak else 1., 'intent': .01 if weak else 1.})


@pytest.fixture
def source(monkeypatch):
    if not os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_'):
        pytest.skip('Use disposable_database.py; never mutates the working database.')
    monkeypatch.setattr(settings, 'memory_source_gate_enabled', False)
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', True)
    monkeypatch.setattr(settings, 'retrieval_strengthen_writes', False)
    monkeypatch.setattr('src.retrieval.orchestrator.extract_entities', lambda _text, _encoder=None: [])
    monkeypatch.setattr('src.retrieval.reranker.score_pairs',
        lambda pairs: [5. if 'cobalt harbor' in text else -5. for _, text in pairs])
    with SessionLocal() as db:
        cid, sid, bid = (uuid.uuid4() for _ in range(3))
        when = datetime.now(timezone.utc) - timedelta(days=500)
        db.add(Conversation(id=cid, memory_scope_type='none')); db.flush()
        db.add(ColdStorage(id=sid, conversation_id=cid, batch_id=bid,
            raw_text=RAW, timestamp=when, embedding=VEC, ts_provenance='original',
            source_spans=chat_provenance('My recovery phrase is cobalt harbor.', 'Noted.'),
            cluster_ids=[]))
        db.commit()
        try:
            yield db, cid, sid, bid, when
        finally:
            db.rollback()
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(ColdStorage).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False)
            db.commit()


@pytest.mark.parametrize('weak', [False, True])
@pytest.mark.parametrize('timescope', [CURRENT, TimeScope(mode='evolution')])
def test_ordinary_and_undated_evolution_reach_final_budget(source, weak, timescope):
    db, cid, sid, _, _ = source
    reader = HybridRetrievalOrchestrator(db, Encoder())
    reader.max_retrieval_tokens = 100
    selected = reader.retrieve(classification(weak), str(cid), VEC,
        {'conversation_id': str(cid), 'timescope': to_scope_dict(timescope)}, defer_exposure=True)
    assert any(f.source_batch_id == str(sid) and 'cobalt harbor' in f.text for f in selected)
    assert sum(f.token_count for f in selected) <= 100
    assert db.get(ColdStorage, sid).raw_text == RAW and db.get(EpisodicMemory, sid) is None


def test_lexical_search_does_not_admit_nonmatching_vector_rows_or_empty_browse(source):
    db, cid, sid, _, when = source
    reader = HybridRetrievalOrchestrator(db, Encoder())
    assert not reader._cold_lookup(set(), str(cid))
    assert not reader._cold_lookup({'unrelated'}, str(cid))
    assert reader._cold_lookup({'recovery'}, str(cid))[0].source_batch_id == str(sid)
    reader._active_timescope = TimeScope(mode='range', axis='unknown',
        t0=when - timedelta(days=1), t1=when + timedelta(days=1))
    assert not reader._cold_lookup(set(), str(cid))
    reader._active_timescope = TimeScope(mode='range',
        t0=when - timedelta(days=1), t1=when + timedelta(days=1))
    assert reader._cold_lookup(set(), str(cid))[0].source_batch_id == str(sid)
    reader._active_timescope = TimeScope(mode='range',
        t0=when + timedelta(days=1), t1=when + timedelta(days=2))
    assert not reader._cold_lookup({'recovery'}, str(cid), prompt_embedding=VEC)


@pytest.mark.parametrize('weak', [False, True])
def test_explicit_cold_ablation_changes_retrieval_but_timescope_off_does_not(source, weak):
    db, cid, sid, _, _ = source
    scope = {'conversation_id': str(cid)}
    active = ConfigurableOrchestrator(db, Encoder(), {'timescope': False})
    disabled = ConfigurableOrchestrator(db, Encoder(), {'cold': False})
    assert any(f.source_batch_id == str(sid) for f in
        active.retrieve(classification(weak), str(cid), VEC, scope, defer_exposure=True))
    assert not disabled.retrieve(classification(weak), str(cid), VEC, scope, defer_exposure=True)


def test_undated_scope_and_cap_apply_before_ranking(source, monkeypatch):
    db, cid, sid, bid, when = source
    foreign = uuid.uuid4()
    db.add(ColdStorage(id=foreign, raw_text=RAW, is_private=True, timestamp=when,
        embedding=VEC, cluster_ids=[])); db.commit()
    try:
        reader = HybridRetrievalOrchestrator(db, Encoder())
        monkeypatch.setattr(settings, 'timescope_cold_limit', 1)
        hits = reader._cold_lookup(set(), prompt_embedding=VEC)
        assert [f.source_batch_id for f in hits] == [str(sid)]
        assert not reader._cold_lookup(set(), str(cid),
            {'exclude_conversation_ids': [cid]}, VEC)
        assert not reader._cold_lookup(set(), str(cid), {'batch_ids': []}, VEC)
        assert reader._cold_lookup(set(), str(cid), {'batch_ids': [bid]}, VEC)
        assert not reader._cold_lookup(set(), scope={'conversation_ids': []}, prompt_embedding=VEC)
    finally:
        db.rollback(); db.query(ColdStorage).filter_by(id=foreign).delete(); db.commit()


@pytest.mark.parametrize('weak', [False, True])
def test_candidate_or_budget_rejection_does_not_restore(source, monkeypatch, weak):
    db, cid, sid, _, _ = source
    monkeypatch.setattr(settings, 'retrieval_strengthen_writes', True)
    reader = HybridRetrievalOrchestrator(db, Encoder())
    reader.max_retrieval_tokens = 100
    selected = reader.retrieve(classification(weak), str(cid), VEC,
        {'conversation_id': str(cid)}, defer_exposure=True)
    assert selected and db.get(EpisodicMemory, sid) is None
    reader.record_exposure([])  # Final prompt eviction must not resurrect candidacy.
    assert db.get(EpisodicMemory, sid) is None
    reader.max_retrieval_tokens = 1
    # The wide-net budget has a separate floor; test real normal-path rejection.
    assert not reader.retrieve(classification(), str(cid), VEC,
        {'conversation_id': str(cid)}, defer_exposure=True)
    assert db.get(ColdStorage, sid).raw_text == RAW


@pytest.mark.parametrize('weak', [False, True])
def test_direct_selected_read_restores_only_at_probation(source, monkeypatch, weak):
    db, cid, sid, _, when = source
    monkeypatch.setattr(settings, 'retrieval_strengthen_writes', True)
    reader = HybridRetrievalOrchestrator(db, Encoder())
    selected = reader.retrieve(classification(weak), str(cid), VEC,
        {'conversation_id': str(cid)})
    db.expire_all()
    restored = db.get(EpisodicMemory, sid)
    assert selected and restored.raw_text == RAW and restored.timestamp == when
    assert restored.decay_score == settings.timescope_probation_score
    assert db.get(ColdStorage, sid) is None


@pytest.mark.parametrize('weak', [False, True])
def test_actual_decay_cold_final_preparation_and_restore(source, monkeypatch, weak):
    db, cid, sid, bid, when = source
    cold = db.get(ColdStorage, sid)
    spans = cold.source_spans
    db.delete(cold); db.flush()
    key, session, chunk = str(uuid.uuid4()), uuid.uuid4(), uuid.uuid4()
    db.add(EpisodicMemory(id=sid, conversation_id=cid, batch_id=bid, timestamp=when,
        raw_text=RAW, source_spans=spans, embedding=VEC, ts_provenance='original',
        decay_score=.001, is_archived=True, context_reliance='Long_Term_Memory',
        inject_raw=True, session_id=session, idempotency_key=key)); db.flush()
    db.add(EpisodicChunk(id=chunk, turn_id=sid, chunk_index=0,
        chunk_text='My recovery phrase is cobalt harbor.', embedding=VEC)); db.commit()
    apply_decay(); db.expire_all()
    archived = db.get(ColdStorage, sid)
    assert archived.raw_text == RAW and archived.source_spans == spans
    assert db.get(EpisodicMemory, sid) is None and db.get(ColdChunk, chunk) is not None
    args = dict(db=db, classifier=NS(embedder=Encoder()), classification=classification(weak),
        conversation_id=str(cid), user_message=QUESTION, scope={'conversation_id': str(cid)},
        turn_count=1, total_tokens=40, total_budget=16000, serving_window=32768,
        base_retrieve=True, memory_slots=[], bookmarked_texts=[])
    result = memory_preparation.prepare_memory_context(**args)
    assert result.retrieve and result.prepared.ledger.fits()
    prompt = '\n'.join(m['content'] for m in result.prepared.messages)
    assert 'cobalt harbor' in prompt and when.strftime('%Y-%m-%dT%H:%M:%S') in prompt
    assert result.fragments and db.get(EpisodicMemory, sid) is None  # write-off read
    monkeypatch.setattr(settings, 'retrieval_strengthen_writes', True)
    result = memory_preparation.prepare_memory_context(**args)
    db.expire_all()
    restored = db.get(EpisodicMemory, sid)
    assert result.fragments and restored.raw_text == RAW and restored.source_spans == spans
    assert restored.batch_id == bid and restored.timestamp == when and restored.session_id == session
    assert restored.idempotency_key == key and list(restored.embedding) == VEC
    assert restored.decay_score == settings.timescope_probation_score
    assert restored.access_count == 1 and not restored.is_archived
    assert db.get(ColdStorage, sid) is None and db.get(ColdChunk, chunk) is None
    assert db.get(EpisodicChunk, chunk).chunk_text == 'My recovery phrase is cobalt harbor.'
    repeated = memory_preparation.prepare_memory_context(**args)
    db.expire_all()
    assert any(f.source_batch_id == str(sid) for f in repeated.fragments)
    assert db.get(EpisodicMemory, sid).decay_score > settings.timescope_probation_score
    assert db.get(EpisodicMemory, sid).raw_text == RAW


@pytest.mark.parametrize('weak', [False, True])
@pytest.mark.parametrize('score,archived', [(.08, True), (.12, False)])
def test_warm_archived_and_probation_sources_can_compete(source, weak, score, archived):
    db, cid, sid, bid, when = source
    spans = db.get(ColdStorage, sid).source_spans
    db.delete(db.get(ColdStorage, sid)); db.flush()
    db.add(EpisodicMemory(id=sid, conversation_id=cid, batch_id=bid, timestamp=when,
        raw_text=RAW, source_spans=spans, embedding=VEC, ts_provenance='original',
        decay_score=score, is_archived=archived, context_reliance='Long_Term_Memory',
        inject_raw=True, idempotency_key=str(uuid.uuid4()))); db.commit()
    reader = HybridRetrievalOrchestrator(db, Encoder())
    scope = {'conversation_id': str(cid)}
    got = reader.retrieve(classification(weak), str(cid), VEC, scope, defer_exposure=True)
    assert any(f.source_batch_id == str(sid) for f in got)
    scope['timescope'] = to_scope_dict(TimeScope(mode='range',
        t0=when + timedelta(days=1), t1=when + timedelta(days=2)))
    dated = reader.retrieve(classification(weak), str(cid), VEC, scope, defer_exposure=True)
    assert not any(f.source_batch_id == str(sid) for f in dated)
    original = db.get(EpisodicMemory, sid)
    assert original.raw_text == RAW and original.decay_score == score
