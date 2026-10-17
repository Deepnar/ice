"""Procedural clocks/retirement through real PostgreSQL and final budgeting."""

import io
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import ColdStorage, Conversation, EpisodicMemory, ProceduralMemory
from src.memory.procedural_support import pattern_sources
from src.memory.source import chat_provenance
from src.memory.source_order import attach_sequence, sequence_receipt
from src.memory.tokens import count
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import procedural_decay
from test_procedural_support import fake_review


def _retrieval(monkeypatch, db, conversation, *, actual=False, scope=None):
    query = 'How do I want my patch reviews organized?'
    classification = NS(prompt=query, context_reliance='Long_Term_Memory',
        topic_tags=['Software_&_Tech'], intent_tags=['Factual_Retrieval'],
        head_confidences={'topic': 1., 'intent': 1.}, max_confidence=1.)
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', True)
    monkeypatch.setattr(settings, 'retrieval_rerank_min_score', None)
    import src.retrieval.reranker as reranker
    if not actual:
        monkeypatch.setattr(reranker, 'score_pairs', lambda pairs: [5.] * len(pairs))
    reader = HybridRetrievalOrchestrator(db, None)
    reader.max_retrieval_tokens = 256
    for name in ('_bm25_episodic', '_vector_episodic', '_facts_leg',
                 '_batch_summary_lookup', '_cold_lookup', '_relevant_cluster_ids'):
        monkeypatch.setattr(reader, name, lambda *_a, **_kw: [])
    final = reader.retrieve(classification, str(conversation), [1.] + [0.] * 1023,
                            scope=scope, defer_exposure=True)
    assert sum(f.token_count for f in final) <= 256
    assert all(f.token_count == count(f.text) for f in final)
    return final


def exercise_source_clocks(monkeypatch, *, actual=False):
    if not os.getenv('ICE_TEST_DATABASE', '').startswith('ice_test_'):
        raise RuntimeError('requires disposable PostgreSQL')
    description = 'The user prefers patch reviews to list failing tests before proposed changes.'
    cid, hidden_cid = uuid.uuid4(), uuid.uuid4()
    ids, batches, pid = [uuid.uuid4() for _ in range(3)], [uuid.uuid4() for _ in range(3)], uuid.uuid4()
    now = datetime(2026, 10, 9, tzinfo=timezone.utc)
    with SessionLocal() as db:
        db.add_all([Conversation(id=cid), Conversation(id=hidden_cid)])
        db.flush()
        for i, date in enumerate((datetime(2010, 1, 1, tzinfo=timezone.utc), now,
                                  datetime(2025, 5, 5, tzinfo=timezone.utc))):
            raw = f'User: {description}\n\nAssistant: Noted.'
            spans = chat_provenance(description, 'Noted.')
            if i < 2:
                spans = attach_sequence(spans, raw, sequence_receipt(raw, 'a' * 64, i + 1))
            model = ColdStorage if i == 1 else EpisodicMemory
            db.add(model(id=ids[i], batch_id=batches[i], conversation_id=cid if i < 2 else hidden_cid,
                session_id=uuid.uuid4(), raw_text=raw, source_spans=spans, timestamp=date,
                ts_provenance='synthetic_raw_import' if i == 1 else 'original',
                context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4())))
        db.flush()
        originals = pattern_sources(db, set(batches))
        proof = fake_review(description, originals, kind='explicit_instruction')
        pattern = ProceduralMemory(id=pid, pattern_name='Patch review order', pattern_description=description,
            embedding=[1.] + [0.] * 1023, source_batch_ids=batches,
            trigger_conditions={'_source_support': proof}, first_observed=now, last_observed=now,
            reinforcement_count=3, confidence_score=.8, is_active=True)
        db.add(pattern); db.commit()
        before = {s.batch_id: s.raw_sha256 for s in originals}
        try:
            scope = {'exclude_conversation_ids': [str(hidden_cid)]}
            final = _retrieval(monkeypatch, db, cid, actual=actual, scope=scope)
            procedures = [f for f in final if f.source_type == 'procedural']
            assert len(procedures) == 1
            rendered = procedures[0].text
            assert rendered.startswith('[Explicit user instruction]')
            assert 'sources recorded: 2010-01-01T00:00:00Z' in rendered
            assert 'source calendar unknown: 1' in rendered
            assert f'verified order in conversation {str(cid)[:8]}: 1 to 2' in rendered
            assert 'ICE observations: 2026-10-09T00:00:00Z' in rendered
            assert '2025-05-05' not in rendered and str(hidden_cid)[:8] not in rendered
            assert {s.batch_id: s.raw_sha256 for s in pattern_sources(db, set(batches))} == before
            assert db.get(ColdStorage, ids[1]) is not None  # read-only exposure
            return dict(source_dates_separated=True, cold_source_preserved=True,
                        hidden_date_absent=True, final_tokens=procedures[0].token_count,
                        budget=256, actual_reranker=actual)
        finally:
            db.rollback()
            db.query(ProceduralMemory).filter_by(id=pid).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter(EpisodicMemory.id.in_(ids)).delete(synchronize_session=False)
            db.query(ColdStorage).filter(ColdStorage.id.in_(ids)).delete(synchronize_session=False)
            db.query(Conversation).filter(Conversation.id.in_([cid, hidden_cid])).delete(synchronize_session=False)
            db.commit()


def exercise_retirement(monkeypatch, *, actual=False):
    if not os.getenv('ICE_TEST_DATABASE', '').startswith('ice_test_'):
        raise RuntimeError('requires disposable PostgreSQL')
    now = datetime(2026, 10, 9, tzinfo=timezone.utc)
    cutoff = now - timedelta(days=180)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now
    monkeypatch.setattr(procedural_decay, 'datetime', Clock)
    monkeypatch.setattr(settings, 'procedural_stale_days', 180)
    monkeypatch.setattr(settings, 'procedural_min_reinforcement', 3)
    cid = uuid.uuid4()
    cases = [('stale_single_sitting', cutoff - timedelta(seconds=1), 1, True),
             ('stale_recurrent', cutoff - timedelta(days=20), 3, True),
             ('boundary', cutoff, 1, True), ('recent', now, 1, True),
             ('already_inactive', cutoff - timedelta(days=20), 1, False)]
    rows = []
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        for name, last, recurrence, active in cases:
            row = ProceduralMemory(pattern_name=name,
                pattern_description='The user prefers patch reviews to list failing tests before proposed changes.',
                embedding=[1.] + [0.] * 1023, source_batch_ids=[], trigger_conditions={},
                first_observed=last, last_observed=last, reinforcement_count=recurrence,
                confidence_score=.8, is_active=active)
            db.add(row); db.flush(); rows.append(row)
        db.commit()
        original = [(r.id, r.pattern_description, r.last_observed, r.reinforcement_count) for r in rows]
        try:
            before = _retrieval(monkeypatch, db, cid, actual=actual)
            assert len([f for f in before if f.source_type == 'procedural']) > 0
            receipt = procedural_decay.decay_procedural_patterns(cycles=999)
            assert receipt['active_before'] == 4 and receipt['stale_protected'] == 1 and receipt['retired'] == 1
            assert receipt['clock_basis'] == 'ice_observation' and receipt['cutoff'] == cutoff.isoformat()
            db.expire_all()
            assert [r.is_active for r in rows] == [False, True, True, True, False]
            assert [(r.id, r.pattern_description, r.last_observed, r.reinforcement_count) for r in rows] == original
            # Reads before/after neither corroborate nor change retirement dates.
            after = _retrieval(monkeypatch, db, cid, actual=actual)
            assert any(f.source_type == 'procedural' for f in after)
            # Production registry → worker → replay receipt → derived report.
            from scripts.z1 import report_v3_replay, seed_v3
            monkeypatch.setattr(settings, 'maintenance_intervals', {'decay_procedural': 5400})
            sink = io.StringIO()
            seed_v3.due_maintenance(db, sink, now, cid, {})
            emitted = [json.loads(line) for line in sink.getvalue().splitlines()]
            assert len(emitted) == 1 and emitted[0]['worker_status'] == 'complete'
            second = emitted[0]['result']
            assert second['active_before'] == 3 and second['stale_protected'] == 1 and second['retired'] == 0
            assert db.query(ProceduralMemory).count() == 5
            with TemporaryDirectory() as temporary:
                path = Path(temporary) / 'seed.jsonl'
                path.write_text(json.dumps(dict(event='run', meta={'extra': {}})) + '\n' + sink.getvalue())
                observed = report_v3_replay.summarize(path)['procedural_retirement']['counts']
                assert observed['observed_calls'] == 1 and observed['retired'] == 0
            return dict(first=receipt, second=second, stored=5,
                        unchanged_observation_clocks=True, collector_report_verified=True,
                        actual_reranker=actual)
        finally:
            db.rollback()
            db.query(ProceduralMemory).filter(ProceduralMemory.id.in_([r.id for r in rows])).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False)
            db.commit()


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='disposable PostgreSQL required')
def test_source_clock_reaches_final_context(monkeypatch):
    exercise_source_clocks(monkeypatch)


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='disposable PostgreSQL required')
def test_retirement_and_final_reader(monkeypatch):
    exercise_retirement(monkeypatch)
