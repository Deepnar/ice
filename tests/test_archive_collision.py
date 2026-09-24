"""v3 archive retry cannot discard a newer source or privacy correction."""
import os
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import ColdStorage, Conversation, EpisodicMemory
from src.memory.source import single_provenance
from src.workers.decay import apply_decay

assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
assert make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE']


def test_archive_collision_preserves_current_evidence():
    cid, sid, bid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    stamp = datetime.now(timezone.utc) - timedelta(days=100)
    raw = 'Correction: never enable Redis persistence.'
    provenance = single_provenance(raw, 'user')
    vector = [1.] + [0.] * 1023
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        db.add(EpisodicMemory(id=sid, conversation_id=cid, batch_id=bid,
            raw_text=raw, summary_text='Current summary metadata.',
            source_spans=provenance, timestamp=stamp, ts_provenance='original',
            is_private=True, embedding=vector, topic_tags=['Technology'],
            decay_score=.001, is_archived=True,
            context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4())))
        db.add(ColdStorage(id=sid, raw_text='Enable Redis persistence.',
            timestamp=stamp - timedelta(days=1), is_private=False,
            conversation_id=None, source_spans=None, embedding=None))
        db.commit()
        try:
            apply_decay()
            db.expire_all()
            assert db.get(EpisodicMemory, sid) is None
            cold = db.get(ColdStorage, sid)
            assert cold.raw_text == raw and cold.summary_text == 'Current summary metadata.'
            assert cold.source_spans == provenance
            assert cold.timestamp == stamp and cold.ts_provenance == 'original'
            assert cold.is_private and cold.conversation_id == cid and cold.batch_id == bid
            assert cold.topic_tags == ['Technology']
            assert list(cold.embedding) == vector
        finally:
            db.rollback()
            db.query(EpisodicMemory).filter_by(id=sid).delete()
            db.query(ColdStorage).filter_by(id=sid).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()


def test_cold_metadata_migration_roundtrip():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    from src.api.db import engine
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, 'c8f60224d395')
    command.downgrade(cfg, 'b7e5f013c284')
    assert 'representation_verification' not in {c['name'] for c in inspect(engine).get_columns('cold_storage')}
    command.upgrade(cfg, 'c8f60224d395')


def test_archive_read_restore_preserves_verification_and_identity(monkeypatch):
    from types import SimpleNamespace as NS
    from src.memory.representation import verify_representations
    from src.memory.support import verify_support
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    from src.retrieval.timescope import TimeScope

    cid, sid, bid, session = (uuid.uuid4() for _ in range(4))
    stamp = datetime.now(timezone.utc) - timedelta(days=100)
    raw, summary = 'I use Atlas on port 8391.', 'The user uses Atlas on port 8391.'
    key = str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        turn = EpisodicMemory(id=sid, conversation_id=cid, batch_id=bid,
            raw_text=raw, summary_text=summary, source_spans=single_provenance(raw, 'user'),
            timestamp=stamp, ts_provenance='original', session_id=session,
            summary_coverage=1., abstract_text=summary, inject_raw=False, lossless_flag=False,
            intent_tags=['Factual_Retrieval'], embedding=[1.] + [0.] * 1023,
            decay_score=.001, is_archived=True, context_reliance='Long_Term_Memory',
            idempotency_key=key)
        verdict = verify_representations(turn, summary, summary,
            verifier=lambda p, h: verify_support(p, h, scorer=lambda pairs:
                [dict(entailment=.99, neutral=.005, contradiction=.005)]))
        turn.representation_verification = verdict
        db.add(turn); db.commit()
        try:
            apply_decay()
            db.expire_all()
            orch = HybridRetrievalOrchestrator(db, NS())
            orch._active_timescope = TimeScope(mode='range',
                t0=stamp - timedelta(days=1), t1=stamp + timedelta(days=1))
            hits = orch._cold_lookup({'atlas'}, str(cid))
            assert len(hits) == 1 and summary in hits[0].text
            assert not hits[0].covers_entire_source
            monkeypatch.setattr(settings, 'retrieval_strengthen_writes', False)
            orch._resurrect_cold_hits(hits)
            assert db.get(ColdStorage, sid) is not None
            assert db.get(EpisodicMemory, sid) is None
            monkeypatch.setattr(settings, 'retrieval_strengthen_writes', True)
            orch._resurrect_cold_hits(hits)
            db.expire_all()
            restored = db.get(EpisodicMemory, sid)
            assert restored is not None and db.get(ColdStorage, sid) is None
            assert restored.representation_verification == verdict
            assert restored.summary_coverage == 1. and restored.abstract_text == summary
            assert restored.inject_raw is False and restored.lossless_flag is False
            assert restored.session_id == session and restored.idempotency_key == key
            assert restored.intent_tags == ['Factual_Retrieval']
            assert restored.raw_text == raw and restored.summary_text == summary
            assert restored.timestamp == stamp and restored.batch_id == bid
        finally:
            db.rollback()
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
            db.query(ColdStorage).filter_by(conversation_id=cid).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()


def test_restore_idempotency_conflict_keeps_cold_source(monkeypatch):
    from types import SimpleNamespace as NS
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    from src.retrieval.timescope import TimeScope

    cid, sid = uuid.uuid4(), uuid.uuid4()
    stamp = datetime.now(timezone.utc) - timedelta(days=100)
    key = str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        db.add(EpisodicMemory(conversation_id=cid, raw_text='Existing live source.',
            idempotency_key=key, context_reliance='Long_Term_Memory', batch_id=uuid.uuid4()))
        db.add(ColdStorage(id=sid, conversation_id=cid, raw_text='Archived Atlas evidence.',
            timestamp=stamp, idempotency_key=key))
        db.commit()
        try:
            monkeypatch.setattr(settings, 'retrieval_strengthen_writes', True)
            orch = HybridRetrievalOrchestrator(db, NS())
            orch._active_timescope = TimeScope(mode='range',
                t0=stamp - timedelta(days=1), t1=stamp + timedelta(days=1))
            hits = orch._cold_lookup({'atlas'}, str(cid))
            assert len(hits) == 1
            orch._resurrect_cold_hits(hits)
            assert db.get(ColdStorage, sid).raw_text == 'Archived Atlas evidence.'
            assert db.get(EpisodicMemory, sid) is None
        finally:
            db.rollback()
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
            db.query(ColdStorage).filter_by(conversation_id=cid).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()
