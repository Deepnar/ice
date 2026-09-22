"""v3 archived excerpts survive actual decay, small-budget retrieval and restore."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import ColdChunk, ColdStorage, Conversation, EpisodicChunk, EpisodicMemory
from src.memory.source import single_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.timescope import TimeScope
from src.workers.decay import apply_decay

assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
assert make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE']
VEC = [1.] + [0.] * 1023


@pytest.mark.parametrize('collision', [False, True])
@pytest.mark.parametrize('vector', [False, True])
def test_late_excerpt_archive_pack_restore(monkeypatch, vector, collision):
    cid, sid, chunk_id, other = (uuid.uuid4() for _ in range(4))
    stamp = datetime.now(timezone.utc) - timedelta(days=100)
    correction = 'Final correction: never enable Redis persistence.'
    raw = 'Discuss deployment alternatives. ' * 900 + correction
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        db.add(EpisodicMemory(id=sid, conversation_id=cid, batch_id=uuid.uuid4(),
            raw_text=raw, source_spans=single_provenance(raw, 'user'),
            timestamp=stamp, decay_score=.001, is_archived=True, embedding=VEC,
            idempotency_key=str(uuid.uuid4()), context_reliance='Long_Term_Memory'))
        db.flush()
        db.add(EpisodicChunk(id=chunk_id, turn_id=sid, chunk_index=9,
                            chunk_text=correction, embedding=VEC)); db.commit()
        try:
            apply_decay(); db.expire_all()
            archived = db.get(ColdChunk, chunk_id)
            assert archived.chunk_text == correction and list(archived.embedding) == VEC
            assert db.get(EpisodicChunk, chunk_id) is None
            orch = HybridRetrievalOrchestrator(db, NS())
            orch._active_timescope = TimeScope(mode='range',
                t0=stamp - timedelta(days=1), t1=stamp + timedelta(days=1))
            probe = VEC if vector else None
            hits = orch._cold_lookup({'redis'}, str(cid), prompt_embedding=probe)
            assert len(hits) == 2
            selected = orch._enforce_token_budget(hits, max_tokens=100,
                                                  relevance_order=True)
            assert len(selected) == 1 and correction in selected[0].text
            assert selected[0].leg == 'cold_chunk'
            assert not selected[0].covers_entire_source
            assert not orch._cold_lookup({'redis'}, str(cid),
                {'exclude_conversation_ids': [cid]}, prompt_embedding=probe)
            if collision:
                db.add(EpisodicMemory(id=other, conversation_id=cid, batch_id=uuid.uuid4(),
                    raw_text='Other source.', idempotency_key=str(uuid.uuid4()),
                    context_reliance='Long_Term_Memory'))
                db.flush()
                db.add(EpisodicChunk(id=chunk_id, turn_id=other, chunk_index=0,
                                    chunk_text='Other source.', embedding=VEC))
                db.commit()
            monkeypatch.setattr(settings, 'retrieval_strengthen_writes', True)
            orch._resurrect_cold_hits(selected); db.expire_all()
            if collision:
                assert db.get(EpisodicMemory, sid) is None
                assert db.get(ColdStorage, sid) is not None
                assert db.get(ColdChunk, chunk_id).chunk_text == correction
                assert db.get(EpisodicChunk, chunk_id).turn_id == other
                return
            restored = db.get(EpisodicChunk, chunk_id)
            assert restored.chunk_text == correction and restored.chunk_index == 9
            assert list(restored.embedding) == VEC
            assert db.get(ColdStorage, sid) is None and db.get(ColdChunk, chunk_id) is None
        finally:
            db.rollback()
            db.query(EpisodicChunk).filter(EpisodicChunk.turn_id.in_([sid, other])).delete()
            db.query(EpisodicMemory).filter(EpisodicMemory.id.in_([sid, other])).delete()
            db.query(ColdStorage).filter_by(id=sid).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()


def test_cold_chunk_migration_roundtrip():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    from src.api.db import engine
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, 'ea182446f5b7')
    command.downgrade(cfg, 'd9071335e4a6')
    assert 'cold_chunks' not in inspect(engine).get_table_names()
    command.upgrade(cfg, 'ea182446f5b7')
