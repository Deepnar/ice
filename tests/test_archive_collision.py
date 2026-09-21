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
