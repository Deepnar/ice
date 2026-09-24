"""v3 cold visibility keeps known-unlinked distinct from unknown membership."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import ColdStorage, ContextCluster, Conversation, EpisodicClusterLink, EpisodicMemory
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.timescope import TimeScope
from src.workers.decay import apply_decay

assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
assert make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE']


def test_cold_scope_filters_before_limit(monkeypatch):
    a, b, cid, batch = (uuid.uuid4() for _ in range(4))
    stamp = datetime.now(timezone.utc)
    with SessionLocal() as db:
        rows = [ColdStorage(id=uuid.uuid4(), raw_text=f'Atlas evidence {name}',
            conversation_id=cid, batch_id=batch, timestamp=stamp, cluster_ids=clusters)
            for name, clusters in [('a', [a]), ('b', [b]), ('both', [a, b]),
                                   ('unlinked', []), ('unknown', None)]]
        db.add_all(rows); db.commit()
        try:
            orch = HybridRetrievalOrchestrator(db, NS())
            orch._active_timescope = TimeScope(mode='range',
                t0=stamp - timedelta(days=1), t1=stamp + timedelta(days=1))
            def found(scope):
                return {f.source_batch_id for f in orch._cold_lookup({'atlas'}, str(cid), scope)}
            ids = [str(r.id) for r in rows]
            assert found({}) == set(ids)
            assert found({'cluster_ids': [a]}) == {ids[0], ids[2], ids[3]}
            assert found({'exclude_cluster_ids': [b]}) == {ids[0], ids[3]}
            assert found({'cluster_ids': [a], 'exclude_cluster_ids': [b]}) == {ids[0], ids[3]}
            assert not found({'batch_ids': []})
            assert found({'batch_ids': [batch]}) == set(ids)
            assert not found({'batch_ids': [uuid.uuid4()]})
            monkeypatch.setattr(settings, 'timescope_cold_limit', 1)
            assert len(found({'cluster_ids': [a], 'exclude_cluster_ids': [b]})) == 1
        finally:
            db.rollback(); db.query(ColdStorage).filter_by(conversation_id=cid).delete(); db.commit()


def test_cluster_membership_roundtrip(monkeypatch):
    cid, sid, a, b = (uuid.uuid4() for _ in range(4))
    stamp = datetime.now(timezone.utc) - timedelta(days=100)
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        db.add_all([ContextCluster(id=a, name='Atlas'), ContextCluster(id=b, name='Boreal')]); db.flush()
        db.add(EpisodicMemory(id=sid, conversation_id=cid, cluster_id=a,
            raw_text='Atlas historical evidence.', timestamp=stamp, is_archived=True,
            decay_score=.001, idempotency_key=str(uuid.uuid4()), batch_id=uuid.uuid4(),
            context_reliance='Long_Term_Memory')); db.flush()
        db.add_all([EpisodicClusterLink(episodic_id=sid, cluster_id=c) for c in [a, b]])
        db.commit()
        try:
            apply_decay(); db.expire_all()
            cold = db.get(ColdStorage, sid)
            assert cold.cluster_id == a and set(cold.cluster_ids) == {a, b}
            assert not db.query(EpisodicClusterLink).filter_by(episodic_id=sid).all()
            # A deleted cluster is not recreated by restoring the memory.
            db.query(ContextCluster).filter_by(id=b).delete(); db.commit()
            orch = HybridRetrievalOrchestrator(db, NS())
            orch._active_timescope = TimeScope(mode='range',
                t0=stamp - timedelta(days=1), t1=stamp + timedelta(days=1))
            hits = orch._cold_lookup({'atlas'}, str(cid), {'cluster_ids': [a]})
            assert len(hits) == 1
            monkeypatch.setattr(settings, 'retrieval_strengthen_writes', True)
            orch._resurrect_cold_hits(hits); db.expire_all()
            assert db.get(EpisodicMemory, sid).cluster_id == a
            assert {r.cluster_id for r in db.query(EpisodicClusterLink).filter_by(episodic_id=sid)} == {a}
            assert db.get(ColdStorage, sid) is None
        finally:
            db.rollback()
            db.query(EpisodicClusterLink).filter_by(episodic_id=sid).delete()
            db.query(EpisodicMemory).filter_by(id=sid).delete()
            db.query(ColdStorage).filter_by(id=sid).delete()
            db.query(ContextCluster).filter(ContextCluster.id.in_([a,b])).delete()
            db.query(Conversation).filter_by(id=cid).delete(); db.commit()


def test_cluster_migration_roundtrip():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    from src.api.db import engine
    cfg = Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url', settings.database_url.replace('%', '%%'))
    command.stamp(cfg, 'd9071335e4a6')
    command.downgrade(cfg, 'c8f60224d395')
    assert 'cluster_ids' not in {c['name'] for c in inspect(engine).get_columns('cold_storage')}
    command.upgrade(cfg, 'd9071335e4a6')
