"""v3 long-turn late evidence through SQL candidates, ranking and real packing."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import ColdStorage, Conversation, EpisodicMemory, EpisodicChunk
from src.memory.source import single_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
import src.retrieval.reranker as ranking

assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
assert make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE']
VEC = [1.] + [0.] * 1023


@pytest.mark.parametrize('wide', [False, True])
@pytest.mark.parametrize('ranked', [False, True])
def test_late_correction_survives_small_budget_through_actual_retrieve(monkeypatch, ranked, wide):
    cid = uuid.uuid4()
    raw = 'Discuss deployment alternatives. ' * 900 + 'Final correction: never enable Redis persistence.'
    correction = 'Final correction: never enable Redis persistence.'
    classification = NS(prompt='Should we enable Redis persistence?', max_confidence=1.,
                        context_reliance='Long_Term_Memory',
                        intent_tags=['Factual_Retrieval'], topic_tags=[])
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        turn = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(),
            raw_text=raw, source_spans=single_provenance(raw, 'user'), inject_raw=True,
            embedding=VEC, context_reliance='Long_Term_Memory',
            idempotency_key=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc))
        db.add(turn); db.flush()
        db.add(EpisodicChunk(turn_id=turn.id, chunk_index=1,
                            chunk_text=correction, embedding=VEC))
        db.commit()
        try:
            orch = HybridRetrievalOrchestrator(db, NS())
            orch.max_retrieval_tokens = 100
            candidates = orch._vector_episodic(VEC, classification, None, str(cid))
            assert len(candidates) == 2
            parent = next(f for f in candidates if f.covers_entire_source)
            assert raw in parent.text and correction in parent.text
            # Keep both real SQL legs; stub unrelated subsystems only.
            for name in ('_codex_graph', '_codex_claims', '_relevant_cluster_ids',
                         '_procedural_lookup', '_batch_summary_lookup', '_cold_lookup'):
                monkeypatch.setattr(orch, name, lambda *a, **kw: [])
            monkeypatch.setattr(settings, 'retrieval_rerank_enabled', ranked)
            monkeypatch.setattr(settings, 'retrieval_max_frags_per_turn', 1)
            monkeypatch.setattr(ranking, 'score_pairs', lambda pairs:
                [5. if len(t) > 1000 else 4. for _,t in pairs])
            if wide:
                monkeypatch.setattr(settings, 'retrieval_wide_net_budget_floor', 100)
                selected = orch._wide_net_fallback(classification, VEC, str(cid),
                                                   {'conversation_id': str(cid)})
            else:
                selected = orch.retrieve(classification, str(cid), VEC,
                                         scope={'conversation_id': str(cid)})
            assert len(selected) == 1 and correction in selected[0].text
            assert not selected[0].covers_entire_source
            assert selected[0].token_count <= 100
            assert selected[0].source_batch_id == str(turn.id)
        finally:
            db.rollback()
            db.query(EpisodicChunk).filter_by(turn_id=turn.id).delete()
            db.query(EpisodicMemory).filter_by(id=turn.id).delete()
            db.query(Conversation).filter_by(id=cid).delete()
            db.commit()


def test_cold_reader_keeps_late_source_instead_of_unverified_summary():
    from src.retrieval.timescope import TimeScope

    cid, sid = uuid.uuid4(), uuid.uuid4()
    now = datetime.now(timezone.utc)
    raw = 'Discuss deployment alternatives. ' * 400 + 'Never enable Redis persistence.'
    with SessionLocal() as db:
        db.add(ColdStorage(id=sid, conversation_id=cid, raw_text=raw,
            summary_text='Enable Redis persistence.', timestamp=now,
            source_spans=single_provenance(raw, 'user'), embedding=VEC))
        db.commit()
        try:
            orch = HybridRetrievalOrchestrator(db, NS())
            orch._active_timescope = TimeScope(mode='range',
                t0=now - timedelta(days=1), t1=now + timedelta(days=1))
            hits = orch._cold_lookup({'redis'}, str(cid), prompt_embedding=VEC)
            assert len(hits) == 1
            assert raw in hits[0].text
            assert 'Enable Redis persistence.' not in hits[0].text
            assert hits[0].covers_entire_source
        finally:
            db.rollback()
            db.query(ColdStorage).filter_by(id=sid).delete()
            db.commit()
