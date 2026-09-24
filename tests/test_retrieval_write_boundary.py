"""v3 retrieval popularity must not manufacture graph evidence."""

import uuid
from types import SimpleNamespace

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexEdge, CodexEntity, Conversation, EpisodicMemory
from src.retrieval import orchestrator as retrieval


def test_repeated_graph_candidates_do_not_promote(monkeypatch):
    with SessionLocal() as db:
        try:
            a = CodexEntity(canonical_name=f"atlas_{uuid.uuid4().hex}",
                            context_payload="Atlas uses Beacon.")
            b = CodexEntity(canonical_name=f"beacon_{uuid.uuid4().hex}")
            db.add_all([a, b])
            db.flush()
            edge = CodexEdge(source_id=a.id, target_id=b.id, relation="uses",
                             source_batch=uuid.uuid4(), strength=1.9,
                             extraction_confidence=1.0, confidence="pending")
            db.add(edge)
            db.flush()
            o = retrieval.HybridRetrievalOrchestrator(db, None)
            monkeypatch.setattr(retrieval, "extract_entities", lambda *args: [a.canonical_name])
            monkeypatch.setattr(o, "_match_entities_by_similarity", lambda *args: [a])
            monkeypatch.setattr(o, "_match_entities_exact", lambda *args: [a])
            monkeypatch.setattr(o, "_relation_fit", lambda *args: ({}, 0.0))
            monkeypatch.setattr(settings, "codex_max_depth", 0)
            commits = []
            monkeypatch.setattr(db, "commit", lambda: commits.append(True))
            classification = SimpleNamespace(prompt="What does Atlas use?", intent_tags=[])
            for _ in range(5):
                fragments = o._codex_graph(classification, None)
                assert fragments
                assert fragments[0].origin_edge_ids == (str(edge.id),)
                assert "uses" in fragments[0].text
            db.flush()
            db.refresh(edge)
            assert edge.strength == 1.9 and edge.confidence == "pending"
            assert commits == []
        finally:
            db.rollback()


def test_multiple_chunks_count_as_one_access_and_off_means_no_writes(monkeypatch):
    with SessionLocal() as db:
        try:
            conv = Conversation(id=uuid.uuid4())
            db.add(conv)
            db.flush()
            turn = EpisodicMemory(conversation_id=conv.id, batch_id=uuid.uuid4(),
                raw_text="Synthetic long source.", context_reliance="Long_Term_Memory",
                decay_score=0.4, access_count=0, idempotency_key=str(uuid.uuid4()))
            db.add(turn)
            db.flush()
            fragments = [retrieval.ContextFragment(text=t, source_type="episodic",
                score=1.0, token_count=3, source_batch_id=str(turn.id)) for t in ("chunk A", "chunk B")]
            o = retrieval.HybridRetrievalOrchestrator(db, None)
            monkeypatch.setattr(db, "commit", db.flush)
            monkeypatch.setattr(settings, "retrieval_strengthen_writes", True)
            monkeypatch.setattr(settings, "decay_strengthen_amount", 0.15)
            o._strengthen_retrieved(fragments)
            db.refresh(turn)
            assert turn.access_count == 1 and abs(turn.decay_score - 0.55) < 1e-9
            monkeypatch.setattr(settings, "retrieval_strengthen_writes", False)
            o._strengthen_retrieved(fragments)
            # A sentinel would fail if restoration touched it with writes off.
            o._cold_hits = {str(turn.id): object()}
            o._resurrect_cold_hits(fragments)
            db.refresh(turn)
            assert turn.access_count == 1 and abs(turn.decay_score - 0.55) < 1e-9
        finally:
            db.rollback()


def test_cold_restoration_preserves_vector_and_unknown_timestamp(monkeypatch):
    from datetime import datetime, timezone

    import numpy as np
    from sqlalchemy import text

    from src.memory.models import ColdStorage

    with SessionLocal() as db:
        try:
            conv = Conversation(id=uuid.uuid4())
            db.add(conv)
            db.flush()
            monkeypatch.setattr(db, "commit", db.flush)
            monkeypatch.setattr(settings, "retrieval_strengthen_writes", True)
            # No encoder: restoration must not generate a new representation.
            o = retrieval.HybridRetrievalOrchestrator(db, None)
            for vector in ([0.125] * 1024, None):
                rid = uuid.uuid4()
                raw = "Preserve the complete source. " * 200
                cold = ColdStorage(id=rid, conversation_id=conv.id,
                    batch_id=uuid.uuid4(), raw_text=raw, summary_text="A short summary.",
                    timestamp=datetime.now(timezone.utc), embedding=vector,
                    is_private=False, ts_provenance=None)
                db.add(cold)
                db.flush()
                # Exercise the driver representation used by the actual cold leg.
                row = db.execute(text("SELECT * FROM cold_storage WHERE id=:id"),
                                 {"id": rid}).one()
                o._cold_hits = {str(rid): row}
                o._resurrect_cold_hits([retrieval.ContextFragment(raw, "episodic", 1.0,
                    100, source_batch_id=str(rid))])
                restored = db.get(EpisodicMemory, rid)
                assert restored is not None
                assert restored.raw_text == raw and restored.ts_provenance == "unknown"
                if vector is None:
                    assert restored.embedding is None
                else:
                    np.testing.assert_array_equal(restored.embedding, vector)
                assert db.execute(text("SELECT id FROM cold_storage WHERE id=:id"),
                                  {"id": rid}).first() is None
        finally:
            db.rollback()
