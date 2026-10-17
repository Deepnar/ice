"""v3 retrieval popularity must not manufacture codex evidence."""

import uuid

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexFact, CodexFactSource, Conversation, EpisodicMemory
from src.retrieval import orchestrator as retrieval


def test_repeated_fact_candidates_do_not_promote(monkeypatch):
    """Reading a fact never strengthens it: support counts re-extraction and
    usage counts FINAL-prompt exposure (usage.record_graph_access), never a
    candidate the leg merely returned."""
    with SessionLocal() as db:
        try:
            conv = Conversation(id=uuid.uuid4())
            db.add(conv)
            db.flush()
            turn = EpisodicMemory(conversation_id=conv.id, batch_id=uuid.uuid4(),
                raw_text="User: Atlas uses Beacon.\n\nAssistant: Noted.",
                context_reliance="Long_Term_Memory", idempotency_key=str(uuid.uuid4()))
            db.add(turn)
            db.flush()
            fact = CodexFact(text="The user said Atlas uses Beacon.", text_sha256=uuid.uuid4().hex,
                             speaker="user", modality="asserted", support_count=1, usage=0.0,
                             mentioned_at=turn.timestamp, valid_from=turn.timestamp)
            db.add(fact)
            db.flush()
            db.add(CodexFactSource(fact_id=fact.id, source_batch=turn.batch_id,
                                   window_sha256=uuid.uuid4().hex, episodic_id=turn.id,
                                   conversation_id=conv.id, window={}, quote="Atlas uses Beacon",
                                   recorded_at=turn.timestamp, verification={}))
            db.flush()
            o = retrieval.HybridRetrievalOrchestrator(db, None)
            monkeypatch.setattr(retrieval, "extract_entities", lambda *args, **kw: [])
            commits = []
            monkeypatch.setattr(db, "commit", lambda: commits.append(True))
            for _ in range(5):
                fragments = o._facts_leg("What does Atlas use Beacon?", None, None)
                assert fragments
                assert fragments[0].origin_fact_ids == (str(fact.id),)
                assert "Atlas uses Beacon" in fragments[0].text
            db.flush()
            db.refresh(fact)
            assert fact.support_count == 1 and fact.usage == 0.0
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
