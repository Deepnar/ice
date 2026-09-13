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
            monkeypatch.setattr(o, "_relation_facts", lambda *args: ([], [], 0.0))
            monkeypatch.setattr(settings, "codex_max_depth", 0)
            commits = []
            monkeypatch.setattr(db, "commit", lambda: commits.append(True))
            classification = SimpleNamespace(prompt="What does Atlas use?", intent_tags=[])
            for _ in range(5):
                assert o._codex_graph(classification, None)
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
