"""v3 control integrity through real shared preparation and PostgreSQL retrieval.

Run with tests/support/disposable_database.py. Fixed embeddings, disabled
source judge/reranker and empty non-vector legs isolate the temporal and
wide-net paths; this is not an answer-quality or model-accuracy measurement.
"""
import sys
import uuid
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import event

from scripts.z1 import seed_v3, snapshot
from scripts.z1.production_parity import Preamble
from src.api.config import settings
from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import Conversation, EpisodicMemory
from src.model_registry import registry
from src.retrieval.orchestrator import HybridRetrievalOrchestrator


def main():
    seed_v3.isolated_database()
    vector = [1.0] + [0.0] * 1023
    classifier = SimpleNamespace(embedder=SimpleNamespace(encode=lambda *_a, **_k: vector))
    source_time = datetime(2025, 2, 1, tzinfo=timezone.utc)
    temporal = {"mode": "range", "t0": "2025-01-01T00:00:00+00:00",
                "t1": "2025-01-31T23:59:59+00:00"}
    captured = []
    with SessionLocal() as db:
        conversation = Conversation()
        db.add(conversation)
        db.flush()
        db.add(EpisodicMemory(conversation_id=conversation.id, batch_id=uuid.uuid4(),
            idempotency_key="synthetic-control-source", raw_text="User: Project Nimbus used port 7813.",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc), embedding=vector,
            context_reliance="Long_Term_Memory", inject_raw=True,
            lossless_flag=True, decay_score=1.0))
        db.commit()
        def query_seen(_conn, _cursor, statement, _params, _context, _many):
            if "<=>" in statement:
                captured.append(statement)
        event.listen(db.get_bind(), "before_cursor_execute", query_seen)
        try:
            with (patch.object(settings, "memory_source_gate_enabled", False),
                  patch.object(settings, "context_use_serving_window", False),
                  patch.object(registry, "get_model_context_window", lambda *_: 8192),
                  patch.object(settings, "retrieval_rerank_enabled", False),
                  patch.object(settings, "confidence_fallback_threshold", 0.5),
                  patch.object(HybridRetrievalOrchestrator, "_extract_prompt_keywords", lambda *_: set())):
                # Non-vector model-dependent legs are outside this fixture.
                with ExitStack() as stack:
                    for method in ("_bm25_episodic", "_codex_graph", "_codex_claims",
                                   "_procedural_lookup", "_batch_summary_lookup", "_cold_lookup"):
                        stack.enter_context(patch.object(HybridRetrievalOrchestrator, method, lambda *_a, **_k: []))
                    for confidence, timescope in ((0.9, temporal), (0.0, None)):
                        classification = ClassificationResult(
                            topic_tags=["Science_&_Technology"], intent_tags=["Factual_Retrieval"],
                            context_reliance="Long_Term_Memory", raw_probs=[], max_confidence=confidence,
                            head_confidences={"topic": confidence, "intent": confidence},
                            prompt="Which port did Project Nimbus use?", p_ltm=1.0)
                        scope = {"conversation_id": str(conversation.id)}
                        if timescope:
                            scope["timescope"] = timescope
                        pre = Preamble(classification, scope, vector, str(conversation.id),
                                       1, 12, 6000, "synthetic-control", "range" if timescope else "current", True, 1.0)
                        before = snapshot.fingerprints()
                        results = {}
                        for arm in seed_v3.PROMPT_ARMS:
                            calls_before = len(captured)
                            with seed_v3.probe_observation(db):
                                results[arm] = seed_v3.prepare_with_trace(
                                    db, deepcopy(pre), classifier, arm=arm, source_time=source_time)
                            if arm == "recent_only":
                                assert len(captured) == calls_before, "recent control performed a store similarity query"
                        assert snapshot.fingerprints() == before, "control observation changed the memory store"
                        assert not results["recent_only"][0].retrieve
                        assert all(not rows for rows in results["recent_only"][1:])
                        assert not results["recent_only"][0].fragments
                        if timescope:
                            for arm in ("full", "no_codex"):
                                fragments = results[arm][0].fragments
                                assert len(fragments) == 1 and "Closest matches" in fragments[0].text
                                assert "2024-01" in fragments[0].text
                            assert not results["vector_only"][0].fragments
                        else:
                            assert {f.leg for f in results["full"][0].fragments} == {"fallback"}
                            assert {f.leg for f in results["no_codex"][0].fragments} == {"fallback"}
                            assert {f.leg for f in results["vector_only"][0].fragments} == {"vector"}
                            assert results["vector_only"][1]["vector"]
                        assert settings.confidence_fallback_threshold == 0.5
        finally:
            event.remove(db.get_bind(), "before_cursor_execute", query_seen)
    print("v3 controls: temporal nearest-era and low-confidence wide-net paths checked through shared preparation; "
          "recent performs no similarity search, vector stays on its leg, full/no-Codex behavior and store unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
