"""The real chat handler uses final prepared evidence for rescue and skip."""
import asyncio
import os
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks

from scripts.classifier.pipeline.style_variants import _source_context
from src.api import main, memory_preparation
from src.api.config import settings
from src.api.db import SessionLocal
from src.api.source_need import SourceNeed
from src.classifier.schemas import ClassificationResult
from src.memory.models import Conversation, EpisodicMemory
from src.retrieval.orchestrator import ContextFragment


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"), reason="disposable DB required")
@pytest.mark.parametrize("mode", ["rescue", "withhold", "skip"])
def test_actual_handler_source_decisions(monkeypatch, mode):
    monkeypatch.setattr(settings, "memory_source_gate_enabled", True)
    monkeypatch.setattr(settings, "memory_source_rescue_enabled", True)
    monkeypatch.setattr(main, "core", None)
    monkeypatch.setattr(main, "classifier", SimpleNamespace(
        classify=lambda *_a, **_k: ClassificationResult([], [], "Zero_Shot", [], .99, p_ltm=.01),
        embedder=SimpleNamespace(encode=lambda *_a, **_k: [1.0] + [0.0] * 1023)))
    monkeypatch.setattr(main, "decide_memory_retrieval", lambda *_a, **_k:
                        SimpleNamespace(retrieve=mode == "skip", breakdown={}))
    monkeypatch.setattr(main, "get_model_context_window", lambda *_a: 8192)
    monkeypatch.setattr(main, "serving_window", lambda *_a: 8192)
    monkeypatch.setattr(main, "log_window_truth", lambda *_a: None)
    monkeypatch.setattr(main, "record_graph_access", lambda *_a, **_k: None)
    exposed, judged = [], []
    fragment = ContextFragment("User: For this setup I checked GPU memory with gpu-scan --mem --device 7.", "episodic", 1, 25)
    class Orchestrator:
        def __init__(self, *_a): self.recent_token_budget = 1000
        def set_budget_from_turn_count(self, *_a, **_k): pass
        def retrieve(self, **kw):
            assert kw["defer_exposure"] is True
            return [fragment]
        def record_exposure(self, fragments): exposed.extend(fragments)
    monkeypatch.setattr(memory_preparation, "HybridRetrievalOrchestrator", Orchestrator)
    def judge(messages, *, current_only=False):
        assert current_only == (mode == "skip")
        judged.append(messages)
        if mode == "skip": return SourceNeed("visible_evidence", "Our port is 7813.")
        if len(judged) == 1: return SourceNeed("older_memory")
        return (SourceNeed("visible_evidence", "gpu-scan --mem --device 7")
                if mode == "rescue" else SourceNeed("general_knowledge"))
    monkeypatch.setattr(memory_preparation, "judge_source_need", judge)
    async def run(db, conversation_id):
        class Request:
            headers = {"X-ICE-Conversation-ID": str(conversation_id)}
            async def json(self):
                prompt = ("Our port is 7813. Which port?" if mode == "skip"
                          else "Which exact command did I run to check GPU memory during our setup?")
                return dict(model="controlled-local", messages=[dict(role="user", content=prompt)])
        response = await main.chat_completions(Request(), BackgroundTasks(), db)
        events = []
        # Preflight events precede upstream generation; no fake answer is needed.
        for _ in range(3): events.append(await anext(response.body_iterator))
        await response.body_iterator.aclose()
        return "".join(events)
    with SessionLocal() as db:
        conversation_id, _, _ = _source_context(db)
        try:
            events = asyncio.run(run(db, conversation_id))
        finally:
            db.rollback()
            db.query(EpisodicMemory).filter_by(conversation_id=conversation_id).delete()
            db.query(Conversation).filter_by(id=conversation_id).delete()
            db.commit()
    if mode == "rescue":
        assert len(judged) == 2 and exposed == [fragment]
        assert "gpu-scan" not in "\n".join(m["content"] for m in judged[0])
        assert "gpu-scan" in "\n".join(m["content"] for m in judged[1])
        assert '"fragments_count": 1' in events
    else:
        assert not exposed and '"fragments_count": 0' in events
    if mode == "skip": assert len(judged) == 1
