"""G32(b): the actual chat route selects native Ollama and stores only success."""

import asyncio
import json
import os
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import BackgroundTasks

from src.api import main
from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import Conversation, EpisodicMemory


class _Request:
    def __init__(self, conversation_id, model="controlled-local"):
        self.headers = {"X-ICE-Conversation-ID": str(conversation_id)}
        self.model = model

    async def json(self):
        return {"model": self.model, "messages": [
            {"role": "user", "content": "What is one plus one?"}]}


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
def test_route_native_stream_and_failure_postflight(monkeypatch):
    model_vector = [1.0] + [0.0] * 1023
    enqueued = []
    runtime = SimpleNamespace(
        note_user_activity=lambda: None,
        generation_started=lambda: None,
        generation_finished=lambda: None,
        notify_work_unit=lambda *_a, **_k: None,
        enqueue=lambda name, **kw: enqueued.append((name, kw)),
    )
    monkeypatch.setattr(main, "core", SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(main, "classifier", SimpleNamespace(
        classify=lambda *_a, **_k: ClassificationResult(
            [], [], "Zero_Shot", [], 0.99, p_ltm=0.01),
        embedder=SimpleNamespace(encode=lambda *_a, **_k: model_vector)))
    monkeypatch.setattr(main, "decide_memory_retrieval", lambda *_a, **_k:
                        SimpleNamespace(retrieve=False, breakdown={}))
    monkeypatch.setattr(main, "get_model_context_window", lambda *_a: 8192)
    monkeypatch.setattr(main, "serving_window", lambda *_a: 8192)
    monkeypatch.setattr(main, "log_window_truth", lambda *_a: None)
    monkeypatch.setattr(main, "get_fallback_model", lambda: "controlled-local")
    monkeypatch.setattr(main, "conversation_summary_block", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "record_graph_access", lambda *_a, **_k: None)

    class _Ledger:
        def fits(self):
            return True

        def log(self, _log):
            pass

    monkeypatch.setattr(main, "assemble_budgeted_prompt", lambda **_k:
                        SimpleNamespace(messages=[{"role": "user", "content": "What is one plus one?"}],
                                        ledger=_Ledger(), removed=[],
                                        item_counts={"slots": 0, "bookmarks": 0}))

    requests = []
    status = {"code": 200}
    native_ok = "".join(json.dumps(frame) + "\n" for frame in [
        {"message": {"role": "assistant", "content": "Two."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 2},
    ])

    def handler(request):
        requests.append(request)
        if status["code"] == "external_timeout" and request.url.host == "provider.example":
            raise httpx.ReadTimeout("controlled timeout", request=request)
        if status["code"] == 400:
            return httpx.Response(400, json={"error": {
                "type": "exceed_context_size_error", "n_prompt_tokens": 9000,
                "n_ctx": 8192}})
        return httpx.Response(200, text=native_ok)

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(main.httpx, "AsyncClient", lambda **kw:
                        real_async_client(transport=transport, **kw))

    async def run(conversation_id, model="controlled-local"):
        tasks = BackgroundTasks()
        with SessionLocal() as db:
            response = await main.chat_completions(_Request(conversation_id, model), tasks, db)
            stream = "".join([chunk async for chunk in response.body_iterator])
        await tasks()
        return stream

    success_id, failed_id, fallback_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    success = asyncio.run(run(success_id))
    assert "Two." in success and "data: [DONE]" in success
    assert requests[-1].url.path == "/api/chat"
    assert json.loads(requests[-1].content)["options"]["num_ctx"] > 0
    with SessionLocal() as db:
        turn = db.query(EpisodicMemory).filter_by(conversation_id=success_id).one()
        assert turn.raw_text.endswith("Assistant: Two.")

    status["code"] = 400
    failure = asyncio.run(run(failed_id))
    assert "exceed_context_size_error" in failure
    assert len(enqueued) == 1
    with SessionLocal() as db:
        assert db.query(EpisodicMemory).filter_by(conversation_id=failed_id).count() == 0
    monkeypatch.setattr(main, "find_best_model", lambda *_a: (
        "cloud-answer", "https://provider.example"))
    status["code"] = "external_timeout"
    fallback = asyncio.run(run(fallback_id, "ice-proxy"))
    assert "primary_model_timeout" in fallback and "Two." in fallback
    assert [req.url.path for req in requests[-2:]] == [
        "/v1/chat/completions", "/api/chat"]
    assert requests[-1].url.host == "localhost"
    assert enqueued[-1][1]["model_used"] == "controlled-local"
    with SessionLocal() as db:
        assert db.query(EpisodicMemory).filter_by(conversation_id=fallback_id).count() == 1
        db.query(EpisodicMemory).filter_by(conversation_id=success_id).delete()
        db.query(EpisodicMemory).filter_by(conversation_id=fallback_id).delete()
        db.query(Conversation).filter(Conversation.id.in_([
            success_id, failed_id, fallback_id])).delete()
        db.commit()
