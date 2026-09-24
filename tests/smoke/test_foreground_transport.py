"""G32(b): native Ollama and compatibility streams preserve one SSE contract."""

import asyncio
import json
import math
from types import SimpleNamespace

import httpx
import pytest

from src.api.config import settings
from src.api.foreground_transport import (
    UpstreamGenerationError,
    build_generation_request,
    stream_generation_response,
)
from src.api.main import _extract_usage, _parse_sse_text, store_turn_async


class _Parts(httpx.AsyncByteStream):
    def __init__(self, *parts):
        self.parts = parts

    async def __aiter__(self):
        for part in self.parts:
            yield part.encode()


def _response(*parts, status=200):
    return httpx.Response(status, request=httpx.Request("POST", "http://test/api/chat"),
                          stream=_Parts(*parts))


async def _collect(response, request):
    state = {}
    chunks = []
    async for chunk in stream_generation_response(response, request, state):
        chunks.append(chunk)
    return "".join(chunks), state


def test_native_request_sends_context_and_compat_does_not(monkeypatch):
    monkeypatch.setattr(settings, "ollama_send_num_ctx", True)
    monkeypatch.setattr(settings, "ollama_num_ctx_mode", "fit")
    messages = [{"role": "user", "content": "hello"}]
    native = build_generation_request("small", messages, 1000, 8192,
                                      "http://local:11434", True)
    assert native.url == "http://local:11434/api/chat"
    assert native.body["options"]["num_ctx"] == min(
        8192, math.ceil(1000 * settings.token_count_safety_margin)
        + settings.context_generation_reserve)
    compat = build_generation_request("cloud", messages, 1000, 8192,
                                      "https://provider.example/v1", False)
    assert compat.url == "https://provider.example/v1/chat/completions"
    assert "options" not in compat.body
    assert compat.body["stream_options"] == {"include_usage": True}

    monkeypatch.setattr(settings, "ollama_num_ctx_mode", "max")
    assert build_generation_request("small", messages, 1000, 8192,
                                    "http://local:11434", True).body["options"]["num_ctx"] == 8192
    monkeypatch.setattr(settings, "ollama_send_num_ctx", False)
    assert "options" not in build_generation_request(
        "small", messages, 1000, 8192, "http://local:11434", True).body


def test_native_stream_content_usage_and_thinking_separation():
    request = build_generation_request("small", [], 10, 4096,
                                       "http://test", True)
    # Shape captured from the running Ollama 0.32.13 /api/chat stream.
    frames = [
        {"message": {"role": "assistant", "thinking": "private reasoning",
                     "content": "OK"}, "done": False},
        {"message": {"role": "assistant", "content": "."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "done_reason": "stop", "prompt_eval_count": 33, "eval_count": 2},
    ]
    raw = "".join(json.dumps(frame) + "\n" for frame in frames)
    stream, state = asyncio.run(_collect(_response(raw[:17], raw[17:]), request))
    assert state["complete"] is True
    assert _parse_sse_text(stream)[0] == "OK."
    assert "private reasoning" not in stream
    assert _extract_usage(stream) == {
        "prompt_tokens": 33, "completion_tokens": 2, "total_tokens": 35}
    assert stream.endswith("data: [DONE]\n\n")


def test_native_missing_terminal_frame_and_http_context_error_are_loud():
    request = build_generation_request("small", [], 10, 4096,
                                       "http://test", True)
    partial = json.dumps({"message": {"role": "assistant", "content": "partial"},
                          "done": False}) + "\n"
    with pytest.raises(UpstreamGenerationError, match="incomplete_native_stream"):
        asyncio.run(_collect(_response(partial), request))
    length_stop = json.dumps({"message": {"role": "assistant", "content": "cut"},
                              "done": True, "done_reason": "length",
                              "prompt_eval_count": 5, "eval_count": 2}) + "\n"
    with pytest.raises(UpstreamGenerationError, match="generation_length_limit"):
        asyncio.run(_collect(_response(length_stop), request))
    body = json.dumps({"error": {"type": "exceed_context_size_error",
                                 "n_prompt_tokens": 6012, "n_ctx": 2048}})
    with pytest.raises(UpstreamGenerationError) as caught:
        asyncio.run(_collect(_response(body, status=400), request))
    assert (caught.value.reason, caught.value.status_code,
            caught.value.prompt_tokens, caught.value.context_tokens) == (
                "exceed_context_size_error", 400, 6012, 2048)


def test_compat_stream_requires_real_done_line():
    request = build_generation_request("cloud", [], 10, None,
                                       "https://provider.example", False)
    good = 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    stream, state = asyncio.run(_collect(
        _response(good, "data: [DO", "NE]\n\n"), request))
    assert state["complete"] is True and stream == good + "data: [DONE]\n\n"
    with pytest.raises(UpstreamGenerationError, match="incomplete_compat_stream"):
        asyncio.run(_collect(_response(good), request))


def test_failed_or_empty_generation_never_enters_postflight(monkeypatch):
    import src.api.main as main

    monkeypatch.setattr(main, "classifier", SimpleNamespace(
        embedder=SimpleNamespace(encode=lambda *_a, **_k: pytest.fail(
            "failed generation reached the embedder"))))
    kwargs = dict(correlation_id="failed", user_message="hello",
                  conversation_id=None, topic_tags=[], intent_tags=[],
                  context_reliance="Zero_Shot")
    asyncio.run(store_turn_async(**kwargs, raw_stream_segments=[[]],
                                 generation_state={"complete": False}))
    asyncio.run(store_turn_async(**kwargs, raw_stream_segments=[[
        'data: {"choices":[{"delta":{"content":""}}]}\n\n']],
        generation_state={"complete": True}))
