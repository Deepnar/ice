"""Foreground generation transport: native Ollama locally, OpenAI SSE elsewhere."""

import json
import math
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
import structlog

from src.api.config import settings

logger = structlog.get_logger("ice.api.foreground_transport")


@dataclass(frozen=True)
class GenerationRequest:
    url: str
    body: dict
    native: bool


class UpstreamGenerationError(Exception):
    def __init__(self, reason: str, status_code: int = 0,
                 prompt_tokens: int | None = None, context_tokens: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens


def build_generation_request(model: str, messages: list, prompt_tokens: int,
                             serving_window: int | None, base_url: str,
                             local_ollama: bool) -> GenerationRequest:
    body = {"model": model, "messages": messages, "stream": True}
    base = base_url.rstrip("/")
    if not local_ollama:
        if settings.token_usage_reconciliation:
            body["stream_options"] = {"include_usage": True}
        suffix = "/chat/completions" if base.endswith("/v1") else "/v1/chat/completions"
        return GenerationRequest(f"{base}{suffix}", body, False)

    if settings.ollama_send_num_ctx:
        ceiling = max(1, int(settings.ollama_num_ctx_max))
        if serving_window:
            ceiling = min(ceiling, int(serving_window))
        if settings.ollama_num_ctx_mode == "fit":
            requested = (math.ceil(prompt_tokens * settings.token_count_safety_margin)
                         + settings.context_generation_reserve)
            num_ctx = min(ceiling, requested)
        elif settings.ollama_num_ctx_mode == "max":
            requested = ceiling
            num_ctx = ceiling
        else:
            raise ValueError(f"Unknown ollama_num_ctx_mode: {settings.ollama_num_ctx_mode}")
        body["options"] = {"num_ctx": max(1, num_ctx)}
        if num_ctx < requested:
            logger.warning("ollama_num_ctx_clamped", requested=requested, selected=num_ctx,
                           ceiling=ceiling)
    else:
        logger.warning("ollama_num_ctx_disabled", model=model,
                       reason="native request has no explicit context window")
    return GenerationRequest(f"{base}/api/chat", body, True)


async def _check_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    raw = await response.aread()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        data = {}
    error = data.get("error", {}) if isinstance(data, dict) else {}
    if not isinstance(error, dict):
        error = {}
    reason = str(error.get("type") or error.get("code") or "upstream_http_error")
    raise UpstreamGenerationError(
        reason, response.status_code,
        prompt_tokens=error.get("n_prompt_tokens"),
        context_tokens=error.get("n_ctx"),
    )


def _sse(data: dict) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


async def stream_generation_response(response: httpx.Response,
                                     request: GenerationRequest,
                                     state: dict) -> AsyncIterator[str]:
    """Emit the existing SSE contract and set complete only on a true terminator."""
    await _check_status(response)
    state["complete"] = False
    if not request.native:
        pending = ""
        async for chunk in response.aiter_text():
            pending += chunk
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                if line.strip() == "data: [DONE]":
                    state["complete"] = True
            yield chunk
        if pending.strip() == "data: [DONE]":
            state["complete"] = True
        if not state["complete"]:
            raise UpstreamGenerationError("incomplete_compat_stream")
        return

    async for line in response.aiter_lines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UpstreamGenerationError("invalid_native_frame") from exc
        if not isinstance(data, dict):
            raise UpstreamGenerationError("invalid_native_frame")
        if "error" in data:
            raise UpstreamGenerationError("native_stream_error")
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(data.get("done"), bool):
            raise UpstreamGenerationError("invalid_native_frame")
        if message.get("tool_calls"):
            raise UpstreamGenerationError("native_tool_calls_unsupported")
        content = message.get("content")
        if content is None:  # thinking-only chunks can omit answer content
            content = ""
        if not isinstance(content, str):
            raise UpstreamGenerationError("invalid_native_frame")
        if content:
            yield _sse({"choices": [{"index": 0, "delta": {"content": content},
                                     "finish_reason": None}]})
        if data["done"]:
            finish = data.get("done_reason") or "stop"
            yield _sse({"choices": [{"index": 0, "delta": {},
                                     "finish_reason": finish}]})
            if finish == "length":
                raise UpstreamGenerationError("generation_length_limit")
            prompt_count = data.get("prompt_eval_count")
            eval_count = data.get("eval_count")
            if isinstance(prompt_count, int) and isinstance(eval_count, int):
                yield _sse({"choices": [], "usage": {
                    "prompt_tokens": prompt_count,
                    "completion_tokens": eval_count,
                    "total_tokens": prompt_count + eval_count,
                }})
            else:
                logger.warning("native_usage_unavailable", model=request.body["model"])
            state["complete"] = True
            yield "data: [DONE]\n\n"
            return
    raise UpstreamGenerationError("incomplete_native_stream")
