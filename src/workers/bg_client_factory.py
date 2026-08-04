"""Factory for the background LLM client + model name (C7 D7/G2/G12).

Shared mode (the default): background work reuses the main Ollama server and,
when settings.background_model_name is None, the chat model itself (registry
fallback) — no second model in VRAM. Dedicated mode is the power-user path: a
separate OpenAI-compatible server on :8002, started manually (./ice no longer
launches it).
"""

import structlog
from openai import OpenAI

from src.api.config import settings

logger = structlog.get_logger("ice.workers.bg_client")

# Dedicated-mode default when background_model_name is unset — matches the
# documented manual vLLM invocation in ./ice.
DEDICATED_DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
DEDICATED_BASE_URL = "http://localhost:8002/v1"


class _NoReasoningCompletions:
    """Adds `reasoning_effort="none"` to every background completion.

    This lives at the factory rather than at the ~17 call sites deliberately:
    the parameter is not a property of any one task, it is a property of *how
    ICE uses a chat model for structured background work*, and 17 copies of it
    is exactly the drift G29 catalogues. It is also the seam G32(a) needs — when
    background calls move to Ollama's native endpoint, this class becomes the
    place that translates, and no caller changes.

    Why it exists at all (measured 2026-08-03, see PROVENANCE): a reasoning
    model spends the whole `max_tokens` budget inside a thinking block that
    Ollama strips before returning, so `content` is `""` and ICE's entire
    background layer silently produced nothing. A caller that genuinely wants
    reasoning passes its own `extra_body={"reasoning_effort": ...}`, which is
    respected.
    """

    def __init__(self, inner):
        self._inner = inner

    def create(self, *args, **kwargs):
        extra = dict(kwargs.pop("extra_body", None) or {})
        extra.setdefault("reasoning_effort", "none")
        try:
            completion = self._inner.create(*args, extra_body=extra, **kwargs)
        except Exception as exc:
            # G32(a): a server that cannot do constrained decoding must SAY so.
            # Silently dropping a capability it does not understand is exactly
            # what Ollama's /v1 shim does to us; repeating it one layer up is
            # what this item exists to prevent. Retry unconstrained so the
            # caller still gets an answer, but never quietly.
            if kwargs.get("response_format") is None or not _looks_like_schema_refusal(exc):
                raise
            logger.warning(
                "bg_schema_unsupported_retrying_unconstrained",
                model=kwargs.get("model"),
                response_format=str(kwargs.get("response_format"))[:120],
                error=str(exc)[:200],
            )
            kwargs.pop("response_format", None)
            completion = self._inner.create(*args, extra_body=extra, **kwargs)
        # A fallback must be observable (CLAUDE.md standing rule): an empty
        # answer is the exact shape of the outage this parameter exists to
        # prevent, so say so rather than letting each caller's default hide it.
        try:
            choice = completion.choices[0]
            if not (choice.message.content or "").strip():
                logger.warning(
                    "bg_model_returned_empty_content",
                    model=kwargs.get("model"),
                    finish_reason=choice.finish_reason,
                    max_tokens=kwargs.get("max_tokens"),
                    reasoning_chars=len(
                        (choice.message.model_extra or {}).get("reasoning") or ""),
                )
            elif choice.finish_reason == "length":
                # G32(a), measured 2026-08-04: at the production max_tokens=500
                # this fired on 2 of 12 turns and the JSON was unparseable every
                # time. Constrained decoding guarantees a valid *prefix*, not a
                # complete document — so the schema does NOT make this go away,
                # and every caller's salvage path reads a truncated answer as a
                # thin one. `finish_reason` was on every response all along and
                # nothing in src/ read it.
                logger.warning(
                    "bg_model_output_truncated",
                    model=kwargs.get("model"),
                    max_tokens=kwargs.get("max_tokens"),
                    content_chars=len(choice.message.content or ""),
                    constrained=kwargs.get("response_format") is not None,
                )
        except (AttributeError, IndexError, TypeError):
            pass
        return completion

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _Chat:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _NoReasoningCompletions(inner.completions)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _BgClient:
    def __init__(self, inner: OpenAI):
        self._inner = inner
        self.chat = _Chat(inner.chat)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _looks_like_schema_refusal(exc) -> bool:
    """Did the server reject us *for the schema*, or for something else?

    Only a refusal of the constraint may be retried without it — retrying a
    timeout or an out-of-memory unconstrained would silently downgrade output
    quality for an unrelated reason.
    """
    status = getattr(exc, "status_code", None)
    if status is not None and status not in (400, 415, 422, 501):
        return False
    blob = str(getattr(exc, "message", "") or exc).lower()
    return any(k in blob for k in
               ("response_format", "json_schema", "schema", "format",
                "unsupported", "not supported", "unrecognized"))


def json_schema(name: str, schema: dict) -> dict:
    """Build the `response_format` for JSON-schema constrained decoding.

    One constructor rather than a copy at each call site — the nesting is easy
    to get subtly wrong, and getting it wrong is SILENT: measured 2026-08-04,
    `{"type": "json_object"}` (the shape `maintenance_agent` used to send) is
    honoured by nobody and scored 0/8, exactly as if no constraint had been
    sent at all, while `{"type": "json_schema", ...}` scored 8/8 through the
    very same endpoint. Both return 200. See PROVENANCE, the G32 audit.

    Works on Ollama's OpenAI-compatible endpoint AND on vLLM/SGLang, so it
    needs no transport branch; a server that refuses it is logged and retried
    unconstrained by `_NoReasoningCompletions`.
    """
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def get_bg_client():
    """Return an OpenAI-compatible client for the background model.

    Wrapped so structured background work never loses its answer to a hidden
    reasoning block (`bg_disable_reasoning`, default on).
    """
    if settings.background_model_mode == "shared":
        base_url = f"{settings.ollama_base_url}/v1"
    else:
        base_url = DEDICATED_BASE_URL
    client = OpenAI(base_url=base_url, api_key="dummy")
    if not settings.bg_disable_reasoning:
        return client
    return _BgClient(client)


def get_bg_model_name() -> str:
    """settings.background_model_name wins; None ⇒ the chat model (shared)
    or the dedicated default."""
    if settings.background_model_name:
        return settings.background_model_name
    if settings.background_model_mode == "shared":
        # Lazy import: registry is a JSON read, but keep the dependency out
        # of module import time for the pure-logic tests.
        from src.model_registry.registry import get_fallback_model
        return get_fallback_model()
    return DEDICATED_DEFAULT_MODEL


def bg_timeout(max_tokens: int) -> float:
    """G12: scale the request timeout with the asked-for output size —
    base × clamp(max_tokens/500, 1, 6). Retries ride the maintenance
    runtime's backoff, not per-call loops."""
    return settings.bg_timeout_base_seconds * min(max(max_tokens / 500.0, 1.0), 6.0)
