"""Factory for the background LLM client + model name (C7 D7/G2/G12).

Shared mode (the default): background work reuses the main Ollama server and,
when settings.background_model_name is None, the chat model itself (registry
fallback) — no second model in VRAM. Dedicated mode is the power-user path: a
separate OpenAI-compatible server on :8002, started manually (./ice no longer
launches it).
"""

from openai import OpenAI

from src.api.config import settings

# Dedicated-mode default when background_model_name is unset — matches the
# documented manual vLLM invocation in ./ice.
DEDICATED_DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
DEDICATED_BASE_URL = "http://localhost:8002/v1"


def get_bg_client() -> OpenAI:
    """Return an OpenAI-compatible client for the background model."""
    if settings.background_model_mode == "shared":
        base_url = f"{settings.ollama_base_url}/v1"
    else:
        base_url = DEDICATED_BASE_URL
    return OpenAI(base_url=base_url, api_key="dummy")


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
