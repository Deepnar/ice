"""Factory that returns the background LLM client based on settings."""

from openai import OpenAI
from src.api.config import settings

# def get_bg_client() -> OpenAI:
#     """Return an OpenAI-compatible client for the background model."""
#     if settings.background_model_mode == "shared":
#         base_url = f"{settings.ollama_base_url}/v1"
#     else:
#         base_url = "http://localhost:8002/v1"
#     return OpenAI(base_url=base_url, api_key="dummy")

# def get_bg_model_name() -> str:
#     if settings.background_model_mode == "shared":
#         return "qwen3:4b-instruct-bg"   # ← new line
#     else:
#         return "Qwen/Qwen2.5-3B-Instruct-AWQ"

"""Factory that returns the background LLM client based on settings.
   For the ablation runs the background model is the same SGLang‑served
   Qwen3‑14B‑AWQ as the experiment probes, with thinking explicitly
   DISABLED so structured‑JSON extraction works reliably.
"""

from openai import OpenAI
from src.api.config import settings

# ── Wrappers that force non‑thinking mode on every chat call ─────────────

class _NonThinkingCompletions:
    """Inject enable_thinking=False into every request."""
    def __init__(self, completions):
        self._c = completions

    def create(self, **kwargs):
        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        return self._c.create(**kwargs)


class _NonThinkingChat:
    def __init__(self, chat):
        self.completions = _NonThinkingCompletions(chat.completions)


class _NonThinkingClient:
    """Drop‑in replacement for OpenAI client that always runs in non‑thinking mode."""
    def __init__(self, base_url):
        self._client = OpenAI(base_url=base_url, api_key="dummy")

    @property
    def chat(self):
        return _NonThinkingChat(self._client.chat)


def get_bg_client():
    """Return a client that talks to the SGLang experiment server and
    disables thinking mode so the model returns clean JSON immediately."""
    return _NonThinkingClient("http://localhost:8001/v1")


def get_bg_model_name() -> str:
    """Return the model name for background tasks."""
    return "Qwen/Qwen3-14B-AWQ"