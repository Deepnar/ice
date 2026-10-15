"""G32(a)/G4: background generation over Ollama's own /api/chat (2026-10-15).

Background callers keep speaking the OpenAI SDK's `chat.completions.create`
shape; the factory routes the call here in shared mode, and this module
translates it to the native request and the answer back to a ChatCompletion.

Why it exists — measured on Ollama 0.32.13, 2026-10-15:
  * the OpenAI-compatible /v1 door drops `num_ctx`, so background work runs at
    whatever the host loaded (32,768 on a 24 GB card). A ~50K-token input came
    back HTTP 200 / "stop" with its head silently dropped, and the model
    INVENTED the dropped fact. 11 of history C's 325 turns exceed that window
    in the turn summariser alone.
  * the native door does not refuse an oversized input either (a 69K-token
    prompt at num_ctx 65,536 kept 32,771 tokens, HTTP 200). So ICE counts the
    input itself before sending and checks Ollama's own count afterwards.

What it is NOT: not a model change and not a prompt change. The same model
reads the same messages; only the door, the context size and the residency
instruction differ. Dedicated mode (vLLM/SGLang) never comes here.
"""

import math
import time

import httpx
import structlog
from openai.types.chat import ChatCompletion

from src.api.config import settings
from src.workers.completion_text import IncompleteCompletion

logger = structlog.get_logger("ice.workers.bg_native_transport")

# kwargs the translation understands; anything else is reported, not dropped.
_KNOWN = {"model", "messages", "temperature", "max_tokens", "timeout",
          "response_format", "extra_body"}
# Truncation keeps about half the window, so a real cut reads far below the
# estimate. The embedder's tokenizer is not the generator's (measured up to
# ~16% apart on code), so the threshold sits well clear of that disagreement.
_TRUNCATION_RATIO = 0.7


class NativeCallError(Exception):
    """An HTTP refusal, shaped like the SDK's errors for the schema-retry path."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def use_native() -> bool:
    return (settings.background_model_mode == "shared"
            and settings.background_transport in ("auto", "ollama"))


def estimate_input_tokens(messages) -> int:
    from src.memory.tokens import count_messages
    return count_messages(messages)


def model_maximum(model):
    from src.model_registry.runtime_probe import observed_context_window
    return observed_context_window(model)["gguf_max"]


def fit_num_ctx(required: int, maximum) -> int:
    """Smallest floor·2^k that holds *required*, capped by the model maximum.

    Doubling rather than an exact fit: every distinct num_ctx makes Ollama
    reload the runner (~4 s measured), so sizes are bucketed and the common
    case stays on the floor the host already loads.
    """
    size = settings.bg_num_ctx_floor
    while size < required:
        size *= 2
    if maximum:
        size = min(size, int(maximum))
    return size


def check_prompt_count(model, estimated, actual, *, transport, num_ctx=None):
    """Raise when the server read far less input than ICE sent.

    A silently cut input produces a confident answer about text the model never
    saw (the 2026-10-15 probe invented a fact), so the call fails loudly and the
    caller's existing failure path runs instead of storing invented content.
    """
    if type(actual) is not int or estimated <= 0:
        logger.warning("bg_prompt_count_unavailable", model=model, transport=transport,
                       estimated=estimated)
        return
    if actual < estimated * _TRUNCATION_RATIO:
        logger.warning("bg_input_truncated", model=model, transport=transport,
                       estimated=estimated, server_prompt_tokens=actual, num_ctx=num_ctx,
                       reason="the server read far less input than was sent")
        raise IncompleteCompletion("background input was truncated by the server",
                                   finish_reason="input_truncated",
                                   details={"estimated": estimated, "prompt_tokens": actual})


def _format(response_format):
    if not response_format:
        return None
    kind = response_format.get("type")
    if kind == "json_schema":
        return (response_format.get("json_schema") or {}).get("schema")
    if kind == "json_object":
        return "json"
    logger.warning("bg_native_response_format_unknown", response_format=str(response_format)[:120])
    return None


def chat_completion(**kwargs):
    """One background completion over /api/chat, returned as a ChatCompletion."""
    from src.workers.bg_client_factory import native_call_timings
    unknown = sorted(set(kwargs) - _KNOWN)
    if unknown:
        logger.warning("bg_native_params_not_translated", params=unknown,
                       model=kwargs.get("model"))
    model, messages = kwargs["model"], kwargs["messages"]
    max_tokens = int(kwargs.get("max_tokens") or 0)
    extra = dict(kwargs.get("extra_body") or {})
    estimated = estimate_input_tokens(messages)
    required = math.ceil(estimated * settings.token_count_safety_margin) + max_tokens
    maximum = model_maximum(model)
    if maximum and required > maximum:
        logger.warning("bg_input_over_model_window", model=model, required=required,
                       model_maximum=maximum,
                       reason="input plus output cannot fit the model at any context size")
        raise IncompleteCompletion("background input exceeds the model's maximum context",
                                   finish_reason="input_over_model_window",
                                   details={"required": required, "maximum": maximum})
    num_ctx = fit_num_ctx(required, maximum)
    options = {"num_ctx": num_ctx}
    if kwargs.get("temperature") is not None:
        options["temperature"] = kwargs["temperature"]
    if max_tokens:
        options["num_predict"] = max_tokens
    body = {"model": model, "messages": messages, "stream": False,
            "keep_alive": settings.bg_keep_alive, "options": options}
    effort = extra.pop("reasoning_effort", None)
    if effort is not None:
        body["think"] = effort != "none"
    if extra:
        logger.warning("bg_native_params_not_translated", params=sorted(extra), model=model)
    fmt = _format(kwargs.get("response_format"))
    if fmt is not None:
        body["format"] = fmt
    timeout = (float(kwargs.get("timeout") or settings.bg_timeout_base_seconds)
               + estimated / 1000.0 * settings.bg_timeout_seconds_per_1k_input)
    response = httpx.post(settings.ollama_base_url.rstrip("/") + "/api/chat",
                          json=body, timeout=timeout)
    if response.status_code >= 400:
        try:
            message = str(response.json().get("error"))
        except ValueError:
            message = response.text[:300]
        raise NativeCallError(response.status_code, message)
    data = response.json()
    reason = data.get("done_reason")
    if data.get("done") is not True or reason not in ("stop", "length"):
        raise IncompleteCompletion(f"native background response incomplete: {reason!r}",
                                   finish_reason=reason)
    actual, output = data.get("prompt_eval_count"), data.get("eval_count")
    logger.info("bg_native_call", model=model, num_ctx=num_ctx, estimated_input=estimated,
                prompt_tokens=actual, completion_tokens=output, finish_reason=reason,
                **native_call_timings(data))
    check_prompt_count(model, estimated, actual, transport="ollama", num_ctx=num_ctx)
    usage = None
    if type(actual) is int and type(output) is int:
        usage = dict(prompt_tokens=actual, completion_tokens=output, total_tokens=actual + output)
    message = dict(role="assistant", content=(data.get("message") or {}).get("content"))
    thinking = (data.get("message") or {}).get("thinking")
    if thinking:
        message["reasoning"] = thinking
    return ChatCompletion(id="ice-bg-native", created=int(time.time()), model=model,
                          object="chat.completion", usage=usage,
                          choices=[dict(index=0, finish_reason=reason, message=message)])
