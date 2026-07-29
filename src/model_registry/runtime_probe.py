"""C16 — what context window is the model server ACTUALLY giving us?

ICE derives a token budget from `models/model_registry.json`'s
`context_window` and has never checked it against reality. Measured on this
machine for one loaded model, three sources disagreed three ways:

    registry.json          64,000
    Ollama's live runner   32,768      <- the number that decides truncation
    the GGUF's own max    262,144
    derive_total_budget    40,000      <- 22% ABOVE what the server allocated

Nothing reserved room for the answer either, and ICE never sends `num_ctx`, so
the server applies its own default and truncates silently. Truncation keeps the
newest messages, which means the most likely casualty is the retrieved-memory
block ICE just spent a request building. The only reason this has not been
catastrophic is that `growth_cap` accidentally kept retrieval tiny.

This module reads the truth. It changes no decision on its own — it exists so
that the budget can be reconciled against reality before anything starts
trusting it, and so the reconciliation is a logged number rather than an
argument.
"""

import time

import httpx
import structlog

from src.api.config import settings

logger = structlog.get_logger("ice.registry.runtime_probe")

PROBE_TIMEOUT = 2.0          # never block a request on the probe
CACHE_TTL_SECONDS = 300      # a loaded runner's window does not move often

_cache: dict = {}            # model_name -> (expires_at, dict)


def _fresh(model_name: str):
    hit = _cache.get(model_name)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    return None


def observed_context_window(model_name: str) -> dict:
    """What the serving stack says about *model_name*'s window.

    Returns ``{"runtime": int|None, "gguf_max": int|None, "source": str}``.

    ``runtime`` is the live runner's allocated context (``/api/ps``) — the
    number that actually governs truncation, and the one to budget against.
    ``gguf_max`` is the architecture's ceiling (``/api/show``) — the most you
    could ask for with ``num_ctx``. Both are None when the server is
    unreachable; callers fall back to the registry and say so.
    """
    cached = _fresh(model_name)
    if cached is not None:
        return cached

    out = {"runtime": None, "gguf_max": None, "source": "unavailable"}
    base = settings.ollama_base_url.rstrip("/")
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            ps = client.get(f"{base}/api/ps")
            if ps.status_code == 200:
                for m in (ps.json().get("models") or []):
                    if m.get("model") == model_name or m.get("name") == model_name:
                        out["runtime"] = m.get("context_length")
                        out["source"] = "api/ps"
                        break
            show = client.post(f"{base}/api/show", json={"model": model_name})
            if show.status_code == 200:
                info = show.json().get("model_info") or {}
                # The key is architecture-prefixed (gemma4.context_length,
                # qwen3.context_length, ...) — read whatever arch this is
                # rather than maintaining a list of prefixes that rots.
                for k, v in info.items():
                    if k.endswith(".context_length") and isinstance(v, int):
                        out["gguf_max"] = v
                        break
                if out["source"] == "unavailable":
                    out["source"] = "api/show"
    except Exception as exc:
        logger.warning("context_window_probe_failed", model=model_name,
                       error=str(exc))

    _cache[model_name] = (time.monotonic() + CACHE_TTL_SECONDS, out)
    return out


def log_window_truth(log, model_name: str, registry_window, budget_derived: int) -> dict:
    """Emit the quadruple, and return it for the ledger.

    One log line per request is the whole point: it is what turns "the budget
    is probably fine" into a number somebody can read a week later, and it is
    what tells us whether the registry is lying about any given model.
    """
    probe = observed_context_window(model_name)
    truth = {
        "registry": registry_window,
        "runtime_ps": probe["runtime"],
        "gguf_max": probe["gguf_max"],
        "budget_derived": budget_derived,
        "probe_source": probe["source"],
    }
    # The condition that matters: we are about to build a prompt for a window
    # smaller than the one we budgeted against. Check EVERY ceiling we can
    # see, not just the live runner — the first real request this shipped
    # against proved why: `tinyllama` is registered at 8,192, its GGUF maximum
    # is 2,048, and the derived budget was 6,144. Nothing was resident, so a
    # runtime-only check reported `over_serving_window=False` about a budget
    # 3x the model's architectural ceiling.
    ceilings = [c for c in (probe["runtime"], probe["gguf_max"]) if c]
    truth["over_serving_window"] = bool(ceilings and budget_derived > min(ceilings))
    if truth["over_serving_window"]:
        truth["ceiling"] = min(ceilings)
        log.warning("context_budget_exceeds_serving_window", **truth)
    else:
        log.info("context_window_truth", **truth)
    return truth


def serving_window(model_name: str, registry_window):
    """The window to actually budget against.

    The live runner's allocation if visible, else the registry's claim — and
    in both cases clamped by the GGUF's architectural maximum, because the
    registry can and does over-claim (measured: 8,192 registered against a
    2,048 ceiling). Explicit function so no caller has to remember the
    precedence.
    """
    probe = observed_context_window(model_name)
    window = probe["runtime"] or registry_window
    if probe["gguf_max"]:
        window = min(window, probe["gguf_max"]) if window else probe["gguf_max"]
    return window
