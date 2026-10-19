"""Z2 cloud model roles (spec Z2_EVALUATION D6), all on OpenCode Go.

Profiles are defined here rather than in `experiments/lme/cloud_provider.py`,
whose bytes are fingerprinted by frozen tuning and judging bundles (TRAPS 131).
The free Zen variants are refused to API clients (D6b, 2026-10-17), so none
appear here. Every call records the profile's public metadata.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.lme.cloud_provider import PROFILES, ProviderProfile


def _go(name, model):
    return ProviderProfile(name=name, endpoint="chat_completions", model=model,
                           base_url_env="PROBE_API_BASE_URL", api_key_env="PROBE_API_KEY")


Z2_PROFILES = {
    "deepseek": PROFILES["opencode-deepseek-v41-flash"],
    "mimo": _go("opencode-mimo26-flash", "mimo-v2.6-flash"),
    "longcat": PROFILES["opencode-longcat20"],
    "luna": PROFILES["opencode-luna6"],
    "muse": PROFILES["opencode-muse13"],
}

# D6: the two ledger/question writer families; each verifies the other. Muse 1.3
# breaks ties (maintainer, 2026-10-18: on 34 reviewed dev disputes Muse matched the
# maintainer's keep/drop 29/34 and erred strict; LongCat 28/34, erred lenient).
# Writers never answer or judge.
WRITERS = ("deepseek", "mimo")
TIE_BREAK = "muse"


# Writers run with thinking off. With it left to the model, DeepSeek V4.1 Flash
# once spent a whole 16,000-token reply thinking and returned no text (dev window
# 1, 2026-10-17); a run before it on the same window had not thought at all.
# `reasoning_effort: "none"` turns it off on both writer families (measured).
# JSON mode (2026-10-18): MiMo closed long replies with a missing or extra brace —
# 14 unparseable replies on history B, 4 windows lost after the retry; replies are
# never repaired, so the provider is asked for valid JSON instead (both accept it).
WRITER_EXTRA_BODY = {"reasoning_effort": "none", "response_format": {"type": "json_object"}}


class _Injecting:
    """Adds fixed request fields to every chat completion (TextGenerator has no
    hook for them, and its module is fingerprinted, so it is not edited)."""

    def __init__(self, client, extra_body):
        self._client, self._extra = client, extra_body
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        kwargs["extra_body"] = {**self._extra, **(kwargs.get("extra_body") or {})}
        return self._client.chat.completions.create(**kwargs)


def _thinking_off(role):
    """Writers, and any chat-completions tie-breaker, run with thinking off. Muse
    is served on the Responses API, which this wrapper does not cover; it runs
    with the provider's default reasoning, recorded in its metadata as such."""
    return (role in WRITERS or role == TIE_BREAK) and Z2_PROFILES[role].endpoint == "chat_completions"


# Transient provider failures are retried, never quota or auth (ProviderAccessError
# stops the run for operator action). One MiMo call returned HTTP 500 at dev window
# 5 of 5 (2026-10-18) and, with no retry, stopped the whole run.
RETRY_WAITS = (10, 30, 60)


def transient(exc) -> bool:
    import openai
    from experiments.lme.cloud_provider import CloudCompletionError
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    # A generation the provider aborted mid-way (status "abort", 2026-10-19 on B's
    # change detection) is transient; a length cut is not (it would cut again).
    if isinstance(exc, CloudCompletionError) and "abort" in str(exc):
        return True
    return isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", 0) >= 500


def retrying(generate, role, *, waits=RETRY_WAITS, sleep=None):
    import sys
    import time
    sleep = sleep or time.sleep

    def call(*args, **kwargs):
        for attempt, wait in enumerate((*waits, None), 1):
            try:
                return generate(*args, **kwargs)
            except Exception as exc:                          # noqa: BLE001
                if wait is None or not transient(exc):
                    raise
                print(f"WARNING: {role} call attempt {attempt} failed "
                      f"({type(exc).__name__} {getattr(exc, 'status_code', '')}); retrying in {wait}s",
                      file=sys.stderr, flush=True)
                sleep(wait)
    return call


def generator(role):
    from experiments.lme.cloud_provider import TextGenerator, load_selected_env
    load_selected_env()
    generator = TextGenerator(Z2_PROFILES[role])
    if _thinking_off(role):
        generator.client = _Injecting(generator.client, WRITER_EXTRA_BODY)
    generator.generate = retrying(generator.generate, role)
    return generator


def metadata(role):
    """Public identity of a role's calls, including any fixed request fields."""
    meta = dict(Z2_PROFILES[role].metadata())
    if _thinking_off(role):
        meta["extra_body"] = WRITER_EXTRA_BODY
    return meta
