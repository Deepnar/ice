"""B2 — principled, classifier-trusting memory-retrieval decision.

Replaces the scattered *hard* Long-Term-Memory overrides that masked the
classifier ("safe, not smart"):

  * main.py:      ``turn_count > 10 or max_confidence < 0.95  ⇒ LTM``
  * classifier:   ``Creative_&_Media ⇒ LTM`` and ``Software+referential ⇒ LTM``
                  (the old ``_apply_hard_overrides``)
  * DI3:          the anaphora / reference rule force-returned ``Long_Term_Memory``
  * orchestrator: ``Zero_Shot + conversation_id ⇒ LTM`` and ``Creative ⇒ LTM``

…with ONE posterior in log-odds space that *prefers* memory but never *forces* it:

    logit(P_need_mem) = logit(P_ltm) + bias
                        + β · logit(P_len)              # memory pressure (see below)
                        + Σ feature bumps               # reference / creative / referential / low-conf
    retrieve  ⇔  P_need_mem > τ

Design notes
------------
* **Prefers, doesn't force.** ``ltm_prior_bias`` tilts the default toward
  retrieval, but a *confident* Zero_Shot (low ``p_ltm``) still stays out — no
  reflexive flip on turn count.

* **Memory-pressure length prior, not a turn cliff.** ``P_len`` is a smooth
  function of how much conversation history sits **beyond the sliding window**
  (the orchestrator reserves only a fraction of the token budget for recent
  turns). If the window already carries the recent turns, a recent reference is
  covered and pressure ≈ 0; pressure rises only as history the window can't see
  accumulates. This is the "sliding window + total turns + total context"
  signal, per user direction.

* **Built on a classifier B1 will retrain.** We consume ``p_ltm`` as a *scalar*,
  so B1's softmax-3 → multi-label-sigmoid change only alters how the scalar is
  produced (classifier side), not this decision. Every weight is a setting so
  this is **re-tuned, not rewritten**, after the retrain — and we genuinely
  cannot know yet whether that retrain lands better or worse.

The function is pure (no DB, no I/O) and returns a full ``breakdown`` for
telemetry (a prime candidate to promote to the F5 SSE attribution layer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.classifier.schemas import ClassificationResult

# Referential / anaphoric cues — a *light* signal (a bump, not a slam). Kept
# here (not in the classifier) so the whole memory decision reads from one place.
REFERENTIAL_WORDS = {
    "my", "our", "mine", "ours", "we", "us",
    "this", "that", "these", "those", "the",
    "it", "they", "them", "their",
    "previous", "last", "before", "yesterday", "earlier",
    "again", "still", "same",
}

# Mirrors the orchestrator's recent-turn budget curve (_compute_recent_fraction
# base term) so the "beyond the window" estimate matches what actually gets
# reserved for recent turns. If that curve moves, move this with it.
_TOTAL_CONTEXT_BUDGET = 23_000
_OVERHEAD_RESERVE = 1_800


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def derive_total_budget(context_window, settings) -> int:
    """C16 (model-aware half): total context budget derived from the routed
    model's context window — ``fraction × window``, clamped to [min, max] —
    instead of a hardcoded 23k. Unknown model (None/0) → configured fallback.
    The max guardrail stays until C16's need-based filling makes huge budgets
    safe; all knobs live in settings."""
    if not context_window:
        return settings.context_budget_fallback
    budget = int(context_window * settings.context_input_fraction)
    return max(settings.context_budget_min,
               min(settings.context_budget_max, budget))


def estimate_recent_window_tokens(turn_count: int, total_budget: float = None) -> float:
    """Token budget the orchestrator reserves for recent turns (the sliding
    window), estimated from conversation length. Mirrors the base fraction in
    ``HybridRetrievalOrchestrator._compute_recent_fraction``. ``total_budget``
    is the model-derived total (C16); defaults to the legacy 23k mirror."""
    total = total_budget if total_budget else _TOTAL_CONTEXT_BUDGET
    if turn_count < 10:
        frac = 0.3
    elif turn_count < 200:
        frac = 0.2
    else:
        frac = 0.15
    return frac * (total - _OVERHEAD_RESERVE)


def memory_pressure(total_tokens: float, window_tokens: float,
                    midpoint_tokens: float, scale_tokens: float) -> float:
    """P_len ∈ [0.5, 1): a *one-sided* logistic in the volume of history sitting
    beyond the sliding window.

    Neutral (0.5 → contributes 0 in log-odds) while the conversation still fits
    the window plus a ``midpoint`` deadzone; rises toward 1 only as out-of-window
    history accumulates. One-sided by design: memory pressure should *add* push
    toward retrieval when there's unseen history, never *penalise* a short
    conversation (that's the classifier's job)."""
    beyond = max(0.0, total_tokens - window_tokens)
    deficit = max(0.0, beyond - midpoint_tokens)
    return _sigmoid(deficit / max(1.0, scale_tokens))


def _has_referential_word(prompt: str) -> bool:
    if not prompt:
        return False
    tokens = set(prompt.lower().split())
    return bool(tokens & REFERENTIAL_WORDS)


@dataclass
class MemoryDecision:
    retrieve: bool
    p_need_mem: float
    breakdown: dict = field(default_factory=dict)


def decide_memory_retrieval(
    result: ClassificationResult,
    turn_count: int,
    total_tokens: float,
    settings,
    recent_window_tokens: Optional[float] = None,
    timescope_mode: Optional[str] = None,
    coding_scope: bool = False,
) -> MemoryDecision:
    """Decide whether to run long-term-memory retrieval for this turn.

    Pure function of the classification + cheap conversation stats + settings.
    ``total_tokens`` is the estimated size of the whole conversation so far
    (chars/4 is fine); ``recent_window_tokens`` defaults to the orchestrator's
    reserved recent-turn budget. ``timescope_mode`` (T2) is a kwarg, not a
    ClassificationResult field — keeps this pure and leaves the label question
    to B1; a non-current mode is a large bump, never a hard override (an
    explicit "what did I think in 2025" is definitionally a memory query, but
    the posterior still decides).
    """
    if recent_window_tokens is None:
        recent_window_tokens = estimate_recent_window_tokens(turn_count)

    # 1. Classifier prior — trust it (this is the whole point of B2).
    lo = _logit(result.p_ltm) + settings.ltm_prior_bias

    # 2. Memory-pressure length prior (beyond-window history).
    p_len = memory_pressure(
        total_tokens, recent_window_tokens,
        settings.ltm_pressure_midpoint_tokens, settings.ltm_pressure_scale_tokens,
    )
    lo += settings.ltm_length_weight * _logit(p_len)

    # 3. Feature bumps — the old hard signals, demoted to additive nudges.
    creative = "Creative_&_Media" in result.topic_tags
    referential = _has_referential_word(result.prompt)
    low_conf = result.max_confidence < settings.confidence_fallback_threshold
    if creative:
        lo += settings.ltm_bump_creative
    if result.reference_signal:
        lo += settings.ltm_bump_reference
    if referential:
        lo += settings.ltm_bump_referential
    if low_conf:
        lo += settings.ltm_bump_low_confidence
    # T2 + B1 D7: the detector and the Temporal_Recall label are EQUIVALENT
    # evidence for the retrieval bump — OR, never AND, and never twice. A
    # v2 classifier catches "what did I think about this back then" (no parseable
    # date, detector silent); the detector catches "in March 2026" on a prompt the
    # head reads as ordinary. Note what the label does NOT do: only the
    # deterministic detector ever sets a time WINDOW. A sigmoid inventing
    # "two years ago" would be a hallucinated filter.
    detector_fired = bool(timescope_mode and timescope_mode != "current")
    temporal_label = result.p_temporal >= settings.temporal_label_threshold
    if detector_fired or temporal_label:
        lo += settings.ltm_bump_timescope
    # E1 (D11): a project-attached conversation almost always wants context —
    # pointers are cheap. A bump, not a force (B2 idiom).
    if coding_scope:
        lo += settings.ltm_bump_coding

    p_need = _sigmoid(lo)
    retrieve = p_need > settings.ltm_decision_threshold

    return MemoryDecision(
        retrieve=retrieve,
        p_need_mem=p_need,
        breakdown={
            "p_ltm": round(result.p_ltm, 4),
            "ctx_confidence": round(result.ctx_confidence, 4),
            "p_len": round(p_len, 4),
            "total_tokens": int(total_tokens),
            "window_tokens": int(recent_window_tokens),
            "creative": creative,
            "reference_signal": result.reference_signal,
            "referential": referential,
            "low_confidence": low_conf,
            "timescope": timescope_mode,
            "p_temporal": round(result.p_temporal, 4),
            "temporal_label": temporal_label,
            "coding_scope": coding_scope,
            "logit": round(lo, 4),
            "p_need_mem": round(p_need, 4),
        },
    )
