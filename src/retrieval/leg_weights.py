"""G9: the intent-dependent leg weights, as configuration rather than code.

These thirty-odd numbers decide how much each retrieval leg contributes to the
fused ranking, and they were local literals rebuilt inside `retrieve()` on
every request — the first thing Z1's stage-2 sweep wants to turn, and the one
thing that could not be turned without editing the hot path.

**Why a dict and not thirty scalar settings.** The profiles are reasoned about
as a table: a row is one intent's opinion about all four legs, and reading it
as `retrieval_weight_factual_vector`, `..._factual_bm25` destroys the shape
that makes it reviewable. `maintenance_intervals` set the precedent.

**Why the dict is validated on every read.** A dict setting cannot be
type-checked by pydantic the way a field can, so a mistyped intent
(`Factual_Retreival`) would simply never match and that row would be silently
dead — the same shape as the ablation flag that zeroed the wrong module's copy
for a year (ROADMAP G19). Validation happens at READ, not at config load,
because Z1's sweeper writes these values at runtime: an import-time check
would validate the defaults and wave through everything the sweep sets.

The two halves are validated differently on purpose:

* **Leg names** are a closed set owned by this file. Anything else is a bug in
  every case, so it raises.
* **Intent and topic names** come from the live label schema, which legitimately
  changes with the checkpoint — a rollback to the v1 model drops `Code_Change`.
  Raising there would take retrieval down on a supported rollback, so an
  unrecognised label warns once and its row is ignored, which is exactly what
  happens today.
"""

from typing import Dict, Iterable, List

import structlog

from src.api.config import settings

logger = structlog.get_logger("ice.retrieval.leg_weights")

# The legs that carry an RRF weight. Closed, and owned here rather than by
# config, because adding a leg is a code change in this module's neighbourhood.
LEGS = ("bm25", "vector", "codex", "procedural", "timeline")

# Warn once per offending label rather than once per request — a hot-path
# warning that repeats on every turn trains everyone to ignore it, which is the
# failure the warning exists to prevent (TRAPS #13b).
_warned: set = set()


def _known_labels(head: str) -> frozenset:
    """Labels the LIVE checkpoint's schema declares for *head*.

    Empty frozenset when the schema cannot be read — in which case label
    validation is skipped rather than the request failed.
    """
    try:
        from src.classifier.schema import load_schema
        return frozenset(load_schema().head(head).labels)
    except Exception:
        return frozenset()


def _check_legs(where: str, weights: Dict[str, float]) -> None:
    unknown = set(weights) - set(LEGS)
    if unknown:
        raise ValueError(
            f"{where} names retrieval legs that do not exist: {sorted(unknown)}. "
            f"Valid legs are {list(LEGS)}. A weight on a non-existent leg is "
            f"silently ignored by fusion, so this is refused rather than dropped."
        )


def _check_labels(where: str, keys: Iterable[str], head: str) -> set:
    """Return the subset of *keys* the live schema recognises, warning on the rest."""
    known = _known_labels(head)
    if not known:
        return set(keys)
    unknown = set(keys) - known
    for label in sorted(unknown):
        mark = (where, label)
        if mark not in _warned:
            _warned.add(mark)
            logger.warning("leg_weight_unknown_label", table=where, label=label,
                           head=head,
                           reason="not in the live label schema; this row never "
                                  "matches and contributes nothing")
    return set(keys) - unknown


def _unrankable(weights: Dict[str, float]) -> bool:
    """True when no leg can contribute a ranking signal.

    `timeline` is excluded because it is pinned to its base value after the
    blend, so it is never the leg that collapsed — counting it would make this
    check unsatisfiable.
    """
    rankable = [w for leg, w in weights.items() if leg != "timeline"]
    return bool(rankable) and all(w <= 0.0 for w in rankable)


def resolve(intent_tags: List[str], topic_tags: List[str]) -> Dict[str, float]:
    """Blend the per-intent profiles into one weight per leg.

    Each active intent contributes equally: an intent with a profile row
    contributes that row, an intent without one contributes the base weights.
    Topic overrides are cumulative on top. The timeline weight is pinned to its
    base value because its firing condition — a matched anchor with history —
    is the gate, not the intent.
    """
    base = dict(settings.retrieval_leg_base_weights)
    _check_legs("retrieval_leg_base_weights", base)

    profiles = settings.retrieval_leg_profiles
    for intent, override in profiles.items():
        _check_legs(f"retrieval_leg_profiles[{intent!r}]", override)
    valid_intents = _check_labels("retrieval_leg_profiles", profiles.keys(), "intent")

    active = list(intent_tags or [])
    num_active = len(active) if active else 1

    blended = {leg: 0.0 for leg in base}
    for tag in active:
        override = profiles.get(tag) if tag in valid_intents else None
        source = override if override else base
        for leg, w in source.items():
            blended[leg] = blended.get(leg, 0.0) + w / num_active

    # Nothing matched at all (no active intents) — fall back to base.
    if all(v == 0.0 for v in blended.values()):
        blended = dict(base)

    # Snapshot before the deltas — the all-zero guard below needs something
    # meaningful to fall back to, and "the weights minus the override that
    # broke them" is the closest honest answer.
    pre_override = dict(blended)

    # Cumulative topic overrides.
    topic_overrides = settings.retrieval_leg_topic_overrides
    for topic, override in topic_overrides.items():
        _check_legs(f"retrieval_leg_topic_overrides[{topic!r}]", override)
    valid_topics = _check_labels("retrieval_leg_topic_overrides",
                                 topic_overrides.keys(), "topic")
    for topic in set(topic_tags or []):
        if topic in valid_topics:
            for leg, delta in topic_overrides.get(topic, {}).items():
                blended[leg] = blended.get(leg, 0.0) + delta

    # A weight of 0.0 does NOT switch a leg off. `_apply_rrf` still registers
    # that leg's fragments; they simply score 0.0. So an all-zero weight set
    # does not mean "retrieve nothing" — it means "admit every candidate at
    # score 0 and let _apply_bonuses order them", i.e. recency decides and
    # relevance is discarded entirely. That state is UNRANKABLE, not
    # restrictive, and its output is indistinguishable from a legitimately bad
    # retrieval, which is why it is refused here rather than served.
    #
    # Unreachable with the shipped table (every override is positive). It
    # becomes reachable the moment a negative delta meets an already-suppressed
    # intent blend — and Z1 stage 2 writes these values AT RUNTIME, so the
    # sweep can land on it without anyone having typed it.
    #
    # Loud every time, not once per process: this substitutes a default for a
    # real answer, which is the one case the warn-once idiom above must not
    # cover. It cannot spam a healthy system, because a healthy system never
    # reaches it.
    if _unrankable(blended):
        replacement = pre_override if not _unrankable(pre_override) else dict(base)
        logger.warning(
            "leg_weight_all_zero",
            intent_tags=list(intent_tags or []), topic_tags=list(topic_tags or []),
            collapsed=dict(blended), falling_back_to=dict(replacement),
            reason="every rankable leg resolved to <= 0, which admits all "
                   "candidates at score 0 rather than retrieving nothing; "
                   "check the topic overrides for this label combination",
        )
        blended = replacement

    # T4: the timeline weight is constant across intent profiles.
    if "timeline" in base:
        blended["timeline"] = base["timeline"]

    return {leg: max(0.0, w) for leg, w in blended.items()}
