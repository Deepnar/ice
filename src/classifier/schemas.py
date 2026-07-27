"""Shared classification dataclasses to avoid circular imports."""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class ClassificationResult:
    topic_tags: List[str]
    intent_tags: List[str]
    context_reliance: str
    raw_probs: List[float]        # schema-wide: 27 under v2, 25 under a v1 checkpoint
    max_confidence: float
    prompt: str = ""

    # ── B2: context-reliance confidence, consumed by the memory-retrieval
    # decision (src/api/memory_decision.py). Kept as scalars so B1's
    # softmax-3 → multi-label-sigmoid retrain only changed how they're
    # populated, not who reads them — which is exactly what happened. ──
    p_ltm: float = 0.0            # P(needs memory); v2 = the Needs_Memory sigmoid
    p_rts: float = 0.0            # P(needs live info); v2 = the Needs_Live_Info sigmoid
    ctx_confidence: float = 0.0   # top1−top2 margin of the derived 3-way (D6)
    # (A `reference_signal` flag lived here until D8, 2026-07-27. It was set only
    #  by DI3's anaphora rule and read by B2's bump and T2's joint gate; both
    #  measured better without it, so the flag went with DI3. The surviving
    #  anaphora signal is memory_decision.REFERENTIAL_WORDS, computed from the
    #  prompt where it is used.)

    # ── B1 v2: the two signals the 3-way could never express. ──
    # p_temporal: memory query with a time dimension. memory_decision treats
    # ≥ settings.temporal_label_threshold as evidence equivalent to a fired
    # TimeScope for the retrieval *bump* (D7, OR never AND) — but ONLY the
    # deterministic detector (retrieval/timescope.py) ever sets a window.
    p_temporal: float = 0.0
    # p_complex: "the strongest model would answer this materially better".
    # The cloud/routing toggle signal — B3 consumes it; nothing routes on it yet.
    p_complex: float = 0.0

    # Per-head peakedness {"topic": .., "intent": .., "context_reliance": ..},
    # populated by the classifier which knows its own schema. C15's wide-net
    # trigger and B3's router read these instead of re-slicing raw_probs by hand.
    head_confidences: Dict[str, float] = field(default_factory=dict)
