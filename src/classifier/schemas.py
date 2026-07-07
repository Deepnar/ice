"""Shared classification dataclasses to avoid circular imports."""

from dataclasses import dataclass, field
from typing import List

@dataclass
class ClassificationResult:
    topic_tags: List[str]
    intent_tags: List[str]
    context_reliance: str
    raw_probs: List[float]        # 25 probabilities
    max_confidence: float
    prompt: str = ""

    # ── B2: context-reliance confidence, consumed by the memory-retrieval
    # decision (src/api/memory_decision.py). Kept as scalars so B1's
    # softmax-3 → multi-label-sigmoid retrain only changes how they're
    # populated, not who reads them. ──
    p_ltm: float = 0.0            # P(Long_Term_Memory) from the ctx head
    p_rts: float = 0.0            # P(Real_Time_Search) — orthogonal (B1); routes nowhere yet
    ctx_confidence: float = 0.0   # top1−top2 margin of the ctx head (how sure about reliance)
    reference_signal: bool = False  # strong anaphora detected (DI3 reference rule)