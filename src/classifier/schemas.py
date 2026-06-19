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