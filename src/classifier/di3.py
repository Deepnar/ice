"""DI3 – Dynamic Intent Inferencer (pre‑classification stage)."""

from typing import List, Optional

from src.classifier.schemas import ClassificationResult
from src.classifier.di3_signals import extract_signals
from src.classifier.di3_config import (
    DI3_ENABLED,
    DI3_CODE_DENSITY_THRESHOLD,
    DI3_SENTIMENT_DENSITY_THRESHOLD,
    DI3_META_DENSITY_THRESHOLD,
    DI3_NOISE_DENSITY_THRESHOLD,
    DI3_REFERENCE_DENSITY_THRESHOLD,
    DI3_LTM_REFERENCE_DENSITY_THRESHOLD,
)
from src.classifier.di3_logger import log_di3_decision, log_di3_passed_to_ml


def run_di3(
    prompt: str,
    conversation_length: int = 0,
    conversation_history: Optional[List[str]] = None,
) -> Optional[ClassificationResult]:
    """
    Attempt to classify the prompt using signal‑based heuristics.
    Returns a ClassificationResult when confident, or None to fall back to the ML model.
    """

    if not DI3_ENABLED:
        return None

    signals = extract_signals(prompt)

    # 1. Noise / gibberish
    if signals["noise_density"] > DI3_NOISE_DENSITY_THRESHOLD:
        log_di3_decision("noise", 0.95)
        return ClassificationResult(
            topic_tags=["Null_Noise"],
            intent_tags=["Casual_Banter"],
            context_reliance="Zero_Shot",
            raw_probs=[0.0] * 25,
            max_confidence=0.95,
            prompt=prompt,
        )

    # 2. Code
    if signals["code_density"] > DI3_CODE_DENSITY_THRESHOLD:
        ctx = "Long_Term_Memory" if conversation_length > 0 else "Zero_Shot"
        log_di3_decision("code", 0.9)
        return ClassificationResult(
            topic_tags=["Software_&_Tech"],
            intent_tags=["Generation"],
            context_reliance=ctx,
            raw_probs=[0.0] * 25,
            max_confidence=0.9,
            prompt=prompt,
        )

    # 3. Sentiment / emotional
    if signals["sentiment_density"] > DI3_SENTIMENT_DENSITY_THRESHOLD:
        ctx = "Long_Term_Memory" if conversation_length > 5 else "Zero_Shot"
        log_di3_decision("sentiment", 0.85)
        return ClassificationResult(
            topic_tags=["Lifestyle_&_Health", "Social_&_Relationships"],
            intent_tags=["Emotional_Processing"],
            context_reliance=ctx,
            raw_probs=[0.0] * 25,
            max_confidence=0.85,
            prompt=prompt,
        )

    # 4. Meta‑AI
    if signals["meta_density"] > DI3_META_DENSITY_THRESHOLD:
        log_di3_decision("meta", 0.9)
        return ClassificationResult(
            topic_tags=["Meta_AI"],
            intent_tags=["Factual_Retrieval"],
            context_reliance="Zero_Shot",
            raw_probs=[0.0] * 25,
            max_confidence=0.9,
            prompt=prompt,
        )

    # 5. Reference (anaphora) – force LTM even with low confidence
    ref_threshold = (
        DI3_LTM_REFERENCE_DENSITY_THRESHOLD
        if conversation_length > 10
        else DI3_REFERENCE_DENSITY_THRESHOLD
    )
    if signals["reference_density"] > ref_threshold:
        log_di3_decision("reference", 0.7)
        return ClassificationResult(
            topic_tags=[],
            intent_tags=[],
            context_reliance="Long_Term_Memory",
            raw_probs=[0.0] * 25,
            max_confidence=0.7,
            prompt=prompt,
        )

    # Fall through – let the ML classifier decide
    log_di3_passed_to_ml(signals)
    return None