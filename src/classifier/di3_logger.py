"""DI3 event logger – thin wrapper around structlog."""

import structlog

logger = structlog.get_logger("ice.di3")


def log_di3_decision(signal: str, confidence: float) -> None:
    logger.info("di3_decided", signal=signal, confidence=confidence)


def log_di3_passed_to_ml(signals: dict) -> None:
    logger.info("di3_passed_to_ml", signal_scores=signals)