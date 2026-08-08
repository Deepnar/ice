"""Procedural Memory Decay – deactivates stale, low-confidence patterns."""

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal

logger = structlog.get_logger("ice.workers.procedural_decay")


def decay_procedural_patterns(cycles: int = 1):
    """Deactivate procedural patterns that are stale and weak.

    ``cycles`` exists for interface uniformity with the other decay jobs
    (C7 D5) but is a no-op here: staleness is an absolute-age cutoff, not a
    multiplicative score, so catch-up needs no compression.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.procedural_stale_days)
        db.execute(text("""
            UPDATE procedural_memory
            SET is_active = FALSE
            WHERE is_active = TRUE
              AND last_observed < :cutoff
              AND reinforcement_count < :min_reinf
        """), {"cutoff": cutoff, "min_reinf": settings.procedural_min_reinforcement})
        db.commit()
        logger.info("procedural_decay_cycle_complete")
    except Exception as exc:
        db.rollback()
        logger.error("procedural_decay_failed", error=str(exc))
        raise
    finally:
        db.close()
