"""Retire stale procedures below the independent-recurrence exemption."""

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
        # One statement's snapshot reports the old eligible population and the
        # actual UPDATE count. Reading a procedure never refreshes this clock.
        counts = db.execute(text("""
            WITH retired AS (
                UPDATE procedural_memory SET is_active = FALSE
                WHERE is_active = TRUE AND last_observed < :cutoff
                  AND reinforcement_count < :min_reinf
                RETURNING id
            )
            SELECT (SELECT count(*) FROM procedural_memory WHERE is_active = TRUE) AS active_before,
                   (SELECT count(*) FROM procedural_memory WHERE is_active = TRUE
                       AND last_observed < :cutoff
                       AND reinforcement_count >= :min_reinf) AS stale_protected,
                   (SELECT count(*) FROM retired) AS retired
        """), {"cutoff": cutoff, "min_reinf": settings.procedural_min_reinforcement}).mappings().one()
        db.commit()
        receipt = dict(method="procedural_retirement_v1", clock_basis="ice_observation",
            cutoff=cutoff.isoformat(), stale_days=settings.procedural_stale_days,
            min_reinforcement=settings.procedural_min_reinforcement, **dict(counts))
        logger.info("procedural_decay_cycle_complete", **receipt)
        return receipt
    except Exception as exc:
        db.rollback()
        logger.error("procedural_decay_failed", error=str(exc))
        raise
    finally:
        db.close()
