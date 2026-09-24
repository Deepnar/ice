"""Codex Edge Decay – periodically reduces strength of unreinforced edges.

Runs on the maintenance runtime's cadence; ``cycles`` compresses missed runs
closed-form (C7 D5), with a retention floor applied once at the end —
identical to N sequential runs.
"""

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.workers.decay import per_cycle as decay_per_cycle

logger = structlog.get_logger("ice.workers.codex_decay")


def decay_rate() -> float:
    """Per-cycle multiplier compounding to settings.codex_decay_daily per day.

    Derived from THIS job's own cadence (`decay_codex`), not the episodic one —
    they happen to share 5400 s today, and a shared constant would hide it the
    day they stop. See decay.cycles_per_day for why this is not a setting.
    """
    return decay_per_cycle(settings.codex_decay_daily, job="decay_codex")


def decay_codex_edges(cycles: int = 1):
    """Decay retention priority; preserve confidence and historical validity."""
    cycles = max(1, min(int(cycles), settings.runtime_cycles_cap))
    db = SessionLocal()
    try:
        # Nonuse reduces retention priority, not support or temporal validity.
        # A quiet true fact stays queryable; correction/deletion owns retirement.
        db.execute(text("""
            UPDATE codex_edges
            SET strength = GREATEST(:floor, strength * POWER(:rate, :cycles))
            WHERE valid_until IS NULL
              AND source = 'conversation'
        """), {"rate": decay_rate(), "cycles": cycles,
               "floor": settings.codex_retention_floor})

        db.commit()
        logger.info("codex_decay_cycle_complete", cycles=cycles)
    except Exception as exc:
        db.rollback()
        logger.error("codex_decay_failed", error=str(exc))
        raise
    finally:
        db.close()
