"""Codex usage decay — fades facts' prompt-usage counts (CODEX_V4 D7).

Runs on the maintenance runtime's cadence; ``cycles`` compresses missed runs
closed-form (C7 D5) — identical to N sequential runs.
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


def decay_fact_usage(cycles: int = 1):
    """CODEX_V4 D7: fade facts' prompt usage (the third of the strength prior
    that is not support). No floor — a fact nobody reads drifts back to its
    support-only rank. Validity and support are never touched."""
    cycles = max(1, min(int(cycles), settings.runtime_cycles_cap))
    db = SessionLocal()
    try:
        db.execute(text("""
            UPDATE codex_facts SET usage = usage * POWER(:rate, :cycles)
            WHERE usage > 0 AND valid_until IS NULL
        """), {"rate": decay_rate(), "cycles": cycles})
        db.commit()
        logger.info("codex_decay_cycle_complete", cycles=cycles)
    except Exception as exc:
        db.rollback()
        logger.error("codex_decay_failed", error=str(exc))
        raise
    finally:
        db.close()
