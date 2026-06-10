"""Procedural Memory Decay – deactivates stale, low‑confidence patterns."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.procedural_decay")
STALE_DAYS = 180
MIN_REINFORCEMENT = 3


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def decay_procedural_patterns(self):
    """Periodic task: deactivate procedural patterns that are stale and weak."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)
        db.execute(text("""
            UPDATE procedural_memory
            SET is_active = FALSE
            WHERE is_active = TRUE
              AND last_observed < :cutoff
              AND reinforcement_count < :min_reinf
        """), {"cutoff": cutoff, "min_reinf": MIN_REINFORCEMENT})
        db.commit()
        logger.info("procedural_decay_cycle_complete")
    except Exception as exc:
        db.rollback()
        logger.error("procedural_decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()