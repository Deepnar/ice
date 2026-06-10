"""Codex Edge Decay – periodically reduces strength of unreinforced edges."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import CodexEdge
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.codex_decay")
DECAY_RATE = 0.99          # 1% decay per run
DEMOTION_THRESHOLD = 0.3   # strength below this -> pending


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def decay_codex_edges(self):
    """Daily task: decay strength of active Codex edges, demote weak ones."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # 1. Decay all active edges
        db.execute(text("""
            UPDATE codex_edges
            SET strength = strength * :rate
            WHERE confidence = 'active'
        """), {"rate": DECAY_RATE})

        # 2. Demote edges that fell below threshold
        db.execute(text("""
            UPDATE codex_edges
            SET confidence = 'pending'
            WHERE confidence = 'active' AND strength < :thresh
        """), {"thresh": DEMOTION_THRESHOLD})

        db.commit()
        logger.info("codex_decay_cycle_complete")
    except Exception as exc:
        db.rollback()
        logger.error("codex_decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()