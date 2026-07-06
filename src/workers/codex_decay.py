"""Codex Edge Decay – periodically reduces strength of unreinforced edges."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import CodexEdge
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.codex_decay")
CYCLES_PER_DAY = 16.0
DECAY_RATE = 0.99 ** (1.0 / CYCLES_PER_DAY)   # ≈0.9994  (same effective daily rate)
DEMOTION_THRESHOLD = 0.3   # strength below this -> pending
EXPIRY_THRESHOLD = 0.1     # pending edges below this are expired (A3 garbage collection)


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def decay_codex_edges(self):
    """Daily task: decay strength of live Codex edges, demote weak ones."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # 1. Decay ALL live edges — pending included (A3). Previously only
        #    active edges decayed, so a retrieval-reinforced pending edge
        #    could inflate forever without ever entering the decay cycle.
        db.execute(text("""
            UPDATE codex_edges
            SET strength = strength * :rate
            WHERE valid_until IS NULL
        """), {"rate": DECAY_RATE})
        # 2. Demote active edges that fell below threshold
        db.execute(text("""
            UPDATE codex_edges
            SET confidence = 'pending'
            WHERE confidence = 'active'
              AND valid_until IS NULL
              AND strength < :thresh
        """), {"thresh": DEMOTION_THRESHOLD})
        # 3. A3 garbage collection: pending edges that decayed to near-zero
        #    without ever being corroborated or retrieved are expired —
        #    uncorroborated low-trust residue (e.g. grounding-rejected
        #    triplets) leaves the live graph instead of accumulating.
        db.execute(text("""
            UPDATE codex_edges
            SET valid_until = NOW()
            WHERE confidence = 'pending'
              AND valid_until IS NULL
              AND strength < :expiry
        """), {"expiry": EXPIRY_THRESHOLD})

        db.commit()
        logger.info("codex_decay_cycle_complete")
    except Exception as exc:
        db.rollback()
        logger.error("codex_decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()