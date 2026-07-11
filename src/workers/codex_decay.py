"""Codex Edge Decay – periodically reduces strength of unreinforced edges.

Runs on the maintenance runtime's cadence; ``cycles`` compresses missed runs
closed-form (C7 D5), with demotion/expiry thresholds applied once at the end —
identical to N sequential runs.
"""

import structlog
from sqlalchemy import text

from src.api.db import SessionLocal
from src.workers.runtime import CYCLES_CAP

logger = structlog.get_logger("ice.workers.codex_decay")
CYCLES_PER_DAY = 16.0
DECAY_RATE = 0.99 ** (1.0 / CYCLES_PER_DAY)   # ≈0.9994  (same effective daily rate)
DEMOTION_THRESHOLD = 0.3   # strength below this -> pending
EXPIRY_THRESHOLD = 0.1     # pending edges below this are expired (A3 garbage collection)


def decay_codex_edges(cycles: int = 1):
    """Decay strength of live Codex edges, demote weak ones."""
    cycles = max(1, min(int(cycles), CYCLES_CAP))
    db = SessionLocal()
    try:
        # 1. Decay ALL live edges — pending included (A3). Previously only
        #    active edges decayed, so a retrieval-reinforced pending edge
        #    could inflate forever without ever entering the decay cycle.
        db.execute(text("""
            UPDATE codex_edges
            SET strength = strength * POWER(:rate, :cycles)
            WHERE valid_until IS NULL
        """), {"rate": DECAY_RATE, "cycles": cycles})
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
        logger.info("codex_decay_cycle_complete", cycles=cycles)
    except Exception as exc:
        db.rollback()
        logger.error("codex_decay_failed", error=str(exc))
        raise
    finally:
        db.close()
