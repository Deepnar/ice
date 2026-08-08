"""Codex Edge Decay – periodically reduces strength of unreinforced edges.

Runs on the maintenance runtime's cadence; ``cycles`` compresses missed runs
closed-form (C7 D5), with demotion/expiry thresholds applied once at the end —
identical to N sequential runs.
"""

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal

logger = structlog.get_logger("ice.workers.codex_decay")


def decay_rate() -> float:
    """Per-cycle multiplier compounding to settings.codex_decay_daily per day.

    G9: derived, not stored — see the note on decay.per_cycle.
    """
    return settings.codex_decay_daily ** (1.0 / settings.codex_decay_cycles_per_day)


def decay_codex_edges(cycles: int = 1):
    """Decay strength of live Codex edges, demote weak ones."""
    cycles = max(1, min(int(cycles), settings.runtime_cycles_cap))
    db = SessionLocal()
    try:
        # 1. Decay ALL live conversation edges — pending included (A3).
        #    Previously only active edges decayed, so a retrieval-reinforced
        #    pending edge could inflate forever without ever entering the
        #    decay cycle. E1b (D3): derived memory (static_analysis/derived
        #    sources) is decay-EXEMPT — the code graph is regenerated from
        #    source, not earned through use; forgetting it would just desync
        #    the map from the repo.
        db.execute(text("""
            UPDATE codex_edges
            SET strength = strength * POWER(:rate, :cycles)
            WHERE valid_until IS NULL
              AND source = 'conversation'
        """), {"rate": decay_rate(), "cycles": cycles})
        # 2. Demote active edges that fell below threshold
        db.execute(text("""
            UPDATE codex_edges
            SET confidence = 'pending'
            WHERE confidence = 'active'
              AND valid_until IS NULL
              AND source = 'conversation'
              AND strength < :thresh
        """), {"thresh": settings.codex_demotion_threshold})
        # 3. A3 garbage collection: pending edges that decayed to near-zero
        #    without ever being corroborated or retrieved are expired —
        #    uncorroborated low-trust residue (e.g. grounding-rejected
        #    triplets) leaves the live graph instead of accumulating.
        db.execute(text("""
            UPDATE codex_edges
            SET valid_until = NOW()
            WHERE confidence = 'pending'
              AND valid_until IS NULL
              AND source = 'conversation'
              AND strength < :expiry
        """), {"expiry": settings.codex_expiry_threshold})

        db.commit()
        logger.info("codex_decay_cycle_complete", cycles=cycles)
    except Exception as exc:
        db.rollback()
        logger.error("codex_decay_failed", error=str(exc))
        raise
    finally:
        db.close()
