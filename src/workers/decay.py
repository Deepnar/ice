"""Decay Worker – applies access-weighted memory decay and archival.

Runs on the maintenance runtime's cadence (maintenance_intervals
["decay_episodic"], 1.5h). Catch-up after downtime is closed-form (C7 D5):
the runtime passes ``cycles`` = whole intervals elapsed since this job's last
ledger finish, and one UPDATE applies ``rate ** cycles`` — the multipliers are
exponential, so any gap collapses into a single statement.
"""

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import text

from src.api.db import SessionLocal
from src.workers.runtime import CYCLES_CAP

logger = structlog.get_logger("ice.workers.decay")
# Effective daily decay targets: 0.95 (unaccessed), 0.98 (accessed)
# Cycles per day (from maintenance_intervals): 24h / 1.5h = 16
CYCLES_PER_DAY = 16.0
DECAY_RATE_UNACCESSED = 0.95 ** (1.0 / CYCLES_PER_DAY)   # ≈0.9968
DECAY_RATE_ACCESSED   = 0.98 ** (1.0 / CYCLES_PER_DAY)   # ≈0.9987

# Creative slow decay (1% per day effective)
CREATIVE_DECAY_RATE = 0.99 ** (1.0 / CYCLES_PER_DAY)     # ≈0.9994
STRENGTHEN_AMOUNT = 0.15
ARCHIVE_THRESHOLD = 0.1
COLD_THRESHOLD = 0.05


def apply_decay(cycles: int = 1):
    """Decay old turns, archive stale ones, move to cold storage.

    ``cycles`` (clamped to [1, 96]; rate**96 ≈ 0.73 for the unaccessed rate)
    compresses missed runs into one pass — thresholds and the creative floor
    apply once at the end, exactly as they would after N sequential runs.
    """
    cycles = max(1, min(int(cycles), CYCLES_CAP))
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        # Access-weighted decay: unaccessed non-creative turns
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * POWER(:rate, :cycles)
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND access_count = 0
              AND NOT ('Creative_&_Media' = ANY(topic_tags))
        """), {"rate": DECAY_RATE_UNACCESSED, "cutoff": cutoff, "cycles": cycles})

        # Access-weighted decay: previously-accessed non-creative turns
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * POWER(:rate, :cycles)
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND access_count > 0
              AND NOT ('Creative_&_Media' = ANY(topic_tags))
        """), {"rate": DECAY_RATE_ACCESSED, "cutoff": cutoff, "cycles": cycles})

        # Slow decay for creative turns (1% per day)
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * POWER(:rate, :cycles)
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND 'Creative_&_Media' = ANY(topic_tags)
        """), {"rate": CREATIVE_DECAY_RATE, "cutoff": cutoff, "cycles": cycles})

        # Creative floor: never drop below 0.3
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = 0.3
            WHERE 'Creative_&_Media' = ANY(topic_tags)
              AND decay_score < 0.3
        """))

        # Archive turns below threshold
        db.execute(text("""
            UPDATE episodic_memory
            SET is_archived = TRUE
            WHERE decay_score < :archive_threshold AND is_archived = FALSE
        """), {"archive_threshold": ARCHIVE_THRESHOLD})

        # Move extremely stale archived turns to cold_storage
        cold_rows = db.execute(text("""
            SELECT id, raw_text, summary_text, topic_tags, timestamp
            FROM episodic_memory
            WHERE is_archived = TRUE AND decay_score < :cold_threshold
        """), {"cold_threshold": COLD_THRESHOLD}).fetchall()

        for row in cold_rows:
            db.execute(text("""
                INSERT INTO cold_storage (id, archived_at, raw_text, summary_text, topic_tags, timestamp)
                VALUES (:id, :now, :raw, :summary, :tags, :ts)
                ON CONFLICT (id) DO NOTHING
            """), {
                "id": row.id,
                "now": datetime.now(timezone.utc),
                "raw": row.raw_text,
                "summary": row.summary_text,
                "tags": row.topic_tags,
                "ts": row.timestamp
            })
            db.execute(text("DELETE FROM episodic_memory WHERE id = :id"), {"id": row.id})

        db.commit()
        logger.info("decay_cycle_complete", archived=len(cold_rows), cycles=cycles)

    except Exception as exc:
        db.rollback()
        logger.error("decay_failed", error=str(exc))
        raise
    finally:
        db.close()
