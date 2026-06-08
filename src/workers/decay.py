"""Decay Worker – applies time‑based memory decay and archival."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ColdStorage
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.decay")
DECAY_RATE = 0.97          # 3% decay per day
ARCHIVE_THRESHOLD = 0.1
COLD_THRESHOLD = 0.05


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def apply_decay(self):
    """Daily task: decay old turns, archive stale ones, move to cold storage."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # 1. Decay turns older than 7 days, not bookmarked, not decay_immune
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
        """), {"rate": DECAY_RATE, "cutoff": cutoff})

        # 2. Strengthen turns retrieved in the last 24 hours
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = LEAST(decay_score + 0.15, 1.0)
            WHERE access_count > 0
              AND timestamp < :recent
        """), {"recent": recent})

        # 3. Archive turns below threshold
        db.execute(text("""
            UPDATE episodic_memory
            SET is_archived = TRUE
            WHERE decay_score < :archive_threshold AND is_archived = FALSE
        """), {"archive_threshold": ARCHIVE_THRESHOLD})

        # 4. Move extremely stale archived turns to cold_storage
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
        logger.info("decay_cycle_complete", archived=len(cold_rows))

    except Exception as exc:
        db.rollback()
        logger.error("decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()