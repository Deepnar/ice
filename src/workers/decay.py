"""Decay Worker – applies access‑weighted memory decay and archival."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ColdStorage
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.decay")
DECAY_RATE_UNACCESSED = 0.95   # 5% decay per day for never‑accessed turns
DECAY_RATE_ACCESSED = 0.98     # 2% decay for turns that have been accessed
STRENGTHEN_AMOUNT = 0.15
ARCHIVE_THRESHOLD = 0.1
COLD_THRESHOLD = 0.05


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def apply_decay(self):
    """Daily task: decay old turns, archive stale ones, move to cold storage."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        # Access‑weighted decay: unaccessed turns decay faster
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND access_count = 0
        """), {"rate": DECAY_RATE_UNACCESSED, "cutoff": cutoff})

        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND access_count > 0
        """), {"rate": DECAY_RATE_ACCESSED, "cutoff": cutoff})

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
        logger.info("decay_cycle_complete", archived=len(cold_rows))

    except Exception as exc:
        db.rollback()
        logger.error("decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()