"""Periodic task that merges similar clusters based on embedding similarity."""

import structlog
from datetime import datetime, timezone
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import ContextCluster, EpisodicMemory
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.cluster_merge")

@app.task(bind=True, max_retries=2, default_retry_delay=120)
def merge_similar_clusters(self):
    """Scan clusters and merge those with cosine similarity > 0.85."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        clusters = db.query(ContextCluster).filter(ContextCluster.embedding != None).all()
        if len(clusters) < 2:
            return

        # Pairwise comparison (O(n²), acceptable for the current cluster count)
        merged = set()
        for i, c1 in enumerate(clusters):
            if c1.id in merged:
                continue
            for c2 in clusters[i+1:]:
                if c2.id in merged:
                    continue
                dot = sum(a*b for a, b in zip(c1.embedding, c2.embedding))
                if dot >= 0.85:
                    # Merge c2 into c1
                    db.execute(
                        text("UPDATE episodic_memory SET cluster_id = :c1 WHERE cluster_id = :c2"),
                        {"c1": c1.id, "c2": c2.id}
                    )
                    # Average the embeddings
                    avg_emb = [(a+b)/2 for a, b in zip(c1.embedding, c2.embedding)]
                    c1.embedding = avg_emb
                    c1.updated_at = datetime.now(timezone.utc)
                    db.delete(c2)
                    merged.add(c2.id)
                    logger.info("clusters_merged", kept=c1.name, removed=c2.name)

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("cluster_merge_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()