"""Clustering Worker – groups unassigned episodic turns into named clusters."""

import structlog
import json
import re
from datetime import datetime, timezone
from sqlalchemy import text
from openai import OpenAI

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ContextCluster
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.clustering")
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def cluster_turns(self):
    """Periodic task: scan unassigned turns and propose clusters."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # Find turns with no cluster assigned
        unassigned = db.query(EpisodicMemory).filter_by(cluster_id=None).limit(100).all()
        if not unassigned:
            return

        # Compile raw texts and ask the model to suggest cluster names
        texts = "\n---\n".join([t.raw_text[:200] for t in unassigned])
        prompt = (
            "Given the following conversation fragments, suggest 1‑3 cluster names that group related topics. "
            "Each name should be a short, descriptive phrase (e.g., \"PostgreSQL Schema Design\", \"Creative Writing – FLAW Lore\"). "
            "Output a JSON array of strings, e.g. [\"ICE Development\", \"Story Writing\"]. "
            "If only one theme is present, output a single‑element array."
        )
        completion = bg_client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are a topic clustering engine."},
                {"role": "user", "content": f"{prompt}\n\n{texts}"}
            ],
            temperature=0.0,
            max_tokens=100,
            timeout=30.0
        )
        raw = completion.choices[0].message.content.strip()

        # Robust JSON extraction to handle markdown fences
        try:
            json_match = re.search(r"\[\s*.*?\s*\]", raw, re.DOTALL)
            if not json_match:
                return
            cluster_names = json.loads(json_match.group(0))
        except Exception as e:
            logger.error("clustering_json_parse_error", error=str(e))
            return

        # Distribute turns as evenly as possible
        if not cluster_names:
            return

        num_clusters = len(cluster_names)
        total_turns = len(unassigned)
        base_size = total_turns // num_clusters
        remainder = total_turns % num_clusters

        start = 0
        for i, name in enumerate(cluster_names):
            chunk_size = base_size + (1 if i < remainder else 0)
            if chunk_size == 0:
                continue
            chunk = unassigned[start:start + chunk_size]
            cluster = db.query(ContextCluster).filter_by(name=name).first()
            if not cluster:
                cluster = ContextCluster(name=name, description="", created_at=datetime.now(timezone.utc))
                db.add(cluster)
                db.flush()
            for turn in chunk:
                turn.cluster_id = cluster.id
            db.commit()
            logger.info("cluster_assigned", cluster_name=name, turns_assigned=len(chunk))
            start += chunk_size

    except Exception as exc:
        db.rollback()
        logger.error("clustering_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()