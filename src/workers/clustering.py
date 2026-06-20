"""Clustering Worker – groups unassigned episodic turns into named clusters."""

from sentence_transformers import SentenceTransformer
import structlog
import json
import re
from datetime import datetime, timezone
from sqlalchemy import text
from openai import OpenAI
import re
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ContextCluster
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy, is_user_active

logger = structlog.get_logger("ice.workers.clustering")
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
bg_client = get_bg_client()

@app.task(bind=True, max_retries=2, default_retry_delay=60)
def cluster_turns(self):
    """Periodic task: scan unassigned turns and propose clusters."""
    if is_gpu_busy():
        raise self.retry(countdown=60)
    if settings.background_model_mode == "shared" and is_user_active():
        raise self.retry(countdown=30)
    db = SessionLocal()
    try:
        # Find turns with no cluster assigned
        unassigned = db.query(EpisodicMemory).filter_by(cluster_id=None).limit(50).all()
        if not unassigned:
            return

        # Compile raw texts and ask the model to suggest cluster names
        # Extract topic tags and a brief snippet to keep the prompt tiny
        descriptions = []
        for t in unassigned:
            tags = ", ".join(t.topic_tags) if t.topic_tags else "untagged"
            snippet = t.raw_text[:80].replace("\n", " ")
            descriptions.append(f"[{tags}] {snippet}")
        texts = "\n".join(descriptions)
        prompt = (
            "The following are topic tags and brief snippets from several conversation turns. "
            "Suggest 1-3 cluster names that group related topics. "
            "Output ONLY a valid JSON array of strings, e.g. [\"Database Systems\", \"Creative Writing\"]. "
            "Each string must be a short, descriptive phrase. "
            "Do NOT output anything else. Do NOT include markdown or explanations."
        )
        completion = bg_client.chat.completions.create(
            model=get_bg_model_name(),
            messages=[
                {"role": "system", "content": "You are a topic clustering engine."},
                {"role": "user", "content": f"{prompt}\n\n{texts}"}
            ],
            temperature=0.0,
            max_tokens=100,
            timeout=30.0
        )
        raw = completion.choices[0].message.content.strip()
        logger.info("clustering_raw_response", raw=raw)

        # Robust JSON extraction to handle markdown fences
                # Robust JSON extraction to handle markdown fences and missing quotes
        try:
            json_match = re.search(r"\[\s*.*?\s*\]", raw, re.DOTALL)
            if not json_match:
                logger.error("clustering_no_json_match", raw=raw)
                return
            json_str = json_match.group(0)
            # Try to parse as-is
            try:
                cluster_names = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback: add missing quotes around unquoted strings
                fixed = re.sub(r'([\[\s,])([A-Za-z0-9_&+]+)([, \s\]])', r'\1"\2"\3', json_str)
                try:
                    cluster_names = json.loads(fixed)
                except json.JSONDecodeError:
                    logger.error("clustering_json_parse_error", raw=raw)
                    return
        except Exception as e:
            logger.error("clustering_json_parse_error", error=str(e))
            return
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
                embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", truncate_dim=384)
                emb = embedder.encode(name, convert_to_tensor=False).tolist()
                cluster = ContextCluster(
                    name=name, description="",
                    embedding=emb,
                    created_at=datetime.now(timezone.utc)
                )
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