"""Procedural Extractor – identifies recurring behavioural patterns."""

import hashlib
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import structlog
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ProceduralMemory, IdempotencyKey
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.procedural")
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

# Load the embedding model once globally – prevents disk I/O starvation
pattern_embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def encode_pattern(text: str):
    return pattern_embedder.encode(text, convert_to_tensor=False).tolist()


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def extract_procedural(self, batch_id: str):
    """Scan the exchange for recurring workflows or habits."""
    log = logger.bind(batch_id=batch_id)

    if is_gpu_busy():
        raise self.retry(countdown=30)

    idempotency_key = hashlib.sha256(f"procedural:{batch_id}".encode()).hexdigest()
    db = SessionLocal()
    try:
        if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
            return

        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn:
            return

        # Call the 1.5B model to detect patterns
        prompt = (
            "Identify any recurring workflows, decision sequences, or behavioural patterns "
            "in this exchange that represent how the user approaches problems. "
            "If no recurring pattern is evident, output 'NONE'. "
            "Otherwise output a short one‑sentence description of the pattern."
        )
        completion = bg_client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are a behavioural pattern detector."},
                {"role": "user", "content": f"Text:\n{turn.raw_text}\n\n{prompt}"}
            ],
            temperature=0.0,
            max_tokens=80,
            timeout=30.0
        )
        pattern_text = completion.choices[0].message.content.strip()
        if pattern_text.upper() == "NONE" or not pattern_text:
            return

        # Encode the pattern for similarity matching
        embedding = encode_pattern(pattern_text)

        # Force PostgreSQL to accept the list as a vector via explicit cast
        similarity_query = text("""
            SELECT id, 1 - (embedding <=> CAST(:emb AS vector)) AS sim
            FROM procedural_memory
            WHERE embedding IS NOT NULL
            ORDER BY sim DESC LIMIT 1
        """)
        row = db.execute(similarity_query, {"emb": str(embedding)}).first()

        if row and row.sim > 0.85:
            # Reinforce existing pattern
            existing = db.query(ProceduralMemory).get(row.id)
            existing.reinforcement_count += 1
            existing.last_observed = datetime.now(timezone.utc)
            if existing.reinforcement_count >= 3 and existing.confidence_score < 0.8:
                existing.confidence_score = 0.8
                existing.is_active = True
        else:
            # Insert new pending pattern
            new_pattern = ProceduralMemory(
                pattern_name=pattern_text[:80],
                pattern_description=pattern_text,
                topic_tags=turn.topic_tags or [],
                trigger_conditions={},
                reinforcement_count=1,
                confidence_score=0.3,
                first_observed=datetime.now(timezone.utc),
                last_observed=datetime.now(timezone.utc),
                is_active=False,
                source_batch_ids=[uuid.UUID(batch_id)],
                embedding=embedding
            )
            db.add(new_pattern)

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("procedural_extraction_complete", pattern=pattern_text[:50])

    except Exception as exc:
        db.rollback()
        log.error("procedural_extraction_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()