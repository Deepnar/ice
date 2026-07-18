"""Procedural Extractor – identifies recurring behavioural patterns."""

import hashlib
import uuid
from datetime import datetime, timezone
import structlog
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ProceduralMemory, IdempotencyKey

logger = structlog.get_logger("ice.workers.procedural")
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name
bg_client = get_bg_client()
# Load the embedding model once globally – prevents disk I/O starvation
pattern_embedder = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cpu",
    truncate_dim=384
)

def encode_pattern(text: str):
    return pattern_embedder.encode(text, convert_to_tensor=False).tolist()


def extract_procedural(batch_id: str, model_used: str = ""):
    """Scan the exchange for recurring workflows or habits. Plain callable
    since C7 — gating/retries live in the maintenance runtime."""
    log = logger.bind(batch_id=batch_id)

    idempotency_key = hashlib.sha256(f"procedural:{batch_id}".encode()).hexdigest()
    db = SessionLocal()
    try:
        if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
            return

        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn:
            return
        # E1 (D1): a pattern observed inside a project-attached conversation
        # is a project convention — scoped by project_id, not a fourth store.
        project_id = db.execute(
            text("SELECT project_id FROM conversations WHERE id = :cid"),
            {"cid": turn.conversation_id}).scalar()

        prompt = (
            "Analyze the following conversation exchange and identify any recurring workflow or behavioural pattern "
            "that the user consistently follows. Examples of patterns: "
            "\"When debugging, the user first asks for the error message, then the surrounding code.\" "
            "\"When starting a new project, the user creates a Pydantic model before the FastAPI route.\"\n\n"
            "If you detect a clear, repeatable pattern, output a one‑sentence description of that pattern.\n"
            "If no pattern is evident, output 'NONE'.\n"
            "Do NOT include any other text or explanation."
        )
        model_name = model_used if model_used else get_bg_model_name()
        completion = bg_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a behavioural pattern detector."},
                {"role": "user", "content": f"Text:\n{turn.raw_text}\n\n{prompt}"}
            ],
            temperature=0.0,
            max_tokens=80,
            timeout=bg_timeout(80)
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
                embedding=embedding,
                project_id=project_id,
            )
            db.add(new_pattern)

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("procedural_extraction_complete", pattern=pattern_text[:50])

    except Exception as exc:
        db.rollback()
        log.error("procedural_extraction_failed", error=str(exc))
        raise
    finally:
        db.close()