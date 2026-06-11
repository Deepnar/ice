"""Codex Extractor Subsystem – Structural Ingestion Plane."""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import structlog

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import (
    CodexEntity, CodexEdge, CodexEvent, IdempotencyKey, EpisodicMemory
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy, is_user_active

logger = structlog.get_logger("ice.workers.codex")
# Dedicated extraction client (port 8003)
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
bg_client = get_bg_client()
CODEX_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


def generate_uuid5(canonical_name: str) -> uuid.UUID:
    """Derive deterministic UUIDv5 identifier for a canonical entity node."""
    return uuid.uuid5(CODEX_NAMESPACE, canonical_name.strip().lower())


def extract_triplets(text: str, model_override: str = "") -> list:
    prompt = (
        "You are an entity extraction tool. Extract subject-relation-object triplets from the text.\n"
        "Each triplet must capture a fact: a subject (entity or concept), a relation (verb or verb phrase), "
        "and an object (another entity or concept).\n"
        "Return ONLY a valid JSON array of objects. Each object must have exactly three keys: "
        "\"subject\", \"relation\", \"object\".\n"
        "If no factual triplets exist, return an empty array [].\n"
        "Do NOT include any other text, markdown, or explanation.\n\n"
        "Example:\n"
        "Text: \"ICE uses PostgreSQL for memory storage and Redis for task management.\"\n"
        "Output: [{\"subject\":\"ICE\",\"relation\":\"uses\",\"object\":\"PostgreSQL\"}, "
        "{\"subject\":\"ICE\",\"relation\":\"uses\",\"object\":\"Redis\"}]\n\n"
        "Now process this text:"
    )
    try:
        model_name = model_override if model_override else get_bg_model_name()
        completion = bg_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a JSON-only entity extraction tool. Never output anything but JSON."},
                {"role": "user", "content": f"Text:\n{text}\n\n{prompt}"}
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=30.0
        )
        raw = completion.choices[0].message.content.strip()
        logger.debug("extraction_raw_response", raw=raw)

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Parse first valid JSON value
        decoder = json.JSONDecoder()
        parsed = None
        try:
            parsed, _ = decoder.raw_decode(raw)
        except json.JSONDecodeError:
            # Fallback regex for individual triplet objects
            triplet_pattern = re.compile(
                r'\{\s*"subject"\s*:\s*"([^"]+)"\s*,\s*"relation"\s*:\s*"([^"]+)"\s*,\s*"object"\s*:\s*"([^"]+)"\s*\}',
                re.DOTALL
            )
            matches = triplet_pattern.findall(raw)
            if matches:
                return [{"subject": s, "relation": r, "object": o} for s, r, o in matches]
            return []

        if isinstance(parsed, list):
            valid = []
            for item in parsed:
                if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                    valid.append(item)
            return valid

        return []

    except Exception as err:
        logger.error("triplet_parsing_boundary_failed", error=str(err))
        return []




def get_or_create_entity(db, name: str) -> CodexEntity:
    """Resolves structural identity records across global name and alias spaces."""
    canonical = name.strip().lower()
    entity = db.query(CodexEntity).filter_by(canonical_name=canonical).first()
    if entity:
        return entity

    entity = db.query(CodexEntity).filter(CodexEntity.aliases.any(canonical)).first()
    if entity:
        return entity

    new_entity = CodexEntity(
        id=generate_uuid5(canonical),
        canonical_name=canonical,
        aliases=[name],
        tags=[],
        properties={},
        context_payload="",
        last_updated=datetime.now(timezone.utc)
    )
    db.add(new_entity)
    db.flush()
    return new_entity


def handle_triplet(db, subject_name: str, relation: str, object_name: str, batch_id: str):
    """Integrates extraction assertions into the transaction context."""
    subj = get_or_create_entity(db, subject_name)
    obj = get_or_create_entity(db, object_name)

    existing_edge = db.query(CodexEdge).filter(
        CodexEdge.source_id == subj.id,
        CodexEdge.target_id == obj.id,
        CodexEdge.valid_until == None
    ).first()

    if existing_edge:
        if existing_edge.relation == relation:
            # Corroboration pass logic
            existing_edge.strength += 1.0
            if existing_edge.strength >= 2.0 and existing_edge.confidence == "pending":
                existing_edge.confidence = "active"

            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_strengthened",
                payload={
                    "edge_id": str(existing_edge.id),
                    "relation": relation,
                    "target_id": str(obj.id)
                },
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
        else:
            # Contradiction resolution pass logic (INV-4)
            existing_edge.valid_until = datetime.now(timezone.utc)
            
            # Explicit Client-Side UUID Generation to guarantee valid event tracking logs
            new_edge_id = uuid.uuid4()
            db.add(CodexEdge(
                id=new_edge_id,
                source_id=subj.id,
                target_id=obj.id,
                relation=relation,
                strength=1.0,
                source_batch=batch_id,
                confidence="pending",
                valid_from=datetime.now(timezone.utc)
            ))
            
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_expired",
                payload={
                    "edge_id": str(existing_edge.id),
                    "relation": existing_edge.relation
                },
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_added",
                payload={
                    "edge_id": str(new_edge_id),
                    "relation": relation,
                    "target_id": str(obj.id)
                },
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
    else:
        new_edge_id = uuid.uuid4()
        db.add(CodexEdge(
            id=new_edge_id,
            source_id=subj.id,
            target_id=obj.id,
            relation=relation,
            strength=1.0,
            source_batch=batch_id,
            confidence="pending",
            valid_from=datetime.now(timezone.utc)
        ))
        db.add(CodexEvent(
            entity_id=subj.id,
            event_type="edge_added",
            payload={
                "edge_id": str(new_edge_id),
                "relation": relation,
                "target_id": str(obj.id)
            },
            timestamp=datetime.now(timezone.utc),
            batch_source=batch_id
        ))


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def extract_codex(self, batch_id: str, model_used: str = ""):
    """Executes background semantic link mutations across target graph states."""
    log = logger.bind(batch_id=batch_id)

    if is_gpu_busy():
        raise self.retry(countdown=30)
    if settings.background_model_mode == "shared" and is_user_active():
        raise self.retry(countdown=30)
    idempotency_key = hashlib.sha256(f"codex:{batch_id}".encode()).hexdigest()
    db = SessionLocal()
    
    try:
        if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
            return

        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn or not turn.lossless_flag:
            return

        triplets = extract_triplets(turn.raw_text, model_used)
        for triplet in triplets:
            s = triplet.get("subject", "").strip()
            r = triplet.get("relation", "").strip()
            o = triplet.get("object", "").strip()
            if s and r and o:
                handle_triplet(db, s, r, o, batch_id)

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("codex_graph_assertions_committed", extracted_count=len(triplets))

    except Exception as exc:
        db.rollback()
        log.error("codex_extraction_aborted", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()