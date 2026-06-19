"""Codex Extractor Subsystem – Structural Ingestion Plane."""

import hashlib
import json
import re
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import structlog
from sentence_transformers import SentenceTransformer
from src.api.config import settings
from src.api.db import SessionLocal
from sqlalchemy.orm.attributes import flag_modified

from src.memory.models import (
    CodexEntity, CodexEdge, CodexEvent, IdempotencyKey, EpisodicMemory
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy, is_user_active

# Module‑level embedder for new entity embeddings
embedder = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cpu",
    truncate_dim=384
)

logger = structlog.get_logger("ice.workers.codex")

# Dedicated extraction client (port 8003)
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
bg_client = get_bg_client()
CODEX_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

# -----------------------------------------------------------------
# Controlled relation vocabulary for Codex 2.0
# -----------------------------------------------------------------
PROPERTY_RELATIONS = {
    "name", "age", "description", "species", "home", "role", "profession",
    "title", "status", "type"
}
"""Relations that update the source entity's properties JSONB and expire previous edges."""

MULTI_VALUED_RELATIONS = {
    "uses", "imports", "includes", "member_of", "depends_on",
    "friend", "ally", "enemy", "colleague", "tag", "category",
    "works_with", "collaborates_with", "attended", "participated_in",
    "studied", "taught", "wrote", "published", "released",
    "features", "contains"
}
"""Relations that allow multiple active edges simultaneously (no auto‑expiry)."""

SINGLE_VALUED_RELATIONS = {
    "part_of", "works_on", "created", "located_in", "has", "is",
    "offers", "requires", "provides", "ranks", "connects_to",
    "employs", "studies", "applies_to", "extends", "implements",
    "calls", "returns", "founded", "founded_by", "works_at",
    "studies_at", "lives_in", "born_in", "died_in", "married_to",
    "parent_of", "child_of", "sibling_of", "mentor_of",
    "supervised_by", "owned_by", "operated_by", "manufactured_by",
    "sold_by", "purchased_from"
}
"""Single‑valued relations: a new edge auto‑expires any previous active edge with the same source and relation."""

ALLOWED_RELATIONS = PROPERTY_RELATIONS | MULTI_VALUED_RELATIONS | SINGLE_VALUED_RELATIONS


def generate_uuid5(canonical_name: str) -> uuid.UUID:
    """Derive deterministic UUIDv5 identifier for a canonical entity node."""
    return uuid.uuid5(CODEX_NAMESPACE, canonical_name.strip().lower())


def extract_triplets(text: str, model_override: str = "", topic_tags: Optional[List[str]] = None) -> list:
    """Extract structured triplets using a controlled relation vocabulary."""
    relation_list = ", ".join(sorted(ALLOWED_RELATIONS))
    
    prompt = (
        "You are a precise fact extractor. Convert the given text into a JSON array of "
        "subject‑relation‑object triplets.\n\n"
        "STRICT RULES:\n"
        "1. Use ONLY these relations:\n"
        f"   {relation_list}\n"
        "   If a fact does not fit any of these relations, SKIP IT – never invent a new relation.\n"
        "2. Canonicalise subjects and objects: lowercase, singular, no punctuation, concise.\n"
        "   Example: \"PostgreSQL\" → \"postgresql\", \"the goo blade\" → \"goo blade\".\n"
        "3. For facts that describe a property of something (e.g., name, age, role, profession, description), "
        "use the property relation itself as the relation. Example:\n"
        "   \"Kael is a fire mage\" → {\"subject\":\"kael\",\"relation\":\"role\",\"object\":\"fire mage\"}\n"
        "4. Output ONLY a JSON array. No markdown, no explanation.\n\n"
        "EXAMPLES:\n"
        "Text: \"ICE uses PostgreSQL for memory and Redis for tasks.\"\n"
        "Output: [{\"subject\":\"ice\",\"relation\":\"uses\",\"object\":\"postgresql\"},"
        " {\"subject\":\"ice\",\"relation\":\"uses\",\"object\":\"redis\"}]\n\n"
        "Text: \"My character Kael is a fire mage from the northern kingdom.\"\n"
        "Output: [{\"subject\":\"kael\",\"relation\":\"role\",\"object\":\"fire mage\"},"
        " {\"subject\":\"kael\",\"relation\":\"home\",\"object\":\"northern kingdom\"}]\n\n"
        "Text: \"FastAPI extends Starlette and depends on Pydantic.\"\n"
        "Output: [{\"subject\":\"fastapi\",\"relation\":\"extends\",\"object\":\"starlette\"},"
        " {\"subject\":\"fastapi\",\"relation\":\"depends_on\",\"object\":\"pydantic\"}]\n"
    )

    # Optional code‑specific instructions
    code_prompt = ""
    if topic_tags and "Software_&_Tech" in topic_tags:
        code_prompt = (
            "\nAdditionally, extract code‑specific entities like function names, class names, "
            "library names, and technical dependencies. Use relations such as "
            "\"uses\", \"imports\", \"extends\", \"implements\", \"calls\", \"returns\".\n"
            "Examples:\n"
            "Text: \"Function calculate_total uses library numpy.\"\n"
            "Output: [{\"subject\":\"calculate_total\",\"relation\":\"uses\",\"object\":\"numpy\"}]\n"
            "Text: \"Class DataLoader extends Dataset.\"\n"
            "Output: [{\"subject\":\"dataloader\",\"relation\":\"extends\",\"object\":\"dataset\"}]\n"
        )
    full_prompt = prompt + code_prompt + "\nNow process this text:"

    try:
        model_name = model_override if model_override else get_bg_model_name()
        completion = bg_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a JSON-only fact extraction tool. Never output anything but JSON."},
                {"role": "user", "content": f"Text:\n{text}\n\n{full_prompt}"}
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

        # Parse JSON
        decoder = json.JSONDecoder()
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
                if isinstance(item, dict) and all(k in item for k in ("subject","relation","object")):
                    # Filter out relations not in our vocabulary (just in case)
                    if item["relation"] in ALLOWED_RELATIONS:
                        valid.append(item)
            return valid

        return []

    except Exception as err:
        logger.error("triplet_parsing_failed", error=str(err))
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
        embedding=embedder.encode(canonical, convert_to_tensor=False).tolist(),
        last_updated=datetime.now(timezone.utc)
    )
    db.add(new_entity)
    db.flush()
    return new_entity

def _regenerate_context_payload(entity: CodexEntity, db) -> None:
    """Rebuild context_payload from current properties and active edges."""
    parts = []
    if entity.properties:
        for k, v in entity.properties.items():
            parts.append(f"{k}: {v}")
    # Add a one‑line summary of active edges (using the same db session)
    active_edges = db.query(CodexEdge).filter(
        CodexEdge.source_id == entity.id,
        CodexEdge.valid_until == None
    ).limit(10).all()
    for edge in active_edges:
        target = db.query(CodexEntity).get(edge.target_id)
        target_name = target.canonical_name if target else "?"
        parts.append(f"{edge.relation} → {target_name}")
    entity.context_payload = "; ".join(parts) if parts else ""

def handle_triplet(db, subject_name: str, relation: str, object_name: str, batch_id: str):
    """Integrates extraction assertions into the transaction context,
    with property‑aware updates, auto‑expiry, multi‑valued support,
    and immediate contradiction activation."""

    subj = get_or_create_entity(db, subject_name)
    obj  = get_or_create_entity(db, object_name)

    # ── 1. Property relations: update entity properties, expire previous edges ──
    if relation in PROPERTY_RELATIONS:
        # Expire any existing active edge of the same relation for this source
        for old_edge in db.query(CodexEdge).filter(
            CodexEdge.source_id == subj.id,
            CodexEdge.relation == relation,
            CodexEdge.valid_until == None
        ).all():
            old_edge.valid_until = datetime.now(timezone.utc)
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_expired",
                payload={"edge_id": str(old_edge.id), "relation": relation},
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))

        # Create a new active edge with strength 3.0
        new_edge_id = uuid.uuid4()
        db.add(CodexEdge(
            id=new_edge_id,
            source_id=subj.id,
            target_id=obj.id,
            relation=relation,
            strength=3.0,
            source_batch=batch_id,
            confidence="active",
            valid_from=datetime.now(timezone.utc)
        ))
        db.add(CodexEvent(
            entity_id=subj.id,
            event_type="edge_added",
            payload={"edge_id": str(new_edge_id), "relation": relation, "target_id": str(obj.id)},
            timestamp=datetime.now(timezone.utc),
            batch_source=batch_id
        ))

        # Update entity properties
        # Update entity properties (JSONB requires explicit flagging for in‑place changes)
        if subj.properties is None:
            subj.properties = {}
        subj.properties[relation] = object_name.strip()
        flag_modified(subj, "properties")          # tell SQLAlchemy the JSONB changed
        subj.last_updated = datetime.now(timezone.utc)

        # Regenerate context payload
        # --- Enhanced context_payload regeneration ---
        _regenerate_context_payload(subj, db)
        return        

    # ── 2. Non‑property relations ──
    existing_active = db.query(CodexEdge).filter(
        CodexEdge.source_id == subj.id,
        CodexEdge.target_id == obj.id,
        CodexEdge.valid_until == None
    ).first()

    if existing_active:
        # Same source‑target pair, same relation → reinforcement
        if existing_active.relation == relation:
            existing_active.strength += 1.0
            if existing_active.strength >= 2.0 and existing_active.confidence == "pending":
                existing_active.confidence = "active"
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_strengthened",
                payload={"edge_id": str(existing_active.id), "relation": relation, "target_id": str(obj.id)},
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
        else:
            # Same pair, different relation → expire old, create new active
            existing_active.valid_until = datetime.now(timezone.utc)
            new_edge_id = uuid.uuid4()
            db.add(CodexEdge(
                id=new_edge_id,
                source_id=subj.id,
                target_id=obj.id,
                relation=relation,
                strength=3.0,
                source_batch=batch_id,
                confidence="active",
                valid_from=datetime.now(timezone.utc)
            ))
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_expired",
                payload={"edge_id": str(existing_active.id), "relation": existing_active.relation},
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_added",
                payload={"edge_id": str(new_edge_id), "relation": relation, "target_id": str(obj.id)},
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
    else:
        # No existing edge between this source and target
        # If the relation is single‑valued, expire any other active edge with the same source and relation
        previous_expired = False
        if relation not in MULTI_VALUED_RELATIONS:
            previous = db.query(CodexEdge).filter(
                CodexEdge.source_id == subj.id,
                CodexEdge.relation == relation,
                CodexEdge.valid_until == None
            ).first()
            if previous:
                previous.valid_until = datetime.now(timezone.utc)
                previous_expired = True
                db.add(CodexEvent(
                    entity_id=subj.id,
                    event_type="edge_expired",
                    payload={"edge_id": str(previous.id), "relation": relation},
                    timestamp=datetime.now(timezone.utc),
                    batch_source=batch_id
                ))

        # Create a new edge – immediately active if a previous edge was expired
        new_edge_id = uuid.uuid4()
        new_strength = 3.0 if previous_expired else 1.0
        new_confidence = "active" if previous_expired else "pending"
        db.add(CodexEdge(
            id=new_edge_id,
            source_id=subj.id,
            target_id=obj.id,
            relation=relation,
            strength=new_strength,
            source_batch=batch_id,
            confidence=new_confidence,
            valid_from=datetime.now(timezone.utc)
        ))
        db.add(CodexEvent(
            entity_id=subj.id,
            event_type="edge_added",
            payload={"edge_id": str(new_edge_id), "relation": relation, "target_id": str(obj.id)},
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

        triplets = extract_triplets(turn.raw_text, model_used, topic_tags=turn.topic_tags)
        for triplet in triplets:
            if isinstance(triplet, dict):
                s_raw = triplet.get("subject")
                r_raw = triplet.get("relation")
                o_raw = triplet.get("object")
                if isinstance(s_raw, str) and isinstance(r_raw, str) and isinstance(o_raw, str):
                    s = s_raw.strip()
                    r = r_raw.strip()
                    o = o_raw.strip()
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