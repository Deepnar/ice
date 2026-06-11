"""Reflection Worker – full implementation: session synthesis, pattern crystallization,
   memory slot evolution, Codex enrichment, motif detection."""

import structlog, json, re, uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from openai import OpenAI

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import (
    EpisodicMemory, SessionSummary, MemorySlot, CodexEntity, CodexEvent,
    ProceduralMemory, ContextCluster
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy, is_user_active

logger = structlog.get_logger("ice.workers.reflection")
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
bg_client = get_bg_client()
# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
SUMMARY_PROMPT = (
    "Generate a structured session summary from the following conversation turns.\n"
    "Output ONLY a valid JSON object with these keys:\n"
    "  - \"topics_covered\": a list of strings (e.g., [\"PostgreSQL\", \"FastAPI\"])\n"
    "  - \"decisions_made\": a string describing any decisions\n"
    "  - \"unresolved_items\": a string describing any unresolved questions\n"
    "  - \"entities_updated\": a list of canonical entity names that appeared\n"
    "  - \"patterns_observed\": a list of strings describing observed behavioural patterns\n\n"
    "If a field has no content, use an empty list [] for lists, or an empty string \"\" for strings.\n"
    "Do NOT include markdown or additional text."
)

CRYSTALLIZATION_PROMPT = (
    "Below are snippets from multiple recent conversation sessions. Identify any recurring "
    "behavioural patterns or workflows that the user consistently follows. For each pattern, "
    "output a single descriptive sentence. Return ONLY a JSON array of strings. If no patterns "
    "are found, return an empty array []."
)

SLOT_EVOLUTION_PROMPT = (
    "You are analysing a user's recent conversations. Based on the content, suggest if any of "
    "the following persistent memory slots should be updated:\n"
    "- project_context: what the user is currently working on\n"
    "- user_preferences: how the user likes to interact\n"
    "- guidance: rules the AI should follow\n\n"
    "Output ONLY a JSON object with keys matching the slot names (if an update is needed) and "
    "the proposed new content as the value. If no update is needed for a slot, omit the key. "
    "The proposed content should be a concise paragraph. Do NOT include markdown."
)

ENRICHMENT_PROMPT = (
    "The following is a context payload for a knowledge graph entity. It is currently very thin. "
    "Given additional conversation passages, write an enriched, factual description of the entity. "
    "Output ONLY the enriched description, no markdown."
)

MOTIF_PROMPT = (
    "Below are conversations from multiple recent sessions. Identify any recurring thematic motifs "
    "that do not yet correspond to a named project or cluster. For each motif, suggest a short, "
    "descriptive cluster name. Output ONLY a JSON array of strings. If none, return []."
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _robust_json(raw: str) -> dict:
    """Try to extract a JSON object from model output, fall back to empty dict."""
    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {}
    except Exception:
        return {}

def _robust_list(raw: str) -> list:
    try:
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else []
    except Exception:
        return []

# ------------------------------------------------------------------
# Main task
# ------------------------------------------------------------------
@app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_reflection(self):
    """Execute a full reflection pass: synthesis, patterns, slots, enrichment, motifs."""
    if is_gpu_busy():
        raise self.retry(countdown=60)
    if settings.background_model_mode == "shared" and is_user_active():
        raise self.retry(countdown=30)
    db = SessionLocal()
    try:
        # 1. Load recent turns (last 200 across all conversations, for breadth)
        recent = db.query(EpisodicMemory).order_by(
            EpisodicMemory.timestamp.desc()
        ).limit(200).all()
        if not recent:
            return
        recent.reverse()  # chronological

        # ---- Session Synthesis ----
        _synthesize_session(db, recent)

        # ---- Pattern Crystallization ----
        _crystallize_patterns(db, recent)

        # ---- Memory Slot Evolution ----
        _evolve_memory_slots(db, recent)

        # ---- Codex Enrichment ----
        _enrich_codex_entities(db)

        # ---- Motif Detection ----
        _detect_motifs(db, recent)

        db.commit()
        logger.info("reflection_full_pass_complete")

    except Exception as exc:
        db.rollback()
        logger.error("reflection_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


# ------------------------------------------------------------------
# Session Synthesis (existing logic, kept)
# ------------------------------------------------------------------
def _synthesize_session(db, turns):
    full_text = "\n\n".join([t.raw_text for t in turns])
    words = full_text.split()
    if len(words) > 3000:
        full_text = " ".join(words[-3000:])
    completion = bg_client.chat.completions.create(
        model=get_bg_model_name(),
        messages=[
            {"role": "system", "content": "You are a session analysis engine. Output only JSON."},
            {"role": "user", "content": f"{SUMMARY_PROMPT}\n\n{full_text}"}
        ],
        temperature=0.0, max_tokens=400, timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()
    data = _robust_json(raw)

    summary = SessionSummary(
        topics_covered=data.get("topics_covered", []),
        decisions_made=data.get("decisions_made", ""),
        unresolved_items=data.get("unresolved_items", ""),
        entities_updated=data.get("entities_updated", []),
        patterns_observed=data.get("patterns_observed", [])
    )
    db.add(summary)

    # Update pending_items if unresolved items found
    unresolved = data.get("unresolved_items")
    if unresolved and isinstance(unresolved, str) and unresolved.strip():
        slot = db.query(MemorySlot).filter_by(slot_name="pending_items").first()
        if slot:
            slot.content = (slot.content or "") + "\n" + unresolved
            slot.version += 1
            slot.last_updated = datetime.now(timezone.utc)
            slot.updated_by = "reflection_worker"


# ------------------------------------------------------------------
# Pattern Crystallization
# ------------------------------------------------------------------
def _crystallize_patterns(db, turns):
    # Build a compact representation (last 1500 words)
    text = "\n".join([t.raw_text[:200] for t in turns])
    if len(text.split()) > 1500:
        text = " ".join(text.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a behavioural pattern detector."},
            {"role": "user", "content": f"{CRYSTALLIZATION_PROMPT}\n\n{text}"}
        ],
        temperature=0.0, max_tokens=200, timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()
    patterns = _robust_list(raw)
    for desc in patterns:
        if not isinstance(desc, str) or not desc.strip():
            continue
        # Check for existing pattern by embedding similarity
        from src.workers.procedural_extractor import encode_pattern
        emb = encode_pattern(desc)
        try:
            similar = db.execute(
                text("SELECT id, 1 - (embedding <=> CAST(:emb AS vector)) AS sim FROM procedural_memory WHERE embedding IS NOT NULL ORDER BY sim DESC LIMIT 1"),
                {"emb": str(emb)}
            ).first()
            if similar and similar.sim > 0.85:
                existing = db.query(ProceduralMemory).get(similar.id)
                existing.reinforcement_count += 1
                existing.last_observed = datetime.now(timezone.utc)
                if existing.reinforcement_count >= 3 and existing.confidence_score < 0.8:
                    existing.confidence_score = 0.8
                    existing.is_active = True
            else:
                new_pat = ProceduralMemory(
                    pattern_name=desc[:80],
                    pattern_description=desc,
                    topic_tags=turns[0].topic_tags if turns else [],
                    trigger_conditions={},
                    reinforcement_count=1,
                    confidence_score=0.3,
                    first_observed=datetime.now(timezone.utc),
                    last_observed=datetime.now(timezone.utc),
                    is_active=False,
                    source_batch_ids=[t.batch_id for t in turns[:10]],
                    embedding=emb
                )
                db.add(new_pat)
        except Exception as e:
            logger.error("pattern_crystallization_error", error=str(e))


# ------------------------------------------------------------------
# Memory Slot Evolution
# ------------------------------------------------------------------
def _evolve_memory_slots(db, turns):
    text = "\n".join([t.raw_text[:200] for t in turns])
    if len(text.split()) > 1500:
        text = " ".join(text.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a memory slot analyst. Output only JSON."},
            {"role": "user", "content": f"{SLOT_EVOLUTION_PROMPT}\n\n{text}"}
        ],
        temperature=0.0, max_tokens=300, timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()
    proposals = _robust_json(raw)
    for slot_name, content in proposals.items():
        if slot_name in ("project_context", "user_preferences", "guidance") and isinstance(content, str) and content.strip():
            # Insert into review_queue for user confirmation (Phase C)
            db.execute(
                text("INSERT INTO review_queue (item_type, item_content) VALUES ('memory_slot_update', :payload)"),
                {"payload": json.dumps({"slot_name": slot_name, "proposed_content": content})}
            )


# ------------------------------------------------------------------
# Codex Enrichment
# ------------------------------------------------------------------
def _enrich_codex_entities(db):
    # Find entities with short context_payload (less than 100 chars)
    thin_entities = db.query(CodexEntity).filter(
        CodexEntity.context_payload == None
    ).all()[:10]  # limit to 10 per run
    for entity in thin_entities:
        if entity.context_payload and len(entity.context_payload) > 100:
            continue
        # Find episodic turns that mention this entity
        batch_ids = db.execute(
            text("SELECT batch_source FROM codex_events WHERE entity_id = :eid"),
            {"eid": entity.id}
        ).fetchall()
        if not batch_ids:
            continue
        passages = []
        for (bid,) in batch_ids:
            turn = db.query(EpisodicMemory).filter_by(batch_id=bid).first()
            if turn:
                passages.append(turn.raw_text[:500])
        if not passages:
            continue
        combined = "\n".join(passages)
        completion = bg_client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are a knowledge graph enricher. Write a factual description."},
                {"role": "user", "content": f"{ENRICHMENT_PROMPT}\nCurrent payload: {entity.context_payload or ''}\nRelevant passages:\n{combined[:2000]}"}
            ],
            temperature=0.0, max_tokens=300, timeout=30.0
        )
        enriched = completion.choices[0].message.content.strip()
        entity.context_payload = enriched
        entity.last_updated = datetime.now(timezone.utc)
        db.add(CodexEvent(
            entity_id=entity.id,
            event_type="context_appended",
            payload={"enriched_from_reflection": True},
            batch_source=uuid.uuid4()
        ))


# ------------------------------------------------------------------
# Motif Detection
# ------------------------------------------------------------------
def _detect_motifs(db, turns):
    text = "\n".join([t.raw_text[:200] for t in turns])
    if len(text.split()) > 1500:
        text = " ".join(text.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a thematic motif detector. Output only JSON."},
            {"role": "user", "content": f"{MOTIF_PROMPT}\n\n{text}"}
        ],
        temperature=0.0, max_tokens=150, timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()
    motifs = _robust_list(raw)
    for motif in motifs:
        if isinstance(motif, str) and motif.strip():
            db.execute(
                text("INSERT INTO review_queue (item_type, item_content) VALUES ('new_cluster_proposal', :payload)"),
                {"payload": json.dumps({"cluster_name": motif})}
            )