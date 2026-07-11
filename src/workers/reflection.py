"""Reflection Worker – full implementation: session synthesis, pattern crystallization,
   memory slot evolution, Codex enrichment, motif detection."""

import structlog, json, re, uuid
from datetime import datetime, timezone
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import (
    EpisodicMemory, SessionSummary, MemorySlot, CodexEntity, CodexEvent,
    ProceduralMemory
)

logger = structlog.get_logger("ice.workers.reflection")
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name
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
    "Write a concise, factual encyclopedia-style note describing the entity named below, "
    "using ONLY the information in the passages. Cover what it is and its most important "
    "facts, role, and relationships. Be domain-agnostic — it may be a person, place, "
    "software component, function, concept, organization, event, product, dataset, or a "
    "fictional entity. No preamble, no speculation, no meta-commentary, no markdown — just "
    "the description, under ~120 words."
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
def run_reflection():
    """Execute a full reflection pass: synthesis, patterns, slots, enrichment,
    motifs. Plain callable since C7 — gating/retries live in the runtime."""
    db = SessionLocal()
    try:
        # Get distinct conversation IDs from the last 200 turns
        conv_ids = [row[0] for row in db.execute(text(
            "SELECT conversation_id FROM ("
            "  SELECT DISTINCT ON (conversation_id) conversation_id, timestamp "
            "  FROM episodic_memory WHERE is_private = FALSE ORDER BY conversation_id, timestamp DESC"
            ") sub ORDER BY timestamp DESC LIMIT 200"
        )).fetchall()]
        for conv_id in conv_ids:
            recent = db.query(EpisodicMemory).filter_by(
                conversation_id=conv_id, is_private=False
            ).order_by(EpisodicMemory.timestamp.desc()).limit(200).all()
            if len(recent) < 10:
                continue
            recent.reverse()
            _synthesize_session(db, recent)
            _crystallize_patterns(db, recent)
            _evolve_memory_slots(db, recent)
            _detect_motifs(db, recent)

        # Codex enrichment is global — it works on thin entities regardless of conversation
        _enrich_codex_entities(db)

        db.commit()
        logger.info("reflection_full_pass_complete")

    except Exception as exc:
        db.rollback()
        logger.error("reflection_failed", error=str(exc))
        raise
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
        temperature=0.0, max_tokens=400, timeout=bg_timeout(400)
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
    txt = "\n".join([t.raw_text[:200] for t in turns])
    if len(txt.split()) > 1500:
        txt = " ".join(txt.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model=get_bg_model_name(),
        messages=[
            {"role": "system", "content": "You are a behavioural pattern detector."},
            {"role": "user", "content": f"{CRYSTALLIZATION_PROMPT}\n\n{txt}"}
        ],
        temperature=0.0, max_tokens=200, timeout=bg_timeout(200)
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
    txt = "\n".join([t.raw_text[:200] for t in turns])
    if len(txt.split()) > 1500:
        txt = " ".join(txt.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model=get_bg_model_name(),
        messages=[
            {"role": "system", "content": "You are a memory slot analyst. Output only JSON."},
            {"role": "user", "content": f"{SLOT_EVOLUTION_PROMPT}\n\n{txt}"}
        ],
        temperature=0.0, max_tokens=300, timeout=bg_timeout(300)
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
ENRICH_LIMIT = 25            # entities enriched per reflection run (was 10)
ENRICH_REFRESH_DAYS = 14     # re-enrich an entity if its note is this old and it grew


def _enrich_codex_entities(db):
    """A7.3: fill each entity's `description` (the rich 'note body') by
    summarising the conversation passages that mention it, then reassemble
    context_payload so the note = description + properties + links + backlinks.
    Writes to `description`, NOT context_payload (which is auto-generated and
    would otherwise be overwritten on the next edge). Priority: entities with
    an EMPTY description first (the big win), then a few stale ones to refresh.
    Domain-general — works for code, research, business, people, or lore."""
    from src.workers.codex_extractor import _regenerate_context_payload

    # Priority 1: no description yet — richest-mentioned first.
    to_enrich = db.execute(text("""
        SELECT e.id
        FROM codex_entities e
        JOIN codex_events ev ON ev.entity_id = e.id
        WHERE COALESCE(e.description, '') = ''
        GROUP BY e.id
        ORDER BY COUNT(ev.id) DESC
        LIMIT :lim
    """), {"lim": ENRICH_LIMIT}).fetchall()
    ids = [r[0] for r in to_enrich]

    # Priority 2: fill remaining budget by refreshing stale, well-mentioned notes.
    if len(ids) < ENRICH_LIMIT:
        stale = db.execute(text("""
            SELECT e.id
            FROM codex_entities e
            JOIN codex_events ev ON ev.entity_id = e.id
            WHERE COALESCE(e.description, '') <> ''
              AND e.last_updated < NOW() - (:days || ' days')::interval
            GROUP BY e.id
            HAVING COUNT(ev.id) >= 3
            ORDER BY COUNT(ev.id) DESC
            LIMIT :lim
        """), {"days": ENRICH_REFRESH_DAYS, "lim": ENRICH_LIMIT - len(ids)}).fetchall()
        ids.extend(r[0] for r in stale)

    for eid in ids:
        entity = db.query(CodexEntity).get(eid)
        if not entity:
            continue
        rows = db.execute(
            text("SELECT DISTINCT batch_source FROM codex_events WHERE entity_id = :eid"),
            {"eid": entity.id}
        ).fetchall()
        passages = []
        for (bid,) in rows:
            turn = db.query(EpisodicMemory).filter_by(batch_id=bid).first()
            if turn and turn.raw_text:
                passages.append(turn.raw_text[:600])
            if len(passages) >= 8:
                break
        if not passages:
            continue
        try:
            completion = bg_client.chat.completions.create(
                model=get_bg_model_name(),
                messages=[
                    {"role": "system", "content": "You write concise factual knowledge-graph entity notes."},
                    {"role": "user", "content": f"{ENRICHMENT_PROMPT}\n\nEntity: {entity.canonical_name}\n\nPassages:\n" + "\n---\n".join(passages)[:2500]}
                ],
                temperature=0.0, max_tokens=200, timeout=bg_timeout(200)
            )
        except Exception as exc:
            logger.error("codex_enrich_failed", entity=entity.canonical_name, error=str(exc))
            continue
        note = (completion.choices[0].message.content or "").strip()
        if not note:
            continue
        entity.description = note
        _regenerate_context_payload(entity, db)   # note = description + props + links + backlinks
        entity.last_updated = datetime.now(timezone.utc)
        db.add(CodexEvent(
            entity_id=entity.id,
            event_type="context_appended",
            payload={"enriched_from_reflection": True, "chars": len(note)},
            batch_source=uuid.uuid4()
        ))
    logger.info("codex_enrichment_done", enriched=len(ids))


# ------------------------------------------------------------------
# Motif Detection
# ------------------------------------------------------------------
def _detect_motifs(db, turns):
    txt = "\n".join([t.raw_text[:200] for t in turns])
    if len(txt.split()) > 1500:
        txt = " ".join(txt.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model=get_bg_model_name(),
        messages=[
            {"role": "system", "content": "You are a thematic motif detector. Output only JSON."},
            {"role": "user", "content": f"{MOTIF_PROMPT}\n\n{txt}"}
        ],
        temperature=0.0, max_tokens=150, timeout=bg_timeout(150)
    )
    raw = completion.choices[0].message.content.strip()
    motifs = _robust_list(raw)
    for motif in motifs:
        if isinstance(motif, str) and motif.strip():
            db.execute(
                text("INSERT INTO review_queue (item_type, item_content) VALUES ('new_cluster_proposal', :payload)"),
                {"payload": json.dumps({"cluster_name": motif})}
            )