"""Reflection Worker – produces higher‑order knowledge from accumulated episodic content."""

import structlog
import json
import re
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from openai import OpenAI

from src.api.db import SessionLocal
from src.memory.models import (
    EpisodicMemory, SessionSummary, MemorySlot, CodexEntity, CodexEvent
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.reflection")
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

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


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_reflection(self, conversation_id: str = None):
    """Execute a reflection pass. If conversation_id is given, reflect on that session."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # If a specific conversation is requested
        if conversation_id:
            turns = db.query(EpisodicMemory).filter_by(
                conversation_id=conversation_id
            ).order_by(EpisodicMemory.timestamp.asc()).all()
            if turns:
                _synthesize_session(db, turns, conversation_id)
            return

        # Default: process most recent 50 turns as a fake session
        recent_turns = db.query(EpisodicMemory).order_by(
            EpisodicMemory.timestamp.desc()
        ).limit(20).all()
        if recent_turns:
            recent_turns.reverse()  # chronological order for the model
            _synthesize_session(db, recent_turns, None)

    except Exception as exc:
        db.rollback()
        logger.error("reflection_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


def _synthesize_session(db, turns, conversation_id):
    """Create a session summary from a list of turns."""
    # Build the text, truncating to last 3000 words to avoid context overflow
    full_text = "\n\n".join([t.raw_text for t in turns])
    words = full_text.split()
    if len(words) > 3000:
        full_text = " ".join(words[-3000:])

    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a session analysis engine. Output only JSON."},
            {"role": "user", "content": f"{SUMMARY_PROMPT}\n\n{full_text}"}
        ],
        temperature=0.0,
        max_tokens=400,
        timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()

    # Robust JSON extraction
    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(json_match.group(0)) if json_match else {}
    except Exception:
        data = {}

    summary = SessionSummary(
        conversation_id=conversation_id,
        topics_covered=data.get("topics_covered", []),
        decisions_made=data.get("decisions_made", ""),
        unresolved_items=data.get("unresolved_items", ""),
        entities_updated=data.get("entities_updated", []),      # ← must be a plain list of strings
        patterns_observed=data.get("patterns_observed", [])     # ← same here
    )
    db.add(summary)

    # Optionally update pending_items slot if unresolved items were found
    unresolved = data.get("unresolved_items")
    if unresolved and isinstance(unresolved, str):
        slot = db.query(MemorySlot).filter_by(slot_name="pending_items").first()
        if slot:
            existing = slot.content or ""
            slot.content = existing + "\n" + unresolved if existing else unresolved
            slot.version += 1
            slot.last_updated = datetime.now(timezone.utc)
            slot.updated_by = "reflection_worker"

    db.commit()
    logger.info("session_synthesized", conversation_id=conversation_id)