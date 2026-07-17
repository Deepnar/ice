"""Turn-level user-control service (E0): bookmarks, turn lookups, and
ice_remember's note branch.

Bookmarking promotes a turn to lossless + decay-immune and (C7) enqueues
codex extraction for it on the maintenance runtime — extract_codex's own
idempotency key makes a re-run a no-op.
"""
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from src.memory.models import Conversation, EpisodicMemory
from src.services.errors import NotFoundError
from src.workers.runtime import get_runtime

logger = structlog.get_logger("ice.services.bookmarks")

# ice_remember notes land in one dedicated conversation (deterministic id) —
# scope "auto" so they're globally retrievable like any non-private turn.
NOTES_CONVERSATION_ID = uuid.uuid5(uuid.NAMESPACE_URL, "ice://mcp-notes")


def _bookmark_out(turn: EpisodicMemory) -> dict:
    return {
        "id": str(turn.id),
        "timestamp": turn.timestamp.isoformat() if turn.timestamp else "",
        "raw_text": turn.raw_text[:200] + "…" if len(turn.raw_text) > 200 else turn.raw_text,
        "summary_text": turn.summary_text,
        "is_bookmarked": turn.is_bookmarked,
        "decay_immune": turn.decay_immune,
    }


def _enqueue_extraction(batch_id: str) -> None:
    runtime = get_runtime()
    if runtime is not None:
        runtime.enqueue("codex_extract", batch_id=batch_id, priority=True)
    else:
        logger.error("maintenance_core_not_started_codex_extract_skipped",
                     batch_id=batch_id)


def bookmark_turn(db: Session, turn_id: str) -> dict:
    """C1: flag a turn bookmarked + lossless + decay-immune, then make sure
    codex extraction runs for it."""
    turn = db.query(EpisodicMemory).filter_by(id=uuid.UUID(turn_id)).first()
    if not turn:
        raise NotFoundError("Turn not found")
    turn.is_bookmarked = True
    turn.lossless_flag = True
    turn.decay_immune = True
    out = _bookmark_out(turn)          # capture before commit (expiry trap)
    batch_id = str(turn.batch_id)
    db.commit()
    _enqueue_extraction(batch_id)
    return out


def list_bookmarks(db: Session, conversation_id: Optional[str] = None) -> list[dict]:
    query = db.query(EpisodicMemory).filter_by(is_bookmarked=True)
    if conversation_id:
        query = query.filter_by(conversation_id=uuid.UUID(conversation_id))
    return [_bookmark_out(t) for t in
            query.order_by(EpisodicMemory.timestamp.desc()).all()]


def latest_turn(db: Session, conversation_id: str) -> dict:
    """TUI helper: the id of a conversation's most recent turn."""
    turn = db.query(EpisodicMemory).filter_by(
        conversation_id=uuid.UUID(conversation_id)
    ).order_by(EpisodicMemory.timestamp.desc()).first()
    if not turn:
        raise NotFoundError("No turns found")
    return {"turn_id": str(turn.id)}


def remember_note(db: Session, text: str, embedder=None) -> dict:
    """ice_remember's bookmark branch (spec rev 6): store *text* as a
    bookmarked, decay-immune, lossless note in the dedicated MCP-notes
    conversation and enqueue codex extraction — the same promotion
    bookmark_turn performs on chat turns."""
    conv = db.query(Conversation).filter_by(id=NOTES_CONVERSATION_ID).first()
    if not conv:
        db.add(Conversation(id=NOTES_CONVERSATION_ID, memory_scope_type="auto"))
        db.flush()
    embedding = None
    if embedder is not None:
        vec = embedder.encode(text, convert_to_tensor=False)
        embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    now = datetime.now(timezone.utc)
    turn = EpisodicMemory(
        conversation_id=NOTES_CONVERSATION_ID,
        batch_id=uuid.uuid4(),
        timestamp=now,
        topic_tags=[],
        intent_tags=[],
        context_reliance="Zero_Shot",
        raw_text=text,
        embedding=embedding,
        lossless_flag=True,
        inject_raw=True,
        is_bookmarked=True,
        decay_immune=True,
        decay_score=1.0,
        idempotency_key=hashlib.sha256(
            f"mcp-note:{now.isoformat()}:{text}".encode()).hexdigest(),
    )
    db.add(turn)
    db.flush()
    out = {"id": str(turn.id), "conversation_id": str(NOTES_CONVERSATION_ID),
           "stored": True, "bookmarked": True}   # capture before commit
    batch_id = str(turn.batch_id)
    db.commit()
    _enqueue_extraction(batch_id)
    return out
