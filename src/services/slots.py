"""Memory-slot service (E0) — ICE's persistent working memory.

The seven fixed slot names live HERE (single constant; C9's tier rework
widens this list in one place). All adapters — the REST router, ice-mcp's
`ice_slots`/`ice_remember`, C11's chat commands — call these functions.
"""
from datetime import datetime, timezone

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.memory.models import MemorySlot
from src.services.errors import ConflictError, NotFoundError, ValidationError

logger = structlog.get_logger("ice.services.slots")

# Allowed slot names – must match the architecture (§2.1). C9 seam: widen here.
VALID_SLOTS = [
    "persona",
    "user_preferences",
    "tool_guidelines",
    "project_context",
    "guidance",
    "pending_items",
    "session_patterns",
]


def _estimate_tokens(text: str) -> int:
    """Rough token count (words * 1.33), same heuristic as the orchestrator."""
    return int(len(text.split()) * 1.33)


def _format_slot(slot: MemorySlot) -> dict:
    """Return a JSON-safe dict for a MemorySlot row."""
    return {
        "id": str(slot.id),
        "slot_name": slot.slot_name,
        "content": slot.content or "",
        "token_count": slot.token_count,
        "version": slot.version,
        "last_updated": slot.last_updated.isoformat() if slot.last_updated else "",
        "updated_by": slot.updated_by,
        "is_active": slot.is_active,
    }


def _require_valid_name(slot_name: str) -> None:
    if slot_name not in VALID_SLOTS:
        raise ValidationError(f"Invalid slot name. Must be one of {VALID_SLOTS}")


def list_slots(db: Session) -> list[dict]:
    """All currently active memory slots."""
    return [_format_slot(s) for s in
            db.query(MemorySlot).filter_by(is_active=True).all()]


def get_slot(db: Session, slot_name: str) -> dict:
    """A single memory slot by name."""
    _require_valid_name(slot_name)
    slot = db.query(MemorySlot).filter_by(slot_name=slot_name, is_active=True).first()
    if not slot:
        raise NotFoundError("Slot not found or inactive")
    return _format_slot(slot)


def update_slot(db: Session, slot_name: str, content: str,
                updated_by: str = "user") -> dict:
    """Update a slot's content (created with version 1 if it doesn't exist)."""
    _require_valid_name(slot_name)
    slot = db.query(MemorySlot).filter_by(slot_name=slot_name).first()
    if not slot:
        slot = MemorySlot(
            slot_name=slot_name,
            content=content,
            token_count=_estimate_tokens(content),
            version=1,
            last_updated=datetime.now(timezone.utc),
            updated_by=updated_by,
            is_active=True,
        )
        db.add(slot)
    else:
        slot.content = content
        slot.token_count = _estimate_tokens(content)
        slot.version += 1
        slot.last_updated = datetime.now(timezone.utc)
        slot.updated_by = updated_by
        if not slot.is_active:
            slot.is_active = True
    db.commit()
    db.refresh(slot)
    return _format_slot(slot)


def append_to_slot(db: Session, slot_name: str, text: str,
                   updated_by: str = "user") -> dict:
    """ice_remember's slot branch: newline-append through the same versioned
    update path (never a silent overwrite)."""
    _require_valid_name(slot_name)
    slot = db.query(MemorySlot).filter_by(slot_name=slot_name).first()
    existing = (slot.content or "") if slot else ""
    content = f"{existing}\n{text}" if existing else text
    return update_slot(db, slot_name, content, updated_by=updated_by)


def initialize_slots(db: Session) -> dict:
    """Create the seven default memory slots with empty content. Skips any
    slot that already exists; the unique constraint guards the concurrent-
    initialization race."""
    created = []
    for name in VALID_SLOTS:
        existing = db.query(MemorySlot).filter_by(slot_name=name).first()
        if not existing:
            db.add(MemorySlot(
                slot_name=name,
                content="",
                token_count=0,
                version=1,
                last_updated=datetime.now(timezone.utc),
                updated_by="system",
                is_active=True,
            ))
            created.append(name)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        logger.warning("initialization_race_condition_prevented", error=str(e))
        raise ConflictError("Initialization conflict. Slots may already exist.")
    return {
        "status": "ok",
        "created": created,
        "skipped": [n for n in VALID_SLOTS if n not in created],
    }
