"""
Memory Slots router – CRUD endpoints for ICE's persistent working memory.
"""

import logging
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.api.db import get_db
from src.memory.models import MemorySlot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed slot names – must match the architecture (§2.1)
# ---------------------------------------------------------------------------
VALID_SLOTS = [
    "persona",
    "user_preferences",
    "tool_guidelines",
    "project_context",
    "guidance",
    "pending_items",
    "session_patterns",
]

router = APIRouter(prefix="/memory-slots", tags=["memory-slots"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class SlotOut(BaseModel):
    id: str
    slot_name: str
    content: str
    token_count: int
    version: int
    last_updated: str
    updated_by: str
    is_active: bool

    model_config = {"from_attributes": True}


class SlotUpdate(BaseModel):
    content: str = Field(..., min_length=0, description="New content for the slot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Rough token count (words * 1.33), same heuristic as the orchestrator."""
    return int(len(text.split()) * 1.33)


def _format_slot(slot: MemorySlot) -> dict:
    """Return a JSON‑safe dict for a MemorySlot row."""
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/", response_model=List[SlotOut])
def list_slots(db: Session = Depends(get_db)):
    """Return all currently active memory slots."""
    slots = db.query(MemorySlot).filter_by(is_active=True).all()
    return [_format_slot(s) for s in slots]


@router.get("/{slot_name}", response_model=SlotOut)
def get_slot(slot_name: str, db: Session = Depends(get_db)):
    """Return a single memory slot by name."""
    if slot_name not in VALID_SLOTS:
        raise HTTPException(status_code=400, detail=f"Invalid slot name. Must be one of {VALID_SLOTS}")
    slot = db.query(MemorySlot).filter_by(slot_name=slot_name, is_active=True).first()
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found or inactive")
    return _format_slot(slot)


@router.put("/{slot_name}", response_model=SlotOut)
def update_slot(slot_name: str, update: SlotUpdate, db: Session = Depends(get_db)):
    """
    Update the content of a memory slot.  If the slot does not exist, it is created
    (with version 1).  The `updated_by` field is always set to "user" for this endpoint.
    """
    if slot_name not in VALID_SLOTS:
        raise HTTPException(status_code=400, detail=f"Invalid slot name. Must be one of {VALID_SLOTS}")

    slot = db.query(MemorySlot).filter_by(slot_name=slot_name).first()

    if not slot:
        # Create a new slot – the model's default=uuid.uuid4 handles the ID automatically
        slot = MemorySlot(
            slot_name=slot_name,
            content=update.content,
            token_count=_estimate_tokens(update.content),
            version=1,
            last_updated=datetime.now(timezone.utc),
            updated_by="user",
            is_active=True,
        )
        db.add(slot)
    else:
        slot.content = update.content
        slot.token_count = _estimate_tokens(update.content)
        slot.version += 1
        slot.last_updated = datetime.now(timezone.utc)
        slot.updated_by = "user"
        if not slot.is_active:
            slot.is_active = True

    db.commit()
    db.refresh(slot)
    return _format_slot(slot)


@router.post("/initialize")
def initialize_slots(db: Session = Depends(get_db)):
    """
    Create the seven default memory slots with empty content.
    Skips any slot that already exists. Protected against concurrent initialization.
    """
    created = []
    for name in VALID_SLOTS:
        existing = db.query(MemorySlot).filter_by(slot_name=name).first()
        if not existing:
            # No manual 'id' – the model's default generates the UUID
            slot = MemorySlot(
                slot_name=name,
                content="",
                token_count=0,
                version=1,
                last_updated=datetime.now(timezone.utc),
                updated_by="system",
                is_active=True,
            )
            db.add(slot)
            created.append(name)

    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        logger.warning("initialization_race_condition_prevented", error=str(e))
        raise HTTPException(
            status_code=409,
            detail="Initialization conflict. Slots may already exist.",
        )

    return {
        "status": "ok",
        "created": created,
        "skipped": [n for n in VALID_SLOTS if n not in created],
    }