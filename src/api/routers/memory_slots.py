"""Memory Slots router – thin REST adapter over the slot service (E0).

All operation logic lives in src/services/slots.py; this file only parses,
delegates, and translates domain errors (same URLs, same responses as the
pre-E0 router).
"""
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.db import get_db
from src.api.routers.adapter import service_errors
from src.services import slots as slots_svc

router = APIRouter(prefix="/memory-slots", tags=["memory-slots"])


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


@router.get("/", response_model=List[SlotOut])
def list_slots(db: Session = Depends(get_db)):
    """Return all currently active memory slots."""
    return slots_svc.list_slots(db)


@router.get("/{slot_name}", response_model=SlotOut)
def get_slot(slot_name: str, db: Session = Depends(get_db)):
    """Return a single memory slot by name."""
    with service_errors():
        return slots_svc.get_slot(db, slot_name)


@router.put("/{slot_name}", response_model=SlotOut)
def update_slot(slot_name: str, update: SlotUpdate, db: Session = Depends(get_db)):
    """Update a slot's content (created with version 1 if missing);
    `updated_by` is always "user" for this endpoint."""
    with service_errors():
        return slots_svc.update_slot(db, slot_name, update.content, updated_by="user")


@router.post("/initialize")
def initialize_slots(db: Session = Depends(get_db)):
    """Create the seven default memory slots with empty content."""
    with service_errors():
        return slots_svc.initialize_slots(db)
