"""Memory Slots router – thin REST adapter over the slot service (E0).

All operation logic lives in src/services/slots.py; this file only parses,
delegates, and translates domain errors (same URLs, same responses as the
pre-E0 router; C9 adds optional tier query params + tier fields in the
response — F4's panel and C11's commands ride the same service).
"""
from typing import List, Optional

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
    scope_tier: str = "global"
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    truncated: Optional[bool] = None       # set when the G14 cap fired
    warning: Optional[str] = None

    model_config = {"from_attributes": True}


class SlotUpdate(BaseModel):
    content: str = Field(..., min_length=0, description="New content for the slot")


@router.get("/", response_model=List[SlotOut])
def list_slots(scope_tier: Optional[str] = None,
               project_id: Optional[str] = None,
               conversation_id: Optional[str] = None,
               db: Session = Depends(get_db)):
    """All currently active memory slots (optionally one tier)."""
    return slots_svc.list_slots(db, scope_tier=scope_tier,
                                project_id=project_id,
                                conversation_id=conversation_id)


@router.get("/{slot_name}", response_model=SlotOut)
def get_slot(slot_name: str, scope_tier: str = "global",
             project_id: Optional[str] = None,
             conversation_id: Optional[str] = None,
             db: Session = Depends(get_db)):
    """Return a single memory slot by name (+ tier anchor for non-global)."""
    with service_errors():
        return slots_svc.get_slot(db, slot_name, scope_tier=scope_tier,
                                  project_id=project_id,
                                  conversation_id=conversation_id)


@router.put("/{slot_name}", response_model=SlotOut)
def update_slot(slot_name: str, update: SlotUpdate,
                scope_tier: str = "global",
                project_id: Optional[str] = None,
                conversation_id: Optional[str] = None,
                db: Session = Depends(get_db)):
    """Update a slot's content (created with version 1 if missing);
    `updated_by` is always "user" for this endpoint."""
    with service_errors():
        return slots_svc.update_slot(db, slot_name, update.content,
                                     updated_by="user", scope_tier=scope_tier,
                                     project_id=project_id,
                                     conversation_id=conversation_id)


@router.post("/initialize")
def initialize_slots(db: Session = Depends(get_db)):
    """Create the seven default global memory slots with empty content."""
    with service_errors():
        return slots_svc.initialize_slots(db)
