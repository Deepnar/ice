"""User‑guided memory control endpoints (Phase C)."""

import uuid, json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.model_registry.registry import load_registry, populate_from_ollama
from src.api.db import get_db
from src.memory.models import (
    EpisodicMemory, CodexEntity, CodexEdge, CodexEvent,
    CuratedLabel, Conversation, ContextCluster, MemorySlot, ReviewQueue
)
from src.workers.codex_extractor import extract_codex

router = APIRouter(prefix="/user-control", tags=["user-control"])

# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------
class BookmarkOut(BaseModel):
    id: str
    timestamp: str
    raw_text: str
    summary_text: Optional[str] = None
    is_bookmarked: bool
    decay_immune: bool

    class Config:
        from_attributes = True

class LabelOverride(BaseModel):
    batch_id: str
    topic_labels: List[str] = []
    intent_labels: List[str] = []
    context_reliance: str

class ScopeUpdate(BaseModel):
    memory_scope_type: str   # none, auto, project, manual
    cluster_ids: Optional[List[str]] = None
    custom_filter: Optional[str] = None

class ClusterCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class ClusterAssign(BaseModel):
    turn_ids: List[str]

class ReviewApprove(BaseModel):
    slot_name: Optional[str] = None   # for memory_slot_update
    cluster_name: Optional[str] = None  # for new_cluster_proposal

@router.get("/model-registry")
def get_model_registry():
    return load_registry()

@router.post("/model-registry/refresh")
def refresh_model_registry():
    return populate_from_ollama()

# ------------------------------------------------------------------
# C1 — Bookmarking
# ------------------------------------------------------------------
@router.post("/turns/{turn_id}/bookmark", response_model=BookmarkOut)
def bookmark_turn(turn_id: str, db: Session = Depends(get_db)):
    """Mark a turn as bookmarked, force lossless, decay‑immune, and re‑extract Codex."""
    turn = db.query(EpisodicMemory).filter_by(id=uuid.UUID(turn_id)).first()
    if not turn:
        raise HTTPException(status_code=404, detail="Turn not found")

    turn.is_bookmarked = True
    turn.lossless_flag = True
    turn.decay_immune = True
    db.commit()

    # Trigger priority Codex extraction for this turn (bypasses GPU check via immediate task)
    extract_codex.delay(batch_id=str(turn.batch_id))

    return BookmarkOut(
        id=str(turn.id),
        timestamp=turn.timestamp.isoformat() if turn.timestamp else "",
        raw_text=turn.raw_text[:200] + "…" if len(turn.raw_text) > 200 else turn.raw_text,
        summary_text=turn.summary_text,
        is_bookmarked=turn.is_bookmarked,
        decay_immune=turn.decay_immune,
    )


@router.get("/bookmarks", response_model=List[BookmarkOut])
def list_bookmarks(conversation_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Return all bookmarked turns, optionally filtered by conversation."""
    query = db.query(EpisodicMemory).filter_by(is_bookmarked=True)
    if conversation_id:
        query = query.filter_by(conversation_id=uuid.UUID(conversation_id))
    turns = query.order_by(EpisodicMemory.timestamp.desc()).all()
    return [
        BookmarkOut(
            id=str(t.id),
            timestamp=t.timestamp.isoformat() if t.timestamp else "",
            raw_text=t.raw_text[:200] + "…" if len(t.raw_text) > 200 else t.raw_text,
            summary_text=t.summary_text,
            is_bookmarked=t.is_bookmarked,
            decay_immune=t.decay_immune,
        )
        for t in turns
    ]


# ------------------------------------------------------------------
# C3 — Manual label correction
# ------------------------------------------------------------------
@router.post("/batch/override-tags")
def override_tags(override: LabelOverride, db: Session = Depends(get_db)):
    """Record a user‑corrected classification for a batch."""
    entry = CuratedLabel(
        batch_id=uuid.UUID(override.batch_id),
        prompt="",  # we can later fill from episodic_memory, but not required for fine‑tuning
        corrected_topic_labels=override.topic_labels,
        corrected_intent_labels=override.intent_labels,
        corrected_context_reliance=override.context_reliance,
        created_at=datetime.now(timezone.utc)
    )
    db.add(entry)
    db.commit()
    return {"status": "ok", "id": str(entry.id)}


# ------------------------------------------------------------------
# C4 — Conversation scoping
# ------------------------------------------------------------------
@router.put("/conversations/{conv_id}/scope")
def set_conversation_scope(conv_id: str, scope: ScopeUpdate, db: Session = Depends(get_db)):
    """Set the memory scope for a conversation."""
    conv = db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.memory_scope_type = scope.memory_scope_type
    conv.cluster_ids = [uuid.UUID(cid) for cid in (scope.cluster_ids or [])]
    conv.custom_filter = scope.custom_filter
    db.commit()
    return {"status": "ok", "conversation_id": str(conv.id), "memory_scope_type": conv.memory_scope_type}


# ------------------------------------------------------------------
# C5 — Explicit cluster creation & assignment
# ------------------------------------------------------------------
@router.post("/clusters", response_model=dict)
def create_cluster(body: ClusterCreate, db: Session = Depends(get_db)):
    """Manually create a named cluster."""
    cluster = ContextCluster(name=body.name, description=body.description,
                             created_at=datetime.now(timezone.utc))
    db.add(cluster)
    db.commit()
    db.refresh(cluster)
    return {"id": str(cluster.id), "name": cluster.name}

@router.put("/clusters/{cluster_id}/assign")
def assign_turns_to_cluster(cluster_id: str, body: ClusterAssign, db: Session = Depends(get_db)):
    """Assign specific turns to a cluster."""
    cluster = db.query(ContextCluster).filter_by(id=uuid.UUID(cluster_id)).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    for tid in body.turn_ids:
        turn = db.query(EpisodicMemory).filter_by(id=uuid.UUID(tid)).first()
        if turn:
            turn.cluster_id = cluster.id
    db.commit()
    return {"assigned": len(body.turn_ids)}


# ------------------------------------------------------------------
# C6 — Review queue (memory slot update confirmation)
# ------------------------------------------------------------------
@router.get("/review-queue", response_model=List[dict])
def get_review_queue(status: Optional[str] = "pending", db: Session = Depends(get_db)):
    """List review items, default pending."""
    items = db.query(ReviewQueue).filter_by(status=status).all()
    return [
        {"id": str(i.id), "item_type": i.item_type, "item_content": i.item_content,
         "status": i.status, "created_at": i.created_at.isoformat() if i.created_at else ""}
        for i in items
    ]

@router.post("/review-queue/{item_id}/approve")
def approve_review_item(item_id: str, body: ReviewApprove = None, db: Session = Depends(get_db)):
    """Approve a review item and execute its action."""
    item = db.query(ReviewQueue).filter_by(id=uuid.UUID(item_id)).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    item.status = "approved"

    # Execute action based on item_type
    if item.item_type == "memory_slot_update":
        slot_name = item.item_content.get("slot_name")
        content = item.item_content.get("proposed_content")
        if slot_name and content:
            slot = db.query(MemorySlot).filter_by(slot_name=slot_name).first()
            if slot:
                slot.content = content
                slot.version += 1
                slot.last_updated = datetime.now(timezone.utc)
                slot.updated_by = "user"

    elif item.item_type == "new_cluster_proposal":
        name = item.item_content.get("cluster_name")
        if name:
            cluster = ContextCluster(name=name, created_at=datetime.now(timezone.utc))
            db.add(cluster)

    elif item.item_type == "sentinel_review":
        # Sentinel review items are informational; just mark approved
        pass

    db.commit()
    return {"status": "approved"}