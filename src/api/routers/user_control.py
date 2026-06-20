"""User‑guided memory control endpoints (Phase C + model registry)."""

import uuid, json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.db import get_db
from src.memory.models import (
    EpisodicMemory, CodexEntity, CodexEdge, CodexEvent,
    CuratedLabel, Conversation, ContextCluster, MemorySlot, ReviewQueue
)
from src.workers.codex_extractor import extract_codex
from src.model_registry.registry import load_registry, save_registry, populate_from_ollama

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
    memory_scope_type: str
    cluster_ids: Optional[List[str]] = None
    custom_filter: Optional[str] = None

class ClusterCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class ClusterAssign(BaseModel):
    turn_ids: List[str]

class ReviewApprove(BaseModel):
    slot_name: Optional[str] = None
    cluster_name: Optional[str] = None


# ------------------------------------------------------------------
# C1 — Bookmarking
# ------------------------------------------------------------------
@router.post("/turns/{turn_id}/bookmark", response_model=BookmarkOut)
def bookmark_turn(turn_id: str, db: Session = Depends(get_db)):
    turn = db.query(EpisodicMemory).filter_by(id=uuid.UUID(turn_id)).first()
    if not turn:
        raise HTTPException(status_code=404, detail="Turn not found")
    turn.is_bookmarked = True
    turn.lossless_flag = True
    turn.decay_immune = True
    db.commit()
    extract_codex.apply_async(kwargs={"batch_id": str(turn.batch_id), "priority": True})
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
    entry = CuratedLabel(
        batch_id=uuid.UUID(override.batch_id),
        prompt="",
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
    conv = db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.memory_scope_type = scope.memory_scope_type
    conv.cluster_ids = [uuid.UUID(cid) for cid in (scope.cluster_ids or [])]
    conv.custom_filter = scope.custom_filter
    db.commit()
    return {"status": "ok", "conversation_id": str(conv.id), "memory_scope_type": conv.memory_scope_type}

@router.get("/conversations/{conv_id}/scope")
def get_conversation_scope(conv_id: str, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": str(conv.id),
        "memory_scope_type": conv.memory_scope_type,
        "cluster_ids": [str(cid) for cid in (conv.cluster_ids or [])],
        "custom_filter": conv.custom_filter,
    }


# ------------------------------------------------------------------
# C5 — Explicit cluster creation & assignment
# ------------------------------------------------------------------
@router.post("/clusters", response_model=dict)
def create_cluster(body: ClusterCreate, db: Session = Depends(get_db)):
    cluster = ContextCluster(name=body.name, description=body.description,
                             created_at=datetime.now(timezone.utc))
    db.add(cluster)
    db.commit()
    db.refresh(cluster)
    return {"id": str(cluster.id), "name": cluster.name}

@router.put("/clusters/{cluster_id}/assign")
def assign_turns_to_cluster(cluster_id: str, body: ClusterAssign, db: Session = Depends(get_db)):
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
# C6 — Review queue
# ------------------------------------------------------------------
@router.get("/review-queue", response_model=List[dict])
def get_review_queue(status: Optional[str] = "pending", db: Session = Depends(get_db)):
    items = db.query(ReviewQueue).filter_by(status=status).all()
    return [
        {"id": str(i.id), "item_type": i.item_type, "item_content": i.item_content,
         "status": i.status, "created_at": i.created_at.isoformat() if i.created_at else ""}
        for i in items
    ]

@router.post("/review-queue/{item_id}/approve")
def approve_review_item(item_id: str, body: ReviewApprove = None, db: Session = Depends(get_db)):
    item = db.query(ReviewQueue).filter_by(id=uuid.UUID(item_id)).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    item.status = "approved"

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
        pass

    db.commit()
    return {"status": "approved"}


# ------------------------------------------------------------------
# Extra endpoints for TUI
# ------------------------------------------------------------------
@router.get("/conversations/{conv_id}/latest-turn")
def get_latest_turn(conv_id: str, db: Session = Depends(get_db)):
    turn = db.query(EpisodicMemory).filter_by(
        conversation_id=uuid.UUID(conv_id)
    ).order_by(EpisodicMemory.timestamp.desc()).first()
    if not turn:
        raise HTTPException(status_code=404, detail="No turns found")
    return {"turn_id": str(turn.id)}


# ------------------------------------------------------------------
# Model Registry endpoints
# ------------------------------------------------------------------
@router.get("/model-registry")
def get_model_registry():
    return load_registry()

@router.post("/model-registry/refresh")
def refresh_model_registry():
    return populate_from_ollama()

@router.put("/model-registry/{model_name}")
def update_model_entry(model_name: str, data: dict):
    reg = load_registry()
    if model_name not in reg["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    entry = reg["models"][model_name]
    if "topic_tags" in data:
        entry["topic_tags"] = data["topic_tags"]
    if "intent_tags" in data:
        entry["intent_tags"] = data["intent_tags"]
    if "confirmed" in data:
        entry["confirmed"] = data["confirmed"]
    if "base_url" in data:
        entry["base_url"] = data["base_url"]
    save_registry(reg)
    return {"status": "updated"}

@router.delete("/model-registry/{model_name}")
def delete_model(model_name: str):
    reg = load_registry()
    if model_name in reg["models"]:
        del reg["models"][model_name]
        save_registry(reg)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Model not found")