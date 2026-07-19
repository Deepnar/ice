"""User-guided memory control endpoints (Phase C + model registry) — thin
REST adapter over the E0 services (bookmarks / scoping / clusters / review /
registry_svc). Parse → service → format; all operation logic lives in
src/services/ (ice-mcp and C11's chat commands are adapters two and three
over the same functions)."""

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.db import get_db
from src.api.routers.adapter import service_errors
from src.services import bookmarks as bookmarks_svc
from src.services import clusters as clusters_svc
from src.services import conversations as conversations_svc
from src.services import registry_svc
from src.services import review as review_svc
from src.services import scoping as scoping_svc

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
    # E1 (D11): slug/name/id attaches to a project; "" detaches; None = leave.
    project: Optional[str] = None


class CommitNotice(BaseModel):
    commit: Optional[str] = None

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
    with service_errors():
        return bookmarks_svc.bookmark_turn(db, turn_id)

@router.get("/bookmarks", response_model=List[BookmarkOut])
def list_bookmarks(conversation_id: Optional[str] = None, db: Session = Depends(get_db)):
    return bookmarks_svc.list_bookmarks(db, conversation_id)


# ------------------------------------------------------------------
# C3 — Manual label correction
# ------------------------------------------------------------------
@router.post("/batch/override-tags")
def override_tags(override: LabelOverride, db: Session = Depends(get_db)):
    with service_errors():
        return scoping_svc.override_tags(
            db, override.batch_id, override.topic_labels,
            override.intent_labels, override.context_reliance)


# ------------------------------------------------------------------
# C10 — Conversation deletion (manifest-first; ?dry_run=true previews)
# ------------------------------------------------------------------
@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, dry_run: bool = False,
                        db: Session = Depends(get_db)):
    with service_errors():
        return conversations_svc.delete_conversation(db, conv_id,
                                                     dry_run=dry_run)


# ------------------------------------------------------------------
# C4 — Conversation scoping
# ------------------------------------------------------------------
@router.put("/conversations/{conv_id}/scope")
def set_conversation_scope(conv_id: str, scope: ScopeUpdate, db: Session = Depends(get_db)):
    with service_errors():
        return scoping_svc.set_scope(db, conv_id, scope.memory_scope_type,
                                     scope.cluster_ids, scope.custom_filter,
                                     project=scope.project)


# ------------------------------------------------------------------
# E3 — commit notification (the post-commit hook's target; spec rev 4)
# ------------------------------------------------------------------
@router.post("/projects/{project_ref}/commit")
def notify_project_commit(project_ref: str, body: CommitNotice = None,
                          db: Session = Depends(get_db)):
    with service_errors():
        from src.services import projects as projects_svc
        return projects_svc.notify_commit(
            db, project_ref, body.commit if body else None)

@router.get("/conversations/{conv_id}/scope")
def get_conversation_scope(conv_id: str, db: Session = Depends(get_db)):
    with service_errors():
        return scoping_svc.get_scope(db, conv_id)


# ------------------------------------------------------------------
# C5 — Explicit cluster creation & assignment
# ------------------------------------------------------------------
@router.post("/clusters", response_model=dict)
def create_cluster(body: ClusterCreate, db: Session = Depends(get_db)):
    return clusters_svc.create_cluster(db, body.name, body.description)

@router.put("/clusters/{cluster_id}/assign")
def assign_turns_to_cluster(cluster_id: str, body: ClusterAssign, db: Session = Depends(get_db)):
    with service_errors():
        return clusters_svc.assign_turns(db, cluster_id, body.turn_ids)


# ------------------------------------------------------------------
# C6 — Review queue
# ------------------------------------------------------------------
@router.get("/review-queue", response_model=List[dict])
def get_review_queue(status: Optional[str] = "pending", db: Session = Depends(get_db)):
    return review_svc.list_items(db, status)

@router.post("/review-queue/{item_id}/approve")
def approve_review_item(item_id: str, body: ReviewApprove = None, db: Session = Depends(get_db)):
    with service_errors():
        return review_svc.approve(db, item_id)


# ------------------------------------------------------------------
# Extra endpoints for TUI
# ------------------------------------------------------------------
@router.get("/conversations/{conv_id}/latest-turn")
def get_latest_turn(conv_id: str, db: Session = Depends(get_db)):
    with service_errors():
        return bookmarks_svc.latest_turn(db, conv_id)


# ------------------------------------------------------------------
# Model Registry endpoints
# ------------------------------------------------------------------
@router.get("/model-registry")
def get_model_registry():
    return registry_svc.get_registry()

@router.post("/model-registry/refresh")
def refresh_model_registry():
    return registry_svc.refresh_registry()

@router.put("/model-registry/{model_name}")
def update_model_entry(model_name: str, data: dict):
    with service_errors():
        return registry_svc.update_model(model_name, data)

@router.delete("/model-registry/{model_name}")
def delete_model(model_name: str):
    with service_errors():
        return registry_svc.delete_model(model_name)
