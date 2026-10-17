"""Review-queue service (E0): list / approve / reject / undo.

G77 (2026-10-17): background decisions apply themselves and the queue is
their audit log — a self-applied decision is written ``status="applied"``
with enough payload to reverse it, and ``undo`` reverses it (slot text,
decision supersession, entity merge). Still gated on approval:
``forget_request`` and ``finetune_proposal``.

Approval APPLIES the item, not just flips status (D1/D2 spec D6):

  * ``entity_merge``         → ``codex_ops.merge_entities`` (legacy pending
                               proposals; since CODEX_V4 D12 the agent applies
                               model-confirmed merges itself and writes an
                               ``applied`` audit row)

The queue holds *agent/worker proposals only* (D8), plus the ONE deliberate
exception: ``forget_request`` (C10/C11) — /forget is fuzzy AND destructive,
so it queues here and applies only on approval (D1-tier consistency: precise
writes apply, fuzzy destructive ops queue). Besides pending/approved/
rejected, items the maintenance agent settles itself carry
``status="resolved"``, and C10's conversation deletion marks pending items
that referenced the deleted conversation's batches ``status="stale"``.
Undone items carry ``status="undone"``.
"""
import uuid
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from src.memory.models import ReviewQueue
from src.services.errors import ConflictError, NotFoundError, ValidationError

logger = structlog.get_logger("ice.services.review")


def list_items(db: Session, status: Optional[str] = "pending") -> list[dict]:
    items = db.query(ReviewQueue).filter_by(status=status).all()
    return [
        {"id": str(i.id), "item_type": i.item_type, "item_content": i.item_content,
         "status": i.status, "created_at": i.created_at.isoformat() if i.created_at else ""}
        for i in items
    ]


def approve(db: Session, item_id: str, *, keep_edge_ids: Optional[list[str]] = None) -> dict:
    # keep_edge_ids is accepted for API compatibility; the edge-conflict arm it
    # served was retired with the legacy codex (CODEX_V4).
    item = db.query(ReviewQueue).filter_by(id=uuid.UUID(item_id)).first()
    if not item:
        raise NotFoundError("Item not found")
    if item.item_type in ("codex_contradiction", "codex_reconciliation"):
        # CODEX_V4: the legacy edge-conflict queue is retired (facts decide
        # same / replaces / both_true at write time; old rows are archived).
        raise ValidationError("Legacy graph conflict items are archived and cannot be applied.")
    if item.item_type == "new_cluster_proposal":
        # G77 (maintainer, 2026-10-17): motif proposals are retired — approving
        # one created an EMPTY named cluster that nothing ever filled; clusters
        # are named from their members (cluster_labels).
        raise ValidationError("Motif cluster proposals are retired and cannot be applied.")
    item.status = "approved"

    if item.item_type == "memory_slot_update":
        # C9 (D7): apply through the slots service — tier validation + the
        # G14 cap live there; updated_by records the PROPOSER (the human
        # approving is the gate, not the author).
        from src.services import slots as slots_svc
        slot_name = item.item_content.get("slot_name")
        content = item.item_content.get("proposed_content")
        if slot_name and content:
            slots_svc.update_slot(
                db, slot_name, content,
                updated_by=item.item_content.get("proposed_by", "agent"),
                scope_tier=item.item_content.get("scope_tier", "global"),
                project_id=item.item_content.get("project_id"),
                conversation_id=item.item_content.get("conversation_id"))

    elif item.item_type == "entity_merge":
        # D1/D2 arm: the import stays lazy, like the other dispatch arms.
        from src.workers.codex_ops import merge_entities
        merge_entities(
            db,
            keep_id=uuid.UUID(item.item_content["keep_id"]),
            absorb_id=uuid.UUID(item.item_content["absorb_id"]),
            agent_run_id=item.item_content.get("agent_run_id"),
        )

    elif item.item_type == "forget_request":
        # C10/C11 arm: /forget queues, approval applies — turn deletes +
        # journaled edge expiries live in the conversations service.
        from src.services import conversations as conversations_svc
        conversations_svc.apply_forget(db, item.item_content or {})

    db.commit()
    return {"status": "approved"}


def undo(db: Session, item_id: str) -> dict:
    """G77: reverse one background decision that applied itself. Only an
    `applied` row is undoable; the row becomes `undone`, so a second call is
    refused instead of reversing twice."""
    item = db.query(ReviewQueue).filter_by(id=uuid.UUID(item_id)).first()
    if not item:
        raise NotFoundError("Item not found")
    if item.status != "applied":
        raise ValidationError(f"Only an applied item can be undone (this one is {item.status!r}).")
    content = item.item_content or {}
    if item.item_type == "memory_slot_update":
        from src.services import slots as slots_svc
        result = {"slot": slots_svc.undo_journaled(db, content)}
    elif item.item_type == "decision_supersession":
        from sqlalchemy import text
        restored = db.execute(text("""
            UPDATE decisions SET valid_until = NULL, superseded_by = NULL
            WHERE id = :old AND superseded_by = :new"""),
            {"old": content.get("old_id"), "new": content.get("new_id")}).rowcount
        if not restored:
            raise ConflictError("the old decision is no longer superseded by this one; nothing to undo")
        result = {"decision": content.get("old_id"), "current": True}
    elif item.item_type == "entity_merge" and content.get("merge_event_id"):
        from src.workers.codex_ops import unmerge_entities
        result = unmerge_entities(db, content["merge_event_id"])
        if result.get("status") == "skipped":
            raise ConflictError(f"merge cannot be undone: {result.get('reason')}")
    else:
        raise ValidationError(f"{item.item_type!r} changed nothing, so there is nothing to undo.")
    item.status = "undone"
    db.commit()
    logger.info("review_item_undone", item_type=item.item_type, item_id=item_id)
    return {"status": "undone", **result}


def reject(db: Session, item_id: str) -> dict:
    """Flip to rejected (no application). D1's detectors query rejected rows
    so a rejected proposal is never re-proposed."""
    item = db.query(ReviewQueue).filter_by(id=uuid.UUID(item_id)).first()
    if not item:
        raise NotFoundError("Item not found")
    item.status = "rejected"
    db.commit()
    return {"status": "rejected"}
