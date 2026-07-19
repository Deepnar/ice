"""Review-queue service (E0): list / approve / reject.

Approval APPLIES the item, not just flips status (D1/D2 spec D6). The two
agent arms are live:

  * ``entity_merge``         → ``codex_ops.merge_entities`` (D5's real merge —
                               the maintenance agent's Tier-2 proposals land
                               here and merge on approval)
  * ``codex_reconciliation`` → apply the supersession the in-line reconciler
                               refused to guess at: expire the old edge
                               (journaled ``edge_expired``, reason
                               "supersession")

The queue holds *agent/worker proposals only* (D8), plus the ONE deliberate
exception: ``forget_request`` (C10/C11) — /forget is fuzzy AND destructive,
so it queues here and applies only on approval (D1-tier consistency: precise
writes apply, fuzzy destructive ops queue). Besides pending/approved/
rejected, items the maintenance agent settles itself carry
``status="resolved"``, and C10's conversation deletion marks pending items
that referenced the deleted conversation's batches ``status="stale"``.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from src.memory.models import ContextCluster, ReviewQueue
from src.services.errors import NotFoundError

logger = structlog.get_logger("ice.services.review")


def list_items(db: Session, status: Optional[str] = "pending") -> list[dict]:
    items = db.query(ReviewQueue).filter_by(status=status).all()
    return [
        {"id": str(i.id), "item_type": i.item_type, "item_content": i.item_content,
         "status": i.status, "created_at": i.created_at.isoformat() if i.created_at else ""}
        for i in items
    ]


def approve(db: Session, item_id: str) -> dict:
    item = db.query(ReviewQueue).filter_by(id=uuid.UUID(item_id)).first()
    if not item:
        raise NotFoundError("Item not found")
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

    elif item.item_type == "new_cluster_proposal":
        name = item.item_content.get("cluster_name")
        if name:
            db.add(ContextCluster(name=name,
                                  created_at=datetime.now(timezone.utc)))

    elif item.item_type == "entity_merge":
        # D1/D2 arm: heavy import stays lazy (codex_ops pulls codex_extractor,
        # which loads an embedder at import).
        from src.workers.codex_ops import merge_entities
        merge_entities(
            db,
            keep_id=uuid.UUID(item.item_content["keep_id"]),
            absorb_id=uuid.UUID(item.item_content["absorb_id"]),
            agent_run_id=item.item_content.get("agent_run_id"),
        )

    elif item.item_type == "codex_reconciliation":
        from src.workers.codex_extractor import _expire_edge  # lazy: embedder at import
        _expire_edge(db, uuid.UUID(item.item_content["old_edge_id"]),
                     batch_id=uuid.uuid4(), reason="supersession")
        logger.info("codex_reconcile", type="supersession",
                    decision="expire_old_approved",
                    old_edge_id=item.item_content["old_edge_id"])

    elif item.item_type == "forget_request":
        # C10/C11 arm: /forget queues, approval applies — turn deletes +
        # journaled edge expiries live in the conversations service.
        from src.services import conversations as conversations_svc
        conversations_svc.apply_forget(db, item.item_content or {})

    db.commit()
    return {"status": "approved"}


def reject(db: Session, item_id: str) -> dict:
    """Flip to rejected (no application). D1's detectors query rejected rows
    so a rejected proposal is never re-proposed."""
    item = db.query(ReviewQueue).filter_by(id=uuid.UUID(item_id)).first()
    if not item:
        raise NotFoundError("Item not found")
    item.status = "rejected"
    db.commit()
    return {"status": "rejected"}
