"""Conversation-scoping + classification-curation service (E0).

Covers the router's C4 (conversation scope, incl. the G16 privacy re-sync)
and C3 (manual label correction — user curation of a batch's classification,
grouped here as the second "user control over what memory sees" surface).
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from src.memory.models import Conversation, CuratedLabel, EpisodicMemory
from src.services.errors import NotFoundError


def set_scope(db: Session, conv_id: str, memory_scope_type: str,
              cluster_ids: Optional[List[str]] = None,
              custom_filter: Optional[str] = None) -> dict:
    conv = db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).first()
    if not conv:
        raise NotFoundError("Conversation not found")
    was_private = conv.memory_scope_type == "none"
    conv.memory_scope_type = memory_scope_type
    conv.cluster_ids = [uuid.UUID(cid) for cid in (cluster_ids or [])]
    conv.custom_filter = custom_filter
    # G16: privacy is denormalised onto episodic rows for the retrieval-time
    # visibility invariant — keep it in sync when the scope crosses the
    # none-boundary in either direction.
    now_private = memory_scope_type == "none"
    if was_private != now_private:
        db.query(EpisodicMemory).filter_by(conversation_id=conv.id).update(
            {"is_private": now_private}
        )
    out = {"status": "ok", "conversation_id": str(conv.id),
           "memory_scope_type": conv.memory_scope_type}
    db.commit()
    return out


def get_scope(db: Session, conv_id: str) -> dict:
    conv = db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).first()
    if not conv:
        raise NotFoundError("Conversation not found")
    return {
        "conversation_id": str(conv.id),
        "memory_scope_type": conv.memory_scope_type,
        "cluster_ids": [str(cid) for cid in (conv.cluster_ids or [])],
        "custom_filter": conv.custom_filter,
    }


def override_tags(db: Session, batch_id: str, topic_labels: List[str],
                  intent_labels: List[str], context_reliance: str) -> dict:
    """C3: record a manual label correction (feeds the consent-gated
    fine-tune's curated set)."""
    entry = CuratedLabel(
        batch_id=uuid.UUID(batch_id),
        prompt="",
        corrected_topic_labels=topic_labels,
        corrected_intent_labels=intent_labels,
        corrected_context_reliance=context_reliance,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.flush()
    out = {"status": "ok", "id": str(entry.id)}   # capture before commit
    db.commit()
    return out
