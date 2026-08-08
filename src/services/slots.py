"""Memory-slot service (E0/C9) — ICE's persistent working memory.

The valid slot names PER TIER live HERE (single constant — the C9 tier dict;
the pre-C9 seven-name list is the 'global' tier). All adapters — the REST
router, ice-mcp's `ice_slots`/`ice_remember`, C11's chat commands, the review
queue's slot-proposal arm — call these functions.

C9 (D5–D7): three tiers (global / project / conversation). Global rows keep
NULL anchors; project rows require a project_id; conversation rows require a
conversation_id (ValidationError names the missing attachment). G14 lives
here too: content is hard-capped at settings.slot_token_cap tokens on every write
(truncated + flagged in the response — never silently).
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.api.config import settings
import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.memory.models import MemorySlot
from src.memory.tokens import count as count_tokens
from src.services.errors import ConflictError, NotFoundError, ValidationError

logger = structlog.get_logger("ice.services.slots")

# Valid slot names per tier (C9 D5). ONE constant — routers/MCP/C11 inherit.
VALID_SLOTS = {
    "global": [
        "persona",
        "user_preferences",
        "tool_guidelines",
        "project_context",
        "guidance",
        "pending_items",
        "session_patterns",
    ],
    "project": [
        "project_context",
        "conventions",
        "pending_items",
        "guidance",
    ],
    "conversation": [
        "conversation_focus",
        "pending_items",
    ],
}

# G14 (folded into C9): hard per-slot content cap.


def _cap_content(content: str) -> tuple[str, bool]:
    """Enforce the G14 cap: hard-truncate past settings.slot_token_cap tokens.

    C16: real tokens, and the truncation is a real token truncation. The old
    pair (`words * 1.33` to measure, `settings.slot_token_cap / 1.33` to cut) let a
    300-token cap admit ~550 real tokens of dense content, and every slot is
    prepended to EVERY prompt — with 13 slot names permitted, that was the
    largest silent overrun in the whole context budget.
    """
    if count_tokens(content) <= settings.slot_token_cap:
        return content, False
    words, kept, used = content.split(), [], 0
    for w in words:
        used += count_tokens(w + " ")
        if used > settings.slot_token_cap:
            break
        kept.append(w)
    return " ".join(kept), True


def _format_slot(slot: MemorySlot, truncated: bool = False) -> dict:
    """Return a JSON-safe dict for a MemorySlot row."""
    out = {
        "id": str(slot.id),
        "slot_name": slot.slot_name,
        "content": slot.content or "",
        "token_count": slot.token_count,
        "version": slot.version,
        "last_updated": slot.last_updated.isoformat() if slot.last_updated else "",
        "updated_by": slot.updated_by,
        "is_active": slot.is_active,
        "scope_tier": slot.scope_tier or "global",
        "project_id": str(slot.project_id) if slot.project_id else None,
        "conversation_id": str(slot.conversation_id) if slot.conversation_id else None,
    }
    if truncated:
        out["truncated"] = True
        out["warning"] = (f"content exceeded the {settings.slot_token_cap}-token slot "
                          "cap and was hard-truncated")
    return out


def _validate_tier(slot_name: str, scope_tier: str,
                   project_id, conversation_id) -> None:
    if scope_tier not in VALID_SLOTS:
        raise ValidationError(
            f"Invalid scope_tier {scope_tier!r}. Must be one of "
            f"{sorted(VALID_SLOTS)}")
    if slot_name not in VALID_SLOTS[scope_tier]:
        raise ValidationError(
            f"Invalid slot name for the {scope_tier} tier. Must be one of "
            f"{VALID_SLOTS[scope_tier]}")
    if scope_tier == "project" and not project_id:
        raise ValidationError(
            "a project-tier slot requires project_id — attach the "
            "conversation to a project or pass the project explicitly")
    if scope_tier == "conversation" and not conversation_id:
        raise ValidationError("a conversation-tier slot requires conversation_id")
    if scope_tier == "global" and (project_id or conversation_id):
        raise ValidationError("a global slot takes no project/conversation anchor")


def _tier_anchor(scope_tier: str, project_id, conversation_id) -> dict:
    """Filter kwargs pinning one slot row within its tier."""
    return {
        "scope_tier": scope_tier,
        "project_id": uuid.UUID(str(project_id)) if (
            scope_tier == "project" and project_id) else None,
        "conversation_id": uuid.UUID(str(conversation_id)) if (
            scope_tier == "conversation" and conversation_id) else None,
    }


def list_slots(db: Session, scope_tier: Optional[str] = None,
               project_id=None, conversation_id=None) -> list[dict]:
    """Active memory slots. No filters → every tier (adapters that only want
    one tier pass scope_tier [+ its anchor])."""
    q = db.query(MemorySlot).filter_by(is_active=True)
    if scope_tier:
        q = q.filter_by(scope_tier=scope_tier)
        if scope_tier == "project" and project_id:
            q = q.filter_by(project_id=uuid.UUID(str(project_id)))
        if scope_tier == "conversation" and conversation_id:
            q = q.filter_by(conversation_id=uuid.UUID(str(conversation_id)))
    return [_format_slot(s) for s in q.all()]


def get_slot(db: Session, slot_name: str, scope_tier: str = "global",
             project_id=None, conversation_id=None) -> dict:
    """A single memory slot by name (+ tier anchor for non-global tiers)."""
    _validate_tier(slot_name, scope_tier, project_id, conversation_id)
    slot = db.query(MemorySlot).filter_by(
        slot_name=slot_name, is_active=True,
        **_tier_anchor(scope_tier, project_id, conversation_id)).first()
    if not slot:
        raise NotFoundError("Slot not found or inactive")
    return _format_slot(slot)


def update_slot(db: Session, slot_name: str, content: str,
                updated_by: str = "user", scope_tier: str = "global",
                project_id=None, conversation_id=None) -> dict:
    """Update a slot's content (created with version 1 if it doesn't exist).
    Content is G14-capped; the response carries `truncated` when it was."""
    _validate_tier(slot_name, scope_tier, project_id, conversation_id)
    content, truncated = _cap_content(content)
    if truncated:
        logger.warning("slot_content_truncated", slot=slot_name,
                       tier=scope_tier, cap=settings.slot_token_cap)
    anchor = _tier_anchor(scope_tier, project_id, conversation_id)
    slot = db.query(MemorySlot).filter_by(slot_name=slot_name, **anchor).first()
    if not slot:
        slot = MemorySlot(
            slot_name=slot_name,
            content=content,
            token_count=count_tokens(content),
            version=1,
            last_updated=datetime.now(timezone.utc),
            updated_by=updated_by,
            is_active=True,
            **anchor,
        )
        db.add(slot)
    else:
        slot.content = content
        slot.token_count = count_tokens(content)
        slot.version += 1
        slot.last_updated = datetime.now(timezone.utc)
        slot.updated_by = updated_by
        if not slot.is_active:
            slot.is_active = True
    db.commit()
    db.refresh(slot)
    return _format_slot(slot, truncated=truncated)


def append_to_slot(db: Session, slot_name: str, text: str,
                   updated_by: str = "user", scope_tier: str = "global",
                   project_id=None, conversation_id=None) -> dict:
    """ice_remember's slot branch: newline-append through the same versioned
    update path (never a silent overwrite)."""
    _validate_tier(slot_name, scope_tier, project_id, conversation_id)
    slot = db.query(MemorySlot).filter_by(
        slot_name=slot_name,
        **_tier_anchor(scope_tier, project_id, conversation_id)).first()
    existing = (slot.content or "") if slot else ""
    content = f"{existing}\n{text}" if existing else text
    return update_slot(db, slot_name, content, updated_by=updated_by,
                       scope_tier=scope_tier, project_id=project_id,
                       conversation_id=conversation_id)


def initialize_slots(db: Session) -> dict:
    """Create the seven default GLOBAL memory slots with empty content
    (project/conversation slots are created on first write). Skips any slot
    that already exists; the unique index guards the concurrent-
    initialization race."""
    created = []
    for name in VALID_SLOTS["global"]:
        existing = db.query(MemorySlot).filter_by(
            slot_name=name, scope_tier="global",
            project_id=None, conversation_id=None).first()
        if not existing:
            db.add(MemorySlot(
                slot_name=name,
                content="",
                token_count=0,
                version=1,
                last_updated=datetime.now(timezone.utc),
                updated_by="system",
                is_active=True,
                scope_tier="global",
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
        "skipped": [n for n in VALID_SLOTS["global"] if n not in created],
    }
