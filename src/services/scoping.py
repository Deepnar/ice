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
from src.services.errors import NotFoundError, ValidationError


#: C6: the scope vocabulary is CLOSED. `set_scope` used to accept any string
#: and store it verbatim, and main.py only branched on 'none'/'project' — so a
#: typo'd or invented mode was silently retrieved as 'auto'. Rejecting is the
#: only honest option: a scope that quietly means something else is a privacy
#: surface, not a convenience.
VALID_SCOPE_TYPES = ("none", "auto", "project", "manual")


def _uuid_list(values: Optional[List[str]]) -> Optional[List[uuid.UUID]]:
    """None = leave the column unchanged; [] = clear it. Every id-set
    parameter below shares these semantics (`cluster_ids` included, as of C6 —
    it used to wipe on None, which is why /scope had to re-send the current
    value on every call just to avoid destroying it)."""
    if values is None:
        return None
    return [v if isinstance(v, uuid.UUID) else uuid.UUID(str(v)) for v in values]


def set_scope(db: Session, conv_id: str, memory_scope_type: str,
              cluster_ids: Optional[List[str]] = None,
              project: Optional[str] = None,
              included_conversation_ids: Optional[List[str]] = None,
              excluded_conversation_ids: Optional[List[str]] = None,
              excluded_cluster_ids: Optional[List[str]] = None) -> dict:
    """*project* (E1 D11): slug/name/id attaches the conversation to a
    project (coding scope); empty string detaches; None leaves it unchanged.
    Incognito ('none') conversations refuse attachment — private turns must
    never feed a project's shared context.

    C6 id sets — all `None` = unchanged, `[]` = clear:
      *included_conversation_ids* — the cross-chat set a 'manual' conversation
        reads besides itself (ignored in other modes; kept, not wiped, so
        flipping auto→manual→auto doesn't destroy the pick).
      *excluded_conversation_ids* / *excluded_cluster_ids* — never retrieve
        these, in EVERY mode. The middle ground between adding to scope and
        deleting: keep the memory, stop reading it.
    """
    if memory_scope_type not in VALID_SCOPE_TYPES:
        raise ValidationError(
            f"unknown memory_scope_type {memory_scope_type!r} — "
            f"valid: {', '.join(VALID_SCOPE_TYPES)}")
    conv = db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).first()
    if not conv:
        raise NotFoundError("Conversation not found")
    was_private = conv.memory_scope_type == "none"
    conv.memory_scope_type = memory_scope_type
    for attr, value in (
        ("cluster_ids", cluster_ids),
        ("included_conversation_ids", included_conversation_ids),
        ("excluded_conversation_ids", excluded_conversation_ids),
        ("excluded_cluster_ids", excluded_cluster_ids),
    ):
        parsed = _uuid_list(value)
        if parsed is not None:
            setattr(conv, attr, parsed)
    if project is not None:
        if project == "":
            conv.project_id = None
        else:
            from src.services.projects import resolve_project
            if memory_scope_type == "none":
                raise ValidationError(
                    "an incognito ('none') conversation cannot attach to a project")
            conv.project_id = resolve_project(db, project).id
    # G16: privacy is denormalised onto episodic rows for the retrieval-time
    # visibility invariant — keep it in sync when the scope crosses the
    # none-boundary in either direction.
    now_private = memory_scope_type == "none"
    if was_private != now_private:
        db.query(EpisodicMemory).filter_by(conversation_id=conv.id).update(
            {"is_private": now_private}
        )
    out = {"status": "ok", "conversation_id": str(conv.id),
           "memory_scope_type": conv.memory_scope_type,
           "project_id": str(conv.project_id) if conv.project_id else None}
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
        "included_conversation_ids": [
            str(cid) for cid in (conv.included_conversation_ids or [])],
        "excluded_conversation_ids": [
            str(cid) for cid in (conv.excluded_conversation_ids or [])],
        "excluded_cluster_ids": [
            str(cid) for cid in (conv.excluded_cluster_ids or [])],
        "project_id": str(conv.project_id) if conv.project_id else None,
    }


def resolve_retrieval_scope(db: Session, conv: Optional[Conversation]) -> dict:
    """C6: the ONE place a conversation row becomes a retrieval scope dict.

    Both request paths call this — `chat_completions` and `retrieval_svc.
    context_for` (the MCP `ice_context` pull). They used to build the dict
    separately, and the copies had already drifted: the service path never set
    the incognito flags, so an MCP pull inside a 'none' conversation ran the
    RAG and procedural legs against global memory. One resolver, no drift.

    Precedence, most explicit first:
      none      → incognito: reads only itself, codex resolves the empty set,
                  rag/procedural are skipped downstream. Exclusions are moot.
      manual    → the user's own pick wins: a CLOSED conversation set (itself +
                  included_conversation_ids). A project attachment still
                  contributes `project_id` — the code graph and project
                  conventions are not a conversation, and detaching is a
                  separate act — but it no longer chooses the conversations.
      attached  → the project's conversations (E1/D11).
      project   → this conversation alone.
      auto      → unscoped; the legs' `is_private = FALSE` invariant applies.

    Explicitly chosen clusters are marked `cluster_ids_explicit` so the
    orchestrator's automatic picker cannot overwrite them (it did: a hand-
    picked set only survived when the picker happened to return nothing).
    """
    scope: dict = {}
    if conv is None:
        return scope

    if conv.memory_scope_type == "none":
        scope["conversation_id"] = str(conv.id)
        scope["isolated"] = True
        scope["incognito"] = True
        return scope

    if conv.project_id:
        scope["project_id"] = str(conv.project_id)

    if conv.memory_scope_type == "manual":
        picked = [conv.id] + list(conv.included_conversation_ids or [])
        seen, ordered = set(), []
        for cid in picked:
            if cid not in seen:
                seen.add(cid)
                ordered.append(str(cid))
        scope["conversation_ids"] = ordered
    elif conv.project_id:
        proj_convs = db.query(Conversation.id).filter(
            Conversation.project_id == conv.project_id,
            Conversation.memory_scope_type != "none",
        ).all()
        scope["conversation_ids"] = [str(c.id) for c in proj_convs]
    elif conv.memory_scope_type == "project":
        scope["conversation_id"] = str(conv.id)

    if conv.cluster_ids:
        scope["cluster_ids"] = [str(cid) for cid in conv.cluster_ids]
        scope["cluster_ids_explicit"] = True

    # Exclusions apply in every mode (user decision 2026-07-28) — an exclusion
    # you set is an exclusion that holds, with no hidden precondition on how
    # this conversation happens to be scoped.
    if conv.excluded_conversation_ids:
        excluded = {str(cid) for cid in conv.excluded_conversation_ids}
        scope["exclude_conversation_ids"] = sorted(excluded)
        # Subtract from the closed set too, so an excluded conversation cannot
        # be re-admitted by having been picked. Postgres `= ANY('{}')` is
        # false, so an emptied set correctly matches nothing (_conv_scope_filter
        # tests the key for presence, not truthiness).
        if scope.get("conversation_ids") is not None:
            scope["conversation_ids"] = [
                cid for cid in scope["conversation_ids"] if cid not in excluded]
    if conv.excluded_cluster_ids:
        scope["exclude_cluster_ids"] = [
            str(cid) for cid in conv.excluded_cluster_ids]
    return scope


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
