"""Retrieval service (E0 D4): explicit context pulls, recent turns,
conventions, and the session-start block.

`context_for` is ice_context's engine, F1's preview-retrieval backend, and
C11's "search specifically for Y" backend. It wraps the chat pipeline's
stages — classify → memory-decision → orchestrate — over the LIVE
classifier/embedder reached through the core object (one process = one model
load, G13). It returns STRUCTURED fragments, never a rendered prompt:
rendering belongs to each adapter (the assembler for chat, markdown for MCP,
JSON for REST).
"""
import re
import uuid
from pathlib import PurePosixPath
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from src.api.config import settings
from src.memory.models import (
    Decision,
    EpisodicMemory,
    ProceduralMemory,
    SessionSummary,
)
from src.services import slots as slots_svc

logger = structlog.get_logger("ice.services.retrieval")

# path-ish tokens in a task ("src/api/main.py", "config.py", "docker-compose.yml")
_PATHISH = re.compile(r"[\w./-]*\w\.\w{1,5}\b|[\w-]+/[\w./-]+")


def constraints_for_task(db: Session, task_text: str,
                         project_id: Optional[str] = None, cap: int = 5) -> list[dict]:
    """E8 (D7): active `constraint` decisions whose files the task mentions —
    the do-not-touch payoff. A constraint hits when one of its files_affected
    appears in the task text, or its basename matches a mentioned path's
    basename. Surfaced FIRST by context_for."""
    mentioned = {m.group(0).strip(".,;:") for m in _PATHISH.finditer(task_text or "")}
    if not mentioned:
        return []
    mentioned_low = {m.lower() for m in mentioned}
    mentioned_base = {PurePosixPath(m).name for m in mentioned_low}
    task_low = (task_text or "").lower()
    q = db.query(Decision).filter(
        Decision.decision_type == "constraint",
        Decision.valid_until.is_(None))
    if project_id:
        q = q.filter(Decision.project_id == uuid.UUID(str(project_id)))
    hits = []
    for c in q.limit(200).all():
        for f in c.files_affected or []:
            fl = f.lower()
            if fl in task_low or PurePosixPath(fl).name in mentioned_base:
                hits.append({
                    "text": f"[Constraint] {c.decision}"
                            + (f" — {c.rationale}" if c.rationale else "")
                            + f" (files: {', '.join(c.files_affected or [])})",
                    "source_type": "constraint",
                    "score": 10.0,
                    "token_count": int(len(c.decision.split()) * 1.33),
                    "source_batch_id": str(c.id),
                    "conversation_id": None,
                })
                break
        if len(hits) >= cap:
            break
    return hits


def _get_classifier():
    # Lazy: src.api.core would drag the runtime import chain into every
    # adapter that only wants recent_turns/conventions.
    from src.api.core import get_core
    core = get_core()
    if core is None:
        raise RuntimeError(
            "ICE core not started — create_core() must run before context_for")
    return core.classifier


def context_for(db: Session, task_text: str, scope: Optional[dict] = None,
                budget: Optional[int] = None) -> dict:
    """Classify → memory-decision → orchestrate, returning structured
    fragments. An explicit pull always orchestrates (spec rev 5): the B2
    decision is *reported* in the result, never used to answer empty-handed.
    First call in a headless process loads the classifier (one-time latency).
    """
    from src.api.memory_decision import decide_memory_retrieval
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    from src.retrieval.timescope import detect_timescope, to_scope_dict

    classifier = _get_classifier()
    result = classifier.classify(task_text)
    result.prompt = task_text
    tscope = detect_timescope(
        task_text,
        intent_tags=result.intent_tags,
        p_ltm=getattr(result, "p_ltm", 0.0),
        reference_signal=getattr(result, "reference_signal", False),
    )
    decision = decide_memory_retrieval(
        result, turn_count=0, total_tokens=0.0, settings=settings,
        timescope_mode=tscope.mode,
    )
    result.context_reliance = "Long_Term_Memory"   # explicit pull: orchestrate
    scope = dict(scope or {})
    ts_dict = to_scope_dict(tscope)
    if ts_dict:
        scope["timescope"] = ts_dict

    # E1 (D11) parity with the chat path: a conversation attached to a
    # project pulls project scope (code-graph allowance + the project's
    # conversations) — without this, ice_context under a coding conversation
    # would retrieve as if unattached.
    if scope.get("conversation_id") and not scope.get("project_id"):
        from src.memory.models import Conversation
        conv_row = db.query(Conversation).filter_by(
            id=uuid.UUID(str(scope["conversation_id"]))).first()
        if conv_row is not None and conv_row.project_id is not None \
                and conv_row.memory_scope_type != "none":
            scope["project_id"] = str(conv_row.project_id)
            proj_convs = db.query(Conversation.id).filter(
                Conversation.project_id == conv_row.project_id,
                Conversation.memory_scope_type != "none").all()
            scope["conversation_ids"] = [str(c.id) for c in proj_convs]

    # E11: a project-scoped pull freshens that project's working tree first,
    # so retrieved code pointers match the tree being edited right now.
    if scope.get("project_id"):
        from src.coding.reconciler import freshen_working_tree
        from src.services.projects import resolve_project
        try:
            freshen_working_tree(
                db, resolve_project(db, scope["project_id"]))
        except Exception as exc:
            db.rollback()
            logger.warning("freshen_failed_read_stays_commit_fresh",
                           project=str(scope["project_id"]), error=str(exc))

    vec = classifier.embedder.encode(task_text, convert_to_tensor=False)
    prompt_embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

    orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)
    orchestrator.set_budget_from_turn_count(
        0, total_tokens=0, classification=result,
        total_budget=budget or settings.context_budget_fallback,
    )
    fragments = orchestrator.retrieve(
        classification=result,
        conversation_id=scope.get("conversation_id"),
        prompt_embedding=prompt_embedding,
        scope=scope,
    )
    # E8: constraints FIRST whenever the task mentions their files — a
    # do-not-touch rule outranks every retrieved fragment by construction.
    constraint_frags = constraints_for_task(
        db, task_text, project_id=(scope or {}).get("project_id"))
    logger.info("context_for", query_words=len(task_text.split()),
                fragments=len(fragments), constraints=len(constraint_frags),
                timescope=tscope.mode)
    return {
        "query": task_text,
        "topic_tags": result.topic_tags,
        "intent_tags": result.intent_tags,
        "timescope": tscope.mode,
        "memory_decision": {"retrieve": decision.retrieve,
                            "p_need_mem": round(decision.p_need_mem, 4)},
        "fragments": constraint_frags + [
            {
                "text": f.text,
                "source_type": f.source_type,
                "score": f.score,
                "token_count": f.token_count,
                "source_batch_id": f.source_batch_id,
                "conversation_id": f.conversation_id,
            }
            for f in fragments
        ],
    }


def recent_turns(db: Session, conversation_id: Optional[str] = None,
                 limit: int = 10, project: Optional[str] = None) -> list[dict]:
    """Most recent stored turns. Unscoped calls honor the G16 visibility
    invariant (private turns are readable only when explicitly scoped to
    their own conversation). *project* (E1): all of that project's
    conversations — never private ones (incognito can't attach, and the
    filter is the backstop)."""
    q = db.query(EpisodicMemory)
    if conversation_id:
        q = q.filter_by(conversation_id=uuid.UUID(conversation_id))
    elif project:
        from src.memory.models import Conversation
        from src.services.projects import resolve_project
        proj = resolve_project(db, project)
        conv_ids = [c.id for c in db.query(Conversation.id).filter(
            Conversation.project_id == proj.id).all()]
        q = q.filter(EpisodicMemory.conversation_id.in_(conv_ids),
                     EpisodicMemory.is_private.is_(False))
    else:
        q = q.filter_by(is_private=False)
    rows = q.order_by(EpisodicMemory.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": str(t.id),
            "conversation_id": str(t.conversation_id),
            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
            "text": t.summary_text or t.raw_text[:300],
            "topic_tags": t.topic_tags or [],
            "is_bookmarked": t.is_bookmarked,
        }
        for t in rows
    ]


def conventions(db: Session, limit: int = 15) -> list[dict]:
    """Active procedural patterns — the user's observed recurring behaviors."""
    rows = db.query(ProceduralMemory).filter_by(is_active=True).order_by(
        ProceduralMemory.confidence_score.desc()).limit(limit).all()
    return [
        {
            "pattern": p.pattern_name,
            "description": p.pattern_description or "",
            "confidence": p.confidence_score,
            "reinforcements": p.reinforcement_count,
            "last_observed": p.last_observed.isoformat() if p.last_observed else None,
        }
        for p in rows
    ]


def session_start_block(db: Session) -> dict:
    """Data behind the ice://session-start resource: active slots + the most
    recent session summary. E4 enriches THIS function into the full
    welcome-back block later (resource URI stays stable)."""
    summary = db.query(SessionSummary).order_by(
        SessionSummary.session_date.desc()).first()
    return {
        "slots": slots_svc.list_slots(db),
        "last_session": None if summary is None else {
            "date": summary.session_date.isoformat() if summary.session_date else None,
            "topics_covered": summary.topics_covered or [],
            "decisions_made": summary.decisions_made or "",
            "unresolved_items": summary.unresolved_items or "",
            "entities_updated": summary.entities_updated or [],
            "patterns_observed": summary.patterns_observed or [],
        },
    }
