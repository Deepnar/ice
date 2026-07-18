"""Codex-graph service (E0 D3) — the one surface for reading and editing
entities. Wraps T4's evolution functions AS BUILT (build_entity_timeline /
entity_diff / history_exists in src/retrieval/evolution.py); F3's graph view
and D1's agent consume these same functions.

The A7/F3 rule is enforced here and nowhere else: edits go to `description`
(journaled via T's log_description_update, D13), and `context_payload` is
DERIVED — regenerated after every edit, never written directly.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from src.api.config import settings
from src.memory.models import CodexEdge, CodexEntity
from src.retrieval.evolution import build_entity_timeline, log_description_update
from src.retrieval.evolution import entity_diff as _evolution_diff
from src.services.errors import NotFoundError, ValidationError

logger = structlog.get_logger("ice.services.graph")


def _resolve(db: Session, name_or_id: str) -> CodexEntity:
    """Entity by UUID, canonical name (casefolded), or alias."""
    try:
        entity = db.query(CodexEntity).filter_by(id=uuid.UUID(name_or_id)).first()
        if entity:
            return entity
    except (ValueError, AttributeError):
        pass
    canonical = str(name_or_id).strip().lower()
    entity = (db.query(CodexEntity).filter_by(canonical_name=canonical).first()
              or db.query(CodexEntity).filter(CodexEntity.aliases.any(canonical)).first())
    if not entity:
        raise NotFoundError(f"Entity not found: {name_or_id}")
    return entity


def _edge_out(db: Session, edge: CodexEdge, anchor_id) -> dict:
    other_id = edge.target_id if edge.source_id == anchor_id else edge.source_id
    other = db.query(CodexEntity).filter_by(id=other_id).first()
    return {
        "relation": edge.relation,
        "direction": "out" if edge.source_id == anchor_id else "in",
        "other": other.canonical_name if other else "?",
        "negated": edge.negated,
        "strength": edge.strength,
        "confidence": edge.confidence,
        "extraction_confidence": edge.extraction_confidence,
        "since": edge.valid_from.isoformat() if edge.valid_from else None,
        "until": edge.valid_until.isoformat() if edge.valid_until else None,
    }


def entity_view(db: Session, name_or_id: str) -> dict:
    """The note + typed links/backlinks — what F3 renders and ice_why returns."""
    entity = _resolve(db, name_or_id)
    out_edges = db.query(CodexEdge).filter(
        CodexEdge.source_id == entity.id, CodexEdge.valid_until.is_(None)
    ).order_by(CodexEdge.strength.desc()).limit(20).all()
    in_edges = db.query(CodexEdge).filter(
        CodexEdge.target_id == entity.id, CodexEdge.valid_until.is_(None)
    ).order_by(CodexEdge.strength.desc()).limit(20).all()
    return {
        "id": str(entity.id),
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type,
        "aliases": entity.aliases or [],
        "description": entity.description or "",
        "note": entity.context_payload or "",
        "properties": entity.properties or {},
        "links": [_edge_out(db, e, entity.id) for e in out_edges],
        "backlinks": [_edge_out(db, e, entity.id) for e in in_edges],
        "last_updated": entity.last_updated.isoformat() if entity.last_updated else None,
    }


def entity_edit(db: Session, name_or_id: str, description: str,
                source: str = "manual_edit") -> dict:
    """Manual description editing is BORN here (T-spec rev note 5): journal
    the overwrite (D13), regenerate the derived note, never accept a
    context_payload write."""
    if description is None:
        raise ValidationError("entity_edit requires a description")
    entity = _resolve(db, name_or_id)
    # Heavy import stays lazy — codex_extractor loads an embedder at import.
    from src.workers.codex_extractor import _regenerate_context_payload
    old = entity.description
    entity.description = description
    log_description_update(db, entity, old, description, source=source)
    _regenerate_context_payload(entity, db)
    entity.last_updated = datetime.now(timezone.utc)
    name = entity.canonical_name          # capture before commit (expiry trap)
    db.commit()
    logger.info("entity_edited", entity=name, source=source)
    return entity_view(db, name)


def edges_list(db: Session, name_or_id: str, include_expired: bool = False) -> dict:
    entity = _resolve(db, name_or_id)
    q = db.query(CodexEdge).filter(
        (CodexEdge.source_id == entity.id) | (CodexEdge.target_id == entity.id))
    if not include_expired:
        q = q.filter(CodexEdge.valid_until.is_(None))
    edges = q.order_by(CodexEdge.strength.desc()).all()
    return {"entity": entity.canonical_name,
            "edges": [_edge_out(db, e, entity.id) for e in edges]}


def entity_timeline(db: Session, name_or_id: str,
                    max_transitions: Optional[int] = None) -> dict:
    """T4's `log` porcelain. timeline is None when the entity has no
    event-backed supersession history (a story of purely-current facts would
    just repeat the note)."""
    entity = _resolve(db, name_or_id)
    timeline = build_entity_timeline(
        db, entity,
        max_transitions=max_transitions or settings.timeline_max_transitions)
    return {"entity": entity.canonical_name, "timeline": timeline,
            "has_history": timeline is not None}


def _parse_ts(value, name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        ts = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be an ISO datetime, got {value!r}")
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def entity_diff(db: Session, name_or_id: str, t0, t1) -> dict:
    """T4's `diff` porcelain over [t0, t1) — JSON-safe (dates → ISO)."""
    entity = _resolve(db, name_or_id)
    diff = _evolution_diff(db, entity, _parse_ts(t0, "t0"), _parse_ts(t1, "t1"))
    return {"entity": entity.canonical_name, **{
        kind: [{**item, "date": item["date"].isoformat()} for item in items]
        for kind, items in diff.items()
    }}


def where_symbol(db: Session, symbol: str) -> dict:
    """ice_where's engine (E1b swap-in-place): the code graph first —
    definitions with file:line pointers — falling back to codex name/alias
    resolution for non-code names. Results carry project + path."""
    from src.coding.code_graph import where_symbol_rows
    matches = where_symbol_rows(db, symbol)
    if matches:
        return {"resolved": True, "engine": "code_graph", "matches": matches}
    try:
        return {"resolved": True, "engine": "codex name/alias resolution",
                **entity_view(db, symbol)}
    except NotFoundError:
        return {"resolved": False, "engine": "code_graph + codex",
                "note": f"no code symbol or memory entity matches {symbol!r}"}


def render_architecture_doc(db: Session, project_ref) -> str:
    """E8 (D8): the architecture doc is a RENDER over the stores, never a
    hand-maintained file — module tree with docstring one-liners, key
    decisions with rationale, constraints, conventions (procedural rows),
    and the recent git timeline read LIVE (git IS the timeline; commits are
    never re-stored). If a section looks wrong, fix the underlying store."""
    import subprocess
    from pathlib import Path

    from src.memory.models import Decision, ProceduralMemory
    from src.services.projects import resolve_project

    project = resolve_project(db, project_ref)
    lines = [f"# {project.name} — architecture (rendered "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC)", ""]
    if (project.settings or {}).get("unreachable"):
        lines += ["> ⚠ project root currently unreachable — this render may be stale", ""]

    # 1. Module tree with per-module one-liners.
    modules = db.query(CodexEntity).filter_by(
        project_id=project.id, source="static_analysis",
        entity_type="module").order_by(CodexEntity.canonical_name).all()
    lines.append("## Modules")
    if modules:
        for m in modules:
            props = m.properties or {}
            doc = props.get("docstring_summary") or ""
            lines.append(f"- `{props.get('display_name', m.canonical_name)}`"
                         + (f" — {doc}" if doc else "")
                         + f"  ({props.get('file_path')})")
    else:
        lines.append("(no code graph yet — register/reconcile the project)")
    lines.append("")

    # 2. Key decisions (active, by recency) + constraints.
    decisions = db.query(Decision).filter(
        Decision.project_id == project.id, Decision.valid_until.is_(None),
        Decision.decision_type != "constraint",
    ).order_by(Decision.valid_from.desc()).limit(10).all()
    lines.append("## Key decisions")
    lines += [f"- {d.decision}" + (f" — *{d.rationale}*" if d.rationale else "")
              for d in decisions] or ["(none recorded)"]
    lines.append("")
    constraints = db.query(Decision).filter(
        Decision.project_id == project.id, Decision.valid_until.is_(None),
        Decision.decision_type == "constraint",
    ).order_by(Decision.valid_from.desc()).limit(10).all()
    lines.append("## Constraints")
    lines += [f"- {c.decision}"
              + (f" (files: {', '.join(c.files_affected or [])})"
                 if c.files_affected else "")
              for c in constraints] or ["(none recorded)"]
    lines.append("")

    # 3. Conventions — project-scoped procedural rows.
    conventions = db.query(ProceduralMemory).filter_by(
        project_id=project.id, is_active=True).order_by(
        ProceduralMemory.confidence_score.desc()).limit(10).all()
    lines.append("## Conventions")
    lines += [f"- {p.pattern_description or p.pattern_name}"
              for p in conventions] or ["(none observed yet)"]
    lines.append("")

    # 4. Recent git timeline — read live.
    lines.append("## Recent commits")
    root = Path(project.roots[0]) if project.roots else None
    log_out = None
    if root and root.is_dir():
        try:
            proc = subprocess.run(
                ["git", "log", "--oneline", "-15"], cwd=str(root),
                capture_output=True, text=True, timeout=15)
            log_out = proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            log_out = None
    lines.append(log_out if log_out else "(git log unavailable)")
    return "\n".join(lines)
