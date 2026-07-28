"""Projects service (E1/E3/E4) — register/bootstrap, state, tasks, decisions
CRUD, the commit notification, and the session-start renderer. E0 pattern:
plain functions, ``fn(db, ...) -> dict``, zero FastAPI/HTTP concepts; the
REST router, MCP tools, and future F-track UI are thin adapters over these.
"""
import re
import stat
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from src.api.config import settings
from src.memory.models import Decision, Project, ProjectState, ProjectTask
from src.services.errors import NotFoundError, ValidationError

logger = structlog.get_logger("ice.services.projects")

HOOK_MARKER = "# installed by ICE (E3 reconcile-on-commit)"
HOOK_TEMPLATE = """#!/bin/sh
{marker}
# Notifies ICE of new commits; falls back to a marker file the poll job
# picks up when the app is down. Always exits 0 — never blocks a commit.
hash=$(git rev-parse HEAD)
curl -fsS -m 2 -X POST "http://localhost:8000/user-control/projects/{project_id}/commit" \\
     -H 'Content-Type: application/json' -d "{{\\"commit\\": \\"$hash\\"}}" \\
     >/dev/null 2>&1 || echo "$hash" > "$(git rev-parse --git-dir)/ice_pending_commit"
exit 0
"""


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


def resolve_project(db: Session, ref) -> Project:
    """Project by UUID, slug, or name."""
    try:
        proj = db.query(Project).filter_by(id=uuid.UUID(str(ref))).first()
        if proj:
            return proj
    except (ValueError, AttributeError, TypeError):
        pass
    proj = (db.query(Project).filter_by(slug=str(ref)).first()
            or db.query(Project).filter_by(name=str(ref)).first())
    if not proj:
        raise NotFoundError(f"Project not found: {ref}")
    return proj


# ── Registration + bootstrap (USER-REQUIRED entry point) ────────────────────

def install_hook(project: Project) -> dict:
    """Consent-installed post-commit hook (D5). Never overwrites a hook we
    didn't write; always report what happened."""
    root = Path(project.roots[0])
    git_dir_out = subprocess.run(
        ["git", "rev-parse", "--git-dir"], cwd=str(root),
        capture_output=True, text=True)
    if git_dir_out.returncode != 0:
        return {"installed": False, "reason": "not a git repository"}
    git_dir = Path(git_dir_out.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    hook = git_dir / "hooks" / "post-commit"
    if hook.exists() and HOOK_MARKER not in hook.read_text(errors="replace"):
        return {"installed": False,
                "reason": "an existing post-commit hook was found — not "
                          "overwriting it; add the ICE curl line manually or "
                          "rely on the 10-min poll"}
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(HOOK_TEMPLATE.format(marker=HOOK_MARKER,
                                         project_id=project.id))
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {"installed": True, "path": str(hook)}


def register_project(db: Session, name: str, root: str,
                     slug: Optional[str] = None,
                     install_git_hook: bool = False,
                     replay_git_log: bool = False) -> dict:
    """Register a repo root and bootstrap it: full E1b parse + E9 facts.
    Done = the returned report lists entities/edges/facts. git-log replay is
    OPTIONAL and off by default (heavy; F10 fast-forward semantics) — the
    flag is accepted and recorded now, honored when F10's importer lands."""
    from src.coding.code_graph import CodeExtractor
    from src.coding.project_facts import derive_project_facts
    from src.coding.reconciler import _git

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValidationError(f"root is not a directory: {root_path}")
    name = name.strip()
    if not name:
        raise ValidationError("project name required")
    if db.query(Project).filter_by(name=name).first():
        raise ValidationError(f"project name already registered: {name}")
    slug = _slugify(slug or name)
    if db.query(Project).filter_by(slug=slug).first():
        raise ValidationError(f"project slug already registered: {slug}")

    project = Project(name=name, slug=slug, roots=[str(root_path)],
                      settings={"replay_git_log": bool(replay_git_log)})
    db.add(project)
    db.flush()
    state = ProjectState(project_id=project.id,
                         current_branch=_git(root_path, "branch", "--show-current"),
                         last_reconciled_commit=_git(root_path, "rev-parse", "HEAD"),
                         updated_at=datetime.now(timezone.utc))
    db.add(state)
    db.commit()

    extractor = CodeExtractor(project)
    graph_stats = extractor.full_sync(db)
    fact_stats = derive_project_facts(db, project)

    hook_result = {"installed": False, "reason": "declined (poll fallback, ≤10 min lag)"}
    if install_git_hook:
        hook_result = install_hook(project)
        proj_settings = dict(project.settings or {})
        proj_settings["hook_installed"] = hook_result["installed"]
        project.settings = proj_settings
        db.commit()

    report = {
        "status": "registered",
        "project_id": str(project.id), "name": name, "slug": slug,
        "root": str(root_path),
        "graph": graph_stats, "facts": fact_stats, "hook": hook_result,
        "replay_git_log": bool(replay_git_log),
    }
    logger.info("project_registered", **{k: v for k, v in report.items()
                                         if k not in ("graph", "facts")})
    return report


def list_projects(db: Session) -> list[dict]:
    out = []
    for p in db.query(Project).order_by(Project.created_at).all():
        state = db.query(ProjectState).filter_by(project_id=p.id).first()
        out.append({
            "project_id": str(p.id), "name": p.name, "slug": p.slug,
            "roots": p.roots, "settings": p.settings or {},
            "goal": state.goal if state else None,
            "current_branch": state.current_branch if state else None,
            "last_reconciled_commit": state.last_reconciled_commit if state else None,
        })
    return out


def project_status(db: Session, ref) -> dict:
    from src.memory.models import CodexEntity
    project = resolve_project(db, ref)
    state = db.query(ProjectState).filter_by(project_id=project.id).first()
    counts = {
        "code_entities": db.query(CodexEntity).filter_by(
            project_id=project.id, source="static_analysis").count(),
        "facts": db.query(CodexEntity).filter_by(
            project_id=project.id, source="derived").count(),
        "decisions_active": db.query(Decision).filter(
            Decision.project_id == project.id,
            Decision.valid_until.is_(None)).count(),
        "tasks_open": db.query(ProjectTask).filter(
            ProjectTask.project_id == project.id,
            ProjectTask.status.in_(("pending", "active"))).count(),
    }
    return {
        "project_id": str(project.id), "name": project.name,
        "slug": project.slug, "roots": project.roots,
        "unreachable": bool((project.settings or {}).get("unreachable")),
        "goal": state.goal if state else None,
        "current_branch": state.current_branch if state else None,
        "last_reconciled_commit": state.last_reconciled_commit if state else None,
        "last_session_at": state.last_session_at.isoformat()
        if state and state.last_session_at else None,
        **counts,
    }


def set_goal(db: Session, ref, goal: str) -> dict:
    project = resolve_project(db, ref)
    state = db.query(ProjectState).filter_by(project_id=project.id).first()
    if state is None:
        state = ProjectState(project_id=project.id)
        db.add(state)
    state.goal = goal
    state.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"status": "ok", "project": project.slug, "goal": goal}


def notify_commit(db: Session, ref, commit_hash: Optional[str] = None) -> dict:
    """The hook endpoint's engine: route the commit signal into the runtime's
    work-unit table. No runtime (headless tests) → reconcile inline."""
    from src.coding.reconciler import reconcile_project
    from src.workers.runtime import get_runtime
    project = resolve_project(db, ref)
    runtime = get_runtime()
    if runtime is not None:
        runtime.notify_work_unit("commit", project_id=str(project.id),
                                 commit_hash=commit_hash)
        return {"status": "queued", "project": project.slug}
    result = reconcile_project(db, project_id=project.id)
    return {"status": "reconciled_inline", "project": project.slug,
            "result": result.get("status")}


# ── Tasks CRUD ──────────────────────────────────────────────────────────────

def task_add(db: Session, ref, title: str, status: str = "pending") -> dict:
    project = resolve_project(db, ref)
    if not (title or "").strip():
        raise ValidationError("task title required")
    if status not in ("pending", "active"):
        raise ValidationError("new task status must be pending or active")
    now = datetime.now(timezone.utc)
    task = ProjectTask(project_id=project.id, title=title.strip(),
                       status=status, created_at=now, updated_at=now)
    db.add(task)
    db.flush()
    out = {"status": "ok", "task_id": str(task.id), "title": task.title}
    db.commit()
    return out


def task_list(db: Session, ref, status: Optional[str] = None) -> list[dict]:
    project = resolve_project(db, ref)
    q = db.query(ProjectTask).filter_by(project_id=project.id)
    if status:
        q = q.filter_by(status=status)
    return [
        {"task_id": str(t.id), "title": t.title, "status": t.status,
         "commits": len(t.commit_hashes or []),
         "updated_at": t.updated_at.isoformat() if t.updated_at else None}
        for t in q.order_by(ProjectTask.updated_at.desc()).all()
    ]


def task_set_status(db: Session, task_id: str, status: str) -> dict:
    if status not in ("pending", "active", "done", "dropped"):
        raise ValidationError(f"unknown task status: {status}")
    task = db.query(ProjectTask).filter_by(id=uuid.UUID(str(task_id))).first()
    if task is None:
        raise NotFoundError(f"Task not found: {task_id}")
    task.status = status
    task.updated_at = datetime.now(timezone.utc)
    out = {"status": "ok", "task_id": str(task.id), "new_status": status}
    db.commit()
    return out


# ── Decisions CRUD (manual path; E8's extractor is the automatic one) ──────

def decisions_list(db: Session, ref, decision_type: Optional[str] = None,
                   include_superseded: bool = False, limit: int = 25) -> list[dict]:
    project = resolve_project(db, ref)
    q = db.query(Decision).filter_by(project_id=project.id)
    if not include_superseded:
        q = q.filter(Decision.valid_until.is_(None))
    if decision_type:
        q = q.filter_by(decision_type=decision_type)
    return [
        {"id": str(d.id), "decision": d.decision, "rationale": d.rationale,
         "type": d.decision_type, "files": d.files_affected or [],
         "since": d.valid_from.isoformat() if d.valid_from else None,
         "until": d.valid_until.isoformat() if d.valid_until else None,
         "superseded_by": str(d.superseded_by) if d.superseded_by else None}
        for d in q.order_by(Decision.valid_from.desc()).limit(limit).all()
    ]


def decision_add(db: Session, ref, decision: str, rationale: str = "",
                 decision_type: str = "decision",
                 files: Optional[list] = None) -> dict:
    """Manual decision entry (MCP `decisions_add` write path — applies
    immediately, D8). Goes through E8's `reconcile_and_insert`, so a manual
    entry dedupes and supersedes conflicting older decisions exactly like an
    extracted one. Status is E8's vocabulary — `recorded` | `duplicate`
    (nothing written, `existing` names the row) | `superseded` |
    `conflict_queued` — never a bare "ok": a caller writing a decision that
    silently collided with an active one needs to be told."""
    from src.workers.decision_extractor import _embed, reconcile_and_insert
    project = resolve_project(db, ref)
    if not (decision or "").strip():
        raise ValidationError("decision text required")
    if decision_type not in ("decision", "constraint", "incident"):
        raise ValidationError(f"unknown decision_type: {decision_type}")
    text_ = decision.strip()[:500]
    out = reconcile_and_insert(db, project.id, {
        "decision": text_,
        "rationale": (rationale or "").strip()[:1000],
        "alternatives_rejected": [],
        "files_affected": [str(f)[:300] for f in (files or [])[:10]],
        "type": decision_type,
    }, _embed(text_), None, origin="manual")
    out["type"] = decision_type
    return out


# ── Session-start block (E4 D6) ─────────────────────────────────────────────

def _diffstat_since(root: Path, since: Optional[datetime]) -> Optional[str]:
    from src.coding.reconciler import _git
    if since is None:
        return None
    # reflog-based @{<date>} — local single-user reality; errors → no diffstat.
    anchor = f"HEAD@{{{since.strftime('%Y-%m-%d %H:%M:%S')}}}"
    out = _git(root, "diff", "--stat", anchor, "HEAD")
    if not out:
        return None
    lines = out.splitlines()
    return "\n".join(lines[-6:]) if len(lines) > 6 else out


def session_start_data(db: Session, project: Project) -> dict:
    """The block's data: project_state + diffstat since last_session_at +
    open constraints + ≤3 pending/stale tasks + ≤3 freshest decisions.
    Advances last_session_at only when it was itself stale past the session
    gap (rev 8: repeated renders inside one sitting don't move it)."""
    state = db.query(ProjectState).filter_by(project_id=project.id).first()
    now = datetime.now(timezone.utc)
    last_session = state.last_session_at if state else None
    root = Path(project.roots[0]) if project.roots else None

    constraints = db.query(Decision).filter(
        Decision.project_id == project.id,
        Decision.decision_type == "constraint",
        Decision.valid_until.is_(None),
    ).order_by(Decision.valid_from.desc()).limit(5).all()
    decisions = db.query(Decision).filter(
        Decision.project_id == project.id,
        Decision.decision_type != "constraint",
        Decision.valid_until.is_(None),
    ).order_by(Decision.valid_from.desc()).limit(3).all()
    tasks = db.query(ProjectTask).filter(
        ProjectTask.project_id == project.id,
        ProjectTask.status.in_(("active", "pending")),
    ).order_by(ProjectTask.updated_at.asc()).limit(3).all()

    last_task = None
    if state and state.last_task_id:
        t = db.query(ProjectTask).filter_by(id=state.last_task_id).first()
        last_task = t.title if t else None

    data = {
        "project": project.name, "slug": project.slug,
        "goal": state.goal if state else None,
        "branch": state.current_branch if state else None,
        "last_task": last_task,
        "last_session_at": last_session.isoformat() if last_session else None,
        "diffstat": _diffstat_since(root, last_session) if root else None,
        "constraints": [c.decision for c in constraints],
        "tasks": [{"title": t.title, "status": t.status,
                   "days_idle": int((now - t.updated_at).total_seconds() // 86400)
                   if t.updated_at else None} for t in tasks],
        "decisions": [{"decision": d.decision,
                       "since": d.valid_from.strftime("%Y-%m-%d")
                       if d.valid_from else None} for d in decisions],
    }
    if state is not None and (
            last_session is None
            or now - last_session > timedelta(minutes=settings.session_gap_minutes)):
        state.last_session_at = now
        db.commit()
    return data


def render_session_start(data: dict) -> str:
    """Markdown renderer shared by the chat assembler and E7's
    ice://session-start resource (same block, two adapters)."""
    lines = [f"## Project: {data['project']} (branch: {data['branch'] or '?'})"]
    if data.get("goal"):
        lines.append(f"Goal: {data['goal']}")
    if data.get("last_task"):
        lines.append(f"Last worked task: {data['last_task']}")
    if data.get("diffstat"):
        lines.append("Changes since last session:\n" + data["diffstat"])
    if data.get("constraints"):
        lines.append("Constraints (do not violate):")
        lines += [f"- {c}" for c in data["constraints"]]
    if data.get("tasks"):
        lines.append("Open tasks:")
        lines += [f"- [{t['status']}] {t['title']}"
                  + (f" (idle {t['days_idle']}d)" if t.get("days_idle") else "")
                  for t in data["tasks"]]
    if data.get("decisions"):
        lines.append("Recent decisions:")
        lines += [f"- {d['decision']}"
                  + (f" (since {d['since']})" if d.get("since") else "")
                  for d in data["decisions"]]
    return "\n".join(lines)


def chat_session_start(db: Session, project_id) -> Optional[str]:
    """The assembler's one-call entry: rendered block or None."""
    project = db.query(Project).filter_by(id=uuid.UUID(str(project_id))).first()
    if project is None:
        return None
    return render_session_start(session_start_data(db, project))
