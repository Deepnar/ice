"""E3 — reconcile-on-commit (spec D5). Commit is the semantic signal; polling
is the fallback, never the design.

Trigger paths into ``reconcile_project`` (a cpu-lane runtime job — pure
git+DB, no LLM):

  1. consent-installed post-commit hook → POST /user-control/projects/{id}/
     commit → ``notify_work_unit("commit", ...)`` → the handler registered in
     ``create_core()`` enqueues this job (C7's reserved seam, zero runtime
     changes);
  2. hook fallback marker file ``$GIT_DIR/ice_pending_commit`` (written when
     the POST fails — app down) and plain HEAD drift, both picked up by
     ``poll_projects`` (interval job, 600s — the ≤10-min lag when the hook
     was declined).

Steps per commit-range since ``project_state.last_reconciled_commit``:
changed-file incremental re-parse (E1b) → hash-gated fact re-derive (E9) →
task linking → decision extraction over commit messages (cue-gated, enqueued
to the gpu-lane ``decision_extract`` job) → advance the state row. Stale-work
detection is NOT here — it's the maintenance agent's ``stale_work`` detector
(the worklist re-derives staleness itself; no parallel notifier).
"""
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from src.memory.models import Project, ProjectState, ProjectTask

logger = structlog.get_logger("ice.coding.reconciler")

MAX_COMMITS_SCANNED = 20    # decision extraction per reconcile, newest first


def _git(root, *args, strip: bool = True):
    """Run git in *root*; None on any failure (callers treat git errors as
    'skip this step', never crash the reconcile)."""
    try:
        proc = subprocess.run(["git", *args], cwd=str(root),
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("git_call_failed", args=args, error=str(exc))
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() if strip else proc.stdout


def marker_path(root) -> Path:
    git_dir = _git(root, "rev-parse", "--git-dir")
    base = Path(git_dir) if git_dir and Path(git_dir).is_absolute() \
        else Path(root) / (git_dir or ".git")
    return base / "ice_pending_commit"


def make_commit_handler(runtime):
    """The "commit" work-unit handler (registered by create_core): turn the
    signal into the reconcile event job — nothing else happens in-line."""
    def _handle_commit(project_id=None, commit_hash=None, **_):
        if not project_id:
            logger.warning("commit_work_unit_without_project")
            return
        runtime.enqueue("project_reconcile", project_id=str(project_id))
    return _handle_commit


def _mark_unreachable(db, project, state, reachable: bool) -> bool:
    """Track root reachability in project settings; tools read this to say so
    instead of silently serving a stale graph (spec §4)."""
    settings = dict(project.settings or {})
    if not reachable:
        if not settings.get("unreachable"):
            settings["unreachable"] = True
            project.settings = settings
            db.commit()
            logger.warning("project_unreachable", project=project.slug,
                           roots=project.roots)
        return False
    if settings.get("unreachable"):
        settings.pop("unreachable", None)
        project.settings = settings
        db.commit()
        logger.info("project_reachable_again", project=project.slug)
    return True


def reconcile_project(db, project_id=None, force_full: bool = False) -> dict:
    """One reconcile pass for one project (D5 steps 1–5)."""
    from src.coding.code_graph import CodeExtractor
    from src.coding.project_facts import derive_project_facts

    project = db.query(Project).filter_by(id=uuid.UUID(str(project_id))).first()
    if project is None:
        logger.warning("reconcile_unknown_project", project_id=str(project_id))
        return {"status": "unknown_project"}
    state = db.query(ProjectState).filter_by(project_id=project.id).first()
    if state is None:
        state = ProjectState(project_id=project.id)
        db.add(state)
        db.flush()

    root = Path(project.roots[0]) if project.roots else None
    if root is None or not root.is_dir():
        _mark_unreachable(db, project, state, reachable=False)
        return {"status": "unreachable"}
    _mark_unreachable(db, project, state, reachable=True)

    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        logger.warning("reconcile_not_a_repo", project=project.slug)
        return {"status": "no_git"}
    branch = _git(root, "branch", "--show-current") or "(detached)"
    last = state.last_reconciled_commit

    # consume the hook's fallback marker regardless of outcome below
    try:
        marker_path(root).unlink(missing_ok=True)
    except OSError:
        pass

    if last == head and not force_full:
        if state.current_branch != branch:
            state.current_branch = branch
            state.updated_at = datetime.now(timezone.utc)
            db.commit()
        return {"status": "up_to_date", "head": head}

    # Changed files since last reconcile. Unknown old commit (force-push /
    # history rewrite) or no anchor yet → full re-parse fallback (spec §4).
    changed = None
    if last and not force_full:
        diff = _git(root, "diff", "--name-only", last, head, strip=False)
        if diff is not None:
            changed = [ln.strip() for ln in diff.splitlines() if ln.strip()]
        else:
            logger.warning("reconcile_unknown_base_full_reparse",
                           project=project.slug, last=last)

    extractor = CodeExtractor(project)
    if changed is None:
        graph_stats = extractor.full_sync(db)
        fact_stats = derive_project_facts(db, project)
        commits = [head]
    else:
        py_changed = [f for f in changed if f.endswith(".py")
                      and not any(part in extractor.ignores
                                  for part in Path(f).parts)]
        graph_stats = extractor.sync_files(db, py_changed) if py_changed else {}
        fact_stats = derive_project_facts(db, project, changed_files=changed)
        rev_list = _git(root, "rev-list", f"{last}..{head}", strip=False)
        commits = [c.strip() for c in (rev_list or "").splitlines() if c.strip()] \
            or [head]

    # D5 step 3: link the range's commits to the active task (active =
    # status 'active', else the most recent pending).
    task = (db.query(ProjectTask)
            .filter_by(project_id=project.id, status="active")
            .order_by(ProjectTask.updated_at.desc()).first()
            or db.query(ProjectTask)
            .filter_by(project_id=project.id, status="pending")
            .order_by(ProjectTask.updated_at.desc()).first())
    if task is not None:
        existing = set(task.commit_hashes or [])
        new_hashes = [c for c in commits if c not in existing]
        if new_hashes:
            task.commit_hashes = (task.commit_hashes or []) + new_hashes
            if changed:
                merged = set(task.files_changed or []) | set(changed)
                task.files_changed = sorted(merged)
            task.updated_at = datetime.now(timezone.utc)
            state.last_task_id = task.id

    # E8 at reconcile: cue-gated decision extraction over commit messages —
    # the LLM stage rides the gpu-lane decision_extract job, never this one.
    from src.workers.decision_extractor import decision_cue
    from src.workers.runtime import get_runtime
    runtime = get_runtime()
    queued_msgs = 0
    for commit in commits[:MAX_COMMITS_SCANNED]:
        msg = _git(root, "log", "-1", "--format=%B", commit)
        if not msg or decision_cue(msg) is None:
            continue
        if runtime is not None:
            runtime.enqueue("decision_extract", project_id=str(project.id),
                            text_content=msg, origin=f"commit:{commit[:12]}")
            queued_msgs += 1
        else:
            logger.info("decision_cue_no_runtime", commit=commit[:12])

    state.last_reconciled_commit = head
    state.current_branch = branch
    state.updated_at = datetime.now(timezone.utc)
    db.commit()
    result = {"status": "reconciled", "head": head, "branch": branch,
              "changed_files": len(changed) if changed is not None else -1,
              "graph": graph_stats, "facts": fact_stats,
              "decision_msgs_queued": queued_msgs,
              "task_linked": str(task.id) if task else None}
    logger.info("project_reconciled", project=project.slug, **{
        k: v for k, v in result.items() if k not in ("graph", "facts")})
    return result


def poll_projects(db) -> dict:
    """The 10-min fallback (D5): for every registered project, reconcile when
    the hook's marker file exists or HEAD drifted past the state row."""
    out = {"checked": 0, "reconciled": 0}
    for project in db.query(Project).all():
        out["checked"] += 1
        root = Path(project.roots[0]) if project.roots else None
        state = db.query(ProjectState).filter_by(project_id=project.id).first()
        if root is None or not root.is_dir():
            if state is not None:
                _mark_unreachable(db, project, state, reachable=False)
            continue
        head = _git(root, "rev-parse", "HEAD")
        if head is None:
            continue
        marker = marker_path(root)
        drifted = state is None or state.last_reconciled_commit != head
        if marker.exists() or drifted:
            reconcile_project(db, project_id=project.id)
            out["reconciled"] += 1
    if out["reconciled"]:
        logger.info("project_poll", **out)
    return out
