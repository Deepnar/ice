"""Ingestion service (E0 adapter seam for F10/F14).

`start_import` and `import_status` are the operation logic; REST
(`POST /user-control/import`, `GET /user-control/import[/{id}]`) and
`ice_control action="import_status"` are thin adapters over them (parity
harness, like every other E0 service). The heavy engine lives in
`src/ingestion/` — this module only validates input, manages the ImportRun
lifecycle row, enforces one-import-at-a-time, prints the D5 cost estimate,
and hands off to the runtime's `import_replay` job (or runs inline for
dry-run / test).
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.api.config import settings
import structlog
from sqlalchemy.orm import Session

from src.ingestion import importer
from src.memory.models import ImportRun
from src.services.errors import ConflictError, NotFoundError, ValidationError

logger = structlog.get_logger("ice.services.ingestion")


def _run_dict(run: ImportRun) -> dict:
    return {
        "id": str(run.id), "source_path": run.source_path,
        "source_format": run.source_format, "policy": run.policy,
        "kind": run.kind, "status": run.status,
        "total_conversations": run.total_conversations,
        "total_turns": run.total_turns,
        "done_conversations": run.done_conversations,
        "done_turns": run.done_turns,
        "skipped_conversations": run.skipped_conversations,
        "failed_turns": run.failed_turns,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error": run.error, "report": run.report,
    }


def _reap_stale(db: Session) -> None:
    """A `running` row whose heartbeat is older than the staleness window is a
    crashed import — mark it aborted so a fresh start isn't blocked (rev 10;
    resume rides the hash ledger, not the row)."""
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=settings.import_stale_run_seconds)
    for run in db.query(ImportRun).filter(ImportRun.status == "running").all():
        stamp = run.updated_at or run.started_at
        if stamp is None or stamp < cutoff:
            run.status = "aborted"
            run.error = "heartbeat stale — presumed crashed"
            run.finished_at = datetime.now(timezone.utc)
    db.commit()


def start_import(db: Session, source_path: str, policy: str = "hybrid",
                 *, dry_run: bool = False, runtime=None, classifier=None) -> dict:
    """Validate + kick off an import. Returns the ImportRun dict (dry_run adds
    a parse-only preview + cost estimate and creates NO run row).

    One import at a time: a live `running` row (fresh heartbeat) raises
    ConflictError; a stale one is reaped first.
    """
    from src.ingestion import formats

    if policy not in importer.POLICIES:
        raise ValidationError(
            f"unknown decay policy '{policy}' — one of {importer.POLICIES}")
    if not source_path or not os.path.exists(source_path):
        raise NotFoundError(f"import source not found: {source_path}")

    fmt = formats.detect_format(source_path)

    # Parse/slice to count work + estimate cost (also the dry-run payload).
    source, total_turns = importer._build_source(source_path, fmt)
    convs = list(source)
    total_convs = len(convs)
    est = importer.estimate_seconds(total_turns)
    preview = {
        "source_format": fmt, "policy": policy,
        "total_conversations": total_convs, "total_turns": total_turns,
        "estimate_seconds": round(est), "estimate_human": _human(est),
        "note": ("Background-model extraction runs per turn — keep the machine "
                 "on. Imported conversations arrive auto-scoped and non-private; "
                 "re-scope any of them afterward."),
    }
    logger.info("import_estimate", **{k: preview[k] for k in
                ("source_format", "policy", "total_conversations",
                 "total_turns", "estimate_seconds")})
    if dry_run:
        return {"dry_run": True, **preview}

    _reap_stale(db)
    if db.query(ImportRun).filter(ImportRun.status == "running").first():
        raise ConflictError(
            "an import is already running — wait for it or check import_status")

    run = ImportRun(
        source_path=source_path, source_format=fmt, policy=policy,
        kind="replay", status="running",
        total_conversations=total_convs, total_turns=total_turns)
    db.add(run)
    db.commit()
    run_id = run.id

    if runtime is not None:
        # Real path: hand off to the gpu-lane sliced job (yields to live chat).
        runtime.enqueue("import_replay", import_id=str(run_id), seq=0)
        logger.info("import_enqueued", import_id=str(run_id))
    else:
        # No runtime (tests / headless one-shot): replay inline, then finalize.
        report = importer.import_conversations(
            db, None, convs, policy, run_id=run_id, classifier=classifier)
        run = db.get(ImportRun, run_id)
        importer._bump_run(db, run_id, report)
        run.report = report
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        db.commit()

    return _run_dict(db.get(ImportRun, run_id))


def import_status(db: Session, import_id: Optional[str] = None) -> dict:
    """Latest import (or a specific one by id) as a status dict; a list of the
    most recent runs under ``recent``."""
    if import_id:
        try:
            run = db.get(ImportRun, uuid.UUID(import_id))
        except (ValueError, AttributeError):
            run = None
        if run is None:
            raise NotFoundError(f"no import run {import_id}")
        return _run_dict(run)
    runs = (db.query(ImportRun)
            .order_by(ImportRun.started_at.desc().nullslast()).limit(10).all())
    if not runs:
        return {"status": "none", "recent": []}
    return {**_run_dict(runs[0]), "recent": [_run_dict(r) for r in runs]}


def _human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"~{round(seconds / 60)} min"
    return f"~{seconds / 3600:.1f} h"
