"""C12: the watch folder — drop a file in `ingest_inbox/`, it becomes memory.

The honest replacement for `workers/drop_zone.py`, which this deletes. What
was wrong with it, since the shape of the fix follows from the failure:

  * it was a `watchdog` observer in a `while True` loop with its own
    `main()` — a process nothing ever started. `./ice` launches uvicorn; C7
    deleted Celery and drop_zone was never moved into the runtime's job table.
    The folder sat there doing nothing.
  * it instantiated a SECOND `PyTorchClassifier`, doubling the encoder in
    VRAM during ingestion (that is G13, closed here).
  * it wrote 512-word slices straight into `rag_chunks` with its own inline
    token estimate — no codex, no clustering, no summaries, no scope. A
    searchable archive bolted onto the side, which is exactly what C12's entry
    says a document must not be.

Now it is an ordinary cadence job in the maintenance runtime: scan, hand each
file to the document service (which does the real ingestion through the
pipeline), move it to `processed/`. One file per run by default, because
ingestion is LLM-bound and the runtime's lane discipline should decide
throughput, not this loop.
"""

import os
import shutil

import structlog

logger = structlog.get_logger("ice.ingestion.documents.watch_folder")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
WATCH_DIR = os.path.join(REPO_ROOT, "ingest_inbox")
PROCESSED_DIR = os.path.join(WATCH_DIR, "processed")
FAILED_DIR = os.path.join(WATCH_DIR, "failed")
FILES_PER_RUN = 1


def _settled(path: str, min_age_seconds: float = 5.0) -> bool:
    """A file still being written must not be ingested half-formed. Age beats
    the old size-polling loop: a cadence job runs every few minutes anyway, so
    a file that stopped changing is simply a file whose mtime is not fresh."""
    import time
    try:
        return (time.time() - os.path.getmtime(path)) >= min_age_seconds
    except OSError:
        return False


def pending_files(watch_dir: str = WATCH_DIR) -> list:
    if not os.path.isdir(watch_dir):
        return []
    out = []
    for name in sorted(os.listdir(watch_dir)):
        path = os.path.join(watch_dir, name)
        if os.path.isfile(path) and not name.startswith(".") and _settled(path):
            out.append(path)
    return out


def run_watch_folder(db, conversation_id: str = None,
                     limit: int = FILES_PER_RUN) -> int:
    """Ingest up to `limit` dropped files. Returns how many were ingested.

    Dropped files have no conversation of their own to belong to, so they are
    ingested unattached: the document exists and is listed in the library, and
    the user enables it where they want it. That is the same opt-in rule
    uploads follow — the folder is not a back door into every chat.
    """
    from src.services import documents as documents_svc

    files = pending_files()[:limit]
    if not files:
        return 0

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)
    done = 0
    for path in files:
        try:
            result = documents_svc.add_document(
                db, conversation_id=conversation_id, path=path,
                origin="watch_folder", runtime=None)
            shutil.move(path, os.path.join(PROCESSED_DIR,
                                           os.path.basename(path)))
            done += 1
            logger.info("watch_folder_ingested", path=path,
                        document=result.get("id"))
        except Exception as exc:
            # Move it aside rather than retrying the same broken file every
            # cadence forever — a failing file must not block the queue.
            shutil.move(path, os.path.join(FAILED_DIR, os.path.basename(path)))
            logger.error("watch_folder_failed", path=path, error=str(exc))
    return done
