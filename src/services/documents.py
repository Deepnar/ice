"""Document service (C12) — the E0 seam every document surface goes through.

REST (`/user-control/documents`), `ice_control` document actions, the CLI
(`scripts/ice_add_document.py`) and the watch folder are all thin adapters
over these five functions. The heavy lifting lives in
`src/ingestion/documents/`; this module owns the *registry* semantics:
de-duplication, the enable/disable toggles, the knowledge-promotion latch, and
the lifecycle row a long ingest reports progress into.

Two behaviors here are load-bearing and easy to get wrong:

* **Enabling is not `included_conversation_ids`.** That field is user-authored
  `manual` scope; documents must work identically under `auto`, where it is
  never read. Enablement lives in `document_links` and is applied by
  `scoping._apply_document_visibility`.
* **A link row is never deleted on disable.** `first_enabled_at` is the
  evidence the promotion latch reads — "has a second conversation ever reached
  for this document" cannot be answered by rows that get cleaned up.
"""

import hashlib
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from src.ingestion.documents import extract, ingest, parsers
from src.ingestion.documents import kind as kind_mod
from src.memory.models import Conversation, Document, DocumentLink
from src.services.errors import NotFoundError, ValidationError

logger = structlog.get_logger("ice.services.documents")

ORIGINS = ("upload", "paste", "watch_folder", "project")


def _doc_dict(db: Session, doc: Document,
              conversation_id: Optional[str] = None) -> dict:
    enabled = None
    if conversation_id:
        link = db.get(DocumentLink, (doc.id, uuid.UUID(str(conversation_id))))
        enabled = bool(link and link.enabled)
    return {
        "id": str(doc.id),
        "conversation_id": str(doc.conversation_id),
        "filename": doc.filename,
        "file_type": doc.file_type,
        "kind": doc.kind,
        "origin": doc.origin,
        "sha256": doc.sha256,
        "byte_size": doc.byte_size,
        "n_sections": doc.n_sections,
        "page_count": doc.page_count,
        "project_id": str(doc.project_id) if doc.project_id else None,
        "knowledge_shared": doc.knowledge_shared,
        "shared_at": doc.shared_at.isoformat() if doc.shared_at else None,
        "status": doc.status,
        "error": doc.error,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "enabled_here": enabled,
    }


def _require_conversation(db: Session, conversation_id: str) -> Conversation:
    try:
        conv = db.get(Conversation, uuid.UUID(str(conversation_id)))
    except (ValueError, AttributeError):
        conv = None
    if conv is None:
        raise NotFoundError(f"no conversation {conversation_id}")
    if conv.memory_scope_type == "none":
        # D5: `none` means "reads only itself" and C6 enforces that before
        # anything else. Carving an exception here would put a hole in the one
        # scope rule the user is entitled to trust. Precedent: E1 refuses
        # project attachment for incognito conversations for the same reason.
        raise ValidationError(
            "incognito conversations read only themselves — change this "
            "conversation's scope to use documents here")
    return conv


# ── ingestion ───────────────────────────────────────────────────────────────

def add_document(db: Session, *, conversation_id: Optional[str] = None,
                 path: Optional[str] = None, blob: Optional[str] = None,
                 filename: Optional[str] = None, doc_kind: Optional[str] = None,
                 origin: str = "upload", project_id: Optional[str] = None,
                 dry_run: bool = False, runtime=None, classifier=None,
                 embedder=None, llm=None, kind_llm=None) -> dict:
    """Register + ingest a document. Exactly one of `path` / `blob`.

    `doc_kind` ('document' | 'transcript') is the explicit user choice and
    always wins; when it is None the content is classified by
    `detect_blob_kind` — never by counting speaker labels (D6).

    `conversation_id` is optional: a file dropped in the watch folder belongs
    to no conversation, so it lands in the library enabled NOWHERE. That is
    the same opt-in rule uploads follow — a drop folder must not be a back
    door into every chat.
    """
    if (path is None) == (blob is None):
        raise ValidationError("pass exactly one of path or blob")
    if origin not in ORIGINS:
        raise ValidationError(f"unknown origin '{origin}' — one of {ORIGINS}")
    conv = _require_conversation(db, conversation_id) if conversation_id else None

    if path is not None:
        if not os.path.exists(path):
            raise NotFoundError(f"file not found: {path}")
        raw = parsers.read_bytes(path)
        filename = filename or os.path.basename(path)
        file_type = parsers.detect_file_type(path)
    else:
        raw = blob.encode("utf-8")
        filename = filename or "pasted text"
        file_type = "txt"

    sha = hashlib.sha256(raw).hexdigest()
    existing = db.query(Document).filter_by(sha256=sha).first()
    if existing is not None:
        # Same bytes = same document. Enable it here instead of ingesting a
        # second copy — re-uploading the spec into a new chat is the normal
        # way a user says "I want this here".
        logger.info("document_already_present", document=str(existing.id),
                    filename=filename)
        if conversation_id:
            set_document_enabled(db, str(existing.id), conversation_id, True)
        return {**_doc_dict(db, existing, conversation_id), "deduplicated": True}

    # The kind decides the destination, so it is settled before anything is
    # written. For a file this reads the extracted text, not the bytes.
    if file_type in ("csv", "xlsx", "pptx", "docx"):
        # Structured formats are documents by construction — a spreadsheet is
        # not a chat log, and asking a model would only add a way to be wrong.
        resolved_kind = kind_mod.DOCUMENT
        if doc_kind == kind_mod.TRANSCRIPT:
            raise ValidationError(
                f"a .{file_type} file cannot be ingested as a transcript")
    else:
        parse_for_kind = (extract.extract(path, file_type) if path
                          else parsers.parse_blob(blob, filename, file_type))
        sample = "\n\n".join(b.text for b in parse_for_kind.blocks)[:20_000]
        resolved_kind = kind_mod.detect_blob_kind(
            sample, explicit=doc_kind, llm=kind_llm)

    if resolved_kind == kind_mod.TRANSCRIPT:
        return _ingest_transcript(
            db, conv, raw_text=(blob if blob is not None
                                else raw.decode("utf-8", errors="replace")),
            filename=filename, sha=sha, origin=origin, project_id=project_id,
            path=path, dry_run=dry_run, runtime=runtime,
            classifier=classifier, embedder=embedder, llm=llm)

    parse = (extract.extract(path, file_type) if path
             else parsers.parse_blob(blob, filename, file_type))
    sections = ingest.build_sections(parse)
    if not sections:
        raise ValidationError("nothing to ingest — the document has no text")

    if dry_run:
        seconds = ingest.estimate_seconds(len(sections))
        return {"dry_run": True, "filename": filename, "file_type": file_type,
                "kind": resolved_kind, "n_sections": len(sections),
                "page_count": parse.page_count,
                "estimate_seconds": round(seconds),
                "estimate_human": _human(seconds),
                "note": ("Background-model extraction runs per section — keep "
                         "the machine on. The document is enabled only in this "
                         "conversation until you enable it elsewhere.")}

    doc = _create_document(
        db, conv, filename=filename, file_type=file_type, kind=resolved_kind,
        origin=origin, sha=sha, byte_size=len(raw), path=path,
        project_id=project_id, page_count=parse.page_count,
        n_sections=len(sections), source_text=blob)

    if runtime is not None:
        doc.status = "ingesting"
        db.commit()
        runtime.enqueue("ingest_document", document_id=str(doc.id))
        logger.info("document_enqueued", document=str(doc.id))
        return _doc_dict(db, doc, conversation_id)

    _run_ingest(db, doc, parse, runtime=None, classifier=classifier,
                embedder=embedder, llm=llm)
    return _doc_dict(db, doc, conversation_id)


def _ingest_transcript(db, conv, *, raw_text, filename, sha, origin,
                       project_id, path, dry_run, runtime, classifier,
                       embedder, llm) -> dict:
    """C12b: a text dump that IS a conversation goes through F14's amnesia
    slicer and lands as a ghost conversation with real user/assistant turns —
    not as document sections, which would flatten the exchange into prose."""
    from src.ingestion.importer import import_conversations
    from src.ingestion.raw_slicer import slice_raw_text

    normalized = slice_raw_text(raw_text, filename, llm=llm)
    if normalized is None or not normalized.turns:
        raise ValidationError("could not extract any turns from this transcript")
    if dry_run:
        seconds = ingest.estimate_seconds(len(normalized.turns))
        return {"dry_run": True, "filename": filename, "kind": "transcript",
                "n_sections": len(normalized.turns),
                "estimate_seconds": round(seconds),
                "estimate_human": _human(seconds)}

    doc = _create_document(
        db, conv, filename=filename, file_type="txt", kind="transcript",
        origin=origin, sha=sha, byte_size=len(raw_text.encode("utf-8")),
        path=path, project_id=project_id, page_count=None,
        n_sections=len(normalized.turns))
    doc.status = "ingesting"
    db.commit()

    try:
        # The ghost conversation IS the document's conversation, so scoping,
        # sharing and deletion treat a pasted chat log exactly like a PDF.
        report = import_conversations(
            db, runtime, [normalized], "fresh", classifier=classifier,
            embedder=embedder, llm=llm,
            conversation_id_override=doc.conversation_id)
        doc.n_sections = report.get("turns", 0)
        doc.status = "ready"
    except Exception as exc:
        db.rollback()
        doc = db.get(Document, doc.id)
        doc.status, doc.error = "failed", str(exc)[:1000]
        logger.error("transcript_ingest_failed", document=str(doc.id),
                     error=str(exc))
    doc.updated_at = datetime.now(timezone.utc)
    db.commit()
    return _doc_dict(db, doc, str(conv.id) if conv is not None else None)


def _create_document(db, conv, *, filename, file_type, kind, origin, sha,
                     byte_size, path, project_id, page_count,
                     n_sections, source_text=None) -> Document:
    """The document's own conversation + the registry row + the first link.

    The document conversation is `auto`-scoped, which sounds like it makes the
    document globally readable and does not: `kind <> 'chat'` is what scope
    resolution reads, and it makes the document opt-in everywhere regardless.
    `auto` here means only that the document itself has no scope opinion."""
    inherited_project = (uuid.UUID(str(project_id)) if project_id
                         else (conv.project_id if conv is not None else None))
    doc_conv = Conversation(
        id=uuid.uuid4(), kind=kind, memory_scope_type="auto",
        project_id=inherited_project)
    db.add(doc_conv)
    db.flush()          # FK parent before children (the C6 trap)

    doc = Document(
        id=uuid.uuid4(), conversation_id=doc_conv.id, filename=filename,
        file_type=file_type, kind=kind, origin=origin, sha256=sha,
        byte_size=byte_size, n_sections=n_sections, page_count=page_count,
        source_path=path, source_text=source_text,
        project_id=doc_conv.project_id, status="pending")
    db.add(doc)
    db.flush()
    if conv is not None:
        db.add(DocumentLink(document_id=doc.id, conversation_id=conv.id,
                            enabled=True))
    db.commit()
    logger.info("document_registered", document=str(doc.id), filename=filename,
                kind=kind, sections=n_sections,
                conversation=str(conv.id) if conv is not None else None)
    return doc


def _run_ingest(db, doc, parse, *, runtime, classifier, embedder, llm,
                deadline=None) -> dict:
    doc.status = "ingesting"
    doc.updated_at = datetime.now(timezone.utc)
    db.commit()
    try:
        report = ingest.ingest_document(
            db, doc, parse, runtime=runtime, classifier=classifier,
            embedder=embedder, llm=llm, deadline=deadline)
    except Exception as exc:
        db.rollback()
        doc = db.get(Document, doc.id)
        doc.status, doc.error = "failed", str(exc)[:1000]
        doc.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.error("document_ingest_failed", document=str(doc.id),
                     error=str(exc))
        raise
    doc = db.get(Document, doc.id)
    doc.n_sections = report["stored"]
    if not report["complete"]:
        doc.status = "ingesting"          # a slice ended; the job re-enqueues
    elif report["stored"] == 0:
        # "no silent failure": a document whose every section failed used to be
        # reported READY with n_sections = 0 — indistinguishable, to anything
        # downstream or to the user, from a document that genuinely held
        # nothing. It is a failure and it says so.
        doc.status = "failed"
        doc.error = (f"every section failed to ingest "
                     f"({report['failed']} of {report['failed']})")
        logger.error("document_ingest_empty", document=str(doc.id),
                     failed=report["failed"])
    else:
        doc.status = "ready"
        if report["failed"]:
            # Partial success is still success, but it must be visible.
            doc.error = (f"{report['failed']} of "
                         f"{report['failed'] + report['stored']} sections "
                         "failed to ingest")
            logger.warning("document_ingest_partial", document=str(doc.id),
                           stored=report["stored"], failed=report["failed"])
    doc.updated_at = datetime.now(timezone.utc)
    db.commit()
    return report


def run_document_ingest(db: Session, document_id: str) -> None:
    """The runtime job: re-read the source and replay. Kill-safe — per-section
    idempotency keys make a re-run skip what already landed.

    The source is a file OR the stored text. A blob has never had a file, so
    reading `source_path` unconditionally is what made every blob ingested
    through REST/MCP die as "source file is gone" — both adapters pass a
    runtime, so both take this path, and the test suite never did.
    """
    doc = db.get(Document, uuid.UUID(document_id))
    if doc is None:
        logger.error("document_missing", document_id=document_id)
        return
    if doc.status == "ready":
        return
    if doc.source_path and os.path.exists(doc.source_path):
        parse = extract.extract(doc.source_path, doc.file_type)
    elif doc.source_text:
        parse = parsers.parse_blob(doc.source_text, doc.filename, doc.file_type)
    else:
        doc.status, doc.error = "failed", (
            "source file is gone — re-upload it")
        db.commit()
        return
    from src.api.core import get_core
    core = get_core()
    _run_ingest(db, doc, parse, runtime=_get_runtime(),
                classifier=core.classifier if core is not None else None,
                embedder=None, llm=None)


def _get_runtime():
    from src.workers.runtime import get_runtime
    return get_runtime()


# ── registry surface ────────────────────────────────────────────────────────

def list_documents(db: Session, conversation_id: Optional[str] = None) -> dict:
    """Every document, newest first. With `conversation_id`, each carries
    `enabled_here` — that pair is the whole document-library UI."""
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return {"documents": [_doc_dict(db, d, conversation_id) for d in docs]}


def document_status(db: Session, document_id: str,
                    conversation_id: Optional[str] = None) -> dict:
    doc = _require_document(db, document_id)
    return _doc_dict(db, doc, conversation_id)


def set_document_enabled(db: Session, document_id: str, conversation_id: str,
                         enabled: bool) -> dict:
    """The live toggle (D3). Also the only place the promotion latch trips."""
    doc = _require_document(db, document_id)
    conv = _require_conversation(db, conversation_id)
    conv_uuid = conv.id

    link = db.get(DocumentLink, (doc.id, conv_uuid))
    if link is None:
        if not enabled:
            return _doc_dict(db, doc, conversation_id)   # already off
        link = DocumentLink(document_id=doc.id, conversation_id=conv_uuid,
                            enabled=True)
        db.add(link)
    else:
        link.enabled = enabled
        link.updated_at = datetime.now(timezone.utc)
    db.flush()

    # D4: a SECOND distinct conversation reaching for this document is the
    # signal that it is real working knowledge rather than a one-off dump, so
    # the facts extracted from it join the graph for good. The document's TEXT
    # stays opt-in forever — that is what keeps a 300-page manual out of
    # chats that never asked for it.
    if enabled and not doc.knowledge_shared:
        reach = db.execute(sql_text(
            "SELECT count(*) FROM document_links WHERE document_id = :d"),
            {"d": doc.id}).scalar()
        if reach and reach >= 2:
            doc.knowledge_shared = True
            doc.shared_at = datetime.now(timezone.utc)
            logger.info("document_knowledge_promoted", document=str(doc.id),
                        filename=doc.filename, conversations=reach)
    db.commit()
    logger.info("document_toggled", document=str(doc.id),
                conversation=str(conv_uuid), enabled=enabled)
    return _doc_dict(db, doc, conversation_id)


def delete_document(db: Session, document_id: str, dry_run: bool = False) -> dict:
    """Delete a document = delete its conversation, through C10's one cascade.

    Deliberately not a second deletion path: C10 already expires sole-support
    codex edges, prunes procedural patterns, stales review items and writes the
    manifest. A document is a conversation, so it gets all of that for free —
    which is also how "un-sharing" a promoted document works."""
    from src.services.conversations import delete_conversation

    doc = _require_document(db, document_id)
    conv_id = str(doc.conversation_id)
    manifest = delete_conversation(db, conv_id, dry_run=dry_run)
    if not dry_run:
        # The registry row cascades from the conversation; say so plainly
        # rather than letting the caller wonder.
        manifest["document"] = {"id": str(doc.id), "filename": doc.filename}
    return manifest


def _require_document(db: Session, document_id: str) -> Document:
    try:
        doc = db.get(Document, uuid.UUID(str(document_id)))
    except (ValueError, AttributeError):
        doc = None
    if doc is None:
        raise NotFoundError(f"no document {document_id}")
    return doc


def _human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"~{round(seconds / 60)} min"
    return f"~{seconds / 3600:.1f} h"
