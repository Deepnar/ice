"""C12: a document LIVES THROUGH the pipeline, it is not filed next to it.

The shape, and why it is this shape: a document becomes **its own
conversation** whose turns are its sections. That single decision is what makes
the rest free — C6 scoping and exclusion, C4's per-conversation summary (which
for a document conversation simply *is* the document summary, the top of C3's
raw → summary → abstract hierarchy), C2/C3 chunk-level retrieval, C5
clustering, and the codex/procedural extraction chain all already operate on
conversations and turns. No new retrieval leg, no parallel store.

The loop below is deliberately the sibling of `importer.import_conversations`
(F10): store a turn with its real timestamp, call the real `evaluate_turn`
chain, cluster the finished conversation, then summarise. If you find yourself
adding a step here that the importer does not have, check whether it belongs in
the shared pipeline instead.
"""

import hashlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from src.api.config import settings
from src.ingestion.documents import parsers
from src.memory.chunking import chunk_text
from src.memory.models import Conversation, EpisodicMemory
from src.memory.session import resolve_session_id

logger = structlog.get_logger("ice.ingestion.documents.ingest")

# One document = one sitting. Sections are stamped a second apart so their
# order is stable and `resolve_session_id` keeps them in a single session,
# which is what makes C5's session affinity cluster a document's sections
# together instead of scattering them.
SECTION_SECONDS = 1
SECONDS_PER_SECTION = 6.0        # D5 cost estimate, same constant as F10


class Section:
    __slots__ = ("text", "meta", "index")

    def __init__(self, text: str, meta: dict, index: int):
        self.text, self.meta, self.index = text, meta, index


def build_sections(parse: parsers.DocumentParse) -> list:
    """Blocks → sections, using THE SHARED CHUNKER (C2's sentence/code-aware
    greedy packer, which already makes markdown headings hard boundaries — C3).

    A block is never merged with its neighbour, so a section never straddles a
    page, slide, heading or sheet: the provenance header on each section is
    therefore always true, which is the whole point of carrying one.
    """
    sections: list = []
    for block in parse.blocks:
        pieces = chunk_text(block.text)
        for piece in pieces:
            if piece.strip():
                sections.append(Section(piece, block.meta, len(sections)))
    return sections


def section_header(title: str, meta: dict, index: int, total: int) -> str:
    """The one-line provenance stamp prefixed to a section's stored text.

    It rides INSIDE raw_text on purpose: BM25 then finds "report.pdf" (people
    do ask "what did the spec say"), the prompt assembler renders it with no
    change (it falls through cleanly for text that is not a "User: …/Assistant:
    …" exchange), and `reembed._episodic_source` reproduces it deterministically
    on a re-embed.
    """
    label = parsers.block_label(meta)
    where = f" · {label}" if label else ""
    return f"[{title}{where} · section {index + 1}/{total}]"


def _section_idempotency_key(doc_id, index: int, text: str) -> str:
    body = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return hashlib.sha256(
        f"ice-document:{doc_id}:{index}:{body}".encode("utf-8")).hexdigest()


def ingest_document(db, doc, parse: parsers.DocumentParse, *,
                    runtime=None, classifier=None, embedder=None, llm=None,
                    deadline: Optional[float] = None,
                    run_summary: bool = True) -> dict:
    """Replay one parsed document into its own conversation.

    Idempotent and resumable at section granularity: a re-run skips sections
    whose key already exists, so a crash mid-document costs only the sections
    that had not committed.
    """
    from src.workers.post_flight import evaluate_turn

    sections = build_sections(parse)
    report = {"sections": 0, "stored": 0, "failed": 0, "complete": True}
    if not sections:
        report["complete"] = False
        return report

    conv_id = doc.conversation_id
    if db.get(Conversation, conv_id) is None:
        raise RuntimeError(f"document conversation {conv_id} does not exist")

    base_ts = doc.created_at or datetime.now(timezone.utc)
    if base_ts.tzinfo is None:
        base_ts = base_ts.replace(tzinfo=timezone.utc)
    total = len(sections)

    for section in sections:
        if deadline is not None and time.monotonic() > deadline:
            report["complete"] = False
            break
        if runtime is not None:
            # Yield to the user exactly as the importer does — never replay
            # into the pipeline while a live chat stream is generating.
            while runtime.generation_in_flight:
                time.sleep(0.5)

        header = section_header(parse.title, section.meta, section.index, total)
        body = f"{header}\n{section.text}"
        ts = base_ts + timedelta(seconds=SECTION_SECONDS * section.index)
        try:
            stored = _store_section(db, doc, conv_id, body, ts, section.index,
                                    classifier, embedder)
            if stored is not None:
                batch_id = stored
                evaluate_turn(batch_id=str(batch_id), prompt=body, response="",
                              conversation_id=str(conv_id), model_used="",
                              source_kind="document",
                              source_title=parse.title)
                report["stored"] += 1
            report["sections"] += 1
        except Exception as exc:        # one bad section must not kill the doc
            db.rollback()
            report["failed"] += 1
            logger.error("document_section_failed", document=str(doc.id),
                         index=section.index, error=str(exc))

    try:
        from src.workers.clustering import run_cluster_assignment
        run_cluster_assignment(db, conversation_ids=[str(conv_id)])
    except Exception as exc:
        logger.warning("document_clustering_failed", document=str(doc.id),
                       error=str(exc))

    if run_summary and report["stored"]:
        # The document-level summary. C4's worker writes one evolving summary
        # per conversation — for a document conversation that row IS the
        # document summary, and the cross-conversation half of
        # `_batch_summary_lookup` can then surface the document by its gist.
        try:
            from src.workers.conversation_summary import run_conversation_summaries
            run_conversation_summaries(db, llm=llm, embedder=embedder,
                                       conversation_ids=[str(conv_id)])
        except Exception as exc:
            logger.warning("document_summary_failed", document=str(doc.id),
                           error=str(exc))

    logger.info("document_ingested", document=str(doc.id), title=parse.title,
                sections=report["sections"], stored=report["stored"],
                failed=report["failed"], complete=report["complete"])
    return report


def _store_section(db, doc, conv_id, body: str, ts: datetime, index: int,
                   classifier, embedder):
    """One section → one episodic turn. Returns its batch_id, or None if the
    section is already stored (per-section idempotency).

    ⚠ `is_document` is deliberately NOT set (trap 1 in the spec). The vector
    leg EXCLUDES `is_document` rows on the assumption that chunks will be
    searched instead — but a section is already chunk-sized, so the chunker
    produces none for it and the row would be represented nowhere at all.
    Sections are ordinary turns; what marks them as document content is the
    KIND of the conversation holding them, which is what scope resolution
    reads.
    """
    idem = _section_idempotency_key(doc.id, index, body)
    if db.query(EpisodicMemory.id).filter_by(idempotency_key=idem).first():
        return None

    topic_tags: list = []
    intent_tags: list = []
    context_reliance = "Zero_Shot"
    if classifier is not None:
        try:
            result = classifier.classify(body, conversation_id=str(conv_id))
            topic_tags = list(result.topic_tags or [])
            intent_tags = list(result.intent_tags or [])
            context_reliance = result.context_reliance or "Zero_Shot"
        except Exception as exc:
            logger.warning("document_classify_failed", error=str(exc))

    enc = embedder if embedder is not None else _lazy_embedder()
    vec = enc.encode(body, convert_to_tensor=False)
    embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

    session_id, _started, _gap = resolve_session_id(
        db, conv_id, ts, settings.session_gap_minutes)

    batch_id = uuid.uuid4()
    db.add(EpisodicMemory(
        conversation_id=conv_id, batch_id=batch_id, session_id=session_id,
        is_private=False, timestamp=ts, ts_provenance="document_ingest",
        topic_tags=topic_tags, intent_tags=intent_tags,
        context_reliance=context_reliance, raw_text=body,
        embedding=embedding, idempotency_key=idem))
    db.commit()
    return batch_id


def _lazy_embedder():
    from src.memory.embedder import get_embedder
    return get_embedder()


def estimate_seconds(n_sections: int) -> float:
    return n_sections * SECONDS_PER_SECTION
