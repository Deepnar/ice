"""Document chunker (C2): big pasted inputs stop entering context whole.

When post_flight flags a turn `is_document` (>2000 words, <3 assistant
markers), this worker splits its raw_text with the shared chunker
(src/memory/chunking.py — the A1 sentence/code-aware greedy packer) and
stores each chunk with its own embedding in `episodic_chunks`. Retrieval then
works at chunk granularity: the vector leg searches chunk embeddings directly
(a whole-doc embedding is semantic mush — the C3 ceiling), and BM25-found doc
rows inject only their keyword-relevant chunks.

Two entry points, both thin wrappers over plain callables (the C5/C7
composability pattern):
  * chunk_document(batch_id)   — event-driven from post_flight per new doc;
  * chunk_pending_documents()  — beat catch-up that heals legacy documents
    ingested before C2 (and any dispatch that failed), a few per run.

Chunking runs for private (incognito) documents too — chunks are turn-local
and inherit visibility through the parent-turn join at query time, so nothing
leaks; the conversation needs its own chunks for self-retrieval.
"""

import uuid
import structlog

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, EpisodicChunk
from src.memory.chunking import chunk_text, CHUNK_TOKENS, OVERLAP_WORDS
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy, is_user_active
# Reuse the worker process's already-loaded embedder (no extra model copy, G13).
from src.workers.codex_extractor import embedder as shared_embedder

logger = structlog.get_logger("ice.workers.document_chunker")

CATCHUP_DOCS_PER_RUN = 5


def run_chunk_turn(db, turn) -> int:
    """Chunk one document turn. Idempotent: returns 0 if chunks already exist
    or the text doesn't actually split. Returns the number of chunks stored."""
    existing = db.query(EpisodicChunk).filter_by(turn_id=turn.id).count()
    if existing:
        return 0
    chunks = chunk_text(turn.raw_text or "", max_tokens=CHUNK_TOKENS,
                        overlap_words=OVERLAP_WORDS)
    if len(chunks) <= 1:
        return 0
    for i, chunk in enumerate(chunks):
        db.add(EpisodicChunk(
            id=uuid.uuid4(),
            turn_id=turn.id,
            chunk_index=i,
            chunk_text=chunk,
            embedding=shared_embedder.encode(chunk, convert_to_tensor=False).tolist(),
        ))
    db.commit()
    logger.info("document_chunked", turn_id=str(turn.id), chunks=len(chunks))
    return len(chunks)


def run_pending_documents(db, limit: int = CATCHUP_DOCS_PER_RUN) -> int:
    """Catch-up callable: chunk turns that should have chunks but don't —
    is_document pastes (C2) and all long turns (C3, > ~LONG_TURN_CHUNK_WORDS,
    approximated in SQL by char length since word count isn't stored). Heals
    legacy pre-C2/C3 turns and lost dispatches."""
    from sqlalchemy import func, or_
    pending = (
        db.query(EpisodicMemory)
        .filter(
            or_(
                EpisodicMemory.is_document == True,  # noqa: E712
                func.length(EpisodicMemory.raw_text) > 4200,  # ≈600+ words
            ),
            ~EpisodicMemory.id.in_(
                db.query(EpisodicChunk.turn_id).distinct()
            ),
        )
        .order_by(EpisodicMemory.timestamp.desc())
        .limit(limit)
        .all()
    )
    done = 0
    for turn in pending:
        done += 1 if run_chunk_turn(db, turn) else 0
    return done


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def chunk_document(self, batch_id: str):
    """Event-driven from post_flight when a turn is flagged is_document."""
    if is_gpu_busy():
        raise self.retry(countdown=30)
    db = SessionLocal()
    try:
        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn:
            raise self.retry(countdown=10)
        run_chunk_turn(db, turn)
    except Exception as exc:
        db.rollback()
        logger.error("document_chunking_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


@app.task(bind=True, max_retries=1, default_retry_delay=120)
def chunk_pending_documents(self):
    """Beat catch-up: heal unchunked documents a few at a time."""
    if is_gpu_busy():
        raise self.retry(countdown=120)
    if settings.background_model_mode == "shared" and is_user_active():
        raise self.retry(countdown=60)
    db = SessionLocal()
    try:
        n = run_pending_documents(db)
        if n:
            logger.info("pending_documents_chunked", count=n)
    except Exception as exc:
        db.rollback()
        logger.error("pending_document_chunking_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
