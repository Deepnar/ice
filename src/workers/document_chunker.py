"""Document chunker (C2): big pasted inputs stop entering context whole.

When post_flight flags a turn `is_document` (>2000 words, <3 assistant
markers), this worker splits its raw_text with the shared chunker
(src/memory/chunking.py — the A1 sentence/code-aware greedy packer) and
stores each chunk with its own embedding in `episodic_chunks`. Retrieval then
works at chunk granularity: the vector leg searches chunk embeddings directly
(a whole-doc embedding is semantic mush — the C3 ceiling), and BM25-found doc
rows inject only their keyword-relevant chunks.

Two plain callables (the C5/C7 composability pattern):
  * run_chunk_turn(db, turn)      — direct call from post_flight per new doc;
  * run_pending_documents(db)     — the maintenance runtime's 2h catch-up that
    heals legacy documents ingested before C2 (and any dispatch that failed),
    a few per run.

Chunking runs for private (incognito) documents too — chunks are turn-local
and inherit visibility through the parent-turn join at query time, so nothing
leaks; the conversation needs its own chunks for self-retrieval.
"""

import uuid

from src.api.config import settings
import structlog

from src.memory.chunking import chunk_text
from src.memory.models import EpisodicChunk, EpisodicMemory

# Reuse the process's already-loaded embedder (no extra model copy, G13).
from src.workers.codex_extractor import embedder as shared_embedder

logger = structlog.get_logger("ice.workers.document_chunker")



def run_chunk_turn(db, turn) -> int:
    """Chunk one document turn. Idempotent: returns 0 if chunks already exist
    or the text doesn't actually split. Returns the number of chunks stored."""
    existing = db.query(EpisodicChunk).filter_by(turn_id=turn.id).count()
    if existing:
        return 0
    chunks = chunk_text(turn.raw_text or "")
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


def run_pending_documents(db, limit: int = settings.document_catchup_docs_per_run) -> int:
    """Catch-up callable: chunk turns that should have chunks but don't —
    is_document pastes (C2) and all long turns (C3, > ~settings.turn_long_turn_chunk_words,
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


