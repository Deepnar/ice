"""Batch Summarization Worker – coalesces old turns into high‑level summaries."""

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import or_

from src.api.db import SessionLocal
from src.memory.embedder import get_embedder
from src.memory.models import BatchSummary, EpisodicMemory
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name

logger = structlog.get_logger("ice.workers.batch_summarizer")
bg_client = get_bg_client()
# The process-shared native-width embedder (G13/G23).
embedder = get_embedder()

# G11: a turn this old compresses regardless of decay score. Decay alone left
# old-but-accessed turns in long conversations uncompressed forever, and long
# conversations are the case compression exists for. A module constant for now,
# like the other worker tuning constants; G9 sweeps these into settings.
BATCH_SUMMARY_AGE_DAYS = 30

def batch_summarize():
    """Compress old turns into per-conversation batch summaries. Plain callable
    since C7 — gating/retries live in the maintenance runtime.

    G11 (2026-08-08) changed two things here, and the second was the reason the
    first could not simply be widened.

    **Coverage is now recorded.** Turns carry `batch_summary_id`; NULL means
    "not yet summarised" and is the selection predicate. Until this column
    existed there was no way to express that — the docstring claimed this worker
    found turns "that haven't been batch-summarised" while *nothing* excluded
    them, so every cadence pass (7200 s) re-selected the same turns, re-ran the
    LLM over them, and appended another `BatchSummary` row for the retrieval leg
    to inject as duplicates. `start_turn_index`/`end_turn_index` could not serve
    as the record: they are positions in whatever filtered, decay-ordered list
    the producing run built, and that list shifts between runs.

    **Selection is age OR decay, not decay alone.** Only decayed turns were
    compressing, so a very old but frequently-accessed turn in a long
    conversation never did — and long conversations are exactly where compression
    matters. A turn now qualifies on `decay_score < 0.3` **or** simply being
    older than `BATCH_SUMMARY_AGE_DAYS`, whichever comes first.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=BATCH_SUMMARY_AGE_DAYS)
        stale_turns = db.query(EpisodicMemory).filter(
            EpisodicMemory.is_private == False,   # G16: incognito never summarised into shared stores
            EpisodicMemory.batch_summary_id.is_(None),   # G11: not already covered
            or_(EpisodicMemory.decay_score < 0.3,
                EpisodicMemory.timestamp < cutoff),      # G11: age OR decay
            EpisodicMemory.lossless_flag == False,
            EpisodicMemory.is_document == False
        ).order_by(EpisodicMemory.conversation_id, EpisodicMemory.timestamp).all()

        # Group by conversation in batches of 50 turns
        conv_groups = {}
        for turn in stale_turns:
            conv_id = str(turn.conversation_id)
            conv_groups.setdefault(conv_id, []).append(turn)

        for conv_id, turns in conv_groups.items():
            # G4(a): one conversation's summary per iteration, each committed
            # as it completes, so standing down here loses nothing already done.
            from src.workers.runtime import yield_if_user_active
            yield_if_user_active("batch_summarize.conversation")
            for i in range(0, len(turns), 50):
                batch = turns[i:i+50]
                if len(batch) < 5:
                    continue  # skip tiny batches
                # Assemble the raw text
                combined = "\n\n".join(t.raw_text for t in batch)
                prompt = (
                    "Summarise the following conversation excerpt in 2‑3 paragraphs. "
                    "Preserve all names, numbers, decisions, and specific facts. "
                    "Output only the summary."
                )
                completion = bg_client.chat.completions.create(
                    model=get_bg_model_name(),
                    messages=[
                        {"role": "system", "content": "You are a concise summarisation engine."},
                        {"role": "user", "content": f"{prompt}\n\n{combined}"}
                    ],
                    temperature=0.0,
                    max_tokens=500,
                    # prefill-heavy (up to 50 turns in the prompt): keep the
                    # old 60s floor — G12's formula scales with output only.
                    timeout=max(60.0, bg_timeout(500))
                )
                summary_text = completion.choices[0].message.content.strip()
                if not summary_text:
                    continue

                # Store with embedding
                embedding = embedder.encode(summary_text, convert_to_tensor=False).tolist()
                summary = BatchSummary(
                    conversation_id=batch[0].conversation_id,
                    # ⚠ Write-only legacy (see models.py): positions in THIS
                    # run's filtered list, not turn identity. Coverage is the
                    # `batch_summary_id` stamp below.
                    start_turn_index=i,
                    end_turn_index=min(i+49, len(turns)-1),
                    summary_text=summary_text,
                    embedding=embedding
                )
                db.add(summary)
                db.flush()          # need the id before stamping the turns
                # G11: close the loop — mark exactly the turns this summary
                # covers, in the SAME transaction as the summary. Committing the
                # summary without the stamps would recreate the duplicate-work
                # bug on the next pass, so these cannot be separated.
                for t in batch:
                    t.batch_summary_id = summary.id
                db.commit()
                logger.info("batch_summary_created", conv_id=conv_id,
                            turns=len(batch), summary_id=str(summary.id))

    except Exception as exc:
        db.rollback()
        logger.error("batch_summarization_failed", error=str(exc))
        raise
    finally:
        db.close()