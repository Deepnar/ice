"""Batch Summarization Worker – coalesces decayed turns into high‑level summaries."""

import structlog

from src.api.db import SessionLocal
from src.memory.embedder import get_embedder
from src.memory.models import BatchSummary, EpisodicMemory
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name

logger = structlog.get_logger("ice.workers.batch_summarizer")
bg_client = get_bg_client()
# The process-shared native-width embedder (G13/G23).
embedder = get_embedder()

def batch_summarize():
    """Summarize old, decayed turns grouped by conversation. Plain callable
    since C7 — gating/retries live in the maintenance runtime."""
    db = SessionLocal()
    try:
        # Find conversations with old, non‑lossless turns that haven't been batch‑summarised
        stale_turns = db.query(EpisodicMemory).filter(
            EpisodicMemory.is_private == False,   # G16: incognito never summarised into shared stores
            EpisodicMemory.decay_score < 0.3,
            EpisodicMemory.lossless_flag == False,
            EpisodicMemory.is_document == False
        ).order_by(EpisodicMemory.conversation_id, EpisodicMemory.timestamp).all()

        # Group by conversation in batches of 50 turns
        conv_groups = {}
        for turn in stale_turns:
            conv_id = str(turn.conversation_id)
            conv_groups.setdefault(conv_id, []).append(turn)

        for conv_id, turns in conv_groups.items():
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
                db.add(BatchSummary(
                    conversation_id=batch[0].conversation_id,
                    start_turn_index=i,
                    end_turn_index=min(i+49, len(turns)-1),
                    summary_text=summary_text,
                    embedding=embedding
                ))
                db.commit()
                logger.info("batch_summary_created", conv_id=conv_id, turns=len(batch))

    except Exception as exc:
        db.rollback()
        logger.error("batch_summarization_failed", error=str(exc))
        raise
    finally:
        db.close()