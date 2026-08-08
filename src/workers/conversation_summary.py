"""C4 — one evolving summary per conversation ("the whole conversation so
far, current" — never a batch_summaries range row).

Runs in the session-end burst (quartet member since C4) and on its cadence;
both are cheap: only conversations with turns newer than their summary's
``covers_through`` do any work, and a summary row is first created only once
the conversation outgrows the sliding window (the D3a condition — a 2-turn
chat never earns one). Maintenance is incremental: prompt = existing summary
+ the NEW turns' C1 representations, folded in bounded chunks, grounded
C1-style (must-keep terms from the chunk, one retry on coverage miss).
A failed LLM call leaves the old row untouched — the next burst retries.

Consumers: the assembler (active conversation past the window condition,
``=== CONVERSATION SUMMARY ===``) and the batch-summary retrieval leg
(cross-conversation hits; excludes private conversations and the active one).
Incognito conversations DO get summaries — their own context; the retrieval
consumer's join is the privacy shield. T-track era digests read these rows
as-is later.
"""
from datetime import datetime, timezone

from src.api.config import settings
import structlog

from src.api.memory_decision import estimate_recent_window_tokens
from src.memory.models import ConversationSummary, EpisodicMemory
from src.memory.tokens import estimate_from_chars
from src.workers.turn_density import (
    extract_key_terms,
    must_terms,
    summary_coverage,
)

logger = structlog.get_logger("ice.workers.conversation_summary")



def _representation(turn) -> str:
    """C1 read-side idiom (same preference as the assembler's recent window):
    raw when flagged inject_raw, else the grounded summary, else a raw head —
    capped so documents/monsters can't blow the prompt."""
    if turn.inject_raw and turn.raw_text:
        text_ = turn.raw_text
    elif turn.summary_text:
        text_ = turn.summary_text
    else:
        text_ = (turn.raw_text or "")[:300]
    words = text_.split()
    if len(words) > settings.conversation_summary_per_turn_words:
        text_ = " ".join(words[:settings.conversation_summary_per_turn_words]) + "…"
    return text_


def _default_llm(prompt: str, max_tokens: int = 400) -> str:
    from src.workers.bg_client_factory import (
        bg_timeout,
        get_bg_client,
        get_bg_model_name,
    )
    completion = get_bg_client().chat.completions.create(
        model=get_bg_model_name(),
        messages=[
            {"role": "system", "content": "You are a precise summarisation engine."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        # prefill-heavy (existing summary + a turn chunk): keep the 60s floor,
        # G12's formula scales with output only.
        timeout=max(60.0, bg_timeout(max_tokens)),
    )
    return (completion.choices[0].message.content or "").strip()


def _shared_embedder():
    # Lazy: codex_extractor loads the SentenceTransformer at import (G13 —
    # one load per process; never instantiate another).
    from src.workers.codex_extractor import embedder
    return embedder


def _fold_prompt(existing: str, chunk_text: str, terms: list,
                 missing: list = None) -> str:
    must_block = ""
    if terms:
        must_block = (
            "\nMUST-PRESERVE TERMS (every one of these must appear verbatim "
            f"in your summary): {', '.join(terms)}\n")
    retry_block = ""
    if missing:
        retry_block = (
            "\nYour previous summary DROPPED these required terms — include "
            f"each of them verbatim this time: {', '.join(missing)}\n")
    existing_block = (
        f"EXISTING SUMMARY OF THE CONVERSATION SO FAR:\n{existing}\n\n"
        if existing else "")
    return (
        "You maintain ONE evolving summary of a whole conversation. Revise "
        "the existing summary to also cover the new turns below — do not "
        "drop information the existing summary carries unless the new turns "
        "supersede it (then state what changed). Preserve names, numbers, "
        "decisions, and open questions. Output ONLY the revised summary, "
        f"at most {settings.conversation_summary_max_words} words."
        f"{must_block}{retry_block}\n\n"
        f"{existing_block}NEW TURNS:\n{chunk_text}"
    )


def _summarize_chunk(existing: str, chunk_text: str, llm, embedder) -> str:
    """One grounded fold: must-keep terms from the chunk, one retry on a
    coverage miss (post_flight idiom). Returns "" on an empty/failed call."""
    key_terms = extract_key_terms(chunk_text, embedder)
    terms = must_terms(key_terms)
    summary = llm(_fold_prompt(existing, chunk_text, terms), max_tokens=400)
    if not summary:
        return ""
    coverage = summary_coverage(summary, key_terms)
    if coverage < settings.turn_summary_coverage_threshold and terms:
        low = summary.lower()
        missing = [t for t in terms if t.lower() not in low]
        retry = llm(_fold_prompt(existing, chunk_text, terms, missing),
                    max_tokens=400)
        if retry and summary_coverage(retry, key_terms) > coverage:
            summary = retry
    words = summary.split()
    if len(words) > settings.conversation_summary_max_words + 50:      # tolerance, then hard cap
        summary = " ".join(words[:settings.conversation_summary_max_words])
    return summary


def run_conversation_summaries(db, llm=None, embedder=None,
                               conversation_ids=None) -> dict:
    """One pass: for every conversation with turns newer than its summary's
    covers_through, fold the new turns into the evolving summary. Row
    creation is gated on the D3a window condition (with the legacy default
    budget — the job doesn't know the routed model; the assembler re-checks
    at injection). *llm*/*embedder* are injectable and *conversation_ids*
    restricts the pass (tests process only fixture conversations — the full
    scan is the burst/cadence behavior)."""
    stats = {"scanned": 0, "created": 0, "updated": 0,
             "below_window": 0, "failed": 0}
    if llm is None:
        llm = _default_llm
    lazy_embedder = embedder

    from sqlalchemy import text as sql_text
    rows = db.execute(sql_text("""
        SELECT e.conversation_id AS cid, count(*) AS n_turns,
               max(e.timestamp) AS last_ts,
               coalesce(sum(length(e.raw_text)), 0) AS chars
        FROM episodic_memory e
        GROUP BY e.conversation_id
    """)).fetchall()
    if conversation_ids is not None:
        wanted = {str(c) for c in conversation_ids}
        rows = [r for r in rows if str(r.cid) in wanted]
    summaries = {s.conversation_id: s
                 for s in db.query(ConversationSummary).all()}

    for row in rows:
        # G4(a): per-conversation boundary, and `covers_through` means a
        # conversation already folded is not redone on the requeued run.
        from src.workers.runtime import yield_if_user_active
        yield_if_user_active("conversation_summary.conversation")
        stats["scanned"] += 1
        existing_row = summaries.get(row.cid)
        if existing_row is not None and existing_row.covers_through is not None \
                and row.last_ts is not None \
                and row.last_ts <= existing_row.covers_through:
            continue                                  # nothing new
        if existing_row is None:
            total_tokens = estimate_from_chars(row.chars)
            if total_tokens <= estimate_recent_window_tokens(row.n_turns):
                stats["below_window"] += 1            # still fits the window
                continue

        q = db.query(EpisodicMemory).filter(
            EpisodicMemory.conversation_id == row.cid)
        if existing_row is not None and existing_row.covers_through is not None:
            q = q.filter(EpisodicMemory.timestamp > existing_row.covers_through)
        new_turns = q.order_by(EpisodicMemory.timestamp.asc()).all()
        if not new_turns:
            continue

        if lazy_embedder is None:
            lazy_embedder = _shared_embedder()

        # Fold in bounded chunks; any failed call aborts THIS conversation
        # with the old row intact (retry next burst — never half-advance).
        summary = existing_row.summary_text if existing_row else ""
        chunk, chunk_words, ok = [], 0, True
        for turn in new_turns:
            rep = _representation(turn)
            chunk.append(rep)
            chunk_words += len(rep.split())
            if chunk_words >= settings.conversation_summary_chunk_words:
                summary = _summarize_chunk(summary, "\n\n".join(chunk),
                                           llm, lazy_embedder)
                if not summary:
                    ok = False
                    break
                chunk, chunk_words = [], 0
        if ok and chunk:
            summary = _summarize_chunk(summary, "\n\n".join(chunk),
                                       llm, lazy_embedder)
            ok = bool(summary)
        if not ok:
            stats["failed"] += 1
            logger.warning("conversation_summary_failed",
                           conversation_id=str(row.cid))
            db.rollback()
            continue

        vec = lazy_embedder.encode(summary, convert_to_tensor=False)
        embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        now = datetime.now(timezone.utc)
        if existing_row is None:
            db.add(ConversationSummary(
                conversation_id=row.cid, summary_text=summary,
                covers_through=new_turns[-1].timestamp,
                covers_turns=row.n_turns, embedding=embedding, updated_at=now))
            stats["created"] += 1
        else:
            existing_row.summary_text = summary
            existing_row.covers_through = new_turns[-1].timestamp
            existing_row.covers_turns = row.n_turns
            existing_row.embedding = embedding
            existing_row.updated_at = now
            stats["updated"] += 1
        db.commit()

    if stats["created"] or stats["updated"] or stats["failed"]:
        logger.info("conversation_summaries", **stats)
    return stats
