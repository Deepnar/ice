"""C4 — one evolving summary per conversation ("the whole conversation so
far, current" — never a batch_summaries range row).

Runs in the session-end burst (quartet member since C4) and on its cadence;
unchanged source snapshots do no generation work. A summary row is first created
once the conversation outgrows the sliding window (the D3a condition — a 2-turn
chat never earns one). Maintenance is incremental: prompt = existing summary
+ the NEW turns' C1 representations, folded in bounded chunks, grounded
C1-style (must-keep terms from the chunk, one retry on coverage miss).
Edited/deleted sources and older imports rebuild from surviving representations;
strictly newer additions can extend a valid checkpoint. A failed LLM call leaves
the old row untouched — the next burst retries; stale snapshots are not injected.

Consumers: the assembler (active conversation past the window condition,
``=== CONVERSATION SUMMARY ===``) and the batch-summary retrieval leg
(cross-conversation hits; excludes private conversations and the active one).
Incognito conversations DO get summaries — their own context; the retrieval
consumer's join is the privacy shield. T-track era digests read these rows
as-is later.
"""
from datetime import datetime, timezone

import structlog

from src.api.config import settings
from src.api.memory_decision import estimate_recent_window_tokens
from src.memory.models import ConversationSummary, EpisodicMemory
from src.memory.representation import choose_representation
from src.memory.summary_snapshot import bind_snapshot, snapshot_matches, source_snapshot
from src.memory.time_format import recorded_stamp
from src.memory.tokens import estimate_from_chars
from src.workers.completion_text import complete_text
from src.workers.turn_density import (
    extract_key_terms,
    must_terms,
    retry_on_coverage_miss,
)

logger = structlog.get_logger("ice.workers.conversation_summary")


def _representation(turn) -> str:
    """Complete eligible evidence; neither unchecked summaries nor source heads."""
    selected = choose_representation(turn)[0]
    if not selected:
        return ""
    return f"{recorded_stamp(turn.timestamp, turn.ts_provenance)} {selected}"


def _source_chunks(turns):
    """Pack whole turn representations; the request boundary checks hard capacity."""
    chunk, size = [], 0
    target = settings.conversation_summary_chunk_words
    for turn in turns:
        representation = _representation(turn)
        if not representation:
            continue
        words = len(representation.split())
        if chunk and size + words > target:
            yield "\n\n".join(chunk)
            chunk, size = [], 0
        chunk.append(representation)
        size += words
    if chunk:
        yield "\n\n".join(chunk)


def _default_llm(prompt: str, max_tokens: int = 400) -> str:
    from src.workers.bg_client_factory import (
        bg_timeout,
        get_bg_client,
        get_bg_model_name,
    )
    from src.memory.tokens import count_messages, with_margin
    from src.model_registry.registry import get_model_context_window
    from src.model_registry.runtime_probe import serving_window
    from src.workers.completion_text import IncompleteCompletion

    model = get_bg_model_name()
    messages = [
        {"role": "system", "content": "You are a precise summarisation engine."},
        {"role": "user", "content": prompt},
    ]
    window = int(settings.ollama_num_ctx_max)
    if settings.background_model_mode == "shared":
        observed = serving_window(model, get_model_context_window(model))
        if observed:
            window = min(window, int(observed)) if window > 0 else int(observed)
    required = with_margin(count_messages(messages), settings.token_count_safety_margin) + max_tokens
    if window <= 0 or required > window:
        logger.warning("conversation_summary_input_over_budget", required=required,
                       window=window, model=model,
                       reason="complete evidence cannot fit; retaining previous checkpoint")
        raise IncompleteCompletion("complete conversation-summary input exceeds known capacity")
    completion = get_bg_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        # prefill-heavy (existing summary + a turn chunk): keep the 60s floor,
        # G12's formula scales with output only.
        timeout=max(60.0, bg_timeout(max_tokens)),
    )
    return complete_text(completion)


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
    """Coverage-guided generation, not a semantic-support verdict.

    Empty custom output returns an empty string; provider failures propagate.
    """
    key_terms = extract_key_terms(chunk_text, embedder)
    terms = must_terms(key_terms)
    summary = llm(_fold_prompt(existing, chunk_text, terms), max_tokens=400)
    if not summary:
        return ""
    # G29: shared with post_flight — this copy used to adopt a better retry
    # without updating its coverage.
    summary, _coverage = retry_on_coverage_miss(
        summary, key_terms,
        lambda missing: llm(_fold_prompt(existing, chunk_text, terms, missing),
                            max_tokens=400))
    return summary


def run_conversation_summaries(db, llm=None, embedder=None,
                               conversation_ids=None) -> dict:
    """Refresh changed source snapshots, incrementing only strictly newer turns. Row
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
        # Yield between conversations; source identity detects unchanged work
        # as well as edits/deletions/backfills that timestamp cursors missed.
        from src.workers.runtime import yield_if_user_active
        yield_if_user_active("conversation_summary.conversation")
        stats["scanned"] += 1
        existing_row = summaries.get(row.cid)
        sources = source_snapshot(db, row.cid)
        previous_valid = existing_row is not None and snapshot_matches(
            existing_row.source_manifest, existing_row.summary_text, sources,
            allow_newer=True)
        if previous_valid and snapshot_matches(
                existing_row.source_manifest, existing_row.summary_text, sources):
            continue
        if existing_row is None:
            total_tokens = estimate_from_chars(row.chars)
            if total_tokens <= estimate_recent_window_tokens(row.n_turns):
                stats["below_window"] += 1            # still fits the window
                continue

        q = db.query(EpisodicMemory).filter(
            EpisodicMemory.conversation_id == row.cid)
        if previous_valid:
            covered_ids = [item["id"] for item in existing_row.source_manifest["sources"]]
            q = q.filter(EpisodicMemory.id.notin_(covered_ids))
        new_turns = q.order_by(EpisodicMemory.timestamp.asc(), EpisodicMemory.id.asc()).all()
        if not new_turns:
            continue

        if lazy_embedder is None:
            lazy_embedder = _shared_embedder()

        # Fold in bounded chunks; any failed call aborts THIS conversation
        # with the old row intact (retry next burst — never half-advance).
        summary = existing_row.summary_text if previous_valid else ""
        ok = False
        for chunk_text in _source_chunks(new_turns):
            summary = _summarize_chunk(summary, chunk_text, llm, lazy_embedder)
            ok = bool(summary)
            if not ok:
                break
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
                covers_turns=row.n_turns, embedding=embedding, updated_at=now,
                source_manifest=bind_snapshot(sources, summary)))
            stats["created"] += 1
        else:
            existing_row.source_manifest = bind_snapshot(sources, summary)
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
