"""Batch Summarization Worker – coalesces old turns into high‑level summaries."""

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory import tokens
from src.memory.embedder import get_embedder
from src.memory.models import BatchSummary, ColdStorage, EpisodicMemory
from src.memory.summary_snapshot import (
    SUMMARY_SOURCES_SQL, batch_snapshot_readable, bind_snapshot, compose_parts,
    source_snapshot,
)
from src.memory.support import verify_support
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name
from src.workers.completion_text import IncompleteCompletion, complete_text
from src.workers.conversation_summary import _original_groups, _source_note, _summary_embedding


logger = structlog.get_logger("ice.workers.batch_summarizer")
bg_client = get_bg_client()
# The process-shared native-width embedder (G13/G23).
embedder = get_embedder()

# Rough sizing for initial grouping only. _batch_llm checks the complete
# attributed request plus output reserve against the serving window.
_PROMPT_OVERHEAD_TOKENS = 128


def _content_budget() -> int:
    """Tokens available for the TURNS THEMSELVES in one summarisation call.

    ⚑ WHY THIS EXISTS. Batches were a fixed 50 turns with no token budget. On
    2026-08-17 that sent **37,359 tokens at a 32,768-token window** — a hard
    400 which, because the whole loop shared one `try`, killed every LATER
    batch as well, including two that would have fitted. The arm finished with
    **zero** summaries and `summary_synthesis` (40 probes) was unscoreable, and
    the failure was recorded as "transient" for a session because the only
    evidence was a swallowed exception.
    """
    window = int(settings.ollama_num_ctx_max)
    usable = window - int(settings.batch_summary_max_tokens) - _PROMPT_OVERHEAD_TOKENS
    # Divide rather than multiply: we need the CONTENT to still fit once the
    # margin is applied to it (`tokens.with_margin` is the other direction).
    return max(1, int(usable / float(settings.token_count_safety_margin)))


def _token_batches(turns, budget):
    """Greedy `(start_index, batch)` pairs whose content fits *budget* tokens.

    Always advances by at least one turn, so a single turn larger than the
    whole budget becomes a batch of ONE rather than an infinite loop — the
    `< 5` floor then skips it, which shows up in the log instead of as a 400.
    """
    batch, used, start = [], 0, 0
    for idx, turn in enumerate(turns):
        n = tokens.count(turn.raw_text or "")
        if batch and used + n > budget:
            yield start, batch
            batch, used, start = [], 0, idx
        batch.append(turn)
        used += n
    if batch:
        yield start, batch


def _batch_llm(prompt, max_tokens):
    messages = [
        {"role": "system", "content": "You are a precise summarisation engine."},
        {"role": "user", "content": prompt},
    ]
    from src.model_registry.registry import get_model_context_window
    from src.model_registry.runtime_probe import serving_window
    model = get_bg_model_name()
    window = int(settings.ollama_num_ctx_max)
    if settings.background_model_mode == "shared":
        observed = serving_window(model, get_model_context_window(model))
        if observed:
            window = min(window, int(observed)) if window > 0 else int(observed)
    required = tokens.with_margin(tokens.count_messages(messages),
                                  settings.token_count_safety_margin) + max_tokens
    if window <= 0 or required > window:
        logger.warning("batch_summary_input_over_budget", required=required, window=window)
        raise IncompleteCompletion("complete batch-summary input exceeds known capacity")
    return complete_text(bg_client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
        timeout=max(60.0, bg_timeout(max_tokens))))


def _repair_stale_caches(db):
    for summary in db.query(BatchSummary).order_by(BatchSummary.created_at, BatchSummary.id):
        if batch_snapshot_readable(db, summary):
            continue
        db.query(EpisodicMemory).filter_by(batch_summary_id=summary.id).update(
            {EpisodicMemory.batch_summary_id: None}, synchronize_session='fetch')
        db.query(ColdStorage).filter_by(batch_summary_id=summary.id).update(
            {ColdStorage.batch_summary_id: None}, synchronize_session='fetch')
        db.delete(summary)
    db.commit()


def _eligible(turn, conv_id, cutoff):
    return (str(turn.conversation_id) == conv_id
            and turn.is_private is False and turn.is_document is False
            and turn.lossless_flag is False and turn.batch_summary_id is None
            and ((turn.decay_score is not None and turn.decay_score < .3)
                 or turn.timestamp < cutoff))


def _source_rows(db, ids):
    return db.execute(text(f'''
        SELECT * FROM ({SUMMARY_SOURCES_SQL}) e
        WHERE e.id = ANY(CAST(:ids AS uuid[]))
    '''), {'ids': list(ids)}).all()


def _lock_tier(db, table, ids):
    if not ids:
        return
    if table not in ('episodic_memory', 'cold_storage'):
        raise ValueError('unknown batch source tier')
    rows = db.execute(text(f'''
        SELECT id FROM {table} WHERE id = ANY(CAST(:ids AS uuid[]))
        ORDER BY id FOR UPDATE
    '''), {'ids': ids}).all()
    if {str(row.id) for row in rows} != set(ids):
        raise ValueError('batch source moved or deleted during generation')


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
    older than `settings.batch_summary_age_days`, whichever comes first.
    """
    db = SessionLocal()
    try:
        _repair_stale_caches(db)
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.batch_summary_age_days)
        stale_turns = db.execute(text(f'''
            SELECT * FROM ({SUMMARY_SOURCES_SQL}) e
            WHERE e.conversation_id IS NOT NULL
              AND e.is_private IS FALSE
              AND e.batch_summary_id IS NULL
              AND e.lossless_flag IS FALSE
              AND e.is_document IS FALSE
              AND ((e.decay_score IS NOT NULL AND e.decay_score < 0.3)
                   OR e.timestamp < :cutoff)
            ORDER BY e.conversation_id, e.timestamp, e.id
        '''), {'cutoff': cutoff}).all()

        # Group by conversation; the batching inside is by TOKEN BUDGET,
        # not a turn count — see `_token_batches`.
        conv_groups = {}
        for turn in stale_turns:
            conv_id = str(turn.conversation_id)
            conv_groups.setdefault(conv_id, []).append(turn)

        for conv_id, turns in conv_groups.items():
            # G4(a): one conversation's summary per iteration, each committed
            # as it completes, so standing down here loses nothing already done.
            from src.workers.runtime import yield_if_user_active
            yield_if_user_active("batch_summarize.conversation")
            budget = _content_budget()
            for start, batch in _token_batches(turns, budget):
                if len(batch) < 5:
                    # G11's floor, kept (a 4-turn summary is not worth a model
                    # call). Logged, not silent: a token-CUT tail has to be
                    # distinguishable from a conversation that was just short.
                    logger.info("batch_summary_skipped_small",
                                conv_id=conv_id, turns=len(batch))
                    continue
                batch_tokens = sum(tokens.count(t.raw_text or "") for t in batch)
                logger.info("batch_summary_batch_sized", conv_id=conv_id,
                            turns=len(batch), tokens=batch_tokens, budget=budget)
                # ⚑ PER-BATCH, NOT PER-PASS. The outer `try` still re-raises
                # genuinely fatal errors (the DB going away), but one batch the
                # model refuses must not take the other batches with it — that
                # is precisely how a single 400 produced an arm with zero
                # summaries while two fitting batches never ran.
                try:
                    ids = {str(turn.id) for turn in batch}
                    # The tier may change between initial selection and this
                    # refresh. Generate from the refreshed complete originals.
                    refreshed_batch = _source_rows(db, ids)
                    if len(refreshed_batch) != len(batch) or any(
                            not _eligible(turn, conv_id, cutoff) for turn in refreshed_batch):
                        raise ValueError("batch source eligibility changed")
                    batch = sorted(refreshed_batch, key=lambda turn: (turn.timestamp, str(turn.id)))
                    batch_tokens = sum(tokens.count(turn.raw_text or '') for turn in batch)
                    before = [item for item in source_snapshot(db, conv_id) if item['id'] in ids]
                    if len(before) != len(batch):
                        raise ValueError("batch source identity changed while loading")
                    parts = []
                    for group, source, known_roles in _original_groups(batch):
                        note = _source_note(group, source, known_roles, _batch_llm,
                                            verify_support,
                                            max_tokens=settings.batch_summary_max_tokens)
                        if note is None:
                            raise IncompleteCompletion("empty batch source note")
                        parts.append(note)
                    summary_text = compose_parts(parts)
                    embedding = _summary_embedding(summary_text, embedder)
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    # Lock both physical tiers before validating source identity
                    # and writing coverage. A concurrent archive/restore moves
                    # a row between them and invalidates this attempt.
                    warm_ids = [str(t.id) for t in batch if t.storage_tier == 'warm']
                    cold_ids = [str(t.id) for t in batch if t.storage_tier == 'cold']
                    _lock_tier(db, 'episodic_memory', warm_ids)
                    _lock_tier(db, 'cold_storage', cold_ids)
                    locked = _source_rows(db, ids)
                    if (len(locked) != len(batch)
                            or {str(t.id): t.storage_tier for t in locked}
                               != {str(t.id): t.storage_tier for t in batch}
                            or any(not _eligible(t, conv_id, cutoff) for t in locked)):
                        raise ValueError("batch source eligibility or tier changed during generation")
                    after = [item for item in source_snapshot(db, conv_id) if item['id'] in ids]
                    if before != after or len(before) != len(batch):
                        raise ValueError("batch sources changed during generation")
                    summary = BatchSummary(
                        conversation_id=batch[0].conversation_id,
                        # ⚠ Write-only legacy (see models.py): positions in THIS
                        # run's filtered list, not turn identity. Coverage is the
                        # `batch_summary_id` stamp below.
                        start_turn_index=start,
                        end_turn_index=start + len(batch) - 1,
                        summary_text=summary_text,
                        embedding=embedding,
                        source_manifest=bind_snapshot(before, summary_text, parts=parts)
                    )
                    db.add(summary)
                    db.flush()          # need the id before stamping the turns
                    # G11: close the loop — mark exactly the turns this summary
                    # covers, in the SAME transaction as the summary. Committing the
                    # summary without the stamps would recreate the duplicate-work
                    # bug on the next pass, so these cannot be separated.
                    for table, tier_ids in (('episodic_memory', warm_ids),
                                            ('cold_storage', cold_ids)):
                        if tier_ids:
                            stamped = db.execute(text(f'''
                                UPDATE {table} SET batch_summary_id = :summary_id
                                WHERE id = ANY(CAST(:ids AS uuid[]))
                                  AND batch_summary_id IS NULL
                                RETURNING id
                            '''), {'ids': tier_ids, 'summary_id': summary.id}).all()
                            if len(stamped) != len(tier_ids):
                                raise ValueError("batch coverage changed before stamp")
                    db.commit()
                    logger.info("batch_summary_created", conv_id=conv_id,
                                turns=len(batch), summary_id=str(summary.id))
                except Exception as exc:
                    # WARNING and EVERY time, with the size that caused it —
                    # the RATE is the finding (CLAUDE.md: a silent fallback
                    # hides an outage). rollback() first: the batch may have
                    # already added and flushed its summary row.
                    db.rollback()
                    from src.workers.runtime import JobYielded
                    if isinstance(exc, JobYielded):
                        raise
                    logger.warning(
                        "batch_summary_batch_failed", conv_id=conv_id,
                        turns=len(batch), tokens=batch_tokens, budget=budget,
                        error=type(exc).__name__,
                    )
                    continue

    except Exception as exc:
        db.rollback()
        from src.workers.runtime import JobYielded
        if isinstance(exc, JobYielded):
            raise
        logger.error("batch_summarization_failed", error=type(exc).__name__)
        raise
    finally:
        db.close()
