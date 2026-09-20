"""Original-source conversation notes, composed without recursive generation.

Unchanged source groups reuse their output-bound cache. Appends create independent
notes; edits, deletions and backfills rebuild from originals. Only supported
compression replaces a group; uncertainty retains its complete source evidence.
The composed block still obeys foreground budgeting, and can be large when
long-source verification is unavailable. It is not a globally reconciled narrative.
"""
from dataclasses import asdict
from datetime import datetime, timezone

import structlog

from src.api.config import settings
from src.api.memory_decision import estimate_recent_window_tokens
from src.memory.models import ConversationSummary, EpisodicMemory
from src.memory.representation import representation_source
from src.memory.summary_snapshot import (
    bind_snapshot, compose_parts, snapshot_matches, source_snapshot,
)
from src.memory.support import supported_current, verify_support
from src.memory.time_format import recorded_stamp
from src.memory.tokens import estimate_from_chars
from src.workers.completion_text import complete_text


logger = structlog.get_logger("ice.workers.conversation_summary")


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


def _original_source(turn):
    """Do not make a generated representation the premise for another summary."""
    source = representation_source(turn)
    known_roles = source is not None
    if source is None:
        import json
        source = "Source with unknown speaker attribution: " + json.dumps(
            turn.raw_text or "", ensure_ascii=False)
    return f"{recorded_stamp(turn.timestamp, turn.ts_provenance)} {source}", known_roles


def _original_groups(turns):
    group, texts, size, known = [], [], 0, True
    for turn in turns:
        source, roles_known = _original_source(turn)
        words = len(source.split())
        if group and size + words > settings.conversation_summary_chunk_words:
            yield group, "\n\n".join(texts), known
            group, texts, size, known = [], [], 0, True
        group.append(turn)
        texts.append(source)
        size += words
        known = known and roles_known
    if group:
        yield group, "\n\n".join(texts), known


def _note_prompt(source):
    return (
        "Summarize only the original source group below. Preserve who said what, "
        "uncertainty, conditions, negation, corrections, decisions and open questions. "
        "An assistant suggestion is not a user decision. Do not invent a connection "
        "to include a name or number. Do not resolve changes using recorded time "
        "as if it were event time. Output only the source-grounded note, aiming for "
        f"at most {settings.conversation_summary_max_words} words.\n\n"
        f"ORIGINAL SOURCE GROUP:\n{source}"
    )


def _source_note(turns, source, known_roles, llm, verifier):
    generated = llm(_note_prompt(source), max_tokens=400)
    if not generated or not generated.strip():
        return None
    verdict = asdict(verifier(source, generated)) if known_roles else None
    supported = supported_current(verdict, source, generated)
    if not supported:
        logger.warning("conversation_note_source_fallback",
                       reason=verdict["reason"] if verdict else "source_roles_unknown",
                       status=verdict["status"] if verdict else "unknown",
                       source_turns=len(turns))
    return {"source_ids": [str(t.id) for t in turns],
            "mode": "supported" if supported else "source",
            "recorded_range": (f"{recorded_stamp(turns[0].timestamp, turns[0].ts_provenance)} to "
                               f"{recorded_stamp(turns[-1].timestamp, turns[-1].ts_provenance)}"),
            "text": generated if supported else source,
            "verification": verdict}


def _summary_embedding(summary, embedder):
    """Cover the complete composition even when it exceeds the encoder window."""
    import numpy as np

    tokenizer = getattr(embedder, "tokenizer", None)
    capacity = getattr(embedder, "max_seq_length", None)
    # Injected encoders without a tokenizer own their input contract (tests).
    if tokenizer is None or not isinstance(capacity, int):
        return embedder.encode(summary, convert_to_tensor=False)
    if capacity < 16:
        raise ValueError("summary encoder capacity is unusable")

    def fits(value):
        return len(tokenizer(value, truncation=False)["input_ids"]) <= capacity

    if fits(summary):
        return embedder.encode(summary, convert_to_tensor=False)
    spans, start = [], 0
    while start < len(summary):
        low, high = 1, len(summary) - start
        if not fits(summary[start:start + 1]):
            raise ValueError("single summary character exceeds encoder capacity")
        while low < high:
            mid = (low + high + 1) // 2
            if fits(summary[start:start + mid]):
                low = mid
            else:
                high = mid - 1
        spans.append(summary[start:start + low])
        start += low
    vectors = [np.asarray(embedder.encode(span, convert_to_tensor=False), dtype=float)
               for span in spans]
    pooled = np.average(vectors, axis=0, weights=[len(span) for span in spans])
    norm = float(np.linalg.norm(pooled))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("invalid pooled summary embedding")
    return pooled / norm


def run_conversation_summaries(db, llm=None, embedder=None,
                               conversation_ids=None, verifier=None) -> dict:
    """Refresh independent source notes, incrementing only strictly newer turns. Row
    creation is gated on the D3a window condition (with the legacy default
    budget — the job doesn't know the routed model; the assembler re-checks
    at injection). *llm*/*embedder* are injectable and *conversation_ids*
    restricts the pass (tests process only fixture conversations — the full
    scan is the burst/cadence behavior)."""
    stats = {"scanned": 0, "created": 0, "updated": 0,
             "below_window": 0, "failed": 0}
    if llm is None:
        llm = _default_llm
    verifier = verifier or verify_support
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
        previous_valid = existing_row is not None and bool(
            (existing_row.source_manifest or {}).get("parts")) and snapshot_matches(
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

        # Reuse output-bound notes, never their text as a generation premise.
        parts = list(existing_row.source_manifest["parts"]) if previous_valid else []
        ok = True
        for turns, source, known_roles in _original_groups(new_turns):
            from src.workers.runtime import yield_if_user_active
            yield_if_user_active("conversation_summary.source_group")
            part = _source_note(turns, source, known_roles, llm, verifier)
            if part is None:
                ok = False
                break
            parts.append(part)
        if not ok:
            stats["failed"] += 1
            logger.warning("conversation_summary_failed",
                           conversation_id=str(row.cid))
            db.rollback()
            continue
        summary = compose_parts(parts)

        vec = _summary_embedding(summary, lazy_embedder)
        embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        now = datetime.now(timezone.utc)
        if existing_row is None:
            db.add(ConversationSummary(
                conversation_id=row.cid, summary_text=summary,
                covers_through=new_turns[-1].timestamp,
                covers_turns=row.n_turns, embedding=embedding, updated_at=now,
                source_manifest=bind_snapshot(sources, summary, parts=parts)))
            stats["created"] += 1
        else:
            existing_row.source_manifest = bind_snapshot(sources, summary, parts=parts)
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
