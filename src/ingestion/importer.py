"""F10 D1: the one ingestion engine — replay history through the pipeline.

`import_conversations` walks a source of NormalizedConversations in
timestamp order and, per turn pair, *lives through* the pipeline: store the
episodic turn (ORIGINAL timestamp + gap-derived session), run the real
post-flight chain (density → summary → codex → procedural → decision), then
cluster the finished conversation and, at the end, batch + C4 summaries. The
result is mature memory, not an archive.

This is the REPLAY path. G23's `scripts/ice_import.py` is the sibling
STATE-COPY path (id-preserving row restore); the two are distinguished by the
`kind` field on the manifest / ImportRun. Front-ends: the format adapters
(formats.py), F14's raw slicer (raw_slicer.py), and — when FINAL is built —
its lme/synth adapters relocate here (spec §5).

Runtime shape (spec rev 9): the real import runs as the self-re-enqueueing
`import_replay` gpu-lane job — each dispatch processes conversations until a
slice budget expires or a live generation appears, then re-enqueues the next
slice, so an hours-long import never starves live chat. Tests call
`import_conversations` directly with stub LLM/embedder and no deadline.
"""

import hashlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import structlog
from sqlalchemy import text as sql_text

from src.api.config import settings
from src.memory.models import (
    Conversation,
    EpisodicMemory,
    ImportConversation,
    ImportRun,
)
from src.memory.session import resolve_session_id
from src.workers.decay import (
    ARCHIVE_THRESHOLD,
    COLD_THRESHOLD,
    CREATIVE_DECAY_RATE,
    CYCLES_PER_DAY,
    DECAY_RATE_UNACCESSED,
)

logger = structlog.get_logger("ice.ingestion.importer")

# Deterministic conversation-id namespace (D8): a provider+source-id maps to
# the same conversation across re-runs so per-turn keys can dedupe.
NS_ICE_IMPORT = uuid.UUID("2f6c0b1e-9a3d-5e4b-8c7a-1d2e3f405162")

# Per-day decay rates DERIVED from decay.py's per-cycle constants (rev 1) —
# fast_forward/hybrid ages memory in one closed-form pass at insert; the live
# apply_decay job keeps aging it forward from there.
DAILY_UNACCESSED = DECAY_RATE_UNACCESSED ** CYCLES_PER_DAY      # ≈0.95/day
DAILY_CREATIVE = CREATIVE_DECAY_RATE ** CYCLES_PER_DAY          # ≈0.99/day
CREATIVE_FLOOR = 0.3                                            # decay.py floor

POLICIES = ("hybrid", "preserve", "fast_forward", "fresh")
IMMUNE_WINDOW_DAYS = 14      # preserve/hybrid: memory feels present, then earns
RECENT_DAYS = 30             # hybrid: ≤30d preserve, older fast-forward from 30
SECONDS_PER_TURN = 6.0       # D5 cost estimate (bg-model extraction per turn)
CREATIVE_TAG = "Creative_&_Media"

# A running import older than this heartbeat is treated as crashed (rev 10).
STALE_RUN_SECONDS = 600.0


# ── decay policy ────────────────────────────────────────────────────────────

def compute_decay(policy: str, turn_ts: datetime, now: datetime,
                  is_creative: bool):
    """(decay_score, decay_immune_until, is_archived) for an imported turn.

    hybrid (default) — turns ≤RECENT_DAYS old preserve (score 1.0 + 14d
    immunity); older turns fast-forward with aging counted from the threshold,
    so the ramp is smooth instead of a cliff at day 30. preserve — all fresh
    + immune. fast_forward — all aged to their real date. fresh — all 1.0, no
    immunity. Fast-forwarded scores floor at COLD_THRESHOLD: the importer
    never deletes a row it just created (rev 4); the next natural decay cycle
    takes truly-dead rows cold through the normal machinery.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown decay policy '{policy}' — one of {POLICIES}")
    age_days = max(0.0, (now - turn_ts).total_seconds() / 86400.0)

    if policy == "fresh":
        return 1.0, None, False
    if policy == "preserve":
        return 1.0, now + timedelta(days=IMMUNE_WINDOW_DAYS), False
    if policy == "hybrid":
        if age_days <= RECENT_DAYS:
            return 1.0, now + timedelta(days=IMMUNE_WINDOW_DAYS), False
        aging_days = age_days - RECENT_DAYS
    else:   # fast_forward
        aging_days = age_days

    rate = DAILY_CREATIVE if is_creative else DAILY_UNACCESSED
    score = rate ** aging_days
    if is_creative:
        score = max(score, CREATIVE_FLOOR)
        return score, None, False
    score = max(score, COLD_THRESHOLD)
    is_archived = score < ARCHIVE_THRESHOLD
    return score, None, is_archived


# ── turn pairing (rev 13) ───────────────────────────────────────────────────

def _merge_and_pair(turns: list):
    """Merge consecutive same-role messages (``\\n\\n``), then pair
    user→assistant. Returns (pairs, skipped_leading_assistant). A trailing
    user turn pairs with an empty assistant half; a leading assistant turn is
    skipped and counted."""
    merged = []
    for t in turns:
        if merged and merged[-1]["role"] == t.role:
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + t.text
            if merged[-1]["ts"] is None:
                merged[-1]["ts"] = t.ts
        else:
            merged.append({"role": t.role, "text": t.text, "ts": t.ts})

    i, skipped = 0, 0
    while i < len(merged) and merged[i]["role"] == "assistant":
        skipped += 1
        i += 1
    pairs = []
    while i < len(merged):
        user = merged[i]
        if i + 1 < len(merged) and merged[i + 1]["role"] == "assistant":
            assistant = merged[i + 1]
            i += 2
        else:
            assistant = None
            i += 1
        pairs.append((user, assistant))
    return pairs, skipped


def _turn_idempotency_key(conv_id: uuid.UUID, index: int,
                          user_text: str, assistant_text: str) -> str:
    pair_hash = hashlib.sha256(
        (user_text + "\x00" + assistant_text).encode("utf-8")).hexdigest()[:16]
    raw = f"ice-import:{conv_id}:{index}:{pair_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── the engine ──────────────────────────────────────────────────────────────

def import_conversations(db, runtime, source: Iterable, policy: str = "hybrid",
                         *, run_id: Optional[uuid.UUID] = None,
                         classifier=None, embedder=None, llm=None,
                         run_batch_summary: bool = True,
                         run_conversation_summary: bool = True,
                         deadline: Optional[float] = None,
                         progress_cb=None,
                         conversation_id_override: Optional[uuid.UUID] = None) -> dict:
    """Replay ``source`` (iterable of NormalizedConversation) through the full
    pipeline. Idempotent + resumable: a fully-replayed conversation is
    recorded in import_conversations by content hash and skipped on re-run; a
    mid-conversation kill re-runs it and per-turn idempotency keys dedupe.

    ``deadline`` (monotonic seconds) bounds one slice — when passed, the
    engine returns ``report["complete"] = False`` with the remaining work
    still un-hashed, so the caller re-enqueues. ``runtime`` (optional) gates
    on live generations: the engine pauses between turns while a chat stream
    is in flight (never store into the pipeline behind the user's back).

    ``conversation_id_override`` (C12b) replays a single source into an
    already-created conversation instead of the provider-derived uuid5 — how a
    pasted chat log becomes a ghost conversation the document registry owns.
    Single-source only; with several conversations they would collide.
    """
    from src.ingestion.formats import content_hash
    from src.workers.post_flight import evaluate_turn

    if policy not in POLICIES:
        raise ValueError(f"unknown decay policy '{policy}' — one of {POLICIES}")

    now = datetime.now(timezone.utc)
    report = {
        "policy": policy, "conversations": 0, "skipped_conversations": 0,
        "turns": 0, "failed_turns": 0, "branch_messages_skipped": 0,
        "leading_assistant_skipped": 0, "ts_synthesized_conversations": 0,
        "complete": True, "touched_conversation_ids": [],
    }

    already = {row[0] for row in db.execute(sql_text(
        "SELECT content_hash FROM import_conversations")).fetchall()}

    for conv in source:
        if deadline is not None and time.monotonic() > deadline:
            report["complete"] = False
            break

        chash = content_hash(conv)
        if chash in already:
            report["skipped_conversations"] += 1
            continue
        if not conv.turns:
            report["skipped_conversations"] += 1
            continue

        conv_id = (conversation_id_override if conversation_id_override
                   else uuid.uuid5(NS_ICE_IMPORT,
                                   f"{conv.provider}:{conv.source_id}"))
        pairs, skipped_leading = _merge_and_pair(conv.turns)
        report["leading_assistant_skipped"] += skipped_leading
        report["branch_messages_skipped"] += conv.branch_messages_skipped
        if conv.ts_synthesized or conv.ts_synthetic:
            report["ts_synthesized_conversations"] += 1
        if not pairs:
            report["skipped_conversations"] += 1
            continue

        # Flush the parent conversation before its FK children (trap).
        first_ts = pairs[0][0]["ts"] or now
        if db.get(Conversation, conv_id) is None:
            db.add(Conversation(id=conv_id, created_at=conv.created_at or first_ts,
                                memory_scope_type="auto"))
            db.flush()

        ts_prov = "synthetic_raw_import" if conv.ts_synthetic else "original"
        history: list = []
        n_stored = 0
        for index, (user, assistant) in enumerate(pairs):
            if runtime is not None:
                # Yield to the user: never replay into the pipeline while a
                # live chat stream's post-flight could be running.
                while runtime.generation_in_flight:
                    time.sleep(0.5)
            assistant_text = assistant["text"] if assistant else ""
            turn_ts = user["ts"] or first_ts
            idem = _turn_idempotency_key(conv_id, index, user["text"], assistant_text)

            try:
                stored = _store_turn(
                    db, conv_id, user["text"], assistant_text, turn_ts, idem,
                    policy, now, ts_prov, classifier, embedder)
                if stored is not None:
                    batch_id, prompt, response = stored
                    # The real post-flight chain (density/summary → codex →
                    # procedural → decision), one direct call per turn.
                    evaluate_turn(batch_id=str(batch_id), prompt=prompt,
                                  response=response, conversation_id=str(conv_id),
                                  model_used="")
                    n_stored += 1
                report["turns"] += 1
            except Exception as exc:      # one bad turn must not kill the run
                db.rollback()
                report["failed_turns"] += 1
                logger.error("import_turn_failed", conversation=conv.source_id,
                             index=index, error=str(exc))
            history.append(user["text"])

        # Cluster the finished conversation (direct call, cpu lane logic).
        try:
            from src.workers.clustering import run_cluster_assignment
            run_cluster_assignment(db, conversation_ids=[str(conv_id)])
        except Exception as exc:
            logger.warning("import_clustering_failed",
                           conversation=conv.source_id, error=str(exc))

        # Record the hash ONLY after the conversation fully replayed (resume
        # boundary). A kill before here re-runs it; per-turn keys dedupe.
        db.add(ImportConversation(
            content_hash=chash, import_id=run_id,
            conversation_id=conv_id, title=conv.title[:500], n_turns=n_stored))
        db.commit()
        already.add(chash)

        report["conversations"] += 1
        report["touched_conversation_ids"].append(str(conv_id))
        logger.info("import_progress", conversation=conv.source_id,
                    title=conv.title[:80], turns=n_stored,
                    total_conversations=report["conversations"])
        if progress_cb is not None:
            progress_cb(report)
        if run_id is not None:
            _bump_run(db, run_id, report)

    # End-of-run summaries over exactly the touched conversations.
    if report["touched_conversation_ids"]:
        if run_conversation_summary:
            try:
                from src.workers.conversation_summary import run_conversation_summaries
                run_conversation_summaries(
                    db, llm=llm, embedder=embedder,
                    conversation_ids=report["touched_conversation_ids"])
            except Exception as exc:
                logger.warning("import_conversation_summary_failed", error=str(exc))
        if run_batch_summary:
            try:
                from src.workers.batch_summarizer import batch_summarize
                batch_summarize()
            except Exception as exc:
                logger.warning("import_batch_summary_failed", error=str(exc))

    return report


def _store_turn(db, conv_id, user_text, assistant_text, turn_ts, idem,
                policy, now, ts_prov, classifier, embedder):
    """Store one episodic turn with store_turn_async parity (rev 15): raw_text
    ``User: … \\n\\nAssistant: …``, embedding of the USER half only, tags from
    the classifier when available, gap-derived session from the ORIGINAL
    timestamp. Returns (batch_id, prompt, response) or None if the turn is
    already present (per-turn idempotency)."""
    if db.query(EpisodicMemory.id).filter_by(idempotency_key=idem).first():
        return None

    topic_tags: list = []
    intent_tags: list = []
    context_reliance = "Zero_Shot"
    if classifier is not None:
        try:
            result = classifier.classify(user_text, conversation_id=str(conv_id))
            topic_tags = list(result.topic_tags or [])
            intent_tags = list(result.intent_tags or [])
            context_reliance = result.context_reliance or "Zero_Shot"
        except Exception as exc:
            logger.warning("import_classify_failed", error=str(exc))

    is_creative = CREATIVE_TAG in topic_tags
    decay_score, immune_until, is_archived = compute_decay(
        policy, turn_ts, now, is_creative)

    enc = embedder if embedder is not None else _lazy_embedder()
    vec = enc.encode(user_text, convert_to_tensor=False)
    embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

    session_id, _started, _gap = resolve_session_id(
        db, conv_id, turn_ts, settings.session_gap_minutes)

    batch_id = uuid.uuid4()
    turn = EpisodicMemory(
        conversation_id=conv_id, batch_id=batch_id, session_id=session_id,
        is_private=False, timestamp=turn_ts, ts_provenance=ts_prov,
        topic_tags=topic_tags, intent_tags=intent_tags,
        context_reliance=context_reliance,
        raw_text=f"User: {user_text}\n\nAssistant: {assistant_text}",
        embedding=embedding, decay_score=decay_score,
        decay_immune_until=immune_until, is_archived=is_archived,
        idempotency_key=idem)
    db.add(turn)
    db.commit()
    return batch_id, user_text, assistant_text


def _lazy_embedder():
    from src.memory.embedder import get_embedder
    return get_embedder()


# ── ImportRun bookkeeping ───────────────────────────────────────────────────

def _bump_run(db, run_id, report) -> None:
    run = db.get(ImportRun, run_id)
    if run is None:
        return
    run.done_conversations = report["conversations"]
    run.skipped_conversations = report["skipped_conversations"]
    run.done_turns = report["turns"]
    run.failed_turns = report["failed_turns"]
    run.updated_at = datetime.now(timezone.utc)
    db.commit()


def estimate_seconds(total_turns: int) -> float:
    return total_turns * SECONDS_PER_TURN


# ── the runtime job (spec rev 9) ────────────────────────────────────────────

SLICE_BUDGET_SECONDS = 600.0    # ≈10 min per gpu-lane slice, then yield


def run_import_replay(db, import_id: str, seq: int = 0) -> None:
    """The self-re-enqueueing gpu-lane import job. Rebuilds the source from the
    ImportRun's path+format, replays one slice, and re-enqueues the next slice
    until the export is exhausted. Kill-safe: resume rides the
    import_conversations hash ledger + per-turn keys."""
    run = db.get(ImportRun, uuid.UUID(import_id))
    if run is None:
        logger.error("import_run_missing", import_id=import_id)
        return
    if run.status in ("completed", "aborted"):
        return

    try:
        source, total = _build_source(run.source_path, run.source_format)
    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)[:1000]
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
        logger.error("import_source_build_failed", import_id=import_id, error=str(exc))
        return

    deadline = time.monotonic() + SLICE_BUDGET_SECONDS
    from src.api.core import get_core
    core = get_core()
    classifier = core.classifier if core is not None else None

    report = import_conversations(
        db, _get_runtime(), source, run.policy, run_id=run.id,
        classifier=classifier, deadline=deadline,
        # end-of-run summaries only fire on the FINAL slice (below)
        run_batch_summary=False, run_conversation_summary=False)

    _bump_run(db, run.id, report)
    if report["complete"]:
        # Final slice: run the end-of-run summaries over everything imported.
        _finalize_run(db, run)
    else:
        _enqueue_next_slice(run.id, seq + 1)


def _finalize_run(db, run) -> None:
    conv_ids = [r[0] for r in db.execute(sql_text(
        "SELECT conversation_id FROM import_conversations WHERE import_id = :i"),
        {"i": run.id}).fetchall()]
    if conv_ids:
        try:
            from src.workers.conversation_summary import run_conversation_summaries
            run_conversation_summaries(db, conversation_ids=[str(c) for c in conv_ids])
        except Exception as exc:
            logger.warning("import_finalize_conv_summary_failed", error=str(exc))
        try:
            from src.workers.batch_summarizer import batch_summarize
            batch_summarize()
        except Exception as exc:
            logger.warning("import_finalize_batch_summary_failed", error=str(exc))
    run.status = "completed"
    run.finished_at = datetime.now(timezone.utc)
    run.updated_at = run.finished_at
    db.commit()
    logger.info("import_completed", import_id=str(run.id),
                conversations=run.done_conversations, turns=run.done_turns)


def _build_source(path: str, fmt: str):
    """(source iterable, total_turns) for a run's file. raw → F14 slicer."""
    from src.ingestion import formats
    if fmt == "raw":
        from src.ingestion.raw_slicer import slice_raw_file
        convs = slice_raw_file(path)
    else:
        _fmt, convs = formats.normalize_file(path, fmt)
    total = sum(len(c.turns) for c in convs)
    return convs, total


def _enqueue_next_slice(run_id, seq: int) -> None:
    rt = _get_runtime()
    if rt is not None:
        rt.enqueue("import_replay", import_id=str(run_id), seq=seq)


def _get_runtime():
    from src.workers.runtime import get_runtime
    return get_runtime()
