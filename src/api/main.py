"""ICE FastAPI Middleware.

Intercepts OpenAI-compatible chat requests, classifies the prompt,
forwards to Ollama, streams the response back, and safely stores records
without blocking the event loop or triggering race conditions.
"""

import asyncio
import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.chat_commands import command_sse_stream, try_handle
from src.api.config import settings
from src.api.context_ledger import ContextLedger, effective_memory_budget
from src.api.core import ICECore, create_core
from src.api.db import SessionLocal, get_db
from src.api.memory_decision import (
    decide_memory_retrieval,
    derive_total_budget,
    estimate_recent_window_tokens,
)
from src.api.prompt_assembler import assemble_prompt, conversation_summary_block
from src.api.routers import memory_slots, user_control
from src.classifier.classifier import PyTorchClassifier
from src.memory.models import Conversation, EpisodicMemory, MemorySlot
from src.memory.session import resolve_session_id
from src.memory.tokens import count_messages as count_tokens_messages
from src.memory.tokens import estimate_from_chars as estimate_tokens_from_chars
from src.model_registry.registry import (
    find_best_model,
    get_fallback_model,
    get_model_context_window,
)
from src.model_registry.runtime_probe import log_window_truth, serving_window
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.timescope import detect_timescope, to_scope_dict
from src.services.scoping import resolve_retrieval_scope

logger = structlog.get_logger("ice.api")
classifier: Optional[PyTorchClassifier] = None
core: Optional[ICECore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager – the maintenance core (C7) + its classifier."""
    global classifier, core
    logger.info("Loading classifier...")
    core = create_core()
    # E0/G13: the classifier lives on the core (one process, one model load —
    # retrieval_svc reaches it via get_core()); touch it eagerly so the load
    # still happens at boot, not on the first request.
    classifier = core.classifier
    logger.info("Classifier loaded. ICE Proxy ready.")
    yield
    await core.stop()


app = FastAPI(
    title="ICE Proxy",
    description="Infinite Context Engine — OpenAI-compatible memory middleware",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(memory_slots.router)
app.include_router(user_control.router)


@app.get("/health")
def health():
    return {"status": "ok"}


def _ollama_body(model: str, messages: list,
                 prompt_tokens: int = 0) -> dict:
    """The generation request body.

    C16: `stream_options.include_usage` asks the server to append a final
    chunk carrying `usage.prompt_eval_count` — the model server's OWN count of
    the prompt it received. That is the only ground truth for how many tokens
    ICE actually spends, and it is what calibrates the embedder-tokenizer
    prediction (`memory/tokens.py`) against the generation model's vocabulary
    instead of arguing about the difference. The chunk has no
    `choices[0].delta`, so the storage parser already skips it.
    """
    body = {"model": model, "messages": messages, "stream": True}
    if settings.token_usage_reconciliation:
        body["stream_options"] = {"include_usage": True}
    # C16: ask for the window this prompt actually needs. Off by default —
    # a bigger KV cache costs VRAM and can evict the generation model, which
    # trades ~300 ms of prompt handling for something worse. Only safe now
    # that the prompt is bounded and measured.
    if settings.ollama_send_num_ctx and prompt_tokens:
        want = prompt_tokens + settings.context_generation_reserve
        body["options"] = {"num_ctx": min(int(settings.ollama_num_ctx_max),
                                          int(want))}
    return body


def _extract_usage(full_raw_stream: str) -> Optional[dict]:
    """Pull the trailing usage chunk out of an SSE stream, if the server sent
    one. Returns None when it did not — which is itself worth logging, because
    a serving stack that ignores `stream_options` leaves the prediction
    unverifiable and the safety margin is all that stands behind it."""
    for line in reversed(full_raw_stream.split("\n")):
        line = line.strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        try:
            data = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data.get("usage"), dict):
            return data["usage"]
    return None


async def store_turn_async(
    correlation_id: str,
    user_message: str,
    conversation_id: uuid.UUID,
    topic_tags: list,
    intent_tags: list,
    context_reliance: str,
    raw_stream_chunks: list[str],
    model_used: str = "",
    is_private: bool = False,
    predicted_prompt_tokens: int = 0,
):
    """Async post-flight task.

    Assembles streaming fragments, parses clean SSE text, calculates embeddings
    via thread pool offloading, and commits write-once transactions.
    """
    log = logger.bind(correlation_id=correlation_id)

    # 1. Join raw fragments FIRST to repair broken line boundaries from socket splits
    full_raw_stream = "".join(raw_stream_chunks)

    # C16: reconcile the prediction against the server's own count. This is
    # the only place ICE can learn whether its ruler is right — the embedder's
    # tokenizer is not the generation model's, and the whole token-efficiency
    # claim rests on the difference being small. Logged, never enforced.
    usage = _extract_usage(full_raw_stream)
    if predicted_prompt_tokens and usage and usage.get("prompt_tokens"):
        actual = usage["prompt_tokens"]
        log.info("token_prediction_reconciled",
                 predicted=predicted_prompt_tokens, actual=actual,
                 ratio=round(predicted_prompt_tokens / max(1, actual), 4),
                 model=model_used)
    elif predicted_prompt_tokens and settings.token_usage_reconciliation:
        # A serving stack that ignores stream_options leaves the prediction
        # unverifiable — say so rather than assuming it was fine.
        log.info("token_usage_unavailable", predicted=predicted_prompt_tokens,
                 model=model_used)

    clean_fragments = []

    for line in full_raw_stream.split("\n"):
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        if line == "data: [DONE]":
            continue
        try:
            data = json.loads(line[5:].strip())
            content = data["choices"][0]["delta"].get("content", "")
            if content:
                clean_fragments.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    full_assistant_text = "".join(clean_fragments)

    # 2. Offload CPU-heavy tensor tasks to avoid event loop starvation
    embedding = await asyncio.to_thread(
        classifier.embedder.encode, user_message, convert_to_tensor=False
    )
    embedding_list = (
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    )

    # 3. Establish deterministic idempotency boundaries
    raw_key = f"{correlation_id}:{user_message}"
    idempotency_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    write_db = SessionLocal()
    try:
        # C6: session identity — same sitting while the silence stays within
        # session_gap_minutes; a longer gap opens a new session.
        now = datetime.now(timezone.utc)
        session_id, session_started, gap_seconds = resolve_session_id(
            write_db, conversation_id, now, settings.session_gap_minutes
        )
        if session_started:
            log.info("session_started", session_id=str(session_id),
                     conversation_id=str(conversation_id))

        turn = EpisodicMemory(
            conversation_id=conversation_id,
            batch_id=uuid.uuid4(),
            session_id=session_id,
            is_private=is_private,
            timestamp=now,
            topic_tags=topic_tags,
            intent_tags=intent_tags,
            context_reliance=context_reliance,
            entropy_score=None,          # set by Post‑Flight Evaluator
            lossless_flag=None,          # NULL = not yet evaluated
            raw_text=f"User: {user_message}\n\nAssistant: {full_assistant_text}",
            summary_text=None,
            embedding=embedding_list,
            decay_score=1.0,
            idempotency_key=idempotency_key,
        )
        write_db.add(turn)
        write_db.commit()
        log.info("turn_stored", episodic_id=str(turn.id), session_id=str(session_id))

        # C7: in-process dispatch (the celery .delay + jsonl-buffer fallback
        # and the redis publish died with the broker — an in-process enqueue's
        # only failure mode is the app being down, in which case the turn
        # wasn't stored either; idempotency keys remain the at-least-once
        # guard). E7: runtime may be absent (create_core(start_runtime=False)
        # in tests) — a lease-deferred core still has a standby runtime that
        # runs this process's event jobs.
        if core is not None and core.runtime is not None:
            core.runtime.enqueue(
                "post_flight",
                batch_id=str(turn.batch_id),
                prompt=user_message,
                response=full_assistant_text,
                conversation_id=str(conversation_id),
                model_used=model_used,
            )
            if session_started:
                # The "new sitting" work-unit: cluster freshening + an
                # immediate overdue pass (decay catch-up rides the ledger).
                core.runtime.notify_work_unit(
                    "session_gap",
                    conversation_id=str(conversation_id),
                    gap_seconds=gap_seconds,
                )
        else:
            # Only reachable when store_turn_async is driven outside the app
            # lifespan (standalone tests) — surface it, never swallow it.
            log.error("maintenance_core_not_started_post_flight_skipped",
                      batch_id=str(turn.batch_id))

    except Exception as exc:
        write_db.rollback()
        log.error("failed_to_store_turn", error=str(exc))
    finally:
        write_db.close()


def sse_event(event_type: str, data: dict) -> str:
    """Build a properly formatted SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    body = await request.json()
    messages = body.get("messages", [])
    model_name = body.get("model", "default")

    correlation_id = str(uuid.uuid4())
    log = logger.bind(correlation_id=correlation_id)

    # C7 D7: in-process activity signal — the runtime's idle gating for
    # background work keys off this (replaces the redis last-chat flag).
    if core is not None and core.runtime is not None:
        core.runtime.note_user_activity()

    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        log.warning("No user message found in request")
        return JSONResponse(
            status_code=400,
            content={"error": "No user message found in the request."},
        )

    # G26: conversation + scope must resolve BEFORE classification —
    # classify() reads conversation_id for its CL7 context prefix (this block
    # previously sat ~40 lines below the classify call, a C16-reorder
    # casualty: UnboundLocalError on every request).
    conversation_id_str = request.headers.get("X-ICE-Conversation-ID")
    if conversation_id_str:
        conversation_id = uuid.UUID(conversation_id_str)
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            conversation = Conversation(id=conversation_id)
            db.add(conversation)
            db.commit()
    else:
        conversation = Conversation()
        db.add(conversation)
        db.commit()
        conversation_id = conversation.id

    # ── Scope from conversation metadata ──
    # C6: built by the shared resolver (services/scoping.py), which the MCP
    # pull path uses too — this block and retrieval_svc's copy had already
    # drifted apart. Modes, precedence and the exclusion sets are documented
    # there; "auto" resolves to an empty dict → retrieval searches globally,
    # private turns excluded by the legs' is_private = FALSE invariant.
    conv_row = db.query(Conversation).filter_by(id=conversation_id).first()
    scope = resolve_retrieval_scope(db, conv_row)
    is_private_conversation = bool(conv_row and conv_row.memory_scope_type == "none")

    # C11: deterministic slash commands — parsed AFTER conversation/scope
    # resolution (they need the conv row) and BEFORE classification (a
    # command must never burn pre-flight latency or reach the model as a
    # prompt; unknown /x gets a help hint, never silent fallthrough). Handled
    # commands stream a normal SSE completion and skip storage/post-flight
    # entirely — the chat_command journal is the record.
    if user_message.split("\n", 1)[0].lstrip().startswith("/"):
        cmd_result = await asyncio.to_thread(
            try_handle, db,
            core.runtime if core is not None else None,
            conv_row, user_message, scope,
        )
        if cmd_result is not None:
            return StreamingResponse(
                command_sse_stream(cmd_result["text"]),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-ICE-Conversation-ID": str(conversation_id),
                },
            )

    # CL7: classifier will fetch & truncate the last 3 turns internally
    result = classifier.classify(
        user_message,
        conversation_id=str(conversation_id),
    )

    # T2: deterministic temporal-intent detection (joint-gated on the
    # classification signals). A non-current TimeScope rides inside the scope
    # dict; every retrieval leg reads it from there.
    tscope = detect_timescope(
        user_message,
        p_ltm=getattr(result, "p_ltm", 0.0),
        p_temporal=getattr(result, "p_temporal", 0.0),
    )
    if tscope.mode != "current":
        log.info(
            "timescope_detected",
            mode=tscope.mode,
            t0=tscope.t0.isoformat() if tscope.t0 else None,
            t1=tscope.t1.isoformat() if tscope.t1 else None,
            matched=tscope.matched_text,
        )
    ts_dict = to_scope_dict(tscope)
    if ts_dict:
        scope["timescope"] = ts_dict
    # ── Session stickiness: prevent model switching on a single off-topic turn ──
    # C7 D9 (G8): state lives on the conversation row (sticky_model /
    # consecutive_shifts), not an in-memory dict; the previous turn's tags for
    # the shift-overlap check come from the latest episodic row (written
    # post-stream, so a rapid-fire message may compare against the turn
    # before — same fallback as a fresh dict entry had).
    prev_tags = db.query(
        EpisodicMemory.topic_tags, EpisodicMemory.intent_tags,
        EpisodicMemory.timestamp,
    ).filter_by(
        conversation_id=conversation_id
    ).order_by(EpisodicMemory.timestamp.desc()).first()

    topic_overlap = set(result.topic_tags) & set(prev_tags.topic_tags or [] if prev_tags else [])
    intent_overlap = set(result.intent_tags) & set(prev_tags.intent_tags or [] if prev_tags else [])
    if topic_overlap or intent_overlap:
        conv_row.consecutive_shifts = 0
    else:
        conv_row.consecutive_shifts = (conv_row.consecutive_shifts or 0) + 1
    log.info(
        "classified",
        topic_tags=result.topic_tags,
        intent_tags=result.intent_tags,
        context_reliance=result.context_reliance,
        max_confidence=result.max_confidence,
    )

    if result.max_confidence < settings.confidence_fallback_threshold:
        log.info(
            "low_confidence_fallback",
            max_confidence=result.max_confidence,
            threshold=settings.confidence_fallback_threshold,
        )

    # ── Model selection via registry (with stickiness) ──
    # C16: selection happens BEFORE budgeting/retrieval so the context budget
    # can derive from the routed model's actual context window (the old design
    # assembled first and selected after — with two duplicated selection blocks,
    # the first silently overwritten, and a stray ollama_url override that
    # killed registry base_url routing entirely; all removed, G20).
    if body.get("model", "default") == "ice-proxy":
        if conv_row.sticky_model and conv_row.consecutive_shifts < 3:
            model_name = conv_row.sticky_model
            model_base_url = None
            log.info("mini_moe_sticky", model=model_name, shifts=conv_row.consecutive_shifts)
        else:
            model_name, model_base_url = find_best_model(result.topic_tags, result.intent_tags)
            conv_row.sticky_model = model_name
            conv_row.consecutive_shifts = 0
            log.info("mini_moe_routing", selected_model=model_name,
                     topic_tags=result.topic_tags, intent_tags=result.intent_tags)
        ollama_url = f"{model_base_url or settings.ollama_base_url}/v1/chat/completions"
    else:
        model_name = body.get("model", get_fallback_model())
        ollama_url = f"{settings.ollama_base_url}/v1/chat/completions"
    db.commit()  # persist the stickiness state (shift counter + sticky model)

    # C16: total context budget from the window the SERVER actually allocated,
    # falling back to the registry's claim. Those two disagreed by 2x on a live
    # model — the registry said 64,000, the runner had allocated 32,768, and
    # the derived budget was 40,000, i.e. already over the real window before
    # a single token of answer. log_window_truth emits all four every request.
    registry_window = get_model_context_window(model_name)
    effective_window = registry_window
    if settings.context_use_serving_window:
        effective_window = serving_window(model_name, registry_window)
    total_budget = derive_total_budget(effective_window, settings)
    log_window_truth(log, model_name, registry_window, total_budget)

    # C16: the user's own message is part of the arithmetic now. It never was:
    # on the turn where someone pastes 10,000 words, ICE added its full memory
    # allowance ON TOP, which is precisely the case where stopping matters most.
    memory_budget = effective_memory_budget(
        total_budget, user_message,
        generation_reserve=settings.context_generation_reserve,
        floor=settings.context_budget_floor)
    if memory_budget < total_budget:
        log.info("budget_after_question", total=total_budget,
                 memory=memory_budget,
                 question_tokens=total_budget - memory_budget
                 - settings.context_generation_reserve)
    total_budget = memory_budget

    # ── Retrieval & prompt assembly ──
    result.prompt = user_message
    fragments = []
    memory_slots_list = []
    bookmarked_texts = []
    hyde_used = False

    # B2: one principled, classifier-trusting decision replaces the old hard
    # overrides (turn_count>10 / conf<0.95 / creative / referential). Prefers
    # memory, never forces it. Weights are settings (re-tuned after B1).
    turn_count = db.query(EpisodicMemory).filter_by(
        conversation_id=conversation_id
    ).count()
    total_chars = db.query(
        func.coalesce(func.sum(func.length(EpisodicMemory.raw_text)), 0)
    ).filter_by(conversation_id=conversation_id).scalar() or 0
    total_tokens = estimate_tokens_from_chars(total_chars)

    mem_decision = decide_memory_retrieval(
        result, turn_count=turn_count, total_tokens=total_tokens, settings=settings,
        recent_window_tokens=estimate_recent_window_tokens(turn_count, total_budget),
        timescope_mode=tscope.mode,
        coding_scope=bool(scope.get("project_id")),
    )
    log.info("memory_decision", retrieve=mem_decision.retrieve, **mem_decision.breakdown)
    # Recent-turn budget for prompt assembly — principled default that also
    # applies when we don't retrieve (a long convo we chose not to search still
    # deserves a scaled recent window).
    recent_budget = estimate_recent_window_tokens(turn_count, total_budget)

    if mem_decision.retrieve:
        # Downstream (orchestrator gates, episodic storage, telemetry) still
        # keys off the label, so reflect the decision there.
        result.context_reliance = "Long_Term_Memory"

        embedding_tensor = await asyncio.to_thread(
            classifier.embedder.encode, user_message, convert_to_tensor=False
        )
        prompt_embedding = embedding_tensor.tolist() if hasattr(embedding_tensor, "tolist") else list(embedding_tensor)

        orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)
        # CL4: dynamic token budget from conversation length, ceiling from the
        # routed model's context window (C16).
        orchestrator.set_budget_from_turn_count(
            turn_count, total_tokens=total_tokens, classification=result,
            total_budget=total_budget,
        )
        fragments = await asyncio.to_thread(
            orchestrator.retrieve,
            classification=result,
            conversation_id=str(conversation_id),
            prompt_embedding=prompt_embedding,
            scope=scope,
        )

        # HyDE usage detection (the orchestrator sets a flag internally; we approximate)
        hyde_used = getattr(orchestrator, "_hyde_used", False)
        recent_budget = getattr(orchestrator, "recent_token_budget", recent_budget)

    # ── Persistent memory + assembly run on EVERY turn ──
    # Memory slots (standing user memory) and bookmarks are user-level context,
    # not retrieval results — they must be injected even when B2 decides a
    # confident standalone turn needs no long-term retrieval. Only the retrieval
    # `fragments` are conditional (empty here when we didn't retrieve).
    bookmarked_turns = await asyncio.to_thread(
        lambda: db.query(EpisodicMemory).filter_by(
            is_bookmarked=True, conversation_id=conversation_id
        ).order_by(EpisodicMemory.timestamp.desc()).limit(5).all()
    )
    for bt in bookmarked_turns:
        text = bt.raw_text if bt.inject_raw else (bt.summary_text or bt.raw_text[:300])
        words = text.split()
        if len(words) > 500:
            text = " ".join(words[:500]) + "…"
        bookmarked_texts.append(text)

    memory_slots_list = await asyncio.to_thread(
        lambda: db.query(MemorySlot).filter_by(is_active=True).all()
    )

    # E4 (D6): coding-scoped conversations get the project session-start
    # block, rendered only when this turn OPENS a sitting (rev 8: the latest
    # stored turn is older than the session gap, or absent).
    session_start_text = None
    if scope.get("project_id"):
        last_ts = prev_tags.timestamp if prev_tags else None
        new_sitting = last_ts is None or (
            datetime.now(timezone.utc) - last_ts
            > timedelta(minutes=settings.session_gap_minutes))
        if new_sitting:
            from src.services import projects as projects_svc
            session_start_text = await asyncio.to_thread(
                projects_svc.chat_session_start, db, scope["project_id"])
            log.info("project_session_start",
                     project_id=scope["project_id"],
                     rendered=bool(session_start_text))

    # C4 (D3a): once the conversation outgrew the sliding window, its evolving
    # summary gives the model global shape (stamped when the burst is behind).
    conversation_summary_text = await asyncio.to_thread(
        conversation_summary_block, db, str(conversation_id),
        turn_count, total_tokens, recent_budget)

    # Separate fragments by type for token trimming (both empty when not retrieving)
    episodic_frags = [f for f in fragments if f.source_type == "episodic"]
    procedural_frags = [f for f in fragments if f.source_type == "procedural"]

    messages = assemble_prompt(
        memory_slots_list, fragments, user_message,
        db_session=db, conversation_id=str(conversation_id),
        bookmarked_texts=bookmarked_texts,
        classification=result,
        scope=scope,
        max_recent_tokens=recent_budget,
        session_start_text=session_start_text,
        conversation_summary_text=conversation_summary_text,
    )

    # C16: the prompt is measured, not guessed. What stood here was a loop
    # that trimmed fragments against a hardcoded 4096-token window, and it was
    # broken four ways at once: (1) it ignored the model-derived budget
    # entirely; (2) it measured `messages[0]`, the SYSTEM message, while
    # fragments are appended as their own user message — so popping a fragment
    # could never change its own loop condition; (3) `reduced` was built as the
    # COMPLEMENT of the survivors, i.e. exactly the fragments it had just
    # decided to drop; and (4) because the condition was invariant it ran until
    # both lists were empty, at which point `reduced` == every fragment and all
    # of them were restored. Net behaviour: none. Net cost: one re-assembly per
    # fragment, each re-running the recent-turns query, on the latency path.
    prompt_tokens = count_tokens_messages(messages)

    # C16: one account, in one unit, checked against the window the server
    # actually allocated. When it does not fit, static blocks are shed before
    # evidence — with E8 constraints exempt, because they are user preferences
    # and losing one is the failure that feature exists to prevent.
    ledger = ContextLedger(
        serving_window=effective_window or 0,
        generation_reserve=settings.context_generation_reserve,
        safety_margin=settings.token_count_safety_margin)
    ledger.add("system_prompt", messages[0]["content"] if messages else "")
    ledger.add("recent_turns", sum(
        count_tokens_messages([m]) for m in messages[1:-1]
        if not m["content"].startswith("=== RETRIEVED CONTEXT")))
    ledger.add("evidence", sum(f.token_count for f in fragments))
    ledger.add("user_message", user_message)

    plan = ledger.evict_plan()
    if plan:
        log.warning("context_evicting", plan=plan, overflow=ledger.overflow(),
                    window=effective_window)
        messages = assemble_prompt(
            memory_slots_list if "slots" not in plan else [],
            [] if "evidence" in plan else fragments,
            user_message,
            db_session=db, conversation_id=str(conversation_id),
            bookmarked_texts=None if "bookmarks" in plan else bookmarked_texts,
            classification=result, scope=scope,
            max_recent_tokens=64 if "recent_turns" in plan else recent_budget,
            session_start_text=None if "session_start" in plan else session_start_text,
            conversation_summary_text=(
                None if "conversation_summary" in plan else conversation_summary_text),
        )
        prompt_tokens = count_tokens_messages(messages)
    ledger.log(log)

    log.info("prompt_measured", prompt_tokens=prompt_tokens,
             total_budget=total_budget, model=model_name,
             fragments=len(fragments),
             episodic=len(episodic_frags), procedural=len(procedural_frags))

    log.info(
        "context_injection_complete",
        retrieved=mem_decision.retrieve,
        injected_fragments=len(fragments),
        active_slots=len(memory_slots_list),
        bookmarked_count=len(bookmarked_texts),
        hyde_used=hyde_used,
    )

    # (Model selection happens above, before budgeting/retrieval — C16.)

    # ── Streaming generation with fallback model ──
    accumulated_raw_chunks = []
    model_to_use = model_name

    async def generate():
        nonlocal model_to_use

        # C7 D7: the in-flight flag is the shared-mode contention gate — no
        # background gpu job dispatches while a generation streams.
        if core is not None and core.runtime is not None:
            core.runtime.generation_started()

        # SSE: classified
        yield sse_event("classified", {
            "topic_tags": result.topic_tags,
            "intent_tags": result.intent_tags,
            "context_reliance": result.context_reliance,
            "max_confidence": result.max_confidence,
        })

        # SSE: retrieval
        yield sse_event("retrieval", {
            "active_legs": list({f.source_type for f in fragments}),
            "hyde_used": hyde_used,
            "tokens_injected": sum(f.token_count for f in fragments),
        })

        # SSE: context_ready
        yield sse_event("context_ready", {
            "fragments_count": len(fragments),
            "sources": {
                "codex": sum(1 for f in fragments if f.source_type == "codex"),
                "episodic": sum(1 for f in fragments if f.source_type == "episodic"),
                "procedural": sum(1 for f in fragments if f.source_type == "procedural"),
                "rag": sum(1 for f in fragments if f.source_type == "rag"),
            },
            "total_tokens": sum(f.token_count for f in fragments),
        })

        # SSE: generating
        yield sse_event("generating", {"model": model_to_use})

        # Primary request with tight timeout. The outer try/finally guarantees
        # generation_finished fires even on a client disconnect mid-stream.
        try:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    async with client.stream(
                        "POST",
                        ollama_url,
                        json=_ollama_body(model_to_use, messages,
                                          prompt_tokens),
                    ) as response:
                        async for chunk in response.aiter_text():
                            accumulated_raw_chunks.append(chunk)
                            yield chunk
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                log.warning("primary_model_timeout", model=model_to_use, error=str(e))
                yield sse_event("degraded", {"reason": "primary_model_timeout", "fallback": get_fallback_model()})
                model_to_use = get_fallback_model()
                yield sse_event("generating", {"model": model_to_use})
                async with httpx.AsyncClient(timeout=30.0) as client2:
                    async with client2.stream(
                        "POST",
                        ollama_url,
                        json=_ollama_body(model_to_use, messages,
                                          prompt_tokens),
                    ) as response2:
                        async for chunk in response2.aiter_text():
                            accumulated_raw_chunks.append(chunk)
                            yield chunk
            except Exception as e:
                yield sse_event("degraded", {"reason": "streaming_error", "error": str(e)})
        finally:
            if core is not None and core.runtime is not None:
                core.runtime.generation_finished()

    # Enqueue post‑flight storage (via BackgroundTasks, which runs after response)
    background_tasks.add_task(
        store_turn_async,
        correlation_id=correlation_id,
        user_message=user_message,
        conversation_id=conversation_id,
        topic_tags=result.topic_tags,
        intent_tags=result.intent_tags,
        context_reliance=result.context_reliance,
        raw_stream_chunks=accumulated_raw_chunks,
        model_used=model_to_use,
        is_private=is_private_conversation,
        predicted_prompt_tokens=prompt_tokens,
    )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-ICE-Conversation-ID": str(conversation_id),
        },
    )