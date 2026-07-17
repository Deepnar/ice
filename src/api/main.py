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
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.core import ICECore, create_core
from src.api.db import SessionLocal, get_db
from src.api.memory_decision import (
    decide_memory_retrieval,
    derive_total_budget,
    estimate_recent_window_tokens,
)
from src.api.prompt_assembler import assemble_prompt
from src.api.routers import memory_slots, user_control
from src.classifier.classifier import PyTorchClassifier
from src.memory.models import Conversation, EpisodicMemory, MemorySlot
from src.memory.session import resolve_session_id
from src.model_registry.registry import (
    find_best_model,
    get_fallback_model,
    get_model_context_window,
)
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.timescope import detect_timescope, to_scope_dict

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
):
    """Async post-flight task.

    Assembles streaming fragments, parses clean SSE text, calculates embeddings
    via thread pool offloading, and commits write-once transactions.
    """
    log = logger.bind(correlation_id=correlation_id)

    # 1. Join raw fragments FIRST to repair broken line boundaries from socket splits
    full_raw_stream = "".join(raw_stream_chunks)
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
    scope = {}
    conv_row = db.query(Conversation).filter_by(id=conversation_id).first()
    if conv_row and conv_row.memory_scope_type == "project":
        scope["conversation_id"] = str(conversation_id)
        if conv_row.cluster_ids:
            scope["cluster_ids"] = [str(cid) for cid in conv_row.cluster_ids]
    elif conv_row and conv_row.memory_scope_type == "none":
        # G16 incognito ("store private + read nothing"): episodic legs search
        # only this conversation; codex resolves an empty scope set (A5
        # `isolated`); rag/procedural legs are skipped by the orchestrator; the
        # turn is stored is_private so no other scope ever retrieves it and the
        # derivative pipelines (codex/procedural/clustering/summaries) skip it.
        scope["conversation_id"] = str(conversation_id)
        scope["isolated"] = True
        scope["incognito"] = True
    # For "auto", scope stays empty → retrieval searches globally (private
    # turns excluded by the legs' is_private = FALSE visibility invariant)
    is_private_conversation = bool(conv_row and conv_row.memory_scope_type == "none")

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
        intent_tags=result.intent_tags,
        p_ltm=getattr(result, "p_ltm", 0.0),
        reference_signal=getattr(result, "reference_signal", False),
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
    prev_tags = db.query(EpisodicMemory.topic_tags, EpisodicMemory.intent_tags).filter_by(
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

    # C16: total context budget from the routed model's context window
    # (fraction × window, clamped), not a hardcoded 23k.
    model_ctx_window = get_model_context_window(model_name)
    total_budget = derive_total_budget(model_ctx_window, settings)
    log.info("context_budget", model=model_name,
             context_window=model_ctx_window, total_budget=total_budget)

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
    total_tokens = int(total_chars) / 4.0  # rough chars→tokens estimate

    mem_decision = decide_memory_retrieval(
        result, turn_count=turn_count, total_tokens=total_tokens, settings=settings,
        recent_window_tokens=estimate_recent_window_tokens(turn_count, total_budget),
        timescope_mode=tscope.mode,
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

    # Separate fragments by type for token trimming (both empty when not retrieving)
    episodic_frags = [f for f in fragments if f.source_type == "episodic"]
    procedural_frags = [f for f in fragments if f.source_type == "procedural"]

    messages = assemble_prompt(
        memory_slots_list, fragments, user_message,
        db_session=db, conversation_id=str(conversation_id),
        bookmarked_texts=bookmarked_texts,
        classification=result,
        max_recent_tokens=recent_budget,
    )

    # Token budget check (crude: words * 1.33 ≈ tokens, aim for 90% of 4096)
    def word_count(text: str) -> int:
        return len(text.split()) if text else 0

    total_words = word_count(messages[0]["content"]) + word_count(user_message)
    max_words = int(0.9 * 4096 / 1.33)

    while total_words > max_words and (episodic_frags or procedural_frags):
        if procedural_frags:
            procedural_frags.pop()
        elif episodic_frags:
            episodic_frags.pop()
        # Reassemble with the reduced fragment list
        reduced = [f for f in fragments if f not in (set(episodic_frags) | set(procedural_frags))]
        messages = assemble_prompt(memory_slots_list, reduced, user_message,
                                   db_session=db, conversation_id=str(conversation_id),
                                   bookmarked_texts=bookmarked_texts,
                                   max_recent_tokens=recent_budget)
        total_words = word_count(messages[0]["content"]) + word_count(user_message)

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
                        json={"model": model_to_use, "messages": messages, "stream": True},
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
                        json={"model": model_to_use, "messages": messages, "stream": True},
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