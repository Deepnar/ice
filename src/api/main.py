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
import redis.asyncio as aioredis
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.db import SessionLocal, get_db
from src.api.prompt_assembler import assemble_prompt
from src.api.routers import memory_slots, user_control
from src.classifier.classifier import PyTorchClassifier
from src.memory.models import Conversation, EpisodicMemory, MemorySlot
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers.post_flight import evaluate_turn
from src.model_registry.registry import find_best_model, get_fallback_model
SESSION_STATE: dict = {}
logger = structlog.get_logger("ice.api")
classifier: Optional[PyTorchClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager – initialises the classifier at startup."""
    global classifier
    logger.info("Loading classifier...")
    classifier = PyTorchClassifier(
        model_path=settings.classifier_model_path,
        schema_path=settings.label_schema_path,
    )
    logger.info("Classifier loaded. ICE Proxy ready.")
    yield


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
        turn = EpisodicMemory(
            conversation_id=conversation_id,
            batch_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
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
        log.info("turn_stored", episodic_id=str(turn.id))

        # Enqueue post‑flight evaluation (with graceful fallback to local buffer)
        try:
            evaluate_turn.delay(
                batch_id=str(turn.batch_id),
                prompt=user_message,
                response=full_assistant_text,
                conversation_id=str(conversation_id),
                model_used=model_used,
            )
        except Exception as celery_err:
            log.error("celery_enqueue_failed", error=str(celery_err))
            buffer_entry = {
                "batch_id": str(turn.batch_id),
                "prompt": user_message,
                "response": full_assistant_text,
                "conversation_id": str(conversation_id),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            with open("data/post_flight_buffer.jsonl", "a") as buf:
                buf.write(json.dumps(buffer_entry) + "\n")

        # Emit CHAT_COMPLETED event to Redis + update last‑chat timestamp
        try:
            r = aioredis.from_url(settings.redis_url)
            await r.publish("chat:completed", json.dumps({
                "correlation_id": correlation_id,
                "conversation_id": str(conversation_id),
                "batch_id": str(turn.batch_id),
                "idempotency_key": idempotency_key
            }))
            await r.set("ice:last_chat_completed", datetime.now(timezone.utc).isoformat())  # ← new line
            await r.close()
        except Exception as redis_err:
            log.error("redis_publish_failed", error=str(redis_err))

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

    # CL7: classifier will fetch & truncate the last 3 turns internally
    result = classifier.classify(
        user_message,
        conversation_id=str(conversation_id),
    ) 
    # ── Session stickiness: prevent model switching on a single off‑topic turn ──
    global SESSION_STATE
    conv_state = SESSION_STATE.get(str(conversation_id), {
        "model": None,
        "consecutive_shifts": 0,
        "last_topic_tags": [],
        "last_intent_tags": [],
    })

    # Determine if a hard topic shift occurred (no overlap with previous turn's tags)
    topic_overlap = set(result.topic_tags) & set(conv_state["last_topic_tags"])
    intent_overlap = set(result.intent_tags) & set(conv_state["last_intent_tags"])
    if topic_overlap or intent_overlap:
        conv_state["consecutive_shifts"] = 0
    else:
        conv_state["consecutive_shifts"] += 1

    conv_state["last_topic_tags"] = result.topic_tags
    conv_state["last_intent_tags"] = result.intent_tags
    SESSION_STATE[str(conversation_id)] = conv_state
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

    # State tracking boundary
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
    # For "auto" or "none", scope stays empty → retrieval searches globally

    # ── Retrieval & prompt assembly ──
    result.prompt = user_message
    fragments = []
    memory_slots_list = []
    bookmarked_texts = []
    hyde_used = False
    # CL2: LTM Bias – force memory retrieval for long conversations or uncertain classification
    if result.context_reliance == "Zero_Shot":
        turn_count = db.query(EpisodicMemory).filter_by(
            conversation_id=conversation_id
        ).count()
        if turn_count > 10 or result.max_confidence < 0.95:
            result.context_reliance = "Long_Term_Memory"
            log.info(
                "ltm_bias_override",
                turn_count=turn_count,
                max_confidence=result.max_confidence,
            )

    if (result.context_reliance == "Long_Term_Memory" or
        result.max_confidence < settings.confidence_fallback_threshold):

        embedding_tensor = await asyncio.to_thread(
            classifier.embedder.encode, user_message, convert_to_tensor=False
        )
        prompt_embedding = embedding_tensor.tolist() if hasattr(embedding_tensor, "tolist") else list(embedding_tensor)

        orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)
        # CL4: dynamic token budget from conversation length
        turn_count = db.query(EpisodicMemory).filter_by(
            conversation_id=conversation_id
        ).count()
        orchestrator.set_budget_from_turn_count(turn_count, classification=result)
        fragments = await asyncio.to_thread(
            orchestrator.retrieve,
            classification=result,
            conversation_id=str(conversation_id),
            prompt_embedding=prompt_embedding,
            scope=scope,
        )

        # HyDE usage detection (the orchestrator sets a flag internally; we approximate)
        hyde_used = getattr(orchestrator, "_hyde_used", False)

        # Bookmarked turns
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

        # Memory slots
        memory_slots_list = await asyncio.to_thread(
            lambda: db.query(MemorySlot).filter_by(is_active=True).all()
        )

        # Separate fragments by type for token trimming
        episodic_frags = [f for f in fragments if f.source_type == "episodic"]
        procedural_frags = [f for f in fragments if f.source_type == "procedural"]

        messages = assemble_prompt(
            memory_slots_list, fragments, user_message,
            db_session=db, conversation_id=str(conversation_id),
            bookmarked_texts=bookmarked_texts,
            classification=result,
            max_recent_tokens=getattr(orchestrator, 'recent_token_budget', 4000),
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
                                       bookmarked_texts=bookmarked_texts)
            total_words = word_count(messages[0]["content"]) + word_count(user_message)

        log.info(
            "context_injection_complete",
            injected_fragments=len(fragments),
            active_slots=len(memory_slots_list),
            bookmarked_count=len(bookmarked_texts),
            hyde_used=hyde_used,
        )

    # ── Model selection via registry (with stickiness) ──────────────────────────
    if body.get("model", "default") == "ice-proxy":
        if conv_state["model"] and conv_state["consecutive_shifts"] < 3:
            model_name = conv_state["model"]
            model_base_url = None
            log.info("mini_moe_sticky", model=model_name, shifts=conv_state["consecutive_shifts"])
        else:
            model_name, model_base_url = find_best_model(result.topic_tags, result.intent_tags)
            conv_state["model"] = model_name
            conv_state["consecutive_shifts"] = 0
            log.info("mini_moe_routing", selected_model=model_name,
                     topic_tags=result.topic_tags, intent_tags=result.intent_tags)
        ollama_url = f"{model_base_url or settings.ollama_base_url}/v1/chat/completions"
    else:
        model_name = body.get("model", get_fallback_model())
        ollama_url = f"{settings.ollama_base_url}/v1/chat/completions"

    # ── Model selection (moved here so we can use assembled token count) ──
    # Estimate token count of the assembled prompt
    def _word_count(text: str) -> int:
        return len(text.split()) if text else 0
    system_words = _word_count(messages[0]["content"]) if messages else 0
    user_words = _word_count(user_message)
    required_tokens = int((system_words + user_words) * 1.33)

    if body.get("model", "default") == "ice-proxy":
        if conv_state["model"] and conv_state["consecutive_shifts"] < 3:
            model_name = conv_state["model"]
            model_base_url = None
            log.info("mini_moe_sticky", model=model_name, shifts=conv_state["consecutive_shifts"])
        else:
            model_name, model_base_url = find_best_model(
                result.topic_tags, result.intent_tags, required_tokens
            )
            conv_state["model"] = model_name
            conv_state["consecutive_shifts"] = 0
            log.info("mini_moe_routing", selected_model=model_name,
                     topic_tags=result.topic_tags, intent_tags=result.intent_tags)
        ollama_url = f"{model_base_url or settings.ollama_base_url}/v1/chat/completions"
    else:
        model_name = body.get("model", get_fallback_model())
        ollama_url = f"{settings.ollama_base_url}/v1/chat/completions"

    # ── Streaming generation with fallback model ──
    ollama_url = f"{settings.ollama_base_url}/v1/chat/completions"   # keep or remove? we'll keep
    # ... (the existing ollama_url line can be removed since we already set it above)

    # ── Streaming generation with fallback model ──
    accumulated_raw_chunks = []
    model_to_use = model_name

    async def generate():
        nonlocal model_to_use

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

        # Primary request with tight timeout
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