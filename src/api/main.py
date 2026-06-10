"""ICE FastAPI Middleware.

Intercepts OpenAI-compatible chat requests, classifies the prompt,
forwards to Ollama, streams the response back, and safely stores records.
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
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.db import SessionLocal, get_db
from src.classifier.classifier import PyTorchClassifier
from src.memory.models import Conversation, EpisodicMemory
from src.workers.post_flight import evaluate_turn
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.prompt_assembler import assemble_prompt
from src.memory.models import MemorySlot


logger = structlog.get_logger("ice.api")
classifier: Optional[PyTorchClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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

from src.api.routers import memory_slots

app.include_router(memory_slots.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "ice-proxy",
                "object": "model",
                "created": 1,
                "owned_by": "ice",
            }
        ],
    }


async def store_turn_async(
    correlation_id: str,
    user_message: str,
    conversation_id: uuid.UUID,
    topic_tags: list,
    intent_tags: list,
    context_reliance: str,
    raw_stream_chunks: list[str],
):
    """Async post-flight task – parses SSE, computes embedding, writes to DB."""
    log = logger.bind(correlation_id=correlation_id)

    # 1. Parse SSE to extract clean assistant text
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

    # 2. Embedding offloaded to thread pool
    embedding = await asyncio.to_thread(
        classifier.embedder.encode, user_message, convert_to_tensor=False
    )
    embedding_list = (
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    )

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
            entropy_score=None,
            lossless_flag=None,
            raw_text=f"User: {user_message}\n\nAssistant: {full_assistant_text}",
            summary_text=None,
            embedding=embedding_list,
            decay_score=1.0,
            idempotency_key=idempotency_key,
        )
        write_db.add(turn)
        write_db.commit()
        log.info("turn_stored", episodic_id=str(turn.id))
        evaluate_turn.delay(
            batch_id=str(turn.batch_id),
            prompt=user_message,
            response=full_assistant_text,
            conversation_id=str(conversation_id),
        )
    except Exception as exc:
        write_db.rollback()
        log.error("failed_to_store_turn", error=str(exc))
    finally:
        write_db.close()


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

    ollama_url = f"{settings.ollama_base_url}/v1/chat/completions"

    # ───────────────────────────────────────────────────────────
    # FILTER: OpenWebUI internal prompts (title, tags, follow‑ups)
    # ───────────────────────────────────────────────────────────
    if "### Task:" in user_message or "### Chat History:" in user_message:
        log.info("skipping_internal_task")
        async def passthrough():
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    ollama_url,
                    json={"model": model_name, "messages": messages, "stream": True},
                ) as ollama_response:
                    async for chunk in ollama_response.aiter_text():
                        yield chunk
        return StreamingResponse(passthrough(), media_type="text/event-stream")

    # ───────────────────────────────────────────────────────────
    # REAL USER PROMPT – classify
    # ───────────────────────────────────────────────────────────
    result = classifier.classify(user_message)
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

    # ───────────────────────────────────────────────────────────
    # Conversation: create a new one per request (V1 default)
    # ───────────────────────────────────────────────────────────
    conversation = Conversation()
    db.add(conversation)
    db.commit()
    conversation_id = conversation.id

    # ───────────────────────────────────────────────────────────
    # RETRIEVAL & PROMPT ASSEMBLY (Long_Term_Memory or low confidence)
    # ───────────────────────────────────────────────────────────
    result.prompt = user_message

    if (result.context_reliance == "Long_Term_Memory" or
            result.max_confidence < settings.confidence_fallback_threshold):

        # 1. Offload CPU‑bound embedding to a worker thread
        embedding_tensor = await asyncio.to_thread(
            classifier.embedder.encode, user_message, convert_to_tensor=False
        )
        prompt_embedding = (
            embedding_tensor.tolist()
            if hasattr(embedding_tensor, "tolist")
            else list(embedding_tensor)
        )

        orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)

        # 2. Offload synchronous PostgreSQL retrieval queries to a worker thread
        fragments = await asyncio.to_thread(
            orchestrator.retrieve,
            classification=result,
            conversation_id=str(conversation_id),
            prompt_embedding=prompt_embedding,
            scope=None
        )

        # 3. Safe database fetch for memory slots
        memory_slots = await asyncio.to_thread(
            lambda: db.query(MemorySlot).filter_by(is_active=True).all()
        )

        # Assemble final prompt
        messages = assemble_prompt(memory_slots, fragments, user_message,
                           db_session=db, conversation_id=str(conversation_id))

        logger.info(
            "context_injection_complete",
            injected_fragments=len(fragments),
            active_slots=len(memory_slots)
        )

    # ───────────────────────────────────────────────────────────
    # Stream from Ollama and schedule background storage
    # ───────────────────────────────────────────────────────────
    accumulated_raw_chunks = []

    async def generate():
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                ollama_url,
                json={"model": model_name, "messages": messages, "stream": True},
            ) as ollama_response:
                async for chunk in ollama_response.aiter_text():
                    accumulated_raw_chunks.append(chunk)
                    yield chunk

    background_tasks.add_task(
        store_turn_async,
        correlation_id=correlation_id,
        user_message=user_message,
        conversation_id=conversation_id,
        topic_tags=result.topic_tags,
        intent_tags=result.intent_tags,
        context_reliance=result.context_reliance,
        raw_stream_chunks=accumulated_raw_chunks,
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