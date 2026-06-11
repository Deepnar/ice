"""Post-Flight Evaluation Celery Worker Node."""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import structlog

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, IdempotencyKey
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy, is_user_active
from src.workers.codex_extractor import extract_codex
from src.workers.procedural_extractor import extract_procedural


logger = structlog.get_logger("ice.workers.post_flight")

# Dedicated backend inference client targeting isolated LLM instance
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
bg_client = get_bg_client()

def is_lossless(text: str) -> bool:
    """Analyzes text density to resolve the Asymmetrical Value Problem.
    
    Ensures true proper nouns are tracked while filtering out standard sentence starts.
    """
    if "```" in text:
        return True

    if len(text.split()) > 500:
        return True

    # Strip out standard sentence boundaries (. ! ?) followed by whitespace and capitals
    # to avoid false positives on standard sentence starts
    cleaned_text = re.sub(r'(?:^[A-Z]|\b[\.\!\?]\s+[A-Z])[a-z]+\b', '', text)
    
    # Track true remaining capitalized words (e.g., inline proper nouns)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', cleaned_text)
    if len(proper_nouns) >= 3:
        return True

    return False


def generate_summary(prompt: str, response: str, model_used: str = "") -> str:
    """Invokes the background 3B model to produce a tight, fact‑dense summary."""
    try:
        completion = bg_client.chat.completions.create(
            model = model_used if model_used else get_bg_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise summarisation engine. "
                        "Summarize the following user/assistant exchange in 2‑3 sentences. "
                        "Focus ONLY on concrete facts, decisions, code snippets, or named entities. "
                        "Do NOT include pleasantries, speculation, or meta‑commentary. "
                        "If the exchange contains code, describe what the code does."
                    )
                },
                {"role": "user", "content": f"User: {prompt}\nAssistant: {response}"},
            ],
            temperature=0.0,
            max_tokens=200,
            timeout=30.0,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("background_summarization_failed", error=str(exc))
        return ""


@app.task(bind=True, max_retries=5, default_retry_delay=15)
def evaluate_turn(self, batch_id: str, prompt: str, response: str, conversation_id: str, model_used: str = ""):
    """Executes structural density qualification and data post-processing routines."""
    log = logger.bind(batch_id=batch_id, conversation_id=conversation_id)

    # 1. Active GPU Resource Gate (INV-5)
    if is_gpu_busy():
        log.info("gpu_saturation_yielding", message="Rescheduling worker target thread.")
        raise self.retry(countdown=15)
    if settings.background_model_mode == "shared" and is_user_active():
        raise self.retry(countdown=30)

    # 2. Border Idempotency Verification (INV-6)
    idempotency_key = hashlib.sha256(batch_id.encode()).hexdigest()
    db = SessionLocal()
    
    try:
        existing = db.query(IdempotencyKey).filter_by(key=idempotency_key).first()
        if existing:
            log.info("task_execution_skipped_idempotent")
            return

        # 3. Defensive Record Fetching (Mitigates DB Commit Race Condition)
        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn:
            log.warn("record_visibility_lag_retry")
            raise self.retry(countdown=5)  # Back off briefly to let the API commit finish

        # 4. Text Ingestion & Processing
                # 4. Text Ingestion & Processing
        lossless = is_lossless(response)
        full_text = f"User: {prompt}\nAssistant: {response}"
        word_count = len(full_text.split())
        has_code = "```" in response

        # By default, inject raw text
        inject_raw = True

        if lossless and word_count > 500 and not has_code:
            # Long lossless turn without code → summarise and mark for summary injection
            summary = generate_summary(prompt, response, model_used)
            turn.summary_text = summary
            inject_raw = False
        elif not lossless:
            # Non‑lossless turn → summarise normally
            summary = generate_summary(prompt, response, model_used)
            turn.summary_text = summary
            inject_raw = False

        # 5. Core Write-Once Persistence
        turn.lossless_flag = lossless
        turn.inject_raw = inject_raw
        
        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("post_flight_evaluation_complete", lossless=lossless)
        if lossless:
            extract_codex.delay(batch_id=batch_id, model_used=model_used)
        
        extract_procedural.delay(batch_id=batch_id, model_used=model_used)

    except Exception as exc:
        db.rollback()
        log.error("worker_transaction_execution_failure", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()

    # Pipeline Trigger Event Stub
    print(f"BATCH_PROCESSED: {batch_id}")