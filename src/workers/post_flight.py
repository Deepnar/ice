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
from src.workers.codex_extractor import extract_codex, embedder as shared_embedder
from src.workers.procedural_extractor import extract_procedural
from src.workers.turn_density import (
    extract_key_terms, compute_entropy, decide_representation,
    summary_coverage, must_terms,
    DENSITY_LOSSLESS_THRESHOLD, SUMMARY_COVERAGE_THRESHOLD,
)


logger = structlog.get_logger("ice.workers.post_flight")

# Dedicated backend inference client targeting isolated LLM instance
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
bg_client = get_bg_client()

def _summary_llm_call(prompt: str, response: str, model_name: str,
                      terms: list, missing: list = None) -> str:
    """One grounded summarisation call. *terms* is the MUST-PRESERVE list;
    *missing* (retry only) names the terms the first attempt dropped."""
    must_block = ""
    if terms:
        must_block = (
            "\nMUST-PRESERVE TERMS (every one of these must appear verbatim in "
            f"your summary): {', '.join(terms)}\n"
        )
    retry_block = ""
    if missing:
        retry_block = (
            "\nYour previous summary DROPPED these required terms — include "
            f"each of them verbatim this time: {', '.join(missing)}\n"
        )
    system = (
        "You are a precise summarisation engine. "
        "Summarize the following user/assistant exchange in 4-6 sentences. "
        "RULES:\n"
        "- Preserve ALL named entities (people, characters, places, tools, project names).\n"
        "- Preserve ALL numbers, dates, versions, and their specific assignments.\n"
        "- Preserve ALL specific facts, decisions, and their rationale.\n"
        "- Do NOT include pleasantries, speculation, or meta-commentary.\n"
        "- If the exchange contains code, describe what the code does.\n"
        f"{must_block}{retry_block}"
        "- End with one final line formatted exactly as:\n"
        "  Key terms: <comma-separated list of the named entities, figures, and "
        "identifiers that appear in the exchange>"
    )
    completion = bg_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"User: {prompt}\nAssistant: {response}"},
        ],
        temperature=0.0,
        max_tokens=300,
        timeout=30.0,
    )
    return completion.choices[0].message.content.strip()


def generate_summary(prompt: str, response: str, key_terms: dict,
                     model_used: str = ""):
    """C1 grounded summarisation: MUST-PRESERVE terms are injected into the
    prompt (ground-then-generate, like Codex A2), the result is *measured*
    (``summary_coverage``), and one retry names any dropped terms. Returns
    ``(summary_text, coverage)`` — ("", 0.0) on failure so the caller's
    raw-wins fallback engages."""
    model_name = model_used if model_used else get_bg_model_name()
    terms = must_terms(key_terms)
    try:
        summary = _summary_llm_call(prompt, response, model_name, terms)
        coverage = summary_coverage(summary, key_terms)
        if coverage < SUMMARY_COVERAGE_THRESHOLD and terms:
            low = summary.lower()
            missing = [t for t in terms if t.lower() not in low]
            retry = _summary_llm_call(prompt, response, model_name, terms, missing)
            retry_cov = summary_coverage(retry, key_terms)
            if retry_cov > coverage:
                summary, coverage = retry, retry_cov
        return summary, coverage
    except Exception as exc:
        logger.error("background_summarization_failed", error=str(exc))
        return "", 0.0


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

        # 4. Density evaluation + representation decision (C1).
        # Replaces the crude is_lossless heuristic (≥3 capitalized words fired
        # on nearly everything) and the backwards branch that summarised the
        # DENSEST long turns into 2-3 sentences while leaving banter raw.
        full_text = f"User: {prompt}\nAssistant: {response}"
        word_count = len(full_text.split())
        has_code = "```" in response
        is_creative = bool(
            (turn.topic_tags and "Creative_&_Media" in turn.topic_tags) or
            (turn.intent_tags and "Emotional_Processing" in turn.intent_tags)
        )
        # ML4: document turns (long, low conversational density). Folded into
        # the decision matrix — the old code set turn.inject_raw here and then
        # overwrote it two lines later (documents silently lost raw injection).
        raw_words = len(turn.raw_text.split())
        assistant_count = turn.raw_text.count("Assistant:")
        is_document = raw_words > 2000 and assistant_count < 3

        key_terms = extract_key_terms(full_text, shared_embedder)
        entropy = compute_entropy(full_text, key_terms, has_code)
        # lossless = "valuable enough for lossless treatment" — gates codex
        # extraction below and exempts from batch summarisation. Generous by
        # design (codex was historically starved).
        lossless = has_code or is_creative or entropy >= DENSITY_LOSSLESS_THRESHOLD

        decision = decide_representation(
            word_count=word_count, entropy=entropy, has_code=has_code,
            is_creative=is_creative, is_document=is_document,
        )
        inject_raw = decision["inject_raw"]
        summary, coverage = None, None
        if decision["want_summary"]:
            summary, coverage = generate_summary(prompt, response, key_terms, model_used)
            summary = summary or None
            if decision["summary_decides"]:
                # The retrievability gate: inject the summary only if it
                # measurably preserved the key terms — else raw wins and the
                # summary remains as metadata.
                inject_raw = not (summary and coverage >= SUMMARY_COVERAGE_THRESHOLD)
            log.info(
                "summary_quality",
                coverage=coverage,
                injected="summary" if not inject_raw else "raw",
                reason=decision["reason"],
                must_terms=len(must_terms(key_terms)),
            )

        turn.entropy_score = entropy
        turn.is_document = is_document
        turn.lossless_flag = lossless
        turn.summary_text = summary
        turn.summary_coverage = coverage if summary else None
        turn.inject_raw = inject_raw
        log.info(
            "representation_decided",
            entropy=entropy, lossless=lossless, inject_raw=inject_raw,
            reason=decision["reason"], words=word_count,
        )
        
        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("post_flight_evaluation_complete", lossless=lossless)

        # G16 incognito: private turns keep their per-turn evaluation (summary/
        # lossless are internal to the row) but never feed the shared stores —
        # no codex entities, no procedural patterns (clustering/batch-summary/
        # reflection skip them on their own side).
        if turn.is_private:
            log.info("private_turn_pipelines_skipped")
        else:
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