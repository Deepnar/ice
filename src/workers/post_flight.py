"""Post-flight evaluation — density qualification + representation decision.

Runs as the maintenance runtime's "post_flight" event job (gpu lane) whenever
a turn is stored, and chains codex + procedural extraction as direct calls in
the same job (C7). Retries/backoff and idle gating live in the runtime, not
here.
"""

import hashlib
import re
import uuid
from datetime import datetime, timezone

import structlog

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, IdempotencyKey
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name
from src.workers.codex_extractor import embedder as shared_embedder
from src.workers.codex_extractor import extract_codex
from src.workers.document_chunker import run_chunk_turn
from src.workers.procedural_extractor import extract_procedural
from src.workers.turn_density import (
    DENSITY_LOSSLESS_THRESHOLD,
    LONG_TURN_CHUNK_WORDS,
    SUMMARY_COVERAGE_THRESHOLD,
    compute_entropy,
    decide_representation,
    extract_key_terms,
    must_terms,
    summary_coverage,
)

logger = structlog.get_logger("ice.workers.post_flight")

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
        "- End with two final lines formatted exactly as:\n"
        "  Key terms: <comma-separated list of the named entities, figures, and "
        "identifiers that appear in the exchange>\n"
        "  Abstract: <ONE sentence, max ~20 words, stating what this exchange "
        "was about>"
    )
    completion = bg_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"User: {prompt}\nAssistant: {response}"},
        ],
        temperature=0.0,
        max_tokens=300,
        timeout=bg_timeout(300),
    )
    return completion.choices[0].message.content.strip()


def _split_abstract(summary: str):
    """C3: pull the trailing 'Abstract:' line out of the generated summary.
    Returns (summary_without_abstract, abstract_or_None)."""
    m = re.search(r"(?im)^\s*abstract:\s*(.+)\s*$", summary or "")
    if not m:
        return summary, None
    body = (summary[:m.start()] + summary[m.end():]).strip()
    return body, m.group(1).strip()


def generate_summary(prompt: str, response: str, key_terms: dict,
                     model_used: str = ""):
    """C1/C3 grounded summarisation: MUST-PRESERVE terms are injected into the
    prompt (ground-then-generate, like Codex A2), the result is *measured*
    (``summary_coverage``), one retry names any dropped terms, and the one-line
    abstract (hierarchy level 3) rides in the same call. Returns
    ``(summary_text, coverage, abstract)`` — ("", 0.0, None) on failure so the
    caller's raw-wins fallback engages."""
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
        summary, abstract = _split_abstract(summary)
        # coverage was measured on the full text incl. the Key-terms line;
        # re-measure on the stored body so the gate reflects what's injected.
        coverage = summary_coverage(summary, key_terms)
        return summary, coverage, abstract
    except Exception as exc:
        logger.error("background_summarization_failed", error=str(exc))
        return "", 0.0, None


def evaluate_turn(batch_id: str, prompt: str, response: str,
                  conversation_id: str, model_used: str = ""):
    """Density qualification + representation decision, then the derivative
    pipelines (chunking, codex, procedural) as direct calls.

    Idempotency layout (C7 rev 2026-07-11): the density/summary stage is
    guarded by this job's own key, but the chained stages run on every entry —
    each is self-idempotent (codex + procedural keep their own keys, the
    chunker checks for existing chunks), so a retry after a partial chain
    failure completes the missing stages instead of hitting an early return.
    Raising here is deliberate: the maintenance runtime owns backoff retries.
    """
    log = logger.bind(batch_id=batch_id, conversation_id=conversation_id)

    idempotency_key = hashlib.sha256(batch_id.encode()).hexdigest()
    db = SessionLocal()

    try:
        # Defensive record fetch (the enqueue happens after the storing
        # commit, so visibility lag is near-impossible now; a raise lets the
        # runtime's backoff handle the freak case).
        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn:
            raise RuntimeError(f"turn for batch {batch_id} not visible yet")

        already_evaluated = db.query(IdempotencyKey).filter_by(
            key=idempotency_key).first() is not None

        if already_evaluated:
            log.info("task_execution_skipped_idempotent")
        else:
            # Density evaluation + representation decision (C1). Replaces the
            # crude is_lossless heuristic (≥3 capitalized words fired on
            # nearly everything) and the backwards branch that summarised the
            # DENSEST long turns into 2-3 sentences while leaving banter raw.
            full_text = f"User: {prompt}\nAssistant: {response}"
            word_count = len(full_text.split())
            has_code = "```" in response
            is_creative = bool(
                (turn.topic_tags and "Creative_&_Media" in turn.topic_tags) or
                (turn.intent_tags and "Emotional_Processing" in turn.intent_tags)
            )
            # ML4: document turns (long, low conversational density). Folded
            # into the decision matrix — the old code set turn.inject_raw here
            # and then overwrote it two lines later (documents silently lost
            # raw injection).
            raw_words = len(turn.raw_text.split())
            assistant_count = turn.raw_text.count("Assistant:")
            is_document = raw_words > 2000 and assistant_count < 3

            key_terms = extract_key_terms(full_text, shared_embedder)
            entropy = compute_entropy(full_text, key_terms, has_code)
            # lossless = "valuable enough for lossless treatment" — gates codex
            # extraction below and exempts from batch summarisation. Generous
            # by design (codex was historically starved).
            lossless = has_code or is_creative or entropy >= DENSITY_LOSSLESS_THRESHOLD

            decision = decide_representation(
                word_count=word_count, entropy=entropy, has_code=has_code,
                is_creative=is_creative, is_document=is_document,
            )
            inject_raw = decision["inject_raw"]
            summary, coverage, abstract = None, None, None
            if decision["want_summary"]:
                summary, coverage, abstract = generate_summary(prompt, response, key_terms, model_used)
                summary = summary or None
                if decision["summary_decides"]:
                    # The retrievability gate: inject the summary only if it
                    # measurably preserved the key terms — else raw wins and
                    # the summary remains as metadata.
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
            turn.abstract_text = abstract if summary else None
            turn.inject_raw = inject_raw
            log.info(
                "representation_decided",
                entropy=entropy, lossless=lossless, inject_raw=inject_raw,
                reason=decision["reason"], words=word_count,
            )

            db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
            db.commit()
            log.info("post_flight_evaluation_complete", lossless=lossless)

        # ── Chained stages (direct calls, one job — C7) ──
        # C2/C3: documents AND all long turns get chunked for chunk-level
        # retrieval (also for private turns — chunks are turn-local and
        # inherit visibility through the parent join, and the conversation
        # needs them for self-retrieval). run_chunk_turn is a no-op when
        # chunks already exist.
        if turn.is_document or len((turn.raw_text or "").split()) > LONG_TURN_CHUNK_WORDS:
            run_chunk_turn(db, turn)

        # G16 incognito: private turns keep their per-turn evaluation (summary/
        # lossless are internal to the row) but never feed the shared stores —
        # no codex entities, no procedural patterns (clustering/batch-summary/
        # reflection skip them on their own side).
        if turn.is_private:
            log.info("private_turn_pipelines_skipped")
        else:
            if turn.lossless_flag:
                extract_codex(batch_id=batch_id, model_used=model_used)
            extract_procedural(batch_id=batch_id, model_used=model_used)

    except Exception as exc:
        db.rollback()
        log.error("worker_transaction_execution_failure", error=str(exc))
        raise
    finally:
        db.close()
