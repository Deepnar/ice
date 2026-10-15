"""Post-flight evaluation — density qualification + representation decision.

Runs as the maintenance runtime's "post_flight" event job (gpu lane) whenever
a turn is stored, and chains codex + procedural extraction as direct calls in
the same job (C7). Retries/backoff and idle gating live in the runtime, not
here.
"""

import re
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.embedder import get_embedder
from src.memory.models import EpisodicMemory, IdempotencyKey
from src.memory.representation import choose_representation, verify_representations
from src.memory.summary_support import (
    SummaryCandidate, SummaryResult, assertion_instruction, assertion_text,
    METHOD_V3, generate_claims, original_units, source_windows, window_prompt,
)
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name, json_schema
from src.workers.codex_extractor import extract_codex
from src.workers.completion_text import complete_text
from src.workers.document_chunker import run_chunk_turn
from src.workers.idempotency import job_key
from src.workers.procedural_extractor import extract_procedural
from src.workers.turn_density import (
    compute_entropy,
    decide_representation,
    extract_key_terms,
    must_terms,
    retry_on_coverage_miss,
    summary_coverage,
)


logger = structlog.get_logger("ice.workers.post_flight")

bg_client = get_bg_client()
shared_embedder = get_embedder()

def _summary_llm_call(prompt: str, response: str, model_name: str,
                      terms: list, missing: list = None,
                      source_kind: str = "chat", source_title: str = "",
                      windows=None, max_tokens=None, schema=None,
                      abstract: bool = True) -> str:
    """One grounded summarisation call. *terms* is the MUST-PRESERVE list;
    *missing* (retry only) names the terms the first attempt dropped.

    C12: for a document section the instruction says so. Telling the model to
    "summarize the following user/assistant exchange" when it is looking at a
    page of a specification invites it to invent an exchange."""
    # Historical term-preservation wording, measured for plain E4B prose.
    # The 2026-10-08 cited JSON contract is separately qualified; the old
    # percentages below do not describe this new generation format.
    # Measured 2026-08-27, n=70 turns, one changed
    # substring, blind judge gated on planted defects first:
    #
    #   gemma4:e4b   hard "must appear verbatim"  15.7% fabricated, 82.9% faithful
    #   gemma4:e4b   the wording below             2.9% fabricated, 95.7% faithful
    #                                              incomplete UNCHANGED (1 -> 1)
    #
    # Why the hard form was worse: a model that cannot ground a must-term is
    # ordered to include it anyway, so it invents context to carry it — and
    # `summary_coverage` then scores it well for having done so. The order
    # manufactured the defect the metric rewarded.
    #
    # ⚠ THIS IS MODEL-COUPLED, exactly like the extraction prompt is coupled to
    # NuExtract3 (G63). The SAME change measured on `qwen3:4b-instruct` was a
    # LOSS — 54% -> 34% faithful, incomplete 7% -> 56%: that model converts the
    # freedom into omission rather than accuracy. **Changing the background
    # model means re-measuring this prompt, not editing a config value.**
    # Protocol + full tables: docs/specs/BG_LAYER_FIXES.md §0, §1.1.
    #
    # ⚠ Expect LOWER `summary_coverage` (0.784 -> 0.640). That is the metric
    # noticing the model stopped padding, not a regression — coverage is a
    # presence score and cannot see invention (TRAPS #52).
    must_block = ""
    if terms:
        must_block = (
            "\nKEY TERMS from this turn — preserve each one the turn actually "
            "supports, and simply omit any you cannot state faithfully. Never "
            f"introduce a fact to accommodate a term: {', '.join(terms)}\n"
        )
    retry_block = ""
    if missing:
        # ⚑ THE RETRY IS THE HARDER ORDER AND MUST MOVE WITH IT. It fired on
        # 38 of 70 turns in the measured run, so leaving "include each of them
        # verbatim this time" here would have kept the strongest form of the
        # instruction on more than half of all summaries.
        retry_block = (
            "\nYour previous summary omitted these key terms. Include any the "
            "turn genuinely supports; leave out any that would require "
            f"inventing a detail: {', '.join(missing)}\n"
        )
    if source_kind == "document":
        titled = f" from the document \"{source_title}\"" if source_title else ""
        what = f"Extract a compact memory note from the following excerpt{titled}. "
        noun = "excerpt"
    else:
        what = "Extract a compact memory note from the following user/assistant exchange. "
        noun = "exchange"
    system = (
        "You are a precise summarisation engine. " + what +
        "RULES:\n"
        "- Preserve ALL named entities (people, characters, places, tools, project names).\n"
        "- Preserve ALL numbers, dates, versions, and their specific assignments.\n"
        "- Preserve substantive facts, decisions, and their rationale.\n"
        "- Omit repeated background narration; do not describe the source or discussion process.\n"
        "- Do NOT include pleasantries, speculation, or meta-commentary.\n"
        f"- If the {noun} contains code, describe what the code does.\n"
        f"{must_block}{retry_block}"
        + assertion_instruction(abstract=abstract)
    )
    body = window_prompt(windows) if windows else (prompt if source_kind == "document"
                                                  else f"User: {prompt}\nAssistant: {response}")
    completion = bg_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ],
        temperature=0.0,
        # TRAPS #17: a literal here is a knob nobody can reach. The cap scales
        # with the piece's input (summary_support.output_cap); the setting is
        # its floor.
        max_tokens=max_tokens or settings.turn_summary_max_tokens,
        timeout=bg_timeout(max_tokens or settings.turn_summary_max_tokens),
        **({"response_format": json_schema("summary_claims", schema)} if schema else {}),
    )
    return complete_text(completion)


def _split_abstract(summary: str):
    """C3: pull the trailing 'Abstract:' line out of the generated summary.
    Returns (summary_without_abstract, abstract_or_None)."""
    m = re.search(r"(?im)^\s*abstract:\s*(.+)\s*$", summary or "")
    if not m:
        return summary, None
    body = (summary[:m.start()] + summary[m.end():]).strip()
    return body, m.group(1).strip()


def generate_summary(prompt: str, response: str, key_terms: dict,
                     model_used: str = "", source_kind: str = "chat",
                     source_title: str = "", source_row=None):
    """C1/C3 grounded summarisation: MUST-PRESERVE terms are injected into the
    prompt (ground-then-generate, like Codex A2), the result is *measured*
    (``summary_coverage``), one retry names any dropped terms, and the one-line
    abstract (hierarchy level 3) rides in the same call. Returns
    ``(summary_text, coverage, abstract)`` plus attached source proof when available;
    ("", 0.0, None) on failure so the
    caller's raw-wins fallback engages."""
    model_name = get_bg_model_name()
    terms = must_terms(key_terms)
    try:
        if source_row is None:
            from types import SimpleNamespace
            from src.memory.source import chat_provenance, single_provenance
            raw = prompt if source_kind == "document" else f"User: {prompt}\n\nAssistant: {response}"
            source_row = SimpleNamespace(raw_text=raw, source_spans=(
                single_provenance(raw, "document") if source_kind == "document"
                else chat_provenance(prompt, response)))
        windows = source_windows(original_units([source_row]))

        def claims(missing=None):
            # SUMMARY_CLAIMS_V3 S5: one call per piece, merged onto this turn.
            def call(piece, cap, schema, abstract):
                return _summary_llm_call(prompt, response, model_name, terms, missing,
                                         source_kind=source_kind, source_title=source_title,
                                         windows=piece, max_tokens=cap, schema=schema,
                                         abstract=abstract)
            parsed, _ = generate_claims(call, windows, floor=settings.turn_summary_max_tokens,
                                        abstract=True)
            parsed.setdefault("abstract", [])
            return SummaryCandidate(parsed, windows)

        summary = claims()
        # G29: shared with conversation_summary's fold (turn_density).
        summary, coverage = retry_on_coverage_miss(summary, key_terms, claims)
        proof = None
        if isinstance(summary, SummaryCandidate):
            proof = {**summary.assertions, "windows": summary.windows}
            abstract = assertion_text(summary.assertions["abstract"]) or None
            summary = str(summary)
        else:
            summary, abstract = _split_abstract(summary)
        # coverage was measured on the full text incl. the Key-terms line;
        # re-measure on the stored body so the gate reflects what's injected.
        coverage = summary_coverage(summary, key_terms)
        return SummaryResult(summary, coverage, abstract, proof)
    except Exception as exc:
        from src.workers.runtime import JobYielded

        if isinstance(exc, JobYielded):
            raise
        logger.warning("background_summarization_failed", error=str(exc))
        return "", 0.0, None


def evaluate_turn(batch_id: str, prompt: str, response: str,
                  conversation_id: str, model_used: str = "",
                  source_kind: str = "chat", source_title: str = ""):
    """Density qualification + representation decision, then the derivative
    pipelines (chunking, codex, procedural) as direct calls.

    Document sections retain lossless representation eligibility, and their
    summaries use a document prompt. Extraction is independent of that flag:
    every non-private turn can contribute facts, even a short correction.

    Idempotency layout (C7 rev 2026-07-11): the density/summary stage is
    guarded by this job's own key, but the chained stages run on every entry —
    each is self-idempotent (codex + procedural keep their own keys, the
    chunker checks for existing chunks), so a retry after a partial chain
    failure completes the missing stages instead of hitting an early return.
    Raising here is deliberate: the maintenance runtime owns backoff retries.
    """
    log = logger.bind(batch_id=batch_id, conversation_id=conversation_id)

    # G29: was a bare sha256(batch_id) — the un-namespaced key in a table shared
    # with every other job. Safe to re-namespace here because episodic_memory is
    # empty; on a populated store this orphans existing markers and re-processes
    # their turns, so a future change to this string needs that check first.
    idempotency_key = job_key("post_flight", batch_id)
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
            is_doc_source = source_kind == "document"
            full_text = (prompt if is_doc_source
                         else f"User: {prompt}\nAssistant: {response}")
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
            # Lossless controls representation and batch-summary eligibility.
            # Every non-private turn can carry facts, including short corrections.
            lossless = (is_doc_source or has_code or is_creative
                        or entropy >= settings.turn_density_lossless_threshold)

            decision = decide_representation(
                word_count=word_count, entropy=entropy, has_code=has_code,
                is_creative=is_creative, is_document=is_document,
            )
            inject_raw = decision["inject_raw"]
            summary, coverage, abstract = None, None, None
            if decision["want_summary"]:
                generated = generate_summary(
                    prompt, response, key_terms, model_used,
                    source_kind=source_kind, source_title=source_title, source_row=turn)
                summary, coverage, abstract = generated
                summary = summary or None
                if decision["summary_decides"]:
                    # Coverage is only a retention prerequisite. Independent
                    # source support below controls actual substitution.
                    inject_raw = not (summary and coverage >= settings.turn_summary_coverage_threshold)
                log.info(
                    "summary_quality",
                    coverage=coverage,
                    coverage_candidate="summary" if not inject_raw else "raw",
                    reason=decision["reason"],
                    must_terms=len(must_terms(key_terms)),
                )

            turn.entropy_score = entropy
            turn.is_document = is_document
            turn.lossless_flag = lossless
            turn.summary_text = summary
            turn.summary_coverage = coverage if summary else None
            turn.abstract_text = abstract if summary else None
            proof = getattr(generated, "proof", None) if decision["want_summary"] else None
            if proof is None:
                turn.representation_verification = verify_representations(turn, summary, turn.abstract_text)
            else:
                turn.representation_verification = verify_representations(
                    turn, summary, turn.abstract_text, proof=proof)
                records = turn.representation_verification or {}
                if (records.get("summary") or {}).get("method") == METHOD_V3:
                    # Per-claim acceptance: the stored text is exactly the
                    # supported claims, so coverage and the raw/summary choice
                    # are re-decided on what is actually kept.
                    kept = assertion_text(records["summary"]["assertions"]) or None
                    summary = kept
                    coverage = summary_coverage(kept, key_terms) if kept else None
                    turn.summary_text, turn.summary_coverage = kept, coverage
                    if decision["summary_decides"]:
                        inject_raw = not (kept and coverage >= settings.turn_summary_coverage_threshold)
                    abstract_record = records.get("abstract") or {}
                    turn.abstract_text = (assertion_text(abstract_record.get("assertions") or [])
                                          or None) if kept else None
                    log.info("summary_claims_kept", kept=len(records["summary"]["assertions"]),
                             dropped=len(records["summary"].get("dropped") or []),
                             coverage=coverage)
            turn.inject_raw = inject_raw
            if not inject_raw:
                preferred, _, _ = choose_representation(turn)
                turn.inject_raw = preferred != summary
            inject_raw = turn.inject_raw
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
        if turn.is_document or len((turn.raw_text or "").split()) > settings.turn_long_turn_chunk_words:
            run_chunk_turn(db, turn)

        # G16 incognito: private turns keep their per-turn evaluation (summary/
        # lossless are internal to the row) but never feed the shared stores —
        # no codex entities, no procedural patterns (clustering/batch-summary/
        # reflection skip them on their own side).
        if turn.is_private:
            log.info("private_turn_pipelines_skipped")
        else:
            # Independent stages have their own sessions/completion keys. Let
            # both make progress, then propagate failure for the runtime retry.
            # Never turn a partial chain into a successful completion.
            from src.workers.runtime import JobYielded

            failures = []
            for stage_name, extractor in (("codex", extract_codex),
                                          ("procedural", extract_procedural)):
                try:
                    extractor(batch_id=batch_id, model_used=model_used)
                except JobYielded:
                    raise
                except Exception as exc:
                    log.warning("post_flight_stage_failed", stage=stage_name,
                                error_type=type(exc).__name__)
                    failures.append(exc)
            # E8: project-attached turns also run cue-gated decision
            # extraction — a direct call in this gpu job (C7 chain style);
            # the cue check inside is deterministic, so no-cue turns cost
            # nothing (A6: no cue, no LLM).
            project_id = db.execute(
                text("SELECT project_id FROM conversations WHERE id = :cid"),
                {"cid": turn.conversation_id}).scalar()
            if project_id:
                from src.workers.decision_extractor import run_decision_extraction
                run_decision_extraction(db, project_id=project_id,
                                        text_content=turn.raw_text,
                                        source_batch=turn.batch_id,
                                        origin="turn")
            if failures:
                raise failures[0]

    except Exception as exc:
        from src.workers.runtime import JobYielded

        db.rollback()
        if not isinstance(exc, JobYielded):
            log.error("worker_transaction_execution_failure", error=str(exc))
        raise
    finally:
        db.close()
