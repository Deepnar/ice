"""Procedural Extractor – identifies recurring behavioural patterns."""

import re
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.embedder import get_embedder
from src.memory.models import EpisodicMemory, IdempotencyKey, ProceduralMemory, StoreMeta
from src.memory.pattern_support import equivalence_receipts
from src.memory.procedural_support import pattern_sources, proof_current, raise_unknown, review_sources, supported_ids
from src.memory.source import source_units
from src.memory.tokens import count_messages, with_margin
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name
from src.workers.completion_text import IncompleteCompletion, complete_text
from src.workers.idempotency import job_key


logger = structlog.get_logger("ice.workers.procedural")
bg_client = get_bg_client()
# The process-shared native-width embedder (G13/G23).
pattern_embedder = get_embedder()

def encode_pattern(text: str):
    return pattern_embedder.encode(text, convert_to_tensor=False).tolist()


def _parse_pattern(reply: str):
    """Split ``PATTERN: … | EVIDENCE: 1, 3`` into (text, {1, 3}).

    Tolerant of a model that ignores the format: if no EVIDENCE clause is
    found the evidence set comes back empty, and the caller rejects it. That is
    the safe direction — an unparsed reply must not become a pattern that looks
    corroborated.
    """
    if not reply:
        return "", set()
    text_part, evidence = reply, set()
    m = re.search(r"EVIDENCE\s*:\s*([0-9,\s]+)", reply, re.IGNORECASE)
    if m:
        evidence = {int(n) for n in re.findall(r"\d+", m.group(1))}
        text_part = reply[:m.start()]
    text_part = re.sub(r"^\s*PATTERN\s*:\s*", "", text_part.strip(),
                       flags=re.IGNORECASE)
    return text_part.strip().rstrip("|").strip(), evidence


def _user_text(turn: EpisodicMemory) -> str:
    """Use writer-attributed user spans; an unknown speaker is not evidence."""
    return "\n".join(u.text.strip() for u in source_units(turn)
                     if u.role == "user" and u.text.strip())


def _source_session_ids(db, batch_ids: set) -> set | None:
    """Resolve *every* cited source; missing/ambiguous provenance is unknown."""
    if not batch_ids:
        return None
    rows = db.execute(text("""
        SELECT batch_id, session_id FROM (
            SELECT batch_id, session_id FROM episodic_memory WHERE batch_id = ANY(:ids)
            UNION ALL
            SELECT batch_id, session_id FROM cold_storage WHERE batch_id = ANY(:ids)
        ) sources
    """), {"ids": list(batch_ids)}).fetchall()
    sessions_by_batch = {}
    for batch, session in rows:
        if session is None or (batch in sessions_by_batch and sessions_by_batch[batch] != session):
            return None
        sessions_by_batch[batch] = session
    if set(sessions_by_batch) != batch_ids:
        return None
    return set(sessions_by_batch.values())


def pattern_prompt(numbered: str, count: int) -> str:
    """Audit A10 (2026-10-15). A procedural memory is an answer preference the
    assistant should follow (ICE_Architecture §3.3: "always asks for tests
    after code"). Qualified on 25 real blocks (dev session + history B,
    logs/audit-a10-procedural-prompt-2026-10-15-r1/): the earlier wording gave
    "the user seeks validation/feedback" on ~16/25 and personality notes on 3
    ("feelings of inadequacy" reached a store); a negative rule ("validation is
    NOT a habit") was ignored by gemma4:e4b (~12/25); a positive target with
    example sentences had the model copy an example 8/25, at least once with
    citations that did not show it. This positive, example-free wording gave
    0 validation, 0 personality and 0 copies — all answer preferences, though
    often the broad "prefers detailed, comprehensive answers" (13/25)."""
    return (
        f"Below are {count} messages the SAME user wrote during one session, in order.\n\n"
        f"{numbered}\n\n"
        "Find ONE answer preference this user shows in two or more messages: how they want the "
        "assistant's answers shaped. It must be about the answer's FORMAT (list, table, code, "
        "prose), STRUCTURE (summary first, consolidated version, sections), LENGTH or DETAIL, "
        "ORDER of steps, TOOLS, or LANGUAGE. Describe it in your own words from what the "
        "cited messages actually ask for.\n\n"
        "Rules:\n"
        "- It MUST be visible in two or more of the numbered messages.\n"
        "- Write it in this exact format: PATTERN: The user wants <answer preference> | "
        "EVIDENCE: <comma-separated message numbers>\n"
        "- Only an answer preference counts. If the messages show none, output exactly: NONE\n"
        "Output nothing else."
    )


def cited_topics(session_turns, cited_batch_ids) -> list:
    """Topic tags of the cited turns: the pattern's relevance trigger.

    `_procedural_lookup` already filters on trigger_conditions.topic_tags, but
    nothing ever wrote them, so every pattern matched every prompt (A10)."""
    cited = {str(b) for b in cited_batch_ids}
    return sorted({tag for t in session_turns if str(t.batch_id) in cited
                   for tag in (t.topic_tags or [])})


def extract_procedural(batch_id: str, model_used: str = ""):
    """Identify a habit from a SESSION of the user's prompts.

    **G42 (2026-08-12): this used to read one turn and ask for a *recurring*
    pattern.** One turn cannot evidence recurrence, so the model invented it
    every time — all 247 patterns in the measured store were single-turn
    restatements wrapped in "the user consistently…", and because one-per-turn
    generation makes near-duplicates rather than genuine repeats, only 2 ever
    reached the reinforcement threshold that promotes a pattern to active. The
    mechanism was not weak; it was never given the input it needs.

    Three things follow, and each is deliberate:

    * **A session, not a turn.** A habit is visible across a sitting.
    * **The user's prompts only.** A habit is a property of what the *user*
      does; the assistant's replies are most of the tokens and none of the
      signal.
    * **Reinforcement counts SESSIONS, not extractions.** Re-reading one
      session as it grows must not look like a repeat, or the count inflates
      itself and promotion becomes meaningless again.

    Plain callable since C7 — gating/retries live in the maintenance runtime.
    """
    log = logger.bind(batch_id=batch_id)

    db = SessionLocal()
    try:
        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn:
            return
        if turn.is_private or turn.is_document:
            log.warning("procedural_skipped_ineligible_source",
                        private=bool(turn.is_private), document=bool(turn.is_document))
            return

        # A row with no session cannot be reasoned about as a sitting. Say so at
        # WARNING with the reason — a silent skip here would read as "no habits
        # found" forever (CLAUDE.md: a silent fallback hides an outage).
        if not turn.session_id:
            log.warning("procedural_skipped_no_session",
                        reason="episodic row has no session_id; procedural "
                               "extraction is session-scoped since G42",
                        episodic_id=str(turn.id))
            return

        session_turns = (
            db.query(EpisodicMemory)
            .filter(EpisodicMemory.session_id == turn.session_id,
                    EpisodicMemory.conversation_id == turn.conversation_id,
                    EpisodicMemory.is_private == False,  # noqa: E712
                    EpisodicMemory.is_document == False)  # noqa: E712
            .order_by(EpisodicMemory.timestamp)
            .all()
        )
        n_turns = len(session_turns)
        if n_turns < settings.procedural_min_session_turns:
            log.info("procedural_session_too_short", turns=n_turns,
                     needed=settings.procedural_min_session_turns,
                     session_id=str(turn.session_id))
            return

        # Re-extract as the session grows, but in steps: keyed per turn this
        # would be one model call per turn again, which is the cost G42 removes.
        bucket = n_turns // max(1, settings.procedural_session_step)
        idempotency_key = job_key("procedural", f"{turn.session_id}:{bucket}")
        if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
            # ⚑ SAY SO. This returned silently until 2026-08-27, and the cost was
            # a diagnosis, not an outage: deleting every `procedural_memory` row
            # changed nothing — the KEYS survive — and the job kept declining to
            # work at DEBUG with **no output whatsoever**. Correct behaviour,
            # invisible reason. CLAUDE.md's standing rule is that a component
            # substituting a default for a real answer emits at WARNING with the
            # reason, every time; INFO is right here because this is the designed
            # path rather than a fallback, but it must not be mute.
            log.info("procedural_already_extracted", session_id=str(turn.session_id),
                     bucket=bucket, turns=n_turns,
                     note="session unchanged since last extraction at this step")
            return
        # E1 (D1): a pattern observed inside a project-attached conversation
        # is a project convention — scoped by project_id, not a fourth store.
        project_id = db.execute(
            text("SELECT project_id FROM conversations WHERE id = :cid"),
            {"cid": turn.conversation_id}).scalar()

        # The user's own prompts, newest-last, trimmed from the FRONT so the
        # most recent behaviour survives the cap.
        user_prompts = [(t.batch_id, _user_text(t)) for t in session_turns]
        user_prompts = [(bid, p) for bid, p in user_prompts if bid and p]
        if len(user_prompts) < settings.procedural_min_session_turns:
            log.info("procedural_too_few_user_prompts", found=len(user_prompts),
                     needed=settings.procedural_min_session_turns)
            return
        block, total = [], 0
        for bid, p in reversed(user_prompts):
            if total + len(p) > settings.procedural_max_prompt_chars and block:
                break
            block.append((bid, p))
            total += len(p)
        block.reverse()
        numbered = "\n".join(f"{i}. {p}" for i, (_, p) in enumerate(block, 1))

        prompt = pattern_prompt(numbered, len(block))
        model_name = get_bg_model_name()
        messages = [
            {"role": "system", "content": "You are a behavioural pattern detector."},
            {"role": "user", "content": prompt},
        ]
        from src.model_registry.registry import get_model_context_window
        from src.model_registry.runtime_probe import serving_window
        window = int(settings.ollama_num_ctx_max)
        if settings.background_model_mode == "shared":
            observed = serving_window(model_name, get_model_context_window(model_name))
            if observed:
                window = min(window, int(observed)) if window > 0 else int(observed)
        output_cap = settings.procedural_extraction_max_tokens
        required = with_margin(count_messages(messages), settings.token_count_safety_margin) + output_cap
        if window <= 0 or required > window:
            log.warning("procedural_input_over_budget", required=required, window=window, model=model_name)
            raise IncompleteCompletion("complete procedural input exceeds known capacity")
        completion = bg_client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.0,
            max_tokens=output_cap, timeout=bg_timeout(output_cap),
        )
        raw_reply = complete_text(completion)
        if raw_reply.upper() == "NONE":
            # Empty success is not failure. Record this bucket without adding
            # pattern support, so unchanged negative results are not re-read.
            db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
            db.commit()
            log.info("procedural_no_repeated_pattern", session_id=str(turn.session_id),
                     bucket=bucket, session_turns=n_turns)
            return

        pattern_text, evidence = _parse_pattern(raw_reply)
        if not pattern_text:
            log.warning("procedural_unparsable_reply", reply_chars=len(raw_reply))
            return
        # The evidence requirement is the whole point: a pattern resting on one
        # message is the single-turn fabrication under a new name. Enforced
        # here rather than trusted to the prompt, because a prompt is a request
        # and this is a rule.
        if len(evidence) < 2 or any(n < 1 or n > len(block) for n in evidence):
            log.warning("procedural_rejected_invalid_evidence",
                        cited_count=len(evidence), shown_count=len(block),
                        session_id=str(turn.session_id))
            return
        cited_batch_ids = {block[n - 1][0] for n in evidence}
        if len(cited_batch_ids) < 2:
            log.warning("procedural_rejected_duplicate_sources",
                        cited_count=len(cited_batch_ids),
                        session_id=str(turn.session_id))
            return

        sources = pattern_sources(db, cited_batch_ids)
        if sources is None:
            raise IncompleteCompletion('procedural cited originals are unresolved')
        source_proof = review_sources(pattern_text, sources)
        review_key = 'procedural_review:' + idempotency_key
        review = db.get(StoreMeta, review_key)
        if review is None:
            review = StoreMeta(key=review_key)
            db.add(review)
        review.value = dict(proposal=pattern_text, session_id=str(turn.session_id), bucket=bucket,
                            proof=source_proof)
        if source_proof['status'] != 'supported':
            if source_proof['status'] == 'unsupported':
                db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
                db.commit()
                log.warning('procedural_rejected_semantic_evidence', cited_count=len(cited_batch_ids),
                            supported_count=len(supported_ids(source_proof)))
                return
            # A diagnostic receipt is not a false completion. Preserve bounded
            # runtime retries for a verifier outage/truncation/bounds failure.
            db.commit()
            raise_unknown(source_proof)
        cited_batch_ids = {uuid.UUID(b) for b in supported_ids(source_proof)}

        # Similarity nominates a bounded pool; it never proves equivalence.
        embedding = encode_pattern(pattern_text)

        # Force PostgreSQL to accept the list as a vector via explicit cast
        similarity_query = text("""
            SELECT id, 1 - (embedding <=> CAST(:emb AS vector)) AS sim
            FROM procedural_memory
            WHERE embedding IS NOT NULL
              AND project_id IS NOT DISTINCT FROM CAST(:project_id AS uuid)
            ORDER BY sim DESC, id LIMIT :candidate_limit
        """)
        candidates = db.execute(similarity_query, {
            "emb": str(embedding), "project_id": str(project_id) if project_id else None,
            "candidate_limit": settings.procedural_match_candidates,
        }).all()
        existing_patterns = [db.get(ProceduralMemory, row.id) for row in candidates]
        proofs = equivalence_receipts(pattern_text, [p.pattern_description for p in existing_patterns])
        checks = [dict(candidate_id=str(row.id), cosine=float(row.sim), proof=proof)
                  for row, proof in zip(candidates, proofs)]
        for candidate, proof in zip(candidates, proofs):
            log.info('procedural_equivalence', candidate_id=str(candidate.id),
                     cosine=float(candidate.sim), status=proof['status'], reason=proof['reason'])
        matched = None
        for p, proof in zip(existing_patterns, proofs):
            if proof['status'] != 'equivalent':
                continue
            prior_sources = pattern_sources(db, set(p.source_batch_ids or []))
            if prior_sources is None or not proof_current((p.trigger_conditions or {}).get('_source_support'),
                                                          p.pattern_description, prior_sources):
                log.warning('procedural_prior_source_unqualified', candidate_id=str(p.id))
                continue
            matched = p, proof
            break

        if matched is not None:
            existing, proof = matched
            # ⚑ Only a DIFFERENT session is a repeat. Re-reading this one as it
            # grows (see the bucket key above) would otherwise reinforce the
            # pattern against itself, and `reinforcement_count >= 3` — the whole
            # promotion gate — would be satisfied by one sitting read three
            # times. Recurrence has to cross sittings or it is not recurrence.
            seen = set(existing.source_batch_ids or [])
            seen_sessions = _source_session_ids(db, seen)
            if seen_sessions is not None and turn.session_id not in seen_sessions:
                existing.last_observed = datetime.now(timezone.utc)
                existing.reinforcement_count += 1
                existing.source_batch_ids = sorted(seen | cited_batch_ids, key=str)
                for group in source_proof['observations']:
                    group['equivalence'] = proof
                old_proof = existing.trigger_conditions['_source_support']
                combined_proof = {**old_proof,
                    'observations': old_proof['observations'] + source_proof['observations'],
                    'kind': ('explicit_instruction' if 'explicit_instruction' in
                             {old_proof['kind'], source_proof['kind']} else 'observed_behavior')}
                existing.trigger_conditions = {**(existing.trigger_conditions or {}),
                    '_last_equivalence': proof, '_equivalence_candidates': checks,
                    '_source_support': combined_proof,
                    'topic_tags': sorted(set((existing.trigger_conditions or {}).get('topic_tags') or [])
                                         | set(cited_topics(session_turns, cited_batch_ids)))}
                if (existing.reinforcement_count >= 3 or
                        len(existing.source_batch_ids) >= settings.procedural_min_cited_turns):
                    existing.confidence_score = 0.8
                    existing.is_active = True
                log.info("procedural_reinforced_across_sessions",
                         reinforcement=existing.reinforcement_count,
                         is_active=existing.is_active)
            else:
                log.info("procedural_session_not_independent",
                         reason=("same_session" if seen_sessions is not None else
                                 "unresolved_prior_source"),
                         reinforcement=existing.reinforcement_count)
        else:
            # Insert new pending pattern
            new_pattern = ProceduralMemory(
                pattern_name=pattern_text[:80],
                pattern_description=pattern_text,
                topic_tags=turn.topic_tags or [],
                trigger_conditions={'_equivalence_candidates': checks, '_source_support': source_proof,
                                    'topic_tags': cited_topics(session_turns, cited_batch_ids)},
                reinforcement_count=1,
                first_observed=datetime.now(timezone.utc),
                last_observed=datetime.now(timezone.utc),
                # ⚑ Activate on EVIDENCE (G49). A pattern citing enough distinct
                # turns qualifies without a later session re-describing it.
                # Reinforcement still activates below; this is a second path,
                # not a replacement.
                is_active=len(cited_batch_ids) >= settings.procedural_min_cited_turns,
                confidence_score=(0.8 if len(cited_batch_ids)
                                  >= settings.procedural_min_cited_turns else 0.3),
                source_batch_ids=sorted(cited_batch_ids, key=str),
                embedding=embedding,
                project_id=project_id,
            )
            db.add(new_pattern)

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("procedural_extraction_complete", cited_count=len(cited_batch_ids),
                 session_id=str(turn.session_id), session_turns=n_turns,
                 evidence=sorted(evidence))

    except Exception as exc:
        db.rollback()
        log.error("procedural_extraction_failed", error_type=type(exc).__name__)
        raise
    finally:
        db.close()
