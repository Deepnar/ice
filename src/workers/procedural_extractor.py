"""Procedural Extractor – identifies recurring behavioural patterns."""

import re
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.workers.completion_text import complete_text
from src.api.db import SessionLocal
from src.memory.embedder import get_embedder
from src.memory.models import EpisodicMemory, IdempotencyKey, ProceduralMemory
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name
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


def _user_half(raw_text: str) -> str:
    """The user's own words out of a stored turn.

    Turns are written as ``User: …\\n\\nAssistant: …`` by main.py, so this
    parses OUR OWN delimiter — not the user's writing convention — which is the
    kind of parsing CLAUDE.md's style-invariance rule explicitly allows (it
    RESOLVES a value rather than inferring intent). Falls back to the whole
    text if the marker is absent, so pre-format rows still contribute.
    """
    if raw_text.startswith("User:"):
        head = raw_text.split("\n\nAssistant:", 1)[0]
        return head[len("User:"):].strip()
    return raw_text.strip()


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
            .filter(EpisodicMemory.session_id == turn.session_id)
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
        user_prompts = [_user_half(t.raw_text or "") for t in session_turns]
        user_prompts = [p for p in user_prompts if p]
        if len(user_prompts) < settings.procedural_min_session_turns:
            log.info("procedural_too_few_user_prompts", found=len(user_prompts),
                     needed=settings.procedural_min_session_turns)
            return
        block, total = [], 0
        for p in reversed(user_prompts):
            if total + len(p) > settings.procedural_max_prompt_chars and block:
                break
            block.append(p)
            total += len(p)
        block.reverse()
        numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(block, 1))

        # The old prompt asked for a pattern "the user consistently follows"
        # while showing ONE turn — an instruction that cannot be satisfied
        # honestly, so it was answered dishonestly. It now shows several of the
        # user's prompts and demands the recurrence be pointed at.
        prompt = (
            f"Below are {len(block)} messages the SAME user wrote during one "
            "session, in order.\n\n"
            f"{numbered}\n\n"
            "Identify ONE habit or workflow this user repeats — something visible "
            "in at least TWO of the messages above. Examples: \"asks for the error "
            "message before showing code\", \"restates the plan before agreeing to "
            "it\".\n\n"
            "Rules:\n"
            "- It MUST be supported by two or more of the numbered messages.\n"
            "- Cite them, in this exact format: PATTERN: <one sentence> | "
            "EVIDENCE: <comma-separated message numbers>\n"
            "- Describe what the user DOES, not what they talked about.\n"
            "- If no habit repeats across messages, output exactly: NONE\n"
            "Output nothing else."
        )
        model_name = get_bg_model_name()
        completion = bg_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a behavioural pattern detector."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=160,
            timeout=bg_timeout(160)
        )
        raw_reply = complete_text(completion)
        if raw_reply.upper().startswith("NONE"):
            return

        pattern_text, evidence = _parse_pattern(raw_reply)
        if not pattern_text:
            log.warning("procedural_unparsable_reply", reply=raw_reply[:120])
            return
        # The evidence requirement is the whole point: a pattern resting on one
        # message is the single-turn fabrication under a new name. Enforced
        # here rather than trusted to the prompt, because a prompt is a request
        # and this is a rule.
        if len(evidence) < 2:
            log.info("procedural_rejected_single_evidence",
                     pattern=pattern_text[:80], evidence=sorted(evidence),
                     session_id=str(turn.session_id))
            return

        # Encode the pattern for similarity matching
        embedding = encode_pattern(pattern_text)

        # Force PostgreSQL to accept the list as a vector via explicit cast
        similarity_query = text("""
            SELECT id, 1 - (embedding <=> CAST(:emb AS vector)) AS sim
            FROM procedural_memory
            WHERE embedding IS NOT NULL
            ORDER BY sim DESC LIMIT 1
        """)
        row = db.execute(similarity_query, {"emb": str(embedding)}).first()

        session_batch_ids = [t.batch_id for t in session_turns if t.batch_id]

        if row and row.sim > settings.procedural_similarity_threshold:
            existing = db.query(ProceduralMemory).get(row.id)
            existing.last_observed = datetime.now(timezone.utc)
            # ⚑ Only a DIFFERENT session is a repeat. Re-reading this one as it
            # grows (see the bucket key above) would otherwise reinforce the
            # pattern against itself, and `reinforcement_count >= 3` — the whole
            # promotion gate — would be satisfied by one sitting read three
            # times. Recurrence has to cross sittings or it is not recurrence.
            seen = set(existing.source_batch_ids or [])
            if seen.isdisjoint(session_batch_ids):
                existing.reinforcement_count += 1
                existing.source_batch_ids = list(seen | set(session_batch_ids))
                if existing.reinforcement_count >= 3 and existing.confidence_score < 0.8:
                    existing.confidence_score = 0.8
                    existing.is_active = True
                log.info("procedural_reinforced_across_sessions",
                         pattern=existing.pattern_name[:60],
                         reinforcement=existing.reinforcement_count,
                         is_active=existing.is_active)
            else:
                log.info("procedural_same_session_not_counted",
                         pattern=existing.pattern_name[:60],
                         reinforcement=existing.reinforcement_count)
        else:
            # Insert new pending pattern
            new_pattern = ProceduralMemory(
                pattern_name=pattern_text[:80],
                pattern_description=pattern_text,
                topic_tags=turn.topic_tags or [],
                trigger_conditions={},
                reinforcement_count=1,
                first_observed=datetime.now(timezone.utc),
                last_observed=datetime.now(timezone.utc),
                # ⚑ Activate on EVIDENCE (G49). A pattern citing enough distinct
                # turns is supported whether or not a later session happens to
                # re-describe it similarly — and re-description is what the
                # system provably cannot produce (0.708 against a 0.85 bar).
                # Reinforcement still activates below; this is a second path,
                # not a replacement.
                is_active=len(session_batch_ids or []) >= settings.procedural_min_cited_turns,
                confidence_score=(0.8 if len(session_batch_ids or [])
                                  >= settings.procedural_min_cited_turns else 0.3),
                # The whole session is the evidence now, not the one turn that
                # happened to trigger the job — which is also what lets a later
                # session be recognised as a genuinely separate observation.
                source_batch_ids=session_batch_ids,
                embedding=embedding,
                project_id=project_id,
            )
            db.add(new_pattern)

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("procedural_extraction_complete", pattern=pattern_text[:50],
                 session_id=str(turn.session_id), session_turns=n_turns,
                 evidence=sorted(evidence))

    except Exception as exc:
        db.rollback()
        log.error("procedural_extraction_failed", error=str(exc))
        raise
    finally:
        db.close()
