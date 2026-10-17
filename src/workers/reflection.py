"""Reflection Worker – session synthesis and the conversation's own slots.
   Procedural patterns have one cited-source writer.

G77 (maintainer, 2026-10-17): reflection writes only the CONVERSATION-tier
`conversation_focus` and `pending_items` of the conversation its evidence came
from — quoted claims, each NLI-checked against its quote, applied with an
undo journal (slots.write_journaled). It never writes a global slot (those are
the user's: /remember, the slot API). Retired: global slot proposals (they
rewrote the global project_context with each batch's topic) and motif cluster
proposals (approval created an empty cluster)."""

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, SessionSummary
from src.workers.bg_client_factory import (
    bg_timeout,
    get_bg_client,
    get_bg_model_name,
    json_schema,
)
from src.workers.llm_json import parse_object
from src.workers.runtime import JobYielded

logger = structlog.get_logger("ice.workers.reflection")
bg_client = get_bg_client()
# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
SUMMARY_PROMPT = (
    "Generate a structured session summary from the following conversation turns.\n"
    "Output ONLY a valid JSON object with these keys:\n"
    "  - \"topics_covered\": a list of strings (e.g., [\"PostgreSQL\", \"FastAPI\"])\n"
    "  - \"decisions_made\": a string describing any decisions\n"
    "  - \"unresolved_items\": a string describing any unresolved questions\n"
    "  - \"entities_updated\": a list of canonical entity names that appeared\n"
    "  - \"patterns_observed\": a list of strings describing observed behavioural patterns\n\n"
    "If a field has no content, use an empty list [] for lists, or an empty string \"\" for strings.\n"
    "Do NOT include markdown or additional text."
)

CONVERSATION_SLOTS_PROMPT = (
    "Below are the most recent messages of ONE conversation, split into windows. "
    "Write two short lists about where this conversation stands at its end:\n"
    "- focus (at most 3): what the USER is working on, deciding or asking about, "
    "taken from the user's own words - their goal, task or question, written as "
    "'The user wants ...', 'The user is trying to ...' or 'The user asked ...'.\n"
    "- open (at most 5): something still unresolved at the end, and the sentence "
    "must state that unresolved act itself: a question someone asked ('The assistant "
    "asked whether ...', 'The user asked ...'), a task someone said is still to be "
    "done ('The user still needs to ...', 'The assistant offered to ... once ...'), "
    "or a choice not yet made ('The user has not yet chosen between ...'). Not an "
    "explanation, an answer, advice already given, a description of what someone "
    "said, or a feeling, event or personal detail the user shared.\n"
    "For each entry, in this order: (1) sources: the one window (at most two) that "
    "states it; (2) quote: a short exact copy, up to about 25 words, of the words in "
    "that window that state it - copy, do not paraphrase; (3) speaker: who said those "
    "words (the role shown on that window); (4) text: one complete sentence that the "
    "quote supports, naming the speaker. An assistant proposal is not a user "
    "decision. Use [] for a list when nothing qualifies - an empty list is better "
    "than an entry that does not fit. Never put window IDs inside text. Output only "
    "JSON."
)

OPEN_CHECK_PROMPT = (
    "Below is the end of a conversation, then numbered notes about it. For each note "
    "answer two questions about what the messages say:\n"
    "1. pending: is the note about something that still needs a response or an "
    "action - a question someone asked, a step someone said to do, or an offer or "
    "choice put to the user? Answer false for an explanation, an answer, a "
    "conclusion, an opinion, encouragement or a remark.\n"
    "2. closed_later: does a LATER message in the conversation answer it, do it, "
    "decline it, or move on from it?\n"
    "Use only what the messages say. Output only JSON."
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _robust_json(raw: str, *, where: str = "?") -> dict:
    """G29: delegates to the shared salvage. Kept as a named wrapper because
    this module's callers expect {} rather than None — the loudness that made
    this the canonical copy now lives in llm_json."""
    return parse_object(raw, where=f"reflection.{where}") or {}


# ------------------------------------------------------------------
# Main task
# ------------------------------------------------------------------
def run_reflection():
    """Execute a full reflection pass: synthesis and the conversation's own
    slots. Plain callable since C7 — gating/retries live in the runtime."""
    db = SessionLocal()
    try:
        # Get distinct conversation IDs from the last 200 turns
        conv_ids = [row[0] for row in db.execute(text(
            "SELECT conversation_id FROM ("
            "  SELECT DISTINCT ON (conversation_id) conversation_id, timestamp "
            "  FROM episodic_memory WHERE is_private = FALSE ORDER BY conversation_id, timestamp DESC"
            ") sub ORDER BY timestamp DESC LIMIT 200"
        )).fetchall()]
        for conv_id in conv_ids:
            recent = db.query(EpisodicMemory).filter_by(
                conversation_id=conv_id, is_private=False
            ).order_by(EpisodicMemory.timestamp.desc()).limit(200).all()
            if len(recent) < 10:
                continue
            recent.reverse()
            _synthesize_session(db, recent)
            db.commit()
            # The session summary retains patterns_observed as non-authoritative
            # observations. Only extract_procedural may write retrievable habits:
            # this periodic pass has no numbered, writer-attributed citations and
            # used to reinforce the same sitting on every run.
            try:
                _refresh_conversation_slots(db, conv_id, recent)
            except JobYielded:
                raise
            except Exception as exc:                               # noqa: BLE001
                # One conversation's failed slot call must not abort the pass
                # for every other conversation (TRAPS 144); said every time.
                db.rollback()
                logger.warning("reflection_conversation_slots_failed",
                               conversation_id=str(conv_id),
                               error=f"{type(exc).__name__}: {exc}"[:200])

        # CODEX_V4 D11.1: no model-written entity notes — an entity's page is
        # its verified facts. The receipt stays so replay reports see the state.
        enrichment = dict(method="entity_enrichment_receipt_v1", selected=0, attempted=0,
                          skipped="codex_v4_facts")

        db.commit()
        logger.info("reflection_full_pass_complete")
        return {"entity_enrichment": enrichment}

    except Exception as exc:
        db.rollback()
        logger.error("reflection_failed", error=str(exc))
        raise
    finally:
        db.close()


# ------------------------------------------------------------------
# Session Synthesis (existing logic, kept)
# ------------------------------------------------------------------
def _synthesize_session(db, turns):
    full_text = "\n\n".join([t.raw_text for t in turns])
    words = full_text.split()
    if len(words) > 3000:
        full_text = " ".join(words[-3000:])
    completion = bg_client.chat.completions.create(
        model=get_bg_model_name(),
        messages=[
            {"role": "system", "content": "You are a session analysis engine. Output only JSON."},
            {"role": "user", "content": f"{SUMMARY_PROMPT}\n\n{full_text}"}
        ],
        temperature=0.0, max_tokens=400, timeout=bg_timeout(400),
        response_format=json_schema("session_summary", {
            "type": "object",
            "properties": {
                "topics_covered": {"type": "array", "items": {"type": "string"}},
                "decisions_made": {"type": "string"},
                "unresolved_items": {"type": "string"},
                "entities_updated": {"type": "array", "items": {"type": "string"}},
                "patterns_observed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["topics_covered", "decisions_made", "unresolved_items",
                         "entities_updated", "patterns_observed"],
        }),
    )
    raw = (completion.choices[0].message.content or "").strip()
    data = _robust_json(raw, where="session_summary")

    summary = SessionSummary(
        topics_covered=data.get("topics_covered", []),
        decisions_made=data.get("decisions_made", ""),
        unresolved_items=data.get("unresolved_items", ""),
        entities_updated=data.get("entities_updated", []),
        patterns_observed=data.get("patterns_observed", [])
    )
    db.add(summary)

    # G77: unresolved items no longer append to the GLOBAL pending_items slot
    # (every prompt, uncapped); the conversation's own pending_items slot is
    # written from quoted, checked claims by _refresh_conversation_slots.


# ------------------------------------------------------------------
# The conversation's own slots (G77)
# ------------------------------------------------------------------
# How many of the newest turns are cut into windows before the newest ones
# that fit one prompt piece are kept — a bound on tokenizer work, not a tunable.
_SLOT_SOURCE_TURNS = 20
_SLOT_LISTS = {"focus": ("conversation_focus", 3), "open": ("pending_items", 5)}
# Output stop for 8 quoted entries: measured 2026-10-17, 8 entries used 639 tokens
# and the next case hit 700 (finish_reason length). A cap, not a length control.
_SLOT_OUTPUT_TOKENS = 1500


def _slot_windows(turns):
    """Windows of the newest turns that fit one piece (settings.summary_piece_tokens).
    A turn without recorded speaker boundaries is left out, counted."""
    from src.memory import tokens
    from src.memory.summary_support import original_units, source_windows, window_prompt
    units, unknown = [], 0
    for turn in turns[-_SLOT_SOURCE_TURNS:]:
        try:
            units.extend(original_units([turn]))
        except ValueError:
            unknown += 1
    if not units:
        return [], unknown
    chosen, used = [], 0
    for window in reversed(source_windows(units)):
        size = tokens.count(window_prompt([window])) + 2
        if chosen and used + size > settings.summary_piece_tokens:
            break
        chosen.insert(0, window)
        used += size
    return chosen, unknown


def _slot_schema(windows):
    ids = [w["id"] for w in windows]
    roles = sorted({w["role"] for w in windows})
    row = {"type": "object", "additionalProperties": False,
           "required": ["sources", "quote", "speaker", "text"],
           "properties": {"sources": {"type": "array", "items": {"type": "string", "enum": ids},
                                      "minItems": 1, "maxItems": 2},
                          "quote": {"type": "string", "minLength": 1},
                          "speaker": {"type": "string", "enum": roles},
                          "text": {"type": "string", "minLength": 1}}}
    # A focus is the USER's goal, so it is quoted from the user's own words —
    # by the authoritative speaker spans, not by how the sentence is phrased.
    focus_row = dict(row, properties=dict(row["properties"],
                                          speaker={"type": "string", "enum": ["user"]}))
    items = {"focus": focus_row, "open": row}
    return {"type": "object", "additionalProperties": False, "required": list(_SLOT_LISTS),
            "properties": {k: {"type": "array", "items": items[k],
                               "maxItems": n if k != "focus" or "user" in roles else 0}
                           for k, (_, n) in _SLOT_LISTS.items()}}


def _check_claim(row, windows):
    """(premise, None) when the claim may go to NLI, else (None, reason): the
    mechanical checks of the claim pass — cited windows exist, the quote is in
    them word for word, the speaker said it."""
    import json as _json

    from src.memory.summary_support import _collapse, cited_premise
    if (not isinstance(row, dict) or set(row) != {"sources", "quote", "speaker", "text"}
            or not all(isinstance(row[k], str) and row[k].strip() for k in ("quote", "speaker", "text"))
            or not isinstance(row["sources"], list) or not 1 <= len(row["sources"]) <= 2):
        return None, "invalid_claim_shape"
    try:
        premise, intervals = cited_premise(windows, row["sources"])
    except (ValueError, TypeError):
        return None, "unresolved_source"
    quote, roles = _collapse(row["quote"]), []
    for part, interval in zip(premise.split("\n"), intervals):
        if quote in _collapse(_json.loads(part[len(f"The {interval['role']} said: "):])):
            roles.append(interval["role"])
    if not roles:
        return None, "quote_not_in_source"
    if row["speaker"] not in roles:
        return None, "speaker_mismatch"
    return premise, None


def _verify(premise, claim):
    """NLI verdict for one claim; over the verifier's token bound is 'unknown'."""
    from dataclasses import asdict

    from src.memory.support import support_tokenizer, verify_support
    if len(support_tokenizer()(premise, claim, truncation=False)["input_ids"]) \
            > settings.source_support_max_tokens:
        return {"status": "unknown", "reason": "over_nli_token_bound", "scores": {}}
    return asdict(verify_support(premise, claim))


def _last_reflected_turn(db, conv_id):
    row = db.execute(text("""
        SELECT item_content->>'through_turn_id' FROM review_queue
        WHERE item_type = 'memory_slot_update' AND status IN ('applied', 'undone')
          AND item_content->>'proposed_by' = 'reflection'
          AND item_content->>'conversation_id' = :cid
        ORDER BY created_at DESC LIMIT 1"""), {"cid": str(conv_id)}).first()
    return row[0] if row else None


def _open_still_pending(windows, texts):
    """G77 (measured 2026-10-17): the writer keeps listing finished matters —
    closing remarks ("The assistant concluded …"), answered questions — as open,
    whatever the instruction says (TRAPS 138). One call asks what must be true
    of each item (TRAPS 136); kept only if pending and not closed later. A failed
    call keeps NO open item (unknown is never asserted) and warns."""
    from src.memory.summary_support import window_prompt
    from src.workers.completion_text import complete_text
    if not texts:
        return []
    row = {"type": "object", "additionalProperties": False,
           "required": ["pending", "closed_later"],
           "properties": {"pending": {"type": "boolean"}, "closed_later": {"type": "boolean"}}}
    schema = {"type": "object", "additionalProperties": False, "required": ["notes"],
              "properties": {"notes": {"type": "array", "items": row,
                                       "minItems": len(texts), "maxItems": len(texts)}}}
    notes = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    try:
        completion = bg_client.chat.completions.create(
            model=get_bg_model_name(),
            messages=[{"role": "system", "content": OPEN_CHECK_PROMPT},
                      {"role": "user", "content": f"{window_prompt(windows)}\n\nNotes:\n{notes}"}],
            temperature=0.0, max_tokens=60 * len(texts) + 60,
            timeout=bg_timeout(60 * len(texts) + 60),
            response_format=json_schema("open_check", schema))
        answers = _robust_json(complete_text(completion), where="open_check").get("notes")
        if not isinstance(answers, list) or len(answers) != len(texts):
            raise ValueError("open_check_shape")
        return [bool(a.get("pending")) and not bool(a.get("closed_later")) for a in answers]
    except JobYielded:
        raise
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("reflection_open_check_failed", error=f"{type(exc).__name__}: {exc}"[:200],
                       fallback="no_open_items", items=len(texts))
        return [False] * len(texts)


def slot_claims(turns, *, conversation_id=None, prompt=None):
    """The claim half of the conversation-slot write, with no database: windows
    of the newest turns, one model call, and each claim through the mechanical
    checks and NLI. Returns (windows, {list: kept claims}, dropped), or None when
    no turn has speaker boundaries. *prompt* lets a qualification run compare
    instructions on the same windows."""
    from src.memory.summary_support import window_prompt
    from src.workers.completion_text import complete_text
    windows, unknown = _slot_windows(turns)
    if unknown:
        logger.warning("reflection_turns_without_speaker_spans",
                       conversation_id=str(conversation_id), skipped_turns=unknown)
    if not windows:
        return None
    completion = bg_client.chat.completions.create(
        model=get_bg_model_name(),
        messages=[{"role": "system", "content": prompt or CONVERSATION_SLOTS_PROMPT},
                  {"role": "user", "content": window_prompt(windows)}],
        temperature=0.0, max_tokens=_SLOT_OUTPUT_TOKENS, timeout=bg_timeout(_SLOT_OUTPUT_TOKENS),
        response_format=json_schema("conversation_slots", _slot_schema(windows)))
    parsed = _robust_json(complete_text(completion), where="conversation_slots")
    by_list, dropped = {}, []
    for key, (_slot_name, limit) in _SLOT_LISTS.items():
        rows = parsed.get(key)
        kept = []
        for row in (rows if isinstance(rows, list) else [])[:limit]:
            premise, problem = _check_claim(row, windows)
            verdict = _verify(premise, row["text"].strip()) if premise else None
            if verdict is None or verdict["status"] != "supported":
                dropped.append({"list": key, "reason": problem or verdict["status"],
                                "text": row.get("text") if isinstance(row, dict) else None})
                continue
            kept.append({"text": row["text"].strip(), "quote": row["quote"],
                         "speaker": row["speaker"], "sources": row["sources"],
                         "scores": verdict.get("scores")})
        by_list[key] = kept
    if by_list.get("open"):
        verdicts = _open_still_pending(windows, [c["text"] for c in by_list["open"]])
        dropped += [{"list": "open", "reason": "not_pending_or_closed", "text": c["text"]}
                    for c, ok in zip(by_list["open"], verdicts) if not ok]
        by_list["open"] = [c for c, ok in zip(by_list["open"], verdicts) if ok]
    return windows, by_list, dropped


def _refresh_conversation_slots(db, conv_id, turns):
    """G77: write this conversation's `conversation_focus` and `pending_items`
    from quoted claims that pass the claim pass's checks and NLI. A list with no
    supported claim leaves its slot untouched; unchanged text is not rewritten.
    Skipped when the newest turn was already reflected on."""
    from src.services import slots as slots_svc
    from src.services.errors import NotFoundError
    through = str(turns[-1].id)
    if _last_reflected_turn(db, conv_id) == through:
        return {"status": "skipped", "reason": "no_new_turns"}
    claims = slot_claims(turns, conversation_id=conv_id)
    if claims is None:
        return {"status": "skipped", "reason": "no_source_windows"}
    windows, by_list, dropped = claims
    written = {}
    for key, (slot_name, _limit) in _SLOT_LISTS.items():
        kept = by_list[key]
        if not kept:
            continue
        content = "\n".join(c["text"] for c in kept)
        try:
            current = slots_svc.get_slot(db, slot_name, "conversation",
                                         conversation_id=conv_id)["content"]
        except NotFoundError:
            current = None
        if current == content:
            continue
        slots_svc.write_journaled(
            db, slot_name, content, updated_by="reflection", scope_tier="conversation",
            conversation_id=conv_id,
            audit={"reason": "reflection: quoted claims, NLI-supported",
                   "through_turn_id": through, "claims": kept,
                   "windows": [{k: w[k] for k in ("id", "unit", "start", "end")}
                               for w in windows]})
        written[slot_name] = len(kept)
    if dropped:
        logger.warning("reflection_slot_claims_dropped", conversation_id=str(conv_id),
                       count=len(dropped), reasons=sorted({d["reason"] for d in dropped}))
    logger.info("reflection_conversation_slots", conversation_id=str(conv_id),
                written=written, dropped=len(dropped))
    return {"status": "done", "written": written, "dropped": dropped}
