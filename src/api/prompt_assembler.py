"""Context Assembly – multi‑message format with external budget control.

TWO CHANGES FROM THE PREVIOUS VERSION:

1. Removed the literal "<|think|>" token from the system prompt. This
   looks like a model-specific reasoning-mode control token (some Qwen/
   DeepSeek variants use a literal <think> tag this way), but the
   experiment runs against gemma4:26b-a4b-it-q4_K_M as SINGLE_MODEL, plus
   whatever model find_best_model() routes to for *_moe conditions. If
   gemma doesn't recognize it as special, it's just 4 wasted tokens of
   literal text. If some OTHER model in the MoE-routed stack DOES treat it
   as a special control token, the vector_rag_moe / full_ice_moe
   conditions could silently get different reasoning behavior than the
   generalist conditions purely because of this token — confounding the
   six-condition comparison in a way that would be very hard to notice
   from the outside (answers would just look subtly different, with no
   obvious cause). Removed for safety; the same instruction is conveyed in
   plain English in the line right after it, so nothing is lost.

2. Added an explicit boundary marker between "recent history" messages and
   the live question, instead of relying on positional inference alone.
   The multi-message structure already does most of the work (real
   alternating user/assistant roles instead of one fused string), but
   without an explicit marker the model still has to INFER that the very
   last user message is the live one rather than another history turn —
   which is the same class of ambiguity that caused the original
   "answers the previous turn's prompt" symptom, just less severe. A short
   system-level instruction now states this rule directly instead of
   leaving it implicit.
"""

import uuid as _uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from src.api.config import settings
from src.memory.models import (
    Conversation,
    ConversationSummary,
    EpisodicMemory,
    MemorySlot,
)
from src.memory.tokens import count as _estimate_tokens
from src.retrieval.orchestrator import ContextFragment


def conversation_summary_block(
    db_session: Session,
    conversation_id: str,
    turn_count: int,
    total_tokens: float,
    recent_window_tokens: float,
) -> Optional[str]:
    """C4 (D3a): the active conversation's evolving summary, injected only
    once the conversation outgrew the sliding window (B2's memory-pressure
    condition — the window's reach is `recent_window_tokens`). A summary the
    burst hasn't caught up with is still injected, stamped with how far
    behind it runs. Returns the block text or None."""
    if not conversation_id or total_tokens <= recent_window_tokens:
        return None
    row = db_session.query(ConversationSummary).filter_by(
        conversation_id=_uuid.UUID(str(conversation_id))).first()
    if row is None or not row.summary_text:
        return None
    behind = max(0, int(turn_count) - int(row.covers_turns or 0))
    stamp = f"(as of {behind} turns ago) " if behind > 0 else ""
    return f"{stamp}{row.summary_text}"


def _turn_text(t) -> str:
    """The representation to show for one turn (C1's storage-side hint)."""
    if t.inject_raw and t.raw_text:
        return t.raw_text
    if t.summary_text:
        return t.summary_text
    return (t.raw_text or "")[:300]


def _split_pair(text: str):
    """ICE's own stored turn format → (user_part, assistant_part_or_None).

    This parses a separator ICE itself wrote (`main.py` builds every raw_text
    as ``User: … \\n\\nAssistant: …``), so it is a protocol read, not a guess
    about how a person writes.
    """
    if not text.startswith("User: "):
        return text, None
    body = text[6:]
    i = body.find("\n\nAssistant: ")
    if i == -1:
        return body, None
    return body[:i], body[i + len("\n\nAssistant: "):]


def _fit(text: str, budget: int) -> str:
    """Trim to a real token budget, on a word boundary."""
    if budget <= 0 or not text:
        return ""
    if _estimate_tokens(text) <= budget:
        return text
    words, kept, used = text.split(), [], 0
    for w in words:
        used += _estimate_tokens(w + " ")
        if used > budget:
            break
        kept.append(w)
    return (" ".join(kept) + "…") if kept else ""


def _recent_turn_rows(db_session, conversation_id, scope_mode: str,
                      bridge_turns: int, hard_max: int):
    """The turns of the CURRENT SITTING, newest first, plus a bridge.

    C16/C6: the recent window is *continuity*, not evidence — "what were we
    just doing" — so its natural boundary is the sitting, not a turn count and
    not relevance. `session_id` has been written by all three ingest paths and
    indexed since C6 and read by NOTHING in retrieval; this is its first reader.

    When the current turn opens a NEW sitting, the whole previous sitting is
    the conversation summary's job (C4) — except its last turn, which comes
    along so a 31-minute coffee break does not amputate the thread. The clock
    gap is not a meaning boundary and `recent_window_bridge_turns` exists
    because of that.

    The filters match every retrieval leg's: private turns, document sections
    and promoted pastes have no business in a chat's continuity window. The old
    query had none of them — so a >2,000-word turn that the vector leg excludes
    *for being too big* walked straight into the window.
    """
    q = (db_session.query(EpisodicMemory)
         .join(Conversation, Conversation.id == EpisodicMemory.conversation_id)
         .filter(EpisodicMemory.conversation_id == conversation_id,
                 EpisodicMemory.is_private.is_(False),
                 EpisodicMemory.is_document.is_(False),
                 EpisodicMemory.promoted_document_id.is_(None),
                 Conversation.kind == "chat")
         .order_by(EpisodicMemory.timestamp.desc()))

    if scope_mode != "session":
        return q.limit(hard_max).all()

    rows = q.limit(hard_max).all()
    if not rows:
        return []
    current = rows[0].session_id
    if current is None:
        return rows            # pre-C6 rows carry no session — count mode
    same = [r for r in rows if r.session_id == current]
    if len(same) < len(rows) and bridge_turns > 0:
        # The last turn(s) of the sitting before this one.
        older = [r for r in rows if r.session_id != current][:bridge_turns]
        same += older
    return same


def get_recent_turns(
    db_session: Session,
    conversation_id: str,
    max_tokens: int = 4000,
    max_count: int = None,
) -> List[dict]:
    """The recent-turn window, newest-first and bounded by the sitting.

    Two behaviours changed here and both were bugs.

    **It used to discard the NEWEST turns.** The query took the 10 newest
    descending, reversed them, then `break`-ed on the first turn that would
    overflow the budget — so one fat old turn cost you every turn after it,
    including the one you had just sent. On a long sitting the "recent window"
    showed the model the OLDEST turns it had. Selection now runs newest-first
    and the output is reversed at the end, so the budget is spent on the most
    recent context and the oldest turns are what fall off.

    **`max_count` was hardcoded to 10 and no caller ever overrode it**, so a
    model with a 40k window still saw ten turns. The bound is now the sitting,
    with `recent_window_max_turns` as a rail for a marathon session.

    A turn too big for its share degrades through C1/C3's chain (raw → summary
    → abstract) instead of being word-truncated mid-sentence.
    """
    scope_mode = getattr(settings, "recent_window_scope", "session")
    hard_max = int(max_count or getattr(settings, "recent_window_max_turns", 40))
    turns = _recent_turn_rows(
        db_session, conversation_id, scope_mode,
        int(getattr(settings, "recent_window_bridge_turns", 1)), hard_max)
    if not turns:
        return []

    # One turn may not eat the window. Nothing enforced this before: the
    # per-turn cap was a bracket ladder over `max_tokens // len(turns) // 2`.
    max_turn_frac = float(getattr(settings, "recent_window_max_turn_frac", 0.35))
    per_turn_cap = max(64, int(max_tokens * max_turn_frac))

    pairs, used = [], 0
    for t in turns:                              # newest first
        text = _turn_text(t)
        if _estimate_tokens(text) > per_turn_cap:
            # C1/C3 degrade-before-truncate: prefer a form that was written to
            # be short over a sentence cut in half.
            for alt in (t.summary_text, t.abstract_text):
                if alt and _estimate_tokens(alt) <= per_turn_cap:
                    text = alt
                    break
        user_part, assistant_part = _split_pair(text)
        share = min(per_turn_cap, max(64, max_tokens - used))
        if assistant_part is not None:
            u = _fit(user_part, share // 2)
            a = _fit(assistant_part, share // 2)
            cost = _estimate_tokens(u) + _estimate_tokens(a)
            pair = [{"role": "user", "content": u},
                    {"role": "assistant", "content": a}]
        else:
            u = _fit(user_part, share)
            cost = _estimate_tokens(u)
            pair = [{"role": "user", "content": u}]
        if pairs and used + cost > max_tokens:
            break                                 # the OLDEST turns fall off
        used += cost
        pairs.append(pair)

    # Selection ran newest-first; the model reads oldest-first.
    out = []
    for pair in reversed(pairs):
        out.extend(pair)
    return out


def assemble_prompt(
    memory_slots: List[MemorySlot],
    retrieved_fragments: List[ContextFragment],
    user_message: str,
    db_session: Optional[Session] = None,
    conversation_id: Optional[str] = None,
    bookmarked_texts: Optional[List[str]] = None,
    classification=None,
    scope: Optional[dict] = None,
    max_recent_tokens: int = 4000,
    session_start_text: Optional[str] = None,
    conversation_summary_text: Optional[str] = None,
) -> List[dict]:
    """Build a multi‑message prompt. The caller controls the total budget
    via *max_recent_tokens*; retrieval fragments are passed as‑is (already
    budgeted by the orchestrator).
    """

    # T1 date-grounding: without a today-anchor, even dated fragments can't
    # resolve relative time ("two years ago"); with it, the [YYYY-MM-DD]
    # fragment stamps become usable for ordering and era-telling.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_msg = {
        "role": "system",
        "content": (
            f"Today's date: {today}. "
            "You have access to the user's conversation history below, shown as a "
            "sequence of earlier user/assistant message pairs, followed by retrieved "
            "background context, followed by the user's CURRENT question as the final "
            "message in this conversation. The final message is the ONLY one you are "
            "answering right now — earlier messages are history for context, not the "
            "question to respond to. "
            "Think step-by-step through all the relevant facts before answering. "
            "When facts have changed over time (a role was reassigned, a name changed, "
            "a decision was reversed), mention the earlier version and what it changed to. "
            "Retrieved memory fragments are prefixed with the date they were written, "
            "like [2025-11-04]; facts may show (since YYYY-MM) or a [Timeline: …] "
            "history. Use these dates to order events and to tell earlier versions "
            "from the current one. "
            "Be specific — reference chapter numbers, timestamps, or quotes from the context. "
            "Answer accurately, thoroughly and in deep detail, drawing on the given context to make the "
            "response complete and well-grounded."
        ),
    }

    # C9 (D6): three slot tiers under one PERSISTENT CONTEXT header, in
    # order global → project → conversation. Project slots render only for
    # the attached project (scope["project_id"] — the E1 payoff: coding
    # conversations inherit them automatically); conversation slots only for
    # this conversation.
    project_id = str((scope or {}).get("project_id") or "") or None
    conv_id_str = str(conversation_id) if conversation_id else None

    def _tier(slot) -> str:
        return getattr(slot, "scope_tier", None) or "global"

    active = [s for s in memory_slots if s.is_active and s.content]
    slot_lines = [f"[{s.slot_name.upper()}]\n{s.content.strip()}"
                  for s in active if _tier(s) == "global"]
    slot_lines += [f"[PROJECT · {s.slot_name.upper()}]\n{s.content.strip()}"
                   for s in active if _tier(s) == "project"
                   and project_id and str(s.project_id) == project_id]
    slot_lines += [f"[CONVERSATION · {s.slot_name.upper()}]\n{s.content.strip()}"
                   for s in active if _tier(s) == "conversation"
                   and conv_id_str and str(s.conversation_id) == conv_id_str]
    if slot_lines:
        system_msg["content"] += "\n\n=== PERSISTENT CONTEXT ===\n" + "\n\n".join(slot_lines)

    # E4 (D6): coding-scoped conversations open a sitting with the project's
    # where-was-I block (state + diffstat + constraints + tasks + decisions).
    # The caller renders it only at session start — not every turn.
    if session_start_text:
        system_msg["content"] += "\n\n=== PROJECT SESSION START ===\n" + session_start_text

    # C4 (D3a): global conversation shape once the window can't hold it all —
    # orientation, not evidence (retrieval fragments stay the evidence).
    if conversation_summary_text:
        system_msg["content"] += ("\n\n=== CONVERSATION SUMMARY ===\n"
                                  + conversation_summary_text)

    # C16: bookmarks are INJECTED. This parameter was declared here and never
    # read — the caller queried up to five bookmarked turns, word-capped each
    # one, logged `bookmarked_count` as though they had been injected, and
    # dropped them on the floor. A turn the user explicitly pinned is the
    # highest-intent signal in the store and it was reaching the model in
    # exactly none of the requests that reported it.
    if bookmarked_texts:
        system_msg["content"] += (
            "\n\n=== BOOKMARKED BY THE USER ===\n"
            "(turns the user pinned as important — treat as standing context)\n"
            + "\n\n".join(bookmarked_texts))

    recent_messages = []
    if db_session and conversation_id:
        recent_messages = get_recent_turns(db_session, conversation_id, max_tokens=max_recent_tokens)

    fragments_block = ""
    if retrieved_fragments:
        fragments_block = "\n\n".join(f.text for f in retrieved_fragments)

    messages = [system_msg]
    messages.extend(recent_messages)

    if fragments_block:
        cluster_names = []
        if scope and scope.get("cluster_ids"):
            try:
                from src.memory.models import ContextCluster
                clusters = (
                    db_session.query(ContextCluster)
                    .filter(ContextCluster.id.in_(scope["cluster_ids"]))
                    .all()
                )
                cluster_names = [c.name for c in clusters]
            except Exception:
                pass
        header = "=== RETRIEVED CONTEXT ==="
        if cluster_names:
            header += f" (clusters: {', '.join(cluster_names)})"
        messages.append({"role": "user", "content": f"{header}\n{fragments_block}"})
        messages.append({
            "role": "assistant",
            "content": "Understood — I have the background context. What would you like to know?",
        })

    messages.append({"role": "user", "content": user_message})
    return messages