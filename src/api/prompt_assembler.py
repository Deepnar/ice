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

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from src.memory.models import EpisodicMemory, MemorySlot
from src.retrieval.orchestrator import ContextFragment


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.33)


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "…"
    return text


def get_recent_turns(
    db_session: Session,
    conversation_id: str,
    max_tokens: int = 4000,
    max_count: int = 10,
) -> List[dict]:
    """Return the most recent turns that fit under *max_tokens* total.

    Per‑turn word caps scale dynamically: a larger budget allows
    fuller turns; a tiny budget keeps everything ultra‑compact.
    """
    turns = (
        db_session.query(EpisodicMemory)
        .filter_by(conversation_id=conversation_id)
        .order_by(EpisodicMemory.timestamp.desc())
        .limit(max_count)
        .all()
    )
    turns.reverse()

    if max_tokens <= 1000:
        per_turn_words = 80
    elif max_tokens <= 3000:
        per_turn_words = 150
    else:
        per_turn_words = min(500, max(100, max_tokens // max(1, len(turns)) // 2))

    result = []
    tokens_used = 0
    for t in turns:
        if t.inject_raw and t.raw_text:
            text = t.raw_text
        elif t.summary_text:
            text = t.summary_text
        else:
            text = (t.raw_text or "")[:300]

        if text.startswith("User: "):
            user_part = text[6:]
            assistant_start = user_part.find("\n\nAssistant: ")
            if assistant_start != -1:
                assistant_part = user_part[assistant_start + len("\n\nAssistant: "):]
                user_part = user_part[:assistant_start]
                u = _trim_words(user_part, per_turn_words)
                a = _trim_words(assistant_part, per_turn_words)
                pair_tokens = _estimate_tokens(u) + _estimate_tokens(a)
                if tokens_used + pair_tokens > max_tokens and result:
                    break
                tokens_used += pair_tokens
                result.append({"role": "user", "content": u})
                result.append({"role": "assistant", "content": a})
            else:
                u = _trim_words(user_part, per_turn_words)
                t_tok = _estimate_tokens(u)
                if tokens_used + t_tok > max_tokens and result:
                    break
                tokens_used += t_tok
                result.append({"role": "user", "content": u})
        else:
            trimmed = _trim_words(text, per_turn_words)
            t_tok = _estimate_tokens(trimmed)
            if tokens_used + t_tok > max_tokens and result:
                break
            tokens_used += t_tok
            result.append({"role": "user", "content": trimmed})
    return result


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

    slot_lines = []
    for slot in memory_slots:
        if slot.is_active and slot.content:
            slot_lines.append(f"[{slot.slot_name.upper()}]\n{slot.content.strip()}")
    if slot_lines:
        system_msg["content"] += "\n\n=== PERSISTENT CONTEXT ===\n" + "\n\n".join(slot_lines)

    # E4 (D6): coding-scoped conversations open a sitting with the project's
    # where-was-I block (state + diffstat + constraints + tasks + decisions).
    # The caller renders it only at session start — not every turn.
    if session_start_text:
        system_msg["content"] += "\n\n=== PROJECT SESSION START ===\n" + session_start_text

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