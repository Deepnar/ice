"""B1 D3: the classifier's input templates, in ONE place.

The v1 classifier rendered its inference input inline in ``classifier.py`` while
the trainer embedded bare prompt text — so the model was trained on one
distribution and served another. That mismatch is the single biggest known
defect B1 fixes, and it is fixed *by construction*: both sides import from here.

**This module is load-bearing, not cosmetic.** If training stops calling
``render()``, the mismatch silently returns and nothing fails loudly.

Templates are VERSIONED alongside the schema:

* ``V1_*`` — lifted verbatim from the pre-B1 ``classifier.py`` (the exact
  strings the live ``ice_classifier_v3_qwen_ft3.pt`` checkpoint was served with).
  Frozen forever: D5's non-regression gate must render the OLD model's input the
  way that model actually saw it, or the comparison is rigged against it.
* ``V2_*`` — names the real v2 categories (the four independent reliance
  signals, the coding intents). The frozen encoder maps the whole string to one
  vector; the prefix steers which subspace, so naming the actual label set is a
  small, free win now that the head is retrained from scratch anyway.

A checkpoint records its ``template_version``, so a loaded model always knows how
its input must be rendered — the pairing can't drift even with both generations
on disk.
"""

from __future__ import annotations

from typing import Optional

# ── v1 (frozen — verbatim from classifier.py @ 0dc5d89) ──────────────────────

V1_WITH_CONTEXT = (
    "Conversation context (summarized):\n{context_text}\n\n"
    "Given the above conversation and the user's latest prompt, "
    "predict:\n"
    "1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
    "2. INTENT: what is the user trying to do (Factual_Retrieval, Troubleshooting, etc.)\n"
    "3. CONTEXT RELIANCE: does the user need memory (Zero_Shot, Long_Term_Memory, Real_Time_Search)\n\n"
    "User prompt: {prompt}"
)

V1_NO_CONTEXT = (
    "Given a user prompt, predict:\n"
    "1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
    "2. INTENT: what is the user trying to do (Factual_Retrieval, Troubleshooting, etc.)\n"
    "3. CONTEXT RELIANCE: does the user need memory (Zero_Shot, Long_Term_Memory, Real_Time_Search)\n\n"
    "User prompt: {prompt}"
)

# ── v2 (B1) ──────────────────────────────────────────────────────────────────

V2_WITH_CONTEXT = (
    "Conversation context (summarized):\n{context_text}\n\n"
    "Given the above conversation and the user's latest prompt, "
    "predict:\n"
    "1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
    "2. INTENT: what is the user trying to do (Factual_Retrieval, Codebase_Query, Code_Change, etc.)\n"
    "3. CONTEXT RELIANCE: what does answering need (Needs_Memory, Temporal_Recall, Needs_Live_Info, High_Complexity)\n\n"
    "User prompt: {prompt}"
)

V2_NO_CONTEXT = (
    "Given a user prompt, predict:\n"
    "1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
    "2. INTENT: what is the user trying to do (Factual_Retrieval, Codebase_Query, Code_Change, etc.)\n"
    "3. CONTEXT RELIANCE: what does answering need (Needs_Memory, Temporal_Recall, Needs_Live_Info, High_Complexity)\n\n"
    "User prompt: {prompt}"
)

_TEMPLATES = {
    1: (V1_NO_CONTEXT, V1_WITH_CONTEXT),
    2: (V2_NO_CONTEXT, V2_WITH_CONTEXT),
}

DEFAULT_VERSION = 2

# The context builder's shape (``classifier._get_context_turns``). Named here so
# the offline pipeline reproduces the live prefix EXACTLY — same unit, same
# per-turn cap, same total budget.
#
# The unit is an EXCHANGE, not a message: `main.py` stores one episodic row per
# turn as "User: …\n\nAssistant: …", and the live context block is the last three
# of those. An offline builder that took the last three *messages* would show the
# model roughly half the history it sees in production, in a different surface
# form — the exact train/inference drift this module exists to prevent.
CONTEXT_TURNS = 3
CONTEXT_MAX_WORDS = 500
TURN_WORD_CAP = 150            # per exchange, before the total budget applies
EXCHANGE_FORMAT = "User: {user}\n\nAssistant: {assistant}"


def render(prompt: str, context_text: Optional[str] = None,
           version: int = DEFAULT_VERSION) -> str:
    """Render the classifier's encoder input.

    *context_text* is the prior-turn summary block (empty/None ⇒ the standalone
    template). Training and inference MUST both come through here.
    """
    try:
        no_ctx, with_ctx = _TEMPLATES[int(version)]
    except KeyError:
        raise ValueError(f"unknown template version {version!r} "
                         f"(have {sorted(_TEMPLATES)})") from None
    if context_text:
        return with_ctx.format(context_text=context_text, prompt=prompt)
    return no_ctx.format(prompt=prompt)


def cap_turn(text: str, max_words: int = TURN_WORD_CAP) -> str:
    """Trim ONE exchange to the live per-turn cap (mirrors `_get_context_turns`)."""
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max_words]) + "…" if len(words) > max_words else text


def context_from_messages(messages, n_turns: int = CONTEXT_TURNS,
                          max_total_words: int = CONTEXT_MAX_WORDS) -> str:
    """Build the live-shaped context prefix from an ordered message list.

    *messages* is ``[(role, text), …]`` in chronological order, ending just
    before the prompt being classified. Messages are paired into user→assistant
    exchanges the way the runtime stores them, each exchange is capped, and the
    last *n_turns* are joined under the total budget.

    This is the offline twin of ``classifier._get_context_turns``. The one thing
    it cannot reproduce is that the live path prefers a turn's `summary_text`
    when the background summarizer has written one; raw text is the fallback in
    both, and a fresh conversation has no summaries either way.
    """
    exchanges, pending_user = [], None
    for role, text in messages:
        text = (text or "").strip()
        if not text:
            continue
        if role == "user":
            if pending_user is not None:
                exchanges.append(EXCHANGE_FORMAT.format(user=pending_user, assistant=""))
            pending_user = text
        else:
            exchanges.append(EXCHANGE_FORMAT.format(user=pending_user or "", assistant=text))
            pending_user = None
    if pending_user is not None:
        exchanges.append(EXCHANGE_FORMAT.format(user=pending_user, assistant=""))

    recent = [cap_turn(e) for e in exchanges[-n_turns:]]
    return truncate_context(recent, max_total_words=max_total_words)


def truncate_context(turn_texts, max_total_words: int = CONTEXT_MAX_WORDS) -> str:
    """Join prior-turn texts under a word budget, mirroring the live path.

    ``classifier._get_context_turns`` applies this to rows pulled from the DB;
    the offline pipeline applies it to rows pulled from files. Same function so
    the two can't diverge (a context prefix that is longer offline than online
    would reintroduce the very mismatch D3 exists to kill).
    """
    parts, total = [], 0
    for text in turn_texts:
        if not text:
            continue
        words = text.split()
        if total + len(words) > max_total_words:
            remaining = max_total_words - total
            if remaining > 20:
                parts.append(" ".join(words[:remaining]) + "…")
            break
        parts.append(text)
        total += len(words)
    return "\n".join(parts)
