"""F14 D6: raw-log extraction v2 — unformatted dumps → NormalizedConversation.

For a text file with no roles and no timestamps (a pasted transcript, an old
plain-text log), the v1 "amnesia method" split the text into isolated cold
chunks and guessed speakers per chunk with no shared context — boundaries
landed mid-turn. v2 replaces that:

  1. slice at word boundaries into ~2,000-word windows with 200-word overlaps
     (the shared C2 greedy packer);
  2. extract turns per slice with one bg-model call;
  3. **one open seam call per adjacent pair** reconciles the overlap — the
     model sees slice A's boundary turns + slice B's head turns + the raw
     overlap ONCE and returns the corrected boundary (no cold guessing);
  4. dedup on a normalized-alphanumeric hash across the whole dump;
  5. synthetic timestamps END at the file mtime (a file's mtime marks when
     its conversation ended), spaced 1/min backwards, and every turn is
     flagged ``ts_synthetic`` so T-timelines caveat those dates.

The bg-model seams are injectable (``llm``) so tests run stubbed; the real
path uses the shared background client. Empirical deferral (spec): if boundary
errors show in >10% of spot-checked seams, raise ``RAW_OVERLAP_WORDS`` to 400
before touching the prompt.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import structlog

from src.ingestion.formats import NormalizedConversation, NormalizedTurn
from src.memory.chunking import chunk_text

logger = structlog.get_logger("ice.ingestion.raw_slicer")

RAW_SLICE_TOKENS = 2667      # ≈2,000 prose words (rev 12)
RAW_OVERLAP_WORDS = 200
SEAM_TURNS = 3               # boundary turns each side handed to the seam call
_NORM_RE = re.compile(r"[^a-z0-9]+")

_EXTRACT_SYS = "You segment raw chat transcripts into speaker turns."
_EXTRACT_PROMPT = (
    "The text below is a slice of a raw conversation between a user and an AI "
    "assistant, with the speaker labels lost. Split it into turns. Output ONLY "
    "a JSON array of objects {\"role\": \"user\"|\"assistant\", \"text\": \"...\"} "
    "in order. Do not invent content.\n\nText:\n")
_SEAM_PROMPT = (
    "Two adjacent slices of one conversation overlap. Below are the last turns "
    "of slice A, the first turns of slice B, and the raw overlapping text. Some "
    "turns appear in both, and one turn may be split across the boundary. Return "
    "ONLY a JSON array {\"role\", \"text\"} of the CORRECT turns for slice B's "
    "head — drop turns already fully present in slice A, and complete any turn "
    "split across the seam.\n\n")


# ── bg-model seam (injectable) ──────────────────────────────────────────────

def _default_llm(prompt: str, system: str = _EXTRACT_SYS) -> str:
    from src.workers.bg_client_factory import (
        bg_timeout,
        get_bg_client,
        get_bg_model_name,
        json_schema,
    )
    client = get_bg_client()
    completion = client.chat.completions.create(
        model=get_bg_model_name(),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=1500, timeout=bg_timeout(1500),
        response_format=json_schema("turns", {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"role": {"type": "string",
                                              "enum": ["user", "assistant"]},
                                     "text": {"type": "string"}},
                      "required": ["role", "text"]},
        }))
    return completion.choices[0].message.content or ""


def _parse_turns(raw: str, fallback_text: str) -> list:
    """Parse a JSON turn array from a (possibly fenced) model reply. On any
    failure, fall back to one user turn holding the whole text — never drop
    content."""
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        turns = []
        for obj in data:
            role = (obj.get("role") or "").lower()
            text = (obj.get("text") or "").strip()
            if role in ("user", "assistant") and text:
                turns.append({"role": role, "text": text})
        if turns:
            return turns
        logger.warning("raw_slice_no_turns_parsed",
                       chars=len(raw or ""), head=(raw or "")[:120])
    except (json.JSONDecodeError, AttributeError, TypeError, KeyError) as exc:
        # Loud: the fallback below stores the ENTIRE slice as one user turn, so
        # a silent failure here turns a conversation into an undifferentiated
        # blob and nothing says the roles were lost.
        logger.warning("raw_slice_unparseable", error=str(exc)[:120],
                       head=(raw or "")[:120])
    stripped = fallback_text.strip()
    return [{"role": "user", "text": stripped}] if stripped else []


def _extract_slice_turns(slice_text: str, llm: Callable) -> list:
    return _parse_turns(llm(_EXTRACT_PROMPT + slice_text, _EXTRACT_SYS), slice_text)


def _resolve_seam(a_tail: list, b_head: list, overlap_text: str,
                  llm: Callable) -> list:
    """One open seam call: reconcile slice B's head against slice A's tail +
    the raw overlap. Returns the corrected turns for B's head. On failure,
    returns b_head unchanged (the end dedup still removes exact duplicates)."""
    payload = (_SEAM_PROMPT
               + "Slice A tail:\n" + json.dumps(a_tail, ensure_ascii=False)
               + "\n\nSlice B head:\n" + json.dumps(b_head, ensure_ascii=False)
               + "\n\nRaw overlap:\n" + overlap_text)
    try:
        parsed = _parse_turns(llm(payload, _EXTRACT_SYS), "")
        return parsed if parsed else b_head
    except Exception as exc:
        logger.warning("raw_seam_resolve_failed", error=str(exc))
        return b_head


# ── dedup + synthetic time ──────────────────────────────────────────────────

def _norm(text: str) -> str:
    return _NORM_RE.sub("", text.lower())


def _dedup(turns: list) -> list:
    """Drop later turns whose normalized-alnum text repeats an earlier one
    (rev 12) — the end sweep that cleans residual overlap duplicates."""
    seen = set()
    out = []
    for t in turns:
        h = _norm(t["text"])
        if h and h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out


def _assign_times(turns: list, mtime: datetime) -> list:
    """Synthetic timestamps ending at mtime, one minute apart, backwards."""
    n = len(turns)
    out = []
    for i, t in enumerate(turns):
        ts = mtime - timedelta(minutes=(n - 1 - i))
        out.append(NormalizedTurn(role=t["role"], text=t["text"], ts=ts))
    return out


# ── entry points ────────────────────────────────────────────────────────────

def slice_raw_text(raw: str, title: str, *, llm: Optional[Callable] = None,
                   mtime: Optional[datetime] = None) -> Optional[NormalizedConversation]:
    """Turn one raw dump into a NormalizedConversation (rev 12). ``llm`` is a
    ``(prompt, system) -> str`` callable (stubbed in tests); ``mtime`` anchors
    the synthetic times (defaults to now)."""
    if not raw.strip():
        return None
    llm = llm or _default_llm
    mtime = mtime or datetime.now(timezone.utc)

    slices = chunk_text(raw, max_tokens=RAW_SLICE_TOKENS,
                        overlap_words=RAW_OVERLAP_WORDS)
    if not slices:
        return None

    merged = _extract_slice_turns(slices[0], llm)
    for i in range(1, len(slices)):
        b_turns = _extract_slice_turns(slices[i], llm)
        if not b_turns:
            continue
        a_tail = merged[-SEAM_TURNS:]
        b_head = b_turns[:SEAM_TURNS]
        overlap = " ".join(slices[i].split()[:RAW_OVERLAP_WORDS])
        reconciled_head = _resolve_seam(a_tail, b_head, overlap, llm)
        merged = merged + reconciled_head + b_turns[SEAM_TURNS:]

    deduped = _dedup(merged)
    turns = _assign_times(deduped, mtime)
    logger.info("raw_slice_complete", title=title[:80], slices=len(slices),
                turns=len(turns))
    return NormalizedConversation(
        provider="raw", source_id=title, title=title, turns=turns,
        created_at=turns[0].ts if turns else mtime, updated_at=mtime,
        ts_synthetic=True)


def slice_raw_file(path: str, *, llm: Optional[Callable] = None) -> list:
    """[NormalizedConversation] for a raw ``.txt`` dump — one conversation per
    file, mtime-anchored."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    except OSError:
        mtime = datetime.now(timezone.utc)
    conv = slice_raw_text(raw, os.path.basename(path), llm=llm, mtime=mtime)
    return [conv] if conv is not None else []
