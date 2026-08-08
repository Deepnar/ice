"""Shared chunking primitive (C2) — extracted verbatim from codex_extractor
(A1) so one implementation serves all consumers, per the density sub-project
rule ("build one chunker"):

  * Codex extraction windows (A1 — the original home),
  * big pasted-input document chunks (C2, workers/document_chunker.py),
  * chunk-aware retrieval (C3) and the doc pipeline (C12) later.

Mechanics (A1, unchanged): fenced code blocks are isolated from prose; atomic
units — sentences for prose, non-blank lines for code — are never split across
a chunk boundary; units greedy-pack to ~``max_tokens`` with ``overlap_words``
of trailing context carried into the next chunk; code is token-estimated
heavier than prose; a single oversized unit is hard word-split as a last
resort.
"""

import re

from src.api.config import settings
from src.memory.tokens import count as count_tokens


_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# C3: markdown headings are hard section boundaries — a chunk never spans one.
_HEADING_RE = re.compile(r"(?m)^(#{1,6}\s.*)$")


def estimate_tokens(text: str, is_code: bool = False) -> int:
    """Token count for the packer. C16: this is now the REAL count.

    It used to be `words * 1.33` (with a char-based floor for code), which
    measured 0.53x on fenced Python — so a chunk the packer believed was 550
    tokens was really ~1,000, and the chunk budget meant nothing for exactly the
    content where chunk size matters most. `is_code` is now vestigial for
    accuracy and kept only so callers need no change; it costs one tokenizer
    call per atomic unit (a sentence or a code line), ~1 ms per turn.

    Behavioural consequence, stated plainly: chunks are now sized correctly,
    which means code chunks get SMALLER and prose chunks get slightly LARGER
    than before. Nothing needed re-chunking because the store was empty when
    this landed.
    """
    return count_tokens(text)


def split_segments(text: str):
    """Split *text* into ordered (segment, is_code) pairs, isolating fenced
    code blocks from surrounding prose so each gets its own unit strategy."""
    segments = []
    idx = 0
    for m in _CODE_FENCE_RE.finditer(text):
        if m.start() > idx:
            segments.append((text[idx:m.start()], False))
        segments.append((m.group(0), True))
        idx = m.end()
    if idx < len(text):
        segments.append((text[idx:], False))
    return segments


def atomic_units(text: str):
    """Break *text* into atomic units that must never be split across chunks:
    sentences for prose, non-blank lines for code. Each unit is
    ``(unit, is_code, is_boundary)`` — C3: markdown headings are boundary
    units (section starts), and the packer flushes before them so a chunk
    never spans a section break."""
    units = []
    for seg, is_code in split_segments(text):
        if is_code:
            units.extend((ln, True, False) for ln in seg.splitlines() if ln.strip())
        else:
            # Split the prose segment on heading lines first (section-level
            # decomposition), then sentences within each block.
            parts = _HEADING_RE.split(seg)
            for part in parts:
                if not part.strip():
                    continue
                if _HEADING_RE.fullmatch(part.strip()):
                    units.append((part.strip(), False, True))
                else:
                    units.extend((s, False, False)
                                 for s in _SENTENCE_SPLIT_RE.split(part.strip()) if s.strip())
    return units


def _hard_split(unit_text: str, max_tokens: int) -> list:
    """Last resort for a single unit larger than the whole budget.

    Words first. Then characters, because a word boundary is not guaranteed to
    exist: minified JS, a base64 blob and a comma-joined identifier list are
    all ONE "word" and the word-split silently returned them whole — a 3,000-
    token chunk from a 550-token cap, which then blows every budget downstream
    while claiming to be a chunk.
    """
    out, piece, used = [], [], 0
    for w in unit_text.split():
        wt = count_tokens(w + " ")
        if wt > max_tokens:
            if piece:
                out.append(" ".join(piece))
                piece, used = [], 0
            # No word boundary to use — cut on characters, sized from this
            # blob's own measured density rather than an assumed ratio.
            per_char = max(1e-6, wt / max(1, len(w)))
            step = max(1, int(max_tokens / per_char))
            out.extend(w[i:i + step] for i in range(0, len(w), step))
            continue
        if piece and used + wt > max_tokens:
            out.append(" ".join(piece))
            piece, used = [], 0
        piece.append(w)
        used += wt
    if piece:
        out.append(" ".join(piece))
    return out


def chunk_text(text: str, max_tokens: int = None, overlap_words: int = None) -> list:
    """Split *text* into ~max_tokens chunks on sentence/code-line boundaries,
    carrying overlap_words of context into each subsequent chunk. A single
    unit larger than the budget is hard word-split as a last resort.

    G9: both default to None and resolve from settings here rather than in the
    signature — a default argument is evaluated once at import, which would
    freeze the value against a Z1 sweep."""
    if max_tokens is None:
        max_tokens = settings.chunk_tokens
    if overlap_words is None:
        overlap_words = settings.chunk_overlap_words
    units = atomic_units(text)
    if not units:
        return [text] if text.strip() else []

    chunks = []
    current = []          # list of unit strings in the chunk being built
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append("\n".join(current))
            current, current_tokens = [], 0

    for unit_text, is_code, is_boundary in units:
        # +1 for the "\n" this unit contributes when the chunk is joined.
        # Units are counted individually but stored joined, and the tokenizer
        # does not merge across that newline for free — without this the
        # packer overshot the cap by ~24% on code, where units are lines and
        # there are many of them.
        ut = estimate_tokens(unit_text, is_code) + 1
        if is_boundary and current:
            # C3: a heading starts a new section — never let a chunk span it.
            # (No overlap carried across a section boundary either: the
            # previous section's tail isn't context for the new one.)
            flush()
        if ut > max_tokens:
            # Oversized single unit (e.g. a minified line): flush, then hard-split.
            flush()
            # Word-step the split, then verify: the old `max_tokens / 1.33`
            # inverse assumed the same wrong ratio the estimate did, so an
            # oversized code line was split into pieces still over budget.
            chunks.extend(_hard_split(unit_text, max_tokens))
            continue
        if current and current_tokens + ut > max_tokens:
            prev = "\n".join(current)
            flush()
            overlap = " ".join(prev.split()[-overlap_words:]) if overlap_words else ""
            if overlap:
                current = [overlap]
                current_tokens = estimate_tokens(overlap)
        current.append(unit_text)
        current_tokens += ut
    flush()
    return chunks
