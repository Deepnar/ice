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

CHUNK_TOKENS = 550                    # target tokens per chunk (A1; shared with A2 NER)
OVERLAP_WORDS = 50                    # word overlap carried into the next chunk

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str, is_code: bool = False) -> int:
    """Rough token estimate. Code tokenizes far heavier than prose (symbols,
    no word spacing), so for code we take the larger of a word- and a
    char-based estimate rather than the prose words*1.33 heuristic."""
    word_est = len(text.split()) * 1.33
    if is_code:
        return int(max(word_est, len(text) / 3.0))
    return int(word_est)


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
    sentences for prose, non-blank lines for code. Each unit is (unit, is_code)."""
    units = []
    for seg, is_code in split_segments(text):
        if is_code:
            units.extend((ln, True) for ln in seg.splitlines() if ln.strip())
        else:
            units.extend((s, False) for s in _SENTENCE_SPLIT_RE.split(seg.strip()) if s.strip())
    return units


def chunk_text(text: str, max_tokens: int = CHUNK_TOKENS, overlap_words: int = OVERLAP_WORDS) -> list:
    """Split *text* into ~max_tokens chunks on sentence/code-line boundaries,
    carrying overlap_words of context into each subsequent chunk. A single
    unit larger than the budget is hard word-split as a last resort."""
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

    for unit_text, is_code in units:
        ut = estimate_tokens(unit_text, is_code)
        if ut > max_tokens:
            # Oversized single unit (e.g. a minified line): flush, then hard-split.
            flush()
            words = unit_text.split()
            step = max(1, int(max_tokens / 1.33))
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + step]))
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
