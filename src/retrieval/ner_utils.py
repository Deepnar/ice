"""Shared micro-NER entity extraction — used by BOTH the retrieval
orchestrator (pre-flight: extracting entities from a live user prompt to
match against the Codex graph) and the clustering worker (post-flight:
extracting entities from a completed turn's raw_text to decide cluster
membership).

WHY THIS IS A SEPARATE MODULE
  Previously, the orchestrator had its own _extract_entities_with_ner
  method, and the clustering worker used a much weaker regex-only fallback
  ("capitalized word, not in a small stoplist") instead of the real model.
  The stated reason for the regex fallback was an assumption that the NER
  model "may not be available in worker context" — but the model loading
  is just torch.load() of a local .pt file plus a HuggingFace tokenizer by
  name, with no dependency on request/API context at all. There was no
  real obstacle; clustering.py already loads its own SentenceTransformer
  at module level the same way. Using the real model here means narrative
  clustering benefits from the SAME entity-recognition quality used for
  Codex graph matching, instead of a weaker proxy that has known false
  positives (sentence-initial capitalized words, chapter headers, etc —
  the old regex version needed an explicit stoplist to work around this).

WHY THIS GENERALIZES BEYOND NARRATIVE CONTENT
  Entity overlap as a clustering signal isn't narrative-specific: a
  technical conversation that repeatedly mentions "Codex", "FastAPI",
  "PostgreSQL" across different turns has the same kind of recurring-named-
  thing continuity that character names provide in a story. The mechanism
  was already general-purpose; only the extraction QUALITY needed fixing.

LOADED ONCE, MODULE-LEVEL
  Both the orchestrator and the clustering worker get a singleton model/
  tokenizer instance via get_ner_extractor(), rather than each maintaining
  its own copy — avoids loading the model twice in the same process and
  keeps the loading code in exactly one place to prevent the two callers
  from drifting apart again in the future.
"""

import os
import re
from typing import List, Optional

import torch
from transformers import AutoTokenizer

_ner_model = None
_ner_tokenizer = None
_load_attempted = False


def _load_model():
    from src.classifier.ner_model import MicroNER
    model = MicroNER()
    path = "models/ner/ner_model.pt"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    return None


def _load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    except Exception:
        return None


def _ensure_loaded():
    global _ner_model, _ner_tokenizer, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    _ner_model = _load_model()
    _ner_tokenizer = _load_tokenizer()


# Pronouns / articles / conversational boilerplate that the context-free NER
# sometimes tags as entities — dropped from output to improve precision (the
# same junk the regex-fallback stoplist guards against, applied to model output).
_NER_STOP = {
    "the", "a", "an", "he", "she", "it", "they", "them", "his", "her", "its",
    "their", "this", "that", "these", "those", "i", "we", "you", "me", "us",
    "user", "assistant", "chapter",
}


def _snap_to_words(text: str, start: int, end: int):
    """Expand a char span to whole-word boundaries. The per-subword tagger can
    stop mid-word (e.g. 'Pyd' inside 'Pydantic'); snapping recovers the full
    word so downstream entity matching (and A2 grounding) sees 'pydantic'."""
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        start -= 1
    while end < len(text) and (text[end].isalnum() or text[end] == "_"):
        end += 1
    return start, end


# Function words that are never the head/tail of a real entity — trimmed from
# both ends to clean the context-free tagger's boundary bleed ('on Pydantic'
# -> 'Pydantic'). Verb-led bleed ('uses PostgreSQL') is NOT handled here; that
# needs a context-aware model (roadmap A9).
_EDGE_TRIM = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "with", "for",
    "at", "by", "from", "over", "under", "as", "that", "this", "into", "onto",
}


def _clean_entity(s: str) -> str:
    """Trim leading/trailing function words from an entity span."""
    parts = s.strip().split()
    while parts and parts[0].lower() in _EDGE_TRIM:
        parts.pop(0)
    while parts and parts[-1].lower() in _EDGE_TRIM:
        parts.pop()
    return " ".join(parts).strip()


def extract_entities(text: str, embedder, max_chars: Optional[int] = None) -> List[str]:
    """Extract entity strings from *text* using the real MicroNER model.

    Falls back to a capitalized-word regex ONLY if the model or tokenizer
    genuinely failed to load (e.g. the .pt file is missing) — this is a
    safety net for a broken deployment, not the intended default path.
    *embedder* must be a SentenceTransformer-compatible object with
    .encode(), since MicroNER classifies over token embeddings rather than
    raw token ids.

    *max_chars*: the orchestrator's original use case was a short live user
    prompt (10-30 words) — this function was never exercised against much
    longer text. Clustering calls this on FULL CHAPTERS (800-1200+ words),
    where tokenizing and embedding the entire text token-by-token is
    unnecessary overhead for a background worker processing up to 50 turns
    per run (latency compounds across the batch in a way it never did for
    a single live prompt). Pass max_chars to truncate before NER runs — a
    recurring cast is almost always introduced early in a chapter/scene, so
    truncating to the first N characters captures the same entities at a
    fraction of the cost. None (default) means no truncation, preserving
    the original behavior for the orchestrator's live-prompt call site.
    """
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]

    _ensure_loaded()

    if _ner_model is None or _ner_tokenizer is None:
        # Genuine fallback only — log-worthy if this fires in normal
        # operation, since it means the real model isn't loadable.
        candidates = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        common_caps = {"The", "This", "That", "These", "Those", "When", "Where",
                       "What", "Who", "How", "Why", "After", "Before", "Then",
                       "User", "Assistant", "Chapter"}
        return list({c for c in candidates if c not in common_caps})

    encoding = _ner_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    if not token_ids:
        return []
    token_strs = _ner_tokenizer.convert_ids_to_tokens(token_ids)
    offsets = encoding["offset_mapping"]

    embeddings = embedder.encode(token_strs, convert_to_tensor=True, show_progress_bar=False)
    model_device = next(_ner_model.parameters()).device
    if embeddings.device != model_device:
        embeddings = embeddings.to(model_device)
    if embeddings.dtype != torch.float32:
        embeddings = embeddings.float()

    with torch.no_grad():
        logits = _ner_model(embeddings.unsqueeze(0))
        preds = torch.argmax(logits, dim=-1).squeeze(0)

    entities = []
    current_start = None
    current_end = None

    for i, p in enumerate(preds.tolist()):
        tok_start, tok_end = offsets[i]
        if p == 0:  # B-ENT
            if current_start is not None:
                entities.append((current_start, current_end,
                                 text[current_start:current_end].strip()))
            current_start = tok_start
            current_end = tok_end
        elif p == 1 and current_start is not None:  # I-ENT
            current_end = tok_end
        else:
            if current_start is not None:
                entities.append((current_start, current_end,
                                 text[current_start:current_end].strip()))
                current_start = None
    if current_start is not None:
        entities.append((current_start, current_end,
                         text[current_start:current_end].strip()))

    # Glue consecutive entities separated only by whitespace
    if len(entities) >= 2:
        glued = []
        prev_start, prev_end, prev_str = entities[0]
        for i in range(1, len(entities)):
            curr_start, curr_end, curr_str = entities[i]
            if text[prev_end:curr_start].strip() == "":
                prev_end = curr_end
                prev_str = text[prev_start:prev_end].strip()
            else:
                glued.append((prev_start, prev_end, prev_str))
                prev_start, prev_end, prev_str = curr_start, curr_end, curr_str
        glued.append((prev_start, prev_end, prev_str))
        entities = glued

    # Per-entity cleanup (applied after gluing so it can't trigger new merges):
    # snap partial-word spans to whole words (fixes 'Pyd' -> 'Pydantic'), strip
    # leading articles, drop pronoun/boilerplate junk, dedup.
    cleaned, seen = [], set()
    for s, e, _ in entities:
        s, e = _snap_to_words(text, s, e)
        c = _clean_entity(text[s:e].strip())
        cl = c.lower()
        if c and cl not in _NER_STOP and cl not in seen:
            seen.add(cl)
            cleaned.append(c)
    return cleaned