"""Codex Extractor Subsystem – Structural Ingestion Plane."""

import hashlib
import json
import re
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import structlog
from sentence_transformers import SentenceTransformer
from src.api.config import settings
from src.api.db import SessionLocal
from sqlalchemy.orm.attributes import flag_modified

from src.memory.models import (
    CodexEntity, CodexEdge, CodexEvent, IdempotencyKey, EpisodicMemory, ReviewQueue
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy, is_user_active
from src.retrieval.ner_utils import extract_entities

# Module‑level embedder for new entity embeddings
embedder = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cpu",
    truncate_dim=384
)

logger = structlog.get_logger("ice.workers.codex")

# Dedicated extraction client (port 8003)
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
bg_client = get_bg_client()
CODEX_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

# ===================================================================
# Controlled relation vocabulary for Codex 2.0 — v2 (redesigned)
#
# Design rules:
# 1. One canonical relation per *meaning*. Near-synonyms that described
#    the exact same fact (is_married_to/married_to, duplicate reports_to,
#    duplicate follows/subscribes_to, etc.) were collapsed to a single
#    relation. Relations that *look* similar but describe genuinely
#    different facts (knows vs friend vs colleague) were kept distinct.
# 2. Generic dumping-ground relations ("is", "has", "applies_to",
#    "connects_to") were removed. Every fact they could have carried has
#    a more specific relation below — forcing specificity at extraction
#    time is what keeps the graph queryable later. If the LLM has
#    nowhere specific to put a fact, it now correctly skips it instead
#    of burying it under "is".
# 3. Relations are grouped into categories (dicts, not just comments).
#    The prompt below renders these grouped, instead of one flat sorted
#    list of 150+ strings. This is the main lever against synonym
#    fragmentation: an LLM choosing within a labeled 5-item group is far
#    more consistent than choosing within one flat 150-item list.
# 4. Single vs multi-valued reassigned by real-world cardinality:
#    works_on, founded_by, created, manufactured_by, sold_by,
#    published_by, produced_by, distributed_by moved from single->multi
#    because real entities can have multiple of each simultaneously.
# ===================================================================

# --- PROPERTY_RELATIONS ---
# Facts about the SUBJECT itself. Written to properties JSONB.
# Each must hold "at most one current value" semantics.

PROPERTY_RELATIONS_BY_CATEGORY = {
    "identity": {
        "name", "alias", "nickname", "full_name", "title", "description",
    },
    "demographics": {
        "age", "gender", "species", "nationality", "religion",
        "birthday", "blood_type",
    },
    "appearance": {
        "height", "weight", "eye_color", "hair_color",
    },
    "professional": {
        "role", "occupation", "profession", "affiliation", "status",
    },
    "contact_location": {
        "email", "url", "home", "phone",
    },
    "metadata_generic": {
        "type", "genre", "format", "license", "version", "language",
    },
    "task_management": {
        "difficulty", "priority", "deadline", "budget", "duration",
    },
    "attribution": {
        "author",
    },
    "technical_specs": {
        # performance/measurement facts about a model, pipeline, script,
        # or system — distinct from "metadata_generic" because these are
        # numeric/measured facts, not descriptive labels
        "accuracy", "throughput", "runtime", "latency", "size",
        "resolution", "capacity",
    },
    "narrative_metadata": {
        # facts about a fictional entity (FLAW characters, sagas, systems)
        # that don't fit the person-demographic categories above
        "power_system", "universe", "timeline_position",
    },
    "abstract_metadata": {
        # facts about a concept / theory / taxonomy entity
        "definition", "scope", "unit",
    },
}
PROPERTY_RELATIONS = set().union(*PROPERTY_RELATIONS_BY_CATEGORY.values())
"""Relations that update the source entity's properties JSONB and expire previous edges."""

# --- MULTI_VALUED_RELATIONS ---
# Many active edges of this relation can coexist from the same source.

MULTI_VALUED_RELATIONS_BY_CATEGORY = {
    "technical_dependency": {
        "uses", "imports", "depends_on", "supports", "integrates_with",
        "calls", "returns", "references", "cites", "extends", "implements",
    },
    "technical_distribution": {
        "manufactured_by", "sold_by", "published_by", "produced_by",
        "distributed_by", "purchased_from",
    },
    "structural_containment": {
        "contains", "features",
    },
    "social_relationship": {
        "friend", "ally", "enemy", "colleague", "knows",
    },
    "social_action": {
        "follows", "subscribes_to", "watches", "listens_to", "reads",
        "comments_on", "reacts_to", "reviews", "shares",
    },
    "organisational_collab": {
        "member_of", "partners_with", "competes_with", "contributes_to",
        "works_with", "collaborates_on", "co_authors", "edits",
        "moderates", "administers", "contributes_code_to",
    },
    "support_endorsement": {
        "funds", "sponsors", "endorses", "criticises",
    },
    "activity_participation": {
        "owns", "writes", "maintains", "teaches", "enrolled_in",
        "attends", "participated_in", "participates_in",
        "competes_in", "volunteers_for",
    },
    "categorisation": {
        "tag", "category",
    },
    "works_on_projects": {
        "works_on",
    },
    "founding_creation_multi": {
        "founded_by", "created",
    },
    "data_lineage": {
        # ML/data-pipeline lineage facts — directly relevant to ICE's
        # dataset combiner/dedup/classifier-training pipeline
        "derived_from", "trained_on", "configured_with",
        "evaluated_on", "benchmarks_against",
    },
    "code_structure": {
        # where code/artifacts live and how they're verified
        "defined_in", "located_in", "tested_by", "documents",
    },
    "research_relations": {
        # for the ICE research papers — distinct from "cites" (a citation
        # can be incidental; these describe an actual methodological or
        # evidentiary relationship between findings/papers)
        "builds_on", "validates", "contradicts", "replicates",
        "extends_findings_of",
    },
    "narrative_structure": {
        # FLAW-specific: character/saga/system relationships that aren't
        # personal relationships (married_to etc.) or generic containment
        "appears_in", "wields", "possesses", "mirrors",
        "foreshadows", "inspired_by",
    },
    "conceptual": {
        # for abstract entities: theories, taxonomies, cosmological layers
        "derived_from_theory", "complements", "opposes",
        "exemplifies", "measures",
    },
}
MULTI_VALUED_RELATIONS = set().union(*MULTI_VALUED_RELATIONS_BY_CATEGORY.values())
"""Relations that allow multiple active edges simultaneously (no auto‑expiry)."""

# --- SINGLE_VALUED_RELATIONS ---
# A new edge of the same relation from the same source auto-expires any
# previous active edge of that relation (regardless of target).

SINGLE_VALUED_RELATIONS_BY_CATEGORY = {
    "organisational_position": {
        "part_of", "works_at", "reports_to", "managed_by", "assigned_to",
        "supervised_by", "supervises", "manages", "directs",
        "employs", "is_employed_by", "is_contracted_by",
    },
    "executive_role": {
        "is_ceo_of", "is_founder_of", "is_president_of",
        "represents", "acts_on_behalf_of", "is_delegated_by",
    },
    "succession": {
        "succeeds", "precedes", "replaces", "supersedes",
    },
    "education": {
        "studies", "studies_at", "graduated_from", "is_educated_in",
        "mentor_of", "student_of", "taught",
    },
    "production_singular": {
        "released", "published", "acquired_by",
    },
    "personal_relationship": {
        "married_to", "is_engaged_to", "is_dating",
        "is_divorced_from", "is_separated_from",
        "parent_of", "child_of", "sibling_of",
    },
    "biography_location": {
        "lives_in", "born_in", "died_in", "is_based_in", "operates_in",
        "is_raised_in",
    },
    "deployment_ownership": {
        "hosted_on", "deployed_to", "owned_by", "operated_by",
    },
    "logical_requirement": {
        "offers", "requires", "provides", "ranks",
    },
    "narrative_singular": {
        # a saga/arc has one primary setting; an entity has one current
        # transformation state at a time (binary-universe/three-phase
        # entities like Orien fit this — only one active phase at once)
        "set_in", "transforms_into",
    },
    "conceptual_singular": {
        # taxonomy/classification facts — an entity is one specific
        # instance/subtype at a time
        "instance_of", "subtype_of",
    },
}
SINGLE_VALUED_RELATIONS = set().union(*SINGLE_VALUED_RELATIONS_BY_CATEGORY.values())
"""Single‑valued relations: a new edge auto‑expires any previous active edge
with the same source and relation."""

# --- Sanity check at import time: catch accidental re-overlap early ---
_overlap = (
    (PROPERTY_RELATIONS & MULTI_VALUED_RELATIONS)
    | (PROPERTY_RELATIONS & SINGLE_VALUED_RELATIONS)
    | (MULTI_VALUED_RELATIONS & SINGLE_VALUED_RELATIONS)
)
assert not _overlap, f"Relation(s) appear in more than one bucket: {_overlap}"

ALLOWED_RELATIONS = PROPERTY_RELATIONS | MULTI_VALUED_RELATIONS | SINGLE_VALUED_RELATIONS

# Grouped view used by the prompt builder below — category label -> sorted relations
_ALL_CATEGORIES_GROUPED = {
    **{f"property: {k}": sorted(v) for k, v in PROPERTY_RELATIONS_BY_CATEGORY.items()},
    **{f"multi-valued: {k}": sorted(v) for k, v in MULTI_VALUED_RELATIONS_BY_CATEGORY.items()},
    **{f"single-valued: {k}": sorted(v) for k, v in SINGLE_VALUED_RELATIONS_BY_CATEGORY.items()},
}

# Raw category keys (e.g. "social_relationship", without the "multi-valued:"
# prefix) — used only to DETECT when the model has mistakenly output a
# category header instead of an actual relation, so it can be logged
# clearly instead of silently vanishing. We deliberately do NOT auto-remap
# these to a specific child relation (e.g. defaulting "social_relationship"
# to "friend") because guessing wrong would silently write an incorrect
# fact (e.g. recording an enemy as a friend) — worse than dropping it.
_CATEGORY_KEYS_ONLY = {
    k.split(": ", 1)[1] if ": " in k else k
    for k in _ALL_CATEGORIES_GROUPED
}


def generate_uuid5(canonical_name: str) -> uuid.UUID:
    """Derive deterministic UUIDv5 identifier for a canonical entity node."""
    return uuid.uuid5(CODEX_NAMESPACE, canonical_name.strip().lower())

# -----------------------------------------------------------------
# Extraction chunking (roadmap A1)
# -----------------------------------------------------------------
# WHY SMALL CHUNKS: a 3–4B extractor's attention dilutes past ~1k tokens,
# so oversized chunks drop mid-passage entities and confuse subject/object
# (the `fastapi uses fastapi` failure). We target ~550 tokens so the same
# chunks can also feed the NER-grounding step (roadmap A2) in one pass.
# WHY SENTENCE/CODE-AWARE BOUNDARIES: raw word windows cut facts in half;
# packing whole sentences (prose) and whole lines (code) keeps each fact
# intact, which is the bigger quality lever than size alone.
CHUNK_TOKENS = 550                    # target tokens per extraction chunk (shared with A2 NER)
OVERLAP_WORDS = 50                    # word overlap carried into the next chunk
MAX_EXTRACTION_TOKENS = 6000          # legacy constant; retained for import compatibility

# A3 — extraction-confidence seeding (stored on codex_edges.extraction_confidence).
CONF_GROUNDED = 0.9     # both terms confirmed by NER grounding
CONF_UNGROUNDED = 0.7   # NER found no entities in the chunk; nothing to ground against
CONF_REJECTED = 0.35    # failed grounding — stored anyway, gated out of retrieval until corroborated

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str, is_code: bool = False) -> int:
    """Rough token estimate. Code tokenizes far heavier than prose (symbols,
    no word spacing), so for code we take the larger of a word- and a
    char-based estimate rather than the prose words*1.33 heuristic."""
    word_est = len(text.split()) * 1.33
    if is_code:
        return int(max(word_est, len(text) / 3.0))
    return int(word_est)


def _split_segments(text: str):
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


def _atomic_units(text: str):
    """Break *text* into atomic units that must never be split across chunks:
    sentences for prose, non-blank lines for code. Each unit is (unit, is_code)."""
    units = []
    for seg, is_code in _split_segments(text):
        if is_code:
            units.extend((ln, True) for ln in seg.splitlines() if ln.strip())
        else:
            units.extend((s, False) for s in _SENTENCE_SPLIT_RE.split(seg.strip()) if s.strip())
    return units


def _chunk_text(text: str, max_tokens: int = CHUNK_TOKENS, overlap_words: int = OVERLAP_WORDS) -> list:
    """Split *text* into ~max_tokens chunks on sentence/code-line boundaries,
    carrying overlap_words of context into each subsequent chunk. A single
    unit larger than the budget is hard word-split as a last resort."""
    units = _atomic_units(text)
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
        ut = _estimate_tokens(unit_text, is_code)
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
                current_tokens = _estimate_tokens(overlap)
        current.append(unit_text)
        current_tokens += ut
    flush()
    return chunks


# -----------------------------------------------------------------
# NER grounding (roadmap A2)
# -----------------------------------------------------------------
# The CPU micro-NER model is the trusted anchor for *which entities exist*.
# The extraction LLM's only job is to relate them — so a triplet naming an
# entity NER never confirmed is treated as a hallucination and dropped.
# This is the seam for A3: instead of dropping `rejected`, A3 will keep them
# as low-confidence edges. Property relations are special: their object is a
# value/descriptor (e.g. role="fire mage"), so only the subject is grounded.
def _normalize_term(s: str) -> str:
    """Lowercase, drop a leading article, strip punctuation, collapse spaces —
    so NER's verbatim strings and the LLM's canonicalised output compare fairly."""
    s = s.strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _ground_triplets(triplets: list, ner_entities: List[str]):
    """Split *triplets* into (grounded, rejected) against the NER-confirmed
    entity list. A term is grounded if its normalised form equals a confirmed
    entity or its token set is a subset (either direction) of one — so
    shortened ('citadel' ⊂ 'obsidian citadel') and qualified mentions still
    ground, while invented entities with no token overlap are rejected."""
    norm_entities = set()
    entity_token_sets = []
    for e in ner_entities:
        ne = _normalize_term(e)
        if ne:
            norm_entities.add(ne)
            entity_token_sets.append(frozenset(ne.split()))

    def _grounded(term: str) -> bool:
        nt = _normalize_term(term)
        if not nt:
            return False
        if nt in norm_entities:
            return True
        t_tokens = frozenset(nt.split())
        if not t_tokens:
            return False
        return any(t_tokens <= e or e <= t_tokens for e in entity_token_sets)

    keep, drop = [], []
    for t in triplets:
        subj_ok = _grounded(t.get("subject", ""))
        # Property relations carry a value object, not an entity — ground subject only.
        obj_ok = True if t.get("relation") in PROPERTY_RELATIONS else _grounded(t.get("object", ""))
        (keep if (subj_ok and obj_ok) else drop).append(t)
    return keep, drop


def _build_grouped_relation_block() -> str:
    """Render ALLOWED_RELATIONS as labeled groups instead of one flat list.

    This is the main lever against synonym fragmentation (e.g. the model
    picking "knows" vs "friend" vs "follows" inconsistently for the same
    kind of fact): choosing within a small labeled group is far more
    consistent than choosing within one 150-item flat list.

    IMPORTANT: the "# category:" line is a section header for human/LLM
    readability ONLY. It is never itself a valid relation value — only the
    individual words listed under each header are valid. This is rendered
    as a comment-style header (not bracketed inline) specifically so it
    cannot be mistaken for an item in the list that follows.
    """
    lines = []
    for label, relations in _ALL_CATEGORIES_GROUPED.items():
        lines.append(f"   # category: {label} (the category name itself is NOT a relation)")
        lines.append(f"   {', '.join(relations)}")
    return "\n".join(lines)


def extract_triplets(text: str, model_override: str = "", topic_tags: Optional[List[str]] = None) -> list:
    """Extract structured triplets using a controlled relation vocabulary."""
    grouped_relations = _build_grouped_relation_block()

    prompt = (
        "You are a precise fact extractor. Convert the given text into a JSON array of "
        "subject‑relation‑object triplets.\n\n"
        "STRICT RULES:\n"
        "1. Use ONLY the individual relation words listed below (e.g. uses, friend, lives_in). "
        "Relations are grouped under '# category: ...' comment headers purely to help you pick "
        "the most precise word when several similar relations exist — the category header itself "
        "is NEVER a valid relation value. Pick one specific word from inside a group, never the "
        "header text above it.\n"
        f"{grouped_relations}\n"
        "   If a fact does not naturally and clearly fit any of these individual relation words, "
        "SKIP IT – never invent a new relation, never output a category name, and never force a "
        "fact into a relation that doesn't truly describe it.\n"
        "2. Canonicalise subjects and objects: lowercase, singular, no punctuation, concise.\n"
        "   Example: \"PostgreSQL\" → \"postgresql\", \"the goo blade\" → \"goo blade\".\n"
        "3. For facts that describe a property of something (e.g., name, age, role, profession, description), "
        "use the property relation itself as the relation. Example:\n"
        "   \"Kael is a fire mage\" → {\"subject\":\"kael\",\"relation\":\"role\",\"object\":\"fire mage\"}\n"
        "4. A relation must make logical sense. If the subject and object could not appear in a "
        "real‑world sentence using that relation, do NOT output it.\n"
        "   BAD:  {\"subject\":\"shinchan\",\"relation\":\"makes\",\"object\":\"shinchan blush\"}\n"
        "   GOOD: {\"subject\":\"shinchan\",\"relation\":\"competes_with\",\"object\":\"rika miyamoto\"}\n"
        "5. NEVER output a category header as a relation. Example:\n"
        "   BAD:  {\"subject\":\"shinchan\",\"relation\":\"social_relationship\",\"object\":\"kazama\"}\n"
        "   GOOD: {\"subject\":\"shinchan\",\"relation\":\"friend\",\"object\":\"kazama\"}\n"
        "6. NEGATION: if the text says a relationship does NOT hold or stopped holding "
        "(e.g. \"X no longer uses Y\", \"X is not allied with Z\", \"they are no longer friends\"), "
        "use the SAME positive relation word from the list above and add \"negated\": true to that "
        "triplet. Only negate a relation that exists in the list; if the negative idea has no "
        "matching relation word, SKIP it. Examples:\n"
        "   \"ICE no longer uses PostgreSQL\" → {\"subject\":\"ice\",\"relation\":\"uses\","
        "\"object\":\"postgresql\",\"negated\":true}\n"
        "   \"Kael and Orien are no longer allies\" → {\"subject\":\"kael\",\"relation\":\"ally\","
        "\"object\":\"orien\",\"negated\":true}\n"
        "7. Output ONLY a JSON array. No markdown, no explanation.\n\n"
        "EXAMPLES:\n"
        "Text: \"ICE uses PostgreSQL for memory and Redis for tasks.\"\n"
        "Output: [{\"subject\":\"ice\",\"relation\":\"uses\",\"object\":\"postgresql\"},"
        " {\"subject\":\"ice\",\"relation\":\"uses\",\"object\":\"redis\"}]\n\n"
        "Text: \"My character Kael is a fire mage from the northern kingdom.\"\n"
        "Output: [{\"subject\":\"kael\",\"relation\":\"role\",\"object\":\"fire mage\"},"
        " {\"subject\":\"kael\",\"relation\":\"home\",\"object\":\"northern kingdom\"}]\n\n"
        "Text: \"FastAPI extends Starlette and depends on Pydantic.\"\n"
        "Output: [{\"subject\":\"fastapi\",\"relation\":\"extends\",\"object\":\"starlette\"},"
        " {\"subject\":\"fastapi\",\"relation\":\"depends_on\",\"object\":\"pydantic\"}]\n"
    )

    # Optional code‑specific instructions
    code_prompt = ""
    if topic_tags and "Software_&_Tech" in topic_tags:
        code_prompt = (
            "\nAdditionally, extract code‑specific entities like function names, class names, "
            "library names, and technical dependencies. Use relations such as "
            "\"uses\", \"imports\", \"extends\", \"implements\", \"calls\", \"returns\".\n"
            "Examples:\n"
            "Text: \"Function calculate_total uses library numpy.\"\n"
            "Output: [{\"subject\":\"calculate_total\",\"relation\":\"uses\",\"object\":\"numpy\"}]\n"
            "Text: \"Class DataLoader extends Dataset.\"\n"
            "Output: [{\"subject\":\"dataloader\",\"relation\":\"extends\",\"object\":\"dataset\"}]\n"
        )

    try:
        model_name = model_override if model_override else get_bg_model_name()

        # --- Chunking: sentence/code-aware ~CHUNK_TOKENS windows (roadmap A1).
        # Short turns come back as a single chunk; the chunker decides.
        chunks = _chunk_text(text)
        if len(chunks) > 1:
            logger.info("extraction_chunking", n_chunks=len(chunks),
                        estimated_tokens=_estimate_tokens(text))

        all_triplets = []
        for chunk in chunks:
            # NER grounding (roadmap A2): confirm entities on the CPU first, then
            # constrain the LLM to relate only those. Reuses A1's chunk.
            ner_entities = extract_entities(chunk, embedder)
            entity_block = ""
            if ner_entities:
                confirmed = ", ".join(dict.fromkeys(ner_entities))  # dedup, keep order
                entity_block = (
                    "\n\nCONFIRMED ENTITIES (use ONLY these as subjects, and as objects "
                    "for relations between two entities; do NOT introduce named entities "
                    f"not in this list):\n{confirmed}"
                )

            chunk_prompt = prompt + code_prompt + "\nNow process this text:"
            completion = bg_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a JSON-only fact extraction tool. Never output anything but JSON."},
                    {"role": "user", "content": f"Text:\n{chunk}{entity_block}\n\n{chunk_prompt}"}
                ],
                temperature=0.0,
                max_tokens=500,
                timeout=30.0
            )
            raw = completion.choices[0].message.content.strip()
            logger.debug("extraction_raw_response", raw=raw[:200])

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # Parse JSON
            decoder = json.JSONDecoder()
            try:
                parsed, _ = decoder.raw_decode(raw)
            except json.JSONDecodeError:
                # Fallback regex for individual triplet objects
                triplet_pattern = re.compile(
                    r'\{\s*"subject"\s*:\s*"([^"]+)"\s*,\s*"relation"\s*:\s*"([^"]+)"\s*,\s*"object"\s*:\s*"([^"]+)"\s*\}',
                    re.DOTALL
                )
                matches = triplet_pattern.findall(raw)
                if matches:
                    chunk_triplets = [{"subject": s, "relation": r, "object": o} for s, r, o in matches]
                else:
                    chunk_triplets = []
            else:
                if isinstance(parsed, list):
                    chunk_triplets = [item for item in parsed if isinstance(item, dict) and all(k in item for k in ("subject","relation","object"))]
                else:
                    chunk_triplets = []

            # Keep only triplets with allowed relations
            chunk_triplets = [t for t in chunk_triplets if t.get("relation") in ALLOWED_RELATIONS]

            # Sanity filter: remove triplets where object is clearly a verb phrase
            suspicious_objects = {"blush", "laugh", "cry", "smile", "angry", "sad", "happy", "mad"}
            chunk_triplets = [t for t in chunk_triplets
                              if t.get("object", "").strip().lower() not in suspicious_objects]

            # Drop self-referential triplets ("fastapi uses fastapi") — an
            # attention-dilution artifact the A1/A2 work targets; grounding
            # alone can't catch it since both terms are confirmed entities.
            chunk_triplets = [t for t in chunk_triplets
                              if _normalize_term(t.get("subject", "")) != _normalize_term(t.get("object", ""))]

            # NER grounding → extraction confidence (A3, completing the A2 seam):
            # grounded triplets are trusted high; grounding-REJECTED triplets are
            # no longer dropped — they enter the graph at low confidence, where
            # retrieval's dynamic thresholds keep them out of context until
            # corroborated (or they decay out). No-NER chunks get mid confidence
            # (nothing to ground against).
            if ner_entities:
                grounded, rejected = _ground_triplets(chunk_triplets, ner_entities)
                for t in grounded:
                    t["confidence"] = CONF_GROUNDED
                for t in rejected:
                    t["confidence"] = CONF_REJECTED
                if rejected:
                    logger.info("codex_grounding",
                                kept=len(grounded), rejected=len(rejected),
                                samples=[f'{t.get("subject")}|{t.get("relation")}|{t.get("object")}'
                                         for t in rejected[:8]])
                chunk_triplets = grounded + rejected
            else:
                for t in chunk_triplets:
                    t["confidence"] = CONF_UNGROUNDED

            all_triplets.extend(chunk_triplets)

        # Deduplicate by (subject, relation, object, negated), keeping the highest
        # confidence seen. Polarity is part of the key: "uses" and "NOT uses" of
        # the same pair are distinct facts (A8).
        by_key = {}
        for t in all_triplets:
            key = (t["subject"].strip().lower(), t["relation"],
                   t["object"].strip().lower(), bool(t.get("negated", False)))
            prev = by_key.get(key)
            if prev is None or t.get("confidence", 0) > prev.get("confidence", 0):
                by_key[key] = t
        return list(by_key.values())

    except Exception as err:
        logger.error("triplet_parsing_failed", error=str(err))
        return []



def get_or_create_entity(db, name: str) -> CodexEntity:
    """Resolves structural identity records across global name and alias spaces."""
    canonical = name.strip().lower()
    entity = db.query(CodexEntity).filter_by(canonical_name=canonical).first()
    if entity:
        return entity

    entity = db.query(CodexEntity).filter(CodexEntity.aliases.any(canonical)).first()
    if entity:
        return entity

    new_entity = CodexEntity(
        id=generate_uuid5(canonical),
        canonical_name=canonical,
        aliases=[name],
        tags=[],
        properties={},
        context_payload="",
        embedding=embedder.encode(canonical, convert_to_tensor=False).tolist(),
        last_updated=datetime.now(timezone.utc)
    )
    db.add(new_entity)
    db.flush()
    return new_entity

# A7: relation → likely type of the relation's SOURCE entity. Used to infer a
# structural entity_type from how an entity is talked about. Deterministic code
# types (function/class/file/module) are set directly by the code graph (E1b).
_TYPE_HINTS = {
    "person": {"role", "occupation", "profession", "married_to", "is_dating",
               "is_divorced_from", "is_separated_from", "parent_of", "child_of",
               "sibling_of", "friend", "enemy", "ally", "colleague", "knows",
               "mentor_of", "student_of", "reports_to", "works_at", "born_in",
               "lives_in", "died_in", "wields", "possesses", "age", "gender"},
    "software": {"uses", "imports", "depends_on", "extends", "implements", "calls",
                 "returns", "integrates_with", "supports", "hosted_on", "deployed_to",
                 "references", "cites", "trained_on", "evaluated_on", "configured_with",
                 "defined_in", "tested_by", "derived_from"},
    "place": {"located_in", "contains", "set_in", "operates_in", "is_based_in", "capital_of"},
    "organization": {"founded_by", "member_of", "employs", "acquired_by", "partners_with",
                     "is_ceo_of", "is_founder_of"},
    "concept": {"instance_of", "subtype_of", "exemplifies", "derived_from_theory",
                "complements", "opposes", "measures"},
}
# Tags we accept directly as a structural type. ICE is a general-purpose memory
# for ALL domains — coding, research, academic, business, personal, creative —
# so the vocabulary spans them; a normaliser folds common synonyms onto a
# canonical type. entity_type is open: any tag can be a type, these are just the
# recognised ones with inference support.
_TYPE_SYNONYMS = {"location": "place", "org": "organization", "company": "organization",
                  "tool": "software", "library": "software", "framework": "software",
                  "app": "software", "npc": "character", "char": "character",
                  "paper": "document", "article": "document", "metric": "concept",
                  "theory": "concept", "topic": "concept"}
_KNOWN_TYPES = set(_TYPE_HINTS.keys()) | {
    "person", "place", "organization", "event", "concept", "object",    # universal
    "software", "function", "class", "file", "module", "dataset",       # coding / research
    "document", "product",                                              # academic / business
    "character", "location", "item", "creature", "faction",             # creative / narrative
}


def _infer_entity_type(relations, tags, current: str) -> str:
    """Infer a structural type: an explicit known type-tag wins; else vote by the
    entity's outgoing relations; else keep the current value."""
    for t in (tags or []):
        tl = t.lower()
        tl = _TYPE_SYNONYMS.get(tl, tl)
        if tl in _KNOWN_TYPES:
            return tl
    votes = {}
    for rel in relations:
        for etype, rels in _TYPE_HINTS.items():
            if rel in rels:
                votes[etype] = votes.get(etype, 0) + 1
    if votes:
        return max(votes, key=votes.get)
    return current or "entity"


class _N:  # sentinel: a missing target/source entity renders as "?"
    canonical_name = "?"


def _regenerate_context_payload(entity: CodexEntity, db) -> None:
    """A7: rebuild context_payload as a rich, bidirectional 'note': the enriched
    description (note body), then properties, then outgoing links AND incoming
    backlinks (Obsidian-style). Also infers entity_type from the relations."""
    out_edges = db.query(CodexEdge).filter(
        CodexEdge.source_id == entity.id,
        CodexEdge.valid_until == None
    ).order_by(CodexEdge.strength.desc()).limit(20).all()
    in_edges = db.query(CodexEdge).filter(
        CodexEdge.target_id == entity.id,
        CodexEdge.valid_until == None
    ).order_by(CodexEdge.strength.desc()).limit(20).all()

    entity.entity_type = _infer_entity_type(
        [e.relation for e in out_edges] + [e.relation for e in in_edges],
        entity.tags, entity.entity_type)

    parts = []
    if entity.description:
        parts.append(entity.description.strip())
    if entity.properties:
        props = "; ".join(f"{k}: {v}" for k, v in entity.properties.items())
        if props:
            parts.append(f"Properties: {props}")
    # A8: positive edges → Links/Backlinks; negated edges → a Negations section.
    out_pos = [e for e in out_edges if not e.negated][:10]
    out_neg = [e for e in out_edges if e.negated][:6]
    in_pos = [e for e in in_edges if not e.negated][:10]
    in_neg = [e for e in in_edges if e.negated][:6]
    if out_pos:
        parts.append("Links: " + "; ".join(
            f"{e.relation} → {(db.query(CodexEntity).get(e.target_id) or _N).canonical_name}"
            for e in out_pos))
    if in_pos:
        parts.append("Backlinks: " + "; ".join(
            f"{(db.query(CodexEntity).get(e.source_id) or _N).canonical_name} --{e.relation}→"
            for e in in_pos))
    neg_lines = [f"NOT {e.relation} → {(db.query(CodexEntity).get(e.target_id) or _N).canonical_name}"
                 for e in out_neg]
    neg_lines += [f"{(db.query(CodexEntity).get(e.source_id) or _N).canonical_name} --NOT {e.relation}→"
                  for e in in_neg]
    if neg_lines:
        parts.append("Negations: " + "; ".join(neg_lines))
    entity.context_payload = "\n".join(parts)

# ===================================================================
# A6 — Self-correcting graph (bounded reconciliation loop)
# ===================================================================
# Fixed rules can't catch cross-turn contradictions ("uses postgres" then
# "migrated off postgres") or relationship reversals (friend -> enemy). A6
# adds a CHEAP deterministic conflict check before the fixed rules run;
# antonym reversals resolve deterministically (newer state supersedes), and
# only genuinely ambiguous supersessions touch the LLM (or go to review) —
# so a small model never gets blanket delete/merge authority over the graph.
SUPERSESSION_CUES = (
    "migrated off", "moved off", "no longer", "stopped using", "switched from",
    "switched to", "replaced", "instead of", "deprecated", "abandoned",
    "dropped", "gave up on", "used to", "moved away from", "ditched",
)
_ANTONYM_PAIRS = [
    ("friend", "enemy"), ("ally", "enemy"),
    ("married_to", "is_divorced_from"), ("is_dating", "is_separated_from"),
    ("endorses", "criticises"),
]  # all in the controlled vocabulary (verified); add new pairs only for real relations
ANTONYM_OF: dict = {}
for _a, _b in _ANTONYM_PAIRS:
    ANTONYM_OF.setdefault(_a, set()).add(_b)
    ANTONYM_OF.setdefault(_b, set()).add(_a)


def _entity_name(db, entity_id) -> str:
    if entity_id is None:
        return "?"
    e = db.query(CodexEntity).get(entity_id)
    return e.canonical_name if e else "?"


def check_conflict(db, subj_id, relation: str, obj_id, turn_text: Optional[str]):
    """A6 deterministic conflict pre-filter. Returns a conflict dict or None.
    Runs a DB query only when the relation has a known antonym, or when a
    multi-valued relation coincides with a supersession cue in the turn — so
    the ~95% of triplets with neither take a dict-lookup fast path."""
    antonyms = ANTONYM_OF.get(relation)
    if antonyms:
        old = db.query(CodexEdge).filter(
            CodexEdge.source_id == subj_id,
            CodexEdge.target_id == obj_id,
            CodexEdge.relation.in_(list(antonyms)),
            CodexEdge.valid_until == None,
        ).first()
        if old:
            return {"type": "antonym", "old_edge_id": old.id, "old_relation": old.relation,
                    "old_target_id": old.target_id}

    if relation in MULTI_VALUED_RELATIONS and turn_text:
        tl = turn_text.lower()
        if any(cue in tl for cue in SUPERSESSION_CUES):
            old = db.query(CodexEdge).filter(
                CodexEdge.source_id == subj_id,
                CodexEdge.relation == relation,
                CodexEdge.target_id != obj_id,
                CodexEdge.valid_until == None,
            ).first()
            if old:
                return {"type": "supersession", "old_edge_id": old.id,
                        "old_relation": old.relation, "old_target_id": old.target_id}
    return None


def _expire_edge(db, edge_id, batch_id, reason: str):
    edge = db.query(CodexEdge).get(edge_id)
    if edge and edge.valid_until is None:
        edge.valid_until = datetime.now(timezone.utc)
        db.add(CodexEvent(entity_id=edge.source_id, event_type="edge_expired",
                          payload={"edge_id": str(edge_id), "reason": reason},
                          timestamp=datetime.now(timezone.utc), batch_source=batch_id))


def reconcile_conflict(db, conflict, subj, relation, obj, batch_id,
                       turn_text: Optional[str], reconciler) -> bool:
    """Resolve a detected conflict. Antonym reversals are deterministic (the
    newly-asserted state supersedes its opposite — no LLM). Ambiguous
    supersessions go to *reconciler* (the bounded LLM) if provided, else to
    human review — never auto-expire on a guess. Returns True if the new edge
    should still be written. Callable as a unit so Track D's agent can drive
    it with its own reconciler."""
    if conflict["type"] == "antonym":
        _expire_edge(db, conflict["old_edge_id"], batch_id, "antonym_superseded")
        logger.info("codex_reconcile", type="antonym", decision="expire_old",
                    relation=relation, old_relation=conflict["old_relation"])
        return True

    # supersession — genuinely ambiguous ("migrated off X" vs "considered it").
    decision = "review"
    if reconciler is not None:
        try:
            decision = reconciler({
                "subject": subj.canonical_name, "relation": relation,
                "object": obj.canonical_name, "old_relation": conflict["old_relation"],
                "old_object": _entity_name(db, conflict.get("old_target_id")),
                "turn": turn_text or "",
            }) or "review"
        except Exception as err:
            logger.error("codex_reconcile_llm_failed", error=str(err))
            decision = "review"

    if decision == "expire_old":
        _expire_edge(db, conflict["old_edge_id"], batch_id, "supersession")
    elif decision == "reject_new":
        logger.info("codex_reconcile", type="supersession", decision="reject_new")
        return False
    elif decision != "keep_both":  # review / unknown → keep both, flag human
        db.add(ReviewQueue(item_type="codex_reconciliation", item_content={
            "new": {"subject": subj.canonical_name, "relation": relation, "object": obj.canonical_name},
            "conflict_type": "supersession", "old_edge_id": str(conflict["old_edge_id"]),
            "old_relation": conflict["old_relation"],
            "old_object": _entity_name(db, conflict.get("old_target_id")),
            "turn_excerpt": (turn_text or "")[:300],
        }))
        decision = "review"
    logger.info("codex_reconcile", type="supersession", decision=decision)
    return True


def make_llm_reconciler():
    """A bounded reconciler backed by the background model: one word out, five
    tokens max. Returned as a callable so it can be swapped/stubbed."""
    def _reconcile(ctx: dict) -> str:
        prompt = (
            "Two facts about the same subject may conflict. Using ONLY the conversation "
            "text, decide how to reconcile them.\n"
            f"Existing fact: {ctx['subject']} {ctx['old_relation']} {ctx['old_object']}\n"
            f"New fact: {ctx['subject']} {ctx['relation']} {ctx['object']}\n"
            f"Conversation text: {ctx['turn'][:600]}\n\n"
            "Reply with exactly ONE word:\n"
            "expire_old  — the new fact replaces/supersedes the old one\n"
            "keep_both   — both are true at the same time\n"
            "reject_new  — the new fact is wrong or not actually asserted"
        )
        resp = bg_client.chat.completions.create(
            model=get_bg_model_name(),
            messages=[{"role": "system", "content": "You output exactly one word."},
                      {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10, timeout=20.0)  # >5 so 'expire_old' can't truncate
        out = (resp.choices[0].message.content or "").strip().lower().replace(" ", "_").replace("-", "_")
        for d in ("expire_old", "keep_both", "reject_new"):
            if d in out:
                return d
        return "review"
    return _reconcile


def handle_triplet(db, subject_name: str, relation: str, object_name: str, batch_id: str,
                   extraction_confidence: float = 1.0, turn_text: Optional[str] = None,
                   reconciler=None, negated: bool = False):
    """Integrates extraction assertions into the transaction context,
    with property‑aware updates, auto‑expiry, multi‑valued support,
    and immediate contradiction activation. *extraction_confidence* (A3)
    is the grounding-seeded trust stored on new edges; on reinforcement the
    edge keeps the highest confidence seen (corroboration raises trust).
    *turn_text* / *reconciler* drive the A6 reconciliation loop (below).
    *negated* (A8) stores the relation's negative polarity."""

    subj = get_or_create_entity(db, subject_name)
    obj  = get_or_create_entity(db, object_name)

    # ── A8: negated assertion ("X no longer uses Y", "X distrusts Y") ──
    # A negation retracts the matching POSITIVE edge (the fact stopped being
    # true), and is itself stored as a negative fact so retrieval can surface
    # "X does NOT relate to Y". Handled up front, separate from the positive
    # write rules below.
    if negated:
        for pos in db.query(CodexEdge).filter(
            CodexEdge.source_id == subj.id, CodexEdge.target_id == obj.id,
            CodexEdge.relation == relation, CodexEdge.negated == False,
            CodexEdge.valid_until == None,
        ).all():
            pos.valid_until = datetime.now(timezone.utc)
            db.add(CodexEvent(entity_id=subj.id, event_type="edge_expired",
                              payload={"edge_id": str(pos.id), "relation": relation,
                                       "reason": "negated"},
                              timestamp=datetime.now(timezone.utc), batch_source=batch_id))
        existing_neg = db.query(CodexEdge).filter(
            CodexEdge.source_id == subj.id, CodexEdge.target_id == obj.id,
            CodexEdge.relation == relation, CodexEdge.negated == True,
            CodexEdge.valid_until == None,
        ).first()
        if existing_neg:
            existing_neg.strength += 1.0
            existing_neg.extraction_confidence = max(
                existing_neg.extraction_confidence or 1.0, extraction_confidence)
        else:
            neg_id = uuid.uuid4()
            db.add(CodexEdge(id=neg_id, source_id=subj.id, target_id=obj.id, relation=relation,
                             strength=1.0, source_batch=batch_id, confidence="active",
                             extraction_confidence=extraction_confidence, negated=True,
                             valid_from=datetime.now(timezone.utc)))
            db.add(CodexEvent(entity_id=subj.id, event_type="edge_added",
                              payload={"edge_id": str(neg_id), "relation": relation,
                                       "target_id": str(obj.id), "negated": True},
                              timestamp=datetime.now(timezone.utc), batch_source=batch_id))
        _regenerate_context_payload(subj, db)
        return

    # ── 1. Property relations: update entity properties, expire previous edges ──
    if relation in PROPERTY_RELATIONS:
        # Expire any existing active edge of the same relation for this source
        for old_edge in db.query(CodexEdge).filter(
            CodexEdge.source_id == subj.id,
            CodexEdge.relation == relation,
            CodexEdge.valid_until == None
        ).all():
            old_edge.valid_until = datetime.now(timezone.utc)
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_expired",
                payload={"edge_id": str(old_edge.id), "relation": relation},
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))

        # Create a new active edge with strength 3.0
        new_edge_id = uuid.uuid4()
        db.add(CodexEdge(
            id=new_edge_id,
            source_id=subj.id,
            target_id=obj.id,
            relation=relation,
            strength=3.0,
            source_batch=batch_id,
            confidence="active",
            extraction_confidence=extraction_confidence,
            valid_from=datetime.now(timezone.utc)
        ))
        db.add(CodexEvent(
            entity_id=subj.id,
            event_type="edge_added",
            payload={"edge_id": str(new_edge_id), "relation": relation, "target_id": str(obj.id)},
            timestamp=datetime.now(timezone.utc),
            batch_source=batch_id
        ))

        # Update entity properties
        # Update entity properties (JSONB requires explicit flagging for in‑place changes)
        if subj.properties is None:
            subj.properties = {}
        subj.properties[relation] = object_name.strip()
        flag_modified(subj, "properties")          # tell SQLAlchemy the JSONB changed
        subj.last_updated = datetime.now(timezone.utc)

        # Regenerate context payload
        # --- Enhanced context_payload regeneration ---
        _regenerate_context_payload(subj, db)
        return        

    # ── 2. Non‑property relations ──
    # A6: reconcile cross-turn conflicts before the fixed rules apply. Cheap
    # unless a real conflict is detected; may expire a superseded edge, or
    # reject this assertion entirely (reject_new → don't write).
    conflict = check_conflict(db, subj.id, relation, obj.id, turn_text)
    if conflict and not reconcile_conflict(db, conflict, subj, relation, obj,
                                           batch_id, turn_text, reconciler):
        return

    existing_active = db.query(CodexEdge).filter(
        CodexEdge.source_id == subj.id,
        CodexEdge.target_id == obj.id,
        CodexEdge.valid_until == None
    ).first()

    if existing_active:
        # Same source‑target pair, same relation → reinforcement
        if existing_active.relation == relation:
            existing_active.strength += 1.0
            # A3: corroborating re-extraction raises trust to the best seen.
            existing_active.extraction_confidence = max(
                existing_active.extraction_confidence or 1.0, extraction_confidence)
            if existing_active.strength >= 2.0 and existing_active.confidence == "pending":
                existing_active.confidence = "active"
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_strengthened",
                payload={"edge_id": str(existing_active.id), "relation": relation, "target_id": str(obj.id)},
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
        else:
            # Same pair, different relation.
            # FIX: only expire the old edge if the OLD relation is single-valued.
            # Previously this branch expired the old edge unconditionally, which
            # silently broke multi-valued semantics — e.g. if "knows" (multi-valued)
            # was active between two entities and a later turn asserted "friend"
            # (also multi-valued) between the same pair, the old "knows" edge was
            # deleted even though both relations are supposed to coexist.
            # Multi-valued relations between the same pair should simply add a
            # second, independent edge instead of replacing the first.
            if existing_active.relation not in MULTI_VALUED_RELATIONS:
                existing_active.valid_until = datetime.now(timezone.utc)
                db.add(CodexEvent(
                    entity_id=subj.id,
                    event_type="edge_expired",
                    payload={"edge_id": str(existing_active.id), "relation": existing_active.relation},
                    timestamp=datetime.now(timezone.utc),
                    batch_source=batch_id
                ))

            new_edge_id = uuid.uuid4()
            db.add(CodexEdge(
                id=new_edge_id,
                source_id=subj.id,
                target_id=obj.id,
                relation=relation,
                strength=3.0,
                source_batch=batch_id,
                confidence="active",
                extraction_confidence=extraction_confidence,
                valid_from=datetime.now(timezone.utc)
            ))
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_added",
                payload={"edge_id": str(new_edge_id), "relation": relation, "target_id": str(obj.id)},
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
    else:
        # No existing edge between this source and target
        # If the relation is single‑valued, expire any other active edge with the same source and relation
        previous_expired = False
        if relation not in MULTI_VALUED_RELATIONS:
            previous = db.query(CodexEdge).filter(
                CodexEdge.source_id == subj.id,
                CodexEdge.relation == relation,
                CodexEdge.valid_until == None
            ).first()
            if previous:
                previous.valid_until = datetime.now(timezone.utc)
                previous_expired = True
                db.add(CodexEvent(
                    entity_id=subj.id,
                    event_type="edge_expired",
                    payload={"edge_id": str(previous.id), "relation": relation},
                    timestamp=datetime.now(timezone.utc),
                    batch_source=batch_id
                ))

        # Create a new edge – immediately active if a previous edge was expired
        new_edge_id = uuid.uuid4()
        new_strength = 3.0 if previous_expired else 1.0
        new_confidence = "active" if previous_expired else "pending"
        db.add(CodexEdge(
            id=new_edge_id,
            source_id=subj.id,
            target_id=obj.id,
            relation=relation,
            strength=new_strength,
            source_batch=batch_id,
            confidence=new_confidence,
            extraction_confidence=extraction_confidence,
            valid_from=datetime.now(timezone.utc)
        ))
        db.add(CodexEvent(
            entity_id=subj.id,
            event_type="edge_added",
            payload={"edge_id": str(new_edge_id), "relation": relation, "target_id": str(obj.id)},
            timestamp=datetime.now(timezone.utc),
            batch_source=batch_id
        ))


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def extract_codex(self, batch_id: str, model_used: str = "", priority: bool = False):
    """Executes background semantic link mutations across target graph states."""
    log = logger.bind(batch_id=batch_id)

    if not priority:
        if is_gpu_busy():
            raise self.retry(countdown=30)
        if settings.background_model_mode == "shared" and is_user_active():
            raise self.retry(countdown=30)
    idempotency_key = hashlib.sha256(f"codex:{batch_id}".encode()).hexdigest()
    db = SessionLocal()
    
    try:
        if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
            return

        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn or not turn.lossless_flag:
            return

        triplets = extract_triplets(turn.raw_text, model_used, topic_tags=turn.topic_tags)
        reconciler = make_llm_reconciler()   # A6: bounded LLM for ambiguous supersessions
        for triplet in triplets:
            if isinstance(triplet, dict):
                s_raw = triplet.get("subject")
                r_raw = triplet.get("relation")
                o_raw = triplet.get("object")
                if isinstance(s_raw, str) and isinstance(r_raw, str) and isinstance(o_raw, str):
                    s = s_raw.strip()
                    r = r_raw.strip()
                    o = o_raw.strip()
                    if s and r and o:
                        handle_triplet(db, s, r, o, batch_id,
                                       extraction_confidence=float(triplet.get("confidence", 1.0)),
                                       turn_text=turn.raw_text, reconciler=reconciler,
                                       negated=bool(triplet.get("negated", False)))

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("codex_graph_assertions_committed", extracted_count=len(triplets))

    except Exception as exc:
        db.rollback()
        log.error("codex_extraction_aborted", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()