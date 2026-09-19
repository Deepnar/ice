"""Codex Extractor Subsystem – Structural Ingestion Plane."""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import structlog
from sqlalchemy import and_, or_, text
from sqlalchemy.orm.attributes import flag_modified

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.claims import excerpt_is_current, source_for_claim, store_claims
from src.memory.embedder import get_embedder
from src.memory.models import (
    CodexClaim,
    CodexClaimLink,
    CodexEdge,
    CodexEntity,
    CodexEvent,
    CodexRelationGap,
    EpisodicMemory,
    IdempotencyKey,
    ReviewQueue,
)
from src.memory.source import source_units
from src.retrieval.ner_utils import extract_entities
from src.workers.extraction_result import ExtractionOutputError, parse_extraction_response
from src.workers.idempotency import job_key

# G50: shared identity key. Safe at module level — maintenance_agent's own
# codex_extractor imports are all lazy (inside functions), so there is no cycle,
# and it pulls in nothing heavier than settings at import time.
from src.workers.maintenance_agent import merge_key

# The process-shared native-width embedder (G13/G23) — document_chunker,
# decision_extractor and conversation_summary reach the SAME instance
# through this module's name.
embedder = get_embedder()

logger = structlog.get_logger("ice.workers.codex")

from src.workers.bg_client_factory import (
    bg_timeout,
    get_bg_client,
    get_bg_model_name,
    json_schema,
)

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
# C2: the chunker moved to src/memory/chunking.py (the shared primitive for
# extraction windows, document chunks, and future C3/C12). Old underscore
# names kept as aliases so this module's callers/tests are unchanged.
from src.memory.chunking import (
    chunk_text as _chunk_text,
)
from src.memory.chunking import (
    estimate_tokens as _estimate_tokens,
)

# G68/P3: the same probe the request path budgets against — runner allocation
# from /api/ps, clamped by the GGUF ceiling, cached. Reused rather than
# re-derived so there is one answer to "how big is this model's window".
from src.model_registry.runtime_probe import serving_window

# A3 — extraction-confidence seeding (stored on codex_edges.extraction_confidence).
# G9: settings.codex_conf_* (grounded / ungrounded / rejected)

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


def _ground_triplets(triplets: list, ner_entities: List[str], source_text: str = ""):
    """Split *triplets* into (grounded, rejected) against the NER-confirmed
    entity list. A term is grounded if its normalised form equals a confirmed
    entity or its token set is a subset (either direction) of one — so
    shortened ('citadel' ⊂ 'obsidian citadel') and qualified mentions still
    ground, while invented entities with no token overlap are rejected.

    G43: a PROPERTY relation's object is a value, not an entity, so NER has
    nothing to say about it — and until 2026-08-12 that meant it was not
    checked at all. `november --eye_color--> golden black` and
    `krishna --role--> god of love` both reached the live graph that way, with
    the invented half sitting in the position nobody looked at. Property
    objects are now required to OCCUR IN THE SOURCE TEXT. That is the one
    check available for a value: it cannot be confirmed as an entity, but a
    value the turn never contains was not read out of the turn.

    Deliberately verbatim (after normalisation) rather than fuzzy. Rejection is
    not deletion here — a rejected triplet still enters the graph at
    `codex_conf_rejected` — so the cost of being strict is a true fact landing
    at low confidence, while the cost of being loose is a fabricated one landing
    at high confidence. Measured on the 293-turn store: a strict rule of this
    shape marks 4.8% of existing edges (201 of 4,170), so it is not a purge.
    """
    norm_source = _normalize_term(source_text) if source_text else ""
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

    def _value_in_source(term: str) -> bool:
        """G43: a property VALUE has to appear in the turn it is attributed to.
        No source text (legacy callers, tests) means no opinion — unchanged
        behaviour rather than a silent mass-rejection."""
        if not norm_source:
            return True
        nt = _normalize_term(term)
        return bool(nt) and nt in norm_source

    keep, drop = [], []
    for t in triplets:
        subj_ok = _grounded(t.get("subject", ""))
        # Property relations carry a value object, not an entity: NER cannot
        # ground it, so it is checked against the source text instead.
        if t.get("relation") in PROPERTY_RELATIONS:
            obj_ok = _value_in_source(t.get("object", ""))
        else:
            obj_ok = _grounded(t.get("object", ""))
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


_TRIPLET_SHAPE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            # Deliberately NOT enum-constrained by default — see
            # settings.codex_constrain_relation_enum and the Z2 note there.
            "relation": {"type": "string"},
            "object": {"type": "string"},
            "negated": {"type": "boolean"},
        },
        "required": ["subject", "relation", "object"],
    },
}

# Helper verbs and articles a model puts in front of a relation it means
# correctly ("is located in" for `located_in`). Stripping them is safe here
# because no two vocabulary entries collide once separators are removed —
# verified over all 197 (2026-08-04), so a normalized form can never land on
# the wrong relation.
_LEAD_FILLER = {"is", "was", "are", "were", "be", "been", "being",
                "get", "gets", "got", "the", "a", "an", "have", "had"}
_PREP_SYNONYM = {"upon": "on", "onto": "on", "unto": "to", "into": "in"}


def _relation_forms(raw: str):
    """Every spelling of *raw* worth testing against the vocabulary, cheapest
    first. Deterministic and lossless — this RESOLVES a surface form, it does
    not INFER intent, so it is not the kind of rule CLAUDE.md's
    style-invariance rule forbids."""
    yield raw
    n = raw.lower().strip()
    n = re.sub(r"[^a-z0-9\s_-]", " ", n).replace("-", " ").replace("_", " ")
    n = re.sub(r"\s+", "_", n.strip())
    if not n:
        return
    yield n
    toks = [_PREP_SYNONYM.get(w, w) for w in n.split("_")]
    yield "_".join(toks)
    while toks and toks[0] in _LEAD_FILLER:
        toks = toks[1:]
        if toks:
            yield "_".join(toks)
            yield "is_" + "_".join(toks)


def _is_inverse_pair(a: str, b: str) -> bool:
    """True when *a* and *b* are the active/passive forms of one relation.

    G45: `built`/`built_by`, `creates`/`created_by` and `makes`/`made_by` all
    appear spontaneously in real extraction output, and every similarity measure
    scores them near-identical. Collapsing them **writes the fact backwards** —
    `A built_by B` becoming `A built B` swaps subject and object. So this is
    checked BEFORE any similarity test and is deterministic, not a threshold:
    a guard that a threshold can skip is not a guard.
    """
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b or a == b:
        return False
    # Known converses first — similarity CANNOT distinguish these (measured:
    # before/after 0.8791, above the merge threshold), so the only reliable
    # signal is the curated map. See _RELATION_SEPARATION_PAIRS for why this is not
    # optional.
    _ant = globals().get("RELATION_SEPARATION_OF") or {}
    if b in _ant.get(a, ()) or a in _ant.get(b, ()):
        return True
    # ⚑ POLARITY — settled here, deterministically, because similarity cannot
    # see it. `can`/`cannot` score 0.9134 on the live encoder and
    # `is_the_same_as`/`is_not_the_same_as` 0.9182 — both above the 0.90 merge
    # threshold, and neither is caught by RELATION_SEPARATION_OF (a hand list cannot
    # enumerate an open vocabulary) nor by the passive rule below, which
    # compares True against True for any `is_*`/`is_*` pair. Merging a relation
    # into its own negation asserts the opposite of what the turn said.
    if _is_negated(a) != _is_negated(b):
        return True

    # ⚑ ARGUMENT ROLE — same reasoning. A trailing preposition names WHICH
    # argument the object fills, so changing it changes the fact:
    # `is_used_by`/`is_used_for` 0.9327, `is_needed_by`/`is_needed_for` 0.9547,
    # `results_from`/`results_in` 0.9266. All three clear 0.90 and all three are
    # invisible to the passive rule. The worst single case is `is` -> `in`,
    # which turns a copula into a containment claim.
    #
    # Blocking is the safe direction, by the asymmetry the passive rule already
    # argues: a missed merge leaves two correct relations, a wrong merge writes
    # a fact backwards.
    tail_a, tail_b = _role_tail(a), _role_tail(b)
    if tail_a != tail_b and (tail_a or tail_b):
        return True

    # Blanket direction rule, and deliberately blunt: two relations may not be
    # merged when exactly ONE of them is marked passive. Matching stems was the
    # first attempt and it failed on irregulars — `makes`/`made_by` share no
    # stem, and that pair appears in the real harvested data. Being blunt costs
    # at most a missed merge between an active and a passive form, which leaves
    # two correct relations; being clever costs a fact written backwards.
    return _is_passive(a) != _is_passive(b)


# Negation carried INSIDE a relation word. G45 opened the vocabulary, so the
# model now says `is_not`, `cannot`, `does_not`, `lacks` as relation words
# rather than setting A8's `negated` flag — which is why these have to be
# recognised as strings here rather than trusted to the flag.
_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "cannot", "cant", "without", "lacks", "lack",
    "isnt", "wasnt", "arent", "werent", "doesnt", "dont", "didnt",
    "hasnt", "havent", "hadnt", "wont", "shouldnt", "couldnt", "wouldnt",
})

# Trailing prepositions that name which argument the object fills. A change of
# tail is a change of meaning, never a synonym.
_ROLE_TAILS = frozenset({
    "by", "for", "from", "to", "in", "on", "as", "with", "of", "at",
    "into", "onto", "about", "against", "over", "under", "through",
})

# `properties` keys that are ICE's own bookkeeping, never a claim about the
# entity, and so never rendered into a retrieval fragment.
_INTERNAL_PROPERTY_KEYS = frozenset({"merge_key", "merged_into"})


def _is_negated(rel: str) -> bool:
    """Whether a relation word carries its own negation.

    ⚑ WHOLE TOKENS ONLY, never a prefix. A `non`/`un`/`dis` prefix rule was
    written here first and was wrong within minutes of meeting real data: it
    read `discusses` (20 edges) and `understands` (9) as negations, because
    English prefixes are not negation markers — they are spelling. That is
    CLAUDE.md's own rule, that no decision may depend on HOW a thing is
    written, violated inside the guard added to enforce it.

    The cost of dropping it is that a genuine `unrelated_to` is not recognised.
    That is the safe direction: an unrecognised negation is stored as an
    oddly-named positive relation, whereas a false positive here changes what
    `_is_inverse_pair` blocks and what `_strip_negation` rewrites.
    """
    r = (rel or "").strip().lower()
    if not r:
        return False
    return bool(set(r.split("_")) & _NEGATION_TOKENS)


# Contracted / fused negatives and the positive they carry. Only forms where
# the positive is mechanical — never a lexical judgement.
_FUSED_NEGATIVES = {
    "cannot": "can", "cant": "can", "wont": "will", "isnt": "is",
    "wasnt": "was", "arent": "are", "werent": "were", "doesnt": "does",
    "dont": "do", "didnt": "did", "hasnt": "has", "havent": "have",
    "hadnt": "had", "shouldnt": "should", "couldnt": "could",
    "wouldnt": "would",
}

# Bare negation tokens that can simply be deleted from a relation word.
_STRIPPABLE_NEGATION_TOKENS = frozenset({"not", "never"})


def _strip_negation(rel: str) -> Optional[str]:
    """The positive form of a negation-shaped relation, or None.

    ⚑ MECHANICAL ONLY. Deleting a `not` token or expanding a contraction is a
    string operation and cannot invent a meaning. LEXICAL negatives are
    deliberately excluded — `lacks` is not turned into `has`, and `without` is
    not turned into `with`, because that is a semantic judgement and
    `handle_triplet`'s A8 branch **expires the matching positive edge**. A
    wrong positive form would therefore destroy a true fact, which is exactly
    the failure G51 was just fixed for. A missed normalisation leaves a
    correctly-stored, if oddly-named, relation; a wrong one deletes evidence.
    """
    r = (rel or "").strip().lower()
    if not r:
        return None
    toks = r.split("_")
    out = []
    changed = False
    for tok in toks:
        if tok in _STRIPPABLE_NEGATION_TOKENS:
            changed = True
            continue
        if tok in _FUSED_NEGATIVES:
            out.append(_FUSED_NEGATIVES[tok])
            changed = True
            continue
        out.append(tok)
    if not changed or not out:
        return None
    positive = "_".join(out)
    # `no_longer_uses` → `longer_uses` would be worse than leaving it alone, and
    # a result that is still negation-shaped means the strip did not finish.
    return None if (positive == r or _is_negated(positive)) else positive


def _role_tail(rel: str) -> Optional[str]:
    """The trailing preposition naming the object's argument role, or None."""
    r = (rel or "").strip().lower()
    if not r:
        return None
    tail = r.rsplit("_", 1)[-1]
    return tail if tail in _ROLE_TAILS else None


def _is_passive(rel: str) -> bool:
    """Whether a relation names the object→subject direction (`built_by`,
    `is_used`). Marker-based, so it holds for irregular verbs too."""
    r = (rel or "").strip().lower()
    return r.endswith("_by") or r.startswith(("is_", "was_", "been_"))


_RELATION_SEED: Optional[List[str]] = None


def _seed_relations() -> List[str]:
    """G45: the harvested starter vocabulary, loaded once per process.

    A fresh store holds no relations, so canonicalisation has nothing to match
    against and the first few hundred turns each invent their own synonym —
    exactly the spread the mechanism exists to prevent. These 111 relations are
    the ones several independent models produced for the same idea, so they are
    the safest possible starting set. **They are a SEED, not a gate:** nothing
    is rejected for being absent from them.
    """
    global _RELATION_SEED
    if _RELATION_SEED is None:
        _RELATION_SEED = []
        try:
            from src.paths import REPO_ROOT
            p = REPO_ROOT / "data" / "relation_seed.json"
            if p.exists():
                _RELATION_SEED = list(json.loads(p.read_text()).get("relations") or [])
        except Exception as err:                        # noqa: BLE001
            logger.warning("codex_relation_seed_unreadable", error=str(err)[:120])
    return _RELATION_SEED


def known_relations(db=None) -> list:
    """The open vocabulary: what the graph holds, plus the harvested seed.

    Fetched ONCE per extraction call and passed down, not queried per triplet:
    a chunk yields dozens of relations and this is on the background path, not
    the request path, but a query per relation is still gratuitous.
    """
    own = db is None
    if own:
        from src.api.db import SessionLocal
        db = SessionLocal()
    try:
        # ⚑ ORDER BY IS LOAD-BEARING, NOT TIDINESS (2026-08-23). Without it
        # Postgres returns rows in whatever order the plan happens to produce,
        # and that changes as the table grows and is vacuumed — so the relation
        # vocabulary arrived in a DIFFERENT ORDER on every run.
        #
        # `canonical_relation` scores this list by cosine and takes the argmax.
        # Real candidates sit within thousandths of each other (`has`/`have`
        # 0.9469, `fails`/`failed` 0.9431), so list order decides near-ties —
        # and every accepted relation is fed back into the vocabulary the rest
        # of the run canonicalises against. One flipped tie on turn 1 therefore
        # changes what turn 2 sees, and so on.
        #
        # Measured: two IDENTICAL seeds diverged on 35 of 48 turns starting at
        # turn 1, with real content differences (`attention --affects--> exams`
        # in one run, `9.65 cgpa --does_see--> father` in the other). A model
        # warm-up cut the edge gap from -17% to -2.9%; this is the remainder.
        # The model itself is deterministic at temperature 0 — verified 12
        # alternating warm calls, one distinct output each.
        live = [r[0] for r in db.execute(text(
            "SELECT DISTINCT relation FROM codex_edges "
            "WHERE relation IS NOT NULL ORDER BY relation")).all() if r[0]]
    except Exception as err:
        logger.warning("codex_known_relations_failed", error=str(err)[:120])
        live = []
    finally:
        if own:
            db.close()
    return sorted(set(live) | set(_seed_relations()))


def is_clausal_relation(rel: str) -> bool:
    """G49: True when *rel* is a sentence fragment rather than a predicate.

    The open vocabulary (G45) accepts any relation word, which is right — but
    nothing checked the SHAPE, so whole clauses became edges:
    `taking_what_youve_built_and_what_youve_understood`,
    `was_waiting_for_a_reason_to_come_back`. Those are unmatchable by
    construction: no later mention will ever produce the same string, so the
    fact is stored and can never be found again.

    Counted by words, not characters — a length-in-characters rule would punish
    long single words, which is the kind of writing-convention bet G28 exists to
    kill. Callers DEMOTE on a True; they must not drop (G45).
    """
    return len([w for w in str(rel or "").split("_") if w]) > settings.codex_relation_max_words


def canonical_relation(raw: str, known=None):
    """G45: resolve *raw* onto a relation the graph already uses, or accept it.

    The closed-vocabulary gate this replaces returned None for anything not on a
    197-word list, and the caller **discarded** those triplets — 1,259 of them in
    a single 293-turn seed, including true facts like `i --didnt_get--> csi`.
    Dropping kept 32% of what the model produced; forcing the enum kept 100% and
    got ~78% of them wrong. The list itself was the defect.

    Three tiers, cheapest first:
      1. an exact vocabulary form — unchanged behaviour, keeps existing
         relations stable;
      2. a relation ALREADY IN THE GRAPH within
         `codex_relation_canonical_threshold` — write-time drift control, so the
         graph converges instead of sprouting a synonym per turn;
      3. otherwise the relation itself — new is not the same as wrong.

    Inverse pairs are never collapsed (see `_is_inverse_pair`).
    """
    if not raw:
        return None
    cleaned = re.sub(r"\s+", "_", str(raw).strip().lower())
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned).strip("_")
    if not cleaned:
        return None

    for form in _relation_forms(raw):
        if form in ALLOWED_RELATIONS:
            return form

    if not settings.codex_relation_open_vocabulary:
        return None

    if known:
        try:
            candidates = [k for k in known
                          if k and k != cleaned and not _is_inverse_pair(cleaned, k)]
            if candidates:
                vecs = embedder.encode([cleaned] + candidates,
                                       convert_to_tensor=False,
                                       show_progress_bar=False)
                arr = np.asarray(vecs, dtype=np.float64)
                target, rest = arr[0], arr[1:]
                sims = rest @ target / (
                    np.linalg.norm(rest, axis=1) * np.linalg.norm(target) + 1e-12)
                best = int(np.argmax(sims))
                if float(sims[best]) >= settings.codex_relation_canonical_threshold:
                    if candidates[best] != cleaned:
                        logger.info("codex_relation_canonicalised",
                                    incoming=cleaned, reused=candidates[best],
                                    similarity=round(float(sims[best]), 4))
                    return candidates[best]
        except Exception as err:
            # Never lose the relation because canonicalisation failed — the
            # whole point of this item is that a relation survives.
            logger.warning("codex_relation_canonicalise_failed",
                           relation=cleaned, error=str(err)[:120])

    return cleaned


def normalize_relation(raw: str):
    """Map a model-emitted relation onto the vocabulary, or None.

    ⚠ **G45 superseded this as the write-path decision.** `canonical_relation`
    is what extraction calls now; this stays for callers that genuinely want
    "is this one of the 197 controlled words" (and for the closed-vocabulary
    kill-switch, `codex_relation_open_vocabulary=False`).

    ⚠ Scope is deliberately narrow. Measured over 1,152 real out-of-vocabulary
    relations (2026-08-04, PROVENANCE): this recovers ~1%, and the *full*
    ladder — lemmatisation, fuzzy matching, token containment, embeddings —
    only reaches 5.7% while introducing direction inversions (`is_used` →
    `uses`, `is created by` → `created`) that write a fact backwards. The
    remaining 94% are not near-misses at all: they are concepts the vocabulary
    lacks (`is`, `has`, `includes`), and no string method invents those. So
    the aggressive tiers are deliberately NOT here — they wait on the
    vocabulary repair (Z2), after which the residual failures genuinely will
    be near-misses and the extra tiers start paying.
    """
    if not raw:
        return None
    for form in _relation_forms(raw):
        if form in ALLOWED_RELATIONS:
            return form
    return None


def _grounding_ner_labels() -> Optional[List[str]]:
    """The label set the grounding whitelist asks the background NER for.

    None means "whatever that tier defaults to", which keeps the micro-NER path
    and the unconfigured background path byte-identical. A non-empty
    `codex_grounding_ner_extra_types` puts those types back for THIS consumer
    only — see the setting's comment for why widening the shared default would
    regress clustering and key-term extraction.
    """
    raw = (settings.codex_grounding_ner_extra_types or "").strip()
    if not raw:
        return None
    from src.retrieval.ner_utils import _background_labels
    extra = [t.strip() for t in raw.split(",") if t.strip()]
    return _background_labels(extra_types=extra)


def extract_triplets(text: str, model_override: str = "",
                     topic_tags: Optional[List[str]] = None,
                     gaps: Optional[list] = None, source_sentences: Optional[list] = None) -> list:
    """Extract complete facts; failures propagate for runtime retry.

    `gaps` is an optional sink: when supplied, every triplet whose relation
    cannot be mapped onto the vocabulary is appended to it instead of simply
    vanishing. Appended rather than returned so the existing callers and stubs
    keep working unchanged (TRAPS #8).
    """
    grouped_relations = _build_grouped_relation_block()
    # G45: the open vocabulary, read once per call rather than per triplet.
    _known_rels = (known_relations()
                   if settings.codex_relation_open_vocabulary else [])

    prompt = (
        "You are a precise fact extractor. Convert the given text into a JSON array of "
        "subject‑relation‑object triplets.\n\n"
        "STRICT RULES:\n"
        "1. Prefer the individual relation words listed below (e.g. uses, friend, lives_in). "
        "Relations are grouped under '# category: ...' comment headers purely to help you pick "
        "the most precise word when several similar relations exist — the category header itself "
        "is NEVER a valid relation value. Pick one specific word from inside a group, never the "
        "header text above it.\n"
        f"{grouped_relations}\n"
        "   If a fact does not naturally fit any of these words, use the clearest short verb "
        "phrase of your own instead (lowercase, underscores, e.g. didnt_get, applies_to). "
        "NEVER output a category name, and never force a fact into a relation that doesn't "
        "truly describe it — a precise new relation is better than a wrong listed one, and "
        "far better than dropping the fact.\n"
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
        + (
            # 2026-08-17: the whitelist constrains WHICH entities are legal but
            # never what SHAPE a subject may take, so the model returns spans it
            # copied out of the turn — `what the flaw`, `to krishna`, `when
            # orien`. `_ground_triplets` then admits every one of them, because
            # a fragment containing a confirmed entity is structurally identical
            # to a legitimate qualified mention (`emotional validation`). No
            # rule over set-membership or source-presence separates the two, so
            # the only place left to act is where the string is produced.
            # Default OFF until measured; flip after the paired run.
            "7. A subject or object must be a NOUN PHRASE NAMING A THING — never a "
            "clause, a question fragment, or a verb phrase. If the text does not give "
            "you a nameable thing, skip the fact rather than inventing a span.\n"
            "   BAD:  {\"subject\":\"what the flaw\",...}  (question fragment)\n"
            "   BAD:  {\"subject\":\"began flaw\",...}     (verb phrase)\n"
            "   BAD:  {\"subject\":\"to krishna\",...}     (preposition + name)\n"
            "   GOOD: {\"subject\":\"flaw\",...}  {\"subject\":\"emotional validation\",...}\n"
            if settings.codex_extraction_entity_shape_rule else ""
        )
        + (
            # ⚑ DIRECTION. Measured 2026-08-22: ~25-28% of stored triplets are
            # REVERSED — right entities, right relation, backwards — and that
            # rate did not move when 1,611 direction-changing canonicalisation
            # merges were blocked, so the reversals are produced HERE, at
            # generation, not downstream. Until now this prompt said nothing
            # about which argument goes in `subject`, and all three examples
            # below were active SVO, while ALLOWED_RELATIONS mixes voices inside
            # a single category (`created` active, `founded_by` passive;
            # `manufactured_by`/`published_by` passive-only). A model asked to
            # render "Ford manufactured the car" therefore had a passive-only
            # relation available and no template for using it.
            "8. DIRECTION — which argument is the subject:\n"
            "   · With an ACTIVE relation, the subject DOES the action.\n"
            "     \"Ford manufactured the car\" -> "
            "{\"subject\":\"ford\",\"relation\":\"created\",\"object\":\"car\"}\n"
            "   · With a relation ending in _by, the subject RECEIVES the action "
            "and the object is the doer. These are mirror images — never both.\n"
            "     \"The car was manufactured by Ford\" -> "
            "{\"subject\":\"car\",\"relation\":\"manufactured_by\",\"object\":\"ford\"}\n"
            "   · A passive sentence does NOT mean you must use a passive "
            "relation. Rewrite it actively if the active word exists.\n"
            "     \"ICE was written by Deepesh\" -> "
            "{\"subject\":\"deepesh\",\"relation\":\"created\",\"object\":\"ice\"}\n"
            "   · Read the fact back before emitting it: subject-relation-object "
            "must be true IN THAT ORDER. \"car created ford\" is false; "
            "\"ford created car\" is true.\n"
            if settings.codex_extraction_direction_rule else ""
        )
        + "9. Output ONLY a JSON array. No markdown, no explanation.\n\n"
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
        + (
            # A PASSIVE worked example, because every example above is active
            # and examples carry more weight than rules.
            "\nText: \"The paper was reviewed by Dr. Rao, and the grant was "
            "awarded to our lab.\"\n"
            "Output: [{\"subject\":\"paper\",\"relation\":\"reviewed_by\","
            "\"object\":\"dr. rao\"},"
            " {\"subject\":\"our lab\",\"relation\":\"received\",\"object\":\"grant\"}]\n"
            if settings.codex_extraction_direction_rule else ""
        )
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
        # ⚑ G63: extraction has its OWN model, and this is the only background
        # job that does. `settings.codex_extraction_model` sits between the
        # per-call override and the shared background model so a run can pin an
        # arm from the command line, a deployment can pin the specialist in
        # .env, and everything else — summaries, cluster names, reflection —
        # keeps using the general model. Empty ⇒ the old behaviour exactly.
        model_name = (model_override or settings.codex_extraction_model
                      or get_bg_model_name())

        # --- Chunking: sentence/code-aware windows (roadmap A1).
        # ⚑ G68/P3: sized to the MODEL's window, not a fixed 550. A turn split
        # into three pieces is a turn whose entities are introduced in one call
        # and described in another, and nothing can link them across that
        # boundary. The ceiling exists because large chunks are UNMEASURED —
        # see `codex_extraction_chunk_max` in config.py.
        chunk_budget = settings.chunk_tokens
        window_source = "fixed"
        if settings.codex_extraction_chunk_adaptive:
            window = serving_window(model_name, None)
            if window:
                reserve = (_estimate_tokens(prompt + code_prompt)
                           + settings.codex_extraction_max_tokens)
                usable = window - reserve
                chunk_budget = max(settings.chunk_tokens,
                                   min(usable, settings.codex_extraction_chunk_max))
                window_source = "probe"
            else:
                # A silent fallback hides an outage (CLAUDE.md): say so, every
                # time, because the symptom is "extraction got quietly worse".
                logger.warning("extraction_chunk_window_unavailable",
                               model=model_name, falling_back_to=chunk_budget)
                window_source = "probe_failed"

        chunks = _chunk_text(text, max_tokens=chunk_budget)
        # ⚑ Log the REASONING, not just the result: n_chunks alone cannot tell
        # you whether a split was necessary or an artifact of the budget.
        logger.info("extraction_chunking", n_chunks=len(chunks),
                    estimated_tokens=_estimate_tokens(text),
                    chunk_budget=chunk_budget, window_source=window_source,
                    model=model_name)

        all_triplets = []
        log_relation_repairs = []
        for chunk in chunks:
            # G4(a): the chunk boundary is the cheap place to stand down — one
            # LLM call per chunk, and abandoning between them leaves nothing
            # half-written (the graph write happens later, in extract_codex).
            from src.workers.runtime import yield_if_user_active
            yield_if_user_active("codex_extract.chunk")
            # NER grounding (roadmap A2): confirm entities with a tagger first,
            # then constrain the LLM to relate only those. Reuses A1's chunk.
            # The tier picks the tagger — see `codex_extraction_ner_tier`; both
            # honour their own device setting, so neither is CPU-only.
            ner_entities = extract_entities(
                chunk, embedder, tier=settings.codex_extraction_ner_tier,
                labels=_grounding_ner_labels())
            entity_block = ""
            if ner_entities:
                confirmed = ", ".join(dict.fromkeys(ner_entities))  # dedup, keep order
                entity_block = (
                    "\n\nCONFIRMED ENTITIES (use ONLY these as subjects, and as objects "
                    "for relations between two entities; do NOT introduce named entities "
                    f"not in this list):\n{confirmed}"
                )

            # ⚑ G63: TWO PROMPT SHAPES, and a model is only usable in its own.
            # `template` sends a JSON schema to FILL and nothing else — no
            # rules, no examples, no code block. Sending the nine-rule prompt to
            # a specialist measurably produces garbage, and sending the bare
            # template to a generalist produces clause-triplets it cannot parse.
            # See `codex_extraction_mode` in config.py for the measurements.
            template_mode = settings.codex_extraction_mode == "template"
            if template_mode:
                user_content = (
                    "<|input|>\n### Template:\n"
                    + (json.dumps({'facts': [dict(subject='', relation='', object='',
                        **({'source_sentence': ''} if settings.codex_sentence_claims else {}))]}) + '\n')
                    + f"### Text:\n{chunk}{entity_block}\n<|output|>\n"
                )
                system_content = None
            else:
                claim_prompt = ("\nInclude source_sentence: an exact quotation of the complete source sentence for each fact."
                                if settings.codex_sentence_claims else "")
                chunk_prompt = prompt + code_prompt + claim_prompt + "\nNow process this text:"
                user_content = f"Text:\n{chunk}{entity_block}\n\n{chunk_prompt}"
                system_content = ("You are a JSON-only fact extraction tool. "
                                  "Never output anything but JSON.")

            messages = ([] if system_content is None
                        else [{"role": "system", "content": system_content}])
            messages.append({"role": "user", "content": user_content})
            call_kwargs = dict(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=settings.codex_extraction_max_tokens,
                timeout=bg_timeout(settings.codex_extraction_max_tokens),
            )
            # ⚑ The schema constraint FIGHTS a template model — it is trained to
            # emit its own envelope, and forcing a different one is a second
            # format instruction. Skipped in template mode, unchanged otherwise.
            if settings.codex_constrain_shape and not template_mode:
                shape = json.loads(json.dumps(_TRIPLET_SHAPE))
                if settings.codex_sentence_claims:
                    shape["items"]["properties"]["source_sentence"] = {"type": "string"}
                    shape["items"]["required"].append("source_sentence")
                if settings.codex_constrain_relation_enum:
                    shape["items"]["properties"]["relation"]["enum"] = sorted(ALLOWED_RELATIONS)
                call_kwargs["response_format"] = json_schema("codex_triplets", shape)
            completion = bg_client.chat.completions.create(**call_kwargs)
            choice = completion.choices[0]
            chunk_triplets = parse_extraction_response(
                choice.message.content,
                getattr(choice, "finish_reason", None),
                template_mode=template_mode,
            )

            if source_sentences is not None:
                for fact in chunk_triplets:
                    sentence = fact.get("source_sentence")
                    if isinstance(sentence, str) and sentence.strip():
                        source_sentences.append(sentence.strip())
                    else:
                        raise ExtractionOutputError("missing source sentence")

            # Map each relation onto the vocabulary; keep what maps, RECORD what
            # does not. This line used to be a bare filter with no log, and it
            # was discarding 67.8% of everything the model produced (measured
            # over 300 real turns, 2026-08-04 — see PROVENANCE). A drop that
            # large, invisible, is the silent-fallback failure CLAUDE.md names:
            # extraction looked like it ran, and mostly it deleted its own work.
            kept, dropped = [], []
            for t in chunk_triplets:
                # ⚑ A8/G45: the prompt already asks for a POSITIVE relation
                # plus `"negated": true`, and the model ignores it — it coins
                # `is_not`, `cannot`, `does_not` as relation words instead.
                # Measured on the arm-B store: 686 such edges, **0** of them
                # flagged, against 11 `negated=True` edges in the whole graph.
                # They were therefore stored as ordinary positive facts, gained
                # strength through reinforcement like any other, and rendered
                # under `Links:` — telling the answering model that the exact
                # opposite of the turn was true. Normalise here, before
                # canonicalisation, so the positive form is what gets
                # canonicalised and `handle_triplet`'s A8 branch does the rest.
                positive = _strip_negation(t.get("relation", ""))
                if positive:
                    t["relation"] = positive
                    t["negated"] = True
                # G45: canonicalise, do not gate. This returned None for
                # anything off a 197-word list and the triplet was destroyed —
                # 1,259 in one seed, `i --didnt_get--> csi` among them.
                mapped = canonical_relation(t.get("relation", ""), known=_known_rels)
                if mapped:
                    if mapped != t.get("relation"):
                        log_relation_repairs.append((t.get("relation"), mapped))
                    t["relation"] = mapped
                    # ⚑ G45/G50: feed the accepted relation back into the set
                    # THIS turn is canonicalising against. `known_relations()`
                    # is read once per call, and a turn yields 24.5 distinct
                    # relations on average (164 at the worst), so without this
                    # every variant in the same turn is blind to the others and
                    # they diverge permanently. That is why `have` merges into
                    # `has` on demand at 0.9469 and yet the store still holds
                    # has 491 / have 78 / had 60 — they were all first written
                    # in the same batch window, before either could see the
                    # other. Canonicalisation converges to one attractor only
                    # if the set grows as the turn proceeds.
                    if mapped not in _known_rels:
                        _known_rels.append(mapped)
                    kept.append(t)
                else:
                    dropped.append(t)
            if dropped:
                # WARNING, not debug, and every time — the rate IS the finding.
                logger.warning(
                    "codex_relation_out_of_vocabulary",
                    dropped=len(dropped), kept=len(kept),
                    relations=sorted({str(t.get("relation"))[:40] for t in dropped})[:10],
                )
                if gaps is not None:
                    gaps.extend(dropped)
            chunk_triplets = kept

            # Source support, not an object-word blacklist or equality of
            # endpoints, determines whether a claim is meaningful. Reflexive
            # relations and emotion values can both be valid facts.

            # NER grounding → extraction confidence (A3, completing the A2 seam):
            # grounded triplets are trusted high; grounding-REJECTED triplets are
            # no longer dropped — they enter the graph at low confidence, where
            # retrieval's dynamic thresholds keep them out of context until
            # corroborated (or they decay out). No-NER chunks get mid confidence
            # (nothing to ground against).
            if ner_entities:
                grounded, rejected = _ground_triplets(chunk_triplets, ner_entities,
                                                      source_text=chunk)
                for t in grounded:
                    t["confidence"] = settings.codex_conf_grounded
                for t in rejected:
                    t["confidence"] = settings.codex_conf_rejected
                if rejected:
                    logger.info("codex_grounding",
                                kept=len(grounded), rejected=len(rejected),
                                samples=[f'{t.get("subject")}|{t.get("relation")}|{t.get("object")}'
                                         for t in rejected[:8]])
                chunk_triplets = grounded + rejected
            else:
                for t in chunk_triplets:
                    t["confidence"] = settings.codex_conf_ungrounded

            # G49: a clause is not a predicate. Demote AFTER grounding so the
            # two judgements compose — a triplet can be well-grounded and still
            # carry an unusable relation, and it keeps the lowest verdict.
            # Demoted, never dropped: G45 measured that dropping destroys true
            # facts, and a low-confidence edge stays out of context until it is
            # corroborated rather than being deleted.
            clausal = [t for t in chunk_triplets
                       if is_clausal_relation(t.get("relation", ""))]
            for t in clausal:
                t["confidence"] = min(float(t.get("confidence", 1.0)),
                                      settings.codex_conf_rejected)
            if clausal:
                logger.info(
                    "codex_relation_clausal",
                    demoted=len(clausal), of=len(chunk_triplets),
                    max_words=settings.codex_relation_max_words,
                    samples=sorted({str(t.get("relation"))[:60] for t in clausal})[:6],
                )

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
        # Cooperative yields must reach the runtime without consuming a retry.
        from src.workers.runtime import JobYielded

        if not isinstance(err, JobYielded):
            logger.warning("codex_extraction_incomplete", error=str(err))
        raise



# G44: tokens that cannot name a node, whatever the writing style.
#
# ⚠ NOT A LENGTH RULE, and deliberately so. "short name = junk" is a bet on how
# people write, which is the class of rule CLAUDE.md's invariance rule exists to
# kill and which G45 calls out by name in `_stem`. `ai`, `ml`, `q4` and `eq` are
# four characters or fewer and are perfectly good nodes.
#
# What these have in common is FUNCTIONAL, not stylistic: none of them refers to
# anything on its own. A node called `i` collects every speaker who ever said
# "I"; a node called `8` collects every unrelated eight. They can never be
# looked up and they merge things that are not the same, which is the defect —
# measured 2026-08-12, the store held `8`, `3`, `6`, `2` and `d` typed as
# **person**. The set is closed and universal (pronouns, articles, bare
# numerals, punctuation), so it infers no intent and carries no convention.
_NON_REFERRING = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us", "our",
    "ours", "they", "them", "their", "theirs", "this", "that", "these",
    "those", "the", "a", "an", "there", "here", "who", "what", "which",
    "someone", "something", "anyone", "anything", "everyone", "everything",
}


def is_unusable_entity_name(name: str) -> bool:
    """True when *name* cannot serve as a graph node. See `_NON_REFERRING`."""
    n = (name or "").strip().lower()
    if not n:
        return True
    if n in _NON_REFERRING:
        return True
    # Pure punctuation or whitespace — `~`, `- -`, `...`. A name needs at least
    # one letter OR DIGIT to refer to anything.
    #
    # ⚑ THIS USED TO REQUIRE A LETTER, AND IT DESTROYED TRUE FACTS. Measured
    # 2026-08-15 over 586 seeded turns: 2,273 refusals, of which **94 were
    # numbers** — `2023`, `9.65`, `3.80`, `3/80`, `19`, `8`. Years, exam scores
    # and GPAs are referable entities; `3.80 --score--> maths` is a fact, and
    # the rule deleted its subject. The original reasoning ("a node named 8 can
    # never be looked up") is true of a bare pronoun and false of a number,
    # which is exactly as lookup-able as any other short name.
    #
    # This is a PRODUCT defect, not a corpus quirk: versions (`v2.1`), model
    # sizes (`128k`), hardware (`4090`), prices, years and scores are numeric
    # for every user. What still goes is punctuation, which refers to nothing.
    if not any(ch.isalnum() for ch in n):
        return True
    return False


def get_or_create_entity(db, name: str, protect_ids=None) -> CodexEntity:
    """Resolves structural identity records across global name and alias spaces.

    *protect_ids* names entities the caller is still holding — they are exempt
    from promotion below. See the promotion block for why that is not optional.
    """
    canonical = name.strip().lower()
    entity = db.query(CodexEntity).filter_by(canonical_name=canonical).first()
    if entity:
        return entity

    # ⚑ ORDERED (2026-08-23). `aliases` is an ARRAY and nothing makes its
    # contents unique across entities, so two entities can legitimately carry
    # the same alias — and `.first()` with no ORDER BY then returns whichever
    # row Postgres' plan happens to yield, which changes as the table grows.
    # Whichever one wins absorbs the fact, so the graph differs between runs.
    # Ordered by `canonical_name`, not by time or id: CodexEntity has no
    # `created_at`, `last_updated` MUTATES on every touch (so it cannot express
    # "which came first"), and `id` is a random uuid that differs per run —
    # which would defeat the entire point. `canonical_name` is UNIQUE, so it is
    # a total order, and it is derived from content rather than from history.
    entity = (db.query(CodexEntity)
              .filter(CodexEntity.aliases.any(canonical))
              .order_by(CodexEntity.canonical_name.asc())
              .first())
    if entity:
        return entity

    # ── G50: normalisation-equal match, the deterministic tier ───────────────
    # Both lookups above are EXACT-STRING, and `canonical_name` is UNIQUE with
    # aliases mirroring it — so `gemma-4-e4b q4` and `gemma-4-e4b-q4` both miss
    # and both get minted as separate entities. Relations have had a three-tier
    # ladder (`canonical_relation`) since G45; entities had none at all, and the
    # asymmetry was never a decision.
    #
    # This is where the fix belongs rather than in the background sweep: the
    # sweep repairs duplicates AFTER they exist, while this stops them being
    # minted. Measured 2026-08-17 on 8,280 entities — 51 merge_key groups store
    # wide, and every safety probe holds (`~15 gb`≠`~17 gb`, `8 gb`≠`4 gb`,
    # `12th boards`≠`10th boards`), because merge_key preserves token ORDER and
    # splits digit/letter runs rather than sorting anything.
    #
    # ⚠ NOT gated on `codex_node_promotion`. That flag guards PROMOTION, which
    # merges two *different* names and can re-attribute facts. This tier merges
    # names that normalise identically, which is an equality test, not a
    # judgement — and it never deletes a row.
    key = merge_key(canonical)
    if key:
        # ⚑ ORDERED (2026-08-23), and this tier needs it MORE than the alias
        # lookup above: a merge_key is deliberately many-to-one — that is its
        # entire purpose, collapsing `gemma-4-e4b q4` and `gemma-4-e4b-q4` onto
        # one key — so a populated store routinely has SEVERAL rows matching.
        # `.first()` picked whichever the plan returned, so the surviving name
        # for a merged group changed between runs, and every later exact-string
        # lookup then resolved differently.
        #
        # This is the residue the ORDER BY on `known_relations` left behind:
        # after that fix the first 19 turns reproduced exactly and divergence
        # began at turn 20, once enough entities existed for near-misses to
        # appear — `a roh --is_pulled_back_in--> aroh` in one run and not the
        # other. Early turns have too few entities to collide.
        hit = (db.query(CodexEntity).filter(
            CodexEntity.properties["merge_key"].astext == key,
            CodexEntity.properties["merged_into"].astext.is_(None),
        ).order_by(CodexEntity.canonical_name.asc()).first())
        if hit is not None:
            # Record the surface form so the exact-string tiers catch it next
            # time without reaching this query at all.
            if canonical not in (hit.aliases or []):
                hit.aliases = [*(hit.aliases or []), canonical]
                flag_modified(hit, "aliases")
                db.flush()
            logger.info("codex_entity_merge_key_hit",
                        name=canonical, resolved_to=hit.canonical_name, key=key)
            return hit

    # ── G44 second half: node promotion ──────────────────────────────────────
    # A generic node is one nothing can usefully be walked to. When a strictly
    # MORE SPECIFIC name arrives ("master plan" for a stored "plan"), Graphiti's
    # move is to promote: the specific node becomes canonical and the generic
    # name resolves to it, so later mentions land on the node worth traversing
    # rather than sprouting a second one.
    #
    # ⚠ Only STUB nodes are eligible — zero edges by default. Promotion merges
    # two identities, and "plan" is not always "master plan"; doing it to a node
    # that already carries facts would silently re-attribute them. A stub has no
    # facts to re-attribute, so the merge cannot destroy anything. Off by
    # default regardless (`codex_node_promotion`).
    #
    # ⚑ "ZERO EDGES" MEANS ZERO *COMMITTED* EDGES, AND THAT IS THE WHOLE TRAP.
    # `record_triplet` resolves its subject, then its object, then writes the
    # edge between them. Resolving the object can promote the SUBJECT away —
    # `plan` folded into `the big plan` — because at that instant the edge the
    # triplet is about to write does not exist, so the subject counts as a
    # disposable stub. The caller then writes an edge pointing at a deleted row
    # and Postgres rejects the whole batch on the foreign key, costing that
    # turn its codex AND procedural extraction (post_flight re-raises).
    # Reproduced deterministically 2026-08-15; 1 turn in 6 on a smoke seed.
    # ⇒ an entity the caller is still holding is never a disposable stub.
    if settings.codex_node_promotion:
        try:
            tokens = set(canonical.split())
            if len(tokens) > 1:
                protected = set(protect_ids or ())
                # ⚑ ORDERED (2026-08-23). `tokens` is a SET, so `list(tokens)`
                # is already arbitrary — and the loop below mutates the store
                # (it promotes a stub into this name and deletes the stub), so
                # which candidate is reached first decides the outcome. Sorting
                # the tokens and ordering the query makes that choice stable.
                # Currently latent: promotion is default-OFF and fired 0 times
                # in a 586-turn run. It is fixed anyway because a dormant
                # order-dependency is exactly what surfaced at turn 20 once the
                # earlier ones were removed.
                for generic in (db.query(CodexEntity)
                                .filter(CodexEntity.canonical_name.in_(
                                    sorted(tokens)))
                                .order_by(CodexEntity.canonical_name.asc())
                                .all()):
                    if generic.id in protected:
                        logger.info("codex_node_promotion_skipped_in_flight",
                                    generic=generic.canonical_name,
                                    candidate=canonical)
                        continue
                    degree = db.query(CodexEdge).filter(
                        (CodexEdge.source_id == generic.id)
                        | (CodexEdge.target_id == generic.id)).count()
                    if degree > settings.codex_node_promotion_max_degree:
                        continue
                    new_specific = CodexEntity(
                        id=generate_uuid5(canonical),
                        canonical_name=canonical,
                        aliases=list({*(generic.aliases or []), generic.canonical_name, name}),
                        tags=list(generic.tags or []),
                        # G50: the promoted node carries the GENERIC's properties,
                        # so its merge_key would be the generic's — stamp the new
                        # canonical name's key or promotion silently poisons the
                        # tier above with a key that no longer describes the row.
                        properties={**dict(generic.properties or {}),
                                    "merge_key": merge_key(canonical)},
                        context_payload=generic.context_payload or "",
                        entity_type=generic.entity_type,
                        description=generic.description,
                        embedding=embedder.encode(canonical,
                                                  convert_to_tensor=False).tolist(),
                        last_updated=datetime.now(timezone.utc),
                    )
                    # SAVEPOINT, so a failed promotion undoes ITSELF and not the
                    # caller's open transaction. This used to fall through to a
                    # bare `db.rollback()`, which discards every uncommitted
                    # write in the batch — the same class of defect as the
                    # in-flight deletion above, and silent where that one is loud.
                    with db.begin_nested():
                        db.add(new_specific)
                        db.delete(generic)
                        db.flush()
                    logger.info("codex_node_promoted",
                                generic=generic.canonical_name,
                                promoted_to=canonical, generic_degree=degree)
                    return new_specific
        except Exception as err:
            # Promotion is an optimisation; never lose the entity over it. The
            # savepoint has already undone the failed attempt, so the caller's
            # transaction is intact and resolution falls through to a new node.
            logger.warning("codex_node_promotion_failed",
                           name=canonical, error=str(err)[:120])

    new_entity = CodexEntity(
        id=generate_uuid5(canonical),
        canonical_name=canonical,
        aliases=[name],
        tags=[],
        # G50: stamped at birth, because the tier above matches on it. An
        # entity created without it is invisible to normalisation matching
        # forever — the migration backfills existing rows for the same reason.
        properties={"merge_key": merge_key(canonical)},
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


# G33: relations whose two ends are the SAME kind of thing, so an incoming edge
# says as much about the target's type as an outgoing one says about the source's.
# Everything else is asymmetric and may not vote from the far end — see
# _infer_entity_type.
_SYMMETRIC_RELATIONS = {
    "married_to", "is_dating", "is_divorced_from", "is_separated_from",
    "sibling_of", "friend", "enemy", "ally", "colleague", "knows",
    "partners_with", "complements", "opposes",
}


def _infer_entity_type(outgoing, incoming, tags, current: str) -> str:
    """Infer a structural type: an explicit known type-tag wins; else vote by the
    entity's relations; else keep the current value.

    ⚠ **Direction matters, and ignoring it produced confidently wrong types**
    (G33, measured 2026-08-08). `_TYPE_HINTS` maps a relation to the type of the
    entity that has it *outgoing*: `role` outgoing means "this thing HAS a role",
    i.e. a person. Read from the other end it means the opposite — `fire mage`
    with an incoming `role` edge **IS** a role, and is not a person. The old
    signature took one flat list of both directions, so `kael --role→ fire mage`
    typed `fire mage` as `person`.

    So incoming relations vote **only when the relation is symmetric**
    (`married_to`, `colleague`, …), where both ends genuinely are the same kind
    of thing. Asymmetric incoming edges are ignored rather than guessed at: the
    result is `entity` — unknown, which is honest — instead of a wrong type that
    downstream tag filters and `_codex_enumeration` would trust.

    (A full fix would carry a second hint table for the object side —
    `works_at` incoming ⇒ organization, `located_in` incoming ⇒ place. That is
    worth doing when something depends on it; under-claiming beats mis-claiming
    in the meantime.)
    """
    for t in (tags or []):
        tl = t.lower()
        tl = _TYPE_SYNONYMS.get(tl, tl)
        if tl in _KNOWN_TYPES:
            return tl
    voting = list(outgoing) + [r for r in incoming if r in _SYMMETRIC_RELATIONS]
    votes = {}
    for rel in voting:
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
    backlinks (Obsidian-style). Also infers entity_type from the relations.

    ⚠ The `flush` is load-bearing, not hygiene. `SessionLocal` is built with
    `autoflush=False` (api/db.py), so the two queries below see the graph **as
    of the last commit** — not including the edge the caller just added or the
    expiry it just set. Every one of this function's nine call sites writes
    edges and then calls it, so every payload in the store was ONE WRITE
    BEHIND.

    Measured before the fix: assert "proj uses postgres" twice and the payload
    is EMPTY both times (the insert had not flushed); then negate it, and the
    payload becomes `Links: uses → postgres` — the retracted fact, rendered as
    current, with no Negations section, because the expiry had not flushed
    either. A fact the user explicitly took back was being injected as true,
    and a freshly-extracted entity contributed nothing at all.
    """
    db.flush()
    out_edges = db.query(CodexEdge).filter(
        CodexEdge.source_id == entity.id,
        CodexEdge.valid_until == None
    ).order_by(CodexEdge.strength.desc()).limit(20).all()
    in_edges = db.query(CodexEdge).filter(
        CodexEdge.target_id == entity.id,
        CodexEdge.valid_until == None
    ).order_by(CodexEdge.strength.desc()).limit(20).all()

    entity.entity_type = _infer_entity_type(
        [e.relation for e in out_edges], [e.relation for e in in_edges],
        entity.tags, entity.entity_type)

    parts = []
    if entity.description:
        parts.append(entity.description.strip())
    if entity.properties:
        # ⚑ Internal bookkeeping is not a fact about the entity. `merge_key` is
        # G50's normalisation key, stamped on every entity at birth, and it was
        # rendering into 6,271 of 6,271 payloads — so for any entity with no
        # real properties it was the ONLY `Properties:` line, and the answering
        # model read a lookup key as a stated attribute:
        #     jee main | Properties: merge_key: jee main
        props = "; ".join(f"{k}: {v}" for k, v in entity.properties.items()
                          if k not in _INTERNAL_PROPERTY_KEYS)
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
# Vocabulary separation prevents wrong canonical merges; it never establishes
# contradiction. Conflict candidates need source-aware reconciliation.

_RELATION_SEPARATION_PAIRS = [
    ("friend", "enemy"), ("ally", "enemy"),
    ("married_to", "is_divorced_from"), ("is_dating", "is_separated_from"),
    ("endorses", "criticises"),
    # Converses remain separate even when their embeddings are similar.
    # The same subject can both buy from and sell to another entity.
    ("before", "after"), ("parent_of", "child_of"),
    ("teaches", "learns_from"), ("follows", "precedes"),
    ("supports", "opposes"), ("above", "below"),
    ("buys", "sells"), ("wins", "loses"),
    ("member_of", "contains"), ("part_of", "has_part"),
]
RELATION_SEPARATION_OF: dict = {}
for _a, _b in _RELATION_SEPARATION_PAIRS:
    RELATION_SEPARATION_OF.setdefault(_a, set()).add(_b)
    RELATION_SEPARATION_OF.setdefault(_b, set()).add(_a)


# These nominate a question for the reconciler, never an automatic expiry.
_OPPOSITION_PAIRS = [
    ("friend", "enemy"), ("ally", "enemy"),
    ("married_to", "is_divorced_from"), ("is_dating", "is_separated_from"),
    ("endorses", "criticises"),
]
OPPOSITION_OF: dict = {}
for _a, _b in _OPPOSITION_PAIRS:
    OPPOSITION_OF.setdefault(_a, set()).add(_b)
    OPPOSITION_OF.setdefault(_b, set()).add(_a)


def _entity_name(db, entity_id) -> str:
    if entity_id is None:
        return "?"
    e = db.query(CodexEntity).get(entity_id)
    return e.canonical_name if e else "?"


def conflict_candidates(db, subj_id, relation: str, obj_id, negated=False):
    """Candidate identity is structural; source evidence decides replacement."""
    alternatives = [and_(CodexEdge.relation == relation,
        or_(CodexEdge.target_id != obj_id, CodexEdge.negated != negated))]
    oppositions = OPPOSITION_OF.get(relation)
    if oppositions:
        alternatives.append(and_(CodexEdge.target_id == obj_id,
                                  CodexEdge.relation.in_(sorted(oppositions))))
    return [{"type": "source_comparison", "old_edge_id": e.id,
             "old_relation": e.relation, "old_target_id": e.target_id,
             "old_negated": e.negated}
            for e in db.query(CodexEdge).filter(CodexEdge.source_id == subj_id,
                CodexEdge.valid_until.is_(None), or_(*alternatives))
                .order_by(CodexEdge.id).all()]


def check_conflict(db, subj_id, relation: str, obj_id, turn_text: Optional[str]):
    """Compatibility lookup; the writer processes every candidate."""
    return next(iter(conflict_candidates(db, subj_id, relation, obj_id)), None)


def _reconciliation_evidence(db, conflict, new_claims):
    """Resolve complete authoritative units; unknown authority cannot expire."""
    old_claims = db.query(CodexClaim).join(CodexClaimLink).filter(
        CodexClaimLink.edge_id == conflict["old_edge_id"]).all()

    def units(claims):
        found = {}
        for claim in claims or []:
            row = source_for_claim(db, claim)
            if (row is None or not excerpt_is_current(row, claim)
                    or claim.role == "unknown" or row.ts_provenance != "original"
                    or row.timestamp is None):
                return None
            unit = next((u for u in source_units(row) if u.role == claim.role
                         and u.start <= claim.start and claim.end <= u.end), None)
            if unit is None:
                return None
            key = (str(row.conversation_id), claim.role, row.timestamp, unit.text)
            found[key] = dict(conversation_id=str(row.conversation_id), role=claim.role,
                              recorded_at=row.timestamp, text=unit.text)
        return next(iter(found.values())) if len(found) == 1 else None

    old, new = units(old_claims), units(new_claims)
    if (old is None or new is None or old["conversation_id"] != new["conversation_id"]
            or old["role"] != new["role"] or old["recorded_at"] >= new["recorded_at"]):
        return None
    return old, new


def _refresh_property_projection(db, subj, relation):
    """Derive displayed property values from every live positive edge."""
    if relation in PROPERTY_RELATIONS:
        # Preserve every live value. This JSON is a display projection, not
        # an independent last-write-wins assertion or an expiry authority.
        values = sorted({name for (name,) in db.query(CodexEntity.canonical_name)
            .join(CodexEdge, CodexEdge.target_id == CodexEntity.id)
            .filter(CodexEdge.source_id == subj.id, CodexEdge.relation == relation,
                    CodexEdge.negated.is_(False), CodexEdge.valid_until.is_(None)).all()})
        props = dict(subj.properties or {})
        if values:
            props[relation] = values[0] if len(values) == 1 else values
        else:
            props.pop(relation, None)
        subj.properties = props
        subj.last_updated = datetime.now(timezone.utc)


def _expire_edge(db, edge_id, batch_id, reason: str, source: Optional[str] = None):
    """*source* (G17/D4): who caused the expiry ("maintenance_agent", ...);
    omitted for the in-line extraction path, whose provenance is the batch."""
    edge = db.query(CodexEdge).get(edge_id)
    if edge and edge.valid_until is None:
        edge.valid_until = datetime.now(timezone.utc)
        payload = {"edge_id": str(edge_id), "reason": reason}
        if source:
            payload["source"] = source
        db.add(CodexEvent(entity_id=edge.source_id, event_type="edge_expired",
                          payload=payload,
                          timestamp=datetime.now(timezone.utc), batch_source=batch_id))
        db.flush()
        for entity_id in (edge.source_id, edge.target_id):
            entity = db.get(CodexEntity, entity_id)
            if entity is not None:
                if entity_id == edge.source_id:
                    _refresh_property_projection(db, entity, edge.relation)
                _regenerate_context_payload(entity, db)


def reconcile_conflict(db, conflict, subj, relation, obj, batch_id,
                       turn_text: Optional[str], reconciler,
                       source: Optional[str] = None, source_claims=None, negated=False) -> bool:
    """Reconcile candidates from source text; relation names alone never expire.

    Missing evidence/reconciler retains both claims and records a review item.
    Returns False only for an explicit, source-backed reject_new decision.
    """
    decision = "review"
    evidence = _reconciliation_evidence(db, conflict, source_claims)
    if reconciler is not None and evidence is not None:
        try:
            decision = reconciler({
                "subject": subj.canonical_name, "relation": relation,
                "object": obj.canonical_name, "old_relation": conflict["old_relation"],
                "old_object": _entity_name(db, conflict.get("old_target_id")),
                "turn": evidence[1]["text"], "old_source": evidence[0],
                "new_source": evidence[1], "negated": negated,
                "old_negated": conflict.get("old_negated", False),
            }) or "review"
        except Exception as err:
            from src.workers.runtime import JobYielded
            if isinstance(err, JobYielded):
                raise
            logger.warning("codex_reconcile_llm_failed", error_type=type(err).__name__)
            decision = "review"

    if decision == "expire_old":
        _expire_edge(db, conflict["old_edge_id"], batch_id, "supersession",
                     source=source)
    elif decision == "reject_new":
        logger.info("codex_reconcile", type="supersession", decision="reject_new")
        return False
    elif decision != "keep_both":  # review / unknown → keep both, flag human
        content = {
            "new": {"subject": subj.canonical_name, "relation": relation,
                    "object": obj.canonical_name, "negated": negated},
            "conflict_type": conflict["type"], "old_edge_id": str(conflict["old_edge_id"]),
            "old_relation": conflict["old_relation"],
            "old_object": _entity_name(db, conflict.get("old_target_id")),
            "new_batch_id": str(batch_id),
            "reason": "source_comparison_unresolved",
        }
        # Repeating an observed batch is not a new review question.
        identity = {k: content[k] for k in ("old_edge_id", "new_batch_id", "new")}
        if not db.query(ReviewQueue.id).filter(
                ReviewQueue.item_type == "codex_reconciliation",
                ReviewQueue.item_content.contains(identity)).first():
            db.add(ReviewQueue(item_type="codex_reconciliation", item_content=content))
        decision = "review"
    logger.info("codex_reconcile", type="supersession", decision=decision)
    return True


def reconciliation_prompt(ctx):
    """Complete source context shared by inline and maintenance consumers."""
    return (
        "Two facts about the same subject may conflict. Using ONLY the conversation "
        "text, decide how to reconcile them. Proposals, hypotheticals and quoted "
        "denials are not adopted facts. Different dates or contexts can coexist. "
        "Expire only if the same speaker clearly replaces the old assertion; "
        "otherwise keep both.\n"
        f"Existing candidate (negated={ctx.get('old_negated', False)}): "
        f"{ctx['subject']} {ctx['old_relation']} {ctx['old_object']}\n"
        f"New candidate (negated={ctx.get('negated', False)}): "
        f"{ctx['subject']} {ctx['relation']} {ctx['object']}\n"
        f"Original source with role and recorded time: {ctx.get('old_source')}\n"
        f"New source with role and recorded time: {ctx.get('new_source')}\n"
        f"Conversation text: {ctx['turn']}\n\n"
        "Reply with exactly ONE word:\n"
        "expire_old  — the new fact replaces/supersedes the old one\n"
        "keep_both   — both are true at the same time\n"
        "reject_new  — the new fact is wrong or not actually asserted"
    )


def make_llm_reconciler():
    """A bounded reconciler backed by the background model: one word out, ten
    tokens max. Returned as a callable so it can be swapped/stubbed.

    ⚑ DELIBERATELY NOT `settings.codex_extraction_model` (G63, 2026-08-26).
    This function lives in the extractor and is therefore the natural thing to
    sweep along when "the extractor" is repointed at NuExtract3 — do not. It
    does not extract anything: it READS two facts and REASONS about which
    survives, then answers in one word. An extraction specialist is trained to
    fill a JSON template from stated content and has no path to that judgement,
    so pointing this at one would turn a decision into a coin flip while every
    log line still looked healthy. It follows the general background model.
    """
    def _reconcile(ctx: dict) -> str:
        prompt = reconciliation_prompt(ctx)
        from src.memory.tokens import count
        if count(prompt) > settings.codex_reconcile_input_tokens:
            logger.warning("codex_reconcile_source_too_long")
            return "review"
        resp = bg_client.chat.completions.create(
            model=get_bg_model_name(),
            messages=[{"role": "system", "content": "You output exactly one word."},
                      {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10, timeout=bg_timeout(10))  # >5 so 'expire_old' can't truncate
        choice = resp.choices[0]
        out = (choice.message.content or "").strip().lower()
        if getattr(choice, "finish_reason", None) not in (None, "stop"):
            logger.warning("codex_reconcile_incomplete_response")
            return "review"
        if out not in {"expire_old", "keep_both", "reject_new"}:
            logger.warning("codex_reconcile_invalid_response")
            return "review"
        return out
    return _reconcile


def _observe_edge(edge, batch_id, extraction_confidence):
    """Count a source batch once; retrieval popularity cannot promote support."""
    batch = uuid.UUID(str(batch_id))
    seen = {uuid.UUID(str(b)) for b in (edge.observed_batches or [])}
    if edge.source_batch:
        seen.add(uuid.UUID(str(edge.source_batch)))
    if batch in seen:
        return False
    seen.add(batch)
    edge.observed_batches = sorted(seen, key=str)
    edge.strength = min(settings.codex_retention_cap, (edge.strength or 0.0) + 1.0)
    old_conf = edge.extraction_confidence if edge.extraction_confidence is not None else 1.0
    edge.extraction_confidence = max(old_conf, extraction_confidence)
    if len(seen) >= 2 and edge.confidence == "pending":
        edge.confidence = "active"
    return True


def handle_triplet(db, subject_name: str, relation: str, object_name: str, batch_id: str,
                   extraction_confidence: float = 1.0, turn_text: Optional[str] = None,
                   reconciler=None, negated: bool = False, source_claims=None):
    """Integrates extraction assertions into the transaction context,
    with source-qualified reconciliation and polarity-aware observation. *extraction_confidence* (A3)
    is the grounding-seeded trust stored on new edges; on reinforcement the
    edge keeps the highest confidence seen (corroboration raises trust).
    *turn_text* / *reconciler* drive the A6 reconciliation loop (below).
    *negated* (A8) stores the relation's negative polarity."""

    # G44: an edge is only as useful as its endpoints. A triplet hanging off a
    # node nobody can look up is not a weaker fact, it is an unreachable one —
    # so it is refused here, at the single write boundary, rather than filtered
    # downstream. Loud on purpose: the RATE is the finding, and a silent skip
    # would read as a clean graph (CLAUDE.md — a silent fallback hides an outage).
    for role, candidate in (("subject", subject_name), ("object", object_name)):
        if is_unusable_entity_name(candidate):
            logger.warning("codex_entity_name_unusable", role=role,
                           name=str(candidate)[:40], relation=relation,
                           subject=str(subject_name)[:40],
                           object=str(object_name)[:40],
                           reason="not a referring expression — cannot be a node")
            return

    subj = get_or_create_entity(db, subject_name)
    # ⚑ The subject is now IN FLIGHT: this triplet is about to write an edge
    # from it, but that edge does not exist yet, so promotion would read it as a
    # zero-edge stub and delete it while resolving the object. Passing the id
    # exempts it. See the promotion block in get_or_create_entity.
    obj  = get_or_create_entity(db, object_name, protect_ids={subj.id})

    # All assertion types use the same source boundary. Neither a property
    # category nor polarity establishes that a different assertion became false.
    with db.begin_nested() as changes:
        for conflict in conflict_candidates(db, subj.id, relation, obj.id, negated):
            if not reconcile_conflict(db, conflict, subj, relation, obj, batch_id,
                                      turn_text, reconciler, source_claims=source_claims,
                                      negated=negated):
                # A later comparison may reject the candidate. Do not leave
                # earlier expiries behind without the proposed replacement.
                changes.rollback()
                return

    existing_active = db.query(CodexEdge).filter(
        CodexEdge.source_id == subj.id, CodexEdge.target_id == obj.id,
        CodexEdge.relation == relation, CodexEdge.negated == negated,
        CodexEdge.valid_until.is_(None)).first()
    if existing_active:
        if _observe_edge(existing_active, batch_id, extraction_confidence):
            db.add(CodexEvent(entity_id=subj.id, event_type="edge_strengthened",
                payload={"edge_id": str(existing_active.id), "relation": relation,
                         "target_id": str(obj.id), "reason": "distinct_source_batch"},
                timestamp=datetime.now(timezone.utc), batch_source=batch_id))
        edge = existing_active
    else:
        edge = CodexEdge(id=uuid.uuid4(), source_id=subj.id, target_id=obj.id,
            relation=relation, negated=negated, strength=1.0, source_batch=batch_id,
            confidence="pending", extraction_confidence=extraction_confidence,
            valid_from=datetime.now(timezone.utc))
        db.add(edge)
        db.add(CodexEvent(entity_id=subj.id, event_type="edge_added",
            payload={"edge_id": str(edge.id), "relation": relation,
                     "target_id": str(obj.id), "negated": negated},
            timestamp=datetime.now(timezone.utc), batch_source=batch_id))
    db.flush()
    _refresh_property_projection(db, subj, relation)
    _regenerate_context_payload(subj, db)
    _regenerate_context_payload(obj, db)
    return edge


def _record_relation_gaps(db, dropped: list, turn, log) -> int:
    """Persist facts the vocabulary cannot express, instead of losing them.

    Deliberately best-effort and never fatal: this is a diagnostic ledger, and
    an extraction run must not fail because its diagnostics could not be
    written. It IS, however, loud on failure — a silent diagnostic is worth
    nothing, which is the whole reason this table exists.
    """
    written = 0
    try:
        for t in dropped:
            raw = str(t.get("relation") or "").strip()
            if not raw:
                continue
            subject = str(t.get("subject") or "").strip()
            obj = str(t.get("object") or "").strip()
            if not subject or not obj:
                continue
            forms = list(_relation_forms(raw))
            db.add(CodexRelationGap(
                proposed_relation=forms[-1] if forms else raw.lower(),
                raw_relation=raw[:200],
                subject=subject[:500],
                object=obj[:500],
                negated=bool(t.get("negated", False)),
                batch_id=turn.batch_id,
                conversation_id=getattr(turn, "conversation_id", None),
                status="pending",
            ))
            written += 1
        db.flush()
        log.info("codex_relation_gaps_recorded", count=written)
    except Exception as exc:
        db.rollback()
        log.warning("codex_relation_gap_write_failed", error=str(exc)[:200],
                    attempted=len(dropped))
        return 0
    return written


def extract_codex(batch_id: str, model_used: str = "", priority: bool = False):
    """Executes background semantic link mutations across target graph states.

    Plain callable since C7 — the maintenance runtime's gpu lane + idle gating
    replace the old per-task GPU/user-activity checks (``priority`` is kept
    for signature stability; gating now lives outside the callable).
    """
    log = logger.bind(batch_id=batch_id)

    # G29 consolidated every job onto job_key(); this one kept hashing by hand.
    # Byte-identical output (both are sha256("codex:<batch_id>")), verified — so
    # no existing marker is orphaned and no turn is re-processed. The point is
    # that the namespace is now expressed rather than spelled out, which is what
    # stops the un-namespaced-key bug G29 fixed from coming back here.
    idempotency_key = job_key("codex", batch_id)
    db = SessionLocal()
    
    try:
        if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
            log.info("codex_already_extracted")
            return

        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(str(batch_id))).first()
        if turn is None:
            raise RuntimeError(f"turn for batch {batch_id} not visible yet")
        if turn.is_private:
            log.info("codex_private_turn_skipped")
            return

        relation_gaps = []
        triplets, sentences = [], []
        for unit in source_units(turn):
            if not unit.text.strip():
                continue
            # Foreground answer model is provenance, not an extraction override.
            extracted = extract_triplets(unit.text,
                topic_tags=turn.topic_tags, gaps=relation_gaps,
                source_sentences=sentences if settings.codex_sentence_claims else None)
            for triplet in extracted:
                triplet["_source_role"] = unit.role
                triplet["_source_start"] = unit.start
                triplet["_source_end"] = unit.end
            triplets.extend(extracted)
        claims = (store_claims(db, turn, sentences, encoder=embedder)
                  if settings.codex_sentence_claims else [])
        if relation_gaps:
            _record_relation_gaps(db, relation_gaps, turn, log)
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
                        matched_claims = [c for c in claims if
                            c.sentence == triplet.get("source_sentence", "").strip()
                            and c.role == triplet.get("_source_role")
                            and triplet["_source_start"] <= c.start
                            and c.end <= triplet["_source_end"]]
                        edge = handle_triplet(db, s, r, o, batch_id,
                                       extraction_confidence=float(triplet.get("confidence", 1.0)),
                                       turn_text=turn.raw_text, reconciler=reconciler,
                                       negated=bool(triplet.get("negated", False)),
                                       source_claims=matched_claims)
                        if edge is not None:
                            for claim in matched_claims:
                                key = {"claim_id": claim.id, "edge_id": edge.id}
                                if db.get(CodexClaimLink, (claim.id, edge.id)) is None:
                                    db.add(CodexClaimLink(**key))

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("codex_graph_assertions_committed", extracted_count=len(triplets))

    except Exception as exc:
        from src.workers.runtime import JobYielded

        db.rollback()
        if not isinstance(exc, JobYielded):
            log.error("codex_extraction_aborted", error=str(exc))
        raise
    finally:
        db.close()