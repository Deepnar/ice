"""Hybrid Retrieval Orchestrator – Phase A hardened: decay filtering, access-weighting,
wide‑net full‑vector, Codex/Procedural scoping, HyDE rewriting, procedural trigger matching,
micro‑NER integration, and dynamic token budget."""

import hashlib
import math
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog
from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy import bindparam, or_, text
from sqlalchemy.orm import Session

from src.api.config import settings
from src.classifier.schemas import ClassificationResult
from src.memory.tokens import count as count_tokens
from src.retrieval import coverage
from src.memory.models import (
    CodexEdge,
    CodexEntity,
    EpisodicMemory,
    ProceduralMemory,
)
from src.retrieval.evolution import build_entity_timeline, history_exists
from src.retrieval.ner_utils import extract_entities
from src.retrieval.timescope import CURRENT, from_scope
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name

logger = structlog.get_logger("ice.retrieval")


@dataclass(frozen=True)
class ContextFragment:
    text: str
    source_type: str          # "episodic", "codex", "procedural", "timeline"
    score: float              # RRF fused score
    token_count: int
    source_batch_id: Optional[str] = None
    conversation_id: Optional[str] = None
    # C1: the *trusted* compact form (grounded summary with passing coverage)
    # when text is the raw representation — lets the token budget degrade a
    # too-big fragment to its summary instead of dropping it entirely. None
    # when text already is the summary, no trusted summary exists, or the
    # keyword that matched lives only in the raw.
    degrade_text: Optional[str] = None
    # C3: the one-line abstract — the LAST degradation step (raw → summary →
    # abstract); attached whenever the turn's summary is trusted, never
    # preferred by the chooser.
    abstract_text: Optional[str] = None
    # C16: the SPECIFIC retrieval mechanism, alongside the coarse source_type.
    # Eight legs report as five source_types — bm25, vector, chunks, cold and
    # the wide net all say "episodic" — which is why no experiment ICE has run
    # could say whether BM25, vector search or chunk retrieval did the episodic
    # work. Exp 2 could report Codex at 3.3% of fragments and nothing about the
    # other 96%.
    #
    # ⚠ ATTRIBUTION, NEVER ENTITLEMENT. The leg-diversity guarantee in
    # `_enforce_token_budget` keys on `source_type` on purpose: re-keying it on
    # this field would hand cold storage and the wide net guaranteed slots they
    # do not have today, as a side effect of a measurement change. Conditional
    # legs stay bounded by their own firing conditions.
    leg: Optional[str] = None

# C8: recency INSIDE the vector score (the candidate set is LIMIT 100 by
# score — post-fusion bonuses can't rescue a recent turn that never made the
# cut). Mirrors A11's codex formula, gentler because episodic already has
# post-hoc recency bonuses; skipped for creative (recent meta turns are noise,
# same rule as _apply_bonuses).
EPISODIC_RECENCY_BOOST = 0.25
EPISODIC_RECENCY_TAU_DAYS = 30.0

# C15: wide-net budget derives from the model-aware retrieval budget instead
# of a hardcoded 2,000 tokens.
WIDE_NET_BUDGET_FRACTION = 0.3
WIDE_NET_BUDGET_FLOOR = 1500


def _head_confidences(classification):
    """C15: honest per-head (topic, intent) confidences.

    B1: the classifier publishes these directly now — it is the one component
    that knows which schema its checkpoint was trained on. Two fallbacks behind
    that: slice `raw_probs` using whichever schema matches their width (25 = a v1
    checkpoint, 27 = v2), then the result's own max_confidence for a result with
    no probabilities at all. That last case had one producer, DI3, and D8 deleted
    it — the branch stays as a guard (see schema.empty_probs).
    """
    from src.classifier.schema import INTENT, TOPIC, resolve_by_width

    heads = getattr(classification, "head_confidences", None) or {}
    if heads.get(TOPIC) is not None and heads.get(INTENT) is not None:
        return float(heads[TOPIC]), float(heads[INTENT])

    probs = getattr(classification, "raw_probs", None) or []
    schema = resolve_by_width(len(probs)) if probs else None
    if schema is not None and any(p > 0.0 for p in probs):
        return (max(probs[schema.slice(TOPIC)]), max(probs[schema.slice(INTENT)]))

    mc = getattr(classification, "max_confidence", 1.0)
    return mc, mc


def _truncate_at_sentence(text: str, word_cap: int) -> str:
    """C3 smarter truncation: cap at *word_cap* words but cut on the last
    sentence boundary inside the cap (when one exists past 60% of it), so
    fragments stop mid-thought less often. Falls back to the hard word cut."""
    words = text.split()
    if len(words) <= word_cap:
        return text
    hard = ' '.join(words[:word_cap])
    best = -1
    for m in re.finditer(r'[.!?](?:\s|$)', hard):
        best = m.end()
    if best > len(hard) * 0.6:
        return hard[:best].rstrip() + ' …'
    return hard + '…'


# ---------------------------------------------------------------------------
# Scoring constants – single source for all additive bonuses
# ---------------------------------------------------------------------------
BONUS_BOOKMARKED = 0.5            # +50% of base score
BONUS_RECENT_TOP_10PCT = 1.0      # +100% if in the most recent 10% of the conversation
BONUS_RECENT_TOP_30PCT = 0.5      # +50% if in the most recent 30%
BONUS_LONG_NARRATIVE = 1.5        # +150% if >800 words (likely a full chapter)
BONUS_SUBSTANTIAL = 0.5           # +50% if >400 words
PENALTY_SHORT = -0.7              # −70% if <80 words
BONUS_KEYWORD_MATCH = 1.0         # +100% if fragment contains a prompt keyword
MAX_TOTAL_BONUS_MULTIPLIER = 4.0  # maximum bonus sum (score can be multiplied by up to 5.0)

# Soft meta‑discussion downweight – classifier‑driven, not string‑matching
NARRATIVE_FACT_INTENTS = {"Factual_Retrieval", "Decision_Making"}
META_LEANING_INTENTS = {"Analysis_&_Summarization"}
META_DOWNWEIGHT_FACTOR = 0.55     # multiply score by this if source turn leans meta

# A4: process-wide cache of (relation_names, gloss_embeddings) for relation
# detection — built lazily on first use by _relation_gloss_cache().
_RELATION_GLOSSES = None

class HybridRetrievalOrchestrator:
    def __init__(self, db: Session, embedder):
        self.db = db
        self.embedder = embedder
        self.bg_client = get_bg_client()
        self.max_retrieval_tokens = 5000
        self._coverage_record = None   # C16: last coverage decision, for audit
        self._force_hyde = False
        # A4: entity resolution mode (ablation `fuzzy_match` flag maps here).
        self.use_fuzzy_match = True
        # A4: relation/tag-driven enumeration for entity-less category queries
        # ("list all the characters"). Replaces MERA (−0.21 in the ablation):
        # same capability, but grounded in the controlled vocabulary + entity
        # tags via embedding similarity — no LLM call in the hot path.
        # The ablation `mera` flag maps onto this.
        self.enable_enumeration = True

        # A4 — relation detection. Empirical note: prompt↔gloss similarity is
        # only reliable as a *joint* signal (top-3 accuracy is good, but neutral
        # prompts score ~0.69 absolute), so detected relations are never a
        # trigger on their own — they only boost/annotate edges of matched
        # entities, or drive enumeration when explicit cue words are present.
        # Top-k is recall-only: a detected relation surfaces facts only when a
        # matched entity actually has such an edge (the join is the precision),
        # so k=5 is safe — feasibility probe: top-3 11/12, top-5 12/12.
        self.RELATION_TOP_K = 5
        self.RELATION_SIM_FLOOR = 0.45           # below this a gloss match is noise
        self.RELATION_OVERLAP_BOOST = 0.25       # entity-hit ∩ relation-hit fragment boost
        self.EXPANSION_MAX_TERMS = 8             # grounded query expansion cap (BM25)
        self.ENUM_EDGE_LIMIT = 15                # enumeration: max fact edges surfaced
        self.ENUM_ENTITY_LIMIT = 8               # enumeration: max entities surfaced
        self._last_matched_entities = []         # per-call: for grounded expansion

        # A3 — edge confidence/strength as a live retrieval signal.
        # An edge's effective trust is strength * extraction_confidence:
        # strength carries usage dynamics (reinforcement/decay), confidence
        # carries extraction trust (NER grounding, corroboration).
        self.CODEX_MAX_DEPTH = 3                 # traversal ceiling (gated below)
        self.CODEX_DIRECT_TRUST_FLOOR = 0.5      # matched-entity edge must clear this to expand/reinforce
        self.CODEX_DEEP_STRENGTH_FLOOR = 1.0     # deep hops require this much effective trust
        self.CODEX_REINFORCE_INCREMENT = 0.15    # per-retrieval boost on anchor edges (episodic analog)
        self.CODEX_STRENGTH_CAP = 10.0           # soft ceiling so retrieval can't inflate forever
        self.CODEX_PROMOTE_STRENGTH = 2.0        # reinforced pending edge promotes to active (enters decay cycle)
        self.CODEX_PROMOTE_MIN_CONFIDENCE = 0.5  # ...but never on low-trust extractions (needs corroboration first)
        # A11: mild recency boost on edge trust — a recently-asserted fact
        # outranks a stale one of equal strength. Rewards recent assertion
        # (valid_from); never penalises age (decay already handles that).
        self.CODEX_RECENCY_BOOST = 0.3           # max multiplier bump for a just-asserted edge
        self.CODEX_RECENCY_TAU_DAYS = 30.0       # e-folding time of the boost

        # T2/T3: the request's TimeScope (set from scope["timescope"] at the
        # top of retrieve()/_wide_net_fallback(); one orchestrator instance
        # per request, so an instance attr is safe). timescope_allowed is the
        # ablation seam — the configurable subclass sets it False to force
        # CURRENT regardless of scope.
        self._active_timescope = CURRENT
        self.timescope_allowed = True
        self._cold_hits = {}
        # E1b (D3): the request's attached project (scope["project_id"]) —
        # source-visibility for derived code/fact entities keys off this,
        # same instance-attr pattern as _active_timescope.
        self._scope_project_id = None
        # C6: the request's exclusion deny sets (same pattern). Empty means
        # "nothing excluded", which is the default and the common case.
        self._denied_batch_ids = set()
        self._denied_entity_ids = set()

        # Load micro‑NER model (fallback to None if not available)
    def _relevant_cluster_ids(self, prompt_embedding, classification=None, conversation_id=None, top_k=10, scope=None):
        """Return a list of cluster_id strings for the clusters most
        relevant to the prompt, using both embedding similarity and
        topic‑tag overlap with the current classification.
        """
        try:
            # G29: the shared scope filter — under project scope conv_id is
            # None and the ids live in scope["conversation_ids"]; hand-rolling
            # `if conversation_id` here searched every cluster in the store.
            conv_filter, conv_params, _ = self._conv_scope_filter(scope, conversation_id)
            query = text(f"""
                SELECT id, 1 - (embedding <=> :emb) AS sim, tags, name, description
                FROM context_clusters
                WHERE embedding IS NOT NULL
                {conv_filter}
                ORDER BY sim DESC
                LIMIT :limit
            """).bindparams(bindparam("emb", type_=PgVector))
            params = {"emb": prompt_embedding, "limit": top_k * 3, **conv_params}
            rows = self.db.execute(query, params).fetchall()
        except Exception:
            return []

        if not rows:
            return []

        # Boost clusters whose tags overlap with the current topic tags.
        # C5: the per-row name/description embedder.encode is GONE — it cost
        # up to top_k*3 model forward passes in the synchronous hot path for
        # a signal redundant with the centroid (the description text ≈ the
        # cluster content ≈ what the centroid already encodes).
        topic_tags = set(classification.topic_tags) if classification else set()
        scored = []
        for row in rows:
            cluster_tags = set(row.tags or [])
            tag_overlap = len(topic_tags & cluster_tags) if topic_tags else 0
            combined = row.sim + (0.3 * tag_overlap)
            scored.append((combined, str(row.id)))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or scored[0][0] < 0.50:
            # No cluster is confidently relevant – fall back to full conversation search
            return []
        # C5: adaptive band instead of a flat top-10 — keep clusters whose
        # score is within 80% of the best match (capped at top_k). A single
        # dominant cluster scopes tightly; several comparably-relevant
        # clusters all stay in; weak tails drop out instead of padding the
        # scope to a fixed count.
        best = scored[0][0]
        band = [cid for s, cid in scored if s >= 0.8 * best]
        return band[:top_k]
    
    def _apply_bonuses(self, fragments, classification, conv_id, prompt_keywords):
        creative = bool({"Creative_&_Media"} & set(classification.topic_tags))
        wants_narrative_fact = bool(NARRATIVE_FACT_INTENTS & set(classification.intent_tags))

        out = []
        for f in fragments:
            bonus = 0.0
            text_lower = f.text.lower()
            word_count = len(f.text.split())

            # Keyword match
            if prompt_keywords and any(
                kw in text_lower or kw.rstrip('s') in text_lower for kw in prompt_keywords
            ):
                bonus += BONUS_KEYWORD_MATCH

            # Length
            if word_count > 800:
                bonus += BONUS_LONG_NARRATIVE
            elif word_count > 400:
                bonus += BONUS_SUBSTANTIAL
            elif word_count < 80:
                bonus += PENALTY_SHORT

            # Recency (skip for creative – recent meta turns are noise;
            # T3: skipped under any non-current mode — this bonus points at now)
            if (f.source_type == "episodic" and not creative and conv_id
                    and self._active_timescope.mode == "current"):
                bonus += self._recency_bonus(f, conv_id)

            # Soft meta downweight
            if wants_narrative_fact and f.source_type == "episodic" and f.source_batch_id:
                if self._turn_leans_meta(f.source_batch_id):
                    bonus -= (1.0 - META_DOWNWEIGHT_FACTOR)

            bonus = max(-0.9, min(MAX_TOTAL_BONUS_MULTIPLIER, bonus))
            new_score = f.score * (1.0 + bonus)
            out.append(replace(f, score=new_score))
        return out

    def _recency_bonus(self, fragment, conv_id):
        try:
            turn = self.db.query(EpisodicMemory).get(uuid.UUID(fragment.source_batch_id)) \
                if fragment.source_batch_id else None
            if not turn:
                return 0.0
            total = self.db.query(EpisodicMemory).filter_by(conversation_id=conv_id).count()
            if total <= 20:
                return 0.0
            newer_count = self.db.query(EpisodicMemory).filter(
                EpisodicMemory.conversation_id == conv_id,
                EpisodicMemory.timestamp > turn.timestamp
            ).count()
            recency_pct = newer_count / total
            if recency_pct < 0.10:
                return BONUS_RECENT_TOP_10PCT
            elif recency_pct < 0.30:
                return BONUS_RECENT_TOP_30PCT
            return 0.0
        except Exception:
            return 0.0

    def _turn_leans_meta(self, source_batch_id):
        """Check the source turn's intent_tags for meta/analytical leaning."""
        try:
            turn = self.db.query(EpisodicMemory).get(uuid.UUID(source_batch_id))
            if not turn or not turn.intent_tags:
                return False
            return bool(META_LEANING_INTENTS & set(turn.intent_tags))
        except Exception:
            return False
    def _extract_prompt_keywords(self, prompt_text):
        words = set(re.sub(r'[^\w\s]', ' ', prompt_text).lower().split())
        common = {"the","is","of","and","a","to","in","that","it","for","was","on","are",
                  "with","what","when","where","who","how","i","you","me","my","we","our",
                  "so","be","do","did","does","get","got","very","too","now","this","that",
                  "these","those","some","many","each","every","other","more","gonna","wanna"}
        return words - common



    TOTAL_CONTEXT_BUDGET = 23_000          # fallback ceiling when no model-derived budget is passed (C16)
    OVERHEAD_RESERVE = 1_800               # system message + slots + question


    def set_budget_from_turn_count(
        self, turn_count: int, total_tokens: int = 0, classification=None,
        total_budget: int = None,
    ):
        # C16 (model-aware half): the caller derives total_budget from the routed
        # model's context window (derive_total_budget); the class constant is
        # only the fallback for legacy/direct callers.
        total = total_budget if total_budget else self.TOTAL_CONTEXT_BUDGET
        available = total - self.OVERHEAD_RESERVE
        fraction = self._compute_recent_fraction(turn_count, total_tokens, classification)
        recent_budget = int(available * fraction)
        raw_retrieval = available - recent_budget

        # Growth-based retrieval cap (same as before)
        if turn_count < 30:
            growth_cap = 2_000 + turn_count * 150
        elif turn_count < 100:
            growth_cap = 5_000 + (turn_count - 30) * 100
        elif turn_count < 500:
            growth_cap = 10_000 + (turn_count - 100) * 30
        else:
            growth_cap = raw_retrieval

        retrieval_budget = min(raw_retrieval, growth_cap)

        # NO reallocation — leftover stays unused. This is what makes ICE
        # token-efficient compared to the vector baseline.

        self.recent_token_budget = recent_budget
        self.max_retrieval_tokens = retrieval_budget

        # Remove the assertion — the sum is intentionally less than 'available'


    def _compute_recent_fraction(self, turn_count: int, total_tokens: int = 0, classification=None) -> float:
        """Return 0.0‑1.0 – how much of the context budget goes to recent turns."""
        # Base – conversation length
        if turn_count < 10:
            base = 0.3
        elif turn_count < 50:
            base = 0.2
        elif turn_count < 200:
            base = 0.2
        elif turn_count < 500:
            base = 0.15
        else:
            base = 0.15

        # Token‑density adjustment: if average tokens/turn is huge, shift some
        # budget toward retrieval to avoid recent‑only context dominated by a
        # couple of very long turns.
        if turn_count > 0 and total_tokens > 0:
            avg_tokens_per_turn = total_tokens / turn_count
            if avg_tokens_per_turn > 3000:
                base -= 0.15
            elif avg_tokens_per_turn > 1500:
                base -= 0.10
            elif avg_tokens_per_turn > 800:
                base -= 0.05

        modifier = 0.0
        if classification is not None:
            intents = set(classification.intent_tags)
            if intents & {"Factual_Retrieval", "Troubleshooting", "Analysis_&_Summarization"}:
                modifier -= 0.10
            if intents & {"Emotional_Processing", "Casual_Banter"}:
                modifier += 0.10
            topics = set(classification.topic_tags)
            if "Creative_&_Media" in topics:
                modifier += 0.05
            if "Software_&_Tech" in topics:
                modifier -= 0.05
            if topics & {"Social_&_Relationships", "Lifestyle_&_Health"}:
                modifier += 0.05

        return max(0.05, min(0.85, base + modifier))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def retrieve(
        self,
        classification: ClassificationResult,
        conversation_id: str,
        prompt_embedding: list[float],
        scope: Optional[dict] = None,
    ) -> List[ContextFragment]:
        # The retrieve/no-retrieve decision is now made upstream in one place
        # (B2, src/api/memory_decision.py) — retrieve() is only called when that
        # decision says so, and main.py sets context_reliance accordingly. The
        # old Zero_Shot+conversation and Creative belt-and-suspenders forces
        # (which silently overrode that decision) are gone. These early returns
        # stay purely as a defensive guard if retrieve() is ever called directly.
        self._hyde_used = False
        self._last_hyde_query = None
        # T2: resolve the request's TimeScope once; every leg below reads it.
        self._active_timescope = self._resolve_timescope(scope)
        # E1b (D3): attached project — gates derived-entity visibility.
        self._scope_project_id = self._resolve_project_scope(scope)
        # C6: the exclusion deny sets, resolved once for every leg below.
        self._resolve_exclusion_sets(scope)
        self._cold_hits = {}
        if classification.context_reliance == "Zero_Shot":
            return []
        if classification.context_reliance == "Real_Time_Search":
            return []

        # C15: the wide net fires on HONEST per-head uncertainty (topic AND
        # intent both weak), not the legacy max-over-all-25-probs — which was
        # dominated by whichever head happened to be peaked (the same broken
        # measure B2 replaced for the LTM decision). Wide net is a degraded
        # single-leg mode, so it fires only when the classifier is genuinely
        # lost about WHAT the user wants.
        topic_conf, intent_conf = _head_confidences(classification)
        if (topic_conf < settings.confidence_fallback_threshold
                and intent_conf < settings.confidence_fallback_threshold):
            logger.info("wide_net_fallback_triggered",
                        topic_confidence=round(topic_conf, 3),
                        intent_confidence=round(intent_conf, 3))
            return self._wide_net_fallback(classification, prompt_embedding, conversation_id, scope)

        # Conversation scope filter for episodic legs
        # Conversation scope filter for episodic legs
        conv_id = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]

        # ── Cluster‑scoped retrieval: find the most relevant clusters
        #     and add them to the scope so the episodic legs only search
        #     those clusters.  Falls back gracefully if no clusters exist.
        # C6: an EXPLICIT cluster choice is not a suggestion. This used to
        # overwrite scope["cluster_ids"] unconditionally, so a hand-picked set
        # only survived when the automatic picker happened to return nothing
        # (best combined score < 0.50) — the user's pick was, in effect, a
        # fallback for the machine's. The user's choice now wins outright.
        if scope and scope.get("cluster_ids_explicit"):
            logger.debug("cluster_scope_explicit_kept",
                         clusters=len(scope.get("cluster_ids") or []))
        else:
            cluster_ids = self._relevant_cluster_ids(prompt_embedding, classification=classification, conversation_id=conv_id, top_k=10, scope=scope)
            if cluster_ids and scope is not None:
                scope["cluster_ids"] = cluster_ids

        # HyDE query rewriting
        # hyde_prompt = None
        # if self._force_hyde or classification.context_reliance == "Long_Term_Memory":
        #     hyde_prompt = self._hyde_rewrite(classification.prompt, conversation_id)
        #     search_prompt = hyde_prompt if hyde_prompt else classification.prompt
        #     if self._force_hyde:
        #         self._hyde_used = hyde_prompt is not None
        #         self._last_hyde_query = hyde_prompt

        # # Re‑compute embedding if the search prompt changed
        # if hyde_prompt:
        #     prompt_embedding = self.embedder.encode(search_prompt, convert_to_tensor=False).tolist()
        # A4: run the codex leg first — the entities it resolves ground the
        # query expansion for the lexical leg below.
        codex_fragments = self._codex_graph(classification, scope,
                                            prompt_embedding=prompt_embedding)
        # T4: timelines ride back with the codex fragments but fuse and
        # budget as their own leg — RRF weight is keyed by legs-dict entry,
        # the budget's round-robin lane by source_type.
        timeline_fragments = [f for f in codex_fragments if f.source_type == "timeline"]
        codex_fragments = [f for f in codex_fragments if f.source_type != "timeline"]

        # A4 grounded query expansion (the sane replacement for HyDE): append
        # matched entities' canonical names + aliases to the BM25 search prompt
        # so lexical search hits turns that use the full/other name. Nothing is
        # generated — expansion terms come from the graph, not a model.
        search_prompt = classification.prompt
        expansion = self._expansion_terms()
        if expansion:
            search_prompt = f"{classification.prompt} {' '.join(expansion)}"
            logger.info("grounded_query_expansion", terms=expansion)

        # G16 incognito: episodic legs are conversation-scoped via conv_id;
        # codex resolves an empty scope set (A5 `isolated`); the user-global
        # user-global leg (procedural patterns) reads nothing at all.
        incognito = bool(scope and scope.get("incognito"))

        # Prompt keywords: used by the cold leg's ILIKE patterns and the
        # post-fusion bonus pass.
        prompt_keywords = self._extract_prompt_keywords(classification.prompt) if classification.prompt else set()

        # Execute the remaining retrieval legs
        legs: Dict[str, List[ContextFragment]] = {
            "bm25": self._bm25_episodic(classification, scope, conv_id, search_prompt),
            "vector": self._vector_episodic(prompt_embedding, classification, scope, conv_id),
            "codex": codex_fragments,
            "procedural": [] if incognito else self._procedural_lookup(prompt_embedding, classification, scope),
            # C4: the cross-conversation summary half is a user-global read —
            # gated off under incognito like procedural; the conv-scoped
            # batch half (own context) still runs.
            "batch_summary": self._batch_summary_lookup(
                prompt_embedding, conv_id, include_cross=not incognito),
            # T3: cold storage joins time-scoped queries only (no-op leg
            # otherwise); fragments are episodic-typed, so budget fairness
            # treats them as memories — the leg name only affects RRF weight.
            "cold": self._cold_lookup(prompt_keywords, conv_id, scope,
                                      prompt_embedding),
            "timeline": timeline_fragments,
        }

        # ── Dynamic leg weighting (blended over all active intents) ──

        # Base balanced weights
        base_weights = {
            "bm25": 0.8,
            "vector": 1.0,
            "codex": 0.5,
            "procedural": 0.2,
            "timeline": 0.6,   # T4: pinned below — constant across profiles
        }

        # Profile definitions: each profile → (intents, weight_override)
        PROFILES = [
            ({"Factual_Retrieval", "Utility_Formatting"},
            {"vector": 1.2, "bm25": 0.8, "codex": 0.1, "procedural": 0.1}),
            ({"Troubleshooting", "Strategic_Planning"},
            {"vector": 1.0, "bm25": 0.8, "codex": 0.3, "procedural": 1.2}),
            ({"Generation", "Ideation", "Open_Exploration"},
            {"vector": 0.6, "bm25": 0.6, "codex": 1.2, "procedural": 0.1}),
            ({"Emotional_Processing", "Analysis_&_Summarization", "Decision_Making"},
            {"vector": 1.1, "bm25": 0.6, "codex": 0.9, "procedural": 0.0}),
            ({"Casual_Banter", "Null_Noise"},
            {"vector": 0.5, "bm25": 0.2, "codex": 0.0, "procedural": 0.0}),
            # B1 D9: the v2 coding intent. Code_Change leans procedural — how
            # this user/project does changes (conventions, past fixes) matters
            # more than definitions.
            # Starting values — Z1-prep's tuning pass owns the final numbers.
            #
            # E12 (2026-07-27) measured whether this row is actually reached,
            # because the blend below divides every profile by len(active_intents)
            # and the roadmap's worry was that a 3-intent row leaves a third of a
            # nudge. Over 9,441 held-out rows: the head fires Code_Change on
            # 5.75%, carrying a mean of 1.79 intents (alone on 27%), so it lands
            # ~56% of its intended weight and shifts the biggest leg by 0.291
            # against base weights of 0.2–1.2. That is a real effect, so the row
            # STAYS. Its usual companion is Troubleshooting (269 of 543 rows),
            # which agrees on procedural 1.2 and disagrees on codex (0.3 vs 1.0)
            # — so the dilution that does happen mostly costs codex weight.
            #
            # D9's second row, Codebase_Query → {codex 1.3, bm25 0.9,
            # vector 0.8, procedural 0.6}, is deliberately absent: the label was
            # dropped from label_schema.json in B1's run 2 (test F1 0.10 — see
            # dropped_labels there). It cannot appear in intent_tags, so a
            # profile for it would be unreachable. Those weights are the tuned
            # starting point if E7's MCP traffic earns the label back.
            ({"Code_Change"},
            {"procedural": 1.2, "codex": 1.0, "vector": 0.6, "bm25": 0.6}),
        ]

        # Build a mapping from intent label → its profile’s override
        intent_to_profile_weights = {}
        for intents_in_profile, override in PROFILES:
            for intent in intents_in_profile:
                intent_to_profile_weights[intent] = override

        # Blend: each active intent contributes equally
        active_intents = classification.intent_tags
        num_active = len(active_intents) if active_intents else 1

        blend_weights = {leg: 0.0 for leg in base_weights}
        for tag in active_intents:
            profile_weights = intent_to_profile_weights.get(tag)
            if profile_weights:
                for leg, w in profile_weights.items():
                    blend_weights[leg] += w / num_active
            else:
                # Unknown intent – use base weights
                for leg, w in base_weights.items():
                    blend_weights[leg] += w / num_active

        # Fallback to base weights if nothing matched
        if all(v == 0.0 for v in blend_weights.values()):
            blend_weights = dict(base_weights)

        # --- TOPIC OVERRIDES (cumulative) ---
        if "Creative_&_Media" in set(classification.topic_tags):
            blend_weights["codex"] = blend_weights.get("codex", 0.5) + 0.3
        if "Software_&_Tech" in set(classification.topic_tags):
            blend_weights["procedural"] = blend_weights.get("procedural", 0.2) + 0.4

        # T4: the timeline weight is constant across intent profiles — its
        # firing condition (history_exists on a matched anchor) is the gate,
        # not the intent. The profile dicts don't carry it, so pin it here.
        blend_weights["timeline"] = base_weights["timeline"]

        # Clamp to zero (no negative weights)
        for leg in blend_weights:
            blend_weights[leg] = max(0.0, blend_weights[leg])

        # Fuse, diversify, deduplicate, trim
        fused = self._apply_rrf(legs, alpha_map=blend_weights)
        fused = self._apply_bonuses(fused, classification, conv_id, prompt_keywords)
        fused.sort(key=lambda x: x.score, reverse=True)
                # ── Leg diversity guarantee: always include the top‑ranked fragment
        #     from each leg, even if RRF would have dropped it.

        # ── Keyword‑aware re‑ranking: fragments containing probe keywords
        #     get a massive score boost so they survive token‑budget enforcement.


        # (delete the old leg‑diversity block entirely)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        deduped = self._deduplicate(diversified)
        deduped = self._collapse_provenance(deduped)
        deduped = self._apply_coverage(deduped, prompt_embedding)
        final = self._enforce_token_budget(deduped)

        # T3 honest emptiness: a windowed query with nothing in the window
        # says so — never silently widens.
        final = self._append_empty_window_note(final, prompt_embedding, conv_id, scope)

        # Strengthen retrieved turns (access count + decay boost)
        self._strengthen_retrieved(final)

        # T3 (D-U1): budget-surviving cold hits get their probation
        # resurrection (after strengthening — probation starts exactly at
        # the configured score, not score+0.15).
        self._resurrect_cold_hits(final)

        return final

    def _resolve_timescope(self, scope):
        """T2: the request's TimeScope from scope["timescope"], forced to
        CURRENT when the kill-switch (settings) or the ablation seam
        (timescope_allowed) is off."""
        if not settings.timescope_enabled or not getattr(self, "timescope_allowed", True):
            return CURRENT
        return from_scope(scope)

    def _resolve_project_scope(self, scope):
        """E1b (D3/D11): the attached project's id (UUID) from
        scope["project_id"], or None. Tolerant of garbage — no project."""
        if not scope or not scope.get("project_id"):
            return None
        try:
            return uuid.UUID(str(scope["project_id"]))
        except (ValueError, AttributeError, TypeError):
            return None

    def _entity_source_filters(self):
        """D3 derived-memory visibility, as SQLAlchemy clauses for the entity
        matching/enumeration queries: conversation entities always; derived
        (static_analysis / project-fact) entities only when the query's
        attached project matches. Extend-don't-fork — the same clause list is
        appended wherever entities are resolved (G16 invariant style)."""
        base = or_(CodexEntity.source.is_(None),
                   CodexEntity.source == "conversation")
        if self._scope_project_id is not None:
            return [or_(base, CodexEntity.project_id == self._scope_project_id)]
        return [base]

    def _entity_visible(self, entity) -> bool:
        """Python-side twin of _entity_source_filters for traversal expansion
        (neighbors arrive via edges, not the filtered queries)."""
        source = getattr(entity, "source", None)
        if source is None or source == "conversation":
            return True
        return (self._scope_project_id is not None
                and entity.project_id == self._scope_project_id)

    def _conv_scope_filter(self, scope, conv_id, column="conversation_id"):
        """D11/C6 seam: the episodic legs' conversation filter — a single
        conversation (today's behavior, byte-identical) or a conversation-id
        list (project scope resolves to the project's conversations). Returns
        (sql_snippet, params, scoped) — *scoped* drives the privacy filter
        (explicit scoping is the only door to private turns, G16).

        C6: the list is tested for PRESENCE, not truthiness. An exclusion set
        can empty a closed conversation set, and an empty set must match
        nothing — `= ANY('{}')` is false, which is exactly right. Under the
        old truthiness test an emptied list fell through to "no filter at
        all", i.e. a scoped query silently going global."""
        conv_ids = scope.get("conversation_ids") if scope else None
        if conv_ids is not None:
            return (f"AND {column} = ANY(:conv_ids)",
                    {"conv_ids": [str(c) for c in conv_ids]}, True)
        if conv_id:
            return f"AND {column} = :conv_id", {"conv_id": conv_id}, True
        return "", {}, False

    def _exclusion_filters(self, scope, conv_column="conversation_id",
                           id_column="episodic_memory.id"):
        """C6: the negated scope — "keep this memory, stop retrieving it".

        The middle ground that never existed between C6's add-to-scope and
        C10's delete-forever: an abandoned project you may come back to should
        stop surfacing without being destroyed. Honored in EVERY scope mode
        (user decision 2026-07-28), so an exclusion has no hidden precondition.

        *id_column* None skips the cluster arm — cold_storage rows carry no
        episodic_cluster_links. Returns (sql_snippet, params)."""
        sql, params = "", {}
        if not scope:
            return sql, params
        excluded_convs = scope.get("exclude_conversation_ids")
        if excluded_convs:
            sql += f"\n              AND {conv_column} <> ALL(:excl_conv_ids)"
            params["excl_conv_ids"] = [str(c) for c in excluded_convs]
        excluded_clusters = scope.get("exclude_cluster_ids")
        if excluded_clusters and id_column:
            sql += (f"\n              AND NOT EXISTS ("
                    f"SELECT 1 FROM episodic_cluster_links xl "
                    f"WHERE xl.episodic_id = {id_column} "
                    f"AND xl.cluster_id = ANY(:excl_cluster_ids))")
            params["excl_cluster_ids"] = [str(c) for c in excluded_clusters]
        return sql, params

    def _timescope_leg_filters(self, prefix=""):
        """T3: param-driven filter snippets for the episodic legs — one query
        text per leg, never a timescoped twin (G19). current: archived hidden,
        decay floor 0.2, no window (today's behavior). Non-current: window
        predicate (as_of/range only), archived visible (D10 — "archived" means
        not-relevant-to-*now*, which a time query is not asking), and the
        decay floor dropped (archived rows sit below 0.1 by construction —
        spec rev note 3)."""
        ts = self._active_timescope
        params = {}
        time_filter = ""
        if ts.mode in ("as_of", "range") and ts.t0 and ts.t1:
            time_filter = (f"AND {prefix}timestamp >= :ts_t0 "
                           f"AND {prefix}timestamp < :ts_t1")
            params["ts_t0"], params["ts_t1"] = ts.t0, ts.t1
        archived_filter = f"AND {prefix}is_archived = false" if ts.mode == "current" else ""
        min_decay = 0.2 if ts.mode == "current" else 0.0
        return time_filter, archived_filter, params, min_decay

    def _recency_params(self, creative: bool):
        """T3 (D9): mode-aware recency — the same exponential with a movable
        origin, returned as (center, boost, tau_days). current: center=now,
        boost as today (0 for creative) — numerically identical to the old
        NOW()-based formula for past rows. as_of: center = window midpoint,
        boost applies even for creative (this is target-proximity relevance,
        not freshness), tau widens with the window. range/evolution: flat —
        the user asked for a period, not proximity to its midpoint."""
        ts = self._active_timescope
        now = datetime.now(timezone.utc)
        if ts.mode == "as_of" and ts.t0 and ts.t1:
            center = ts.t0 + (ts.t1 - ts.t0) / 2
            window_days = (ts.t1 - ts.t0).total_seconds() / 86400.0
            return center, EPISODIC_RECENCY_BOOST, max(EPISODIC_RECENCY_TAU_DAYS, window_days / 2)
        if ts.mode in ("range", "evolution"):
            return now, 0.0, EPISODIC_RECENCY_TAU_DAYS
        return now, (0.0 if creative else EPISODIC_RECENCY_BOOST), EPISODIC_RECENCY_TAU_DAYS

    # ------------------------------------------------------------------
    # HyDE rewriting
    # ------------------------------------------------------------------
    def _hyde_rewrite(self, prompt: str, conversation_id: str = None) -> Optional[str]:
        # Collect the last 5 turns as context for the rewrite
        context_text = ""
        if conversation_id:
            try:
                recent = self.db.query(EpisodicMemory).filter_by(
                    conversation_id=conversation_id
                ).order_by(EpisodicMemory.timestamp.desc()).limit(5).all()
                recent.reverse()
                parts = []
                for t in recent:
                    parts.append(t.raw_text[:300])
                context_text = "\n".join(parts)
            except Exception:
                pass

        rewrite_prompt = (
            f"Recent conversation:\n{context_text}\n\nUser's question:\n{prompt}"
            if context_text else prompt
        )

        try:
            resp = self.bg_client.chat.completions.create(
                model=get_bg_model_name(),
                messages=[
                    {"role": "system", "content": (
                        "You are a query rewriting engine. Take the user's question and rewrite it "
                        "as a dense, factual search query that would retrieve the relevant past conversation. "
                        "Include key entities and omit polite phrasing. Output ONLY the rewritten query, no other text."
                    )},
                    {"role": "user", "content": rewrite_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=15.0,
            )
            rewritten = resp.choices[0].message.content.strip()
            return rewritten if rewritten else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # BM25 episodic (full‑text search)
    # ------------------------------------------------------------------
    def _bm25_episodic(self, classification, scope, conv_id=None, search_prompt=None):
        prompt_text = search_prompt if search_prompt else classification.prompt
        # Remove all non‑alpha characters and split into words
        clean_prompt = re.sub(r'[^a-zA-Z]', ' ', prompt_text)
        words = [w.strip().lower() for w in clean_prompt.split() if len(w.strip()) > 2]
        # Keep only words that look like real English tokens
        stop_words = {"the","and","for","you","that","this","with","from","have","are","was","were",
                      "will","would","could","should","about","also","just","like","then","than","over",
                      "into","only","more","some","such","each","every","other","many","most","its",
                      "our","his","her","they","them","these","those","not","but","can","all","been",
                      "had","has","did","does","get","got","very","too","now","how"}
        valid_words = [w for w in words[:30] if w not in stop_words]
        # Build individual to_tsquery tokens and join with OR
        try:
            tokens = []
            for w in valid_words:
                try:
                    tokens.append(w)
                except Exception:
                    continue
            if tokens:
                search_terms = " | ".join(tokens)
            else:
                search_terms = prompt_text
        except Exception:
            search_terms = prompt_text

        topic_filter = ""
        # D11: single conversation OR the project's conversation list.
        conv_filter, conv_params, conv_scoped = self._conv_scope_filter(scope, conv_id)
        # C6: and the negated scope — excluded conversations/clusters.
        excl_filter, excl_params = self._exclusion_filters(scope)
        # G16 visibility invariant: global search never sees private (incognito)
        # turns; explicit conversation scoping is the only door to them.
        privacy_filter = "" if conv_scoped else "AND is_private = FALSE"
        # T3: window / archived-visibility / decay-floor, driven by the mode.
        time_filter, archived_filter, ts_params, min_decay = self._timescope_leg_filters()

        # ── Cluster filter (new) ──
        cluster_filter = ""
        if scope and scope.get("cluster_ids"):
            cluster_filter = """
                AND (
                    EXISTS (
                        SELECT 1 FROM episodic_cluster_links l
                        WHERE l.episodic_id = episodic_memory.id
                          AND l.cluster_id = ANY(:cluster_ids)
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM episodic_cluster_links l
                        WHERE l.episodic_id = episodic_memory.id
                    )
                )
            """

        query = text(f"""
            SELECT id, raw_text, summary_text, summary_coverage, abstract_text, lossless_flag, inject_raw, conversation_id, is_bookmarked, timestamp,
                   ts_rank(
                       to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')),
                       query
                   ) as score
            FROM episodic_memory,
                 LATERAL (SELECT
                     CASE WHEN length(:search_terms) > 0
                          THEN to_tsquery('english', :search_terms)
                          ELSE plainto_tsquery('english', :prompt_text)
                     END AS query) AS q
            WHERE to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')) @@ query
              {topic_filter}
              {conv_filter}
              {excl_filter}
              {privacy_filter}
              {cluster_filter}
              {time_filter}
              AND decay_score > :min_decay
              {archived_filter}
            ORDER BY score DESC
            LIMIT 100
        """)
        params = {
            "search_terms": search_terms,
            "prompt_text": prompt_text,
            "min_decay": min_decay,
            **ts_params,
            **conv_params,
            **excl_params,
        }
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic", prompt_text=search_prompt or classification.prompt, classification=classification)
        except Exception as err:
            logger.error("bm25_retrieval_failed", error=str(err))
            self.db.rollback()
            # Final fallback: use plainto_tsquery (AND) if everything fails
            try:
                query2 = text(f"""
                    SELECT id, raw_text, summary_text, summary_coverage, abstract_text, lossless_flag, inject_raw, conversation_id, is_bookmarked, timestamp,
                           ts_rank(
                               to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')),
                               plainto_tsquery('english', :prompt_text)
                           ) as score
                    FROM episodic_memory
                    WHERE to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, ''))
                          @@ plainto_tsquery('english', :prompt_text)
                      {conv_filter}
                      {excl_filter}
                      {privacy_filter}
                      {cluster_filter}
                      {time_filter}
                      AND decay_score > :min_decay
                      {archived_filter}
                    ORDER BY score DESC
                    LIMIT 100
                """)
                p = {"prompt_text": prompt_text, "min_decay": min_decay,
                     **ts_params, **conv_params, **excl_params}
                if scope and scope.get("cluster_ids"):
                    p["cluster_ids"] = scope["cluster_ids"]
                rows = self.db.execute(query2, p).fetchall()
                return self._rows_to_fragments(rows, "episodic", prompt_text=search_prompt or classification.prompt, classification=classification)        
            except Exception:
                return []

    # ------------------------------------------------------------------
    # Vector episodic (pgvector cosine similarity)
    # ------------------------------------------------------------------
    def _vector_episodic(self, prompt_embedding, classification, scope, conv_id: Optional[str] = None) -> List[ContextFragment]:
        topic_filter = ""
        # D11: single conversation OR the project's conversation list.
        conv_filter, conv_params, conv_scoped = self._conv_scope_filter(scope, conv_id)
        # C6: and the negated scope — excluded conversations/clusters.
        excl_filter, excl_params = self._exclusion_filters(scope)
        # G16 visibility invariant: global search never sees private (incognito)
        # turns; explicit conversation scoping is the only door to them.
        privacy_filter = "" if conv_scoped else "AND is_private = FALSE"
        # T3: window / archived-visibility / decay-floor, driven by the mode.
        time_filter, archived_filter, ts_params, min_decay = self._timescope_leg_filters()

        # ── Cluster filter (new) ──
        cluster_filter = ""
        if scope and scope.get("cluster_ids"):
            cluster_filter = """
                AND (
                    EXISTS (
                        SELECT 1 FROM episodic_cluster_links l
                        WHERE l.episodic_id = episodic_memory.id
                          AND l.cluster_id = ANY(:cluster_ids)
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM episodic_cluster_links l
                        WHERE l.episodic_id = episodic_memory.id
                    )
                )
            """

        # C2: document turns are EXCLUDED from turn-level vector search — one
        # embedding over thousands of words is semantic mush (the C3 ceiling).
        # Their chunks compete instead, via _vector_chunks below.
        # C8 recency, T3/D9 origin-parameterized: ABS(timestamp - :ts_center)
        # with center=now is numerically identical to the old NOW()-based
        # GREATEST form for past rows; as_of re-anchors center to the window
        # midpoint. One formula, mode-driven params — never fork the leg SQL.
        query = text(f"""
            SELECT id, raw_text, summary_text, summary_coverage, abstract_text, lossless_flag, inject_raw, conversation_id, is_bookmarked, timestamp,
                (1 - (embedding <=> :prompt_embedding)) * COALESCE(decay_score, 1.0)
                  * (1 + :recency_boost * EXP(-ABS(EXTRACT(EPOCH FROM (timestamp - :ts_center))) / 86400.0 / :recency_tau)) as score
            FROM episodic_memory
            WHERE embedding IS NOT NULL
            AND is_document = false
            {topic_filter}
            {conv_filter}
            {excl_filter}
            {privacy_filter}
            {cluster_filter}
            {time_filter}
            AND decay_score > :min_decay
            {archived_filter}
            ORDER BY score DESC
            LIMIT :cand_limit
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        # C8: creative skips the in-score recency weight (same rule as the
        # post-fusion bonuses — recent meta turns are noise for narrative);
        # T3: except in as_of mode, where the boost is target-proximity.
        creative = "Creative_&_Media" in (classification.topic_tags or [])
        ts_center, rec_boost, rec_tau = self._recency_params(creative)
        # T4: evolution mode widens the candidate pool (no window, flat
        # recency already via _timescope_leg_filters/_recency_params) so the
        # era stratifier below has the idea's whole lifespan to sample from.
        ts = self._active_timescope
        params = {"prompt_embedding": prompt_embedding, "min_decay": min_decay,
                  "recency_boost": rec_boost, "recency_tau": rec_tau,
                  "ts_center": ts_center,
                  "cand_limit": 300 if ts.mode == "evolution" else 100,
                  **ts_params, **conv_params, **excl_params}
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]

        try:
            rows = self.db.execute(query, params).fetchall()
            if ts.mode == "evolution":
                rows = self._stratify_by_era(rows)
            fragments = self._rows_to_fragments(rows, "episodic", prompt_text=classification.prompt, classification=classification)
            # C3 dedupe: a chunk drops out when its parent turn is already in
            # the turn-level results (the parent covers the content); doc
            # parents are excluded above, so document chunks always compete.
            parent_ids = {str(r.id) for r in rows}
            fragments.extend(f for f in self._vector_chunks(prompt_embedding, scope, conv_id,
                                                            recency_boost=rec_boost)
                             if f.source_batch_id not in parent_ids)
            return fragments
        except Exception as err:
            logger.error("vector_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    def _stratify_by_era(self, rows):
        """T4: era-stratified sampling for evolution mode — sort candidates
        by timestamp, cut into equal-count buckets, keep each bucket's top
        scorers. Similarity alone lets one era (usually the recent, densest
        one) soak the whole result; an evolution answer needs the idea's
        early life represented too. Python-side on purpose (executor-proof,
        no SQL fork) and row-shape-agnostic — needs only .timestamp and
        .score, so C4's era digests can join the same candidate list later."""
        if not rows:
            return rows
        by_time = sorted(rows, key=lambda r: r.timestamp)
        bucket_size = math.ceil(len(by_time) / settings.evolution_era_buckets)
        kept = []
        for i in range(0, len(by_time), bucket_size):
            bucket = by_time[i:i + bucket_size]
            kept.extend(sorted(bucket, key=lambda r: r.score, reverse=True)
                        [:settings.evolution_per_era])
        return kept

    def _vector_chunks(self, prompt_embedding, scope, conv_id=None,
                       recency_boost=EPISODIC_RECENCY_BOOST) -> List[ContextFragment]:
        """C2: chunk-level vector search over document turns. Visibility
        (decay, archive, privacy, conversation, cluster scope) is enforced
        through the parent turn; provenance points at the parent so
        strengthening/decay land on the turn. Max 3 chunks per document."""
        conv_filter, conv_params, conv_scoped = self._conv_scope_filter(
            scope, conv_id, column="e.conversation_id")
        # C6: exclusions ride the parent turn like every other visibility rule.
        excl_filter, excl_params = self._exclusion_filters(
            scope, conv_column="e.conversation_id", id_column="e.id")
        privacy_filter = "" if conv_scoped else "AND e.is_private = FALSE"
        # T3: visibility rides the parent turn — window on e.timestamp,
        # archived/decay-floor on the parent's flags.
        time_filter, archived_filter, ts_params, min_decay = self._timescope_leg_filters(prefix="e.")
        ts_center, _, rec_tau = self._recency_params(creative=False)  # boost passed in by caller
        cluster_filter = ""
        if scope and scope.get("cluster_ids"):
            cluster_filter = """
                AND (
                    EXISTS (
                        SELECT 1 FROM episodic_cluster_links l
                        WHERE l.episodic_id = e.id
                          AND l.cluster_id = ANY(:cluster_ids)
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM episodic_cluster_links l
                        WHERE l.episodic_id = e.id
                    )
                )
            """
        query = text(f"""
            SELECT c.chunk_text, c.chunk_index, e.id AS parent_id,
                   e.conversation_id, e.is_bookmarked, e.timestamp,
                   (1 - (c.embedding <=> :prompt_embedding)) * COALESCE(e.decay_score, 1.0)
                     * (1 + :recency_boost * EXP(-ABS(EXTRACT(EPOCH FROM (e.timestamp - :ts_center))) / 86400.0 / :recency_tau)) AS score
            FROM episodic_chunks c
            JOIN episodic_memory e ON e.id = c.turn_id
            WHERE c.embedding IS NOT NULL
              AND e.decay_score > :min_decay
              {archived_filter}
              {time_filter}
              {conv_filter}
              {excl_filter}
              {privacy_filter}
              {cluster_filter}
            ORDER BY score DESC
            LIMIT 30
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        params = {"prompt_embedding": prompt_embedding, "min_decay": min_decay,
                  "recency_boost": recency_boost, "recency_tau": rec_tau,
                  "ts_center": ts_center, **ts_params, **conv_params,
                  **excl_params}
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]
        try:
            rows = self.db.execute(query, params).fetchall()
        except Exception as err:
            logger.error("chunk_vector_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

        fragments, per_parent = [], {}
        for row in rows:
            pid = str(row.parent_id)
            if per_parent.get(pid, 0) >= 3:
                continue
            per_parent[pid] = per_parent.get(pid, 0) + 1
            score_val = float(row.score)
            if row.is_bookmarked:
                score_val *= (1.0 + BONUS_BOOKMARKED)
            # T1: chunks are episodic fragments too — date them from the parent turn.
            chunk_text = row.chunk_text
            if row.timestamp:
                chunk_text = row.timestamp.strftime("[%Y-%m-%d] ") + chunk_text
            fragments.append(ContextFragment(
                text=chunk_text,
                source_type="episodic",
                score=score_val,
                token_count=count_tokens(chunk_text),
                source_batch_id=pid,
                conversation_id=str(row.conversation_id) if row.conversation_id else None,
            ))
        return fragments
        
    
    def _match_entities_by_similarity(self, entity_strings: List[str], threshold: float = 0.85) -> List:
        """Match extracted entity strings to CodexEntity rows using vector similarity.
        Falls back to canonical name / alias exact match when embeddings are unavailable."""
        if not entity_strings:
            return []

        # Embed all candidate strings
        candidate_embeddings = self.embedder.encode(entity_strings, convert_to_tensor=False, show_progress_bar=False)
        # Fetch all entities that have embeddings
        all_entities = self.db.query(CodexEntity).filter(CodexEntity.embedding != None).all()

        matched = []
        seen_ids = set()
        for candidate_str, candidate_emb in zip(entity_strings, candidate_embeddings):
            # 1) Vector similarity
            best_score = 0.0
            best_entity = None
            for ent in all_entities:
                if ent.id in seen_ids:
                    continue
                emb = ent.embedding
                if emb is None:
                    continue
                # cosine similarity = dot product of unit vectors. True since
                # C17: native-width encode IS unit-norm (the old truncate_dim
                # prefixes were NOT — this comparison ran deflated for the
                # whole 384 era).
                dot = sum(a * b for a, b in zip(candidate_emb, emb))
                score = dot
                if score > best_score and score >= threshold:
                    best_score = score
                    best_entity = ent
            if best_entity is not None:
                matched.append(best_entity)
                seen_ids.add(best_entity.id)
                continue

            # 2) Fallback: exact canonical name / alias match
            from sqlalchemy import or_
            norm = candidate_str.lower().strip()
            fallback = self.db.query(CodexEntity).filter(
                or_(CodexEntity.canonical_name == norm, CodexEntity.aliases.any(norm)),
                *self._entity_source_filters(),
            ).first()
            if fallback and fallback.id not in seen_ids:
                matched.append(fallback)
                seen_ids.add(fallback.id)

        return matched

    # ------------------------------------------------------------------
    # A4: relation detection + enumeration + grounded expansion helpers
    # ------------------------------------------------------------------
    _ENUM_CUES = ("list", "all", "who are", "what are", "every", "each",
                  "which", "name the", "tell me about", "enumerate")

    def _relation_gloss_cache(self):
        """Lazily embed the controlled relation vocabulary (as 'inspired by'
        style glosses) once per process. ~200 relations × native dims —
        trivial to hold and scan. (A4's per-vector re-normalization died with
        C17: native-width encode is already unit-norm.)"""
        global _RELATION_GLOSSES
        if _RELATION_GLOSSES is None:
            from src.workers.codex_extractor import ALLOWED_RELATIONS
            rels = sorted(ALLOWED_RELATIONS)
            embs = self.embedder.encode([r.replace("_", " ") for r in rels],
                                        convert_to_tensor=False, show_progress_bar=False)
            _RELATION_GLOSSES = (rels, [list(e) for e in embs])
        return _RELATION_GLOSSES

    @staticmethod
    def _stem(word: str) -> str:
        """Crude suffix-stripper so 'inspired'/'inspires'/'inspiring' all meet
        the relation lexeme 'inspired'. Both sides are stemmed identically, so
        crudeness cancels out."""
        w = word.lower()
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    def _detect_relations(self, prompt: str, prompt_embedding) -> List[str]:
        """Controlled-vocabulary relations relevant to the prompt, from two
        channels: (1) lexical — a relation's own content word appears in the
        prompt ('who inspired X' → inspired_by), which is a direct grounded
        hit; (2) embedding — top-k gloss cosine for paraphrases ('who is X's
        wife' → married_to). Joint-signal only (see __init__ note): callers
        must pair the result with matched entities or enumeration cues, never
        use it alone."""
        try:
            rels, gloss_embs = self._relation_gloss_cache()
            detected: List[str] = []

            # Channel 1 — lexical hit on relation content words.
            func_words = {"by", "of", "in", "to", "at", "on", "from", "is", "the", "with", "for"}
            prompt_stems = {self._stem(w.strip(".,!?'\"")) for w in prompt.lower().split()}
            for rel in rels:
                lexemes = {self._stem(w) for w in rel.split("_") if w not in func_words}
                if not lexemes:
                    continue
                # Single-word relations hit on that word; multi-word relations
                # require all content words (one common word alone is too loose).
                hit = (lexemes <= prompt_stems) if len(lexemes) > 1 else bool(lexemes & prompt_stems)
                if hit:
                    detected.append(rel)

            # Channel 2 — embedding paraphrase channel (true cosine — both
            # sides unit-norm natively since C17).
            if prompt_embedding is not None:
                p = prompt_embedding
                scored = []
                for rel, emb in zip(rels, gloss_embs):
                    if rel in detected:
                        continue
                    sim = sum(a * b for a, b in zip(p, emb))
                    if sim >= self.RELATION_SIM_FLOOR:
                        scored.append((sim, rel))
                scored.sort(reverse=True)
                detected.extend(rel for _, rel in scored[:self.RELATION_TOP_K])
            return detected
        except Exception as err:
            logger.error("relation_detection_failed", error=str(err))
            return []

    def _codex_scope_sets(self, scope: Optional[dict]):
        """A5: resolve a scope into (allowed_entity_ids, allowed_batch_ids).
        Both None means UNSCOPED — search the whole graph (auto).

        The scope is resolved down to a **set of batch_ids**, which is the
        forward-compatible primitive for the C6 scoping rework: today only a
        single `conversation_id` is populated (project scope), but any future
        scoping — several ticked conversations (cross-chat), a session_id, an
        @-mentioned conversation/turn — is just a different way of computing
        this same batch set, and the traversal filters downstream never change.
        C6 will populate one of:
          scope["batch_ids"]         — pre-resolved (session / @-mention)
          scope["conversation_ids"]  — several conversations (cross-chat)
          scope["conversation_id"]   — one conversation (today's project scope)
          scope["isolated"] = True   — incognito / none: empty set, matches nothing
        """
        if not scope:
            return None, None
        if scope.get("isolated"):                 # C6 'none' = true incognito
            return set(), set()
        try:
            batch_ids = set()
            if scope.get("batch_ids"):
                batch_ids = {b if not isinstance(b, str) else b for b in scope["batch_ids"]}
            else:
                conv_ids = scope.get("conversation_ids")
                if not conv_ids and scope.get("conversation_id"):
                    conv_ids = [scope["conversation_id"]]
                if conv_ids:
                    rows = self.db.execute(
                        text("SELECT DISTINCT batch_id FROM episodic_memory "
                             "WHERE conversation_id = ANY(:cids)"),
                        {"cids": list(conv_ids)}
                    ).fetchall()
                    batch_ids = {row.batch_id for row in rows}
            # E1b (D11): the code-graph allowance — an attached project adds
            # its derived entities + the deterministic static-edge batch to
            # the allowed sets, and a project scope NEVER falls back to
            # unscoped (a fresh project conversation with zero turns still
            # reads only its own graph).
            pid = self._scope_project_id
            if pid is None and not batch_ids:
                return None, None
            entity_ids = set()
            if batch_ids:
                event_rows = self.db.execute(
                    text("SELECT DISTINCT entity_id FROM codex_events "
                         "WHERE batch_source = ANY(:bids)"),
                    {"bids": list(batch_ids)}
                ).fetchall()
                entity_ids = {row.entity_id for row in event_rows}
            if pid is not None:
                from src.coding.code_graph import code_graph_batch_id
                derived = self.db.execute(
                    text("SELECT id FROM codex_entities "
                         "WHERE project_id = :pid AND source != 'conversation'"),
                    {"pid": pid}).fetchall()
                entity_ids |= {row.id for row in derived}
                batch_ids.add(code_graph_batch_id(pid))
            if self._denied_batch_ids:
                batch_ids -= self._denied_batch_ids
                entity_ids -= self._denied_entity_ids
            return entity_ids, batch_ids
        except Exception:
            self.db.rollback()
            return None, None

    def _resolve_exclusion_sets(self, scope: Optional[dict]) -> None:
        """C6: compute the codex/procedural DENY sets once per request.

        An allowed-set cannot express exclusion under `auto`, where the graph
        is deliberately unscoped (allowed = None = everything): "everything
        except X" has no allow-list form short of enumerating the store. So
        exclusion is carried as its own deny set and subtracted at each
        admission point, which also keeps the `auto` case — the common one —
        working without a scope.

        An ENTITY is denied only when every codex_event that evidences it
        comes from an excluded conversation. One first extracted in the
        abandoned project but discussed in five live ones is not that
        project's property, and must survive.

        C12: the deny set reads `exclude_knowledge_conversation_ids` WHEN
        PRESENT, else falls back to `exclude_conversation_ids` (so every C6
        path is byte-identical). The two diverge only for documents: a
        document whose knowledge has been promoted (a second conversation
        enabled it — D4) stays excluded from the TEXT everywhere it is
        switched off, while the facts extracted from it remain in the graph.
        Text visibility and knowledge visibility are different questions."""
        self._denied_batch_ids = set()
        self._denied_entity_ids = set()
        scope = scope or {}
        excluded = scope.get("exclude_knowledge_conversation_ids")
        if excluded is None:
            excluded = scope.get("exclude_conversation_ids")
        if not excluded:
            return
        try:
            rows = self.db.execute(
                text("SELECT DISTINCT batch_id FROM episodic_memory "
                     "WHERE conversation_id = ANY(:cids)"),
                {"cids": [str(c) for c in excluded]}).fetchall()
            self._denied_batch_ids = {r.batch_id for r in rows if r.batch_id}
            if not self._denied_batch_ids:
                return
            rows = self.db.execute(
                # NULL batch_source counts as evidence from OUTSIDE the
                # excluded set: unproven provenance keeps an entity visible
                # rather than hiding memory the user never excluded.
                text("SELECT entity_id FROM codex_events "
                     "GROUP BY entity_id "
                     "HAVING bool_and(batch_source IS NOT NULL "
                     "                AND batch_source = ANY(:bids))"),
                {"bids": list(self._denied_batch_ids)}).fetchall()
            self._denied_entity_ids = {r.entity_id for r in rows}
        except Exception:
            self.db.rollback()
            self._denied_batch_ids = set()
            self._denied_entity_ids = set()

    def _match_entities_exact(self, entity_strings: List[str]) -> List:
        """Entity resolution by exact canonical name / alias only (no vectors).
        Production fallback stage and the ablation `fuzzy_match=False` path."""
        from sqlalchemy import or_
        matched, seen = [], set()
        for candidate_str in entity_strings:
            norm = candidate_str.lower().strip()
            ent = self.db.query(CodexEntity).filter(
                or_(CodexEntity.canonical_name == norm, CodexEntity.aliases.any(norm)),
                *self._entity_source_filters(),
            ).first()
            if ent and ent.id not in seen:
                matched.append(ent)
                seen.add(ent.id)
        return matched

    def _match_entities_by_payload(self, entity_strings: List[str]) -> List:
        """A4 descriptor fallback: when name/alias/vector matching fails, look
        for the *descriptor* inside entity payloads — 'main fortress' matches
        the entity whose context_payload mentions 'fortress'. Closes part of
        the semantic-vs-lexical gap without a schema change."""
        stop = {"main", "this", "that", "what", "where", "when", "primary", "the"}
        words = {w.lower() for s in entity_strings for w in s.split()
                 if len(w) >= 4 and w.lower() not in stop}
        if not words:
            return []
        try:
            scored = {}
            for w in words:
                rows = self.db.query(CodexEntity).filter(
                    CodexEntity.context_payload.ilike(f"%{w}%"),
                    *self._entity_source_filters(),
                ).limit(20).all()
                for ent in rows:
                    scored[ent.id] = (scored.get(ent.id, (0, ent))[0] + 1, ent)
            ranked = sorted(scored.values(), key=lambda t: t[0], reverse=True)
            return [ent for hits, ent in ranked[:2] if hits >= 1]
        except Exception:
            self.db.rollback()
            return []

    def _edge_valid_filters(self):
        """T3 (D4): the bi-temporal valid_at(T) read the schema was built for.
        as_of/range: an edge counts if established by T = window end and not
        yet expired at T ("state as of then" = everything established *by*
        then). Legacy NULL valid_from = -infinity (passes). current keeps the
        exact old predicate (valid_until IS NULL ≡ valid_at(now), since
        expiries are stamped at write time). Used by every codex read —
        traversal (both directions), relation facts, enumeration."""
        ts = getattr(self, "_active_timescope", CURRENT)
        if ts.mode in ("as_of", "range") and ts.t1:
            return [or_(CodexEdge.valid_from.is_(None), CodexEdge.valid_from <= ts.t1),
                    or_(CodexEdge.valid_until.is_(None), CodexEdge.valid_until > ts.t1)]
        return [CodexEdge.valid_until.is_(None)]

    def _fact_line(self, src, edge, tgt) -> str:
        """Render one codex fact. T1: dated with the edge's valid_from at month
        precision (day precision would imply false exactness); legacy NULL
        valid_from renders undated. Negated edges render as NOT (the note
        renderer already did this; fact lines previously lied by omission)."""
        rel = f"NOT {edge.relation}" if getattr(edge, "negated", False) else edge.relation
        vf = getattr(edge, "valid_from", None)
        since = f" (since {vf.strftime('%Y-%m')})" if vf else ""
        return f"[Fact: {src.canonical_name} --{rel}--> {tgt.canonical_name}{since}]"

    def _relation_facts(self, matched, relations: List[str], allowed_batch_ids):
        """A4: surface explicit edge facts where a matched entity participates
        in a detected relation (either direction). The entity∩relation joint
        hit is the precision anchor; these edges also join the reinforcement
        anchors because they directly answered the query."""
        if not matched or not relations:
            return [], []
        matched_ids = [e.id for e in matched]
        q = self.db.query(CodexEdge).filter(
            *self._edge_valid_filters(),
            CodexEdge.relation.in_(relations),
            ((CodexEdge.source_id.in_(matched_ids)) | (CodexEdge.target_id.in_(matched_ids)))
        )
        if allowed_batch_ids is not None:
            q = q.filter(CodexEdge.source_batch.in_(allowed_batch_ids))
        if self._denied_batch_ids:   # C6 exclusion
            q = q.filter(CodexEdge.source_batch.notin_(self._denied_batch_ids))
        lines, fact_edges = [], []
        for edge in q.order_by(CodexEdge.strength.desc()).limit(10).all():
            if self._edge_trust(edge) < self.CODEX_DIRECT_TRUST_FLOOR:
                continue
            src = self.db.query(CodexEntity).get(edge.source_id)
            tgt = self.db.query(CodexEntity).get(edge.target_id)
            if src and tgt:
                lines.append(self._fact_line(src, edge, tgt))
                fact_edges.append(edge)
        return lines, fact_edges

    def _codex_enumeration(self, prompt: str, relations: List[str],
                           allowed_entity_ids, allowed_batch_ids) -> List[ContextFragment]:
        """A4: re-homed MERA. Entity-less category/enumeration queries ('list
        all the characters') answered from the graph itself. Joint gate:
        an explicit enumeration cue AND a grounded signal (a tag matching a
        prompt token, or a detected relation) — no LLM, no loose triggers."""
        pl = prompt.lower()
        if not any(cue in pl for cue in self._ENUM_CUES):
            return []
        tokens = {w.strip(".,!?'s\"") for w in pl.split()}
        candidate_tags = {t for w in tokens if len(w) >= 4 for t in (w, w.rstrip("s"))}
        # A10: enumeration also emits per-entity fragments (+ one facts fragment)
        # so each competes on its own in fusion/budget rather than as one blob.
        fragments: List[ContextFragment] = []
        seen_entities = set()
        try:
            # (a) tag-matched entities: "characters" → tag 'character'
            if candidate_tags:
                from sqlalchemy import or_
                q = self.db.query(CodexEntity).filter(
                    or_(*[CodexEntity.tags.any(t) for t in candidate_tags]),
                    *self._entity_source_filters())
                for ent in q.limit(self.ENUM_ENTITY_LIMIT).all():
                    if allowed_entity_ids is not None and ent.id not in allowed_entity_ids:
                        continue
                    if ent.id in self._denied_entity_ids:   # C6 exclusion
                        continue
                    if ent.id not in seen_entities and ent.context_payload:
                        seen_entities.add(ent.id)
                        t = f"[Entity: {ent.canonical_name}]\n{ent.context_payload}"
                        fragments.append(ContextFragment(text=t, source_type="codex", score=1.0,
                                                         token_count=count_tokens(t)))
            # (b) relation-driven facts: "who inspired ..." → inspired_by edges,
            #     grouped into a single facts fragment (they're a list answer).
            fact_lines = []
            if relations:
                q = self.db.query(CodexEdge).filter(
                    *self._edge_valid_filters(),
                    CodexEdge.relation.in_(relations))
                if allowed_batch_ids is not None:
                    q = q.filter(CodexEdge.source_batch.in_(allowed_batch_ids))
                if self._denied_batch_ids:   # C6 exclusion
                    q = q.filter(CodexEdge.source_batch.notin_(self._denied_batch_ids))
                for edge in q.order_by(CodexEdge.strength.desc()).limit(self.ENUM_EDGE_LIMIT).all():
                    if self._edge_trust(edge) < self.CODEX_DIRECT_TRUST_FLOOR:
                        continue
                    src = self.db.query(CodexEntity).get(edge.source_id)
                    tgt = self.db.query(CodexEntity).get(edge.target_id)
                    if src and tgt:
                        fact_lines.append(self._fact_line(src, edge, tgt))
            if fact_lines:
                t = "\n".join(fact_lines)
                fragments.append(ContextFragment(text=t, source_type="codex", score=1.0,
                                                 token_count=count_tokens(t)))
            if fragments:
                logger.info("codex_enumeration", entities=len(seen_entities),
                            facts=len(fact_lines), relations=relations)
            return fragments
        except Exception as err:
            logger.error("codex_enumeration_failed", error=str(err))
            self.db.rollback()
            return []

    def _expansion_terms(self) -> List[str]:
        """A4 grounded query expansion (the sane replacement for HyDE): expand
        the BM25 search prompt with the *canonical names and aliases* of the
        entities the prompt actually matched — 'citadel' pulls in 'the obsidian
        citadel' so lexical search hits turns using the full name. Grounded
        only: nothing is generated, so nothing can be hallucinated."""
        terms, seen = [], set()
        for ent in self._last_matched_entities:
            for term in [ent.canonical_name] + list(ent.aliases or []):
                t = (term or "").strip().lower()
                if t and t not in seen:
                    seen.add(t)
                    terms.append(t)
                if len(terms) >= self.EXPANSION_MAX_TERMS:
                    return terms
        return terms

    # ------------------------------------------------------------------
    # Codex graph traversal (conversation‑scoped, NER‑powered)
    # ------------------------------------------------------------------
    def _codex_graph(self, classification, scope: Optional[dict] = None,
                     prompt_embedding=None) -> List[ContextFragment]:
        prompt = classification.prompt
        self._last_matched_entities = []
        entity_strings = extract_entities(prompt, self.embedder)

        matched = []
        if entity_strings:
            matched = (self._match_entities_by_similarity(entity_strings)
                       if self.use_fuzzy_match
                       else self._match_entities_exact(entity_strings))
            if not matched:
                # A4 descriptor fallback: 'main fortress' → payload mentions 'fortress'
                matched = self._match_entities_by_payload(entity_strings)

        # A4: relations relevant to the prompt — lexical + embedding channels
        # (joint signal only).
        detected_relations = self._detect_relations(prompt, prompt_embedding)
        # A5: project-scope sets (both None when unscoped).
        allowed_entity_ids, allowed_batch_ids = self._codex_scope_sets(scope)

        if not matched:
            # A4: re-homed MERA — entity-less enumeration ("list all the characters").
            if self.enable_enumeration:
                return self._codex_enumeration(prompt, detected_relations,
                                               allowed_entity_ids, allowed_batch_ids)
            return []

        self._last_matched_entities = matched   # grounded query expansion (BM25)

        try:
            # A10: emit ONE fragment per anchor entity (its full note + trust-gated
            # neighbor previews + its relation facts), scored by that anchor's own
            # edge trust. Each anchor gets its OWN visited set so a shared neighbor
            # appears (as a preview) in each anchor's self-contained fragment; the
            # OTHER anchors are excluded from being absorbed as neighbors, so every
            # matched entity keeps its own fragment (A7.2 bidirectional traversal
            # would otherwise merge connected anchors via a shared visited set).
            fragments: List[ContextFragment] = []
            timeline_frags: List[ContextFragment] = []   # T4: own leg (RRF weight + budget lane)
            all_anchor_edges = []   # A3: reinforced across all anchors at the end
            any_relation_hit = False
            anchor_ids = {a.id for a in matched}
            ts = self._active_timescope
            timeline_cap = (settings.timeline_max_fragments_evolution
                            if ts.mode == "evolution"
                            else settings.timeline_max_fragments)
            for anchor in matched:
                if allowed_entity_ids is not None and anchor.id not in allowed_entity_ids:
                    continue
                if anchor.id in self._denied_entity_ids:   # C6 exclusion
                    continue
                local_texts, direct_edges = [], []
                self._traverse_graph(anchor, 0, self.CODEX_MAX_DEPTH, set(),
                                     local_texts, direct_edges,
                                     allowed_entity_ids, allowed_batch_ids,
                                     exclude_ids=anchor_ids - {anchor.id})
                if not local_texts:
                    continue
                # Per-anchor score from THIS anchor's direct-edge trust (A3).
                mean_trust = (sum(self._edge_trust(e) for e in direct_edges) / len(direct_edges)
                              if direct_edges else 0.0)
                score = 1.0 + min(0.5, 0.25 * mean_trust)
                # A4: relation-aware facts for this anchor → boost just this fragment.
                if detected_relations:
                    fact_lines, fact_edges = self._relation_facts(
                        [anchor], detected_relations, allowed_batch_ids)
                    if fact_lines:
                        local_texts.extend(fact_lines)
                        direct_edges.extend(fact_edges)
                        score += self.RELATION_OVERLAP_BOOST
                        any_relation_hit = True
                text = "\n\n".join(local_texts)
                fragments.append(ContextFragment(
                    text=text, source_type="codex", score=score,
                    token_count=count_tokens(text)))
                all_anchor_edges.extend(direct_edges)

                # T4: attach the anchor's evolution timeline whenever it
                # carries real supersession history (D-U2: provided in any
                # mode — including current, the saga case — never forced;
                # the model decides how much to narrate). 0.9× the anchor's
                # own score; capped per D7 so a life story can't eat the
                # window.
                if len(timeline_frags) < timeline_cap and history_exists(self.db, anchor.id):
                    tl = build_entity_timeline(
                        self.db, anchor, allowed_batch_ids,
                        t0=ts.t0, t1=ts.t1,
                        max_transitions=settings.timeline_max_transitions)
                    if tl:
                        timeline_frags.append(ContextFragment(
                            text=tl, source_type="timeline", score=0.9 * score,
                            token_count=count_tokens(tl)))

            if any_relation_hit:
                logger.info("codex_relation_overlap", relations=detected_relations,
                            fragments=len(fragments))
            # A3: retrieval-reinforcement across every anchor's edges.
            self._reinforce_codex_edges(all_anchor_edges)
            return fragments + timeline_frags
        except Exception as err:
            logger.error("codex_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    def _edge_trust(self, edge) -> float:
        """Effective trust = strength (A3 usage dynamics) x extraction_confidence
        (A3 grounding/corroboration) x recency (A11). Legacy edges with NULL
        confidence count as fully trusted (they predate grounding)."""
        conf = edge.extraction_confidence if edge.extraction_confidence is not None else 1.0
        base = (edge.strength or 0.0) * conf
        # A11: reward recently-asserted facts; old edges tend to 1.0 (no penalty).
        # T3 (D9): under a window, "recent" means near the window end — the
        # multiplier stays >= 1.0, so A11's never-penalize-age invariant
        # survives the re-anchoring.
        vf = getattr(edge, "valid_from", None)
        if vf is not None:
            try:
                ts = getattr(self, "_active_timescope", CURRENT)
                if ts.mode in ("as_of", "range") and ts.t1:
                    age_days = abs((ts.t1 - vf).total_seconds()) / 86400.0
                else:
                    age_days = max(0.0, (datetime.now(timezone.utc) - vf).total_seconds() / 86400.0)
                base *= 1.0 + self.CODEX_RECENCY_BOOST * math.exp(-age_days / self.CODEX_RECENCY_TAU_DAYS)
            except Exception:
                pass
        return base

    def _render_codex_entity(self, entity, depth, out_edges, in_edges,
                             allowed_batch_ids, context_texts):
        """A7.2 depth-graded rendering (Obsidian reading model): the anchor
        (depth 0) injects its FULL rich note; deeper neighbors inject a compact
        one-line preview (name + type + a snippet), so navigation is rich but
        token-efficient."""
        # E1b: derived entities (code graph / project facts) render their full
        # pointer payload even under scope — it's built from the project
        # itself, so the "leaks other conversations" rationale doesn't apply.
        derived = getattr(entity, "source", None) not in (None, "conversation")
        if depth == 0:
            if allowed_batch_ids is None or derived:
                # unscoped: the stored rich note (description + props + links + backlinks).
                if entity.context_payload:
                    context_texts.append(f"[Entity: {entity.canonical_name}]\n{entity.context_payload}")
            else:
                # A5 scoped: rebuild from this conversation's edges only (both
                # directions), no global description — it would leak other convos.
                lines = []
                for e in out_edges[:10]:
                    t = self.db.query(CodexEntity).get(e.target_id)
                    if t:
                        rel = f"NOT {e.relation}" if getattr(e, "negated", False) else e.relation
                        lines.append(f"{rel} → {t.canonical_name}")
                for e in in_edges[:10]:
                    s = self.db.query(CodexEntity).get(e.source_id)
                    if s:
                        rel = f"NOT {e.relation}" if getattr(e, "negated", False) else e.relation
                        lines.append(f"{s.canonical_name} --{rel}→")
                if lines:
                    context_texts.append(f"[Entity: {entity.canonical_name}]\n" + "; ".join(lines))
        else:
            etype = getattr(entity, "entity_type", None) or "entity"
            preview = f"[{entity.canonical_name} ({etype})]"
            if allowed_batch_ids is None:  # description is global → only show unscoped
                desc = (entity.description or "").strip()
                if desc:
                    preview += ": " + " ".join(desc.split()[:20])
            context_texts.append(preview)

    def _traverse_graph(self, entity, depth, max_depth, visited, context_texts, anchor_edges=None,
                        allowed_entity_ids=None, allowed_batch_ids=None, exclude_ids=None):
        if entity.id in visited or depth > max_depth:
            return
        visited.add(entity.id)

        # A7.2: fetch BOTH directions — outgoing links and incoming backlinks —
        # so the graph is navigable both ways (Obsidian backlinks). T3: the
        # validity predicate is valid_at(T) under a window (D4); evolution
        # mode deliberately navigates the CURRENT graph (D5 — history is the
        # T4 timeline builder's job, dead-edge walking makes incoherent
        # neighborhoods).
        valid = self._edge_valid_filters()
        out_edges = self.db.query(CodexEdge).filter(
            CodexEdge.source_id == entity.id, *valid
        ).order_by(CodexEdge.strength.desc()).all()
        in_edges = self.db.query(CodexEdge).filter(
            CodexEdge.target_id == entity.id, *valid
        ).order_by(CodexEdge.strength.desc()).all()
        # A5: scope filter (both directions) — no cross-conversation leakage.
        if allowed_batch_ids is not None:
            out_edges = [e for e in out_edges if e.source_batch in allowed_batch_ids]
            in_edges = [e for e in in_edges if e.source_batch in allowed_batch_ids]
        # C6: and the negated scope — an excluded conversation's edges never
        # traverse, in any mode (the allowed set is None under `auto`).
        if self._denied_batch_ids:
            out_edges = [e for e in out_edges if e.source_batch not in self._denied_batch_ids]
            in_edges = [e for e in in_edges if e.source_batch not in self._denied_batch_ids]

        self._render_codex_entity(entity, depth, out_edges, in_edges,
                                  allowed_batch_ids, context_texts)

        # A7.2: traverse both directions (into the target of outgoing edges and
        # the source of incoming ones), trust-gated and scope-bounded as before.
        for edge, other_id in ([(e, e.target_id) for e in out_edges] +
                               [(e, e.source_id) for e in in_edges]):
            # A8: a negated edge ("X does NOT use Y") is a stored fact, not a
            # navigable link — it's rendered in the note (Negations section) but
            # we don't walk into it as if the relationship held.
            if getattr(edge, "negated", False):
                continue
            trust = self._edge_trust(edge)
            if depth == 0:
                # A3 dynamic threshold: a matched entity's edge expands (and is
                # reinforced as a query anchor) only above the direct floor.
                if trust < self.CODEX_DIRECT_TRUST_FLOOR:
                    continue
                if anchor_edges is not None:
                    anchor_edges.append(edge)
            # A3: trust-gate deep hops — weak/decayed edges don't expand the frontier.
            elif trust < self.CODEX_DEEP_STRENGTH_FLOOR:
                continue
            # A5: traversal never leaves the conversation's entity set under scope.
            if allowed_entity_ids is not None and other_id not in allowed_entity_ids:
                continue
            if other_id in self._denied_entity_ids:   # C6 exclusion
                continue
            # A10/A7.2: don't absorb another matched anchor as a neighbor — it has
            # its own fragment.
            if exclude_ids and other_id in exclude_ids:
                continue
            other = self.db.query(CodexEntity).get(other_id)
            # E1b (D3): derived entities stay invisible outside their project
            # even when reachable through a conversation edge.
            if other and self._entity_visible(other):
                self._traverse_graph(other, depth + 1, max_depth, visited,
                                     context_texts, anchor_edges,
                                     allowed_entity_ids, allowed_batch_ids, exclude_ids)

    def _reinforce_codex_edges(self, edges):
        """A3 retrieval-reinforcement (episodic analog): the anchor edges of the
        matched query entities gain a little strength each time they're surfaced,
        so repeatedly-useful facts self-promote through use — balanced by the
        codex_decay worker (which decays ALL live edges, so this loop is closed).
        A reinforced pending edge is promoted to active once it crosses
        CODEX_PROMOTE_STRENGTH — but only if its extraction_confidence clears
        CODEX_PROMOTE_MIN_CONFIDENCE, so a low-trust extraction cannot promote
        through retrieval popularity alone; it needs corroboration first.
        Scoped to anchor edges only to avoid diluting the signal across the
        whole traversed neighborhood. Write-on-read, like _strengthen_retrieved
        does for episodic turns."""
        if not edges:
            return
        try:
            seen = set()
            for e in edges:
                if e.id in seen:
                    continue
                seen.add(e.id)
                e.strength = min((e.strength or 0.0) + self.CODEX_REINFORCE_INCREMENT,
                                 self.CODEX_STRENGTH_CAP)
                conf = e.extraction_confidence if e.extraction_confidence is not None else 1.0
                if (e.confidence == "pending"
                        and e.strength >= self.CODEX_PROMOTE_STRENGTH
                        and conf >= self.CODEX_PROMOTE_MIN_CONFIDENCE):
                    e.confidence = "active"
            self.db.commit()
        except Exception:
            self.db.rollback()

    # ------------------------------------------------------------------
    # Procedural lookup (scoped + trigger‑condition evaluation)
    # ------------------------------------------------------------------
    def _procedural_lookup(self, prompt_embedding, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
        # C9 (D4): the leg always runs — the old 3-intent whitelist
        # ({Strategic_Planning, Generation, Open_Exploration}) is why nobody
        # ever *felt* procedural memory. Precision comes from the three
        # signals that remain: embedding rank (LIMIT 5), the confidence floor
        # in the SQL, and _procedural_trigger_match.
        # G29 (4th leak site) / C6: this hand-rolled copy read
        # scope["conversation_id"] ONLY, so under project or manual scope —
        # where the ids live in scope["conversation_ids"] and there is no
        # single conversation_id — it resolved no batch set at all and the leg
        # ran against every pattern in the store. It never matched the grep
        # that found the other three sites because it resolves a batch set
        # instead of emitting SQL. Now the shared A5 resolver, which handles
        # all four scope forms (batch_ids / conversation_ids / conversation_id
        # / isolated) plus the project code-graph allowance.
        # C6: patterns supported ONLY by excluded conversations drop out.
        # Partial overlap does not — a habit seen in ten conversations is
        # still evidenced by the nine you did not exclude. Resolved here too
        # because this leg is also a direct entry point (tests, direct calls);
        # idempotent and query-free when reached via retrieve().
        self._resolve_exclusion_sets(scope)
        _, allowed_batch_ids = self._codex_scope_sets(scope)
        excluded_batch_ids = self._denied_batch_ids

        # T3: under a window, a pattern is relevant only if its observation
        # span overlaps it (a habit first seen after the window didn't exist
        # then).
        ts = self._active_timescope
        time_filter, params = "", {
            "prompt_embedding": prompt_embedding,
            "min_conf": settings.procedural_min_conf,
        }
        if ts.mode in ("as_of", "range") and ts.t0 and ts.t1:
            time_filter = "AND first_observed <= :ts_t1 AND last_observed >= :ts_t0"
            params["ts_t0"], params["ts_t1"] = ts.t0, ts.t1
        # C9: the PgVector bindparam is load-bearing — the plain-list bind
        # raised `vector <=> double precision[]` on every call, so this leg
        # silently returned [] (rollback + empty) since the psycopg3 move.
        # The intent whitelist hid the crash; killing the gate exposed it.
        query = text(f"""
            SELECT id, pattern_description,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM procedural_memory
            WHERE embedding IS NOT NULL AND is_active = true
              AND confidence_score >= :min_conf
            {time_filter}
            ORDER BY score DESC
            LIMIT 5
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        try:
            rows = self.db.execute(query, params).fetchall()
            fragments = []
            pid = self._scope_project_id
            for r in rows:
                pattern = self.db.query(ProceduralMemory).get(r.id)
                if not pattern:
                    continue
                # E1 (D1): project-scoped conventions are invisible outside
                # their project; inside it they pass regardless of batch scope
                # (they belong to the project, not to one conversation).
                in_project = pid is not None and pattern.project_id == pid
                if pattern.project_id is not None and not in_project:
                    continue
                if allowed_batch_ids is not None and not in_project:
                    if not any(bid in allowed_batch_ids for bid in (pattern.source_batch_ids or [])):
                        continue
                source_batches = pattern.source_batch_ids or []
                if (excluded_batch_ids and source_batches
                        and all(bid in excluded_batch_ids for bid in source_batches)):
                    continue
                if not self._procedural_trigger_match(pattern, classification):
                    continue
                fragments.append(ContextFragment(
                    text=pattern.pattern_description,
                    source_type="procedural",
                    score=r.score,
                    token_count=count_tokens(pattern.pattern_description)
                ))
            return fragments
        except Exception:
            self.db.rollback()
            return []

    def _procedural_trigger_match(self, pattern: ProceduralMemory, classification: ClassificationResult) -> bool:
        conditions = pattern.trigger_conditions or {}
        if not conditions:
            return True
        required_topics = set(conditions.get("topic_tags", []))
        required_intents = set(conditions.get("intent_tags", []))
        if required_topics and not required_topics.intersection(classification.topic_tags):
            return False
        if required_intents and not required_intents.intersection(classification.intent_tags):
            return False
        return True

    # ------------------------------------------------------------------
    # (C12: the RAG leg is GONE — see the entry. `_rag_lookup` read
    # `rag_chunks` with no scope filter at all, behind a gate that required
    # the prompt to contain one of five English nouns
    # (document/pdf/reference/manual/guide), and its only writer was a
    # watchdog script nothing started. Documents now enter as their own
    # conversations and are retrieved by the ordinary episodic/codex/cluster
    # legs, so there is no lexical trigger left to be style-dependent.)
    # ------------------------------------------------------------------

    def _batch_summary_lookup(self, prompt_embedding, conv_id: Optional[str] = None,
                              include_cross: bool = True) -> List[ContextFragment]:
        # T3 (D14): skipped under any non-current mode — a summary's created_at
        # is long after its content's period, which is underivable without a
        # turn-index→timestamp join; serving it under a window would mislead.
        if self._active_timescope.mode != "current":
            return []
        fragments: List[ContextFragment] = []
        # Half 1 (as built): this conversation's batch summaries.
        if conv_id:
            try:
                query = text("""
                    SELECT summary_text, created_at,
                           1 - (embedding <=> :prompt_embedding) as score
                    FROM batch_summaries
                    WHERE conversation_id = :conv_id
                      AND embedding IS NOT NULL
                    ORDER BY score DESC
                    LIMIT 3
                """).bindparams(bindparam("prompt_embedding", type_=PgVector))
                rows = self.db.execute(query, {"prompt_embedding": prompt_embedding, "conv_id": conv_id}).fetchall()
                # T1: summaries are written long after the turns they compress,
                # so they get a "[summary, <created>]" prefix, not a turn date.
                fragments += [ContextFragment(
                    text=(f"[summary, {r.created_at.strftime('%Y-%m-%d')}] " if r.created_at else "") + r.summary_text,
                    source_type="batch_summary",
                    score=r.score,
                    token_count=count_tokens(r.summary_text)
                ) for r in rows]
            except Exception:
                self.db.rollback()
        # Half 2 (C4 D3b): OTHER conversations' evolving whole-conversation
        # summaries — cross-conversation overview hits. The active
        # conversation's own summary is excluded (the assembler injects it —
        # double-inject trap) and private conversations never leave their
        # scope (memory_scope_type join). Skipped entirely under incognito
        # (a user-global read — G16 "read nothing").
        if not include_cross:
            return fragments
        try:
            query = text("""
                SELECT s.summary_text, s.updated_at,
                       1 - (s.embedding <=> :prompt_embedding) as score
                FROM conversation_summaries s
                JOIN conversations c ON c.id = s.conversation_id
                WHERE s.embedding IS NOT NULL
                  AND c.memory_scope_type != 'none'
                  AND (CAST(:conv_id AS uuid) IS NULL
                       OR s.conversation_id != CAST(:conv_id AS uuid))
                ORDER BY score DESC
                LIMIT 2
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = self.db.execute(query, {
                "prompt_embedding": prompt_embedding,
                "conv_id": conv_id,
            }).fetchall()
            fragments += [ContextFragment(
                text=(f"[conversation summary, {r.updated_at.strftime('%Y-%m-%d')}] "
                      if r.updated_at else "[conversation summary] ") + r.summary_text,
                source_type="batch_summary",
                score=r.score,
                token_count=count_tokens(r.summary_text)
            ) for r in rows]
        except Exception:
            self.db.rollback()
        return fragments
    # ------------------------------------------------------------------
    # T3: cold-storage leg + resurrection (D-U1) + honest emptiness
    # ------------------------------------------------------------------
    def _cold_lookup(self, prompt_keywords, conv_id: Optional[str] = None,
                     scope: Optional[dict] = None,
                     prompt_embedding=None) -> List[ContextFragment]:
        """Cold storage joins time-scoped queries ONLY (never normal ones, and
        never the wide net). Requires a window — the window is what bounds the
        search when there are no keywords (a pure "what was I thinking about
        in march" browse); evolution without a window skips cold (unbounded
        browse is noise). Scoping goes through the SAME _conv_scope_filter the
        live episodic legs use (G29 — a hand-rolled `if conv_id` copy here let
        project-scoped queries read the whole global cold store)."""
        ts = self._active_timescope
        if ts.mode not in ("as_of", "range", "evolution") or not ts.t0:
            return []
        t1 = ts.t1 or datetime.now(timezone.utc)

        # Up to 6 %kw% patterns from prompt keywords + grounded expansion
        # terms (codex leg runs first, so _last_matched_entities is set).
        terms, seen = [], set()
        for term in self._expansion_terms() + sorted(
                (k for k in prompt_keywords if len(k) >= 3),
                key=lambda k: (-len(k), k)):
            if term not in seen:
                seen.add(term)
                terms.append(term)
            if len(terms) >= 6:
                break
        pats = [f"%{t}%" for t in terms]

        conv_filter, conv_params, conv_scoped = self._conv_scope_filter(scope, conv_id)
        # C6: cold rows carry no cluster links — conversation exclusion only.
        excl_filter, excl_params = self._exclusion_filters(scope, id_column=None)
        privacy_filter = "" if conv_scoped else "AND is_private = FALSE"
        # C16: rank by MEANING when the archived row carries a vector, and fall
        # back to the keyword patterns only for rows that predate
        # `cold_storage.embedding`. The ILIKE-over-a-40-word-English-stoplist
        # form was the last purely lexical query left in retrieval — it could
        # not find a paraphrase, and it fired on any fragment that happened to
        # share vocabulary. Rows written from now on carry the same vector the
        # live turn was matched on, so a cold hit means the same thing a warm
        # one does.
        vector_rank = ""
        if prompt_embedding is not None:
            vector_rank = ("CASE WHEN embedding IS NULL THEN 1 "
                           "ELSE (embedding <=> :cold_probe) END ASC,")
        query = text(f"""
            SELECT id, conversation_id, batch_id, raw_text, summary_text,
                   topic_tags, timestamp, is_private, embedding
            FROM cold_storage
            WHERE timestamp >= :t0 AND timestamp < :t1
              {conv_filter}
              {excl_filter}
              {privacy_filter}
              AND (
                    embedding IS NOT NULL
                    OR :no_kw
                    OR raw_text ILIKE ANY(:pats)
                    OR summary_text ILIKE ANY(:pats)
              )
            ORDER BY {vector_rank} timestamp DESC
            LIMIT :cold_limit
        """)
        if prompt_embedding is not None:
            query = query.bindparams(bindparam("cold_probe", type_=PgVector))
        params = {"t0": ts.t0, "t1": t1, "no_kw": not pats, "pats": pats or ["%"],
                  "cold_limit": settings.timescope_cold_limit, **conv_params,
                  **excl_params}
        if prompt_embedding is not None:
            params["cold_probe"] = prompt_embedding
        try:
            rows = self.db.execute(query, params).fetchall()
        except Exception as err:
            logger.error("cold_lookup_failed", error=str(err))
            self.db.rollback()
            return []

        fragments = []
        for row in rows:
            body = _truncate_at_sentence(row.summary_text or row.raw_text, 300)
            stamp = row.timestamp.strftime("[%Y-%m-%d] ") if row.timestamp else ""
            text_ = stamp + body
            fragments.append(ContextFragment(
                text=text_, source_type="episodic", score=0.6,
                token_count=count_tokens(text_),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None,
            ))
            self._cold_hits[str(row.id)] = row
        if fragments:
            logger.info("cold_leg_hits", count=len(fragments), mode=ts.mode)
        return fragments

    def _resurrect_cold_hits(self, final: List[ContextFragment]):
        """D-U1 second chance: a cold memory that was actually *injected*
        (survived the budget — never mere candidacy) moves back into
        episodic_memory on probation: ORIGINAL timestamp (it's an old memory),
        decay just above the archive line — unengaged, normal decay re-archives
        it within days; engaged, write-on-read strengthening saves it. Never
        restored at full strength. Legacy rows (NULL conversation_id) are
        cite-only. Runs AFTER _strengthen_retrieved (the row isn't episodic
        yet during that pass), so probation starts exactly at 0.12."""
        if not self._cold_hits:
            return
        for f in final:
            row = self._cold_hits.get(f.source_batch_id) if f.source_batch_id else None
            if row is None:
                continue
            if row.conversation_id is None:
                logger.info("cold_cited_only", cold_id=str(row.id))
                continue
            try:
                emb = self.embedder.encode((row.summary_text or row.raw_text)[:2000],
                                           convert_to_tensor=False)
                emb = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                res = self.db.execute(text("""
                    INSERT INTO episodic_memory
                        (id, conversation_id, batch_id, timestamp, topic_tags,
                         intent_tags, context_reliance, raw_text, summary_text,
                         embedding, decay_score, access_count, is_archived,
                         is_private, inject_raw, idempotency_key)
                    VALUES (:id, :conv, :batch, :ts, :tags, :itags,
                            'Long_Term_Memory', :raw, :summary, :emb, :score,
                            1, FALSE, :priv, TRUE, :ikey)
                    ON CONFLICT (id) DO NOTHING
                """).bindparams(bindparam("emb", type_=PgVector)), {
                    "id": row.id, "conv": row.conversation_id,
                    "batch": row.batch_id or uuid.uuid4(), "ts": row.timestamp,
                    "tags": list(row.topic_tags or []), "itags": [],
                    "raw": row.raw_text, "summary": row.summary_text,
                    "emb": emb, "score": settings.timescope_probation_score,
                    "priv": row.is_private,
                    "ikey": f"cold-resurrect-{row.id}",
                })
                if res.rowcount == 0:
                    # id somehow still live in episodic — keep the cold row.
                    logger.info("cold_resurrect_conflict", cold_id=str(row.id))
                else:
                    self.db.execute(text("DELETE FROM cold_storage WHERE id = :id"),
                                    {"id": row.id})
                    logger.info("cold_resurrected", episodic_id=str(row.id),
                                decay_score=settings.timescope_probation_score)
                self.db.commit()
            except Exception as err:
                logger.error("cold_resurrect_failed", cold_id=str(row.id), error=str(err))
                self.db.rollback()

    def _append_empty_window_note(self, final, prompt_embedding, conv_id,
                                  scope=None):
        """T3 honest emptiness (§2.6): a windowed query with no episodic
        matches says so explicitly — naming the window and, when an unwindowed
        probe finds anything, the nearest eras — instead of silently widening
        (codex/timeline fragments may still be present and still answer).

        The nearest-era probe is scoped through _conv_scope_filter like every
        other episodic read (G29): it drops the window, never the scope, or a
        project-scoped question gets told about another project's eras."""
        ts = self._active_timescope
        if ts.mode not in ("as_of", "range") or not (ts.t0 and ts.t1):
            return final
        if any(f.source_type == "episodic" for f in final):
            return final
        note = (f"[Memory note] No stored memories match this query between "
                f"{ts.t0.date()} and {ts.t1.date()}.")
        try:
            conv_filter, conv_params, conv_scoped = self._conv_scope_filter(scope, conv_id)
            # C6: an excluded conversation must not be advertised as a
            # "closest match" either — the note is a read of the same store.
            excl_filter, excl_params = self._exclusion_filters(scope)
            privacy_filter = "" if conv_scoped else "AND is_private = FALSE"
            probe = text(f"""
                SELECT timestamp FROM episodic_memory
                WHERE embedding IS NOT NULL {conv_filter} {excl_filter} {privacy_filter}
                ORDER BY embedding <=> :emb
                LIMIT 3
            """).bindparams(bindparam("emb", type_=PgVector))
            params = {"emb": prompt_embedding, **conv_params, **excl_params}
            eras = sorted({r.timestamp.strftime("%Y-%m")
                           for r in self.db.execute(probe, params).fetchall()
                           if r.timestamp})
            if eras:
                note += (" Closest matches outside that window are from "
                         + ", ".join(eras) + ".")
        except Exception:
            self.db.rollback()
        logger.info("timescope_empty_window", t0=str(ts.t0.date()), t1=str(ts.t1.date()))
        final.append(ContextFragment(
            text=note, source_type="episodic", score=5.0,
            token_count=count_tokens(note)))
        return final

    # ------------------------------------------------------------------
    # RRF fusion, diversification, dedup, token budget
    # ------------------------------------------------------------------
    def _apply_rrf(self, legs: Dict[str, List[ContextFragment]], alpha_map: Dict[str, float] = None, k: int = 60) -> List[ContextFragment]:
        from dataclasses import replace
        if alpha_map is None:
            alpha_map = {}

        rrf_scores: Dict[str, float] = {}
        fragment_registry: Dict[str, ContextFragment] = {}

        # C16: the leg name is right here and was being thrown away. Stamp it —
        # first leg to produce the fragment wins, which is also the
        # highest-ranked contribution for that fragment within that leg. This
        # is the only place in the pipeline that knows which of the eight
        # mechanisms actually found a given piece of memory.
        frag_leg: Dict[str, str] = {}

        for leg_name, fragments in legs.items():
            weight = alpha_map.get(leg_name, 1.0)
            fragments.sort(key=lambda x: x.score, reverse=True)
            for rank, frag in enumerate(fragments, start=1):
                frag_hash = hashlib.sha256(frag.text.encode('utf-8')).hexdigest()
                if frag_hash not in fragment_registry:
                    fragment_registry[frag_hash] = frag
                    frag_leg[frag_hash] = leg_name
                rrf_scores[frag_hash] = rrf_scores.get(frag_hash, 0.0) + (weight / (k + rank))

        fused = []
        for frag_hash, score in rrf_scores.items():
            original = fragment_registry[frag_hash]
            new_frag = replace(original, score=score,
                               leg=original.leg or frag_leg.get(frag_hash))
            fused.append(new_frag)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def _collapse_provenance(self, fragments):
        """C16: collapse fragments that are the SAME MEMORY, by identity.

        A chunk and its parent turn carry the same `source_batch_id`, and today
        that is deduped only *within* the vector leg — so a BM25 hit on the
        parent plus a vector hit on one of its chunks both survive and the same
        text is injected twice. `_deduplicate` cannot catch it: it hashes the
        exact string, and a chunk is a substring, not a copy.

        This runs before any similarity math because it is an ID join — a
        resolver, not a lexicon — so it is exact, free, and cannot be fooled by
        how anything is written. It is also the only arm in C16 that provably
        cannot lose information: what it drops is, by construction, already
        present.
        """
        if not getattr(settings, "retrieval_collapse_enabled", True):
            return fragments
        cap = int(getattr(settings, "retrieval_max_frags_per_turn", 2))
        seen: Dict[str, int] = {}
        out, dropped = [], 0
        for f in fragments:            # already score-ordered — best survive
            key = f.source_batch_id
            if not key:
                out.append(f)
                continue
            n = seen.get(key, 0)
            if n >= cap:
                dropped += 1
                continue
            seen[key] = n + 1
            out.append(f)
        if dropped:
            logger.info("provenance_collapsed", dropped=dropped,
                        kept=len(out), cap=cap)
        return out

    def _fragment_vectors(self, fragments):
        """Vectors for coverage, fetched — never encoded at request time.

        Encoding 100 fragments costs ~380 ms even on the GPU, which is not
        something to spend on the pre-flight path. Every episodic fragment
        already has a stored vector reachable by `source_batch_id`; anything
        else returns None and is admitted without competing (see
        `select_by_coverage`), so a leg with no stored embedding is
        under-filtered rather than silently dropped.
        """
        ids = [f.source_batch_id for f in fragments if f.source_batch_id]
        if not ids:
            return {}
        try:
            rows = self.db.execute(text(
                "SELECT id, embedding FROM episodic_memory "
                "WHERE id = ANY(:ids) AND embedding IS NOT NULL"
            ), {"ids": [str(i) for i in ids]}).fetchall()
        except Exception as exc:
            self.db.rollback()
            logger.warning("coverage_vector_fetch_failed", error=str(exc))
            return {}
        out = {}
        for r in rows:
            vec = r.embedding
            if isinstance(vec, str):          # pgvector text form
                try:
                    vec = [float(x) for x in vec.strip("[]").split(",")]
                except ValueError:
                    continue
            out[str(r.id)] = vec
        return out

    def _apply_coverage(self, fragments, prompt_embedding):
        """The "is this enough?" decision. Off by default until measured."""
        if not getattr(settings, "retrieval_coverage_enabled", False):
            return fragments
        if not fragments or prompt_embedding is None:
            return fragments

        scores = [float(f.score) for f in fragments]
        if getattr(settings, "retrieval_set_floor_enabled", False):
            if not coverage.set_floor_passes(
                    scores, float(settings.retrieval_set_floor)):
                # Honest emptiness: nothing here is worth the tokens. Better a
                # short prompt than the best of a bad set injected as evidence.
                logger.info("coverage_set_floor_rejected_all",
                            best=max(scores), floor=settings.retrieval_set_floor)
                return []

        vectors = self._fragment_vectors(fragments)
        candidates = [
            (f, vectors.get(str(f.source_batch_id)) if f.source_batch_id else None,
             float(f.score))
            for f in fragments
        ]
        selected, record = coverage.select_by_coverage(
            prompt_embedding, candidates,
            alpha=float(settings.coverage_alpha),
            min_gain=float(settings.coverage_min_gain),
            min_keep=int(settings.coverage_min_keep),
            max_keep=int(settings.coverage_max_keep),
            knee_enabled=bool(settings.coverage_knee_enabled),
            knee_min_prominence=float(settings.coverage_knee_min_prominence),
        )
        # The counterfactual: what fill-to-cap WOULD have spent, recorded so a
        # later experiment can price the decision without re-running retrieval.
        self._coverage_record = {
            **record,
            "tokens_if_no_stop": sum(f.token_count for f in fragments),
            "tokens_selected": sum(f.token_count for f in selected),
            "legs_in": sorted({f.leg or f.source_type for f in fragments}),
            "legs_out": sorted({f.leg or f.source_type for f in selected}),
        }
        logger.info("coverage_applied", **{
            k: v for k, v in self._coverage_record.items() if k != "gains"})
        return selected

    def _session_diversify(self, fragments, current_id, max_per_conversation=3):
        counts: Dict[str, int] = {}
        result = []
        for f in fragments:
            cid = f.conversation_id
            if not cid:
                result.append(f)
            elif cid == current_id:
                result.append(f)
            else:
                counts[cid] = counts.get(cid, 0) + 1
                if counts[cid] <= max_per_conversation:
                    result.append(f)
        return result

    def _deduplicate(self, fragments):
        seen = set()
        unique = []
        for f in fragments:
            h = hashlib.sha256(f.text.encode('utf-8')).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(f)
        return unique

    def _enforce_token_budget(self, fragments, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_retrieval_tokens
        from collections import deque

        # Phase 1 – leg-diversity guarantee: each leg's single best fragment first.
        best_per_leg = {}
        for f in fragments:
            leg = f.source_type
            if leg not in best_per_leg or f.score > best_per_leg[leg].score:
                best_per_leg[leg] = f
        guaranteed = sorted(best_per_leg.values(), key=lambda x: x.score, reverse=True)

        def _degraded(f, budget_left):
            """C1/C3 degrade-before-drop chain: swap a too-big fragment for
            its trusted summary, or failing that its one-line abstract,
            instead of losing it entirely. Returns the first level that fits."""
            from dataclasses import replace as dc_replace
            for alt in (f.degrade_text, f.abstract_text):
                if not alt or alt == f.text:
                    continue
                tokens = count_tokens(alt)
                if tokens <= budget_left:
                    return dc_replace(f, text=alt, token_count=tokens,
                                      degrade_text=None, abstract_text=None)
            return None

        total, result, used = 0, [], set()
        for f in guaranteed:
            if total + f.token_count <= max_tokens:
                result.append(f); total += f.token_count; used.add(id(f))
            else:
                d = _degraded(f, max_tokens - total)
                if d:
                    result.append(d); total += d.token_count; used.add(id(f))

        # Phase 2 – round-robin-with-slack across legs (A10 budget fairness).
        # Each round, every leg contributes its next-best fragment (highest-scoring
        # leg first). This stops episodic — which emits dozens of fragments — from
        # soaking the whole remainder, while still filling fully when other legs
        # are sparse (exhausted legs drop out and their share goes to the rest).
        queues = {}
        for f in fragments:
            if id(f) not in used:
                queues.setdefault(f.source_type, []).append(f)
        for leg in queues:
            queues[leg].sort(key=lambda x: x.score, reverse=True)
        queues = {leg: deque(q) for leg, q in queues.items() if q}

        active = list(queues.keys())
        while active and total < max_tokens:
            active.sort(key=lambda leg: queues[leg][0].score, reverse=True)
            progressed = False
            for leg in list(active):
                q = queues[leg]
                f = q.popleft()
                if total + f.token_count <= max_tokens:
                    result.append(f); total += f.token_count; progressed = True
                else:
                    # C1/C3: fragment too big — degrade (summary, then
                    # abstract) before giving up on it (else skip; the leg's
                    # next fragment gets its chance next round).
                    d = _degraded(f, max_tokens - total)
                    if d:
                        result.append(d); total += d.token_count; progressed = True
                if not q:
                    active.remove(leg)
            if not progressed:
                break
        return result
    # ------------------------------------------------------------------
    # Strengthening (access count + decay boost)
    # ------------------------------------------------------------------
    def _strengthen_retrieved(self, fragments: List[ContextFragment]):
        for frag in fragments:
            if frag.source_type != "episodic" or not frag.source_batch_id:
                continue
            try:
                turn = self.db.query(EpisodicMemory).get(uuid.UUID(frag.source_batch_id))
                if turn:
                    turn.access_count = (turn.access_count or 0) + 1
                    turn.decay_score = min(1.0, (turn.decay_score or 0.0) + 0.15)
                    self.db.commit()
            except Exception:
                self.db.rollback()

    # ------------------------------------------------------------------
    # Wide‑net fallback (now uses full vector search)
    # ------------------------------------------------------------------
    def _wide_net_fallback(self, classification, prompt_embedding, conversation_id, scope):
        # C6/G16: the wide net widens *ranking*, not *visibility* — it must
        # honor the same scope rules as the normal legs. Previously it ignored
        # scope entirely (searched every conversation, codex unscoped, RAG
        # always on), which leaked project- and incognito-scoped memory.
        # T2: set here too — the wide net is also a public entry point (tests,
        # direct calls); idempotent when reached via retrieve().
        self._active_timescope = self._resolve_timescope(scope)
        self._scope_project_id = self._resolve_project_scope(scope)
        self._resolve_exclusion_sets(scope)
        scope_conv = scope.get("conversation_id") if scope else None
        # (C12: `incognito` was read here only to gate the RAG leg, which is
        # gone. The wide net's isolation now comes from _conv_scope_filter +
        # the privacy filter below, like every other leg.)
        conv_filter, conv_params, conv_scoped = self._conv_scope_filter(scope, scope_conv)
        excl_filter, excl_params = self._exclusion_filters(scope)
        privacy_filter = "" if conv_scoped else "AND is_private = FALSE"
        # G29/C6: the cluster filter was silently absent here while all three
        # normal episodic legs applied it — so a cluster-scoped conversation
        # that tripped the wide net widened its VISIBILITY, not just its
        # ranking, which is exactly what the comment above forbids. Same
        # predicate as the other legs, unlinked turns still allowed through.
        cluster_filter = ""
        if scope and scope.get("cluster_ids"):
            cluster_filter = """
                  AND (
                      EXISTS (
                          SELECT 1 FROM episodic_cluster_links l
                          WHERE l.episodic_id = episodic_memory.id
                            AND l.cluster_id = ANY(:cluster_ids)
                      )
                      OR NOT EXISTS (
                          SELECT 1 FROM episodic_cluster_links l
                          WHERE l.episodic_id = episodic_memory.id
                      )
                  )
            """
        # T3: same window/archived/floor rules as the normal legs (the wide
        # net widens ranking, not visibility — and not time either).
        time_filter, archived_filter, ts_params, min_decay = self._timescope_leg_filters()
        try:
            query = text(f"""
                SELECT id, raw_text, summary_text, summary_coverage, abstract_text, lossless_flag, inject_raw, conversation_id, is_document, timestamp,
                       (1 - (embedding <=> :prompt_embedding))
                         * (1 + :recency_boost * EXP(-ABS(EXTRACT(EPOCH FROM (timestamp - :ts_center))) / 86400.0 / :recency_tau)) as score
                FROM episodic_memory
                WHERE embedding IS NOT NULL
                  {archived_filter}
                  {time_filter}
                  AND decay_score > :min_decay
                  {conv_filter}
                  {excl_filter}
                  {privacy_filter}
                  {cluster_filter}
                ORDER BY score DESC
                LIMIT 100
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            creative = "Creative_&_Media" in (classification.topic_tags or [])
            ts_center, rec_boost, rec_tau = self._recency_params(creative)
            params = {"prompt_embedding": prompt_embedding, "min_decay": min_decay,
                      "recency_boost": rec_boost, "recency_tau": rec_tau,
                      "ts_center": ts_center, **ts_params, **conv_params,
                      **excl_params}
            if scope and scope.get("cluster_ids"):
                params["cluster_ids"] = scope["cluster_ids"]
            rows = self.db.execute(query, params).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic", prompt_text=classification.prompt, classification=classification)
        except Exception:
            self.db.rollback()
            fragments = []

        fragments.extend(self._codex_graph(classification, scope,
                                           prompt_embedding=prompt_embedding))

        fused = self._apply_rrf({"fallback": fragments}, alpha_map={"fallback": 1.0})
        prompt_keywords = self._extract_prompt_keywords(classification.prompt) if classification.prompt else set()
        fused = self._apply_bonuses(fused, classification, conversation_id, prompt_keywords)
        fused.sort(key=lambda x: x.score, reverse=True)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        # C15: dynamic ceiling — a fraction of the (model-aware, C16) retrieval
        # budget with a floor, replacing the hardcoded 2,000 tokens.
        wide_budget = max(WIDE_NET_BUDGET_FLOOR,
                          int(self.max_retrieval_tokens * WIDE_NET_BUDGET_FRACTION))
        return self._enforce_token_budget(self._deduplicate(diversified), max_tokens=wide_budget)

    # ------------------------------------------------------------------
    # Helper: convert raw DB rows to ContextFragment list
    # ------------------------------------------------------------------
    INTENTS_PREFER_RAW = {"Factual_Retrieval", "Troubleshooting"}
    INTENTS_PREFER_SUMMARY = {"Analysis_&_Summarization", "Strategic_Planning",
                              "Ideation", "Open_Exploration"}

    def _choose_representation(self, row, classification, prompt_keywords):
        """C1 read-time representation choice (user design: both forms are
        stored; NOTHING is permanently raw or permanently summary — the query
        context decides). Returns ``(text, degrade_text)``.

        Order of authority:
          1. availability/trust — no summary, or coverage below threshold
             (a summary that dropped must-terms is never used) → raw;
          2. keyword protection — the matched keyword lives in raw but not in
             the summary → raw, and NOT degradable (degrading would remove
             the very term that made this fragment relevant);
          3. intent preference — exactness intents prefer raw (degradable);
             compression-tolerant intents prefer the trusted summary;
          4. otherwise the storage-side default hint (inject_raw), with raw
             degradable to the trusted summary under budget pressure.

        Returns ``(text, degrade_text, abstract_text)`` — the abstract (C3,
        the last degradation level) rides along under the same trust and
        keyword-protection rules; the chooser never *prefers* it.
        """
        raw = row.raw_text
        summ = row.summary_text
        abstract = getattr(row, "abstract_text", None)
        if not raw:
            return (summ, None, abstract) if summ else (None, None, None)
        if not summ:
            return (raw if row.inject_raw else raw[:300]), None, None
        cov = getattr(row, "summary_coverage", None)
        # NULL coverage = legacy pre-C1 summary → keep status-quo trust
        trusted = cov is None or cov >= 0.7
        if not trusted:
            return raw, None, None

        if prompt_keywords:
            raw_l, summ_l = raw.lower(), summ.lower()
            kw_raw = any(kw in raw_l or kw.rstrip('s') in raw_l for kw in prompt_keywords)
            kw_summ = any(kw in summ_l or kw.rstrip('s') in summ_l for kw in prompt_keywords)
            if kw_raw and not kw_summ:
                return raw, None, None

        if classification is not None:
            intents = set(classification.intent_tags or [])
            if intents & self.INTENTS_PREFER_SUMMARY and not intents & self.INTENTS_PREFER_RAW:
                return summ, None, abstract
            if intents & self.INTENTS_PREFER_RAW:
                return raw, summ, abstract

        if row.inject_raw:
            return raw, summ, abstract
        return summ, None, abstract

    def _relevant_doc_chunks(self, turn_id, prompt_keywords, limit: int = 2):
        """C2: pick a document's most query-relevant chunks (keyword-hit count,
        earliest-first tiebreak; the opening chunk as fallback — it usually
        identifies the document). Returns joined text or None when the doc has
        no chunks yet (legacy pre-C2; the catch-up worker heals those)."""
        try:
            rows = self.db.execute(text("""
                SELECT chunk_text, chunk_index FROM episodic_chunks
                WHERE turn_id = :tid ORDER BY chunk_index ASC
            """), {"tid": turn_id}).fetchall()
        except Exception:
            self.db.rollback()
            return None
        if not rows:
            return None
        if prompt_keywords:
            def hits(txt):
                low = txt.lower()
                return sum(1 for kw in prompt_keywords
                           if kw in low or kw.rstrip('s') in low)
            scored = sorted(rows, key=lambda r: (-hits(r.chunk_text), r.chunk_index))
            if hits(scored[0].chunk_text) > 0:
                chosen = sorted(scored[:limit], key=lambda r: r.chunk_index)
                return "\n[…]\n".join(r.chunk_text for r in chosen)
        return rows[0].chunk_text

    def _rows_to_fragments(self, rows, source_type, prompt_text: Optional[str] = None,
                           classification=None):
        fragments = []
        prompt_keywords = self._extract_prompt_keywords(prompt_text) if prompt_text else set()

        for row in rows:
            text, degrade_text, abstract_text = self._choose_representation(row, classification, prompt_keywords)
            if not text:
                continue

            # Word cap (keyword-aware). C2: documents are NEVER injected whole
            # anymore (the old word_cap=999999 bypass) — a doc row found by
            # BM25/text search injects only its keyword-relevant chunks.
            is_doc = getattr(row, "is_document", False)
            word_cap = 500
            if is_doc:
                chunk_text_ = self._relevant_doc_chunks(row.id, prompt_keywords)
                if chunk_text_:
                    text = chunk_text_
                    degrade_text = None
                    word_cap = 999999   # already a bounded selection (≤2 chunks)
                # else: legacy doc without chunks (pre-C2, catch-up worker will
                # heal it) — falls through to the normal 500-word cap instead
                # of dumping the whole document.
            elif prompt_keywords:
                text_lower = text.lower()
                if any(kw in text_lower or kw.rstrip('s') in text_lower for kw in prompt_keywords):
                    word_cap = 1500

            text = _truncate_at_sentence(text, word_cap)

            if not text:
                continue

            score_val = float(getattr(row, "score", 1.0))
            if getattr(row, "is_bookmarked", False):
                score_val *= (1.0 + BONUS_BOOKMARKED)

            # T1 date-grounding: every episodic fragment carries the date its
            # turn was written, so the model can order events against the
            # "Today's date" anchor in the system prompt. Dates only, never
            # clock times; degraded forms keep the stamp (degradation must not
            # lose the date).
            if getattr(row, "timestamp", None):
                stamp = row.timestamp.strftime("[%Y-%m-%d] ")
                text = stamp + text
                if degrade_text:
                    degrade_text = stamp + degrade_text
                if abstract_text:
                    abstract_text = stamp + abstract_text

            fragments.append(ContextFragment(
                text=text,
                source_type=source_type,
                score=score_val,
                token_count=count_tokens(text),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None,
                degrade_text=degrade_text,
                abstract_text=abstract_text,
            ))
        return fragments