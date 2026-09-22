"""Hybrid Retrieval Orchestrator – Phase A hardened: decay filtering, access-weighting,
wide‑net full‑vector, Codex/Procedural scoping, grounded query expansion, procedural
trigger matching, micro‑NER integration, and dynamic token budget."""

import hashlib
import math
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import structlog
from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy import bindparam, or_, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.config import settings
from src.classifier.schemas import ClassificationResult
from src.memory.models import (
    CodexClaim,
    CodexClaimLink,
    CodexEdge,
    CodexEntity,
    ColdStorage,
    EpisodicMemory,
    ProceduralMemory,
)
from src.memory.claims import claim_representation, excerpt_is_current, source_for_claim
from src.memory.representation import choose_representation
from src.memory.summary_snapshot import batch_snapshot_readable, summary_snapshot_readable
from src.memory.time_format import format_time, recorded_stamp
from src.memory.tokens import count as count_tokens
from src.retrieval import coverage, leg_weights
from src.retrieval.evolution import build_entity_timeline, history_exists
from src.retrieval.ner_utils import extract_entities
from src.retrieval.reranker import rerank
from src.retrieval.timescope import CURRENT, from_scope

logger = structlog.get_logger("ice.retrieval")


def _count_by(items, key) -> dict:
    """Count *items* by *key*, for log payloads. Never raises on a bad key —
    an observability helper that can break the call it is observing is worse
    than the missing number."""
    out: dict = {}
    for it in items:
        try:
            k = str(key(it))
        except Exception:                                    # noqa: BLE001
            k = "?"
        out[k] = out.get(k, 0) + 1
    return out


@dataclass(frozen=True)
class ContextFragment:
    text: str
    source_type: str          # "episodic", "codex", "procedural", "timeline"
    score: float              # RRF fused score
    token_count: int
    source_batch_id: Optional[str] = None
    conversation_id: Optional[str] = None
    # ⚑ WHICH TURNS THIS FRAGMENT WAS DERIVED FROM (G48). `source_batch_id` is
    # one turn and only episodic fragments ever set it, so every recall number
    # ICE has produced credits the episodic leg alone: measured on one harvest,
    # 4,124 episodic fragments earned 249 gold credits while 474 codex, 400
    # procedural and 207 timeline fragments earned **zero** — not because they
    # were wrong, but because nothing could attribute them to a turn.
    #
    # A codex facts fragment aggregates many edges and a procedural pattern
    # cites many turns, so this is a TUPLE, not a single id. The provenance
    # already exists in the store (all 9,662 edges carry a source_batch that
    # maps to a real turn; patterns carry source_batch_ids) — it was simply
    # never carried onto the fragment. Frozen dataclass, hence a tuple.
    origin_batch_ids: tuple = ()
    # C1: the *trusted* compact form (grounded summary with passing coverage)
    # when text is the raw representation — lets the token budget degrade a
    # too-big fragment to its summary instead of dropping it entirely. None
    # when text already is the summary, no trusted summary exists, or the
    # keyword that matched lives only in the raw.
    degrade_text: Optional[str] = None
    # C3: a source-extractive abstract, independently eligible for degradation.
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
    origin_edge_ids: tuple = ()  # exact rendered fact lines, not traversed candidates
    covers_entire_source: bool = False  # complete raw turn, not summary/excerpt

# G9 (2026-08-08): every tunable number in this module moved to settings, so
# Z1 can sweep it without editing code. What remains here are label SETS —
# they name classifier labels, which is schema, not tuning.
#
# C8's recency-inside-the-vector-score, C15's wide-net budget, the whole bonus
# family and every leg candidate limit are now `settings.retrieval_*`; the
# codex traversal/relation knobs (formerly `self.CODEX_*` / `self.RELATION_*`
# attributes set in __init__) are `settings.codex_*`. They are read at the
# point of use, never cached on the instance — see the config.py note.


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



# Soft meta‑discussion downweight – classifier‑driven, not string‑matching.
# The label sets stay here (schema); the factor is settings.
NARRATIVE_FACT_INTENTS = {"Factual_Retrieval", "Decision_Making"}
META_LEANING_INTENTS = {"Analysis_&_Summarization"}

# A4: process-wide cache of (relation_names, gloss_embeddings) for relation
# detection — built lazily on first use by _relation_gloss_cache().
_RELATION_GLOSSES = None

class HybridRetrievalOrchestrator:
    def __init__(self, db: Session, embedder):
        self.db = db
        self.embedder = embedder
        # Query expansion is graph-grounded. The shared local reranker scores
        # evidence without generating text or creating a background API client.
        self.max_retrieval_tokens = 5000
        self._coverage_record = None   # C16: last coverage decision, for audit
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
        # (G9: the numbers are settings.codex_relation_* now; the feasibility
        # probe behind the default k=5 was top-3 11/12, top-5 12/12.)
        self._last_matched_entities = []         # per-call: for grounded expansion

        # T2/T3: the request's TimeScope (set from scope["timescope"] at the
        # top of retrieve()/_wide_net_fallback(); one orchestrator instance
        # per request, so an instance attr is safe). timescope_allowed is the
        # ablation seam — the configurable subclass sets it False to force
        # CURRENT regardless of scope.
        self._active_timescope = CURRENT
        self.timescope_allowed = True
        self._cold_hits = {}
        self._source_times = {}
        # E1b (D3): the request's attached project (scope["project_id"]) —
        # source-visibility for derived code/fact entities keys off this,
        # same instance-attr pattern as _active_timescope.
        self._scope_project_id = None
        # C6: the request's exclusion deny sets (same pattern). Empty means
        # "nothing excluded", which is the default and the common case.
        self._denied_batch_ids = set()
        self._denied_entity_ids = set()
        # G36: set when a scope/exclusion resolver failed. Both used to fail
        # OPEN (unscoped / nothing excluded), so a broken scope query widened
        # visibility. The flag makes _codex_scope_sets fail CLOSED instead.
        self._scope_resolution_failed = False
        # G35: the request's prompt embedding, so _traverse_graph can rank its
        # frontier by relation fit (same instance-attr pattern as
        # _active_timescope — one orchestrator per request). None means "rank by
        # trust alone", which is the pre-G35 behaviour and the honest fallback
        # when the encoder is degraded.
        self._prompt_embedding = None

        # Load micro‑NER model (fallback to None if not available)

    # ------------------------------------------------------------------
    # G36: the one place a retrieval leg is allowed to give up
    # ------------------------------------------------------------------
    def _leg_degraded(self, leg: str, err: Exception) -> None:
        """Announce that *leg* swallowed *err* and is returning nothing.

        Every ``except`` on this path used to catch, roll back and return
        ``[]`` — so a leg whose SQL was broken by a schema change or an
        unbound parameter said *"no memory found"* and nothing anywhere
        disagreed. `_procedural_lookup` sat dead that way from the psycopg3
        move until C9 happened to remove the gate hiding it. One event name
        (`retrieval_leg_failed`) with the leg as a FIELD, because the question
        worth asking is "did any leg fail on this request", and thirty-one
        distinct event names cannot answer it.

        Logged EVERY time, never first-occurrence-only: the rate is the
        signal. Measured baseline before this shipped — 476 checks across 15
        seeded suites raised nothing here, so a line means something.

        **The rollback is conditional and that is the point.** After a
        *database* error Postgres refuses every later statement in the
        transaction, so rolling back is mandatory or one broken leg kills all
        the ones behind it *and* the post-flight write on the same session.
        After a *Python* error the transaction is intact, and rolling back
        anyway expires the whole identity map on a session `main.py` keeps
        using for the rest of the request. So: DB errors only.
        """
        # SQLAlchemy exception strings include statement parameters (raw memory,
        # role provenance and verifier inputs). Diagnose without logging evidence.
        logger.warning("retrieval_leg_failed", leg=leg,
                       error_type=type(err).__name__,
                       sqlstate=getattr(getattr(err, "orig", None), "sqlstate", None))
        if isinstance(err, SQLAlchemyError):
            self.db.rollback()

    def _relevant_cluster_ids(self, prompt_embedding, classification=None, conversation_id=None, top_k=None, scope=None):
        """Return a list of cluster_id strings for the clusters most
        relevant to the prompt, using both embedding similarity and
        topic‑tag overlap with the current classification.
        """
        if top_k is None:
            top_k = settings.retrieval_cluster_top_k
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
            params = {"emb": prompt_embedding,
                      "limit": top_k * settings.retrieval_cluster_candidate_multiplier,
                      **conv_params}
            rows = self.db.execute(query, params).fetchall()
        except Exception as err:
            self._leg_degraded("cluster_scope", err)
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
                bonus += settings.retrieval_bonus_keyword_match

            # Length
            if word_count > settings.retrieval_long_narrative_words:
                bonus += settings.retrieval_bonus_long_narrative
            elif word_count > settings.retrieval_substantial_words:
                bonus += settings.retrieval_bonus_substantial
            elif word_count < settings.retrieval_short_words:
                bonus += settings.retrieval_penalty_short

            # Recency (skip for creative – recent meta turns are noise;
            # T3: skipped under any non-current mode — this bonus points at now)
            if (f.source_type == "episodic" and not creative and conv_id
                    and self._active_timescope.mode == "current"):
                bonus += self._recency_bonus(f, conv_id)

            # Soft meta downweight
            if wants_narrative_fact and f.source_type == "episodic" and f.source_batch_id:
                if self._turn_leans_meta(f.source_batch_id):
                    bonus -= (1.0 - settings.retrieval_meta_downweight_factor)

            bonus = max(settings.retrieval_min_total_bonus,
                        min(settings.retrieval_max_total_bonus_multiplier, bonus))
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
            if total <= settings.retrieval_recency_min_turns:
                return 0.0
            newer_count = self.db.query(EpisodicMemory).filter(
                EpisodicMemory.conversation_id == conv_id,
                EpisodicMemory.timestamp > turn.timestamp
            ).count()
            recency_pct = newer_count / total
            if recency_pct < settings.retrieval_recent_top_pct:
                return settings.retrieval_bonus_recent_top_10pct
            elif recency_pct < settings.retrieval_recent_mid_pct:
                return settings.retrieval_bonus_recent_top_30pct
            return 0.0
        except Exception as err:
            self._leg_degraded("bonus.recency", err)
            return 0.0

    def _turn_leans_meta(self, source_batch_id):
        """Check the source turn's intent_tags for meta/analytical leaning."""
        try:
            turn = self.db.query(EpisodicMemory).get(uuid.UUID(source_batch_id))
            if not turn or not turn.intent_tags:
                return False
            return bool(META_LEANING_INTENTS & set(turn.intent_tags))
        except Exception as err:
            self._leg_degraded("bonus.meta_lean", err)
            return False
    def _extract_prompt_keywords(self, prompt_text):
        words = set(re.sub(r'[^\w\s]', ' ', prompt_text).lower().split())
        common = {"the","is","of","and","a","to","in","that","it","for","was","on","are",
                  "with","what","when","where","who","how","i","you","me","my","we","our",
                  "so","be","do","did","does","get","got","very","too","now","this","that",
                  "these","those","some","many","each","every","other","more","gonna","wanna"}
        return words - common



    # G9: the fallback ceiling and the overhead reserve are settings now
    # (context_total_budget_fallback / context_overhead_reserve). Kept as
    # properties so external callers reading the class attribute still work.
    @property
    def TOTAL_CONTEXT_BUDGET(self) -> int:
        return settings.context_total_budget_fallback

    @property
    def OVERHEAD_RESERVE(self) -> int:
        return settings.context_overhead_reserve


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

        # Growth-based retrieval cap. G9: the brackets are
        # settings.context_growth_cap_ladder — [turn_count_below, base,
        # per_turn], with no cap past the last bracket.
        growth_cap = raw_retrieval
        prev_edge = 0
        for edge, base_cap, per_turn in settings.context_growth_cap_ladder:
            if turn_count < edge:
                growth_cap = base_cap + (turn_count - prev_edge) * per_turn
                break
            prev_edge = edge

        retrieval_budget = min(raw_retrieval, growth_cap)

        # NO reallocation — leftover stays unused. This is what makes ICE
        # token-efficient compared to the vector baseline.

        self.recent_token_budget = recent_budget
        self.max_retrieval_tokens = retrieval_budget

        # Remove the assertion — the sum is intentionally less than 'available'


    def _compute_recent_fraction(self, turn_count: int, total_tokens: int = 0, classification=None) -> float:
        """Return 0.0-1.0 - how much of the context budget goes to recent turns.

        G9: every bracket and modifier is settings.context_recent_* now. The
        shape is unchanged - a length ladder, a token-density adjustment for
        conversations made of very long turns, and label groups that each
        apply at most once however many of their labels are active.
        """
        base = settings.context_recent_fraction_default
        for edge, value in settings.context_recent_fraction_ladder:
            if turn_count < edge:
                base = value
                break

        # Token-density adjustment: when the average turn is huge, shift budget
        # toward retrieval so the recent window is not two enormous turns.
        if turn_count > 0 and total_tokens > 0:
            avg_tokens_per_turn = total_tokens / turn_count
            for threshold, delta in settings.context_recent_density_ladder:
                if avg_tokens_per_turn > threshold:
                    base += delta
                    break

        modifier = 0.0
        if classification is not None:
            active = {
                "intent": set(classification.intent_tags or []),
                "topic": set(classification.topic_tags or []),
            }
            for group in settings.context_recent_fraction_groups:
                if active.get(group["head"], set()) & set(group["labels"]):
                    modifier += group["delta"]

        return max(settings.context_recent_fraction_min,
                   min(settings.context_recent_fraction_max, base + modifier))

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
        # T2: resolve the request's TimeScope once; every leg below reads it.
        self._active_timescope = self._resolve_timescope(scope)
        # E1b (D3): attached project — gates derived-entity visibility.
        self._scope_project_id = self._resolve_project_scope(scope)
        # C6: the exclusion deny sets, resolved once for every leg below.
        self._resolve_exclusion_sets(scope)
        self._cold_hits = {}
        self._source_times = {}
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

        # ⚑ G53: TWO DIFFERENT QUESTIONS, AND THEY USED TO SHARE ONE VARIABLE.
        #
        #   conv_id      — "restrict the search to this conversation". Set only
        #                  when the SCOPE says so, because that is a scoping
        #                  decision: `none`/`project` scope it, `auto` does not.
        #   own_conv_id  — "which conversation are we in". Always known, because
        #                  the caller passes it, and it is never a filter.
        #
        # `resolve_retrieval_scope` returns `{}` for an `auto` conversation —
        # the DEFAULT mode — so `conv_id` is None there by design. But the
        # recency bonus and the conversation's own batch summaries were both
        # gated on `conv_id`, so on the default mode ICE could not tell which
        # conversation it was in: it never weighted its own recent turns and
        # never read its own summaries. `conversation_id` was sitting in the
        # signature unused for exactly this the whole time.
        #
        # Search scoping is deliberately unchanged: `auto` still searches
        # everything. This only restores the "which conversation is mine"
        # half.
        conv_id = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]
        own_conv_id = str(conversation_id) if conversation_id else conv_id

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
            # G29: top_k omitted so settings.retrieval_cluster_top_k decides. The
            # literal 10 that used to sit here matched the setting, so nothing
            # moves — but it made the knob unreachable, same as the
            # max_per_conversation case below.
            cluster_ids = self._relevant_cluster_ids(prompt_embedding, classification=classification, conversation_id=conv_id, scope=scope)
            if cluster_ids and scope is not None:
                scope["cluster_ids"] = cluster_ids

        # (A commented-out HyDE call sat here, and had since before
        # `v2-paper-eval`. Deleted with the method itself, 2026-08-09, G36 —
        # see the note where `_hyde_rewrite` used to be. The grounded query
        # expansion a few lines down is what replaced it.)

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
            "codex": codex_fragments + self._codex_claims(classification.prompt, prompt_embedding, scope, conv_id),
            "procedural": [] if incognito else self._procedural_lookup(prompt_embedding, classification, scope),
            # C4: the cross-conversation summary half is a user-global read —
            # gated off under incognito like procedural; the conv-scoped
            # batch half (own context) still runs.
            # G53: `own_conv_id`, not `conv_id` — "my own summaries" is a
            # question about which conversation this is, not about how wide the
            # search may go. Under `auto` (the default) `conv_id` is None, so
            # this half never ran and `batch_summary` appeared in zero of 15
            # recorded scoring runs. The cross half's self-exclusion needs the
            # same id for the same reason: with None it excluded nothing and
            # could re-inject this conversation's own summary.
            "batch_summary": self._batch_summary_lookup(
                prompt_embedding, own_conv_id, include_cross=not incognito,
                scope=scope, search_conv_id=conv_id),
            # T3: cold storage joins time-scoped queries only (no-op leg
            # otherwise); fragments are episodic-typed, so budget fairness
            # treats them as memories — the leg name only affects RRF weight.
            "cold": self._cold_lookup(prompt_keywords, conv_id, scope,
                                      prompt_embedding),
            "timeline": timeline_fragments,
        }

        # ── Dynamic leg weighting (blended over all active intents) ──
        # G9: the table itself is settings.retrieval_leg_* and the blend lives
        # in leg_weights.resolve(), which validates it on every read. It used
        # to be ~50 lines of literals rebuilt here on every single request.
        blend_weights = leg_weights.resolve(classification.intent_tags,
                                            classification.topic_tags)

        # Fuse, diversify, deduplicate, trim
        fused = self._apply_rrf(legs, alpha_map=blend_weights)
        # G53: `own_conv_id` — the recency bonus ranks a fragment by its
        # position within THIS conversation (`_recency_bonus` counts turns in
        # it), which is meaningless as a scope filter and was dead under `auto`.
        # `_wide_net_fallback` already passed the parameter here; the main path
        # did not, and the two disagreed.
        fused = self._apply_bonuses(fused, classification, own_conv_id, prompt_keywords)
        fused.sort(key=lambda x: x.score, reverse=True)
        fused, reranked = rerank(classification.prompt, fused)
        # G29: no max_per_conversation here — passing the literal 3 is what made
        # settings.retrieval_max_per_conversation unreachable on every live path.
        deduped = self._deduplicate(fused)
        if not reranked:
            deduped = self._apply_coverage(deduped, prompt_embedding)
        final = self._enforce_token_budget(deduped, relevance_order=reranked,
                                           current_conversation_id=own_conv_id)

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
        except (ValueError, AttributeError, TypeError) as err:
            # Tolerant by design — but "no project" silently WIDENS what the
            # derived-entity filters admit, so it says so.
            self._leg_degraded("scope.project", err)
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

    def _cluster_filter(self, scope, id_column="episodic_memory.id") -> str:
        """C5 cluster scoping as a SQL fragment — G29: written out four times.

        The copies were verbatim apart from the row alias (`episodic_memory.id`
        on three legs, `e.id` on the chunk leg), which is why they read as
        identical and could not be diffed by eye. The wide net had **no** copy
        at all until C6 added one, and that omission widened *visibility* rather
        than ranking — the failure this consolidation is meant to make
        impossible to repeat.

        The `OR NOT EXISTS` half is load-bearing and easy to mistake for
        redundancy: a turn that no clustering pass has reached yet belongs to no
        cluster, and must stay visible rather than be scoped out of existence by
        a filter about clusters it was never eligible for.

        `id_column` is a caller-supplied SQL identifier, never user input; the
        cluster ids themselves are bound as `:cluster_ids`.
        """
        if not (scope and scope.get("cluster_ids")):
            return ""
        return f"""
                AND (
                    EXISTS (
                        SELECT 1 FROM episodic_cluster_links l
                        WHERE l.episodic_id = {id_column}
                          AND l.cluster_id = ANY(:cluster_ids)
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM episodic_cluster_links l
                        WHERE l.episodic_id = {id_column}
                    )
                )
            """

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
            return center, settings.retrieval_episodic_recency_boost, max(settings.retrieval_episodic_recency_tau_days, window_days / 2)
        if ts.mode in ("range", "evolution"):
            return now, 0.0, settings.retrieval_episodic_recency_tau_days
        return now, (0.0 if creative else settings.retrieval_episodic_recency_boost), settings.retrieval_episodic_recency_tau_days

    # ------------------------------------------------------------------
    # (A `_hyde_rewrite` method used to live here. DELETED 2026-08-09, G36.)
    #
    # It asked the background model to rewrite the prompt into a dense search
    # query before the lexical leg ran. It was never real HyDE — it never
    # fabricated a hypothetical answer document, only reformulated the
    # question — and the post-paper review (roadmap P0.1) rejected shipping
    # the real thing: it would invent specifics about the user's PRIVATE
    # history with a small model, and hallucinated specifics poison the
    # noise-sensitive BM25 leg. **Its replacement is A4 grounded query
    # expansion** in retrieve(): the BM25 search prompt gains the canonical
    # names and aliases of the entities the codex leg actually matched, so
    # the expansion terms come from the graph rather than from a generator.
    #
    # Why it was deleted rather than left commented: the call site had been
    # commented out since before `v2-paper-eval`, `ConfigurableOrchestrator`
    # never read its own `hyde` flag, and nothing set `_force_hyde` — so the
    # method was unreachable by every path, including the ablation one the
    # docs claimed. Experiment 1's `full_ice_no_hyde` arm was therefore the
    # same configuration as `full_ice`; see ROADMAP G36 and PROVENANCE.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # BM25 episodic (full‑text search)
    # ------------------------------------------------------------------
    def _bm25_episodic(self, classification, scope, conv_id=None, search_prompt=None):
        prompt_text = search_prompt if search_prompt else classification.prompt
        # Match the stored tsvector normalization without losing numbers,
        # Unicode or late query terms. Never interpret user text as tsquery.
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
        cluster_filter = self._cluster_filter(scope)

        query = text(f"""
            SELECT id, raw_text, summary_text, summary_coverage, representation_verification, source_spans, abstract_text, lossless_flag, inject_raw, conversation_id, is_bookmarked, timestamp, ts_provenance,
                   ts_rank(
                       to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')),
                       query
                   ) as score
            FROM episodic_memory,
                 LATERAL (SELECT
                     string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
                     FROM unnest(tsvector_to_array(
                         to_tsvector('english', :prompt_text))) AS terms(lexeme)
                 ) AS q
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
            LIMIT :cand_limit
        """)
        params = {
            "cand_limit": settings.retrieval_bm25_candidate_limit,
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
            self._leg_degraded("bm25", err)
            # Final fallback: use plainto_tsquery (AND) if everything fails
            try:
                query2 = text(f"""
                    SELECT id, raw_text, summary_text, summary_coverage, representation_verification, source_spans, abstract_text, lossless_flag, inject_raw, conversation_id, is_bookmarked, timestamp, ts_provenance,
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
                    LIMIT :cand_limit
                """)
                p = {"cand_limit": settings.retrieval_bm25_candidate_limit,
                     "prompt_text": prompt_text, "min_decay": min_decay,
                     **ts_params, **conv_params, **excl_params}
                if scope and scope.get("cluster_ids"):
                    p["cluster_ids"] = scope["cluster_ids"]
                rows = self.db.execute(query2, p).fetchall()
                return self._rows_to_fragments(rows, "episodic", prompt_text=search_prompt or classification.prompt, classification=classification)
            except Exception as err2:
                self._leg_degraded("bm25.fallback", err2)
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
        cluster_filter = self._cluster_filter(scope)

        # C2: document turns are EXCLUDED from turn-level vector search — one
        # embedding over thousands of words is semantic mush (the C3 ceiling).
        # Their chunks compete instead, via _vector_chunks below.
        # C8 recency, T3/D9 origin-parameterized: ABS(timestamp - :ts_center)
        # with center=now is numerically identical to the old NOW()-based
        # GREATEST form for past rows; as_of re-anchors center to the window
        # midpoint. One formula, mode-driven params — never fork the leg SQL.
        query = text(f"""
            SELECT id, raw_text, summary_text, summary_coverage, representation_verification, source_spans, abstract_text, lossless_flag, inject_raw, conversation_id, is_bookmarked, timestamp, ts_provenance,
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
                  "cand_limit": (settings.retrieval_vector_candidate_limit_evolution
                                 if ts.mode == "evolution"
                                 else settings.retrieval_vector_candidate_limit),
                  **ts_params, **conv_params, **excl_params}
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]

        try:
            rows = self.db.execute(query, params).fetchall()
            if ts.mode == "evolution":
                rows = self._stratify_by_era(rows)
            fragments = self._rows_to_fragments(rows, "episodic", prompt_text=classification.prompt, classification=classification)
            # A candidate parent may never fit. Keep its excerpts until packing.
            fragments.extend(self._vector_chunks(prompt_embedding, scope, conv_id,
                                                  recency_boost=rec_boost))
            return fragments
        except Exception as err:
            self._leg_degraded("vector", err)
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
                       recency_boost=None) -> List[ContextFragment]:
        """C2: chunk-level vector search over document turns. Visibility
        (decay, archive, privacy, conversation, cluster scope) is enforced
        through the parent turn; provenance points at the parent so
        strengthening/decay land on the turn. Max 3 chunks per document.

        G9: `recency_boost` defaults to None rather than to the setting — a
        default argument is evaluated once at import, which would freeze the
        value and make a Z1 sweep of it silently do nothing."""
        if recency_boost is None:
            recency_boost = settings.retrieval_episodic_recency_boost
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
        cluster_filter = self._cluster_filter(scope, "e.id")
        query = text(f"""
            SELECT c.chunk_text, c.chunk_index, e.id AS parent_id,
                   e.conversation_id, e.is_bookmarked, e.timestamp, e.ts_provenance,
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
            LIMIT :cand_limit
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        params = {"cand_limit": settings.retrieval_chunk_candidate_limit,
                  "prompt_embedding": prompt_embedding, "min_decay": min_decay,
                  "recency_boost": recency_boost, "recency_tau": rec_tau,
                  "ts_center": ts_center, **ts_params, **conv_params,
                  **excl_params}
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]
        try:
            rows = self.db.execute(query, params).fetchall()
        except Exception as err:
            self._leg_degraded("vector_chunks", err)
            return []

        fragments, per_parent = [], {}
        for row in rows:
            pid = str(row.parent_id)
            if per_parent.get(pid, 0) >= 3:
                continue
            per_parent[pid] = per_parent.get(pid, 0) + 1
            score_val = float(row.score)
            if row.is_bookmarked:
                score_val *= (1.0 + settings.retrieval_bonus_bookmarked)
            # T1: chunks are episodic fragments too — date them from the parent turn.
            chunk_text = row.chunk_text
            if row.timestamp:
                chunk_text = recorded_stamp(row.timestamp, getattr(row, "ts_provenance", None)) + chunk_text
            fragments.append(ContextFragment(
                text=chunk_text,
                source_type="episodic",
                score=score_val,
                token_count=count_tokens(chunk_text),
                source_batch_id=pid,
                conversation_id=str(row.conversation_id) if row.conversation_id else None,
            ))
        return fragments
        
    
    def _match_entities_by_similarity(self, entity_strings: List[str], threshold: float = None) -> List:
        """Match extracted entity strings to CodexEntity rows using vector similarity.
        Falls back to canonical name / alias exact match when embeddings are unavailable."""
        if threshold is None:
            threshold = settings.codex_entity_match_threshold
        if not entity_strings:
            return []

        # Embed all candidate strings
        candidate_embeddings = self.embedder.encode(entity_strings, convert_to_tensor=False, show_progress_bar=False)

        matched = []
        seen_ids = set()
        for candidate_str, candidate_emb in zip(entity_strings, candidate_embeddings):
            # G41: the nearest entity, ranked BY POSTGRES.
            #
            # This used to fetch every CodexEntity carrying an embedding into
            # Python and score them in a pure-Python loop — O(candidates x
            # entities x 1024) multiply-adds per prompt, unbounded in the size of
            # the graph. Invisible on an empty store, and the reason G34's entry
            # singled it out as "fine on an empty store and not fine on Z2's".
            #
            # The rewrite is EXACTLY equivalent, not merely close. Two facts make
            # it so: (a) since C17 native-width encodes are unit-norm, cosine
            # similarity IS the dot product the old loop computed, and pgvector's
            # `<=>` is cosine distance, so `1 - (a <=> b)` is that same number;
            # (b) the loop only ever wanted the single best entity above the
            # threshold that had not already been claimed, and ordering by
            # distance puts it at rank 1 — so `LIMIT 1 + len(seen_ids)` is
            # guaranteed to contain it however many earlier candidates matched.
            rows = self.db.execute(text("""
                SELECT id, 1 - (embedding <=> CAST(:emb AS vector)) AS sim
                FROM codex_entities
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :lim
            """), {
                # ⚠ float(), not list(): candidate_emb is a numpy array, and
                # str(list(ndarray)) renders "[np.float32(-0.019), ...]", which
                # pgvector rejects as invalid vector syntax.
                "emb": str([float(x) for x in candidate_emb]),
                "lim": 1 + len(seen_ids),
            }).all()
            best_entity = None
            for row in rows:
                if row.id in seen_ids:
                    continue
                if row.sim >= threshold:
                    best_entity = self.db.query(CodexEntity).get(row.id)
                break   # rank 1 among the unclaimed; a worse one cannot win
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
            # G45: the vocabulary is open, so relation fit must be scored over
            # the relations the graph ACTUALLY uses — not a 197-word list it has
            # outgrown. G34 needs an embedding per relation, never a closed set,
            # so this is the one function that changes: score against the union
            # and the feature keeps working instead of going quietly inert on
            # every relation invented after it was written.
            rels = set(ALLOWED_RELATIONS)
            try:
                rels |= {r[0] for r in self.db.execute(text(
                    "SELECT DISTINCT relation FROM codex_edges "
                    "WHERE relation IS NOT NULL")).all() if r[0]}
            except Exception as err:
                self._leg_degraded("codex.relation_vocab", err)
            rels = sorted(rels)
            embs = self.embedder.encode([r.replace("_", " ") for r in rels],
                                        convert_to_tensor=False, show_progress_bar=False)
            # G41: cached as ONE float64 ndarray, not a list of 197 lists.
            # Measured: the matmul against these is 0.018 ms, but converting the
            # list-of-lists to an array on every call cost 2.97 ms — so the
            # per-prompt bill was the marshalling, not the arithmetic. Vectorising
            # the loop without fixing this bought 25%; fixing this buys ~165x on
            # the same line.
            _RELATION_GLOSSES = (rels, np.asarray(embs, dtype=np.float64))
        return _RELATION_GLOSSES

    @staticmethod
    def _stem(word: str) -> str:
        """Crude suffix-stripper so 'inspired'/'inspires'/'inspiring' all meet
        the relation lexeme 'inspired'. Both sides are stemmed identically, so
        crudeness cancels out.

        G34: the guard was `len(w) > 4`, which made the symmetry claim above
        MEASURABLY FALSE for short vocabulary words — `uses`/`using`,
        `uses`/`use`, `owns`/`owning`, `has`/`have` all failed to meet, and both
        `uses` and `owns` are live vocabulary. End to end, "what does Kael use
        for the ritual" produced zero channel-1 hits and "what does Kael own"
        found `owned_by` while missing `owns`: one concept, two entries,
        reachability decided by word length. The floor is now the length of what
        REMAINS, so stripping never empties a word and short entries still stem.

        ⚠ This makes the docstring true; it does not make the channel sound.
        Lexical matching is a bet on how a thing is written, which CLAUDE.md's
        invariance rule forbids and [G28] owns. G34 removed this channel from the
        anchor path entirely for that reason — it survives only for
        `_codex_enumeration`, whose gate is explicitly lexical anyway.
        """
        w = word.lower()
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 2:
                return w[: -len(suf)]
        return w

    def _detect_relations(self, prompt: str, prompt_embedding) -> List[str]:
        """Controlled-vocabulary relations relevant to the prompt, from two
        channels: (1) lexical — a relation's own content word appears in the
        prompt ('who inspired X' → inspired_by), which is a direct grounded
        hit; (2) embedding — top-k gloss cosine for paraphrases ('who is X's
        wife' → married_to). Joint-signal only (see __init__ note): callers
        must pair the result with matched entities or enumeration cues, never
        use it alone.

        ⚠ G34 (2026-08-11): the ANCHOR path no longer calls this. It ranks each
        anchor against its own relations instead (`_relation_fit`), because this
        function could not be made to say "no" — measured, three candidate fixes,
        none separating. `_codex_enumeration` is now the only caller, and there
        the explicit cue word carries the precision this cannot.

        The kill-switch therefore governs enumeration alone, and returns before
        any work so "off" costs nothing."""
        if not settings.codex_relation_detection_enabled:
            return []
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
                # G41: 197 x 1024 as one matmul rather than 197 Python loops —
                # this is the median 12.2 ms that used to sit on every
                # pre-flight, now confined to enumeration-cue prompts (G34).
                all_sims = gloss_embs @ np.asarray(prompt_embedding,
                                                   dtype=np.float64)
                scored = []
                for rel, sim in zip(rels, all_sims):
                    if rel in detected:
                        continue
                    sim = float(sim)
                    if sim >= settings.codex_relation_sim_floor:
                        scored.append((sim, rel))
                scored.sort(reverse=True)
                detected.extend(rel for _, rel in scored[:settings.codex_relation_top_k])
            return detected
        except Exception as err:
            self._leg_degraded("codex.relations", err)
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
        if self._scope_resolution_failed:
            # G36: the exclusion resolver could not work out what to deny.
            # Reading the graph unscoped would serve the excluded material;
            # match nothing instead. Checked before the `not batch_ids`
            # branch below.
            return set(), set()
        # G37: did the caller RESTRICT the conversation set, or merely scope by
        # something else? Tested by key PRESENCE, not truthiness — C6's rule,
        # because an exclusion can empty a closed set and `[]` must still mean
        # "these conversations, of which there are none". A cluster-only or
        # timescope-only scope names no conversations at all and stays
        # unscoped, which is what keeps `auto` traffic working.
        restricted = ("batch_ids" in scope or "conversation_ids" in scope
                      or "conversation_id" in scope)
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
                # G37: a named conversation set that resolves to no batches
                # reads NOTHING, not everything. It resolves to nothing on the
                # first turn of every manually-scoped conversation (no stored
                # turns yet) and whenever exclusions empty a closed set — and
                # the old `return None, None` meant UNSCOPED, so exactly then
                # the graph leg served the whole store. C6 already settled this
                # for the episodic legs (`_conv_scope_filter`: "an empty set
                # must match nothing"); the codex and procedural legs never got
                # it. A project attachment was exempted long ago by the `pid`
                # test above, with a comment naming this same hazard.
                # User decision 2026-08-09: starting fresh should give nothing,
                # which is what codex scoping was built for.
                if restricted:
                    return set(), set()
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
        except Exception as err:
            # G36: this used to return (None, None) — which does NOT mean
            # "no scope", it means UNSCOPED, i.e. search the entire graph.
            # A failure resolving the scope the caller asked for therefore
            # WIDENED visibility, silently, in the one direction G16 forbids.
            # Fail closed instead: the empty sets match nothing, which is the
            # same value the `isolated` (incognito) branch above returns, so
            # every downstream admission point already handles it.
            # Reached only when a scope EXISTS (`if not scope` returns above),
            # so this never turns an ordinary `auto` query into a dead one.
            self._leg_degraded("scope.codex_sets", err)
            self._scope_resolution_failed = True
            return set(), set()

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
        # G36: this runs at the top of every entry point (retrieve,
        # _wide_net_fallback, _procedural_lookup), so it is where the
        # fail-closed flag is cleared for a fresh resolution.
        self._scope_resolution_failed = False
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
        except Exception as err:
            # G36: clearing the deny sets here meant a failure to compute what
            # the user EXCLUDED silently un-excluded it — the graph then served
            # exactly the conversations they asked it not to read. Reached only
            # when an exclusion list exists (the `if not excluded` return
            # above), so failing closed costs nothing in the common case:
            # _codex_scope_sets sees the flag and scopes the codex and
            # procedural legs to nothing rather than to everything.
            self._leg_degraded("scope.exclusions", err)
            self._denied_batch_ids = set()
            self._denied_entity_ids = set()
            self._scope_resolution_failed = True

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
                ).limit(settings.codex_entity_payload_match_limit).all()
                for ent in rows:
                    scored[ent.id] = (scored.get(ent.id, (0, ent))[0] + 1, ent)
            ranked = sorted(scored.values(), key=lambda t: t[0], reverse=True)
            return [ent for hits, ent in ranked[:2] if hits >= 1]
        except Exception as err:
            self._leg_degraded("codex.payload_match", err)
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

    def _prime_edge_times(self, edges):
        """Fetch source dates in batches, for edges already filtered by scope."""
        cache = getattr(self, "_source_times", None)
        if cache is None:
            cache = self._source_times = {}
        batches = {e.source_batch for e in edges if getattr(e, "source_batch", None)
                   and str(e.source_batch) not in cache}
        if not batches:
            return
        rows = self.db.query(EpisodicMemory.batch_id, EpisodicMemory.timestamp,
                             EpisodicMemory.ts_provenance, EpisodicMemory.is_private).filter(
            EpisodicMemory.batch_id.in_(batches)).all()
        missing = batches - {row.batch_id for row in rows}
        if missing:
            rows += self.db.query(ColdStorage.batch_id, ColdStorage.timestamp,
                                 ColdStorage.ts_provenance, ColdStorage.is_private).filter(
                ColdStorage.batch_id.in_(missing)).all()
        private = getattr(self, "_private_source_batches", None)
        if private is None:
            private = self._private_source_batches = set()
        private.update(str(b) for b, _, _, is_private in rows if is_private)
        cache.update({str(b): None for b in batches})
        cache.update({str(b): (stamp, provenance) for b, stamp, provenance, _ in rows})

    def _fact_line(self, src, edge, tgt) -> str:
        """Separate source-recorded time from when ICE learned the claim."""
        self._prime_edge_times([edge])
        if str(getattr(edge, "source_batch", None)) in getattr(self, "_private_source_batches", set()):
            return ""
        source = self._source_times.get(str(getattr(edge, "source_batch", None)))
        dates = []
        if source:
            dates.append(recorded_stamp(*source).strip().strip("[]"))
        else:
            dates.append("source time unknown")
        learned = getattr(edge, "learned_at", None)
        valid = getattr(edge, "valid_from", None)
        if learned:
            dates.append(f"learned: {format_time(learned)}")
        if valid and format_time(valid) != format_time(learned):
            dates.append(f"recorded valid from: {format_time(valid)}")
        linked = self.db.query(CodexClaim).join(CodexClaimLink,
            CodexClaimLink.claim_id == CodexClaim.id).filter(
                CodexClaimLink.edge_id == edge.id,
                CodexClaim.source_batch == edge.source_batch).order_by(
                    CodexClaim.start, CodexClaim.id).all() if getattr(edge, "id", None) and self.db is not None else []
        if linked:
            lines = []
            for claim in linked:
                original = source_for_claim(self.db, claim)
                if original is None or original.is_private or not excerpt_is_current(original, claim):
                    continue
                lines.append(f"[Source excerpt; speaker: {claim.role}; {'; '.join(dates)}] "
                             f"{claim_representation(claim)}")
            return "\n".join(dict.fromkeys(lines))
        rel = f"NOT {edge.relation}" if getattr(edge, "negated", False) else edge.relation
        return (f"[Unverified graph relation: {src.canonical_name} --{rel}--> {tgt.canonical_name}"
                f" ({'; '.join(dates)})]")

    def _relation_fit(self, relations: List[str], prompt_embedding):
        """G34: score each of *relations* against the prompt, and report how
        sharply the best one stands out. Returns ``(scores, fit)``.

        This is the inverted question. The old detector asked "which of the 197
        vocabulary glosses is this prompt near?" and could not be made to answer
        — measured 2026-08-11, three candidate fixes, none separating, because
        absolute cosine is anti-correlated with relational content ("ok" clears
        197/197 at 0.844; "who inspired Kael" clears 43 at 0.575). Asking instead
        which of an anchor's OWN handful of relations the prompt points at scored
        93.2% top-1 against 8.0% for strength ordering, at a 10-edge anchor.

        It works because it never has to say "no". Ranking a set we are already
        rendering is a strictly weaker claim than judging relational intent.

        ``fit`` — best minus mean across *relations* — is the one place a "no"
        is expressed, and it is expressed as a continuous zero rather than a
        gate: a contentless prompt is uniformly near everything, so the spread
        collapses and the caller's bonus vanishes without any threshold deciding
        that it should.
        """
        if not relations or prompt_embedding is None:
            return {}, 0.0
        try:
            vocab, embs = self._relation_gloss_cache()
            index = {r: i for i, r in enumerate(vocab)}
            known = [r for r in relations if r in index]
            if len(known) < 2:
                # Ranking one item is theatre, and the spread of one is zero.
                return {}, 0.0
            # G41: fancy-index the cached array and take one matmul. Building an
            # intermediate list here would reintroduce the marshalling cost that
            # was the whole bill (see _relation_gloss_cache).
            sims = embs[[index[r] for r in known]] @ np.asarray(
                prompt_embedding, dtype=np.float64)
            scores = dict(zip(known, (float(s) for s in sims)))
            vals = list(scores.values())
            fit = max(vals) - (sum(vals) / len(vals))
            return scores, max(0.0, min(1.0, fit))
        except Exception as err:
            self._leg_degraded("codex.relation_fit", err)
            return {}, 0.0

    def _relation_facts(self, matched, prompt_embedding, allowed_batch_ids):
        """A4: surface an anchor's explicit edge facts, ordered by how well each
        edge's relation answers the question. Returns ``(lines, edges, fit)``.

        G34 inverted this. It used to take a list of relations detected from the
        prompt and filter edges to them — but with ~197 relations detected on
        every prompt that filter was a **no-op**, so the leg returned the
        anchor's strongest edges regardless of what was asked, and A4's
        documented "entity ∩ relation joint hit" was never actually a joint hit.
        Now every valid edge of the anchor is a candidate and the *ordering*
        carries the relation signal.

        The candidate pool stays bounded by strength (G35's concern: a hub
        entity has hundreds of edges), but at a multiple of the output size, so
        strength decides only who competes and fit decides who wins.
        """
        if not matched:
            return [], [], 0.0
        matched_ids = [e.id for e in matched]
        q = self.db.query(CodexEdge).filter(
            *self._edge_valid_filters(),
            ((CodexEdge.source_id.in_(matched_ids)) | (CodexEdge.target_id.in_(matched_ids)))
        )
        if allowed_batch_ids is not None:
            q = q.filter(CodexEdge.source_batch.in_(allowed_batch_ids))
        if self._denied_batch_ids:   # C6 exclusion
            q = q.filter(CodexEdge.source_batch.notin_(self._denied_batch_ids))
        pool = q.order_by(CodexEdge.strength.desc()).limit(
            settings.codex_entity_edge_limit
            * settings.codex_relation_pool_multiplier).all()
        # A negative source statement can answer a question; only navigation
        # treats negation as a reason not to walk the relationship.
        pool = [e for e in pool if self._edge_trust(e) >= settings.codex_direct_trust_floor]
        if not pool:
            return [], [], 0.0

        scores, fit = self._relation_fit(
            sorted({e.relation for e in pool}), prompt_embedding)
        # No usable scores ⇒ keep the strength order the pool already carries.
        if scores:
            pool.sort(key=lambda e: (scores.get(e.relation, 0.0), self._edge_trust(e)),
                      reverse=True)

        self._prime_edge_times(pool[:settings.codex_entity_edge_limit])
        lines, fact_edges = [], []
        for edge in pool[:settings.codex_entity_edge_limit]:
            src = self.db.query(CodexEntity).get(edge.source_id)
            tgt = self.db.query(CodexEntity).get(edge.target_id)
            if src and tgt:
                line = self._fact_line(src, edge, tgt)
                if line:
                    lines.append(line)
                    fact_edges.append(edge)
        return lines, fact_edges, fit

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
                for ent in q.limit(settings.codex_enum_entity_limit).all():
                    if allowed_entity_ids is not None and ent.id not in allowed_entity_ids:
                        continue
                    if ent.id in self._denied_entity_ids:   # C6 exclusion
                        continue
                    if ent.id not in seen_entities and ent.context_payload:
                        seen_entities.add(ent.id)
                        texts, rendered = [], []
                        self._traverse_graph(ent, 0, 0, set(), texts,
                            allowed_entity_ids=allowed_entity_ids, allowed_batch_ids=allowed_batch_ids,
                            rendered_edges=rendered)
                        if texts:
                            t = "\n\n".join(texts)
                            fragments.append(ContextFragment(text=t, source_type="codex", score=1.0,
                                token_count=count_tokens(t),
                                origin_batch_ids=tuple({str(e.source_batch) for e in rendered}),
                                origin_edge_ids=tuple({str(e.id) for e in rendered})))
            # (b) relation-driven facts: "who inspired ..." → inspired_by edges,
            #     grouped into a single facts fragment (they're a list answer).
            fact_lines = []
            fact_batches: list = []
            fact_ids: list = []
            if relations:
                q = self.db.query(CodexEdge).filter(
                    *self._edge_valid_filters(),
                    CodexEdge.relation.in_(relations))
                if allowed_batch_ids is not None:
                    q = q.filter(CodexEdge.source_batch.in_(allowed_batch_ids))
                if self._denied_batch_ids:   # C6 exclusion
                    q = q.filter(CodexEdge.source_batch.notin_(self._denied_batch_ids))
                edges = q.order_by(CodexEdge.strength.desc()).limit(settings.codex_enum_edge_limit).all()
                self._prime_edge_times(edges)
                for edge in edges:
                    if self._edge_trust(edge) < settings.codex_direct_trust_floor:
                        continue
                    src = self.db.query(CodexEntity).get(edge.source_id)
                    tgt = self.db.query(CodexEntity).get(edge.target_id)
                    if src and tgt:
                        line = self._fact_line(src, edge, tgt)
                        if not line:
                            continue
                        fact_lines.append(line)
                        fact_ids.append(str(edge.id))
                        if edge.source_batch:
                            fact_batches.append(str(edge.source_batch))
            if fact_lines:
                t = "\n".join(fact_lines)
                fragments.append(ContextFragment(text=t, source_type="codex", score=1.0,
                                                 token_count=count_tokens(t),
                                                 origin_batch_ids=tuple(fact_batches),
                                                 origin_edge_ids=tuple(fact_ids)))
            if fragments:
                logger.info("codex_enumeration", entities=len(seen_entities),
                            facts=len(fact_lines), relations=relations)
            return fragments
        except Exception as err:
            self._leg_degraded("codex.enumeration", err)
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
                if len(terms) >= settings.codex_expansion_max_terms:
                    return terms
        return terms

    # ------------------------------------------------------------------
    # Codex graph traversal (conversation‑scoped, NER‑powered)
    # ------------------------------------------------------------------
    def _codex_claims(self, prompt, prompt_embedding, scope=None, conv_id=None):
        """Search attributed sentences without requiring an entity match."""
        if (not settings.codex_sentence_claims or self._scope_resolution_failed
                or (scope and (scope.get("isolated") or scope.get("incognito")))):
            return []
        if scope and scope.get("cluster_ids_explicit") and not scope.get("cluster_ids"):
            return []
        conv_filter, conv_params, _ = self._conv_scope_filter(
            scope, (scope or {}).get("conversation_id") or conv_id, "e.conversation_id")
        excl_filter, excl_params = self._exclusion_filters(scope, "e.conversation_id", "e.id")
        cluster_filter = self._cluster_filter(scope, "e.id")
        time_filter, _, ts_params, _ = self._timescope_leg_filters("e.")
        params = {"prompt": prompt, "limit": settings.codex_claim_candidate_limit,
                  **conv_params, **excl_params, **ts_params}
        filters = ""
        if scope and "batch_ids" in scope:
            filters += " AND c.source_batch = ANY(:batches)"
            params["batches"] = list(scope["batch_ids"] or [])
        if self._denied_batch_ids:
            filters += " AND c.source_batch <> ALL(:denied)"
            params["denied"] = list(self._denied_batch_ids)
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]
        cold_allowed = not (scope and (scope.get("cluster_ids") or scope.get("cluster_ids_explicit")
                                      or scope.get("exclude_cluster_ids")))
        source_rows = "SELECT id, batch_id, conversation_id, is_private, timestamp FROM episodic_memory"
        if cold_allowed:
            source_rows += (" UNION ALL SELECT cold.id, cold.batch_id, cold.conversation_id, cold.is_private, cold.timestamp "
                            "FROM cold_storage cold WHERE NOT EXISTS (SELECT 1 FROM episodic_memory warm WHERE warm.id = cold.id)")
        base = f"""FROM codex_claims c JOIN ({source_rows}) e ON e.id = c.episodic_id AND e.batch_id = c.source_batch
            WHERE e.is_private = false {conv_filter} {excl_filter} {cluster_filter}
            {time_filter} {filters}"""
        try:
            lexical = text(f"""SELECT c.id {base}
                AND to_tsvector('english', c.text) @@ plainto_tsquery('english', :prompt)
                ORDER BY ts_rank(to_tsvector('english', c.text), plainto_tsquery('english', :prompt)) DESC, c.id
                LIMIT :limit""")
            channels = [list(self.db.execute(lexical, params).scalars())]
            if prompt_embedding is not None:
                semantic = text(f"""SELECT c.id {base} AND c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> :embedding, c.id LIMIT :limit""").bindparams(
                        bindparam("embedding", type_=PgVector))
                channels.append(list(self.db.execute(semantic, {**params, "embedding": prompt_embedding}).scalars()))
            scores = {}
            for channel in channels:
                for rank, ident in enumerate(channel, 1):
                    scores[ident] = scores.get(ident, 0.0) + 1.0 / (settings.retrieval_rrf_k + rank)
            fragments, sources = [], {}
            for ident in sorted(scores, key=lambda k: (-scores[k], str(k)))[:settings.codex_claim_candidate_limit]:
                claim = self.db.get(CodexClaim, ident)
                if claim.source_batch not in sources:
                    sources[claim.source_batch] = source_for_claim(self.db, claim)
                source = sources[claim.source_batch]
                if source is None or not excerpt_is_current(source, claim):
                    logger.warning("claim_source_stale", claim_id=str(ident))
                    continue
                rendered = (f"[Source excerpt; speaker: {claim.role}] "
                            f"{recorded_stamp(source.timestamp, source.ts_provenance)}\n"
                            f"{claim_representation(claim)}")
                fragments.append(ContextFragment(rendered, "codex", scores[ident],
                    count_tokens(rendered), conversation_id=str(source.conversation_id),
                    origin_batch_ids=(str(claim.source_batch),), leg="codex"))
            return fragments
        except Exception as exc:
            self._leg_degraded("codex.claims", exc)
            return []

    def _codex_graph(self, classification, scope: Optional[dict] = None,
                     prompt_embedding=None) -> List[ContextFragment]:
        prompt = classification.prompt
        self._last_matched_entities = []
        # G35: hand the traversal the request's prompt so it can rank its
        # frontier by relation fit rather than by trust alone.
        self._prompt_embedding = prompt_embedding
        entity_strings = extract_entities(prompt, self.embedder)

        matched = []
        if entity_strings:
            matched = (self._match_entities_by_similarity(entity_strings)
                       if self.use_fuzzy_match
                       else self._match_entities_exact(entity_strings))
            if not matched:
                # A4 descriptor fallback: 'main fortress' → payload mentions 'fortress'
                matched = self._match_entities_by_payload(entity_strings)

        # A5: project-scope sets (both None when unscoped).
        allowed_entity_ids, allowed_batch_ids = self._codex_scope_sets(scope)

        if not matched:
            # A4: re-homed MERA — entity-less enumeration ("list all the characters").
            if self.enable_enumeration:
                # G34: _detect_relations is computed HERE, not at the top of the
                # leg. It is now enumeration's only consumer — the anchor path
                # ranks against each anchor's own relations instead — and
                # enumeration additionally needs an explicit cue word, so the
                # 12.2 ms gloss loop went from every prompt to the rare ones
                # that could use it.
                return self._codex_enumeration(
                    prompt, self._detect_relations(prompt, prompt_embedding),
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
            best_fit = 0.0   # G34: max across anchors, not the last one's
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
                local_texts, direct_edges, rendered_edges = [], [], []
                self._traverse_graph(anchor, 0, settings.codex_max_depth, set(),
                                     local_texts, direct_edges,
                                     allowed_entity_ids, allowed_batch_ids,
                                     exclude_ids=anchor_ids - {anchor.id}, rendered_edges=rendered_edges)
                if not local_texts:
                    continue
                # Per-anchor score from THIS anchor's direct-edge trust (A3).
                mean_trust = (sum(self._edge_trust(e) for e in direct_edges) / len(direct_edges)
                              if direct_edges else 0.0)
                score = 1.0 + min(0.5, 0.25 * mean_trust)
                # A4/G34: this anchor's facts, ordered by how well each edge's
                # relation answers the question. Runs unconditionally now — the
                # old `if detected_relations:` gate was always true (the detector
                # fired on every prompt), so it decided nothing while looking
                # like precision.
                fact_lines, fact_edges, fit = self._relation_facts(
                    [anchor], prompt_embedding, allowed_batch_ids)
                if fact_lines:
                    existing_text = "\n\n".join(local_texts)
                    local_texts.extend(line for line in fact_lines if line not in existing_text)
                    direct_edges.extend(fact_edges)
                    # G34: proportional to how sharply one relation stood out,
                    # replacing a flat +0.25 that was applied on every prompt
                    # because its condition could never be false. A contentless
                    # prompt scores every relation alike, so fit collapses to 0
                    # and this adds nothing — without a threshold saying so.
                    score += settings.codex_relation_fit_weight * fit
                    best_fit = max(best_fit, fit)
                text = "\n\n".join(local_texts)
                fragments.append(ContextFragment(
                    text=text, source_type="codex", score=score,
                    token_count=count_tokens(text),
                    # G48: the turns these facts were extracted FROM. This is the
                    # path that actually produces the codex fragments retrieval
                    # returns — the enumeration path below is a different one —
                    # and without this a codex fragment can never be credited to
                    # a gold turn, which is why recall has only ever scored the
                    # episodic leg.
                    origin_batch_ids=tuple(
                        {str(e.source_batch) for e in rendered_edges + fact_edges if e.source_batch}),
                    origin_edge_ids=tuple({str(e.id) for e in rendered_edges + fact_edges})))

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

            if best_fit > 0.0:
                # G34: reports the measured fit rather than a list of "detected"
                # relations. The old field listed whatever cleared the floor —
                # typically all 197 — so it logged noise as though it were a hit.
                logger.info("codex_relation_overlap", best_fit=round(best_fit, 4),
                            fragments=len(fragments))
            return fragments + timeline_frags
        except Exception as err:
            self._leg_degraded("codex", err)
            return []

    def _edge_trust(self, edge) -> float:
        """Source quality gates entry; bounded retention/recency rank candidates.

        This ranking score is not a calibrated probability of truth. Legacy NULL
        confidence retains its historical behavior pending evidence migration.
        """
        conf = edge.extraction_confidence if edge.extraction_confidence is not None else 1.0
        # Usage must never lift a low-support assertion across the quality gate.
        if conf < settings.codex_direct_trust_floor:
            return conf
        retention = min(1.0, max(0.0, edge.strength or 0.0) / settings.codex_retention_cap)
        base = conf * (1.0 + settings.codex_retention_rank_weight * retention)
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
                base *= 1.0 + settings.codex_recency_boost * math.exp(-age_days / settings.codex_recency_tau_days)
            except Exception as err:
                self._leg_degraded("codex.edge_recency", err)
        return base

    def _render_codex_entity(self, entity, depth, out_edges, in_edges,
                             allowed_batch_ids, context_texts, rendered_edges=None):
        """Rich source notes at anchors; compact navigation at deeper nodes."""
        derived = getattr(entity, "source", None) not in (None, "conversation")
        if derived and depth == 0:
            if entity.context_payload:
                context_texts.append(f"[Entity: {entity.canonical_name}]\n{entity.context_payload}")
            return
        if depth > 0:
            context_texts.append(f"[{entity.canonical_name} ({getattr(entity, 'entity_type', None) or 'entity'})]")
            return
        lines = []
        self._prime_edge_times(out_edges + in_edges)
        has_private_source = any(str(e.source_batch) in self._private_source_batches
                                 for e in out_edges + in_edges) if hasattr(self, "_private_source_batches") else False
        unscoped_current = (allowed_batch_ids is None and not self._denied_batch_ids
                            and not has_private_source and self._active_timescope.mode == "current")
        if unscoped_current:
            note = (entity.description or "").strip()
            if not note and not out_edges and not in_edges:
                note = (entity.context_payload or "").strip()
            if note:
                lines.append(f"[Unverified stored note] {note}")
        edges = list({e.id: e for e in out_edges + in_edges}.values())
        edges.sort(key=lambda e: (-self._edge_trust(e), str(e.id)))
        self._prime_edge_times(edges[:settings.codex_entity_edge_limit])
        for edge in edges[:settings.codex_entity_edge_limit]:
            if self._edge_trust(edge) < settings.codex_direct_trust_floor:
                continue
            source = self.db.get(CodexEntity, edge.source_id)
            target = self.db.get(CodexEntity, edge.target_id)
            if source and target:
                line = self._fact_line(source, edge, target)
                if line:
                    lines.append(line)
                    if rendered_edges is not None:
                        rendered_edges.append(edge)
        if lines:
            context_texts.append(f"[Entity: {entity.canonical_name}]\n" + "\n".join(dict.fromkeys(lines)))

    def _traverse_graph(self, entity, depth, max_depth, visited, context_texts, anchor_edges=None,
                        allowed_entity_ids=None, allowed_batch_ids=None, exclude_ids=None,
                        rendered_edges=None):
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
                                  allowed_batch_ids, context_texts, rendered_edges)

        # A7.2: traverse both directions (into the target of outgoing edges and
        # the source of incoming ones), trust-gated and scope-bounded as before.
        # G35: collect the surviving frontier, then expand only the best
        # `codex_max_fanout` of it. Two changes here, both load-bearing:
        #   * a cap exists at all. Expansion was unbounded, so the size and the
        #     cost of a codex fragment were set by graph topology — how many
        #     edges the anchor happens to have — instead of by relevance to the
        #     question. Invisible on a sparse store, unavoidable on a hub.
        #   * candidates from BOTH directions now compete in ONE ranking keyed
        #     on _edge_trust. The old loop walked every out_edge and then every
        #     in_edge, each ordered by raw `strength` in SQL — a different
        #     ordering from the _edge_trust that gates them, and one under which
        #     a well-connected outgoing set starves every backlink.
        frontier = []
        for edge, other_id in ([(e, e.target_id) for e in out_edges] +
                               [(e, e.source_id) for e in in_edges]):
            # A8: a negated edge ("X does NOT use Y") is a stored fact, not a
            # navigable link — it's rendered in the note (Negations section) but
            # we don't walk into it as if the relationship held.
            if getattr(edge, "negated", False):
                continue
            trust = self._edge_trust(edge)
            if depth == 0:
                # Direct-edge trust gates candidate expansion, never promotion.
                if trust < settings.codex_direct_trust_floor:
                    continue
                # Candidate lineage/trust summary, not proof of final exposure.
                if anchor_edges is not None:
                    anchor_edges.append(edge)
            # A3: trust-gate deep hops — weak/decayed edges don't expand the frontier.
            elif trust < settings.codex_deep_strength_floor:
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
            frontier.append((trust, other_id, edge.relation))

        # G35's second half, unblocked by G34: rank the frontier by how well each
        # edge's RELATION answers the question, then by trust. Which edges we
        # walk into was previously decided by trust alone — i.e. by how often the
        # graph had seen them, never by what was asked. Relation fit leads
        # because trust is the tiebreak *within* an equally relevant set, not a
        # substitute for relevance.
        #
        # When there is no prompt embedding, or fewer than two distinct relations
        # to choose between, `scores` is empty and this degrades to exactly the
        # trust ordering it replaces — the same fallback the fact path takes.
        scores, _ = self._relation_fit(
            sorted({r for _, _, r in frontier}),
            getattr(self, "_prompt_embedding", None))
        # Highest first. A node already in `visited`, or one that fails the
        # visibility check below, still consumes a slot — the cap bounds the
        # WORK done per node, which is the property being bought.
        frontier.sort(key=lambda c: (scores.get(c[2], 0.0), c[0]), reverse=True)
        for _trust, other_id, _rel in frontier[:settings.codex_max_fanout]:
            other = self.db.query(CodexEntity).get(other_id)
            # E1b (D3): derived entities stay invisible outside their project
            # even when reachable through a conversation edge.
            if other and self._entity_visible(other):
                self._traverse_graph(other, depth + 1, max_depth, visited,
                                     context_texts, anchor_edges,
                                     allowed_entity_ids, allowed_batch_ids, exclude_ids, rendered_edges)

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
            "proc_limit": settings.retrieval_procedural_limit,
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
            LIMIT :proc_limit
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
                    token_count=count_tokens(pattern.pattern_description),
                    # G48: the turns this habit was inferred from. Already
                    # computed above for scoping and then thrown away, so a
                    # procedural fragment could never be credited to a turn.
                    origin_batch_ids=tuple(str(b) for b in source_batches),
                ))
            return fragments
        except Exception as err:
            # The proof case for G36: this exact handler hid a psycopg3 bind
            # mismatch (`vector <=> double precision[]`) for months — the leg
            # returned [] on every call and the intent whitelist above it made
            # the silence look like a design choice.
            self._leg_degraded("procedural", err)
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
                              include_cross: bool = True, scope=None,
                              search_conv_id=None) -> List[ContextFragment]:
        # T3 (D14): skipped under any non-current mode — a summary's created_at
        # is long after its content's period, which is underivable without a
        # turn-index→timestamp join; serving it under a window would mislead.
        if self._active_timescope.mode != "current":
            return []
        source_scope, source_params, _ = self._conv_scope_filter(
            scope, search_conv_id, column="covered_source.conversation_id")
        source_exclusions, exclusion_params = self._exclusion_filters(
            scope, conv_column="covered_source.conversation_id",
            id_column="covered_source.id")
        source_clusters = self._cluster_filter(scope, id_column="covered_source.id")
        # The active conversation identity has a separate SQL name; the scope
        # helper may bind :conv_id to a different, explicitly searched conversation.
        scope_params = {**source_params, **exclusion_params}
        if scope and scope.get("cluster_ids"):
            scope_params["cluster_ids"] = scope["cluster_ids"]
        source_allowed = f"TRUE {source_scope} {source_exclusions} {source_clusters}"
        if scope and scope.get("batch_ids") is not None:
            source_allowed += " AND covered_source.batch_id = ANY(:summary_batch_ids)"
            scope_params["summary_batch_ids"] = [str(b) for b in scope["batch_ids"]]
        fragments: List[ContextFragment] = []
        # Half 1 (as built): this conversation's batch summaries.
        if conv_id:
            try:
                query = text(f"""
                    SELECT bs.id, bs.conversation_id, bs.source_manifest, bs.summary_text, bs.created_at,
                           1 - (bs.embedding <=> :prompt_embedding) as score,
                           (SELECT array_agg(em.batch_id::text)
                              FROM episodic_memory em
                             WHERE em.batch_summary_id = bs.id) AS covered
                    FROM batch_summaries bs
                    WHERE bs.conversation_id = :own_conv_id
                      AND bs.embedding IS NOT NULL
                      AND EXISTS (SELECT 1 FROM episodic_memory em
                                  WHERE em.batch_summary_id = bs.id)
                      AND NOT EXISTS (
                          SELECT 1 FROM episodic_memory covered_source
                          WHERE covered_source.batch_summary_id = bs.id
                            AND NOT ({source_allowed}))
                    ORDER BY score DESC
                    LIMIT :bs_limit
                """).bindparams(bindparam("prompt_embedding", type_=PgVector))
                rows = self.db.execute(query, {
                    "prompt_embedding": prompt_embedding, "own_conv_id": conv_id,
                    **scope_params,
                    "bs_limit": settings.retrieval_batch_summary_limit}).fetchall()
                # T1: summaries are written long after the turns they compress,
                # so they get a "[summary, <created>]" prefix, not a turn date.
                #
                # ⚑ G48/G53: `origin_batch_ids` carries the turns this summary
                # compresses, taken from `episodic_memory.batch_summary_id`.
                # Without it a summary fragment is structurally uncreditable in
                # every recall-style metric in the repo — it can be returned,
                # spend budget, and answer the question, and still score zero,
                # which is the TRAPS #32 blindness one leg further on. The link
                # already existed as an FK; nothing read it.
                fragments += [ContextFragment(
                    text=(rendered := f"[summary created: {format_time(r.created_at)}] " + r.summary_text),
                    source_type="batch_summary",
                    score=r.score,
                    token_count=count_tokens(rendered),
                    origin_batch_ids=tuple(r.covered or ()),
                    conversation_id=str(conv_id),
                ) for r in rows if batch_snapshot_readable(self.db, r)]
            except Exception as err:
                self._leg_degraded("batch_summary.own", err)
        # Half 2 (C4 D3b): OTHER conversations' evolving whole-conversation
        # summaries — cross-conversation overview hits. The active
        # conversation's own summary is excluded (the assembler injects it —
        # double-inject trap) and private conversations never leave their
        # scope (memory_scope_type join). Skipped entirely under incognito
        # (a user-global read — G16 "read nothing").
        if not include_cross:
            return fragments
        try:
            query = text(f"""
                SELECT s.summary_text, s.updated_at, s.source_manifest, s.conversation_id,
                       1 - (s.embedding <=> :prompt_embedding) as score
                FROM conversation_summaries s
                JOIN conversations c ON c.id = s.conversation_id
                WHERE s.embedding IS NOT NULL
                  AND c.memory_scope_type != 'none'
                  AND NOT EXISTS (
                      SELECT 1 FROM episodic_memory private_source
                      WHERE private_source.conversation_id = s.conversation_id
                        AND private_source.is_private = TRUE)
                  AND (CAST(:own_conv_id AS uuid) IS NULL
                       OR s.conversation_id != CAST(:own_conv_id AS uuid))
                  AND NOT EXISTS (
                      SELECT 1 FROM episodic_memory covered_source
                      WHERE covered_source.conversation_id = s.conversation_id
                        AND s.source_manifest @> jsonb_build_object(
                            'sources', jsonb_build_array(jsonb_build_object(
                                'id', covered_source.id::text)))
                        AND NOT ({source_allowed}))
                ORDER BY score DESC
                LIMIT :cs_limit
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = self.db.execute(query, {
                "prompt_embedding": prompt_embedding,
                "own_conv_id": conv_id,
                **scope_params,
                "cs_limit": settings.retrieval_conversation_summary_limit,
            }).fetchall()
            fragments += [ContextFragment(
                text=(rendered := f"[conversation summary updated: {format_time(r.updated_at)}] " + r.summary_text),
                source_type="batch_summary",
                score=r.score,
                token_count=count_tokens(rendered),
                conversation_id=str(r.conversation_id),
                origin_batch_ids=tuple(item['batch_id'] for item in r.source_manifest['sources'])
            ) for r in rows if summary_snapshot_readable(self.db, r)]
        except Exception as err:
            self._leg_degraded("batch_summary.cross", err)
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
        excl_filter, excl_params = self._exclusion_filters(scope, id_column=None)
        if scope and (scope.get("cluster_ids") or scope.get("exclude_cluster_ids")):
            excl_filter += " AND cluster_ids IS NOT NULL"
        if scope and scope.get("cluster_ids"):
            excl_filter += " AND (cardinality(cluster_ids) = 0 OR cluster_ids && CAST(:cold_clusters AS uuid[]))"
            excl_params["cold_clusters"] = [str(c) for c in scope["cluster_ids"]]
        if scope and scope.get("exclude_cluster_ids"):
            excl_filter += " AND NOT (cluster_ids && CAST(:cold_excluded_clusters AS uuid[]))"
            excl_params["cold_excluded_clusters"] = [str(c) for c in scope["exclude_cluster_ids"]]
        if scope and scope.get("batch_ids") is not None:
            excl_filter += " AND batch_id = ANY(CAST(:cold_batches AS uuid[]))"
            excl_params["cold_batches"] = [str(b) for b in scope["batch_ids"]]
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
                   topic_tags, timestamp, is_private, embedding, source_spans, ts_provenance,
                   summary_coverage, representation_verification, abstract_text, lossless_flag, inject_raw, session_id, intent_tags, context_reliance, idempotency_key, cluster_id, cluster_ids
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
            self._leg_degraded("cold", err)
            return []

        fragments = []
        for row in rows:
            body = choose_representation(row)[0] or ""
            stamp = recorded_stamp(row.timestamp, getattr(row, "ts_provenance", None)) if row.timestamp else ""
            text_ = stamp + body
            fragments.append(ContextFragment(
                text=text_, source_type="episodic", score=0.6,
                covers_entire_source=body == row.raw_text,
                token_count=count_tokens(text_),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None,
            ))
            try:
                fragments.extend(self._cold_chunk_candidates(row, prompt_keywords, prompt_embedding))
            except Exception as err:
                self._leg_degraded("cold.chunks", err)
            self._cold_hits[str(row.id)] = row
        if fragments:
            logger.info("cold_leg_hits", count=len(fragments), mode=ts.mode)
        return fragments

    def _cold_chunk_candidates(self, parent, keywords, prompt_embedding):
        """Only called for already eligible cold parents; text stays complete."""
        if prompt_embedding is not None:
            query = text("""
                SELECT chunk_text, 1 - (embedding <=> :probe) AS score
                FROM cold_chunks WHERE turn_id = :id AND embedding IS NOT NULL
                ORDER BY embedding <=> :probe, chunk_index, id LIMIT 3
            """).bindparams(bindparam("probe", type_=PgVector))
            rows = self.db.execute(query, {"id": parent.id, "probe": prompt_embedding}).all()
        else:
            rows = self.db.execute(text("""
                SELECT chunk_text, 0.0 AS score FROM cold_chunks
                WHERE turn_id = :id ORDER BY chunk_index, id
            """), {"id": parent.id}).all()
            terms = {str(k).casefold() for k in keywords if k}
            rows = sorted(rows, key=lambda r: -sum(k in r.chunk_text.casefold() for k in terms))[:3]
        stamp = recorded_stamp(parent.timestamp, getattr(parent, "ts_provenance", None))
        return [ContextFragment(
            text=(rendered := stamp + row.chunk_text), source_type="episodic",
            score=float(row.score or 0), token_count=count_tokens(rendered),
            source_batch_id=str(parent.id),
            conversation_id=str(parent.conversation_id) if parent.conversation_id else None,
            leg="cold_chunk", covers_entire_source=False,
        ) for row in rows]

    def _resurrect_cold_hits(self, final: List[ContextFragment]):
        """D-U1 second chance: a cold memory selected by retrieval budgeting
        (survived the budget — never mere candidacy) moves back into
        episodic_memory on probation: ORIGINAL timestamp (it's an old memory),
        decay just above the archive line — unengaged, normal decay re-archives
        it within days; engaged, write-on-read strengthening saves it. Never
        restored at full strength. Legacy rows (NULL conversation_id) are
        cite-only. Runs AFTER _strengthen_retrieved (the row isn't episodic
        yet during that pass), so probation starts exactly at 0.12."""
        if not settings.retrieval_strengthen_writes or not self._cold_hits:
            return
        seen = set()
        for f in final:
            if f.source_batch_id in seen:
                continue
            seen.add(f.source_batch_id)
            row = self._cold_hits.get(f.source_batch_id) if f.source_batch_id else None
            if row is None:
                continue
            if row.conversation_id is None:
                logger.info("cold_cited_only", cold_id=str(row.id))
                continue
            try:
                # Restore the representation that found this memory, not a
                # newly truncated summary embedding. Legacy NULL stays NULL.
                emb = getattr(row, "embedding", None)
                if isinstance(emb, str):
                    emb = [float(value) for value in emb.strip("[]").split(",")]
                elif hasattr(emb, "tolist"):
                    emb = emb.tolist()
                if emb is None:
                    logger.warning("cold_restored_without_vector", cold_id=str(row.id),
                                   reason="archived_vector_missing")
                res = self.db.execute(text("""
                    INSERT INTO episodic_memory
                        (id, conversation_id, batch_id, timestamp, topic_tags,
                         intent_tags, context_reliance, raw_text, summary_text,
                         embedding, decay_score, access_count, is_archived,
                         is_private, inject_raw, idempotency_key, source_spans, ts_provenance,
                         summary_coverage, representation_verification, abstract_text,
                         lossless_flag, session_id, cluster_id)
                    VALUES (:id, :conv, :batch, :ts, :tags, :itags,
                            :context_reliance, :raw, :summary, :emb, :score,
                            1, FALSE, :priv, :inject_raw, :ikey, :source_spans, :ts_provenance,
                            :summary_coverage, :representation_verification, :abstract_text,
                            :lossless_flag, :session_id,
                            (SELECT id FROM context_clusters WHERE id = :primary_cluster))
                    ON CONFLICT (id) DO NOTHING
                """).bindparams(bindparam("emb", type_=PgVector),
                                 bindparam("source_spans", type_=JSONB),
                                 bindparam("representation_verification", type_=JSONB)), {
                    "id": row.id, "conv": row.conversation_id,
                    "primary_cluster": getattr(row, "cluster_id", None),
                    "batch": row.batch_id or uuid.uuid4(), "ts": row.timestamp,
                    "tags": list(row.topic_tags or []),
                    "itags": list(getattr(row, "intent_tags", None) or []),
                    "context_reliance": getattr(row, "context_reliance", None) or "Long_Term_Memory",
                    "inject_raw": (getattr(row, "inject_raw", None)
                                   if getattr(row, "inject_raw", None) is not None else True),
                    **{key: getattr(row, key, None) for key in (
                        "summary_coverage", "representation_verification", "abstract_text",
                        "lossless_flag", "session_id")},
                    "raw": row.raw_text, "summary": row.summary_text,
                    "source_spans": getattr(row, "source_spans", None),
                    "ts_provenance": getattr(row, "ts_provenance", None) or "unknown",
                    "emb": emb, "score": settings.timescope_probation_score,
                    "priv": row.is_private,
                    "ikey": getattr(row, "idempotency_key", None) or f"cold-resurrect-{row.id}",
                })
                if res.rowcount == 0:
                    # id somehow still live in episodic — keep the cold row.
                    logger.info("cold_resurrect_conflict", cold_id=str(row.id))
                else:
                    self.db.execute(text("""
                        INSERT INTO episodic_chunks (id, turn_id, chunk_index, chunk_text, embedding)
                        SELECT id, turn_id, chunk_index, chunk_text, embedding
                        FROM cold_chunks WHERE turn_id = :id
                    """), {"id": row.id})
                    self.db.execute(text("""
                        INSERT INTO episodic_cluster_links (episodic_id, cluster_id)
                        SELECT :id, id FROM context_clusters
                        WHERE id = ANY(CAST(:clusters AS uuid[]))
                        ON CONFLICT DO NOTHING
                    """), {"id": row.id, "clusters": [str(c) for c in
                             (getattr(row, "cluster_ids", None) or [])]})
                    self.db.execute(text("DELETE FROM cold_storage WHERE id = :id"),
                                    {"id": row.id})
                    logger.info("cold_resurrected", episodic_id=str(row.id),
                                decay_score=settings.timescope_probation_score)
                self.db.commit()
            except Exception as err:
                self._leg_degraded(f"cold.resurrect[{row.id}]", err)

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
        except Exception as err:
            self._leg_degraded("timescope.era_probe", err)
        logger.info("timescope_empty_window", t0=str(ts.t0.date()), t1=str(ts.t1.date()))
        final.append(ContextFragment(
            text=note, source_type="episodic", score=5.0,
            token_count=count_tokens(note)))
        return final

    # ------------------------------------------------------------------
    # RRF fusion, diversification, dedup, token budget
    # ------------------------------------------------------------------
    def _apply_rrf(self, legs: Dict[str, List[ContextFragment]], alpha_map: Dict[str, float] = None, k: int = None) -> List[ContextFragment]:
        from dataclasses import replace
        if alpha_map is None:
            alpha_map = {}
        if k is None:
            k = settings.retrieval_rrf_k

        rrf_scores: Dict[str, float] = {}
        fragment_registry: Dict[str, ContextFragment] = {}

        # C16: the leg name is right here and was being thrown away. Stamp it —
        # first leg to produce the fragment wins, which is also the
        # highest-ranked contribution for that fragment within that leg. This
        # is the only place in the pipeline that knows which of the eight
        # mechanisms actually found a given piece of memory.
        # ⚑ ALL producing legs, not the first — G46 asked for this and only the
        # first half landed. `legs` is a dict and `bm25` is first in it, so a
        # fragment found by bm25 AND vector was stamped `bm25` and the vector
        # leg read as dead. That near-miss was reported once as "the vector leg
        # is dead"; it is not — it returns 87-89 candidates on its own. Legs are
        # joined with "+" (`bm25+vector`) so the field stays a string and the
        # overlap is visible rather than silently resolved to whichever leg the
        # dict happened to yield first.
        frag_legs: Dict[str, set] = {}

        for leg_name, fragments in legs.items():
            weight = alpha_map.get(leg_name, 1.0)
            fragments.sort(key=lambda x: x.score, reverse=True)
            for rank, frag in enumerate(fragments, start=1):
                frag_hash = hashlib.sha256(frag.text.encode('utf-8')).hexdigest()
                if frag_hash not in fragment_registry:
                    fragment_registry[frag_hash] = frag
                frag_legs.setdefault(frag_hash, set()).add(leg_name)
                rrf_scores[frag_hash] = rrf_scores.get(frag_hash, 0.0) + (weight / (k + rank))

        fused = []
        for frag_hash, score in rrf_scores.items():
            original = fragment_registry[frag_hash]
            produced = "+".join(sorted(frag_legs.get(frag_hash, ())))
            new_frag = replace(original, score=score,
                               leg=original.leg or produced or None)
            fused.append(new_frag)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def _collapse_provenance(self, fragments):
        """Limit per-source survivors at admission, never pre-budget candidates.

        Sharing a source ID does not imply identical information: disjoint chunks
        can both matter. This is a configured diversity bound, not lossless dedup.
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
            self._leg_degraded("coverage.vectors", exc)
            return {}
        out = {}
        unparsed = 0
        last_err = None
        for r in rows:
            vec = r.embedding
            if isinstance(vec, str):          # pgvector text form
                try:
                    vec = [float(x) for x in vec.strip("[]").split(",")]
                except ValueError as err:
                    # G36: counted, not logged per row. A vector that will not
                    # parse is a SHAPE problem, so it hits every row — one line
                    # each would be a flood saying one thing.
                    unparsed += 1
                    last_err = err
                    continue
            out[str(r.id)] = vec
        if unparsed:
            self._leg_degraded(f"coverage.vector_parse[{unparsed}/{len(rows)}]",
                               last_err)
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

    def _session_diversify(self, fragments, current_id, max_per_conversation=None):
        if max_per_conversation is None:
            max_per_conversation = settings.retrieval_max_per_conversation
        counts: Dict[str, int] = {}
        result = []
        for f in fragments:
            cid = str(f.conversation_id) if f.conversation_id is not None else None
            if not cid:
                result.append(f)
            elif cid == str(current_id):
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

    def _enforce_token_budget(self, fragments, max_tokens=None, *, relevance_order=False,
                              current_conversation_id=None):
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
                                      degrade_text=None, abstract_text=None,
                                      covers_entire_source=False)
            return None

        total, result = 0, []

        def admit(fragment):
            nonlocal total
            fitted = (fragment if fragment.token_count <= max_tokens - total
                      else _degraded(fragment, max_tokens - total))
            if fitted is None:
                return False
            if settings.retrieval_collapse_enabled and fitted.source_batch_id:
                same = [f for f in result if f.source_batch_id == fitted.source_batch_id]
                if same and (fitted.covers_entire_source or
                             any(f.covers_entire_source for f in same)):
                    return False
            trial = result + [fitted]
            if len(self._collapse_provenance(trial)) != len(trial):
                return False
            if len(self._session_diversify(trial, current_conversation_id)) != len(trial):
                return False
            result.append(fitted)
            total += fitted.token_count
            return True

        if relevance_order:
            for f in fragments:
                admit(f)
            self._log_leg_budget_share(result, total, max_tokens)
            return result

        used = set()
        for f in guaranteed:
            admit(f)
            used.add(id(f))

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
            for leg in list(active):
                q = queues[leg]
                admit(q.popleft())
                if not q:
                    active.remove(leg)
            # Every queue advances even if nothing fits this round. A later
            # smaller excerpt can still fit; do not stop at an oversized head.
        self._log_leg_budget_share(result, total, max_tokens)
        return result

    def _log_leg_budget_share(self, fragments, total, max_tokens):
        """G35: report what share of the window each leg actually took.

        The entry's look-ahead asks for this by name, and the reason is that the
        total is the wrong number to watch. The budget already bounds total
        injected tokens, so an over-eager leg never blows the window — it crowds
        the other legs out from *inside* it, which is the harder failure to see
        and the one a total-tokens check cannot show. Phase 2's round-robin makes
        that unlikely; this is what would reveal it happening anyway.

        Emitted at INFO on every retrieval, not sampled: a share that has to be
        reproduced before it can be looked at is a share nobody looks at.
        """
        if not fragments:
            return
        try:
            per_leg = {}
            for f in fragments:
                per_leg[f.source_type] = per_leg.get(f.source_type, 0) + f.token_count
            logger.info(
                "leg_budget_share",
                total_tokens=total,
                max_tokens=max_tokens,
                utilisation=round(total / max_tokens, 3) if max_tokens else None,
                # share of what was SPENT, not of the cap — a leg taking 80% of a
                # barely-used window is not the same problem as taking 80% of a
                # full one, and the raw totals below keep both readable.
                share={leg: round(t / total, 3) for leg, t in per_leg.items()} if total else {},
                tokens=per_leg,
                fragments={leg: sum(1 for f in fragments if f.source_type == leg)
                           for leg in per_leg},
                # G46/C16: `source_type` is the FAMILY (episodic, codex,
                # timeline); `leg` is the leg that actually produced the
                # fragment (bm25 / vector / chunk inside episodic). That
                # distinction is the whole point of C16's attribution work, and
                # its only reader was the coverage block — which is OFF by
                # default, so on a default config no run could say which leg did
                # the episodic work. Reported here instead, where the log
                # already fires on every retrieval, because a measurement gated
                # behind a BEHAVIOUR flag cannot be turned on without changing
                # the thing being measured (coverage also stops retrieval early).
                producing_legs=_count_by(fragments, lambda f: f.leg or f.source_type))
        except Exception as err:
            self._leg_degraded("budget.share_log", err)

    # ------------------------------------------------------------------
    # Strengthening (access count + decay boost)
    # ------------------------------------------------------------------
    def _strengthen_retrieved(self, fragments: List[ContextFragment]):
        # G38/Z1: the write is gated as a whole. decay_strengthen_amount=0 stops
        # the score moving but leaves access_count incrementing, and that alone
        # reorders equal-scoring candidates on the next query — 26/40 identical
        # result sets with it on, 40/40 with it off, on one store and one config.
        if not settings.retrieval_strengthen_writes:
            return
        ids = {uuid.UUID(str(f.source_batch_id)) for f in fragments
               if f.source_type == "episodic" and f.source_batch_id}
        if not ids:
            return
        try:
            self.db.execute(text("""
                UPDATE episodic_memory
                SET access_count = COALESCE(access_count, 0) + 1,
                    decay_score = LEAST(1.0, COALESCE(decay_score, 0.0) + :amount)
                WHERE id = ANY(:ids)
            """), {"ids": list(ids), "amount": settings.decay_strengthen_amount})
            self.db.commit()
        except Exception as err:
            self._leg_degraded("strengthen", err)

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
        cluster_filter = self._cluster_filter(scope)
        # T3: same window/archived/floor rules as the normal legs (the wide
        # net widens ranking, not visibility — and not time either).
        time_filter, archived_filter, ts_params, min_decay = self._timescope_leg_filters()
        try:
            query = text(f"""
                SELECT id, raw_text, summary_text, summary_coverage, representation_verification, source_spans, abstract_text, lossless_flag, inject_raw, conversation_id, is_document, timestamp, ts_provenance,
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
                LIMIT :cand_limit
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            creative = "Creative_&_Media" in (classification.topic_tags or [])
            ts_center, rec_boost, rec_tau = self._recency_params(creative)
            params = {"prompt_embedding": prompt_embedding, "min_decay": min_decay,
                      "cand_limit": settings.retrieval_wide_net_candidate_limit,
                      "recency_boost": rec_boost, "recency_tau": rec_tau,
                      "ts_center": ts_center, **ts_params, **conv_params,
                      **excl_params}
            if scope and scope.get("cluster_ids"):
                params["cluster_ids"] = scope["cluster_ids"]
            rows = self.db.execute(query, params).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic", prompt_text=classification.prompt, classification=classification)
        except Exception as err:
            self._leg_degraded("wide_net", err)
            fragments = []

        fragments.extend(self._vector_chunks(prompt_embedding, scope, scope_conv,
            recency_boost=self._recency_params(
                "Creative_&_Media" in (classification.topic_tags or []))[1]))

        fragments.extend(self._codex_graph(classification, scope,
                                           prompt_embedding=prompt_embedding))

        fragments.extend(self._codex_claims(classification.prompt, prompt_embedding, scope, conversation_id))
        fused = self._apply_rrf({"fallback": fragments}, alpha_map={"fallback": 1.0})
        prompt_keywords = self._extract_prompt_keywords(classification.prompt) if classification.prompt else set()
        fused = self._apply_bonuses(fused, classification, conversation_id, prompt_keywords)
        fused.sort(key=lambda x: x.score, reverse=True)
        fused, reranked = rerank(classification.prompt, fused)
        # C15: dynamic ceiling — a fraction of the (model-aware, C16) retrieval
        # budget with a floor, replacing the hardcoded 2,000 tokens.
        wide_budget = max(settings.retrieval_wide_net_budget_floor,
                          int(self.max_retrieval_tokens * settings.retrieval_wide_net_budget_fraction))
        deduped = self._deduplicate(fused)
        return self._enforce_token_budget(deduped, max_tokens=wide_budget,
                                          relevance_order=reranked,
                                          current_conversation_id=conversation_id)

    # ------------------------------------------------------------------
    # Helper: convert raw DB rows to ContextFragment list
    # ------------------------------------------------------------------
    def _choose_representation(self, row, classification, prompt_keywords):
        """Use the same eligible representations as chat and explicit reads."""
        return choose_representation(row, classification, prompt_keywords)

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
        except Exception as err:
            self._leg_degraded("document_chunks", err)
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

            # Query-selected document excerpts are alternatives, never a prefix
            # of an otherwise complete turn. Legacy/no-chunk rows keep raw text.
            complete_source = text == getattr(row, "raw_text", None)
            if getattr(row, "is_document", False):
                chunk_text_ = self._relevant_doc_chunks(row.id, prompt_keywords)
                if chunk_text_:
                    text = chunk_text_
                    degrade_text = None
                    abstract_text = None
                    complete_source = False

            if not text:
                continue

            score_val = float(getattr(row, "score", 1.0))
            if getattr(row, "is_bookmarked", False):
                score_val *= (1.0 + settings.retrieval_bonus_bookmarked)

            # Preserve source time/provenance on every scored representation.
            if getattr(row, "timestamp", None):
                stamp = recorded_stamp(row.timestamp, getattr(row, "ts_provenance", None))
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
                covers_entire_source=complete_source,
            ))
        return fragments
