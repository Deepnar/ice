"""Hybrid Retrieval Orchestrator – Phase A hardened: decay filtering, access-weighting,
wide‑net full‑vector, Codex/Procedural scoping, grounded query expansion, procedural
trigger matching, micro‑NER integration, and dynamic token budget."""

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog
from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy import bindparam, func, literal, or_, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.config import settings
from src.classifier.schemas import ClassificationResult
from src.memory.conversation_notes import indexed_parts, note_matches, render_note
from src.memory.models import (
    BatchNote,
    CodexEntity,
    CodexFact,
    ConversationNote,
    EpisodicMemory,
    ProceduralMemory,
)
from src.memory.representation import choose_representation
from src.memory.recent_window import base_recent_fraction
from src.memory.summary_snapshot import SUMMARY_SOURCES_SQL, batch_snapshot_readable, summary_snapshot_readable
from src.memory.time_format import format_time, source_stamp
from src.memory.tokens import count as count_tokens
from src.retrieval import coverage, leg_weights
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
    source_type: str          # "episodic", "codex", "procedural", "batch_summary", … ("timeline": no producer since CODEX_V4)
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
    covers_entire_source: bool = False  # complete raw turn, not summary/excerpt
    # An indexed, manifest-checked note that contains exactly one complete
    # original turn. Final packing can share its slot with that episodic row.
    source_note_row_id: Optional[str] = None
    source_note_body: Optional[str] = None
    # Exact stored rows, separate from source-turn and rendered-text lineage.
    stored_artifact_ids: tuple = ()
    # CODEX_V4: the episodic ROW ids of a fact's visible sources (the same id
    # space as an episodic fragment's source_batch_id and the recent-turn
    # receipts) — final assembly skips a fact whose every source turn is
    # already in the prompt (D8). origin_fact_ids feed usage (D7).
    fact_source_row_ids: tuple = ()
    origin_fact_ids: tuple = ()

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

# Every original turn, warm or archived, with its cluster membership: warm
# membership from current links, archived membership as preserved. A cold row
# is the archived copy of the same id, so warm wins when both exist.
_SOURCE_ROWS = """
    SELECT warm.id, warm.batch_id, warm.conversation_id, warm.is_private, warm.timestamp,
           ARRAY(SELECT l.cluster_id FROM episodic_cluster_links l
                 WHERE l.episodic_id = warm.id) AS cluster_ids
    FROM episodic_memory warm
    UNION ALL
    SELECT cold.id, cold.batch_id, cold.conversation_id, cold.is_private, cold.timestamp,
           cold.cluster_ids
    FROM cold_storage cold
    WHERE NOT EXISTS (SELECT 1 FROM episodic_memory warm WHERE warm.id = cold.id)
"""

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
        # visibility. The facts leg fails CLOSED on it instead.
        self._scope_resolution_failed = False

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
        base = base_recent_fraction(turn_count)

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
        *, defer_exposure: bool = False,
    ) -> List[ContextFragment]:
        # The retrieve/no-retrieve decision is now made upstream in one place
        # (B2 plus optional source refinement) — the caller sets
        # context_reliance before a normal or provisional search. The
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
            final = self._wide_net_fallback(classification, prompt_embedding, conversation_id, scope)
            if not defer_exposure:
                self.record_exposure(final)
            return final

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
        elif settings.retrieval_auto_cluster_scope:
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
        # CODEX_V4 D8: one facts leg; a fact's predecessor rides in its own
        # rendering, so there is no separate timeline leg.
        codex_fragments = self._facts_leg(classification.prompt, prompt_embedding, scope, conv_id)

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
            # Cold originals compete under the same episodic packing lane;
            # a date is a query constraint, not permission to remember.
            "cold": self._cold_lookup(prompt_keywords, conv_id, scope,
                                      prompt_embedding),
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
        if reranked:
            fused = self._fact_strength_bonus(fused)
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
        if not defer_exposure:
            self._strengthen_retrieved(final)

        # T3 (D-U1): budget-surviving cold hits get their probation
        # resurrection (after strengthening — probation starts exactly at
        # the configured score, not score+0.15).
        if not defer_exposure:
            self._resurrect_cold_hits(final)

        return final

    def record_exposure(self, fragments):
        """Credit admitted surviving sources after a provisional retrieval."""
        self._strengthen_retrieved(fragments)
        self._resurrect_cold_hits(fragments)

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

    def _cluster_filter(self, scope, id_column="episodic_memory.id", *, membership_column=None) -> str:
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
        if membership_column:
            return (f" AND {membership_column} IS NOT NULL AND "
                    f"(cardinality({membership_column}) = 0 OR "
                    f"{membership_column} && CAST(:cluster_ids AS uuid[]))")
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
                           id_column="episodic_memory.id", *, membership_column=None):
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
        if excluded_clusters and membership_column:
            sql += (f" AND {membership_column} IS NOT NULL AND NOT "
                    f"({membership_column} && CAST(:excl_cluster_ids AS uuid[]))")
            params["excl_cluster_ids"] = [str(c) for c in excluded_clusters]
        elif excluded_clusters and id_column:
            sql += (f"\n              AND NOT EXISTS ("
                    f"SELECT 1 FROM episodic_cluster_links xl "
                    f"WHERE xl.episodic_id = {id_column} "
                    f"AND xl.cluster_id = ANY(:excl_cluster_ids))")
            params["excl_cluster_ids"] = [str(c) for c in excluded_clusters]
        return sql, params

    def _timescope_leg_filters(self, prefix=""):
        """T3: param-driven filter snippets for the episodic legs — one query
        text per leg, never a timescoped twin (G19). A storage state or low
        decay score is not a truth/relevance veto: warm archived and probation
        originals compete under the same caps, decay ranking and reranker.
        Only explicit source-clock windows restrict recording dates."""
        ts = self._active_timescope
        params = {}
        time_filter = ""
        if ts.source_clock and ts.mode in ("as_of", "range") and ts.t0 and ts.t1:
            time_filter = (f"AND {prefix}timestamp >= :ts_t0 "
                           f"AND {prefix}timestamp < :ts_t1")
            params["ts_t0"], params["ts_t1"] = ts.t0, ts.t1
        return time_filter, "", params, 0.0

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
        if ts.mode != 'current' and not ts.source_clock:
            return now, 0.0, settings.retrieval_episodic_recency_tau_days
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
            SELECT id, raw_text, summary_text, summary_coverage, representation_verification, source_spans, abstract_text, lossless_flag, inject_raw, is_document, conversation_id, is_bookmarked, timestamp, ts_provenance,
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
                    SELECT id, raw_text, summary_text, summary_coverage, representation_verification, source_spans, abstract_text, lossless_flag, inject_raw, is_document, conversation_id, is_bookmarked, timestamp, ts_provenance,
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
            SELECT c.id AS chunk_id, c.chunk_text, c.chunk_index, e.id AS parent_id,
                   e.conversation_id, e.is_bookmarked, e.timestamp, e.ts_provenance,
                   e.raw_text, e.source_spans,
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
            chunk_text = source_stamp(row) + chunk_text
            fragments.append(ContextFragment(
                text=chunk_text,
                source_type="episodic",
                score=score_val,
                token_count=count_tokens(chunk_text),
                source_batch_id=pid,
                stored_artifact_ids=(("episodic_chunks", str(row.chunk_id)),),
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
            # G29: reuse the exact/payload matchers' SQLAlchemy visibility
            # predicate *before* LIMIT. Filtering the nearest raw result in
            # Python would let a hidden project entity crowd out a visible one.
            # Coerce numpy scalars to Python floats for pgvector's binder.
            vector = [float(x) for x in candidate_emb]
            distance = CodexEntity.embedding.cosine_distance(vector)
            rows = (self.db.query(CodexEntity.id, (1 - distance).label("sim"))
                    .filter(CodexEntity.embedding.is_not(None),
                            *self._entity_source_filters())
                    .order_by(distance)
                    .limit(1 + len(seen_ids)).all())
            best_entity = None
            for row in rows:
                if row.id in seen_ids:
                    continue
                if row.sim >= threshold:
                    best_entity = self.db.get(CodexEntity, row.id)
                break   # rank 1 among the unclaimed; a worse one cannot win
            if best_entity is not None:
                matched.append(best_entity)
                seen_ids.add(best_entity.id)
                continue

            # 2) Fallback: exact canonical name / alias match
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

    def _scope_batch_ids(self, scope: Optional[dict]):
        """A5: resolve a scope into the set of batch_ids it admits, for the
        legs that filter by source batch (procedural). None means UNSCOPED.

        One of: scope["batch_ids"] (pre-resolved), scope["conversation_ids"]
        (several conversations), scope["conversation_id"] (one), or
        scope["isolated"] = True (incognito / none: empty set, matches nothing).
        Until CODEX_V4 this also resolved the conversation graph's entity set
        (`_codex_scope_sets`); facts are scoped by their own source rows.
        """
        if not scope:
            return None
        if scope.get("isolated"):                 # C6 'none' = true incognito
            return set()
        if self._scope_resolution_failed:
            # G36: the exclusion resolver could not work out what to deny.
            # Reading unscoped would serve the excluded material; match nothing.
            return set()
        # G37: did the caller RESTRICT the conversation set, or merely scope by
        # something else? Tested by key PRESENCE, not truthiness — an exclusion
        # can empty a closed set and `[]` must still mean "these conversations,
        # of which there are none". A cluster-only or timescope-only scope names
        # no conversations at all and stays unscoped.
        restricted = ("batch_ids" in scope or "conversation_ids" in scope
                      or "conversation_id" in scope)
        try:
            batch_ids = set()
            if scope.get("batch_ids"):
                batch_ids = {uuid.UUID(b) if isinstance(b, str) else b
                             for b in scope["batch_ids"]}
            else:
                conv_ids = scope.get("conversation_ids")
                if not conv_ids and scope.get("conversation_id"):
                    conv_ids = [scope["conversation_id"]]
                if conv_ids:
                    rows = self.db.execute(
                        text("SELECT batch_id FROM episodic_memory "
                             "WHERE conversation_id = ANY(:cids) AND is_private = FALSE "
                             "UNION SELECT batch_id FROM cold_storage "
                             "WHERE conversation_id = ANY(:cids) AND is_private = FALSE "
                             "AND batch_id IS NOT NULL"),
                        {"cids": list(conv_ids)}
                    ).fetchall()
                    batch_ids = {row.batch_id for row in rows}
            # E1b (D11): an attached project adds the deterministic static-edge
            # batch, and a project scope NEVER falls back to unscoped.
            pid = self._scope_project_id
            if pid is None and not batch_ids:
                # G37: a named conversation set that resolves to no batches
                # reads NOTHING, not everything (first turn of a manually-scoped
                # conversation; exclusions emptying a closed set). User decision
                # 2026-08-09: starting fresh should give nothing.
                return set() if restricted else None
            if pid is not None:
                from src.coding.code_graph import code_graph_batch_id
                batch_ids.add(code_graph_batch_id(pid))
            if self._denied_batch_ids:
                batch_ids -= self._denied_batch_ids
            return batch_ids
        except Exception as err:
            # G36: fail CLOSED. None would mean unscoped — a failure resolving
            # the scope the caller asked for must not widen visibility.
            self._leg_degraded("scope.batch_ids", err)
            self._scope_resolution_failed = True
            return set()

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
                text("SELECT batch_id FROM episodic_memory "
                     "WHERE conversation_id = ANY(:cids) "
                     "UNION SELECT batch_id FROM cold_storage "
                     "WHERE conversation_id = ANY(:cids) AND batch_id IS NOT NULL"),
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
            # _scope_batch_ids and the facts leg see the flag and scope the
            # codex and procedural legs to nothing rather than to everything.
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

    def _facts_leg(self, prompt, prompt_embedding, scope=None, conv_id=None):
        """CODEX_V4 D8: verified fact sentences by keyword, meaning and one-hop entity.

        A fact is visible when ANY of its source turns is visible under this
        request's scope (TRAPS 65 — the first source is not the only one).
        Current facts only, unless an as-of/range timescope asks for the state
        then. Fragments carry no conversation_id: they are compact facts, not
        turns, so the per-conversation turn cap does not apply to them.
        """
        self._last_matched_entities = []
        self._fact_strength = {}
        if (self._scope_resolution_failed
                or (scope and (scope.get("isolated") or scope.get("incognito")))):
            return []
        if scope and scope.get("cluster_ids_explicit") and not scope.get("cluster_ids"):
            return []
        conv_filter, conv_params, _ = self._conv_scope_filter(
            scope, (scope or {}).get("conversation_id") or conv_id, "e.conversation_id")
        excl_filter, excl_params = self._exclusion_filters(
            scope, "e.conversation_id", "e.id", membership_column="e.cluster_ids")
        cluster_filter = self._cluster_filter(scope, "e.id", membership_column="e.cluster_ids")
        time_filter, _, ts_params, _ = self._timescope_leg_filters("e.")
        ts = getattr(self, "_active_timescope", CURRENT)
        if time_filter and ts.mode == "as_of":
            # As of a moment: the state THEN — anything stated by the window's
            # end that was still valid at it, however long before it was said.
            time_filter = "AND e.timestamp < :ts_t1"
        limit = settings.codex_fact_candidate_limit
        params = {"prompt": prompt, "limit": limit, **conv_params, **excl_params, **ts_params}
        source_filters = ""
        if scope and "batch_ids" in scope:
            source_filters += " AND s.source_batch = ANY(:batches)"
            params["batches"] = list(scope["batch_ids"] or [])
        if self._denied_batch_ids:
            source_filters += " AND s.source_batch <> ALL(:denied)"
            params["denied"] = list(self._denied_batch_ids)
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]
        # A fact expired by /forget or deletion is never history: only a
        # REPLACED fact (superseded_by) is served from the past.
        if time_filter and ts.mode == "as_of":
            validity = ("f.valid_from <= :valid_at AND (f.valid_until IS NULL OR "
                        "(f.valid_until > :valid_at AND f.superseded_by IS NOT NULL))")
            params["valid_at"] = ts.t1
        elif time_filter:
            # A range: what was stated in it, current or replaced after it began.
            validity = ("(f.valid_until IS NULL OR "
                        "(f.valid_until > :ts_t0 AND f.superseded_by IS NOT NULL))")
        else:
            validity = "f.valid_until IS NULL"
        visible = f"""SELECT s.fact_id, s.episodic_id, s.source_batch, e.timestamp
            FROM codex_fact_sources s JOIN ({_SOURCE_ROWS}) e
              ON e.id = s.episodic_id AND e.batch_id = s.source_batch
            WHERE e.is_private = false {conv_filter} {excl_filter} {cluster_filter}
              {time_filter} {source_filters}"""
        base = f"FROM codex_facts f WHERE {validity} AND f.id IN (SELECT fact_id FROM ({visible}) v)"
        try:
            channels = [list(self.db.execute(text(f"""SELECT f.id {base}
                AND to_tsvector('english', f.text) @@ plainto_tsquery('english', :prompt)
                ORDER BY ts_rank(to_tsvector('english', f.text), plainto_tsquery('english', :prompt)) DESC, f.id
                LIMIT :limit"""), params).scalars())]
            if prompt_embedding is not None:
                semantic = text(f"""SELECT f.id {base} AND f.embedding IS NOT NULL
                    ORDER BY f.embedding <=> :embedding, f.id LIMIT :limit""").bindparams(
                        bindparam("embedding", type_=PgVector))
                channels.append(list(self.db.execute(
                    semantic, {**params, "embedding": prompt_embedding}).scalars()))
            matched = self._facts_entities(prompt)
            code_anchors = self._code_graph_anchors(matched)
            matched = [e for e in matched if getattr(e, "source", None) in (None, "conversation")]
            if matched:
                channels.append(list(self.db.execute(text(f"""SELECT f.id {base}
                    AND f.id IN (SELECT fact_id FROM codex_fact_entities WHERE entity_id = ANY(:ents))
                    ORDER BY f.support_count DESC, f.mentioned_at DESC, f.id LIMIT :limit"""),
                    {**params, "ents": [e.id for e in matched]}).scalars()))
            scores = {}
            for channel in channels:
                for rank, ident in enumerate(channel, 1):
                    scores[ident] = scores.get(ident, 0.0) + 1.0 / (settings.retrieval_rrf_k + rank)
            chosen = sorted(scores, key=lambda k: (-scores[k], str(k)))[:limit]
            if not chosen:
                return code_anchors
            sources = {}
            for fact_id, row_id, batch, _when in self.db.execute(
                    text(f"SELECT * FROM ({visible}) v WHERE v.fact_id = ANY(:chosen)"),
                    {**params, "chosen": chosen}):
                sources.setdefault(fact_id, []).append((str(row_id), str(batch)))
            current = validity == "f.valid_until IS NULL"
            # The predecessor clause obeys the same visibility as the fact: a
            # replaced fact whose source was deleted, forgotten or is out of
            # scope must not reach the prompt through its successor.
            earlier = {}
            if current:
                for successor, ident in self.db.execute(text(f"""
                        SELECT f.superseded_by, f.id FROM codex_facts f
                        WHERE f.superseded_by = ANY(:chosen) AND f.valid_until IS NOT NULL
                          AND f.id IN (SELECT fact_id FROM ({visible}) v)
                        ORDER BY f.valid_until DESC"""), {**params, "chosen": chosen}):
                    earlier.setdefault(successor, ident)
            fragments = []
            for ident in chosen:
                fact = self.db.get(CodexFact, ident)
                previous = self.db.get(CodexFact, earlier[ident]) if ident in earlier else None
                rendered = self._render_fact(fact, current=current, earlier=previous)
                rows = sources.get(ident, [])
                self._fact_strength[str(ident)] = (fact.support_count or 1, fact.usage or 0.0)
                fragments.append(ContextFragment(
                    rendered, "codex", scores[ident], count_tokens(rendered), leg="facts",
                    origin_batch_ids=tuple(sorted({b for _, b in rows})),
                    fact_source_row_ids=tuple(sorted({r for r, _ in rows})),
                    origin_fact_ids=(str(ident),),
                    stored_artifact_ids=(("codex_facts", str(ident)),)))
            return code_anchors + fragments
        except Exception as exc:
            self._leg_degraded("codex.facts", exc)
            return []

    def _code_graph_anchors(self, entities):
        """E1b under CODEX_V4: a derived (code-graph / project-fact) entity the
        question names renders its stored pointer payload, exactly as the legacy
        graph leg did for derived anchors. Visible only inside its project (the
        name matchers already apply _entity_source_filters)."""
        out = []
        for entity in entities:
            if getattr(entity, "source", None) in (None, "conversation") or not entity.context_payload:
                continue
            rendered = f"[Entity: {entity.canonical_name}]\n{entity.context_payload}"
            out.append(ContextFragment(rendered, "codex", 1.0 / (settings.retrieval_rrf_k + 1),
                                       count_tokens(rendered), leg="code_graph",
                                       stored_artifact_ids=(("codex_entities", str(entity.id)),)))
        return out

    def _facts_entities(self, prompt):
        """Entities the question names; also the grounded BM25 expansion's input (A4).

        The store's own entity names found whole-word in the question, plus the
        micro-NER's matches. Measured 2026-10-16 on the 67 reviewed development
        questions (CODEX_V4 §1b): micro-NER alone reached gold-turn facts on 15,
        whole-word lookup on 25, both on 29 at 1.5 entities per question;
        trigram word-similarity >= 0.6 reached 31 but at 3.1 entities.
        """
        matched = []
        if prompt:
            try:
                names = extract_entities(prompt, self.embedder)
            except Exception as exc:                                # noqa: BLE001
                # One channel's failure must not take the keyword and meaning
                # channels down with it; the store-name lookup still runs.
                self._leg_degraded("codex.facts.ner", exc)
                names = []
            if names:
                matched = (self._match_entities_by_similarity(names) if self.use_fuzzy_match
                           else self._match_entities_exact(names))
            matched += [e for e in self._known_names_in(prompt) if e.id not in {m.id for m in matched}]
        self._last_matched_entities = matched
        return matched

    def _known_names_in(self, prompt):
        """Conversation entities whose name occurs whole-word in *prompt*
        (punctuation folded to spaces on both sides; names of 3+ characters)."""
        padded = " " + " ".join(re.sub(r"[^\w\s]", " ", prompt.lower()).split()) + " "
        name = func.regexp_replace(func.regexp_replace(CodexEntity.canonical_name, r"[^\w\s]+", " ", "g"),
                                   r"\s+", " ", "g")
        try:
            return (self.db.query(CodexEntity)
                    .filter(*self._entity_source_filters(), func.length(CodexEntity.canonical_name) >= 3,
                            literal(padded).like(func.concat("% ", func.trim(name), " %")))
                    .order_by(func.length(CodexEntity.canonical_name).desc())
                    .limit(settings.codex_enum_entity_limit).all())
        except Exception as exc:                                    # noqa: BLE001
            self._leg_degraded("codex.facts.names", exc)
            return []

    def _render_fact(self, fact, *, current, earlier=None):
        """D1: the sentence is what the model reads; one predecessor clause for a
        fact that replaced another (knowledge-update questions). *earlier* is the
        newest replaced predecessor that is visible under this request's scope."""
        state = "current" if fact.valid_until is None else f"replaced {fact.valid_until:%Y-%m-%d}"
        rendered = (f"[Fact; {fact.speaker}, {fact.modality}; {fact.mentioned_at:%Y-%m-%d}; {state}] "
                    f"{fact.text}")
        if current and earlier is not None:
            rendered += f" (earlier: {earlier.text} — until {earlier.valid_until:%Y-%m-%d})"
        return rendered

    def _fact_strength_bonus(self, fragments):
        """D7: after the reranker (a prior before it only moves the candidate cut).
        support_count carries two thirds, decayed prompt usage one third; both
        saturate (TRAPS 135). It ranks, it never admits."""
        bound = settings.codex_fact_strength_bonus
        strength = getattr(self, "_fact_strength", {})
        if not bound or not strength:
            return fragments
        out = []
        for frag in fragments:
            ids = getattr(frag, "origin_fact_ids", ())
            if frag.leg == "facts" and ids and ids[0] in strength:
                support, usage = strength[ids[0]]
                bonus = bound * (2 / 3 * min(1.0, (support - 1) / 3) + 1 / 3 * min(1.0, usage / 5))
                frag = replace(frag, score=frag.score + bonus)
            out.append(frag)
        out.sort(key=lambda f: f.score, reverse=True)
        return out


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
        allowed_batch_ids = self._scope_batch_ids(scope)
        excluded_batch_ids = self._denied_batch_ids

        # T3: under a window, a pattern is relevant only if its observation
        # span overlaps it (a habit first seen after the window didn't exist
        # then).
        ts = self._active_timescope
        time_filter, params = "", {
            "prompt_embedding": prompt_embedding,
            "min_conf": settings.procedural_min_conf,
            "proc_limit": settings.retrieval_procedural_limit,
            "proc_project": self._scope_project_id,
            "proc_unscoped": allowed_batch_ids is None,
            "proc_allowed": list(allowed_batch_ids or []),
            "proc_excluded": list(excluded_batch_ids),
            "proc_topics": list(classification.topic_tags or []),
            "proc_intents": list(classification.intent_tags or []),
        }
        if ts.source_clock and ts.mode in ("as_of", "range") and ts.t0 and ts.t1:
            time_filter = "AND first_observed <= :ts_t1 AND last_observed >= :ts_t0"
            params["ts_t0"], params["ts_t1"] = ts.t0, ts.t1
        # C9: the PgVector bindparam is load-bearing — the plain-list bind
        # raised `vector <=> double precision[]` on every call, so this leg
        # silently returned [] (rollback + empty) since the psycopg3 move.
        # The intent whitelist hid the crash; killing the gate exposed it.
        # Eligibility precedes the cap: an invisible nearest neighbor must not
        # consume the only slot before Python removes it. Reserved proof keys
        # do not participate in topic/intent trigger matching.
        trigger_filters = "\n".join(f"""
            AND CASE
                WHEN COALESCE(trigger_conditions->'{labels}', '[]'::jsonb) = '[]'::jsonb
                    THEN TRUE
                WHEN jsonb_typeof(trigger_conditions->'{labels}') = 'array'
                    THEN (trigger_conditions->'{labels}') ?| CAST(:{parameter} AS text[])
                ELSE FALSE
            END
        """ for labels, parameter in (("topic_tags", "proc_topics"), ("intent_tags", "proc_intents")))
        query = text(f"""
            SELECT id, pattern_description,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM procedural_memory
            WHERE embedding IS NOT NULL AND is_active = true
              AND confidence_score >= :min_conf
              AND (project_id IS NULL OR project_id = CAST(:proc_project AS uuid))
              AND (:proc_unscoped OR project_id = CAST(:proc_project AS uuid)
                   OR source_batch_ids && CAST(:proc_allowed AS uuid[]))
              AND (COALESCE(cardinality(source_batch_ids), 0) = 0
                   OR NOT (source_batch_ids <@ CAST(:proc_excluded AS uuid[])))
            {trigger_filters}
            {time_filter}
            ORDER BY score DESC, id
            LIMIT :proc_limit
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        try:
            rows = self.db.execute(query, params).fetchall()
            fragments = []
            pid = self._scope_project_id
            for r in rows:
                pattern = self.db.get(ProceduralMemory, r.id)
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
                # Typed source status qualifies the wording supplied to the
                # answerer. Legacy or stale source proofs remain explicit.
                from src.memory.procedural_support import pattern_sources, proof_current, source_time_context
                proof = (pattern.trigger_conditions or {}).get('_source_support')
                originals = pattern_sources(self.db, set(source_batches)) if source_batches else None
                verified = originals is not None and proof_current(proof, pattern.pattern_description, originals)
                label = ('Explicit user instruction' if verified and proof['kind'] == 'explicit_instruction'
                         else 'Observed user behavior' if verified else 'Unverified stored procedure')
                # The stored observation interval is an ICE knowledge clock,
                # never the date a cited user instruction/action occurred.
                visible = [s for s in (originals or []) if uuid.UUID(s.batch_id) not in excluded_batch_ids
                           and (allowed_batch_ids is None or in_project
                                or uuid.UUID(s.batch_id) in allowed_batch_ids)]
                source_clock = source_time_context(visible)
                observed = (f'ICE observations: {format_time(pattern.first_observed)}'
                            f' to {format_time(pattern.last_observed)}')
                rendered = f'[{label}] [{source_clock}; {observed}] {pattern.pattern_description}'
                fragments.append(ContextFragment(
                    text=rendered,
                    source_type="procedural",
                    score=r.score,
                    token_count=count_tokens(rendered),
                    # G48: the turns this habit was inferred from. Already
                    # computed above for scoping and then thrown away, so a
                    # procedural fragment could never be credited to a turn.
                    origin_batch_ids=tuple(str(b) for b in source_batches),
                    stored_artifact_ids=(("procedural_memory", str(pattern.id)),),
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
            scope, conv_column="covered_source.conversation_id", id_column=None,
            membership_column="covered_source.cluster_ids")
        source_clusters = self._cluster_filter(scope,
            membership_column="covered_source.cluster_ids")
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
        # Own batch parts use the same manifest-bound indexing contract as
        # rolling notes. Keep full valid roots available until index backfill.
        own_limit = settings.retrieval_batch_summary_limit
        if conv_id and own_limit > 0:
            try:
                parent_eligible = """
                    bs.conversation_id = :own_conv_id
                    AND EXISTS (SELECT 1 FROM summary_sources em
                                WHERE em.batch_summary_id = bs.id)
                """
                allowed_root = f"""
                    {parent_eligible}
                    AND NOT EXISTS (
                        SELECT 1 FROM summary_sources covered_source
                        WHERE covered_source.batch_summary_id = bs.id
                          AND NOT ({source_allowed}))
                """
                allowed_part = f"""
                    {parent_eligible}
                    AND NOT EXISTS (
                        SELECT 1 FROM summary_sources covered_source
                        WHERE covered_source.batch_summary_id = bs.id
                          AND n.source_ids @> jsonb_build_array(covered_source.id::text)
                          AND NOT ({source_allowed}))
                """
                params = {"prompt_embedding": prompt_embedding, "own_conv_id": conv_id,
                          **scope_params, "candidate_limit": max(64, own_limit * 16)}
                query = text(f"""
                    WITH summary_sources AS ({SUMMARY_SOURCES_SQL})
                    SELECT bs.id, bs.conversation_id, bs.source_manifest, bs.summary_text,
                           bs.created_at, n.ordinal, n.text, n.mode, n.recorded_range,
                           n.source_ids, n.batch_ids,
                           ({allowed_root}) AS root_scope_ok,
                           1 - (n.embedding <=> :prompt_embedding) AS score
                    FROM batch_notes n JOIN batch_summaries bs ON bs.id = n.summary_id
                    WHERE {allowed_part}
                    ORDER BY score DESC, bs.id, n.ordinal
                    LIMIT :candidate_limit
                """).bindparams(bindparam("prompt_embedding", type_=PgVector))
                rows = self.db.execute(query, params).fetchall()
                checked, aggregate_fallbacks = {}, set()
                for r in rows:
                    if r.id not in checked:
                        current = batch_snapshot_readable(self.db, r)
                        notes = self.db.query(BatchNote).filter_by(summary_id=r.id).all()
                        checked[r.id] = (current, current and indexed_parts(r.source_manifest, notes))
                    current, indexed = checked[r.id]
                    if not current:
                        continue
                    if not indexed:
                        if r.id in aggregate_fallbacks:
                            continue
                        # The compatibility aggregate covers every parent source.
                        # A part-scoped query must never expose hidden siblings.
                        if not r.root_scope_ok:
                            continue
                        logger.warning('batch_note_index_mismatch', summary_id=str(r.id),
                                       reason='using current complete aggregate until index repair')
                        rendered = (f"[summary created: {format_time(r.created_at)}] "
                                    + r.summary_text)
                        batches = tuple(item['batch_id'] for item in r.source_manifest['sources'])
                        aggregate_fallbacks.add(r.id)
                    else:
                        part = r.source_manifest['parts'][r.ordinal - 1]
                        if not note_matches(r, part, r.source_manifest, r.ordinal):
                            continue
                        rendered = (f"[batch note created: {format_time(r.created_at)}] "
                                    + render_note(r))
                        batches = tuple(r.batch_ids)
                    fragments.append(ContextFragment(
                        text=rendered, source_type="batch_summary", score=r.score,
                        token_count=count_tokens(rendered), origin_batch_ids=batches,
                        conversation_id=str(conv_id),
                        source_note_row_id=(r.source_ids[0] if indexed and r.mode == 'source'
                                            and len(r.source_ids) == 1 else None),
                        source_note_body=(r.text if indexed and r.mode == 'source'
                                          and len(r.source_ids) == 1 else None),
                        stored_artifact_ids=(("batch_notes", json.dumps([str(r.id), r.ordinal],
                            separators=(",", ":"))),) if indexed else (("batch_summaries", str(r.id)),)))
                    if len(fragments) >= own_limit:
                        break
                if len(fragments) < own_limit:
                    fallback = text(f"""
                        WITH summary_sources AS ({SUMMARY_SOURCES_SQL})
                        SELECT bs.id, bs.conversation_id, bs.source_manifest,
                               bs.summary_text, bs.created_at,
                               1 - (bs.embedding <=> :prompt_embedding) AS score
                        FROM batch_summaries bs
                        WHERE {allowed_root} AND bs.embedding IS NOT NULL
                          AND NOT EXISTS (SELECT 1 FROM batch_notes n WHERE n.summary_id = bs.id)
                        ORDER BY score DESC
                        LIMIT :candidate_limit
                    """).bindparams(bindparam("prompt_embedding", type_=PgVector))
                    for r in self.db.execute(fallback, params).fetchall():
                        if not batch_snapshot_readable(self.db, r):
                            continue
                        logger.warning('batch_note_index_missing', summary_id=str(r.id),
                                       reason='using current complete aggregate until index backfill')
                        rendered = (f"[summary created: {format_time(r.created_at)}] "
                                    + r.summary_text)
                        fragments.append(ContextFragment(
                            text=rendered, source_type="batch_summary", score=r.score,
                            token_count=count_tokens(rendered), conversation_id=str(conv_id),
                            origin_batch_ids=tuple(item['batch_id']
                                                   for item in r.source_manifest['sources']),
                            stored_artifact_ids=(("batch_summaries", str(r.id)),)))
                        if len(fragments) >= own_limit:
                            break
            except Exception as err:
                self._leg_degraded("batch_summary.own", err)
        # Half 2 (C4 D3b): OTHER conversations' independently indexed source
        # notes. The active
        # conversation's own summary is excluded (the assembler injects it —
        # double-inject trap) and private conversations never leave their
        # scope (memory_scope_type join). Skipped entirely under incognito
        # (a user-global read — G16 "read nothing").
        if not include_cross or settings.retrieval_conversation_summary_limit <= 0:
            return fragments
        try:
            query = text(f"""
                WITH summary_sources AS ({SUMMARY_SOURCES_SQL})
                SELECT n.ordinal, n.text, n.mode, n.recorded_range, n.source_ids,
                       n.batch_ids, s.summary_text, s.updated_at,
                       s.source_manifest, s.conversation_id,
                       1 - (n.embedding <=> :prompt_embedding) as score
                FROM conversation_notes n
                JOIN conversation_summaries s ON s.conversation_id = n.conversation_id
                JOIN conversations c ON c.id = s.conversation_id
                WHERE c.memory_scope_type != 'none'
                  AND NOT EXISTS (
                      SELECT 1 FROM summary_sources private_source
                      WHERE private_source.conversation_id = s.conversation_id
                        AND private_source.is_private = TRUE)
                  AND (CAST(:own_conv_id AS uuid) IS NULL
                       OR s.conversation_id != CAST(:own_conv_id AS uuid))
                  AND NOT EXISTS (
                      SELECT 1 FROM summary_sources covered_source
                      WHERE covered_source.conversation_id = s.conversation_id
                        AND s.source_manifest @> jsonb_build_object(
                            'sources', jsonb_build_array(jsonb_build_object(
                                'id', covered_source.id::text)))
                        AND NOT ({source_allowed}))
                ORDER BY score DESC
                LIMIT :cs_candidate_limit
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = self.db.execute(query, {
                "prompt_embedding": prompt_embedding,
                "own_conv_id": conv_id,
                **scope_params,
                "cs_candidate_limit": max(64, settings.retrieval_conversation_summary_limit * 16),
            }).fetchall()
            checked = {}
            aggregate_fallbacks = set()
            cross_added = 0
            for r in rows:
                cid = str(r.conversation_id)
                if cid not in checked:
                    notes = self.db.query(ConversationNote).filter_by(
                        conversation_id=r.conversation_id).all()
                    current = summary_snapshot_readable(self.db, r)
                    checked[cid] = (current, current and indexed_parts(r.source_manifest, notes))
                current, indexed = checked[cid]
                if not current:
                    continue
                if not indexed:
                    if cid not in aggregate_fallbacks:
                        logger.warning('conversation_note_index_mismatch',
                                       conversation_id=cid,
                                       reason='using current complete aggregate until index repair')
                        rendered = (f"[conversation summary updated: {format_time(r.updated_at)}] "
                                    + r.summary_text)
                        fragments.append(ContextFragment(
                            text=rendered, source_type="batch_summary", score=r.score,
                            token_count=count_tokens(rendered), conversation_id=cid,
                            origin_batch_ids=tuple(item['batch_id']
                                                   for item in r.source_manifest['sources']),
                            stored_artifact_ids=(("conversation_summaries", cid),)))
                        aggregate_fallbacks.add(cid)
                        cross_added += 1
                    if cross_added >= settings.retrieval_conversation_summary_limit:
                        break
                    continue
                part = r.source_manifest['parts'][r.ordinal - 1]
                if not note_matches(r, part, r.source_manifest, r.ordinal):
                    continue
                rendered = (f"[conversation note updated: {format_time(r.updated_at)}] "
                            + render_note(r))
                fragments.append(ContextFragment(
                    text=rendered, source_type="batch_summary", score=r.score,
                    token_count=count_tokens(rendered), conversation_id=cid,
                    origin_batch_ids=tuple(r.batch_ids),
                    source_note_row_id=(r.source_ids[0] if r.mode == 'source'
                                        and len(r.source_ids) == 1 else None),
                    source_note_body=(r.text if r.mode == 'source'
                                      and len(r.source_ids) == 1 else None),
                    stored_artifact_ids=(("conversation_notes", json.dumps([cid, r.ordinal],
                        separators=(",", ":"))),)))
                cross_added += 1
                if cross_added >= settings.retrieval_conversation_summary_limit:
                    break
            # Existing snapshot-valid roots remain available until maintenance
            # materializes their part index; this branch is a compatibility read.
            if cross_added < settings.retrieval_conversation_summary_limit:
                legacy = text(f"""
                    WITH summary_sources AS ({SUMMARY_SOURCES_SQL})
                    SELECT s.summary_text, s.updated_at, s.source_manifest,
                           s.conversation_id,
                           1 - (s.embedding <=> :prompt_embedding) as score
                    FROM conversation_summaries s
                    JOIN conversations c ON c.id = s.conversation_id
                    WHERE s.embedding IS NOT NULL
                      AND c.memory_scope_type != 'none'
                      AND NOT EXISTS (SELECT 1 FROM conversation_notes n
                                      WHERE n.conversation_id = s.conversation_id)
                      AND NOT EXISTS (SELECT 1 FROM summary_sources private_source
                                      WHERE private_source.conversation_id = s.conversation_id
                                        AND private_source.is_private = TRUE)
                      AND (CAST(:own_conv_id AS uuid) IS NULL
                           OR s.conversation_id != CAST(:own_conv_id AS uuid))
                      AND NOT EXISTS (
                          SELECT 1 FROM summary_sources covered_source
                          WHERE covered_source.conversation_id = s.conversation_id
                            AND s.source_manifest @> jsonb_build_object(
                                'sources', jsonb_build_array(jsonb_build_object(
                                    'id', covered_source.id::text)))
                            AND NOT ({source_allowed}))
                    ORDER BY score DESC LIMIT :cs_limit
                """).bindparams(bindparam("prompt_embedding", type_=PgVector))
                old_rows = self.db.execute(legacy, {
                    "prompt_embedding": prompt_embedding, "own_conv_id": conv_id,
                    **scope_params,
                    "cs_limit": settings.retrieval_conversation_summary_limit - cross_added,
                }).fetchall()
                for r in old_rows:
                    if not summary_snapshot_readable(self.db, r):
                        continue
                    logger.warning('conversation_note_index_missing',
                                   conversation_id=str(r.conversation_id),
                                   reason='cross read using complete aggregate until backfill')
                    rendered = (f"[conversation summary updated: {format_time(r.updated_at)}] "
                                + r.summary_text)
                    fragments.append(ContextFragment(
                        text=rendered, source_type="batch_summary", score=r.score,
                        token_count=count_tokens(rendered),
                        conversation_id=str(r.conversation_id),
                        origin_batch_ids=tuple(item['batch_id']
                                               for item in r.source_manifest['sources']),
                        stored_artifact_ids=(("conversation_summaries", str(r.conversation_id)),)))
        except Exception as err:
            self._leg_degraded("batch_summary.cross", err)
        return fragments
    # ------------------------------------------------------------------
    # T3: cold-storage leg + resurrection (D-U1) + honest emptiness
    # ------------------------------------------------------------------
    def _cold_lookup(self, prompt_keywords, conv_id: Optional[str] = None,
                     scope: Optional[dict] = None,
                     prompt_embedding=None) -> List[ContextFragment]:
        """Query-relevant cold originals compete under the existing cap.
        Source-clock windows filter recording dates; event/unknown axes retain
        meaning-ranked sources from other recording dates. An undated search
        needs an embedding or lexical terms. Scoping uses _conv_scope_filter the
        live episodic legs use (G29 — a hand-rolled `if conv_id` copy here let
        project-scoped queries read the whole global cold store)."""
        ts = self._active_timescope
        t1 = ts.t1 or datetime.now(timezone.utc)
        source_window = bool(ts.mode != "current" and ts.t0 and ts.source_clock)

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
        if prompt_embedding is None and not pats and not source_window:
            return []

        conv_filter, conv_params, conv_scoped = self._conv_scope_filter(scope, conv_id)
        excl_filter, excl_params = self._exclusion_filters(scope, id_column=None,
                                                              membership_column="cluster_ids")
        excl_filter += self._cluster_filter(scope, membership_column="cluster_ids")
        if scope and scope.get("cluster_ids"):
            excl_params["cluster_ids"] = [str(c) for c in scope["cluster_ids"]]
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
        vector_match = ""
        if prompt_embedding is not None:
            vector_rank = ("CASE WHEN embedding IS NULL THEN 1 "
                           "ELSE (embedding <=> :cold_probe) END ASC,")
            vector_match = "embedding IS NOT NULL OR"
        time_filter = "timestamp >= :t0 AND timestamp < :t1" if source_window else "TRUE"
        query = text(f"""
            SELECT id, conversation_id, batch_id, raw_text, summary_text,
                   topic_tags, timestamp, is_private, embedding, source_spans, ts_provenance,
                   summary_coverage, representation_verification, abstract_text, lossless_flag, is_document, inject_raw, session_id, intent_tags, context_reliance, idempotency_key, cluster_id, cluster_ids, batch_summary_id
            FROM cold_storage
            WHERE {time_filter}
              {conv_filter}
              {excl_filter}
              {privacy_filter}
              AND (
                    {vector_match} :browse_window
                    OR raw_text ILIKE ANY(:pats)
                    OR summary_text ILIKE ANY(:pats)
              )
            ORDER BY {vector_rank} timestamp DESC
            LIMIT :cold_limit
        """)
        if prompt_embedding is not None:
            query = query.bindparams(bindparam("cold_probe", type_=PgVector))
        params = {"t0": ts.t0, "t1": t1, "browse_window": source_window and not pats,
                  "pats": pats,
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
            stamp = source_stamp(row)
            text_ = stamp + body
            fragments.append(ContextFragment(
                text=text_, source_type="episodic", score=0.6,
                covers_entire_source=body == row.raw_text,
                token_count=count_tokens(text_),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None,
                stored_artifact_ids=(("cold_storage", str(row.id)),),
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
                SELECT id, chunk_text, 1 - (embedding <=> :probe) AS score
                FROM cold_chunks WHERE turn_id = :id AND embedding IS NOT NULL
                ORDER BY embedding <=> :probe, chunk_index, id LIMIT 3
            """).bindparams(bindparam("probe", type_=PgVector))
            rows = self.db.execute(query, {"id": parent.id, "probe": prompt_embedding}).all()
        else:
            rows = self.db.execute(text("""
                SELECT id, chunk_text, 0.0 AS score FROM cold_chunks
                WHERE turn_id = :id ORDER BY chunk_index, id
            """), {"id": parent.id}).all()
            terms = {str(k).casefold() for k in keywords if k}
            rows = sorted(rows, key=lambda r: -sum(k in r.chunk_text.casefold() for k in terms))[:3]
        stamp = source_stamp(parent)
        return [ContextFragment(
            text=(rendered := stamp + row.chunk_text), source_type="episodic",
            score=float(row.score or 0), token_count=count_tokens(rendered),
            source_batch_id=str(parent.id),
            conversation_id=str(parent.conversation_id) if parent.conversation_id else None,
            leg="cold_chunk", covers_entire_source=False,
            stored_artifact_ids=(("cold_chunks", str(row.id)),),
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
                         lossless_flag, is_document, session_id, cluster_id, batch_summary_id)
                    VALUES (:id, :conv, :batch, :ts, :tags, :itags,
                            :context_reliance, :raw, :summary, :emb, :score,
                            1, FALSE, :priv, :inject_raw, :ikey, :source_spans, :ts_provenance,
                            :summary_coverage, :representation_verification, :abstract_text,
                            :lossless_flag, :is_document, :session_id,
                            (SELECT id FROM context_clusters WHERE id = :primary_cluster),
                            (SELECT id FROM batch_summaries WHERE id = :batch_summary_id))
                    ON CONFLICT (id) DO NOTHING
                """).bindparams(bindparam("emb", type_=PgVector),
                                 bindparam("source_spans", type_=JSONB),
                                 bindparam("representation_verification", type_=JSONB)), {
                    "id": row.id, "conv": row.conversation_id,
                    "primary_cluster": getattr(row, "cluster_id", None),
                    "batch_summary_id": getattr(row, "batch_summary_id", None),
                    "batch": row.batch_id or uuid.uuid4(), "ts": row.timestamp,
                    "tags": list(row.topic_tags or []),
                    "itags": list(getattr(row, "intent_tags", None) or []),
                    "context_reliance": getattr(row, "context_reliance", None) or "Long_Term_Memory",
                    "inject_raw": (getattr(row, "inject_raw", None)
                                   if getattr(row, "inject_raw", None) is not None else True),
                    **{key: getattr(row, key, None) for key in (
                        "summary_coverage", "representation_verification", "abstract_text",
                        "lossless_flag", "is_document", "session_id")},
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
        if not ts.source_clock or ts.mode not in ("as_of", "range") or not (ts.t0 and ts.t1):
            return final
        if any(f.source_type == "episodic" for f in final):
            return final
        note = (f"[Memory note] No episodic source records were selected for this query between "
                f"{ts.t0.date()} and {ts.t1.date()}.")
        available = self.max_retrieval_tokens - sum(f.token_count for f in final)
        base_note = note
        if count_tokens(base_note) > available:
            logger.info("timescope_empty_window_note_omitted", reason="remaining_budget",
                        available_tokens=available, required_tokens=count_tokens(base_note))
            return final
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
                note += (" Nearby source records are dated "
                         + ", ".join(eras) + ".")
        except Exception as err:
            self._leg_degraded("timescope.era_probe", err)
        logger.info("timescope_empty_window", t0=str(ts.t0.date()), t1=str(ts.t1.date()))
        if count_tokens(note) > available:
            logger.info("timescope_empty_window_eras_omitted", reason="remaining_budget",
                        available_tokens=available, required_tokens=count_tokens(note))
            note = base_note
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

    @staticmethod
    def _source_note_contains_episodic(note, episodic):
        return (note.source_type == "batch_summary" and note.source_note_row_id
                and episodic.source_type == "episodic" and episodic.covers_entire_source
                and str(episodic.source_batch_id) == note.source_note_row_id)

    @staticmethod
    def _same_source_excerpt_contains(source, excerpt):
        """Exact retained text and contributors, never parent-only deduplication."""
        return bool(
            source.source_type == excerpt.source_type == "episodic"
            and source.source_batch_id and source.conversation_id
            and source.source_batch_id == excerpt.source_batch_id
            and source.conversation_id == excerpt.conversation_id
            and excerpt.text and excerpt.text in source.text
            and source.stored_artifact_ids and excerpt.stored_artifact_ids
            and set(excerpt.stored_artifact_ids) <= set(source.stored_artifact_ids)
            and set(excerpt.origin_batch_ids) <= set(source.origin_batch_ids)
            and (not excerpt.covers_entire_source or source.covers_entire_source)
        )

    @classmethod
    def _omit_covered_source_excerpts(cls, fragments):
        """Spend selected overlapping source text once without budget backfill."""
        return [excerpt for index, excerpt in enumerate(fragments)
                if not any(other != index
                    and (len(source.text) > len(excerpt.text) or other < index)
                    and cls._same_source_excerpt_contains(source, excerpt)
                    for other, source in enumerate(fragments))]

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
            if fragment.source_type == "episodic" and any(
                    self._source_note_contains_episodic(note, fragment)
                    for note in result):
                return False
            redundant = [held for held in result
                         if self._source_note_contains_episodic(fragment, held)]
            reclaimable = sum(held.token_count for held in redundant)
            fitted = (fragment if fragment.token_count <= max_tokens - total + reclaimable
                      else _degraded(fragment, max_tokens - total))
            if fitted is None:
                return False
            # A degraded summary need not contain what the raw source
            # contained. Recheck against the exact admitted text.
            redundant = [held for held in result
                         if self._source_note_contains_episodic(fitted, held)]
            if fitted.token_count > max_tokens - total + sum(
                    held.token_count for held in redundant):
                return False
            if settings.retrieval_collapse_enabled and fitted.source_batch_id:
                same = [f for f in result if f.source_batch_id == fitted.source_batch_id]
                if same and (fitted.covers_entire_source or
                             any(f.covers_entire_source for f in same)):
                    return False
            trial = [f for f in result if f not in redundant] + [fitted]
            if len(self._collapse_provenance(trial)) != len(trial):
                return False
            if len(self._session_diversify(trial, current_conversation_id)) != len(trial):
                return False
            result[:] = trial
            total += fitted.token_count - sum(claim.token_count for claim in redundant)
            return True

        # Successful reranking owns its order. With the fallback diversity
        # strategy explicitly off, fill by fused score through the same guards.
        # Guarding only Phase 1 would leave Phase 2's first fair round intact
        # and make the control ineffective.
        if relevance_order or not settings.retrieval_leg_guarantee_enabled:
            ordered = (fragments if relevance_order else
                       sorted(fragments, key=lambda f: f.score, reverse=True))
            for f in ordered:
                admit(f)
            result = self._omit_covered_source_excerpts(result)
            total = sum(fragment.token_count for fragment in result)
            self._log_leg_budget_share(
                result, total, max_tokens,
                selection_mode="reranked_relevance" if relevance_order else "fusion_relevance")
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
        result = self._omit_covered_source_excerpts(result)
        total = sum(fragment.token_count for fragment in result)
        self._log_leg_budget_share(result, total, max_tokens,
                                   selection_mode="fusion_family_diversity")
        return result

    def _log_leg_budget_share(self, fragments, total, max_tokens, *, selection_mode=None):
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
                selection_mode=selection_mode,
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

        fragments.extend(self._facts_leg(classification.prompt, prompt_embedding, scope, scope_conv))
        prompt_keywords = self._extract_prompt_keywords(classification.prompt) if classification.prompt else set()
        cold = self._cold_lookup(prompt_keywords, scope_conv, scope, prompt_embedding)
        fused = self._apply_rrf({"fallback": fragments, "cold": cold},
                                alpha_map={"fallback": 1.0, "cold": 1.0})
        fused = self._apply_bonuses(fused, classification, conversation_id, prompt_keywords)
        fused.sort(key=lambda x: x.score, reverse=True)
        fused, reranked = rerank(classification.prompt, fused)
        if reranked:
            fused = self._fact_strength_bonus(fused)   # D7, as on the main path
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

    def _relevant_doc_chunks(self, turn_id, prompt_keywords, limit: int = 2, *, artifact_ids=None):
        """C2: pick a document's most query-relevant chunks (keyword-hit count,
        earliest-first tiebreak; the opening chunk as fallback — it usually
        identifies the document). Returns joined text or None when the doc has
        no chunks yet (legacy pre-C2; the catch-up worker heals those)."""
        try:
            rows = self.db.execute(text("""
                SELECT id, chunk_text, chunk_index FROM episodic_chunks
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
                if artifact_ids is not None:
                    artifact_ids.extend(("episodic_chunks", str(r.id)) for r in chosen)
                return "\n[…]\n".join(r.chunk_text for r in chosen)
        if artifact_ids is not None:
            artifact_ids.append(("episodic_chunks", str(rows[0].id)))
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
            artifact_ids = [("episodic_memory", str(row.id))]
            if getattr(row, "is_document", False):
                chunk_artifacts = []
                chunk_text_ = self._relevant_doc_chunks(row.id, prompt_keywords,
                                                        artifact_ids=chunk_artifacts)
                if chunk_text_:
                    text = chunk_text_
                    degrade_text = None
                    abstract_text = None
                    complete_source = False
                    artifact_ids = chunk_artifacts

            if not text:
                continue

            score_val = float(getattr(row, "score", 1.0))
            if getattr(row, "is_bookmarked", False):
                score_val *= (1.0 + settings.retrieval_bonus_bookmarked)

            # Preserve source time/provenance on every scored representation.
            stamp = source_stamp(row)
            if stamp:
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
                stored_artifact_ids=tuple(artifact_ids),
            ))
        return fragments
