"""Hybrid Retrieval Orchestrator – Phase A hardened: decay filtering, access-weighting,
wide‑net full‑vector, Codex/Procedural scoping, HyDE rewriting, procedural trigger matching,
micro‑NER integration, and dynamic token budget."""

from datetime import datetime, timezone
import hashlib
import math
import os
import re
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass, replace
from openai import OpenAI
import structlog
from sqlalchemy.orm import Session
from sqlalchemy import bindparam, text
from pgvector.sqlalchemy import Vector as PgVector
from src.retrieval.ner_utils import extract_entities
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name

from src.api.config import settings
from src.memory.models import (
    EpisodicMemory,
    CodexEntity,
    CodexEdge,
    ProceduralMemory,
    MemorySlot,
)
from src.classifier.schemas import ClassificationResult
logger = structlog.get_logger("ice.retrieval")


@dataclass(frozen=True)
class ContextFragment:
    text: str
    source_type: str          # "episodic", "codex", "procedural", "rag"
    score: float              # RRF fused score
    token_count: int
    source_batch_id: Optional[str] = None
    conversation_id: Optional[str] = None

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

        # Load micro‑NER model (fallback to None if not available)
    def _relevant_cluster_ids(self, prompt_embedding, classification=None, conversation_id=None, top_k=10):
        """Return a list of cluster_id strings for the clusters most
        relevant to the prompt, using both embedding similarity and
        topic‑tag overlap with the current classification.
        """
        try:
            conv_filter = "AND conversation_id = :conv_id" if conversation_id else ""
            query = text(f"""
                SELECT id, 1 - (embedding <=> :emb) AS sim, tags, name, description
                FROM context_clusters
                WHERE embedding IS NOT NULL
                {conv_filter}
                ORDER BY sim DESC
                LIMIT :limit
            """).bindparams(bindparam("emb", type_=PgVector))
            params = {"emb": prompt_embedding, "limit": top_k * 3}
            if conversation_id:
                params["conv_id"] = conversation_id
            rows = self.db.execute(query, params).fetchall()
        except Exception:
            return []

        if not rows:
            return []

        # Boost clusters whose tags overlap with the current topic tags
        topic_tags = set(classification.topic_tags) if classification else set()
        scored = []
        for row in rows:
            cluster_tags = set(row.tags or [])
            tag_overlap = len(topic_tags & cluster_tags) if topic_tags else 0

            # Name/description similarity (small weight – secondary signal)
            name_desc_text = (row.name + " " + (row.description or "")).strip()
            name_sim = 0.0
            if name_desc_text:
                name_desc_emb = self.embedder.encode(name_desc_text, convert_to_tensor=False)
                name_sim = sum(a * b for a, b in zip(prompt_embedding, name_desc_emb))

            combined = row.sim + (0.3 * tag_overlap) + (0.15 * name_sim)
            scored.append((combined, str(row.id)))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or scored[0][0] < 0.50:
            # No cluster is confidently relevant – fall back to full conversation search
            return []
        return [cid for _, cid in scored[:top_k]]
    
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

            # Recency (skip for creative – recent meta turns are noise)
            if f.source_type == "episodic" and not creative and conv_id:
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
        if classification.context_reliance == "Zero_Shot":
            return []
        if classification.context_reliance == "Real_Time_Search":
            return []

        if classification.max_confidence < settings.confidence_fallback_threshold:
            logger.info("wide_net_fallback_triggered", confidence=classification.max_confidence)
            return self._wide_net_fallback(classification, prompt_embedding, conversation_id, scope)

        # Conversation scope filter for episodic legs
        # Conversation scope filter for episodic legs
        conv_id = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]

        # ── Cluster‑scoped retrieval: find the most relevant clusters
        #     and add them to the scope so the episodic legs only search
        #     those clusters.  Falls back gracefully if no clusters exist.
        cluster_ids = self._relevant_cluster_ids(prompt_embedding, classification=classification, conversation_id=conv_id, top_k=10)
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
        # legs (procedural patterns, RAG documents) read nothing at all.
        incognito = bool(scope and scope.get("incognito"))

        # Execute the remaining retrieval legs
        legs: Dict[str, List[ContextFragment]] = {
            "bm25": self._bm25_episodic(classification, scope, conv_id, search_prompt),
            "vector": self._vector_episodic(prompt_embedding, classification, scope, conv_id),
            "codex": codex_fragments,
            "procedural": [] if incognito else self._procedural_lookup(prompt_embedding, classification, scope),
            "rag": [] if incognito else self._rag_lookup(prompt_embedding, classification),
            "batch_summary": self._batch_summary_lookup(prompt_embedding, conv_id),
        }

        # ── Dynamic leg weighting (blended over all active intents) ──

        # Base balanced weights
        base_weights = {
            "bm25": 0.8,
            "vector": 1.0,
            "codex": 0.5,
            "procedural": 0.2,
            "rag": 1.0,
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

        # Clamp to zero (no negative weights)
        for leg in blend_weights:
            blend_weights[leg] = max(0.0, blend_weights[leg])

        # Fuse, diversify, deduplicate, trim
        fused = self._apply_rrf(legs, alpha_map=blend_weights)
                # Compute prompt keywords once for the bonus pass
        prompt_keywords = self._extract_prompt_keywords(classification.prompt) if classification.prompt else set()
        fused = self._apply_bonuses(fused, classification, conv_id, prompt_keywords)
        fused.sort(key=lambda x: x.score, reverse=True)
                # ── Leg diversity guarantee: always include the top‑ranked fragment
        #     from each leg, even if RRF would have dropped it.

        # ── Keyword‑aware re‑ranking: fragments containing probe keywords
        #     get a massive score boost so they survive token‑budget enforcement.


        # (delete the old leg‑diversity block entirely)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        deduped = self._deduplicate(diversified)
        final = self._enforce_token_budget(deduped)

        # Strengthen retrieved turns (access count + decay boost)
        self._strengthen_retrieved(final)

        return final

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
        conv_filter = "AND conversation_id = :conv_id" if conv_id else ""
        # G16 visibility invariant: global search never sees private (incognito)
        # turns; explicit conversation scoping is the only door to them.
        privacy_filter = "" if conv_id else "AND is_private = FALSE"

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
            SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id, is_bookmarked,
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
              {privacy_filter}
              {cluster_filter}
              AND decay_score > :min_decay
              AND is_archived = false
            ORDER BY score DESC
            LIMIT 100
        """)
        params = {
            "search_terms": search_terms,
            "prompt_text": prompt_text,
            "min_decay": 0.2
        }
        if conv_id:
            params["conv_id"] = conv_id
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic", prompt_text=search_prompt or classification.prompt)
        except Exception as err:
            logger.error("bm25_retrieval_failed", error=str(err))
            self.db.rollback()
            # Final fallback: use plainto_tsquery (AND) if everything fails
            try:
                query2 = text(f"""
                    SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id, is_bookmarked,
                           ts_rank(
                               to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')),
                               plainto_tsquery('english', :prompt_text)
                           ) as score
                    FROM episodic_memory
                    WHERE to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, ''))
                          @@ plainto_tsquery('english', :prompt_text)
                      {conv_filter}
                      {privacy_filter}
                      {cluster_filter}
                      AND decay_score > :min_decay
                      AND is_archived = false
                    ORDER BY score DESC
                    LIMIT 100
                """)
                p = {"prompt_text": prompt_text, "min_decay": 0.2}
                if conv_id:
                    p["conv_id"] = conv_id
                if scope and scope.get("cluster_ids"):
                    p["cluster_ids"] = scope["cluster_ids"]
                rows = self.db.execute(query2, p).fetchall()
                return self._rows_to_fragments(rows, "episodic", prompt_text=search_prompt or classification.prompt)        
            except Exception:
                return []

    # ------------------------------------------------------------------
    # Vector episodic (pgvector cosine similarity)
    # ------------------------------------------------------------------
    def _vector_episodic(self, prompt_embedding, classification, scope, conv_id: Optional[str] = None) -> List[ContextFragment]:
        topic_filter = ""
        conv_filter = "AND conversation_id = :conv_id" if conv_id else ""
        # G16 visibility invariant: global search never sees private (incognito)
        # turns; explicit conversation scoping is the only door to them.
        privacy_filter = "" if conv_id else "AND is_private = FALSE"

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
            SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id, is_bookmarked,
                (1 - (embedding <=> :prompt_embedding)) * COALESCE(decay_score, 1.0) as score
            FROM episodic_memory
            WHERE embedding IS NOT NULL
            {topic_filter}
            {conv_filter}
            {privacy_filter}
            {cluster_filter}
            AND decay_score > :min_decay
            AND is_archived = false
            ORDER BY score DESC
            LIMIT 100
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        params = {"prompt_embedding": prompt_embedding, "min_decay": 0.2}
        if conv_id:
            params["conv_id"] = conv_id
        if scope and scope.get("cluster_ids"):
            params["cluster_ids"] = scope["cluster_ids"]

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic", prompt_text=classification.prompt)
        except Exception as err:
            logger.error("vector_retrieval_failed", error=str(err))
            self.db.rollback()
            return []
        
    
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
                # cosine similarity = dot product of normalized vectors
                dot = sum(a * b for a, b in zip(candidate_emb, emb))
                score = dot  # embeddings are already normalised by SentenceTransformer
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
                or_(CodexEntity.canonical_name == norm, CodexEntity.aliases.any(norm))
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

    @staticmethod
    def _unit(vec):
        """L2-normalise a vector. Needed because truncate_dim=384 slices a
        longer normalised embedding, breaking unit norm — raw dot products
        would sit well below any cosine threshold."""
        norm = sum(a * a for a in vec) ** 0.5
        return [a / norm for a in vec] if norm > 0 else list(vec)

    def _relation_gloss_cache(self):
        """Lazily embed the controlled relation vocabulary (as 'inspired by'
        style glosses) once per process, unit-normalised. ~200 relations ×
        384 dims — trivial to hold and scan."""
        global _RELATION_GLOSSES
        if _RELATION_GLOSSES is None:
            from src.workers.codex_extractor import ALLOWED_RELATIONS
            rels = sorted(ALLOWED_RELATIONS)
            embs = self.embedder.encode([r.replace("_", " ") for r in rels],
                                        convert_to_tensor=False, show_progress_bar=False)
            _RELATION_GLOSSES = (rels, [self._unit(list(e)) for e in embs])
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

            # Channel 2 — embedding paraphrase channel (true cosine).
            if prompt_embedding is not None:
                p = self._unit(prompt_embedding)
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
            if not batch_ids:
                return None, None
            event_rows = self.db.execute(
                text("SELECT DISTINCT entity_id FROM codex_events WHERE batch_source = ANY(:bids)"),
                {"bids": list(batch_ids)}
            ).fetchall()
            return {row.entity_id for row in event_rows}, batch_ids
        except Exception:
            self.db.rollback()
            return None, None

    def _match_entities_exact(self, entity_strings: List[str]) -> List:
        """Entity resolution by exact canonical name / alias only (no vectors).
        Production fallback stage and the ablation `fuzzy_match=False` path."""
        from sqlalchemy import or_
        matched, seen = [], set()
        for candidate_str in entity_strings:
            norm = candidate_str.lower().strip()
            ent = self.db.query(CodexEntity).filter(
                or_(CodexEntity.canonical_name == norm, CodexEntity.aliases.any(norm))
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
                    CodexEntity.context_payload.ilike(f"%{w}%")
                ).limit(20).all()
                for ent in rows:
                    scored[ent.id] = (scored.get(ent.id, (0, ent))[0] + 1, ent)
            ranked = sorted(scored.values(), key=lambda t: t[0], reverse=True)
            return [ent for hits, ent in ranked[:2] if hits >= 1]
        except Exception:
            self.db.rollback()
            return []

    def _relation_facts(self, matched, relations: List[str], allowed_batch_ids):
        """A4: surface explicit edge facts where a matched entity participates
        in a detected relation (either direction). The entity∩relation joint
        hit is the precision anchor; these edges also join the reinforcement
        anchors because they directly answered the query."""
        if not matched or not relations:
            return [], []
        matched_ids = [e.id for e in matched]
        q = self.db.query(CodexEdge).filter(
            CodexEdge.valid_until == None,
            CodexEdge.relation.in_(relations),
            ((CodexEdge.source_id.in_(matched_ids)) | (CodexEdge.target_id.in_(matched_ids)))
        )
        if allowed_batch_ids is not None:
            q = q.filter(CodexEdge.source_batch.in_(allowed_batch_ids))
        lines, fact_edges = [], []
        for edge in q.order_by(CodexEdge.strength.desc()).limit(10).all():
            if self._edge_trust(edge) < self.CODEX_DIRECT_TRUST_FLOOR:
                continue
            src = self.db.query(CodexEntity).get(edge.source_id)
            tgt = self.db.query(CodexEntity).get(edge.target_id)
            if src and tgt:
                lines.append(f"[Fact: {src.canonical_name} --{edge.relation}--> {tgt.canonical_name}]")
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
                    or_(*[CodexEntity.tags.any(t) for t in candidate_tags]))
                for ent in q.limit(self.ENUM_ENTITY_LIMIT).all():
                    if allowed_entity_ids is not None and ent.id not in allowed_entity_ids:
                        continue
                    if ent.id not in seen_entities and ent.context_payload:
                        seen_entities.add(ent.id)
                        t = f"[Entity: {ent.canonical_name}]\n{ent.context_payload}"
                        fragments.append(ContextFragment(text=t, source_type="codex", score=1.0,
                                                         token_count=int(len(t.split()) * 1.33)))
            # (b) relation-driven facts: "who inspired ..." → inspired_by edges,
            #     grouped into a single facts fragment (they're a list answer).
            fact_lines = []
            if relations:
                q = self.db.query(CodexEdge).filter(
                    CodexEdge.valid_until == None,
                    CodexEdge.relation.in_(relations))
                if allowed_batch_ids is not None:
                    q = q.filter(CodexEdge.source_batch.in_(allowed_batch_ids))
                for edge in q.order_by(CodexEdge.strength.desc()).limit(self.ENUM_EDGE_LIMIT).all():
                    if self._edge_trust(edge) < self.CODEX_DIRECT_TRUST_FLOOR:
                        continue
                    src = self.db.query(CodexEntity).get(edge.source_id)
                    tgt = self.db.query(CodexEntity).get(edge.target_id)
                    if src and tgt:
                        fact_lines.append(f"[Fact: {src.canonical_name} --{edge.relation}--> {tgt.canonical_name}]")
            if fact_lines:
                t = "\n".join(fact_lines)
                fragments.append(ContextFragment(text=t, source_type="codex", score=1.0,
                                                 token_count=int(len(t.split()) * 1.33)))
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
            all_anchor_edges = []   # A3: reinforced across all anchors at the end
            any_relation_hit = False
            anchor_ids = {a.id for a in matched}
            for anchor in matched:
                if allowed_entity_ids is not None and anchor.id not in allowed_entity_ids:
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
                    token_count=int(len(text.split()) * 1.33)))
                all_anchor_edges.extend(direct_edges)

            if any_relation_hit:
                logger.info("codex_relation_overlap", relations=detected_relations,
                            fragments=len(fragments))
            # A3: retrieval-reinforcement across every anchor's edges.
            self._reinforce_codex_edges(all_anchor_edges)
            return fragments
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
        vf = getattr(edge, "valid_from", None)
        if vf is not None:
            try:
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
        if depth == 0:
            if allowed_batch_ids is None:
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
        # so the graph is navigable both ways (Obsidian backlinks).
        out_edges = self.db.query(CodexEdge).filter(
            CodexEdge.source_id == entity.id, CodexEdge.valid_until == None
        ).order_by(CodexEdge.strength.desc()).all()
        in_edges = self.db.query(CodexEdge).filter(
            CodexEdge.target_id == entity.id, CodexEdge.valid_until == None
        ).order_by(CodexEdge.strength.desc()).all()
        # A5: scope filter (both directions) — no cross-conversation leakage.
        if allowed_batch_ids is not None:
            out_edges = [e for e in out_edges if e.source_batch in allowed_batch_ids]
            in_edges = [e for e in in_edges if e.source_batch in allowed_batch_ids]

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
            # A10/A7.2: don't absorb another matched anchor as a neighbor — it has
            # its own fragment.
            if exclude_ids and other_id in exclude_ids:
                continue
            other = self.db.query(CodexEntity).get(other_id)
            if other:
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
        activating = {"Strategic_Planning", "Generation", "Open_Exploration"}
        if not any(i in classification.intent_tags for i in activating):
            return []

        allowed_batch_ids = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]
            try:
                batch_rows = self.db.execute(
                    text("SELECT DISTINCT batch_id FROM episodic_memory WHERE conversation_id = :cid"),
                    {"cid": conv_id}
                ).fetchall()
                allowed_batch_ids = [row.batch_id for row in batch_rows]
            except Exception:
                self.db.rollback()

        query = text("""
            SELECT id, pattern_description,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM procedural_memory
            WHERE embedding IS NOT NULL AND is_active = true
            ORDER BY score DESC
            LIMIT 5
        """)
        try:
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding}).fetchall()
            fragments = []
            for r in rows:
                pattern = self.db.query(ProceduralMemory).get(r.id)
                if not pattern:
                    continue
                if allowed_batch_ids is not None:
                    if not any(bid in allowed_batch_ids for bid in (pattern.source_batch_ids or [])):
                        continue
                if not self._procedural_trigger_match(pattern, classification):
                    continue
                fragments.append(ContextFragment(
                    text=pattern.pattern_description,
                    source_type="procedural",
                    score=r.score,
                    token_count=int(len(pattern.pattern_description.split()) * 1.33)
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
    # RAG lookup
    # ------------------------------------------------------------------
    def _rag_lookup(self, prompt_embedding, classification) -> List[ContextFragment]:
        if classification.context_reliance != "Long_Term_Memory":
            return []
        if not ("Factual_Retrieval" in classification.intent_tags or
                "Analysis_&_Summarization" in classification.intent_tags):
            return []
        if not any(w in classification.prompt.lower() for w in ["document", "pdf", "reference", "manual", "guide"]):
            return []

        query = text("""
            SELECT chunk_text,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM rag_chunks
            ORDER BY score DESC
            LIMIT 5
        """)
        try:
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding}).fetchall()
            return [ContextFragment(
                text=r.chunk_text,
                source_type="rag",
                score=r.score,
                token_count=int(len(r.chunk_text.split()) * 1.33)
            ) for r in rows]
        except Exception:
            self.db.rollback()
            return []
    def _batch_summary_lookup(self, prompt_embedding, conv_id: Optional[str] = None) -> List[ContextFragment]:
        if not conv_id:
            return []
        try:
            from src.memory.models import BatchSummary
            query = text("""
                SELECT summary_text,
                       1 - (embedding <=> :prompt_embedding) as score
                FROM batch_summaries
                WHERE conversation_id = :conv_id
                  AND embedding IS NOT NULL
                ORDER BY score DESC
                LIMIT 3
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding, "conv_id": conv_id}).fetchall()
            return [ContextFragment(
                text=r.summary_text,
                source_type="batch_summary",
                score=r.score,
                token_count=int(len(r.summary_text.split()) * 1.33)
            ) for r in rows]
        except Exception:
            self.db.rollback()
            return []
    # ------------------------------------------------------------------
    # RRF fusion, diversification, dedup, token budget
    # ------------------------------------------------------------------
    def _apply_rrf(self, legs: Dict[str, List[ContextFragment]], alpha_map: Dict[str, float] = None, k: int = 60) -> List[ContextFragment]:
        from dataclasses import replace
        if alpha_map is None:
            alpha_map = {}

        rrf_scores: Dict[str, float] = {}
        fragment_registry: Dict[str, ContextFragment] = {}

        for leg_name, fragments in legs.items():
            weight = alpha_map.get(leg_name, 1.0)
            fragments.sort(key=lambda x: x.score, reverse=True)
            for rank, frag in enumerate(fragments, start=1):
                frag_hash = hashlib.sha256(frag.text.encode('utf-8')).hexdigest()
                if frag_hash not in fragment_registry:
                    fragment_registry[frag_hash] = frag
                rrf_scores[frag_hash] = rrf_scores.get(frag_hash, 0.0) + (weight / (k + rank))

        fused = []
        for frag_hash, score in rrf_scores.items():
            original = fragment_registry[frag_hash]
            new_frag = replace(original, score=score)
            fused.append(new_frag)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

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

        total, result, used = 0, [], set()
        for f in guaranteed:
            if total + f.token_count <= max_tokens:
                result.append(f); total += f.token_count; used.add(id(f))

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
                # else: fragment too big — skip it, try this leg's next one next round
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
        scope_conv = scope.get("conversation_id") if scope else None
        incognito = bool(scope and scope.get("incognito"))
        conv_filter = "AND conversation_id = :conv_id" if scope_conv else ""
        privacy_filter = "" if scope_conv else "AND is_private = FALSE"
        try:
            query = text(f"""
                SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id,
                       1 - (embedding <=> :prompt_embedding) as score
                FROM episodic_memory
                WHERE embedding IS NOT NULL
                  AND is_archived = false
                  AND decay_score > :min_decay
                  {conv_filter}
                  {privacy_filter}
                ORDER BY score DESC
                LIMIT 100
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            params = {"prompt_embedding": prompt_embedding, "min_decay": 0.2}
            if scope_conv:
                params["conv_id"] = scope_conv
            rows = self.db.execute(query, params).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic", prompt_text=classification.prompt)
        except Exception:
            self.db.rollback()
            fragments = []

        fragments.extend(self._codex_graph(classification, scope,
                                           prompt_embedding=prompt_embedding))
        if not incognito:
            fragments.extend(self._rag_lookup(prompt_embedding, classification))

        fused = self._apply_rrf({"fallback": fragments}, alpha_map={"fallback": 1.0})
        prompt_keywords = self._extract_prompt_keywords(classification.prompt) if classification.prompt else set()
        fused = self._apply_bonuses(fused, classification, conversation_id, prompt_keywords)
        fused.sort(key=lambda x: x.score, reverse=True)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        return self._enforce_token_budget(self._deduplicate(diversified), max_tokens=2000)

    # ------------------------------------------------------------------
    # Helper: convert raw DB rows to ContextFragment list
    # ------------------------------------------------------------------
    def _rows_to_fragments(self, rows, source_type, prompt_text: Optional[str] = None):
        fragments = []
        prompt_keywords = self._extract_prompt_keywords(prompt_text) if prompt_text else set()

        for row in rows:
            if row.inject_raw and row.raw_text:
                text = row.raw_text
            elif row.summary_text:
                text = row.summary_text
            elif row.raw_text:
                text = row.raw_text[:300]
            else:
                continue

            # Word cap (document override / keyword-aware cap)
            is_doc = getattr(row, "is_document", False)
            word_cap = 500
            if is_doc:
                word_cap = 999999
            elif prompt_keywords:
                text_lower = text.lower()
                if any(kw in text_lower or kw.rstrip('s') in text_lower for kw in prompt_keywords):
                    word_cap = 1500

            words = text.split()
            if len(words) > word_cap:
                text = ' '.join(words[:word_cap]) + '…'

            if not text:
                continue

            score_val = float(getattr(row, "score", 1.0))
            if getattr(row, "is_bookmarked", False):
                score_val *= (1.0 + BONUS_BOOKMARKED)
            if hasattr(row, "timestamp") and row.timestamp:
                age_hours = (datetime.now(timezone.utc) - row.timestamp).total_seconds() / 3600.0
                recency_tiebreaker = max(0.0, 0.25 * (1.0 - age_hours / (30 * 24)))
                score_val += recency_tiebreaker
            
                        # Turn‑based recency tiebreaker (max +0.1 for the most recent turn)
            if hasattr(row, "timestamp") and row.timestamp and row.conversation_id:
                turn_count = self.db.query(EpisodicMemory).filter_by(
                    conversation_id=row.conversation_id
                ).count()
                if turn_count > 1:
                    newer_count = self.db.query(EpisodicMemory).filter(
                        EpisodicMemory.conversation_id == row.conversation_id,
                        EpisodicMemory.timestamp > row.timestamp
                    ).count()
                    recency_frac = 1.0 - (newer_count / turn_count)
                    score_val += recency_frac * 0.1
            fragments.append(ContextFragment(
                text=text,
                source_type=source_type,
                score=score_val,
                token_count=int(len(text.split()) * 1.33),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None
            ))
        return fragments