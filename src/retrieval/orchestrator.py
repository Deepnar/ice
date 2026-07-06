"""Hybrid Retrieval Orchestrator – Phase A hardened: decay filtering, access-weighting,
wide‑net full‑vector, Codex/Procedural scoping, HyDE rewriting, procedural trigger matching,
micro‑NER integration, and dynamic token budget."""

from datetime import datetime, timezone
import hashlib
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

class HybridRetrievalOrchestrator:
    def __init__(self, db: Session, embedder):
        self.db = db
        self.embedder = embedder
        self.bg_client = get_bg_client()
        self.max_retrieval_tokens = 5000
        self._force_hyde = False
        # MERA (category/enumeration fallback) is disabled by default: it scored
        # −0.21 in the buildup ablation and its capability is being re-homed into
        # relation-aware retrieval (roadmap A4). The ablation ConfigurableOrchestrator
        # can still enable it via its own `mera` flag. See ROADMAP.md P0.2.
        self.enable_mera = False

        # A3 — edge confidence/strength as a live retrieval signal.
        self.CODEX_MAX_DEPTH = 3                 # traversal ceiling (gated below)
        self.CODEX_DEEP_STRENGTH_FLOOR = 1.0     # deep hops require this much strength
        self.CODEX_REINFORCE_INCREMENT = 0.15    # per-retrieval boost on anchor edges (episodic analog)
        self.CODEX_STRENGTH_CAP = 10.0           # soft ceiling so retrieval can't inflate forever

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



    TOTAL_CONTEXT_BUDGET = 23_000          # hard ceiling for all context
    OVERHEAD_RESERVE = 1_800               # system message + slots + question


    def set_budget_from_turn_count(
        self, turn_count: int, total_tokens: int = 0, classification=None
    ):
        available = self.TOTAL_CONTEXT_BUDGET - self.OVERHEAD_RESERVE
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
        # ── Safety override: if we have a conversation scope, always check memory ──
        if classification.context_reliance == "Zero_Shot" and conversation_id:
            classification.context_reliance = "Long_Term_Memory"

        # Keep the explicit creative/lore guard (belt and suspenders)
        if "Creative_&_Media" in classification.topic_tags:
            classification.context_reliance = "Long_Term_Memory"
        # Reset HyDE tracking flags for this retrieval call
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
        search_prompt = classification.prompt
        # Execute all retrieval legs
        legs: Dict[str, List[ContextFragment]] = {
            "bm25": self._bm25_episodic(classification, scope, conv_id, search_prompt),
            "vector": self._vector_episodic(prompt_embedding, classification, scope, conv_id),
            "codex": self._codex_graph(classification, scope),
            "procedural": self._procedural_lookup(prompt_embedding, classification, scope),
            "rag": self._rag_lookup(prompt_embedding, classification),
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
    # Codex graph traversal (conversation‑scoped, NER‑powered)
    # ------------------------------------------------------------------
    def _codex_graph(self, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
        prompt = classification.prompt
        entity_strings = extract_entities(prompt, self.embedder)
        if entity_strings:
            matched = self._match_entities_by_similarity(entity_strings)
        elif self.enable_mera:
            from src.retrieval.mera import is_mera_candidate, map_category_to_filters, enumerate_entities
            if is_mera_candidate(prompt):
                filters = map_category_to_filters(self.db, prompt)
                matched = enumerate_entities(self.db, filters.get("tags", []), filters.get("relations", []))
            else:
                return []
        else:
            return []

        if not matched:
            return []

        try:
            allowed_entity_ids = None
            if scope and "conversation_id" in scope:
                conv_id = scope["conversation_id"]
                try:
                    batch_rows = self.db.execute(
                        text("SELECT DISTINCT batch_id FROM episodic_memory WHERE conversation_id = :cid"),
                        {"cid": conv_id}
                    ).fetchall()
                    batch_ids = [row.batch_id for row in batch_rows]
                    if batch_ids:
                        event_rows = self.db.execute(
                            text("SELECT DISTINCT entity_id FROM codex_events WHERE batch_source = ANY(:bids)"),
                            {"bids": batch_ids}
                        ).fetchall()
                        allowed_entity_ids = {row.entity_id for row in event_rows}
                except Exception:
                    self.db.rollback()

            visited = set()
            context_texts = []
            anchor_edges = []   # A3: direct edges of matched entities → reinforced on retrieval
            for entity in matched:
                if allowed_entity_ids is not None and entity.id not in allowed_entity_ids:
                    continue
                self._traverse_graph(entity, 0, self.CODEX_MAX_DEPTH, visited,
                                     context_texts, anchor_edges)

            if context_texts:
                combined = "\n\n".join(context_texts)
                # A3: graded score from matched entities' edge strength, replacing
                # the coarse binary 1.5x. Bounded to 1.0–1.5 so codex doesn't
                # dominate fusion on strength alone.
                matched_edges = self.db.query(CodexEdge).filter(
                    CodexEdge.source_id.in_([e.id for e in matched]),
                    CodexEdge.valid_until == None
                ).all()
                mean_strength = (sum((e.strength or 0.0) for e in matched_edges) / len(matched_edges)
                                 if matched_edges else 0.0)
                score = 1.0 + min(0.5, 0.25 * mean_strength)

                # A3: retrieval-reinforcement — the codex analog of episodic
                # access_count/decay_score strengthening (only the anchor edges).
                self._reinforce_codex_edges(anchor_edges)

                return [ContextFragment(
                    text=combined,
                    source_type="codex",
                    score=score,
                    token_count=int(len(combined.split()) * 1.33)
                )]
            return []
        except Exception as err:
            logger.error("codex_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    def _traverse_graph(self, entity, depth, max_depth, visited, context_texts, anchor_edges=None):
        if entity.id in visited or depth > max_depth:
            return
        visited.add(entity.id)
        if entity.context_payload:
            context_texts.append(f"[Entity: {entity.canonical_name}]\n{entity.context_payload}")
        edges = self.db.query(CodexEdge).filter(
            CodexEdge.source_id == entity.id,
            CodexEdge.valid_until == None
        ).all()
        for edge in edges:
            # A3: direct edges of a matched (depth-0) entity are the query anchors.
            if depth == 0 and anchor_edges is not None:
                anchor_edges.append(edge)
            # A3: strength-gate deep hops — weak/decayed edges don't expand the
            # frontier, cutting the depth-3 pollution. Direct edges always expand.
            elif depth >= 1 and (edge.strength or 0.0) < self.CODEX_DEEP_STRENGTH_FLOOR:
                continue
            target = self.db.query(CodexEntity).get(edge.target_id)
            if target:
                self._traverse_graph(target, depth + 1, max_depth, visited,
                                     context_texts, anchor_edges)

    def _reinforce_codex_edges(self, edges):
        """A3 retrieval-reinforcement (episodic analog): the anchor edges of the
        matched query entities gain a little strength each time they're surfaced,
        so repeatedly-useful facts self-promote through use — balanced by the
        codex_decay worker. Scoped to anchor edges only to avoid diluting the
        signal across the whole traversed neighborhood. Write-on-read, like
        _strengthen_retrieved does for episodic turns."""
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

        # Step 1 – guarantee each leg that produced results gets its best fragment in
        best_per_leg = {}
        for f in fragments:
            leg = f.source_type
            if leg not in best_per_leg or f.score > best_per_leg[leg].score:
                best_per_leg[leg] = f

        guaranteed = list(best_per_leg.values())
        guaranteed.sort(key=lambda x: x.score, reverse=True)

        total = 0
        result = []
        for f in guaranteed:
            if total + f.token_count <= max_tokens:
                result.append(f)
                total += f.token_count

        # Step 2 – greedily fill the rest of the budget with remaining fragments
        remaining = [f for f in fragments if f not in guaranteed]
        for f in remaining:
            if total + f.token_count <= max_tokens:
                result.append(f)
                total += f.token_count

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
        try:
            query = text("""
                SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id,
                       1 - (embedding <=> :prompt_embedding) as score
                FROM episodic_memory
                WHERE embedding IS NOT NULL
                  AND is_archived = false
                  AND decay_score > :min_decay
                ORDER BY score DESC
                LIMIT 100
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = self.db.execute(query, {
                "prompt_embedding": prompt_embedding,
                "min_decay": 0.2
            }).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic", prompt_text=classification.prompt)
        except Exception:
            self.db.rollback()
            fragments = []

        fragments.extend(self._codex_graph(classification))
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