"""Hybrid Retrieval Orchestrator – Phase A hardened: decay filtering, access-weighting,
wide‑net full‑vector, Codex/Procedural scoping, HyDE rewriting, procedural trigger matching."""

import hashlib
import re
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass
from openai import OpenAI
import structlog
from sqlalchemy.orm import Session
from sqlalchemy import bindparam, text
from pgvector.sqlalchemy import Vector as PgVector
from src.api.config import settings
from src.memory.models import (
    EpisodicMemory,
    CodexEntity,
    CodexEdge,
    ProceduralMemory,
    MemorySlot,
)
from src.classifier.classifier import ClassificationResult

logger = structlog.get_logger("ice.retrieval")


@dataclass
class ContextFragment:
    text: str
    source_type: str          # "episodic", "codex", "procedural", "rag"
    score: float              # RRF fused score
    token_count: int
    source_batch_id: Optional[str] = None
    conversation_id: Optional[str] = None


class HybridRetrievalOrchestrator:
    def __init__(self, db: Session, embedder):
        self.db = db
        self.embedder = embedder
        self.bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

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
        """Orchestrate multi‑source retrieval."""
        if classification.context_reliance == "Zero_Shot":
            return []
        if classification.context_reliance == "Real_Time_Search":
            return []

        if classification.max_confidence < settings.confidence_fallback_threshold:
            logger.info("wide_net_fallback_triggered", confidence=classification.max_confidence)
            return self._wide_net_fallback(classification, prompt_embedding, conversation_id, scope)

        # Conversation scope filter for episodic legs
        conv_id = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]

        # HyDE query rewriting
        hyde_prompt = None
        if classification.context_reliance == "Long_Term_Memory":
            hyde_prompt = self._hyde_rewrite(classification.prompt)
        search_prompt = hyde_prompt if hyde_prompt else classification.prompt

        # Re‑compute embedding if the search prompt changed
        if hyde_prompt:
            prompt_embedding = self.embedder.encode(search_prompt, convert_to_tensor=False).tolist()

        # Execute all retrieval legs
        legs: Dict[str, List[ContextFragment]] = {
            "bm25": self._bm25_episodic(classification, scope, conv_id, search_prompt),
            "vector": self._vector_episodic(prompt_embedding, classification, scope, conv_id),
            "codex": self._codex_graph(classification, scope),
            "procedural": self._procedural_lookup(prompt_embedding, classification, scope),
            "rag": self._rag_lookup(prompt_embedding, classification),
        }

        fused = self._apply_rrf(legs)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        deduped = self._deduplicate(diversified)
        final = self._enforce_token_budget(deduped, max_tokens=2000)

        # Strengthen retrieved turns (access count + decay boost)
        self._strengthen_retrieved(final)

        return final

    # ------------------------------------------------------------------
    # HyDE rewriting
    # ------------------------------------------------------------------
    def _hyde_rewrite(self, prompt: str) -> Optional[str]:
        try:
            resp = self.bg_client.chat.completions.create(
                model="Qwen/Qwen2.5-3B-Instruct-AWQ",
                messages=[
                    {"role": "system", "content": (
                        "You are a query rewriting engine. Take the user's question and rewrite it "
                        "as a dense, factual search query that would retrieve the relevant past conversation. "
                        "Include key entities and omit polite phrasing. Output ONLY the rewritten query, no other text."
                    )},
                    {"role": "user", "content": prompt},
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
    def _bm25_episodic(self, classification, scope, conv_id: Optional[str] = None, search_prompt: Optional[str] = None) -> List[ContextFragment]:
        prompt_text = search_prompt if search_prompt else classification.prompt
        clean_prompt = re.sub(r'[^\w\s]', ' ', prompt_text)
        search_words = [w for w in clean_prompt.split() if w][:30]
        search_terms = " & ".join(search_words) if search_words else "ice"

        topic_filter = "AND topic_tags && :topics" if classification.topic_tags else ""
        conv_filter = "AND conversation_id = :conv_id" if conv_id else ""

        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id,
                   ts_rank(
                       to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')),
                       query
                   ) as score
            FROM episodic_memory, to_tsquery('english', :search_terms) query
            WHERE to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')) @@ query
              {topic_filter}
              {conv_filter}
              AND decay_score > :min_decay
              AND is_archived = false
            ORDER BY score DESC
            LIMIT 10
        """)

        params = {"search_terms": search_terms, "min_decay": 0.2}
        if classification.topic_tags:
            params["topics"] = classification.topic_tags
        if conv_id:
            params["conv_id"] = conv_id

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic")
        except Exception as err:
            logger.error("bm25_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    # ------------------------------------------------------------------
    # Vector episodic (pgvector cosine similarity)
    # ------------------------------------------------------------------
    def _vector_episodic(self, prompt_embedding, classification, scope, conv_id: Optional[str] = None) -> List[ContextFragment]:
        topic_filter = "AND topic_tags && :topics" if classification.topic_tags else ""
        conv_filter = "AND conversation_id = :conv_id" if conv_id else ""

        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id,
                1 - (embedding <=> :prompt_embedding) as score
            FROM episodic_memory
            WHERE embedding IS NOT NULL
            {topic_filter}
            {conv_filter}
            AND decay_score > :min_decay
            AND is_archived = false
            ORDER BY score DESC
            LIMIT 10
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        params = {"prompt_embedding": prompt_embedding, "min_decay": 0.2}
        if classification.topic_tags:
            params["topics"] = classification.topic_tags
        if conv_id:
            params["conv_id"] = conv_id

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic")
        except Exception as err:
            logger.error("vector_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    # ------------------------------------------------------------------
    # Codex graph traversal (conversation‑scoped)
    # ------------------------------------------------------------------
    def _codex_graph(self, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
        prompt = classification.prompt
        candidates = set(re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', prompt))
        if not candidates:
            return []

        normalized = [c.lower().strip() for c in candidates]
        from sqlalchemy import or_
        alias_conditions = [CodexEntity.aliases.any(name) for name in normalized]

        try:
            entities = self.db.query(CodexEntity).filter(
                or_(CodexEntity.canonical_name.in_(normalized), *alias_conditions)
            ).all()

            # Scoping: restrict to entities that appear in the target conversation
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
            for entity in entities:
                if allowed_entity_ids is not None and entity.id not in allowed_entity_ids:
                    continue
                self._traverse_graph(entity, 0, 2, visited, context_texts)

            if context_texts:
                combined = "\n\n".join(context_texts)
                return [ContextFragment(
                    text=combined,
                    source_type="codex",
                    score=1.0,
                    token_count=int(len(combined.split()) * 1.33)
                )]
            return []
        except Exception as err:
            logger.error("codex_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    def _traverse_graph(self, entity, depth, max_depth, visited, context_texts):
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
            target = self.db.query(CodexEntity).get(edge.target_id)
            if target:
                self._traverse_graph(target, depth + 1, max_depth, visited, context_texts)

    # ------------------------------------------------------------------
    # Procedural lookup (scoped + trigger‑condition evaluation)
    # ------------------------------------------------------------------
    def _procedural_lookup(self, prompt_embedding, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
        activating = {"Strategic_Planning", "Generation", "Open_Exploration"}
        if not any(i in classification.intent_tags for i in activating):
            return []

        # Scoping: collect batch_ids for the target conversation
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
                # Scope filter
                if allowed_batch_ids is not None:
                    if not any(bid in allowed_batch_ids for bid in (pattern.source_batch_ids or [])):
                        continue
                # Trigger condition match
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
    # RAG lookup (unchanged)
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

    # ------------------------------------------------------------------
    # RRF fusion, diversification, dedup, token budget
    # ------------------------------------------------------------------
    def _apply_rrf(self, legs: Dict[str, List[ContextFragment]], k: int = 60) -> List[ContextFragment]:
        rrf_scores: Dict[str, float] = {}
        fragment_registry: Dict[str, ContextFragment] = {}

        for leg_name, fragments in legs.items():
            fragments.sort(key=lambda x: x.score, reverse=True)
            for rank, frag in enumerate(fragments, start=1):
                frag_hash = hashlib.sha256(frag.text.encode('utf-8')).hexdigest()
                if frag_hash not in fragment_registry:
                    fragment_registry[frag_hash] = frag
                rrf_scores[frag_hash] = rrf_scores.get(frag_hash, 0.0) + (1.0 / (k + rank))

        fused = []
        for frag_hash, score in rrf_scores.items():
            fragment_registry[frag_hash].score = score
            fused.append(fragment_registry[frag_hash])

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

    def _enforce_token_budget(self, fragments, max_tokens=2000):
        total = 0
        result = []
        for f in fragments:
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
                LIMIT 10
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = self.db.execute(query, {
                "prompt_embedding": prompt_embedding,
                "min_decay": 0.2
            }).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic")
        except Exception:
            self.db.rollback()
            fragments = []

        fragments.extend(self._codex_graph(classification))
        fragments.extend(self._rag_lookup(prompt_embedding, classification))

        fused = self._apply_rrf({"fallback": fragments})
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        return self._enforce_token_budget(self._deduplicate(diversified), max_tokens=2000)

    # ------------------------------------------------------------------
    # Helper: convert raw DB rows to ContextFragment list
    # ------------------------------------------------------------------
    def _rows_to_fragments(self, rows, source_type):
        fragments = []
        for row in rows:
            if row.inject_raw and row.raw_text:
                text = row.raw_text
            elif row.summary_text:
                text = row.summary_text
            elif row.raw_text:
                text = row.raw_text[:300]
            else:
                continue

            words = text.split()
            if len(words) > 500:
                text = ' '.join(words[:500]) + '…'

            if not text:
                continue

            score_val = getattr(row, "score", 1.0)
            fragments.append(ContextFragment(
                text=text,
                source_type=source_type,
                score=float(score_val),
                token_count=int(len(text.split()) * 1.33),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None
            ))
        return fragments