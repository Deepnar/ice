"""Hybrid Retrieval Orchestrator – Production implementation with true RRF."""

import hashlib
import re
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
        # Gate: Zero_Shot → no retrieval
        if classification.context_reliance == "Zero_Shot":
            return []
        if classification.context_reliance == "Real_Time_Search":
            return []  # web search stub

        # Confidence fallback → wide‑net
        if classification.max_confidence < settings.confidence_fallback_threshold:
            logger.info("wide_net_fallback_triggered", confidence=classification.max_confidence)
            return self._wide_net_fallback(classification, prompt_embedding, conversation_id, scope)

        # Execute all retrieval legs
        legs: Dict[str, List[ContextFragment]] = {
            "bm25": self._bm25_episodic(classification, scope),
            "vector": self._vector_episodic(prompt_embedding, classification, scope),
            "codex": self._codex_graph(classification),
            "procedural": self._procedural_lookup(prompt_embedding, classification),
            "rag": self._rag_lookup(prompt_embedding, classification),
        }

        # Fuse with true RRF
        fused = self._apply_rrf(legs)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        deduped = self._deduplicate(diversified)
        return self._enforce_token_budget(deduped, max_tokens=2000)

    # ------------------------------------------------------------------
    # BM25 episodic (full‑text search)
    # ------------------------------------------------------------------
    def _bm25_episodic(self, classification, scope) -> List[ContextFragment]:
        clean_prompt = re.sub(r'[^\w\s]', ' ', classification.prompt)
        search_words = [w for w in clean_prompt.split() if w][:30]
        search_terms = " & ".join(search_words) if search_words else "ice"

        topic_filter = "AND topic_tags && :topics" if classification.topic_tags else ""
        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, conversation_id,
                   ts_rank(
                       to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')),
                       query
                   ) as score
            FROM episodic_memory, to_tsquery('english', :search_terms) query
            WHERE to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')) @@ query
              {topic_filter}
              AND is_archived = false
            ORDER BY score DESC
            LIMIT 10
        """)

        params = {"search_terms": search_terms}
        if classification.topic_tags:
            params["topics"] = classification.topic_tags

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
    def _vector_episodic(self, prompt_embedding, classification, scope) -> List[ContextFragment]:
        topic_filter = "AND topic_tags && :topics" if classification.topic_tags else ""
        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, conversation_id,
                1 - (embedding <=> :prompt_embedding) as score
            FROM episodic_memory
            WHERE embedding IS NOT NULL
            {topic_filter}
            AND is_archived = false
            ORDER BY score DESC
            LIMIT 10
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        params = {"prompt_embedding": prompt_embedding}
        if classification.topic_tags:
            params["topics"] = classification.topic_tags

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic")
        except Exception as err:
            logger.error("vector_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    # ------------------------------------------------------------------
    # Codex graph traversal (NER → entity lookup → 1‑2 hop traversal)
    # ------------------------------------------------------------------
    def _codex_graph(self, classification) -> List[ContextFragment]:
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

            visited = set()
            context_texts = []
            for entity in entities:
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
    # Procedural lookup (only for certain intents)
    # ------------------------------------------------------------------
    def _procedural_lookup(self, prompt_embedding, classification) -> List[ContextFragment]:
        activating = {"Strategic_Planning", "Generation", "Open_Exploration"}
        if not any(i in classification.intent_tags for i in activating):
            return []

        query = text("""
            SELECT pattern_description,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM procedural_memory
            WHERE embedding IS NOT NULL AND is_active = true
            ORDER BY score DESC
            LIMIT 5
        """)
        try:
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding}).fetchall()
            return [ContextFragment(
                text=r.pattern_description,
                source_type="procedural",
                score=r.score,
                token_count=int(len(r.pattern_description.split()) * 1.33)
            ) for r in rows]
        except Exception:
            self.db.rollback()
            return []

    # ------------------------------------------------------------------
    # RAG lookup (only for factual / analysis with reference language)
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
    # Reciprocal Rank Fusion (architecture‑specified k=60)
    # ------------------------------------------------------------------
    def _apply_rrf(self, legs: Dict[str, List[ContextFragment]], k: int = 60) -> List[ContextFragment]:
        """True RRF: rank each leg independently, then 1/(k + rank)."""
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

    # ------------------------------------------------------------------
    # Session diversification, deduplication, token budget
    # ------------------------------------------------------------------
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
    # Wide‑net fallback (confidence safety net)
    # ------------------------------------------------------------------
    def _wide_net_fallback(self, classification, prompt_embedding, conversation_id, scope):
        """Fallback retrieval when classifier confidence is low."""
        try:
            rows = self.db.execute(text("""
                SELECT id, raw_text, summary_text, lossless_flag, conversation_id
                FROM episodic_memory
                WHERE is_archived = false
                ORDER BY timestamp DESC
                LIMIT 20
            """)).fetchall()
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
            text = row.raw_text if row.lossless_flag else (row.summary_text or row.raw_text[:300])
            if not text:
                continue
            score_val = getattr(row, "score", 1.0)   # safe fallback for queries without score
            fragments.append(ContextFragment(
                text=text,
                source_type=source_type,
                score=float(score_val),
                token_count=int(len(text.split()) * 1.33),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None
            ))
        return fragments