"""Configurable Hybrid Retrieval Orchestrator — feature‑flagged for ablation experiments.

Inherits from HybridRetrievalOrchestrator and allows individual retrieval legs
and post‑processing steps to be toggled on/off via an `overrides` dict.

Flags (all default ON unless noted):
    vector, bm25, rrf, hyde (default OFF), cluster_restrict, session_diversify,
    codex, mera, fuzzy_match, procedural, batch_summary,
    dynamic_budget, sliding_window, keyword_boost, recency_boost
"""

from typing import List, Optional, Dict
import hashlib
from dataclasses import replace
from sqlalchemy.orm import Session
import structlog

from src.classifier.schemas import ClassificationResult
from src.memory.models import CodexEntity
from src.retrieval.ner_utils import extract_entities   # ← ADD THIS LINE
from src.retrieval.orchestrator import (
    HybridRetrievalOrchestrator, ContextFragment,
    BONUS_RECENT_TOP_10PCT, BONUS_RECENT_TOP_30PCT,
)

logger = structlog.get_logger("ice.retrieval.configurable")


class ConfigurableOrchestrator(HybridRetrievalOrchestrator):
    """HybridRetrievalOrchestrator with per‑feature toggles for ablation studies."""

    def __init__(self, db: Session, embedder, overrides: Optional[Dict[str, bool]] = None):
        """Initialise with an optional overrides dict.

        Default (overrides=None or key missing): feature is ON (True).
        Set a key to False to disable that feature.
        Special: 'hyde' defaults to False (OFF) — set to True to enable.
        """
        super().__init__(db, embedder)
        self.overrides = overrides or {}

    # ── Convenience helpers ──────────────────────────────────────────────

    def _on(self, flag: str) -> bool:
        """Return True if *flag* is enabled.  'hyde' defaults to False; all others default to True."""
        default = False if flag == "hyde" else True
        return self.overrides.get(flag, default)

    def _off(self, flag: str) -> bool:
        """Return True if *flag* is explicitly disabled."""
        return not self._on(flag)

    # ── Override retrieval legs ──────────────────────────────────────────

    def _bm25_episodic(self, classification, scope, conv_id=None, search_prompt=None):
        if self._off("bm25"):
            return []
        return super()._bm25_episodic(classification, scope, conv_id, search_prompt)

    def _vector_episodic(self, prompt_embedding, classification, scope, conv_id=None):
        if self._off("vector"):
            return []
        return super()._vector_episodic(prompt_embedding, classification, scope, conv_id)

    def _procedural_lookup(self, prompt_embedding, classification, scope=None):
        if self._off("procedural"):
            return []
        return super()._procedural_lookup(prompt_embedding, classification, scope)

    def _batch_summary_lookup(self, prompt_embedding, conv_id=None):
        if self._off("batch_summary"):
            return []
        return super()._batch_summary_lookup(prompt_embedding, conv_id)

    def _rag_lookup(self, prompt_embedding, classification):
        if self._off("rag"):
            return []
        return super()._rag_lookup(prompt_embedding, classification)

    # ── Codex graph — fully overridden to support mera / fuzzy_match flags ──

    def _codex_graph(self, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
        """Codex graph traversal with optional MERA and fuzzy‑match toggles."""
        if self._off("codex"):
            return []

        prompt = classification.prompt
        entity_strings = extract_entities(prompt, self.embedder)
        if entity_strings:
            if self._off("fuzzy_match"):
                matched = self._match_entities_exact(entity_strings)
            else:
                matched = self._match_entities_by_similarity(entity_strings)
        else:
            if self._off("mera"):
                return []
            from src.retrieval.mera import is_mera_candidate, map_category_to_filters, enumerate_entities
            if is_mera_candidate(prompt):
                filters = map_category_to_filters(self.db, prompt)
                matched = enumerate_entities(self.db, filters.get("tags", []), filters.get("relations", []))
            else:
                return []

        if not matched:
            return []

        try:
            allowed_entity_ids = None
            if scope and "conversation_id" in scope:
                conv_id = scope["conversation_id"]
                try:
                    from sqlalchemy import text
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
            for entity in matched:
                if allowed_entity_ids is not None and entity.id not in allowed_entity_ids:
                    continue
                self._traverse_graph(entity, 0, 3, visited, context_texts)

            if context_texts:
                combined = "\n\n".join(context_texts)
                score = 1.0
                from src.memory.models import CodexEdge
                if any(e.confidence == "active" and e.strength >= 2.0 for e in
                    self.db.query(CodexEdge).filter(
                        CodexEdge.source_id.in_([e.id for e in matched]),
                        CodexEdge.valid_until == None
                    ).all()):
                    score *= 1.5
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

    def _match_entities_exact(self, entity_strings: List[str]) -> List:
        """Match entity strings using ONLY exact canonical name or alias match (no vector similarity)."""
        from sqlalchemy import or_
        matched = []
        seen = set()
        for candidate_str in entity_strings:
            norm = candidate_str.lower().strip()
            if norm in seen:
                continue
            ent = self.db.query(CodexEntity).filter(
                or_(CodexEntity.canonical_name == norm, CodexEntity.aliases.any(norm))
            ).first()
            if ent and ent.id not in seen:
                matched.append(ent)
                seen.add(ent.id)
        return matched

    # ── Override post‑processing ─────────────────────────────────────────

    def _apply_rrf(self, legs, alpha_map=None, k=60):
        if self._off("rrf"):
            return self._simple_merge(legs)
        return super()._apply_rrf(legs, alpha_map=alpha_map, k=k)

    def _simple_merge(self, legs: Dict[str, List[ContextFragment]]) -> List[ContextFragment]:
        """Concatenate all leg results and sort by original score (no RRF)."""
        merged = []
        seen = set()
        for _leg_name, fragments in legs.items():
            for frag in fragments:
                h = hashlib.sha256(frag.text.encode('utf-8')).hexdigest()
                if h not in seen:
                    seen.add(h)
                    merged.append(frag)
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged

    def _session_diversify(self, fragments, current_id, max_per_conversation=3):
        if self._off("session_diversify"):
            return fragments
        return super()._session_diversify(fragments, current_id, max_per_conversation)

    def _apply_bonuses(self, fragments, classification, conv_id, prompt_keywords):
        """Optionally skip keyword and/or recency boosts."""
        # Both off → skip entirely
        if self._off("keyword_boost") and self._off("recency_boost"):
            return fragments

        # Only recency → clear keywords
        if self._off("keyword_boost") and self._on("recency_boost"):
            return super()._apply_bonuses(fragments, classification, conv_id, set())

        # Only keyword → temporarily zero out recency bonus constants
        if self._on("keyword_boost") and self._off("recency_boost"):
            global BONUS_RECENT_TOP_10PCT, BONUS_RECENT_TOP_30PCT
            saved_top = BONUS_RECENT_TOP_10PCT
            saved_30  = BONUS_RECENT_TOP_30PCT
            BONUS_RECENT_TOP_10PCT = 0.0
            BONUS_RECENT_TOP_30PCT  = 0.0
            try:
                return super()._apply_bonuses(fragments, classification, conv_id, prompt_keywords)
            finally:
                BONUS_RECENT_TOP_10PCT = saved_top
                BONUS_RECENT_TOP_30PCT  = saved_30

        # Both on → normal path
        return super()._apply_bonuses(fragments, classification, conv_id, prompt_keywords)

    # ── Budget overrides ─────────────────────────────────────────────────

    def set_budget_from_turn_count(self, turn_count, total_tokens=0, classification=None):
        if self._off("dynamic_budget"):
            self.max_retrieval_tokens = 8000
            self.recent_token_budget = 4000
        else:
            super().set_budget_from_turn_count(turn_count, total_tokens, classification)

    # ── Cluster restriction override ─────────────────────────────────────

    def _relevant_cluster_ids(self, prompt_embedding, classification=None, conversation_id=None, top_k=3):
        if self._off("cluster_restrict"):
            return []
        return super()._relevant_cluster_ids(prompt_embedding, classification, conversation_id, top_k)