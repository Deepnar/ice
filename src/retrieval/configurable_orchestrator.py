"""Configurable Hybrid Retrieval Orchestrator — feature‑flagged for ablation experiments.

Inherits from HybridRetrievalOrchestrator and allows individual retrieval legs
and post‑processing steps to be toggled on/off via an `overrides` dict.

Flags (all default ON unless noted):
    vector, bm25, rrf, hyde (default OFF), cluster_restrict, session_diversify,
    codex, mera, fuzzy_match, procedural, batch_summary,
    dynamic_budget, sliding_window, keyword_boost, recency_boost, timescope
"""

from typing import List, Optional, Dict
import hashlib
from dataclasses import replace
from sqlalchemy.orm import Session
import structlog

from src.classifier.schemas import ClassificationResult
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
        # T2/T3 ablation seam: timescope=False forces CURRENT in the parent's
        # _resolve_timescope — every temporal branch collapses to pre-T
        # behavior. The legs themselves are untouched (timescope travels
        # inside `scope`, so every overridden super() call keeps working).
        self.timescope_allowed = self._on("timescope")

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

    # ── Codex graph — flag mapping over the base implementation ──────────
    # The base leg (A3/A4) natively supports exact-vs-fuzzy matching and the
    # relation/tag enumeration that replaced MERA, so this override just maps
    # ablation flags onto base-class switches. The `mera` flag now toggles the
    # re-homed enumeration capability (capability-level ablation continuity).

    def _codex_graph(self, classification, scope: Optional[dict] = None,
                     prompt_embedding=None) -> List[ContextFragment]:
        if self._off("codex"):
            return []
        self.use_fuzzy_match = self._on("fuzzy_match")
        self.enable_enumeration = self._on("mera")
        return super()._codex_graph(classification, scope, prompt_embedding)

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

    def set_budget_from_turn_count(self, turn_count, total_tokens=0, classification=None,
                                   total_budget=None):
        if self._off("dynamic_budget"):
            self.max_retrieval_tokens = 8000
            self.recent_token_budget = 4000
        else:
            super().set_budget_from_turn_count(turn_count, total_tokens, classification,
                                               total_budget=total_budget)

    # ── Cluster restriction override ─────────────────────────────────────

    def _relevant_cluster_ids(self, prompt_embedding, classification=None, conversation_id=None, top_k=3):
        if self._off("cluster_restrict"):
            return []
        return super()._relevant_cluster_ids(prompt_embedding, classification, conversation_id, top_k)