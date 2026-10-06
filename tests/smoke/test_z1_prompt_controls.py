"""The instrument suppresses search bypasses without changing full-arm helpers."""
from types import SimpleNamespace

import pytest

from scripts.z1 import seed_v3
from src.api import memory_preparation as mp
from src.api.config import settings
from src.retrieval import orchestrator as retrieval


@pytest.mark.parametrize("arm", seed_v3.PROMPT_ARMS)
@pytest.mark.parametrize("weak", [False, True])
def test_controls_keep_budgeting_but_suppress_out_of_arm_searches(monkeypatch, arm, weak):
    calls = []
    vector = retrieval.ContextFragment("Source turn", "episodic", 1, 3, leg="vector")
    note = retrieval.ContextFragment("Store-derived eras", "episodic", 1, 3)
    class Orchestrator:
        def __init__(self, *_a):
            for name in ("_bm25_episodic", "_codex_graph", "_codex_claims", "_procedural_lookup",
                         "_batch_summary_lookup", "_cold_lookup"):
                setattr(self, name, lambda *_a, **_k: [])
            self._vector_episodic = lambda *_a, **_k: [vector]
        def set_budget_from_turn_count(self, *_a, **_k): calls.append("budget")
        def _enforce_token_budget(self, rows): return rows
        def _append_empty_window_note(self, rows, *_a, **_k):
            calls.append("era_query")
            return rows + [note]
        def retrieve(self, *_a, **_k):
            calls.append("search")
            if weak and settings.confidence_fallback_threshold > 0:
                calls.append("wide_net")
            return self._append_empty_window_note(self._enforce_token_budget(self._vector_episodic()))
    monkeypatch.setattr(retrieval, "HybridRetrievalOrchestrator", Orchestrator)
    monkeypatch.setattr(settings, "confidence_fallback_threshold", 0.5)
    monkeypatch.setattr(settings, "context_use_serving_window", False)
    from src.model_registry import registry
    monkeypatch.setattr(registry, "get_model_context_window", lambda *_: 8192)
    monkeypatch.setattr(seed_v3.pp, "prepare", lambda db, pre, classifier, **kw: prepare(db, classifier))
    def prepare(db, classifier):
        orchestrator = mp.HybridRetrievalOrchestrator(db, classifier)
        orchestrator.set_budget_from_turn_count(3)
        fragments = orchestrator.retrieve()
        return SimpleNamespace(fragments=fragments, retrieve=True, action="keep",
                               prepared=SimpleNamespace(messages=[], ledger=SimpleNamespace(fits=lambda: True)))
    query = SimpleNamespace(filter_by=lambda **kw: SimpleNamespace(all=lambda: []))
    db = SimpleNamespace(query=lambda *_a: query)
    from src.api import prompt_assembler
    monkeypatch.setattr(prompt_assembler, "bookmarked_turn_texts", lambda *_: [])
    pre = SimpleNamespace(model_name="fixture", conversation_id="fixture")
    result = seed_v3.prepare_with_trace(db, pre, None, arm=arm)[0]
    assert calls[0] == "budget" and settings.confidence_fallback_threshold == 0.5
    if arm == "recent_only":
        assert calls == ["budget"] and not result.retrieve and not result.fragments
    elif arm == "vector_only":
        assert calls == ["budget", "search"] and result.fragments == [vector]
    else:
        assert "era_query" in calls and ("wide_net" in calls) is weak
        assert result.fragments == [vector, note]
