"""v3 ranking, representation and budgeting contracts with controlled scores."""

from types import SimpleNamespace

import pytest

from src.api.config import settings
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator
from src.retrieval.reranker import rerank


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_rerank_enabled", True)
    monkeypatch.setattr(settings, "retrieval_rerank_min_score", 0.0)


def fragment(text, kind="episodic", **kwargs):
    return ContextFragment(text, kind, 1.0, 10, **kwargs)


def test_relevance_overrides_leg_entitlement():
    candidates = [
        fragment("distractor", "procedural"),
        fragment("answer"),
        fragment("support"),
    ]
    ranked, ok = rerank("query", candidates, lambda pairs: [-4, 5, 3])
    assert ok
    o = HybridRetrievalOrchestrator(None, None)
    selected = o._enforce_token_budget(ranked, max_tokens=20, relevance_order=ok)
    assert [f.text for f in selected] == ["answer", "support"]


def test_degraded_form_cannot_borrow_relevance():
    raw = "long evidence " * 100
    candidates = [fragment(raw, degrade_text="unrelated summary")]
    ranked, ok = rerank("query", candidates, lambda pairs: [5, -2])
    assert ok and ranked[0].degrade_text is None
    assert not HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        ranked,
        max_tokens=10,
        relevance_order=ok,
    )


def test_better_alternative_preserves_provenance():
    candidates = [
        fragment(
            "long evidence " * 100,
            degrade_text="compact answer",
            source_batch_id="turn",
            origin_batch_ids=("batch",),
            leg="bm25+vector",
        )
    ]
    ranked, ok = rerank("query", candidates, lambda pairs: [1, 5])
    assert ok and ranked[0].text == "compact answer"
    assert ranked[0].source_batch_id == "turn"
    assert ranked[0].origin_batch_ids == ("batch",)
    assert ranked[0].leg == "bm25+vector"


def test_no_relevant_candidates_means_empty_context():
    assert rerank("query", [fragment("distractor")], lambda pairs: [-1]) == ([], True)


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")], [1, 2]])
def test_bad_scores_preserve_complete_fallback(scores):
    candidates = [fragment("evidence")]
    assert rerank("query", candidates, lambda pairs: scores) == (candidates, False)


def test_model_failure_is_visible_each_time(monkeypatch):
    import src.retrieval.reranker as module

    calls = []
    monkeypatch.setattr(module.logger, "warning", lambda *a, **kw: calls.append(kw))

    def fail(pairs):
        raise RuntimeError("private content must not appear in logs")

    candidates = [fragment("private evidence")]
    for _ in range(2):
        assert rerank("private query", candidates, fail) == (candidates, False)
    assert len(calls) == 2 and "private" not in str(calls)


def test_disabled_never_calls_model(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_rerank_enabled", False)
    candidates = [fragment("evidence")]
    assert rerank("query", candidates, lambda pairs: pytest.fail("model called")) == (
        candidates,
        False,
    )


def test_oversized_candidate_does_not_hide_smaller_evidence():
    candidates = [fragment("large " * 500), fragment("small answer")]
    ranked, ok = rerank("query", candidates, lambda pairs: [5, 4])
    selected = HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        ranked,
        max_tokens=10,
        relevance_order=ok,
    )
    assert [f.text for f in selected] == ["small answer"]


@pytest.mark.parametrize("wide", [False, True])
def test_production_retrieve_paths_reach_ranking_and_budget(monkeypatch, wide):
    import src.retrieval.reranker as module

    db = SimpleNamespace(execute=lambda *a, **kw: SimpleNamespace(fetchall=lambda: []))
    o = HybridRetrievalOrchestrator(db, None)
    candidates = [fragment("irrelevant", "codex"), fragment("answer", "codex")]
    monkeypatch.setattr(o, "_codex_graph", lambda *a, **kw: candidates)
    for name in (
        "_relevant_cluster_ids",
        "_bm25_episodic",
        "_vector_episodic",
        "_procedural_lookup",
        "_batch_summary_lookup",
        "_cold_lookup",
    ):
        monkeypatch.setattr(o, name, lambda *a, **kw: [])
    monkeypatch.setattr(o, "_apply_bonuses", lambda f, *a: f)
    calls = []

    def scores(pairs):
        calls.extend(pairs)
        return [5 if doc == "answer" else -5 for query, doc in pairs]

    monkeypatch.setattr(module, "score_pairs", scores)
    c = SimpleNamespace(
        prompt="question",
        context_reliance="Long_Term_Memory",
        max_confidence=0.0 if wide else 1.0,
        intent_tags=[],
        topic_tags=[],
    )
    actual = o.retrieve(c, None, [1.0], scope=None)
    assert len(calls) == 2
    assert [f.text for f in actual] == ["answer"]


def test_ordering_only_default_does_not_claim_rejection(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_rerank_min_score", None)
    ranked, ok = rerank(
        "query", [fragment("weak"), fragment("better")], lambda p: [-5, -1]
    )
    assert ok and [f.text for f in ranked] == ["better", "weak"]


def test_complete_template_limit_is_checked_before_inference(monkeypatch):
    import src.retrieval.reranker as module
    from src.memory import embedder

    monkeypatch.setattr(embedder, "resolve_device", lambda preference: "cpu")
    key = (
        settings.retrieval_rerank_model,
        settings.retrieval_rerank_revision,
        settings.retrieval_rerank_device,
    )
    monkeypatch.setattr(module, "_model_key", key)
    fake = SimpleNamespace(
        preprocess=lambda *a, **kw: {"input_ids": SimpleNamespace(shape=(1, 4097))},
        predict=lambda *a, **kw: pytest.fail("overlength evidence was scored"),
    )
    monkeypatch.setattr(module, "_model", fake)
    monkeypatch.setattr(settings, "retrieval_rerank_max_tokens", 4096)
    with pytest.raises(module.RerankerInputTooLong):
        module.score_pairs([("query", "evidence")])
