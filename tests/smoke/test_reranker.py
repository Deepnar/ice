"""v3 ranking, representation and budgeting contracts with controlled scores."""

from types import SimpleNamespace

import pytest

from src.api.config import settings
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator
from src.retrieval.reranker import candidate_pool, rerank


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


def test_family_nominee_competes_without_increasing_candidate_cap(monkeypatch):
    monkeypatch.setattr(settings, 'retrieval_rerank_candidates', 4)
    candidates = [fragment(f'noise {i}') for i in range(6)] + [fragment('workflow', 'procedural')]
    pool = candidate_pool(candidates, 4)
    assert [f.text for f in pool] == ['noise 0', 'noise 1', 'noise 2', 'workflow']
    calls = []
    ranked, ok = rerank('workflow question', candidates,
        lambda pairs: calls.extend(pairs) or [5 if text == 'workflow' else -5 for _, text in pairs])
    assert ok and len(calls) == 4 and [f.text for f in ranked] == ['workflow']


def test_small_cap_reserves_only_the_highest_fusion_families():
    candidates = [fragment('episodic'), fragment('graph', 'codex'), fragment('workflow', 'procedural')]
    assert candidate_pool(candidates, 2) == candidates[:2]


def exercise_candidate_pool_path(monkeypatch, *, procedural_relevant=True, actual=False):
    import src.retrieval.reranker as module
    from src.memory.tokens import count

    query = ('When you review my patches, how do I want the review organized?'
             if procedural_relevant else 'Which marker color did we choose for the crate?')
    workflow = 'The user prefers patch reviews to list failing tests first, proposed changes second, and remaining risks last.'
    fact = 'We chose amber as the crate marker color.'
    noise = 'Patch review dashboard displays test timestamps, change colors and risk counts. No review-format preference is stated.'
    episodic = [ContextFragment(noise + f' Sample {i}.', 'episodic', 1., count(noise)) for i in range(65)]
    if not procedural_relevant:
        episodic.insert(0, ContextFragment(fact, 'episodic', 2., count(fact)))
    o = HybridRetrievalOrchestrator(SimpleNamespace(), None)
    o.max_retrieval_tokens = count(workflow if procedural_relevant else fact)
    monkeypatch.setattr(o, '_bm25_episodic', lambda *_a, **_kw: episodic)
    monkeypatch.setattr(o, '_vector_episodic', lambda *_a, **_kw: episodic)
    monkeypatch.setattr(o, '_procedural_lookup', lambda *_a, **_kw:
        [ContextFragment(workflow, 'procedural', .001, count(workflow), origin_batch_ids=('a', 'b'))])
    for name in ('_relevant_cluster_ids', '_codex_graph', '_codex_claims',
                 '_batch_summary_lookup', '_cold_lookup'):
        monkeypatch.setattr(o, name, lambda *_a, **_kw: [])
    fused_positions = []
    def capture_fusion(fragments, *_a):
        fused_positions.append(next(i for i, f in enumerate(fragments, 1) if f.source_type == 'procedural'))
        return fragments
    monkeypatch.setattr(o, '_apply_bonuses', capture_fusion)
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', True)
    monkeypatch.setattr(settings, 'retrieval_rerank_min_score', None)
    monkeypatch.setattr(settings, 'retrieval_rerank_candidates', 64)
    calls = []
    real_scorer = module.score_pairs
    def score(pairs):
        calls.extend(pairs)
        if actual:
            return real_scorer(pairs)
        expected = workflow if procedural_relevant else fact
        return [5 if text == expected else -5 for _, text in pairs]
    monkeypatch.setattr(module, 'score_pairs', score)
    classification = SimpleNamespace(prompt=query, context_reliance='Long_Term_Memory',
        head_confidences={'topic': 1., 'intent': 1.}, max_confidence=1.,
        topic_tags=['Software_&_Tech'], intent_tags=['Factual_Retrieval'])
    final = o.retrieve(classification, conversation_id=None, prompt_embedding=[1.], defer_exposure=True)
    expected = workflow if procedural_relevant else fact
    assert len(calls) == 64 and any(text == workflow for _, text in calls)
    assert fused_positions[0] > 64  # The old prefix pool actually excluded it.
    assert [f.text for f in final] == [expected]
    assert sum(f.token_count for f in final) <= o.max_retrieval_tokens
    return dict(procedural_relevant=procedural_relevant, candidates_scored=len(calls),
                final_type=final[0].source_type, final_tokens=final[0].token_count,
                token_cap=o.max_retrieval_tokens, expected_text_selected=True,
                pre_rerank_procedural_rank=fused_positions[0])


@pytest.mark.parametrize('procedural_relevant', [True, False])
def test_actual_retrieve_allows_competition_without_entitlement(monkeypatch, procedural_relevant):
    exercise_candidate_pool_path(monkeypatch, procedural_relevant=procedural_relevant)


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
    assert module.score_pairs([("query", "evidence")]) == [None]


def test_overlength_pair_does_not_disable_other_model_scores(monkeypatch):
    import numpy as np
    import src.retrieval.reranker as module
    from src.memory import embedder
    monkeypatch.setattr(embedder, 'resolve_device', lambda preference: 'cpu')
    monkeypatch.setattr(module, '_model_key', (settings.retrieval_rerank_model,
        settings.retrieval_rerank_revision, settings.retrieval_rerank_device))
    calls=[]
    def predict(batch, **kwargs):
        calls.extend(batch)
        return np.array([2.] * len(batch))
    fake=SimpleNamespace(preprocess=lambda batch, **kw:
        {'input_ids': SimpleNamespace(shape=(1, 5000 if batch[0][1]=='large' else 50))},
        predict=predict)
    monkeypatch.setattr(module, '_model', fake)
    monkeypatch.setattr(settings, 'retrieval_rerank_max_tokens', 4096)
    assert module.score_pairs([('q','large'), ('q','small')]) == [None, 2.]
    assert calls == [('q','small')]
