"""G29: tuning the base window must reach B2, retrieval and summary creation."""

import pytest

from src.api.config import settings
from src.api.memory_decision import decide_memory_retrieval, estimate_recent_window_tokens
from src.classifier.schemas import ClassificationResult
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import conversation_summary


@pytest.mark.parametrize("fraction", [0.12, 0.42])
def test_fraction_reaches_all_base_window_consumers(monkeypatch, fraction):
    monkeypatch.setattr(settings, "context_recent_fraction_ladder", [[10, fraction]])
    orchestrator = object.__new__(HybridRetrievalOrchestrator)
    orchestrator.set_budget_from_turn_count(3, total_budget=12_000)

    expected = fraction * (12_000 - settings.context_overhead_reserve)
    assert estimate_recent_window_tokens(3, 12_000) == pytest.approx(expected)
    assert orchestrator.recent_token_budget == int(expected)
    result = ClassificationResult(
        topic_tags=[], intent_tags=[], context_reliance="Zero_Shot",
        raw_probs=[], max_confidence=0.99, p_ltm=0.4,
    )
    decision = decide_memory_retrieval(
        result, turn_count=3, total_tokens=8_000, settings=settings,
        recent_window_tokens=estimate_recent_window_tokens(3, 12_000),
    )
    assert decision.breakdown["window_tokens"] == int(expected)
    assert (decision.breakdown["p_len"] > 0.7) is (fraction == 0.12)
    # The background summary job uses the fallback model budget by design.
    assert conversation_summary.estimate_recent_window_tokens(3) == pytest.approx(
        fraction * (settings.context_total_budget_fallback
                    - settings.context_overhead_reserve))


def test_default_and_reserve_follow_settings(monkeypatch):
    monkeypatch.setattr(settings, "context_recent_fraction_ladder", [])
    monkeypatch.setattr(settings, "context_recent_fraction_default", 0.27)
    monkeypatch.setattr(settings, "context_total_budget_fallback", 14_000)
    monkeypatch.setattr(settings, "context_overhead_reserve", 2_000)
    orchestrator = object.__new__(HybridRetrievalOrchestrator)
    orchestrator.set_budget_from_turn_count(900)

    assert estimate_recent_window_tokens(900) == pytest.approx(0.27 * 12_000)
    assert orchestrator.recent_token_budget == 0.27 * 12_000
    assert conversation_summary.estimate_recent_window_tokens(900) == 0.27 * 12_000


def test_retrieval_modifiers_remain_after_shared_base(monkeypatch):
    monkeypatch.setattr(settings, "context_recent_fraction_ladder", [[10, 0.30]])
    orchestrator = object.__new__(HybridRetrievalOrchestrator)
    orchestrator.set_budget_from_turn_count(3, total_tokens=9_003,
                                            total_budget=12_000)
    assert orchestrator.recent_token_budget < estimate_recent_window_tokens(3, 12_000)


def test_every_ladder_edge_is_shared(monkeypatch):
    monkeypatch.setattr(settings, "context_recent_fraction_ladder", [
        [10, 0.12], [50, 0.22], [200, 0.32], [500, 0.42],
    ])
    monkeypatch.setattr(settings, "context_recent_fraction_default", 0.52)
    for turns, fraction in [(9, 0.12), (10, 0.22), (49, 0.22),
                            (50, 0.32), (200, 0.42), (500, 0.52)]:
        orchestrator = object.__new__(HybridRetrievalOrchestrator)
        orchestrator.set_budget_from_turn_count(turns, total_budget=12_000)
        expected = fraction * (12_000 - settings.context_overhead_reserve)
        assert estimate_recent_window_tokens(turns, 12_000) == pytest.approx(expected)
        assert orchestrator.recent_token_budget == int(expected)
