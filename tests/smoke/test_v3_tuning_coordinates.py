"""Planned coordinates must move real callers without changing defaults."""

import pytest

from experiments.v3_repair.tuning_plan import applied, coordinates
from src.api.config import settings
from src.api.memory_decision import derive_total_budget, estimate_recent_window_tokens
from src.retrieval.orchestrator import HybridRetrievalOrchestrator


def test_bounded_coordinates_change_one_policy_and_leave_unmeasured_outcomes():
    before = settings.model_dump()
    rows = coordinates(settings)
    assert len({r["id"] for r in rows}) == len(rows)
    assert rows[0]["id"] == "baseline" and rows[0]["overrides"] == {}
    for row in rows:
        assert row["live_delta"] is None and row["cost"] is None and row["verdict"] == "unmeasured"
        assert not row["requires_writer_replay"]
        assert len(row["overrides"]) == (3 if row["stage"] == "budget" else
                                         2 if row["stage"] == "recent_allocation" else
                                         0 if row["stage"] == "baseline" else 1)
    assert settings.model_dump() == before


@pytest.mark.parametrize("cap", [4000, 8000, 16000, 32000])
def test_budget_coordinate_reaches_derived_budget_and_respects_real_window(cap):
    row = next(r for r in coordinates(settings) if r["overrides"].get("context_budget_max") == cap)
    before = settings.model_dump()
    with applied(settings, row):
        assert derive_total_budget(128_000, settings) == cap
        usable = (2048 - settings.context_generation_reserve if 2048 > settings.context_generation_reserve
                  else 1024)
        assert derive_total_budget(2048, settings) <= usable
    assert settings.model_dump() == before


def test_recent_coordinate_moves_shared_callers_and_restores_on_error():
    baseline = estimate_recent_window_tokens(3, 12000)
    row = next(r for r in coordinates(settings) if r["stage"] == "recent_allocation")
    before = settings.model_dump()
    with pytest.raises(RuntimeError, match="interruption"):
        with applied(settings, row):
            modified = estimate_recent_window_tokens(3, 12000)
            assert modified != baseline
            reader = object.__new__(HybridRetrievalOrchestrator)
            reader.set_budget_from_turn_count(3, total_budget=12000)
            assert reader.recent_token_budget == int(modified)
            raise RuntimeError("interruption")
    assert settings.model_dump() == before


def test_altered_coordinate_is_rejected_without_mutating_settings():
    row = next(r for r in coordinates(settings) if r["stage"] == "fusion")
    row["overrides"]["retrieval_rrf_k"] = 123456
    before = settings.model_dump()
    with pytest.raises(ValueError, match="differs"):
        with applied(settings, row): pytest.fail("modified coordinate applied")
    assert settings.model_dump() == before


@pytest.mark.parametrize("window,cap", [(0, 4000), (2048, 4000), (8192, 16000), (8192, True)])
def test_infeasible_or_unknown_hard_caps_refuse(window, cap):
    from scripts.z1.production_parity import capped_window
    with pytest.raises(ValueError): capped_window(window, cap)


def test_hard_cap_bounds_real_ledger_and_preserves_uncapped_behavior():
    from scripts.z1.production_parity import capped_window
    from src.api.context_ledger import ContextLedger
    assert capped_window(8192) == 8192
    ledger = ContextLedger(serving_window=capped_window(8192, 4000),
        generation_reserve=settings.context_generation_reserve, safety_margin=1.0)
    ledger.add("system_prompt", 1000); ledger.add("user_message", 1000); ledger.add("evidence", 2000)
    assert ledger.fits() and ledger.available() == 4000
    ledger.add("evidence", 1)
    assert not ledger.fits()
