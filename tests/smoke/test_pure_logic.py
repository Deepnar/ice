"""Pure-logic smoke checks: B2 decision math, C16 budget math, shared chunker
mechanics. No DB, no GPU, no models."""

from types import SimpleNamespace

from src.api.memory_decision import (
    decide_memory_retrieval,
    derive_total_budget,
    memory_pressure,
)
from src.classifier.schemas import ClassificationResult
from src.memory.chunking import chunk_text, estimate_tokens

SETTINGS = SimpleNamespace(
    ltm_decision_threshold=0.5,
    ltm_prior_bias=0.4,
    ltm_length_weight=0.8,
    ltm_pressure_midpoint_tokens=2000,
    ltm_pressure_scale_tokens=4000,
    ltm_bump_creative=0.7,
    ltm_bump_referential=0.5,
    ltm_bump_low_confidence=0.8,
    ltm_bump_timescope=3.0,
    temporal_label_threshold=0.85,  # B1 D7: label-or-detector, never both twice
    confidence_fallback_threshold=0.55,
    context_budget_fallback=18_000,
    context_input_fraction=0.75,
    context_budget_min=6_000,
    context_budget_max=24_000,
)


def _cr(p_ltm, prompt="hello world", **kw):
    # prompt deliberately avoids REFERENTIAL_WORDS so the referential bump
    # stays out of these assertions.
    return ClassificationResult(
        topic_tags=["Software_&_Tech"],
        intent_tags=["Factual_Retrieval"],
        context_reliance="Zero_Shot",
        raw_probs=[0.0] * 25,
        max_confidence=0.9,
        prompt=prompt,
        p_ltm=p_ltm,
        **kw,
    )


def test_confident_zero_shot_stays_out():
    d = decide_memory_retrieval(_cr(0.05), turn_count=2, total_tokens=500, settings=SETTINGS)
    assert d.retrieve is False


def test_high_p_ltm_retrieves():
    d = decide_memory_retrieval(_cr(0.9), turn_count=2, total_tokens=500, settings=SETTINGS)
    assert d.retrieve is True


def test_out_of_window_history_pushes_toward_memory():
    low = decide_memory_retrieval(_cr(0.35), turn_count=5, total_tokens=500, settings=SETTINGS)
    high = decide_memory_retrieval(
        _cr(0.35), turn_count=300, total_tokens=80_000, settings=SETTINGS
    )
    assert high.p_need_mem > low.p_need_mem


def test_memory_pressure_monotone_and_one_sided():
    base = memory_pressure(0, 4000, 2000, 4000)
    assert base == 0.5  # neutral while history fits the window
    p1 = memory_pressure(10_000, 4000, 2000, 4000)
    p2 = memory_pressure(50_000, 4000, 2000, 4000)
    assert 0.5 < p1 < p2 < 1.0


def test_budget_clamps():
    assert derive_total_budget(None, SETTINGS) == 18_000  # unknown model
    assert derive_total_budget(8_000, SETTINGS) == 6_000  # min clamp
    assert derive_total_budget(200_000, SETTINGS) == 24_000  # max guardrail


def test_chunker_bounds_and_overlap():
    text = ". ".join(f"sentence number {i} with several extra words" for i in range(400))
    chunks = chunk_text(text, max_tokens=550, overlap_words=50)
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c) <= 550 * 1.25  # bounded (greedy pack + slack)
    # short input passes through as a single chunk
    assert chunk_text("just a short line") == ["just a short line"]
