"""G34: the fit scalar that replaced the unconditional relation boost.

The old per-anchor bonus was a flat `+0.25` applied whenever any relation was
detected — and the detector fired on every prompt, so the condition was never
false and the bonus was applied to noise for the whole of v2.

`_relation_fit` replaces it with `best - mean` across the anchor's OWN relations.
That is the one place a "no" is expressed, and it is expressed as a continuous
zero rather than a gate: a contentless prompt sits near everything, so the spread
collapses and the bonus vanishes without any threshold deciding it should.

Glosses are synthetic here. The semantic question — does the right relation win —
is answered by `scripts/oneoff/g34_ablation.py` against the real vocabulary and
recorded in PROVENANCE (93.2% top-1 vs 8.0% for strength ordering). What this
suite covers is the arithmetic and its edges, which is what a regression would
break silently.

Run:  uv run pytest tests/smoke/test_relation_fit.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.orchestrator import HybridRetrievalOrchestrator  # noqa: E402


def _orch(vocab, vectors):
    o = object.__new__(HybridRetrievalOrchestrator)
    o._relation_gloss_cache = lambda: (vocab, vectors)
    o._leg_degraded = lambda leg, err: None
    return o


VOCAB = ["married_to", "owns", "inspired_by"]
#            married  owns  inspired
POINTED = [1.0, 0.0, 0.0]      # sits squarely on `married_to`
FLAT = [0.5, 0.5, 0.5]         # equidistant — the "ok" case
VECTORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_a_pointed_prompt_earns_a_positive_fit():
    scores, fit = _orch(VOCAB, VECTORS)._relation_fit(VOCAB, POINTED)
    assert max(scores, key=scores.get) == "married_to"
    assert fit > 0.0


def test_a_flat_prompt_earns_zero_fit():
    """The failure this item exists to fix — handled by construction, not by a
    threshold. `"ok"` is near every relation, so nothing stands out."""
    _, fit = _orch(VOCAB, VECTORS)._relation_fit(VOCAB, FLAT)
    assert fit == 0.0, f"a contentless prompt earned fit={fit}"


def test_fewer_than_two_relations_scores_nothing():
    """Ranking one item is theatre, and the spread of a single value is zero —
    so an anchor with one relation must fall back to strength order, not to a
    fit of 1.0."""
    for rels in ([], ["owns"]):
        scores, fit = _orch(VOCAB, VECTORS)._relation_fit(rels, POINTED)
        assert (scores, fit) == ({}, 0.0)


def test_a_missing_embedding_scores_nothing():
    """A degraded encoder must not silently become 'no relation matched'."""
    assert _orch(VOCAB, VECTORS)._relation_fit(VOCAB, None) == ({}, 0.0)


def test_unknown_relations_are_skipped_not_fatal():
    """Legacy and hand-written edges carry relations outside the controlled
    vocabulary. They must not crash the leg, and must not be scored."""
    scores, fit = _orch(VOCAB, VECTORS)._relation_fit(
        ["married_to", "owns", "a_legacy_relation"], POINTED)
    assert "a_legacy_relation" not in scores
    assert fit > 0.0


def test_fit_is_clamped_into_zero_one():
    """It feeds a score bonus; an out-of-range value would silently re-weight
    the whole codex leg rather than fail."""
    big = [10.0, 0.0, 0.0]
    _, fit = _orch(VOCAB, VECTORS)._relation_fit(VOCAB, big)
    assert 0.0 <= fit <= 1.0
