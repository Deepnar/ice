"""G29: the C5 cluster-scope SQL fragment, built in one place.

Four legs each wrote this predicate out longhand — the BM25 leg, the vector leg,
the chunk leg and the wide net. They were verbatim apart from the row alias
(`e.id` on the chunk leg, `episodic_memory.id` elsewhere) and one differed only
by two spaces of indentation inside the string, which is why four copies read as
"the same" and could not be diffed by eye.

The failure this guards against is on record: the wide net had **no** copy at all
until C6 added one, so cluster scoping held on three legs and was dropped on the
fourth — widening *visibility* rather than ranking, which the leg's own comment
forbids. One builder makes a missing copy impossible rather than merely unlikely.

No DB: the fragment is a string, and what matters is which rows it would admit.

Run:  uv run pytest tests/smoke/test_cluster_filter.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.orchestrator import HybridRetrievalOrchestrator  # noqa: E402


def _sql(scope, *a):
    return HybridRetrievalOrchestrator._cluster_filter(
        object.__new__(HybridRetrievalOrchestrator), scope, *a)


def test_no_scope_means_no_filter():
    assert _sql(None) == ""
    assert _sql({}) == ""
    assert _sql({"cluster_ids": []}) == "", (
        "an empty cluster list must not emit a filter — it is 'no cluster "
        "scoping', not 'match nothing'")


def test_a_scope_emits_the_predicate():
    sql = _sql({"cluster_ids": ["c1"]})
    assert "episodic_cluster_links" in sql
    assert "ANY(:cluster_ids)" in sql, "cluster ids must stay bound, not inlined"


def test_unlinked_turns_stay_visible():
    """The OR NOT EXISTS half, which reads as redundant and is not.

    A turn no clustering pass has reached yet belongs to no cluster. Without
    this branch a cluster-scoped query would hide every turn that was merely
    too new to have been clustered.
    """
    assert "OR NOT EXISTS" in _sql({"cluster_ids": ["c1"]})


def test_the_alias_is_honoured():
    """The chunk leg joins under a different alias; a builder that ignored the
    argument would emit SQL referencing a table the query does not have."""
    default = _sql({"cluster_ids": ["c1"]})
    aliased = _sql({"cluster_ids": ["c1"]}, "e.id")
    assert "episodic_memory.id" in default and "e.id" not in default
    assert "e.id" in aliased and "episodic_memory.id" not in aliased


def test_every_leg_uses_the_builder():
    """Structural: the point is that no fifth copy is ever hand-written.

    Counted rather than asserted-present, because a leg that quietly stops
    calling it is the exact regression — and the count is the only thing that
    notices a call site being replaced by a literal again.
    """
    import inspect
    src = inspect.getsource(HybridRetrievalOrchestrator)
    calls = src.count("self._cluster_filter(")
    assert calls == 6, f"expected 6 consumers (including claims and summaries), found {calls}"

    # The inclusion predicate binds :cluster_ids. C6's EXCLUSION predicate in
    # _exclusion_filters also reads episodic_cluster_links but binds
    # :excl_cluster_ids — a different question ("which clusters are muted"),
    # legitimately its own SQL, and not a fifth copy of this one.
    outside = src.replace(
        inspect.getsource(HybridRetrievalOrchestrator._cluster_filter), "")
    assert "ANY(:cluster_ids)" not in outside, (
        "a leg is hand-rolling the cluster inclusion predicate again")
