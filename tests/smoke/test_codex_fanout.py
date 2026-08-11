"""G35: _traverse_graph must expand a bounded, best-first frontier.

Before the cap, expansion followed EVERY valid edge from every node. On a sparse
store that is a handful; on a populated one a hub entity (a recurring person, the
user's main project) fans out to hundreds, so the size and cost of a codex
fragment became a property of the corpus rather than of the query. The budget
bounds total injected tokens, so this never blew the window — it crowded the
other legs out from inside it, which is the harder failure to see.

Two properties are asserted, and the second is the one a naive cap gets wrong:

  1. at most `codex_max_fanout` neighbours are expanded per node;
  2. the ones kept are the highest `_edge_trust`, ranked across BOTH edge
     directions together. The pre-cap loop walked every out-edge and then every
     in-edge, each ordered by raw `strength` in SQL — a different ordering from
     the `_edge_trust` that gates them, under which a well-connected outgoing
     set starves every backlink.

No DB and no model. The session is a scripted stub: the question is whether the
frontier is built, ranked and truncated correctly, which is wiring, not data.

Run:  uv run pytest tests/smoke/test_codex_fanout.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings  # noqa: E402
from src.retrieval.orchestrator import HybridRetrievalOrchestrator  # noqa: E402


class _Entity:
    source = "conversation"
    project_id = None
    context_payload = None
    description = ""
    entity_type = "thing"

    def __init__(self, id_):
        self.id = id_
        self.canonical_name = f"E{id_}"


class _Edge:
    negated = False
    valid_from = None
    extraction_confidence = 1.0

    def __init__(self, id_, source_id, target_id, strength):
        self.id = id_
        self.source_id = source_id
        self.target_id = target_id
        self.strength = strength
        self.relation = "relates_to"
        self.source_batch = None


class _Query:
    def __init__(self, rows, entities):
        self._rows, self._entities = rows, entities

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def all(self):
        return self._rows

    def get(self, id_):
        return self._entities.get(id_)


class _DB:
    """Serves the hub's out-edges to the first edge query and nothing after.

    `filter()` cannot inspect its SQLAlchemy expressions here, so out-vs-in is
    scripted by call order instead: query 1 is the hub's outgoing set, query 2
    its (empty) incoming set, and every later query belongs to a leaf.
    """

    def __init__(self, hub_out, entities):
        self._hub_out, self._entities, self._n = hub_out, entities, 0

    def query(self, model):
        if getattr(model, "__name__", "") == "CodexEntity":
            return _Query([], self._entities)
        self._n += 1
        return _Query(self._hub_out if self._n == 1 else [], self._entities)


def _expand(fanout, monkeypatch, n_neighbours=20):
    """Traverse a hub with `n_neighbours` outgoing edges of distinct strength.

    Returns the ids actually expanded into, in the order they were expanded.
    """
    monkeypatch.setattr(settings, "codex_max_fanout", fanout)
    monkeypatch.setattr(HybridRetrievalOrchestrator, "_render_codex_entity",
                        lambda *a, **k: None)

    hub = _Entity(0)
    entities = {i: _Entity(i) for i in range(1, n_neighbours + 1)}
    # strength 0.51..0.99, ascending with id, so the BEST are the highest ids —
    # the reverse of insertion order, or a cap that just takes the first N
    # would pass without ranking anything.
    edges = [_Edge(i, 0, i, 0.50 + i / 100.0) for i in range(1, n_neighbours + 1)]

    o = object.__new__(HybridRetrievalOrchestrator)
    o.db = _DB(edges, entities)
    o._denied_batch_ids = set()
    o._denied_entity_ids = set()
    o._scope_project_id = None

    expanded = []
    real = HybridRetrievalOrchestrator._traverse_graph

    def spy(self, entity, depth, *a, **k):
        if depth > 0:
            expanded.append(entity.id)
        return real(self, entity, depth, *a, **k)

    monkeypatch.setattr(HybridRetrievalOrchestrator, "_traverse_graph", spy)
    spy(o, hub, 0, settings.codex_max_depth, set(), [], [])
    return expanded


def test_fanout_is_capped(monkeypatch):
    assert len(_expand(5, monkeypatch)) == 5


def test_raising_the_cap_expands_more(monkeypatch):
    """The other side — a cap asserted at one value proves nothing about
    whether it was read or the frontier was simply that small."""
    assert len(_expand(12, monkeypatch)) == 12


def test_uncapped_would_have_expanded_everything(monkeypatch):
    """Establishes that 20 candidates really were available, so the two counts
    above are truncations rather than the whole frontier."""
    assert len(_expand(999, monkeypatch)) == 20


def test_the_kept_neighbours_are_the_highest_trust(monkeypatch):
    """Best-first, not first-N. Strength ascends with id, so the top 5 by trust
    are ids 20..16 in that order."""
    assert _expand(5, monkeypatch) == [20, 19, 18, 17, 16]


@pytest.mark.parametrize("fanout", [1, 3, 7])
def test_both_directions_compete_in_one_ranking(fanout, monkeypatch):
    """A backlink stronger than every outgoing edge must survive any cap >= 1.

    The pre-cap loop concatenated out-edges then in-edges, so a truncation
    applied to that order would drop every backlink before touching a weak
    outgoing edge. Ranking across both directions is what prevents it.
    """
    monkeypatch.setattr(settings, "codex_max_fanout", fanout)
    monkeypatch.setattr(HybridRetrievalOrchestrator, "_render_codex_entity",
                        lambda *a, **k: None)

    hub = _Entity(0)
    entities = {i: _Entity(i) for i in range(1, 12)}
    out = [_Edge(i, 0, i, 0.60) for i in range(1, 11)]
    backlink = _Edge(11, 11, 0, 0.99)          # strongest edge, incoming
    entities[11] = _Entity(11)

    class _BothDB(_DB):
        def query(self, model):
            if getattr(model, "__name__", "") == "CodexEntity":
                return _Query([], self._entities)
            self._n += 1
            if self._n == 1:
                return _Query(out, self._entities)
            if self._n == 2:
                return _Query([backlink], self._entities)
            return _Query([], self._entities)

    o = object.__new__(HybridRetrievalOrchestrator)
    o.db = _BothDB(out, entities)
    o._denied_batch_ids = set()
    o._denied_entity_ids = set()
    o._scope_project_id = None

    expanded = []
    real = HybridRetrievalOrchestrator._traverse_graph

    def spy(self, entity, depth, *a, **k):
        if depth > 0:
            expanded.append(entity.id)
        return real(self, entity, depth, *a, **k)

    monkeypatch.setattr(HybridRetrievalOrchestrator, "_traverse_graph", spy)
    spy(o, hub, 0, settings.codex_max_depth, set(), [], [])

    assert 11 in expanded, (
        f"the strongest edge is a backlink and the cap ({fanout}) dropped it — "
        "the two directions are not competing in one ranking")


def test_anchor_edges_are_not_capped(monkeypatch):
    """A3 reinforcement must still see every edge above the direct trust floor.

    Capping traversal is about bounding cost; narrowing what gets reinforced is
    a different behaviour change, and bundling the two would silently alter the
    codex strength dynamics.
    """
    monkeypatch.setattr(settings, "codex_max_fanout", 5)
    monkeypatch.setattr(HybridRetrievalOrchestrator, "_render_codex_entity",
                        lambda *a, **k: None)

    hub = _Entity(0)
    entities = {i: _Entity(i) for i in range(1, 21)}
    edges = [_Edge(i, 0, i, 0.50 + i / 100.0) for i in range(1, 21)]

    o = object.__new__(HybridRetrievalOrchestrator)
    o.db = _DB(edges, entities)
    o._denied_batch_ids = set()
    o._denied_entity_ids = set()
    o._scope_project_id = None

    anchor_edges = []
    o._traverse_graph(hub, 0, settings.codex_max_depth, set(), [], anchor_edges)
    assert len(anchor_edges) == 20, (
        f"the cap leaked into A3 reinforcement: {len(anchor_edges)} of 20 edges")
