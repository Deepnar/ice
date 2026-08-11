"""G35: report what share of the retrieval window each leg actually took.

The entry's look-ahead asks for this by name. The reasoning is that the TOTAL is
the wrong number to watch: the budget already bounds total injected tokens, so an
over-eager leg never blows the window — it crowds the other legs out from inside
it. That failure is invisible to a total-tokens check and visible here.

Run:  uv run pytest tests/smoke/test_leg_budget_share.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.orchestrator import HybridRetrievalOrchestrator  # noqa: E402


class _F:
    def __init__(self, leg, tokens):
        self.source_type, self.token_count = leg, tokens


def _emit(fragments, total, max_tokens=1000):
    seen = {}
    o = object.__new__(HybridRetrievalOrchestrator)
    import src.retrieval.orchestrator as mod
    real = mod.logger.info
    mod.logger.info = lambda ev, **kw: seen.update({"event": ev, **kw})
    try:
        o._log_leg_budget_share(fragments, total, max_tokens)
    finally:
        mod.logger.info = real
    return seen


def test_shares_are_of_what_was_spent():
    """A leg taking 80% of a barely-used window is a different problem from
    taking 80% of a full one, so the share is of the spend and the utilisation
    is reported alongside it."""
    got = _emit([_F("episodic", 60), _F("codex", 40)], total=100)
    assert got["share"] == {"episodic": 0.6, "codex": 0.4}
    assert got["utilisation"] == 0.1
    assert got["tokens"] == {"episodic": 60, "codex": 40}


def test_fragment_counts_ride_along():
    """Tokens alone cannot distinguish one huge fragment from ten small ones,
    and those are different crowding stories."""
    got = _emit([_F("episodic", 10), _F("episodic", 10), _F("codex", 80)], total=100)
    assert got["fragments"] == {"episodic": 2, "codex": 1}


def test_nothing_retrieved_emits_nothing():
    """An empty retrieval has no share to report, and a row of zeroes in the log
    would read as a leg that took nothing rather than a request that asked for
    nothing."""
    assert _emit([], total=0) == {}


def test_a_single_leg_taking_everything_is_visible():
    """The failure this exists to surface, stated as an assertion."""
    got = _emit([_F("episodic", 100)], total=100)
    assert got["share"] == {"episodic": 1.0}
