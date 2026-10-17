"""G34: codex_relation_detection_enabled must make the A4 relation path inert.

The relation detector's switch must avoid its candidate work. Graph retrieval
no longer writes evidence strength or promotes pending edges (v3 repair).
This tests the detector boundary, not every downstream relation-fact path.

No DB and no model. The gloss vocabulary is stubbed and the embedding channel is
skipped (`prompt_embedding=None`), because the question here is whether the flag
reaches the code — wiring, not data. Both sides are asserted: an off-switch that
is checked only in the off position proves nothing (TRAPS #5).

Run:  uv run pytest tests/smoke/test_relation_killswitch.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings  # noqa: E402
from src.retrieval.orchestrator import HybridRetrievalOrchestrator  # noqa: E402

# Two relations from the controlled vocabulary. Only the lexical channel runs,
# so the gloss vectors are never read and may be None.
_RELS = ["inspired_by", "married_to"]


def _detect(monkeypatch, enabled, prompt="who inspired kael", spy=None):
    o = object.__new__(HybridRetrievalOrchestrator)

    def gloss(self):
        if spy is not None:
            spy.append(1)
        return _RELS, [None, None]

    monkeypatch.setattr(HybridRetrievalOrchestrator, "_relation_gloss_cache", gloss)
    monkeypatch.setattr(settings, "codex_relation_detection_enabled", enabled)
    return o._detect_relations(prompt, None)


def test_on_detects_the_relation(monkeypatch):
    """The other side first: if this is empty, the off-test below is vacuous."""
    assert _detect(monkeypatch, True) == ["inspired_by"]


def test_off_detects_nothing(monkeypatch):
    assert _detect(monkeypatch, False) == []


def test_off_does_no_work_at_all(monkeypatch):
    """G34 also owns a per-prompt gloss cosine loop on the synchronous path.
    The switch returns before the vocabulary is even loaded, so 'off' costs
    nothing rather than costing the same and discarding the answer."""
    spy = []
    _detect(monkeypatch, False, spy=spy)
    assert spy == [], "the relation vocabulary was built despite the kill-switch"
    _detect(monkeypatch, True, spy=spy)
    assert spy == [1], "the on-path must build it, or the check above is vacuous"


def test_default_is_on(monkeypatch):
    """Production behaviour is unchanged by this seam existing."""
    assert type(settings).model_fields["codex_relation_detection_enabled"].default is True
