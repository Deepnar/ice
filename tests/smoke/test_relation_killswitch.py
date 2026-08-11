"""G34: codex_relation_detection_enabled must make the A4 relation path inert.

Why this exists. The obvious way to neutralise A4 for a run is to set
`codex_relation_overlap_boost = 0.0`, and it does not work. That setting is read
at exactly one site — the per-anchor score bump in `_codex_lookup` — while a
detected relation has three OTHER effects that survive it:

  1. its fact lines are appended to the anchor's fragment text, so they still
     consume context budget and still change what the model reads;
  2. its fact edges join `all_anchor_edges`, which `_reinforce_codex_edges`
     writes to the store (strength +=, `pending` -> `active`) and COMMITS, so a
     read mutates the graph;
  3. it is passed to `_codex_enumeration` as half of that leg's grounded gate.

So the kill-switch is placed at the source instead: `_detect_relations` returns
an empty list, and every downstream site is guarded by `if detected_relations:`.

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
