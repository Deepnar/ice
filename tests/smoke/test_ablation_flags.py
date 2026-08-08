"""G19(a): the ablation harness's flags must actually reach the scoring code.

Why this exists. `ConfigurableOrchestrator` is a subclass used only by the
flaw-ablation runners, and two of its overrides were broken in ways no test
could see:

  1. `_apply_bonuses` imported the recency constants from the parent module and
     rebound them with `global`. That wrote THIS module's copies while the
     parent scored from its own, so `recency_boost: False` disabled nothing —
     for as long as the flag has existed. Experiment 3's `add_keyword_boost`
     and `full_ice` arms were therefore the same configuration, which their
     near-identical token counts (27,769 vs 27,768) and fragment counts
     (15.1 vs 15.0) show.
  2. `_batch_summary_lookup` never gained the `include_cross` parameter the
     parent took on with C6's incognito rule, so every ablation run raised
     TypeError the moment it reached that leg.

Both are the desync ROADMAP G19 warns about: a shadow subclass that mirrors
parent internals drifts silently. These checks are the guard until G19's fold
removes the mirroring entirely.

No DB and no model — the point is whether a flag CHANGES WHAT THE PARENT SEES,
which is a question about wiring, not about data.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings  # noqa: E402
from src.retrieval.configurable_orchestrator import ConfigurableOrchestrator  # noqa: E402
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator  # noqa: E402


class _Cls:
    topic_tags: list = []
    intent_tags: list = []


class _TS:
    mode = "current"


def _orchestrator(overrides):
    o = object.__new__(ConfigurableOrchestrator)
    o.overrides = overrides
    o._active_timescope = _TS()
    return o


def _observed_recency_bonus(overrides, monkeypatch):
    """What the PARENT reads for the recency bonus while the flag is applied."""
    seen = {}

    def spy(self, fragment, conv_id):
        seen["top10"] = settings.retrieval_bonus_recent_top_10pct
        seen["top30"] = settings.retrieval_bonus_recent_top_30pct
        return 0.0

    monkeypatch.setattr(HybridRetrievalOrchestrator, "_recency_bonus", spy)
    frag = ContextFragment(text="a " * 200, source_type="episodic",
                           score=1.0, token_count=10)
    ConfigurableOrchestrator._apply_bonuses(
        _orchestrator(overrides), [frag], _Cls(), "conv-1", {"a"})
    return seen


def test_recency_flag_off_is_seen_by_the_parent(monkeypatch):
    seen = _observed_recency_bonus(
        {"keyword_boost": True, "recency_boost": False}, monkeypatch)
    assert seen == {"top10": 0.0, "top30": 0.0}, (
        "recency_boost=False did not reach the parent's scoring path — this is "
        "the exact failure that made two Experiment 3 arms identical")


def test_recency_flag_on_leaves_the_bonus_intact(monkeypatch):
    """The other side: off must differ from on, or the check above is vacuous."""
    seen = _observed_recency_bonus(
        {"keyword_boost": True, "recency_boost": True}, monkeypatch)
    assert seen["top10"] == settings.retrieval_bonus_recent_top_10pct
    assert seen["top10"] != 0.0, "the default bonus is 0.0; this test proves nothing"


def test_flag_restores_the_setting_afterwards(monkeypatch):
    before = settings.retrieval_bonus_recent_top_10pct
    _observed_recency_bonus({"keyword_boost": True, "recency_boost": False}, monkeypatch)
    assert settings.retrieval_bonus_recent_top_10pct == before, (
        "the ablation leaked its override into the live settings")


def test_overridden_signatures_still_match_the_parent():
    """The mirroring hazard itself, checked structurally rather than by luck.

    `_batch_summary_lookup` drifted because the parent gained a parameter and
    the override did not. Any override whose parameter names no longer match
    the method it shadows will raise on the first call that uses the new one.
    """
    drifted = []
    for name, child in vars(ConfigurableOrchestrator).items():
        # __init__ is excluded on purpose: taking `overrides` is the whole
        # reason the subclass exists. Every OTHER override is a mirror, and a
        # mirror that stops matching is the bug.
        if not inspect.isfunction(child) or name == "__init__":
            continue
        parent = getattr(HybridRetrievalOrchestrator, name, None)
        if parent is None or parent is child:
            continue
        p_params = list(inspect.signature(parent).parameters)
        c_params = list(inspect.signature(child).parameters)
        if p_params != c_params:
            drifted.append(f"{name}: parent{p_params} vs subclass{c_params}")
    assert not drifted, "ablation override signatures drifted from the parent:\n  " + \
                        "\n  ".join(drifted)
