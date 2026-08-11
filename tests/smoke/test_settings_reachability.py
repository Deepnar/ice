"""G29: a setting that no live call path can reach is a knob wired to nothing.

`retrieval_max_per_conversation` was the case that prompted this. G9 moved it
into settings and `_session_diversify` reads it — but only when the caller omits
the argument, and all three call paths passed a literal 3:

    orchestrator.retrieve()                 max_per_conversation=3
    orchestrator._wide_net_fallback()       max_per_conversation=3
    ConfigurableOrchestrator                default =3, forwarded

So the setting was unreachable. `tests/test_settings_freeze.py` has a row for it
and PASSED throughout, because the declaration was still 3 — a suite vouching
for a knob it cannot reach. The signature-drift check in test_ablation_flags.py
compared parameter NAMES only, so the subclass's third copy went unseen too.

That is the fourth instance of this shape after `recency_boost`, `include_cross`
and G36's HyDE flag (all in PROVENANCE.md), and it is the specific failure that
makes a tuning sweep meaningless: the sweep moves the setting, the hot path
reads its own copy, and whatever the noise favoured gets kept.

These checks are structural — they read call sites and signatures, not results.
No DB and no model.

Run:  uv run pytest tests/smoke/test_settings_reachability.py -q
"""

import inspect
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings  # noqa: E402
from src.retrieval.configurable_orchestrator import ConfigurableOrchestrator  # noqa: E402
from src.retrieval.orchestrator import HybridRetrievalOrchestrator  # noqa: E402


class _Frag:
    def __init__(self, cid):
        self.conversation_id = cid


def test_session_diversify_honours_the_setting(monkeypatch):
    """The behavioural half: move the setting, the cap must move with it.

    Both directions, because a cap asserted at one value proves nothing about
    whether it was read or hardcoded to that same value.
    """
    o = object.__new__(HybridRetrievalOrchestrator)
    frags = [_Frag("other")] * 10

    monkeypatch.setattr(settings, "retrieval_max_per_conversation", 3)
    assert len(o._session_diversify(frags, "current")) == 3

    monkeypatch.setattr(settings, "retrieval_max_per_conversation", 7)
    assert len(o._session_diversify(frags, "current")) == 7, (
        "the cap did not follow the setting — a call site is passing a literal")


# (method, parameter, the setting it falls through to). Each of these reads its
# setting only when the argument is omitted, so any caller passing a literal
# takes the setting out of the loop. All three were pinned; two by a literal
# that happened to equal the setting, which is why nothing looked wrong.
FALLTHROUGH = [
    ("_session_diversify", "max_per_conversation", "retrieval_max_per_conversation"),
    ("_relevant_cluster_ids", "top_k", "retrieval_cluster_top_k"),
    ("_apply_rrf", "k", "retrieval_rrf_k"),
]


@pytest.mark.parametrize("method,param,setting", FALLTHROUGH,
                         ids=[r[0] for r in FALLTHROUGH])
def test_no_call_site_pins_the_value(method, param, setting):
    """The structural half: the behavioural test above only covers the default
    path, so it would still pass if a call site elsewhere pinned the value.

    A literal equal to the setting is still a failure. It produces the right
    number today and silently ignores the setting tomorrow, which is the exact
    state all three of these were found in.
    """
    pinned = []
    for mod in (HybridRetrievalOrchestrator, ConfigurableOrchestrator):
        src = inspect.getsource(inspect.getmodule(mod))
        pinned += re.findall(rf"{method}\([^)]*\b{param}\s*=\s*(\d+)", src)
    assert not pinned, (
        f"a call site pins {param}={pinned} instead of letting "
        f"settings.{setting} decide")


def test_subclass_defaults_match_the_parent():
    """test_ablation_flags.py compares override parameter NAMES; a default that
    diverges is the same drift with a quieter symptom, and is how the third
    copy of `3` survived. Names are checked there, defaults here.
    """
    drifted = []
    for name, child in vars(ConfigurableOrchestrator).items():
        if not inspect.isfunction(child) or name == "__init__":
            continue
        parent = getattr(HybridRetrievalOrchestrator, name, None)
        if parent is None or parent is child:
            continue
        p = {k: v.default for k, v in inspect.signature(parent).parameters.items()}
        c = {k: v.default for k, v in inspect.signature(child).parameters.items()}
        differing = [k for k in p.keys() & c.keys() if p[k] != c[k]]
        if differing:
            drifted.append(
                f"{name}: " + ", ".join(f"{k} parent={p[k]!r} subclass={c[k]!r}"
                                        for k in differing))
    assert not drifted, ("ablation override defaults drifted from the parent:\n  "
                         + "\n  ".join(drifted))
