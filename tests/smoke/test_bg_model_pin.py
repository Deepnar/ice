"""G27: an unpinned background model must announce itself.

In shared mode with no `BACKGROUND_MODEL_NAME`, `get_bg_model_name()` returns the
first `confirmed: true` entry in the model registry. That is **not** what shared
mode is documented to mean (C7 D7 — reuse the model chat is already running), and
two things follow: background work can hold a second ~17 GB model against a 24 GB
card, and no run can state what produced its output, because the choice follows
the registry's file order.

Resolving the session's routed model is deferred to Z2, which will measure which
background model is worth defaulting to. Until then the pin is the supported
answer — and the fallback must not be silent, because a fallback nobody hears is
how this survived unnoticed in the first place (CLAUDE.md, TRAPS #11).

Run:  uv run pytest tests/smoke/test_bg_model_pin.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.workers.bg_client_factory as bg  # noqa: E402
from src.api.config import settings  # noqa: E402


def _run(monkeypatch, pin):
    warnings = []
    monkeypatch.setattr(settings, "background_model_name", pin)
    monkeypatch.setattr(settings, "background_model_mode", "shared")
    monkeypatch.setattr(bg.logger, "warning",
                        lambda ev, **kw: warnings.append((ev, kw)))
    return bg.get_bg_model_name(), warnings


def test_a_pinned_model_is_returned_and_stays_quiet(monkeypatch):
    name, warnings = _run(monkeypatch, "qwen2.5:7b")
    assert name == "qwen2.5:7b"
    assert warnings == [], "a deliberate pin is not a fallback and must not warn"


def test_an_unpinned_model_warns_with_the_name_it_chose(monkeypatch):
    """The other side. The warning has to carry WHICH model, or it tells the
    reader a problem exists without telling them what they got."""
    name, warnings = _run(monkeypatch, None)
    assert len(warnings) == 1
    event, fields = warnings[0]
    assert event == "bg_model_unpinned"
    assert fields.get("chosen") == name, (
        "the warning must name the model actually returned")


def test_it_warns_every_time_not_just_once(monkeypatch):
    """A once-only warning is worse than none: the first run reports it and
    every run afterwards looks clean while doing the same thing."""
    warnings = []
    monkeypatch.setattr(settings, "background_model_name", None)
    monkeypatch.setattr(settings, "background_model_mode", "shared")
    monkeypatch.setattr(bg.logger, "warning",
                        lambda ev, **kw: warnings.append(ev))
    for _ in range(3):
        bg.get_bg_model_name()
    assert len(warnings) == 3
