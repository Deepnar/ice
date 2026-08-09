"""Dynamics invariants — properties that must hold at ANY tuning.

Z1's D5 says background dynamics are *solved, not swept*: decay rates and
promotion thresholds are temporal behaviours a single-turn probe loop cannot
observe, so they are set by writing down target half-lives, solving the rates
closed-form, and asserting the targets in a pure-logic test that fails if
anyone retunes a rate into violating one. This is that file.

**It is a SEED, not the finished set.** Z1 owns filling it in (a
weekly-revisited memory never archives; an untouched casual turn reaches cold
in ~6-8 weeks; a reinforced codex edge promotes after ~3 corroborations). What
is here is the invariant the cycles-per-day derivation guarantees, written
first because that derivation is what makes the rest expressible.

⚠ Nothing here pins a tuned VALUE. Z1 changes the daily targets; these checks
must survive that. If a check here fails after a retune, the retune broke a
stated property — fix the tuning, not the assertion.

Run: uv run pytest tests/test_dynamics_invariants.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.config import settings  # noqa: E402
from src.workers import codex_decay, decay  # noqa: E402

SECONDS_PER_DAY = 86_400.0


@pytest.fixture
def cadence(monkeypatch):
    """Set the decay cadences without touching the rest of the interval table."""
    def _set(seconds, job="decay_episodic"):
        intervals = dict(settings.maintenance_intervals)
        intervals[job] = seconds
        monkeypatch.setattr(settings, "maintenance_intervals", intervals)
    return _set


# ── the invariant the derivation exists for ────────────────────────────────

@pytest.mark.parametrize("seconds", [5400, 3600, 900, 7200, 43200, 86400])
def test_daily_decay_is_preserved_at_any_cadence(cadence, seconds):
    """Running decay more or less often must not change how fast memory fades.

    This is the whole point of storing a DAILY target. The job applies its
    multiplier once per run, so the per-run rate has to be the daily target's
    (86400/cadence)-th root. Before G9's follow-up, cycles-per-day was its own
    setting that could silently disagree with the cadence — and then `0.95/day`
    described nothing that actually happened.
    """
    cadence(seconds)
    runs_per_day = SECONDS_PER_DAY / seconds
    compounded = decay.rate_unaccessed() ** runs_per_day
    assert compounded == pytest.approx(settings.decay_daily_unaccessed, rel=1e-12), (
        f"at a {seconds}s cadence the per-run rate compounds to {compounded}, "
        f"not the stated {settings.decay_daily_unaccessed}/day")


@pytest.mark.parametrize("rate_fn,target", [
    (decay.rate_unaccessed, "decay_daily_unaccessed"),
    (decay.rate_accessed, "decay_daily_accessed"),
    (decay.rate_creative, "decay_daily_creative"),
])
def test_every_episodic_rate_compounds_to_its_stated_target(cadence, rate_fn, target):
    cadence(3600)
    compounded = rate_fn() ** 24
    assert compounded == pytest.approx(getattr(settings, target), rel=1e-12)


def test_codex_decay_follows_its_OWN_cadence(cadence):
    """The two jobs share 5400s today; they must not share a constant.

    decay_codex and decay_episodic are separate entries in the interval table.
    A codex rate derived from the episodic cadence would look right for exactly
    as long as nobody changed one of them.
    """
    cadence(3600, job="decay_codex")          # codex hourly
    cadence(86400, job="decay_episodic")      # episodic daily
    compounded = codex_decay.decay_rate() ** 24
    assert compounded == pytest.approx(settings.codex_decay_daily, rel=1e-12), (
        "codex decay is being derived from the wrong job's cadence")


# ── ordering properties, true at any tuning ────────────────────────────────

def test_accessed_memory_decays_slower_than_unaccessed():
    assert decay.rate_accessed() > decay.rate_unaccessed(), (
        "using a memory must make it survive longer, not shorter")


def test_creative_decays_slowest():
    """Long-form narrative is re-read in bursts months apart, so the usual
    'unused means unwanted' inference is wrong for it."""
    assert decay.rate_creative() > decay.rate_accessed()


def test_thresholds_are_ordered_archive_above_cold():
    """A turn must pass through archived before it reaches cold storage."""
    assert settings.decay_archive_threshold > settings.decay_cold_threshold


def test_creative_floor_sits_above_the_archive_line():
    """Otherwise the floor would hold turns at a score that is already archived,
    which is the opposite of what a floor protecting narrative is for."""
    assert settings.decay_creative_floor > settings.decay_archive_threshold


def test_codex_demotion_sits_above_expiry():
    """An edge demotes to pending before it is garbage-collected (A3)."""
    assert settings.codex_demotion_threshold > settings.codex_expiry_threshold


# ── the fallback is observable, per the standing rule ──────────────────────

def test_missing_cadence_warns_rather_than_guessing_silently(monkeypatch, capsys):
    """structlog renders to stdout, so capsys is the capture, not caplog."""
    intervals = dict(settings.maintenance_intervals)
    intervals.pop("decay_episodic", None)
    monkeypatch.setattr(settings, "maintenance_intervals", intervals)
    got = decay.cycles_per_day()
    out = capsys.readouterr().out
    assert got == 16.0
    assert "decay_cadence_unresolved" in out, \
        "the fallback substituted a default without saying so"
    assert "warning" in out.lower(), "the fallback was announced below WARNING"


def test_a_present_cadence_does_NOT_warn(capsys):
    """The other side: a warning that fires on healthy runs is noise, and noise
    is what trains everyone to ignore the real one (TRAPS #13b)."""
    decay.cycles_per_day()
    assert "decay_cadence_unresolved" not in capsys.readouterr().out
