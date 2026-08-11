"""G29: the grounded-summary coverage retry, once, for both callers.

`post_flight.generate_summary` and `conversation_summary._summarize_chunk` each
wrote the same block out longhand — measure coverage, name the dropped terms,
ask again, keep the retry if it scores better. They had already drifted the way
this consolidation exists to stop:

  * `conversation_summary` adopted a better retry **without updating coverage**.
    It escaped consequence only because it discards the value, while
    `post_flight` returns it — and `summary_coverage` is the trust gate that
    decides whether a summary may be injected at all.
  * the empty-retry guard existed on one side only.

No DB and no LLM: `retry_fn` is a stub, so what is under test is the decision
logic rather than a model's output.

Run:  uv run pytest tests/smoke/test_coverage_retry.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings  # noqa: E402
from src.workers.turn_density import retry_on_coverage_miss  # noqa: E402

# summary_coverage reads entities/figures/identifiers via must_terms.
KEY_TERMS = {"entities": ["Kael", "Aurelio"], "figures": ["15 lakh"],
             "identifiers": ["EC-9c3e"]}
ALL_TERMS = "Kael Aurelio 15 lakh EC-9c3e"


def test_no_retry_when_coverage_is_already_good():
    calls = []

    def retry_fn(missing):
        calls.append(missing)
        return "unused"

    summary, coverage = retry_on_coverage_miss(ALL_TERMS, KEY_TERMS, retry_fn)
    assert calls == [], "retried a summary that already covered every term"
    assert summary == ALL_TERMS
    assert coverage == 1.0


def test_retry_is_told_exactly_what_was_dropped():
    seen = {}

    def retry_fn(missing):
        seen["missing"] = missing
        return ALL_TERMS

    retry_on_coverage_miss("Kael only", KEY_TERMS, retry_fn)
    assert seen["missing"] == ["Aurelio", "15 lakh", "EC-9c3e"]


def test_a_better_retry_is_adopted_with_its_coverage():
    """The drift itself: the summary and the coverage must move together.

    `conversation_summary` used to return the improved text alongside the OLD
    score, which is the trust gate reading a number for a different string.
    """
    summary, coverage = retry_on_coverage_miss(
        "Kael only", KEY_TERMS, lambda missing: ALL_TERMS)
    assert summary == ALL_TERMS
    assert coverage == 1.0, (
        f"summary was replaced but coverage stayed {coverage} — the gate would "
        "score the text it no longer holds")


def test_a_worse_retry_is_discarded():
    """The other side. Without this, 'always take the retry' would pass."""
    summary, coverage = retry_on_coverage_miss(
        "Kael Aurelio", KEY_TERMS, lambda missing: "nothing useful")
    assert summary == "Kael Aurelio"
    assert coverage == 0.5


def test_an_empty_retry_is_discarded():
    """The guard that existed on one side only."""
    summary, coverage = retry_on_coverage_miss(
        "Kael Aurelio", KEY_TERMS, lambda missing: "")
    assert summary == "Kael Aurelio"
    assert coverage == 0.5


def test_no_must_terms_means_no_retry():
    calls = []
    empty = {"entities": [], "figures": [], "identifiers": []}
    retry_on_coverage_miss("anything", empty, lambda m: calls.append(m))
    assert calls == [], "retried despite there being nothing to preserve"


def test_the_threshold_is_the_setting(monkeypatch):
    """Two-sided on the gate itself: the same summary must retry or not
    depending only on the configured threshold."""
    calls = []

    def retry_fn(missing):
        calls.append(missing)
        return ""

    monkeypatch.setattr(settings, "turn_summary_coverage_threshold", 0.4)
    retry_on_coverage_miss("Kael Aurelio", KEY_TERMS, retry_fn)   # 0.5 >= 0.4
    assert calls == []

    monkeypatch.setattr(settings, "turn_summary_coverage_threshold", 0.9)
    retry_on_coverage_miss("Kael Aurelio", KEY_TERMS, retry_fn)   # 0.5 < 0.9
    assert len(calls) == 1
