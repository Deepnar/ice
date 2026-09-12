"""v3 evidence survives compression choices across the actual read adapters."""

from types import SimpleNamespace

import pytest

from src.api import prompt_assembler
from src.api.config import settings
from src.memory.representation import choose_representation
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.services import retrieval_svc


def turn(**changes):
    fields = dict(
        raw_text="User: Atlas uses Redis.\n\nAssistant: Keep Redis persistence disabled.",
        summary_text="Atlas uses Redis.",
        summary_coverage=0.9,
        abstract_text="Atlas enables persistence.",
        inject_raw=False,
        id="test",
        conversation_id="test",
        timestamp=None,
        topic_tags=[],
        is_bookmarked=False,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "coverage", [None, -0.1, 0.69, float("nan"), float("inf"), 1.1, True]
)
def test_unqualified_summary_cannot_replace_source(coverage):
    row = turn(summary_coverage=coverage)
    assert choose_representation(row) == (row.raw_text, None, None)


def test_setting_controls_real_reader(monkeypatch):
    row = turn(summary_coverage=0.8)
    orch = HybridRetrievalOrchestrator(None, None)
    monkeypatch.setattr(settings, "turn_summary_coverage_threshold", 0.85)
    assert orch._choose_representation(row, None, set())[0] == row.raw_text
    monkeypatch.setattr(settings, "turn_summary_coverage_threshold", 0.8)
    assert orch._choose_representation(row, None, set())[0] == row.summary_text


def test_each_matched_detail_survives_compression():
    row = turn()
    assert choose_representation(row, prompt_keywords={"atlas", "persistence"}) == (
        row.raw_text,
        None,
        None,
    )


def test_abstract_needs_own_source_support():
    row = turn(inject_raw=True)
    assert choose_representation(row)[2] is None
    row.abstract_text = "Keep Redis persistence disabled."
    assert choose_representation(row)[2] == row.abstract_text
    assert choose_representation(row, prompt_keywords={"atlas"})[2] is None


def test_no_hidden_raw_prefix_limit():
    row = turn(raw_text="context " * 100 + "The port is 8391.", summary_text=None)
    assert choose_representation(row)[0].endswith("8391.")
    assert prompt_assembler._turn_text(row).endswith("8391.")


def test_missing_source_does_not_authorize_unmeasured_summary():
    assert choose_representation(turn(raw_text=None, summary_coverage=None))[0] is None


def test_chat_budget_cannot_bypass_summary_gate(monkeypatch):
    row = turn(
        raw_text="User: Persistence must stay disabled. " + "context " * 500,
        summary_text="Persistence must be enabled.",
        summary_coverage=0.1,
        abstract_text="Enable persistence.",
    )
    monkeypatch.setattr(prompt_assembler, "_recent_turn_rows", lambda *args: [row])
    messages = prompt_assembler.get_recent_turns(None, "test", max_tokens=200)
    rendered = " ".join(m["content"] for m in messages)
    assert "disabled" in rendered
    assert "enabled" not in rendered
    assert "Enable persistence" not in rendered


def test_explicit_recent_read_uses_same_contract():
    row = turn(summary_coverage=None)

    class Query:
        def filter_by(self, **kwargs):
            return self

        def order_by(self, *args):
            return self

        def limit(self, n):
            return self

        def all(self):
            return [row]

    db = SimpleNamespace(query=lambda *args: Query())
    assert retrieval_svc.recent_turns(db)[0]["text"] == row.raw_text
