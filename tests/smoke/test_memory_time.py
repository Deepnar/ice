"""v3 clock precision, origin labels, and representation budgeting."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.api.prompt_assembler import assemble_prompt
from src.memory.time_format import format_time, recorded_stamp
from src.memory.tokens import count
from src.retrieval.orchestrator import HybridRetrievalOrchestrator


def test_timezone_conversion_and_unknown_timezone():
    local = datetime(
        2026, 9, 13, 6, 5, 4, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    assert format_time(local) == "2026-09-13T00:35:04Z"
    assert (
        format_time(local.replace(tzinfo=None))
        == "2026-09-13T06:05:04 (timezone unknown)"
    )
    assert format_time(None) == "unknown"


def test_import_clock_is_not_claimed_as_source_time():
    t = datetime(2026, 9, 13, tzinfo=timezone.utc)
    assert "source time unknown" in recorded_stamp(t, "synthetic_raw_import")
    assert "provenance unknown" in recorded_stamp(t)
    assert "source recorded" in recorded_stamp(t, "original")


def test_date_survives_each_eligible_representation_and_budget():
    row = SimpleNamespace(
        id="turn",
        conversation_id=None,
        raw_text="detail " * 100 + "Keep Redis disabled.",
        summary_text="Keep Redis disabled.",
        abstract_text="Redis disabled.",
        summary_coverage=0.9,
        inject_raw=True,
        timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        ts_provenance="original",
    )
    from src.memory.source import single_provenance
    from src.memory.representation import verify_representations
    from src.memory.support import verify_support
    row.source_spans = single_provenance(row.raw_text, 'user')
    row.representation_verification = verify_representations(row, row.summary_text,
        row.abstract_text, verifier=lambda source, claim: verify_support(source, claim,
            scorer=lambda pairs: [dict(entailment=.99, neutral=.005, contradiction=.005)]))
    o = HybridRetrievalOrchestrator(None, None)
    f = o._rows_to_fragments([row], "episodic")[0]
    for t in (f.text, f.degrade_text, f.abstract_text):
        assert t.startswith("[source recorded: 2026-01-02T03:04:05Z]")
    selected = o._enforce_token_budget(
        [f], max_tokens=count(f.degrade_text), relevance_order=True
    )
    assert selected[0].text == f.degrade_text
    assert selected[0].token_count == count(selected[0].text)


def test_fact_without_source_time_does_not_invent_event_date():
    o = HybridRetrievalOrchestrator(None, None)
    e = SimpleNamespace(
        relation="uses",
        negated=True,
        source_batch=None,
        learned_at=datetime(2026, 9, 13, 1, 2, tzinfo=timezone.utc),
        valid_from=None,
    )
    rendered = o._fact_line(
        SimpleNamespace(canonical_name="Atlas"),
        e,
        SimpleNamespace(canonical_name="Redis"),
    )
    assert "NOT uses" in rendered and "source time unknown" in rendered
    assert "learned: 2026-09-13T01:02:00Z" in rendered


def test_system_anchor_supplies_clock_time():
    messages = assemble_prompt([], [], "hello")
    assert "Current date and time (UTC):" in messages[0]["content"]
    assert datetime.now(timezone.utc).strftime("%Y-%m-%dT") in messages[0]["content"]
    assert messages[-1]["content"] == "hello"
