"""Logged catch receipts preserve native logs/bindings and fault semantics."""

import json
import sys
from types import SimpleNamespace as NS

import pytest

from experiments.v3_repair.caught_faults import observe
from experiments.v3_repair.fault_receipts import FaultRecorder


class Logger:
    def __init__(self, seen, **bound):
        self.seen, self.bound = seen, bound

    def bind(self, **kwargs):
        return Logger(self.seen, **self.bound, **kwargs)

    def warning(self, event, **kwargs):
        self.seen.append((event, {**self.bound, **kwargs}))


def test_caught_chain_logged_before_handler_exits_and_logger_restored(tmp_path, monkeypatch):
    seen = []
    logger = Logger(seen)
    module = NS(logger=logger)
    monkeypatch.setitem(sys.modules, "src.controlled_caught_fixture", module)
    faults = FaultRecorder(tmp_path / "faults.jsonl", "fixture", private_root=tmp_path)
    context = dict(job="post_flight", turn=12, attempt=1, attempts=2)
    with observe(faults, context):
        bound = module.logger.bind(batch="controlled")
        bound.warning("healthy", status="ok")
        try:
            try:
                raise ValueError("inner field invalid")
            except ValueError as exc:
                raise RuntimeError("outer incomplete") from exc
        except RuntimeError:
            bound.warning("fallback", status="degraded")
    assert module.logger is logger
    assert seen == [("healthy", dict(batch="controlled", status="ok")),
                    ("fallback", dict(batch="controlled", status="degraded"))]
    row = json.loads(faults.path.read_text())
    assert [e["error_type"] for e in row["exception_chain"]] == ["RuntimeError", "ValueError"]
    assert row["context"]["turn"] == 12 and row["context"]["attempt"] == 1
    assert row["context"]["stage"].endswith(":fallback")
    assert "inner field invalid" in row["traceback"] and "locals" not in row
    assert faults.verify() == 1


def test_interrupt_restores_loggers_and_does_not_change_exception(tmp_path, monkeypatch):
    logger = Logger([])
    module = NS(logger=logger)
    monkeypatch.setitem(sys.modules, "src.controlled_caught_fixture", module)
    faults = FaultRecorder(tmp_path / "faults.jsonl", "fixture", private_root=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with observe(faults, dict(job="fixture")):
            raise KeyboardInterrupt("unchanged")
    assert module.logger is logger and faults.verify() == 0


def test_nested_observer_refuses_and_restores_outer_binding(tmp_path, monkeypatch):
    logger = Logger([])
    module = NS(logger=logger)
    monkeypatch.setitem(sys.modules, "src.controlled_caught_fixture", module)
    faults = FaultRecorder(tmp_path / "faults.jsonl", "fixture", private_root=tmp_path)
    with observe(faults, dict(job="outer")):
        outer = module.logger
        with pytest.raises(ValueError, match="overlap"):
            with observe(faults, dict(job="inner")):
                pass
        assert module.logger is outer
    assert module.logger is logger


def test_native_summary_fallback_retains_original_behavior_and_full_fault(tmp_path, monkeypatch):
    from src.memory import embedder
    monkeypatch.setattr(embedder, "get_embedder", lambda: NS(encode=lambda *_a, **_k: [1.] + [0.] * 1023))
    from src.workers import post_flight
    from src.workers.completion_text import IncompleteCompletion
    faults = FaultRecorder(tmp_path / "faults.jsonl", "summary-fixture", private_root=tmp_path)
    def incomplete(*_a, **_k):
        raise IncompleteCompletion("controlled provider length", finish_reason="length",
                                   details=dict(completion_tokens=900))
    monkeypatch.setattr(post_flight, "_summary_llm_call", incomplete)
    with observe(faults, dict(job="post_flight", turn=7, attempt=1, attempts=2)):
        result = post_flight.generate_summary("Authored source.", "Recorded reply.",
            dict(entities=[], figures=[], identifiers=[]), "controlled-model")
    assert result == ("", 0., None)
    rows = [json.loads(line) for line in faults.path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["error_type"] == "IncompleteCompletion"
    assert rows[0]["context"]["stage"].endswith(":background_summarization_failed")
    assert "controlled provider length" in rows[0]["traceback"]
