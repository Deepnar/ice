"""A nested unknown verdict must not disappear behind a present container."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.z1 import report_v3_replay as report


def trace(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in
        [{"event": "run", "meta": {"extra": {}}}, *rows]) + "\n")


def test_nested_verdicts_have_independent_candidate_denominators(tmp_path):
    path = tmp_path / "seed.jsonl"
    trace(path, [
        {"event": "written", "summary_text": "A summary", "abstract_text": "An abstract",
         "summary_support": {"summary": {"status": "unknown", "reason": "SupportInputError"},
                             "abstract": {"status": "supported", "reason": "model_score"}}},
        {"event": "written", "summary_text": "", "abstract_text": None, "summary_support": {}},
        {"event": "written", "summary_text": "Another summary", "summary_support": {}},
        {"event": "written", "summary_text": "A wrong summary", "abstract_text": "Wrong abstract",
         "summary_support": {"summary": {"status": "contradicted", "reason": "model_score"},
                             "abstract": {"status": "unknown", "reason": "SupportInputError"}}},
        {"event": "written", "summary_text": "Malformed record", "summary_support": {"summary": []}},
    ])
    result = report.summarize(path)
    summaries = result["post_flight_representation_support"]["summary"]
    assert summaries["counts"] == dict(written_turns=5, no_candidate=1, candidates=4,
        missing_verdict=1, invalid_verdict=1, supported=0, contradicted=1, unknown=1)
    assert summaries["status_reasons"]["unknown:SupportInputError"] == 1
    abstracts = result["post_flight_representation_support"]["abstract"]["counts"]
    assert abstracts["candidates"] == 2 and abstracts["no_candidate"] == 3
    assert abstracts["supported"] == 1 and abstracts["unknown"] == 1
    assert result["post_flight_writes"]["summary_support_unknown"] == 1
    assert result["report_schema_version"] == 2
    assert len(result["trace_sha256"]) == len(result["reporter_sha256"]) == 64


def test_existing_report_and_trace_are_never_overwritten(monkeypatch):
    logs = Path(report.__file__).resolve().parents[2] / "logs"
    with TemporaryDirectory(dir=logs, prefix="report-smoke-") as temporary:
        root = Path(temporary)
        path, output = root / "seed.jsonl", root / "report.json"
        trace(path, [])
        original = path.read_bytes()
        output.write_text("frozen report")
        monkeypatch.setattr(sys, "argv", ["report", str(path), "--out", str(output)])
        with pytest.raises(ValueError, match="already exists"):
            report.main()
        assert output.read_text() == "frozen report"
        monkeypatch.setattr(sys, "argv", ["report", str(path), "--out", str(path)])
        with pytest.raises(ValueError, match="already exists"):
            report.main()
        assert path.read_bytes() == original


def test_actual_cli_writes_a_new_hash_bound_report(monkeypatch, capsys):
    logs = Path(report.__file__).resolve().parents[2] / "logs"
    with TemporaryDirectory(dir=logs, prefix="report-smoke-") as temporary:
        root = Path(temporary)
        path, output = root / "seed.jsonl", root / "derived" / "report.json"
        trace(path, [{"event": "written", "summary_text": "Summary",
                      "summary_support": {"summary": {"status": "unknown", "reason": "SupportInputError"}}}])
        original = path.read_bytes()
        monkeypatch.setattr(sys, "argv", ["report", str(path), "--out", str(output)])
        assert report.main() == 0
        saved = json.loads(output.read_text())
        assert saved == json.loads(capsys.readouterr().out)
        assert saved["post_flight_writes"]["summary_support_unknown"] == 1
        assert path.read_bytes() == original
        frozen, modified = output.read_bytes(), output.stat().st_mtime_ns
        assert report.main() == 0
        assert output.read_bytes() == frozen and output.stat().st_mtime_ns == modified
