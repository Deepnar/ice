"""Frozen answering must exclude oracle labels and resume only successful calls."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.lme.cloud_provider import TextGenerator
from scripts.z1 import answer_as_of


def test_answer_transport_receipt_and_interrupted_resume(monkeypatch):
    messages = [{"role": "system", "content": "Answer using the supplied history."},
                {"role": "user", "content": "Which port did I set?"}]
    stage = {"prompt_messages": messages, "prompt_tokens": 25, "selected_tokens": 0,
             "gold_fragment_coverage": {"selected": 0}, "selected": [],
             "gate": {"final_retrieve": False}}
    probe = {"probe_id": "p1", "conversation": "synthetic", "split_turn": 80,
             "question": "Which port did I set?", "question_time": "2025-01-01T00:00:00+00:00",
             "question_time_provenance": "synthetic_raw_import", "type": "episodic",
             "gold_turns": [1], "gold_sources": [{"turn": 1,
                 "recorded_at": "2024-01-01T00:00:00+00:00", "prompt": "ORACLE SOURCE CANARY",
                 "response": "ORACLE RESPONSE CANARY", "source_ids": ["source1"]}],
             "expected_answer": "ORACLE EXPECTED CANARY", "preflight": stage}
    monkeypatch.setattr(answer_as_of, "load_probes", lambda *a, **k: [probe])
    monkeypatch.setenv("PROBE_API_BASE_URL", "https://test.invalid/v1")
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("intentional interrupted provider call")
        return {"output_text": "A saved answer", "id": "response-1", "usage": {"input_tokens": 25}}

    client = SimpleNamespace(responses=SimpleNamespace(create=request))
    monkeypatch.setattr(answer_as_of, "TextGenerator", lambda profile: TextGenerator(profile, client=client))
    with tempfile.TemporaryDirectory(prefix="z1-answer-resume-", dir=answer_as_of.LOGS) as root:
        trace, out = Path(root) / "trace.jsonl", Path(root) / "answers.json"
        trace.write_text(json.dumps({"event": "run", "meta": {"extra": {"clock_policy": "simulated"}}}) + "\n")
        argv = ["answer_as_of.py", "--trace", str(trace), "--out", str(out),
                "--arm", "full", "--allow-partial"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(RuntimeError, match="intentional interrupted"):
            answer_as_of.main()
        assert not json.loads(out.read_text())["complete"]
        monkeypatch.setattr(sys, "argv", argv + ["--resume"])
        assert answer_as_of.main() == 0
        result = json.loads(out.read_text())
        row = result["records"][0]
        assert row["answer_input_messages"] == messages
        assert row["answer_input_sha256"] == answer_as_of.input_digest(messages)
        assert row["prompt_evidence_support"] == "unreviewed"
        assert result["failed_attempts"][0]["error"]
        assert result["decoding"]["temperature_sent"] is False
        assert result["decoding"]["temperature_policy"] == "provider_default"
        assert len(calls) == 2
        for request_data in calls:
            assert request_data["input"] == messages
            assert "temperature" not in request_data
            assert request_data["model"] == "gpt-6-luna"
            assert "ORACLE" not in json.dumps(request_data)
            assert request_data["extra_headers"]["x-opencode-session"] == "ice-v3-z1-full-p1"
        assert answer_as_of.main() == 0
        assert len(calls) == 2
        result["records"][0]["answer_input_messages"][0]["content"] = "changed input"
        out.write_text(json.dumps(result))
        with pytest.raises(ValueError, match="differs from the frozen prompt"):
            answer_as_of.main()
