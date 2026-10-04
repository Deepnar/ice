"""Frozen answering must exclude oracle labels and resume only successful calls."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.lme.cloud_provider import TextGenerator
from scripts.z1 import answer_as_of


@pytest.mark.parametrize("failure", ["transport", "incomplete", "empty"])
def test_answer_transport_receipt_and_interrupted_resume(monkeypatch, failure):
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
            if failure == "transport":
                raise RuntimeError("intentional interrupted provider call")
            return {"status": "incomplete" if failure == "incomplete" else "completed",
                    "output_text": "prefix" if failure == "incomplete" else ""}
        return {"status": "completed", "output_text": "A saved answer", "id": "response-1", "usage": {"input_tokens": 25}}

    client = SimpleNamespace(responses=SimpleNamespace(create=request))
    monkeypatch.setattr(answer_as_of, "TextGenerator", lambda profile: TextGenerator(profile, client=client))
    with tempfile.TemporaryDirectory(prefix="z1-answer-resume-", dir=answer_as_of.LOGS) as root:
        trace, out = Path(root) / "trace.jsonl", Path(root) / "answers.json"
        trace.write_text(json.dumps({"event": "run", "meta": {"inputs": [1, 2, 3, 4, 5],
            "extra": {"clock_policy": "simulated"}}}) + "\n")
        audit = Path(root) / "labels.json"
        audit.write_text(json.dumps({"kind": "longterm_probe_label_review", "inputs": [1, 2, 3, 4, 5],
            "records": [{"probe_id": "p1", "conversation": "synthetic", "cutoff_turn": 80,
                "question": probe["question"], "expected_answer": probe["expected_answer"],
                "reviewed_expected_answer": "ORACLE REVIEWED KEY CANARY", "gold_turns": [1],
                "verdict": "valid", "reason": "Reviewed original and intervening history.",
                "reviewed_through_turn": 80, "recent_only_answerable": False,
                "knowledge_scope": "private_history", "task_types": ["episodic_lookup"],
                "question_family_id": "synthetic_family"}]}))
        argv = ["answer_as_of.py", "--trace", str(trace), "--out", str(out),
                "--arm", "full", "--allow-partial", "--validated-probes", str(audit)]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(RuntimeError, match="intentional interrupted|not successful|cloud answer was empty"):
            answer_as_of.main()

        assert not json.loads(out.read_text())["complete"]
        monkeypatch.setattr(sys, "argv", argv + ["--resume"])
        assert answer_as_of.main() == 0
        result = json.loads(out.read_text())
        row = result["records"][0]
        assert row["expected_answer"] == "ORACLE REVIEWED KEY CANARY"
        assert row["catalog_expected_answer"] == "ORACLE EXPECTED CANARY"
        assert row["label_review"]["question_family_id"] == "synthetic_family"
        assert probe["expected_answer"] == "ORACLE EXPECTED CANARY"
        assert row["answer_input_messages"] == messages
        assert row["answer_input_sha256"] == answer_as_of.input_digest(messages)
        assert row["answer_completion_status"] == "completed"
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


@pytest.mark.parametrize("mode", ["isolated", "outage", "quota", "interrupt"])
def test_continue_answers_have_durable_attempt_caps_and_terminal_errors(monkeypatch, mode):
    from experiments.lme.cloud_provider import ProviderAccessError
    probes = []
    for index in range(4):
        stage = {"prompt_messages": [{"role": "user", "content": f"Synthetic question {index}"}],
                 "prompt_tokens": 4, "selected_tokens": 0, "gold_fragment_coverage": {"selected": 0}}
        probes.append({"probe_id": f"p{index}", "conversation": "synthetic", "split_turn": 80,
            "question": f"Synthetic question {index}", "type": "episodic", "gold_turns": [1],
            "gold_sources": [{"turn": 1, "recorded_at": "2025-01-01T00:00:00+00:00",
                             "prompt": "source", "response": "reply", "source_ids": ["row-1"]}],
            "expected_answer": "source", "preflight": stage})
    monkeypatch.setattr(answer_as_of, "load_probes", lambda *a, **k: probes)
    monkeypatch.setattr(answer_as_of.time, "sleep", lambda _: None)
    calls = []
    def generate(*a, **k):
        calls.append(a)
        if mode == "quota":
            raise ProviderAccessError("fixture", "quota", 429, "quota")
        if mode == "outage" or mode == "isolated" and len(calls) <= 2:
            raise ConnectionError("synthetic connection fault")
        if mode == "interrupt" and len(calls) == 1:
            raise KeyboardInterrupt()
        return SimpleNamespace(text="supported answer", usage={}, response_id="fixture", completion_status="completed")
    monkeypatch.setattr(answer_as_of, "TextGenerator", lambda profile: SimpleNamespace(generate=generate))
    with tempfile.TemporaryDirectory(prefix="z1-answer-fault-control-", dir=answer_as_of.LOGS) as root:
        trace, out = Path(root) / "seed.jsonl", Path(root) / "answers.json"
        trace.write_text(json.dumps({"event": "run", "meta": {"extra": {"clock_policy": "fixture"}}}) + "\n")
        argv = ["answer_as_of.py", "--trace", str(trace), "--out", str(out), "--arm", "full",
                "--allow-partial", "--failure-policy", "continue"]
        monkeypatch.setattr(sys, "argv", argv)
        if mode == "isolated":
            assert answer_as_of.main() == 0
        else:
            with pytest.raises(KeyboardInterrupt if mode == "interrupt" else RuntimeError):
                answer_as_of.main()
        saved = json.loads(out.read_text())
        assert len(calls) == {"isolated": 5, "outage": 6, "quota": 1, "interrupt": 1}[mode]
        if mode == "isolated":
            assert saved["processing_complete"] and not saved["complete"] and saved["cloud_errors"] == 1
        if mode == "interrupt":
            assert saved["records"][0]["cloud_attempts"] == 1
            assert saved["records"][0]["error_type"] == "UnconfirmedCloudCall"
        before = len(calls)
        monkeypatch.setattr(answer_as_of, "TextGenerator", lambda profile: SimpleNamespace(generate=lambda *a, **k:
            calls.append(a) or SimpleNamespace(text="supported answer", usage={}, response_id="fixture", completion_status="completed")))
        monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
        assert answer_as_of.main() == 0
        assert len(calls) - before == {"isolated": 0, "outage": 1, "quota": 4, "interrupt": 4}[mode]
        assert json.loads(out.read_text())["processing_complete"]
        assert answer_as_of.main() == 0
        assert len(calls) - before == {"isolated": 0, "outage": 1, "quota": 4, "interrupt": 4}[mode]
