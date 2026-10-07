"""Completed cloud stages must remain immutable while a paired judge resumes."""
import copy
import fcntl
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.z1 import answer_as_of, answer_resume, judge_answers, repair_judge_inputs
from scripts.z1.campaign_progress import ArtifactProgress
from scripts.z1.replay_checkpoint import atomic_json


def test_answer_retry_then_interrupted_judge_survives_upstream_resume(monkeypatch):
    probes = []
    for index in range(3):
        stage = {"prompt_messages": [{"role": "user", "content": f"Which port {index}?"}],
                 "prompt_tokens": 4, "selected_tokens": 0, "gold_fragment_coverage": {"selected": 0}}
        probes.append({"probe_id": f"p{index}", "conversation": "synthetic", "split_turn": 80,
                       "question": f"Which port {index}?", "type": "episodic", "gold_turns": [1],
                       "gold_sources": [{"turn": 1, "recorded_at": "2025-01-01T00:00:00+00:00",
                           "prompt": "Use port 7813", "response": "Agreed", "source_ids": ["source1"]}],
                       "expected_answer": "7813", "preflight": stage, "controls": {"no_codex": stage}})
    monkeypatch.setattr(answer_as_of, "load_probes", lambda *a, **k: probes)
    monkeypatch.setattr(answer_as_of, "load_selected_env", lambda: None)
    monkeypatch.setattr(answer_as_of.time, "sleep", lambda _: None)
    monkeypatch.setenv("PROBE_API_BASE_URL", "https://test.invalid/v1")
    requests = []
    def generate(*a, **k):
        requests.append(k["session_id"])
        attempt = len(requests) % 5
        if attempt in {1, 3, 4}:
            raise ConnectionError("synthetic one recovered request and one exhausted answer")
        return SimpleNamespace(text="7813", usage={}, response_id="synthetic", completion_status="completed")
    monkeypatch.setattr(answer_as_of, "TextGenerator", lambda _: SimpleNamespace(generate=generate))
    with tempfile.TemporaryDirectory(prefix="z1-completed-answer-resume-", dir=answer_as_of.LOGS) as directory:
        root = Path(directory)
        trace = root / "seed.jsonl"
        trace.write_text(json.dumps({"event": "run", "meta": {"extra": {"clock_policy": "synthetic"}}}) + "\n")
        for arm in ("full", "no_codex"):
            monkeypatch.setattr(sys, "argv", ["answer_as_of.py", "--trace", str(trace),
                "--out", str(root / f"answers-{arm}.json"), "--arm", arm,
                "--allow-partial", "--failure-policy", "continue"])
            assert answer_as_of.main() == 0
            saved = json.loads((root / f"answers-{arm}.json").read_text())
            assert saved["cloud_errors"] == 1 and not saved["complete"]
            recovered = saved["records"][0]
            assert recovered["cloud_attempts"] == 2 and recovered["answer"]
            assert not {"error_type", "error_status", "fault_disposition"} & recovered.keys()
            assert saved["failed_attempts"][0]["error"]
        assert len(requests) == 10
        judging = []
        def verdict(*a, **k):
            judging.append(k)
            if len(judging) == 2:
                raise KeyboardInterrupt()
            return {"verdict": "TIE", "reason": "equivalent", "A_grade": "correct", "B_grade": "correct"}
        monkeypatch.setattr(judge_answers, "judge_one", verdict)
        monkeypatch.setattr(judge_answers, "judge_provider_identity", lambda: {"judge_model": "fixture"})
        argv = ["judge_answers.py", "--a", str(root / "answers-full.json"),
                "--b", str(root / "answers-no_codex.json"), "--out", str(root / "judge.json"),
                "--failure-policy", "continue"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(KeyboardInterrupt):
            judge_answers.main()
        before = {arm: (root / f"answers-{arm}.json").read_bytes() for arm in ("full", "no_codex")}
        monkeypatch.setattr(answer_as_of, "TextGenerator", lambda _: pytest.fail("completed answering made a call"))
        for arm in before:
            path = root / f"answers-{arm}.json"
            stamp = path.stat().st_mtime_ns
            monkeypatch.setattr(sys, "argv", ["answer_as_of.py", "--trace", str(trace),
                "--out", str(path), "--arm", arm, "--allow-partial", "--failure-policy", "continue", "--resume"])
            assert answer_as_of.main() == 0
            assert path.read_bytes() == before[arm] and path.stat().st_mtime_ns == stamp
        monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
        assert judge_answers.main() == 0
        assert len(judging) == 5  # Four returned orders; the interrupted order consumes its first attempt.
        result = json.loads((root / "judge.json").read_text())
        assert result["processing_complete"] and result["cloud_errors"] == 1
        assert result["results"][1]["reason"] == "answer_failed"
        # Existing completed artifacts retain the old runner identity and
        # metadata. The closed transition must traverse the real receipt checks
        # without rewriting them or attempting a request.
        transition = json.loads(answer_resume.REGISTRY.read_text())["repairs"][0]
        for arm in before:
            path = root / f"answers-{arm}.json"
            legacy = json.loads(path.read_text())
            legacy["implementation_sha256"]["answer_runner"] = transition["before"]
            for field in ("completed_resume", "completed_resume_registry"):
                legacy["implementation_sha256"].pop(field)
            for row in legacy["records"]:
                if row.get("answer") and not row.get("error"):
                    row.update(error_type="UnconfirmedCloudCall", error_status=None)
                    if row["cloud_attempts"] == 2:
                        row["fault_disposition"] = "degraded_final"
            atomic_json(path, legacy)
            raw, stamp = path.read_bytes(), path.stat().st_mtime_ns
            monkeypatch.setattr(sys, "argv", ["answer_as_of.py", "--trace", str(trace),
                "--out", str(path), "--arm", arm, "--allow-partial", "--failure-policy", "continue", "--resume"])
            assert answer_as_of.main() == 0
            assert path.read_bytes() == raw and path.stat().st_mtime_ns == stamp


def _completed_identity():
    transition = json.loads(answer_resume.REGISTRY.read_text())["repairs"][0]
    current = {"version": "v3", "probe_ids": ["p1"], "trace_sha256": "trace",
               "answer_model": "fixture", "cloud_failure_policy": "continue",
               "implementation_sha256": {"answer_runner": transition["after"], "cloud_adapter": "adapter"}}
    saved = {**copy.deepcopy(current), "processing_complete": True,
             "records": [{"probe_id": "p1", "answer": "7813", "error": None}]}
    saved["implementation_sha256"]["answer_runner"] = transition["before"]
    return saved, current


@pytest.mark.parametrize("change", [None, "trace", "model", "adapter", "unregistered", "partial", "failed", "input_ids"])
def test_registered_completed_transition_is_read_only_and_exact(change):
    saved, current = _completed_identity()
    if change == "trace": saved["trace_sha256"] = "other"
    if change == "model": saved["answer_model"] = "other"
    if change == "adapter": saved["implementation_sha256"]["cloud_adapter"] = "other"
    if change == "unregistered": saved["implementation_sha256"]["answer_runner"] = "other"
    if change == "partial": saved["processing_complete"] = False
    if change == "failed": saved["records"][0]["error"] = "not terminal"
    if change == "input_ids": saved["records"][0]["probe_id"] = "different"
    original = copy.deepcopy(saved)
    assert answer_resume.completed_identity_matches(saved, current) is (change is None)
    assert saved == original


def _input_repair(root, monkeypatch):
    transition = json.loads(answer_resume.REGISTRY.read_text())["repairs"][0]
    atomic_json(root / "campaign.json", {"judge_profile": "opencode-muse13"})
    originals = {}
    for key, arm in (("a", "full"), ("b", "no_codex")):
        answer = {"version": "v3", "processing_complete": True,
                  "implementation_sha256": {"answer_runner": transition["before"]},
                  "records": [{"answer": "7813", "error": None, "error_type": "UnconfirmedCloudCall", "cloud_attempts": 2},
                              {"answer": "", "error": "real fault", "error_type": "UnconfirmedCloudCall",
                               "cloud_attempts": 2, "fault_disposition": "degraded_final"}]}
        originals[key] = json.dumps(answer, indent=2, default=str).encode()
        answer["records"][0]["fault_disposition"] = "degraded_final"
        atomic_json(root / f"answers-{arm}.json", answer)
    monkeypatch.setattr(judge_answers, "judge_provider_identity", lambda: {"judge_model": "fixture"})
    atomic_json(root / "judge-full-vs-no_codex.partial.json", {
        "judge_model": "fixture", "judge_implementation_sha256": hashlib.sha256(Path(judge_answers.__file__).read_bytes()).hexdigest(),
        "judge_rubric_sha256": hashlib.sha256(judge_answers.SYSTEM_V3.encode()).hexdigest(),
        "cloud_recovery_sha256": hashlib.sha256(Path(judge_answers.__file__).with_name("cloud_recovery.py").read_bytes()).hexdigest(),
        "answer_file_sha256": {key: hashlib.sha256(raw).hexdigest() for key, raw in originals.items()},
        "results": [{"sentinel": "preserve grades"}], "cloud_attempts": [{"sentinel": "preserve reservation"}]})
    return originals


@pytest.mark.parametrize("mode", ["repair", "changed_source", "already_one", "active", "campaign_active", "judge_changed"])
def test_hash_proven_input_repair_archives_before_restoring(monkeypatch, mode):
    with tempfile.TemporaryDirectory(prefix="z1-input-repair-", dir=answer_as_of.LOGS) as directory:
        root = Path(directory)
        originals = _input_repair(root, monkeypatch)
        if mode == "changed_source":
            path = root / "answers-no_codex.json"
            altered = json.loads(path.read_text())
            altered["records"][0]["answer"] = "changed actual answer"
            atomic_json(path, altered)
        if mode == "already_one": (root / "answers-full.json").write_bytes(originals["a"])
        if mode == "judge_changed": monkeypatch.setattr(judge_answers, "judge_provider_identity", lambda: {"judge_model": "other"})
        before = {p.name: p.read_bytes() for p in root.glob("*.json")}
        handle = (root / ("campaign.lock" if mode == "campaign_active" else "answers-full.lock")).open("a+")
        try:
            if mode in {"active", "campaign_active"}: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if mode in {"changed_source", "active", "campaign_active", "judge_changed"}:
                with pytest.raises((ValueError, RuntimeError)):
                    repair_judge_inputs.repair(root, apply=True)
                assert not list(root.glob("answer-input-recovery-*"))
                assert {p.name: p.read_bytes() for p in root.glob("*.json")} == before
                return
            planned = repair_judge_inputs.repair(root)
            assert not planned["applied"] and not list(root.glob("answer-input-recovery-*"))
            result = repair_judge_inputs.repair(root, apply=True)
            archive = Path(result["archive"])
            for key, arm in (("a", "full"), ("b", "no_codex")):
                name = f"answers-{arm}.json"
                assert (root / name).read_bytes() == originals[key]
                assert (archive / name).read_bytes() == before[name]
            name = "judge-full-vs-no_codex.partial.json"
            assert (root / name).read_bytes() == (archive / name).read_bytes() == before[name]
            assert not repair_judge_inputs.repair(root, apply=True)["applied"]
        finally:
            handle.close()


def test_progress_does_not_count_successful_stale_fault_markers(tmp_path):
    atomic_json(tmp_path / "answers-full.json", {"records": [
        {"probe_id": "success", "answer": "7813", "error": None, "fault_disposition": "degraded_final"},
        {"probe_id": "failed", "answer": "", "error": "real fault", "fault_disposition": "degraded_final"}]})
    observation = ArtifactProgress(tmp_path, "answers", arm="full", admitted=2).observe()
    assert observation.done == 1 and "ungraded errors 1" in observation.detail
