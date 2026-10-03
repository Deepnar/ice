"""One manual entry point must fail closed and preserve resumable stages."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.z1 import run_v3_campaign as campaign
from scripts.z1.label_review import blank_review, validate_packet, validate_review


def config():
    identity = "a" * 32
    return {"format": "ice-v3-manual-campaign-1", "campaign_id": identity,
            "database": "ice_v3_campaign_" + identity[:16],
            "snapshot_arm": "manual-v3-" + identity,
            "answer_profile": "opencode-luna6", "checkpoint_every": 10,
            "background_model": "writer", "codex_extraction_model": "extractor",
            "probe_panel": "existing"}


def labels(root):
    row = {"verdict": "valid", "reason": "Complete source and recent review.",
           "cutoff_turn": 80, "reviewed_through_turn": 80,
           "recent_only_answerable": False, "knowledge_scope": "private_history",
           "task_types": ["episodic_lookup"]}
    (root / "labels-source-linked.json").write_text(json.dumps({"records": [row]}))
    (root / "labels-native.json").write_text('{"records":[]}')


def providers(monkeypatch):
    monkeypatch.setenv("PROBE_API_KEY", "synthetic-test-credential")
    monkeypatch.setenv("PROBE_API_BASE_URL", "https://test.invalid/v1")
    monkeypatch.setenv("PROBE_MODEL", "synthetic-judge")


def test_default_status_never_connects_database_or_provider(tmp_path, monkeypatch):
    (tmp_path / "campaign.json").write_text(json.dumps(config()))
    monkeypatch.setattr(campaign, "private_path", lambda p: p)
    monkeypatch.setattr(campaign, "database_environment", lambda _: pytest.fail("status created database"))
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_a, **_k: pytest.fail("status called child"))
    monkeypatch.setattr(sys, "argv", ["run_v3_campaign.py", "--run-dir", str(tmp_path)])
    assert campaign.main() == 0
    assert not campaign.status(tmp_path, config())["cloud_answers_ready"]


def test_status_distinguishes_structural_exclusions_and_source_review(tmp_path, monkeypatch):
    labels(tmp_path)
    path = tmp_path / "labels-source-linked.json"
    valid = json.loads(path.read_text())["records"][0]
    valid.update(reviewer="fixture reviewer", review_scope="complete_history_through_cutoff")
    path.write_text(json.dumps({"records": [valid,
        {"verdict": "invalid", "reviewer": "window eligibility", "reason": "cutoff is inside recent window"},
        {"verdict": "uncertain", "reviewer": "fixture reviewer", "review_scope": "complete_history_through_cutoff"},
        {"verdict": None}]}))
    monkeypatch.setattr(campaign, "database_environment", lambda _: pytest.fail("status created database"))
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_a, **_k: pytest.fail("status called a stage"))
    counts = campaign.status(tmp_path, config())["ground_truth"]["source_linked_review"]
    assert (counts["valid"], counts["invalid"], counts["uncertain"], counts["unreviewed"]) == (1, 1, 1, 1)
    assert counts["source_answer_reviews_recorded"] == 2
    assert counts["without_recorded_source_answer_review"] == 2
    assert campaign.campaign_report(tmp_path, config())["ground_truth"]["source_linked_review"] == counts
    assert "still need" in campaign.status(tmp_path, config())["limits"][0]

    rows = json.loads(path.read_text())["records"]
    for row in rows:
        row.update(verdict="uncertain", reviewer="fixture reviewer",
                   review_scope="complete_history_through_cutoff")
    path.write_text(json.dumps({"records": rows}))
    result = campaign.status(tmp_path, config())
    assert result["ground_truth"]["source_linked_review"]["without_recorded_source_answer_review"] == 0
    assert "every row" in result["limits"][0]
    assert "invalid/uncertain cases remain excluded" in result["limits"][0]
    assert not result["cloud_answers_ready"] and not result["score_of_record"]


def test_manual_report_exposes_reviewed_strata_without_a_cloud_call(tmp_path, monkeypatch):
    from scripts.z1.judge_answers import reviewed_outcome_strata
    monkeypatch.setattr(campaign, "status", lambda *_a: {"version": "v3", "stages": {}})
    strata = reviewed_outcome_strata([{
        "label_review": {"knowledge_scope": "private_history", "task_types": ["procedural"]},
        "winner": "vector_only", "arm_a_grade": "incorrect", "arm_b_grade": "correct"}])
    (tmp_path / "judge-full-vs-vector_only.json").write_text(json.dumps({
        "complete": True, "reviewed_outcome_strata": strata, "results": [{}]}))
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_a, **_k: pytest.fail("report launched a stage"))
    result = campaign.campaign_report(tmp_path, config())
    receipt = result["artifacts"]["judge-full-vs-vector_only.json"]
    assert receipt["reviewed_outcome_strata"] == strata
    assert receipt["source_grade_repetitions_are_not_independent"]


def test_unreviewed_full_campaign_blocks_before_database_or_api(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "database_environment", lambda _: pytest.fail("blocked run connected"))
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_a, **_k: pytest.fail("blocked run called child"))
    assert campaign.execute(tmp_path, config(), "all") == 2
    assert not (tmp_path / "stage-status.json").exists()


@pytest.mark.parametrize("failed_stage", campaign.STAGES)
def test_every_stage_failure_retains_status_and_stops_later_children(tmp_path, monkeypatch, failed_stage):
    monkeypatch.setattr(campaign, "status", lambda *_a: {"reviewed_label_candidates_available": True})
    monkeypatch.setattr(campaign, "validate_labels", lambda *_a: 1)
    monkeypatch.setattr(campaign, "progress_totals", lambda *_a: (1, 1))
    monkeypatch.setattr(campaign, "database_environment", lambda *_a: {})
    monkeypatch.setattr(campaign, "load_selected_env", lambda: None)
    monkeypatch.setattr(campaign, "PROFILES", {"opencode-luna6": SimpleNamespace(
        resolved_base_url=lambda: "https://test.invalid/v1", resolved_api_key=lambda: "fixture")})
    providers(monkeypatch)
    scripts = {"seed_v3.py": "seed", "snapshot.py": "snapshot", "answer_as_of.py": "answers",
               "judge_answers.py": "judge", "report_v3_replay.py": "report"}
    seen = []

    def child(command, **kwargs):
        current = scripts[command[1].rsplit("/", 1)[-1]]
        seen.append(current)
        assert kwargs["check"] and kwargs["stderr"] == subprocess.STDOUT
        if current == failed_stage:
            kwargs["stdout"].write("synthetic child failure retained\n")
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", child)
    with pytest.raises(subprocess.CalledProcessError):
        campaign.execute(tmp_path, config(), "all")
    state = json.loads((tmp_path / "stage-status.json").read_text())
    assert state[failed_stage] == "interrupted_or_failed_resume_required"
    index = campaign.STAGES.index(failed_stage)
    assert all(s not in state and s not in seen for s in campaign.STAGES[index + 1:])
    assert all(state[s] == "complete_diagnostic" for s in campaign.STAGES[:index])


def test_top_level_child_failure_is_readable_and_does_not_hide_log(tmp_path, monkeypatch, capsys):
    (tmp_path / "campaign.json").write_text(json.dumps(config()))
    monkeypatch.setattr(campaign, "private_path", lambda p: p)

    def fail(*_args):
        raise subprocess.CalledProcessError(1, [sys.executable, "/fixture/seed_v3.py"])

    monkeypatch.setattr(campaign, "execute", fail)
    monkeypatch.setattr(sys, "argv", ["run_v3_campaign.py", "--run-dir", str(tmp_path), "--run"])
    assert campaign.main() == 1
    display = capsys.readouterr().err
    assert str(tmp_path / "seed_v3.log") in display and "Saved artifacts retained" in display
    assert "Traceback" not in display


def test_label_age_and_a_valid_word_are_not_long_term_truth(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps({"records": [
        {"verdict": "valid", "reason": "old source", "cutoff_turn": 80}]}))
    counts = campaign.label_counts(tmp_path / "labels.json", "verdict")
    assert counts["valid"] == 0 and counts["marked_valid_but_incomplete"] == 1
    with pytest.raises(ValueError, match="exact cutoff"):
        validate_review(blank_review(), 80)
    labels(tmp_path)
    row = json.loads((tmp_path / "labels-source-linked.json").read_text())["records"][0]
    row["recent_only_answerable"] = True
    with pytest.raises(ValueError, match="recent history"):
        validate_review(row, 80)


def test_readiness_verifies_trace_and_all_arm_plan_without_cloud(tmp_path, monkeypatch):
    labels(tmp_path)
    (tmp_path / "seed.jsonl").write_text("incomplete trace")
    seen = []
    def planned(command, **kwargs):
        seen.append(command)
        assert "--plan" in command and kwargs["capture_output"]
        return SimpleNamespace(returncode=1, stdout="")
    monkeypatch.setattr(campaign.subprocess, "run", planned)
    result = campaign.status(tmp_path, config())
    assert result["reviewed_label_candidates_available"] and not result["cloud_answers_ready"]
    assert len(seen) == 1


def test_interrupted_answer_stage_is_resumed_by_same_manual_command(tmp_path, monkeypatch):
    labels(tmp_path)
    providers(monkeypatch)
    monkeypatch.setattr(campaign, "validate_labels", lambda _: 1)
    seen = []
    def failed(command, **_kwargs):
        seen.append(command)
        (tmp_path / "answers-full.json").write_text('{"complete":false}')
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(campaign.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        campaign.execute(tmp_path, config(), "answers")
    assert json.loads((tmp_path / "stage-status.json").read_text())["answers"].startswith("interrupted")
    seen.clear()
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **_kwargs: seen.append(command))
    assert campaign.execute(tmp_path, config(), "answers") == 0
    assert "--resume" in seen[0] and all("--resume" not in c for c in seen[1:])
    assert len(seen) == 4


def test_report_is_published_after_complete_state(tmp_path, monkeypatch):
    (tmp_path / "seed.jsonl").write_text("development trace")
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_a, **_k: None)
    assert campaign.execute(tmp_path, config(), "report") == 0
    report = json.loads((tmp_path / "campaign-report.json").read_text())
    assert report["stages"]["report"] == "complete_diagnostic"
    assert not report["score_of_record"]


def test_init_cannot_launch_a_run_or_overwrite_reviews(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "initialize", lambda _: pytest.fail("invalid invocation initialised"))
    monkeypatch.setattr(sys, "argv", ["run_v3_campaign.py", "--run-dir", str(tmp_path), "--init", "--run"])
    with pytest.raises(SystemExit):
        campaign.main()
    (tmp_path / "campaign.json").write_text(json.dumps(config()))
    # The implementation's existing-bundle guard protects edited private labels.
    from scripts.z1.run_v3_campaign import read_config
    assert read_config(tmp_path) == config()
    changed = config()
    changed["database"] = "ice_db"
    (tmp_path / "campaign.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity"):
        read_config(tmp_path)


def test_background_manifest_change_is_not_the_same_writer(monkeypatch):
    from scripts.z1 import replay_checkpoint
    settings = SimpleNamespace(ollama_base_url="http://test.invalid", codex_extraction_model="extractor:q8")
    monkeypatch.setattr(replay_checkpoint.httpx, "get", lambda *_a, **_k:
        SimpleNamespace(raise_for_status=lambda: None,
                        json=lambda: {"models": [{"name": "writer:latest", "digest": "one"},
                                                 {"name": "extractor:q8", "digest": "two"}]}))
    assert replay_checkpoint.background_model_digests(settings, "writer") == {"writer": "one", "extractor:q8": "two"}
    with pytest.raises(ValueError, match="no installed Ollama digest"):
        replay_checkpoint.background_model_digests(settings, "missing")


def test_packet_preflight_checks_sources_before_any_seed_cost():
    probe = {"probe_id": "p", "question": "fact?", "expected_answer": "fact",
             "conversation": "synthetic", "split_turn": 80}
    row = {**probe, "cutoff_turn": 80, "answer_verdict": "valid", "reason": "source review",
           "reviewed_gold_turns": [1], "reviewed_through_turn": 80,
           "recent_only_answerable": False, "knowledge_scope": "private_history",
           "task_types": ["episodic_lookup"]}
    packet = {"kind": "native_checkpoint_source_review", "inputs": [1, 2, 3, 4],
              "recent_window_turns": 40, "records": [row]}
    assert validate_packet(packet, {"p": probe}, [1, 2, 3, 4, 5], 40, native=True) == 1
    row["reviewed_gold_turns"] = [70]
    with pytest.raises(ValueError, match="recent-history"):
        validate_packet(packet, {"p": probe}, [1, 2, 3, 4, 5], 40, native=True)
    row["reviewed_gold_turns"] = [1]
    row["question"] = "a changed question"
    with pytest.raises(ValueError, match="frozen catalog"):
        validate_packet(packet, {"p": probe}, [1, 2, 3, 4, 5], 40, native=True)


@pytest.mark.parametrize("native", [False, True])
def test_reviewed_key_is_checked_without_rewriting_catalog(native):
    probe = {"probe_id": "p", "question": "What did I remember?", "expected_answer": "catalog key",
             "conversation": "synthetic", "split_turn": 80, "gold_turns": [1]}
    row = {**probe, "cutoff_turn": 80, "verdict": "valid", "answer_verdict": "valid",
           "reviewed_gold_turns": [1], "reason": "Original bounds are tentative.",
           "reviewed_through_turn": 80, "recent_only_answerable": False,
           "knowledge_scope": "private_history", "task_types": ["episodic_lookup"],
           "reviewed_expected_answer": "Required: preserve the uncertain bound."}
    packet = {"kind": "native_checkpoint_source_review" if native else "longterm_probe_label_review",
              "inputs": [1, 2, 3, 4] if native else [1, 2, 3, 4, 5],
              "recent_window_turns": 40, "records": [row]}
    assert validate_packet(packet, {"p": probe}, [1, 2, 3, 4, 5], 40, native=native) == 1
    row["reviewed_expected_answer"] = " "
    with pytest.raises(ValueError, match="nonempty"):
        validate_packet(packet, {"p": probe}, [1, 2, 3, 4, 5], 40, native=native)
    row["reviewed_expected_answer"] = None
    assert validate_packet(packet, {"p": probe}, [1, 2, 3, 4, 5], 40, native=native) == 1
    row["expected_answer"] = "an edited catalog"
    with pytest.raises(ValueError, match="frozen catalog"):
        validate_packet(packet, {"p": probe}, [1, 2, 3, 4, 5], 40, native=native)
