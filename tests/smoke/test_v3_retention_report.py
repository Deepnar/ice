"""Independent retention reports reuse actual scorer artifacts without regrading."""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from sqlalchemy.engine import Engine

from experiments.v3_repair import report_retention as retention
from scripts.z1 import score_frozen_answers as scorer


def answer(phase, *, history="history", family="service", value="correct", tokens=100):
    cutoff = 20 if phase == "before" else 80
    return dict(probe_id=phase, conversation=history, split_turn=cutoff,
        question="Which service port did I choose?", expected_answer="7813",
        question_time=f"2026-01-{1 if phase == 'before' else 2:02d}T00:00:00+00:00",
        question_time_provenance="simulated", gold_source_complete=True,
        gold_turn_text="User: use service port 7813.", gold_turns=[10],
        label_review=dict(knowledge_scope="private_history", task_types=["episodic_lookup"],
                          recent_only_answerable=phase == "before", question_family_id=family),
        development_repeat=dict(policy=retention.REPEAT_POLICY, question_family_id=family,
            phase=phase, base_probe_id="original", recent_window_turns=40, source_age_turns=cutoff - 10),
        answer=value, error=None, prompt_tokens=tokens)


@pytest.fixture
def native(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("retention qualification must not access a provider or database")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(Engine, "connect", forbidden)
    with tempfile.TemporaryDirectory(prefix="v3-retention-report-", dir=retention.ROOT / "logs") as directory:
        root = Path(directory)
        calls = []
        monkeypatch.setattr(scorer, "judge_profile", lambda: SimpleNamespace(metadata=lambda: {"model": "controlled"}))
        def completion(profile, messages, key):
            calls.append(key)
            value = json.loads(messages[1]["content"])["answer"]
            return SimpleNamespace(text=json.dumps(dict(grade=value, reason="Authored test grade", basis="source")),
                completion_status="completed", completion_details=dict(completion_status="completed"))
        monkeypatch.setattr(scorer, "profile_completion", completion)

        def run(data):
            paths = []
            for arm, rows in data.items():
                path = root / (arm + ".json")
                path.write_text(json.dumps(dict(version="v3", tag=arm, processing_complete=True, records=rows)))
                paths.append(path)
            monkeypatch.setattr(sys, "argv", ["score_frozen_answers.py", "--answers", *map(str, paths),
                "--cache-dir", str(root / "scores"), "--run", "--resume"])
            assert scorer.main() == 0
            report, = (root / "scores/reports").glob("*.json")
            return paths, report
        yield root, calls, run


def snapshot(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*.json")}


def test_actual_scorer_and_cli_resolved_movement_is_reused_without_calls(native, monkeypatch):
    root, calls, run = native
    paths, report = run({"full": [answer("before"), answer("after", value="partial", tokens=150)],
                         "vector": [answer("before", value="incorrect"), answer("after", tokens=90)]})
    before = snapshot(root)
    count = len(calls)
    out = root / "retention.json"
    monkeypatch.setattr(sys, "argv", ["report_retention.py", "--answers", *map(str, paths),
        "--score-report", str(report), "--out", str(out)])
    assert retention.main() == 0
    result = json.loads(out.read_text())
    groups = result["retention"]["by_arm"]
    assert groups["full"]["movements"] == {"worsened": 1}
    assert groups["full"]["correct_to_not_correct"] == 1
    assert groups["full"]["retained_correct_fraction"] == 0
    assert groups["full"]["median_after_minus_before_prompt_tokens"] == 50
    assert groups["vector"]["movements"] == {"improved": 1}
    assert groups["vector"]["retained_correct_fraction"] is None
    stamp = (out.read_bytes(), out.stat().st_mtime_ns)
    assert retention.main() == 0 and len(calls) == count
    assert (out.read_bytes(), out.stat().st_mtime_ns) == stamp
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == data for p, data in before.items())
    assert result["retention"]["independent_sample_count"] is None


@pytest.mark.parametrize("change", ["uncertain", "missing", "answer_failure", "policy", "age", "question",
                                    "answer", "source", "source_text", "family", "window", "recent"])
def test_incomplete_or_changed_pair_cannot_count_as_forgetting(native, change):
    _, _, run = native
    before, after = answer("before"), answer("after", value="incorrect")
    rows = [before, after]
    if change == "uncertain": after["answer"] = "uncertain"
    if change == "missing": rows.pop()
    if change == "answer_failure":
        after.update(answer="", error="incomplete_completion", error_type="CloudCompletionError",
                     cloud_attempts=2, fault_disposition="degraded_final")
    if change == "policy": after["development_repeat"]["policy"] = "unknown"
    if change == "age": after["development_repeat"]["source_age_turns"] = 39
    if change == "question": after["question"] = "What is a port?"
    if change == "answer": after["expected_answer"] = "9123"
    if change == "source": after["gold_turns"] = [11]
    if change == "source_text": after["gold_turn_text"] = "User: use service port 9123."
    if change == "family": after["label_review"]["question_family_id"] = "wrong"
    if change == "window": after["development_repeat"]["recent_window_turns"] = 60
    if change == "recent": after["label_review"]["recent_only_answerable"] = True
    paths, report = run({"full": rows})
    result = retention.derive(paths, report)["retention"]
    assert result["by_arm"]["full"]["resolved_families"] == 0
    assert result["by_arm"]["full"]["correct_to_not_correct"] == 0
    assert result["by_arm"]["full"]["retained_correct_fraction"] is None
    assert result["pairs"][0]["correct_to_not_correct"] is None


def test_histories_with_the_same_family_remain_separate(native):
    _, _, run = native
    paths, report = run({"full": [answer(p, history=h) for p in ("before", "after") for h in ("B", "C")]})
    result = retention.derive(paths, report)["retention"]
    assert len(result["pairs"]) == 2
    assert {p["history"] for p in result["pairs"]} == {"B", "C"}
    assert result["by_arm"]["full"]["resolved_families"] == 2


def test_duplicate_phase_refuses_instead_of_choosing_the_preferred_outcome(native):
    _, _, run = native
    duplicate = answer("after")
    duplicate["probe_id"] = "other-after"
    paths, report = run({"full": [answer("before"), answer("after"), duplicate]})
    with pytest.raises(ValueError, match="duplicate repeat phase"):
        retention.derive(paths, report)


def test_no_repeats_and_incomplete_metadata_do_not_imply_perfect_retention(native):
    _, _, run = native
    ordinary, incomplete = answer("after"), answer("before")
    ordinary.pop("development_repeat")
    incomplete["development_repeat"] = dict(phase="before")
    paths, report = run({"full": [ordinary, incomplete]})
    result = retention.derive(paths, report)["retention"]
    assert result["pairs"] == []
    assert result["by_arm"]["full"]["incomplete_metadata_occurrences"] == 1
    assert result["by_arm"]["full"]["retained_correct_fraction"] is None


@pytest.mark.parametrize("target", ["answers", "report", "plan", "cache"])
def test_changed_native_artifact_refuses_before_report_write(native, monkeypatch, target):
    root, _, run = native
    paths, report = run({"full": [answer("before"), answer("after")]})
    candidates = {"answers": paths[0], "report": report,
                  "cache": next((root / "scores/cache").glob("*.json")),
                  "plan": next((root / "scores/plans").glob("*.json"))}
    item = candidates[target]
    data = json.loads(item.read_text())
    if target == "answers": data["records"][0]["answer"] = "partial"
    if target == "report": data["records"][0]["grade"]["grade"] = "partial"
    if target == "plan": data["inputs"]["full"] = "modified"
    if target == "cache": data["attempts"][0]["raw"]["value"]["grade"] = "partial"
    item.write_text(json.dumps(data))
    out = root / "retention.json"
    monkeypatch.setattr(sys, "argv", ["report_retention.py", "--answers", *map(str, paths),
        "--score-report", str(report), "--out", str(out)])
    with pytest.raises(ValueError): retention.main()
    assert not out.exists()


def test_no_available_answers_still_reports_unresolved_without_judge(native):
    _, calls, run = native
    rows = [answer("before"), answer("after")]
    for row in rows:
        row.update(answer="", error="incomplete_completion", error_type="CloudCompletionError",
                   cloud_attempts=2, fault_disposition="degraded_final")
    paths, report = run({"full": rows})
    result = retention.derive(paths, report)
    assert not calls and result["retention"]["by_arm"]["full"]["unresolved_families"] == 1


def test_interrupted_report_publication_resumes_without_changing_native_grades(native, monkeypatch):
    root, calls, run = native
    paths, report = run({"full": [answer("before"), answer("after")]})
    before, count = snapshot(root), len(calls)
    out = root / "retention.json"
    monkeypatch.setattr(sys, "argv", ["report_retention.py", "--answers", *map(str, paths),
        "--score-report", str(report), "--out", str(out)])
    writer = retention.write_or_verify
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(retention, "write_or_verify", interrupt)
    with pytest.raises(KeyboardInterrupt): retention.main()
    assert not out.exists()
    monkeypatch.setattr(retention, "write_or_verify", writer)
    assert retention.main() == 0
    assert len(calls) == count
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == data for p, data in before.items())


def test_actual_reviewed_repeat_generator_reaches_independent_report(native):
    from scripts.z1.development_repeats import repeat_probes
    _, _, run = native
    base = dict(answer("before"), cutoff_kind="native_unlabeled_checkpoint")
    review = dict(gold_turns=[10], reviewer="authored fixture", review_scope="complete_history_through_cutoff",
        knowledge_scope="private_history", task_types=["episodic_lookup"], reviewed_expected_answer="7813",
        reason="Authored unchanged fact in both histories.")
    packet = dict(version="v3", kind="development_repeat_review", inputs=[1, 2, 3, 4, 5], recent_window_turns=40,
        families=[dict(base_probe_id="before", question_family_id="service", expected_answer_unchanged=True,
            before=dict(review, cutoff_turn=20, reviewed_through_turn=20, recent_only_answerable=True),
            after=dict(review, cutoff_turn=80, reviewed_through_turn=80, recent_only_answerable=False))])
    probes = repeat_probes(packet, {"history": [{}] * 80}, {("history", 20): [base], ("history", 80): []},
                           [1, 2, 3, 4, 5], 40)
    for p in probes:
        p["label_review"] = p["review"]
    paths, report = run({"full": probes})
    result = retention.derive(paths, report)["retention"]
    assert result["by_arm"]["full"]["resolved_families"] == 1
    assert result["by_arm"]["full"]["retained_correct_fraction"] == 1
