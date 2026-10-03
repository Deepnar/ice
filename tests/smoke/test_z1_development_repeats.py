"""Reviewed recent→old questions must survive the actual answering/report path."""
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.lme.cloud_provider import TextGenerator
from scripts.z1 import answer_as_of, judge_answers, run_v3_campaign, seed_v3
from scripts.z1.development_repeats import (POLICY, admit_repeats, packet_hash,
                                           repeat_probes, retention_comparison)


def example():
    base = {"probe_id": "native", "conversation": "example", "split_turn": 20,
            "question": "Which port did I choose?", "expected_answer": "catalog port",
            "cutoff_kind": "native_unlabeled_checkpoint", "probe_type": "native", "gold_turns": []}
    review = {"gold_turns": [10], "reviewer": "fixture reviewer",
              "review_scope": "complete_history_through_cutoff", "knowledge_scope": "private_history",
              "task_types": ["episodic_lookup"], "reviewed_expected_answer": "ORACLE_KEY_CANARY port9123",
              "reason": "Both full histories reviewed, no later changes/repetition."}
    packet = {"version": "v3", "kind": "development_repeat_review", "inputs": [1, 2, 3, 4, 5],
              "recent_window_turns": 40, "families": [{"base_probe_id": "native",
                  "question_family_id": "port", "expected_answer_unchanged": True,
                  "before": {**review, "cutoff_turn": 20, "reviewed_through_turn": 20, "recent_only_answerable": True},
                  "after": {**review, "cutoff_turn": 80, "reviewed_through_turn": 80, "recent_only_answerable": False}}]}
    conversations = {"example": [{}] * 80}
    schedule = {("example", 20): [base], ("example", 80): []}
    return packet, conversations, schedule


def test_repeat_plan_has_distinct_stable_ids_and_preserves_native_question():
    packet, conversations, schedule = example()
    before = copy.deepcopy(schedule)
    probes = repeat_probes(packet, conversations, schedule, [1, 2, 3, 4, 5], 40)
    assert probes == repeat_probes(packet, conversations, schedule, [1, 2, 3, 4, 5], 40)
    assert len({p["probe_id"] for p in probes}) == 2
    assert [p["split_turn"] for p in probes] == [20, 80]
    assert all(p["question"] == schedule[("example", 20)][0]["question"] for p in probes)
    assert [p["review"]["recent_only_answerable"] for p in probes] == [True, False]
    assert schedule == before


@pytest.mark.parametrize("defect", ["inputs", "recent", "cutoff", "review", "key", "gold", "mixed_age", "unchanged", "oversize"])
def test_repeat_admission_rejects_unreviewed_or_confounded_pairs(defect):
    packet, conversations, schedule = example()
    family = packet["families"][0]
    if defect == "inputs": packet["inputs"] = []
    if defect == "recent": family["after"]["recent_only_answerable"] = True
    if defect == "cutoff": family["after"]["cutoff_turn"] = 79
    if defect == "review": family["after"]["reviewed_through_turn"] = 20
    if defect == "key": family["after"]["reviewed_expected_answer"] = "changed"
    if defect == "gold": family["after"]["gold_turns"] = [81]
    if defect == "mixed_age":
        family["before"]["gold_turns"] = family["after"]["gold_turns"] = [1, 10]
        family["before"]["cutoff_turn"] = family["before"]["reviewed_through_turn"] = 45
        schedule[("example", 20)][0]["split_turn"] = 45
    if defect == "unchanged": family["expected_answer_unchanged"] = False
    if defect == "oversize": packet["families"] *= 5
    with pytest.raises(ValueError):
        repeat_probes(packet, conversations, schedule, [1, 2, 3, 4, 5], 40)


def test_retention_summary_keeps_partial_missing_uncertain_and_error_outcomes():
    meta = {"policy": POLICY, "question_family_id": "port", "source_age_turns": 10}
    rows = [{"probe_id": "early", "split_turn": 20, "development_repeat": {**meta, "phase": "before"},
             "winner": "full", "arm_a_grade": "correct", "arm_b_grade": "partial",
             "arm_a_prompt_tokens_est": 100, "arm_b_prompt_tokens_est": 80},
            {"probe_id": "late", "split_turn": 80, "development_repeat": {**meta, "phase": "after", "source_age_turns": 70},
             "winner": "vector_only", "arm_a_grade": "incorrect", "arm_b_grade": "correct",
             "arm_a_prompt_tokens_est": 140, "arm_b_prompt_tokens_est": 90}]
    report = retention_comparison(rows)
    assert report["paired_families"] == 1 and report["independent_sample_count"] is None
    assert report["summary"]["arm_a"]["correct_to_not_correct"] == 1
    assert report["summary"]["arm_b"]["not_correct_to_correct"] == 1
    assert report["pairs"][0]["arms"]["arm_a"]["after_minus_before_prompt_tokens_est"] == 40
    assert retention_comparison(rows[:1])["pairs"][0]["missing_phases"] == ["after"]
    for grade, winner in (("uncertain", "UNCERTAIN"), ("correct", "ERROR")):
        rows[1].update(arm_a_grade=grade, winner=winner)
        assert retention_comparison(rows)["summary"]["arm_a"]["grade_movements"] == {"unresolved": 1}


def test_repeat_freeze_answer_judge_report_and_interrupted_resume(monkeypatch):
    packet, conversations, schedule = example()
    plans = repeat_probes(packet, conversations, schedule, [1, 2, 3, 4, 5], 40)
    monkeypatch.setattr(seed_v3, "load_plan", lambda **_k: (conversations, {("example", p["split_turn"]): [p] for p in plans}, [], []))
    monkeypatch.setattr(answer_as_of, "EXPECTED", {"example": 80})
    monkeypatch.setattr(answer_as_of, "SOURCE_TIMESTAMP_PROVENANCE", {"example": "synthetic_raw_import"})
    answer_calls, judge_calls = [], []
    def respond(**kwargs):
        answer_calls.append(kwargs)
        if len(answer_calls) == 2: raise RuntimeError("intentional provider interruption")
        text = json.dumps(kwargs["input"])
        return {"output_text": "wrong" if "port1111" in text else "port9123", "usage": {"input_tokens": 30}}
    client = SimpleNamespace(responses=SimpleNamespace(create=respond))
    monkeypatch.setattr(answer_as_of, "TextGenerator", lambda profile: TextGenerator(profile, client=client))
    monkeypatch.setenv("PROBE_API_BASE_URL", "https://test.invalid/v1")
    def judge(_question, _source, a, b, **_kwargs):
        judge_calls.append((a, b))
        if len(judge_calls) == 2: return {"verdict": "ERROR", "reason": "intentional second-order interruption"}
        ag, bg = ("incorrect" if a == "wrong" else "correct"), ("incorrect" if b == "wrong" else "correct")
        return {"verdict": "TIE" if ag == bg else "A" if ag == "correct" else "B",
                "reason": "equivalent" if ag == bg else "more_grounded", "A_grade": ag, "B_grade": bg}
    monkeypatch.setattr(judge_answers, "judge_one", judge)
    with tempfile.TemporaryDirectory(prefix="z1-development-repeat-", dir=answer_as_of.LOGS) as root:
        root = Path(root)
        packet_path = root / "development-repeat-review.json"
        packet_path.write_text(json.dumps(packet))
        header = {"event": "run", "meta": {"inputs": [1, 2, 3, 4, 5],
            "settings_resolved": {"recent_window_max_turns": 40}, "extra": {"version": "v3",
                "development_repeat_policy": POLICY, "development_repeat_review_sha256": packet_hash(packet_path),
                "timestamp_provenance_by_conversation": {"example": "synthetic_raw_import"},
                "planned_probes": [[p["probe_id"], "example", p["split_turn"]] for p in plans]}}}
        rows = [header]
        for turn in range(1, 81):
            stamp = f"2025-01-01T{(turn-1)//60:02d}:{(turn-1)%60:02d}:00+00:00"
            rows += [{"event": "turn", "conversation": "example", "turn": turn, "before_turns": turn - 1, "recorded_at": stamp},
                     {"event": "written", "conversation": "example", "turn": turn, "episodic_id": f"row{turn}", "batch_id": f"batch{turn}"}]
            for plan in plans:
                if plan["split_turn"] != turn: continue
                def stage(arm):
                    wrong = turn == 80 and arm == "full"
                    return {"prompt_messages": [{"role": "system", "content": "port1111" if wrong else "port9123"},
                            {"role": "user", "content": plan["question"]}], "prompt_tokens": 140 if turn == 80 else 100,
                            "selected_tokens": 20, "gold_fragment_coverage": {"selected_or_source_note": int(not wrong)},
                            "lineage": {"future_source_ids": []}}
                rows.append({**plan, "type": plan["probe_type"], "event": "as_of_probe", "state_turns": turn,
                    "question_time": stamp, "question_time_provenance": "synthetic_raw_import",
                    "gold_sources": [{"turn": 10, "source_ids": ["row10", "batch10"],
                        "recorded_at": "2025-01-01T00:09:00+00:00", "prompt": "ORACLE_SOURCE_CANARY", "response": "source response"}],
                    "preflight": stage("full"), "controls": {arm: stage(arm) for arm in ("no_codex", "vector_only", "recent_only")}})
        rows.append({"event": "complete", "complete_selected_corpus": True, "turns_by_conversation": {"example": 80}, "as_of_probes": 2, "expected_as_of_probes": 2})
        trace = root / "seed.jsonl"
        trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert len(answer_as_of.load_probes(trace, allow_partial=False)) == 2
        def answers(arm, resume=False):
            monkeypatch.setattr(sys, "argv", ["answer_as_of.py", "--trace", str(trace), "--out", str(root / f"answers-{arm}.json"),
                "--arm", arm, "--development-repeat-review", str(packet_path), *(["--resume"] if resume else [])])
            return answer_as_of.main()
        with pytest.raises(RuntimeError, match="intentional provider"):
            answers("full")
        assert answers("full", True) == 0 and len(answer_calls) == 3
        assert answers("full", True) == 0 and len(answer_calls) == 3
        assert answers("vector_only") == 0 and len(answer_calls) == 5
        assert all("ORACLE" not in json.dumps(call) for call in answer_calls)
        for resume in (False, True, True):
            monkeypatch.setattr(sys, "argv", ["judge_answers.py", "--a", str(root / "answers-full.json"),
                "--b", str(root / "answers-vector_only.json"), "--out", str(root / "judge-full-vs-vector_only.json"),
                *(["--resume"] if resume else [])])
            assert judge_answers.main() == (1 if not resume else 0)
        assert len(judge_calls) == 5
        result = json.loads((root / "judge-full-vs-vector_only.json").read_text())
        comparison = result["retention_comparison"]
        assert comparison["summary"]["arm_a"]["correct_to_not_correct"] == 1
        assert comparison["pairs"][0]["arms"]["arm_a"]["after_minus_before_selected_source_count"] == -1
        assert result["reviewed_outcome_strata"]["development_recent_controls"]["occurrences"] == 1
        assert result["reviewed_outcome_strata"]["knowledge_scope"]["private_history"]["occurrences"] == 1
        partial = json.loads((root / "judge-full-vs-vector_only.partial.json").read_text())
        assert partial["retention_comparison"] == comparison
        monkeypatch.setattr(run_v3_campaign, "status", lambda *_a: {"version": "v3", "stages": {}})
        manual = run_v3_campaign.campaign_report(root, {})
        assert manual["artifacts"]["judge-full-vs-vector_only.json"]["retention_comparison"] == comparison
        with pytest.raises(ValueError, match="pinned replay"):
            admit_repeats(header, rows, plans, "changed packet hash")


def test_manual_entry_point_passes_review_packet_to_seed_and_every_answer_arm(tmp_path, monkeypatch):
    from test_z1_manual_campaign import config, providers
    (tmp_path / "development-repeat-review.json").write_text("reviewed packet fixture")
    monkeypatch.setattr(run_v3_campaign, "status", lambda *_a: {"reviewed_label_candidates_available": True})
    monkeypatch.setattr(run_v3_campaign, "validate_labels", lambda *_a: 4)
    monkeypatch.setattr(run_v3_campaign, "progress_totals", lambda *_a: (1, 4))
    monkeypatch.setattr(run_v3_campaign, "database_environment", lambda *_a: {})
    providers(monkeypatch)
    calls = []
    monkeypatch.setattr(run_v3_campaign.subprocess, "run", lambda argv, **_k: calls.append(argv))
    assert run_v3_campaign.execute(tmp_path, config(), "seed") == 0
    assert run_v3_campaign.execute(tmp_path, config(), "answers") == 0
    assert len(calls) == 5
    assert all(c[c.index("--development-repeat-review") + 1] == str(tmp_path / "development-repeat-review.json") for c in calls)
