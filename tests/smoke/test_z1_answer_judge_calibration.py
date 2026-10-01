"""Calibration must detect position bias, wrong grades and missing calls."""
from __future__ import annotations

import copy
import json
import sys

import pytest

from scripts.z1 import calibrate_answer_judge as calibration


def test_authored_controls_cover_all_grades_without_claiming_human_labels():
    packet = calibration.authored_controls()
    cases = calibration.validate_packet(packet)
    assert len(cases) == 12
    assert packet["kind"] == "authored_answer_judge_controls"
    assert {c["reference"][arm] for c in cases for arm in ("a_grade", "b_grade")} == {
        "correct", "partial", "incorrect", "uncertain"}
    assert len(next(c["source"] for c in cases if c["case_id"] == "long_tail")) > 12000


def test_swapped_orders_map_grades_and_preferences_to_same_answer():
    case = calibration.authored_controls()["cases"][0]
    ab = calibration.mapped_result(case, "ab", {
        "verdict": "A", "reason": "contradicts_source",
        "A_grade": "correct", "B_grade": "incorrect"})
    ba = calibration.mapped_result(case, "ba", {
        "verdict": "B", "reason": "contradicts_source",
        "A_grade": "incorrect", "B_grade": "correct"})
    summary = calibration.summarize([case], [ab, ba])
    assert summary["complete"] and summary["preference_matches"] == 2
    assert summary["grade_matches"] == summary["grade_denominator"] == 4
    assert summary["order_consistent_cases"] == summary["order_pair_denominator"] == 1


def test_always_first_bias_and_api_errors_cannot_disappear_from_denominators():
    case = calibration.authored_controls()["cases"][0]
    verdict = {"verdict": "A", "reason": "more_grounded",
               "A_grade": "correct", "B_grade": "incorrect"}
    biased = [calibration.mapped_result(case, order, verdict) for order in ("ab", "ba")]
    summary = calibration.summarize([case], biased)
    assert summary["preference_matches"] == 1 and summary["preference_denominator"] == 2
    assert summary["order_consistent_cases"] == 0
    failed = calibration.mapped_result(case, "ba", {"verdict": "ERROR", "reason": "api"})
    assert failed["reason"] == "api"
    summary = calibration.summarize([case], [biased[0], failed])
    assert not summary["complete"] and summary["errors"] == 1
    assert summary["order_pair_denominator"] == 1 and summary["order_comparable_cases"] == 0
    assert summary["grade_denominator"] == 4
    assert sum(r["count"] for r in summary["confusion"] if r["judge"] == "ERROR") == 2


def test_missing_duplicate_or_unknown_order_is_not_complete():
    case = calibration.authored_controls()["cases"][0]
    row = calibration.mapped_result(case, "ab", {
        "verdict": "A", "reason": "more_grounded",
        "A_grade": "correct", "B_grade": "incorrect"})
    assert not calibration.summarize([case], [row])["complete"]
    with pytest.raises(ValueError, match="duplicated"):
        calibration.summarize([case], [row, row])
    with pytest.raises(ValueError, match="unknown"):
        calibration.summarize([case], [{**row, "order": "aa"}])


def test_unreviewed_last_reference_blocks_all_cloud_calls(tmp_path, monkeypatch):
    packet = calibration.authored_controls()
    packet["cases"][-1]["reference"]["reason"] = ""
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet))
    monkeypatch.setattr(calibration, "private_path", lambda p: p)
    monkeypatch.setattr(sys, "argv", ["calibrate", "--packet", str(path),
                                    "--out", str(tmp_path / "out.json")])
    calls = []
    monkeypatch.setattr(calibration.judge_answers, "judge_one", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="references"):
        calibration.main()
    assert not calls


def test_main_uses_campaign_request_both_orders_and_saves_diagnostic(tmp_path, monkeypatch):
    packet = copy.deepcopy(calibration.authored_controls())
    packet["cases"] = packet["cases"][:1]
    path, out = tmp_path / "packet.json", tmp_path / "out.json"
    path.write_text(json.dumps(packet))
    monkeypatch.setattr(calibration, "private_path", lambda p: p)
    monkeypatch.setattr(calibration, "run_meta", lambda **kw: {"test": True})
    monkeypatch.setattr(sys, "argv", ["calibrate", "--packet", str(path), "--out", str(out)])
    calls = []
    def judge(question, source, a, b, *, expected_answer, question_time, session_id):
        calls.append((question, source, a, b, expected_answer, question_time))
        return {"verdict": "A" if a == "7412" else "B", "reason": "contradicts_source",
                "A_grade": "correct" if a == "7412" else "incorrect",
                "B_grade": "correct" if b == "7412" else "incorrect"}
    monkeypatch.setattr(calibration.judge_answers, "judge_one", judge)
    assert calibration.main() == 0
    saved = json.loads(out.read_text())
    assert len(calls) == 2 and calls[0][2:4] == calls[1][2:4][::-1]
    assert calls[0][1] == packet["cases"][0]["source"]
    assert saved["summary"]["complete"]
    assert saved["score_of_record"] is False and saved["qualified"] is False
    assert saved["identity"]["packet_kind"] == "authored_answer_judge_controls"
    monkeypatch.setattr(sys, "argv", ["calibrate", "--packet", str(path), "--out", str(out), "--resume"])
    assert calibration.main() == 0 and len(calls) == 2


def test_repeated_outage_stops_calls_and_preserves_original_error(tmp_path, monkeypatch):
    path, out = tmp_path / "packet.json", tmp_path / "out.json"
    path.write_text(json.dumps(calibration.authored_controls()))
    monkeypatch.setattr(calibration, "private_path", lambda p: p)
    monkeypatch.setattr(calibration, "run_meta", lambda **kw: {"test": True})
    monkeypatch.setattr(sys, "argv", ["calibrate", "--packet", str(path), "--out", str(out)])
    calls = []
    def outage(*args, **kwargs):
        calls.append(args)
        return {"verdict": "ERROR", "reason": "api", "note": "HTTP 400 (MissingSessionID)"}
    monkeypatch.setattr(calibration.judge_answers, "judge_one", outage)
    assert calibration.main() == 1
    saved = json.loads(out.read_text())
    assert len(calls) == len(saved["results"]) == 3
    assert saved["summary"]["planned_calls"] == 24
    assert not saved["summary"]["complete"]
    assert all(r["reason"] == "api" and "MissingSessionID" in r["note"] for r in saved["results"])
