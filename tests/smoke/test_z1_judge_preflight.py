"""Paired answer judgements require complete, matched source and models."""

import json
import sys

import pytest

from scripts.z1 import judge_answers


def _record(conversation="first", *, model="gpt-6-luna", source="full source"):
    return {
        "conversation": conversation, "gold_turns": [1], "question": "Same wording?",
        "gold_source_complete": True, "gold_turn_text": source,
        "answer_model": model, "answer_profile": "opencode-luna6",
        "answer": "supported answer", "error": None,
    }


def _run(tmp_path, monkeypatch, left, right):
    paths = []
    for name, records in (("a", left), ("b", right)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"tag": name, "records": records}))
        paths.append(path)
    monkeypatch.setattr(judge_answers, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["judge_answers.py", "--a", str(paths[0]),
                                      "--b", str(paths[1])])
    return judge_answers.main()


def test_same_question_in_two_conversations_stays_two_pairs(tmp_path, monkeypatch):
    seen = []
    def judge(_question, source, _a, _b):
        seen.append(source)
        return {"verdict": "TIE", "reason": "equivalent"}
    monkeypatch.setattr(judge_answers, "judge_one", judge)
    left = [_record("first", source="source one"),
            _record("second", source="source two")]
    assert _run(tmp_path, monkeypatch, left, list(reversed(left))) == 0
    assert seen == ["source one", "source two"]


def test_mismatched_model_or_incomplete_source_blocks_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_answers, "judge_one",
                        lambda *_args: pytest.fail("judge called before preflight"))
    with pytest.raises(ValueError, match="different answering models"):
        _run(tmp_path, monkeypatch, [_record()], [_record(model="other")])
    with pytest.raises(RuntimeError, match="Complete gold source unavailable"):
        old = _record()
        old.pop("gold_source_complete")
        old["gold_turn_text"] = "prefix only"
        monkeypatch.setattr(judge_answers, "_GOLD_IDX", {})
        _run(tmp_path, monkeypatch, [old], [old])


def test_repeated_v3_question_at_two_checkpoints_has_two_identities():
    first = {**_record(), "probe_id": "checkpoint-1", "split_turn": 80}
    later = {**_record(), "probe_id": "checkpoint-2", "split_turn": 160}
    assert judge_answers.probe_key(first) != judge_answers.probe_key(later)


def test_v3_absolute_grades_follow_arms_after_blind_shuffle(tmp_path, monkeypatch):
    calls = []
    def grade(_question, _source, first, second, *, expected_answer, question_time, session_id):
        calls.append((first, second, expected_answer))
        if len(calls) == 2:
            partial = json.loads(judge_answers.PARTIAL.read_text())
            assert partial["complete"] is False
            assert partial["pending_probe"]["first_order"]["verdict"] in {"A", "B"}
        left_first = first == "left answer"
        return {"verdict": "A" if left_first else "B", "reason": "more_grounded",
                "A_grade": "correct" if left_first else "incorrect",
                "B_grade": "incorrect" if left_first else "correct", "note": "source"}
    monkeypatch.setattr(judge_answers, "judge_one", grade)
    monkeypatch.setattr(judge_answers, "OUT", tmp_path)
    left = {**_record(), "probe_id": "checkpoint-1", "split_turn": 80,
            "expected_answer": "The supported fact", "answer": "left answer"}
    right = {**left, "answer": "right answer"}
    paths = []
    for name, record in (("full", left), ("vector_only", right)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"version": "v3", "tag": name,
                                    "complete": True, "trace_sha256": "trace",
                                    "probe_ids": ["checkpoint-1"],
                                    "label_audit_sha256": {"existing": "labels"},
                                    "records": [record]}))
        paths.append(path)
    monkeypatch.setattr(sys, "argv", ["judge_answers.py", "--a", str(paths[0]),
                                      "--b", str(paths[1])])
    assert judge_answers.main() == 0
    output = next(tmp_path.glob("*_judge.json"))
    judged = json.loads(output.read_text())
    row = judged["results"][0]
    assert calls[0][2] == "The supported fact"
    assert len(calls) == 2 and calls[0][:2] == tuple(reversed(calls[1][:2]))
    assert row["arm_a_grade"] == "correct"
    assert row["arm_b_grade"] == "incorrect"
    assert row["winner"] == "full"
    assert row["relative_order_consistent"] and row["absolute_order_consistent"]
    assert judged["complete"] is True
    assert judged["calibration_status"] == "qualification_pending"
    assert judged["score_of_record"] is False


def test_v3_judge_request_sees_historical_clock_and_complete_source(monkeypatch):
    import io
    observed = []
    headers = []
    result = {"verdict": "TIE", "reason": "equivalent",
              "A_grade": "correct", "B_grade": "correct"}
    def request(req, **_kwargs):
        observed.append(json.loads(req.data))
        headers.append(dict(req.header_items()))
        return io.BytesIO(json.dumps({"choices": [{"message": {
            "content": json.dumps(result)}}]}).encode())
    monkeypatch.setattr(judge_answers, "_env", lambda key: {
        "PROBE_API_KEY": "test-only", "PROBE_API_BASE_URL": "https://opencode.ai/zen/go/v1",
        "PROBE_MODEL": "test-judge"}[key])
    monkeypatch.setattr(judge_answers.urllib.request, "urlopen", request)
    source = "Source turn 1 recorded at 2025-01-01T00:00:00+00:00\nUser: first fact"
    assert judge_answers.judge_one("What did I say last month?", source, "first fact", "first fact",
                                  expected_answer="first fact",
                                  question_time="2025-02-01T00:00:00+00:00") == result
    prompt = observed[0]["messages"][1]["content"]
    assert "HISTORICAL QUESTION TIME\n2025-02-01T00:00:00+00:00" in prompt
    assert source in prompt and "EXPECTED ANSWER\nfirst fact" in prompt
    assert headers[0]["User-agent"] == "ice-research/3.0"
    assert headers[0]["X-opencode-session"].startswith("ice-z1-judge-")


def _v3_arm(name, records, declared=None):
    return {"version": "v3", "tag": name, "complete": True,
            "trace_sha256": "trace", "seed_clock_policy": "historical reader clock",
            "decoding": {"max_output_tokens": 1500, "temperature": 0},
            "probe_ids": declared if declared is not None else [r["probe_id"] for r in records],
            "label_audit_sha256": {"existing": "labels"}, "records": records}


def _run_v3(tmp_path, monkeypatch, left, right):
    paths = []
    for index, arm in enumerate((left, right)):
        path = tmp_path / f"arm-{index}.json"
        path.write_text(json.dumps(arm))
        paths.append(path)
    monkeypatch.setattr(judge_answers, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["judge_answers.py", "--a", str(paths[0]),
                                      "--b", str(paths[1])])
    return judge_answers.main()


def test_v3_missing_declared_pair_or_expected_answer_blocks_all_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_answers, "judge_one",
                        lambda *_args, **_kwargs: pytest.fail("judge called before complete preflight"))
    row = {**_record(), "probe_id": "p1", "split_turn": 80, "expected_answer": "fact"}
    with pytest.raises(ValueError, match="every declared probe"):
        _run_v3(tmp_path, monkeypatch, _v3_arm("full", [row], ["p1", "p2"]),
                _v3_arm("vector_only", [row], ["p1", "p2"]))
    missing = {**row, "probe_id": "p2", "expected_answer": ""}
    with pytest.raises(ValueError, match="reviewed expected answer"):
        _run_v3(tmp_path, monkeypatch, _v3_arm("full", [row, missing]),
                _v3_arm("vector_only", [row, missing]))


@pytest.mark.parametrize("response,reason", [
    ({"verdict": "TIE", "reason": "invented", "A_grade": "correct", "B_grade": "correct"},
     "bad_reason"),
    ({"verdict": "TIE", "reason": "both_failed", "A_grade": "correct", "B_grade": "incorrect"},
     "inconsistent_failure_verdict"),
    ({"verdict": "A", "reason": "more_complete", "A_grade": "incorrect", "B_grade": "incorrect"},
     "inconsistent_failure_verdict"),
    ({"verdict": "TIE", "reason": "equivalent", "A_grade": "correct", "B_grade": "incorrect"},
     "inconsistent_grade_preference"),
])
def test_v3_inconsistent_judgements_are_errors(response, reason):
    result = judge_answers.validate_verdict(response, absolute=True)
    assert result["verdict"] == "ERROR"
    assert result["reason"] == reason
    valid = {"verdict": "TIE", "reason": "both_failed",
             "A_grade": "incorrect", "B_grade": "incorrect"}
    assert judge_answers.validate_verdict(valid, absolute=True) == valid


def test_judge_error_marks_output_incomplete_and_returns_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_answers, "judge_one",
                        lambda *_args, **_kwargs: {"verdict": "ERROR", "reason": "api"})
    row = {**_record(), "probe_id": "p1", "split_turn": 80, "expected_answer": "fact"}
    assert _run_v3(tmp_path, monkeypatch, _v3_arm("full", [row]),
                   _v3_arm("vector_only", [row])) == 1
    result = json.loads(next(tmp_path.glob("*_judge.json")).read_text())
    assert result["complete"] is False
    assert result["results"][0]["winner"] == "ERROR"


def test_v3_position_preference_stays_uncertain_through_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_answers, "judge_one", lambda *_args, **_kwargs: {
        "verdict": "A", "reason": "more_grounded",
        "A_grade": "correct", "B_grade": "correct"})
    row = {**_record(), "probe_id": "p1", "split_turn": 80, "expected_answer": "fact"}
    assert _run_v3(tmp_path, monkeypatch, _v3_arm("full", [row]),
                   _v3_arm("vector_only", [row])) == 0
    result = json.loads(next(tmp_path.glob("*_judge.json")).read_text())
    judged = result["results"][0]
    assert judged["winner"] == "UNCERTAIN" and judged["reason"] == "order_unstable"
    assert judged["arm_a_grade"] == judged["arm_b_grade"] == "correct"
    assert {v["winner"] for v in judged["order_verdicts"]} == {"full", "vector_only"}
    assert result["order_checks"] == {"pairs": 1, "relative_consistent": 0,
                                      "absolute_consistent": 1, "uncertain_preferences": 1}
    assert result["complete"] is True and result["score_of_record"] is False


def test_v3_grade_disagreement_preserves_raw_grades():
    first = {"verdict": "A", "reason": "more_grounded",
             "A_grade": "correct", "B_grade": "incorrect"}
    reverse = {"verdict": "B", "reason": "more_grounded",
               "A_grade": "incorrect", "B_grade": "partial"}
    result = judge_answers.combine_orders(first, reverse, a_is_first=True,
                                          name_a="full", name_b="vector_only")
    assert result["winner"] == "full" and result["relative_order_consistent"]
    assert result["arm_a_grade"] == "uncertain" and result["arm_b_grade"] == "incorrect"
    assert not result["absolute_order_consistent"]
    assert [r["raw"] for r in result["order_verdicts"]] == [first, reverse]


def test_v3_second_order_error_keeps_api_reason_and_incomplete_status(tmp_path, monkeypatch):
    responses = iter([{"verdict": "TIE", "reason": "equivalent",
                       "A_grade": "correct", "B_grade": "correct"},
                      {"verdict": "ERROR", "reason": "api_http_400", "note": "MissingSessionID"}])
    monkeypatch.setattr(judge_answers, "judge_one", lambda *_args, **_kwargs: next(responses))
    row = {**_record(), "probe_id": "p1", "split_turn": 80, "expected_answer": "fact"}
    assert _run_v3(tmp_path, monkeypatch, _v3_arm("full", [row]),
                   _v3_arm("vector_only", [row])) == 1
    result = json.loads(next(tmp_path.glob("*_judge.json")).read_text())
    assert result["complete"] is False
    judged = result["results"][0]
    assert judged["winner"] == "ERROR" and judged["reason"] == "api_http_400"
    assert judged["note"] == "MissingSessionID" and judged["arm_a_grade"] is None
    assert len(judged["order_verdicts"]) == 2
