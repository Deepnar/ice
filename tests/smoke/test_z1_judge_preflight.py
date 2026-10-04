"""Paired answer judgements require complete, matched source and models."""

import json
import sys
from pathlib import Path

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


def test_natural_repeat_keeps_cutoffs_and_reports_family_on_resume(tmp_path, monkeypatch):
    import tempfile
    from scripts.z1.answer_as_of import LOGS
    calls = []
    monkeypatch.setattr(judge_answers, "judge_one", lambda *a, **k: calls.append(a) or {
        "verdict": "TIE", "reason": "equivalent", "A_grade": "correct", "B_grade": "correct"})
    rows = [{**_record(), "probe_id": f"p{cutoff}", "split_turn": cutoff,
             "expected_answer": "A habit remembered at this cutoff",
             "label_review": {"question_family_id": "one_reviewed_habit"}}
            for cutoff in (80, 160)]
    arms = (_v3_arm("full", rows), _v3_arm("vector_only", rows))
    with tempfile.TemporaryDirectory(prefix="z1-repeat-family-", dir=LOGS) as root:
        out = str(Path(root) / "judge.json")
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out]) == 0
        result = json.loads(Path(out).read_text())
        assert [r["split_turn"] for r in result["results"]] == [80, 160]
        families = result["question_families"]
        assert families["occurrences"] == 2 and families["declared_families"] == 1
        assert families["occurrences_per_declared_family"] == {"one_reviewed_habit": 2}
        assert families["independent_sample_count"] is None
        assert len(calls) == 4
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out, "--resume"]) == 0
        assert len(calls) == 4
        assert json.loads(Path(out).read_text())["question_families"] == families


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


def _run_v3(tmp_path, monkeypatch, left, right, extra=()):
    paths = []
    for index, arm in enumerate((left, right)):
        path = tmp_path / f"arm-{index}.json"
        path.write_text(json.dumps(arm))
        paths.append(path)
    monkeypatch.setattr(judge_answers, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["judge_answers.py", "--a", str(paths[0]),
                                      "--b", str(paths[1]), *extra])
    return judge_answers.main()


def test_reviewed_strata_keep_errors_unknowns_and_overlapping_tasks_visible():
    review = {"knowledge_scope": "private_history", "task_types": ["temporal", "knowledge_update"],
              "question_family_id": "one_fact"}
    rows = [{"label_review": review, "winner": "full", "arm_a_grade": "correct",
             "arm_b_grade": "partial", "arm_a_prompt_tokens_est": 300,
             "arm_b_prompt_tokens_est": 200},
            {"label_review": review, "winner": "ERROR", "arm_a_grade": None,
             "arm_b_grade": None},
            {"winner": "UNCERTAIN", "arm_a_grade": "uncertain", "arm_b_grade": "incorrect"}]
    strata = judge_answers.reviewed_outcome_strata(rows)
    private = strata["knowledge_scope"]["private_history"]
    assert private["occurrences"] == 2 and private["judge_errors"] == 1
    assert private["absolute_grades"]["arm_a"]["ungraded"] == 1
    assert private["paired_correctness"]["a_only_correct"] == 1
    assert private["paired_correctness"]["unresolved"] == 1
    assert private["paired_prompt_cost"]["pairs_with_estimates"] == 1
    assert private["question_families"]["occurrences_per_declared_family"] == {"one_fact": 2}
    assert strata["knowledge_scope"]["public_knowledge"]["status"] == "unrepresented"
    assert strata["knowledge_scope"]["unreviewed"]["paired_correctness"]["unresolved"] == 1
    assert strata["task_types"]["temporal"]["occurrences"] == 2
    assert strata["task_types"]["knowledge_update"]["occurrences"] == 2
    assert strata["task_types"]["negative"]["occurrences"] == 0
    assert strata["task_groups_overlap"] and strata["independent_sample_count"] is None


def test_campaign_strata_expose_private_failure_hidden_by_public_controls_and_resume(tmp_path, monkeypatch):
    import tempfile
    from scripts.z1.answer_as_of import LOGS
    calls = []

    def grade(question, _source, first, second, **_kwargs):
        calls.append(question)
        if question == "A private remembered fact?":
            a_correct = first == "right private answer"
            return {"verdict": "A" if a_correct else "B", "reason": "more_grounded",
                    "A_grade": "correct" if a_correct else "incorrect",
                    "B_grade": "incorrect" if a_correct else "correct"}
        return {"verdict": "TIE", "reason": "equivalent", "A_grade": "correct", "B_grade": "correct"}

    monkeypatch.setattr(judge_answers, "judge_one", grade)
    public = [{**_record(), "probe_id": f"public-{n}", "split_turn": 80 + n,
               "expected_answer": "A public fact", "probe_type": "episodic",
               "label_review": {"knowledge_scope": "public_knowledge", "task_types": ["episodic_lookup"]},
               "prompt_tokens": 100} for n in range(4)]
    private = {**_record(), "probe_id": "private", "split_turn": 100,
               "question": "A private remembered fact?", "expected_answer": "private fact",
               "probe_type": "episodic", "answer": "wrong private answer", "prompt_tokens": 800,
               "label_review": {"knowledge_scope": "private_history", "task_types": ["procedural"]}}
    left = public + [private]
    right = [{**r, "answer": "right private answer" if r["probe_id"] == "private" else r["answer"],
              "prompt_tokens": 80} for r in left]
    with tempfile.TemporaryDirectory(prefix="z1-reviewed-strata-", dir=LOGS) as root:
        out = str(Path(root) / "judge.json")
        arms = (_v3_arm("full", left), _v3_arm("vector_only", right))
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out]) == 0
        result = json.loads(Path(out).read_text())
        assert result["absolute_by_type"]["episodic"]["arm_a"]["correct"] == 4
        strata = result["reviewed_outcome_strata"]
        scopes = strata["knowledge_scope"]
        assert scopes["public_knowledge"]["paired_correctness"]["both_correct"] == 4
        assert scopes["private_history"]["paired_correctness"]["b_only_correct"] == 1
        assert scopes["private_history"]["absolute_grades"]["arm_a"]["correct"] == 0
        assert scopes["public_knowledge"]["paired_prompt_cost"]["median_a_minus_b_tokens_est"] == 20
        assert scopes["private_history"]["paired_prompt_cost"]["median_a_minus_b_tokens_est"] == 720
        assert strata["task_types"]["knowledge_update"]["status"] == "unrepresented"
        partial = json.loads(Path(out).with_suffix(".partial.json").read_text())
        assert partial["reviewed_outcome_strata"] == strata
        assert len(calls) == 10
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out, "--resume"]) == 0
        assert len(calls) == 10
        assert json.loads(Path(out).read_text())["reviewed_outcome_strata"] == strata


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


def test_judge_resume_reuses_first_order_and_complete_pairs(tmp_path, monkeypatch):
    import tempfile
    from scripts.z1.answer_as_of import LOGS
    good = {"verdict": "TIE", "reason": "equivalent",
            "A_grade": "correct", "B_grade": "correct"}
    row = {**_record(), "probe_id": "p1", "split_turn": 80, "expected_answer": "fact"}
    arms = (_v3_arm("full", [row]), _v3_arm("vector_only", [row]))
    calls = []
    def interrupted(*args, **_kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return good
    with tempfile.TemporaryDirectory(prefix="z1-judge-resume-", dir=LOGS) as root:
        out = str(Path(root) / "judge.json")
        monkeypatch.setattr(judge_answers, "judge_one", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out])
        seen = []
        monkeypatch.setattr(judge_answers, "judge_one", lambda *a, **k: seen.append(a) or good)
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out, "--resume"]) == 0
        assert len(seen) == 1
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out, "--resume"]) == 0
        assert len(seen) == 1


def test_judge_resume_preserves_error_attempt_and_stops_outage(tmp_path, monkeypatch):
    import tempfile
    from scripts.z1.answer_as_of import LOGS
    rows = [{**_record(), "probe_id": f"p{i}", "split_turn": 80, "expected_answer": "fact"}
            for i in range(2)]
    arms = (_v3_arm("full", rows), _v3_arm("vector_only", rows))
    calls = []
    monkeypatch.setattr(judge_answers, "judge_one", lambda *a, **k: calls.append(a) or
                        {"verdict": "ERROR", "reason": "api"})
    with tempfile.TemporaryDirectory(prefix="z1-judge-retry-", dir=LOGS) as root:
        out = str(Path(root) / "judge.json")
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out]) == 1
        assert len(calls) == 1
        good = {"verdict": "TIE", "reason": "equivalent",
                "A_grade": "correct", "B_grade": "correct"}
        monkeypatch.setattr(judge_answers, "judge_one", lambda *a, **k: good)
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out, "--resume"]) == 0
        result = json.loads(Path(out).read_text())
        assert result["complete"] and len(result["results"]) == 2
        assert result["failed_attempts"][0]["reason"] == "api"


def test_failed_second_order_resume_keeps_successful_first(tmp_path, monkeypatch):
    import tempfile
    from scripts.z1.answer_as_of import LOGS
    row = {**_record(), "probe_id": "p1", "split_turn": 80, "expected_answer": "fact"}
    arms = (_v3_arm("full", [row]), _v3_arm("vector_only", [row]))
    good = {"verdict": "TIE", "reason": "equivalent", "A_grade": "correct", "B_grade": "correct"}
    responses = iter([good, {"verdict": "ERROR", "reason": "api_http_429"}])
    monkeypatch.setattr(judge_answers, "judge_one", lambda *_a, **_k: next(responses))
    with tempfile.TemporaryDirectory(prefix="z1-second-order-retry-", dir=LOGS) as root:
        out = str(Path(root) / "judge.json")
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out]) == 1
        seen = []
        monkeypatch.setattr(judge_answers, "judge_one", lambda *a, **_k: seen.append(a) or good)
        assert _run_v3(tmp_path, monkeypatch, *arms, extra=["--out", out, "--resume"]) == 0
        assert len(seen) == 1
        result = json.loads(Path(out).read_text())
        assert result["failed_attempts"][0]["reason"] == "api_http_429"
        assert result["results"][0]["order_verdicts"][0]["raw"] == good


def test_v3_quota_error_makes_one_transport_attempt(monkeypatch):
    import io
    import urllib.error
    monkeypatch.setattr(judge_answers, "_env", lambda key: {
        "PROBE_API_KEY": "synthetic-test-credential", "PROBE_API_BASE_URL": "https://test.invalid/v1",
        "PROBE_MODEL": "synthetic-judge"}.get(key))
    calls = []
    def quota(request, **_kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 429, "quota", {},
            io.BytesIO(b'{"error":{"type":"insufficient_quota"}}'))
    monkeypatch.setattr(judge_answers.urllib.request, "urlopen", quota)
    monkeypatch.setattr(judge_answers.time, "sleep", lambda _: pytest.fail("quota retried"))
    result = judge_answers.judge_one("question", "source", "a", "b", expected_answer="fact")
    assert len(calls) == 1 and result["reason"] == "api_http_429"


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", "stop", None])
def test_judge_rejects_truncation_even_with_parseable_verdict(monkeypatch, finish):
    import io
    monkeypatch.setattr(judge_answers, "_env", lambda key: {
        "PROBE_API_KEY": "fixture", "PROBE_API_BASE_URL": "https://test.invalid/v1",
        "PROBE_MODEL": "fixture"}.get(key))
    verdict = {"verdict": "TIE", "reason": "equivalent", "A_grade": "correct", "B_grade": "correct"}
    payload = {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(verdict)}}]}
    calls = []

    def response(*args, **_kwargs):
        calls.append(args)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(judge_answers.urllib.request, "urlopen", response)
    result = judge_answers.judge_one("question", "source", "a", "b", expected_answer="fact")
    assert len(calls) == 1
    if finish in {"stop", None}:
        assert result["verdict"] == "TIE"
    else:
        assert result["verdict"] == "ERROR" and result["reason"] == "incomplete_completion"


@pytest.mark.parametrize("mode", ["isolated", "outage", "quota", "answer_failed", "interrupt"])
def test_continue_cloud_faults_and_resume_preserve_denominators(tmp_path, monkeypatch, mode):
    import tempfile
    from scripts.z1.answer_as_of import LOGS
    from scripts.z1.cloud_recovery import POLICY
    rows = [{**_record(), "probe_id": f"p{i}", "split_turn": 80, "expected_answer": "fact"}
            for i in range(4)]
    left, right = _v3_arm("full", rows), _v3_arm("vector_only", [dict(r) for r in rows])
    good = {"verdict": "TIE", "reason": "equivalent", "A_grade": "correct", "B_grade": "correct"}
    calls = []
    def response(*a, **k):
        calls.append(a)
        if mode == "quota":
            return {"verdict": "ERROR", "reason": "api_http_429", "operator_required": True}
        if mode == "outage" or mode == "isolated" and len(calls) in (2, 3):
            return {"verdict": "ERROR", "reason": "api_http_503"}
        if mode == "interrupt" and len(calls) == 2:
            raise KeyboardInterrupt()
        return good
    if mode == "answer_failed":
        left["records"] = [dict(r) for r in rows]
        left.update(complete=False, processing_complete=True, cloud_failure_policy="continue", cloud_recovery_policy=POLICY)
        left["records"][0].update(answer="", error="No complete answer", error_type="CloudCompletionError",
            cloud_attempts=2, fault_disposition="degraded_final")
    monkeypatch.setattr(judge_answers, "judge_one", response)
    monkeypatch.setattr(judge_answers.time, "sleep", lambda _: None)
    with tempfile.TemporaryDirectory(prefix="z1-cloud-fault-control-", dir=LOGS) as root:
        out = str(Path(root) / "judge.json")
        extra = ["--out", out, "--failure-policy", "continue"]
        if mode == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                _run_v3(tmp_path, monkeypatch, left, right, extra=extra)
        else:
            assert _run_v3(tmp_path, monkeypatch, left, right, extra=extra) == int(mode in {"outage", "quota"})
            result = json.loads(Path(out).read_text())
            if mode in {"isolated", "answer_failed"}:
                assert result["processing_complete"] and not result["complete"]
                assert result["cloud_errors"] == 1 and len(result["results"]) == 4
                assert result["results"][0]["arm_a_grade"] is None
                assert len(calls) == (9 if mode == "isolated" else 6)
            elif mode == "outage":
                assert result["operator_required"] and len(result["results"]) == 3 and len(calls) == 6
            else:
                assert len(calls) == 1 and not result["processing_complete"]
        before = len(calls)
        monkeypatch.setattr(judge_answers, "judge_one", lambda *a, **k: calls.append(a) or good)
        assert _run_v3(tmp_path, monkeypatch, left, right, extra=[*extra, "--resume"]) == 0
        added = len(calls) - before
        assert added == {"isolated": 0, "answer_failed": 0, "outage": 2, "quota": 8, "interrupt": 7}[mode]
        result = json.loads(Path(out).read_text())
        assert result["processing_complete"] and not result["operator_required"]
        assert _run_v3(tmp_path, monkeypatch, left, right, extra=[*extra, "--resume"]) == 0
        assert len(calls) == before + added
