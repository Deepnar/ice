"""Z2 S12 LongMemEval oracle adapter: pinned data, sessions sorted into separate
conversations, evidence turns tracked, question after the history, upstream yes/no."""

from datetime import datetime, timezone

import pytest

from experiments.z2 import oracle as O


def test_pinned_file_loads_all_500():
    data = O.load_instances()
    assert len(data) == 500 and sum(i["question_id"].endswith("_abs") for i in data) == 30


def test_refuses_a_file_that_differs_from_its_hash(tmp_path):
    bad = tmp_path / "oracle"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="pinned"):
        O.load_instances(bad)


def test_layout_sorts_sessions_and_keeps_every_turn():
    inst = dict(question_id="q1", question="?", question_type="multi-session", answer="a",
                question_date="2023/06/01 (Thu) 10:00",
                haystack_session_ids=["late", "early"], haystack_dates=["2023/05/20 (Sat) 02:21", "2023/05/01 (Mon) 09:00"],
                haystack_sessions=[[dict(role="user", content="u2", has_answer=True), dict(role="assistant", content="a2")],
                                   [dict(role="user", content="u1"), dict(role="assistant", content="a1"),
                                    dict(role="user", content="dangling")]])
    plan = O.layout(inst)
    assert [s["session_id"] for s in plan["sessions"]] == ["early", "late"]
    early, late = plan["conversations"]["lme00"], plan["conversations"]["lme01"]
    assert [r["prompt"] for r in early] == ["u1", "dangling"] and early[1]["response"] == ""
    assert early[0]["timestamp"] == datetime(2023, 5, 1, 9, tzinfo=timezone.utc)
    assert plan["evidence"] == [dict(slug="lme01", turn=1, session_id="late")]
    assert plan["eligible"] and plan["question_time"] > late[-1]["timestamp"]


def test_upstream_verdict_parses_the_first_yes_or_no():
    assert O._yes("Yes.") is True and O._yes("no, it misses the date") is False
    assert O._yes("The response is correct: yes") is True and O._yes("unclear") is None


def test_grade_oracle_grades_answers_side_by_side(tmp_path, monkeypatch):
    """S12 grading: LongMemEval's own yes/no and the Muse grade per answer, incomplete ones ungraded
    (answers 4 at a time since 2026-10-28; S12's ~2,500 grades one at a time were ~10 hours)."""
    import json
    from experiments.z2 import oracle as O
    from experiments.z2 import read_passes as R
    inst = dict(question_id="q1", haystack_sessions=[[dict(role="user", content="my dog is Rex", has_answer=True)]])
    monkeypatch.setattr(O, "load_instances", lambda: [inst])
    ask = dict(id="q1", type="single-session-user", question="dog?", answer="Rex", abstention=False)
    answers = tmp_path / "answers"
    answers.mkdir()
    rows = [dict(prompt_sha256="a", finish="complete", text="Rex"), dict(prompt_sha256="b", finish="incomplete", text="")]
    (answers / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (answers / "index.json").write_text(json.dumps(dict(index={"full|q1": "a", "bare_vector|q1": "b"})))
    for cond in ("full", "bare_vector"):
        R.write_unit(R.unit_path(tmp_path / "units", cond, "q1"), dict(condition=cond, ask=ask))
    results = O.grade_oracle(tmp_path / "units", answers, tmp_path / "out",
                             ask=lambda system, prompt: dict(grade="correct", score=5), ask_text=lambda prompt: "yes")
    assert results["full|q1"]["upstream"] is True and results["full|q1"]["grade"] == "correct"
    assert results["bare_vector|q1"]["status"] == "ungraded"


def test_workers_split_the_questions_each_with_its_own_database():
    from experiments.z2 import oracle as O
    argv = ["experiments/z2/oracle.py", "run", "--database", "ice_z2_main_lme", "--out", "logs/x", "--workers", "2"]
    assert O.worker_argv(argv, 1, 2) == ["experiments/z2/oracle.py", "run", "--database", "ice_z2_main_lme_1",
                                         "--out", "logs/x", "--shard", "1/2"]
