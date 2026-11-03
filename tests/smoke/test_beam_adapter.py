"""FINAL BEAM adapter: pairs and dates from BEAM's own anchors, evidence ids to turns,
rubric nuggets to references, and BEAM's own scoring (one nugget per call, tau_norm for order)."""

import json

from experiments.beam import adapter as A


def row():
    m = lambda i, role, text, anchor="None": dict(id=str(i), role=role, content=text, time_anchor=anchor,
                                                  index="None", question_type="None")
    q = {"knowledge_update": [dict(question="api latency?", answer="250ms", rubric="['state 250ms']",
                                   source_chat_ids="{'original_info': [0], 'updated_info': [4]}")],
         "event_ordering": [dict(question="order?", answer="a then b", rubric="['a', 'b']",
                                 source_chat_ids="[0, 4]")]}
    return dict(conversation_id="c1", probing_questions=str(q), chat=[
        [m(0, "user", "api is 400ms", "March-15-2024"), m(1, "assistant", "ok"), m(2, "user", "next"), m(3, "assistant", "sure")],
        [m(4, "user", "now 250ms", "April-05-2024"), m(5, "assistant", "great")]])


def test_originals_asks_and_scoring():
    originals, ids = A.to_originals(row())
    assert [o["turn_number"] for o in originals] == [1, 2, 3]
    assert originals[0]["timestamp"].startswith("2024-03-15T19:00") and originals[2]["timestamp"].startswith("2024-04-05")
    asks = A.to_asks(row(), ids, 3)
    upd = next(a for a in asks if a["type"] == "knowledge_update")
    assert upd["reference"]["source_turns"] == [1, 3] and upd["checkpoint_turn"] == 3
    order = next(a for a in asks if a["type"] == "event_ordering")
    seen = []

    def judge(messages):
        seen.append(messages)
        if messages[0]["role"] == "system":                     # BEAM's llm_equivalence: YES/NO
            first, second = (x.split(":", 1)[1].strip() for x in messages[1]["content"].split("\n") if "snippet:" in x)
            return "YES" if first == second else "NO"
        nugget = messages[0]["content"].split("RUBRIC CRITERION (what to check): ", 1)[1].split("\n", 1)[0]
        return json.dumps(dict(score=1.0 if nugget == "a" else 0.5, reason="r"))
    s = A.score(order, "b\na", judge)
    assert s["llm_judge_score"] == 0.75 and s["nuggets"] == [1.0, 0.5]
    assert s["tau_norm"] == 0.0 and s["beam_score"] == 0.0 and s["f1"] == 1.0          # reversed order
    blank = A.score(order, "b\n\na", judge)              # BEAM keeps a blank line as an unmatched item
    assert round(blank["tau_norm"], 4) == round(1 / 3, 4) and round(blank["f1"], 4) == 0.8
    assert A.score(order, "a\nb", judge)["beam_score"] == 1.0
    nugget_calls = [m for m in seen if m[0]["role"] == "user"]
    assert len(nugget_calls) == 6 and len(nugget_calls[0]) == 1         # 3 answers x 2 nuggets, one per call and A.UNIFIED_JUDGE_PROMPT.split("<rubric_item>")[0] in nugget_calls[0][0]["content"]
    assert all(m[1]["content"].split("Second snippet:")[1].strip() for m in seen if m[0]["role"] == "system")  # blank lines cost no call
    assert A.parse_json_response('```json\n{"score": 0.5}\n```') == {"score": 0.5}


def test_beam_grading_scores_resumes_and_leaves_incomplete_ungraded(tmp_path):
    from experiments.beam import grade as G
    from experiments.z2 import read_passes as R
    originals, ids = A.to_originals(row())
    asks = A.to_asks(row(), ids, 3)
    (tmp_path / "asks.json").write_text(json.dumps(asks))
    units, ans = tmp_path / "units", tmp_path / "ans"
    ans.mkdir()
    index, lines = {}, []
    for i, (cond, finish) in enumerate((("full", "complete"), ("recent_only", "incomplete"))):
        for a in asks:
            R.write_unit(R.unit_path(units, cond, a["id"]), dict(condition=cond, ask=dict(id=a["id"])))
            sha = f"{cond}{a['id']}"
            index[f"{cond}|{a['id']}"] = sha
            lines.append(json.dumps(dict(prompt_sha256=sha, finish=finish, text="250ms; a then b")))
    (ans / "index.json").write_text(json.dumps(dict(index=index)))
    (ans / "answers.jsonl").write_text("\n".join(lines) + "\n")
    calls = []
    def judge(messages):
        calls.append(messages)
        if messages[0]["role"] == "system":
            return "NO"
        return json.dumps(dict(score=1.0, reason="r"))
    assert G.run(units, ans, tmp_path / "asks.json", tmp_path / "out", judge) == 0
    report = json.loads((tmp_path / "out/beam-report.json").read_text())["by_condition_and_type"]
    assert report["full"]["knowledge_update"]["mean_score"] == 1.0
    assert report["recent_only"]["knowledge_update"]["graded"] == 0
    before = len(calls)
    G.run(units, ans, tmp_path / "asks.json", tmp_path / "out", judge)
    assert len(calls) == before                                          # resumed


def test_grouped_source_ids_are_flattened():
    # event_ordering groups one event's messages: [[24, 26], [146], 7] (5 questions in the chosen set)
    from experiments.beam.adapter import source_turns
    id_to_turn = {7: 1, 24: 2, 26: 2, 146: 5}
    assert source_turns(dict(source_chat_ids="[[24, 26], [146], 7]"), id_to_turn) == [1, 2, 5]
    assert source_turns(dict(source_chat_ids={"a": [7, [24]]}), id_to_turn) == [1, 2]


def test_an_unreadable_reply_leaves_the_answer_for_the_next_run(tmp_path, capsys):
    """Never scored 0 for the judge's failure: BEAM would repair the text; this run retries it."""
    from experiments.beam import grade as G
    from experiments.z2 import read_passes as R
    _, ids = A.to_originals(row())
    asks = [a for a in A.to_asks(row(), ids, 3) if a["type"] == "knowledge_update"]
    (tmp_path / "asks.json").write_text(json.dumps(asks))
    units, ans = tmp_path / "units", tmp_path / "ans"
    ans.mkdir()
    R.write_unit(R.unit_path(units, "full", asks[0]["id"]), dict(condition="full", ask=dict(id=asks[0]["id"])))
    (ans / "index.json").write_text(json.dumps(dict(index={f"full|{asks[0]['id']}": "s"})))
    (ans / "answers.jsonl").write_text(json.dumps(dict(prompt_sha256="s", finish="complete", text="250ms")) + "\n")
    assert G.run(units, ans, tmp_path / "asks.json", tmp_path / "out", lambda m: "no json here") == 1
    assert json.loads((tmp_path / "out/status.json").read_text()) == dict(pending=1, finished=False)
    assert "left for the next run" in capsys.readouterr().out


def test_beam_judge_check_plants_and_pass_marks():
    """Z2 F27: BEAM's own judge is checked on planted answers before FINAL uses it."""
    from experiments.beam import judge_check as J
    ask = lambda i, t, ans, nug: dict(id=f"BEAM:c1:{t}:{i}", type=t, question="q", reference=dict(answer=ans, nuggets=nug))
    asks = [ask(i, t, f"{t} answer {i}", [f"{t} point {i}"]) for t in ("knowledge_update", "abstention", "temporal_reasoning")
            for i in range(3)] + [ask(i, "event_ordering", "x", ["e1", "e2", "e3"]) for i in range(2)]
    ps = J.plants(asks)
    kinds = [k for k, _, _ in ps]
    assert kinds.count("gold") == 9 and kinds.count("extra") == 6 and kinds.count("ordered") == 2   # no extra on abstention
    rev = next(p for p in ps if p[0] == "reversed")
    assert rev[2] == "e3\ne2\ne1"
    rows = [(k, a, dict(beam_score={"gold": 1.0, "extra": 0.95, "off_topic": 0.0, "ordered": 1.0, "reversed": 0.0}[k]))
            for k, a, _ in ps]
    assert all(J.summarise(rows)["passed"].values())
    rows = [(k, a, dict(beam_score=0.5 if k == "extra" else s["beam_score"])) for k, a, s in rows]
    assert J.summarise(rows)["passed"]["extra_drop"] is False                  # long answers must not lose points


def test_a_failing_judge_call_is_retried_then_unjudged_never_a_crash(tmp_path):
    """Z3 judging run 1 (2026-11-03): one reply cut by its output cap stopped the whole BEAM check."""
    from experiments.beam import grade as G
    from experiments.beam import judge_check as J
    from experiments.lme.cloud_provider import CloudCompletionError
    from experiments.z2 import read_passes as R

    def cut(messages):
        raise CloudCompletionError("cloud answer completion was not successful: incomplete")
    _, ids = A.to_originals(row())
    asks = [a for a in A.to_asks(row(), ids, 3) if a["type"] == "knowledge_update"]
    (tmp_path / "asks.json").write_text(json.dumps(asks))
    units, ans = tmp_path / "units", tmp_path / "ans"
    ans.mkdir()
    R.write_unit(R.unit_path(units, "full", asks[0]["id"]), dict(condition="full", ask=dict(id=asks[0]["id"])))
    (ans / "index.json").write_text(json.dumps(dict(index={f"full|{asks[0]['id']}": "s"})))
    (ans / "answers.jsonl").write_text(json.dumps(dict(prompt_sha256="s", finish="complete", text="250ms")) + "\n")
    quiet = lambda *_: None
    assert [G.run(units, ans, tmp_path / "asks.json", tmp_path / "out", cut, say=quiet) for _ in range(3)] == [1, 1, 0]
    report = json.loads((tmp_path / "out/beam-report.json").read_text())["by_condition_and_type"]
    assert report["full"]["knowledge_update"] == dict(n=1, graded=0, unjudged=1, mean_score=None)
    plants = [dict(id=f"BEAM:c:{t}:{i}", type=t, question="q", reference=dict(answer=f"a{i}", nuggets=[f"n{i}"]))
              for t in ("knowledge_update", "temporal_reasoning") for i in range(3)]
    for _ in range(3):
        pending = J.run(plants, tmp_path / "check", cut, say=quiet)
    assert pending == 0 and json.loads((tmp_path / "check/report.json").read_text())["unjudged"] > 0
