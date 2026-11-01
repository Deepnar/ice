"""FINAL BEAM adapter: pairs and dates from BEAM's own anchors, evidence ids to turns,
rubric nuggets to references, nugget mean and Kendall tau-b scoring."""

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
    judge = lambda system, prompt: ({"scores": [1, 0.5]} if system is A.NUGGET_SYSTEM else {"positions": [2, 1]})
    order = next(a for a in asks if a["type"] == "event_ordering")
    s = A.score(order, "b then a", judge)
    assert s["nugget_mean"] == 0.75 and s["kendall_tau_b"] == -1.0
    assert A.kendall_tau_b([1, 2, 3]) == 1.0


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
    def judge(system, prompt):
        calls.append(system)
        return {"scores": [1] * len(json.loads(prompt)["NUGGETS"])} if system is A.NUGGET_SYSTEM else {"positions": [1, 2]}
    report = G.run(units, ans, tmp_path / "asks.json", tmp_path / "out", judge)
    assert report["full"]["knowledge_update"]["mean_nugget"] == 1.0
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
