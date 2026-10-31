"""FINAL's streaming cloud runner on a small fake run: whole checkpoints only, cache groups kept
together, the answer index merged across passes, partial grading never marked finished."""
import json
from types import SimpleNamespace as NS

from experiments.z2 import answers as A
from experiments.z2 import read_passes as R


def unit(out, condition, ask, checkpoint, text, session=None):
    stage = dict(prompt_messages=[{"role": "user", "content": text}])
    if session:
        stage["cache_session"] = session
    R.write_unit(R.unit_path(out, condition, ask), dict(condition=condition, ask=dict(id=ask), checkpoint=checkpoint,
                                                        stage=stage))


def fake_run(tmp_path):
    asks = [dict(id=f"q{i}", checkpoint_turn=cp, kind="question") for i, cp in ((1, 25), (2, 25), (3, 50))]
    path = tmp_path / "asks.json"
    path.write_text(json.dumps(asks))
    conditions = {"full": dict(), "long_context": dict()}
    return path, conditions


def test_only_fully_read_checkpoints_are_ready(tmp_path):
    asks, conditions = fake_run(tmp_path)
    reads = tmp_path / "reads"
    for c in conditions:
        for q in ("q1", "q2"):
            unit(reads, c, q, 25, f"{c} {q}")
    unit(reads, "full", "q3", 50, "full q3")                               # checkpoint 50 half read
    assert R.complete_checkpoints(reads, asks, conditions) == {"25"}
    unit(reads, "long_context", "q3", 50, "long q3")
    assert R.complete_checkpoints(reads, asks, conditions) == {"25", "50"}


def test_partial_passes_answer_whole_checkpoints_and_keep_the_index(tmp_path):
    reads, out = tmp_path / "reads", tmp_path / "answers"
    for q, cp in (("q1", 25), ("q2", 25), ("q3", 50)):
        unit(reads, "long_context", q, cp, f"history up to {cp} / {q}", session=f"ice-final-long-X-{cp}")
    order = []
    gen = lambda messages, **kw: order.append((kw["session_id"], messages[0]["content"])) or NS(
        text="answer", finish_reason="stop", usage={})
    A.run(reads, out, generate=gen, meta=dict(model="fake"), checkpoints={"25"})
    assert sorted(t for _, t in order) == ["history up to 25 / q1", "history up to 25 / q2"]
    assert {s for s, _ in order} == {"ice-final-long-X-25"}                 # one cache session, together
    A.run(reads, out, generate=gen, meta=dict(model="fake"), checkpoints={"25", "50"})
    assert [t for _, t in order[2:]] == ["history up to 50 / q3"]             # nothing answered twice
    index = json.loads((out / "index.json").read_text())["index"]
    assert set(index) == {"long_context|q1", "long_context|q2", "long_context|q3"}


def test_partial_grading_is_never_marked_finished(tmp_path, monkeypatch):
    from experiments.z2 import grade as G
    monkeypatch.setattr(G, "load", lambda *a, **k: [])
    monkeypatch.setattr(G, "turn_dates", lambda h: {})
    G.run("X", tmp_path, tmp_path, tmp_path, None, tmp_path, ask=None, verdicts_only=True, partial=True)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["finished"] is False and status["partial"] is True
    G.run("X", tmp_path, tmp_path, tmp_path, None, tmp_path, ask=None, verdicts_only=True)
    assert json.loads((tmp_path / "status.json").read_text())["finished"] is True


def test_a_stage_another_runner_is_running_is_left_alone(tmp_path):
    from experiments.final import stream_cloud as S
    (tmp_path / "state.json").write_text(json.dumps(dict(stages={"cloud-answers-B": dict(status="running")})))
    ctx = NS(dir=tmp_path)
    assert S.busy_elsewhere(ctx, "cloud-answers-B") and not S.busy_elsewhere(ctx, "cloud-answers-C")


def test_a_quota_pause_never_spends_an_items_error_count():
    # a judge limit waits and retries the same call; nothing is counted against the item
    from experiments.lme.cloud_provider import ProviderAccessError
    from experiments.z2 import grade as G
    calls, slept = [], []
    def generate(*_a, **_k):
        calls.append(1)
        if len(calls) < 3:
            raise ProviderAccessError("opencode-muse13", "quota", 429, "weekly limit reached")
        return "ok"
    assert G.waiting(generate, sleep=slept.append, say=lambda *_: None)() == "ok"
    assert len(calls) == 3 and len(slept) == 2


def test_a_line_cut_by_a_stop_is_skipped_not_fatal(tmp_path):
    from experiments.z2.ledger_merge import Calls, complete_lines
    path = tmp_path / "raw.jsonl"
    path.write_text('{"identity": "a", "text": "x"}\n{"identity": "b", "te')
    assert len(complete_lines(path)) == 1 and set(Calls(path).done) == {"a"}
    store = A.Store(tmp_path / "answers.jsonl")
    (tmp_path / "answers.jsonl").write_text('{"prompt_sha256": "p1", "finish": "complete"}\n{"prompt_sha2')
    assert set(A.Store(tmp_path / "answers.jsonl").rows) == {"p1"} and store is not None
