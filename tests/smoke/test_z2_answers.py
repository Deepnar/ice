"""Z2 S10 answers: one answer per distinct prompt, incomplete and too-long answers
recorded (ungraded), the local answerer never truncates, and the cloud path paces
under Luna's 5-hour limit and waits through quota and key errors."""

import json
from types import SimpleNamespace as NS

from experiments.lme.cloud_provider import CloudCompletionError, ProviderAccessError
from experiments.z2 import answers as A
from experiments.z2 import read_passes as R


def units(tmp_path, rows):
    for cond, ask, messages in rows:
        R.write_unit(R.unit_path(tmp_path, cond, ask), dict(condition=cond, ask=dict(id=ask),
                                                            stage=dict(prompt_messages=messages)))
    return tmp_path


def test_identical_prompts_are_answered_once_and_rerun_skips(tmp_path):
    same = [dict(role="user", content="q")]
    src = units(tmp_path / "s9", [("full", "a", same), ("recent_only", "a", same),
                                  ("full", "b", [dict(role="user", content="other")])])
    calls = []
    gen = lambda messages, **_: calls.append(messages) or NS(text="ans", usage={})
    status = A.run(src, tmp_path / "s10", generate=gen, meta={}, say=lambda *_: None, workers=1)
    assert status["prompts"] == 2 and status["answered"] == 2 and status["finished"] and len(calls) == 2
    A.run(src, tmp_path / "s10", generate=gen, meta={}, say=lambda *_: None, workers=1)
    assert len(calls) == 2
    index = json.loads((tmp_path / "s10" / "index.json").read_text())["index"]
    assert index["full|a"] == index["recent_only|a"] != index["full|b"]


def test_incomplete_is_recorded_not_retried(tmp_path):
    def gen(messages, **_):
        raise CloudCompletionError("cloud answer completion was not successful: incomplete", receipt={})
    src = units(tmp_path / "s9", [("full", "a", [dict(role="user", content="q")])])
    status = A.run(src, tmp_path / "s10", generate=gen, meta={}, say=lambda *_: None, workers=1)
    assert status["incomplete"] == 1 and status["finished"]


def test_quota_and_bad_key_are_waited_through():
    replies = [ProviderAccessError("p", "quota/rate-limit", 429, "slow down"),
               ProviderAccessError("p", "authentication", 401, "bad key"), NS(text="ok", usage={})]
    refreshed, slept = [], []
    def gen(messages, **_):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    row = A.answer_one(gen, [], "x" * 64, say=lambda *_: None, sleep=slept.append,
                       reload_env=lambda: refreshed.append(1))
    # the key is re-read after the quota wait too (a swapped key continues the run)
    assert row["finish"] == "complete" and slept == [A.QUOTA_WAIT, A.AUTH_WAIT] and refreshed == [1, 1]


def test_pacer_never_exceeds_the_window():
    now = [0.0]
    pacer = A.Pacer([], limit=3, window=100, sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0])
    for _ in range(3):
        pacer.take(lambda *_: None)
    assert now[0] == 0.0
    pacer.take(lambda *_: None)            # the 4th waits for the first to leave the window
    assert now[0] >= 100


def fake_ollama(reply):
    calls = []
    def post(path, body, timeout=3600):
        calls.append((path, body))
        return reply
    return post, calls


def test_local_answerer_sends_think_seed_ctx_and_reads_the_answer():
    post, calls = fake_ollama(dict(message=dict(content=" Tofu. ", thinking="hmm"), done_reason="stop",
                                   prompt_eval_count=100, eval_count=20))
    local = A.LocalAnswerer(host="http://x", post=post)
    local.estimate_tokens = lambda messages: 100
    out = local.generate([dict(role="user", content="q")], temperature=0, max_output_tokens=8000)
    assert out.text == "Tofu." and out.usage["thinking_chars"] == 3
    body = calls[0][1]
    assert body["think"] is True and body["options"]["num_ctx"] == A.LOCAL_NUM_CTX
    assert body["options"]["seed"] == A.LOCAL_SEED and body["options"]["num_predict"] == 8000
    local.estimate_tokens = lambda messages: 36_000          # cap = what the window leaves
    local.generate([dict(role="user", content="q")], temperature=0, max_output_tokens=16_000)
    assert calls[-1][1]["options"]["num_predict"] == A.LOCAL_NUM_CTX - int(36_000 * A.FIT_MARGIN)


def test_local_cap_is_incomplete_and_overflow_is_too_long_never_truncated(tmp_path):
    post, calls = fake_ollama(dict(message=dict(content="half"), done_reason="length",
                                   prompt_eval_count=100, eval_count=8000))
    local = A.LocalAnswerer(host="http://x", post=post)
    local.estimate_tokens = lambda messages: 100
    src = units(tmp_path / "s9", [("full", "a", [dict(role="user", content="q")])])
    status = A.run(src, tmp_path / "s10", generate=local.generate, meta={}, say=lambda *_: None)
    assert status["incomplete"] == 1 and status["finished"]
    local.estimate_tokens = lambda messages: A.LOCAL_NUM_CTX          # cannot fit: never sent
    src2 = units(tmp_path / "s9b", [("full", "b", [dict(role="user", content="huge")])])
    calls.clear()
    status = A.run(src2, tmp_path / "s10b", generate=local.generate, meta={}, say=lambda *_: None)
    assert status["too_long"] == 1 and not calls


def test_a_different_answerer_never_mixes_into_saved_answers(tmp_path):
    src = units(tmp_path / "s9", [("full", "a", [dict(role="user", content="q")]),
                                  ("full", "b", [dict(role="user", content="r")])])
    gen = lambda messages, **_: NS(text="ans", usage={})
    A.run(src, tmp_path / "s10", generate=gen, meta=dict(model="qwen3.8:27b"), say=lambda *_: None)
    calls = []
    try:
        A.run(src, tmp_path / "s10", generate=lambda m, **_: calls.append(m),
              meta=dict(model="muse-spark-1.2-contributor"), say=lambda *_: None)
        raise AssertionError("mixed answerers were accepted")
    except A.MixedAnswerers:
        assert calls == []


def test_muse12_profile_cap_and_workers():
    from experiments.z2.profiles import Z2_PROFILES
    assert Z2_PROFILES["muse12"].model == "muse-spark-1.2-contributor"
    assert A.OUTPUT_TOKENS["muse12"] == 16_000 and A.WORKERS["muse12"] == 4


def test_allowance_projects_each_stage_from_measured_answers(tmp_path):
    from experiments.z2 import allowance as W
    (tmp_path / "s10-B").mkdir()
    rows = [dict(prompt_sha256=str(i), usage=dict(opencode=dict(allowance=a))) for i, a in enumerate((0.001, 0.002, 0.003))]
    (tmp_path / "s10-B" / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = []
    grand = W.report(tmp_path, totals={"s10-B": 100, "s10-C": 10, "s12-answers": 0}, say=out.append)
    assert abs(grand - 0.002 * 110) < 1e-9 and "s10-B" in out[0] and "from the other stages" in out[1]


def test_prompts_sharing_a_cache_session_run_in_order_on_one_worker(tmp_path):
    """FINAL long context: one session per checkpoint; the group's first call fills the cache,
    the rest follow it on the same worker while other groups run beside it."""
    import threading
    for i in range(3):
        for cp in (10, 20):
            R.write_unit(R.unit_path(tmp_path / "s9", "long_context", f"q{i}@{cp}"),
                         dict(condition="long_context", ask=dict(id=f"q{i}@{cp}"),
                              stage=dict(prompt_messages=[dict(role="user", content=f"history {cp} / q{i}")],
                                         cache_session=f"ice-final-long-H-{cp}")))
    seen = []

    def gen(messages, *, session_id, **_):
        seen.append((session_id, threading.get_ident(), messages[0]["content"]))
        return NS(text="ans", usage={})
    status = A.run(tmp_path / "s9", tmp_path / "s10", generate=gen, meta={}, say=lambda *_: None, workers=2)
    assert status["answered"] == 6
    for cp in (10, 20):
        group = [s for s in seen if s[0] == f"ice-final-long-H-{cp}"]
        assert len(group) == 3 and len({thread for _, thread, _ in group}) == 1


def test_a_provider_refusal_is_its_own_final_outcome(tmp_path):
    """The content filter's refusal is recorded once as `refused` (ungraded, counted), never
    retried as an error; any other unexpected error still propagates to the error store."""
    src = units(tmp_path / "s9", [("full", "a", [dict(role="user", content="q")]),
                                  ("full", "b", [dict(role="user", content="other")])])

    def gen(messages, **_):
        if messages[0]["content"] == "q":
            raise RuntimeError("Error code: 400 - {'error': {'code': 'content_policy_violation'}}")
        raise ValueError("something else")
    status = A.run(src, tmp_path / "s10", generate=gen, meta={}, say=lambda *_: None, workers=1)
    assert status["refused"] == 1 and status["answered"] == 1 and status["pending"] == 1
    rows = [json.loads(l) for l in (tmp_path / "s10" / "answers.jsonl").read_text().splitlines()]
    assert [r["finish"] for r in rows] == ["refused"]
