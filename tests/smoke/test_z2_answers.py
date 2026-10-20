"""Z2 S10 answers: one answer per distinct prompt, pacing under Luna's 5-hour
limit, incomplete answers recorded (ungraded), waits through quota and key errors."""

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
