"""FINAL's full-history upper bound: long_context's construction with every older turn, one
cache session per conversation, ICE's own `full` contexts copied for the same answerer, and a
prompt over the window recorded, never truncated."""

import json

from experiments.final import full_history as F
from experiments.z2 import read_passes as R


def setup(tmp_path, monkeypatch, turns=6):
    originals = [dict(turn_number=t, prompt=f"user says {t}", response=f"assistant says {t}") for t in range(1, turns + 1)]
    monkeypatch.setattr("experiments.z2.ledger.load_history", lambda history: originals)
    read = tmp_path / "read"
    recent = [dict(role="system", content="S"), dict(role="user", content="user says 6"),
              dict(role="assistant", content="assistant says 6"), dict(role="user", content="what did I say first?")]
    for ask in ("q1", "q2"):
        a = dict(id=ask, kind="question", question="what did I say first?")
        R.write_unit(R.unit_path(read, "recent_only", ask), dict(condition="recent_only", ask=a, checkpoint=6,
                                                                 stage=dict(prompt_messages=recent)))
        R.write_unit(R.unit_path(read, "full", ask), dict(condition="full", ask=a, checkpoint=6,
                                                          stage=dict(prompt_messages=recent[:1] + recent[-1:])))
    return read


def test_whole_history_in_long_contexts_envelope(tmp_path, monkeypatch):
    read = setup(tmp_path, monkeypatch)
    out = tmp_path / "upper"
    summary = F.build("BEAM-100K-x", read, out, window=262_144, count=lambda m: 1000)
    assert summary["built"] == 2 and summary["too_long"] == 0
    unit = json.loads(R.unit_path(out, "full_history", "q1").read_text())
    msgs = unit["stage"]["prompt_messages"]
    evidence = next(m["content"] for m in msgs if m["content"].startswith(R.CONTEXT_HEADER))
    assert [m["role"] for m in msgs][-3:] == ["user", "assistant", "user"]           # evidence, ack, question
    assert all(f"says {t}" in evidence for t in range(1, 6)) and "turn 6]" not in evidence   # 6 is in the recent window
    assert evidence.index("turn 1]") < evidence.index("turn 5]")                              # oldest first
    assert unit["stage"]["cache_session"] == "ice-final-upper-BEAM-100K-x"
    assert unit["stage"]["prompt_messages"][-1]["content"] == "what did I say first?"
    assert R.unit_path(out, "full", "q2").exists()


def test_over_the_window_is_recorded_not_truncated(tmp_path, monkeypatch):
    read = setup(tmp_path, monkeypatch)
    summary = F.build("BEAM-100K-x", read, tmp_path / "upper", window=100_000, count=lambda m: 90_000)
    assert summary["built"] == 0 and summary["too_long"] == 2
    assert not R.unit_path(tmp_path / "upper", "full_history", "q1").exists()
