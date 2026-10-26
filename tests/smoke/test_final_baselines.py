"""FINAL memory-system baselines: the pure parts — date parsing, session order, and the
evidence envelope identical to ICE's bare-vector position (constants in sync)."""

import importlib.util

from experiments.z2 import read_passes as R

spec = importlib.util.spec_from_file_location("run_baseline", "experiments/final/baselines/run_baseline.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def test_constants_match_ice_and_envelope_places_memories_like_bare_vector():
    assert B.CONTEXT_HEADER == R.CONTEXT_HEADER and B.CONTEXT_ACK == R.CONTEXT_ACK
    recent = dict(stage=dict(prompt_messages=[dict(role="system", content="S"), dict(role="user", content="Q?")]))
    msgs = B.envelope(recent, ["cat is Tofu (from 2023-05-20)"], "Q?")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"].startswith(R.CONTEXT_HEADER) and msgs[-1]["content"] == "Q?"
    assert B.envelope(recent, [], "Q?") == recent["stage"]["prompt_messages"]


def test_sessions_in_date_order():
    inst = dict(haystack_dates=["2023/05/21 (Sun) 10:00", "2023/05/20 (Sat) 02:21"], haystack_session_ids=["b", "a"],
                haystack_sessions=[[dict(role="user", content="2")], [dict(role="user", content="1")]])
    assert [sid for _, sid, _ in B.sessions(inst)] == ["a", "b"]
