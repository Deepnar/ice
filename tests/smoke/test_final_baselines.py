"""FINAL memory-system baselines (LoCoMo): the evidence envelope identical to ICE's bare-vector
position, sessions on the source clock without blank sides, each speaker's own view for Mem0, and
half-built stores wiped."""

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


def test_locomo_sessions_split_on_the_source_clock_and_half_stores_are_wiped(tmp_path):
    originals = [dict(turn_number=1, timestamp="2023-05-08T13:56:00Z", prompt="Ann: a", response="Bo: b"),
                 dict(turn_number=2, timestamp="2023-05-08T13:59:00Z", prompt="Ann: c", response=""),
                 dict(turn_number=3, timestamp="2023-06-02T09:00:00Z", prompt="Bo: d", response="Ann: e")]
    groups = B.originals_sessions(originals)
    assert [sid for _, sid, _ in groups] == ["s1", "s2"]
    assert [m["content"] for m in groups[0][2]] == ["Ann: a", "Bo: b", "Ann: c"]
    half = tmp_path / "store"
    (half / "x").mkdir(parents=True)
    B.fresh_store(half)
    assert half.exists() and not (half / "x").exists()
    (half / "done.json").write_text("{}")
    (half / "y").mkdir()
    B.fresh_store(half)
    assert (half / "y").exists()                                              # a finished store is kept


def test_amem_is_pinned_to_its_paper_layer():
    import json
    pin = json.loads(open("experiments/final/baselines/AMEM.json").read())
    assert pin["repository"].endswith("WujiangXu/A-mem") and len(pin["commit"]) == 40 and len(pin["sha256"]) == 64
    assert "amem" in B.SYSTEMS



def test_each_speaker_gets_their_own_view_and_names_come_from_the_originals():
    originals = [dict(turn_number=1, timestamp="2023-05-08T13:56:00Z", prompt="", response="Bo: hi"),
                 dict(turn_number=2, timestamp="2023-05-08T13:59:00Z", prompt="Ann: hello", response="Bo: cat?")]
    assert B.speakers(originals) == ("Ann", "Bo")
    msgs = B.originals_sessions(originals)[0][2]
    assert [m["content"] for m in msgs] == ["Bo: hi", "Ann: hello", "Bo: cat?"]          # no blank message
    assert [m["role"] for m in B.from_side(msgs, "Ann")] == ["assistant", "user", "assistant"]
    assert [m["role"] for m in B.from_side(msgs, "Bo")] == ["user", "assistant", "user"]
    assert B.session_text(msgs).splitlines()[0] == "Bo: hi"                               # no role label
