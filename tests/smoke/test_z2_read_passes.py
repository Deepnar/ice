"""Z2 S9 read passes (spec D10): Z2 asks only (never sealed FINAL families), resume
by unit files, checkpoint times from the event log, bare vector in ICE's envelope."""

import json

from experiments.v3_repair.bare_vector import Chunk
from experiments.z2 import read_passes as R
from experiments.z2 import split as S


def test_asks_come_only_from_z2_families_plus_controls(tmp_path):
    families = [dict(id=f"B:fam{i:03d}", type="fact_recall") for i in range(40)]
    S.write_split(tmp_path, "B", families, controls=[dict(id="B:ctl1", checkpoint_turn=25, question="hi", kind="small_talk")])
    occ = [dict(family=f["id"], type="fact_recall", checkpoint_turn=25, repeat_index=0, question=f"q{i}")
           for i, f in enumerate(families)]
    (tmp_path / "occ.json").write_text(json.dumps(occ))
    asks = R.build_asks(tmp_path, tmp_path / "occ.json")
    z2 = {f["id"] for f in json.loads((tmp_path / "z2-families.json").read_text())["families"]}
    questions = [a for a in asks if a["kind"] == "question"]
    assert questions and all(a["id"].split("@")[0] in z2 for a in questions)
    assert len(questions) == len(z2) < len(families)
    assert [a["id"] for a in asks if a["kind"] == "control"] == ["B:ctl1"]


def test_pending_groups_by_checkpoint_and_skips_finished_units(tmp_path):
    asks = [dict(id="a", checkpoint_turn=50), dict(id="b", checkpoint_turn=25)]
    assert list(R.pending(asks, tmp_path)) == [25, 50]
    for c in R.CONDITIONS:
        R.write_unit(R.unit_path(tmp_path, c, "b"), {})
    assert list(R.pending(asks, tmp_path)) == [50]
    asks_path = tmp_path / "asks.json"
    asks_path.write_text(json.dumps(asks))
    assert R.progress(tmp_path, asks_path) == (len(R.CONDITIONS), 2 * len(R.CONDITIONS))


def test_checkpoint_times_read_the_return_time(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in [
        dict(event="decay", job="decay_codex"),
        dict(event="checkpoint", store="cp-25.sql", at="2025-01-02 01:12:00+00:00")]) + "\n")
    times = R.checkpoint_times(path)
    assert list(times) == [25] and times[25].isoformat() == "2025-01-02T01:12:00+00:00"


def test_bare_vector_uses_icess_evidence_pair_before_the_question():
    recent = dict(prompt_messages=[dict(role="system", content="S"), dict(role="user", content="old"),
                                   dict(role="assistant", content="ok"), dict(role="user", content="Q?")])
    chosen = [(Chunk(3, "user", 0, 5, "hello"), 0.9)]
    messages = R.bare_messages(recent, chosen, "Q?")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[3]["content"] == f"{R.CONTEXT_HEADER}\n[User, turn 3] hello"
    assert messages[4]["content"] == R.CONTEXT_ACK and messages[-1]["content"] == "Q?"
    assert R.bare_messages(recent, [], "Q?") == recent["prompt_messages"]
