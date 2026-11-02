"""FINAL memory-system baselines (LoCoMo): the evidence envelope identical to ICE's bare-vector
position, sessions on the source clock without blank sides, each speaker's own view for Mem0,
half-built stores wiped, and search (no ICE) split from the wrap in ICE's frame, safe to stop."""

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


class Fake:
    """A memory system that records its calls; `stop_at` raises mid-ingest or mid-search."""
    added, searched, stop_at = [], [], None

    def __init__(self, store):
        self.store = store

    async def setup(self):
        return None

    async def add(self, when, sid, msgs):
        if Fake.stop_at == ("add", sid):
            raise KeyboardInterrupt
        Fake.added.append(sid)

    async def search(self, question):
        if Fake.stop_at == ("search", question):
            raise KeyboardInterrupt
        Fake.searched.append(question)
        return [f"hit for {question}"]


def fake_run(tmp_path, monkeypatch, on_disk):
    import asyncio
    monkeypatch.setitem(B.SYSTEMS, "fake", Fake)
    monkeypatch.setattr(B, "ON_DISK", {"fake"} if on_disk else set())
    Fake.added, Fake.searched, Fake.stop_at = [], [], None
    originals = [dict(turn_number=1, timestamp="2023-05-08T13:56:00Z", prompt="Ann: a", response="Bo: b"),
                 dict(turn_number=2, timestamp="2023-06-02T09:00:00Z", prompt="Ann: c", response="Bo: d")]
    asks = [dict(id=f"q{i}", kind="question", question=f"Q{i}?") for i in (1, 2, 3)]
    return lambda: asyncio.run(B.search("fake", "LOCOMO-x", originals, asks, tmp_path / "search")), asks


def test_a_stop_mid_search_keeps_the_ingested_store_and_its_hits(tmp_path, monkeypatch):
    go, _ = fake_run(tmp_path, monkeypatch, on_disk=True)
    Fake.stop_at = ("search", "Q2?")
    try:
        go()
    except KeyboardInterrupt:
        pass
    d = tmp_path / "search/instances/LOCOMO-x"
    assert (d / "ingested.json").exists() and not (d / "done.json").exists()
    assert sorted(p.stem for p in (d / "hits").glob("*.json")) == ["q1"] and not list(d.rglob("*.tmp"))
    Fake.stop_at = None
    go()
    assert Fake.added == ["s1", "s2"]                                   # never ingested twice
    assert Fake.searched == ["Q1?", "Q2?", "Q3?"] and (d / "done.json").exists()   # Q1's hit kept, not re-asked
    go()                                                                # a finished one is left alone
    assert Fake.added == ["s1", "s2"]


def test_a_stop_mid_ingest_rebuilds_the_store_from_its_first_session(tmp_path, monkeypatch):
    go, _ = fake_run(tmp_path, monkeypatch, on_disk=True)
    Fake.stop_at = ("add", "s2")
    try:
        go()
    except KeyboardInterrupt:
        pass
    assert not (tmp_path / "search/instances/LOCOMO-x/ingested.json").exists()
    Fake.stop_at = None
    go()
    assert Fake.added == ["s1", "s1", "s2"]                             # the half store was wiped first


def test_a_store_in_memory_is_ingested_again_after_a_stop(tmp_path, monkeypatch):
    go, _ = fake_run(tmp_path, monkeypatch, on_disk=False)              # A-Mem's case
    Fake.stop_at = ("search", "Q2?")
    try:
        go()
    except KeyboardInterrupt:
        pass
    Fake.stop_at = None
    go()
    assert Fake.added == ["s1", "s2", "s1", "s2"] and Fake.searched.count("Q1?") == 2


def test_wrap_places_saved_hits_in_ices_envelope_and_waits_for_the_search(tmp_path, monkeypatch):
    import json

    import pytest
    go, asks = fake_run(tmp_path, monkeypatch, on_disk=True)
    ice = tmp_path / "ice"
    for a in asks:
        R.write_unit(ice / "units" / "recent_only" / f"{a['id']}.json",
                     dict(ask=dict(id=a["id"]), checkpoint=9, question_time="t",
                          stage=dict(prompt_messages=[dict(role="system", content="S"), dict(role="user", content=a["question"])])))
    with pytest.raises(SystemExit):
        B.wrap("fake", "LOCOMO-x", asks, tmp_path / "search", ice, tmp_path / "out")
    go()
    B.wrap("fake", "LOCOMO-x", asks, tmp_path / "search", ice, tmp_path / "out")
    unit = json.loads((tmp_path / "out/units/fake/q2.json").read_text())
    assert unit["condition"] == "fake" and unit["checkpoint"] == 9 and unit["stage"]["memories"] == ["hit for Q2?"]
    assert [m["role"] for m in unit["stage"]["prompt_messages"]] == ["system", "user", "assistant", "user"]
    assert (tmp_path / "out/instances/LOCOMO-x/done.json").exists()


def test_the_session_clock_sets_a_modules_now_and_restores_it():
    import types
    from datetime import datetime, timezone
    mod = types.SimpleNamespace(datetime=datetime)
    when = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    with B.session_clock(when, [mod]):
        assert mod.datetime.now(timezone.utc).date().isoformat() == "2023-05-08"
        assert mod.datetime.now().year == 2023 and mod.datetime.fromisoformat("2024-01-01").year == 2024
    assert mod.datetime is datetime and datetime.now().year >= 2026


def test_a_bare_list_is_wrapped_only_where_the_schema_is_one_list_field():
    """Graphiti's NodeResolutions came back as `[...]` from gemma4:e4b on Ollama (2026-11-02)."""
    from pydantic import BaseModel
    RB = B

    class Item(BaseModel):
        id: int

    class Resolutions(BaseModel):
        entity_resolutions: list[Item]

    class Two(BaseModel):
        a: list[int]
        b: int

    assert RB.wrapped_list([{"id": 1}], Resolutions) == {"entity_resolutions": [{"id": 1}]}
    assert RB.wrapped_list({"entity_resolutions": []}, Resolutions) is None       # already right
    assert RB.wrapped_list([1], Two) is None and RB.wrapped_list([1], None) is None
