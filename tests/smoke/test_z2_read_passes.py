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
    asks = [dict(id="a", checkpoint_turn=50, kind="question", type="x"),
            dict(id="b", checkpoint_turn=25, kind="question", type="x")]
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


def test_budget_curves_and_guard_pair_apply_only_where_specified():
    asks = [dict(id=f"q{i}", kind="question", type=("update" if i % 4 else "temporal"), checkpoint_turn=25)
            for i in range(200)] + [dict(id="c1", kind="control", type="small_talk", checkpoint_turn=25)]
    guard = R.guard_sample(asks)
    assert guard == R.guard_sample(list(reversed(asks)))
    assert 70 <= len(guard) <= 90 and "c1" not in guard
    kinds = {a["id"]: a["type"] for a in asks}
    assert {kinds[g] for g in guard} == {"update", "temporal"}
    control = asks[-1]
    assert not R.applies(R.CONDITIONS["full_b2000"], control, guard)
    assert R.applies(R.CONDITIONS["full_default"], control, guard)
    outside = next(a for a in asks if a["kind"] == "question" and a["id"] not in guard)
    assert not R.applies(R.CONDITIONS["full_guard_every_word"], outside, guard)
    assert R.applies(R.CONDITIONS["bare_b8000"], outside, guard)


def test_ablation_chains_follow_the_frozen_order():
    order = R.ORDER
    assert order[0] == "bm25" and order[-1] == "maintenance" and len(order) == 10
    assert R.CONDITIONS["loo_rerank"]["settings"] == {"retrieval_rerank_enabled": False}
    assert R.CONDITIONS["loo_rerank"]["groups_off"] == ["rerank"]
    first = R.CONDITIONS["add_1_bm25"]
    assert set(first["groups_off"]) == set(order[1:])            # only BM25/hybrid on
    assert R.phase_of(first) == "unmerged"                       # maintenance off -> merges undone
    assert R.phase_of(R.CONDITIONS["add_9_temporal"]) == "unmerged"
    assert set(R.CONDITIONS["add_9_temporal"]["groups_off"]) == {"maintenance"}
    assert R.phase_of(R.CONDITIONS["loo_codex"]) == "neutral"
    assert "no_fold" in R.CONDITIONS["loo_notes"]["harness"]
    from src.api.config import settings
    from src.retrieval.configurable_orchestrator import SUPPORTED_FLAGS
    for name, spec in R.CONDITIONS.items():
        assert set(spec.get("flags", {})) <= SUPPORTED_FLAGS, name
        assert all(hasattr(settings, k) for k in spec.get("settings", {})), name


def test_long_context_packs_the_newest_messages_into_the_budget():
    from experiments.v3_repair.bare_vector import Chunk
    chunks = [Chunk(t, role, 0, 10, f"m{t}{role[0]}") for t in range(1, 6) for role in ("user", "assistant")]
    chosen, used = R.long_context_chunks(chunks, 4, lambda text: 1)
    assert [(c.turn, c.role) for c, _ in chosen] == [(4, "user"), (4, "assistant"), (5, "user"), (5, "assistant")]
    assert used == 4 and R.phase_of(R.CONDITIONS["long_context"]) == "bare"


def test_final_presets_and_sealed_asks(tmp_path):
    main = R.select_conditions("final_main")
    assert list(main) == R.PRESETS["final_main"] and "long_context" in main
    both = R.select_conditions("final_main+final_ablation")
    assert {"loo_codex", "bare_vector"} <= set(both) and len(both) == len(set(both))
    families = [dict(id=f"B:fam{i:03d}", type="fact_recall") for i in range(40)]
    S.write_split(tmp_path, "B", families)
    occ = [dict(family=f["id"], type="fact_recall", checkpoint_turn=25, repeat_index=0, question=f"q{i}")
           for i, f in enumerate(families)]
    (tmp_path / "occ.json").write_text(json.dumps(occ))
    final = {a["id"].split("@")[0] for a in R.build_final_asks(tmp_path, tmp_path / "occ.json")}
    z2 = {a["id"].split("@")[0] for a in R.build_asks(tmp_path, tmp_path / "occ.json")}
    assert final and z2 and not final & z2 and len(final | z2) == 40


def test_long_context_in_time_order_with_one_shared_start_per_checkpoint():
    """Older history before the recent turns, every question at a checkpoint starting at the
    same chunk (the latest of their own fills), each still within its own room."""
    recent = dict(prompt_messages=[dict(role="system", content="S"), dict(role="user", content="r"),
                                   dict(role="assistant", content="ok"), dict(role="user", content="Q?")])
    chosen = [(Chunk(1, "user", 0, 5, "hello"), None)]
    messages = R.time_order_messages(recent, chosen, "Q?")
    assert [m["content"] for m in messages] == ["S", f"{R.CONTEXT_HEADER}\n[User, turn 1] hello", R.CONTEXT_ACK,
                                                "r", "ok", "Q?"]
    chunks = [Chunk(t, role, 0, 10, f"m{t}{role[0]}") for t in range(1, 7) for role in ("user", "assistant")]
    ordered = sorted(chunks, key=lambda c: (c.turn, c.role != "user"))
    plans = {"a": (ordered[:10], 6), "b": (ordered[:8], 7)}        # a: turns 1-5, room 6; b: turns 1-4, room 7
    shared = R.shared_long_context(plans, lambda text: 1)
    assert shared["a"][0][0] is shared["b"][0][0] is ordered[4]    # a's own fill starts at chunk 4, b's at 1
    assert len(shared["a"]) <= 6 and len(shared["b"]) <= 7


def test_r7_sets_existing_switches_only_and_waits_until_chosen(tmp_path):
    """R7 (2026-11-03): the frozen candidate is full ICE plus run 2's chosen values for switches
    that exist; anything else is refused, and before the choice the condition does not exist."""
    import json
    import pytest
    from experiments.z2 import read_passes as R
    assert R.r7_condition(tmp_path / "absent.json") is None
    path = tmp_path / "r7.json"
    path.write_text(json.dumps(dict(settings={"retrieval_evidence_order": "conversation",
                                              "retrieval_rerank_original_margin": 1.0})))
    assert R.r7_condition(path) == dict(overlay="default", arm="full", settings={
        "retrieval_evidence_order": "conversation", "retrieval_rerank_original_margin": 1.0})
    path.write_text(json.dumps(dict(settings={"retrieval_brand_new_trick": True})))
    with pytest.raises(ValueError, match="existing switches only"):
        R.r7_condition(path)
    assert R.PRESETS["r7"] == ["r7_frozen", "recent_only", "bare_vector"] and "r7_frozen" in R.BUDGET_SOURCES
    from src.api.config import settings
    assert all(hasattr(settings, k) for k in R.R7_SWITCHES)            # every allowed switch is real


def test_r7_switches_off_d9_parts_the_way_z2_measured_them(tmp_path):
    """D9 (2026-11-04): fold notes, the batch-summary leg and procedures go off in R7 by the same
    switches Z2's ablations used (groups 5 and 6); any other part is refused."""
    import json
    import pytest
    from experiments.z2 import read_passes as R
    path = tmp_path / "r7.json"
    path.write_text(json.dumps(dict(settings={"turn_summaries_in_reads": False}, off=["notes", "procedural"])))
    spec = R.r7_condition(path)
    assert spec["flags"] == {"batch_summary": False, "procedural": False} and spec["harness"] == ["no_fold"]
    assert spec["overlay"] == "default" and spec["settings"] == {"turn_summaries_in_reads": False}
    path.write_text(json.dumps(dict(settings={}, off=["temporal"])))
    with pytest.raises(ValueError, match="existing switches only"):
        R.r7_condition(path)
