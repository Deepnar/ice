"""A scored v3 replay must use the complete, time-indexed source catalog."""

from scripts.z1.seed_v3 import EXPECTED, gold_fragment_coverage, load_plan
from scripts.z1.snapshot import TABLES
from src.api.config import settings
from src.memory.models import Base


def test_full_source_and_as_of_probe_contract():
    conversations, probes, quarantined, ineligible = load_plan()
    assert {slug: len(rows) for slug, rows in conversations.items()} == EXPECTED
    assert sum(map(len, probes.values())) == 124
    assert len(ineligible) == 66
    assert len(quarantined) == 32
    assert sum(p["cutoff_kind"] == "native_checkpoint"
               for group in probes.values() for p in group) == 11
    assert sum(p["cutoff_kind"] == "generated_at_checkpoint"
               for group in probes.values() for p in group) == 113
    assert {slug: sum(len(group) for (s, _), group in probes.items() if s == slug)
            for slug in conversations} == {"bb558b5f": 14, "ecc64aab": 87,
                                        "355a5709": 23}
    assert all(all(gold <= split for gold in p["gold_turns"])
               for (_, split), group in probes.items() for p in group)
    assert all(split - max(p["gold_turns"]) >= settings.recent_window_max_turns
               for (_, split), group in probes.items() for p in group)
    assert all(p["label_status"] in {"needs_intervening_turn_review",
                                     "needs_gold_source_confirmation"}
               for group in probes.values() for p in group)
    assert all(conversations[slug][split - 1]["turn_number"] == split
               for slug, split in probes)
    assert all(p.get("source_excerpt")
               for group in probes.values() for p in group
               if p["cutoff_kind"] == "generated_at_checkpoint")
    _, immediate, _, no_immediate = load_plan(probe_timing="immediate",
                                              include_typed=True,
                                              include_native=False,
                                              include_generated=False)
    assert sum(map(len, immediate.values())) == 444
    assert not no_immediate
    _, all_longterm, _, _ = load_plan(include_typed=True)
    assert sum(map(len, all_longterm.values())) == 502
    ids = [p["probe_id"] for group in all_longterm.values() for p in group]
    assert len(ids) == len(set(ids))


def test_snapshot_includes_every_current_store_table():
    assert TABLES == sorted(Base.metadata.tables)
    assert {"codex_claims", "codex_claim_links", "codex_snapshots",
            "batch_notes", "conversation_notes", "cold_chunks",
            "idempotency_keys", "maintenance_ledger"} <= set(TABLES)


def test_gold_fragment_funnel_uses_both_source_id_spaces():
    row = {"source_row": "episodic-row", "origin_batches": [], "leg": "vector"}
    derived = {"source_row": None, "origin_batches": ["source-batch"],
               "leg": "codex"}
    stage = {"generated_by_leg": {"vector": [row], "codex_claims": [derived]},
             "budgeted": [row, derived], "selected": [derived]}
    result = gold_fragment_coverage(stage, {9: {"episodic-row", "source-batch"}})
    assert result["gold_turns"] == 1
    assert result["generated_by_leg"] == {"vector": 1, "codex_claims": 1}
    assert result["generated"] == result["budgeted"] == result["selected"] == 1
