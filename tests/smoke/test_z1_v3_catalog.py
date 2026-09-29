"""A scored v3 replay must use the complete, time-indexed source catalog."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

from scripts.z1.answer_as_of import LOGS, load_probes, main as answer_main, reviewed_native_probes
from scripts.z1.run_meta import file_digest
from scripts.z1.seed_v3 import (CORPUS, DERIVED, GENERATED, TYPED, UNIFIED,
                                EXPECTED, gold_fragment_coverage, load_plan)
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
    _, captured, _, _ = load_plan(capture_unlabeled_native=True)
    assert sum(map(len, captured.values())) == 255
    unresolved = [p for group in captured.values() for p in group
                  if p["cutoff_kind"] == "native_unlabeled_checkpoint"]
    assert len(unresolved) == 131
    assert all(not p["gold_turns"] and p["source_split_turn"] is None
               for p in unresolved)


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


def test_reviewed_native_source_can_join_frozen_checkpoint(tmp_path, monkeypatch, capsys):
    _, captured, _, _ = load_plan(capture_unlabeled_native=True)
    probe = next(p for (_, split), group in captured.items() for p in group
                 if p["cutoff_kind"] == "native_unlabeled_checkpoint" and split >= 41)
    stage = {"generated_by_leg": {}, "budgeted": [], "selected": [],
             "prompt_tokens": 1, "selected_tokens": 0,
             "prompt_messages": [{"role": "system", "content": "frozen"}],
             "lineage": {"future_source_ids": []}}
    frozen = {"probe_id": probe["probe_id"], "type": probe["probe_type"],
              "conversation": probe["conversation"],
              "split_turn": probe["split_turn"], "question": probe["question"],
              "cutoff_kind": probe["cutoff_kind"], "gold_turns": [],
              "gold_sources": [], "state_turns": probe["split_turn"],
              "preflight": dict(stage),
              "controls": {"vector_only": dict(stage), "recent_only": dict(stage)}}
    inputs = [file_digest(path) for path in
              (CORPUS, UNIFIED, TYPED, DERIVED, GENERATED)]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in [
        {"event": "run", "meta": {"inputs": inputs,
                                   "settings_resolved": {
                                       "recent_window_max_turns": settings.recent_window_max_turns}}},
        {"event": "written", "conversation": probe["conversation"], "turn": 1,
         "episodic_id": "row-1", "batch_id": "batch-1"},
        {"event": "as_of_probe", **frozen},
        {"event": "complete", "complete_selected_corpus": True,
         "expected_as_of_probes": 1},
    ]) + "\n")
    assert len(load_probes(trace, allow_partial=False)) == 1
    audit = tmp_path / "audit.json"
    packet = {"kind": "native_checkpoint_source_review", "inputs": inputs[:4],
              "recent_window_turns": settings.recent_window_max_turns,
              "records": [{"probe_id": probe["probe_id"],
                           "conversation": probe["conversation"],
                           "cutoff_turn": probe["split_turn"],
                           "question": probe["question"],
                           "answer_verdict": "valid", "reviewed_gold_turns": [1],
                           "reason": "Reviewed source support and intervening turns."}]}
    audit.write_text(json.dumps(packet))
    joined = reviewed_native_probes(trace, audit, [frozen])
    assert len(joined) == 1
    assert joined[0]["gold_turns"] == [1]
    assert joined[0]["gold_sources"][0]["source_ids"] == ["batch-1", "row-1"]
    assert joined[0]["preflight"]["gold_fragment_coverage"]["gold_turns"] == 1
    with tempfile.TemporaryDirectory(prefix="ice-z1-plan-", dir=LOGS) as root:
        private_trace = Path(root) / "trace.jsonl"
        private_audit = Path(root) / "audit.json"
        private_trace.write_bytes(trace.read_bytes())
        private_audit.write_bytes(audit.read_bytes())
        monkeypatch.setattr(sys, "argv", ["answer_as_of.py", "--trace",
                str(private_trace), "--out", str(Path(root) / "unused.json"),
                "--arm", "full", "--plan", "--validated-native-sources",
                str(private_audit)])
        assert answer_main() == 0
        assert json.loads(capsys.readouterr().out)["probes"] == 1
    packet["records"][0]["reason"] = ""
    audit.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="reason"):
        reviewed_native_probes(trace, audit, [frozen])
