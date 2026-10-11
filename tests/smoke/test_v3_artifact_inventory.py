"""Native row revisions/live denominators are distinct from lifetime writes."""

import copy
import json
from types import SimpleNamespace as NS

import pytest

from experiments.v3_repair import artifact_inventory as inventory
from experiments.v3_repair.prepare_inputs import digest
from experiments.v3_repair.report_reader_usage import count_usage
from scripts.z1 import production_parity as pp, seed_v3
from src.classifier.schemas import ClassificationResult

import test_v3_retrieved_artifact_ids as helpers

store = helpers.store


def indexed(db):
    return inventory.validate(inventory.capture(db))


def test_native_payload_changes_are_separate_from_access_decay_state(store):
    row = helpers.source(store)
    equal = helpers.source(store, cid=row.conversation_id)
    ref = ("episodic_memory", str(row.id))
    before = indexed(store)
    assert before[ref]["stored_content_sha256"] == before["episodic_memory", str(equal.id)]["stored_content_sha256"]
    row.decay_score = .2; row.access_count = 8; store.flush()
    later = indexed(store)
    assert later[ref]["stored_content_sha256"] == before[ref]["stored_content_sha256"]
    assert later[ref]["stored_state_sha256"] != before[ref]["stored_state_sha256"]
    row.raw_text += " Edited source."; store.flush()
    assert indexed(store)[ref]["stored_content_sha256"] != before[ref]["stored_content_sha256"]


def test_indexed_keys_and_preserved_archive_ids_remain_physical_and_logical(store):
    warm = helpers.source(store)
    cold = helpers.source(store, cold=True)
    cold.id = warm.id; store.flush()
    row, _root = helpers.notes(store, "conversation", "indexed")
    value = indexed(store)
    assert value["episodic_memory", str(warm.id)]["logical_identity"] == value["cold_storage", str(cold.id)]["logical_identity"]
    assert ("conversation_notes", json.dumps([str(row.conversation_id), 1], separators=(",", ":"))) in value


@pytest.mark.parametrize("fault", ["hash", "policy", "projection", "duplicate", "revision", "logical"])
def test_changed_catalog_cannot_silently_change_denominators(store, fault):
    helpers.source(store)
    value = inventory.capture(store)
    if fault == "hash": value["rows"].clear()
    elif fault == "policy": value["policy"] = "changed"
    elif fault == "projection": value["schema"]["episodic_memory"]["content"] = ["raw_text"]
    elif fault == "duplicate": value["rows"].append(copy.deepcopy(value["rows"][0]))
    elif fault == "revision": value["rows"][0]["stored_content_sha256"] = "not-a-hash"
    else: value["rows"][0]["logical_identity"] = ["wrong", "identity"]
    if fault != "hash": value["catalog_sha256"] = digest({k: v for k, v in value.items() if k != "catalog_sha256"})
    with pytest.raises(ValueError): inventory.validate(value)


def test_inventory_refuses_production_without_queries(store, monkeypatch):
    monkeypatch.setenv("ICE_TEST_DATABASE", "ice_db")
    with pytest.raises(ValueError, match="isolated"): inventory.capture(store)


def test_live_zero_read_rows_and_context_union_have_honest_denominators(store, monkeypatch):
    from src.api.config import settings
    from src.model_registry import registry
    row = helpers.source(store); other = helpers.source(store)
    row.is_bookmarked = True; store.flush()
    catalog = inventory.capture(store)
    monkeypatch.setattr(settings, "context_use_serving_window", False)
    monkeypatch.setattr(registry, "get_model_context_window", lambda *_a: 8192)
    c = ClassificationResult([], [], "Zero_Shot", [0.] * 27, 1., prompt="What did I say?", p_ltm=0.)
    pre = pp.Preamble(c, {"conversation_id": str(row.conversation_id)}, helpers.VEC,
        str(row.conversation_id), 1, 100, 2048, "controlled-answerer", "current", False, 0.)
    values = seed_v3.prepare_with_trace(store, pre, NS(embedder=helpers.Encoder()), source_time=helpers.NOW)
    stage = seed_v3.stage_record(pre, *values, include_prompt=True)
    stage["artifact_inventory"] = json.loads(json.dumps(catalog))
    rows = [dict(event="written", conversation="fixture", turn=1, episodic_id=str(row.id), batch_id=str(row.batch_id)),
            dict(event="as_of_probe", probe_id="q1", conversation="fixture", split_turn=1, state_turns=1,
                 preflight=stage, controls={})]
    result = count_usage(rows, arms=("full",), native_evidence_only=True)
    artifacts = {r["key"]: r for r in result["arms"]["full"]["observed_stored_artifact_frequency"]}
    assert artifacts[str(row.id)]["counts"]["any_final_context"] == 1
    assert artifacts[str(other.id)]["zero_context_reads_at_observed_live_opportunities"] is True
    assert all(r["counts"]["live_catalog_opportunities"] == 1 and r["eligible_opportunities"] == "unobserved"
               for r in artifacts.values())
    assert catalog["lifetime_write_events"] == "unobserved"
    stage["prompt_artifact_blocks"]["bookmarks"]["items"][0]["stored_artifact_ids"] = [["episodic_memory", "absent"]]
    with pytest.raises(ValueError, match="absent"): count_usage(rows, arms=("full",), native_evidence_only=True)
