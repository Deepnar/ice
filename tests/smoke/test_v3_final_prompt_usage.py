"""Final native memory exposure counters include history without answer credit."""

import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from experiments.v3_repair.final_prompt_artifacts import POLICY, observe
from experiments.v3_repair.report_reader_usage import count_usage
from scripts.z1 import production_parity as pp, seed_v3
from src.classifier.schemas import ClassificationResult

import test_v3_native_reader_usage as native_trace
import test_v3_retrieved_artifact_ids as helpers

store = helpers.store


def sha(body):
    return hashlib.sha256(body.encode()).hexdigest()


def captured_stage():
    stage = native_trace.stage("r1")
    body = "\n\n=== PERSISTENT CONTEXT ===\n[PREFERENCE]\nCopper."
    stage["prompt_messages"][0]["content"] += body
    pair = [dict(role="user", content="Copper."), dict(role="assistant", content="Recorded.")]
    stage["prompt_messages"][1:1] = pair
    # Identical wording in distinct native rows is not one stored artifact.
    stage.update(prompt_artifact_policy=POLICY, prompt_artifact_blocks={
        "slots": dict(message_index=0, char_start=len("Standing context."),
            char_end=len("Standing context." + body), rendered_text_sha256=sha(body),
            items=[dict(stored_artifact_ids=[["memory_slots", "s1"], ["memory_slots", "s2"]])]),
        "recent_turns": dict(items=[dict(stored_artifact_ids=[["episodic_memory", "r1"]],
            message_count=2, message_indices=[1, 2], rendered_messages_sha256=sha(
                json.dumps(pair, sort_keys=True, ensure_ascii=False)))])})
    for f in stage["selected"]:
        f.update(stored_artifact_ids=[["episodic_memory", "r1"]], rendered_text_sha256=sha(f["text"]))
    return stage


def records(stage):
    return [dict(event="written", conversation="fixture", turn=1, episodic_id="r1", batch_id="b1"),
            dict(event="as_of_probe", probe_id="q1", conversation="fixture", split_turn=1,
                 state_turns=1, preflight=stage, controls={})]


def test_history_and_retrieved_evidence_union_counts_once_per_occurrence():
    stage = captured_stage()
    rows = records(stage)
    repeated = copy.deepcopy(rows[-1]); repeated["probe_id"] = "q2"; rows.append(repeated)
    result = count_usage(rows, arms=("full",), native_evidence_only=True)
    artifacts = {(r["table"], r["key"]): r for r in result["arms"]["full"]["observed_stored_artifact_frequency"]}
    original = artifacts["episodic_memory", "r1"]
    assert original["counts"]["rendered_selected"] == 2
    assert original["counts"]["final_block:recent_turns"] == 2
    assert original["counts"]["any_final_context"] == 2
    assert len(original["rendered_version_frequency"]) == len(original["final_block_version_frequency"]) == 1
    assert all(artifacts["memory_slots", key]["counts"]["any_final_context"] == 2 for key in ("s1", "s2"))
    assert result["occurrences"][0]["arms"]["full"]["answer_use"] == "unmeasured"
    assert observe(native_trace.stage("r1")) is None


@pytest.mark.parametrize("fault", ["policy", "span", "hash", "header", "index", "duplicate", "gap",
                                  "role", "body", "ref", "evicted", "count", "missing"])
def test_final_block_receipts_refuse_ambiguous_or_unexposed_metadata(fault):
    value = captured_stage()
    blocks = value["prompt_artifact_blocks"]
    recent = blocks["recent_turns"]["items"][0]
    if fault == "policy": value["prompt_artifact_policy"] = "wrong"
    elif fault == "span": blocks["slots"]["char_end"] += 1
    elif fault == "hash": blocks["slots"]["rendered_text_sha256"] = "wrong"
    elif fault == "header": value["prompt_messages"][0]["content"] = "changed"
    elif fault == "index": recent["message_indices"] = [len(value["prompt_messages"]) - 2, len(value["prompt_messages"]) - 1]
    elif fault == "duplicate": blocks["recent_turns"]["items"].append(copy.deepcopy(recent))
    elif fault == "gap": blocks["recent_turns"]["items"] = []
    elif fault == "role": value["prompt_messages"][1]["role"] = "system"
    elif fault == "body": value["prompt_messages"][1]["content"] = "changed"
    elif fault == "ref": recent["stored_artifact_ids"] = [["no-key"]]
    elif fault == "evicted": value["evicted"] = ["slots"]
    elif fault == "count": value["prompt_block_counts"] = dict(slots=0)
    else: del blocks["recent_turns"]
    with pytest.raises(ValueError): observe(value)


def test_anonymous_items_remain_unmapped_in_verified_container():
    value = captured_stage()
    value["prompt_artifact_blocks"]["slots"]["items"][0]["stored_artifact_ids"] = []
    observed = observe(value)
    assert observed["slots"]["refs"] == set() and observed["slots"]["unmapped_items"] == 1


def test_native_research_preparation_forwards_bookmarks_and_serializes_final_blocks(store, monkeypatch):
    from src.api.config import settings
    from src.model_registry import registry

    db = store
    row = helpers.source(db); row.is_bookmarked = True; db.flush()
    monkeypatch.setattr(settings, "context_use_serving_window", False)
    monkeypatch.setattr(registry, "get_model_context_window", lambda *_a: 8192)
    c = ClassificationResult([], [], "Zero_Shot", [0.] * 27, 1., prompt="What did I say?", p_ltm=0.)
    pre = pp.Preamble(c, {"conversation_id": str(row.conversation_id)}, helpers.VEC,
        str(row.conversation_id), 1, 100, 2048, "controlled-answerer", "current", False, 0.)
    memory, generated, ranked, budgeted, notes = seed_v3.prepare_with_trace(
        db, pre, NS(embedder=helpers.Encoder()), source_time=helpers.NOW)
    stage = seed_v3.stage_record(pre, memory, generated, ranked, budgeted, notes, include_prompt=True)
    stage = json.loads(json.dumps(stage))
    observed = observe(stage)
    ref = ("episodic_memory", str(row.id))
    assert observed["bookmarks"]["refs"] == observed["recent_turns"]["refs"] == {ref}
    assert not memory.retrieve and not stage["selected"] and memory.prepared.ledger.fits()
    assert stage["answer_use"] == "unknown_until_new_answer_is_generated"
    value = count_usage([dict(event="written", conversation="fixture", turn=1,
        episodic_id=str(row.id), batch_id=str(row.batch_id)), dict(event="as_of_probe", probe_id="q1",
        conversation="fixture", split_turn=1, state_turns=1, preflight=stage, controls={})],
        arms=("full",), native_evidence_only=True)
    artifact = value["arms"]["full"]["observed_stored_artifact_frequency"][0]
    assert artifact["counts"]["any_final_context"] == 1
    assert artifact["counts"]["final_block:bookmarks"] == artifact["counts"]["final_block:recent_turns"] == 1
    assert artifact["live_inventory"] == "unobserved"
