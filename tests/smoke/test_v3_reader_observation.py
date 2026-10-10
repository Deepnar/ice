"""Opposing accounting controls and native recovery with complete private faults."""

import copy
import io
import json
import stat

import pytest

from experiments.v3_repair.fault_receipts import FaultRecorder
from experiments.v3_repair.report_reader_usage import ARMS, count_usage
from scripts.z1.worker_recovery import WorkerRecovery


def trace():
    fragment = dict(leg="vector", type="episodic", text="The port is7319.",
                    source_row="r1", origin_batches=["b1"], origin_edges=[], tokens=8)
    stage = dict(generated_by_leg={"vector": [fragment]}, ranked_candidates=[fragment],
                 budgeted=[fragment], selected=[fragment], source_note_ids=[],
                 prompt_messages=[dict(role="user", content=fragment["text"])])
    return [dict(event="written", conversation="fixture", turn=1, episodic_id="r1", batch_id="b1"),
            dict(event="written", conversation="fixture", turn=2, episodic_id="r2", batch_id="b2"),
            dict(event="as_of_probe", probe_id="q1", conversation="fixture", split_turn=2, state_turns=2,
                 preflight=copy.deepcopy(stage), controls={arm: copy.deepcopy(stage) for arm in ARMS if arm != "full"}),
            dict(event="as_of_probe", probe_id="q2", conversation="fixture", split_turn=2, state_turns=2,
                 preflight=copy.deepcopy(stage), controls={arm: copy.deepcopy(stage) for arm in ARMS if arm != "full"}),
            dict(event="written", conversation="fixture", turn=3, episodic_id="r3", batch_id="b3")]


def test_unique_sources_repeated_exposure_and_zero_observed_reads_are_separate():
    rows = trace()
    # Repeated fragments/aliases within one question do not create new memories.
    rows[2]["preflight"]["selected"] *= 2
    report = count_usage(rows)
    full = report["arms"]["full"]
    assert full["stored_originals"] == 3 and full["observed_originals"] == 2
    assert full["zero_evidence_reads_observed"] == 1
    one, two, three = full["originals"]
    assert one["final_evidence_occurrences"] == 2
    assert one["counts"]["rendered_selected"] == 2 and one["counts"]["observed_opportunities"] == 2
    assert two["final_evidence_occurrences"] == three["final_evidence_occurrences"] == 0
    assert three["counts"].get("observed_opportunities", 0) == 0
    assert list(full["rendered_fragment_version_frequency"].values()) == [2]
    assert report["occurrences"][0]["arms"]["full"]["stages"]["rendered_selected"]["unique_originals"] == 1
    assert report["arms"]["no_codex"]["totals"]["rendered_selected_fragments"] == 2
    assert all(x["eligible_opportunities"] == "unobserved" for x in full["originals"])


def test_selected_lineage_without_final_text_does_not_count_as_rendered():
    rows = trace()
    rows[2]["preflight"]["prompt_messages"] = [dict(role="user", content="Unrelated text.")]
    report = count_usage(rows)
    one = report["arms"]["full"]["originals"][0]
    assert one["counts"]["selected"] == 2 and one["counts"]["rendered_selected"] == 1
    assert report["arms"]["full"]["totals"]["selected_text_unconfirmed"] == 1
    assert report["occurrences"][0]["arms"]["full"]["answer_use"] == "unmeasured"


def test_notes_and_graph_lineage_are_not_new_artifact_denominators():
    rows = trace()
    stage = rows[2]["preflight"]
    stage["selected"][0]["origin_edges"] = ["edge1", "edge1"]
    stage["source_note_ids"] = ["r2", "b2"]
    stage["selected"][0]["origin_batches"].append("unknown")
    report = count_usage(rows)
    assert report["arms"]["full"]["credited_graph_edge_frequency"] == {"edge1": 1}
    assert report["arms"]["full"]["originals"][1]["counts"]["rendered_note_lineage"] == 1
    assert report["occurrences"][0]["arms"]["full"]["unknown_source_aliases"] == ["unknown"]
    assert report["occurrences"][0]["arms"]["full"]["exact_derived_artifact_inventory"] == "unobserved"


@pytest.mark.parametrize("fault", ["duplicate", "inventory", "roster", "cutoff"])
def test_changed_occurrence_or_stored_denominators_refuse(fault):
    rows = trace(); probes = None
    if fault == "duplicate": rows[3]["probe_id"] = "q1"
    if fault == "inventory": rows[2]["state_turns"] = 99
    if fault == "roster": del rows[2]["controls"]["recent_only"]
    if fault == "cutoff":
        probes = {r["probe_id"]: copy.deepcopy(r) for r in rows if r["event"] == "as_of_probe"}
        probes["q1"]["split_turn"] = 1
    with pytest.raises(ValueError): count_usage(rows, probes)


def test_native_recovery_retains_two_complete_chained_attempts_and_metadata(tmp_path, monkeypatch):
    import scripts.z1.worker_recovery as recovery
    monkeypatch.setattr(recovery.time, "sleep", lambda *_: None)
    path = tmp_path / "faults.jsonl"
    recorder = FaultRecorder(path, "frozen-run", private_root=tmp_path, secrets=["private-test-secret"])
    attempts = []
    def failure():
        try: raise ValueError("inner cause api_key=private-test-secret")
        except ValueError as cause: raise ConnectionError("LONG" * 400 + " Bearer another-secret") from cause
    def observed():
        attempts.append(1)
        return recorder.call(failure, context=dict(conversation="fixture", turn=80, job="post_flight",
            stage="writer", attempt=len(attempts), attempts=2, model="controlled", window=8192, artifact_id="batch1"))
    native = WorkerRecovery(io.StringIO(), "continue")
    assert native.call(observed, job="post_flight", context=dict(conversation="fixture", turn=80), rollback=lambda: None) == (False, None)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(attempts) == native.state["failed_attempts"] == 2
    assert len(rows[0]["error"]) > 1500 and len(native.state["degraded"][0]["error"]) == 500
    assert "inner cause" in rows[0]["traceback"] and "failure" in rows[0]["traceback"]
    assert [e["error_type"] for e in rows[0]["exception_chain"]] == ["ConnectionError", "ValueError"]
    assert "private-test-secret" not in path.read_text() and "another-secret" not in path.read_text()
    assert rows[0]["context"]["turn"] == 80 and rows[1]["context"]["attempt"] == 2
    assert rows[1]["previous_receipt_sha256"] == rows[0]["receipt_sha256"]
    assert rows[0]["policy_changed"] is False and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert recorder.call(lambda: 7, context={}) == 7 and len(path.read_text().splitlines()) == 2


def test_terminal_fault_is_recorded_and_still_pauses_native_policy(tmp_path):
    recorder = FaultRecorder(tmp_path / "faults.jsonl", "run", private_root=tmp_path)
    native = WorkerRecovery(io.StringIO(), "continue")
    original = RuntimeError("controlled structural fault")
    def failure(): raise original
    with pytest.raises(RuntimeError) as caught:
        native.call(lambda: recorder.call(failure, context=dict(job="reflection", turn=10)),
                    job="reflection", context=dict(conversation="fixture", turn=10), rollback=lambda: None)
    assert caught.value is original and len(recorder.path.read_text().splitlines()) == 1
    assert native.state["failed_attempts"] == 0


def test_receipt_resume_detects_changed_chain_and_interrupted_tail(tmp_path):
    path = tmp_path / "faults.jsonl"
    first = FaultRecorder(path, "run", private_root=tmp_path)
    first.record(ValueError("first"), {})
    resumed = FaultRecorder(path, "run", private_root=tmp_path)
    resumed.record(ValueError("second"), {})
    saved = path.read_bytes()
    row = json.loads(path.read_text().splitlines()[0]); row["error"] = "altered"
    path.write_text(json.dumps(row) + "\n" + path.read_text().splitlines()[1] + "\n")
    corrupted = path.read_bytes()
    with pytest.raises(ValueError, match="chain changed"): resumed.record(ValueError("third"), {})
    assert path.read_bytes() == corrupted
    path.write_bytes(saved.rstrip(b"\n"))
    with pytest.raises(ValueError, match="interrupted tail"): resumed.record(ValueError("third"), {})
    assert path.read_bytes() == saved.rstrip(b"\n")


def test_only_safe_provider_receipt_and_explicit_metadata_survive(tmp_path):
    recorder = FaultRecorder(tmp_path / "faults.jsonl", "run", private_root=tmp_path)
    error = RuntimeError("provider failure")
    error.receipt = dict(response_id="response1", completion_status="incomplete", usage=dict(output_tokens=10, messages="PRIVATE USAGE"),
                         messages=["PRIVATE INPUT"], headers=dict(authorization="PRIVATE HEADER"))
    error.status_code = 503
    recorder.record(error, dict(probe_id="q1", profile="controlled", request_id="req1"))
    row = json.loads(recorder.path.read_text())
    assert row["error_status"] == 503 and row["provider_receipt"]["response_id"] == "response1"
    assert "PRIVATE" not in recorder.path.read_text()
    assert "user:pass" not in recorder.redact('postgresql://user:pass@localhost/db')
    assert "hidden" not in recorder.redact('{"api_key":"hidden"}')
    with pytest.raises(ValueError, match="undeclared"): recorder.record(error, dict(headers={}))
    with pytest.raises(ValueError, match="private path"): FaultRecorder(tmp_path.parent / "public", "run", private_root=tmp_path)
