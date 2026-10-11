"""Native source-exposure channels, evidence visibility and immutable loading."""

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from experiments.v3_repair import chronological_sources as sources
from experiments.v3_repair import native_reader_usage as usage
from experiments.v3_repair.report_reader_usage import ARMS
from scripts.z1.replay_checkpoint import prefix_digest


def stage(source=None):
    fragment = dict(leg="vector", type="episodic", text="The port is7319.",
                    source_row=source, origin_batches=[], origin_edges=[], tokens=8)
    fragments = [fragment, copy.deepcopy(fragment)] if source else []
    messages = [dict(role="system", content="Standing context.")]
    if fragments:
        messages += [dict(role="user", content="=== RETRIEVED CONTEXT ===\n" + fragment["text"]),
                     dict(role="assistant", content="Understood — I have the background context. What would you like to know?")]
    messages += [dict(role="user", content="What is the port?")]
    return dict(generated_by_leg={"vector": fragments}, ranked_candidates=fragments,
        budgeted=fragments, selected=fragments, source_note_ids=[], prompt_messages=messages,
        context_ledger=dict(blocks=dict(evidence=16 if fragments else 0)))


def trace():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    originals = [dict(turn_number=i, prompt="Authored original.", response="Recorded reply.",
        timestamp=start + timedelta(hours=i), ts_provenance="original") for i in (1, 2)]
    probes = []
    for cutoff in (1, 2):
        question = "What is the port?"
        review = dict(conversation="fixture", cutoff_turn=cutoff, reviewed_through_turn=cutoff,
            question=question, expected_answer="Scorer7319.", knowledge_scope="private_history",
            recent_only_answerable=True, task_types=["episodic_lookup"])
        probes.append((cutoff, dict(probe_id=f"q{cutoff}", question=question,
            expected_answer=review["expected_answer"], gold_turns=[1],
            label_status="source_reviewed_valid", label_review=review)))
    plan = sources.build("fixture", originals, probes, clock_basis="original")
    identity = dict(format=usage.PARENT, condition="full", source_plan=plan["identity_sha256"],
                    diagnostic_arms=list(ARMS), tables=["controlled_table"], database="controlled_database")
    rows = [dict(event="chronological_meta", version="v3", score_of_record=False,
                 identity=identity, source_plan=plan)]
    for index, event in enumerate(plan["events"]):
        turn = event["turn"]
        if event["kind"] == "source":
            rows.append(dict(event="source_preflight", event_id=event["id"], turn=turn,
                             preflight=stage("r1" if turn == 2 else None)))
            rows.append(dict(event="source_written", event_id=event["id"], turn=turn,
                clock=event["clock"], original_time=event["original_time"], ts_provenance="original",
                episodic_id=f"r{turn}", batch_id=f"b{turn}", source_raw_sha256="controlled_raw_hash",
                source_sequence=dict(position=turn, raw_sha256="controlled_raw_hash",
                    sequence_sha256=plan["identity"]["source_sequence_sha256"])))
        else:
            rows.append(dict(event="chronological_probe", event_id=event["id"],
                probe_id=event["probe_id"], conversation="fixture", split_turn=turn,
                question=probes[turn - 1][1]["question"], question_time=event["clock"],
                clock_basis="original", stages={a: stage("r1") for a in ARMS},
                scorer=probes[turn - 1][1], probe_writes=False, answer_quality_scored=False))
        rows.append(dict(event="chronological_event_complete", event_id=event["id"], completed=index + 1))
    rows.append(dict(event="chronological_complete", version="v3", events=4, source_turns=2,
        diagnostic_occurrences=2, clock_basis="original", condition="full", answer_quality_scored=False))
    return rows


def test_real_preflight_and_readonly_probe_frequencies_remain_separate():
    report = usage.adapt(trace())
    actual = report["actual_preflight_exposure"]["arms"]["full"]
    diagnostic = report["readonly_diagnostic_preparation"]["arms"]["full"]
    assert set(report["actual_preflight_exposure"]["arms"]) == {"full"}
    assert actual["stored_originals"] == diagnostic["stored_originals"] == 2
    assert [r["own_stored_originals"] for r in report["actual_preflight_exposure"]["occurrences"]] == [0, 1]
    assert [r["own_stored_originals"] for r in report["readonly_diagnostic_preparation"]["occurrences"]] == [1, 2]
    assert actual["originals"][0]["final_evidence_occurrences"] == 1
    assert diagnostic["originals"][0]["final_evidence_occurrences"] == 2
    assert actual["originals"][1]["counts"].get("observed_opportunities", 0) == 0
    assert diagnostic["originals"][1]["final_evidence_occurrences"] == 0
    assert all(x["eligible_opportunities"] == "unobserved" for x in actual["originals"])


def test_exact_stored_ids_and_rendered_versions_differ_from_source_lineage():
    rows = trace()
    flight = next(r for r in rows if r["event"] == "source_preflight" and r["turn"] == 2)
    for label in ("generated_by_leg", "ranked_candidates", "budgeted", "selected"):
        fragments = flight["preflight"][label]["vector"] if label == "generated_by_leg" else flight["preflight"][label]
        for i, fragment in enumerate(fragments):
            fragment["stored_artifact_ids"] = [["procedural_memory", f"pattern-{i}"]]
            fragment["rendered_text_sha256"] = hashlib.sha256(fragment["text"].encode()).hexdigest()
    report = usage.adapt(rows)["actual_preflight_exposure"]["arms"]["full"]
    artifacts = report["observed_stored_artifact_frequency"]
    assert len(artifacts) == 2
    assert [r["key"] for r in artifacts] == ["pattern-0", "pattern-1"]
    assert all(r["counts"]["rendered_selected"] == 1 for r in artifacts)
    assert all(list(r["rendered_version_frequency"].values()) == [1] for r in artifacts)
    assert all(r["eligible_opportunities"] == r["live_inventory"] == "unobserved" for r in artifacts)
    assert report["originals"][0]["final_evidence_occurrences"] == 1


@pytest.mark.parametrize("fault", ["ref", "version", "prompt"])
def test_incomplete_native_artifact_receipts_refuse(fault):
    rows = trace()
    flight = next(r for r in rows if r["event"] == "source_preflight" and r["turn"] == 2)
    fragment = flight["preflight"]["selected"][0]
    fragment["stored_artifact_ids"] = [["procedural_memory", "p1"]]
    fragment["rendered_text_sha256"] = hashlib.sha256(fragment["text"].encode()).hexdigest()
    if fault == "ref": fragment["stored_artifact_ids"] = [["missing-key"]]
    elif fault == "version": fragment["rendered_text_sha256"] = "wrong"
    else: del flight["preflight"]["prompt_messages"]
    with pytest.raises(ValueError): usage.adapt(rows)


def test_native_serializer_keeps_exact_ids_without_changing_legacy_payload():
    from scripts.z1.seed_v3 import fragment as serialize
    from src.retrieval.orchestrator import ContextFragment
    native = ContextFragment("Source wording.", "procedural", 1.0, 8)
    legacy = serialize(native, include_text=True)
    assert "stored_artifact_ids" not in legacy and "rendered_text_sha256" not in legacy
    identified = ContextFragment(native.text, native.source_type, native.score, native.token_count,
                                  stored_artifact_ids=(("procedural_memory", "p1"),))
    measured = serialize(identified, include_text=True)
    assert {k: v for k, v in measured.items() if k not in {"stored_artifact_ids", "rendered_text_sha256"}} == legacy
    assert measured["stored_artifact_ids"] == [["procedural_memory", "p1"]]
    assert measured["rendered_text_sha256"] == hashlib.sha256(native.text.encode()).hexdigest()


@pytest.mark.parametrize("location", ["current_question", "recent_history", "zero_evidence_history", "nowhere"])
def test_selected_text_outside_evidence_block_cannot_impersonate_evidence(location):
    rows = trace()
    preflight = next(r for r in rows if r["event"] == "source_preflight" and r["turn"] == 2)
    fragment = preflight["preflight"]["selected"][0]
    messages = [dict(role="system", content="Standing context.")]
    if location == "recent_history":
        messages.extend([dict(role="user", content="=== RETRIEVED CONTEXT ===\n" + fragment["text"]),
                         dict(role="assistant", content="Old reply."), dict(role="user", content="Later history.")])
    if location == "zero_evidence_history":
        messages.extend([dict(role="user", content="=== RETRIEVED CONTEXT ===\n" + fragment["text"]),
                         dict(role="assistant", content="Understood — I have the background context. What would you like to know?")])
        preflight["preflight"]["context_ledger"]["blocks"]["evidence"] = 0
    messages.append(dict(role="user", content=fragment["text"] if location == "current_question" else "Question."))
    preflight["preflight"]["prompt_messages"] = messages
    report = usage.adapt(rows)["actual_preflight_exposure"]["arms"]["full"]
    assert report["originals"][0]["counts"]["selected"] == 1
    assert report["originals"][0]["final_evidence_occurrences"] == 0
    assert report["totals"]["selected_text_unconfirmed"] == 2


@pytest.mark.parametrize("fault", ["partial", "append", "preflight", "write", "receipt", "question",
                                    "source", "alias", "roster", "cutoff", "future", "scorer", "final"])
def test_broken_native_trace_refuses_instead_of_reporting_zero_usage(fault):
    rows = trace()
    flight = next(r for r in rows if r["event"] == "source_preflight")
    written = next(r for r in rows if r["event"] == "source_written")
    probe = next(r for r in rows if r["event"] == "chronological_probe")
    if fault == "partial": rows.pop()
    elif fault == "append": rows.append(dict(event="source_written"))
    elif fault == "preflight": rows.remove(flight)
    elif fault == "write": rows.insert(rows.index(written), written)
    elif fault == "receipt": next(r for r in rows if r["event"] == "chronological_event_complete")["completed"] = 2
    elif fault == "question": probe["question"] = "Changed runtime question."
    elif fault == "source": written["source_sequence"]["position"] = 9
    elif fault == "alias":
        next(r for r in rows if r["event"] == "source_written" and r["turn"] == 2)["episodic_id"] = "b1"
    elif fault == "roster": del probe["stages"]["recent_only"]
    elif fault == "cutoff": probe["split_turn"] = 2
    elif fault == "future": probe["scorer"]["gold_turns"] = [2]
    elif fault == "scorer": probe["scorer"]["expected_answer"] = "Changed key."
    elif fault == "final": rows[-1]["source_turns"] = 3
    with pytest.raises(ValueError): usage.adapt(rows)


def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(usage, "ROOT", tmp_path)
    root = tmp_path / "logs" / "producer"
    directory = root / "recovery" / "generation-fixture"
    directory.mkdir(parents=True)
    rows = trace()
    path = root / "chronological.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pointer = root / "recovery" / "current.json"
    pointer.write_text(json.dumps(dict(generation=directory.name)))
    result = usage.adapt(rows)
    manifest = dict(identity=rows[0]["identity"], generation=directory.name,
        trace_path=str(path.resolve()), trace_bytes=path.stat().st_size,
        trace_sha256=prefix_digest(path, path.stat().st_size), state=dict(complete=True,
            event_prefix=result["event_prefix"], source_ids=result["source_ids"]))
    (directory / "recovery.json").write_text(json.dumps(manifest))
    (directory / "store.sql").write_text("Controlled dump, never restored.\n")
    (directory / "store.counts").write_text("2025-01-01T00:00:00+00:00\ncontrolled_table=0\n")
    (directory / "store.manifest.json").write_text(json.dumps(dict(database="controlled_database",
        tables=["controlled_table"], counts=dict(controlled_table=0), sha256_by_table=dict(controlled_table="controlled"),
        dump_sha256=prefix_digest(directory / "store.sql", (directory / "store.sql").stat().st_size))))
    # The loader binds actual instruments; keep their own paths below its root.
    monkeypatch.setattr(usage, "file_receipt", lambda p: dict(path=str(p), bytes=p.stat().st_size,
        sha256=prefix_digest(p, p.stat().st_size)))
    return root, directory


def test_offline_complete_loader_is_byte_stable_without_store_or_model(tmp_path, monkeypatch):
    root, directory = bundle(tmp_path, monkeypatch)
    originals = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    result = usage.report(root)
    assert result == usage.report(root)
    assert result["event_prefix"]["completed"] == 4
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == saved for p, saved in originals.items())
    assert len(result["identity"]["inputs"]) == 6


@pytest.mark.parametrize("fault", ["trace", "dump", "prefix", "source_ids", "identity", "complete"])
def test_changed_native_checkpoint_refuses_without_touching_inputs(tmp_path, monkeypatch, fault):
    root, directory = bundle(tmp_path, monkeypatch)
    if fault == "trace":
        with (root / "chronological.jsonl").open("a") as f: f.write("{}\n")
    elif fault == "dump": (directory / "store.sql").write_text("Changed dump.")
    else:
        path = directory / "recovery.json"
        value = json.loads(path.read_text())
        if fault == "prefix": value["state"]["event_prefix"]["completed"] = 3
        elif fault == "source_ids": value["state"]["source_ids"]["1"][0] = "changed"
        elif fault == "identity": value["identity"]["condition"] = "changed"
        elif fault == "complete": value["state"]["complete"] = False
        path.write_text(json.dumps(value))
    saved = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, KeyError)): usage.report(root)
    assert all(p.read_bytes() == before for p, before in saved.items())
