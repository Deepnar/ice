"""Native source/read/checkpoint mechanism controls; model boundaries controlled.

These are not corpus precision, retention or answer-quality measurements.
"""

import copy
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx
import pytest

from experiments.v3_repair import chronological_replay as runner
from experiments.v3_repair import chronological_sources as sources
from experiments.v3_repair.fault_receipts import FaultRecorder
from scripts.z1 import production_parity as pp, replay_checkpoint, snapshot
from src.api.config import settings

START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def inputs():
    rows = [dict(turn_number=i, prompt=f"Original choice {i}: port7319.", response=f"Recorded reply {i}.",
                 timestamp=START + timedelta(hours=i - 1), ts_provenance="original") for i in (1, 2)]
    question = "Which port did I choose?"
    review = dict(conversation="fixture", cutoff_turn=1, reviewed_through_turn=1,
                  question=question, expected_answer="SCORER_ONLY_SECRET", knowledge_scope="private_history",
                  recent_only_answerable=True, task_types=["episodic_lookup"])
    probe = dict(probe_id="first", question=question, expected_answer=review["expected_answer"],
                 gold_turns=[1], label_status="source_reviewed_valid", label_review=review)
    probes = [(1, probe)]
    return rows, probes, sources.build("fixture", rows, probes, clock_basis="original")


@pytest.mark.parametrize("kind", ["source", "scorer", "clock", "frozen"])
def test_input_guard_refuses_before_models_or_store(kind, monkeypatch, tmp_path):
    rows, probes, plan = inputs()
    if kind == "source": rows[1]["response"] = "Changed reply."
    elif kind == "scorer": probes[0][1]["expected_answer"] = "Changed key."
    elif kind == "clock": plan = sources.build("fixture", rows, probes, clock_basis="simulated",
                                              replay_times=[START, START + timedelta(days=60)])
    else:
        plan["identity"]["frozen_writer"] = dict(profile="old writer")
        plan["identity_sha256"] = sources.digest(plan["identity"])
    monkeypatch.setattr(runner, "model_pins", lambda: pytest.fail("model pins queried before input refusal"))
    monkeypatch.setattr(runner, "SessionLocal", lambda: pytest.fail("DB opened before input refusal"))
    with pytest.raises(ValueError): runner.run(tmp_path, plan, rows, probes)


@pytest.fixture
def native(empty_replay_database, monkeypatch, tmp_path):
    from src.api.db import SessionLocal
    from src.classifier.schemas import ClassificationResult
    from src.memory import embedder as embedding
    from src.memory import tokens
    from src.model_registry import registry
    from src.retrieval import ner_utils
    from src.workers import post_flight
    from scripts.z1.historical_clock import CLOCK_MODULES

    for name in CLOCK_MODULES: importlib.import_module(name)
    for name, module in list(sys.modules.items()):
        if name.startswith("src.") and hasattr(module, "SessionLocal"):
            monkeypatch.setattr(module, "SessionLocal", SessionLocal)
    monkeypatch.setattr(runner, "SessionLocal", SessionLocal)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "FaultRecorder", lambda p, i: FaultRecorder(p, i, private_root=tmp_path))
    monkeypatch.setattr(runner, "model_pins", lambda: dict(background="controlled-writer",
        writer_digests={"fixture": "no-real-models"}, declared="controlled native mechanism fixture"))
    calls = []
    encoder = NS(encode=lambda *_a, **_k: [1.0] + [0.0] * (settings.embedding_dim - 1))
    class Classifier:
        embedder = encoder
        def classify(self, prompt, conversation_id=None):
            assert conversation_id
            assert "SCORER_ONLY_SECRET" not in prompt
            calls.append(("classify", prompt, conversation_id))
            return ClassificationResult([], [], "Zero_Shot", [0.] * 27, .99, prompt=prompt, p_ltm=.01)
    monkeypatch.setattr(runner, "components", lambda: (Classifier(), encoder))
    monkeypatch.setattr(embedding, "get_embedder", lambda: encoder)
    monkeypatch.setattr(post_flight, "extract_key_terms", lambda *_a: dict(entities=[], figures=[], identifiers=[]))
    monkeypatch.setattr(post_flight, "extract_codex", lambda **kwargs: calls.append(("codex", kwargs)))
    monkeypatch.setattr(post_flight, "extract_procedural", lambda **kwargs: calls.append(("procedural", kwargs)))
    monkeypatch.setattr(settings, "context_use_serving_window", False)
    monkeypatch.setattr(settings, "memory_source_gate_enabled", False)
    monkeypatch.setattr(settings, "maintenance_intervals", {**settings.maintenance_intervals,
        "decay_episodic": 3600, "compaction": 3600})
    monkeypatch.setattr(pp, "get_model_context_window", lambda *_a: 8192)
    monkeypatch.setattr(registry, "get_model_context_window", lambda *_a: 8192)
    monkeypatch.setattr(pp, "find_best_model", lambda *_a: ("controlled-answerer", None))
    monkeypatch.setattr(ner_utils, "extract_entities", lambda *_a, **_k: [])
    # Use the installed native vocabulary, without a remote metadata lookup.
    # A blocked tokenizer request would otherwise test the estimate fallback.
    from transformers import AutoTokenizer
    monkeypatch.setattr(tokens, "_tokenizer", AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B", local_files_only=True))
    monkeypatch.setattr(tokens, "_load_attempted", True)
    http_calls = []
    def deny(*_a, **_k):
        http_calls.append(1)
        raise AssertionError("no model HTTP permitted in mechanism fixture")
    async def async_deny(*_a, **_k): return deny()
    monkeypatch.setattr(httpx.Client, "request", deny)
    monkeypatch.setattr(httpx.AsyncClient, "request", async_deny)
    item = NS(root=tmp_path / "logs" / "parent", calls=calls, sessions=SessionLocal,
              args=inputs(), enabled=("decay_episodic", "compaction"), http_calls=http_calls)
    yield item
    assert http_calls == []


def execute(native, **kwargs):
    originals, probes, plan = native.args
    return runner.run(native.root, plan, originals, probes, enabled=native.enabled, **kwargs)


def test_native_recorded_writer_readonly_probe_gap_jobs_and_complete_resume(native):
    from src.memory.models import EpisodicMemory
    from src.memory.source_order import verified_sequence
    state = execute(native)
    assert state["complete"] and state["event_prefix"]["completed"] == 3
    with native.sessions() as db:
        rows = db.query(EpisodicMemory).order_by(EpisodicMemory.timestamp).all()
        assert len(rows) == 2
        assert [r.raw_text for r in rows] == [f"User: {r['prompt']}\n\nAssistant: {r['response']}" for r in native.args[0]]
        assert [verified_sequence(r)["position"] for r in rows] == [1, 2]
        assert rows[0].session_id != rows[1].session_id
        assert all(r.lossless_flag is not None for r in rows)
        assert all(r.ts_provenance == "original" for r in rows)
    trace = native.root / "chronological.jsonl"
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    bootstrap = [r for r in records if r["event"] == "maintenance_bootstrap"]
    assert len(bootstrap) == 2 and all(r["source_time"] == START.isoformat() for r in bootstrap)
    gap = [r for r in records if r["event"] == "maintenance"]
    assert len(gap) == 2 and all(r["source_time_trigger"] == (START + timedelta(hours=1)).isoformat() for r in gap)
    second = next(i for i, r in enumerate(records) if r["event"] == "source_preflight" and r["turn"] == 2)
    assert all(records.index(r) < second for r in gap)
    probe = next(r for r in records if r["event"] == "chronological_probe")
    assert set(probe["stages"]) == set(runner.seed_v3.PROMPT_ARMS)
    assert all("SCORER_ONLY_SECRET" not in json.dumps(s["prompt_messages"]) for s in probe["stages"].values())
    assert sum(c[0] == "codex" for c in native.calls) == 2
    assert sum(c[0] == "procedural" for c in native.calls) == 2
    before, tables, calls = trace.read_bytes(), snapshot.fingerprints(), copy.deepcopy(native.calls)
    assert execute(native, resume=True) == state
    assert trace.read_bytes() == before and snapshot.fingerprints() == tables and native.calls == calls


def test_interrupt_after_committed_source_restores_prefix_and_skips_completed_events(native, monkeypatch):
    from src.memory.models import EpisodicMemory
    original = runner.write_source
    interrupted = []
    def interrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[3]["turn"] == 2 and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt("after native source commit")
        return result
    monkeypatch.setattr(runner, "write_source", interrupt)
    with pytest.raises(KeyboardInterrupt): execute(native)
    pointer = json.loads((native.root / "recovery" / "current.json").read_text())
    checkpoint = native.root / "recovery" / pointer["generation"]
    manifest = json.loads((checkpoint / "recovery.json").read_text())
    assert manifest["state"]["event_prefix"]["completed"] == 2
    first_id = manifest["state"]["source_ids"]["1"][0]
    before_calls = sum(c[0] == "codex" for c in native.calls)
    restored = []
    def check_restore(*args, **kwargs):
        db, _sink, _state, event = args[:4]
        if event["turn"] == 2:
            assert db.query(EpisodicMemory).count() == 1
            assert str(db.query(EpisodicMemory).one().id) == first_id
            restored.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "write_source", check_restore)
    state = execute(native, resume=True)
    assert restored == [True] and state["complete"]
    assert sum(c[0] == "codex" for c in native.calls) == before_calls + 1
    records = [json.loads(line) for line in (native.root / "chronological.jsonl").read_text().splitlines()]
    assert sum(r["event"] == "maintenance_bootstrap" for r in records) == 2
    assert sum(r["event"] == "chronological_probe" for r in records) == 1
    assert [r["turn"] for r in records if r["event"] == "source_written"] == [1, 2]
    assert list((native.root / "recovery").glob("unfinished-*.jsonl"))
    faults = [json.loads(line) for line in (native.root / "faults.jsonl").read_text().splitlines()]
    assert any(r["error_type"] == "KeyboardInterrupt" for r in faults)


def test_internally_caught_frozen_classifier_failure_cannot_advance_prefix(native, monkeypatch):
    class Broken:
        def __init__(self, *_a): pass
        def classify(self, *_a, **_k): raise ValueError("controlled frozen classifier mismatch")
    monkeypatch.setattr(runner.seed_v3, "_FrozenClassifier", Broken)
    with pytest.raises(RuntimeError, match="swallowed"):
        execute(native)
    pointer = json.loads((native.root / "recovery" / "current.json").read_text())
    manifest = json.loads((native.root / "recovery" / pointer["generation"] / "recovery.json").read_text())
    assert manifest["state"]["event_prefix"]["completed"] == 0
    faults = [json.loads(line) for line in (native.root / "faults.jsonl").read_text().splitlines()]
    caught = [r for r in faults if r["context"]["stage"].endswith(":import_classify_failed")]
    assert len(caught) == 1 and caught[0]["error_type"] == "ValueError"
    assert caught[0]["context"]["turn"] == 1


@pytest.mark.parametrize("failure", ["responsive", "persistent"])
def test_worker_failure_disposition_keeps_source_and_correct_durable_prefix(native, monkeypatch, failure):
    from src.workers import post_flight
    from src.workers.completion_text import IncompleteCompletion
    from src.memory.models import EpisodicMemory
    original = post_flight.evaluate_turn
    attempts = []
    def fail(**kwargs):
        if kwargs["prompt"].startswith("Original choice 2:"):
            attempts.append(kwargs)
            if failure == "responsive":
                raise IncompleteCompletion("controlled incomplete output", finish_reason="length")
            raise ValueError("structural invariant invalid")
        return original(**kwargs)
    monkeypatch.setattr(post_flight, "evaluate_turn", fail)
    if failure == "responsive":
        state = execute(native)
        assert state["complete"] and state["event_prefix"]["completed"] == 3
        assert len(attempts) == 2 and state["worker_recovery"]["failed_attempts"] == 2
        assert len(state["worker_recovery"]["degraded"]) == 1
        with native.sessions() as db: assert db.query(EpisodicMemory).count() == 2
        written = [json.loads(line) for line in (native.root / "chronological.jsonl").read_text().splitlines()
                   if json.loads(line)["event"] == "source_written"]
        assert written[-1]["post_flight_complete"] is False
    else:
        with pytest.raises(ValueError, match="structural"): execute(native)
        assert len(attempts) == 1
        pointer = json.loads((native.root / "recovery" / "current.json").read_text())
        manifest = json.loads((native.root / "recovery" / pointer["generation"] / "recovery.json").read_text())
        assert manifest["state"]["event_prefix"]["completed"] == 2
        monkeypatch.setattr(post_flight, "evaluate_turn", original)
        assert execute(native, resume=True)["complete"]
    faults = [json.loads(line) for line in (native.root / "faults.jsonl").read_text().splitlines()]
    individual = [r for r in faults if r["context"]["stage"] == "post_flight"]
    assert [r["context"]["attempt"] for r in individual] == list(range(1, len(attempts) + 1))
    assert all(r["context"]["turn"] == 2 and r["context"]["batch_id"] for r in individual)


def test_failed_bootstrap_has_no_fabricated_anchor_and_resumes_missing_jobs_only(native, monkeypatch):
    from src.workers import compaction
    from src.workers.completion_text import IncompleteCompletion
    original = compaction.compact_entities
    def fail(): raise IncompleteCompletion("controlled bootstrap output failure", finish_reason="length")
    monkeypatch.setattr(compaction, "compact_entities", fail)
    with pytest.raises(RuntimeError, match="successful cadence anchor"):
        execute(native)
    pointer = json.loads((native.root / "recovery" / "current.json").read_text())
    manifest = json.loads((native.root / "recovery" / pointer["generation"] / "recovery.json").read_text())
    assert manifest["state"]["event_prefix"]["completed"] == 0
    assert manifest["state"]["last_maintenance"] == {"decay_episodic": START.isoformat()}
    monkeypatch.setattr(compaction, "compact_entities", original)
    assert execute(native, resume=True)["complete"]
    records = [json.loads(line) for line in (native.root / "chronological.jsonl").read_text().splitlines()]
    bootstrap = [r["job"] for r in records if r["event"] == "maintenance_bootstrap"]
    assert bootstrap == ["decay_episodic", "compaction"]


def test_probe_cutoff_stays_restorable_after_later_sources_and_complete_resume(native):
    from src.memory.models import EpisodicMemory
    state = execute(native)
    trace = native.root / "chronological.jsonl"
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    probe = next(r for r in records if r["event"] == "chronological_probe")
    completed = next(r for r in records if r["event"] == "chronological_event_complete" and r["event_id"] == "probe:first")
    target = native.root / "recovery" / "evaluation" / completed["evaluation_key"]
    manifest_bytes = (target / "recovery.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["state"]["event_prefix"]["completed"] == 2
    assert manifest["identity"]["probe_snapshots"] == "immutable_native_event_cutoffs"
    assert json.loads((target / "store.manifest.json").read_text())["sha256_by_table"] == probe["table_sha256"]
    final_rows, final_trace = snapshot.fingerprints(), trace.read_bytes()
    assert execute(native, resume=True) == state
    assert (target / "recovery.json").read_bytes() == manifest_bytes
    with patch.object(snapshot, "SNAPDIR", target):
        assert snapshot.restore("store") == 0
    assert snapshot.fingerprints() == probe["table_sha256"]
    with native.sessions() as db:
        assert db.query(EpisodicMemory).count() == 1
        assert db.query(EpisodicMemory).one().raw_text.startswith("User: Original choice 1:")
    with pytest.raises(ValueError, match="completed chronological trace/store changed"):
        execute(native, resume=True)
    identity = records[0]["identity"]
    cp = replay_checkpoint.Checkpoints(native.root / "recovery", trace, identity)
    try: assert cp.recover()["complete"]
    finally: cp.close()
    assert snapshot.fingerprints() == final_rows and trace.read_bytes() == final_trace
    assert execute(native, resume=True) == state
    assert (target / "recovery.json").read_bytes() == manifest_bytes
