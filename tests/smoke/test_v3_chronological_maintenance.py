"""Native ten-job SQL effects and checkpointed interruption without model calls."""

import importlib
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import text

from experiments.v3_repair import chronological_maintenance as cm
from experiments.v3_repair.fault_receipts import FaultRecorder
from experiments.v3_repair.prepare_inputs import digest
from scripts.z1 import replay_checkpoint, snapshot
from scripts.z1.worker_recovery import WorkerOutage, WorkerRecovery
from src.api.config import settings

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def parent(empty_replay_database, monkeypatch, tmp_path):
    from src.api.db import SessionLocal
    from src.memory.models import CodexEdge, CodexEntity, Conversation, EpisodicMemory, ProceduralMemory
    from scripts.z1.historical_clock import CLOCK_MODULES
    for name in CLOCK_MODULES: importlib.import_module(name)
    # Cached module factories must never escape this nested disposable store.
    for name, module in list(sys.modules.items()):
        if name.startswith("src.") and hasattr(module, "SessionLocal"):
            monkeypatch.setattr(module, "SessionLocal", SessionLocal)
    for name, value in dict(decay_daily_unaccessed=.95, decay_daily_accessed=.98,
            decay_daily_creative=.99, decay_creative_floor=.3, decay_archive_threshold=.1,
            decay_cold_threshold=.05, codex_decay_daily=.99, codex_retention_floor=.1,
            runtime_cycles_cap=96, procedural_stale_days=180, procedural_min_reinforcement=3).items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(settings, "maintenance_intervals", {**settings.maintenance_intervals,
        **dict.fromkeys(cm.CORE_JOBS, 3600)})
    http_attempts = []
    def deny(*args, **kwargs):
        http_attempts.append(1)
        raise AssertionError("no model HTTP request permitted in this mechanism fixture")
    async def async_deny(*args, **kwargs): return deny(*args, **kwargs)
    monkeypatch.setattr(httpx.Client, "request", deny)
    monkeypatch.setattr(httpx.AsyncClient, "request", async_deny)
    with SessionLocal() as db:
        conv = Conversation(); db.add(conv); db.flush()
        row = EpisodicMemory(conversation_id=conv.id, batch_id=uuid.uuid4(),
            idempotency_key="chronology_" + uuid.uuid4().hex, raw_text="USER: original value7319",
            source_spans={"controlled": "original"}, context_reliance="independent",
            timestamp=START - timedelta(days=8), decay_score=.051, is_private=True)
        a, b = [CodexEntity(canonical_name="chronology_" + uuid.uuid4().hex) for _ in range(2)]
        db.add_all([row, a, b]); db.flush()
        edge = CodexEdge(source_id=a.id, target_id=b.id, relation="uses", source_batch=row.batch_id,
            strength=1, confidence="pending", extraction_confidence=.9, valid_from=START)
        proc = ProceduralMemory(pattern_name="controlled legacy pattern", reinforcement_count=1,
            first_observed=START - timedelta(days=190), last_observed=START - timedelta(days=181), is_active=True)
        db.add_all([edge, proc]); db.commit()
        identifiers = dict(conversation=str(conv.id), row=row.id, edge=edge.id, procedure=proc.id)
    parents = []
    def make(label="main", enabled=cm.CORE_JOBS):
        root = tmp_path / label; root.mkdir()
        trace = root / "trace.jsonl"
        trace.write_text(json.dumps(dict(event="controlled_original", turn=1)) + "\n")
        sink = trace.open("a")
        recovery = WorkerRecovery(sink, "continue")
        state = dict(completed={"fixture": 1}, probe_count=0,
            last_maintenance={job: START.isoformat() for job in cm.CORE_JOBS}, worker_recovery=recovery.state)
        bound = dict(database=snapshot.DB, chronological_maintenance=cm.identity(enabled, clock_basis="simulated"))
        cp = replay_checkpoint.Checkpoints(root / "recovery", trace, bound)
        cp.capture(sink, state, evaluation_key="initial-prefix")
        faults = FaultRecorder(root / "faults.jsonl", digest(bound), private_root=tmp_path)
        item = NS(root=root, trace=trace, sink=sink, cp=cp, recovery=recovery, state=state,
            faults=faults, enabled=enabled, ids=identifiers, http_attempts=http_attempts, sessions=SessionLocal)
        parents.append(item)
        return item
    main = make()
    main.make = make
    yield main
    for item in parents:
        item.sink.close(); item.cp.close()
    assert http_attempts == []


def run(parent, hours=12):
    with parent.sessions() as db:
        return cm.drain(db, parent.sink, until=START + timedelta(hours=hours),
            state=parent.state, checkpoints=parent.cp, recovery=parent.recovery,
            faults=parent.faults, enabled=parent.enabled, clock_basis="simulated",
            context=dict(conversation=parent.ids["conversation"], history="controlled", turn=2, arm="full"))


def test_all_ten_native_jobs_actual_effects_and_cached_groups_do_not_repeat(parent):
    from src.memory.models import CodexEdge, ColdStorage, EpisodicMemory, ProceduralMemory
    result = run(parent)
    assert result["completed_time_groups"] == 12 and result["disabled"] == []
    assert result["calls"] == {job + ":complete": 12 for job in cm.CORE_JOBS}
    rows = [json.loads(line) for line in parent.trace.read_text().splitlines()][1:]
    assert len(rows) == 120 and all(r["worker_status"] == "complete" for r in rows)
    assert all(r["call_kwargs"] == ({"cycles": 1} if cm.JOBS[r["job"]].pass_cycles else {}) for r in rows)
    with parent.sessions() as db:
        assert db.get(EpisodicMemory, parent.ids["row"]) is None
        cold = db.get(ColdStorage, parent.ids["row"])
        assert cold.raw_text == "USER: original value7319" and cold.source_spans == {"controlled": "original"}
        assert cold.archived_at == START + timedelta(hours=10)
        edge = db.get(CodexEdge, parent.ids["edge"])
        assert edge.strength == pytest.approx(.99 ** .5) and edge.confidence == "pending"
        assert edge.valid_until is None and edge.extraction_confidence == .9
        assert db.get(ProceduralMemory, parent.ids["procedure"]).is_active is False
    saved = parent.trace.read_bytes(); fingerprints = snapshot.fingerprints()
    again = run(parent)
    assert again["completed_time_groups"] == 0 and again["calls"] == {}
    assert parent.trace.read_bytes() == saved and snapshot.fingerprints() == fingerprints


def test_real_snapshot_interrupt_mid_group_restores_and_matches_uninterrupted(parent, monkeypatch):
    from src.workers import decay
    # First get the same native path's uninterrupted target, then restore its
    # immutable initial evaluation snapshot for the interrupted counterpart.
    run(parent, hours=4)
    expected = snapshot.fingerprints()
    with patch.object(snapshot, "SNAPDIR", parent.cp.root / "evaluation/initial-prefix"):
        assert snapshot.restore("store") == 0
    interrupted = parent.make("interrupted")
    original = decay.apply_decay
    raised = []
    def break_after_real_write(cycles=1):
        original(cycles)
        if cm.stamp(decay.datetime.now(timezone.utc)) == START + timedelta(hours=2) and not raised:
            raised.append(1)
            raise KeyboardInterrupt("controlled interruption after committed native SQL")
    monkeypatch.setattr(decay, "apply_decay", break_after_real_write)
    with pytest.raises(KeyboardInterrupt): run(interrupted, hours=4)
    _, saved = interrupted.cp.validate()
    assert set(saved["state"]["last_maintenance"].values()) == {(START + timedelta(hours=1)).isoformat()}
    assert interrupted.faults.verify() == 1
    interrupted.sink.close()
    restored = interrupted.cp.recover()
    assert list(interrupted.cp.root.glob("unfinished-*.jsonl"))
    interrupted.sink = interrupted.trace.open("a")
    interrupted.state = restored
    interrupted.recovery = WorkerRecovery(interrupted.sink, "continue", restored["worker_recovery"])
    resumed = run(interrupted, hours=4)
    assert resumed["completed_time_groups"] == 3
    assert snapshot.fingerprints() == expected
    records = [json.loads(line) for line in interrupted.trace.read_text().splitlines()][1:]
    assert len(records) == 40 and all(r["worker_status"] == "complete" for r in records)
    assert len({(r["job"], r["source_time_trigger"]) for r in records}) == 40


def test_changed_profile_ledger_trace_and_store_refuse_before_any_job(parent, monkeypatch):
    from src.workers import decay
    monkeypatch.setattr(decay, "apply_decay", lambda **kw: pytest.fail("invalid parent executed a job"))
    saved = parent.trace.read_bytes()
    parent.state["last_maintenance"]["decay_episodic"] = (START + timedelta(days=3)).isoformat()
    with pytest.raises(ValueError, match="later than"): run(parent)
    parent.state["last_maintenance"]["decay_episodic"] = START.isoformat()
    with monkeypatch.context() as changed:
        changed.setattr(settings, "decay_daily_unaccessed", .9)
        with pytest.raises(ValueError, match="identity"): run(parent)
    parent.sink.write('{"event":"uncommitted"}\n'); parent.sink.flush()
    with pytest.raises(ValueError, match="durable trace"): run(parent)
    parent.sink.close(); restored = parent.cp.recover()
    parent.sink = parent.trace.open("a")
    parent.state = restored; parent.recovery = WorkerRecovery(parent.sink, "continue", restored["worker_recovery"])
    with parent.sessions() as db:
        db.execute(text("UPDATE episodic_memory SET decay_score=.5")); db.commit()
    with pytest.raises(ValueError, match="durable store"): run(parent)
    assert parent.trace.read_bytes() == saved and parent.faults.verify() == 0


def test_transport_faults_save_complete_receipts_and_pause_after_durable_third(parent, monkeypatch):
    from src.workers import decay
    def fail(cycles=1):
        raise ConnectionError("complete controlled failure " + "x" * 1800)
    monkeypatch.setattr(decay, "apply_decay", fail)
    with pytest.raises(WorkerOutage): run(parent, hours=8)
    assert parent.faults.verify() == 4  # Three native faults and their derived pause.
    _, saved = parent.cp.validate()
    assert saved["state"]["worker_recovery"]["streaks"]["decay_episodic"] == 3
    assert saved["state"]["last_maintenance"]["decay_episodic"] == START.isoformat()
    assert saved["state"]["worker_recovery"]["last_attempt"]["decay_episodic"] == (START + timedelta(hours=3)).isoformat()
    faults = [json.loads(line) for line in parent.faults.path.read_text().splitlines()]
    assert all(len(r["error"]) > 1800 and r["traceback"] and r["context"]["turn"] == 2 for r in faults[:3])
    assert all(r["context"]["attempt"] == r["context"]["attempts"] == 1 for r in faults[:3])
    assert faults[3]["error_type"] == "WorkerOutage" and faults[3]["context"]["stage"] == "outage_check"
    # Recovery after an operator fixes the outage resumes only the next boundary.
    monkeypatch.setattr(decay, "apply_decay", lambda cycles=1: None)
    result = run(parent, hours=4)
    assert result["completed_time_groups"] == 1
    assert parent.recovery.state["streaks"]["decay_episodic"] == 0


def test_disabled_job_subset_keeps_unexecuted_ledgers_and_state(parent):
    selected = parent.make("decay-only", enabled=("decay_codex",))
    result = run(selected, hours=2)
    assert result["calls"] == {"decay_codex:complete": 2}
    assert set(result["disabled"]) == set(cm.CORE_JOBS) - {"decay_codex"}
    assert all(selected.state["last_maintenance"][job] == START.isoformat() for job in result["disabled"])
    from src.memory.models import EpisodicMemory, ProceduralMemory
    with selected.sessions() as db:
        assert db.get(EpisodicMemory, selected.ids["row"]).decay_score == .051
        assert db.get(ProceduralMemory, selected.ids["procedure"]).is_active


@pytest.mark.parametrize("failure", ["observer_after", "checkpoint"])
def test_failed_observation_or_checkpoint_retains_prior_prefix_and_real_sql_restore(parent, monkeypatch, failure):
    from src.memory.models import ProceduralMemory
    before = snapshot.fingerprints()
    pointer = parent.cp.pointer.read_bytes()
    if failure == "observer_after":
        original = cm.seed_v3.semantic_state
        calls = []
        def observe(db, job):
            if job == "decay_procedural":
                calls.append(1)
                if len(calls) == 2: raise ConnectionError("controlled observer failure after real procedure retirement")
            return original(db, job)
        monkeypatch.setattr(cm.seed_v3, "semantic_state", observe)
        error = ConnectionError
    else:
        monkeypatch.setattr(snapshot, "save", lambda *_a, **_k: 1)
        error = RuntimeError
    with pytest.raises(error): run(parent, hours=1)
    assert parent.cp.pointer.read_bytes() == pointer
    assert parent.faults.verify() == 1 and parent.recovery.state["degraded"] == []
    record = json.loads(parent.faults.path.read_text())
    assert record["context"]["stage"] == failure and record["traceback"]
    with parent.sessions() as db:
        assert db.get(ProceduralMemory, parent.ids["procedure"]).is_active is False
    # Fatal observation/storage faults cannot be accepted as degraded success.
    parent.sink.close(); parent.cp.recover()
    assert snapshot.fingerprints() == before
    parent.sink = parent.trace.open("a")


def test_event_anchors_remainders_missing_or_invalid_roster(monkeypatch):
    monkeypatch.setattr(settings, "maintenance_intervals", {"decay_episodic": 3600, "decay_codex": 7200})
    ledger = dict(decay_episodic=START, decay_codex=START + timedelta(minutes=30))
    result = list(cm.events(ledger, {}, START + timedelta(hours=3), ledger))
    assert result == [(START + timedelta(hours=1), ["decay_episodic"]),
        (START + timedelta(hours=2), ["decay_episodic"]),
        (START + timedelta(hours=2, minutes=30), ["decay_codex"]),
        (START + timedelta(hours=3), ["decay_episodic"])]
    with pytest.raises(ValueError, match="unknown or duplicate"): cm.enabled_roster(["decay_codex", "decay_codex"])
    with pytest.raises(ValueError, match="known maintenance"): list(cm.events({}, {}, START, ["decay_codex"]))
    with pytest.raises(ValueError, match="event limit"): list(cm.events(ledger, {}, START + timedelta(days=100), ledger, event_limit=1))
    with pytest.raises(ValueError, match="later than"):
        list(cm.events({"decay_codex": START + timedelta(days=5)}, {"decay_codex": START}, START + timedelta(days=1), ["decay_codex"]))
