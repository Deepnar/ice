"""Chronological native aging, eligibility and exposure/cap opposing controls."""

import copy
import io
import json
import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from experiments.v3_repair.lifecycle_plan import FIELDS, JOBS, build, decay_events
from experiments.v3_repair.prepare_inputs import write_or_verify
from src.api.config import settings

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def profile():
    return {name: getattr(settings, name) for name in FIELDS}


def intervals():
    return {job: settings.maintenance_intervals[job] for job in JOBS}


def test_actual_baseline_daily_units_cap_sensitivity_and_immutable_export(tmp_path):
    value = build(profile(), intervals(), START, START + timedelta(days=30), clock_basis="simulated")
    assert len(value["schedule"]) == sum(r["due_cycles"] for r in value["endpoint_dispatch"].values())
    epi = value["endpoint_dispatch"]["decay_episodic"]
    assert epi == dict(due_cycles=480, endpoint_cycles=96, omitted_cycles=384,
                      capped_aging_days=6, remainder_seconds=0, semantics="numeric_retention")
    assert value["rates"]["unaccessed"]["half_life_eligible_days"] == pytest.approx(13.513407)
    assert value["rates"]["creative"]["cold_eligible_days"] is None
    assert value["rates"]["unaccessed"]["daily_sensitivity"]["half_duration"] == pytest.approx(.95 ** 2)
    assert value["exposure"]["episodic"]["uninterrupted_exposures_to_ceiling"] == 6
    cap_rows = value["exposure"]["graph"]["cap_sensitivity"]
    assert [r["cap"] for r in cap_rows] == [5, 10, 20]
    assert [r["bonus_factor_at_unchanged_strength1"] for r in cap_rows] == [1.1, 1.05, 1.025]
    out = tmp_path / "plan.json"
    write_or_verify(out, value)
    saved = out.read_bytes()
    write_or_verify(out, value)
    changed = copy.deepcopy(value); changed["policy"]["decay_strengthen_amount"] *= 2
    with pytest.raises(ValueError): write_or_verify(out, changed)
    assert out.read_bytes() == saved


def test_calendar_offsets_fractional_remainder_and_unequal_cadences():
    durations = dict(decay_episodic=3600, decay_codex=7200, decay_procedural=5400)
    end = (START + timedelta(seconds=10_001)).astimezone(timezone(timedelta(hours=5, minutes=30)))
    value = build(profile(), durations, START, end, clock_basis="original")
    assert [r["job"] for r in value["schedule"]] == ["decay_episodic", "decay_procedural", "decay_episodic", "decay_codex"]
    assert value["endpoint_dispatch"]["decay_episodic"]["remainder_seconds"] == 2801
    assert value["endpoint_dispatch"]["decay_procedural"]["omitted_cycles"] is None
    assert list(decay_events(START, START + timedelta(seconds=1), durations)) == []


@pytest.mark.parametrize("fault", ["naive", "reverse", "missing_job", "zero_cadence", "bad_rate",
                                  "bad_thresholds", "bad_cap", "unknown_clock", "too_large"])
def test_invalid_or_unbounded_clock_profiles_refuse_before_output(fault):
    policy, durations, start, end = profile(), intervals(), START, START + timedelta(days=1)
    basis, limit = "simulated", 200_000
    if fault == "naive": start = START.replace(tzinfo=None)
    if fault == "reverse": end = START
    if fault == "missing_job": del durations["decay_codex"]
    if fault == "zero_cadence": durations["decay_episodic"] = 0
    if fault == "bad_rate": policy["decay_daily_unaccessed"] = float("nan")
    if fault == "bad_thresholds": policy["decay_cold_threshold"] = .9
    if fault == "bad_cap": policy["runtime_cycles_cap"] = True
    if fault == "unknown_clock": basis = "inferred"
    if fault == "too_large": limit = 1
    with pytest.raises(ValueError): build(policy, durations, start, end, clock_basis=basis, event_limit=limit)


@pytest.mark.parametrize("mode", ["capped_endpoint", "uncapped_endpoint", "chronological"])
def test_real_native_workers_old_and_new_eligibility_are_not_endpoint_equivalent(
        empty_replay_database, monkeypatch, tmp_path, mode):
    from sqlalchemy import create_engine
    from src.api.db import SessionLocal
    from src.memory.models import CodexEdge, CodexEntity, ColdStorage, Conversation, EpisodicMemory
    from src.workers import codex_decay, decay
    from src.workers.runtime import missed_cycles
    from scripts.z1.historical_clock import historical_clock
    from scripts.z1 import seed_v3

    monkeypatch.setattr(decay, "SessionLocal", SessionLocal)
    monkeypatch.setattr(codex_decay, "SessionLocal", SessionLocal)
    for name, value in dict(decay_daily_unaccessed=.95, decay_daily_accessed=.98,
            decay_daily_creative=.99, decay_creative_floor=.3, decay_archive_threshold=.1,
            decay_cold_threshold=.05, codex_decay_daily=.99, codex_retention_floor=.1,
            runtime_cycles_cap=96).items(): monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(settings, "maintenance_intervals", {**settings.maintenance_intervals,
        **dict.fromkeys(JOBS, 5400)})
    with SessionLocal() as db:
        conv = Conversation(); db.add(conv); db.flush()
        turns = {}
        for kind, age, score, accesses, topics, protected in (
            ("old", 8, .08, 0, [], False), ("boundary", 1, 1, 0, [], False),
            ("accessed", 8, 1, 1, [], False), ("creative", 8, .31, 0, ["Creative_&_Media"], False),
            ("protected", 8, 1, 0, [], True)):
            raw = "USER: " + kind + " original value7319\nASSISTANT: acknowledged"
            row = EpisodicMemory(conversation_id=conv.id, batch_id=uuid.uuid4(),
                raw_text=raw, source_spans={"controlled": kind}, timestamp=START - timedelta(days=age),
                idempotency_key="lifecycle_" + uuid.uuid4().hex,
                context_reliance="independent", decay_score=score, access_count=accesses,
                topic_tags=topics, is_bookmarked=protected)
            db.add(row); db.flush(); turns[kind] = (row.id, raw, row.source_spans, row.timestamp)
        a, b = [CodexEntity(canonical_name="clock_" + uuid.uuid4().hex) for _ in range(2)]
        db.add_all([a, b]); db.flush()
        edge = CodexEdge(source_id=a.id, target_id=b.id, relation="uses", source_batch=uuid.uuid4(),
                         strength=1, confidence="pending", extraction_confidence=.9)
        db.add(edge); db.commit(); edge_id, conversation_id = edge.id, conv.id
    end = START + timedelta(days=10)
    engine = create_engine(settings.database_url)
    try:
        if mode == "chronological":
            for event in decay_events(START, end, intervals()):
                if event["job"] == "decay_procedural": continue
                with historical_clock(datetime.fromisoformat(event["source_time"]), engine):
                    (decay.apply_decay if event["job"] == "decay_episodic" else codex_decay.decay_codex_edges)(cycles=1)
        else:
            if mode == "uncapped_endpoint": monkeypatch.setattr(settings, "runtime_cycles_cap", 480)
            count = missed_cycles(10 * 86400, 5400, settings.runtime_cycles_cap)
            with historical_clock(end, engine):
                if mode == "capped_endpoint":
                    from src.workers import procedural_decay
                    monkeypatch.setattr(procedural_decay, "SessionLocal", SessionLocal)
                    monkeypatch.setattr(seed_v3, "MEMORY_JOBS", JOBS)
                    ledger = dict.fromkeys(JOBS, START)
                    sink = io.StringIO()
                    with SessionLocal() as db:
                        seed_v3.due_maintenance(db, sink, end, conversation_id, ledger)
                    records = [json.loads(line) for line in sink.getvalue().splitlines()]
                    assert {r["job"] for r in records} == set(JOBS)
                    assert all(r["call_kwargs"] == {"cycles": 96} and r["worker_status"] == "complete" for r in records)
                    assert set(ledger.values()) == {end}
                    # Omitted earlier cycles are no longer represented by this ledger.
                    assert missed_cycles((end + timedelta(seconds=5400) - ledger["decay_episodic"]).total_seconds(), 5400, 96) == 1
                else:
                    decay.apply_decay(cycles=count); codex_decay.decay_codex_edges(cycles=count)
        with SessionLocal() as db:
            observed = {}
            for kind, (row_id, raw, spans, stamp) in turns.items():
                warm, cold = db.get(EpisodicMemory, row_id), db.get(ColdStorage, row_id)
                assert bool(warm) != bool(cold)
                row = warm or cold
                assert (row.raw_text, row.source_spans, row.timestamp) == (raw, spans, stamp)
                observed[kind] = dict(score=row.decay_score, location="warm" if warm else "cold")
                if cold: observed[kind]["cold_at"] = cold.archived_at.isoformat()
            effective_old_days = 6 if mode == "capped_endpoint" else 10
            boundary_days = 4 if mode == "chronological" else effective_old_days
            # Cold rows stop warm aging at their first strict threshold crossing.
            crossing_cycles = math.floor(math.log(.05 / .08) / math.log(.95 ** (1 / 16))) + 1
            old_days = crossing_cycles / 16 if mode == "chronological" else effective_old_days
            assert observed["old"]["score"] == pytest.approx(.08 * .95 ** old_days)
            assert observed["old"]["location"] == ("warm" if mode == "capped_endpoint" else "cold")
            if mode != "capped_endpoint":
                expected_cold_at = START + timedelta(days=old_days)
                assert datetime.fromisoformat(observed["old"]["cold_at"]) == expected_cold_at
            assert observed["boundary"]["score"] == pytest.approx(.95 ** boundary_days)
            assert observed["accessed"]["score"] == pytest.approx(.98 ** effective_old_days)
            assert observed["creative"]["score"] == pytest.approx(.3)
            assert observed["protected"]["score"] == 1
            edge = db.get(CodexEdge, edge_id)
            assert edge.strength == pytest.approx(.99 ** effective_old_days)
            assert edge.confidence == "pending" and edge.extraction_confidence == .9
            assert edge.valid_until is edge.unlearned_at is None
            (tmp_path / ("native-" + mode + ".json")).write_text(
                json.dumps(dict(mode=mode, observed=observed, graph_strength=edge.strength)))
    finally:
        engine.dispose()


def test_actual_exposure_zero_amount_dedup_switch_and_cap_rank_coupling(empty_replay_database, monkeypatch):
    from src.api.db import SessionLocal
    from src.memory.models import CodexEdge, CodexEntity, Conversation, EpisodicMemory
    from src.memory.usage import record_graph_access
    from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator
    monkeypatch.setattr(settings, "retrieval_strengthen_writes", True)
    monkeypatch.setattr(settings, "decay_strengthen_amount", .15)
    monkeypatch.setattr(settings, "codex_retention_increment", .15)
    monkeypatch.setattr(settings, "codex_retention_cap", 10)
    monkeypatch.setattr(settings, "codex_retention_rank_weight", .5)
    with SessionLocal() as db:
        conv = Conversation(); db.add(conv); db.flush()
        turn = EpisodicMemory(conversation_id=conv.id, batch_id=uuid.uuid4(), raw_text="USER: saved",
            idempotency_key="exposure_" + uuid.uuid4().hex,
            context_reliance="independent", timestamp=START, decay_score=.1, access_count=0)
        a, b = [CodexEntity(canonical_name="exposure_" + uuid.uuid4().hex) for _ in range(2)]
        db.add_all([turn, a, b]); db.flush()
        edge = CodexEdge(source_id=a.id, target_id=b.id, relation="uses", source_batch=uuid.uuid4(),
            strength=9.9, confidence="pending", extraction_confidence=.9, valid_from=START)
        weak = CodexEdge(source_id=a.id, target_id=b.id, relation="likes", source_batch=uuid.uuid4(),
            strength=10, confidence="pending", extraction_confidence=.35, valid_from=START)
        db.add_all([edge, weak]); db.commit()
        orchestrator = HybridRetrievalOrchestrator(db, None)
        episodic = ContextFragment("saved", "episodic", 1, 1, source_batch_id=str(turn.id))
        graph = ContextFragment("fact", "codex", 1, 1, origin_edge_ids=(str(edge.id),))
        orchestrator._strengthen_retrieved([episodic, episodic]); db.refresh(turn)
        assert turn.access_count == 1 and turn.decay_score == pytest.approx(.25)
        monkeypatch.setattr(settings, "decay_strengthen_amount", 0)
        orchestrator._strengthen_retrieved([episodic]); db.refresh(turn)
        assert turn.access_count == 2 and turn.decay_score == pytest.approx(.25)
        record_graph_access(db, [graph, graph], stage="prompt_prepared"); db.refresh(edge)
        assert edge.usage_count == 1 and edge.strength == 10 and edge.extraction_confidence == .9
        score_before = orchestrator._edge_trust(edge)
        monkeypatch.setattr(settings, "codex_retention_cap", 20)
        assert orchestrator._edge_trust(edge) < score_before
        assert orchestrator._edge_trust(weak) == .35
        monkeypatch.setattr(settings, "retrieval_strengthen_writes", False)
        orchestrator._strengthen_retrieved([episodic]); record_graph_access(db, [graph], stage="context_returned")
        db.refresh(turn); db.refresh(edge)
        assert turn.access_count == 2 and edge.usage_count == 1
