"""Checkpointed ideal-cadence research adapter over native core maintenance.

No CLI: a new versioned parent owns source events, conditions, model pins and
prefix restores. This module never attaches to or upgrades an existing replay.
"""

import heapq
import importlib
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from experiments.v3_repair.caught_faults import observe as observe_caught
from experiments.v3_repair.fault_receipts import CONTEXT_FIELDS, FaultRecorder
from experiments.v3_repair.lifecycle_plan import cadence, clock
from experiments.v3_repair.prepare_inputs import digest, fingerprint
from scripts.z1 import replay_checkpoint, seed_v3, snapshot
from scripts.z1.historical_clock import historical_clock
from scripts.z1.worker_recovery import WorkerRecovery
from src.api.config import settings
from src.workers.runtime import JOBS

CORE_JOBS = seed_v3.MEMORY_JOBS


def enabled_roster(enabled):
    enabled = tuple(enabled)
    if len(set(enabled)) != len(enabled) or set(enabled) - set(CORE_JOBS):
        raise ValueError("unknown or duplicate chronological core job")
    return tuple(job for job in CORE_JOBS if job in enabled)


def checkpoint_groups(value):
    if type(value) is not int or not 1 <= value <= 32:
        raise ValueError("checkpoint group interval must be an integer from 1 to 32")
    return value


def identity(enabled=CORE_JOBS, *, clock_basis, checkpoint_every_groups=1):
    enabled = enabled_roster(enabled)
    if clock_basis not in {"original", "simulated"}:
        raise ValueError("declare chronological clock provenance")
    return dict(method="native_chronological_maintenance_v3", enabled=list(enabled),
        clock_basis=clock_basis, simultaneous_order="reverse_job_name", worker_attempts=1,
        checkpoint_every_groups=checkpoint_groups(checkpoint_every_groups),
        registry={job: asdict(JOBS[job]) for job in CORE_JOBS},
        settings_sha256=digest(settings.model_dump(mode="json")),
        native_code_sha256=replay_checkpoint.code_digest(),
        instruments=[fingerprint(Path(__file__)), fingerprint(Path(__file__).with_name("fault_receipts.py")),
                     fingerprint(Path(__file__).with_name("lifecycle_plan.py")),
                     fingerprint(Path(__file__).with_name("caught_faults.py"))])


def stamp(value):
    return clock(datetime.fromisoformat(value) if isinstance(value, str) else value)


def events(ledger, failed_at, until, enabled, *, event_limit=100_000):
    """Keep each job's actual anchor and remainder; never truncate a long gap."""
    until = stamp(until)
    if type(event_limit) is not int or event_limit < 1:
        raise ValueError("invalid chronological event limit")
    anchors, counts = {}, {}
    for name, value in [*ledger.items(), *failed_at.items()]:
        if name in CORE_JOBS and stamp(value) > until:
            raise ValueError("chronological ledger is later than the target clock")
    for job in enabled_roster(enabled):
        if job not in ledger:
            raise ValueError("enabled job has no known maintenance anchor")
        duration = cadence(settings.maintenance_intervals.get(job))
        anchors[job] = max(stamp(ledger[job]), stamp(failed_at.get(job, ledger[job])))
        counts[job] = (until - anchors[job]) // duration
    if sum(counts.values()) > event_limit:
        raise ValueError("chronological interval exceeds event limit; no jobs executed")
    pending = [(anchors[job] + cadence(settings.maintenance_intervals[job]), job, counts[job])
               for job in anchors if counts[job]]
    heapq.heapify(pending)
    while pending:
        target = pending[0][0]
        group = []
        while pending and pending[0][0] == target:
            _, job, remaining = heapq.heappop(pending)
            group.append(job)
            if remaining > 1:
                heapq.heappush(pending, (target + cadence(settings.maintenance_intervals[job]), job, remaining - 1))
        yield target, sorted(group, reverse=True)


def validate_parent(db, sink, state, checkpoints, recovery, faults, binding):
    seed_v3.isolated_database()
    engine = db.get_bind().engine
    if (not isinstance(checkpoints, replay_checkpoint.Checkpoints) or checkpoints.lock.closed
            or Path(sink.name).resolve() != checkpoints.trace.resolve()
            or engine.url.database != snapshot.DB
            or checkpoints.identity.get("database") != snapshot.DB
            or checkpoints.identity.get("chronological_maintenance") != binding):
        raise ValueError("chronological adapter differs from its owned parent identity/store/trace")
    if (not isinstance(recovery, WorkerRecovery) or recovery.policy != "continue" or recovery.sink is not sink
            or state.get("worker_recovery") is not recovery.state
            or not isinstance(state.get("last_maintenance"), dict)):
        raise ValueError("chronological parent recovery/ledger is incomplete")
    if not isinstance(faults, FaultRecorder) or faults.run_identity != digest(checkpoints.identity):
        raise ValueError("chronological fault observer belongs to a different parent")
    faults.verify()
    directory, manifest = checkpoints.validate()
    if json.loads(json.dumps(state, default=str)) != manifest["state"]:
        raise ValueError("restore the durable parent state before chronological continuation")
    if checkpoints.trace.stat().st_size != manifest["trace_bytes"]:
        raise ValueError("restore the durable trace before chronological continuation")
    store = json.loads((directory / "store.manifest.json").read_text())
    if snapshot.fingerprints() != store["sha256_by_table"]:
        raise ValueError("restore the durable store before chronological continuation")
    return engine


def drain(db, sink, *, until, state, checkpoints, recovery, faults, clock_basis,
          context, enabled=CORE_JOBS, event_limit=100_000, checkpoint_every_groups=1):
    """Execute native time groups; the caller must restore after any interruption.

    Completed batches and the final group are durable before outage checks.
    Incomplete groups leave the previous pointer; native recovery archives their
    trace suffix and replays the bounded uncommitted batch.
    Logged caught exceptions are observed; unlogged swallowed errors are not.
    """
    if set(context) - CONTEXT_FIELDS or not context.get("conversation") or type(context.get("turn")) is not int:
        raise ValueError("explicit frozen turn/conversation fault context required")
    enabled = enabled_roster(enabled)
    binding = identity(enabled, clock_basis=clock_basis,
                       checkpoint_every_groups=checkpoint_every_groups)
    schedule = list(events(state.get("last_maintenance", {}),
                           recovery.state["last_attempt"], until, enabled, event_limit=event_limit))
    engine = validate_parent(db, sink, state, checkpoints, recovery, faults, binding)
    functions = {}
    for job in enabled:
        module, function = JOBS[job].path.split(":")
        module = importlib.import_module(module)
        factory = getattr(module, "SessionLocal", None)
        if factory is not None:
            bound = getattr(factory, "kw", {}).get("bind")
            if bound is None or bound.url.database != engine.url.database:
                raise ValueError("native job factory differs from the owned parent store")
        functions[job] = getattr(module, function)
    totals = Counter()
    captures = 0
    for group_number, (target, jobs) in enumerate(schedule, 1):
        # All jobs in a time group must have complete observations before publishing.
        with historical_clock(target, engine):
            for job in jobs:
                spec = JOBS[job]
                kwargs = {"cycles": 1} if spec.pass_cycles else {}
                details = {**context, "job": job, "source_time": target.isoformat(),
                           "profile": binding["method"], "artifact_id": digest([target.isoformat(), job]),
                           "model": None}
                def observe():
                    return seed_v3.memory_state(db), seed_v3.semantic_state(db, job)
                observation_start = time.perf_counter()
                before, semantic_before = faults.call(observe, context={**details, "stage": "observer_before"})
                observer_ms = (time.perf_counter() - observation_start) * 1000
                fn = functions[job]
                def work():
                    with observe_caught(faults, {**details, "stage": "maintenance", "attempt": 1, "attempts": 1}):
                        return fn(db, **kwargs) if spec.needs_db else fn(**kwargs)
                start = time.perf_counter()
                succeeded, result = recovery.call(
                    lambda: faults.call(work, context={**details, "stage": "maintenance", "attempt": 1, "attempts": 1}),
                    job=job, context=context, rollback=db.rollback, attempts=1)
                elapsed = (time.perf_counter() - start) * 1000
                db.expire_all()
                observation_start = time.perf_counter()
                after, semantic_after = faults.call(observe, context={**details, "stage": "observer_after"})
                changes = faults.call(lambda: seed_v3.semantic_changes(semantic_before, semantic_after),
                                      context={**details, "stage": "observer_changes"})
                observer_ms += (time.perf_counter() - observation_start) * 1000
                sink.write(json.dumps(dict(event="maintenance", job=job,
                    trigger_conversation=context["conversation"], source_time_trigger=target.isoformat(),
                    call_kwargs=kwargs, before=before, after=after,
                    semantic_observed_tables=sorted(semantic_before), semantic_changes=changes,
                    observer_elapsed_ms=observer_ms, job_elapsed_ms=elapsed,
                    worker_status="complete" if succeeded else "degraded", result=result,
                    schedule_method=binding["method"], clock_basis=clock_basis,
                    fault_capture="uncaught_boundaries_and_logged_caught_exceptions"), default=str) + "\n")
                sink.flush()
                if succeeded: state["last_maintenance"][job] = target.isoformat()
                else: recovery.state["last_attempt"][job] = target.isoformat()
                totals[job + (":complete" if succeeded else ":degraded")] += 1
        boundary_context = {**context, "source_time": target.isoformat(), "profile": binding["method"],
                            "artifact_id": digest([target.isoformat(), jobs]), "model": None}
        outage = any(streak >= 3 for streak in recovery.state["streaks"].values())
        if (group_number % checkpoint_every_groups == 0
                or group_number == len(schedule) or outage):
            faults.call(lambda: checkpoints.capture(sink, state),
                        context={**boundary_context, "job": "checkpoint", "stage": "checkpoint"})
            captures += 1
            faults.call(recovery.check_outage,
                        context={**boundary_context, "job": "recovery", "stage": "outage_check"})
    return dict(method=binding["method"], until=stamp(until).isoformat(), clock_basis=clock_basis,
        completed_time_groups=len(schedule), calls=dict(totals), disabled=[j for j in CORE_JOBS if j not in enabled],
        checkpoint_every_groups=checkpoint_every_groups, published_checkpoints=captures,
        max_uncommitted_complete_groups=checkpoint_every_groups - 1,
        score_of_record=False, semantic_quality="unqualified",
        fault_capture="uncaught_boundaries_and_logged_caught_exceptions")
