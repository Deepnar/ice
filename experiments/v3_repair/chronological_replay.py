"""New-store native v3 chronological integration, without a launch CLI.

One full-memory history; diagnostic arms are read-only preparations, not writer
ablations. No generated historical replies, cloud answers, tuning or promotion.
"""

import importlib
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from experiments.v3_repair import chronological_maintenance as maintenance
from experiments.v3_repair import chronological_sources as sources
from experiments.v3_repair.caught_faults import observe
from experiments.v3_repair.fault_receipts import FaultRecorder
from experiments.v3_repair.prepare_inputs import ROOT, digest, fingerprint, local_model_profile
from scripts.z1 import replay_checkpoint, seed_v3, snapshot
from scripts.z1.historical_clock import CLOCK_MODULES, historical_clock
from scripts.z1.worker_recovery import WorkerRecovery
from src.api.config import settings
from src.api.db import SessionLocal
from src.ingestion.importer import _store_turn
from src.memory.models import Base, Conversation, EpisodicMemory
from src.memory.source import chat_provenance
from src.memory.source_order import attach_sequence, sequence_receipt, verified_sequence

POLICY = "ice-v3-native-chronological-parent-1"


def input_projection(plan, originals, probes):
    sources.validate(plan)
    if "frozen_writer" in plan["identity"]:
        raise ValueError("a frozen-reader audit is not a new-writer execution plan")
    if any(e["clock"] != e["original_time"] for e in plan["events"] if e["kind"] == "source"):
        raise ValueError("replacement aging clocks need separate source/calendar qualification")
    rebuilt = sources.build(plan["identity"]["history"], originals, probes,
                            clock_basis=plan["identity"]["clock_basis"])
    if rebuilt["identity"] != plan["identity"]:
        raise ValueError("chronological source/probe plan differs from complete supplied inputs")
    by_id = {probe["probe_id"]: probe for _, probe in probes}
    runtime = []
    for i, event in enumerate(plan["events"]):
        argument = (dict(source=originals[event["turn"] - 1]) if event["kind"] == "source"
                    else dict(probe=by_id[event["probe_id"]]))
        runtime.append(sources.runtime_projection(plan, i, **argument))
    return runtime, by_id


def model_pins():
    from src.workers.bg_client_factory import get_bg_model_name
    background = get_bg_model_name()
    return dict(background=background,
                writer_digests=replay_checkpoint.background_model_digests(settings, background),
                declared=local_model_profile(settings),
                files=[fingerprint(Path(value).resolve()) for value in
                       (settings.classifier_model_path, settings.label_schema_path)])


def components():
    from src.classifier.classifier import PyTorchClassifier
    from src.memory.embedder import get_embedder
    return (PyTorchClassifier(model_path=settings.classifier_model_path,
                              schema_path=settings.label_schema_path), get_embedder())


def native_functions(db):
    """Load clock/logger consumers and refuse cached production-store factories."""
    for name in CLOCK_MODULES:
        importlib.import_module(name)
    engine = db.get_bind().engine
    for name, module in list(sys.modules.items()):
        if name.startswith("src.") and hasattr(module, "SessionLocal"):
            bound = getattr(module.SessionLocal, "kw", {}).get("bind")
            if bound is None or bound.url != engine.url:
                raise ValueError("native session factory differs from chronological store: " + name)
    result = {}
    for job in maintenance.CORE_JOBS:
        module, function = maintenance.JOBS[job].path.split(":")
        result[job] = getattr(importlib.import_module(module), function)
    return result


def emit(sink, **row):
    sink.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    sink.flush()


def bootstrap(db, sink, state, cp, recovery, faults, functions, first, enabled):
    """Only actual successful calls establish initial cadence anchors."""
    for job in sorted(enabled, reverse=True):
        if job in state["last_maintenance"]:
            continue
        spec = maintenance.JOBS[job]
        kwargs = dict(cycles=1) if spec.pass_cycles else {}
        context = dict(history=first["history"], conversation=state["conversation_id"],
                       turn=0, job=job, stage="bootstrap", source_time=first["clock"], arm="full",
                       attempt=1, attempts=1)
        def work():
            with historical_clock(datetime.fromisoformat(first["clock"]), db.get_bind().engine):
                with observe(faults, context):
                    fn = functions[job]
                    return fn(db, **kwargs) if spec.needs_db else fn(**kwargs)
        ok, result = recovery.call(lambda: faults.call(work, context=context),
                                  job=job, context=context, rollback=db.rollback, attempts=1)
        if not ok:
            db.rollback()
            cp.capture(sink, state)
            raise RuntimeError("bootstrap job failed without a successful cadence anchor: " + job)
        db.rollback()
        state["last_maintenance"][job] = first["clock"]
        emit(sink, event="maintenance_bootstrap", job=job, source_time=first["clock"],
             result=result, call_kwargs=kwargs, worker_status="complete", clock_basis=first["clock_basis"])
        cp.capture(sink, state)
    state["bootstrapped"] = True
    cp.capture(sink, state)


def preparation(db, event, conv, classifier, embedder, *, arm, exposure):
    pre = seed_v3.pp.build(db, event.get("question", event.get("prompt")), conv,
                          classifier, embedder, source_time=datetime.fromisoformat(event["clock"]))
    memory, generated, ranked, budgeted, notes = seed_v3.prepare_with_trace(
        db, pre, classifier, arm=arm, source_time=datetime.fromisoformat(event["clock"]),
        graph_exposure=exposure)
    return pre, seed_v3.stage_record(pre, memory, generated, ranked, budgeted, notes, include_prompt=True)


def write_source(db, sink, state, event, classifier, embedder, background, recovery, faults):
    from src.workers.post_flight import evaluate_turn
    stamp = datetime.fromisoformat(event["clock"])
    conv = uuid.UUID(state["conversation_id"])
    context = dict(history=event["history"], conversation=str(conv), turn=event["turn"],
                   source_time=event["clock"], arm="full", job="source", stage="preflight")
    seen = {key: datetime.fromisoformat(value) for key, value in state["seen_source_times"].items()}
    with historical_clock(stamp, db.get_bind().engine), observe(faults, context):
        pre, stage = faults.call(lambda: preparation(db, event, conv, classifier, embedder,
                                                    arm="full", exposure=True), context=context)
        stage["lineage"] = seed_v3.lineage_at_cutoff(stage, seen, stamp)
        emit(sink, event="source_preflight", event_id=event["id"], turn=event["turn"], preflight=stage)
        raw = f"User: {event['prompt']}\n\nAssistant: {event['response']}"
        receipt = sequence_receipt(raw, event["source_sequence_sha256"], event["turn"])
        key = digest([POLICY, event["source_sequence_sha256"], event["turn"]])
        frozen = seed_v3._FrozenClassifier(pre.classification, event["prompt"], conv)
        failures = []
        class CheckedClassifier:
            def classify(self, *args, **kwargs):
                try:
                    return frozen.classify(*args, **kwargs)
                except BaseException as exc:
                    failures.append(exc)
                    faults.record(exc, {**context, "stage": "store_classifier"})
                    raise
        context["stage"] = "store"
        stored = faults.call(lambda: _store_turn(db, conv, event["prompt"], event["response"], stamp,
            key, "fresh", stamp, event["provenance"], CheckedClassifier(), embedder, source_sequence=receipt),
            context={**context, "stage": "store"})
        if failures:
            raise RuntimeError("importer swallowed a frozen-preflight classification failure") from failures[0]
        if stored is None:
            raise RuntimeError("unexpected duplicate original in uncompleted event")
        batch_id, prompt, response = stored
        calls = 0
        def postflight():
            nonlocal calls
            calls += 1
            details = {**context, "job": "post_flight", "stage": "post_flight",
                       "batch_id": str(batch_id), "model": background, "attempt": calls, "attempts": 2}
            context.update(details)
            return faults.call(lambda: evaluate_turn(batch_id=str(batch_id), prompt=prompt, response=response,
                conversation_id=str(conv), model_used=background), context=details)
        ok, _ = recovery.call(postflight, job="post_flight", context=context,
                              rollback=db.rollback, attempts=2)
        db.expire_all()
        row = db.query(EpisodicMemory).filter_by(idempotency_key=key).one()
        result = pre.classification
        spans = attach_sequence(chat_provenance(event["prompt"], event["response"]), raw, receipt)
        if (row.raw_text != raw or row.source_spans != spans or verified_sequence(row) != receipt or row.timestamp != stamp
                or row.ts_provenance != event["provenance"]
                or row.topic_tags != list(result.topic_tags or [])
                or row.intent_tags != list(result.intent_tags or [])
                or row.context_reliance != (result.context_reliance or "Zero_Shot")):
            raise RuntimeError("stored original/classification/source authority differs from preflight")
        ids = [str(row.id), str(row.batch_id)]
        state["source_ids"][str(event["turn"])] = ids
        for identifier in ids:
            state["seen_source_times"][identifier] = event["clock"]
            state["source_turn_by_id"][identifier] = f"{event['history']}:{event['turn']}"
        emit(sink, event="source_written", event_id=event["id"], turn=event["turn"],
             clock=event["clock"], original_time=event["original_time"], ts_provenance=row.ts_provenance,
             episodic_id=str(row.id), batch_id=str(batch_id), session_id=str(row.session_id),
             source_raw_sha256=receipt["raw_sha256"], source_sequence=receipt,
             post_flight_complete=ok, summary_text=row.summary_text, abstract_text=row.abstract_text,
             summary_coverage=row.summary_coverage, representation_verification=row.representation_verification,
             lossless=row.lossless_flag, inject_raw=row.inject_raw,
             author_assertions=seed_v3.author_assertion_counts(db, row.batch_id),
             procedural_source_review=seed_v3.procedural_source_review(db, row.batch_id))
    db.rollback()


def read_probe(db, sink, state, event, scorer, classifier, embedder, arms, faults):
    stamp = datetime.fromisoformat(event["clock"])
    conv = uuid.UUID(state["conversation_id"])
    context = dict(history=event["history"], conversation=str(conv), turn=event["turn"],
                   cutoff=event["turn"], probe_id=event["probe_id"], job="probe", stage="preparation",
                   source_time=event["clock"])
    db.rollback()
    before = snapshot.fingerprints()
    stages = {}
    with historical_clock(stamp, db.get_bind().engine), observe(faults, context):
        for arm in arms:
            with seed_v3.probe_observation(db):
                _, stage = faults.call(lambda: preparation(db, event, conv, classifier, embedder,
                                                           arm=arm, exposure=False),
                                      context={**context, "arm": arm})
                seen = {k: datetime.fromisoformat(v) for k, v in state["seen_source_times"].items()}
                stage["lineage"] = seed_v3.lineage_at_cutoff(stage, seen, stamp)
                gold_ids = {turn: set(state["source_ids"][str(turn)]) for turn in scorer["gold_turns"]}
                stage["gold_fragment_coverage"] = seed_v3.gold_fragment_coverage(
                    stage, gold_ids, state["source_turn_by_id"])
                stages[arm] = stage
    if snapshot.fingerprints() != before:
        raise RuntimeError("diagnostic probe changed the historical store")
    # Scorer packet joins only after all question-only preparation calls.
    emit(sink, event="chronological_probe", event_id=event["id"], probe_id=event["probe_id"],
         conversation=event["history"], split_turn=event["turn"], question=event["question"],
         question_time=event["clock"], clock_basis=event["clock_basis"], stages=stages,
         scorer=scorer, table_sha256=before, probe_writes=False, answer_quality_scored=False)


def run(root, plan, originals, probes, *, resume=False, enabled=maintenance.CORE_JOBS,
        diagnostic_arms=seed_v3.PROMPT_ARMS, checkpoint_every_groups=1):
    """Run only explicitly supplied new-store inputs; caller provisions the DB.

    This API is not a qualified final/manual campaign command. Tests can replace
    model boundaries; their receipts must label those controls as such.
    """
    runtime, scorers = input_projection(plan, originals, probes)
    enabled = maintenance.enabled_roster(enabled)
    arms = tuple(diagnostic_arms)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(seed_v3.PROMPT_ARMS):
        raise ValueError("unknown/duplicate diagnostic preparation arms")
    seed_v3.isolated_database()
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT / "logs"):
        raise ValueError("chronological parent artifacts require private logs")
    trace = root / "chronological.jsonl"
    if trace.exists() != resume:
        raise ValueError("new parent needs a fresh trace; resume needs its original trace")
    binding = maintenance.identity(enabled, clock_basis=plan["identity"]["clock_basis"],
                                   checkpoint_every_groups=checkpoint_every_groups)
    pins = model_pins()
    identity = dict(format=POLICY, database=snapshot.DB, source_plan=plan["identity_sha256"],
        code_sha256=replay_checkpoint.code_digest(), chronological_maintenance=binding,
        diagnostic_arms=list(arms), condition="full", models=pins, tables=snapshot.TABLES,
        instruments=[fingerprint(Path(__file__)), fingerprint(Path(sources.__file__)),
                     fingerprint(Path(__file__).with_name("caught_faults.py"))])
    root.mkdir(parents=True, exist_ok=True)
    cp = replay_checkpoint.Checkpoints(root / "recovery", trace, identity)
    faults = FaultRecorder(root / "faults.jsonl", digest(identity))
    try:
        faults.verify()
        with SessionLocal() as db:
            if db.get_bind().engine.url.database != snapshot.DB:
                raise ValueError("parent database differs from snapshot database")
            functions = native_functions(db)
            if resume:
                directory, manifest = cp.validate()
                sources.validate_prefix(plan, manifest["state"]["event_prefix"])
                if manifest["state"].get("complete"):
                    store = json.loads((directory / "store.manifest.json").read_text())
                    if (trace.stat().st_size != manifest["trace_bytes"]
                            or snapshot.fingerprints() != store["sha256_by_table"]):
                        raise ValueError("completed chronological trace/store changed")
                    return manifest["state"]
                state = cp.recover()
                db.expire_all()
            else:
                if cp.pointer.exists() or any(db.execute(text(f'SELECT count(*) FROM "{table}"')).scalar()
                                               for table in Base.metadata.tables):
                    raise ValueError("new chronological parent requires an empty dedicated store")
                db.rollback()
                state = dict(event_prefix=sources.prefix_receipt(plan, 0), conversation_id=None,
                    bootstrapped=False, complete=False, last_maintenance={}, source_ids={},
                    seen_source_times={}, source_turn_by_id={})
            with trace.open("a" if resume else "x") as sink:
                recovery = WorkerRecovery(sink, "continue", state.get("worker_recovery"))
                state["worker_recovery"] = recovery.state
                if not resume:
                    emit(sink, event="chronological_meta", version="v3", identity=identity,
                         source_plan=plan, score_of_record=False)
                    cp.capture(sink, state)
                classifier, embedder = components()
                if state["conversation_id"] is None:
                    conv = Conversation(id=uuid.uuid4(), memory_scope_type="auto",
                                        created_at=datetime.fromisoformat(runtime[0]["clock"]))
                    db.add(conv); db.commit()
                    state["conversation_id"] = str(conv.id)
                    cp.capture(sink, state)
                if not state["bootstrapped"]:
                    bootstrap(db, sink, state, cp, recovery, faults, functions, runtime[0], enabled)
                completed = sources.validate_prefix(plan, state["event_prefix"])
                for index in range(completed, len(runtime)):
                    event = runtime[index]
                    context = dict(history=event["history"], conversation=state["conversation_id"],
                                   turn=event["turn"], source_time=event["clock"], arm="full")
                    start = time.perf_counter()
                    if event["kind"] == "source":
                        maintenance.drain(db, sink, until=datetime.fromisoformat(event["clock"]),
                            state=state, checkpoints=cp, recovery=recovery, faults=faults,
                            clock_basis=event["clock_basis"], context=context, enabled=enabled,
                            checkpoint_every_groups=checkpoint_every_groups)
                        faults.call(lambda: write_source(db, sink, state, event, classifier, embedder,
                            pins["background"], recovery, faults),
                            context={**context, "job": "source", "stage": "source_event"})
                    else:
                        faults.call(lambda: read_probe(db, sink, state, event, scorers[event["probe_id"]],
                            classifier, embedder, arms, faults),
                            context={**context, "job": "probe", "stage": "probe_event"})
                    state["event_prefix"] = sources.prefix_receipt(plan, index + 1)
                    emit(sink, event="chronological_event_complete", event_id=event["id"],
                         completed=index + 1, elapsed_ms=(time.perf_counter() - start) * 1000)
                    db.rollback()
                    cp.capture(sink, state)
                    recovery.check_outage()
                state["complete"] = True
                emit(sink, event="chronological_complete", version="v3", events=len(runtime),
                     source_turns=len(originals), diagnostic_occurrences=len(scorers),
                     clock_basis=plan["identity"]["clock_basis"], condition="full",
                     memory_processing=recovery.health(), answer_quality_scored=False,
                     fault_capture="uncaught_boundaries_and_logged_caught_exceptions")
                cp.capture(sink, state)
                return state
    except BaseException as exc:
        faults.record(exc, dict(job="parent", stage="parent_boundary", history=plan["identity"]["history"]))
        raise
    finally:
        cp.close()
