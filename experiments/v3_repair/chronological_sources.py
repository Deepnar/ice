"""One-history source/probe timeline contract for a new chronological parent.

No CLI, database, model or writer. Runtime projections omit scorer evidence;
source clocks and immutable event prefixes must be checked by the parent.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from experiments.v3_repair.lifecycle_plan import clock
from experiments.v3_repair.prepare_inputs import ROOT, digest, fingerprint
from scripts.z1.label_review import KNOWLEDGE_SCOPES, TASK_TYPES, reviewed_expected_answer

POLICY = "ice-v3-chronological-source-events-1"
QUERY_CLOCK_POLICY = "ice-v3-chronological-source-events-2"
PROVENANCE = {"original", "synthetic_raw_import"}


def timestamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return clock(value).isoformat()


def original(row, number):
    if (not isinstance(row, dict) or type(row.get("turn_number")) is not int or row["turn_number"] != number
            or any(not isinstance(row.get(k), str) or not row[k].strip()
                   for k in ("prompt", "response"))
            or row.get("ts_provenance") not in PROVENANCE or "timestamp" not in row):
        raise ValueError("incomplete/reordered original source or unknown provenance")
    return dict(turn=number, prompt=row["prompt"], response=row["response"],
                original_time=timestamp(row["timestamp"]), provenance=row["ts_provenance"])


def reviewed_occurrence(probe, history, cutoff):
    review = probe.get("label_review")
    if (not isinstance(review, dict) or review.get("conversation") != history
            or type(review.get("cutoff_turn")) is not int or review["cutoff_turn"] != cutoff
            or type(review.get("reviewed_through_turn")) is not int
            or review["reviewed_through_turn"] != cutoff
            or review.get("question") != probe.get("question")
            or probe.get("label_status") != "source_reviewed_valid"
            or type(review.get("recent_only_answerable")) is not bool
            or review.get("knowledge_scope") not in KNOWLEDGE_SCOPES):
        raise ValueError("diagnostic source review is incomplete or differs from occurrence")
    tasks = review.get("task_types")
    if (not isinstance(tasks, list) or not tasks or any(t not in TASK_TYPES for t in tasks)
            or len(tasks) != len(set(tasks))
            or probe.get("expected_answer") != reviewed_expected_answer(review)):
        raise ValueError("diagnostic task/answer binding differs")
    for key in ("gold_turns", "reviewed_gold_turns"):
        if key in review and review[key] != probe["gold_turns"]:
            raise ValueError("diagnostic reviewed source binding differs")


def validate(plan):
    if (not isinstance(plan, dict) or plan.get("version") != "v3"
            or plan.get("identity_sha256") != digest(plan.get("identity"))
            or plan.get("events_sha256") != digest(plan.get("events"))
            or plan["identity"].get("policy") not in {POLICY, QUERY_CLOCK_POLICY}):
        raise ValueError("source timeline identity/events changed")
    events = plan["events"]
    if (not isinstance(events, list) or not events
            or plan["identity"].get("events_sha256") != plan["events_sha256"]):
        raise ValueError("source timeline event binding differs")
    return plan


def build(history, originals, probes, *, clock_basis, replay_times=None, probe_times=None):
    """Freeze supplied clocks; never synthesize unrequested gaps or source facts.

    probes contains (cutoff, full reviewed occurrence), so the key is bound as
    scorer metadata but absent from the answering/ingestion projection.
    """
    if (not isinstance(history, str) or not history.strip()
            or not isinstance(originals, list) or not originals
            or clock_basis not in {"original", "simulated"}):
        raise ValueError("one named complete history and explicit clock basis required")
    sources = [original(row, n) for n, row in enumerate(originals, 1)]
    original_times = [s["original_time"] for s in sources]
    if any(a >= b for a, b in zip(original_times, original_times[1:])):
        raise ValueError("original source clock is not strictly ordered")
    if clock_basis == "original" and (replay_times is not None or any(
            s["provenance"] != "original" for s in sources)):
        raise ValueError("original clock requires authentic dates without replacements")
    if replay_times is not None and len(replay_times) != len(sources):
        raise ValueError("replacement clocks must cover every source")
    times = original_times if replay_times is None else [timestamp(t) for t in replay_times]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("replay clock is not strictly ordered")
    grouped, occurrences = {}, {}
    for cutoff, probe in probes:
        if type(cutoff) is not int or not 1 <= cutoff <= len(sources) or not isinstance(probe, dict):
            raise ValueError("invalid source cutoff/occurrence")
        identifier, gold = probe.get("probe_id"), probe.get("gold_turns")
        if (not isinstance(identifier, str) or not identifier.strip() or identifier in occurrences
                or not isinstance(probe.get("question"), str) or not probe["question"].strip()
                or not isinstance(gold, list) or not gold
                or any(type(t) is not int or not 1 <= t <= cutoff for t in gold)
                or len(set(gold)) != len(gold)):
            raise ValueError("duplicate/unbound/future diagnostic occurrence")
        reviewed_occurrence(probe, history, cutoff)
        occurrences[identifier] = digest(probe)
        grouped.setdefault(cutoff, []).append(probe)
    if probe_times is not None:
        if (clock_basis != "simulated" or not isinstance(probe_times, dict)
                or not occurrences or set(probe_times) != set(occurrences)):
            raise ValueError("simulated probe clocks must cover every reviewed occurrence")
        probe_times = {name: timestamp(value) for name, value in probe_times.items()}
        for cutoff, probes_at_cutoff in grouped.items():
            for probe in probes_at_cutoff:
                when = probe_times[probe["probe_id"]]
                if when < times[cutoff - 1] or (cutoff < len(times) and when >= times[cutoff]):
                    raise ValueError("probe clock precedes its cutoff or includes a future source")
    sequence = digest([dict(turn=s["turn"], prompt=s["prompt"], response=s["response"])
                       for s in sources])
    events = []
    for source, replay_time in zip(sources, times):
        number = source["turn"]
        common = dict(history=history, turn=number, clock=replay_time, clock_basis=clock_basis)
        events.append(dict(common, id=f"source:{number}", kind="source",
                           original_time=source["original_time"], provenance=source["provenance"],
                           source_sha256=digest(source)))
        for probe in sorted(grouped.get(number, []), key=lambda p: (
                probe_times[p["probe_id"]] if probe_times is not None else replay_time, p["probe_id"])):
            events.append(dict(common, clock=probe_times[probe["probe_id"]] if probe_times is not None else replay_time,
                               id="probe:" + probe["probe_id"], kind="probe",
                               probe_id=probe["probe_id"], probe_sha256=occurrences[probe["probe_id"]]))
    identity = dict(policy=POLICY, history=history, source_sequence_sha256=sequence,
                    sources=[digest(s) for s in sources], occurrences=occurrences,
                    clock_basis=clock_basis, replay_times=times, events_sha256=digest(events))
    if probe_times is not None:
        identity.update(policy=QUERY_CLOCK_POLICY, probe_times=probe_times)
    return dict(version="v3", identity=identity, identity_sha256=digest(identity),
                events=events, events_sha256=digest(events), execution_ready=False,
                labels="input bindings only; independent semantic qualification unchanged")


def prefix_receipt(plan, completed):
    validate(plan)
    if type(completed) is not int or not 0 <= completed <= len(plan["events"]):
        raise ValueError("invalid completed event count")
    return dict(identity=plan["identity_sha256"], completed=completed,
                prefix_sha256=digest(plan["events"][:completed]))


def validate_prefix(plan, receipt):
    if not isinstance(receipt, dict) or receipt != prefix_receipt(plan, receipt.get("completed")):
        raise ValueError("completed source-event prefix changed; preserve original state")
    return receipt["completed"]


def runtime_projection(plan, event_index, *, source=None, probe=None):
    """Only supplied turn/question data reaches the future runtime dispatcher."""
    validate(plan)
    if type(event_index) is not int or not 0 <= event_index < len(plan["events"]):
        raise ValueError("invalid source-event index")
    event = plan["events"][event_index]
    common = {k: event[k] for k in ("id", "kind", "history", "turn", "clock", "clock_basis")}
    if event["kind"] == "source":
        if probe is not None or source is None:
            raise ValueError("source projection needs only its complete original")
        resolved = original(source, event["turn"])
        if digest(resolved) != event["source_sha256"]:
            raise ValueError("original source projection changed")
        return dict(common, prompt=resolved["prompt"], response=resolved["response"],
                    original_time=resolved["original_time"], provenance=resolved["provenance"],
                    source_sequence_sha256=plan["identity"]["source_sequence_sha256"])
    if event["kind"] != "probe" or source is not None or not isinstance(probe, dict):
        raise ValueError("probe projection needs only its bound occurrence")
    if digest(probe) != event["probe_sha256"]:
        raise ValueError("diagnostic occurrence projection changed")
    return dict(common, probe_id=event["probe_id"], question=probe["question"])


def frozen_tuning_view(producer, tuning_plan, reader_profile):
    """Explicit old-writer source audit; never bypass fresh preparation guards."""
    from experiments.v3_repair.reader_adoption import validate as validate_reader
    from experiments.v3_repair.run_tuning_contexts import load_coordinate, resolved_baseline, source_probes

    producer, tuning_plan, reader_profile = map(Path, (producer, tuning_plan, reader_profile))
    producer, tuning_plan, reader_profile = (p.resolve() for p in (producer, tuning_plan, reader_profile))
    if any(not p.is_relative_to(ROOT / "logs") for p in (producer, tuning_plan, reader_profile)):
        raise ValueError("frozen tuning view requires private existing bundles")
    config = json.loads((producer / "development-run.json").read_text())
    if config["identity"].get("partition") != "tuning" or len(config["identity"]["expected_turns"]) != 1:
        raise ValueError("frozen source view requires one tuning history; validation is reserved")
    resolved_baseline(config)
    seal, _coordinate = load_coordinate(tuning_plan, "baseline")
    profile = validate_reader(json.loads(reader_profile.read_text()), producer, config, seal)
    report, observed = source_probes(producer)
    with (producer / "seed.jsonl").open() as source:
        inputs = json.loads(next(source))["meta"]["inputs"]
    paths = [Path(r["path"]).resolve() for r in inputs if Path(r["path"]).name == "development-originals.json"]
    if len(paths) != 1 or not paths[0].is_relative_to(ROOT / "logs"):
        raise ValueError("frozen original source path is missing or ambiguous")
    originals_path = paths[0]
    body = originals_path.read_bytes()
    if hashlib.sha256(body).hexdigest() != config["identity"]["originals_sha256"]:
        raise ValueError("frozen original source file changed")
    history, count = next(iter(config["identity"]["expected_turns"].items()))
    selected = json.loads(body)[history]
    if len(selected) != count:
        raise ValueError("frozen original history count differs")
    originals = [dict(turn_number=r["turn_number"], prompt=r["prompt"], response=r["response"],
                      timestamp=r["timestamp"], ts_provenance=r["timestamp_provenance"]) for r in selected]
    probes = []
    for row in observed.values():
        if row["conversation"] != history:
            raise ValueError("frozen source view contains another history")
        probes.append((row["split_turn"], row))
    basis = "original" if all(r["ts_provenance"] == "original" for r in originals) else "simulated"
    plan = build(history, originals, probes, clock_basis=basis)
    plan["identity"]["frozen_writer"] = dict(producer=config["identity_sha256"],
        reader=profile["identity_sha256"], trace=report["trace_sha256"],
        coordinate_seal=seal["identity_sha256"], originals=fingerprint(originals_path),
        source_contract=fingerprint(Path(__file__)))
    plan["identity_sha256"] = digest(plan["identity"])
    return plan, originals, probes
