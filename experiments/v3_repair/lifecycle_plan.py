"""Plan native chronological decay and quantify retention targets; never run jobs."""

import argparse
import fcntl
import heapq
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.prepare_inputs import digest, fingerprint, write_or_verify

JOBS = ("decay_episodic", "decay_codex", "decay_procedural")
FIELDS = ("decay_daily_unaccessed", "decay_daily_accessed", "decay_daily_creative",
          "decay_archive_threshold", "decay_cold_threshold", "decay_creative_floor",
          "decay_strengthen_amount", "retrieval_strengthen_writes", "codex_decay_daily",
          "codex_retention_floor", "codex_retention_increment", "codex_retention_cap",
          "codex_retention_rank_weight", "procedural_stale_days",
          "procedural_min_reinforcement", "runtime_cycles_cap")
CONSUMERS = ("src/api/config.py", "src/workers/runtime.py", "src/workers/decay.py",
             "src/workers/codex_decay.py", "src/workers/procedural_decay.py",
             "src/memory/usage.py", "src/retrieval/orchestrator.py",
             "scripts/z1/seed_v3.py", "scripts/z1/historical_clock.py")


def number(value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError("invalid lifecycle numeric value")
    return value


def clock(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("lifecycle clock must have an explicit timezone")
    return value.astimezone(timezone.utc)


def cadence(value):
    number(value, positive=True)
    duration = timedelta(seconds=value)
    if duration.total_seconds() != value:
        raise ValueError("cadence cannot be represented exactly by the replay clock")
    return duration


def due_counts(start, end, intervals):
    start, end = clock(start), clock(end)
    if end <= start or set(intervals) != set(JOBS):
        raise ValueError("invalid interval or incomplete decay job roster")
    return {job: (end - start) // cadence(intervals[job]) for job in JOBS}


def decay_events(start, end, intervals):
    """Exact due boundaries after a common successful start; no endpoint batching.

    Stable registry order breaks ties. A final runner must merge its other core
    jobs and source events; this is deliberately not the complete replay runner.
    """
    counts = due_counts(start, end, intervals)
    start = clock(start)
    pending = [(start + cadence(intervals[job]), index, job, counts[job])
               for index, job in enumerate(JOBS) if counts[job]]
    heapq.heapify(pending)
    while pending:
        stamp, index, job, remaining = heapq.heappop(pending)
        yield dict(job=job, source_time=stamp.isoformat(), cycles=1)
        if remaining > 1:
            heapq.heappush(pending, (stamp + cadence(intervals[job]), index, job, remaining - 1))


def crossing_days(daily, threshold, floor=0):
    """Continuous eligible-aging boundary from score1; strict SQL crossing is later."""
    if daily == 1 or floor >= threshold:
        return None
    return math.log(threshold) / math.log(daily)


def validate_policy(policy):
    if set(policy) != set(FIELDS):
        raise ValueError("incomplete or unknown lifecycle field")
    for key in FIELDS:
        if key == "retrieval_strengthen_writes":
            if type(policy[key]) is not bool: raise ValueError("invalid exposure switch")
        else:
            number(policy[key])
    for key in ("decay_daily_unaccessed", "decay_daily_accessed", "decay_daily_creative", "codex_decay_daily"):
        if not 0 < policy[key] <= 1: raise ValueError("daily multiplier outside (0,1]")
    if not 0 < policy["decay_cold_threshold"] < policy["decay_archive_threshold"] <= 1:
        raise ValueError("invalid archive/cold ordering")
    if not 0 <= policy["decay_creative_floor"] <= 1:
        raise ValueError("creative floor outside normalized score")
    if not 0 < policy["codex_retention_floor"] <= 1 <= policy["codex_retention_cap"]:
        raise ValueError("invalid graph retention bounds")
    for key in ("runtime_cycles_cap", "procedural_stale_days", "procedural_min_reinforcement"):
        if type(policy[key]) is not int or policy[key] < 1: raise ValueError("invalid lifecycle integer")


def build(policy, intervals, start, end, *, clock_basis, event_limit=200_000):
    validate_policy(policy)
    if clock_basis not in {"original", "simulated"}: raise ValueError("declare lifecycle clock provenance")
    counts = due_counts(start, end, intervals)
    if type(event_limit) is not int or event_limit < 1 or sum(counts.values()) > event_limit:
        raise ValueError("chronological schedule exceeds explicit event limit")
    rates = {}
    for kind in ("unaccessed", "accessed", "creative"):
        daily = policy["decay_daily_" + kind]
        floor = policy["decay_creative_floor"] if kind == "creative" else 0
        rates[kind] = dict(daily_multiplier=daily, daily_loss_fraction=1 - daily,
            half_life_eligible_days=crossing_days(daily, .5, floor),
            archive_eligible_days=crossing_days(daily, policy["decay_archive_threshold"], floor),
            cold_eligible_days=crossing_days(daily, policy["decay_cold_threshold"], floor),
            daily_sensitivity=dict(half_duration=daily ** 2, double_duration=math.sqrt(daily)))
    cap = policy["runtime_cycles_cap"]
    dispatch = {job: dict(due_cycles=count,
        endpoint_cycles=min(cap, max(1, count)) if count else 0,
        omitted_cycles=max(0, count - cap) if job != "decay_procedural" else None,
        capped_aging_days=cap * intervals[job] / 86400 if job != "decay_procedural" else None,
        remainder_seconds=(clock(end) - clock(start)).total_seconds() - count * intervals[job],
        semantics="absolute_observation_age" if job == "decay_procedural" else "numeric_retention")
        for job, count in counts.items()}
    exposure = {}
    for name, amount, initial, ceiling in (
        ("episodic", policy["decay_strengthen_amount"], policy["decay_archive_threshold"], 1.0),
        ("graph", policy["codex_retention_increment"], 1.0, policy["codex_retention_cap"])):
        exposure[name] = dict(amount=amount, initial_strength=initial, ceiling=ceiling,
            uninterrupted_exposures_to_ceiling=math.ceil(max(0, ceiling - initial) / amount) if amount else None,
            increment_sensitivity=sorted(set((0, amount / 2, amount, amount * 2))),
            baseline_writes_enabled=policy["retrieval_strengthen_writes"],
            qualified=False)
    graph_cap = policy["codex_retention_cap"]
    exposure["graph"]["cap_sensitivity"] = [dict(cap=c,
        bonus_factor_at_unchanged_strength1=1 + policy["codex_retention_rank_weight"] * min(1, 1 / c))
        for c in sorted(set((max(1, graph_cap / 2), graph_cap, graph_cap * 2)))]
    return dict(version="v3", method="chronological_retention_plan_v1", score_of_record=False,
        clock_basis=clock_basis, start=clock(start).isoformat(), end=clock(end).isoformat(),
        policy=policy, intervals=intervals, endpoint_dispatch=dispatch, rates=rates,
        exposure=exposure, schedule=list(decay_events(start, end, intervals)),
        assumptions=["Start is a common known successful job boundary; actual per-job ledgers may differ.",
            "Episodic horizons start from score1 after the strict seven-day eligibility boundary.",
            "No exposure, immunity changes or source updates occur in the analytic horizons.",
            "Accessed means any positive lifetime access count, not a recent-use clock.",
            "Archive/cold transitions affect location, not source truth or ordinary bounded reachability.",
            "Cold moves stop warm numeric aging; endpoint batching can overshoot the first crossing.",
            "Graph decay/exposure changes retention priority only, never confidence or validity.",
            "Zero increment still changes counts/access class; use master-off for no exposure writes.",
            "This plans three native decay jobs; other core jobs/events/recovery still need final integration."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--clock-basis", choices=("original", "simulated"), required=True)
    parser.add_argument("--event-limit", type=int, default=200_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from src.api.config import settings
    from src.workers.runtime import JOBS as registry
    if any(not registry[job].pass_cycles for job in JOBS):
        raise ValueError("native decay registry contract changed")
    value = build({name: getattr(settings, name) for name in FIELDS},
        {job: settings.maintenance_intervals[job] for job in JOBS},
        datetime.fromisoformat(args.start), datetime.fromisoformat(args.end),
        clock_basis=args.clock_basis, event_limit=args.event_limit)
    value["identity"] = dict(instrument=fingerprint(Path(__file__)),
        consumers=[fingerprint(ROOT / path) for path in CONSUMERS],
        parameters_sha256=digest({k: value[k] for k in ("policy", "intervals", "start", "end", "clock_basis")}))
    value["identity_sha256"] = digest(value["identity"])
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "logs"):
        raise ValueError("lifecycle reports must stay in private logs")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_or_verify(out, value)
    print(json.dumps(dict(version="v3", events=len(value["schedule"]),
        endpoint_dispatch=value["endpoint_dispatch"], score_of_record=False)))


if __name__ == "__main__": main()
