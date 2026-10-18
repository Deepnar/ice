"""Summarise a `profile_write_path.py` run: per-turn stage times, model calls by
stage, reconcile calls per new fact against store size, maintenance by job.

    uv run --no-sync python experiments/z2/profile_report.py logs/z2-s0-speed-profile-<date>-r1 \
        [--window 101 150]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def walk(node, path=()):
    """Yield (path, node) for a stage tree, depth first."""
    here = path + (node["stage"],)
    yield here, node
    for child in node.get("children", []):
        yield from walk(child, here)


def self_seconds(node):
    return node["seconds"] - sum(c["seconds"] for c in node.get("children", []))


def turns_of(records):
    """Group top-level records into historical turns: preflight stages precede the
    store of their turn; post-flight and maintenance follow it."""
    turns, pending = [], []
    for record in records:
        stage = record["stage"]
        if stage.startswith("preflight"):
            pending.append(record)
        elif stage == "write.store":
            turns.append({"key": record.get("key"), "words": record.get("words"),
                          "records": pending + [record]})
            pending = []
        elif turns and stage in ("post_flight", "maintenance", "probe.observe"):
            turns[-1]["records"].append(record)
    return turns


def number(key):
    return int(str(key).rsplit("-", 1)[-1]) if key else None


def summarise(records, window=None):
    turns = turns_of(records)
    if window:
        turns = [t for t in turns if window[0] <= number(t["key"]) <= window[1]]
    per_stage = defaultdict(float)
    per_stage_self = defaultdict(float)
    calls = defaultdict(lambda: dict(n=0, wall=0.0, prompt=0, completion=0, load=0.0))
    turn_totals = []
    for turn in turns:
        total = 0.0
        for record in turn["records"]:
            total += record["seconds"]
            for path, node in walk(record):
                per_stage[" > ".join(path)] += node["seconds"]
                per_stage_self[" > ".join(path)] += self_seconds(node)
                for call in node.get("calls", []):
                    c = calls[" > ".join(path)]
                    c["n"] += 1
                    c["wall"] += call.get("wall") or 0
                    c["prompt"] += call.get("prompt_tokens") or 0
                    c["completion"] += call.get("completion_tokens") or 0
                    c["load"] += call.get("provider_load_seconds") or 0
        turn_totals.append((number(turn["key"]), turn["words"], round(total, 1)))
    return turns, per_stage, per_stage_self, calls, turn_totals


def reconcile_growth(records):
    """Per post-flight: new-fact candidates and reconcile calls, in turn order."""
    rows = []
    for turn in turns_of(records):
        candidates = reconcile = 0
        for record in turn["records"]:
            for path, node in walk(record):
                if node["stage"] == "facts.candidates":
                    candidates += 1
                elif node["stage"] == "facts.reconcile_call":
                    reconcile += 1
        rows.append((number(turn["key"]), candidates, reconcile))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, nargs=2)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run / "profile.jsonl").open()
               if line.strip()]
    records = [r for r in records if "seconds" in r]
    turns, per_stage, per_stage_self, calls, totals = summarise(records, args.window)
    n = len(turns)
    grand = sum(t[2] for t in totals)
    print(f"turns {n}  total {grand / 60:.1f} min  mean {grand / max(n, 1):.1f} s/turn  "
          f"median {statistics.median([t[2] for t in totals]) if totals else 0:.1f} s")
    print("\nstage (inclusive s | self s | share of total)")
    for path, seconds in sorted(per_stage.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"  {seconds:9.1f} | {per_stage_self[path]:9.1f} | {seconds / grand:6.1%}  {path}")
    print("\nmodel calls by stage (n | wall s | prompt tok | completion tok | load s)")
    for path, c in sorted(calls.items(), key=lambda kv: -kv[1]["wall"]):
        print(f"  {c['n']:5d} | {c['wall']:8.1f} | {c['prompt']:9d} | {c['completion']:8d} | "
              f"{c['load']:6.1f}  {path}")
    print("\nper turn (turn, words, seconds), slowest 10:")
    for row in sorted(totals, key=lambda t: -t[2])[:10]:
        print("  ", row)
    growth = reconcile_growth(records)
    if growth:
        print("\nreconcile calls per turn, by tenths of the run (turns, candidate lookups, calls):")
        step = max(1, len(growth) // 10)
        for i in range(0, len(growth), step):
            chunk = growth[i:i + step]
            print(f"  turns {chunk[0][0]}-{chunk[-1][0]}: lookups {sum(c[1] for c in chunk)}, "
                  f"calls {sum(c[2] for c in chunk)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
