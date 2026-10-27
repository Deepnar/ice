"""FINAL's local phase in parallel lanes (maintainer, 2026-10-28: one history at a time, the local
phase was ~14-17 GPU days with the GPU mostly idle — Z2's S12 overlapped 2 of 8,220 model calls).

Every history replays into its own database, so different histories are independent. This
launcher (1) runs the preparation stages on the main runner and waits; (2) splits the local
stages into LANES groups — each history's chain kept together (replay, audit inventory, read
passes; LoCoMo's memory systems with LoCoMo), balanced by estimated hours — and (3) starts one
runner per group side by side (`run_final.py run --lane localK --only ...`). A stage a lane
finishes counts as done for every runner of the run (run_z2 Context.done_elsewhere), so the
main runner's cloud phase later uses the lanes' work and never redoes it.

Ollama must serve as many requests at once as there are lanes (OLLAMA_NUM_PARALLEL), or the
lanes' model calls queue for one slot. Resumable: Ctrl-C stops every lane; the same command
continues.

    uv run --no-sync python experiments/final/local_lanes.py --lanes 2 --dry-run   # the split
    uv run --no-sync python experiments/final/local_lanes.py --lanes 2             # run it
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUNNER = "experiments/final/run_final.py"


def chain_of(key):
    """The history a local stage belongs to: its chain stays in one lane."""
    for prefix in ("local-replay-", "local-inventory-", "local-read-"):
        if key.startswith(prefix):
            return key[len(prefix):]
    if key.startswith("local-baseline-"):
        return key.rsplit("-", 1)[-1]                 # the memory systems run on LoCoMo's reads
    return key


def plan(stages, ctx, lanes):
    """(preparation keys, [lane stage keys], [lane estimated hours]) — chains in run order,
    the heaviest first into the lightest lane."""
    prep = [s.key for s in stages if s.key.startswith("prep-")]
    chains, hours = {}, {}
    for s in stages:
        if not s.key.startswith("local-"):
            continue
        c = chain_of(s.key)
        chains.setdefault(c, []).append(s.key)
        hours[c] = hours.get(c, 0.0) + (s.total(ctx) or 0) * (getattr(s, "estimate", None) or 6.0) / 3600
    groups, load = [[] for _ in range(lanes)], [0.0] * lanes
    for c in sorted(chains, key=lambda c: -hours[c]):
        k = load.index(min(load))
        groups[k] += chains[c]
        load[k] += hours[c]
    order = {s.key: i for i, s in enumerate(stages)}
    return prep, [sorted(g, key=order.get) for g in groups], load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lanes", type=int, default=2, help="side by side; match OLLAMA_NUM_PARALLEL")
    parser.add_argument("--name", default="main", help="the run (logs/final-run-<name>)")
    parser.add_argument("--dry-run", action="store_true", help="print the split, run nothing")
    args = parser.parse_args()
    from experiments.final import run_final as F
    from experiments.z2 import run_z2 as Z
    stages = F.final_stages(None)
    ctx = Z.Context(args.name, directory=ROOT / "logs" / f"final-run-{args.name}")
    prep, groups, load = plan(stages, ctx, args.lanes)
    for k, (keys, h) in enumerate(zip(groups, load)):
        print(f"lane local{k}: ~{h:.0f} h — {', '.join(keys)}")
    if args.dry_run:
        return 0
    base = [sys.executable, RUNNER, "run", "--name", args.name]
    print("preparation:", ", ".join(prep), flush=True)
    if subprocess.run(base + ["--only", ",".join(prep)], cwd=ROOT).returncode not in (0, 3):
        print("preparation did not finish; the lanes would only find their inputs missing — rerun later")
        return 3
    lanes = [subprocess.Popen(base + ["--lane", f"local{k}", "--only", ",".join(keys)], cwd=ROOT)
             for k, keys in enumerate(groups) if keys]
    try:
        return max(p.wait() for p in lanes)
    except KeyboardInterrupt:                         # Ctrl-C reaches the lanes too; wait for their clean stop
        return max(p.wait() for p in lanes)


if __name__ == "__main__":
    raise SystemExit(main())
