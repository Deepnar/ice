#!/usr/bin/env python3
"""Compare two judgement runs TRIPLET BY TRIPLET, not rate by rate.

⚑ WHY PER-TRIPLET. Two runs can report `correct 20%` and `correct 20%` while
disagreeing on half the individual verdicts — the aggregates cancel and the
instrument looks stable when it is not. Every graph-quality claim in this
project rests on aggregate rates, so this is the check none of them have had.

Two questions, and they are different:

  SELF-CONSISTENCY   same judge, same store, same seed -> the SAME 200 triplets.
                     Any disagreement is the judge alone: not extraction, not
                     sampling. This puts a floor under every number, because a
                     difference smaller than the floor is unreadable.

  AGREEMENT          two DIFFERENT judges on the same 200. Tells you whether a
                     verdict is a property of the graph or of the model that
                     judged it. [TRAPS #34](../../docs/TRAPS.md) is the
                     precedent: this judge once ruled `both_failed` on 30 of 73
                     verdicts at 100% content overlap.

Run it on both, and never confuse them:

    uv run python scripts/z1/compare_judgements.py \
        --a dir-true-run1 --b dir-true-run1-rejudge --kind self
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

JUDGE = Path("experiments/curation_files/judgements")


def load(arm: str) -> dict:
    p = JUDGE / f"codex_quality_{arm}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    d = json.loads(p.read_text())
    rows = d.get("verdicts") or d.get("records") or d.get("results") or []
    out = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = r.get("edge_id") or r.get("id")
        v = r.get("verdict") or r.get("label") or r.get("judgement")
        if k and v:
            out[str(k)] = str(v)
    return {"meta": {k: d.get(k) for k in ("arm", "model", "seed")}, "verdicts": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--kind", choices=["self", "agreement"], default="self")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    va, vb = A["verdicts"], B["verdicts"]
    shared = sorted(set(va) & set(vb))

    print("=" * 70)
    print(f"{'SELF-CONSISTENCY' if args.kind == 'self' else 'JUDGE AGREEMENT'}"
          f"   {args.a}  vs  {args.b}")
    print("=" * 70)
    print(f"  A: {A['meta']}")
    print(f"  B: {B['meta']}")
    print(f"\n  verdicts A={len(va)}  B={len(vb)}  overlapping={len(shared)}")

    if not shared:
        print("\n  ⚠ NO OVERLAPPING TRIPLETS — these two runs did not judge the")
        print("    same facts, so nothing here is comparable. If this was meant")
        print("    to be a self-consistency check, the seed or the store changed.")
        return 1

    agree = sum(1 for k in shared if va[k] == vb[k])
    pct = 100 * agree / len(shared)
    print(f"  identical verdicts: {agree}/{len(shared)}  ({pct:.1f}%)")

    ca = Counter(va[k] for k in shared)
    cb = Counter(vb[k] for k in shared)
    print(f"\n  {'verdict':14s} {'A':>6s} {'B':>6s} {'delta':>7s}")
    for lbl in sorted(set(ca) | set(cb)):
        print(f"  {lbl:14s} {ca[lbl]:6d} {cb[lbl]:6d} {cb[lbl]-ca[lbl]:+7d}")

    flips = Counter((va[k], vb[k]) for k in shared if va[k] != vb[k])
    if flips:
        print(f"\n  most common disagreements (A -> B):")
        for (x, y), n in flips.most_common(6):
            print(f"    {n:4d}  {x} -> {y}")

    # The number that actually matters: how far can a rate move on noise alone?
    worst = max((abs(cb[l] - ca[l]) for l in set(ca) | set(cb)), default=0)
    print(f"\n  ⚑ largest single-category swing: {worst} of {len(shared)} "
          f"= {100*worst/len(shared):.1f} percentage points")
    if args.kind == "self":
        print("    This is the JUDGE'S OWN noise floor. Any claimed difference")
        print("    smaller than it is not readable, no matter how many probes.")
    else:
        print("    Two judges differing by this much means the verdict is a")
        print("    property of the judge as much as of the graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
