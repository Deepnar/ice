#!/usr/bin/env python3
"""G59 direction rule: pooled treatment vs pooled control.

Uses judge_codex.report_power's OWN interval formula so this is the same
measurement method the arms were reported with, not a second one:
    se = sqrt(p(1-p) / n_turns)
i.e. each TURN counts as one effective observation (maximally conservative
about within-turn correlation).

Turn counts come from the run logs, not the artifacts — the judgement JSON
records edge_id but not source turn, so it cannot be re-clustered after the
fact. That is an instrument gap, noted rather than worked around.
"""
import json
import math
from collections import Counter
from pathlib import Path

D = Path("experiments/curation_files/judgements")

# arm -> distinct turns judged (from the RESOLUTION block of each run log)
TURNS = {
    "dir-true-run1-perturn": 124,
    "dir-true-run2-perturn": 124,
    "dir-false-run1-perturn": 124,
    "dir-false-run2-perturn": 124,
}
ARMS = {
    "treatment (rule ON)":  ["dir-true-run1-perturn",  "dir-true-run2-perturn"],
    "control   (rule OFF)": ["dir-false-run1-perturn", "dir-false-run2-perturn"],
}


def tally(arm):
    v = json.load(open(D / f"codex_quality_{arm}.json"))["verdicts"]
    return Counter(x["label"] for x in v), len(v)


def rate_ci(k, n, n_turns):
    p = k / n
    se = math.sqrt(max(p * (1 - p), 1e-9) / n_turns)
    return p, se


print(f"{'arm':<26} {'trip':>5} {'turns':>6}   {'correct':>16}   {'reversed':>16}")
print("-" * 78)

pool = {}
for name, arms in ARMS.items():
    K = Counter()
    N = T = 0
    for a in arms:
        t, n = tally(a)
        K += t
        N += n
        T += TURNS[a]
        pc, sc = rate_ci(t["correct"], n, TURNS[a])
        pr, sr = rate_ci(t["reversed"], n, TURNS[a])
        print(f"  {a:<24} {n:>5} {TURNS[a]:>6}   "
              f"{100*pc:5.1f}% ±{196*sc:4.1f}   {100*pr:5.1f}% ±{196*sr:4.1f}")
    pc, sc = rate_ci(K["correct"], N, T)
    pr, sr = rate_ci(K["reversed"], N, T)
    pool[name] = {"correct": (pc, sc), "reversed": (pr, sr), "n": N, "turns": T}
    print(f"{name:<26} {N:>5} {T:>6}   "
          f"{100*pc:5.1f}% ±{196*sc:4.1f}   {100*pr:5.1f}% ±{196*sr:4.1f}   <- POOLED\n")

print("=" * 78)
print("G59 EFFECT — pooled treatment minus pooled control\n")
for label in ("correct", "reversed"):
    pt, st = pool["treatment (rule ON)"][label]
    pc, sc = pool["control   (rule OFF)"][label]
    d = 100 * (pt - pc)
    sd = math.sqrt(st ** 2 + sc ** 2)
    z = (pt - pc) / sd
    print(f"  {label:<9} ON {100*pt:5.2f}%   OFF {100*pc:5.2f}%   "
          f"delta {d:+5.2f} pts   95% CI [{d-196*sd:+5.1f}, {d+196*sd:+5.1f}]   z={z:+.3f}")

print(f"\n  smallest effect this design could have detected: "
      f"±{196*math.sqrt(pool['treatment (rule ON)']['correct'][1]**2 + pool['control   (rule OFF)']['correct'][1]**2):.1f} pts")
