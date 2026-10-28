"""Decay's effect on answers by the evidence's age (Z2 F18; FINAL's decay test on S4 and S9).

Live decay (`full_default`) minus neutral decay (`full`), paired per question, split by how old
the cited evidence is when the question is asked (on the history's replay clock) and with
update questions (an old value replaced by a new one) on their own. Grades are read as given;
`--per-point` rescores grade-1 verdicts by required points covered (F8).

    uv run --no-sync python experiments/z2/decay_by_age.py --history B \
        --grades logs/z2-partial-B/graded.json --occurrences logs/z2-questions-final/B/occurrences.json --per-point
"""

import argparse
import json
import math
import statistics as S
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUCKETS = ((30, "evidence < 30 days old"), (90, "30-90 days"), (180, "90-180 days"), (10 ** 6, "> 180 days"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--occurrences", type=Path, required=True)
    parser.add_argument("--live", default="full_default")
    parser.add_argument("--neutral", default="full")
    parser.add_argument("--per-point", action="store_true")
    args = parser.parse_args()
    from experiments.z2 import clock
    times = [datetime.fromisoformat(t) for t in clock.table(args.history, clock.load_history(args.history))["turn_times"]]
    graded = json.loads(args.grades.read_text())
    graded = graded.get("graded", graded)

    def value(cond, q):
        r = graded.get(f"{cond}|{q}")
        if not r or r.get("status") != "graded":
            return None
        if args.per_point and r["items_total"]:
            return float(len(set(r["items_covered"])) >= r["items_total"] and not r["superseded_used"])
        return float(r["grade"] == "correct")

    groups = {}
    for o in json.loads(args.occurrences.read_text()):
        q = f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}"
        live, neutral = value(args.live, q), value(args.neutral, q)
        turns = [int(c["turn"]) for c in o["reference"].get("citations", []) if c.get("turn")]
        if live is None or neutral is None or not turns:
            continue
        asked = times[min(int(o["checkpoint_turn"]), len(times)) - 1]
        age = (asked - times[min(turns) - 1]).days                   # the oldest evidence it needs
        bucket = next(label for limit, label in BUCKETS if age < limit)
        for key in ("all", bucket) + (("update questions",) if o.get("type") == "update" else ()):
            groups.setdefault(key, []).append(live - neutral)
    span = (times[-1] - times[0]).days
    print(f"{args.history}: live decay minus neutral decay ({'per-point' if args.per_point else 'judge-correct'}), "
          f"history spans {span} days; + = decay helps")
    order = ["all"] + [label for _, label in BUCKETS] + ["update questions"]
    for key in [k for k in order if k in groups]:
        x = groups[key]
        ci = 196 * S.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0
        print(f"   {key:24} {100 * S.mean(x):+5.1f} ±{ci:4.1f}  (n={len(x)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
