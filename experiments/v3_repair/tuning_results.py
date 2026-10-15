"""Tuning results: graded read coordinates against the baseline and its noise floor.

Reads completed independent-grader reports. Every row is one answer
(arm tag ``coordinate:arm``, occurrence, grade, prompt tokens). Comparisons are
paired by occurrence; an occurrence ungraded on either side drops out of that
pair only.

The noise floor is the disagreement between repeated answers to the *same*
baseline full context (``baseline-repeat`` and any later re-ask). The
pre-registered rule (V3_POST_RESEED_REPAIR §5.2, 2026-10-14): a candidate wins
only if its net correct gain over the baseline exceeds the floor at equal budget,
or its net change is within the floor and its median prompt tokens fall by at
least 10%. Otherwise the current setting stays. Diagnostic only; no promotion is
written anywhere.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

POINTS = {"correct": 2, "partial": 1, "incorrect": 0}


def load(reports, allow_partial=False):
    """{label: {occurrence: (grade, prompt_tokens)}}; later plans' baselines are re-asks."""
    arms = defaultdict(dict)
    for index, path in enumerate(reports):
        data = json.loads(Path(path).read_text())
        if not data.get("complete") and not allow_partial:
            raise ValueError(f"grading report incomplete: {path}")
        for row in data["records"]:
            label = row["arm"]
            if index and label.startswith("baseline:"):
                label = f"baseline-reask{index}:{label.split(':', 1)[1]}"
            grade = row["grade"] if row.get("status") == "graded" else None
            if isinstance(grade, dict): grade = grade.get("grade")
            arms[label][json.dumps(row["occurrence"], sort_keys=True)] = (grade, row.get("prompt_tokens"))
    return arms


def pair(left, right):
    keys = [k for k in left if k in right and left[k][0] in POINTS and right[k][0] in POINTS]
    wins = sum(right[k][0] == "correct" and left[k][0] != "correct" for k in keys)
    losses = sum(left[k][0] == "correct" and right[k][0] != "correct" for k in keys)
    points = sum(POINTS[right[k][0]] - POINTS[left[k][0]] for k in keys)
    return dict(paired=len(keys), wins=wins, losses=losses, net=wins - losses, points_delta=points)


def summary(rows):
    grades = [g for g, _ in rows.values()]
    tokens = [t for _, t in rows.values() if isinstance(t, (int, float))]
    return dict(correct=grades.count("correct"), partial=grades.count("partial"),
                incorrect=grades.count("incorrect"), uncertain=grades.count("uncertain"),
                ungraded=sum(g is None for g in grades),
                median_tokens=statistics.median(tokens) if tokens else None)


def analyse(arms):
    base = arms["baseline:full"]
    repeats = {label: rows for label, rows in arms.items()
               if label != "baseline:full" and label.endswith(":full")
               and (label.startswith("baseline-repeat") or label.startswith("baseline-reask"))}
    floors = {label: pair(base, rows) for label, rows in repeats.items()}
    floor = max((abs(f["net"]) for f in floors.values()), default=None)
    swing = max((f["wins"] + f["losses"] for f in floors.values()), default=None)
    base_tokens = summary(base)["median_tokens"]
    out = []
    for label, rows in sorted(arms.items()):
        if label == "baseline:full" or label in repeats:
            continue
        p = pair(base, rows)
        s = summary(rows)
        token_change = (s["median_tokens"] - base_tokens) / base_tokens if base_tokens and s["median_tokens"] else None
        control = label.startswith("baseline:")
        if control or floor is None:
            verdict = "control" if control else "no noise floor yet"
        elif p["net"] > floor:
            verdict = "WIN: quality above noise"
        elif abs(p["net"]) <= floor and token_change is not None and token_change <= -0.10:
            verdict = "WIN: same quality, >=10% fewer tokens"
        elif p["net"] < -floor:
            verdict = "worse than baseline"
        else:
            verdict = "keep current (within noise)"
        out.append(dict(arm=label, **s, vs_baseline=p, token_change=token_change, verdict=verdict))
    return dict(baseline=summary(base), noise=dict(floor_net=floor, max_disagreement=swing, pairs=floors), rows=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", type=Path, nargs="+", required=True,
                        help="Completed grader reports; the first holds the four-arm baseline.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preview", action="store_true", help="Allow incomplete reports; output is a partial preview.")
    args = parser.parse_args()
    result = analyse(load(args.reports, allow_partial=args.preview))
    if args.preview:
        print("PREVIEW — grading incomplete; not a result")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    b, n = result["baseline"], result["noise"]
    print(f"baseline full: correct {b['correct']} partial {b['partial']} incorrect {b['incorrect']} "
          f"uncertain {b['uncertain']} ungraded {b['ungraded']} | median tokens {b['median_tokens']}")
    print(f"noise floor: net {n['floor_net']} (max answer disagreement {n['max_disagreement']}) from "
          + ", ".join(f"{k} net {v['net']:+d} (+{v['wins']}/-{v['losses']})" for k, v in n['pairs'].items()))
    for r in result["rows"]:
        p = r["vs_baseline"]
        tc = f"{r['token_change']:+.0%}" if r["token_change"] is not None else "n/a"
        print(f"{r['arm']:<40} correct {r['correct']:>2}  vs base net {p['net']:+d} (+{p['wins']}/-{p['losses']}, n={p['paired']})"
              f"  tokens {tc:>5}  {r['verdict']}")


if __name__ == "__main__":
    main()
