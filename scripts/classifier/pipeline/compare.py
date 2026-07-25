#!/usr/bin/env python3
"""Compare two labeling passes — agreement rate, per-head, per-label.

Descended from ``legacy/promt_labeling/compare_labeling.py``. Two uses:

* **Before committing hours to a labeler**, run it over a few hundred rows the
  other labeler already covered and read the agreement rate. A candidate that
  agrees ~90% is interchangeable; one that agrees ~50% is either much better,
  much worse, or reading the rubric differently — and you want to know which
  before, not after, a five-hour pass.
* **After both full passes**, to see where the disagreement actually lives
  before the tiebreak spends anything on it.

It reuses ``label._agreement`` rather than reimplementing the rule, so what this
reports is exactly what the merge will do.

Usage:
    uv run python scripts/classifier/pipeline/compare.py labels_b.jsonl dryrun_gptoss.jsonl
    uv run python scripts/classifier/pipeline/compare.py --examples 8 A.jsonl B.jsonl
"""

import argparse
import os
from collections import Counter

from common import DATA_DIR, read_jsonl
from label import _agreement, _sets

from src.classifier.schema import CONTEXT_RELIANCE, INTENT, TOPIC


def load(path: str) -> dict:
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(DATA_DIR, path)
    return {e["id"]: e for e in read_jsonl(path)}


def main():
    ap = argparse.ArgumentParser(description="B1: compare two labeling passes")
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--examples", type=int, default=5,
                    help="show N disagreeing rows per head")
    args = ap.parse_args()

    left, right = load(args.left), load(args.right)
    shared = [i for i in left if i in right]
    if not shared:
        raise SystemExit("no overlapping row ids — nothing to compare")

    print(f"left  {args.left}: {len(left)} rows")
    print(f"right {args.right}: {len(right)} rows")
    print(f"overlap: {len(shared)} rows\n")

    heads = (TOPIC, INTENT, CONTEXT_RELIANCE)
    agree = Counter()
    full = 0
    per_label = {h: Counter() for h in heads}
    examples = {h: [] for h in heads}

    for row_id in shared:
        a, b = left[row_id], right[row_id]
        result = _agreement(a, b)
        for head, ok in result.items():
            agree[head] += int(ok)
            if not ok and len(examples[head]) < args.examples:
                examples[head].append((a, b))
        if all(result.values()):
            full += 1
        # Which labels the two passes differ on, per head.
        for idx, head in enumerate(heads):
            sa, sb = _sets(a)[idx], _sets(b)[idx]
            for label in sa ^ sb:
                per_label[head][label] += 1

    n = len(shared)
    print("AGREEMENT")
    for head in heads:
        rate = agree[head] / n
        bar = "#" * int(rate * 40)
        print(f"  {head:18} {rate:6.1%}  {bar}")
    print(f"  {'ALL THREE':18} {full / n:6.1%}   ({full}/{n} rows would settle "
          f"without a tiebreak)\n")

    print("MOST-CONTESTED LABELS (times exactly one pass assigned it)")
    for head in heads:
        top = per_label[head].most_common(5)
        if top:
            print(f"  {head}:")
            for label, count in top:
                print(f"    {label:26} {count:5}  ({count / n:.1%} of rows)")
    print()

    for head in heads:
        if not examples[head]:
            continue
        print(f"DISAGREEMENT EXAMPLES — {head}")
        idx = heads.index(head)
        for a, b in examples[head]:
            text = (a.get("text") or b.get("text") or "")[:130].replace("\n", " ")
            print(f"  * {text}")
            print(f"      left : {sorted(_sets(a)[idx])}")
            print(f"      right: {sorted(_sets(b)[idx])}")
        print()


if __name__ == "__main__":
    main()
