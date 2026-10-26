"""FINAL's LongMemEval-S slice (FINAL_experiments.md §2.1, decided 2026-10-26): 100 of the
500 instances, proportional to question type (abstention `_abs` questions proportional
within each type), by a fixed seed, before any result is seen. Native protocol and upstream
prompts; the oracle evidence sessions are the read-time `all_evidence` control.

    uv run --no-sync python experiments/lme/final_slice.py       # write / verify final_slice.json
"""

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "longmemeval_s"
SEED, SIZE = 20261026, 100


def allocate(counts, size):
    """Largest-remainder proportional allocation."""
    total = sum(counts.values())
    exact = {k: size * v / total for k, v in counts.items()}
    out = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[:size - sum(out.values())]:
        out[k] += 1
    return out


def build():
    raw = DATA.read_bytes()
    rows = json.loads(raw)
    rng = random.Random(SEED)
    groups = {}
    for r in sorted(rows, key=lambda r: r["question_id"]):
        groups.setdefault((r["question_type"], r["question_id"].endswith("_abs")), []).append(r)
    take = allocate({k: len(v) for k, v in groups.items()}, SIZE)
    chosen = []
    for key in sorted(groups):
        chosen += rng.sample(groups[key], take[key])
    turns = lambda r: sum(1 for s in r["haystack_sessions"] for m in s if m["role"] == "user")
    return dict(source="xiaowu0162/longmemeval longmemeval_s", sha256=hashlib.sha256(raw).hexdigest(),
                seed=SEED, size=SIZE, rule="proportional to (question_type, abstention); seeded sample",
                user_turns=sum(turns(r) for r in chosen),
                by_type={f"{t}{' (abs)' if a else ''}": take[(t, a)] for t, a in sorted(take)},
                question_ids=sorted(r["question_id"] for r in chosen))


def main() -> int:
    out, path = build(), HERE / "final_slice.json"
    if path.exists() and json.loads(path.read_text()) != out:
        print("final_slice.json differs from a fresh selection: refusing to overwrite", file=sys.stderr)
        return 1
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "question_ids"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
