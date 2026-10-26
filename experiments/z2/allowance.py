"""Allowance check before the cloud block (spec D6d, TRAPS 148): the plan's limit is
spend, so a stage is sized from the allowance real answers report, never from the
request table. Run a few minutes of S10, stop it, then:

    uv run --no-sync python experiments/z2/allowance.py logs/z2-run-main

Per answer stage: answers saved, their median allowance, and the projected allowance
for the stage. Grading calls are not projected (their rows keep no allowance).
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# answer stage → (its S9/S12 units, planned distinct prompts until the units are read)
STAGES = {"s10-B": ("s9-B", 4864), "s10-C": ("s9-C", 1966), "s12-answers": ("s12", 2000)}
PLAN = "plan (D6c, measured): ~$12 per 5 hours, ~$60 per month"


def allowance(row):
    return (((row.get("usage") or {}).get("opencode") or {}).get("allowance"))


def report(run_dir, totals=None, say=print):
    """Projected allowance per stage; *totals* overrides the distinct-prompt counts."""
    from experiments.z2 import answers
    costs, lines = [], []
    for stage, (units, planned) in STAGES.items():
        path = Path(run_dir) / stage / "answers.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        seen = [a for a in map(allowance, rows) if isinstance(a, (int, float))]
        costs += seen
        total = (totals or {}).get(stage)
        if total is None:
            total = answers.progress(Path(run_dir) / units, path.parent)[1] or planned
        lines.append((stage, len(rows), seen, total))
    if not costs:
        say("no answer reports an allowance yet — run a few minutes of S10 first")
        return None
    median = statistics.median(costs)
    grand = 0.0
    for stage, saved, seen, total in lines:
        per = statistics.median(seen) if seen else median
        grand += per * total
        say(f"{stage:12} {saved:5}/{total} answered  median ${per:.5f}/answer"
            f"{'' if seen else ' (from the other stages)'}  stage ≈ ${per * total:.2f}")
    say(f"all answers ≈ ${grand:.2f} from {len(costs)} measured answers — {PLAN}")
    return grand


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else ROOT / "logs" / "z2-run-main")
