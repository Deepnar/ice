"""FINAL's cloud-answerer subset (Z3 D6, D13): do the conclusions depend on the answerer?

25 questions from each of S9 (the ablation conversation), B (real history) and S5 (time), each
under ICE as it runs, plain vector search and long context (225 contexts), chosen by a fixed hash from the read
passes, answered by a model of a non-Meta family (the main answerer is Muse 1.2), and graded
exactly like the main answers. Compare: does full ICE keep its margin over the baselines? The
same contexts feed the cross-family judge check (run_final's crossjudge-* stages).

    uv run --no-sync python experiments/final/cloud_subset.py --run logs/final-run-main \
        --answerer cloud:<role> [--judge muse] [--per-history 25]

Writes subset-read-<h> (the chosen units), subset-answers-<h>, subset-grade-<h>.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HISTORIES = ("S9", "B", "S5")
CONDITIONS = ("full_default", "bare_vector", "long_context")     # the main comparison (2026-10-28)
POLICY = "ice-final-cloud-subset-1"


def choose(read_dir, n, conditions=CONDITIONS):
    """Ask ids read under every condition, ranked by a fixed hash; the first *n*."""
    have = [{p.stem for p in (Path(read_dir) / "units" / c).glob("*.json")} for c in conditions]
    common = set.intersection(*have) if have else set()
    return sorted(common, key=lambda a: hashlib.sha256(f"{POLICY}|{a}".encode()).hexdigest())[:n]


def stage_units(read_dir, out_dir, asks, conditions=CONDITIONS):
    for c in conditions:
        (Path(out_dir) / "units" / c).mkdir(parents=True, exist_ok=True)
        for a in asks:
            shutil.copy2(Path(read_dir) / "units" / c / f"{a}.json", Path(out_dir) / "units" / c / f"{a}.json")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--answerer", required=True, help="cloud:<role>")
    parser.add_argument("--judge", default="muse")
    parser.add_argument("--per-history", type=int, default=25)
    args = parser.parse_args()
    from experiments.final.run_final import grade_stage
    for h in HISTORIES:
        read = args.run / f"local-read-{h}"
        asks = choose(read, args.per_history)
        units = stage_units(read, args.run / f"subset-read-{h}", asks)
        answers = args.run / f"subset-answers-{h}"
        subprocess.run([sys.executable, "experiments/z2/answers.py", "--units", str(units.relative_to(ROOT)),
                        "--out", str(answers.relative_to(ROOT)), "--answerer", args.answerer], check=True, cwd=ROOT)
        grade = grade_stage(h, units=f"subset-read-{h}", answers_dir=f"subset-answers-{h}", key=f"subset-grade-{h}")
        ctx = type("Ctx", (), {"dir": args.run})()
        cmd = grade.cmd_args(ctx, args.run / f"subset-grade-{h}")
        subprocess.run([sys.executable, *cmd, "--judge", args.judge], check=True, cwd=ROOT)
        print(json.dumps(dict(history=h, asks=len(asks), contexts=len(asks) * len(CONDITIONS))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
