"""Z3 judging — the cloud phase after Z2's own cloud run and before the cloud prep (maintainer,
2026-10-29: Z2 first, then this, then cloud prep; never mixed). One resumable command on Z2's
runner (meter, lock, retries, clean stop, resume); state in logs/z3-judging-<name>.

  r5-panel        the planted judge panel (qualify.py) under grade-2, reusing Z2's planted answers
  r5-judge-B/C    judge_check.py: grade-2 on grade-1 verdicts (B 230 + C ~50, invariance pairs)
  r34-answers-*   the retest answers (Muse 1.2) for the order and summary settings, all at 256 candidates
  r34-grade-*     grade-2 verdicts on those answers (--verdicts-only; claims can be added if a
                  decision hinges on them)
R7 (the chosen settings together) is added once R2–R4 have chosen.

It waits while any Z2 runner is active (Z2's cloud run first), and a stage waits ("needs input")
until its inputs exist: Z2's grade-1 grades for the judge check, the GPU day's reads for answers.

    uv run --no-sync python experiments/z2/z3_judging.py run
    uv run --no-sync python experiments/z2/z3_judging.py status --watch
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import run_z2 as Z  # noqa: E402
from experiments.z2.gpu_day import Command, PY  # noqa: E402
from experiments.z2.histories import HISTORIES  # noqa: E402

RUN = ROOT / "logs/z2-run-main"
GPU_DAY = ROOT / "logs/gpu-day-main/state.json"
QUALIFIED = ROOT / "logs/z2-qualify-2026-10-19-r1"        # grade-1's panel: its planted answers are reused
ANSWERED = ("retest_cand256", "retest_c256_order", "retest_c256_originals_first", "retest_c256_policy",
            "retest_c256_wide_originals", "retest_c256_no_summaries")     # all at 256 candidates (2026-10-30)
ASKS = {"B": 174, "C": 75}                                 # questions per history (retests read questions only)
SCALE = {"B": 1.0, "C": 0.22}


def gpu_day_done(key):
    try:
        return json.loads(GPU_DAY.read_text())["stages"][key]["status"] == "done"
    except (OSError, ValueError, KeyError):
        return False


def reads_done(history):
    """Every condition the answers compare has been read for every question of this history."""
    from experiments.z2 import read_passes as R
    asks = ROOT / f"logs/z2-questions-final/{history}/z2-asks.json"
    done, total = R.progress(ROOT / f"logs/z2-retest-{history}", asks, R.select_conditions(",".join(ANSWERED)))
    return bool(total) and done >= total


def lines(path):
    try:
        with open(path) as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


class Counted(Command):
    """A resumable cloud command whose progress is the line count of its durable call file."""

    def __init__(self, key, title, cmd, *, calls, planned, unit="calls", needs=(), inputs=(), estimate=10.0):
        super().__init__(key, title, cmd, needs=needs, estimate=estimate, attempts=5)
        self.calls, self.planned, self.unit, self.inputs = calls, planned, unit, inputs

    def total(self, ctx):
        return self.planned

    def done(self, ctx):
        return self.planned if self.complete(ctx) else min(lines(self.calls), self.planned - 1)

    def missing(self, ctx):
        return [str(m) for m in self.inputs if not (reads_done(m.removeprefix("reads-")) if isinstance(m, str)
                                                    and m.startswith("reads-") else Path(m).exists())]


class Panel(Counted):
    def run(self, ctx):
        out = self.calls.parent
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "plants-raw.jsonl").exists():       # reuse grade-1's planted answers: judge calls only
            shutil.copy(QUALIFIED / "plants-raw.jsonl", out / "plants-raw.jsonl")
        return super().run(ctx)


def stages(_limit=None):
    out, d = [], ROOT / "logs/z3-judging"
    out.append(Panel("r5-panel", "R5 planted judge panel under grade-2",
                     PY + ["experiments/z2/qualify.py", "--out", "logs/z3-judging/qualify"],
                     calls=d / "qualify/judge-raw.jsonl", planned=932,         # grade-1's panel made 932 judge calls
                     inputs=(QUALIFIED / "plants-raw.jsonl",)))
    for h in ("B", "C"):
        grades = RUN / f"s11-{h}/grades.json"
        out.append(Counted(f"r5-judge-{h}", f"R5 judge check {h} (grade-2 on grade-1 verdicts)", PY + [
            "experiments/z2/judge_check.py", "--history", h, "--grades", str(grades.relative_to(ROOT)),
            "--run", "logs/z2-run-main", "--questions", "logs/z2-questions-final",
            "--out", f"logs/z3-judging/judge-check-{h}", "--scale", str(SCALE[h])],
            calls=d / f"judge-check-{h}/grade2-raw.jsonl", planned=round(230 * SCALE[h]) + 20,
            inputs=(grades,)))
    for h in ("B", "C"):
        answers = ROOT / f"logs/z2-retest-{h}-answers"
        out.append(Counted(f"r34-answers-{h}", f"R3/R4 retest answers {h} (Muse 1.2)", PY + [
            "experiments/z2/answers.py", "--units", f"logs/z2-retest-{h}", "--out", str(answers.relative_to(ROOT)),
            "--conditions", ",".join(ANSWERED), "--answerer", "muse12"],
            calls=answers / "answers.jsonl", planned=ASKS[h] * len(ANSWERED), unit="answers",
            inputs=(f"reads-{h}",)))
        hist = HISTORIES[h]
        out.append(Counted(f"r34-grade-{h}", f"R3/R4 retest grades {h} (grade-2 verdicts)", PY + [
            "experiments/z2/grade.py", "--history", h, "--units", f"logs/z2-retest-{h}",
            "--answers", str(answers.relative_to(ROOT)), "--questions", "logs/z2-questions-final",
            "--ledger", str(hist.path("ledger").relative_to(ROOT)), "--changes", str(hist.path("changes").relative_to(ROOT)),
            "--out", f"logs/z2-retest-{h}-grades", "--verdicts-only"],
            calls=ROOT / f"logs/z2-retest-{h}-grades/grade-raw.jsonl", planned=ASKS[h] * len(ANSWERED),
            needs=(f"r34-answers-{h}",)))
    return out


def z2_finished():
    """Z2's runner prerequisites, plus: Z2's own runs (main or cloud lane) have ended."""
    problems = BASE_PREREQUISITES()
    busy = subprocess.run(["pgrep", "-f", r"experiments/z2/run_z2\.py run"], capture_output=True, text=True).stdout.split()
    if busy:
        problems.append("Z2's own run is still active — Z3 judging starts after it (maintainer's order)")
    return problems


BASE_PREREQUISITES = Z.prerequisites

if __name__ == "__main__":
    Z.prerequisites = z2_finished
    raise SystemExit(Z.main(stages, run_dir=lambda name: ROOT / "logs" / f"z3-judging-{name}", prog="z3_judging.py",
                            description=__doc__.split("\n\n")[0], runs="logs/z3-judging-<name>"))
