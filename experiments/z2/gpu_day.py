"""The GPU day after Z2's S12 (results/RETESTS.md): every GPU-bound retest and fix-pass test behind
one resumable command, on Z2's runner (meter, lock, retries, clean stop, resume).

Order and dependencies:
  gate-*   correctness first — the source-check DB tests, then the full disposable rails. If a gate
           fails, every retest READ is held (numbers from broken code would be wrong); the stages
           that do not depend on them still run.
  r8-*     fix-pass evaluations, independent of the gates: G60 reconcile, G75 summary claim check,
           B7 classifier variants (four trainings).
  r2-*     the candidate sweep on B, one stage per setting so each has its own timing; then the
           coverage report.
  r34-*    reads for the conversation-order and summary retests on B and C (answers and grades
           come later, in the cloud).
  r9-*     a LoCoMo replay+read slice on FINAL's runner under its own run name (never FINAL's).
Nothing here decides a setting: decisions are made at R7, after C's grade-1 results (R6), the
judge check (R5) and the retest answers.

Starts only when Z2's S12 has stopped using the GPU (it waits, showing why), so it can be launched
early. Run from the fix-pass worktree:

    uv run --no-sync python experiments/z2/gpu_day.py run                 # everything, resumable
    uv run --no-sync python experiments/z2/gpu_day.py status --watch      # the meter
    uv run --no-sync python experiments/z2/gpu_day.py run --only r2-B-cand128,r2-coverage
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import run_z2 as Z  # noqa: E402

RUN = "z2-run-main"                                   # Z2's replays, questions and answers
HIST = {"B": dict(database="ice_z2_main_b", replay=f"logs/{RUN}/s7-B", asks="logs/z2-questions-final/B/z2-asks.json"),
        "C": dict(database="ice_z2_main_c", replay=f"logs/{RUN}/s7-C", asks="logs/z2-questions-final/C/z2-asks.json")}
GATES = ("gate-source-db", "gate-rails")
SWEEP = ("retest_cand64", "retest_cand128", "retest_cand256", "retest_cand512",
         "retest_cand64_keep", "retest_cand128_keep", "retest_cand256_batch16")
PY = [sys.executable]
DISPOSABLE = PY + ["tests/support/disposable_database.py", "-m", "pytest", "-q"]


class Command(Z.Stage):
    """One command, run once per attempt; done when it exits 0 (a marker file records it)."""

    def __init__(self, key, title, cmd, *, needs=(), estimate=600.0, attempts=1):
        super().__init__(key=key, title=title, needs=tuple(needs), unit="step", estimate=estimate)
        self.cmd, self.attempts = cmd, attempts

    def marker(self, ctx):
        return ctx.dir / f"{self.key}.done"

    def total(self, ctx):
        return 1

    def done(self, ctx):
        return int(self.marker(ctx).exists())

    def complete(self, ctx):
        return self.marker(ctx).exists()

    def run(self, ctx):
        log = ctx.dir / f"{self.key}.log"
        for n in range(self.attempts):
            if self.complete(ctx):
                return "done"
            rc = ctx.run_child(self.cmd, log)
            if rc == 0:
                self.marker(ctx).write_text(f"exit 0 after {n + 1} attempt(s)\n")
                return "done"
            ctx.log("command failed", stage=self.key, rc=rc, tail=Z._tail(log))
            ctx.stage_state(self.key).setdefault("failures", []).append(dict(rc=rc, detail=Z._tail(log)))
            ctx.save()
        return "stuck"


class Reads(Z.Stage):
    """Retest read passes of one history under named conditions (resumable per unit)."""

    def __init__(self, key, history, conditions, *, needs=GATES, estimate=6.0):
        super().__init__(key=key, title=f"reads {history}: {conditions}", needs=tuple(needs),
                         unit="contexts", estimate=estimate)
        self.history, self.conditions = history, conditions

    def out(self, ctx):
        return ROOT / f"logs/z2-retest-{self.history}"

    def _progress(self):
        from experiments.z2 import read_passes as R
        return R.progress(self.out(None), ROOT / HIST[self.history]["asks"], R.select_conditions(self.conditions))

    def total(self, ctx):
        return self._progress()[1] or 0

    def done(self, ctx):
        return self._progress()[0]

    def complete(self, ctx):
        total = self.total(ctx)
        return bool(total) and self.done(ctx) >= total

    def run(self, ctx):
        h = HIST[self.history]
        log = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = PY + ["experiments/z2/read_passes.py", "--history", self.history, "--database", h["database"],
                        "--replay", h["replay"], "--asks", h["asks"], "--out", f"logs/z2-retest-{self.history}",
                        "--conditions", self.conditions]
            return ctx.run_child(cmd, log), Z._tail(log)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


def stages(_limit=None):
    out = [
        Command("gate-source-db", "DB tests: source check never cancels, routes, summaries",
                DISPOSABLE + ["tests/smoke/test_source_need_route.py", "tests/smoke/test_temporal_axis_path.py",
                              "tests/smoke/test_memory_preparation.py", "tests/smoke/test_reranker.py",
                              "tests/test_summary_snapshot.py"], estimate=600),
        Command("gate-rails", "full disposable test rails", DISPOSABLE + ["tests/smoke"], estimate=1200),
        Command("r8-g60", "G60 same-item supersession eval", PY + ["logs/fix-pass/reconcile_eval.py"], estimate=3600),
        Command("r8-g75", "G75 summary claim-check eval", PY + ["logs/fix-pass/claim_check_eval.py"], estimate=3600),
        Command("r8-b7", "B7 classifier variants (4 trainings)", PY + ["logs/fix-pass/b7_variants.py"], estimate=7200),
    ]
    out += [Reads(f"r2-B-{c.removeprefix('retest_')}", "B", c) for c in SWEEP]
    out.append(Command("r2-coverage", "R2 coverage report (B)", PY + [
        "experiments/z2/retest_coverage.py", "--history", "B", "--units", "logs/z2-retest-B/units",
        "--reference", f"logs/{RUN}/s9-B/units", "--occurrences", "logs/z2-questions-final/B/occurrences.json"],
        needs=[f"r2-B-{c.removeprefix('retest_')}" for c in SWEEP], estimate=300))
    out += [Reads("r34-B", "B", "retest_cand64,retest_conversation_order+retest_summaries"),
            Reads("r34-C", "C", "retest_cand64,retest_conversation_order+retest_summaries", estimate=16.0)]
    out.append(Command("r9-locomo-slice", "LoCoMo replay+read slice (FINAL runner, run name 'trial')", PY + [
        "experiments/final/run_final.py", "run", "--name", "trial", "--limit", "30",
        "--only", "prep-locomo-export,local-replay-LOCOMO,local-read-LOCOMO"], needs=GATES, estimate=5400))
    return out


def gpu_free():
    """Z2's own runner prerequisites, plus: no Z2 local GPU stage still running (S12's workers)."""
    problems = BASE_PREREQUISITES()
    busy = subprocess.run(["pgrep", "-f", r"experiments/z2/(oracle|read_passes|replay)\.py .*--out logs/z2-run-"],
                          capture_output=True, text=True).stdout.split()
    if busy:
        problems.append(f"Z2 still on the GPU ({len(busy)} process(es): S12 or a read/replay) — waiting")
    return problems


BASE_PREREQUISITES = Z.prerequisites

if __name__ == "__main__":
    Z.prerequisites = gpu_free                         # Context() reads this name at start
    raise SystemExit(Z.main(stages, run_dir=lambda name: ROOT / "logs" / f"gpu-day-{name}", prog="gpu_day.py",
                            description=__doc__.split("\n\n")[0], runs="logs/gpu-day-<name>"))
