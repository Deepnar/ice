"""Z3 judging — the cloud phase after Z2's own cloud run and before the cloud prep (maintainer,
2026-10-29: Z2 first, then this, then cloud prep; never mixed). One resumable command on Z2's
runner (meter, lock, retries, clean stop, resume); state in logs/z3-judging-<name>.

  r5-panel        the planted judge panel (qualify.py) under grade-2, reusing Z2's planted answers
  r5-judge-B/C    judge_check.py: grade-2 on grade-1 verdicts (B 230 + C ~50, invariance pairs)
  r5-aux-B/C      aux_judge_check.py: the fixed claim check and controls judge (Z2 F27) on Z2's own
                  cases beside the old verdicts, with planted cases of known answer
  r5-beam         experiments/beam/judge_check.py: BEAM's own protocol (F27) on planted answers
  r34-answers-*   the retest answers (Muse 1.2) for the order and summary settings, all at 256 candidates
  r34-grade-*     grade-2 on those answers with the claim check (SCA needs it: Z2 F27); no ranking
  r10-answers     R10's own replay answered (Muse 1.2): shipped, no memory, reranker rescue (bar off) —
                  308 distinct prompts; every bar is replayed offline from them (r10_report.py --grades)
  r10-grade       LongMemEval's own yes/no judge on those answers (--upstream-only)
  r7-answers-*    R7 (RETESTS): the frozen candidate's answers, beside no memory and plain search, on
                  every ask — questions and no-memory controls — reusing Z2's answers to identical
                  prompts; waits for gpu_extra.py's r7-reads-* (which wait for r7_settings.json)
  r7-grade-*      grade-2 with the claim check, controls and context precision, reported with
                  r7_frozen as the system tested (no ranking)

It waits while any Z2 runner is active (Z2's cloud run first), and a stage waits ("needs input")
until its inputs exist: Z2's grade-1 grades for the judge check, the GPU day's reads for answers.

    uv run --no-sync python experiments/z2/z3_judging.py run --only r5-panel,r5-judge-B,r5-judge-C,r5-aux-B,r5-aux-C,r5-beam   # run 1
    (read logs/z3-judging/{judge-check,aux-check}-*/report.txt, beam-check/report.txt and blind-sheet.md; approve: touch logs/z3-judging/judge-approved)
    uv run --no-sync python experiments/z2/z3_judging.py run                                        # run 2
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
# Two runs (maintainer, 2026-10-31): the judge check first; the retest answers start only after the
# maintainer has read its report and blind sheet and this file exists (touch it to approve).
APPROVED = ROOT / "logs/z3-judging/judge-approved"
R10_ANSWERED = ("full", "recent_only", "full_rescue_reranker")   # a refused question's prompt is recent_only's
ASKS = {"B": 174, "C": 75}                                 # questions per history (retests read questions only)
SCALE = {"B": 1.0, "C": 0.22}
CLAIM_SHARE = 1.73                     # grade + claim calls per graded answer (B: 3,120 claim calls / 4,300 graded)
AUX = {"B": 447, "C": 243}             # aux_judge_check.py --dry-run plans
BEAM_CHECK = 800                       # judge_check.py --dry-run: 288 nugget calls + the ordering alignment


def gpu_day_done(key):
    try:
        return json.loads(GPU_DAY.read_text())["stages"][key]["status"] == "done"
    except (OSError, ValueError, KeyError):
        return False


R7_ASKS = {"B": 174 + 34, "C": 75 + 11}                  # questions + no-memory controls
R7_GRADE_CALLS = 2.3                                       # per answered ask: grade + claims + controls + precision


def r7_reads_done(history):
    """R7's three conditions read for every ask of this history (gpu_extra.py r7-reads-<history>)."""
    from experiments.z2 import read_passes as R
    conditions = R.r7_conditions()
    if conditions is None:
        return False
    asks = ROOT / f"logs/z2-questions-final/{history}/z2-asks.json"
    done, total = R.progress(ROOT / f"logs/z2-r7-{history}", asks, conditions)
    return bool(total) and done >= total


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
        """The plan, or — once the work outgrows it — the calls made so far plus one, so the meter
        keeps moving (run 2, 2026-11-04: grade-2 lists claims on almost every answer, B's grading
        passed its planned 1,806 and the bar sat at 1,805 for hours)."""
        made = lines(self.calls)
        return made if self.complete(ctx) else max(self.planned, made + 1)

    def done(self, ctx):
        return self.total(ctx) if self.complete(ctx) else lines(self.calls)

    def missing(self, ctx):
        def ready(m):
            if isinstance(m, str) and m.startswith("r7reads-"):
                return r7_reads_done(m.removeprefix("r7reads-"))
            if isinstance(m, str) and m.startswith("reads-"):
                return reads_done(m.removeprefix("reads-"))
            return Path(m).exists()
        return [str(m) for m in self.inputs if not ready(m)]


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
        report = RUN / f"s11-{h}/report.json"               # Z2's grading finished: its old verdicts are final
        out.append(Counted(f"r5-aux-{h}", f"R5 claim check + controls judge {h} (fixed prompts, planted cases)", PY + [
            "experiments/z2/aux_judge_check.py", "--history", h, "--grades", f"logs/z2-run-main/s11-{h}/grades.json",
            "--run", "logs/z2-run-main", "--questions", "logs/z2-questions-final", "--out", f"logs/z3-judging/aux-check-{h}"],
            calls=d / f"aux-check-{h}/aux-raw.jsonl", planned=AUX[h], inputs=(report,)))
    out.append(Counted("r5-beam", "R5 BEAM's own judge on planted answers", PY + [
        "experiments/beam/judge_check.py", "--out", "logs/z3-judging/beam-check"],
        calls=d / "beam-check/beam-grade-raw.jsonl", planned=BEAM_CHECK, inputs=(ROOT / "logs/final-beam",)))
    for h in ("B", "C"):
        answers = ROOT / f"logs/z2-retest-{h}-answers"
        out.append(Counted(f"r34-answers-{h}", f"R3/R4 retest answers {h} (Muse 1.2)", PY + [
            "experiments/z2/answers.py", "--units", f"logs/z2-retest-{h}", "--out", str(answers.relative_to(ROOT)),
            "--conditions", ",".join(ANSWERED), "--answerer", "muse12"],
            calls=answers / "answers.jsonl", planned=ASKS[h] * len(ANSWERED), unit="answers",
            inputs=(f"reads-{h}", APPROVED)))
        hist = HISTORIES[h]
        out.append(Counted(f"r34-grade-{h}", f"R3/R4 retest grades {h} (grade-2 + claim check)", PY + [
            "experiments/z2/grade.py", "--history", h, "--units", f"logs/z2-retest-{h}",
            "--answers", str(answers.relative_to(ROOT)), "--questions", "logs/z2-questions-final",
            "--ledger", str(hist.path("ledger").relative_to(ROOT)), "--changes", str(hist.path("changes").relative_to(ROOT)),
            "--out", f"logs/z2-retest-{h}-grades", "--no-ranking"],
            calls=ROOT / f"logs/z2-retest-{h}-grades/grade-raw.jsonl", planned=round(ASKS[h] * len(ANSWERED) * CLAIM_SHARE),
            needs=(f"r34-answers-{h}",)))
    # R7 (2026-11-03): the frozen candidate on B and C, after run 2 chose its switches
    for h in ("B", "C"):
        units, answers = f"logs/z2-r7-{h}", ROOT / f"logs/z2-r7-{h}-answers"
        out.append(Counted(f"r7-answers-{h}", f"R7 answers {h}: frozen candidate, no memory, plain search (Muse 1.2)",
                           PY + ["experiments/z2/answers.py", "--units", units, "--out", str(answers.relative_to(ROOT)),
                                 "--answerer", "muse12", "--reuse", f"logs/z2-run-main/s10-{h}"],
                           calls=answers / "answers.jsonl", planned=R7_ASKS[h] * 3, unit="answers",
                           inputs=(f"r7reads-{h}",)))
        hist = HISTORIES[h]
        out.append(Counted(f"r7-grade-{h}", f"R7 grades {h} (grade-2, claim check, controls, precision)", PY + [
            "experiments/z2/grade.py", "--history", h, "--units", units, "--answers", str(answers.relative_to(ROOT)),
            "--questions", "logs/z2-questions-final", "--ledger", str(hist.path("ledger").relative_to(ROOT)),
            "--changes", str(hist.path("changes").relative_to(ROOT)), "--out", f"logs/z2-r7-{h}-grades",
            "--no-ranking", "--primary", "r7_frozen"],
            calls=ROOT / f"logs/z2-r7-{h}-grades/grade-raw.jsonl", planned=round(R7_ASKS[h] * 3 * R7_GRADE_CALLS),
            needs=(f"r7-answers-{h}",)))
    # R10 (Z2 F4, 2026-11-01): the reranker rescue graded — LongMemEval's judge needs no judge check
    r10, r10_answers = RUN / "s12-r10", ROOT / "logs/z3-judging/r10-answers"
    out.append(Counted("r10-answers", "R10 answers: shipped, no memory, reranker rescue (Muse 1.2)", PY + [
        "experiments/z2/answers.py", "--units", str(r10.relative_to(ROOT)), "--out", str(r10_answers.relative_to(ROOT)),
        "--conditions", ",".join(R10_ANSWERED), "--answerer", "muse12"],
        calls=r10_answers / "answers.jsonl", planned=308, unit="answers", inputs=(r10 / "units/full_rescue_reranker",)))
    out.append(Counted("r10-grade", "R10 grades (LongMemEval's own judge)", PY + [
        "experiments/z2/oracle.py", "grade", "--units", str(r10.relative_to(ROOT)),
        "--answers", str(r10_answers.relative_to(ROOT)), "--out", "logs/z3-judging/r10-grade", "--upstream-only"],
        calls=ROOT / "logs/z3-judging/r10-grade/oracle-grade-raw.jsonl", planned=438, unit="grades",
        needs=("r10-answers",)))
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
