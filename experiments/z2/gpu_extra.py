"""The GPU day's follow-up (2026-10-29): the corrected B7 and the 256-candidate variants, one
resumable command on Z2's runner (meter, lock, retries, clean stop, resume); state in
logs/gpu-extra-<name>. Start it by hand after the GPU day; it does not check the GPU.

  b7-calibrated   classifier variants judged like the live model (cap 5, val-fitted threshold, test F1)
  x-B-*           B reads at 256+ candidates: wider legs (300/300/192, rank up to 512), originals first,
                  the reranker keeping the policy's text, wider legs + originals first
  x-C-*           C reads at 64, 256 and wider legs — the same sweep on a second history
  x-funnel-*      where each cited turn is lost (never found / not ranked / over the budget / assembly)
                  and each leg's share of prompt pieces, tokens and cited turns delivered
  r10, r10-report S12's 146 refused LongMemEval questions re-read with the reranker rescue (R10), then
                  r10_report.py on them (its output: logs/gpu-extra-main/r10-report.log)
  nomem, nomem-report  R10's other side: 146 of the maintainer's no-memory messages read on a copy of B's
                  store with the reranker rescue — how often it lets in memory nobody needed, at each bar
                  (nomem_cost.py; its output: logs/gpu-extra-main/nomem-report.log)

    uv run --no-sync python experiments/z2/gpu_extra.py run
    uv run --no-sync python experiments/z2/gpu_extra.py status --watch
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import run_z2 as Z  # noqa: E402
from experiments.z2.gpu_day import GATES, Command, PY, Reads  # noqa: E402

B_VARIANTS = ("retest_c256_wide", "retest_c256_originals_first", "retest_c256_policy", "retest_c256_wide_originals")
C_VARIANTS = ("retest_cand64", "retest_cand256", "retest_c256_wide")


class RescueReread(Z.Oracle):
    """R10 (Z2 F4): the 146 S12 questions B2 refused, replayed and read again with the reranker rescue
    beside full ICE (oracle.py --slice experiments/lme/r10_slice.json --rescue-read)."""
    SLICE = ROOT / "experiments/lme/r10_slice.json"

    def __init__(self):
        Z.Stage.__init__(self, key="r10", title="R10 reranker rescue: S12's refused questions", unit="questions",
                         estimate=320.0)

    def out(self, ctx):
        return ROOT / "logs" / "z2-run-main" / "s12-r10"

    def database(self, ctx):
        return "ice_z2_r10"

    def total(self, ctx):
        from experiments.z2 import oracle
        return oracle.progress(self.out(ctx), total=len(json.loads(self.SLICE.read_text())["question_ids"]))[1]

    def extra_args(self, ctx):
        return ["--slice", str(self.SLICE.relative_to(ROOT)), "--rescue-read"]


class NoMemoryCost(Z.Stage):
    """R10's other side: the maintainer's no-memory messages read with the reranker rescue (nomem_cost.py)."""

    def __init__(self):
        super().__init__(key="nomem", title="R10 other side: no-memory messages", unit="messages", estimate=30.0)

    def out(self, ctx):
        return ROOT / "logs" / "z2-run-main" / "nomem-cost"

    def total(self, ctx):
        from experiments.z2 import nomem_cost
        return nomem_cost.progress(self.out(ctx))[1]

    def done(self, ctx):
        from experiments.z2 import nomem_cost
        return nomem_cost.progress(self.out(ctx))[0]

    def complete(self, ctx):
        return self.done(ctx) >= self.total(ctx)

    def run(self, ctx):
        log = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = PY + ["experiments/z2/nomem_cost.py", "run", "--database", "ice_z2_nomem_b",
                        "--out", str(self.out(ctx).relative_to(ROOT))]
            return ctx.run_child(cmd, log), Z._tail(log)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


def funnel(history, database, conditions, needs):
    return Command(f"x-funnel-{history}", f"where evidence is lost + leg shares ({history})", PY + [
        "experiments/z2/funnel.py", "--history", history, "--database", database,
        "--units", f"logs/z2-retest-{history}/units", "--conditions", ",".join(conditions),
        "--occurrences", f"logs/z2-questions-final/{history}/occurrences.json"], needs=needs, estimate=300)


def stages(_limit=None):
    out = [Command("b7-calibrated", "B7 judged like the live model: cap 5, val-fitted threshold, test F1",
                   PY + ["experiments/z2/b7_calibrated.py"], estimate=3600)]
    out += [Reads(f"x-B-{c.removeprefix('retest_')}", "B", c, needs=()) for c in B_VARIANTS]
    out += [Reads(f"x-C-{c.removeprefix('retest_')}", "C", c, needs=(), estimate=16.0) for c in C_VARIANTS]
    # 2026-10-30: 256 chosen; the order/summary settings the Z3-judging answers compare, read at 256
    out += [Reads(f"x2-B-{c.removeprefix('retest_')}", "B", c, needs=()) for c in ("retest_c256_order", "retest_c256_no_summaries")]
    out += [Reads(f"x2-C-{c.removeprefix('retest_')}", "C", c, needs=(), estimate=16.0)
            for c in ("retest_c256_order", "retest_c256_originals_first", "retest_c256_policy",
                      "retest_c256_wide_originals", "retest_c256_no_summaries")]
    # 2026-10-31: the reranker's inputs (Z2 F22) on B and C, and their funnel
    rq = ("retest_c256_rq_context", "retest_c256_rq_topical", "retest_c256_rq_both")
    out += [Reads(f"x3-{h}-{c.removeprefix('retest_')}", h, c, needs=(), estimate=16.0 if h == "C" else 15.0)
            for h in ("B", "C") for c in rq]
    for h, db in (("B", "ice_z2_main_b"), ("C", "ice_z2_main_c")):
        f = funnel(h, db, ("retest_cand256",) + rq, [f"x3-{h}-{c.removeprefix('retest_')}" for c in rq])
        f.key, f.title = f"x3-funnel-{h}", f"reranker inputs: where evidence is lost ({h})"
        out.append(f)
    # 2026-11-01: R10 — S12's refused questions re-read with the reranker rescue, then its report
    out += [RescueReread(),
            Command("r10-report", "R10 report: answer in the prompt per setup and bar",
                    PY + ["experiments/z2/r10_report.py", "--units", "logs/z2-run-main/s12-r10/units"],
                    needs=("r10",), estimate=60)]
    out += [NoMemoryCost(),
            Command("nomem-report", "R10 other side report: memory let in per bar, needed vs not",
                    PY + ["experiments/z2/nomem_cost.py", "report", "--out", "logs/z2-run-main/nomem-cost"],
                    needs=("nomem",), estimate=30)]
    out.append(funnel("B", "ice_z2_main_b", ("retest_cand64", "retest_cand256") + B_VARIANTS,
                      [f"x-B-{c.removeprefix('retest_')}" for c in B_VARIANTS]))
    out.append(funnel("C", "ice_z2_main_c", C_VARIANTS, [f"x-C-{c.removeprefix('retest_')}" for c in C_VARIANTS]))
    return out


if __name__ == "__main__":
    raise SystemExit(Z.main(stages, run_dir=lambda name: ROOT / "logs" / f"gpu-extra-{name}", prog="gpu_extra.py",
                            description=__doc__.split("\n\n")[0], runs="logs/gpu-extra-<name>"))
