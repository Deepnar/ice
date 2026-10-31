"""The GPU day's follow-up (2026-10-29): the corrected B7 and the 256-candidate variants, one
resumable command on Z2's runner (meter, lock, retries, clean stop, resume); state in
logs/gpu-extra-<name>. Start it by hand after the GPU day; it does not check the GPU.

  b7-calibrated   classifier variants judged like the live model (cap 5, val-fitted threshold, test F1)
  x-B-*           B reads at 256+ candidates: wider legs (300/300/192, rank up to 512), originals first,
                  the reranker keeping the policy's text, wider legs + originals first
  x-C-*           C reads at 64, 256 and wider legs — the same sweep on a second history
  x-funnel-*      where each cited turn is lost (never found / not ranked / over the budget / assembly)
                  and each leg's share of prompt pieces, tokens and cited turns delivered

    uv run --no-sync python experiments/z2/gpu_extra.py run
    uv run --no-sync python experiments/z2/gpu_extra.py status --watch
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import run_z2 as Z  # noqa: E402
from experiments.z2.gpu_day import GATES, Command, PY, Reads  # noqa: E402

B_VARIANTS = ("retest_c256_wide", "retest_c256_originals_first", "retest_c256_policy", "retest_c256_wide_originals")
C_VARIANTS = ("retest_cand64", "retest_cand256", "retest_c256_wide")


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
    out.append(funnel("B", "ice_z2_main_b", ("retest_cand64", "retest_cand256") + B_VARIANTS,
                      [f"x-B-{c.removeprefix('retest_')}" for c in B_VARIANTS]))
    out.append(funnel("C", "ice_z2_main_c", C_VARIANTS, [f"x-C-{c.removeprefix('retest_')}" for c in C_VARIANTS]))
    return out


if __name__ == "__main__":
    raise SystemExit(Z.main(stages, run_dir=lambda name: ROOT / "logs" / f"gpu-extra-{name}", prog="gpu_extra.py",
                            description=__doc__.split("\n\n")[0], runs="logs/gpu-extra-<name>"))
