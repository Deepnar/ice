#!/usr/bin/env python3
"""Z1: merge every bake-off run into one table, and say what is DISQUALIFYING.

**Why a separate reporter.** The bake-off ran in several passes — the first five
jobs, then `lfm2.5:8b` alone after its download finished, then the four added
later. Reading one JSON gives a partial picture, and a partial picture is how a
model gets crowned on the two jobs someone happened to look at.

**⚑ NO COMPOSITE SCORE, ON PURPOSE.** Weighting nine jobs into one number would
be inventing the weights — and the weights would decide the winner, silently.
Instead this prints every job side by side and applies **disqualifiers**: bright
lines where a model is not merely worse but unusable for that job. Ranking
happens among what survives, and the reader can see why anything was cut.

⚠ Every number here is n=30 turns (n=10 for doc_kind, n=8 for procedural, n=9
for the reconciler). Small gaps are NOT resolvable. The disqualifiers are gross
failures, which is exactly what this sample size CAN see.

  uv run python scripts/z1/bakeoff_report.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DIR = "experiments/curation_files/bakeoff"

# Bright lines. Each is a failure of KIND, not of degree — a model tripping one
# cannot do that job at all, so no amount of strength elsewhere redeems it.
MEDIAN_SOURCE_CHARS = 3553   # measured over the same 30 sampled turns


def disqualifiers(j: dict) -> list[str]:
    out = []
    s = j.get("summary") or {}
    # ⚑ COVERAGE REWARDS NOT SUMMARISING, and this is how it shows up.
    # `summary_coverage` counts must-terms present in the prose. A model that
    # echoes the turn back contains every term by construction and scores near
    # 1.0. Measured 2026-08-26: `lfm2.5:8b` emitted prose at **95% of the source
    # length** and took the HIGHEST coverage of any model (0.973). That is not a
    # good summary, it is the absence of summarisation — and coverage is the
    # trust gate deciding whether the summary REPLACES the raw turn.
    # ⚠ This is a SECOND hole in the metric, distinct from the `Key terms:`
    # circularity that TRAPS #45 caught and `strip_generated_index` fixed.
    pc = s.get("prose_chars_median")
    if pc and pc > 0.9 * MEDIAN_SOURCE_CHARS:
        out.append(f"prose is {100*pc/MEDIAN_SOURCE_CHARS:.0f}% of source length "
                   f"— copying, not summarising (coverage is meaningless here)")
    rc = j.get("reconcile") or {}
    if rc.get("accuracy") == 0.0:
        out.append("reconciler 0/9 — never once correct on a 3-way choice")
    if s and s.get("coverage_mean") is not None and s["coverage_mean"] < 0.40:
        out.append(f"summary coverage {s['coverage_mean']:.2f} — loses most must-terms")
    if s and s.get("empty", 0) >= 3:
        out.append(f"{s['empty']} empty summaries")
    if s and s.get("no_abstract", 0) >= 15:
        out.append(f"no Abstract line on {s['no_abstract']}/30 — hierarchy level 3 missing")
    r = j.get("raw_slice") or {}
    if r.get("invented_frac") is not None and r["invented_frac"] >= 0.25:
        out.append(f"slicer invents {100*r['invented_frac']:.0f}% of its output")
    c = j.get("conv_fold") or {}
    if c.get("chars_after_each"):
        ch = c["chars_after_each"]
        if ch and ch[-1] < 0.4 * max(ch):
            out.append(f"fold collapses to {ch[-1]} chars from a peak of {max(ch)}")
        if ch and ch[0] == 0:
            out.append("fold produced nothing on the first chunk")
    if (j.get("doc_kind") or {}).get("VOID"):
        out.append("doc_kind VOID — model calls failed, default supplied answers")
    return out


def main() -> int:
    runs = sorted(glob.glob(f"{DIR}/bakeoff_*.json"))
    if not runs:
        print("no runs found")
        return 1
    merged: dict[str, dict] = {}
    for path in runs:
        for model, jobs in json.load(open(path))["results"].items():
            slot = merged.setdefault(model, {})
            for job, res in jobs.items():
                # Later runs win: the harness was fixed mid-session (doc_kind
                # crediting the DOCUMENT default, the procedural column name),
                # so an older row for the same job is a retracted measurement.
                if job == "_error" and slot:
                    continue
                slot[job] = res
    print(f"merged {len(runs)} run files · {len(merged)} models\n")

    hdr = (f"{'model':22}{'cov':>6}{'noAb':>5}{'role':>6}{'inv':>6}{'recon':>6}"
           f"{'dkind':>6}{'proc':>6}{'bsum':>5}{'fold✓':>6}")
    print(hdr)
    print("-" * len(hdr))
    survivors = []
    for model, j in sorted(merged.items()):
        if "_error" in j and len(j) == 1:
            print(f"{model:22}  ⛔ never measured — {str(j['_error'])[:50]}")
            continue
        s, r = j.get("summary") or {}, j.get("raw_slice") or {}
        rc, dk = j.get("reconcile") or {}, j.get("doc_kind") or {}
        pr, bs = j.get("procedural") or {}, j.get("batch_summary") or {}
        cf = j.get("conv_fold") or {}
        f = lambda v, w=6, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'—':>{w}}")
        fold_ok = "—"
        if cf.get("chars_after_each"):
            fold_ok = "yes" if cf.get("shrank_steps", 9) == 0 else f"{cf['shrank_steps']}↓"
        print(f"{model:22}{f(s.get('coverage_mean'))}{s.get('no_abstract','—'):>5}"
              f"{f(r.get('role_accuracy'))}{f(r.get('invented_frac'))}"
              f"{f(rc.get('accuracy'))}{f(dk.get('accuracy'))}"
              f"{f(pr.get('accept_rate'))}{bs.get('summaries_written','—'):>5}{fold_ok:>6}")
        dq = disqualifiers(j)
        if dq:
            for d in dq:
                print(f"{'':22}  ⛔ {d}")
        else:
            survivors.append((model, j))

    print(f"\nSURVIVORS (no disqualifier): {', '.join(m for m, _ in survivors) or 'none'}")
    print("\n⚠ n=30 turns (doc_kind 10, procedural 8, reconciler 9). Small gaps "
          "are not resolvable;\n  the disqualifiers above are gross failures, "
          "which this sample size CAN see.")
    print("⚠ Every job here is a PRESENCE or FORMAT check. Faithfulness — "
          "whether a summary\n  invents — is stage 2 (`judge_summaries.py`), "
          "and no model is chosen without it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
