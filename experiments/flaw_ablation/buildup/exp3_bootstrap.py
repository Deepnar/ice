#!/usr/bin/env python3
"""
Experiment 3 (Flaw-Buildup Ablation) — Bootstrap Confidence Intervals
=====================================================================
Post-paper armoring pass. The published Exp 3 table reports point deltas only
(bare point estimates on N=67); a findings-first reframe promotes the RRF /
BM25 steps to the paper's headline, so those deltas need the same paired
percentile-bootstrap CIs the paper already uses for Experiment 2
(ICE_paper.tex, "paired score differences are computed within-probe before
resampling"; 10,000 resamples).

The ablation data is fully paired: the same 67 probes are scored under every
condition, so within-probe step deltas are well defined. This script:
  * one-sample bootstraps each condition's mean score,
  * paired-bootstraps every step delta (feature vs previous step),
  * paired-bootstraps every cumulative delta (feature vs bare_vector),
  * paired-bootstraps full_ice vs the single-leg vector_baseline reference,
and flags which intervals exclude zero.

Reads the FROZEN buildup record read-only; writes only its report under
experiments/flaw_ablation/buildup/results/.
"""

import json
import os

import numpy as np

EVAL_FILE = "experiments/flaw_ablation/buildup/intermediates/evaluation_raw.json"
OUT_JSON = "experiments/flaw_ablation/buildup/results/exp3_bootstrap_report.json"
OUT_MD = "experiments/flaw_ablation/buildup/results/exp3_bootstrap_report.md"

# Canonical buildup order (buildup_runner.py:72). vector_baseline is a reference
# point, not a step in the chain, so it is handled separately.
BUILDUP_CHAIN = [
    "bare_vector", "add_bm25", "add_rrf", "add_hyde", "add_cluster_restrict",
    "add_session_diversify", "add_codex", "add_mera", "add_procedural",
    "add_batch_summary", "add_dynamic_budget", "add_sliding_window",
    "add_keyword_boost", "full_ice",
]
REFERENCE = "vector_baseline"

B = 10_000
SEED = 42
CI_LO, CI_HI = 2.5, 97.5


def load_scores():
    """probe_id -> {condition: score}. Only entries with a numeric score kept."""
    eval_raw = json.load(open(EVAL_FILE))
    scores = {}
    for item in eval_raw:
        pid = item["probe_id"]
        asc = item.get("absolute_scores", {})
        row = {}
        for cond, sd in asc.items():
            if isinstance(sd, dict) and sd.get("score") is not None:
                row[cond] = float(sd["score"])
        scores[pid] = row
    return scores


def boot_mean(rng, values):
    """One-sample percentile-bootstrap CI of the mean."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(B, n))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), tuple(np.percentile(means, [CI_LO, CI_HI]))


def boot_paired_delta(rng, scores, cond_a, cond_b):
    """
    Paired percentile-bootstrap of mean(cond_a - cond_b) over probes scored
    under BOTH conditions. Positive => cond_a scores higher than cond_b.
    """
    diffs = [row[cond_a] - row[cond_b]
             for row in scores.values() if cond_a in row and cond_b in row]
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(B, n))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [CI_LO, CI_HI])
    return {
        "point": float(arr.mean()),
        "ci": [float(lo), float(hi)],
        "n_paired": int(n),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def fmt(x):
    return f"{x:+.2f}"


def main():
    scores = load_scores()
    rng = np.random.default_rng(SEED)

    # Per-condition mean-score CIs
    cond_scores = {}
    for cond in BUILDUP_CHAIN + [REFERENCE]:
        vals = [row[cond] for row in scores.values() if cond in row]
        mean, (lo, hi) = boot_mean(rng, vals)
        cond_scores[cond] = {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)],
                             "n": len(vals)}

    # Paired step deltas (curr vs prev in the chain)
    step_deltas = {}
    for i in range(1, len(BUILDUP_CHAIN)):
        prev, curr = BUILDUP_CHAIN[i - 1], BUILDUP_CHAIN[i]
        step_deltas[curr] = boot_paired_delta(rng, scores, curr, prev)

    # Paired cumulative deltas (curr vs bare_vector)
    cum_deltas = {}
    for cond in BUILDUP_CHAIN[1:]:
        cum_deltas[cond] = boot_paired_delta(rng, scores, cond, "bare_vector")

    # Headline contrasts called out explicitly in the paper
    contrasts = {
        # BM25-without-fusion damage, then RRF rescue on top of it
        "bm25_damage (add_bm25 - bare_vector)":
            boot_paired_delta(rng, scores, "add_bm25", "bare_vector"),
        "rrf_rescue (add_rrf - add_bm25)":
            boot_paired_delta(rng, scores, "add_rrf", "add_bm25"),
        "rrf_vs_bare (add_rrf - bare_vector)":
            boot_paired_delta(rng, scores, "add_rrf", "bare_vector"),
        "full_ice_vs_bare (full_ice - bare_vector)":
            boot_paired_delta(rng, scores, "full_ice", "bare_vector"),
        "full_ice_vs_vecbaseline (full_ice - vector_baseline)":
            boot_paired_delta(rng, scores, "full_ice", REFERENCE),
    }

    report = {
        "experiment": "flaw_buildup_bootstrap",
        "method": "paired within-probe percentile bootstrap, B=10,000, seed=42",
        "n_probes": len(scores),
        "condition_scores": cond_scores,
        "step_deltas": step_deltas,
        "cumulative_deltas": cum_deltas,
        "headline_contrasts": contrasts,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(report, open(OUT_JSON, "w"), indent=2)

    # ── Markdown for the paper ──────────────────────────────────────
    lines = []
    lines.append("# Experiment 3 — Bootstrap CIs (paired, B=10,000, seed=42)\n")
    lines.append(f"N = {len(scores)} probes. Deltas are paired within-probe; "
                 "CI is 95% percentile.\n")
    lines.append("## Step deltas (feature vs previous step)\n")
    lines.append("| Step | Δ (point) | 95% CI | n | excl. 0 |")
    lines.append("|------|----------:|:------:|--:|:-------:|")
    for cond in BUILDUP_CHAIN[1:]:
        d = step_deltas[cond]
        star = "**yes**" if d["excludes_zero"] else "no"
        lines.append(f"| {cond} | {fmt(d['point'])} | "
                     f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] | {d['n_paired']} | {star} |")
    lines.append("\n## Cumulative deltas (vs bare_vector)\n")
    lines.append("| Condition | Cum Δ | 95% CI | excl. 0 |")
    lines.append("|-----------|------:|:------:|:-------:|")
    for cond in BUILDUP_CHAIN[1:]:
        d = cum_deltas[cond]
        star = "**yes**" if d["excludes_zero"] else "no"
        lines.append(f"| {cond} | {fmt(d['point'])} | "
                     f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] | {star} |")
    lines.append("\n## Headline contrasts\n")
    lines.append("| Contrast | Δ | 95% CI | n | excl. 0 |")
    lines.append("|----------|--:|:------:|--:|:-------:|")
    for name, d in contrasts.items():
        star = "**yes**" if d["excludes_zero"] else "no"
        lines.append(f"| {name} | {fmt(d['point'])} | "
                     f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] | {d['n_paired']} | {star} |")
    open(OUT_MD, "w").write("\n".join(lines) + "\n")

    # ── Console ─────────────────────────────────────────────────────
    print(f"N = {len(scores)} probes | B={B} | seed={SEED}\n")
    print(f"{'Step':<24} {'Δ':>7} {'95% CI':>20} {'excl0':>6}")
    print("-" * 60)
    for cond in BUILDUP_CHAIN[1:]:
        d = step_deltas[cond]
        ci = f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
        print(f"{cond:<24} {d['point']:>+7.2f} {ci:>20} {'YES' if d['excludes_zero'] else '.':>6}")
    print("\nHeadline contrasts:")
    for name, d in contrasts.items():
        ci = f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
        print(f"  {name:<48} {d['point']:>+6.2f}  {ci}  {'EXCLUDES 0' if d['excludes_zero'] else ''}")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
