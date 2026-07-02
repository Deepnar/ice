#!/usr/bin/env python3
"""
ICE Flaw Subtraction Paper Summary
===================================
Generates a paper‑ready Markdown summary from the subtraction metrics report.

Reads:  experiments/flaw_ablation/subtraction/metrics_report.json
Writes: experiments/flaw_ablation/subtraction/paper_summary.md
"""

import json
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
METRICS_FILE = "experiments/flaw_ablation/subtraction/metrics_report.json"
OUTPUT_FILE  = "experiments/flaw_ablation/subtraction/paper_summary.md"

FEATURE_CATEGORIES = {
    "Retrieval Legs": ["no_vector", "no_bm25"],
    "Fusion & Query": ["no_rrf", "hyde_on"],
    "Knowledge Graph": ["no_codex", "no_mera", "no_fuzzy_match"],
    "Memory Systems": ["no_procedural", "no_batch_summary"],
    "Scoping & Diversity": ["no_cluster_restrict", "no_session_diversify"],
    "Prompt Assembly": ["static_budget", "no_sliding_window", "no_keyword_boost", "no_recency_boost"],
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def fmt_score(val, decimals=2):
    if val is None:
        return "—"
    return f"{val:+.{decimals}f}" if isinstance(val, (int, float)) and val != 0 else f"{val:.{decimals}f}"

def fmt_delta(val, decimals=2):
    if val is None:
        return "—"
    return f"{val:+.{decimals}f}"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    report = load_json(METRICS_FILE)

    with open(OUTPUT_FILE, "w") as f:
        # ── Header ──
        f.write(f"# Flaw Subtraction Ablation — Paper Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"**Probes evaluated**: {report['n_probes_total']}  \n")
        f.write(f"**Checkpoints**: {report['n_checkpoints']} (turns 51 → 1119)  \n")
        f.write(f"**Model**: mattbucci/Qwen3.6-27B-AWQ (SGLang)  \n\n")

        # ── Baseline ──
        base = report["baseline"]
        f.write("## Baseline (All Features ON)\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Mean Score | {base['mean_score']} |\n")
        f.write(f"| Mean Tokens Injected | {base['mean_tokens']} |\n")
        f.write(f"| TUR | {base['tur']} |\n")
        f.write(f"| Hallucination % | {base['hallucination_pct']}% |\n")
        f.write(f"| Fragment Noise | {base.get('fragment_noise_mean', '—')} / 10 |\n\n")

        # ── Global Per‑Condition Table ──
        f.write("## Feature Ablation Results\n\n")
        f.write("| Condition | Score | Δ Score | Tokens | Δ Tokens | Hall% | Δ Hall% | TUR | Long Δ |\n")
        f.write("|-----------|------:|--------:|-------:|---------:|------:|--------:|----:|-------:|\n")

        # Baseline row
        base_long = report.get("longitudinal", {}).get("baseline_all_on", {}).get("mean_probe_delta", 0)
        f.write(f"| **baseline_all_on** | {base['mean_score']} | — | {base['mean_tokens']} | — | {base['hallucination_pct']}% | — | {base['tur']} | {fmt_delta(base_long)} |\n")

        # Condition rows, grouped by category
        for cat_name, cat_conds in FEATURE_CATEGORIES.items():
            f.write(f"| | | | | | | | | |\n")  # spacer
            for cond in cat_conds:
                c = report["conditions"].get(cond)
                if c is None:
                    continue
                long_delta = report.get("longitudinal", {}).get(cond, {}).get("mean_probe_delta", 0)
                f.write(f"| {cond} | {c['mean_score']} | {fmt_delta(c['delta_score'])} | "
                        f"{c['mean_tokens']} | {fmt_delta(c['delta_tokens'], 0)} | "
                        f"{c['hallucination_pct']}% | {fmt_delta(c['delta_hall_pct'])}% | "
                        f"{c['tur']} | {fmt_delta(long_delta)} |\n")

        # Vector baseline separate
        vec = report["conditions"].get("vector_baseline")
        if vec:
            vec_long = report.get("longitudinal", {}).get("vector_baseline", {}).get("mean_probe_delta", 0)
            f.write(f"| | | | | | | | | |\n")
            f.write(f"| **vector_baseline** | {vec['mean_score']} | {fmt_delta(vec['delta_score'])} | "
                    f"{vec['mean_tokens']} | {fmt_delta(vec['delta_tokens'], 0)} | "
                    f"{vec['hallucination_pct']}% | {fmt_delta(vec['delta_hall_pct'])}% | "
                    f"{vec['tur']} | {fmt_delta(vec_long)} |\n")

        # ── Category Summary ──
        f.write("\n## Category Summary\n\n")
        f.write("| Category | Avg Score | Avg Δ from Baseline |\n")
        f.write("|----------|----------:|--------------------:|\n")
        cats = report.get("categories", {})
        for cat_name, cat_data in cats.items():
            f.write(f"| {cat_name} | {cat_data.get('avg_score', '—')} | {fmt_delta(cat_data.get('avg_delta', 0))} |\n")

        # ── Longitudinal Trends ──
        f.write("\n## Longitudinal Score Evolution (Mean Score per Turn‑Index Bin)\n\n")
        long = report.get("longitudinal", {})
        # Collect all bin labels
        all_bins = set()
        for cond_data in long.values():
            all_bins.update(cond_data.get("bins", {}).keys())
        sorted_bins = sorted(all_bins, key=lambda x: int(x.split("-")[0]))

        # Header
        f.write("| Condition | " + " | ".join(sorted_bins) + " | Mean Δ |\n")
        f.write("|-----------|" + "|".join(["------:" for _ in sorted_bins]) + "|-------:|\n")

        # Baseline
        base_bins = long.get("baseline_all_on", {}).get("bins", {})
        base_delta = long.get("baseline_all_on", {}).get("mean_probe_delta", 0)
        base_row = " | ".join(f"{base_bins.get(b, {}).get('mean_score', '—')}" for b in sorted_bins)
        f.write(f"| **baseline_all_on** | {base_row} | {fmt_delta(base_delta)} |\n")

        # Each condition
        for cond in ALL_CONDITIONS:
            if cond == "baseline_all_on":
                continue
            cond_bins = long.get(cond, {}).get("bins", {})
            cond_delta = long.get(cond, {}).get("mean_probe_delta", 0)
            row = " | ".join(f"{cond_bins.get(b, {}).get('mean_score', '—')}" for b in sorted_bins)
            f.write(f"| {cond} | {row} | {fmt_delta(cond_delta)} |\n")

        # Vector baseline
        vec_bins = long.get("vector_baseline", {}).get("bins", {})
        vec_delta = long.get("vector_baseline", {}).get("mean_probe_delta", 0)
        vec_row = " | ".join(f"{vec_bins.get(b, {}).get('mean_score', '—')}" for b in sorted_bins)
        f.write(f"| **vector_baseline** | {vec_row} | {fmt_delta(vec_delta)} |\n")

        # ── Key Findings ──
        f.write("\n## Key Findings\n\n")

        # Find the largest negative delta (most impactful feature to lose)
        worst = max(report["conditions"].items(), key=lambda x: abs(x[1].get("delta_score", 0)), default=(None, {}))
        if worst[0]:
            f.write(f"1. **Largest quality drop**: `{worst[0]}` ({fmt_delta(worst[1].get('delta_score', 0))} pts vs baseline) — "
                    f"disabling this feature caused the biggest retrieval quality degradation.\n")

        # Best longitudinal performer
        best_long = max(long.items(), key=lambda x: x[1].get("mean_probe_delta", 0), default=(None, {}))
        if best_long[0]:
            f.write(f"2. **Strongest longitudinal improvement**: `{best_long[0]}` ({fmt_delta(best_long[1].get('mean_probe_delta', 0))} mean delta) — "
                    f"probes answered under this condition improved most from early to late checkpoints.\n")

        # Token efficiency
        f.write(f"3. **Token efficiency**: Baseline TUR of {base['tur']} vs vector baseline TUR of {report['conditions']['vector_baseline']['tur']} — "
                f"ICE delivers {base['tur'] / report['conditions']['vector_baseline']['tur']:.1f}× more intelligence per token.\n")

    print(f"Paper summary written to {OUTPUT_FILE}")


# Replicate ALL_CONDITIONS for the longitudinal table
ALL_CONDITIONS = [
    "baseline_all_on",
    "vector_baseline",
    "no_vector",
    "no_bm25",
    "no_rrf",
    "hyde_on",
    "no_cluster_restrict",
    "no_session_diversify",
    "no_codex",
    "no_mera",
    "no_fuzzy_match",
    "no_procedural",
    "no_batch_summary",
    "static_budget",
    "no_sliding_window",
    "no_keyword_boost",
    "no_recency_boost",
]

if __name__ == "__main__":
    main()