#!/usr/bin/env python3
"""
ICE Flaw Buildup Paper Summary (updated for step‑by‑step deltas)
==================================================================
Reads the new metrics_report.json and produces a complete Markdown summary.
"""

import json, os
from datetime import datetime

METRICS_FILE = "experiments/flaw_ablation/buildup/results/metrics_report.json"
OUTPUT_FILE  = "experiments/flaw_ablation/buildup/results/paper_summary.md"

STEPS = [
    "bare_vector", "add_bm25", "add_rrf", "add_hyde",
    "add_cluster_restrict", "add_session_diversify", "add_codex", "add_mera",
    "add_procedural", "add_batch_summary", "add_dynamic_budget",
    "add_sliding_window", "add_keyword_boost", "full_ice", "vector_baseline"
]

def load_json(path):
    with open(path, "r") as f: return json.load(f)
def fmt(val, d=2):
    if val is None: return "—"
    return f"{val:+.{d}f}" if isinstance(val, (int, float)) else str(val)

def main():
    report = load_json(METRICS_FILE)
    conds = report["conditions"]
    long  = report.get("longitudinal", {})
    steps = report.get("step_deltas", {})
    highlights = report.get("highlights", {})

    with open(OUTPUT_FILE, "w") as f:
        f.write(f"# Flaw Buildup Ablation — Paper Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"**Probes**: {report['n_probes_total']} | **Model**: Qwen3‑14B‑AWQ\n")
        f.write(f"**Design**: Single‑pass on fully‑mature database (turn 1119).\n\n")

        # ── 1. Cumulative Feature Addition (with step‑by‑step deltas) ─
        f.write("## 1. Cumulative Feature Addition (starting from bare vector)\n\n")
        f.write("| Step | Score | Step Δ | Cum Δ | SPF | Tokens | Frags | Rec Δ |\n")
        f.write("|------|------:|-------:|------:|----:|-------:|------:|------:|\n")

        for i, cond in enumerate(STEPS):
            c = conds.get(cond)
            if not c: continue
            step_delta = fmt(steps.get(cond)) if i > 0 else "  ·"
            rec = long.get(cond, {}).get("recency_delta", 0)
            f.write(f"| {cond} | {c['mean_score']} | {step_delta} | "
                    f"{fmt(c['cumulative_from_bare'])} | {c['spf']} | "
                    f"{c['mean_tokens']} | {c['mean_frags']} | {fmt(rec)} |\n")

        # ── 2. Largest changes ──────────────────────────────────────
        f.write("\n## 2. Largest Stepwise Changes\n\n")
        pos = highlights.get("largest_positive_step", {})
        neg = highlights.get("largest_negative_step", {})
        f.write(f"- **Largest gain**: `{pos.get('condition','?')}` (+{pos.get('delta','?')})\n")
        f.write(f"- **Largest drop**: `{neg.get('condition','?')}` ({neg.get('delta','?')})\n\n")

        # ── 3. Recency Effect ───────────────────────────────────────
        f.write("## 3. Recency Effect — Score by Origin Split (Fact Age)\n\n")
        f.write("| Condition | Early (0‑400) | Mid (400‑800) | Late (800‑1200) | Recency Δ |\n")
        f.write("|-----------|--------------:|--------------:|----------------:|----------:|\n")
        for cond in STEPS:
            bins = long.get(cond, {}).get("bins", {})
            early_vals = [bins.get(f"orig_{l}_{h}", "—") for l,h in [(0,200),(200,400)]]
            mid_vals   = [bins.get(f"orig_{l}_{h}", "—") for l,h in [(400,600),(600,800)]]
            late_vals  = [bins.get(f"orig_{l}_{h}", "—") for l,h in [(800,1000),(1000,1200)]]
            early = "/".join(str(v) for v in early_vals)
            mid   = "/".join(str(v) for v in mid_vals)
            late  = "/".join(str(v) for v in late_vals)
            rec = long.get(cond, {}).get("recency_delta", 0)
            f.write(f"| {cond} | {early} | {mid} | {late} | {fmt(rec)} |\n")

        # ── 4. Key Findings ─────────────────────────────────────────
        f.write("\n## 4. Key Findings\n\n")
        f.write("1. The step‑by‑step deltas show exactly which feature improved retrieval quality at the moment it was introduced.\n")
        f.write("2. A negative step delta indicates the feature may have introduced noise or interacted poorly with previously active features.\n")
        f.write("3. The cumulative delta column shows how far each stage is from bare‑vector, giving the overall progress of the system build‑up.\n")
        f.write("4. SPF (Score per Fragment) measures retrieval precision — higher values mean each injected fragment contributed more to the final answer quality.\n")

    print(f"Paper summary written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()