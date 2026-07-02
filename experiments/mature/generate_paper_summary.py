#!/usr/bin/env python3
"""
ICE‑Mature Comprehensive Paper Summary — v2
Generates all tables: global (with SPF), MoE, per‑conversation, distributions,
temporal quality, longitudinal, leg contributions, fragment noise,
memory maturity, gating failures, and fragment‑score correlation.
"""

import json, os
from datetime import datetime

REPORTS = [
    ("metrics_complete_report.json",          "All Conversations (ICE‑Dev included)"),
    ("metrics_complete_report_no_ice_dev.json", "ICE‑Dev Excluded (fair comparison)"),
]

CORE_CONDITIONS = [
    "vector_rag_baseline_generalist",
    "vector_rag_moe",
    "full_ice_generalist",
    "full_ice_moe",
]

def main():
    for report_file, label in REPORTS:
        path = os.path.join("experiments/mature/results", report_file)
        if not os.path.exists(path):
            print(f"Skipping {report_file} — not found.")
            continue

        data = json.load(open(path))
        out_path = path.replace(".json", "_paper_summary_full.md")

        ice_cond = data["global_summary"].get("full_ice_generalist", {})
        total_probes = ice_cond.get("n_scores", 0)

        with open(out_path, "w") as f:
            f.write(f"# ICE‑Mature Comprehensive Paper Summary — {label}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"**Total Probes**: {total_probes}\n\n")

            # ── 1. Global Comparison (with SPF) ──────────────────────
            f.write("## 1. Global Comparison\n\n")
            f.write("| Condition | Score | Tokens | Fragments | SPF | TUR | Win% | Hall% |\n")
            f.write("|-----------|------:|-------:|----------:|-----:|----:|-----:|------:|\n")
            for cond in CORE_CONDITIONS:
                c = data["global_summary"].get(cond, {})
                f.write(f"| {cond} | {c.get('avg_score','?')} | {c.get('avg_tokens_injected','?')} | "
                        f"{c.get('avg_fragments','?')} | {c.get('spf','?')} | "
                        f"{c.get('tur','?')} | {c.get('win_rate_pct','?')}% | {c.get('hallucination_pct','?')}% |\n")
            dels = data["global_summary"].get("comparative_deltas", {})
            f.write(f"\n- **Token savings vs vector**: {dels.get('token_savings_vs_vector_pct','?')}%\n")
            f.write(f"- **Quality delta vs vector**: {dels.get('quality_gain_vs_vector_pts','?')} pts\n")
            f.write(f"- **Fragment reduction vs vector**: {dels.get('fragment_count_reduction_pct','?')}%\n\n")

            # ── 2. MoE vs Generalist ──────────────────────────────────
            f.write("## 2. MoE vs Generalist\n\n")
            moe = data.get("moe_vs_generalist", {})
            f.write("| Base | MoE Score | Gen Score | MoE Tokens | Gen Tokens | MoE Win% | Gen Win% | MoE Hall% | Gen Hall% |\n")
            f.write("|------|----------:|----------:|-----------:|-----------:|---------:|---------:|----------:|----------:|\n")
            for base in ["vector_rag", "full_ice", "global"]:
                b = moe.get(base, {})
                m = b.get("moe", {})
                g = b.get("generalist", {})
                f.write(f"| {base} | {m.get('avg_score','?')} | {g.get('avg_score','?')} | "
                        f"{m.get('avg_tokens','?')} | {g.get('avg_tokens','?')} | "
                        f"{m.get('win_rate_pct','?')}% | {g.get('win_rate_pct','?')}% | "
                        f"{m.get('hallucination_pct','?')}% | {g.get('hallucination_pct','?')}% |\n")
            global_moe = moe.get("global", {}).get("delta_moe_vs_generalist", {})
            f.write(f"\n- **Global MoE vs Gen score delta**: {global_moe.get('score_gain','?')} pts\n")
            f.write(f"- **Tokens saved by MoE**: {global_moe.get('tokens_saved','?')}\n")
            f.write(f"- **Hallucination reduction by MoE**: {global_moe.get('hallucination_reduction_pct','?')}%\n\n")

            # ── 3. Score Distributions ────────────────────────────────
            f.write("## 3. Score Distributions\n\n")
            f.write("| Score | ICE Gen % | ICE MoE % | Vector Gen % | Vector MoE % |\n")
            f.write("|-------|----------:|----------:|-------------:|-------------:|\n")
            for s in range(1, 6):
                row = f"| {s} |"
                for cond in CORE_CONDITIONS:
                    dist = data["global_summary"].get(cond, {}).get("score_distribution", {})
                    row += f" {dist.get(f'score_{s}_pct',0)}% |"
                f.write(row + "\n")

            # ── 4. Temporal Score Quality ────────────────────────────
            f.write("\n## 4. Temporal Score Quality\n\n")
            tq = data.get("temporal_score_quality", {})
            f.write("| Condition | Good (4-5) | OK (3) | Poor (1-2) |\n")
            f.write("|-----------|-----------:|-------:|-----------:|\n")
            for cond in CORE_CONDITIONS:
                q = tq.get(cond, {})
                f.write(f"| {cond} | {q.get('good_pct','?')}% | {q.get('ok_pct','?')}% | {q.get('poor_pct','?')}% |\n")

            # ── 5. Per‑Conversation ───────────────────────────────────
            f.write("\n## 5. Per‑Conversation Breakdown\n\n")
            f.write("| Conv | ICE Score | Vector Score | ICE Tokens | Vector Tokens | ICE Win% | Vector Win% |\n")
            f.write("|------|----------:|-------------:|-----------:|--------------:|---------:|------------:|\n")
            for conv, conv_data in data["conversation_lenses"].items():
                ice = conv_data.get("full_ice_generalist", {})
                vec = conv_data.get("vector_rag_baseline_generalist", {})
                f.write(f"| {conv} | {ice.get('avg_score','?')} | {vec.get('avg_score','?')} | "
                        f"{ice.get('avg_tokens_injected','?')} | {vec.get('avg_tokens_injected','?')} | "
                        f"{ice.get('win_rate_pct','?')}% | {vec.get('win_rate_pct','?')}% |\n")

            # ── 6. Longitudinal Global ──────────────────────────────
            f.write("\n## 6. Longitudinal Score Evolution (Global Binned)\n\n")
            long_global = data.get("longitudinal_global", {})
            all_bins = sorted({b for v in long_global.values() for b in v.get("turn_bins", [])})
            if all_bins:
                f.write("| Condition | " + " | ".join(f"{b}" for b in all_bins) + " |\n")
                f.write("|-----------|" + "|".join(["------:" for _ in all_bins]) + "|\n")
                for cond in CORE_CONDITIONS:
                    bins = long_global.get(cond, {})
                    scores = bins.get("mean_scores", [])
                    row = " | ".join(f"{s:.2f}" for s in scores) if len(scores) == len(all_bins) else " | ".join("—" for _ in all_bins)
                    f.write(f"| {cond} | {row} |\n")

            # ── 7. Leg Contributions ──────────────────────────────────
            f.write("\n## 7. Leg Contributions (ICE Generalist)\n\n")
            legs = data["leg_contributions_global"].get("full_ice_generalist", {})
            f.write("| Source | Avg Count | % of Total |\n|--------|----------:|-----------:|\n")
            for src, info in sorted(legs.items()):
                if src != "total_avg_fragments":
                    f.write(f"| {src} | {info['avg_count']} | {info['pct_of_total']}% |\n")
            f.write(f"| **Total** | {legs.get('total_avg_fragments','?')} | 100% |\n")

            # ── 8. Fragment Noise ─────────────────────────────────────
            f.write("\n## 8. Fragment Noise\n\n")
            frag_noise = data.get("fragment_noise_global", {})
            f.write("| Condition | Mean Noise |\n|-----------|----------:|\n")
            for cond in CORE_CONDITIONS:
                fn = frag_noise.get(cond, {})
                f.write(f"| {cond} | {fn.get('mean_noise','?')} |\n")

            # ── 9. Memory Maturity ─────────────────────────────────────
            f.write("\n## 9. Memory Maturity\n\n")
            mem = data.get("memory_maturity", {})
            f.write("| Conversation | Max Simulated Days | Max Decay Cycles | Mean Days/Checkpoint |\n")
            f.write("|-------------|-------------------:|-----------------:|----------------------:|\n")
            for conv, mat in mem.items():
                f.write(f"| {conv} | {mat['max_simulated_days']} | {mat['max_decay_cycles']} | {mat['mean_simulated_days_per_checkpoint']} |\n")

            # ── 10. Gating Failures ────────────────────────────────────
            f.write("\n## 10. Gating Failures (Zero\_Shot mis‑classified)\n\n")
            gf = data.get("gating_failures", [])
            f.write(f"Total probes where classifier said Zero_Shot but ICE score < 3: **{len(gf)}**\n\n")
            if gf:
                f.write("| Conversation | Probe | Question | ICE Score |\n")
                f.write("|-------------|-------|----------|----------:|\n")
                for item in gf[:10]:
                    f.write(f"| {item.get('conversation_id','?')[:8]}... | {item.get('probe_id','?')} | {item.get('question','')[:60]}... | {item.get('ice_score','?')} |\n")

            # ── 11. Fragment‑Score Correlation ───────────────────────
            f.write("\n## 11. Fragment‑Count vs Score Correlation\n\n")
            corr = data.get("fragment_score_correlation", {})
            f.write("| Condition | Correlation |\n|-----------|------------:|\n")
            for cond, val in corr.items():
                f.write(f"| {cond} | {val if val is not None else '—'} |\n")

        print(f"Summary saved → {out_path}")

if __name__ == "__main__":
    main()