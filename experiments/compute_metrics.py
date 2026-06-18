#!/usr/bin/env python3
"""
ICE Metrics Aggregator — Final Paper‑Ready Version (v3.1)
==========================================================
Produces a complete statistics report with:
- Cohort‑split summaries (gated / forced / global) with standard deviations
- Flaw‑lens deep dive
- Six‑core benchmark with deltas
- MoE vs Generalist (core conditions only, clean)
- Longitudinal knowledge‑accumulation curves (per question and per cohort)
- Ablation analysis (HyDE, procedural, scope, sliding window)
- Gating failure analysis
- Fragment noise per cohort
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import statistics

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EVAL_RAW = "experiments/results_phase2/evaluation_raw.json"
MASTER_RESULTS = "experiments/results_phase2/master_results.json"
OUTPUT_FILE = "experiments/results_phase2/metrics_complete_report.json"

NAIVE_MAX_TOKENS = 4000   # cap for control_ conditions

# Conversation prefixes that used adaptive gated retrieval (2k token budget)
COHORT_GATED_PREFIXES = [
    "355a5709", "3976f0b7", "4235e04a", "43084df0",
    "48cc67d6", "52535105", "59a652f7", "615e4db3"
]

# Flaw conversation identifier
FLAW_PREFIX = "bb558b5f"

# The six core evaluation conditions (always present)
SIX_CORE = [
    "control_baseline_generalist", "control_moe",
    "vector_rag_baseline_generalist", "vector_rag_moe",
    "full_ice_generalist", "full_ice_moe"
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_cohort_label(checkpoint_id: str) -> str:
    m = re.search(r"EC-([a-f0-9]+)-", checkpoint_id)
    prefix = m.group(1) if m else ""
    if prefix in COHORT_GATED_PREFIXES:
        return "adaptive_gated_retrieval_2k"
    return "forced_long_horizon_retrieval_5k"

def is_flaw(checkpoint_id: str) -> bool:
    return FLAW_PREFIX in checkpoint_id

def turn_index(checkpoint_id: str) -> int:
    match = re.search(r"TURN(\d+)", checkpoint_id)
    return int(match.group(1)) if match else 0

def conversation_prefix(checkpoint_id: str) -> str:
    m = re.search(r"EC-([a-f0-9]+)-", checkpoint_id)
    return m.group(1) if m else "unknown"

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_data():
    master = load_json(MASTER_RESULTS)["evaluation_run_results"]
    eval_raw = load_json(EVAL_RAW)
    eval_dict = {(item["checkpoint_id"], item["probe_id"]): item for item in eval_raw}
    return master, eval_dict

# ---------------------------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------------------------
def compute_metrics(master_entries, eval_dict):
    # -----------------------------------------------------------------------
    # 1. Data structures
    # -----------------------------------------------------------------------
    cohorts = ["adaptive_gated_retrieval_2k", "forced_long_horizon_retrieval_5k", "global_aggregate"]

    # Per‑cohort per‑condition accumulators
    stats = {c: defaultdict(lambda: {
        "scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0
    }) for c in cohorts}

    # Flaw‑only accumulators (same shape)
    flaw_stats = defaultdict(lambda: {
        "scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0
    })

    # Six‑core benchmark (all probes that have the six conditions)
    six_core_stats = defaultdict(lambda: {
        "scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0
    })

    # MoE vs Generalist (only core six conditions, separated by base)
    moe_vs_gen = {
        "control": {"moe": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0},
                    "generalist": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0}},
        "vector_rag": {"moe": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0},
                       "generalist": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0}},
        "full_ice": {"moe": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0},
                     "generalist": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0}},
        "global": {"moe": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0},
                   "generalist": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0}}
    }

    # Fragment noise accumulators per cohort
    frag_noise = {c: {"ice": [], "vector": []} for c in cohorts}
    flaw_frag_noise = {"ice": [], "vector": []}

    # Longitudinal: keyed by (conversation_prefix, question), store list of (turn, score)
    # Also per‑cohort aggregated: cohort -> list of (turn, score) for full_ice_generalist
    long_curves = defaultdict(list)
    cohort_long_curves = {c: [] for c in cohorts}

    # Ablation pairs for Flaw conversation (cond_a is full ICE, cond_b is ablation variant)
    ablation_pairs = {
        "hyde_ablation": ("full_ice_generalist", "full_ice_no_hyde_generalist"),
        "procedural_ablation": ("full_ice_generalist", "full_ice_no_procedural_generalist"),
        "sliding_window_ablation": ("full_ice_generalist", "full_ice_no_sliding_window_generalist"),
        "scope_auto_vs_project": ("full_ice_scope_auto_generalist", "full_ice_scope_project_generalist"),
        "scope_auto_vs_none": ("full_ice_scope_auto_generalist", "full_ice_scope_none_generalist"),
    }
    ablation_data = {k: {"a": [], "b": []} for k in ablation_pairs}

    # Gating failure probes: classifier said Zero_Shot but full_ice_generalist score < 3
    gating_failures = []

    # -----------------------------------------------------------------------
    # 2. Iterate over master entries
    # -----------------------------------------------------------------------
    for entry in master_entries:
        meta = entry["metadata"]
        cid, pid = meta["checkpoint_id"], meta["probe_id"]
        cohort = get_cohort_label(cid)
        turn = turn_index(cid)
        conv_prefix = conversation_prefix(cid)
        question = meta["raw_user_probe"]
        eval_item = eval_dict.get((cid, pid), {})
        is_flaw_probe = is_flaw(cid)

        groups_for_stats = [cohort, "global_aggregate"]
        if is_flaw_probe:
            groups_for_stats.append("flaw_lens")

        # Classification metadata (for gating failure)
        classification_data = None
        first_cond = next(iter(entry["execution_permutations"].values()))
        classification_data = first_cond.get("classification", {})

        # Process each condition in this entry
        for cond, cond_exec in entry["execution_permutations"].items():
            t_raw = cond_exec.get("tokens_injected", 0)
            t_adj = min(t_raw, NAIVE_MAX_TOKENS) if cond.startswith("control_") else t_raw

            # Safe access to absolute_scores (could be None)
            abs_scores_all = eval_item.get("absolute_scores")
            abs_data = abs_scores_all.get(cond, {}) if isinstance(abs_scores_all, dict) else {}
            score_val = abs_data.get("score") if isinstance(abs_data, dict) else None

            # Safe access to hallucination (could be None)
            hall_all = eval_item.get("hallucination")
            hall_dict = hall_all.get(cond, {}) if isinstance(hall_all, dict) else {}
            hal = 1 if isinstance(hall_dict, dict) and hall_dict.get("hallucination_count", 0) > 0 else 0

            # Feed cohort accumulators
            for group in groups_for_stats:
                d = flaw_stats[cond] if group == "flaw_lens" else stats[group][cond]
                if score_val is not None:
                    d["scores"].append(score_val)
                d["tokens"].append(t_adj)
                d["hals"].append(hal)

            # Six‑core (only if condition is one of the six)
            if cond in SIX_CORE:
                d6 = six_core_stats[cond]
                if score_val is not None:
                    d6["scores"].append(score_val)
                d6["tokens"].append(t_adj)
                d6["hals"].append(hal)

            # MoE vs Generalist (only core six)
            if cond in SIX_CORE:
                base = None
                if cond.startswith("control_"):
                    base = "control"
                elif cond.startswith("vector_rag_"):
                    base = "vector_rag"
                elif cond.startswith("full_ice_"):
                    base = "full_ice"
                if base:
                    variant = "moe" if "_moe" in cond else "generalist"
                    for target in [base, "global"]:
                        d_moe = moe_vs_gen[target][variant]
                        if score_val is not None:
                            d_moe["scores"].append(score_val)
                        d_moe["tokens"].append(t_adj)
                        d_moe["hals"].append(hal)

        # Ablation data (only for Flaw probes)
        if is_flaw_probe:
            for ab_name, (cond_a, cond_b) in ablation_pairs.items():
                if cond_a in entry["execution_permutations"] and cond_b in entry["execution_permutations"]:
                    score_a = None
                    score_b = None
                    abs_scores_all = eval_item.get("absolute_scores")
                    if isinstance(abs_scores_all, dict):
                        a_dict = abs_scores_all.get(cond_a)
                        if isinstance(a_dict, dict):
                            score_a = a_dict.get("score")
                        b_dict = abs_scores_all.get(cond_b)
                        if isinstance(b_dict, dict):
                            score_b = b_dict.get("score")
                    if score_a is not None and score_b is not None:
                        ablation_data[ab_name]["a"].append(score_a)
                        ablation_data[ab_name]["b"].append(score_b)

        # Longitudinal: full_ice_generalist score
        ice_gen_score = None
        if "full_ice_generalist" in entry["execution_permutations"]:
            abs_scores_all = eval_item.get("absolute_scores")
            if isinstance(abs_scores_all, dict):
                score_data = abs_scores_all.get("full_ice_generalist")
                if isinstance(score_data, dict) and "score" in score_data:
                    ice_gen_score = score_data["score"]
        if ice_gen_score is not None:
            long_curves[(conv_prefix, question)].append((turn, ice_gen_score))
            cohort_long_curves[cohort].append((turn, ice_gen_score))

        # Tournament rankings (safe access)
        tournament = eval_item.get("tournament")
        rankings = tournament.get("rankings", []) if isinstance(tournament, dict) else []
        if rankings:
            for group in groups_for_stats:
                d_group = flaw_stats if group == "flaw_lens" else stats[group]
                for idx, cond_name in enumerate(rankings):
                    d_group[cond_name]["apps"] += 1
                    if idx == 0:
                        d_group[cond_name]["wins"] += 1

            for idx, cond_name in enumerate(rankings):
                if cond_name in SIX_CORE:
                    six_core_stats[cond_name]["apps"] += 1
                    if idx == 0:
                        six_core_stats[cond_name]["wins"] += 1

            for idx, cond_name in enumerate(rankings):
                if cond_name in SIX_CORE:
                    base = None
                    if cond_name.startswith("control_"):
                        base = "control"
                    elif cond_name.startswith("vector_rag_"):
                        base = "vector_rag"
                    elif cond_name.startswith("full_ice_"):
                        base = "full_ice"
                    if base:
                        variant = "moe" if "_moe" in cond_name else "generalist"
                        for target in [base, "global"]:
                            d_moe = moe_vs_gen[target][variant]
                            d_moe["apps"] += 1
                            if idx == 0:
                                d_moe["wins"] += 1

        # Gating failure
        if classification_data and classification_data.get("context_reliance") == "Zero_Shot":
            if ice_gen_score is not None and ice_gen_score < 3:
                gating_failures.append({
                    "checkpoint_id": cid,
                    "probe_id": pid,
                    "question": question,
                    "ice_score": ice_gen_score
                })

        # Fragment noise (safe access)
        frag_all = eval_item.get("fragment_analysis")
        frag_ice = frag_all.get("full_ice") if isinstance(frag_all, dict) else None
        frag_vec = frag_all.get("vector_rag") if isinstance(frag_all, dict) else None
        for group in [cohort, "global_aggregate"]:
            if isinstance(frag_ice, dict) and frag_ice.get("noise_score") is not None:
                frag_noise[group]["ice"].append(frag_ice["noise_score"])
            if isinstance(frag_vec, dict) and frag_vec.get("noise_score") is not None:
                frag_noise[group]["vector"].append(frag_vec["noise_score"])
        if is_flaw_probe:
            if isinstance(frag_ice, dict) and frag_ice.get("noise_score") is not None:
                flaw_frag_noise["ice"].append(frag_ice["noise_score"])
            if isinstance(frag_vec, dict) and frag_vec.get("noise_score") is not None:
                flaw_frag_noise["vector"].append(frag_vec["noise_score"])

    # -----------------------------------------------------------------------
    # 3. Build per‑condition summary (with standard deviations)
    # -----------------------------------------------------------------------
    def build_summary(cond_dict, include_deltas=True):
        summary = {}
        for cond, d in cond_dict.items():
            avg_score = statistics.mean(d["scores"]) if d["scores"] else 0.0
            std_score = statistics.stdev(d["scores"]) if len(d["scores"]) > 1 else 0.0
            avg_tokens = statistics.mean(d["tokens"]) if d["tokens"] else 0.0
            std_tokens = statistics.stdev(d["tokens"]) if len(d["tokens"]) > 1 else 0.0
            win_rate = (d["wins"] / d["apps"]) if d["apps"] > 0 else 0.0
            hal_rate = statistics.mean(d["hals"]) if d["hals"] else 0.0
            tur = (avg_score / (avg_tokens / 1000)) if avg_tokens > 0 else 0.0
            summary[cond] = {
                "avg_score": round(avg_score, 2),
                "std_score": round(std_score, 2),
                "avg_tokens_injected": int(avg_tokens),
                "std_tokens_injected": int(std_tokens),
                "win_rate_pct": round(win_rate * 100, 1),
                "hallucination_pct": round(hal_rate * 100, 1),
                "tur": round(tur, 2),
                "n_scores": len(d["scores"]),
                "n_tokens": len(d["tokens"])
            }
        if include_deltas:
            ice = summary.get("full_ice_moe") or summary.get("full_ice_generalist")
            vec = summary.get("vector_rag_baseline_generalist") or summary.get("vector_rag_moe")
            naive = summary.get("control_baseline_generalist") or summary.get("control_moe")
            if ice and vec:
                summary["comparative_deltas"] = {
                    "token_savings_vs_vector_pct": round(
                        (1 - (ice["avg_tokens_injected"] / vec["avg_tokens_injected"])) * 100, 1
                    ) if vec["avg_tokens_injected"] > 0 else 0,
                    "token_savings_vs_naive_pct": round(
                        (1 - (ice["avg_tokens_injected"] / naive["avg_tokens_injected"])) * 100, 1
                    ) if naive and naive["avg_tokens_injected"] > 0 else 0,
                    "quality_gain_vs_vector_pts": round(ice["avg_score"] - vec["avg_score"], 2),
                    "hallucination_reduction_vs_vector_pct": round(
                        vec["hallucination_pct"] - ice["hallucination_pct"], 1
                    )
                }
        return summary

    # -----------------------------------------------------------------------
    # 4. Longitudinal curves (per question) and per‑cohort aggregated curve
    # -----------------------------------------------------------------------
    longitudinal = {}
    for (conv, question), points in long_curves.items():
        if len(points) < 2:
            continue
        points.sort(key=lambda x: x[0])
        label = f"{conv}: {question[:80]}"
        longitudinal[label] = {
            "turns": [p[0] for p in points],
            "scores": [p[1] for p in points]
        }

    # Per‑cohort aggregated longitudinal: group turns into bins (e.g., 50-turn bins) and average
    aggregated_longitudinal = {}
    for cohort, points in cohort_long_curves.items():
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        bins = defaultdict(list)
        for t, s in points:
            bin_low = (t // 50) * 50
            bins[bin_low].append(s)
        bin_centers = sorted(bins.keys())
        avg_scores = [statistics.mean(bins[b]) for b in bin_centers]
        aggregated_longitudinal[cohort] = {
            "turn_bins": bin_centers,
            "mean_scores": [round(s, 2) for s in avg_scores]
        }

    # -----------------------------------------------------------------------
    # 5. MoE vs Generalist summary (with std)
    # -----------------------------------------------------------------------
    def build_moe_summary(moe_data):
        result = {}
        for base, data in moe_data.items():
            label = "global_aggregate" if base == "global" else base
            moe = data["moe"]
            gen = data["generalist"]
            moe_score = statistics.mean(moe["scores"]) if moe["scores"] else 0.0
            moe_score_std = statistics.stdev(moe["scores"]) if len(moe["scores"]) > 1 else 0.0
            gen_score = statistics.mean(gen["scores"]) if gen["scores"] else 0.0
            gen_score_std = statistics.stdev(gen["scores"]) if len(gen["scores"]) > 1 else 0.0
            moe_tokens = statistics.mean(moe["tokens"]) if moe["tokens"] else 0.0
            gen_tokens = statistics.mean(gen["tokens"]) if gen["tokens"] else 0.0
            moe_hal = statistics.mean(moe["hals"]) if moe["hals"] else 0.0
            gen_hal = statistics.mean(gen["hals"]) if gen["hals"] else 0.0
            moe_win = (moe["wins"] / moe["apps"]) if moe["apps"] > 0 else 0.0
            gen_win = (gen["wins"] / gen["apps"]) if gen["apps"] > 0 else 0.0
            result[label] = {
                "moe": {
                    "avg_score": round(moe_score, 2),
                    "std_score": round(moe_score_std, 2),
                    "avg_tokens": int(moe_tokens),
                    "win_rate_pct": round(moe_win * 100, 1),
                    "hallucination_pct": round(moe_hal * 100, 1)
                },
                "generalist": {
                    "avg_score": round(gen_score, 2),
                    "std_score": round(gen_score_std, 2),
                    "avg_tokens": int(gen_tokens),
                    "win_rate_pct": round(gen_win * 100, 1),
                    "hallucination_pct": round(gen_hal * 100, 1)
                },
                "delta_moe_vs_generalist": {
                    "score_gain": round(moe_score - gen_score, 2),
                    "tokens_saved": int(gen_tokens - moe_tokens) if gen_tokens > 0 else 0,
                    "hallucination_reduction_pct": round((gen_hal - moe_hal) * 100, 1)
                }
            }
        return result

    # -----------------------------------------------------------------------
    # 6. Ablation analysis summary (from Flaw)
    # -----------------------------------------------------------------------
    ablation_summary = {}
    for ab_name, (cond_a, cond_b) in ablation_pairs.items():
        scores_a = ablation_data[ab_name]["a"]
        scores_b = ablation_data[ab_name]["b"]
        if not scores_a or not scores_b:
            continue
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0
        ablation_summary[ab_name] = {
            "condition_a": cond_a,
            "condition_b": cond_b,
            "mean_score_a": round(mean_a, 2),
            "std_a": round(std_a, 2),
            "mean_score_b": round(mean_b, 2),
            "std_b": round(std_b, 2),
            "delta_a_minus_b": round(mean_a - mean_b, 2),
            "n_pairs": len(scores_a)
        }

    # -----------------------------------------------------------------------
    # 7. Fragment noise summary
    # -----------------------------------------------------------------------
    def build_frag_summary(frag_dict):
        return {
            "full_ice_mean_noise": round(statistics.mean(frag_dict["ice"]), 2) if frag_dict["ice"] else None,
            "full_ice_std_noise": round(statistics.stdev(frag_dict["ice"]), 2) if len(frag_dict["ice"]) > 1 else 0.0,
            "full_ice_n": len(frag_dict["ice"]),
            "vector_rag_mean_noise": round(statistics.mean(frag_dict["vector"]), 2) if frag_dict["vector"] else None,
            "vector_rag_std_noise": round(statistics.stdev(frag_dict["vector"]), 2) if len(frag_dict["vector"]) > 1 else 0.0,
            "vector_rag_n": len(frag_dict["vector"])
        }

    # -----------------------------------------------------------------------
    # 8. Assemble final report
    # -----------------------------------------------------------------------
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {},
        "flaw_lens": {},
        "six_core_benchmark": {},
        "moe_vs_generalist": {},
        "longitudinal_curves": longitudinal,
        "aggregated_longitudinal_curves": aggregated_longitudinal,
        "ablation_analysis": ablation_summary,
        "gating_failures": gating_failures,
        "global_haystack": {}
    }

    for cohort in cohorts:
        report["cohorts"][cohort] = build_summary(stats[cohort], include_deltas=True)
        report["cohorts"][cohort]["fragment_noise_summary"] = build_frag_summary(frag_noise[cohort])

    report["flaw_lens"] = build_summary(flaw_stats, include_deltas=True)
    report["flaw_lens"]["fragment_noise_summary"] = build_frag_summary(flaw_frag_noise)

    report["six_core_benchmark"] = build_summary(six_core_stats, include_deltas=True)

    report["moe_vs_generalist"] = build_moe_summary(moe_vs_gen)

    report["global_haystack"] = build_summary(stats["global_aggregate"], include_deltas=False)

    # -----------------------------------------------------------------------
    # 9. RESEARCH INTEGRITY METADATA (The "Why it happened" block)
    # -----------------------------------------------------------------------
    integrity_metadata = {
        "experiment_id": "ICE-LSREP-PHASE-1-FINAL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_infrastructure": {
            "gpu": "NVIDIA RTX 5090 (24GB GDDR7)",
            "cpu": "Intel Ultra 9 275HX (Arrow Lake)",
            "ram": "64GB DDR5",
            "os": "CachyOS (Linux Kernel: Optimized BORE/Sched-ext)",
            "storage": "2TB NVMe Gen 5 (High-IOPS logging)"
        },
        "judge_engine_specification": {
            "framework": "SGLang (RadixAttention Runtime)",
            "model_id": "mattbucci/gemma-4-12B-AWQ",
            "quantization": "4-bit AWQ (Weight-only)",
            "runtime_parameters": {
                "context_length_limit": 150000,
                "mem_fraction_static": 0.85,
                "max_running_requests": 4,
                "chunked_prefill_size": 8192,
                "attention_backend": "Triton",
                "kv_cache_dtype": "fp8_e4m3 (Hardware-accelerated)",
                "max_total_tokens": 160000
            }
        },
        "context_distillation_protocol": {
            "method": "Recursive Librarian Search (Surgical Oracle v2.1)",
            "retrieval_logic": "Semantic Vector Top-K (all-MiniLM-L6-v2)",
            "evidence_haystack_threshold": {
                "max_context_tokens": 30000,
                "dynamic_top_k": {
                    "standard_conv": 20,
                    "large_horizon_conv": 40,
                    "threshold_tokens": 200000
                }
            },
            "extraction_rules": [
                "Strict Neutrality",
                "Verbatim Anchor Preservation",
                "Temporal Dominance (LTM Overwrite)",
                "Third-Person Attribution"
            ]
        },
        "tested_inference_architecture": {
            "baseline_generalist": "gemma4:26b-a4b-it-q4_K_M (Ollama/Llama.cpp)",
            "moe_expert_selection_logic": "Intent-Tags Overlap Matrix",
            "expert_stack": {
                "Software_&_STEM": "qwen3-coder:30b-a3b-q4_K_M",
                "Creative_&_Social_&_Life": "gemma4:12b",
                "Business_&_Admin_&_Strategy": "qwen3.6:27b",
                "Meta_AI_&_World_&_Formatting": "qwen2.5:7b",
                "Null_Noise": "tinyllama:latest"
            }
        },
        "experimental_constraints_disclaimer": {
            "metabolism_maturity": "Infant State (3 Decay Cycles)",
            "knowledge_graph_mode": "Regex-Heuristic (Pre-MLP/NER)",
            "cluster_granularity": "Max 10 turns per leaf node",
            "token_injection_budgets": {
                "full_ice": 5000,
                "control_naive_sliding_window": 4000
            },
            "total_decay_cycles": 3   # ← added to match print statement
        }
    }

    report["experiment_integrity_metadata"] = integrity_metadata

    return report

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    master, eval_dict = load_data()
    print(f"Master entries: {len(master)}")
    print(f"Eval entries: {len(eval_dict)}")

    print("Computing complete metrics...")
    report = compute_metrics(master, eval_dict)

    save_json(report, OUTPUT_FILE)
    print(f"Complete report saved to {OUTPUT_FILE}")
    print("\n" + "█"*60)
    print(" 🧊 INFINITE CONTEXT ENGINE — FINAL METRICS SUMMARY")
    print("█" * 60)
    print(f" PROBES EVALUATED   : {len(master)}")
    print(f" HARDWARE           : RTX 5090 / Ultra 9 / CachyOS")
    print(f" JUDGE BACKEND      : SGLang (Gemma 12B AWQ)")
    print(f" KV-CACHE DTYPE     : FP8_E4M3 (High-Efficiency)")
    print(f" CONTEXT WINDOW     : 150,000 Tokens")
    print(f" CONCURRENCY        : 4 Parallel Requests")
    print("-" * 60)
    print(f" LTM OVERRIDE       : Enabled (>10 turns)")
    print(f" CODEX STATUS       : Regex-Heuristic (Infant Stage)")
    print(f" DECAY MATURITY     : {report['experiment_integrity_metadata']['experimental_constraints_disclaimer']['total_decay_cycles']} cycles")    
    print("-" * 60)
    print(f" OUTPUT FILE        : {OUTPUT_FILE}")
    print("█" * 60 + "\n")