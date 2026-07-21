#!/usr/bin/env python3
"""
ICE Flaw Subtraction Metrics (with Longitudinal Analysis)
=========================================================
Computes per‑condition metrics and how each condition's retrieval quality
evolves across the 11 checkpoints (turn 51 → 1119).

Reads:  experiments/flaw_ablation/subtraction/master_results.json
        experiments/flaw_ablation/subtraction/evaluation_raw.json
Writes: experiments/flaw_ablation/subtraction/metrics_report.json
"""

import json
import os
import statistics
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MASTER_FILE = "experiments/flaw_ablation/subtraction/intermediates/master_results.json"
EVAL_FILE   = "experiments/flaw_ablation/subtraction/intermediates/evaluation_raw.json"
OUTPUT_FILE = "experiments/flaw_ablation/subtraction/results/metrics_report.json"

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

FEATURE_CATEGORIES = {
    "Retrieval Legs": ["no_vector", "no_bm25"],
    "Fusion & Query": ["no_rrf", "hyde_on"],
    "Knowledge Graph": ["no_codex", "no_mera", "no_fuzzy_match"],
    "Memory Systems": ["no_procedural", "no_batch_summary"],
    "Scoping & Diversity": ["no_cluster_restrict", "no_session_diversify"],
    "Prompt Assembly": ["static_budget", "no_sliding_window", "no_keyword_boost", "no_recency_boost"],
}

# Turn-index bins for longitudinal aggregation
TURN_BINS = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1200)]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    master = load_json(MASTER_FILE)["evaluation_run_results"]
    eval_raw = load_json(EVAL_FILE)

    eval_by_key = {}
    for item in eval_raw:
        key = (item["checkpoint_id"], item["probe_id"])
        eval_by_key[key] = item

    # ── Per‑condition accumulators ────────────────────────────────────
    accum = defaultdict(lambda: {
        "scores": [],
        "tokens": [],
        "hallucinations": [],
        "fragment_noise_ice": [],
        "fragment_noise_vector": [],
        # For longitudinal: list of (turn_index, score)
        "turn_score_pairs": [],
        # For per‑probe tracking: probe_id -> list of (turn_index, score)
        "probe_trajectories": defaultdict(list),
    })

    for entry in master:
        cid = entry["checkpoint_id"]
        pid = entry["probe_id"]
        turn_index = entry.get("turn_index", 0)
        eval_entry = eval_by_key.get((cid, pid), {})

        for cond in ALL_CONDITIONS:
            cond_data = entry.get("conditions", {}).get(cond)
            if not cond_data:
                continue

            tokens = cond_data.get("tokens_injected", 0)
            accum[cond]["tokens"].append(tokens)

            abs_scores = eval_entry.get("absolute_scores", {})
            score_data = abs_scores.get(cond)
            if isinstance(score_data, dict) and score_data.get("score") is not None:
                score = score_data["score"]
                accum[cond]["scores"].append(score)
                accum[cond]["turn_score_pairs"].append((turn_index, score))
                accum[cond]["probe_trajectories"][pid].append((turn_index, score))

            hall_data = eval_entry.get("hallucination", {}).get(cond)
            if isinstance(hall_data, dict):
                has_hall = 1 if hall_data.get("hallucination_count", 0) > 0 else 0
                accum[cond]["hallucinations"].append(has_hall)

        # Fragment noise — only baseline_all_on and vector_baseline
        frag = eval_entry.get("fragment_analysis", {})
        ice_frag = frag.get("full_ice")
        if isinstance(ice_frag, dict) and ice_frag.get("noise_score") is not None:
            accum["baseline_all_on"]["fragment_noise_ice"].append(ice_frag["noise_score"])
        vec_frag = frag.get("vector_rag")
        if isinstance(vec_frag, dict) and vec_frag.get("noise_score") is not None:
            accum["vector_baseline"]["fragment_noise_vector"].append(vec_frag["noise_score"])

    # ── Baseline values ───────────────────────────────────────────────
    baseline = accum.get("baseline_all_on", {})
    baseline_score = statistics.mean(baseline.get("scores", [0])) if baseline.get("scores") else 0.0
    baseline_tokens = statistics.mean(baseline.get("tokens", [0])) if baseline.get("tokens") else 0
    baseline_hall = statistics.mean(baseline.get("hallucinations", [0])) * 100 if baseline.get("hallucinations") else 0.0
    baseline_tur = baseline_score / (baseline_tokens / 1000) if baseline_tokens > 0 else 0.0

    # ── Per‑condition summary ─────────────────────────────────────────
    report = {
        "experiment": "flaw_subtraction",
        "n_checkpoints": len(set(e["turn_index"] for e in master)),
        "n_probes_total": len(master),
        "baseline": {
            "condition": "baseline_all_on",
            "mean_score": round(baseline_score, 2),
            "mean_tokens": int(baseline_tokens),
            "hallucination_pct": round(baseline_hall, 1),
            "tur": round(baseline_tur, 2),
            "fragment_noise_mean": round(statistics.mean(baseline.get("fragment_noise_ice", [0])), 2)
                if baseline.get("fragment_noise_ice") else None,
            "n_probes": len(baseline.get("scores", [])),
        },
        "conditions": {},
        "longitudinal": {},
    }

    for cond in ALL_CONDITIONS:
        if cond == "baseline_all_on":
            continue
        data = accum.get(cond, {})
        scores = data.get("scores", [])
        tokens = data.get("tokens", [])
        halls = data.get("hallucinations", [])

        mean_score = statistics.mean(scores) if scores else 0.0
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        mean_tokens = statistics.mean(tokens) if tokens else 0
        hall_pct = statistics.mean(halls) * 100 if halls else 0.0
        tur = mean_score / (mean_tokens / 1000) if mean_tokens > 0 else 0.0

        delta_score = mean_score - baseline_score
        delta_tokens = mean_tokens - baseline_tokens
        delta_hall = hall_pct - baseline_hall

        frag_noise = None
        if cond == "vector_baseline":
            vec_noises = data.get("fragment_noise_vector", [])
            frag_noise = round(statistics.mean(vec_noises), 2) if vec_noises else None

        report["conditions"][cond] = {
            "mean_score": round(mean_score, 2),
            "std_score": round(std_score, 2),
            "mean_tokens": int(mean_tokens),
            "hallucination_pct": round(hall_pct, 1),
            "tur": round(tur, 2),
            "delta_score": round(delta_score, 2),
            "delta_tokens": int(delta_tokens),
            "delta_hall_pct": round(delta_hall, 1),
            "fragment_noise_mean": frag_noise,
            "n_probes": len(scores),
        }

    # ── Longitudinal: per‑condition score by turn‑index bin ────────────
    for cond in ALL_CONDITIONS:
        pairs = accum[cond]["turn_score_pairs"]
        if not pairs:
            continue

        bin_data = {}
        for low, high in TURN_BINS:
            bin_scores = [s for t, s in pairs if low <= t < high]
            if bin_scores:
                bin_data[f"{low}-{high}"] = {
                    "mean_score": round(statistics.mean(bin_scores), 2),
                    "n": len(bin_scores),
                }

        # Also compute per‑probe trajectories: for probes that appear at ≥3 checkpoints,
        # track first → last score delta
        probe_deltas = []
        for pid, trajectory in accum[cond]["probe_trajectories"].items():
            if len(trajectory) >= 3:
                trajectory.sort(key=lambda x: x[0])
                first_score = trajectory[0][1]
                last_score = trajectory[-1][1]
                probe_deltas.append({
                    "probe_id": pid,
                    "first_turn": trajectory[0][0],
                    "first_score": first_score,
                    "last_turn": trajectory[-1][0],
                    "last_score": last_score,
                    "delta": last_score - first_score,
                })

        avg_delta = statistics.mean([d["delta"] for d in probe_deltas]) if probe_deltas else 0.0

        report["longitudinal"][cond] = {
            "bins": bin_data,
            "probe_trajectories": probe_deltas[:10],   # top 10 for readability
            "n_trajectories": len(probe_deltas),
            "mean_probe_delta": round(avg_delta, 2),
        }

    # ── Category summaries ────────────────────────────────────────────
    categories = {}
    for cat_name, cat_conditions in FEATURE_CATEGORIES.items():
        cat_scores = []
        cat_deltas = []
        for cond in cat_conditions:
            if cond in report["conditions"]:
                cat_scores.append(report["conditions"][cond]["mean_score"])
                cat_deltas.append(report["conditions"][cond]["delta_score"])
        categories[cat_name] = {
            "conditions": cat_conditions,
            "avg_score": round(statistics.mean(cat_scores), 2) if cat_scores else None,
            "avg_delta": round(statistics.mean(cat_deltas), 2) if cat_deltas else None,
        }

    report["categories"] = categories

    save_json(report, OUTPUT_FILE)
    print(f"Metrics report saved to {OUTPUT_FILE}")

    # ── Console summary ───────────────────────────────────────────────
    print(f"\n{'Condition':<30} {'Score':>6} {'ΔScore':>7} {'Tokens':>7} {'ΔTok':>7} {'Hall%':>6} {'LongΔ':>7}")
    print("-" * 85)
    base = report["baseline"]
    base_long = report["longitudinal"].get("baseline_all_on", {}).get("mean_probe_delta", 0)
    print(f"{'BASELINE (all on)':<30} {base['mean_score']:>6.2f} {'--':>7} {base['mean_tokens']:>7} {'--':>7} {base['hallucination_pct']:>5.1f}% {base_long:>+7.2f}")
    for cond in ALL_CONDITIONS:
        if cond == "baseline_all_on":
            continue
        c = report["conditions"].get(cond, {})
        long_delta = report["longitudinal"].get(cond, {}).get("mean_probe_delta", 0)
        print(f"{cond:<30} {c.get('mean_score', 0):>6.2f} {c.get('delta_score', 0):>+7.2f} {c.get('mean_tokens', 0):>7} {c.get('delta_tokens', 0):>+7} {c.get('hallucination_pct', 0):>5.1f}% {long_delta:>+7.2f}")

    # ── Longitudinal highlight ────────────────────────────────────────
    print(f"\n{'─'*85}")
    print("LONGITUDINAL TREND (mean score per turn‑index bin)")
    print(f"{'─'*85}")
    for cond in ["baseline_all_on", "vector_baseline"] + [c for c in ALL_CONDITIONS if c not in ("baseline_all_on", "vector_baseline")]:
        bins = report["longitudinal"].get(cond, {}).get("bins", {})
        if not bins:
            continue
        bin_str = "  ".join(f"{bins[b]['mean_score']:.2f}" for b in sorted(bins.keys()) if b in bins)
        print(f"{cond:<30}  {bin_str}")


if __name__ == "__main__":
    main()