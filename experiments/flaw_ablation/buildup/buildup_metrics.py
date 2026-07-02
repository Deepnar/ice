#!/usr/bin/env python3
"""
ICE Flaw Buildup Metrics (Single‑Pass with Origin‑Split Longitudinal)
=====================================================================
Now includes step‑by‑step deltas, SPF, and highlights the largest changes.

Output: experiments/flaw_ablation/buildup/metrics_report.json
"""

import json, os, statistics
from collections import defaultdict

MASTER_FILE = "experiments/flaw_ablation/buildup/master_results.json"
EVAL_FILE   = "experiments/flaw_ablation/buildup/evaluation_raw.json"
OUTPUT_FILE = "experiments/flaw_ablation/buildup/metrics_report.json"

ALL_CONDITIONS = [
    "bare_vector", "add_bm25", "add_rrf", "add_hyde",
    "add_cluster_restrict", "add_session_diversify", "add_codex", "add_mera",
    "add_procedural", "add_batch_summary", "add_dynamic_budget",
    "add_sliding_window", "add_keyword_boost", "full_ice", "vector_baseline"
]

ORIGIN_BINS = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1200)]

def load_json(path):
    with open(path, "r") as f: return json.load(f)
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2)

def main():
    master = load_json(MASTER_FILE)["evaluation_run_results"]
    eval_raw = load_json(EVAL_FILE)
    eval_by_key = {item["probe_id"]: item for item in eval_raw}

    accum = defaultdict(lambda: {"scores": [], "tokens": [], "frags": [],
                                  "origin_pairs": [], "hallucinations": []})

    for entry in master:
        pid = entry["probe_id"]
        origin_split = entry.get("origin_split", 0)
        eval_entry = eval_by_key.get(pid, {})
        for cond in ALL_CONDITIONS:
            cond_data = entry.get("conditions", {}).get(cond)
            if not cond_data: continue
            accum[cond]["tokens"].append(cond_data.get("tokens_injected", 0))
            accum[cond]["frags"].append(len(cond_data.get("fragment_ids", [])))
            abs_scores = eval_entry.get("absolute_scores", {})
            score_data = abs_scores.get(cond)
            if isinstance(score_data, dict) and score_data.get("score") is not None:
                s = score_data["score"]
                accum[cond]["scores"].append(s)
                accum[cond]["origin_pairs"].append((origin_split, s))
            hall_data = eval_entry.get("hallucination", {}).get(cond)
            if isinstance(hall_data, dict):
                accum[cond]["hallucinations"].append(1 if hall_data.get("hallucination_count", 0) > 0 else 0)

    # ── Per‑condition summary ───────────────────────────────────────
    def summary(cond):
        data = accum.get(cond, {})
        scores = data["scores"]
        tokens = data["tokens"]
        frags  = data["frags"]
        if not scores: return None
        ms = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        mt = statistics.mean(tokens)
        mf = statistics.mean(frags)
        tur = ms / (mt / 1000) if mt > 0 else 0.0
        spf = ms / mf if mf > 0 else 0.0
        hp  = statistics.mean(data["hallucinations"]) * 100 if data["hallucinations"] else 0.0
        return {
            "mean_score": round(ms, 2), "std_score": round(std, 2),
            "mean_tokens": int(mt), "mean_frags": round(mf, 1),
            "spf": round(spf, 2), "tur": round(tur, 2),
            "hallucination_pct": round(hp, 1), "n_probes": len(scores),
        }

    cond_summaries = {}
    for c in ALL_CONDITIONS:
        s = summary(c)
        if s:
            cond_summaries[c] = s

    # ── Step‑by‑step deltas ─────────────────────────────────────────
    steps = [c for c in ALL_CONDITIONS if c in cond_summaries and c != "vector_baseline"]
    step_deltas = {}
    for i in range(1, len(steps)):
        prev = steps[i-1]
        curr = steps[i]
        step_deltas[curr] = round(cond_summaries[curr]["mean_score"] - cond_summaries[prev]["mean_score"], 2)

    # Cumulative from bare
    bare_score = cond_summaries["bare_vector"]["mean_score"]
    for c in cond_summaries:
        cond_summaries[c]["cumulative_from_bare"] = round(cond_summaries[c]["mean_score"] - bare_score, 2)

    # Delta from full_ice
    full_score = cond_summaries["full_ice"]["mean_score"]
    for c in cond_summaries:
        cond_summaries[c]["delta_from_full_ice"] = round(cond_summaries[c]["mean_score"] - full_score, 2)

    # ── Longitudinal ────────────────────────────────────────────────
    long = {}
    for c in ALL_CONDITIONS:
        pairs = accum[c]["origin_pairs"]
        if not pairs: continue
        bins = {}
        for low, high in ORIGIN_BINS:
            bs = [s for o, s in pairs if low <= o < high]
            if bs: bins[f"orig_{low}_{high}"] = round(statistics.mean(bs), 2)
        early = [s for o, s in pairs if o < 400]
        late  = [s for o, s in pairs if o >= 800]
        em = statistics.mean(early) if early else 0.0
        lm = statistics.mean(late) if late else 0.0
        long[c] = {"bins": bins, "early_mean": round(em, 2),
                   "late_mean": round(lm, 2), "recency_delta": round(lm - em, 2)}

    # ── Find largest changes ────────────────────────────────────────
    positive_steps = sorted(step_deltas.items(), key=lambda x: x[1], reverse=True)
    negative_steps = sorted(step_deltas.items(), key=lambda x: x[1])
    largest_positive = positive_steps[0] if positive_steps else ("?", 0)
    largest_negative = negative_steps[0] if negative_steps else ("?", 0)

    # ── Report ──────────────────────────────────────────────────────
    report = {
        "experiment": "flaw_buildup_single_pass",
        "n_probes_total": len(master),
        "conditions": cond_summaries,
        "step_deltas": step_deltas,
        "longitudinal": long,
        "highlights": {
            "largest_positive_step": {"condition": largest_positive[0], "delta": largest_positive[1]},
            "largest_negative_step": {"condition": largest_negative[0], "delta": largest_negative[1]},
        }
    }
    save_json(report, OUTPUT_FILE)

    # ── Console output ──────────────────────────────────────────────
    print(f"\n{'Condition':<30} {'Score':>6} {'StepΔ':>7} {'CumΔ':>7} {'SPF':>6} {'Tokens':>7} {'RecΔ':>7}")
    print("-" * 90)
    for i, c in enumerate(steps):
        s = cond_summaries[c]
        step = f"{step_deltas.get(c, 0):+.2f}" if i > 0 else "  ·"
        rec = long.get(c, {}).get("recency_delta", 0)
        print(f"{c:<30} {s['mean_score']:>6.2f} {step:>7} {s['cumulative_from_bare']:>+7.2f} "
              f"{s['spf']:>6.2f} {s['mean_tokens']:>7} {rec:>+7.2f}")

    print(f"\nLargest positive step: {largest_positive[0]} (+{largest_positive[1]})")
    print(f"Largest negative step: {largest_negative[0]} ({largest_negative[1]})")

    # Longitudinal
    print(f"\n{'─'*70}\nRECENCY EFFECT — Mean score by origin‑split bin\n{'─'*70}")
    bins_labels = [f"orig_{l}_{h}" for l, h in ORIGIN_BINS]
    for c in steps:
        bins = long.get(c, {}).get("bins", {})
        row = "  ".join(f"{bins.get(b, '—'):>5}" for b in bins_labels)
        print(f"{c:<30} {row}")

if __name__ == "__main__":
    main()