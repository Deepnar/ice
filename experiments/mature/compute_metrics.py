#!/usr/bin/env python3
"""
ICE-Mature Metrics Aggregator — Experiment 2 (Complete)
==========================================================
...

Safe version with proper nested defaultdicts, missing‑condition handling,
memory maturity, leg contributions, fragment‑score correlation, and manual override.
"""

import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MATURE_DIR = "experiments/mature"
RESULTS_DIR = os.path.join(MATURE_DIR, "results")
MASTER_RESULTS = os.path.join(RESULTS_DIR, "master_results.json")
EVAL_RAW       = os.path.join(RESULTS_DIR, "evaluation_raw.json")
OUTPUT_FILE    = os.path.join(RESULTS_DIR, "metrics_complete_report.json")
MANUAL_EVAL_FILE = os.path.join(RESULTS_DIR, "manual_evaluation.json")

CORE_CONDITIONS = [
    "vector_rag_baseline_generalist",
    "vector_rag_moe",
    "full_ice_generalist",
    "full_ice_moe",
]

def conversation_label(cid: str) -> str:
    mapping = {
        "633e26f8-5889-5c21-8c70-f4d7ab22cb00": "shinchan",
        "bb558b5f-5365-5bac-9ed0-07219025b5f2": "flaw",
        "a77c15cf-2078-4279-aeaa-8c3a6d58a972": "ice_dev",
        "ecc64aab-1979-5586-b0d8-c53448c0882e": "masters",
    }
    return mapping.get(cid, cid[:8])

def _default_missing_condition():
    return {
        "answer": "ERROR: Condition did not generate",
        "tokens_injected": 0,
        "classification": {
            "topic_tags": [],
            "intent_tags": [],
            "context_reliance": "Zero_Shot",
            "max_confidence": 0.0,
        },
        "hyde_used": False,
        "hyde_rewritten_query": None,
        "fragment_ids": [],
    }
def _is_failed_answer(answer: str) -> bool:
    """Return True if the answer is effectively absent (system failure)."""
    if not answer or not answer.strip():
        return True
    if answer.strip().upper().startswith("ERROR"):
        return True
    if len(answer.strip()) < 5:
        return True
    return False

PAIRED_CONDITION = {
    "vector_rag_baseline_generalist": "vector_rag_moe",
    "vector_rag_moe": "vector_rag_baseline_generalist",
    "full_ice_generalist": "full_ice_moe",
    "full_ice_moe": "full_ice_generalist",
}
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_data():
    master = load_json(MASTER_RESULTS)["evaluation_run_results"]
    eval_raw = load_json(EVAL_RAW) if os.path.exists(EVAL_RAW) else []
    eval_dict = {}
    for item in eval_raw:
        key = (item["conversation_id"], item["checkpoint_id"], item["probe_id"])
        eval_dict[key] = item
    if os.path.exists(MANUAL_EVAL_FILE):
        manual_eval = load_json(MANUAL_EVAL_FILE)
        for manual_entry in manual_eval:
            key = (
                manual_entry["conversation_id"],
                manual_entry["checkpoint_id"],
                manual_entry["probe_id"],
            )
            if key not in eval_dict:
                eval_dict[key] = {
                    "conversation_id": manual_entry["conversation_id"],
                    "checkpoint_id": manual_entry["checkpoint_id"],
                    "probe_id": manual_entry["probe_id"],
                    "question": manual_entry.get("question", ""),
                    "ground_truth": "",
                    "turn_index": 0,
                    "absolute_scores": {},
                    "tournament": None,
                    "hallucination": {},
                    "fragment_analysis": {},
                }
            entry = eval_dict[key]
            manual_abs = manual_entry.get("absolute_scores", {})
            if manual_abs:
                entry["absolute_scores"] = {
                    cond: {"score": score, "reasoning": "manual"}
                    for cond, score in manual_abs.items()
                    if score is not None
                }
            manual_ranking = manual_entry.get("tournament_ranking", [])
            if manual_ranking:
                entry["tournament"] = {
                    "rankings": manual_ranking,
                    "best_reason": "",
                    "worst_reason": "",
                }
            entry["hallucination"] = {}
            entry["fragment_analysis"] = {}
    return master, eval_dict

def load_fragment_sources(fragments_path):
    sources = defaultdict(list)
    if not os.path.exists(fragments_path):
        return sources
    with open(fragments_path, "r") as f:
        for line in f:
            try:
                frag = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            key = (
                frag.get("conversation_id", ""),
                frag.get("probe_id", ""),
                frag.get("checkpoint_id", ""),
                frag.get("condition", ""),
            )
            src = frag.get("source_type", frag.get("source", "unknown"))
            sources[key].append(src)
    return sources

# ---------------------------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------------------------
def compute_metrics(master_entries, eval_dict):
    conv_names = set()
    conv_stats = defaultdict(lambda: defaultdict(lambda: {
        "scores": [], "tokens": [], "hals": [],
        "wins": 0, "apps": 0, "score_counts": defaultdict(int),
        "fragment_counts": [],
    }))
    global_stats = defaultdict(lambda: defaultdict(lambda: {
        "scores": [], "tokens": [], "hals": [],
        "wins": 0, "apps": 0, "score_counts": defaultdict(int),
        "fragment_counts": [],
    }))
    # Explicit initialisation – avoids defaultdict surprises
    core_stats = {
        cond: {
            "scores": [], "tokens": [], "hals": [],
            "wins": 0, "apps": 0, "score_counts": defaultdict(int),
            "fragment_counts": [],
        }
        for cond in CORE_CONDITIONS
    }

    moe_data = {
        "vector_rag": {
            "moe": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0, "fragment_counts": []},
            "generalist": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0, "fragment_counts": []},
        },
        "full_ice": {
            "moe": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0, "fragment_counts": []},
            "generalist": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0, "fragment_counts": []},
        },
        "global": {
            "moe": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0, "fragment_counts": []},
            "generalist": {"scores": [], "tokens": [], "hals": [], "wins": 0, "apps": 0, "fragment_counts": []},
        },
    }

    frag_conv = defaultdict(lambda: defaultdict(list))
    frag_global = defaultdict(list)

    long_probes = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    long_conv_raw = defaultdict(lambda: defaultdict(list))
    long_global_raw = defaultdict(list)

    gating_failures = []

    for entry in master_entries:
        cid = entry["conversation_id"]
        checkpoint_id = entry["checkpoint_id"]
        probe_id = entry["probe_id"]
        turn = entry["turn_index"]
        conv_lbl = conversation_label(cid)
        conv_names.add(conv_lbl)

        eval_key = (cid, checkpoint_id, probe_id)
        eval_item = eval_dict.get(eval_key, {})

        first_cond = None
        for key in CORE_CONDITIONS:
            if key in entry["conditions"]:
                first_cond = entry["conditions"][key]
                break
        if first_cond is None:
            first_cond = _default_missing_condition()
        classification = first_cond.get("classification", {})
        context_reliance = classification.get("context_reliance", "")

        ice_gen_score = None
        ice_gen_data = entry["conditions"].get("full_ice_generalist", {})
        if ice_gen_data:
            abs_all = eval_item.get("absolute_scores", {})
            sc = abs_all.get("full_ice_generalist", {})
            if isinstance(sc, dict) and "score" in sc:
                ice_gen_score = sc["score"]

        # Pre‑compute valid scores for this probe (for imputation)
        valid_scores = {}
        for cond in CORE_CONDITIONS:
            sc = eval_item.get("absolute_scores", {}).get(cond, {})
            if isinstance(sc, dict) and "score" in sc:
                valid_scores[cond] = sc["score"]
        probe_avg = statistics.mean(valid_scores.values()) if valid_scores else None

        for cond_name in CORE_CONDITIONS:
            cond_data = entry["conditions"].get(cond_name)
            if cond_data is None:
                cond_data = _default_missing_condition()

            tokens = cond_data.get("tokens_injected", 0)
            frag_ids = cond_data.get("fragment_ids", [])
            frag_count = len(frag_ids) if isinstance(frag_ids, list) else 0

            abs_all = eval_item.get("absolute_scores", {})
            sc = abs_all.get(cond_name, {})
            score_val = sc.get("score") if isinstance(sc, dict) else None
            if score_val is None:
                answer_text = cond_data.get("answer", "")
                if _is_failed_answer(answer_text):
                    score_val = 1
                else:
                    paired_cond = PAIRED_CONDITION.get(cond_name)
                    if paired_cond and paired_cond in valid_scores:
                        score_val = valid_scores[paired_cond]
                    elif probe_avg is not None:
                        score_val = round(probe_avg)
                    else:
                        score_val = 3   # neutral fallback

            hall_all = eval_item.get("hallucination", {})
            hd = hall_all.get(cond_name, {}) if isinstance(hall_all, dict) else {}
            judge_evaluated_hall = "hallucination_count" in hd
            if judge_evaluated_hall:
                hal_count = hd["hallucination_count"]
                hal = 1 if hal_count > 0 else 0
            else:
                hal = None   # will be skipped – judge didn't evaluate

            for stats_dict, key in [(conv_stats, conv_lbl), (global_stats, "global")]:
                d = stats_dict[key][cond_name]
                d["scores"].append(score_val)
                d["score_counts"][score_val] += 1
                d["tokens"].append(tokens)
                d["fragment_counts"].append(frag_count)
                if hal is not None:
                    d["hals"].append(hal)

            cd = core_stats[cond_name]
            cd["scores"].append(score_val)
            cd["score_counts"][score_val] += 1
            cd["tokens"].append(tokens)
            cd["fragment_counts"].append(frag_count)
            if hal is not None:
                cd["hals"].append(hal)

            base = None
            if cond_name.startswith("vector_rag_"):
                base = "vector_rag"
            elif cond_name.startswith("full_ice_"):
                base = "full_ice"
            if base:
                variant = "moe" if "_moe" in cond_name else "generalist"
                for tgt in [base, "global"]:
                    dm = moe_data[tgt][variant]
                    dm["scores"].append(score_val)
                    dm["tokens"].append(tokens)
                    dm["fragment_counts"].append(frag_count)
                    if hal is not None:
                        dm["hals"].append(hal)

            long_probes[conv_lbl][probe_id][cond_name].append((turn, score_val))
            long_conv_raw[conv_lbl][cond_name].append((turn, score_val))
            long_global_raw[cond_name].append((turn, score_val))

        # Tournament rankings – only count conditions that exist
        tourn = eval_item.get("tournament")
        rankings = tourn.get("rankings", []) if isinstance(tourn, dict) else []
        for idx, cond_name in enumerate(rankings):
            if cond_name not in entry["conditions"]:
                continue
            for stats_dict, key in [(conv_stats, conv_lbl), (global_stats, "global")]:
                stats_dict[key][cond_name]["apps"] += 1
                if idx == 0:
                    stats_dict[key][cond_name]["wins"] += 1
            core_stats[cond_name]["apps"] += 1
            if idx == 0:
                core_stats[cond_name]["wins"] += 1
            base = None
            if cond_name.startswith("vector_rag_"):
                base = "vector_rag"
            elif cond_name.startswith("full_ice_"):
                base = "full_ice"
            if base:
                variant = "moe" if "_moe" in cond_name else "generalist"
                for tgt in [base, "global"]:
                    moe_data[tgt][variant]["apps"] += 1
                    if idx == 0:
                        moe_data[tgt][variant]["wins"] += 1

        # Fragment noise
        frag_all = eval_item.get("fragment_analysis", {})
        for cond_name in entry["conditions"]:
            fd = frag_all.get(cond_name, {})
            if isinstance(fd, dict) and fd.get("noise_score") is not None:
                frag_conv[conv_lbl][cond_name].append(fd["noise_score"])
                frag_global[cond_name].append(fd["noise_score"])

        # Gating failure
        if context_reliance == "Zero_Shot" and ice_gen_score is not None and ice_gen_score < 3:
            gating_failures.append({
                "conversation_id": cid,
                "checkpoint_id": checkpoint_id,
                "probe_id": probe_id,
                "question": entry["question"],
                "ice_score": ice_gen_score,
            })

    # ===== AFTER PER‑ENTRY LOOP =====

    # Memory maturity
    memory_maturity = {}
    conv_decay = defaultdict(list)
    for entry in master_entries:
        cid = entry["conversation_id"]
        conv_lbl = conversation_label(cid)
        days = entry.get("simulated_days", 0)
        cycles = entry.get("decay_cycles_run", 0)
        conv_decay[conv_lbl].append((days, cycles))
    for lbl, data in conv_decay.items():
        max_days = max(d[0] for d in data)
        max_cycles = max(d[1] for d in data)
        avg_days = statistics.mean(d[0] for d in data) if data else 0.0
        memory_maturity[lbl] = {
            "max_simulated_days": max_days,
            "max_decay_cycles": max_cycles,
            "mean_simulated_days_per_checkpoint": round(avg_days, 1),
            "note": "Earliest turns accumulated max_simulated_days of decay by the final checkpoint. Creative turns decay at 1%/day with a 0.3 floor; non‑creative unaccessed turns at 5%/day."
        }

    # Leg‑level contribution analysis
    fragments_path = os.path.join(RESULTS_DIR, "fragments.jsonl")
    frag_sources_lookup = load_fragment_sources(fragments_path)
    leg_conv = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    leg_global = defaultdict(lambda: defaultdict(list))
    for entry in master_entries:
        cid = entry["conversation_id"]
        checkpoint_id = entry["checkpoint_id"]
        probe_id = entry["probe_id"]
        conv_lbl = conversation_label(cid)
        for cond_name, cond_data in entry["conditions"].items():
            frag_ids = cond_data.get("fragment_ids", [])
            if not isinstance(frag_ids, list):
                continue
            lookup_key = (cid, probe_id, checkpoint_id, cond_name)
            src_list = frag_sources_lookup.get(lookup_key, [])
            src_counts = defaultdict(int)
            for src in src_list:
                src_counts[src] += 1
            if len(src_list) < len(frag_ids):
                src_counts["unknown"] += len(frag_ids) - len(src_list)
            for src, cnt in src_counts.items():
                leg_conv[conv_lbl][cond_name][src].append(cnt)
                leg_global[cond_name][src].append(cnt)

    leg_conv_summary = {}
    for lbl, cond_dict in leg_conv.items():
        leg_conv_summary[lbl] = {}
        for cond_name, src_dict in cond_dict.items():
            total_frags = sum(statistics.mean(lst) for lst in src_dict.values())
            summary = {}
            for src, lst in src_dict.items():
                avg = statistics.mean(lst) if lst else 0.0
                summary[src] = {
                    "avg_count": round(avg, 1),
                    "pct_of_total": round(avg / total_frags * 100, 1) if total_frags > 0 else 0.0,
                }
            summary["total_avg_fragments"] = round(total_frags, 1)
            leg_conv_summary[lbl][cond_name] = summary

    leg_global_summary = {}
    for cond_name, src_dict in leg_global.items():
        total_frags = sum(statistics.mean(lst) for lst in src_dict.values())
        summary = {}
        for src, lst in src_dict.items():
            avg = statistics.mean(lst) if lst else 0.0
            summary[src] = {
                "avg_count": round(avg, 1),
                "pct_of_total": round(avg / total_frags * 100, 1) if total_frags > 0 else 0.0,
            }
        summary["total_avg_fragments"] = round(total_frags, 1)
        leg_global_summary[cond_name] = summary

    # Build condition summary
    def build_cond_summary(d):
        scores = d["scores"]
        tokens = d["tokens"]
        hals = d["hals"]
        frags = d["fragment_counts"]
        avg_score = statistics.mean(scores) if scores else 0.0
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        avg_tokens = statistics.mean(tokens) if tokens else 0.0
        std_tokens = statistics.stdev(tokens) if len(tokens) > 1 else 0.0
        hal_rate = statistics.mean(hals) if hals else 0.0
        win_rate = (d["wins"] / d["apps"]) if d["apps"] > 0 else 0.0
        tur = (avg_score / (avg_tokens / 1000)) if avg_tokens > 0 else 0.0
        avg_frags = statistics.mean(frags) if frags else 0.0
        spf = (avg_score / avg_frags) if avg_frags > 0 else 0.0
        std_frags = statistics.stdev(frags) if len(frags) > 1 else 0.0
        total_scores = len(scores)
        dist = {}
        for i in range(1, 6):
            cnt = d["score_counts"].get(i, 0)
            dist[f"score_{i}"] = cnt
            dist[f"score_{i}_pct"] = round(cnt / total_scores * 100, 1) if total_scores > 0 else 0.0
        return {
            "avg_score": round(avg_score, 2),
            "std_score": round(std_score, 2),
            "avg_tokens_injected": int(avg_tokens),
            "std_tokens_injected": int(std_tokens),
            "win_rate_pct": round(win_rate * 100, 1),
            "hallucination_pct": round(hal_rate * 100, 1),
            "tur": round(tur, 2),
            "spf": round(spf, 2),
            "n_scores": total_scores,
            "n_tokens": len(tokens),
            "avg_fragments": round(avg_frags, 1),
            "std_fragments": round(std_frags, 1),
            "score_distribution": dist,
        }

    conv_summaries = {}
    for lbl in sorted(conv_names):
        stats = conv_stats[lbl]
        conv_summaries[lbl] = {}
        for cond, d in stats.items():
            conv_summaries[lbl][cond] = build_cond_summary(d)
        ice = conv_summaries[lbl].get("full_ice_generalist")
        vec = conv_summaries[lbl].get("vector_rag_baseline_generalist")
        if ice and vec:
            conv_summaries[lbl]["comparative_deltas"] = {
                "token_savings_vs_vector_pct": round(
                    (1 - (ice["avg_tokens_injected"] / vec["avg_tokens_injected"])) * 100, 1
                ) if vec["avg_tokens_injected"] > 0 else 0.0,
                "quality_gain_vs_vector_pts": round(ice["avg_score"] - vec["avg_score"], 2),
                "hallucination_reduction_vs_vector_pct": round(
                    vec["hallucination_pct"] - ice["hallucination_pct"], 1
                ),
                "fragment_count_reduction_pct": round(
                    (1 - (ice["avg_fragments"] / vec["avg_fragments"])) * 100, 1
                ) if vec["avg_fragments"] > 0 else 0.0,
            }

    global_summary = {}
    for cond, d in global_stats["global"].items():
        global_summary[cond] = build_cond_summary(d)
    ice_g = global_summary.get("full_ice_generalist")
    vec_g = global_summary.get("vector_rag_baseline_generalist")
    if ice_g and vec_g:
        global_summary["comparative_deltas"] = {
            "token_savings_vs_vector_pct": round(
                (1 - (ice_g["avg_tokens_injected"] / vec_g["avg_tokens_injected"])) * 100, 1
            ) if vec_g["avg_tokens_injected"] > 0 else 0.0,
            "quality_gain_vs_vector_pts": round(ice_g["avg_score"] - vec_g["avg_score"], 2),
            "hallucination_reduction_vs_vector_pct": round(
                vec_g["hallucination_pct"] - ice_g["hallucination_pct"], 1
            ),
            "fragment_count_reduction_pct": round(
                (1 - (ice_g["avg_fragments"] / vec_g["avg_fragments"])) * 100, 1
            ) if vec_g["avg_fragments"] > 0 else 0.0,
        }

    core_summary = {}
    for cond, d in core_stats.items():
        core_summary[cond] = build_cond_summary(d)
    ice_c = core_summary.get("full_ice_generalist")
    vec_c = core_summary.get("vector_rag_baseline_generalist")
    if ice_c and vec_c:
        core_summary["comparative_deltas"] = {
            "token_savings_vs_vector_pct": round(
                (1 - (ice_c["avg_tokens_injected"] / vec_c["avg_tokens_injected"])) * 100, 1
            ) if vec_c["avg_tokens_injected"] > 0 else 0.0,
            "quality_gain_vs_vector_pts": round(ice_c["avg_score"] - vec_c["avg_score"], 2),
            "hallucination_reduction_vs_vector_pct": round(
                vec_c["hallucination_pct"] - ice_c["hallucination_pct"], 1
            ),
            "fragment_count_reduction_pct": round(
                (1 - (ice_c["avg_fragments"] / vec_c["avg_fragments"])) * 100, 1
            ) if vec_c["avg_fragments"] > 0 else 0.0,
        }

    def build_moe_summary(data):
        result = {}
        for base, d in data.items():
            label = "global" if base == "global" else base
            moe = d["moe"]
            gen = d["generalist"]
            moe_score = statistics.mean(moe["scores"]) if moe["scores"] else 0.0
            moe_std = statistics.stdev(moe["scores"]) if len(moe["scores"]) > 1 else 0.0
            gen_score = statistics.mean(gen["scores"]) if gen["scores"] else 0.0
            gen_std = statistics.stdev(gen["scores"]) if len(gen["scores"]) > 1 else 0.0
            moe_tok = statistics.mean(moe["tokens"]) if moe["tokens"] else 0.0
            gen_tok = statistics.mean(gen["tokens"]) if gen["tokens"] else 0.0
            moe_hal = statistics.mean(moe["hals"]) if moe["hals"] else 0.0
            gen_hal = statistics.mean(gen["hals"]) if gen["hals"] else 0.0
            moe_win = (moe["wins"] / moe["apps"]) if moe["apps"] > 0 else 0.0
            gen_win = (gen["wins"] / gen["apps"]) if gen["apps"] > 0 else 0.0
            result[label] = {
                "moe": {
                    "avg_score": round(moe_score, 2),
                    "std_score": round(moe_std, 2),
                    "avg_tokens": int(moe_tok),
                    "win_rate_pct": round(moe_win * 100, 1),
                    "hallucination_pct": round(moe_hal * 100, 1),
                },
                "generalist": {
                    "avg_score": round(gen_score, 2),
                    "std_score": round(gen_std, 2),
                    "avg_tokens": int(gen_tok),
                    "win_rate_pct": round(gen_win * 100, 1),
                    "hallucination_pct": round(gen_hal * 100, 1),
                },
                "delta_moe_vs_generalist": {
                    "score_gain": round(moe_score - gen_score, 2),
                    "tokens_saved": int(gen_tok - moe_tok),
                    "hallucination_reduction_pct": round((gen_hal - moe_hal) * 100, 1),
                }
            }
        return result

    moe_summary = build_moe_summary(moe_data)

    def build_frag_summary(d):
        if not d:
            return {"mean_noise": None, "std_noise": 0.0, "n": 0}
        return {
            "mean_noise": round(statistics.mean(d), 2),
            "std_noise": round(statistics.stdev(d), 2) if len(d) > 1 else 0.0,
            "n": len(d),
        }

    frag_conv_summary = {}
    for lbl, cd in frag_conv.items():
        frag_conv_summary[lbl] = {c: build_frag_summary(lst) for c, lst in cd.items()}
    frag_global_summary = {c: build_frag_summary(lst) for c, lst in frag_global.items()}

    long_probes_final = {}
    for conv_lbl, probes in long_probes.items():
        for pid, cond_dict in probes.items():
            for cond_name, points in cond_dict.items():
                points.sort(key=lambda x: x[0])
                key = f"{conv_lbl}/{pid}/{cond_name}"
                long_probes_final[key] = {
                    "turns": [p[0] for p in points],
                    "scores": [p[1] for p in points],
                }

    long_conv_binned = {}
    for conv_lbl, cond_dict in long_conv_raw.items():
        binned_conv = {}
        for cond_name, points in cond_dict.items():
            points.sort(key=lambda x: x[0])
            bins = defaultdict(list)
            for t, s in points:
                bin_low = (t // 50) * 50
                bins[bin_low].append(s)
            centers = sorted(bins.keys())
            means = [statistics.mean(bins[b]) for b in centers]
            binned_conv[cond_name] = {
                "turn_bins": centers,
                "mean_scores": [round(m, 2) for m in means],
            }
        long_conv_binned[conv_lbl] = binned_conv

    global_binned = {}
    for cond_name, points in long_global_raw.items():
        points.sort(key=lambda x: x[0])
        bins = defaultdict(list)
        for t, s in points:
            bin_low = (t // 50) * 50
            bins[bin_low].append(s)
        centers = sorted(bins.keys())
        means = [statistics.mean(bins[b]) for b in centers]
        global_binned[cond_name] = {
            "turn_bins": centers,
            "mean_scores": [round(m, 2) for m in means],
        }

    temporal_quality = {}
    for cond_name in CORE_CONDITIONS:
        d = global_stats["global"].get(cond_name, {})
        total = len(d["scores"])
        if total == 0:
            continue
        q = {}
        for score in range(1, 6):
            cnt = d["score_counts"].get(score, 0)
            q[f"score_{score}_count"] = cnt
            q[f"score_{score}_pct"] = round(cnt / total * 100, 1)
        good = d["score_counts"].get(4, 0) + d["score_counts"].get(5, 0)
        poor = d["score_counts"].get(1, 0) + d["score_counts"].get(2, 0)
        q["good_pct"] = round(good / total * 100, 1)
        q["ok_pct"] = round(d["score_counts"].get(3, 0) / total * 100, 1)
        q["poor_pct"] = round(poor / total * 100, 1)
        temporal_quality[cond_name] = q

    frag_score_corr = {}
    for cond_name in CORE_CONDITIONS:
        pairs = []
        for entry in master_entries:
            cond_data = entry["conditions"].get(cond_name)
            if not cond_data:
                continue
            cid = entry["conversation_id"]
            checkpoint_id = entry["checkpoint_id"]
            probe_id = entry["probe_id"]
            eval_key = (cid, checkpoint_id, probe_id)
            eval_item = eval_dict.get(eval_key, {})
            abs_all = eval_item.get("absolute_scores", {})
            sc = abs_all.get(cond_name, {})
            score_val = sc.get("score") if isinstance(sc, dict) else None
            frag_ids = cond_data.get("fragment_ids", [])
            frag_count = len(frag_ids) if isinstance(frag_ids, list) else 0
            if score_val is not None:
                pairs.append((frag_count, score_val))
        if len(pairs) >= 3:
            try:
                r = statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs])
                frag_score_corr[cond_name] = round(r, 3)
            except statistics.StatisticsError:
                frag_score_corr[cond_name] = None
        else:
            frag_score_corr[cond_name] = None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conversation_lenses": conv_summaries,
        "global_summary": global_summary,
        "core_benchmark": core_summary,
        "moe_vs_generalist": moe_summary,
        "temporal_score_quality": temporal_quality,
        "gating_failures": gating_failures,
        "longitudinal_per_probe": long_probes_final,
        "longitudinal_per_conversation": long_conv_binned,
        "longitudinal_global": global_binned,
        "fragment_noise_per_conversation": frag_conv_summary,
        "fragment_noise_global": frag_global_summary,
        "memory_maturity": memory_maturity,
        "leg_contributions_per_conversation": leg_conv_summary,
        "leg_contributions_global": leg_global_summary,
        "fragment_score_correlation": frag_score_corr,
        "research_integrity_metadata": {
            "experiment_id": "ICE-MATURE-PHASE-2",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": {"gpu": "NVIDIA RTX 5090 (24 GB GDDR7)", "cpu": "Intel Ultra 9 275HX", "ram": "64 GB DDR5", "os": "CachyOS"},
            "judge_engine": {"model": "mattbucci/gemma-4-12B-AWQ", "framework": "SGLang", "context_window": 150000, "temperature": 0.0},
            "retrieval_architecture": {
                "classifier": "v3_qwen_ft3", "embedder": "Qwen3‑Embedding‑0.6B", "background_model": "Qwen2.5‑3B‑AWQ",
                "retrieval_legs": ["BM25", "Vector", "Codex", "Procedural", "RAG", "Batch Summaries"],
                "fusion": "RRF", "token_budget": "Unified dynamic (23k total)", "recent_window": "Dynamic",
                "clustering": "v4 – entity‑overlap (MicroNER)",
            },
            "evaluated_models": {"generalist": "gemma4:26b‑a4b‑it‑q4_K_M"},
            "conditions": CORE_CONDITIONS,
        },
    }
    return report

if __name__ == "__main__":
    print("Loading data...")
    master, eval_dict = load_data()
    print(f"Master entries: {len(master)}")
    print(f"Eval entries:   {len(eval_dict)}")
    print("Computing exhaustive metrics...")
    report = compute_metrics(master, eval_dict)
    save_json(report, OUTPUT_FILE)
    print(f"\nComplete report saved to {OUTPUT_FILE}")
    g = report["global_summary"]
    ice = g.get("full_ice_generalist", {})
    vec = g.get("vector_rag_baseline_generalist", {})
    print("\n" + "█" * 70)
    print(" ICE-MATURE — GLOBAL METRICS SNAPSHOT")
    print("█" * 70)
    print(f" ICE generalist  | score: {ice.get('avg_score')}±{ice.get('std_score')}  tokens: {ice.get('avg_tokens_injected')}  TUR: {ice.get('tur')}  frags: {ice.get('avg_fragments')}  wins: {ice.get('win_rate_pct')}%  hal: {ice.get('hallucination_pct')}%")
    print(f" Vector generalist | score: {vec.get('avg_score')}±{vec.get('std_score')}  tokens: {vec.get('avg_tokens_injected')}  TUR: {vec.get('tur')}  frags: {vec.get('avg_fragments')}  wins: {vec.get('win_rate_pct')}%  hal: {vec.get('hallucination_pct')}%")
    print(f" ICE score distribution: {ice.get('score_distribution')}")
    print(f" Vector score distribution: {vec.get('score_distribution')}")
    dels = g.get("comparative_deltas", {})
    print(f" Token savings vs vector: {dels.get('token_savings_vs_vector_pct')}%")
    print(f" Quality gain vs vector:  {dels.get('quality_gain_vs_vector_pts')} pts")
    print(f" Fragment reduction vs vec: {dels.get('fragment_count_reduction_pct')}%")
    print(" Fragment‑count vs score correlations:", report.get("fragment_score_correlation"))
    print("█" * 70)