#!/usr/bin/env python3
"""
Compare Experiment 2 Flaw final‑checkpoint scores (same probes as buildup)
against the buildup results.

Filters Exp 2 master_results.json to Flaw, turn 1119, kept splits only.
Uses already‑corrected evaluation_raw.json.
"""

import json, os, re, statistics
from collections import defaultdict

MATURE_DIR = "experiments/mature"
MASTER_FILE = os.path.join(MATURE_DIR, "intermediates", "master_results.json")
EVAL_FILE   = os.path.join(MATURE_DIR, "intermediates", "evaluation_raw.json")

FLAW_CID    = "bb558b5f-5365-5bac-9ed0-07219025b5f2"
FINAL_TURN  = 1119
KEPT_CHECKPOINTS = {51, 170, 285, 397, 492, 604, 735, 834, 959, 1053, 1119}

CORE = [
    "vector_rag_baseline_generalist",
    "full_ice_generalist",
]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def origin_split(probe_id):
    m = re.match(r"(\d+)-GEN-", probe_id)
    return int(m.group(1)) if m else FINAL_TURN   # manual probes treated as final

def main():
    master = load_json(MASTER_FILE)["evaluation_run_results"]
    eval_raw = load_json(EVAL_FILE) if os.path.exists(EVAL_FILE) else []
    eval_dict = {}
    for item in eval_raw:
        key = (item["conversation_id"], item["checkpoint_id"], item["probe_id"])
        eval_dict[key] = item

    accum = defaultdict(lambda: {"scores": [], "tokens": [], "frags": [], "hals": []})

    for entry in master:
        if entry["conversation_id"] != FLAW_CID:
            continue
        if entry.get("turn_index") != FINAL_TURN:
            continue
        pid = entry["probe_id"]
        split = origin_split(pid)
        if split not in KEPT_CHECKPOINTS:
            continue

        cid = entry["conversation_id"]
        checkpoint_id = entry["checkpoint_id"]
        ekey = (cid, checkpoint_id, pid)
        eitem = eval_dict.get(ekey, {})

        for cond in CORE:
            cond_data = entry.get("conditions", {}).get(cond)
            if not cond_data:
                continue
            tokens = cond_data.get("tokens_injected", 0)
            frags  = len(cond_data.get("fragment_ids", []))
            accum[cond]["tokens"].append(tokens)
            accum[cond]["frags"].append(frags)

            sc = eitem.get("absolute_scores", {}).get(cond, {})
            score = sc.get("score") if isinstance(sc, dict) else None
            if score is not None:
                accum[cond]["scores"].append(score)

            hd = eitem.get("hallucination", {}).get(cond, {})
            hal = 1 if hd.get("hallucination_count", 0) > 0 else 0
            accum[cond]["hals"].append(hal)

    print(f"Probes matched: {len(accum['full_ice_generalist']['scores'])}")
    print(f"\n{'Condition':<40} {'Score':>6} {'Tokens':>7} {'Frags':>6} {'SPF':>6} {'TUR':>6} {'Hall%':>6}")
    print("-" * 90)

    for cond in CORE:
        d = accum[cond]
        scores = d["scores"]
        tokens = d["tokens"]
        frags  = d["frags"]
        hals   = d["hals"]
        if not scores:
            continue
        ms = statistics.mean(scores)
        mt = statistics.mean(tokens)
        mf = statistics.mean(frags)
        spf = ms / mf if mf > 0 else 0.0
        tur = ms / (mt / 1000) if mt > 0 else 0.0
        hp  = statistics.mean(hals) * 100 if hals else 0.0
        print(f"{cond:<40} {ms:>6.2f} {int(mt):>7} {mf:>6.1f} {spf:>6.2f} {tur:>6.2f} {hp:>5.1f}%")

    # Deltas
    ice = accum.get("full_ice_generalist", {})
    vec = accum.get("vector_rag_baseline_generalist", {})
    if ice.get("scores") and vec.get("scores"):
        ice_score = statistics.mean(ice["scores"])
        vec_score = statistics.mean(vec["scores"])
        print(f"\nQuality delta (ICE − Vector): {ice_score - vec_score:+.2f}")
        ice_tok = statistics.mean(ice["tokens"])
        vec_tok = statistics.mean(vec["tokens"])
        print(f"Token delta: {ice_tok - vec_tok:+.0f}  (ICE saves {-(ice_tok-vec_tok)/vec_tok*100:.1f}%)" if vec_tok > 0 else "")

if __name__ == "__main__":
    main()