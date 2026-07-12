"""Bootstrap 95% confidence intervals for the ICE paper's headline numbers.

Replicates the PUBLISHED aggregation from experiments/mature/compute_metrics*.py
exactly — manual_evaluation.json merge, the missing-score imputation chain
(failed answer -> 1, else paired condition's score, else round(probe average),
else 3), and tournament win counting over conditions present in the record —
then bootstraps (10k resamples, percentile method) over probe records.

First naive version of this script dropped None scores and got means ~0.4 off
the paper on ICE-Dev (the 141 failed vector answers are imputed as score 1 in
the published pipeline, not dropped) — hence the faithful replication.

Read-only over the frozen results folder; writes nothing.
Run: uv run python scripts/oneoff/paper_bootstrap_cis.py
"""
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

RESULTS = "experiments/mature/results"
CONDS = ["vector_rag_baseline_generalist", "vector_rag_moe",
         "full_ice_generalist", "full_ice_moe"]
PAIRED = {
    "vector_rag_baseline_generalist": "vector_rag_moe",
    "vector_rag_moe": "vector_rag_baseline_generalist",
    "full_ice_generalist": "full_ice_moe",
    "full_ice_moe": "full_ice_generalist",
}
ICE_DEV = "a77c15cf-2078-4279-aeaa-8c3a6d58a972"
N_BOOT = 10_000
random.seed(42)


def _is_failed_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    if answer.strip().upper().startswith("ERROR"):
        return True
    if len(answer.strip()) < 5:
        return True
    return False


def load_records():
    """One flat record per (probe x checkpoint): imputed scores per condition,
    tournament winner, set of conditions present — the published semantics."""
    master = json.load(open(f"{RESULTS}/master_results.json"))["evaluation_run_results"]
    eval_dict = {}
    for item in json.load(open(f"{RESULTS}/evaluation_raw.json")):
        eval_dict[(item["conversation_id"], item["checkpoint_id"], item["probe_id"])] = item
    manual_path = f"{RESULTS}/manual_evaluation.json"
    if os.path.exists(manual_path):
        for me in json.load(open(manual_path)):
            key = (me["conversation_id"], me["checkpoint_id"], me["probe_id"])
            entry = eval_dict.setdefault(key, {"absolute_scores": {}, "tournament": None})
            if me.get("absolute_scores"):
                entry["absolute_scores"] = {
                    c: {"score": s, "reasoning": "manual"}
                    for c, s in me["absolute_scores"].items() if s is not None}
            if me.get("tournament_ranking"):
                entry["tournament"] = {"rankings": me["tournament_ranking"]}

    records = []
    for entry in master:
        key = (entry["conversation_id"], entry["checkpoint_id"], entry["probe_id"])
        ev = eval_dict.get(key, {})
        abs_all = ev.get("absolute_scores", {}) or {}
        valid = {c: sc["score"] for c in CONDS
                 if isinstance((sc := abs_all.get(c, {})), dict) and "score" in sc}
        probe_avg = statistics.mean(valid.values()) if valid else None

        scores = {}
        for c in CONDS:
            cond_data = entry["conditions"].get(c)
            if c in valid:
                scores[c] = valid[c]
            elif cond_data is not None and _is_failed_answer(cond_data.get("answer", "")):
                scores[c] = 1
            elif PAIRED[c] in valid:
                scores[c] = valid[PAIRED[c]]
            elif probe_avg is not None:
                scores[c] = round(probe_avg)
            else:
                scores[c] = 3

        tourn = ev.get("tournament")
        rankings = tourn.get("rankings", []) if isinstance(tourn, dict) else []
        rankings = [c for c in rankings if c in entry["conditions"]]
        records.append({
            "conversation_id": entry["conversation_id"],
            "scores": scores,
            "winner": rankings[0] if rankings else None,
            "in_tournament": set(rankings),
        })
    return records


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * len(xs))))]


def boot(records, stat):
    point = stat(records)
    n = len(records)
    samples = []
    for _ in range(N_BOOT):
        rs = [records[random.randrange(n)] for _ in range(n)]
        samples.append(stat(rs))
    return point, pctl(samples, 0.025), pctl(samples, 0.975)


def report(label, records):
    print(f"\n== {label} ({len(records)} records) ==")
    for c in CONDS:
        m, mlo, mhi = boot(records, lambda rs, c=c: statistics.mean(r["scores"][c] for r in rs))

        def wr(rs, c=c):
            apps = sum(1 for r in rs if c in r["in_tournament"])
            wins = sum(1 for r in rs if r["winner"] == c)
            return 100.0 * wins / apps if apps else 0.0
        w, wlo, whi = boot(records, wr)
        print(f"  {c:34s} score {m:.2f} [{mlo:.2f}, {mhi:.2f}]   win% {w:.1f} [{wlo:.1f}, {whi:.1f}]")

    def delta(rs):
        return statistics.mean(r["scores"]["full_ice_generalist"]
                               - r["scores"]["vector_rag_baseline_generalist"] for r in rs)
    d, dlo, dhi = boot(records, delta)
    verdict = "EXCLUDES 0" if (dlo > 0 or dhi < 0) else "includes 0 (statistical tie)"
    print(f"  PAIRED ICE-gen minus Vec-gen: {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]  -> {verdict}")


def main():
    records = load_records()
    report("ALL DATA (1,211 probes, paper Tab. exp2-alldata)", records)
    report("EXCLUDING ICE-Dev (1,057 probes, paper Tab. exp2-global)",
           [r for r in records if r["conversation_id"] != ICE_DEV])
    report("ICE-Dev ONLY (154 probes, paper Tab. stress)",
           [r for r in records if r["conversation_id"] == ICE_DEV])


if __name__ == "__main__":
    main()
