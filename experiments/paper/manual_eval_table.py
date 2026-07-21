#!/usr/bin/env python3
"""Human-scored evaluation slice — stats for the paper.

72 hand-written probes were scored entirely by hand (absolute 1-5 + a blind
tournament ranking), independent of the LLM judge. This computes mean score,
win rate, and paired bootstrap CIs so the human slice can sit in the paper as
its own table alongside the automated Experiment-2 results.

Read-only over the frozen results; run: uv run python experiments/paper/manual_eval_table.py
"""
import json
import numpy as np

SRC = "experiments/mature/results/manual_evaluation.json"
CONDS = ["vector_rag_baseline_generalist", "vector_rag_moe",
         "full_ice_generalist", "full_ice_moe"]
LABEL = {"vector_rag_baseline_generalist": "Vector RAG (Gen)",
         "vector_rag_moe": "Vector RAG (MoE)",
         "full_ice_generalist": "Full ICE (Gen)",
         "full_ice_moe": "Full ICE (MoE)"}
B, rng = 10_000, np.random.default_rng(42)


def main():
    data = json.load(open(SRC))
    n = len(data)
    scores = {c: np.array([d["absolute_scores"][c] for d in data], float) for c in CONDS}
    wins = {c: np.array([1.0 if (d.get("tournament_ranking") or [None])[0] == c else 0.0
                         for d in data]) for c in CONDS}

    def ci(arr):
        idx = rng.integers(0, len(arr), size=(B, len(arr)))
        bs = arr[idx].mean(axis=1)
        return arr.mean(), np.percentile(bs, [2.5, 97.5])

    print(f"Human-scored slice: n={n} probes\n")
    print(f"{'Condition':<20}{'Score':>8}{'95% CI':>16}{'Win%':>8}{'95% CI':>16}")
    print("-" * 68)
    for c in CONDS:
        s, sci = ci(scores[c])
        w, wci = ci(wins[c] * 100)
        print(f"{LABEL[c]:<20}{s:>8.2f}  [{sci[0]:.2f},{sci[1]:.2f}]{w:>8.1f}  [{wci[0]:.1f},{wci[1]:.1f}]")

    # paired ICE-gen minus Vec-gen
    d = scores["full_ice_generalist"] - scores["vector_rag_baseline_generalist"]
    idx = rng.integers(0, len(d), size=(B, len(d)))
    dci = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    print(f"\nPaired ICE(Gen) - Vec(Gen): {d.mean():+.2f}  [{dci[0]:+.2f}, {dci[1]:+.2f}]  "
          f"{'EXCLUDES 0' if (dci[0] > 0 or dci[1] < 0) else 'tie'}")


if __name__ == "__main__":
    main()
