#!/usr/bin/env python3
"""Token efficiency as a function of conversation position and per-turn density.

The paper reports a single pooled figure — ICE injects 6.6% more tokens than the
vector baseline. That average hides a crossover, and this script measures it.

Two mechanisms push in opposite directions:

  * Per-turn DENSITY favours ICE. The baseline retrieves a fixed top-30, so its
    injected tokens scale linearly with how large each turn is, unbounded. ICE's
    per-query budget caps its own injection regardless.
  * Conversation LENGTH favours the baseline. ICE's growth cap deliberately
    widens the budget as turns accumulate (2,000 + 150n below 30 turns, then
    5,000 + 100(n-30), then 10,000 + 30(n-100)), while top-30 is indifferent to
    conversation length.

Every comparison here is PAIRED (both conditions answer the same probe) and
bootstrapped, so the trends are reported with intervals rather than as point
estimates from a single pass.

Read-only over the frozen results.
Run: uv run python experiments/mature/token_efficiency_analysis.py
"""
import json
from collections import defaultdict

import numpy as np

SRC = "experiments/mature/intermediates/master_results.json"
ICE = "full_ice_generalist"
VEC = "vector_rag_baseline_generalist"

# Anonymised labels matching the paper's dataset table.
CONVERSATIONS = {
    "633e26f8-5889-5c21-8c70-f4d7ab22cb00": ("A", "Creative Writing", 290),
    "bb558b5f-5365-5bac-9ed0-07219025b5f2": ("B", "Long-Form Creative", 1119),
    "a77c15cf-2078-4279-aeaa-8c3a6d58a972": ("C", "Technical Planning", 325),
    "ecc64aab-1979-5586-b0d8-c53448c0882e": ("D", "Academic Planning", 251),
}

B, SEED = 10_000, 42
rng = np.random.default_rng(SEED)


def load():
    """One record per probe carrying both conditions' injected token counts."""
    rows = []
    for e in json.load(open(SRC))["evaluation_run_results"]:
        c = e.get("conditions", {})
        turn = e.get("turn_index") or 0
        if ICE not in c or VEC not in c or turn <= 0:
            continue
        ice, vec = c[ICE].get("tokens_injected"), c[VEC].get("tokens_injected")
        total = e.get("total_tokens_in_conversation") or 0
        if not ice or not vec:
            continue
        rows.append({
            "conv": e["conversation_id"],
            "turn": turn,
            "ice": ice,
            "vec": vec,
            "density": total / turn if total else None,
        })
    return rows


def boot_ci(arr):
    """95% percentile-bootstrap CI of the mean."""
    arr = np.asarray(arr, dtype=float)
    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    bs = arr[idx].mean(axis=1)
    return float(arr.mean()), tuple(np.percentile(bs, [2.5, 97.5]))


def report(label, subset):
    """Paired ICE-minus-baseline token delta, and the share where ICE is cheaper."""
    diff = [r["ice"] - r["vec"] for r in subset]
    cheaper = [100.0 * (r["ice"] < r["vec"]) for r in subset]
    d, dci = boot_ci(diff)
    p, pci = boot_ci(cheaper)
    ice_m = np.mean([r["ice"] for r in subset])
    vec_m = np.mean([r["vec"] for r in subset])
    print(f"{label:<16}{len(subset):>5}{ice_m:>9.0f}{vec_m:>9.0f}"
          f"{d:>+9.0f}  [{dci[0]:>+7.0f},{dci[1]:>+7.0f}]"
          f"{p:>7.0f}%  [{pci[0]:>3.0f},{pci[1]:>3.0f}]")


def main():
    rows = load()
    print(f"Paired probes: {len(rows)}  |  bootstrap B={B:,}, seed={SEED}\n")
    hdr = (f"{'':<16}{'n':>5}{'ICE':>9}{'base':>9}{'paired Δ':>9}  {'95% CI':^18}"
           f"{'ICE cheaper':>10}  {'95% CI':^10}")

    # ── 1. Within-conversation trend (controls for conversation identity) ──
    print("=" * 100)
    print("1. WITHIN CONVERSATION, BY POSITION — the baseline is flat, ICE climbs")
    print("=" * 100)
    for conv, (letter, name, turns) in CONVERSATIONS.items():
        sub = sorted([r for r in rows if r["conv"] == conv], key=lambda r: r["turn"])
        if len(sub) < 8:
            continue
        print(f"\nDataset {letter} — {name} ({turns} turns)")
        print(hdr)
        q = len(sub) // 4
        for i in range(4):
            chunk = sub[i * q:(i + 1) * q] if i < 3 else sub[3 * q:]
            if chunk:
                report(f"turns {chunk[0]['turn']}-{chunk[-1]['turn']}", chunk)

    # ── 2. Pooled by position ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("2. POOLED, BY CONVERSATION POSITION")
    print("=" * 100)
    print(hdr)
    for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 400), (400, 700), (700, 1200)]:
        sub = [r for r in rows if lo <= r["turn"] < hi]
        if len(sub) >= 5:
            report(f"{lo}-{hi}", sub)

    # ── 3. Pooled by per-turn density ─────────────────────────────────────
    print("\n" + "=" * 100)
    print("3. POOLED, BY PER-TURN DENSITY (tokens per turn in the conversation)")
    print("=" * 100)
    print(hdr)
    dens = [r for r in rows if r["density"]]
    for lo, hi in [(0, 700), (700, 900), (900, 1200), (1200, 10**9)]:
        sub = [r for r in dens if lo <= r["density"] < hi]
        if len(sub) >= 5:
            label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
            report(label, sub)

    # ── 4. Caveat, stated in the output so it travels with the numbers ────
    print("\n" + "=" * 100)
    print("CAVEAT: with four conversations, the density bands correlate strongly with")
    print("conversation identity (the densest band is entirely Dataset C). Section 1 is")
    print("the controlled view — the position trend holds WITHIN every conversation,")
    print("where the baseline is fixed and only ICE's budget moves.")


if __name__ == "__main__":
    main()
