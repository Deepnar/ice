#!/usr/bin/env python3
"""
Manual judge for hand-written probes (P-XX format).
Displays question, ground truth, and all 4 answers truncated.
User enters absolute scores (1-5) and tournament ranking (1-4).
Resumable – skips already-evaluated probes.

Usage:
    uv run python experiments/mature/manual_evaluate.py
"""

import json
import os
from pathlib import Path

MATURE_DIR = Path(__file__).parent
MASTER_FILE = MATURE_DIR / "intermediates" / "master_results.json"
OUTPUT_FILE = MATURE_DIR / "intermediates" / "manual_evaluation.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_evaluated():
    if not OUTPUT_FILE.exists():
        return set()
    data = load_json(OUTPUT_FILE)
    return set((e["conversation_id"], e["probe_id"], e["checkpoint_id"]) for e in data)

def is_handwritten_probe(probe_id: str) -> bool:
    """Hand-written probes follow P-XX format."""
    return bool(probe_id.startswith("P-") and len(probe_id) <= 5)

def main():
    master = load_json(MASTER_FILE)
    entries = master["evaluation_run_results"]

    # Filter to hand-written probes only
    hand_entries = [e for e in entries if is_handwritten_probe(e.get("probe_id", ""))]
    if not hand_entries:
        print("No hand-written probes found in master_results.json.")
        return

    evaluated = load_evaluated()
    pending = [e for e in hand_entries
               if (e["conversation_id"], e["probe_id"], e["checkpoint_id"]) not in evaluated]

    print(f"Hand-written probes: {len(hand_entries)} total, {len(pending)} pending\n")

    if not pending:
        print("All hand-written probes already evaluated.")
        return

    results = []
    if OUTPUT_FILE.exists():
        results = load_json(OUTPUT_FILE)

    for idx, entry in enumerate(pending, 1):
        cid = entry["conversation_id"]
        ckpt = entry["checkpoint_id"]
        pid = entry["probe_id"]
        question = entry["question"]
        ground_truth = entry.get("ground_truth", "")
        turn = entry.get("turn_index", "?")

        print("=" * 70)
        print(f"[{idx}/{len(pending)}]  Probe: {pid}  |  Turn: {turn}  |  Conv: {cid[:8]}...")
        print("=" * 70)
        print(f"\nQUESTION:\n{question}\n")
        print(f"GROUND TRUTH (first 800 chars):\n{ground_truth[:800]}\n")

        conditions = entry.get("conditions", {})
        cond_names = list(conditions.keys())
        answers = {}

        for i, cn in enumerate(cond_names):
            ans = conditions[cn].get("answer", "[NO ANSWER]")
            answers[cn] = ans
            truncated = ans
            print(f"--- [{i+1}] {cn} ---")
            print(f"TOKENS: {conditions[cn].get('tokens_injected', '?')}  |  MODEL: {conditions[cn].get('model_used', '?')}")
            print(f"ANSWER (first  chars):\n{truncated}\n")

        # ── Absolute scores ──
        print("Enter ABSOLUTE SCORES (1-5) for each condition:")
        abs_scores = {}
        for cn in cond_names:
            while True:
                try:
                    s = input(f"  {cn}: ").strip()
                    if s == "":
                        abs_scores[cn] = None
                        break
                    s = int(s)
                    if 1 <= s <= 5:
                        abs_scores[cn] = s
                        break
                    else:
                        print("   Must be 1-5.")
                except ValueError:
                    print("   Invalid input.")
                except (EOFError, KeyboardInterrupt):
                    print("\nInterrupted. Saving progress...")
                    save_json(results, OUTPUT_FILE)
                    return

        # ── Tournament ranking (by reference number) ──
        print(f"\nEnter TOURNAMENT RANKING. Order the conditions from BEST (1) to WORST ({len(cond_names)}).")
        print("Type the reference numbers in order, e.g., 2,1,4,3")
        for i, cn in enumerate(cond_names, 1):
            print(f"  [{i}] {cn}")

        ranking = []
        while True:
            try:
                line = input("Ranking (numbers): ").strip()
                if not line:
                    continue
                # Accept comma‑separated numbers
                nums = [n.strip() for n in line.split(",") if n.strip().isdigit()]
                if len(nums) != len(cond_names):
                    print(f"  Must enter exactly {len(cond_names)} numbers. You entered {len(nums)}.")
                    continue
                indices = [int(n)-1 for n in nums]   # convert to 0‑based
                if sorted(indices) != list(range(len(cond_names))):
                    print(f"  Must use each number 1-{len(cond_names)} exactly once.")
                    continue
                ranking = [cond_names[i] for i in indices]
                break
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted. Saving progress...")
                save_json(results, OUTPUT_FILE)
                return

        # ── Notes ──
        notes = input("\nOptional notes (press Enter to skip): ").strip()

        result = {
            "conversation_id": cid,
            "probe_id": pid,
            "checkpoint_id": ckpt,
            "question": question,
            "absolute_scores": abs_scores,
            "tournament_ranking": ranking,
            "notes": notes,
        }
        results.append(result)
        save_json(results, OUTPUT_FILE)
        print(f"\n✅ Saved. {len(pending)-idx} remaining.\n")

    print(f"\nAll done. {len(results)} evaluations in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()