#!/usr/bin/env python3
"""
Phase 1: Generate curation files with empty probe slots.
Reads simulation_full.jsonl, splits conversations at random points,
writes one JSON file per checkpoint. Shows ALL history turns (not just last N).
No simulation or database interaction.
"""

import json, os, sys, random
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SIMULATION_INPUT = "data/simulation/simulation_full.jsonl"
CURATION_DIR = "experiments/curation_files"
SPLITS_PER_CONV = 3
FUTURE_WINDOW = 10          # how many future turns to show in the reference block
SEED = 42

def main():
    random.seed(SEED)
    os.makedirs(CURATION_DIR, exist_ok=True)

    with open(SIMULATION_INPUT, "r") as f:
        all_turns = [json.loads(line) for line in f if line.strip()]

    convs = defaultdict(list)
    for turn in all_turns:
        cid = turn.get("conversation_id", "unknown")
        convs[cid].append(turn)

    for cid, turns in convs.items():
        L = len(turns)
        if L < 10:
            continue

        MIN_HISTORY = 10
        MIN_PCT = 0.30          # at least 30% of the conversation before the split
        MAX_PCT = 0.95          # up to 95% of the conversation before the split (very late)

        min_idx = max(MIN_HISTORY, int(L * MIN_PCT))
        max_idx = int(L * MAX_PCT)

        # Ensure min <= max and within bounds
        if min_idx >= L:
            continue   # skip – conversation too short to have enough history

        if max_idx >= L:
            max_idx = L - 1

        if max_idx < min_idx:
            # Fallback: force a single split at 50%, but only if it gives enough history
            fallback = max(MIN_HISTORY, int(L * 0.5))
            if fallback < L:
                split_indices = [fallback]
            else:
                continue
        else:
            possible = list(range(min_idx, max_idx + 1))
            if len(possible) < SPLITS_PER_CONV:
                split_indices = possible[:SPLITS_PER_CONV] if possible else []
            else:
                split_indices = sorted(random.sample(possible, SPLITS_PER_CONV))
        for split_n in split_indices:
            history = turns[:split_n]
            future = turns[split_n:split_n + FUTURE_WINDOW]

            # Build FULL history block (every turn before the split)
            history_block = []
            for i, t in enumerate(history, start=1):
                history_block.append({
                    "turn_number": i,
                    "user_input": t["prompt"],      # truncated for readability
                    "ai_response": t["response"],
                })

            future_block = []
            for i, t in enumerate(future, start=split_n + 1):
                future_block.append({
                    "turn_number": i,
                    "user_input": t["prompt"],
                    "ai_response": t["response"],
                })

            probes = []
            for i in range(8):
                probes.append({
                    "probe_id": f"P-{i+1:02d}",
                    "probe_type": "ENTER_TYPE",
                    "user_injected_prompt": "ENTER_PROBE_HERE",
                    "expected_answer": "ENTER_EXPECTED_ANSWER_OR_BLANK",
                    "ground_truth_expected_fragments": []
                })

            checkpoint_id = f"EC-{cid[:8]}-TURN{split_n}"
            curation = {
                "evaluation_checkpoint_id": checkpoint_id,
                "original_conversation_id": cid,
                "simulated_present_timestamp": history[-1]["timestamp"] if history else "",
                "split_turn_index": split_n,
                "total_turns": L,
                "historical_context_block": history_block,
                "future_reference_block": future_block,
                "evaluation_probes": probes
            }

            out_path = os.path.join(CURATION_DIR, f"{checkpoint_id}.json")
            with open(out_path, "w") as f:
                json.dump(curation, f, indent=2)

    print(f"Phase 1 complete. Curation files saved to {CURATION_DIR}")

if __name__ == "__main__":
    main()