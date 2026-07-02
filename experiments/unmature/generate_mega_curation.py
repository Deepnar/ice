#!/usr/bin/env python3
"""Generate a single curation file merging two full conversations into one history."""

import json, os, uuid
from datetime import datetime, timedelta, timezone

SIMULATION_INPUT = "data/simulation/simulation_full.jsonl"
OUTPUT_DIR = "experiments/curation_files"

# The two conversation IDs to merge – Flaw from GPT, Flaw from Claude
CONV_IDS = [
    "bb558b5f-5365-5bac-9ed0-07219025b5f2",   # Flaw (GPT)
    "cca73c87-2068-4211-ab51-f38b6e966b0a",     # Flaw (Claude)
]

# New conversation ID for the merged mega-conversation
MEGA_CID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "flaw_merged_gpt_claude"))

# Start timestamp for the synthetic timeline
BASE_TS = datetime(2025, 1, 1, tzinfo=timezone.utc)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all turns
    with open(SIMULATION_INPUT, "r") as f:
        all_turns = [json.loads(line) for line in f if line.strip()]

    # Collect turns for each conversation in order, preserving internal timestamp order
    merged = []
    for cid in CONV_IDS:
        turns = [t for t in all_turns if t.get("conversation_id") == cid]
        turns.sort(key=lambda x: x["timestamp"])
        merged.extend(turns)

    if not merged:
        print("No turns found for the given conversation IDs.")
        return

    # Assign synthetic timestamps starting from BASE_TS, 5 minutes apart
    current_ts = BASE_TS
    for turn in merged:
        turn["timestamp"] = current_ts.isoformat()
        turn["conversation_id"] = MEGA_CID
        current_ts += timedelta(minutes=5)

    # Build the full history block (every turn)
    history_block = []
    for i, t in enumerate(merged, start=1):
        history_block.append({
            "turn_number": i,
            "user_input": t["prompt"],
            "ai_response": t["response"],
        })

    # No future turns – this is the entire conversation
    future_block = []

    # Empty probe slots
    probes = []
    for i in range(12):   # 12 slots for a big conversation
        probes.append({
            "probe_id": f"P-{i+1:02d}",
            "probe_type": "ENTER_TYPE",
            "user_injected_prompt": "ENTER_PROBE_HERE",
            "expected_answer": "ENTER_EXPECTED_ANSWER_OR_BLANK",
            "ground_truth_expected_fragments": []
        })

    checkpoint_id = f"EC-{MEGA_CID[:8]}-FULL"
    curation = {
        "evaluation_checkpoint_id": checkpoint_id,
        "original_conversation_id": MEGA_CID,
        "simulated_present_timestamp": merged[-1]["timestamp"],
        "split_turn_index": len(merged),     # split at the very end → all history
        "total_turns": len(merged),
        "historical_context_block": history_block,
        "future_reference_block": future_block,
        "evaluation_probes": probes
    }

    out_path = os.path.join(OUTPUT_DIR, f"{checkpoint_id}.json")
    with open(out_path, "w") as f:
        json.dump(curation, f, indent=2)
    print(f"Mega-curation file saved: {out_path}")
    print(f"Total turns in merged history: {len(merged)}")

if __name__ == "__main__":
    main()