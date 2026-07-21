#!/usr/bin/env python3
"""
Correct Experiment 1 tokens_injected for full‑ICE probes by adding the
sliding‑window contribution that was omitted from the original report.

The old get_recent_turns() used the last 10 turns, 500‑word cap per turn.
This script replays that exact logic against simulation_full.jsonl and
adds the resulting token count to every full_ice_* condition.

Control and vector‑rag conditions are not modified — they counted their
own context correctly in the original experiment.
"""

import json, os, re
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MASTER_RESULTS = "experiments/unmature/intermediates/master_results.json"
SIMULATION_INPUT = "data/simulation/simulation_full.jsonl"
OUTPUT_CORRECTED = "experiments/unmature/intermediates/master_results_corrected.json"

# Experiment 1 sliding‑window constants (from the old prompt_assembler.py)
SLIDING_WINDOW_N = 10
SLIDING_WINDOW_WORD_CAP = 500

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def estimate_tokens(text):
    return int(len(text.split()) * 1.33)


def get_sliding_window_tokens(conv_turns, turn_index):
    """
    Replay the old get_recent_turns logic.
    conv_turns: list of {'prompt':..., 'response':...} sorted chronologically.
    turn_index: checkpoint split (0-indexed count of turns in the past).
    Returns estimated token count of the 10 turns before the checkpoint.
    """
    # The checkpoint is at turn_index — everything before it is history
    history = conv_turns[:turn_index]

    # Take the last N turns (or fewer if not enough history)
    recent = history[-SLIDING_WINDOW_N:] if len(history) >= SLIDING_WINDOW_N else history

    total_tokens = 0
    for t in recent:
        raw = f"User: {t['prompt']}\n\nAssistant: {t.get('response', '')}"
        words = raw.split()
        if len(words) > SLIDING_WINDOW_WORD_CAP:
            raw = " ".join(words[:SLIDING_WINDOW_WORD_CAP]) + "…"
        total_tokens += estimate_tokens(raw)
    return total_tokens


def parse_checkpoint(checkpoint_id):
    """Extract (conversation_prefix, turn_number) from 'EC-xxxxxxxx-TURNnnn'."""
    m = re.match(r"EC-([a-f0-9]+)-TURN(\d+)", checkpoint_id)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
print("Loading master results …")
with open(MASTER_RESULTS) as f:
    master = json.load(f)

print("Loading simulation turns …")
conv_turns = defaultdict(list)
prefix_to_full = {}
with open(SIMULATION_INPUT) as f:
    for line in f:
        obj = json.loads(line.strip())
        cid = obj["conversation_id"]
        conv_turns[cid].append(obj)
        prefix_to_full[cid[:8]] = cid

# Sort each conversation by timestamp
for cid in conv_turns:
    conv_turns[cid].sort(key=lambda x: x.get("timestamp", ""))

# ---------------------------------------------------------------------------
# CORRECT
# ---------------------------------------------------------------------------
corrected_count = 0
total_added = 0

for entry in master["evaluation_run_results"]:
    meta = entry.get("metadata", {})
    checkpoint_id = meta.get("checkpoint_id", "")
    prefix, turn_n = parse_checkpoint(checkpoint_id)
    if prefix is None or turn_n is None:
        continue

    full_cid = prefix_to_full.get(prefix)
    if full_cid is None:
        continue

    turns = conv_turns.get(full_cid, [])
    if not turns:
        continue

    sliding_tokens = get_sliding_window_tokens(turns, turn_n)

    for cond_name in list(entry["execution_permutations"].keys()):
        if not cond_name.startswith("full_ice_"):
            continue
        old = entry["execution_permutations"][cond_name].get("tokens_injected", 0)
        new = old + sliding_tokens
        entry["execution_permutations"][cond_name]["tokens_injected"] = new
        corrected_count += 1
        total_added += sliding_tokens

# ---------------------------------------------------------------------------
# SAVE (temporary — verify before replacing original)
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_CORRECTED), exist_ok=True)
with open(OUTPUT_CORRECTED, "w") as f:
    json.dump(master, f, indent=2, ensure_ascii=False)

avg_added = total_added / corrected_count if corrected_count else 0
print(f"Corrected {corrected_count} full‑ICE entries.")
print(f"Average sliding‑window tokens added: {avg_added:.0f}")
print(f"Output: {OUTPUT_CORRECTED}")
print("Verify the numbers, then rename to master_results.json to activate.")