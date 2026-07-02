#!/usr/bin/env python3
"""
Correct tokens_injected for Shinchan probes to include ALL message content
(system prompt, recent turns, retrieved fragments, acknowledgment, question),
not just the first message.

Reads master_results.json and fragments.jsonl, computes honest token counts,
writes a temporary corrected file for verification before replacing the original.
"""

import json
from pathlib import Path
from collections import defaultdict

MATURE_DIR = Path(__file__).parent
RESULTS_DIR = MATURE_DIR / "results"
MASTER_FILE = RESULTS_DIR / "master_results.json"
FRAGMENTS_FILE = RESULTS_DIR / "fragments.jsonl"
OUTPUT_TMP = RESULTS_DIR / "master_results_corrected.json"

SHINCHAN_CID = "633e26f8-5889-5c21-8c70-f4d7ab22cb00"

# ---------------------------------------------------------------------------
# Replicate the unified budget logic (same constants as orchestrator.py)
# ---------------------------------------------------------------------------
TOTAL_CONTEXT_BUDGET = 24_000
OVERHEAD_RESERVE = 1_800

def _compute_recent_fraction(turn_count: int, total_tokens: int, classification: dict) -> float:
    """Copy of orchestrator._compute_recent_fraction."""
    if turn_count < 10:
        base = 0.7
    elif turn_count < 50:
        base = 0.5
    elif turn_count < 200:
        base = 0.3
    elif turn_count < 500:
        base = 0.2
    else:
        base = 0.15

    if turn_count > 0 and total_tokens > 0:
        avg_tokens_per_turn = total_tokens / turn_count
        if avg_tokens_per_turn > 3000:
            base -= 0.15
        elif avg_tokens_per_turn > 1500:
            base -= 0.10
        elif avg_tokens_per_turn > 800:
            base -= 0.05

    modifier = 0.0
    if classification:
        intents = set(classification.get("intent_tags", []))
        if intents & {"Factual_Retrieval", "Troubleshooting", "Analysis_&_Summarization"}:
            modifier -= 0.10
        if intents & {"Emotional_Processing", "Casual_Banter"}:
            modifier += 0.15
        topics = set(classification.get("topic_tags", []))
        if "Creative_&_Media" in topics:
            modifier += 0.10
        if "Software_&_Tech" in topics:
            modifier -= 0.05
        if topics & {"Social_&_Relationships", "Lifestyle_&_Health"}:
            modifier += 0.05

    return max(0.05, min(0.85, base + modifier))

def compute_recent_token_budget(turn_index: int, total_tokens: int, classification: dict) -> int:
    """Return the recent-turns budget that would have been used for this probe."""
    available = TOTAL_CONTEXT_BUDGET - OVERHEAD_RESERVE
    fraction = _compute_recent_fraction(turn_index, total_tokens, classification)
    return int(available * fraction)

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.33)

# ---------------------------------------------------------------------------
# Overhead constants (must match the actual prompts used during Shinchan run)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to the user's personal history. "
    "Use the context below to answer the user's question accurately and directly. "
    "After directly answering, add relevant details from the context that support your answer. "
    "If the context shows that a fact changed over time — for example, a role was reassigned, "
    "a name was updated, or a decision was reversed — mention both the earlier version and "
    "what it changed to. "
    "Do not invent facts. If the context doesn't contain the answer, say so."
)
ACKNOWLEDGMENT = "Understood — I have the background context. What would you like to know?"

SYSTEM_TOKENS = estimate_tokens(SYSTEM_MESSAGE)
ACK_TOKENS = estimate_tokens(ACKNOWLEDGMENT)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(MASTER_FILE, 'r') as f:
    master = json.load(f)

# Build fragment lookup: (probe_id, checkpoint_id, condition) -> [fragment dicts]
fragments_by_probe = defaultdict(list)
if FRAGMENTS_FILE.exists():
    with open(FRAGMENTS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frag = json.loads(line)
            key = (frag["conversation_id"], frag["probe_id"], frag["checkpoint_id"], frag["condition"])
            fragments_by_probe[key].append(frag)

# ---------------------------------------------------------------------------
# Correct each Shinchan probe
# ---------------------------------------------------------------------------
corrected_count = 0
for entry in master["evaluation_run_results"]:
    if entry["conversation_id"] != SHINCHAN_CID:
        continue

    question = entry.get("question", "")
    question_tokens = estimate_tokens(question)
    turn_index = entry.get("turn_index", 0)
    total_tokens_conv = entry.get("total_tokens_in_conversation", 0)

    for cond_name, cond_data in entry.get("conditions", {}).items():
        # Get classification from this condition (same for all conditions, pick first)
        classification = cond_data.get("classification", {})
        recent_budget = compute_recent_token_budget(turn_index, total_tokens_conv, classification)

        # Fragment tokens
        frag_key = (SHINCHAN_CID, entry["probe_id"], entry["checkpoint_id"], cond_name)
        frags = fragments_by_probe.get(frag_key, [])
        frag_tokens = sum(f.get("token_count", 0) for f in frags)
        # If fragment token_count is missing, estimate from text
        if frag_tokens == 0 and frags:
            frag_tokens = sum(estimate_tokens(f.get("text", "")) for f in frags)

        # Did this probe have context? (fragments or recent turns)
        has_context = (frag_tokens > 0) or (recent_budget > 0)

        # Total = system + recent budget + fragments + ack (if context) + question
        total = SYSTEM_TOKENS + recent_budget + frag_tokens + question_tokens
        if has_context:
            total += ACK_TOKENS

        # Also add persistent context tokens (memory slots) — they are empty in Shinchan, so zero.
        # Add cluster header tokens if any clusters were used; we don't know without replaying,
        # but cluster headers are tiny (~10-20 tokens) and won't change the numbers materially.

        old_tokens = cond_data.get("tokens_injected", 0)
        cond_data["tokens_injected"] = total
        corrected_count += 1
        if abs(old_tokens - total) > 500:   # only print significant changes
            print(f"  {entry['probe_id']} @ {entry['checkpoint_id']} [{cond_name}]: {old_tokens} → {total}")

# ---------------------------------------------------------------------------
# Write corrected file (NOT overwriting original)
# ---------------------------------------------------------------------------
with open(OUTPUT_TMP, 'w') as f:
    json.dump(master, f, indent=2, ensure_ascii=False)

print(f"\nCorrected {corrected_count} token counts.")
print(f"Temporary file: {OUTPUT_TMP}")
print("Verify the numbers, then rename it to master_results.json to activate.")