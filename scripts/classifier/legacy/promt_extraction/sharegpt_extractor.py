#!/usr/bin/env python3
"""
ShareGPT prompt extractor.
Source: anon8231489123/ShareGPT_Vicuna_unfiltered on Hugging Face (public).

No language field — uses ASCII ratio as English proxy.
Extracts first human turn only.
Outputs JSONL matching the ICE training data format.
"""

import json
import hashlib
from datasets import load_dataset

OUTPUT_FILE  = "sharegpt_prompts.jsonl"
TARGET_COUNT = 5000


def is_likely_english(text: str) -> bool:
    """True if >85% of characters are ASCII — reasonable English proxy."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) > 0.85


print("Loading ShareGPT dataset...")
dataset = load_dataset(
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
    split="train",
)

seen_hashes = set()
results     = []
counter     = 1

for row in dataset:
    if len(results) >= TARGET_COUNT:
        break

    conversations = row.get("conversations", [])
    if not conversations:
        continue

    first_human = next(
        (turn["value"] for turn in conversations if turn.get("from") == "human"),
        None,
    )
    if not first_human or not first_human.strip():
        continue
    if len(first_human.strip()) < 10:
        continue
    if not is_likely_english(first_human):
        continue

    content_hash = hashlib.sha256(first_human.strip().encode()).hexdigest()
    if content_hash in seen_hashes:
        continue

    seen_hashes.add(content_hash)
    results.append({
        "id":     f"sharegpt_{counter:04d}",
        "source": "sharegpt",
        "prompt": first_human.strip(),
        "label":  None,
    })
    counter += 1

    if counter % 500 == 0:
        print(f"  {len(results)} collected...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in results:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Done. {len(results)} ShareGPT prompts saved to {OUTPUT_FILE}")