#!/usr/bin/env python3
"""
WildChat prompt extractor.
Source: allenai/WildChat-1M on Hugging Face (public).

Uses streaming=True — dataset is 1M rows, don't load it all into RAM.
Extracts first human turn from English conversations only.
Outputs JSONL matching the ICE training data format.
"""

import json
import hashlib
from datasets import load_dataset

OUTPUT_FILE  = "wildchat_prompts.jsonl"
TARGET_COUNT = 5000

print("Loading WildChat dataset (streaming)...")
dataset = load_dataset(
    "allenai/WildChat-1M",
    split="train",
    streaming=True,
)

seen_hashes = set()
results     = []
counter     = 1

for row in dataset:
    if len(results) >= TARGET_COUNT:
        break

    if row.get("language") != "English":
        continue

    conversation = row.get("conversation", [])
    if not conversation:
        continue

    first_user = next(
        (turn["content"] for turn in conversation if turn["role"] == "user"),
        None,
    )
    if not first_user or not first_user.strip():
        continue
    if len(first_user.strip()) < 10:
        continue

    content_hash = hashlib.sha256(first_user.strip().encode()).hexdigest()
    if content_hash in seen_hashes:
        continue

    seen_hashes.add(content_hash)
    results.append({
        "id":     f"wildchat_{counter:04d}",
        "source": "wildchat",
        "prompt": first_user.strip(),
        "label":  None,
    })
    counter += 1

    if counter % 500 == 0:
        print(f"  {len(results)} collected...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in results:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Done. {len(results)} WildChat prompts saved to {OUTPUT_FILE}")