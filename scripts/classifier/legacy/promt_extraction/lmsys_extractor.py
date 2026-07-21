#!/usr/bin/env python3
"""
LMSYS Chatbot Arena prompt extractor.
Source: lmsys/chatbot_arena_conversations on Hugging Face (GATED).
"""

import json
import hashlib
import os
from datasets import load_dataset

OUTPUT_FILE  = "lmsys_prompts.jsonl"
TARGET_COUNT = 5000

hf_token = os.environ.get("HF_TOKEN", None)

print("Loading LMSYS dataset...")
dataset = load_dataset(
    "lmsys/chatbot_arena_conversations",
    split="train",
    token=hf_token,
)

seen_hashes = set()
results     = []
counter     = 1

for row in dataset:
    if len(results) >= TARGET_COUNT:
        break

    if row.get("language") != "English":
        continue

    conversation = row.get("conversation_a", [])
    if not conversation:
        continue

    first_user = next(
        (turn["content"] for turn in conversation if turn["role"] == "user"),  # fixed: "user" not "human"
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
        "id":     f"lmsys_{counter:04d}",
        "source": "lmsys",
        "prompt": first_user.strip(),
        "label":  None,
    })
    counter += 1

    if counter % 500 == 0:
        print(f"  {len(results)} collected...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in results:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Done. {len(results)} LMSYS prompts saved to {OUTPUT_FILE}")