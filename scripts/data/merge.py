#!/usr/bin/env python3
"""Merge multiple JSONL sources into one chronologically sorted, deduplicated file,
   preserving the conversation_id field."""

import json, os

FILES = [
    "data/simulation/gpt.jsonl",
    "data/simulation/claude.jsonl",
    "data/simulation/deepseek.jsonl",
]
OUTPUT = "data/simulation/simulation_full.jsonl"

def main():
    all_turns = []
    seen_prompts = set()

    for fpath in FILES:
        if not os.path.exists(fpath):
            print(f"  Missing: {fpath} — skipping")
            continue
        count = 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                if prompt and prompt not in seen_prompts:
                    seen_prompts.add(prompt)
                    all_turns.append(obj)
                    count += 1
        print(f"  {fpath}: added {count} turns")

    all_turns.sort(key=lambda x: x.get("timestamp", ""))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for turn in all_turns:
            out.write(json.dumps(turn, ensure_ascii=False) + '\n')

    print(f"\nMerged {len(all_turns)} unique turns → {OUTPUT}")

if __name__ == "__main__":
    main()