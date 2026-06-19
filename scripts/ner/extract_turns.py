#!/usr/bin/env python3
"""Combine all prompt sources into one raw_turns.jsonl for NER labelling.

Sources:
1. simulation_full.jsonl (prompt + response) → raw_text = "User: {prompt}\n\nAssistant: {response}"
2. labeled_prompts.jsonl (prompt only) → raw_text = "User: {prompt}"
3. synthetic_prompts_renumbered_labeled.jsonl (prompt only) → raw_text = "User: {prompt}"
"""

import json
import os

# ---- inputs ----
SIMULATION = "data/simulation/simulation_full.jsonl"
LABELED = "data/labeled/labeled_prompts.jsonl"
SYNTHETIC = "data/synthetic/synthetic_prompts_renumbered_labeled.jsonl"

OUTPUT = "data/ner/raw_turns.jsonl"

# ---- helpers ----
def read_simulation(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt", "")
            response = obj.get("response", "")
            if not prompt or not response:
                continue
            yield prompt, response, obj.get("conversation_id", ""), obj.get("timestamp", "")

def read_prompt_only(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt", "")
            if not prompt:
                continue
            yield prompt, None, "", ""

# ---- main ----
def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    total_est = 0
    for p in [SIMULATION, LABELED, SYNTHETIC]:
        try:
            with open(p, "r") as f:
                total_est += sum(1 for _ in f)
        except FileNotFoundError:
            print(f"Warning: {p} not found, skipping.")
            continue

    print(f"Estimated total records: {total_est}")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        idx = 1

        # 1. Simulation
        for prompt, response, conv_id, ts in read_simulation(SIMULATION):
            raw_text = f"User: {prompt}\n\nAssistant: {response}"
            if len(raw_text) < 50:
                continue
            record = {
                "id": f"ner_sim_{idx:05d}",
                "raw_text": raw_text,
                "prompt": prompt,
                "response": response or "",
                "conversation_id": conv_id,
                "timestamp": ts
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            idx += 1

        # 2. Labeled
        for prompt, _, _, _ in read_prompt_only(LABELED):
            raw_text = f"User: {prompt}"
            if len(raw_text) < 10:
                continue
            record = {
                "id": f"ner_label_{idx:05d}",
                "raw_text": raw_text,
                "prompt": prompt,
                "response": "",
                "conversation_id": "labeled",
                "timestamp": ""
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            idx += 1

        # 3. Synthetic
        for prompt, _, _, _ in read_prompt_only(SYNTHETIC):
            raw_text = f"User: {prompt}"
            if len(raw_text) < 10:
                continue
            record = {
                "id": f"ner_synth_{idx:05d}",
                "raw_text": raw_text,
                "prompt": prompt,
                "response": "",
                "conversation_id": "synthetic",
                "timestamp": ""
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            idx += 1

    print(f"Written {idx-1} records to {OUTPUT}")

if __name__ == "__main__":
    main()