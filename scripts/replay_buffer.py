#!/usr/bin/env python3
"""Replay buffered post‑flight events from local JSONL file when Redis is back."""

import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workers.post_flight import evaluate_turn

BUFFER_FILE = "data/post_flight_buffer.jsonl"

def main():
    if not os.path.exists(BUFFER_FILE):
        print("No buffer file found.")
        return
    with open(BUFFER_FILE, "r") as f:
        lines = f.readlines()
    print(f"Replaying {len(lines)} buffered events...")
    for line in lines:
        entry = json.loads(line)
        evaluate_turn.delay(
            batch_id=entry["batch_id"],
            prompt=entry["prompt"],
            response=entry["response"],
            conversation_id=entry["conversation_id"]
        )
    os.remove(BUFFER_FILE)
    print("Buffer replayed and cleared.")

if __name__ == "__main__":
    main()