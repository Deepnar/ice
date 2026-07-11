#!/usr/bin/env python3
"""Replay buffered post-flight events from the local JSONL file.

DEAD SINCE C7 (2026-07-11): the jsonl buffer fallback was deleted with the
Celery broker (spec D8 — an in-process enqueue's only failure mode is the app
being down, in which case the turn wasn't stored either), so nothing writes
data/post_flight_buffer.jsonl anymore and this replayer has nothing to replay.
Kept per the never-delete-scripts rule; `.delay` below is the old Celery API.
"""

import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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