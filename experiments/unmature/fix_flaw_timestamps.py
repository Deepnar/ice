#!/usr/bin/env python3
"""Shift the Flaw conversation (bb558b5f-5365-5bac-9ed0-07219025b5f2) to start in 2025."""

import json, re
from datetime import datetime, timezone, timedelta

INPUT = "data/simulation/simulation_full.jsonl"
OUTPUT = "data/simulation/simulation_full.jsonl"
FLAW_CID = "bb558b5f-5365-5bac-9ed0-07219025b5f2"

def parse_timestamp(ts_str: str) -> datetime:
    """
    Parse any ISO‑8601 timestamp string that may have:
      - trailing 'Z'
      - a numeric offset like +00:00 or -05:00
      - milliseconds or microseconds
    Returns a timezone‑aware datetime.
    """
    ts = ts_str.strip()
    # Remove trailing 'Z' → replace with +00:00 only if no offset already present
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    # If the string already contains an offset (e.g., ends with +HH:MM or -HH:MM),
    # Python 3.11's fromisoformat can handle it.
    return datetime.fromisoformat(ts)

def main():
    # Read all turns
    with open(INPUT, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Find Flaw turns
    flaw_turns = [t for t in lines if t.get("conversation_id") == FLAW_CID]
    if not flaw_turns:
        print("No Flaw turns found.")
        return

    # Parse the earliest Flaw timestamp
    first_ts = parse_timestamp(flaw_turns[0]["timestamp"])
    # Target: January 1, 2025, same time of day as the original first turn
    new_start = first_ts.replace(year=2025, month=1, day=1)
    delta = new_start - first_ts

    # Shift all Flaw turns by the same delta
    for turn in lines:
        if turn.get("conversation_id") == FLAW_CID:
            ts = parse_timestamp(turn["timestamp"])
            new_ts = ts + delta
            turn["timestamp"] = new_ts.isoformat()

    # Write back
    with open(OUTPUT, "w") as f:
        for turn in lines:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    print(f"Updated Flaw timestamps. File saved to {OUTPUT}")

if __name__ == "__main__":
    main()