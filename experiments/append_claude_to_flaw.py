#!/usr/bin/env python3
"""Append Claude Flaw turns to the GPT Flaw conversation in simulation_full.jsonl."""

import json
from datetime import datetime, timedelta, timezone

SIM_INPUT = "data/simulation/simulation_full.jsonl"
GPT_FLAW_ID = "bb558b5f-5365-5bac-9ed0-07219025b5f2"
CLAUDE_FLAW_ID = "cca73c87-2068-4211-ab51-f38b6e966b0a"

def main():
    with open(SIM_INPUT, "r") as f:
        all_turns = [json.loads(line) for line in f if line.strip()]

    # Find existing GPT Flaw turns and the latest timestamp
    gpt_turns = [t for t in all_turns if t.get("conversation_id") == GPT_FLAW_ID]
    gpt_turns.sort(key=lambda x: x["timestamp"])
    if not gpt_turns:
        print("No GPT Flaw turns found – nothing to do.")
        return
    last_ts = datetime.fromisoformat(gpt_turns[-1]["timestamp"].replace("Z", "+00:00"))

    # Find Claude Flaw turns and copy them with new ID + shifted timestamps
    claude_turns = [t for t in all_turns if t.get("conversation_id") == CLAUDE_FLAW_ID]
    claude_turns.sort(key=lambda x: x["timestamp"])
    if not claude_turns:
        print("No Claude Flaw turns found – nothing to do.")
        return

    new_turns = []
    current_ts = last_ts + timedelta(minutes=5)
    for turn in claude_turns:
        new = dict(turn)
        new["conversation_id"] = GPT_FLAW_ID
        new["timestamp"] = current_ts.isoformat()
        new_turns.append(new)
        current_ts += timedelta(minutes=5)

    with open(SIM_INPUT, "a") as f:
        for t in new_turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Appended {len(new_turns)} Claude Flaw turns as GPT Flaw turns.")
    print(f"The GPT Flaw conversation now has {len(gpt_turns) + len(new_turns)} total turns.")

if __name__ == "__main__":
    main()