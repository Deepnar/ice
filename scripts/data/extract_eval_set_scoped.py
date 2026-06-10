#!/usr/bin/env python3
"""
Create a scoped held‑out evaluation set.
Turns are grouped by conversation_id, windows are built inside each conversation,
and test prompts are generated proportionally to conversation size.
"""

import json, os, random, hashlib
from collections import defaultdict
from openai import OpenAI

SIMULATION_INPUT = "/home/deepnar/Programs/ice/data/simulation/simulation_full.jsonl"
OUTPUT = "/home/deepnar/Programs/ice/data/simulation/held_out_set.jsonl"
WINDOW_SIZE = 8
TOTAL_PROBES = 200
SEED = 42
MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
BG_API = "http://localhost:8002/v1"

client = OpenAI(base_url=BG_API, api_key="dummy")

SYSTEM_PROMPT = (
    "You are helping evaluate a long‑term memory system for an AI assistant.\n"
    "You will be given a block of consecutive conversation turns between a user and an AI.\n"
    "Generate a natural user question that someone might ask the AI assistant days or weeks later,\n"
    "which would require remembering the overall topic or decisions discussed in this block.\n"
    "The question should NOT ask for a specific fact that appears in a single line, "
    "but instead require understanding the gist of what was discussed.\n"
    "Examples of good questions:\n"
    "- 'What did we decide about the laptop last month?'\n"
    "- 'What was the problem with the timetabling algorithm we were debugging?'\n"
    "- 'Can you summarise the lore we developed for the second protagonist?'\n"
    "The question must be directed at the AI assistant (use 'you' or 'we'), NOT at the user.\n"
    "Output ONLY the question, nothing else."
)


def generate_question(block_text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": block_text},
            ],
            temperature=0.7,
            max_tokens=120,
            timeout=30.0,
        )
        q = resp.choices[0].message.content.strip()
        if q.startswith('"') and q.endswith('"'):
            q = q[1:-1].strip()
        return q
    except Exception as e:
        print(f"  Model error: {e}")
        return ""


def main():
    random.seed(SEED)

    # 1. Load and group by conversation_id
    with open(SIMULATION_INPUT, "r", encoding="utf-8") as f:
        all_turns = [json.loads(line) for line in f if line.strip()]

    # Remove template fragments
    all_turns = [t for t in all_turns if "{{" not in t.get("prompt", "") and "{%" not in t.get("prompt", "")]
    print(f"Total clean turns: {len(all_turns)}")

    conv_groups = defaultdict(list)
    for turn in all_turns:
        cid = turn.get("conversation_id", "unknown")
        conv_groups[cid].append(turn)

    print(f"Number of conversations: {len(conv_groups)}")

    # 2. Build windows inside each conversation
    conv_windows = {}   # conversation_id -> list of (start_idx, window_turns)
    for cid, turns in conv_groups.items():
        if len(turns) < WINDOW_SIZE:
            continue
        windows = []
        for i in range(0, len(turns) - WINDOW_SIZE + 1, WINDOW_SIZE // 2):  # overlapping
            window_turns = turns[i : i + WINDOW_SIZE]
            avg_len = sum(len(t.get("response", "")) for t in window_turns) / len(window_turns)
            if avg_len > 100:
                windows.append((i, window_turns))
        if windows:
            conv_windows[cid] = windows

    # 3. Allocate probes proportionally to conversation size
    total_turns = sum(len(conv_groups[cid]) for cid in conv_windows)
    probes_per_conv = {}
    for cid in conv_windows:
        proportion = len(conv_groups[cid]) / total_turns
        probes_per_conv[cid] = max(1, int(TOTAL_PROBES * proportion))

    # Ensure total doesn't exceed TOTAL_PROBES due to rounding
    allocated = sum(probes_per_conv.values())
    while allocated > TOTAL_PROBES:
        # Subtract from the conversation with most allocated
        max_cid = max(probes_per_conv, key=probes_per_conv.get)
        probes_per_conv[max_cid] -= 1
        allocated -= 1

    # 4. Generate questions
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    total_written = 0
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for cid, windows in conv_windows.items():
            n_probes = probes_per_conv.get(cid, 0)
            if n_probes <= 0:
                continue
            # Sample windows from this conversation
            sampled = random.sample(windows, min(n_probes, len(windows)))
            for start_idx, turns in sampled:
                # Build block text
                block_text = ""
                for t in turns:
                    user = t["prompt"][:300]
                    assistant = t["response"][:300]
                    block_text += f"User: {user}\nAssistant: {assistant}\n\n"
                if len(block_text) > 2500:
                    block_text = block_text[:2500] + "..."

                question = generate_question(block_text)
                if not question:
                    continue

                window_raw = "".join(t["prompt"] + t["response"] for t in turns)
                window_hash = hashlib.sha256(window_raw.encode()).hexdigest()

                entry = {
                    "test_prompt": question,
                    "conversation_id": cid,
                    "relevant_window_hash": window_hash,
                    "window_start_idx": start_idx,
                    "original_timestamps": [t.get("timestamp") for t in turns],
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_written += 1
                if total_written % 20 == 0:
                    print(f"  {total_written} probes written...")

    print(f"Saved {total_written} test probes → {OUTPUT}")


if __name__ == "__main__":
    main()