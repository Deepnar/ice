#!/usr/bin/env python3
"""Extract GPT (Flaw) conversations from delimited .txt files, with conversation_id."""

import os, re, json, uuid
from datetime import datetime, timedelta, timezone

RAW_DIR = "data/simulation/raw_chats"
OUTPUT = "data/simulation/gpt.jsonl"

# Map file basename -> (conversation_id, start_timestamp)
# gpt1-3 share the same conversation.
FILE_CONFIG = {
    "gpt1.txt": ("flaw_chat", "2024-01-01T00:00:00Z"),
    "gpt2.txt": ("flaw_chat", None),           # will be auto‑continuation
    "gpt3.txt": ("flaw_chat", None),
    "gpt4.txt": ("gpt4_standalone", "2025-01-15T12:00:00Z"),
    "gpt5.txt": ("gpt5_standalone", "2025-06-01T09:00:00Z"),
    "gpt6.txt": ("gpt6_standalone", "2026-02-01T14:00:00Z"),
}

def extract_turns(text: str) -> list[tuple[str, str]]:
    """Split raw text by 'You said:' and 'ChatGPT said:'."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'(You said:|ChatGPT said:)', text)
    pairs = []
    i = 1
    while i < len(parts) - 1:
        if parts[i] == 'You said:':
            prompt = parts[i+1].strip()
            i += 2
            if i < len(parts) and parts[i] == 'ChatGPT said:':
                response = parts[i+1].strip()
                i += 2
                if prompt and response:
                    pairs.append((prompt, response))
            else:
                pass
        else:
            i += 1
    return pairs

def conv_id_from_name(name: str) -> str:
    """Deterministic UUIDv5 for a conversation name."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    all_lines = []

    # State for continuation of flaw chat
    flaw_conv_id = conv_id_from_name("flaw_chat")
    flaw_last_ts = datetime.fromisoformat("2024-01-01T00:00:00Z")

    for fname in sorted(FILE_CONFIG.keys()):
        cid_name, start_ts = FILE_CONFIG[fname]
        if cid_name == "flaw_chat":
            conv_id = flaw_conv_id
            if start_ts:   # gpt1.txt
                current_ts = datetime.fromisoformat(start_ts)
            else:          # gpt2.txt, gpt3.txt – continue from last timestamp
                current_ts = flaw_last_ts + timedelta(minutes=5)
        else:
            conv_id = conv_id_from_name(cid_name)
            current_ts = datetime.fromisoformat(start_ts)

        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  Skipping missing file: {fpath}")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        pairs = extract_turns(content)
        print(f"  {fname}: {len(pairs)} turns, conv_id={conv_id[:8]}…")

        for prompt, response in pairs:
            ts_str = current_ts.isoformat() + 'Z'
            all_lines.append(json.dumps({
                "prompt": prompt,
                "response": response,
                "timestamp": ts_str,
                "conversation_id": conv_id,
            }, ensure_ascii=False))
            current_ts += timedelta(minutes=5)

        if cid_name == "flaw_chat":
            flaw_last_ts = current_ts

    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for line in all_lines:
            out.write(line + '\n')
    print(f"Saved {len(all_lines)} turns to {OUTPUT}")

if __name__ == "__main__":
    main()