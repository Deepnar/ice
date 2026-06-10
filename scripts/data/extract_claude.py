#!/usr/bin/env python3
"""Extract Claude conversations with conversation_id."""

import json, os

INPUT = "data/simulation/raw_chats/claude.json"
OUTPUT = "data/simulation/claude.jsonl"

def extract_content(msg: dict) -> str:
    """Concatenate all text blocks from a message's content array."""
    texts = []
    for block in msg.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
    return " ".join(texts).strip()

def main():
    with open(INPUT, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    all_lines = []
    for conv in conversations:
        conv_id = conv.get("uuid", "")
        msgs = conv.get("chat_messages", [])
        i = 0
        while i < len(msgs) - 1:
            if msgs[i].get("sender") == "human" and msgs[i+1].get("sender") == "assistant":
                prompt = extract_content(msgs[i])
                response = extract_content(msgs[i+1])
                if prompt and response:
                    ts = msgs[i].get("created_at", "")
                    all_lines.append(json.dumps({
                        "prompt": prompt,
                        "response": response,
                        "timestamp": ts,
                        "conversation_id": conv_id,
                    }, ensure_ascii=False))
                i += 2
            else:
                i += 1

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for line in all_lines:
            out.write(line + '\n')
    print(f"Saved {len(all_lines)} turns to {OUTPUT}")

if __name__ == "__main__":
    main()