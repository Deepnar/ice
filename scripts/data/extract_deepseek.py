#!/usr/bin/env python3
"""Extract DeepSeek conversations with conversation_id."""

import json, os

INPUT = "data/simulation/raw_chats/deepseek.json"
OUTPUT = "data/simulation/deepseek.jsonl"

def extract_message_text(node: dict) -> str | None:
    msg = node.get("message")
    if msg is None:
        return None
    fragments = msg.get("fragments", [])
    parts = []
    for frag in fragments:
        content = frag.get("content", "")
        if content:
            parts.append(content)
    return "\n".join(parts).strip() if parts else None

def extract_conversation(conv: dict) -> list[dict]:
    """Flatten the mapping tree into a linear list of messages with role & timestamp."""
    mapping = conv.get("mapping", {})
    nodes = {nid: n for nid, n in mapping.items()}
    visited = set()
    stack = ["root"]
    ordered_msgs = []

    while stack:
        nid = stack.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = nodes.get(nid)
        if node is None:
            continue
        text = extract_message_text(node)
        if text:
            msg_data = node.get("message", {})
            frag_types = [f.get("type") for f in msg_data.get("fragments", [])]
            if "REQUEST" in frag_types:
                role = "user"
            elif "RESPONSE" in frag_types:
                role = "assistant"
            else:
                role = "unknown"
            ordered_msgs.append({
                "role": role,
                "text": text,
                "inserted_at": msg_data.get("inserted_at")
            })
        for child_id in node.get("children", []):
            if child_id not in visited:
                stack.append(child_id)
    return ordered_msgs

def main():
    with open(INPUT, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    all_lines = []
    for conv in conversations:
        conv_id = conv.get("id", "")
        msgs = extract_conversation(conv)
        i = 0
        while i < len(msgs) - 1:
            if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                prompt = msgs[i]["text"]
                response = msgs[i+1]["text"]
                ts = msgs[i]["inserted_at"]
                if prompt and response:
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