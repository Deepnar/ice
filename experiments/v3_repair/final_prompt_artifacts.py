"""Validate native final-block receipts; count exposure, never answer support."""

import hashlib
import json

POLICY = "ice-v3-native-final-blocks-1"
HEADERS = {
    "slots": "\n\n=== PERSISTENT CONTEXT ===\n",
    "bookmarks": "\n\n=== BOOKMARKED BY THE USER ===\n",
    "conversation_summary": "\n\n=== CONVERSATION SUMMARY ===\n",
    "constraints": "\n\n=== PROJECT CONSTRAINTS ===\n",
    "session_start": "\n\n=== PROJECT SESSION START ===\n",
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def references(item):
    values = item.get("stored_artifact_ids")
    if not isinstance(values, (list, tuple)):
        raise ValueError("native final-block references missing")
    for value in values:
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(not isinstance(v, str) or not v for v in value)):
            raise ValueError("native final-block reference malformed")
    return {tuple(value) for value in values}


def observe(stage):
    if "prompt_artifact_blocks" not in stage:
        return None
    blocks = stage["prompt_artifact_blocks"]
    messages = stage.get("prompt_messages")
    if (stage.get("prompt_artifact_policy") != POLICY or not isinstance(blocks, dict)
            or set(blocks) - (set(HEADERS) | {"recent_turns"})
            or "recent_turns" not in blocks or not isinstance(messages, list)
            or len(messages) < 2 or messages[0].get("role") != "system"
            or messages[-1].get("role") != "user"
            or any(not isinstance(m.get("content"), str) for m in messages)):
        raise ValueError("native final-block policy/messages incomplete")
    history_end = len(messages) - 1
    if stage["context_ledger"]["blocks"]["evidence"] > 0:
        if (len(messages) < 4 or messages[-3].get("role") != "user"
                or not messages[-3]["content"].startswith("=== RETRIEVED CONTEXT ===")
                or messages[-2].get("role") != "assistant"
                or messages[-2]["content"] != "Understood — I have the background context. What would you like to know?"):
            raise ValueError("native final-block evidence boundary differs")
        history_end -= 2
    used_spans, recent_indices, result = [], [], {}
    for name, block in blocks.items():
        if not isinstance(block, dict) or not isinstance(block.get("items"), list):
            raise ValueError("native final-block items missing")
        if name in stage.get("evicted", []) and block["items"]:
            raise ValueError("removed native block still credits artifacts")
        if name != "recent_turns":
            start, end = block.get("char_start"), block.get("char_end")
            if (block.get("message_index") != 0 or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(messages[0]["content"])
                    or any(start < b and a < end for a, b in used_spans)):
                raise ValueError("native final-block system span differs")
            body = messages[0]["content"][start:end]
            if not body.startswith(HEADERS[name]) or sha(body) != block.get("rendered_text_sha256"):
                raise ValueError("native final-block system body differs")
            used_spans.append((start, end))
        refs, versions, unmapped = set(), set(), 0
        for item in block["items"]:
            item_refs = references(item)
            if not item_refs:
                unmapped += 1
            if name == "recent_turns":
                indices = item.get("message_indices")
                if (not isinstance(indices, list) or len(indices) != item.get("message_count")
                        or not indices or any(type(i) is not int or not 1 <= i < history_end for i in indices)
                        or indices != list(range(indices[0], indices[0] + len(indices)))):
                    raise ValueError("native recent receipt reaches outside history")
                selected = [messages[i] for i in indices]
                if [m.get("role") for m in selected] not in (["user"], ["user", "assistant"]):
                    raise ValueError("native recent role order differs")
                version = sha(json.dumps(selected, sort_keys=True, ensure_ascii=False))
                if version != item.get("rendered_messages_sha256"):
                    raise ValueError("native recent body hash differs")
                recent_indices.extend(indices)
            else:
                version = item.get("rendered_text_sha256", block["rendered_text_sha256"])
                if (not isinstance(version, str) or len(version) != 64
                        or any(c not in "0123456789abcdef" for c in version)):
                    raise ValueError("native final-block item hash malformed")
            refs.update(item_refs)
            versions.update((ref, version) for ref in item_refs)
        result[name] = dict(refs=refs, versions=versions, items=len(block["items"]),
                            unmapped_items=unmapped,
                            visibility="verified_role_messages" if name == "recent_turns"
                            else "native_items_in_verified_system_container")
    if recent_indices != list(range(1, history_end)):
        raise ValueError("native recent receipts omit/duplicate/reorder historical messages")
    for name in HEADERS:
        cost = stage["context_ledger"]["blocks"].get(name)
        if cost is not None and (cost > 0) != (name in blocks):
            raise ValueError("native final-block ledger presence differs")
    for name in ("slots", "bookmarks"):
        count = stage.get("prompt_block_counts", {}).get(name)
        if count is not None and count != result.get(name, {}).get("items", 0):
            raise ValueError("native final-block item count differs")
    return result
