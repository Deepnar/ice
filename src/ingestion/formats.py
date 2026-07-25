"""F10 D2: format adapters — provider exports → NormalizedConversation.

Four supported shapes (spec rev 4–7), each a pure function unit-tested
against committed synthetic fixtures (`tests/fixtures/ingestion/`):

  * ChatGPT ``conversations.json`` — array of conversations with a
    ``mapping`` node tree and ``current_node``; the current path is walked
    parent-ward from ``current_node`` (other branches counted, not
    imported — B6 territory).
  * Claude export ``conversations.json`` — array with flat
    ``chat_messages`` that INCLUDES abandoned edit-branches; the current
    path is the parent-chain walked up from the latest-created message
    (the globally-latest message is necessarily a leaf of the live path).
  * DeepSeek export — ChatGPT-style mapping tree but with NO
    ``current_node`` and content in ``fragments`` (REQUEST/RESPONSE carry
    the dialogue; THINK/SEARCH/TOOL_* are dropped). At a branch the child
    whose subtree contains the latest ``inserted_at`` wins.
  * Generic JSONL — one ``{"role", "content", "timestamp"[, "conversation",
    "title"]}`` object per line; lines group by ``conversation`` when
    present, else the file is one conversation. The documented escape
    hatch for anything else.

Timestamps are parsed to aware-UTC datetimes. Conversations with missing/
garbage timestamps get a per-conversation monotonic synthesis from whatever
anchors exist (flagged ``ts_synthesized`` for the report — T-honesty).
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

logger = structlog.get_logger("ice.ingestion.formats")

SUPPORTED_FORMATS = ("chatgpt", "claude", "deepseek", "jsonl", "raw")

# Claude's fixed root parent uuid (messages whose parent is unknown also
# terminate the walk, so this is an optimization, not a load-bearing value).
_CLAUDE_ROOT_UUID = "00000000-0000-4000-8000-000000000000"


@dataclass
class NormalizedTurn:
    """One message. The engine merges consecutive same-role messages and
    pairs user→assistant into stored turns (spec rev 13)."""
    role: str                      # "user" | "assistant"
    text: str
    ts: Optional[datetime] = None


@dataclass
class NormalizedConversation:
    provider: str                  # chatgpt | claude | deepseek | jsonl | raw
    source_id: str                 # provider's conversation id (or derived)
    title: str
    turns: list = field(default_factory=list)   # [NormalizedTurn]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    ts_synthetic: bool = False     # F14: ALL times synthesized (mtime-based)
    ts_synthesized: bool = False   # some times invented from anchors (report)
    branch_messages_skipped: int = 0


def content_hash(conv: NormalizedConversation) -> str:
    """D4: the conversation's idempotency identity. Over normalized content
    only — a re-export whose conversation gained turns hashes differently,
    re-runs, and its unchanged turn prefix dedupes on per-turn keys."""
    payload = {
        "title": conv.title,
        "turns": [[t.role, t.text, t.ts.isoformat() if t.ts else None]
                  for t in conv.turns],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── timestamp helpers ───────────────────────────────────────────────────────

def _parse_ts(value) -> Optional[datetime]:
    """ISO string or unix float/int → aware UTC datetime; None/garbage → None."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            if value <= 0:
                return None
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return None


def _synthesize_timestamps(conv: NormalizedConversation) -> None:
    """Fill missing turn timestamps monotonically from whatever anchors exist
    (edge case in §3): forward-fill +1 min from the previous known time,
    back-fill −1 min before the first known time; with no anchors at all,
    end-anchor at conversation updated_at/created_at (else leave for the
    engine's import-time fallback). Real-but-inverted times are clamped to
    keep dialogue order monotonic without flagging."""
    turns = conv.turns
    if not turns:
        return
    known = [i for i, t in enumerate(turns) if t.ts is not None]
    if not known:
        anchor = conv.updated_at or conv.created_at
        if anchor is None:
            anchor = datetime.now(timezone.utc)
        # End-anchored: the export's conversation stamp marks when it ended.
        for i, t in enumerate(turns):
            t.ts = anchor - timedelta(minutes=(len(turns) - 1 - i))
        conv.ts_synthesized = True
        return
    if len(known) < len(turns):
        first = known[0]
        for i in range(first - 1, -1, -1):
            turns[i].ts = turns[i + 1].ts - timedelta(minutes=1)
        last_ts = turns[first].ts
        for i in range(first + 1, len(turns)):
            if turns[i].ts is None:
                turns[i].ts = last_ts + timedelta(minutes=1)
            last_ts = turns[i].ts
        conv.ts_synthesized = True
    # Monotonic clamp (dialogue order is authoritative; sessions need
    # non-decreasing times).
    for i in range(1, len(turns)):
        if turns[i].ts < turns[i - 1].ts:
            turns[i].ts = turns[i - 1].ts + timedelta(seconds=1)


# ── ChatGPT ─────────────────────────────────────────────────────────────────

def _chatgpt_message_text(message: dict) -> str:
    content = message.get("content") or {}
    parts = content.get("parts") or []
    texts = [p for p in parts if isinstance(p, str) and p.strip()]
    return "\n\n".join(t.strip() for t in texts)


def parse_chatgpt(data: list) -> list:
    """ChatGPT ``conversations.json``: mapping tree flattened along the
    ``current_node`` parent chain (D2)."""
    out = []
    for conv in data:
        mapping = conv.get("mapping") or {}
        node_id = conv.get("current_node")
        path = []
        seen = set()
        while node_id and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            path.append(mapping[node_id])
            node_id = mapping[node_id].get("parent")
        path.reverse()

        turns = []
        for node in path:
            message = node.get("message")
            if not message:
                continue
            role = ((message.get("author") or {}).get("role") or "").lower()
            if role not in ("user", "assistant"):
                continue
            if (message.get("metadata") or {}).get(
                    "is_visually_hidden_from_conversation"):
                continue
            text = _chatgpt_message_text(message)
            if not text:
                continue
            turns.append(NormalizedTurn(
                role=role, text=text, ts=_parse_ts(message.get("create_time"))))

        total_messages = sum(
            1 for n in mapping.values()
            if n.get("message")
            and ((n["message"].get("author") or {}).get("role") or "").lower()
            in ("user", "assistant")
            and _chatgpt_message_text(n["message"]))
        norm = NormalizedConversation(
            provider="chatgpt",
            source_id=str(conv.get("conversation_id") or conv.get("id")
                          or conv.get("title") or len(out)),
            title=(conv.get("title") or "Untitled").strip() or "Untitled",
            turns=turns,
            created_at=_parse_ts(conv.get("create_time")),
            updated_at=_parse_ts(conv.get("update_time")),
            branch_messages_skipped=max(0, total_messages - len(turns)),
        )
        _synthesize_timestamps(norm)
        out.append(norm)
    return out


# ── Claude ──────────────────────────────────────────────────────────────────

def _claude_message_text(msg: dict) -> str:
    """The flat ``text`` field renders fuller than the block join (verified
    on a real export) — blocks are the fallback for empty text (rev 6)."""
    text = (msg.get("text") or "").strip()
    if text:
        return text
    blocks = [b.get("text", "").strip()
              for b in (msg.get("content") or [])
              if isinstance(b, dict) and b.get("type") == "text"]
    return "\n\n".join(b for b in blocks if b)


def parse_claude(data: list) -> list:
    """Claude export: ``chat_messages`` contains ALL messages including
    abandoned edit-branches — the current path is the parent chain of the
    latest-created message (rev 6)."""
    out = []
    for conv in data:
        messages = conv.get("chat_messages") or []
        by_uuid = {m.get("uuid"): m for m in messages if m.get("uuid")}
        path = []
        if messages:
            def _created(m):
                return _parse_ts(m.get("created_at")) or datetime.min.replace(
                    tzinfo=timezone.utc)
            leaf = max(messages, key=_created)
            seen = set()
            m = leaf
            while m is not None and m.get("uuid") not in seen:
                seen.add(m.get("uuid"))
                path.append(m)
                parent = m.get("parent_message_uuid")
                m = by_uuid.get(parent) if parent and \
                    parent != _CLAUDE_ROOT_UUID else None
            path.reverse()

        turns = []
        for msg in path:
            sender = (msg.get("sender") or "").lower()
            role = {"human": "user", "assistant": "assistant"}.get(sender)
            if role is None:
                continue
            text = _claude_message_text(msg)
            if not text:
                continue    # tool/attachment-only messages (430/1472 real)
            turns.append(NormalizedTurn(
                role=role, text=text, ts=_parse_ts(msg.get("created_at"))))

        total_messages = sum(
            1 for m in messages
            if (m.get("sender") or "").lower() in ("human", "assistant")
            and _claude_message_text(m))
        norm = NormalizedConversation(
            provider="claude",
            source_id=str(conv.get("uuid") or conv.get("name") or len(out)),
            title=(conv.get("name") or "Untitled").strip() or "Untitled",
            turns=turns,
            created_at=_parse_ts(conv.get("created_at")),
            updated_at=_parse_ts(conv.get("updated_at")),
            branch_messages_skipped=max(0, total_messages - len(turns)),
        )
        _synthesize_timestamps(norm)
        out.append(norm)
    return out


# ── DeepSeek ────────────────────────────────────────────────────────────────

def _deepseek_subtree_latest(mapping: dict, node_id: str, memo: dict) -> str:
    """Max ``inserted_at`` (ISO string; lexicographic-comparable after UTC
    normalization is unnecessary for ordering within one export) in the
    subtree rooted at node_id."""
    if node_id in memo:
        return memo[node_id]
    node = mapping.get(node_id) or {}
    best = ""
    message = node.get("message")
    if message and message.get("inserted_at"):
        ts = _parse_ts(message["inserted_at"])
        best = ts.isoformat() if ts else ""
    for child in node.get("children") or []:
        child_best = _deepseek_subtree_latest(mapping, child, memo)
        if child_best > best:
            best = child_best
    memo[node_id] = best
    return best


def parse_deepseek(data: list) -> list:
    """DeepSeek export: mapping tree without ``current_node`` — at each
    branch follow the child whose subtree has the latest ``inserted_at``
    (latest-edit-wins, rev 5). Dialogue text lives in fragments
    REQUEST (user) / RESPONSE (assistant); THINK/SEARCH/TOOL_* dropped."""
    out = []
    for conv in data:
        mapping = conv.get("mapping") or {}
        memo: dict = {}
        path = []
        node = mapping.get("root")
        seen = set()
        while node is not None:
            node_id = node.get("id")
            if node_id in seen:
                break
            seen.add(node_id)
            if node is not mapping.get("root"):
                path.append(node)
            children = [c for c in (node.get("children") or []) if c in mapping]
            if not children:
                break
            if len(children) == 1:
                node = mapping[children[0]]
            else:
                node = mapping[max(
                    children,
                    key=lambda c: _deepseek_subtree_latest(mapping, c, memo))]

        turns = []
        for n in path:
            message = n.get("message")
            if not message:
                continue
            ts = _parse_ts(message.get("inserted_at"))
            for frag in message.get("fragments") or []:
                ftype = (frag.get("type") or "").upper()
                text = (frag.get("content") or "").strip()
                if not text:
                    continue
                if ftype == "REQUEST":
                    turns.append(NormalizedTurn(role="user", text=text, ts=ts))
                elif ftype == "RESPONSE":
                    turns.append(NormalizedTurn(role="assistant", text=text,
                                                ts=ts))
                # THINK / SEARCH / TOOL_OPEN / TOOL_SEARCH: dropped

        def _frag_count(n):
            msg = n.get("message") or {}
            return sum(1 for f in (msg.get("fragments") or [])
                       if (f.get("type") or "").upper() in
                       ("REQUEST", "RESPONSE") and (f.get("content") or "").strip())
        total_fragments = sum(_frag_count(n) for n in mapping.values())
        norm = NormalizedConversation(
            provider="deepseek",
            source_id=str(conv.get("id") or conv.get("title") or len(out)),
            title=(conv.get("title") or "Untitled").strip() or "Untitled",
            turns=turns,
            created_at=_parse_ts(conv.get("inserted_at")),
            updated_at=_parse_ts(conv.get("updated_at")),
            branch_messages_skipped=max(0, total_fragments - len(turns)),
        )
        _synthesize_timestamps(norm)
        out.append(norm)
    return out


# ── Generic JSONL ───────────────────────────────────────────────────────────

def parse_jsonl(lines, default_title: str = "Imported log") -> list:
    """Escape hatch, two accepted line shapes:

    * **message** — ``{"role", "content", "timestamp"[, "conversation", "title"]}``
    * **pair** — ``{"prompt", "response", "timestamp"[, "conversation_id"]}``,
      one exchange per line, expanded into a user turn and an assistant turn.
      This is the shape ICE's own historical exports use, and a common shape for
      hand-rolled chat logs.

    Lines group by their ``conversation`` / ``conversation_id`` value when present
    (in first-seen order); otherwise the file is one conversation.
    """
    groups: dict = {}
    order: list = []

    def _bucket(key, title=None):
        if key not in groups:
            groups[key] = {"title": title, "turns": []}
            order.append(key)
        if title and not groups[key]["title"]:
            groups[key]["title"] = title
        return groups[key]

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL line {i + 1} is not valid JSON: {exc}")

        # Pair shape: one line is a whole exchange.
        if "role" not in obj and ("prompt" in obj or "response" in obj):
            ts = _parse_ts(obj.get("timestamp"))
            group = _bucket(str(obj.get("conversation_id")
                                or obj.get("conversation") or ""),
                            obj.get("title"))
            for role, field in (("user", "prompt"), ("assistant", "response")):
                text = (obj.get(field) or "").strip()
                if text:
                    group["turns"].append(NormalizedTurn(role=role, text=text, ts=ts))
            continue

        role = (obj.get("role") or "").lower()
        if role not in ("user", "assistant"):
            continue    # system/tool lines are not dialogue
        text = (obj.get("content") or "").strip()
        if not text:
            continue
        group = _bucket(str(obj.get("conversation") or ""), obj.get("title"))
        group["turns"].append(NormalizedTurn(
            role=role, text=text, ts=_parse_ts(obj.get("timestamp"))))

    out = []
    for key in order:
        g = groups[key]
        norm = NormalizedConversation(
            provider="jsonl",
            source_id=key or default_title,
            title=(g["title"] or key or default_title),
            turns=g["turns"],
        )
        _synthesize_timestamps(norm)
        out.append(norm)
    return out


# ── Detection + file entry ──────────────────────────────────────────────────

def detect_format(path: str) -> str:
    """Sniff the export shape. Unknown structure → ValueError naming the
    supported shapes (spec D2); ``.txt``/non-JSON text routes to F14's raw
    slicer."""
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".txt"):
        return "raw"
    try:
        with open(path, encoding="utf-8") as fh:
            head = fh.read(1)
            fh.seek(0)
            if head not in ("[", "{"):
                # First non-JSON byte: JSONL if the first line parses, else raw.
                first = fh.readline().strip()
                try:
                    obj = json.loads(first)
                    if isinstance(obj, dict) and "role" in obj:
                        return "jsonl"
                except json.JSONDecodeError:
                    pass
                return "raw"
            data = json.load(fh)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read export file {path}: {exc}")
    except json.JSONDecodeError:
        return "raw"

    items = data if isinstance(data, list) else [data]
    if items and isinstance(items[0], dict):
        first = items[0]
        if "chat_messages" in first:
            return "claude"
        if "mapping" in first and "current_node" in first:
            return "chatgpt"
        if "mapping" in first and "inserted_at" in first:
            return "deepseek"
    raise ValueError(
        "unrecognized export structure — supported shapes: ChatGPT "
        "conversations.json (mapping + current_node), Claude export "
        "(chat_messages), DeepSeek export (mapping + inserted_at), generic "
        "JSONL ({role, content, timestamp} per line), raw .txt dump")


_PARSERS = {
    "chatgpt": parse_chatgpt,
    "claude": parse_claude,
    "deepseek": parse_deepseek,
}


def normalize_file(path: str, fmt: Optional[str] = None):
    """(format, [NormalizedConversation]) for an export file. ``raw`` is
    routed by the caller to the F14 slicer (it needs the bg model)."""
    fmt = fmt or detect_format(path)
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unknown format '{fmt}' — supported: "
                         f"{', '.join(SUPPORTED_FORMATS)}")
    if fmt == "raw":
        return fmt, []
    if fmt == "jsonl":
        with open(path, encoding="utf-8") as fh:
            convs = parse_jsonl(fh, default_title=os.path.basename(path))
        return fmt, convs
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        data = [data]
    return fmt, _PARSERS[fmt](data)
