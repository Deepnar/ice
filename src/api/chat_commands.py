"""C11 — deterministic chat memory commands (adapter #3 over the E0 services).

`try_handle` is called from `chat_completions` BEFORE classification (D3):
a user message whose first line starts with `/` never reaches the classifier,
retrieval, or the model — handled commands short-circuit the LLM entirely and
the caller streams the returned confirmation as a normal OpenAI-style SSE
completion (`command_sse_stream`), so any frontend renders it. An
unrecognized `/x` gets a help hint, NEVER silent fallthrough — a typo'd
command must not leak into chat as a prompt.

Parsing rules: only the FIRST line is parsed as the command; remaining lines
are payload. Modifiers (`in <slot>`, `@project`, `@conversation`) are
scanned on the first line only, and `in <x>` is stripped only when `x` names
a valid slot for the resolved tier (so "check in tomorrow" stays text).

Every applied command journals: slot writes carry `updated_by=
'chat_command'`, destructive proposals record their proposer, and one
`chat_command` structlog event fires per handled command (the F5 telemetry
stream, mirroring `mcp_tool_call`). Handled command turns are NOT stored to
episodic memory — the journal is the record; a stored "/slots" would pollute
retrieval.

/forget and /delete-conversation follow the D1-tier safety model: precise
writes apply immediately; fuzzy or bulk-destructive ops queue (review
proposal) or demand an explicit two-step confirm.
"""
import json
import re
import time
import uuid
from typing import Optional

from src.api.config import settings
import structlog

from src.services import bookmarks as bookmarks_svc
from src.services import conversations as conversations_svc
from src.services import retrieval_svc
from src.services import scoping as scoping_svc
from src.services import slots as slots_svc
from src.services.errors import ConflictError, NotFoundError, ValidationError

logger = structlog.get_logger("ice.api.chat_commands")

COMMANDS = ("remember", "slots", "bookmark", "search", "scope", "forget",
            "delete-conversation", "help")

# /delete-conversation two-step confirm state: conversation id → monotonic
# deadline. Process-local by design (spec rev 11) — a restart clears pending
# confirmations, which fails safe (the user just re-runs the preview).
_PENDING_DELETES: dict = {}

HELP_TEXT = """**ICE chat commands**

- `/remember <text> [in <slot>] [@project|@conversation]` — append to a memory slot (default: global `pending_items`)
- `/slots [global|project|conversation]` — show persistent memory slots
- `/bookmark` — bookmark the last stored turn (lossless, decay-immune)
- `/search <query>` — search memory directly; dated fragments, no LLM
- `/scope auto|project|none` — set this conversation's memory scope
- `/forget <text>` — propose forgetting matching memories (applies only after review approval)
- `/delete-conversation` — preview the deletion manifest; confirm with `/delete-conversation confirm`
- `/help` — this list"""


def command_sse_stream(text: str, model: str = "ice-commands"):
    """Render a confirmation as a minimal OpenAI chat.completion.chunk SSE
    stream — one content chunk, a finish chunk, [DONE]. Any OpenAI-compatible
    frontend displays it like a normal assistant reply."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _chunk(delta: dict, finish: Optional[str]) -> str:
        return "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    yield _chunk({"role": "assistant", "content": text}, None)
    yield _chunk({}, "stop")
    yield "data: [DONE]\n\n"


def try_handle(db, runtime, conv, text: str,
               scope: Optional[dict] = None) -> Optional[dict]:
    """Parse and execute a slash command. Returns None when *text* is not a
    command (normal chat proceeds); otherwise a dict
    ``{"handled": True, "command": <name>, "text": <confirmation markdown>}``
    — including for unknown commands (help hint) and handler errors (the
    error text IS the confirmation; a failed command never becomes a prompt).

    *runtime* is the process's maintenance runtime (spec D3 signature; the
    services reach it via ``get_runtime()`` — carried for handlers that will
    need direct enqueue access). *scope* is main.py's already-resolved scope
    dict; /search passes it through verbatim for chat-path parity.
    """
    first_line, _, body = (text or "").partition("\n")
    first_line = first_line.strip()
    if not first_line.startswith("/"):
        return None

    parts = first_line[1:].split(None, 1)
    cmd = (parts[0].lower() if parts else "")
    args = parts[1].strip() if len(parts) > 1 else ""
    body = body.strip()
    conv_id = str(conv.id)

    try:
        if cmd == "help":
            out, outcome = HELP_TEXT, "rendered"
        elif cmd == "remember":
            out, outcome = _cmd_remember(db, conv, args, body)
        elif cmd == "slots":
            out, outcome = _cmd_slots(db, conv, args)
        elif cmd == "bookmark":
            out, outcome = _cmd_bookmark(db, conv)
        elif cmd == "search":
            out, outcome = _cmd_search(db, conv, args, body, scope)
        elif cmd == "scope":
            out, outcome = _cmd_scope(db, conv, args)
        elif cmd == "forget":
            out, outcome = _cmd_forget(db, conv, args, body)
        elif cmd == "delete-conversation":
            out, outcome = _cmd_delete_conversation(db, conv, args)
        else:
            out = (f"Unknown command `/{cmd}` — nothing was sent to the "
                   f"model.\n\n{HELP_TEXT}")
            outcome = "unknown"
    except (ValidationError, NotFoundError, ConflictError) as e:
        out, outcome = f"⚠ /{cmd}: {e}", "refused"
    except Exception as e:  # a failed command must still never become a prompt
        logger.exception("chat_command_failed", command=cmd)
        out, outcome = f"⚠ /{cmd} failed: {e}", "error"

    logger.info("chat_command", command=cmd, conversation_id=conv_id,
                outcome=outcome)
    return {"handled": True, "command": cmd, "text": out}


# ── Handlers (each a short hop into an E0 service + rendering) ─────────────

def _resolve_tier(conv, args: str):
    """Pull an `@project`/`@conversation` modifier off the first line."""
    tier, kwargs = "global", {}
    tokens = args.split()
    if "@project" in tokens:
        if not conv.project_id:
            raise ValidationError(
                "this conversation has no attached project — attach one with "
                "`/scope` via the frontend or `ice_control scope_set "
                "project=<slug>`, then retry")
        tier, kwargs = "project", {"project_id": str(conv.project_id)}
        tokens.remove("@project")
    elif "@conversation" in tokens:
        tier, kwargs = "conversation", {"conversation_id": str(conv.id)}
        tokens.remove("@conversation")
    return tier, kwargs, " ".join(tokens)


def _cmd_remember(db, conv, args: str, body: str):
    tier, anchor, rest = _resolve_tier(conv, args)
    slot = "pending_items"
    m = re.search(r"\s+in\s+(\w+)\s*$", rest)
    if m and m.group(1) in slots_svc.VALID_SLOTS[tier]:
        slot = m.group(1)
        rest = rest[:m.start()]
    content = (rest.strip() + ("\n" + body if body else "")).strip()
    if not content:
        raise ValidationError(
            "nothing to remember — usage: /remember <text> [in <slot>] "
            f"[@project|@conversation]; {tier} slots: "
            f"{', '.join(slots_svc.VALID_SLOTS[tier])}")
    res = slots_svc.append_to_slot(db, slot, content,
                                   updated_by="chat_command",
                                   scope_tier=tier, **anchor)
    out = (f"Remembered in **{tier} · {slot}** (now v{res['version']}, "
           f"{res['token_count']} tokens).")
    if res.get("truncated"):
        out += (f"\n⚠ {res['warning']}.")
    return out, "applied"


def _cmd_slots(db, conv, args: str):
    tier = args.split()[0].lower() if args else None
    if tier is not None and tier not in slots_svc.VALID_SLOTS:
        raise ValidationError(
            f"unknown tier {tier!r} — usage: /slots "
            f"[{'|'.join(sorted(slots_svc.VALID_SLOTS))}]")
    sections = []
    want = [tier] if tier else ["global", "project", "conversation"]
    for t in want:
        if t == "project":
            if not conv.project_id:
                if tier == "project":
                    sections.append("_(no project attached — project slots "
                                    "need an attached project)_")
                continue
            rows = slots_svc.list_slots(db, scope_tier="project",
                                        project_id=str(conv.project_id))
        elif t == "conversation":
            rows = slots_svc.list_slots(db, scope_tier="conversation",
                                        conversation_id=str(conv.id))
        else:
            rows = slots_svc.list_slots(db, scope_tier="global")
        for s in rows:
            preview = (s["content"] or "").replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "…"
            sections.append(
                f"- **[{t} · {s['slot_name']}]** v{s['version']}, "
                f"{s['token_count']} tok: {preview or '_(empty)_'}")
    return ("**Memory slots**\n" + "\n".join(sections)
            if sections else "No active slots."), "rendered"


def _cmd_bookmark(db, conv):
    try:
        latest = bookmarks_svc.latest_turn(db, str(conv.id))
    except NotFoundError:
        return ("Nothing to bookmark yet — a turn is stored only after its "
                "stream completes; retry in a moment."), "refused"
    res = bookmarks_svc.bookmark_turn(db, latest["turn_id"])
    return (f"Bookmarked the last stored turn ({res['timestamp'][:19]}): "
            f"“{res['raw_text'][:120]}” — lossless + decay-immune."), "applied"


def _cmd_search(db, conv, args: str, body: str, scope: Optional[dict]):
    query = (args + (" " + body if body else "")).strip()
    if not query:
        raise ValidationError("usage: /search <query>")
    res = retrieval_svc.context_for(db, query, scope=dict(scope or {}))
    frags = res["fragments"]
    if not frags:
        return (f"Memory search for “{query}”: no stored memories "
                "matched."), "rendered"
    lines = [f"**Memory search:** “{query}” — {len(frags)} fragment(s), "
             f"timescope: {res['timescope']}"]
    for i, f in enumerate(frags[:12], 1):
        frag_text = (f["text"] or "").replace("\n", " ")
        if len(frag_text) > 240:
            frag_text = frag_text[:240] + "…"
        lines.append(f"{i}. `[{f['source_type']}]` {frag_text}")
    if len(frags) > 12:
        lines.append(f"…and {len(frags) - 12} more.")
    return "\n".join(lines), "rendered"


def _cmd_scope(db, conv, args: str):
    mode = args.split()[0].lower() if args else ""
    if mode not in scoping_svc.VALID_SCOPE_TYPES:
        raise ValidationError(
            "usage: /scope " + "|".join(scoping_svc.VALID_SCOPE_TYPES))
    # C6: no cluster/filter passthrough any more — every id-set parameter is
    # None = leave unchanged, so a bare /scope changes the mode and nothing
    # else. (It used to overwrite on None, which is why this had to re-send
    # the current value just to avoid destroying it — spec rev 13.)
    scoping_svc.set_scope(db, str(conv.id), mode, project=None)
    picked = len(conv.included_conversation_ids or [])
    detail = {
        "none": "incognito — this conversation stores privately and reads "
                "only itself",
        "auto": "auto — global memory, private turns excluded",
        "project": "project — this conversation's own scope"
                   " (+ its clusters)",
        "manual": (f"manual — reads only this conversation and the {picked} "
                   f"you picked; your choice overrides ICE's automatic one"
                   if picked else
                   "manual — reads only this conversation; pick others to "
                   "widen it (sidebar, or ice_control scope_set)"),
    }[mode]
    return f"Memory scope set: **{detail}**.", "applied"


def _cmd_forget(db, conv, args: str, body: str):
    query = (args + ("\n" + body if body else "")).strip()
    if not query:
        raise ValidationError("usage: /forget <text describing what to forget>")
    res = conversations_svc.propose_forget(db, str(conv.id), query,
                                           proposed_by="chat_command")
    if not res["queued"]:
        return (f"No stored memories matched “{query}” — nothing was "
                "queued."), "rendered"
    lines = [f"**Forget proposal queued** (review item `{res['item_id']}`) — "
             "nothing is deleted until you approve it in the review queue."]
    for t in res["turns"]:
        lines.append(f"- turn {t['date'] or '?'} (sim {t['similarity']}): "
                     f"“{t['excerpt']}”")
    for e in res["edges"]:
        lines.append(f"- fact: {e['description']}")
    return "\n".join(lines), "applied"


def _render_manifest(m: dict) -> str:
    d, c = m["deleted"], m["codex"]
    head = ("**DRY RUN — nothing deleted yet.** Deletion would remove:"
            if m["dry_run"] else "**Conversation deleted.** Removed:")
    lines = [
        f"{head}",
        f"- episodic: {d['episodic_turns']} turns, {d['episodic_chunks']} "
        f"chunks, {d['cluster_links']} cluster links, "
        f"{d['cold_storage_rows']} cold rows",
        f"- clusters: {d['empty_clusters']} emptied → deleted, "
        f"{d['clusters_detached']} detached",
        f"- summaries: {d['batch_summaries']} batch, "
        f"{d['conversation_summaries']} conversation, "
        f"{d['session_summaries']} session; {d['session_replays']} replays",
        f"- codex: {c['edges_expired']} edge(s) expired (sole-support), "
        f"{c['edges_kept_corroborated']} kept (corroborated by other "
        f"conversations), {c['entities_expired']} entity(ies) expired",
        f"- other: {d['conversation_slots']} conversation slot(s), "
        f"{d['curated_labels']} curated label(s), "
        f"{m['decisions_expired']} decision(s) expired, "
        f"{m['procedural_pruned']} procedural pattern(s) pruned "
        f"({m['procedural_deactivated']} deactivated), "
        f"{m['review_items_staled']} review item(s) → stale",
        f"\n⚠ {m['logs_caveat']}",
    ]
    return "\n".join(lines)


def _cmd_delete_conversation(db, conv, args: str):
    arg = args.split()[0].lower() if args else ""
    cid = str(conv.id)
    if arg == "confirm":
        deadline = _PENDING_DELETES.get(cid)
        if deadline is None or time.monotonic() > deadline:
            _PENDING_DELETES.pop(cid, None)
            return ("Nothing pending to confirm — run "
                    "`/delete-conversation` first (the preview is valid for "
                    "10 minutes)."), "refused"
        manifest = conversations_svc.delete_conversation(db, cid,
                                                         dry_run=False)
        _PENDING_DELETES.pop(cid, None)
        return _render_manifest(manifest), "applied"
    if arg:
        raise ValidationError("usage: /delete-conversation "
                              "[confirm]")
    manifest = conversations_svc.delete_conversation(db, cid, dry_run=True)
    _PENDING_DELETES[cid] = time.monotonic() + settings.chat_confirm_ttl_seconds
    return (_render_manifest(manifest)
            + "\n\nTo delete permanently, send `/delete-conversation "
              "confirm` within 10 minutes."), "rendered"
