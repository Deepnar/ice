"""ICE-as-MCP server (E7) — a thin additional entrypoint over the same
install (`ice-mcp`, see pyproject [project.scripts]).

Built on the official `mcp` SDK's FastMCP: 10 tools (6 composite + 4
action-multiplexed micro) + the ice://session-start resource, every handler
a short hop into an E0 service — the SDK owns the protocol, ICE owns only
handlers. Adapter rule: no operation logic here; parsing, dispatch, and
rendering only.

Boot story (D6): postgres IS the shared core state — attach if it answers,
else `docker compose up -d postgres` and wait; then `create_core()` with the
lease-checked runtime (standby when the app owns maintenance — never a second
maintenance dispatcher). On shutdown the core stops but docker LINGERS
(postgres idles at ~zero cost; the next session attaches instantly);
`./stop_ice` / the packaged app remain the explicit all-the-way-down paths.

D8: writes apply immediately + journal — every tool call logs an
`mcp_tool_call` event (the E5 pull-discipline telemetry) and graph writes
journal CodexEvents with source="mcp_edit" (G17 vocabulary).
"""
import argparse
import subprocess
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Optional

import structlog
from mcp.server.fastmcp import FastMCP

from src.services import bookmarks as bookmarks_svc
from src.services import graph as graph_svc
from src.services import registry_svc
from src.services import retrieval_svc
from src.services import review as review_svc
from src.services import scoping as scoping_svc
from src.services import slots as slots_svc
from src.services.errors import NotFoundError

logger = structlog.get_logger("ice.mcp")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_FILE = _REPO_ROOT / "docker" / "docker-compose.yml"
_core = None


# ── Boot (D6) ────────────────────────────────────────────────────────────────

def _db_reachable() -> bool:
    from sqlalchemy import text

    from src.api.db import SessionLocal
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:
        return False


def ensure_db() -> None:
    """Attach if postgres answers; else boot it via docker compose and wait.
    Failure exits nonzero with the exact command to run (the harness surfaces
    stderr)."""
    if _db_reachable():
        logger.info("ice_mcp_attached")
        return
    cmd = ["docker", "compose", "-f", str(_COMPOSE_FILE), "up", "-d", "postgres"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"ice-mcp: could not run docker ({exc}).\n"
              f"Start postgres manually, then retry:\n  {' '.join(cmd)}",
              file=sys.stderr)
        raise SystemExit(1)
    if proc.returncode != 0:
        print(f"ice-mcp: docker compose failed:\n{proc.stderr}\n"
              f"Start postgres manually, then retry:\n  {' '.join(cmd)}",
              file=sys.stderr)
        raise SystemExit(1)
    for _ in range(30):
        if _db_reachable():
            logger.info("ice_mcp_booted_postgres")
            return
        time.sleep(1)
    print("ice-mcp: postgres did not become reachable within 30s.\n"
          f"Check:  docker compose -f {_COMPOSE_FILE} ps", file=sys.stderr)
    raise SystemExit(1)


@asynccontextmanager
async def _lifespan(server):
    global _core
    from src.api.core import create_core
    _core = create_core()   # lease-checked: standby if the app owns maintenance
    logger.info("ice_mcp_core_started",
                runtime_standby=_core.runtime.standby if _core.runtime else None)
    try:
        yield {}
    finally:
        await _core.stop()
        _core = None
        # D6 linger: docker stays up — next session attaches instantly.
        logger.info("ice_core_linger", docker="left-up",
                    note="./stop_ice remains the explicit all-the-way-down path")


mcp = FastMCP(
    "ice",
    instructions=(
        "ICE (Infinite Context Engine) — the user's long-term memory: past "
        "conversations, a knowledge graph of entities/ideas and how they "
        "evolved, observed working conventions, and pinned memory slots. "
        "Call ice_context with your current task BEFORE searching the "
        "codebase or asking the user something they may have already said; "
        "use ice_remember to persist decisions and facts worth keeping."
    ),
    lifespan=_lifespan,
)


# ── Adapter helpers ──────────────────────────────────────────────────────────

@contextmanager
def _session():
    from src.api.db import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _journal(tool: str, **fields) -> None:
    # D8/E5 telemetry: one event per tool call (reads included — pull-
    # discipline is measured from this stream).
    logger.info("mcp_tool_call", tool=tool, **fields)


def _embedder():
    from src.api.core import get_core
    core = get_core()
    return core.classifier.embedder if core is not None else None


def _bad_action(tool: str, action: str, valid: tuple) -> ValueError:
    # Guided self-correction for the calling model (spec §4).
    return ValueError(
        f"{tool}: unknown action {action!r} — valid actions: {', '.join(valid)}")


# ── Composite tools (the in-loop brain) ──────────────────────────────────────

@mcp.tool()
def ice_context(task: str, conversation_id: Optional[str] = None,
                budget: Optional[int] = None) -> dict:
    """Pull the user's relevant long-term memory for a task or question —
    call this BEFORE grepping the codebase or asking the user something they
    may have already told ICE. Returns structured fragments (past turns,
    knowledge-graph notes, procedural patterns, idea timelines) with scores
    and provenance, plus the classifier's read of the task. The first call in
    a fresh session loads the classifier (a few seconds, one-time)."""
    _journal("ice_context", words=len(task.split()))
    scope = {"conversation_id": conversation_id} if conversation_id else None
    with _session() as db:
        return retrieval_svc.context_for(db, task, scope=scope, budget=budget)


@mcp.tool()
def ice_why(name: str) -> dict:
    """The story of an entity/idea/decision: its current note (description,
    typed links and backlinks) plus its evolution timeline — how the idea
    changed over time — when any real supersession history exists."""
    _journal("ice_why", name=name)
    with _session() as db:
        view = graph_svc.entity_view(db, name)
        timeline = graph_svc.entity_timeline(db, name)
    return {**view, "timeline": timeline["timeline"]}


@mcp.tool()
def ice_recent(conversation_id: Optional[str] = None, limit: int = 10) -> list:
    """The most recent stored turns — what the user and ICE did last.
    Optionally scoped to one conversation id. (Project-level scoping lands
    with E1's project registry.)"""
    _journal("ice_recent", conversation_id=conversation_id, limit=limit)
    with _session() as db:
        return retrieval_svc.recent_turns(db, conversation_id, limit)


@mcp.tool()
def ice_conventions() -> list:
    """The user's observed working conventions — active procedural patterns
    (recurring behaviors ICE has learned), strongest first."""
    _journal("ice_conventions")
    with _session() as db:
        return retrieval_svc.conventions(db)


@mcp.tool()
def ice_where(symbol: str) -> dict:
    """Resolve a name/symbol against ICE's memory graph (entities by
    canonical name or alias). HONEST LIMIT: code-graph-aware symbol location
    (definitions, usages, files) lands with E1b — until then this resolves
    memory entities only."""
    _journal("ice_where", symbol=symbol)
    engine = "codex name/alias resolution (code graph lands with E1b)"
    with _session() as db:
        try:
            return {"resolved": True, "engine": engine,
                    **graph_svc.entity_view(db, symbol)}
        except NotFoundError:
            return {"resolved": False, "engine": engine,
                    "note": f"no memory entity matches {symbol!r}"}


@mcp.tool()
def ice_remember(text: str, target: str = "bookmark") -> dict:
    """Persist something worth keeping — applies immediately, journaled.
    target="bookmark" stores a permanent, decay-immune note that also feeds
    the knowledge graph; or name a memory slot (persona, user_preferences,
    tool_guidelines, project_context, guidance, pending_items,
    session_patterns) to append to it."""
    _journal("ice_remember", target=target, words=len(text.split()))
    with _session() as db:
        if target == "bookmark":
            return bookmarks_svc.remember_note(db, text, embedder=_embedder())
        return slots_svc.append_to_slot(db, target, text, updated_by="mcp_edit")


# ── Micro tools (full user control, action-multiplexed) ─────────────────────

@mcp.tool()
def ice_slots(action: str, name: Optional[str] = None,
              content: Optional[str] = None):
    """Memory-slot control. action="list" → all active slots; "get" →
    one slot by name; "set" → replace a slot's content (versioned;
    name + content required)."""
    _journal("ice_slots", action=action, name=name)
    with _session() as db:
        if action == "list":
            return slots_svc.list_slots(db)
        if action == "get":
            return slots_svc.get_slot(db, name)
        if action == "set":
            return slots_svc.update_slot(db, name, content or "",
                                         updated_by="mcp_edit")
    raise _bad_action("ice_slots", action, ("list", "get", "set"))


@mcp.tool()
def ice_graph(action: str, name: str, description: Optional[str] = None,
              t0: Optional[str] = None, t1: Optional[str] = None,
              include_expired: bool = False) -> dict:
    """Knowledge-graph control. action="view" → entity note + links/backlinks;
    "edit" → rewrite the entity's description (name + description; journaled,
    the derived note regenerates — the payload itself is never writable);
    "edges" → its edges (include_expired for history); "timeline" → the
    supersession timeline; "diff" → changes in [t0, t1) (ISO datetimes)."""
    _journal("ice_graph", action=action, name=name)
    with _session() as db:
        if action == "view":
            return graph_svc.entity_view(db, name)
        if action == "edit":
            return graph_svc.entity_edit(db, name, description, source="mcp_edit")
        if action == "edges":
            return graph_svc.edges_list(db, name, include_expired)
        if action == "timeline":
            return graph_svc.entity_timeline(db, name)
        if action == "diff":
            return graph_svc.entity_diff(db, name, t0, t1)
    raise _bad_action("ice_graph", action,
                      ("view", "edit", "edges", "timeline", "diff"))


@mcp.tool()
def ice_control(action: str, conversation_id: Optional[str] = None,
                memory_scope_type: Optional[str] = None,
                cluster_ids: Optional[list] = None,
                item_id: Optional[str] = None, status: str = "pending",
                model: Optional[str] = None, data: Optional[dict] = None):
    """ICE control plane. action="scope_get"/"scope_set" → a conversation's
    memory scope (scope_set: conversation_id + memory_scope_type auto|project|
    none [+ cluster_ids]); "review_list" → pending agent proposals (status
    filter); "review_approve"/"review_reject" → item_id; "registry_view" →
    the model registry; "registry_edit" → model + data (topic_tags/
    intent_tags/confirmed/base_url). (decisions_* actions land with E1.)"""
    _journal("ice_control", action=action)
    with _session() as db:
        if action == "scope_get":
            return scoping_svc.get_scope(db, conversation_id)
        if action == "scope_set":
            return scoping_svc.set_scope(db, conversation_id,
                                         memory_scope_type, cluster_ids)
        if action == "review_list":
            return review_svc.list_items(db, status)
        if action == "review_approve":
            return review_svc.approve(db, item_id)
        if action == "review_reject":
            return review_svc.reject(db, item_id)
        if action == "registry_view":
            return registry_svc.get_registry()
        if action == "registry_edit":
            return registry_svc.update_model(model, data or {})
    raise _bad_action("ice_control", action,
                      ("scope_get", "scope_set", "review_list",
                       "review_approve", "review_reject", "registry_view",
                       "registry_edit"))


@mcp.tool()
def ice_bookmarks(action: str, turn_id: Optional[str] = None,
                  conversation_id: Optional[str] = None):
    """Bookmark control. action="list" → bookmarked turns (optionally
    conversation-scoped); "add" → bookmark turn_id (promotes it to lossless +
    decay-immune and extracts it into the knowledge graph)."""
    _journal("ice_bookmarks", action=action, turn_id=turn_id)
    with _session() as db:
        if action == "list":
            return bookmarks_svc.list_bookmarks(db, conversation_id)
        if action == "add":
            return bookmarks_svc.bookmark_turn(db, turn_id)
    raise _bad_action("ice_bookmarks", action, ("list", "add"))


# ── Session-start resource ───────────────────────────────────────────────────

def _render_session_start(block: dict) -> str:
    lines = ["# ICE session start", "", "## Memory slots"]
    for slot in block["slots"]:
        content = slot["content"].strip() or "(empty)"
        lines += [f"### {slot['slot_name']} (v{slot['version']})", content, ""]
    last = block["last_session"]
    lines.append("## Last session")
    if last is None:
        lines.append("(no session summary recorded yet)")
    else:
        lines.append(f"date: {last['date']}")
        if last["topics_covered"]:
            lines.append("topics: " + ", ".join(last["topics_covered"]))
        if last["decisions_made"]:
            lines.append(f"decisions: {last['decisions_made']}")
        if last["unresolved_items"]:
            lines.append(f"unresolved: {last['unresolved_items']}")
        if last["entities_updated"]:
            lines.append("entities updated: " + ", ".join(last["entities_updated"]))
    return "\n".join(lines)


@mcp.resource("ice://session-start")
def session_start() -> str:
    """Welcome-back block: active memory slots + the last session's summary
    (E4 enriches this resource in place; the URI is stable)."""
    with _session() as db:
        return _render_session_start(retrieval_svc.session_start_block(db))


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ice-mcp",
        description="ICE memory as an MCP server (stdio by default)")
    parser.add_argument("--http", action="store_true",
                        help="serve streamable-http instead of stdio")
    args = parser.parse_args()
    # stdio transport owns stdout — every log line must go to stderr or it
    # corrupts the protocol framing.
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    ensure_db()
    mcp.run(transport="streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
