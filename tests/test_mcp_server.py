"""E7 ice-mcp behavioral test — specs/E0_E7_services_mcp.md §5 checks 5–10
(check 9's lease/boot logic lives in test_services.py's lease section; the
create_core standby/owner behavior asserted there is exactly what ice-mcp's
lifespan gets).

Drives the REAL FastMCP app in-process (mcp.shared.memory client — no stdio
subprocess, no network). The lifespan's create_core is stubbed (no model
load, no runtime); ice_context gets a stub-classifier core; ice_graph edit's
payload regeneration is stubbed (the real regeneration is validated in
test_services.py — here we assert the journal + source tag).

Live Postgres; inserts uniquely-marked rows, deletes them in finally; the
one real slot touched (session_patterns) is snapshot-and-restored.

Run: uv run python tests/test_mcp_server.py
"""
import asyncio
import json
import os
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub codex_extractor BEFORE the server import chain can pull it (it loads a
# SentenceTransformer at import; ice_graph edit only needs the journal here).
_stub_extractor = types.ModuleType("src.workers.codex_extractor")
_stub_extractor._regenerate_context_payload = lambda entity, db: None
_stub_extractor._expire_edge = lambda db, edge_id, batch_id, reason: None
sys.modules["src.workers.codex_extractor"] = _stub_extractor

from sqlalchemy import text

import src.api.core as core_mod
import src.mcp.server as server_mod
from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import (
    CodexEntity,
    Conversation,
    Decision,
    Document,
    DocumentLink,
    EpisodicMemory,
    MemorySlot,
    Project,
)

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


MARK = f"mcpe7{uuid.uuid4().hex[:6]}"
EMB = [0.05] * 1024
NOW = datetime.now(timezone.utc)

EXPECTED_TOOLS = {
    "ice_context", "ice_why", "ice_recent", "ice_conventions", "ice_where",
    "ice_remember", "ice_slots", "ice_graph", "ice_control", "ice_bookmarks",
}


class StubEmbedder:
    def encode(self, texts, convert_to_tensor=False, **kw):
        single = isinstance(texts, str)
        if convert_to_tensor:
            import torch
            return torch.tensor(EMB if single else [EMB for _ in texts])
        return list(EMB) if single else [list(EMB) for _ in texts]


class StubClassifier:
    embedder = StubEmbedder()

    def classify(self, prompt, **kw):
        return ClassificationResult(
            topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
            context_reliance="Long_Term_Memory", raw_probs=[0.0] * 25,
            max_confidence=1.0, prompt=prompt, p_ltm=0.9, ctx_confidence=0.9)


class FakeCore:
    classifier = StubClassifier()
    runtime = None

    async def stop(self):
        pass


def _payload(result):
    """Parse a CallToolResult's JSON payload (dict-returning tools)."""
    if result.structuredContent is not None:
        sc = result.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(result.content[0].text)


def _payload_list(result):
    """List-returning tools: FastMCP emits one content block per item when
    the result isn't structured."""
    if isinstance(result.structuredContent, dict) and "result" in result.structuredContent:
        return result.structuredContent["result"]
    return [json.loads(c.text) for c in result.content]


db = SessionLocal()
slot_snapshot = None
project_ids = []
extra_conv_ids = []
document_id = None

# A snapshot-and-restore cannot clean a row that did not exist to be
# snapshotted. On a clean store there is no `session_patterns` slot, this suite
# CREATES one through `ice_slots set`, and the old `if slot_snapshot:` restore
# then did nothing — leaving an `is_active=True`, `scope_tier='global'` slot
# behind on every run. `main.py` injects every such slot into the system
# prompt, so the leak was in every subsequent turn's context, not merely
# sitting in a table (TRAPS #16). Track creation as well as content, the way
# `test_c10_c11` does for `pending_items`.
slot_created = False
live_slot = db.query(MemorySlot).filter_by(slot_name="session_patterns").first()
if live_slot:
    slot_snapshot = {c: getattr(live_slot, c) for c in
                     ("content", "token_count", "version", "last_updated",
                      "updated_by", "is_active")}
else:
    slot_created = True

try:
    # ═══ Fixtures ════════════════════════════════════════════════════════
    conv = Conversation(memory_scope_type="auto")
    db.add(conv)
    db.commit()
    conv_id = str(conv.id)
    turn = EpisodicMemory(
        conversation_id=conv.id, batch_id=uuid.uuid4(),
        context_reliance="Zero_Shot",
        raw_text=f"{MARK} decided the flux capacitor uses {MARK}core",
        embedding=EMB, timestamp=NOW - timedelta(minutes=5),
        idempotency_key=f"{MARK}-turn",
    )
    entity = CodexEntity(id=uuid.uuid4(), canonical_name=f"{MARK}_entity",
                         aliases=[f"{MARK}_alias"], embedding=EMB,
                         description="original")
    db.add_all([turn, entity])
    db.commit()
    turn_id, entity_id = str(turn.id), str(entity.id)

    project_a = Project(name=f"{MARK} alpha", slug=f"{MARK}alpha", roots=[])
    project_b = Project(name=f"{MARK} beta", slug=f"{MARK}beta", roots=[])
    project_empty = Project(name=f"{MARK} empty", slug=f"{MARK}empty", roots=[])
    db.add_all([project_a, project_b, project_empty])
    db.flush()
    project_ids = [project_a.id, project_b.id, project_empty.id]
    conv_a = Conversation(memory_scope_type="auto", project_id=project_a.id)
    conv_b = Conversation(memory_scope_type="auto", project_id=project_b.id)
    doc_conv = Conversation(kind="document", memory_scope_type="auto")
    db.add_all([conv_a, conv_b, doc_conv])
    db.flush()
    extra_conv_ids = [conv_a.id, conv_b.id, doc_conv.id]
    turn_a = EpisodicMemory(
        conversation_id=conv_a.id, batch_id=uuid.uuid4(),
        context_reliance="Long_Term_Memory",
        raw_text=f"{MARK} alpha config.py constraint source",
        embedding=EMB, idempotency_key=f"{MARK}-alpha")
    turn_b = EpisodicMemory(
        conversation_id=conv_b.id, batch_id=uuid.uuid4(),
        context_reliance="Long_Term_Memory",
        raw_text=f"{MARK} beta config.py constraint source",
        embedding=EMB, idempotency_key=f"{MARK}-beta")
    doc = Document(conversation_id=doc_conv.id, filename=f"{MARK}.md",
                   file_type="md", sha256=uuid.uuid4().hex,
                   source_text=f"{MARK} enabled document", status="ready")
    db.add_all([turn_a, turn_b, doc])
    db.flush()
    document_id = doc.id
    link = DocumentLink(document_id=doc.id, conversation_id=conv_a.id,
                        enabled=True)
    db.add_all([
        link,
        Decision(project_id=project_a.id, decision=f"{MARK} alpha: do not touch config.py",
                 files_affected=["config.py"], decision_type="constraint"),
        Decision(project_id=project_b.id, decision=f"{MARK} beta: do not touch config.py",
                 files_affected=["config.py"], decision_type="constraint"),
    ])
    db.commit()
    from src.services.scoping import resolve_project_pull_scope
    project_scope = resolve_project_pull_scope(db, project_a.slug)
    check("explicit project scope includes only its chats and enabled documents",
          str(conv_a.id) in project_scope["conversation_ids"]
          and str(doc_conv.id) in project_scope["conversation_ids"]
          and str(conv_b.id) not in project_scope["conversation_ids"])
    check("empty project selection stays a closed conversation set",
          resolve_project_pull_scope(db, project_empty.slug)["conversation_ids"] == [])
    link.enabled = False
    db.commit()
    check("disabled document leaves explicit project scope",
          str(doc_conv.id) not in resolve_project_pull_scope(
              db, project_a.slug)["conversation_ids"])
    link.enabled = True
    db.commit()

    # lifespan boot: no model load, no runtime — the real create_core's
    # standby/owner behavior is asserted in test_services.py
    core_mod.create_core = lambda start_runtime=None: FakeCore()
    core_mod._active_core = FakeCore()   # ice_context's get_core()

    from mcp.shared.memory import create_connected_server_and_client_session

    async def run_checks():
        async with create_connected_server_and_client_session(
                server_mod.mcp._mcp_server, raise_exceptions=False) as client:
            # 5) all 10 tools, schemas valid
            tools = (await client.list_tools()).tools
            names = {t.name for t in tools}
            check("all 10 tools listed", names == EXPECTED_TOOLS)
            check("every tool has a description + input schema",
                  all(t.description and t.inputSchema is not None for t in tools))

            # 6) ice_slots set/get round-trip
            r = await client.call_tool("ice_slots", {
                "action": "set", "name": "session_patterns",
                "content": f"{MARK} via mcp"})
            check("ice_slots set applies", not r.isError
                  and _payload(r)["content"] == f"{MARK} via mcp"
                  and _payload(r)["updated_by"] == "mcp_edit")
            r = await client.call_tool("ice_slots", {
                "action": "get", "name": "session_patterns"})
            check("ice_slots get round-trips",
                  _payload(r)["content"] == f"{MARK} via mcp")
            r = await client.call_tool("ice_slots", {"action": "bogus"})
            check("unknown action errors with the action enum",
                  r.isError and "valid actions: list, get, set"
                  in r.content[0].text)

            # 7) ice_graph edit journals with source="mcp_edit"
            r = await client.call_tool("ice_graph", {
                "action": "edit", "name": f"{MARK}_entity",
                "description": f"{MARK} edited via mcp"})
            check("ice_graph edit applies", not r.isError
                  and _payload(r)["description"] == f"{MARK} edited via mcp")
            ev = db.execute(text(
                "SELECT payload FROM codex_events "
                "WHERE event_type = 'description_updated' AND entity_id = :e "
                "ORDER BY timestamp DESC LIMIT 1"), {"e": entity_id}).first()
            check("ice_graph edit journals source=mcp_edit",
                  ev is not None and ev.payload["source"] == "mcp_edit")
            r = await client.call_tool("ice_graph", {
                "action": "view", "name": f"{MARK}_alias"})
            check("ice_graph view resolves aliases",
                  _payload(r)["canonical_name"] == f"{MARK}_entity")

            # 8) ice_context returns structured fragments on the seeded DB
            r = await client.call_tool("ice_context", {
                "task": f"what did we decide about the flux capacitor {MARK}?"})
            payload = _payload(r)
            check("ice_context returns structured fragments", not r.isError
                  and isinstance(payload["fragments"], list)
                  and "memory_decision" in payload)
            check("ice_context finds the seeded turn",
                  any(MARK in f["text"] for f in payload["fragments"]))
            r = await client.call_tool("ice_context", {
                "task": f"edit config.py {MARK}", "project": project_a.slug})
            scoped = _payload(r)
            check("project-selected ice_context returns only its own rule",
                  not r.isError and any(
                      f["source_type"] == "constraint" and "alpha" in f["text"]
                      for f in scoped["fragments"])
                  and not any("beta" in f["text"] for f in scoped["fragments"]))
            check("project-selected ice_context keeps episodic scope closed",
                  any(f["source_batch_id"] == str(turn_a.id)
                      for f in scoped["fragments"])
                  and not any(f["source_batch_id"] == str(turn_b.id)
                              for f in scoped["fragments"]))
            r = await client.call_tool("ice_context", {
                "task": f"edit config.py {MARK}", "project": project_empty.slug})
            check("empty project pull never widens to another project's turns",
                  not r.isError and not any(
                      f["source_batch_id"] in (str(turn_a.id), str(turn_b.id))
                      for f in _payload(r)["fragments"]))
            r = await client.call_tool("ice_context", {
                "task": f"edit config.py {MARK}", "project": project_a.slug,
                "conversation_id": conv_id})
            check("ice_context rejects simultaneous project and conversation choices",
                  r.isError)
            r = await client.call_tool("ice_context", {
                "task": f"edit config.py {MARK}", "project": f"{MARK}missing"})
            check("unknown project selection is an error, never a global search",
                  r.isError)
            r = await client.call_tool("ice_context", {
                "task": f"edit config.py {MARK}", "project": " "})
            check("blank project choice cannot silently widen to global",
                  r.isError)
            r = await client.call_tool("ice_context", {
                "task": f"edit config.py {MARK}"})
            check("projectless ice_context returns no project's rule",
                  not any(f["source_type"] == "constraint"
                          for f in _payload(r)["fragments"]))

            # composite reads
            r = await client.call_tool("ice_recent",
                                       {"conversation_id": conv_id})
            check("ice_recent returns the seeded turn",
                  any(MARK in t["text"] for t in _payload_list(r)))
            r = await client.call_tool("ice_why", {"name": f"{MARK}_entity"})
            check("ice_why merges view + timeline",
                  "timeline" in _payload(r)
                  and _payload(r)["canonical_name"] == f"{MARK}_entity")
            # E1b shipped: the code graph is the primary engine; a non-code
            # name falls back to codex name/alias resolution (engine named).
            r = await client.call_tool("ice_where", {"symbol": f"{MARK}_alias"})
            check("ice_where resolves by alias and names its engine",
                  _payload(r)["resolved"] is True
                  and "codex" in _payload(r)["engine"])
            r = await client.call_tool("ice_where", {"symbol": f"{MARK}_nope"})
            check("ice_where miss is honest, not an error",
                  not r.isError and _payload(r)["resolved"] is False)
            r = await client.call_tool("ice_conventions", {})
            check("ice_conventions callable", not r.isError)

            # writes: remember → bookmark note; bookmarks list/add
            r = await client.call_tool("ice_remember", {
                "text": f"{MARK} remember this note", "target": "bookmark"})
            check("ice_remember(bookmark) stores a note",
                  not r.isError and _payload(r)["bookmarked"] is True)
            r = await client.call_tool("ice_bookmarks", {
                "action": "add", "turn_id": turn_id})
            check("ice_bookmarks add promotes the turn",
                  not r.isError and _payload(r)["is_bookmarked"] is True)
            r = await client.call_tool("ice_bookmarks", {
                "action": "list", "conversation_id": conv_id})
            check("ice_bookmarks list sees it",
                  [b["id"] for b in _payload_list(r)] == [turn_id])

            # control plane: scope round-trip via ice_control
            r = await client.call_tool("ice_control", {
                "action": "scope_set", "conversation_id": conv_id,
                "memory_scope_type": "project"})
            check("ice_control scope_set applies",
                  _payload(r)["memory_scope_type"] == "project")
            r = await client.call_tool("ice_control", {
                "action": "scope_get", "conversation_id": conv_id})
            check("ice_control scope_get round-trips",
                  _payload(r)["memory_scope_type"] == "project")

            # 10) session-start resource renders
            res = await client.read_resource("ice://session-start")
            text_block = res.contents[0].text
            check("session-start resource renders slots block",
                  "# ICE session start" in text_block
                  and "## Memory slots" in text_block
                  and f"{MARK} via mcp" in text_block)

    asyncio.run(run_checks())

finally:
    db.rollback()
    if document_id is not None:
        db.execute(text("DELETE FROM document_links WHERE document_id = :id"),
                   {"id": document_id})
        db.execute(text("DELETE FROM documents WHERE id = :id"),
                   {"id": document_id})
    if project_ids:
        db.execute(text("DELETE FROM decisions WHERE project_id = ANY(:ids)"),
                   {"ids": project_ids})
    db.execute(text(
        "DELETE FROM codex_events WHERE entity_id IN "
        "(SELECT id FROM codex_entities WHERE canonical_name LIKE :m)"),
        {"m": f"{MARK}%"})
    db.execute(text("DELETE FROM codex_entities WHERE canonical_name LIKE :m"),
               {"m": f"{MARK}%"})
    db.query(EpisodicMemory).filter(
        EpisodicMemory.raw_text.like(f"{MARK}%")).delete(synchronize_session=False)
    db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).delete(
        synchronize_session=False)
    if extra_conv_ids:
        db.execute(text("DELETE FROM conversations WHERE id = ANY(:ids)"),
                   {"ids": extra_conv_ids})
    if project_ids:
        db.execute(text("DELETE FROM projects WHERE id = ANY(:ids)"),
                   {"ids": project_ids})
    live = db.query(MemorySlot).filter_by(slot_name="session_patterns").first()
    if live is not None and slot_snapshot:
        for k, v in slot_snapshot.items():
            setattr(live, k, v)
    elif live is not None and slot_created:
        db.delete(live)
    db.commit()
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
