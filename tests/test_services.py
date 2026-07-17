"""E0 service-layer behavioral test — specs/E0_E7_services_mcp.md §5
checks 1, 2, 4 (+ the runtime-lease half of 9 and the grep-gate).

Runs against the live Postgres (docker up). Inserts its own uniquely-marked
rows and deletes them afterwards — never truncates (the dev DB holds real
data). The one real slot it touches (session_patterns) is snapshot-and-
restored. No LLM; the embedder is a stub; the maintenance runtime is a stub
recorder for enqueue assertions; merge_entities is monkeypatched (the real
one is a D1 stub by design).

Run: uv run python tests/test_services.py
"""
import asyncio
import json
import os
import pathlib
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import (
    CodexEdge,
    CodexEntity,
    ContextCluster,
    Conversation,
    EpisodicMemory,
    MemorySlot,
    ProceduralMemory,
    ReviewQueue,
)
from src.services import bookmarks, clusters, graph, registry_svc
from src.services import retrieval_svc, review, scoping, slots
from src.services.errors import NotFoundError, ValidationError

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


# lowercase: entity canonical_names are stored casefolded (graph._resolve)
MARK = f"svce0{uuid.uuid4().hex[:6]}"
EMB = [0.05] * 384
NOW = datetime.now(timezone.utc)


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


class StubRuntime:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, job_name, **kwargs):
        self.enqueued.append((job_name, kwargs))


db = SessionLocal()
stub_rt = StubRuntime()
bookmarks.get_runtime = lambda: stub_rt          # stub the enqueue seam

slot_snapshot = None
notes_conv_preexisting = True
lease_row_snapshot = "absent"

try:
    # ═══ Fixtures ════════════════════════════════════════════════════════
    conv = Conversation(memory_scope_type="auto")
    db.add(conv)
    db.commit()
    conv_id = str(conv.id)
    batch_id = uuid.uuid4()
    turn_ids = []
    for i in range(3):
        t = EpisodicMemory(
            conversation_id=conv.id, batch_id=batch_id,
            context_reliance="Zero_Shot",
            raw_text=f"{MARK} turn {i}: the zephyrglass prototype uses {MARK}lib",
            embedding=EMB, timestamp=NOW - timedelta(minutes=10 - i),
            idempotency_key=f"{MARK}-{i}",
        )
        db.add(t)
        db.flush()
        turn_ids.append(str(t.id))
    db.commit()

    live_slot = db.query(MemorySlot).filter_by(slot_name="session_patterns").first()
    if live_slot:
        slot_snapshot = {c: getattr(live_slot, c) for c in
                         ("content", "token_count", "version", "last_updated",
                          "updated_by", "is_active")}

    # ═══ 1. Slots round-trip ═════════════════════════════════════════════
    print("── slots ──")
    before = slots.get_slot(db, "session_patterns") if live_slot else None
    updated = slots.update_slot(db, "session_patterns", f"{MARK} content v1")
    check("slot update writes content", updated["content"] == f"{MARK} content v1")
    check("slot update bumps version",
          before is None or updated["version"] == before["version"] + 1)
    appended = slots.append_to_slot(db, "session_patterns", f"{MARK} appended",
                                    updated_by="mcp_edit")
    check("append_to_slot newline-appends",
          appended["content"] == f"{MARK} content v1\n{MARK} appended")
    check("append bumps version again",
          appended["version"] == updated["version"] + 1)
    check("mcp writes carry updated_by=mcp_edit", appended["updated_by"] == "mcp_edit")
    try:
        slots.get_slot(db, "not_a_slot")
        check("invalid slot name raises ValidationError", False)
    except ValidationError as e:
        check("invalid slot name raises ValidationError",
              "Invalid slot name" in str(e))

    # ═══ 2. Bookmarks: flags + stub-runtime enqueue ══════════════════════
    print("── bookmarks ──")
    out = bookmarks.bookmark_turn(db, turn_ids[0])
    row = db.query(EpisodicMemory).filter_by(id=uuid.UUID(turn_ids[0])).first()
    check("bookmark sets is_bookmarked+lossless+decay_immune",
          row.is_bookmarked and row.lossless_flag and row.decay_immune)
    check("bookmark enqueues codex_extract with priority",
          ("codex_extract", {"batch_id": str(batch_id), "priority": True})
          in stub_rt.enqueued)
    check("bookmark returns the turn payload", out["id"] == turn_ids[0])
    try:
        bookmarks.bookmark_turn(db, str(uuid.uuid4()))
        check("missing turn raises NotFoundError", False)
    except NotFoundError:
        check("missing turn raises NotFoundError", True)
    listed = bookmarks.list_bookmarks(db, conversation_id=conv_id)
    check("list_bookmarks scoped returns the bookmarked turn",
          [b["id"] for b in listed] == [turn_ids[0]])
    check("latest_turn returns newest", bookmarks.latest_turn(db, conv_id)
          ["turn_id"] == turn_ids[-1])

    notes_conv_preexisting = db.query(Conversation).filter_by(
        id=bookmarks.NOTES_CONVERSATION_ID).first() is not None
    note = bookmarks.remember_note(db, f"{MARK} remembered decision",
                                   embedder=StubEmbedder())
    note_row = db.query(EpisodicMemory).filter_by(id=uuid.UUID(note["id"])).first()
    check("remember_note stores bookmarked decay-immune note",
          note_row is not None and note_row.is_bookmarked
          and note_row.decay_immune and note_row.lossless_flag
          and note_row.conversation_id == bookmarks.NOTES_CONVERSATION_ID)
    check("remember_note embeds and enqueues extraction",
          note_row.embedding is not None
          and ("codex_extract", {"batch_id": str(note_row.batch_id),
                                 "priority": True}) in stub_rt.enqueued)

    # ═══ 3. Scoping: G16 both directions + override_tags ═════════════════
    print("── scoping ──")
    scoping.set_scope(db, conv_id, "none")
    privates = [r.is_private for r in db.query(EpisodicMemory)
                .filter_by(conversation_id=conv.id).all()]
    check("scope→none flips is_private True on all turns", all(privates))
    scoping.set_scope(db, conv_id, "auto")
    privates = [r.is_private for r in db.query(EpisodicMemory)
                .filter_by(conversation_id=conv.id).all()]
    check("scope→auto flips is_private back False", not any(privates))
    got = scoping.get_scope(db, conv_id)
    check("get_scope round-trips", got["memory_scope_type"] == "auto")
    ov = scoping.override_tags(db, str(batch_id), ["Software_&_Tech"],
                               ["Generation"], "Zero_Shot")
    check("override_tags writes a curated label", ov["status"] == "ok")

    # ═══ 4. Clusters ═════════════════════════════════════════════════════
    print("── clusters ──")
    cl = clusters.create_cluster(db, f"{MARK}_cluster", "svc test")
    assigned = clusters.assign_turns(db, cl["id"], turn_ids[:2])
    check("cluster create/assign", assigned["assigned"] == 2 and
          db.query(EpisodicMemory).filter_by(cluster_id=uuid.UUID(cl["id"]))
          .count() == 2)

    # ═══ 5. Review: slot apply + D1/D2 arms + reject ═════════════════════
    print("── review ──")
    marker_slot = MemorySlot(slot_name=f"{MARK}_slot", content="old",
                             token_count=1, version=1, last_updated=NOW,
                             updated_by="user", is_active=False)
    rq1 = ReviewQueue(item_type="memory_slot_update", item_content={
        "slot_name": f"{MARK}_slot", "proposed_content": f"{MARK} approved"})
    db.add_all([marker_slot, rq1])
    db.commit()
    rq1_id = str(rq1.id)
    review.approve(db, rq1_id)
    ms = db.query(MemorySlot).filter_by(slot_name=f"{MARK}_slot").first()
    check("approve applies memory_slot_update (content + version bump)",
          ms.content == f"{MARK} approved" and ms.version == 2)

    merge_calls = []
    import src.workers.codex_ops as codex_ops
    codex_ops.merge_entities = lambda db, keep_id, absorb_id, agent_run_id=None: \
        merge_calls.append((keep_id, absorb_id, agent_run_id))
    keep_u, absorb_u = uuid.uuid4(), uuid.uuid4()
    rq2 = ReviewQueue(item_type="entity_merge", item_content={
        "keep_id": str(keep_u), "absorb_id": str(absorb_u),
        "agent_run_id": f"{MARK}_run"})
    db.add(rq2)
    db.commit()
    review.approve(db, str(rq2.id))
    check("approve dispatches entity_merge to merge_entities (D1 arm)",
          merge_calls == [(keep_u, absorb_u, f"{MARK}_run")])

    e1 = CodexEntity(id=uuid.uuid4(), canonical_name=f"{MARK}_subj",
                     aliases=[f"{MARK}_subj"], embedding=EMB)
    e2 = CodexEntity(id=uuid.uuid4(), canonical_name=f"{MARK}_obj",
                     aliases=[f"{MARK}_obj"], embedding=EMB)
    db.add_all([e1, e2])
    db.flush()   # no relationship() on the models — entities must land first
    edge = CodexEdge(source_id=e1.id, target_id=e2.id, relation="uses",
                     source_batch=batch_id, valid_from=NOW - timedelta(days=30))
    db.add(edge)
    db.commit()
    edge_id = edge.id
    rq3 = ReviewQueue(item_type="codex_reconciliation", item_content={
        "new": {"subject": f"{MARK}_subj", "relation": "uses",
                "object": "newthing"},
        "conflict_type": "supersession", "old_edge_id": str(edge_id),
        "old_relation": "uses", "old_object": f"{MARK}_obj",
        "turn_excerpt": "moved off it"})
    db.add(rq3)
    db.commit()
    review.approve(db, str(rq3.id))
    edge = db.query(CodexEdge).filter_by(id=edge_id).first()
    ev = db.execute(text(
        "SELECT payload FROM codex_events WHERE event_type = 'edge_expired' "
        "AND payload->>'edge_id' = :eid"), {"eid": str(edge_id)}).first()
    check("approve applies codex_reconciliation (edge expired + journaled)",
          edge.valid_until is not None and ev is not None
          and ev.payload["reason"] == "supersession")

    rq4 = ReviewQueue(item_type="sentinel_review", item_content={"note": MARK})
    db.add(rq4)
    db.commit()
    rejected = review.reject(db, str(rq4.id))
    check("reject flips status only",
          rejected["status"] == "rejected" and
          db.query(ReviewQueue).filter_by(id=rq4.id).first().status == "rejected")

    # ═══ 6. Registry under the lock ══════════════════════════════════════
    print("── registry ──")
    import src.model_registry.registry as reg_mod
    reg_path_orig = reg_mod.REGISTRY_PATH
    reg_mod.REGISTRY_PATH = f"/tmp/ice_{MARK}_registry.json"
    try:
        with open(reg_mod.REGISTRY_PATH, "w") as f:
            json.dump({"models": {"m:1b": {"topic_tags": [], "intent_tags": [],
                                           "priority": 5, "context_window": 8192,
                                           "confirmed": False, "base_url": None,
                                           "added_at": 0}},
                       "updated_at": 0}, f)
        registry_svc.update_model("m:1b", {"confirmed": True,
                                           "topic_tags": ["Software_&_Tech"]})
        persisted = json.load(open(reg_mod.REGISTRY_PATH))
        check("registry edit persists under the lock",
              persisted["models"]["m:1b"]["confirmed"] is True
              and os.path.exists(reg_mod.REGISTRY_PATH + ".lock"))
        try:
            registry_svc.update_model("nope", {})
            check("unknown model raises NotFoundError", False)
        except NotFoundError:
            check("unknown model raises NotFoundError", True)
        registry_svc.delete_model("m:1b")
        check("registry delete persists",
              json.load(open(reg_mod.REGISTRY_PATH))["models"] == {})
    finally:
        for p in (reg_mod.REGISTRY_PATH, reg_mod.REGISTRY_PATH + ".lock"):
            if os.path.exists(p):
                os.unlink(p)
        reg_mod.REGISTRY_PATH = reg_path_orig

    # ═══ 7. Graph: view/edit journal + payload regeneration ═════════════
    print("── graph (loads codex_extractor's embedder — one-time) ──")
    view = graph.entity_view(db, f"{MARK}_subj")
    check("entity_view resolves by name with typed links",
          view["canonical_name"] == f"{MARK}_subj" and
          isinstance(view["links"], list) and isinstance(view["backlinks"], list))
    check("entity_view resolves by id too",
          graph.entity_view(db, str(e1.id))["canonical_name"] == f"{MARK}_subj")

    edited = graph.entity_edit(db, f"{MARK}_subj",
                               f"{MARK} edited description", source="manual_edit")
    ev = db.execute(text(
        "SELECT payload FROM codex_events WHERE event_type = 'description_updated' "
        "AND entity_id = :eid ORDER BY timestamp DESC LIMIT 1"),
        {"eid": str(e1.id)}).first()
    check("entity_edit writes description + journals description_updated",
          edited["description"] == f"{MARK} edited description"
          and ev is not None and ev.payload["source"] == "manual_edit"
          and ev.payload["new"].startswith(f"{MARK} edited"))
    check("entity_edit regenerates context_payload (derived, contains description)",
          f"{MARK} edited description" in edited["note"])
    try:
        graph.entity_edit(db, f"{MARK}_subj", None)
        check("entity_edit(None) raises ValidationError", False)
    except ValidationError:
        check("entity_edit(None) raises ValidationError", True)

    tl = graph.entity_timeline(db, f"{MARK}_subj")
    check("entity_timeline surfaces the approved supersession",
          tl["has_history"] and "uses" in (tl["timeline"] or ""))
    diff = graph.entity_diff(db, f"{MARK}_subj",
                             (NOW - timedelta(days=60)).isoformat(),
                             (NOW + timedelta(days=1)).isoformat())
    check("entity_diff reports added+expired with ISO dates",
          len(diff["added"]) == 1 and len(diff["expired"]) == 1
          and isinstance(diff["expired"][0]["date"], str))
    edges_all = graph.edges_list(db, f"{MARK}_subj", include_expired=True)
    check("edges_list include_expired shows the expired edge",
          any(e["until"] for e in edges_all["edges"]))

    # ═══ 8. Retrieval service ════════════════════════════════════════════
    print("── retrieval_svc ──")
    import src.api.core as core_mod

    class FakeCore:
        classifier = StubClassifier()
    core_mod._active_core = FakeCore()
    ctx = retrieval_svc.context_for(db, f"what did I say about zephyrglass {MARK}?")
    check("context_for returns structured fragments (not a prompt)",
          isinstance(ctx["fragments"], list) and "memory_decision" in ctx
          and all({"text", "source_type", "score", "token_count"} <=
                  set(f.keys()) for f in ctx["fragments"]))
    check("context_for finds the seeded marker turn",
          any(MARK in f["text"] for f in ctx["fragments"]))
    core_mod._active_core = None

    recents = retrieval_svc.recent_turns(db, conversation_id=conv_id, limit=5)
    check("recent_turns scoped returns newest-first marker turns",
          len(recents) == 3 and MARK in recents[0]["text"])
    scoping.set_scope(db, conv_id, "none")   # → private
    unscoped = retrieval_svc.recent_turns(db, limit=500)
    check("recent_turns unscoped excludes private turns (G16)",
          not any(r["conversation_id"] == conv_id for r in unscoped))
    scoping.set_scope(db, conv_id, "auto")

    pm = ProceduralMemory(pattern_name=f"{MARK}_pattern",
                          pattern_description="always writes tests",
                          confidence_score=0.99, is_active=True)
    db.add(pm)
    db.commit()
    conv_list = retrieval_svc.conventions(db)
    check("conventions returns active procedural patterns",
          any(c["pattern"] == f"{MARK}_pattern" for c in conv_list))

    block = retrieval_svc.session_start_block(db)
    check("session_start_block returns slots (+ optional last_session)",
          isinstance(block["slots"], list) and "last_session" in block)

    # ═══ 9. Runtime lease → create_core boot logic (E7 D6) ═══════════════
    print("── lease boot logic ──")
    row = db.execute(text("SELECT last_started, last_status FROM maintenance_ledger "
                          "WHERE job_name = 'runtime_lease'")).first()
    lease_row_snapshot = ("present", row.last_started, row.last_status) if row else "absent"

    async def lease_checks():
        from src.api.core import create_core
        # fresh foreign lease ⇒ standby, and standby never stamps the lease
        db.execute(text(
            "INSERT INTO maintenance_ledger (job_name, last_started, last_status, runs) "
            "VALUES ('runtime_lease', :now, 'running', 0) "
            "ON CONFLICT (job_name) DO UPDATE SET last_started = :now, "
            "last_status = 'running'"), {"now": datetime.now(timezone.utc)})
        db.commit()
        stamp_before = db.execute(text(
            "SELECT last_started FROM maintenance_ledger "
            "WHERE job_name = 'runtime_lease'")).scalar()
        core = create_core()
        check("fresh lease ⇒ runtime starts in standby",
              core.runtime is not None and core.runtime.standby is True)
        await asyncio.sleep(0.2)
        stamp_after = db.execute(text(
            "SELECT last_started FROM maintenance_ledger "
            "WHERE job_name = 'runtime_lease'")).scalar()
        check("standby never stamps the lease", stamp_after == stamp_before)
        await core.stop()
        # stale lease ⇒ owner; owner stamps on start and releases on stop
        db.execute(text("UPDATE maintenance_ledger SET last_started = :old "
                        "WHERE job_name = 'runtime_lease'"),
                   {"old": datetime.now(timezone.utc) - timedelta(seconds=600)})
        db.commit()
        core = create_core()
        check("stale lease ⇒ runtime owns maintenance",
              core.runtime is not None and core.runtime.standby is False)
        await asyncio.sleep(0.3)
        stamp = db.execute(text(
            "SELECT last_started FROM maintenance_ledger "
            "WHERE job_name = 'runtime_lease'")).scalar()
        check("owner stamps the lease at start",
              stamp is not None and
              (datetime.now(timezone.utc) - stamp).total_seconds() < 30)
        await core.stop()
        released = db.execute(text(
            "SELECT last_started FROM maintenance_ledger "
            "WHERE job_name = 'runtime_lease'")).scalar()
        check("owner releases the lease on stop", released is None)
        # start_runtime=False ⇒ no runtime at all
        core = create_core(start_runtime=False)
        check("start_runtime=False ⇒ no runtime", core.runtime is None)
        await core.stop()
        # standby promotion when the owner's lease goes stale
        from src.workers.runtime import MaintenanceRuntime
        rt = MaintenanceRuntime(jobs={}, intervals={})
        rt.tick_base, rt.tick_jitter = 0.05, 0.0
        rt.start(SessionLocal, standby=True)
        await asyncio.sleep(0.5)
        check("standby promotes itself once the lease is stale",
              rt.standby is False)
        await rt.stop()

    asyncio.run(lease_checks())

    # ═══ 10. Grep-gate: services are HTTP-free ═══════════════════════════
    print("── grep-gate ──")
    bad = []
    root = pathlib.Path(__file__).resolve().parents[1]
    for tree in ("src/services", "src/mcp"):
        for p in (root / tree).rglob("*.py"):
            for n, line in enumerate(p.read_text().splitlines(), 1):
                if re.search(r"fastapi|HTTPException", line):
                    bad.append(f"{p}:{n}")
    check("no fastapi/HTTPException under src/services or src/mcp", not bad)
    if bad:
        print("   offenders:", bad)

finally:
    db.rollback()
    # ── Marker-keyed cleanup — leave the DB exactly as found ──
    db.query(ReviewQueue).filter(
        text("item_content::text LIKE :m")).params(m=f"%{MARK}%").delete(
        synchronize_session=False)
    db.execute(text("DELETE FROM curated_labels WHERE batch_id = :b"),
               {"b": str(batch_id)})
    db.execute(text(
        "DELETE FROM codex_events WHERE entity_id IN "
        "(SELECT id FROM codex_entities WHERE canonical_name LIKE :m)"),
        {"m": f"{MARK}%"})
    db.execute(text(
        "DELETE FROM codex_edges WHERE source_id IN "
        "(SELECT id FROM codex_entities WHERE canonical_name LIKE :m) "
        "OR target_id IN "
        "(SELECT id FROM codex_entities WHERE canonical_name LIKE :m)"),
        {"m": f"{MARK}%"})
    db.execute(text("DELETE FROM codex_entities WHERE canonical_name LIKE :m"),
               {"m": f"{MARK}%"})
    db.query(EpisodicMemory).filter(
        EpisodicMemory.idempotency_key.like(f"{MARK}%")).delete(
        synchronize_session=False)
    note_rows = db.query(EpisodicMemory).filter(
        EpisodicMemory.conversation_id == bookmarks.NOTES_CONVERSATION_ID,
        EpisodicMemory.raw_text.like(f"{MARK}%"))
    note_rows.delete(synchronize_session=False)
    if not notes_conv_preexisting:
        db.query(Conversation).filter_by(
            id=bookmarks.NOTES_CONVERSATION_ID).delete(synchronize_session=False)
    db.query(ContextCluster).filter_by(name=f"{MARK}_cluster").delete(
        synchronize_session=False)
    db.query(MemorySlot).filter_by(slot_name=f"{MARK}_slot").delete(
        synchronize_session=False)
    db.query(ProceduralMemory).filter_by(pattern_name=f"{MARK}_pattern").delete(
        synchronize_session=False)
    db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).delete(
        synchronize_session=False)
    if slot_snapshot:
        live = db.query(MemorySlot).filter_by(slot_name="session_patterns").first()
        if live:
            for k, v in slot_snapshot.items():
                setattr(live, k, v)
    # restore the runtime_lease row to its pre-test state
    if lease_row_snapshot == "absent":
        db.execute(text("DELETE FROM maintenance_ledger "
                        "WHERE job_name = 'runtime_lease'"))
    else:
        db.execute(text(
            "UPDATE maintenance_ledger SET last_started = :ls, last_status = :st "
            "WHERE job_name = 'runtime_lease'"),
            {"ls": lease_row_snapshot[1], "st": lease_row_snapshot[2]})
    db.commit()
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
