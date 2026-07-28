"""C10/C11 behavioral test — specs/C10_C11_deletion_chat_control.md §4 checks 1–9.

C10: the manifest-first deletion cascade (sole-support vs corroborated codex
edges, entity husking, the full per-store sweep, dry_run identity, the T4
timeline exclusion for `source_deleted`, review-item staling, mid-generation
refusal). C11: the deterministic slash-command parser over the E0 services
(/remember tiers + journaling, /slots, /search with zero LLM, /forget's
review-queue proposal + the D6 apply arm, unknown-command help hint, the
/delete-conversation two-step confirm).

House pattern: live Postgres, uniquely-marked fixture rows, stubbed
classifier/embedder core (no model loads beyond codex_extractor's import),
cleanup in `finally` — NEVER truncates; deletion runs ONLY against fixture
conversations. The /forget proposal created through the command path is
never approved (its fuzzy top-5 may list live rows); the apply arm is
exercised with a hand-built fixture-only item instead.

Run: uv run python tests/test_c10_c11.py
"""
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import (
    BatchSummary,
    CodexEdge,
    CodexEntity,
    CodexEvent,
    ColdStorage,
    ContextCluster,
    Conversation,
    ConversationSummary,
    CuratedLabel,
    Decision,
    EpisodicChunk,
    EpisodicClusterLink,
    EpisodicMemory,
    MemorySlot,
    ProceduralMemory,
    Project,
    ReviewQueue,
    SessionReplay,
    SessionSummary,
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


HEX = uuid.uuid4().hex[:6].translate(str.maketrans("0123456789", "abcdefghij"))
MARK = "ctenmark" + HEX          # letters-only (BM25 legs strip non-alpha)
EMB = [0.05] * 1024
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


class FakeCore:
    classifier = StubClassifier()
    runtime = None

    async def stop(self):
        pass


class BoobyTrappedClient:
    """/search's no-LLM contract: any completion attempt trips the trap."""
    tripped = False

    @property
    def chat(self):
        BoobyTrappedClient.tripped = True
        raise AssertionError("LLM client invoked during a chat command")


db = SessionLocal()

import src.api.core as core_mod
import src.retrieval.orchestrator as orch_mod
import src.workers.runtime as runtime_mod
from src.api.chat_commands import _PENDING_DELETES, try_handle
from src.retrieval.evolution import build_entity_timeline, history_exists
from src.services import conversations as conversations_svc
from src.services import review as review_svc

_orig_get_bg_client = orch_mod.get_bg_client
_orig_active_core = core_mod._active_core
_orig_active_runtime = runtime_mod._active_runtime

# fixture ids (filled as created; cleanup deletes by these — never truncates)
conv_A = conv_B = conv_C = None
conv_id_values: list = []
bA1, bA2, bB1, bS, bF = (uuid.uuid4() for _ in range(5))
turn_ids: list = []
cluster_ids: list = []
entity_ids: list = []
edge_ids: list = []
proc_ids: list = []
review_item_ids: list = []
cold_id = uuid.uuid4()
project_id = None
decision_id = None
slot_ids: list = []
pending_items_snapshot = None
pending_items_created = False

try:
    # ═══ Fixtures ════════════════════════════════════════════════════════
    conv_A = Conversation(memory_scope_type="auto")
    conv_B = Conversation(memory_scope_type="auto")
    conv_C = Conversation(memory_scope_type="auto")
    db.add_all([conv_A, conv_B, conv_C])
    db.commit()
    # Ids are captured as plain values AT creation — instances of rows the
    # cascade deletes raise ObjectDeletedError on attribute access after the
    # service's commit (the house SQLAlchemy trap).
    conv_A_id, conv_B_id, conv_C_id = conv_A.id, conv_B.id, conv_C.id
    conv_id_values.extend([conv_A_id, conv_B_id, conv_C_id])

    def add_turn(conv, batch, text_body, ts, embedding=None):
        t = EpisodicMemory(
            conversation_id=conv.id, batch_id=batch, timestamp=ts,
            context_reliance="Zero_Shot", raw_text=text_body,
            embedding=embedding, inject_raw=True,
            idempotency_key=f"{MARK}-{uuid.uuid4().hex[:8]}",
        )
        db.add(t)
        db.flush()
        turn_ids.append(t.id)
        return t

    tA1_id = add_turn(conv_A, bA1, f"{MARK}aone drafted the saga plan",
                      NOW - timedelta(days=2)).id
    tA2_id = add_turn(conv_A, bA2, f"{MARK}atwo refined the saga plan",
                      NOW - timedelta(days=1)).id
    tB1_id = add_turn(conv_B, bB1, f"{MARK}bone corroborated the ending",
                      NOW - timedelta(days=1)).id
    t_search_id = add_turn(conv_C, bS,
                           f"{MARK}sear the flux capacitor design decision",
                           NOW - timedelta(minutes=30), embedding=EMB).id
    t_forget_id = add_turn(conv_C, bF,
                           f"{MARK}forg the abandoned experiment note",
                           NOW - timedelta(minutes=20), embedding=EMB).id
    db.add(EpisodicChunk(turn_id=tA1_id, chunk_index=0,
                         chunk_text=f"{MARK} chunk body"))

    cl_A = ContextCluster(name=f"{MARK}_clA", conversation_id=conv_A_id)
    cl_shared = ContextCluster(name=f"{MARK}_clShared", conversation_id=conv_A_id)
    db.add_all([cl_A, cl_shared])
    db.flush()
    cl_A_id, cl_shared_id = cl_A.id, cl_shared.id
    cluster_ids.extend([cl_A_id, cl_shared_id])
    db.add_all([
        EpisodicClusterLink(episodic_id=tA1_id, cluster_id=cl_A_id),
        EpisodicClusterLink(episodic_id=tA2_id, cluster_id=cl_A_id),
        EpisodicClusterLink(episodic_id=tA1_id, cluster_id=cl_shared_id),
        EpisodicClusterLink(episodic_id=tB1_id, cluster_id=cl_shared_id),
    ])

    db.add(ColdStorage(id=cold_id, raw_text=f"{MARK} frozen turn",
                       timestamp=NOW - timedelta(days=40),
                       conversation_id=conv_A.id))
    db.add(BatchSummary(conversation_id=conv_A.id, start_turn_index=0,
                        end_turn_index=1, summary_text=f"{MARK} batch summary"))
    db.add(ConversationSummary(conversation_id=conv_A.id,
                               summary_text=f"{MARK} whole-conversation summary",
                               covers_turns=2))
    db.add(SessionReplay(conversation_id=conv_A.id, event_sequence=[]))
    db.add(SessionSummary(conversation_id=conv_A.id))
    slot_A = MemorySlot(slot_name="conversation_focus", scope_tier="conversation",
                        conversation_id=conv_A.id, content=f"{MARK} focus",
                        token_count=2, version=1, updated_by="user",
                        is_active=True)
    db.add(slot_A)
    db.flush()
    slot_ids.append(slot_A.id)
    db.add(CuratedLabel(batch_id=bA1, prompt=f"{MARK} label",
                        corrected_context_reliance="Zero_Shot"))
    db.add(CuratedLabel(batch_id=bF, prompt=f"{MARK} forget label",
                        corrected_context_reliance="Zero_Shot"))

    p1 = ProceduralMemory(pattern_name=f"{MARK}_p1", source_batch_ids=[bA1, bB1],
                          confidence_score=0.5, is_active=True)
    p2 = ProceduralMemory(pattern_name=f"{MARK}_p2", source_batch_ids=[bA2],
                          confidence_score=0.5, is_active=True)
    db.add_all([p1, p2])
    db.flush()
    p1_id, p2_id = p1.id, p2.id
    proc_ids.extend([p1_id, p2_id])

    proj = Project(name=f"{MARK}_proj", slug=f"{MARK}-proj", roots=["/tmp/none"])
    db.add(proj)
    db.flush()
    project_id = proj.id
    dec = Decision(project_id=proj.id, decision=f"{MARK} decided things",
                   source_batch=bA1)
    db.add(dec)
    db.flush()
    decision_id = dec.id

    # Codex: X sole-support (event from A only), Y corroborated (event from
    # B too), Z legacy (no events at all — spec §3 edge case).
    e_solo = CodexEntity(canonical_name=f"{MARK}_solo", source="conversation")
    e_hub = CodexEntity(canonical_name=f"{MARK}_hub", source="conversation")
    e_other = CodexEntity(canonical_name=f"{MARK}_other", source="conversation")
    e_leg = CodexEntity(canonical_name=f"{MARK}_leg", source="conversation")
    e_f1 = CodexEntity(canonical_name=f"{MARK}fone", source="conversation")
    e_f2 = CodexEntity(canonical_name=f"{MARK}ftwo", source="conversation")
    db.add_all([e_solo, e_hub, e_other, e_leg, e_f1, e_f2])
    db.flush()
    e_solo_id, e_hub_id, e_other_id = e_solo.id, e_hub.id, e_other.id
    e_leg_id, e_f1_id, e_f2_id = e_leg.id, e_f1.id, e_f2.id
    entity_ids.extend([e_solo_id, e_hub_id, e_other_id,
                       e_leg_id, e_f1_id, e_f2_id])

    edge_X = CodexEdge(source_id=e_solo_id, target_id=e_hub_id,
                       relation="planned_as", source_batch=bA1)
    edge_Y = CodexEdge(source_id=e_hub_id, target_id=e_other_id,
                       relation="ends_with", source_batch=bA1)
    edge_Z = CodexEdge(source_id=e_leg_id, target_id=e_hub_id,
                       relation="references", source_batch=bA2)
    edge_W = CodexEdge(source_id=e_f1_id, target_id=e_f2_id,
                       relation="uses", source_batch=uuid.uuid4())
    db.add_all([edge_X, edge_Y, edge_Z, edge_W])
    db.flush()
    edge_X_id, edge_Y_id = edge_X.id, edge_Y.id
    edge_Z_id, edge_W_id = edge_Z.id, edge_W.id
    edge_ids.extend([edge_X_id, edge_Y_id, edge_Z_id, edge_W_id])
    db.add_all([
        CodexEvent(entity_id=e_solo_id, event_type="edge_added",
                   payload={"edge_id": str(edge_X_id), "relation": "planned_as",
                            "target_id": str(e_hub_id)}, batch_source=bA1),
        CodexEvent(entity_id=e_hub_id, event_type="edge_added",
                   payload={"edge_id": str(edge_Y_id), "relation": "ends_with",
                            "target_id": str(e_other_id)}, batch_source=bA1),
        CodexEvent(entity_id=e_hub_id, event_type="edge_added",
                   payload={"edge_id": str(edge_Y_id), "relation": "ends_with",
                            "target_id": str(e_other_id)}, batch_source=bB1),
    ])

    recon_item = ReviewQueue(item_type="codex_reconciliation", item_content={
        "old_edge_id": str(edge_X_id), "conflict_type": "supersession"})
    db.add(recon_item)
    db.flush()
    recon_item_id = recon_item.id
    review_item_ids.append(recon_item_id)
    db.commit()

    # ═══ 3) dry_run deletes nothing; manifest identical ═══════════════════
    print("\n── C10: dry run ──")
    manifest_dry = conversations_svc.delete_conversation(
        db, str(conv_A_id), dry_run=True)
    check("dry_run leaves the conversation row",
          db.query(Conversation).filter_by(id=conv_A_id).count() == 1)
    check("dry_run leaves the turns",
          db.query(EpisodicMemory).filter_by(
              conversation_id=conv_A_id).count() == 2)
    db.expire_all()
    edge_X_row = db.query(CodexEdge).filter_by(id=edge_X_id).first()
    check("dry_run leaves edge X live", edge_X_row.valid_until is None)
    check("dry_run manifest flagged", manifest_dry["dry_run"] is True)

    # ═══ 1+2) real deletion: cascade semantics + counts ═══════════════════
    print("\n── C10: deletion cascade ──")
    manifest = conversations_svc.delete_conversation(
        db, str(conv_A_id), dry_run=False)
    d_dry = {k: v for k, v in manifest_dry.items() if k != "dry_run"}
    d_real = {k: v for k, v in manifest.items() if k != "dry_run"}
    check("manifest identical dry vs real", d_dry == d_real)

    db.expire_all()
    edge_X_row = db.query(CodexEdge).filter_by(id=edge_X_id).first()
    edge_Y_row = db.query(CodexEdge).filter_by(id=edge_Y_id).first()
    edge_Z_row = db.query(CodexEdge).filter_by(id=edge_Z_id).first()
    check("sole-support edge X expired", edge_X_row.valid_until is not None)
    x_event = db.query(CodexEvent).filter_by(event_type="edge_expired").filter(
        CodexEvent.payload["edge_id"].astext == str(edge_X_id)).first()
    check("X expiry journaled reason=source_deleted",
          x_event is not None
          and x_event.payload.get("reason") == "source_deleted"
          and x_event.payload.get("source") == "user_deletion")
    check("legacy eventless edge Z expired (own provenance only)",
          edge_Z_row.valid_until is not None)
    check("corroborated edge Y alive", edge_Y_row.valid_until is None)

    e_solo_row = db.query(CodexEntity).filter_by(id=e_solo_id).first()
    e_hub_row = db.query(CodexEntity).filter_by(id=e_hub_id).first()
    e_leg_row = db.query(CodexEntity).filter_by(id=e_leg_id).first()
    check("orphaned entity husked (solo)",
          (e_solo_row.properties or {}).get("deleted_reason") == "source_deleted"
          and "[deleted:" in e_solo_row.canonical_name
          and e_solo_row.embedding is None)
    check("entity_expired event journaled",
          db.query(CodexEvent).filter_by(
              entity_id=e_solo_id, event_type="entity_expired").count() == 1)
    check("legacy-edge entity husked too",
          (e_leg_row.properties or {}).get("deleted_reason") == "source_deleted")
    check("hub entity with surviving edge intact",
          e_hub_row.canonical_name == f"{MARK}_hub"
          and (e_hub_row.properties or {}).get("deleted_reason") is None)
    check("conversation B untouched",
          db.query(EpisodicMemory).filter_by(
              conversation_id=conv_B_id).count() == 1)

    check("episodic rows gone",
          db.query(EpisodicMemory).filter_by(
              conversation_id=conv_A_id).count() == 0)
    check("chunks gone (CASCADE)",
          db.query(EpisodicChunk).filter_by(turn_id=tA1_id).count() == 0)
    check("cluster links gone",
          db.query(EpisodicClusterLink).filter(
              EpisodicClusterLink.episodic_id.in_([tA1_id, tA2_id])).count() == 0)
    check("cold row gone",
          db.query(ColdStorage).filter_by(id=cold_id).count() == 0)
    check("batch summary gone",
          db.query(BatchSummary).filter_by(
              conversation_id=conv_A_id).count() == 0)
    check("conversation summary gone (C4 hook)",
          db.query(ConversationSummary).filter_by(
              conversation_id=conv_A_id).count() == 0)
    check("session replays + summaries gone",
          db.query(SessionReplay).filter_by(conversation_id=conv_A_id).count() == 0
          and db.query(SessionSummary).filter_by(
              conversation_id=conv_A_id).count() == 0)
    check("conversation-tier slot gone",
          db.query(MemorySlot).filter_by(
              conversation_id=conv_A_id).count() == 0)
    check("curated labels gone (training hygiene)",
          db.query(CuratedLabel).filter_by(batch_id=bA1).count() == 0)
    check("empty cluster deleted",
          db.query(ContextCluster).filter_by(id=cl_A_id).count() == 0)
    shared_row = db.query(ContextCluster).filter_by(id=cl_shared_id).first()
    check("shared cluster survives, detached",
          shared_row is not None and shared_row.conversation_id is None
          and db.query(EpisodicClusterLink).filter_by(
              cluster_id=cl_shared_id, episodic_id=tB1_id).count() == 1)
    p1_row = db.query(ProceduralMemory).filter_by(id=p1_id).first()
    p2_row = db.query(ProceduralMemory).filter_by(id=p2_id).first()
    check("procedural pruned keeps other-batch evidence",
          p1_row.is_active and [b for b in p1_row.source_batch_ids] == [bB1])
    check("procedural emptied → deactivated",
          not p2_row.is_active and p2_row.source_batch_ids == [])
    dec_row = db.query(Decision).filter_by(id=decision_id).first()
    check("decision from deleted conversation expired",
          dec_row.valid_until is not None)
    recon_row = db.query(ReviewQueue).filter_by(id=recon_item_id).first()
    check("pending review item → stale", recon_row.status == "stale")
    check("conversation row gone last",
          db.query(Conversation).filter_by(id=conv_A_id).count() == 0)

    m = manifest
    check("manifest counts match",
          m["batches"] == 2
          and m["deleted"]["episodic_turns"] == 2
          and m["deleted"]["episodic_chunks"] == 1
          and m["deleted"]["cluster_links"] == 3
          and m["deleted"]["empty_clusters"] == 1
          and m["deleted"]["clusters_detached"] == 1
          and m["deleted"]["cold_storage_rows"] == 1
          and m["deleted"]["batch_summaries"] == 1
          and m["deleted"]["conversation_summaries"] == 1
          and m["deleted"]["session_replays"] == 1
          and m["deleted"]["session_summaries"] == 1
          and m["deleted"]["conversation_slots"] == 1
          and m["deleted"]["curated_labels"] == 1
          and m["codex"]["edges_expired"] == 2
          and m["codex"]["edges_kept_corroborated"] == 1
          and m["codex"]["entities_expired"] == 2
          and m["decisions_expired"] == 1
          and m["procedural_pruned"] == 2
          and m["procedural_deactivated"] == 1
          and m["review_items_staled"] == 1)
    check("manifest carries the G25 logs caveat",
          "Logs are NOT touched" in m["logs_caveat"])

    # ═══ 4) T4 timeline: deletion is not evolution ════════════════════════
    print("\n── C10: timeline exclusion ──")
    check("history_exists False for deletion-only expiry",
          history_exists(db, e_solo_id) is False
          and history_exists(db, e_hub_id) is False)
    tl = build_entity_timeline(db, e_solo_row)
    check("no timeline built from source_deleted expiries", tl is None)

    # mid-generation refusal (spec §3 / traps)
    rt = runtime_mod.MaintenanceRuntime()
    rt._generation_inflight = 1
    runtime_mod._active_runtime = rt
    try:
        conversations_svc.delete_conversation(db, str(conv_B_id), dry_run=False)
        check("mid-generation deletion refused", False)
    except Exception as e:
        check("mid-generation deletion refused",
              "generation" in str(e).lower())
        db.rollback()
    finally:
        runtime_mod._active_runtime = None

    # ═══ C11: chat commands ═══════════════════════════════════════════════
    print("\n── C11: commands ──")
    core_mod._active_core = FakeCore()
    orch_mod.get_bg_client = lambda: BoobyTrappedClient()
    conv_C_row = db.query(Conversation).filter_by(id=conv_C_id).first()

    # snapshot the LIVE global pending_items slot (restored in finally)
    live_slot = db.query(MemorySlot).filter_by(
        slot_name="pending_items", scope_tier="global",
        project_id=None, conversation_id=None).first()
    if live_slot:
        pending_items_snapshot = {c: getattr(live_slot, c) for c in
                                  ("content", "token_count", "version",
                                   "last_updated", "updated_by", "is_active")}
    else:
        pending_items_created = True

    # 5) /remember — default global tier + journaling. The live
    # pending_items slot may already sit past the G14 cap (legacy content),
    # in which case the append is truncated away and the confirmation says
    # so — assert content-or-warning, never live-data-dependent content.
    r = try_handle(db, None, conv_C_row, f"/remember check the {MARK} rig", {})
    slot_row = db.query(MemorySlot).filter_by(
        slot_name="pending_items", scope_tier="global",
        project_id=None, conversation_id=None).first()
    check("/remember writes global pending_items (or reports the cap)",
          r["handled"] and slot_row is not None
          and "global · pending_items" in r["text"]
          and (f"{MARK} rig" in (slot_row.content or "")
               or "truncated" in r["text"].lower()))
    check("/remember journals updated_by=chat_command",
          slot_row.updated_by == "chat_command")

    # 5b) /remember @conversation with `in <slot>` modifier
    r = try_handle(db, None, conv_C_row,
                   f"/remember {MARK} deep focus in conversation_focus @conversation", {})
    conv_slot = db.query(MemorySlot).filter_by(
        slot_name="conversation_focus", scope_tier="conversation",
        conversation_id=conv_C_id).first()
    check("/remember @conversation writes the conversation tier",
          conv_slot is not None and f"{MARK} deep focus" in conv_slot.content
          and conv_slot.updated_by == "chat_command")
    if conv_slot:
        slot_ids.append(conv_slot.id)

    # 5c) /slots renders (content assertions ride the fixture-owned
    # conversation tier; the global tier only needs to render structurally)
    r = try_handle(db, None, conv_C_row, "/slots", {})
    check("/slots renders tiers",
          "[global · pending_items]" in r["text"]
          and "[conversation · conversation_focus]" in r["text"])
    r = try_handle(db, None, conv_C_row, "/slots conversation", {})
    check("/slots <tier> filters", f"{MARK} deep focus" in r["text"]
          and "[global" not in r["text"])

    # /bookmark — last stored turn of this conversation
    r = try_handle(db, None, conv_C_row, "/bookmark", {})
    db.expire_all()
    t_forget_row = db.query(EpisodicMemory).filter_by(id=t_forget_id).first()
    check("/bookmark flags the latest stored turn",
          "Bookmarked" in r["text"] and t_forget_row.is_bookmarked
          and t_forget_row.decay_immune)

    # 6) /search — dated fragments, deterministic, zero LLM
    r = try_handle(db, None, conv_C_row, f"/search {MARK}sear capacitor", {})
    check("/search returns the seeded memory",
          r["handled"] and f"{MARK}sear" in r["text"])
    check("/search fragments are date-stamped",
          re.search(r"\[\d{4}-\d{2}-\d{2}\]", r["text"]) is not None)
    check("/search never invoked an LLM", BoobyTrappedClient.tripped is False)

    # 7) /forget queues a proposal; the D6 arm applies a fixture-only item
    r = try_handle(db, None, conv_C_row,
                   f"/forget the {MARK}forg note about {MARK}fone", {})
    forget_item = db.query(ReviewQueue).filter_by(
        item_type="forget_request", status="pending").order_by(
        ReviewQueue.created_at.desc()).first()
    forget_item_id = forget_item.id if forget_item else None
    if forget_item_id:
        review_item_ids.append(forget_item_id)
    listed_turns = {t["id"] for t in
                    (forget_item.item_content.get("turns", [])
                     if forget_item else [])}
    listed_edges = {e["id"] for e in
                    (forget_item.item_content.get("edges", [])
                     if forget_item else [])}
    check("/forget queues, never applies",
          forget_item is not None and "queued" in r["text"].lower()
          and db.query(EpisodicMemory).filter_by(id=t_forget_id).count() == 1)
    check("/forget lists matching turns + edges",
          str(t_forget_id) in listed_turns and str(edge_W_id) in listed_edges)

    apply_item = ReviewQueue(item_type="forget_request", item_content={
        "query": "fixture", "conversation_id": str(conv_C_id),
        "turns": [{"id": str(t_forget_id)}],
        "edges": [{"id": str(edge_W_id)}],
        "proposed_by": "chat_command"})
    db.add(apply_item)
    db.flush()
    apply_item_id = apply_item.id
    db.commit()
    review_item_ids.append(apply_item_id)
    review_svc.approve(db, str(apply_item_id))
    db.expire_all()
    edge_W_row = db.query(CodexEdge).filter_by(id=edge_W_id).first()
    w_event = db.query(CodexEvent).filter_by(event_type="edge_expired").filter(
        CodexEvent.payload["edge_id"].astext == str(edge_W_id)).first()
    check("forget approval deletes the turn via the D6 dispatch",
          db.query(EpisodicMemory).filter_by(id=t_forget_id).count() == 0)
    check("forget approval expires the edge, journaled user_forget",
          edge_W_row.valid_until is not None and w_event is not None
          and w_event.payload.get("reason") == "user_forget")
    check("forget approval clears the batch's curated labels",
          db.query(CuratedLabel).filter_by(batch_id=bF).count() == 0)

    # 8) unknown command → help hint, never a prompt
    r = try_handle(db, None, conv_C_row, "/frobnicate all the things", {})
    check("unknown command gets the help hint",
          r is not None and r["handled"] and "Unknown command" in r["text"]
          and "/help" in r["text"])
    check("plain chat text is NOT handled",
          try_handle(db, None, conv_C_row, "hello there", {}) is None)

    # 9) /delete-conversation two-step confirm
    print("\n── C11: /delete-conversation ──")
    r = try_handle(db, None, conv_C_row, "/delete-conversation confirm", {})
    check("bare confirm with nothing pending refused",
          "Nothing pending" in r["text"]
          and db.query(Conversation).filter_by(id=conv_C_id).count() == 1)
    r = try_handle(db, None, conv_C_row, "/delete-conversation", {})
    check("first step renders the dry-run manifest",
          "DRY RUN" in r["text"] and "Logs are NOT touched" in r["text"]
          and db.query(Conversation).filter_by(id=conv_C_id).count() == 1)
    r = try_handle(db, None, conv_C_row, "/delete-conversation confirm", {})
    check("confirm within TTL deletes",
          "Conversation deleted" in r["text"]
          and db.query(Conversation).filter_by(id=conv_C_id).count() == 0)
    db.expire_all()
    forget_item_row = db.query(ReviewQueue).filter_by(
        id=forget_item_id).first() if forget_item_id else None
    check("pending forget proposal from the deleted conversation → stale",
          forget_item_row is not None and forget_item_row.status == "stale")
    # In production the next request would auto-recreate the conversation
    # row (main.py's G26 block); a lightweight stand-in mirrors that here —
    # the deleted ORM instance itself would raise ObjectDeletedError.
    import types
    conv_C_shim = types.SimpleNamespace(id=conv_C_id, project_id=None,
                                        cluster_ids=None,
                                        included_conversation_ids=None)
    r = try_handle(db, None, conv_C_shim, "/delete-conversation confirm", {})
    check("double confirm is an idempotent refusal",
          "Nothing pending" in r["text"])

finally:
    db.rollback()
    core_mod._active_core = _orig_active_core
    runtime_mod._active_runtime = _orig_active_runtime
    orch_mod.get_bg_client = _orig_get_bg_client
    _PENDING_DELETES.clear()
    try:
        # restore/clean the live global pending_items slot
        live_slot = db.query(MemorySlot).filter_by(
            slot_name="pending_items", scope_tier="global",
            project_id=None, conversation_id=None).first()
        if live_slot is not None and pending_items_snapshot is not None:
            for k, v in pending_items_snapshot.items():
                setattr(live_slot, k, v)
        elif live_slot is not None and pending_items_created:
            db.delete(live_slot)
        db.commit()

        for rid in review_item_ids:
            db.query(ReviewQueue).filter_by(id=rid).delete(
                synchronize_session=False)
        db.query(CuratedLabel).filter(
            CuratedLabel.batch_id.in_([bA1, bA2, bB1, bS, bF])).delete(
            synchronize_session=False)
        if entity_ids:
            db.query(CodexEvent).filter(
                CodexEvent.entity_id.in_(entity_ids)).delete(
                synchronize_session=False)
        if edge_ids:
            db.query(CodexEdge).filter(CodexEdge.id.in_(edge_ids)).delete(
                synchronize_session=False)
        if entity_ids:
            db.query(CodexEntity).filter(
                CodexEntity.id.in_(entity_ids)).delete(
                synchronize_session=False)
        if decision_id:
            db.query(Decision).filter_by(id=decision_id).delete(
                synchronize_session=False)
        if project_id:
            db.query(Project).filter_by(id=project_id).delete(
                synchronize_session=False)
        if proc_ids:
            db.query(ProceduralMemory).filter(
                ProceduralMemory.id.in_(proc_ids)).delete(
                synchronize_session=False)
        for sid in slot_ids:
            db.query(MemorySlot).filter_by(id=sid).delete(
                synchronize_session=False)
        if turn_ids:
            db.query(EpisodicClusterLink).filter(
                EpisodicClusterLink.episodic_id.in_(turn_ids)).delete(
                synchronize_session=False)
            db.query(EpisodicMemory).filter(
                EpisodicMemory.id.in_(turn_ids)).delete(
                synchronize_session=False)
        db.query(ColdStorage).filter_by(id=cold_id).delete(
            synchronize_session=False)
        if cluster_ids:
            db.query(ContextCluster).filter(
                ContextCluster.id.in_(cluster_ids)).delete(
                synchronize_session=False)
        for cid in conv_id_values:
            db.query(BatchSummary).filter_by(
                conversation_id=cid).delete(synchronize_session=False)
            db.query(ConversationSummary).filter_by(
                conversation_id=cid).delete(synchronize_session=False)
            db.query(SessionReplay).filter_by(
                conversation_id=cid).delete(synchronize_session=False)
            db.query(SessionSummary).filter_by(
                conversation_id=cid).delete(synchronize_session=False)
            db.query(MemorySlot).filter_by(
                conversation_id=cid).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(
                conversation_id=cid).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(
                synchronize_session=False)
        db.commit()
    except Exception as cleanup_err:  # noqa: BLE001 — report, never mask checks
        db.rollback()
        print(f"  CLEANUP ERROR: {cleanup_err}")
    db.close()

print(f"\n{'='*50}\n  {_passed} passed, {_failed} failed\n{'='*50}")
sys.exit(1 if _failed else 0)
