"""C4/C9 behavioral test — specs/C4_C9_memory_quality.md §4 checks 1–6.

C4: the evolving whole-conversation summary (job incrementality, the D3a
assembler injection condition + staleness stamp, the cross-conversation
retrieval half with privacy + self-exclusion, the C10 cascade). C9: the
procedural leg widened (intent whitelist dead, confidence floor live) and
three-tier slots (per-tier names, anchors, uniqueness, G14 cap, D6 render
order, the D7 review-arm application).

House pattern: live Postgres, uniquely-marked fixture rows, stubbed LLM +
embedder (no model loads), cleanup in `finally` — NEVER truncates. The
summary job runs with conversation_ids=… so it can never touch the live
DB's real conversations.

Run: uv run python tests/test_c4_c9.py
"""
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import (
    Conversation,
    ConversationSummary,
    EpisodicMemory,
    MemorySlot,
    ProceduralMemory,
    Project,
    ReviewQueue,
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


HEX = uuid.uuid4().hex[:6]
# letters-only marker (BM25 legs strip non-alpha)
MARK = "cfourmark" + HEX.translate(str.maketrans("0123456789", "abcdefghij"))
V1 = [1.0] + [0.0] * 383

db = SessionLocal()
conv_ids: list = []
turn_ids: list = []
pattern_ids: list = []
slot_ids: list = []
review_ids: list = []
project_id = None

_orig_min_conf = settings.procedural_min_conf


class StubEmbedder:
    def encode(self, texts, convert_to_tensor=False, **kwargs):
        import torch
        if isinstance(texts, (list, tuple)):
            vecs = torch.zeros((len(texts), 384))
            return vecs if convert_to_tensor else vecs.numpy()
        return torch.tensor(V1) if convert_to_tensor else list(V1)


def add_turn(conv, text_body, ts):
    t = EpisodicMemory(
        conversation_id=conv.id, batch_id=uuid.uuid4(), timestamp=ts,
        context_reliance="Long_Term_Memory", raw_text=text_body,
        decay_score=1.0, idempotency_key=f"c4c9-{uuid.uuid4()}")
    db.add(t)
    db.commit()
    turn_ids.append(t.id)
    return t


try:
    from src.api.prompt_assembler import assemble_prompt, conversation_summary_block
    from src.services import review as review_svc
    from src.services import slots as slots_svc
    from src.services.errors import ValidationError
    from src.workers.conversation_summary import run_conversation_summaries
    from src.workers.runtime import JOBS

    # ═══ 1. summary job: create + incremental update ═════════════════════
    print("── check 1: conversation_summary job (stub LLM) ──")
    check("job registered (JOBS + cadence + burst wiring)",
          "conversation_summary" in JOBS
          and "conversation_summary" in settings.maintenance_intervals)

    conv_a = Conversation(memory_scope_type="auto")
    conv_b = Conversation(memory_scope_type="auto")
    db.add_all([conv_a, conv_b])
    db.commit()
    conv_ids += [conv_a.id, conv_b.id]

    base_ts = datetime.now(timezone.utc) - timedelta(hours=3)
    filler = ("wordfill " * 430).strip()          # ~430 words ≈ 3.4k chars
    for i in range(8):
        add_turn(conv_a, f"User: {MARK} topic {i} {filler}\n\nAssistant: ok",
                 base_ts + timedelta(minutes=i))
    add_turn(conv_b, f"User: tiny {MARK}\n\nAssistant: ok",
             base_ts)                              # below the window

    llm_calls: list = []

    def stub_llm(prompt, max_tokens=400):
        llm_calls.append(prompt)
        return f"Summary v{len(llm_calls)} {MARK} the conversation so far."

    ids = [conv_a.id, conv_b.id]
    stats = run_conversation_summaries(db, llm=stub_llm,
                                       embedder=StubEmbedder(),
                                       conversation_ids=ids)
    row_a = db.query(ConversationSummary).filter_by(
        conversation_id=conv_a.id).first()
    row_b = db.query(ConversationSummary).filter_by(
        conversation_id=conv_b.id).first()
    check("summary row created for the window-crossing conversation",
          stats["created"] == 1 and row_a is not None
          and MARK in row_a.summary_text and row_a.covers_turns == 8
          and row_a.covers_through is not None and row_a.embedding is not None)
    check("2-turn conversation earns no row (D3a condition)",
          stats["below_window"] == 1 and row_b is None)
    first_text = row_a.summary_text
    first_through = row_a.covers_through

    n_before = len(llm_calls)
    add_turn(conv_a, f"User: {MARK} new development alpha\n\nAssistant: noted",
             base_ts + timedelta(minutes=30))
    stats2 = run_conversation_summaries(db, llm=stub_llm,
                                        embedder=StubEmbedder(),
                                        conversation_ids=ids)
    db.expire_all()
    row_a = db.query(ConversationSummary).filter_by(
        conversation_id=conv_a.id).first()
    new_prompts = llm_calls[n_before:]
    check("incremental update: old summary carried into the prompt, "
          "covers_through advances",
          stats2["updated"] == 1
          and any(first_text in p for p in new_prompts)
          and all("topic 0" not in p for p in new_prompts)
          and row_a.covers_through > first_through
          and row_a.covers_turns == 9)
    second_text = row_a.summary_text
    second_through = row_a.covers_through

    add_turn(conv_a, f"User: {MARK} another beat\n\nAssistant: ok",
             base_ts + timedelta(minutes=40))
    stats3 = run_conversation_summaries(db, llm=lambda p, max_tokens=400: "",
                                        embedder=StubEmbedder(),
                                        conversation_ids=ids)
    db.expire_all()
    row_a = db.query(ConversationSummary).filter_by(
        conversation_id=conv_a.id).first()
    check("LLM failure keeps the old row untouched (retry next burst)",
          stats3["failed"] == 1 and row_a.summary_text == second_text
          and row_a.covers_through == second_through)

    # ═══ 2. assembler injection condition + staleness stamp ══════════════
    print("── check 2: D3a injection condition ──")
    block = conversation_summary_block(db, str(conv_a.id), turn_count=10,
                                       total_tokens=500.0,
                                       recent_window_tokens=6000.0)
    check("inside the window ⇒ no injection even though a row exists",
          block is None)
    block = conversation_summary_block(db, str(conv_a.id), turn_count=10,
                                       total_tokens=9000.0,
                                       recent_window_tokens=6000.0)
    check("past the window ⇒ block injected with staleness stamp "
          "(covers 9 of 10 turns)",
          block is not None and second_text in block
          and "(as of 1 turns ago)" in block)
    messages = assemble_prompt([], [], "hello",
                               conversation_summary_text=block)
    check("assembler renders the === CONVERSATION SUMMARY === block",
          "=== CONVERSATION SUMMARY ===" in messages[0]["content"]
          and second_text in messages[0]["content"])
    check("no summary text ⇒ no block header",
          "=== CONVERSATION SUMMARY ===" not in
          assemble_prompt([], [], "hello")[0]["content"])

    # ═══ 3. retrieval consumer: cross-conversation, private, self ════════
    print("── check 3: batch-summary leg union ──")
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    conv_p = Conversation(memory_scope_type="none")     # private
    conv_c = Conversation(memory_scope_type="auto")     # the active one
    db.add_all([conv_p, conv_c])
    db.commit()
    conv_ids += [conv_p.id, conv_c.id]
    db.add(ConversationSummary(
        conversation_id=conv_p.id,
        summary_text=f"{MARK} private things happened",
        covers_turns=5, embedding=V1,
        updated_at=datetime.now(timezone.utc)))
    db.commit()

    orch = HybridRetrievalOrchestrator(db, embedder=None)
    orch._active_timescope = orch._resolve_timescope(None)
    frags = orch._batch_summary_lookup(V1, str(conv_c.id))
    texts = [f.text for f in frags]
    check("another conversation's summary found by embedding",
          any(MARK in t and "conversation summary" in t for t in texts))
    check("private conversation's summary never surfaces",
          not any("private things" in t for t in texts))
    frags_self = orch._batch_summary_lookup(V1, str(conv_a.id))
    check("active conversation's own summary excluded (assembler owns it)",
          not any(second_text in f.text for f in frags_self))
    frags_incog = orch._batch_summary_lookup(V1, str(conv_c.id),
                                             include_cross=False)
    check("incognito path (include_cross=False) reads no cross summaries",
          not any(MARK in f.text for f in frags_incog))

    # ═══ 4. procedural widening: gate dead, floor live ═══════════════════
    print("── check 4: procedural leg ──")
    now = datetime.now(timezone.utc)
    pat_hi = ProceduralMemory(
        pattern_name=f"{MARK} high", pattern_description=f"{MARK} pattern strong",
        trigger_conditions={}, reinforcement_count=3, confidence_score=0.8,
        first_observed=now, last_observed=now, is_active=True,
        source_batch_ids=[], embedding=V1)
    pat_lo = ProceduralMemory(
        pattern_name=f"{MARK} low", pattern_description=f"{MARK} pattern weak",
        trigger_conditions={}, reinforcement_count=1, confidence_score=0.2,
        first_observed=now, last_observed=now, is_active=True,
        source_batch_ids=[], embedding=V1)
    pat_trig = ProceduralMemory(
        pattern_name=f"{MARK} trig", pattern_description=f"{MARK} pattern gated",
        trigger_conditions={"intent_tags": ["Generation"]},
        reinforcement_count=3, confidence_score=0.8,
        first_observed=now, last_observed=now, is_active=True,
        source_batch_ids=[], embedding=V1)
    db.add_all([pat_hi, pat_lo, pat_trig])
    db.commit()
    pattern_ids += [pat_hi.id, pat_lo.id, pat_trig.id]

    clf = ClassificationResult(
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", raw_probs=[0.0] * 25,
        max_confidence=0.9, prompt=f"how do I usually work {MARK}")
    orch2 = HybridRetrievalOrchestrator(db, embedder=None)
    orch2._active_timescope = orch2._resolve_timescope(None)
    orch2._scope_project_id = None
    frags = orch2._procedural_lookup(V1, clf, scope=None)
    texts = [f.text for f in frags]
    check("previously-gated-out intent (Factual_Retrieval) now retrieves",
          any("pattern strong" in t for t in texts))
    check("below the confidence floor ⇒ excluded",
          not any("pattern weak" in t for t in texts))
    check("trigger machinery intact (intent condition still filters)",
          not any("pattern gated" in t for t in texts))
    settings.procedural_min_conf = 0.9
    frags = orch2._procedural_lookup(V1, clf, scope=None)
    check("raising the floor excludes the 0.8 pattern (floor is live)",
          not any("pattern strong" in f.text for f in frags))
    settings.procedural_min_conf = _orig_min_conf

    # ═══ 5. three-tier slots ═════════════════════════════════════════════
    print("── check 5: tiered slots ──")
    project = Project(name=f"c4c9test-{HEX}", slug=f"cf{HEX}",
                      roots=["/tmp/nonexistent-c4c9"])
    db.add(project)
    db.commit()
    project_id = project.id

    try:
        slots_svc.update_slot(db, "conventions", "x", scope_tier="project")
        check("project tier without project_id raises", False)
    except ValidationError as exc:
        check("project tier without project_id raises (names attachment)",
              "project" in str(exc))
    try:
        slots_svc.update_slot(db, "persona", "x", scope_tier="project",
                              project_id=str(project_id))
        check("global-only name rejected on the project tier", False)
    except ValidationError:
        check("global-only name rejected on the project tier", True)
    try:
        slots_svc.update_slot(db, "persona", "x", project_id=str(project_id))
        check("global slot with an anchor rejected", False)
    except ValidationError:
        check("global slot with an anchor rejected", True)

    s_proj = slots_svc.update_slot(db, "conventions", f"{MARK} tabs not spaces",
                                   updated_by="user", scope_tier="project",
                                   project_id=str(project_id))
    slot_ids.append(uuid.UUID(s_proj["id"]))
    s_conv = slots_svc.update_slot(db, "conversation_focus",
                                   f"{MARK} refactor the parser",
                                   updated_by="user", scope_tier="conversation",
                                   conversation_id=str(conv_c.id))
    slot_ids.append(uuid.UUID(s_conv["id"]))
    got = slots_svc.get_slot(db, "conventions", scope_tier="project",
                             project_id=str(project_id))
    check("per-tier create + read round-trip",
          got["content"] == f"{MARK} tabs not spaces"
          and got["scope_tier"] == "project"
          and got["project_id"] == str(project_id)
          and s_conv["scope_tier"] == "conversation")

    dup = MemorySlot(slot_name="conventions", content="dup", token_count=1,
                     version=1, last_updated=now, updated_by="user",
                     is_active=True, scope_tier="project",
                     project_id=project_id)
    db.add(dup)
    import sqlalchemy.exc as sa_exc
    try:
        db.commit()
        check("unique index rejects a duplicate (name, tier, anchor)", False)
        slot_ids.append(dup.id)
    except sa_exc.IntegrityError:
        db.rollback()
        check("unique index rejects a duplicate (name, tier, anchor)", True)

    long_content = "verylongword " * 400          # ≫ 300 tokens
    capped = slots_svc.update_slot(db, "conventions", long_content,
                                   scope_tier="project",
                                   project_id=str(project_id))
    check("G14: 300-token cap hard-truncates with a warning",
          capped.get("truncated") is True and "warning" in capped
          and capped["token_count"] <= 310)
    slots_svc.update_slot(db, "conventions", f"{MARK} tabs not spaces",
                          scope_tier="project", project_id=str(project_id))

    g_slot = MemorySlot(slot_name="persona", content=f"{MARK} the persona",
                        token_count=3, version=1, last_updated=now,
                        updated_by="user", is_active=True, scope_tier="global")
    other_proj_slot = MemorySlot(slot_name="guidance", content="other project",
                                 token_count=2, version=1, last_updated=now,
                                 updated_by="user", is_active=True,
                                 scope_tier="project", project_id=uuid.uuid4())
    proj_slot = db.query(MemorySlot).filter_by(id=uuid.UUID(s_proj["id"])).first()
    conv_slot = db.query(MemorySlot).filter_by(id=uuid.UUID(s_conv["id"])).first()
    msgs = assemble_prompt(
        [conv_slot, other_proj_slot, proj_slot, g_slot], [], "hello",
        conversation_id=str(conv_c.id),
        scope={"project_id": str(project_id)})
    sysc = msgs[0]["content"]
    check("assembler renders three tiers in order with tier prefixes",
          "[PERSONA]" in sysc
          and "[PROJECT · CONVENTIONS]" in sysc
          and "[CONVERSATION · CONVERSATION_FOCUS]" in sysc
          and sysc.index("[PERSONA]") < sysc.index("[PROJECT · CONVENTIONS]")
          < sysc.index("[CONVERSATION · CONVERSATION_FOCUS]"))
    check("another project's slot never renders", "other project" not in sysc)
    msgs2 = assemble_prompt([conv_slot, proj_slot, g_slot], [], "hello")
    check("unattached conversation gets global only",
          "[PERSONA]" in msgs2[0]["content"]
          and "PROJECT · " not in msgs2[0]["content"]
          and "CONVERSATION · " not in msgs2[0]["content"])

    # D7: review-arm applies through the service, recording the proposer
    item = ReviewQueue(item_type="memory_slot_update", item_content={
        "slot_name": "conventions", "proposed_content": f"{MARK} agent says",
        "proposed_by": "agent", "scope_tier": "project",
        "project_id": str(project_id)})
    db.add(item)
    db.commit()
    review_ids.append(item.id)
    review_svc.approve(db, str(item.id))
    applied = slots_svc.get_slot(db, "conventions", scope_tier="project",
                                 project_id=str(project_id))
    check("D7: approval applies via the slots service, updated_by=proposer",
          applied["content"] == f"{MARK} agent says"
          and applied["updated_by"] == "agent")

    # ═══ 6. C10 hook: conversation delete cascades the summary ═══════════
    print("── check 6: cascade delete ──")
    conv_d = Conversation(memory_scope_type="auto")
    db.add(conv_d)
    db.commit()
    db.add(ConversationSummary(conversation_id=conv_d.id,
                               summary_text="doomed", covers_turns=1,
                               updated_at=now))
    db.commit()
    db.delete(conv_d)
    db.commit()
    gone = db.query(ConversationSummary).filter_by(
        conversation_id=conv_d.id).first()
    check("deleting a conversation removes its summary row (CASCADE)",
          gone is None)

finally:
    settings.procedural_min_conf = _orig_min_conf
    try:
        db.rollback()
        if review_ids:
            db.execute(text("DELETE FROM review_queue WHERE id = ANY(:ids)"),
                       {"ids": review_ids})
        if slot_ids:
            db.execute(text("DELETE FROM memory_slots WHERE id = ANY(:ids)"),
                       {"ids": slot_ids})
        if pattern_ids:
            db.execute(text("DELETE FROM procedural_memory WHERE id = ANY(:ids)"),
                       {"ids": pattern_ids})
        if turn_ids:
            db.execute(text("DELETE FROM episodic_memory WHERE id = ANY(:ids)"),
                       {"ids": turn_ids})
        if conv_ids:
            db.execute(text("DELETE FROM conversation_summaries "
                            "WHERE conversation_id = ANY(:ids)"), {"ids": conv_ids})
            db.execute(text("DELETE FROM conversations WHERE id = ANY(:ids)"),
                       {"ids": conv_ids})
        if project_id is not None:
            db.execute(text("DELETE FROM memory_slots WHERE project_id = :pid"),
                       {"pid": project_id})
            db.execute(text("DELETE FROM projects WHERE id = :pid"),
                       {"pid": project_id})
        db.commit()
    finally:
        db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
