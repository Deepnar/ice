"""D1/D2 maintenance-agent behavioral test — specs/D1_D2_maintenance_agent.md
§5 checks 1–10.

Runs against the live Postgres (docker up). Inserts its own uniquely-marked
rows and deletes them in `finally` — NEVER truncates (the dev DB holds real
data). The LLM decider is always a stub. Isolation strategy: detector checks
call the detector against the live DB and process ONLY the fixture items
(real rows may be detected but are never acted on); full-run checks (caps)
monkeypatch the DETECTORS registry so run_maintenance_agent sees fixture
items only. The one real slot touched (pending_items) is snapshot-and-
restored. Fixture entities carry a unique entity_type so the cosine channel
can never pair them with real entities.

Run: uv run python tests/test_maintenance_agent.py
"""
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import (
    CodexEdge,
    CodexEntity,
    CodexEvent,
    MemorySlot,
    ReviewQueue,
)
from src.services import review
from src.workers import maintenance_agent as ma

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


# lowercase: entity canonical_names are stored casefolded
MARK = f"magent{uuid.uuid4().hex[:6]}"
ETYPE = f"t{MARK}"          # unique type: cosine channel can't cross into real data
NOW = datetime.now(timezone.utc)

# near-identical unit vectors (cos ≈ 0.99) orthogonal to each other's pair
V1 = [1.0] + [0.0] * 383
V2 = [0.99, 0.14] + [0.0] * 382
V3 = [0.0, 0.0, 1.0] + [0.0] * 381
V4 = [0.0, 0.0, 0.99, 0.14] + [0.0] * 380

db = SessionLocal()

ent_ids: list = []        # every fixture entity (cleanup)
rq_ids: list = []         # review rows created directly by the test
run_ids: list = []        # agent_run_ids used (cleanup of proposals + events)
slot_snapshot = "absent"  # pending_items snapshot ("absent" | tuple)
orig_detectors = dict(ma.DETECTORS)
orig_dup_cap = ma.DUP_PAIRS_PER_RUN


def mkent(name, aliases=None, emb=None, desc="", n_events=0):
    e = CodexEntity(id=uuid.uuid4(), canonical_name=name,
                    aliases=aliases if aliases is not None else [name],
                    tags=[], entity_type=ETYPE, description=desc,
                    properties={}, context_payload="", embedding=emb)
    db.add(e)
    ent_ids.append(e.id)
    db.flush()   # no relationship() on the models — entities land before edges
    for i in range(n_events):
        db.add(CodexEvent(entity_id=e.id, event_type="edge_added",
                          payload={"fixture": MARK},
                          timestamp=NOW - timedelta(days=n_events - i),
                          batch_source=uuid.uuid4()))
    return e


def mkedge(src, tgt, relation, confidence="active", negated=False,
           strength=1.0, extraction_confidence=1.0, valid_from=None):
    e = CodexEdge(id=uuid.uuid4(), source_id=src.id, target_id=tgt.id,
                  relation=relation, strength=strength,
                  source_batch=uuid.uuid4(), confidence=confidence,
                  extraction_confidence=extraction_confidence, negated=negated,
                  valid_from=valid_from or NOW)
    db.add(e)
    return e


def newrun():
    rid = uuid.uuid4()
    run_ids.append(rid)
    return rid


def agent_events(run_id):
    return db.execute(text(
        "SELECT event_type, payload FROM codex_events "
        "WHERE batch_source = :b"), {"b": run_id}).fetchall()


try:
    # ═══ 1. Tier-0 exact-dup merge (normalization-equal aliases) ═══════════
    print("── 1. Tier-0 auto-merge: exact-duplicate entities ──")
    a = mkent(f"{MARK} dupa", aliases=[f"{MARK} DUPX"], desc="short", n_events=3)
    b = mkent(f"{MARK} dupb", aliases=[f"{MARK} dupx"],
              desc="a much longer note body that should win the merge", n_events=1)
    c = mkent(f"{MARK} target c")
    d = mkent(f"{MARK} target d")
    mkedge(a, c, "uses", strength=2.0)
    dup_edge = mkedge(b, c, "uses", strength=5.0, extraction_confidence=0.9)
    uniq_edge = mkedge(b, d, "depends_on")
    loop_edge = mkedge(b, a, "related_to")
    db.commit()
    a_id, b_id, dup_id, uniq_id, loop_id = a.id, b.id, dup_edge.id, uniq_edge.id, loop_edge.id

    ma.DUP_PAIRS_PER_RUN = 500   # the live DB may outrank fixtures at cap 10
    items = ma._detect_duplicate_entities(db, 500)
    key_ab = ma._pair_key(a_id, b_id)
    mine = [i for i in items if i.payload["pair_key"] == key_ab]
    check("detector 3 finds the alias-overlap pair", len(mine) == 1)
    check("normalization-equal pair is Tier 0", mine and mine[0].tier == 0)
    check("keep-order: richer-mentioned entity kept",
          mine and mine[0].payload["keep_id"] == str(a_id))

    run0 = newrun()
    outcome = ma._process(db, mine[0], None, run0,
                          {"llm_decisions": 0, "applications": 0, "proposals": 0})
    db.expire_all()
    a2 = db.query(CodexEntity).get(a_id)
    b2 = db.query(CodexEntity).get(b_id)
    check("Tier-0 merge applied (no LLM needed)", outcome == "applied")
    check("absorbed entity expired-with-marker, never deleted",
          b2 is not None and (b2.properties or {}).get("merged_into") == str(a_id)
          and "[merged:" in b2.canonical_name)
    check("aliases unioned (absorb canonical + aliases live on keeper)",
          f"{MARK} dupb" in (a2.aliases or []) and f"{MARK} dupx" in (a2.aliases or []))
    check("longer description won (journaled)",
          a2.description.startswith("a much longer"))
    live_out = {(e.relation, e.target_id) for e in db.query(CodexEdge).filter(
        CodexEdge.source_id == a_id, CodexEdge.valid_until == None).all()}  # noqa: E711
    check("unique edge re-pointed to keeper",
          ("depends_on", d.id) in live_out)
    dup2 = db.query(CodexEdge).get(dup_id)
    check("duplicate edge expired, survivor kept max strength",
          dup2.valid_until is not None and
          db.query(CodexEdge).filter(CodexEdge.source_id == a_id,
                                     CodexEdge.relation == "uses",
                                     CodexEdge.valid_until == None)  # noqa: E711
            .first().strength == 5.0)
    check("pair-self-loop edge expired instead of re-pointed",
          db.query(CodexEdge).get(loop_id).valid_until is not None)
    ev0 = agent_events(run0)
    types0 = {r.event_type for r in ev0}
    check("merge journaled: entity_merged + edge_expired + description_updated",
          {"entity_merged", "edge_expired", "description_updated"} <= types0)
    check("absorbed entity's events moved to keeper",
          db.query(CodexEvent).filter_by(entity_id=b_id).count() == 0)

    # ═══ 2. Tier-2 near-dup → proposal → approve dispatch → real merge ═════
    print("── 2. Tier-2 proposal, D6 approve-dispatch, rejected-pair skip ──")
    p1 = mkent(f"{MARK} neara", emb=V1)
    p2 = mkent(f"{MARK} nearb", emb=V2)
    p3 = mkent(f"{MARK} nearc", emb=V3)
    p4 = mkent(f"{MARK} neard", emb=V4)
    db.commit()
    p1_id, p2_id, p3_id, p4_id = p1.id, p2.id, p3.id, p4.id

    items = ma._detect_duplicate_entities(db, 500)
    key_12, key_34 = ma._pair_key(p1_id, p2_id), ma._pair_key(p3_id, p4_id)
    mine12 = [i for i in items if i.payload["pair_key"] == key_12]
    check("cosine-only pair detected", len(mine12) == 1)
    check("cosine-only pair is Tier 2 (never auto-merged)",
          mine12 and mine12[0].tier == 2)

    stub_same = lambda prompt, max_tokens=200: {"verdict": "same"}   # noqa: E731
    run1 = newrun()
    outcome = ma._process(db, mine12[0], stub_same, run1,
                          {"llm_decisions": 0, "applications": 0, "proposals": 0})
    rq = db.query(ReviewQueue).filter(
        ReviewQueue.item_type == "entity_merge",
        ReviewQueue.status == "pending",
        ReviewQueue.item_content["pair_key"].astext == key_12).first()
    check("'same' verdict wrote one pending entity_merge proposal",
          outcome == "proposed" and rq is not None)
    check("proposal is self-describing (names + evidence for F2)",
          rq is not None and rq.item_content.get("keep_name", "").startswith(MARK)
          and rq.item_content.get("evidence", {}).get("cosine") is not None)

    review.approve(db, str(rq.id))          # the EXISTING E0 dispatch
    db.expire_all()
    kept_id = uuid.UUID(rq.item_content["keep_id"])
    absorbed = db.query(CodexEntity).get(uuid.UUID(rq.item_content["absorb_id"]))
    check("approve dispatched the REAL merge (absorb expired into keep)",
          (absorbed.properties or {}).get("merged_into") == str(kept_id))

    # rejected pair is never re-proposed
    items = ma._detect_duplicate_entities(db, 500)
    mine34 = [i for i in items if i.payload["pair_key"] == key_34]
    run2 = newrun()
    ma._process(db, mine34[0], stub_same, run2,
                {"llm_decisions": 0, "applications": 0, "proposals": 0})
    rq34 = db.query(ReviewQueue).filter(
        ReviewQueue.item_type == "entity_merge",
        ReviewQueue.item_content["pair_key"].astext == key_34).first()
    review.reject(db, str(rq34.id))
    items = ma._detect_duplicate_entities(db, 500)
    check("rejected pair never re-proposed (detector skips it)",
          not any(i.payload["pair_key"] == key_34 for i in items))

    # ═══ 3. Pending-edge pileup → Tier-1 dedupe/expire ═════════════════════
    print("── 3. Pileup: pending duplicates expired with events ──")
    hub = mkent(f"{MARK} hub")
    targets = [mkent(f"{MARK} spoke{i}") for i in range(4)]
    pend_ids, act_ids = [], []
    for t in targets:
        act_ids.append(mkedge(hub, t, "uses", confidence="active",
                              extraction_confidence=0.5).id)
        pend_ids.append(mkedge(hub, t, "uses", confidence="pending",
                               extraction_confidence=0.95).id)
    db.commit()
    hub_id = hub.id

    items = ma._detect_pending_pileup(db, 50)
    mine = [i for i in items if i.payload["entity_id"] == str(hub_id)]
    check("detector 2 (ported sentinel query) finds the pileup entity",
          len(mine) == 1)
    run3 = newrun()
    outcome = ma._process(db, mine[0], None, run3,
                          {"llm_decisions": 0, "applications": 0, "proposals": 0})
    db.expire_all()
    pend_expired = all(db.query(CodexEdge).get(i).valid_until is not None
                       for i in pend_ids)
    act_live = all(db.query(CodexEdge).get(i).valid_until is None
                   for i in act_ids)
    check("all pending duplicates expired; actives untouched",
          outcome == "applied" and pend_expired and act_live)
    check("active kept the max extraction_confidence",
          db.query(CodexEdge).get(act_ids[0]).extraction_confidence == 0.95)
    ev3 = agent_events(run3)
    check("pileup expiries journaled with source=maintenance_agent",
          len(ev3) == 4 and all(r.payload.get("source") == "maintenance_agent"
                                and r.payload.get("reason") == "pending_duplicate"
                                for r in ev3))

    # ═══ 4. Contradiction backlog → reconciled (newer supersedes) ══════════
    print("── 4. Contradictions: polarity + antonym pairs ──")
    s1, t1 = mkent(f"{MARK} subj1"), mkent(f"{MARK} obj1")
    pos = mkedge(s1, t1, "uses", valid_from=NOW - timedelta(days=30))
    neg = mkedge(s1, t1, "uses", negated=True, valid_from=NOW - timedelta(days=1))
    s2, t2 = mkent(f"{MARK} subj2"), mkent(f"{MARK} obj2")
    friend = mkedge(s2, t2, "friend", valid_from=NOW - timedelta(days=30))
    enemy = mkedge(s2, t2, "enemy", valid_from=NOW - timedelta(days=1))
    db.commit()
    pos_id, neg_id, friend_id, enemy_id = pos.id, neg.id, friend.id, enemy.id

    items = ma._detect_contradictions(db, 50)
    mine_pol = [i for i in items if i.payload["old_edge_id"] == str(pos_id)]
    mine_ant = [i for i in items if i.payload["old_edge_id"] == str(friend_id)]
    check("polarity clash detected (older positive is the loser)",
          len(mine_pol) == 1 and mine_pol[0].payload["kind"] == "polarity")
    check("antonym pair detected (older friend is the loser)",
          len(mine_ant) == 1 and mine_ant[0].payload["kind"] == "antonym")
    run4 = newrun()
    ctr = {"llm_decisions": 0, "applications": 0, "proposals": 0}
    o1 = ma._process(db, mine_pol[0], None, run4, ctr)
    o2 = ma._process(db, mine_ant[0], None, run4, ctr)
    db.expire_all()
    check("both contradictions reconciled deterministically",
          o1 == "applied" and o2 == "applied"
          and db.query(CodexEdge).get(pos_id).valid_until is not None
          and db.query(CodexEdge).get(neg_id).valid_until is None
          and db.query(CodexEdge).get(friend_id).valid_until is not None
          and db.query(CodexEdge).get(enemy_id).valid_until is None)
    check("reconciliations journaled under the run id",
          len(agent_events(run4)) == 2)

    # ═══ 5. Stale pending_items slot → Tier-2 proposal ═════════════════════
    print("── 5. Stale slot: proposal written, blocker respected ──")
    slot = db.query(MemorySlot).filter_by(slot_name="pending_items").first()
    if slot is not None:
        slot_snapshot = (slot.id, slot.content, slot.last_updated, slot.updated_by)
        slot.content = f"{MARK} old pending stuff"
        slot.last_updated = NOW - timedelta(days=20)
    else:
        slot_snapshot = None   # created by the test → delete on cleanup
        slot = MemorySlot(slot_name="pending_items",
                          content=f"{MARK} old pending stuff",
                          last_updated=NOW - timedelta(days=20),
                          updated_by="user")
        db.add(slot)
    db.commit()

    items = ma._detect_stale_slot(db, 5)
    check("detector 5 (ported absence rule) fires on the stale slot",
          len(items) == 1 and items[0].payload["slot_name"] == "pending_items")
    stub_sugg = lambda prompt, max_tokens=200: {"suggestion": f"{MARK} updated pending list"}  # noqa: E731
    run5 = newrun()
    outcome = ma._process(db, items[0], stub_sugg, run5,
                          {"llm_decisions": 0, "applications": 0, "proposals": 0})
    prop = db.query(ReviewQueue).filter(
        ReviewQueue.item_type == "memory_slot_update",
        ReviewQueue.item_content["agent_run_id"].astext == str(run5)).first()
    check("slot proposal written (memory_slot_update — approve arm applies it)",
          outcome == "proposed" and prop is not None
          and MARK in prop.item_content.get("proposed_content", ""))
    check("pending proposal blocks re-detection",
          ma._detect_stale_slot(db, 5) == [])

    # ═══ 6+8. Leftover re-reconcile: unsure/garbage ⇒ no write; ≥2 skips ═══
    print("── 6+8. Reconciliation leftovers: unsure paths + attempt cap ──")
    ra, rb = mkent(f"{MARK} recon a"), mkent(f"{MARK} recon b")
    old_edge = mkedge(ra, rb, "works_with", valid_from=NOW - timedelta(days=10))
    db.commit()
    old_edge_id = old_edge.id
    leftover = ReviewQueue(item_type="codex_reconciliation", item_content={
        "new": {"subject": ra.canonical_name, "relation": "works_with",
                "object": rb.canonical_name},
        "conflict_type": "supersession", "old_edge_id": str(old_edge_id),
        "old_relation": "works_with", "old_object": rb.canonical_name,
        "turn_excerpt": f"{MARK} ambiguous turn"})
    db.add(leftover)
    db.commit()
    rq_ids.append(leftover.id)
    leftover_id = leftover.id

    items = ma._detect_reconciliation_leftovers(db, 50)
    mine = [i for i in items if i.payload["review_id"] == str(leftover_id)]
    check("detector 1 picks up the pending leftover", len(mine) == 1)

    run6 = newrun()
    stub_none = lambda prompt, max_tokens=200: None                      # unparseable JSON
    stub_junk = lambda prompt, max_tokens=200: {"decision": "banana"}    # out-of-enum
    ctr = {"llm_decisions": 0, "applications": 0, "proposals": 0}
    o1 = ma._process(db, mine[0], stub_none, run6, ctr)
    items = ma._detect_reconciliation_leftovers(db, 50)
    mine = [i for i in items if i.payload["review_id"] == str(leftover_id)]
    o2 = ma._process(db, mine[0], stub_junk, run6, ctr)
    db.expire_all()
    leftover2 = db.query(ReviewQueue).get(leftover_id)
    check("unparseable + out-of-enum ⇒ unsure ⇒ NO write, item stays pending",
          o1 == "still_unsure" and o2 == "still_unsure"
          and leftover2.status == "pending"
          and db.query(CodexEdge).get(old_edge_id).valid_until is None)
    check("agent_attempts incremented per attempt",
          int(leftover2.item_content.get("agent_attempts", 0)) == 2)
    items = ma._detect_reconciliation_leftovers(db, 50)
    check("≥2 attempts ⇒ never retried (detector excludes it)",
          not any(i.payload["review_id"] == str(leftover_id) for i in items))

    # and the applied path: expire_old actually expires + resolves the item
    leftover3 = ReviewQueue(item_type="codex_reconciliation", item_content={
        "new": {"subject": ra.canonical_name, "relation": "works_with",
                "object": rb.canonical_name},
        "conflict_type": "supersession", "old_edge_id": str(old_edge_id),
        "old_relation": "works_with", "old_object": rb.canonical_name,
        "turn_excerpt": f"{MARK} clearer turn"})
    db.add(leftover3)
    db.commit()
    rq_ids.append(leftover3.id)
    stub_expire = lambda prompt, max_tokens=200: {"decision": "expire_old"}  # noqa: E731
    run7 = newrun()
    items = ma._detect_reconciliation_leftovers(db, 50)
    mine = [i for i in items if i.payload["review_id"] == str(leftover3.id)]
    o3 = ma._process(db, mine[0], stub_expire, run7,
                     {"llm_decisions": 0, "applications": 0, "proposals": 0})
    db.expire_all()
    check("expire_old applied: edge expired, item resolved (not 'approved')",
          o3 == "applied"
          and db.query(CodexEdge).get(old_edge_id).valid_until is not None
          and db.query(ReviewQueue).get(leftover3.id).status == "resolved")

    # ═══ 7. Caps (run loop with monkeypatched detector registry) ═══════════
    print("── 7. Caps: 25 LLM decisions / 5 proposals / 10 applications ──")
    ca, cb = mkent(f"{MARK} caps a"), mkent(f"{MARK} caps b")
    cap_edges = [mkedge(ca, cb, f"rel{i:02d}") for i in range(30)]
    db.commit()
    cap_items = []
    for e in cap_edges:
        r = ReviewQueue(item_type="codex_reconciliation", item_content={
            "new": {"subject": ca.canonical_name, "relation": e.relation,
                    "object": cb.canonical_name},
            "old_edge_id": str(e.id), "old_relation": e.relation,
            "old_object": cb.canonical_name, "turn_excerpt": MARK})
        db.add(r)
        db.commit()
        rq_ids.append(r.id)
        cap_items.append(ma.WorkItem("reconciliation_leftover", 1,
                                     {"review_id": str(r.id),
                                      "content": dict(r.item_content)}))
    ma.DETECTORS = {"reconciliation_leftover": lambda db, cap: cap_items[:cap]}
    stub_keep = lambda prompt, max_tokens=200: {"decision": "keep_both"}  # noqa: E731
    result = ma.run_maintenance_agent(db, llm_decider=stub_keep)
    run_ids.append(uuid.UUID(result["agent_run_id"]))
    check("30 seeded items → exactly 25 LLM decisions, 5 skipped by cap",
          result["llm_decisions"] == 25
          and result["outcomes"].get("reconciliation_leftover:skipped_cap") == 5)

    # proposals cap: 8 'same' verdict pairs → 5 proposals
    pairs = []
    for i in range(8):
        x = mkent(f"{MARK} pcap x{i}")
        y = mkent(f"{MARK} pcap y{i}")
        pairs.append(ma.WorkItem("duplicate_entities", 2, {
            "keep_id": str(x.id), "absorb_id": str(y.id),
            "pair_key": ma._pair_key(x.id, y.id), "cosine": 0.95}))
    db.commit()
    ma.DETECTORS = {"duplicate_entities": lambda db, cap: pairs[:cap]}
    result = ma.run_maintenance_agent(db, llm_decider=stub_same)
    run_ids.append(uuid.UUID(result["agent_run_id"]))
    check("8 'same' pairs → exactly 5 proposals (queue-flood guard)",
          result["proposals"] == 5
          and result["outcomes"].get("duplicate_entities:skipped_cap") == 3)

    # applications cap: 12 contradictions → 10 applied
    app_items, app_edge_ids = [], []
    for i in range(12):
        sx = mkent(f"{MARK} acap s{i}")
        tx = mkent(f"{MARK} acap t{i}")
        po = mkedge(sx, tx, "uses", valid_from=NOW - timedelta(days=5))
        ne = mkedge(sx, tx, "uses", negated=True, valid_from=NOW - timedelta(days=1))
        db.flush()
        app_edge_ids.append(po.id)
        app_items.append(ma.WorkItem("contradiction", 1, {
            "kind": "polarity", "old_edge_id": str(po.id),
            "old_relation": "uses", "new_edge_id": str(ne.id),
            "new_relation": "uses", "source_id": str(sx.id),
            "target_id": str(tx.id)}))
    db.commit()
    ma.DETECTORS = {"contradiction": lambda db, cap: app_items[:cap]}
    result = ma.run_maintenance_agent(db, llm_decider=None)   # deterministic tier 1
    run_ids.append(uuid.UUID(result["agent_run_id"]))
    db.expire_all()
    n_expired = sum(1 for eid in app_edge_ids
                    if db.query(CodexEdge).get(eid).valid_until is not None)
    check("12 contradictions → exactly 10 applications (cap), 2 deferred",
          result["applications"] == 10 and n_expired == 10
          and result["outcomes"].get("contradiction:skipped_cap") == 2)

    # ═══ 9. The Sentinel is gone ═══════════════════════════════════════════
    print("── 9. Sentinel removal ──")
    regs = db.execute(text(
        "SELECT to_regclass('sentinel_rules'), to_regclass('sentinel_events')"
    )).first()
    check("sentinel tables dropped", regs[0] is None and regs[1] is None)
    import src.memory.models as mm
    check("Sentinel models gone",
          not hasattr(mm, "SentinelRule") and not hasattr(mm, "SentinelEvent"))
    check("sentinel_monitor.py deleted",
          not os.path.exists(os.path.join(os.path.dirname(__file__), "..",
                                          "src", "workers", "sentinel_monitor.py")))
    from src.api.config import settings
    from src.workers.runtime import JOBS
    check("runtime JOBS: maintenance_agent in (gpu lane), sentinel out",
          "maintenance_agent" in JOBS and "sentinel_monitor" not in JOBS
          and JOBS["maintenance_agent"].lane == "gpu"
          and JOBS["maintenance_agent"].needs_db)
    check("intervals: agent overdue-scheduled (12h), sentinel gone",
          settings.maintenance_intervals.get("maintenance_agent") == 43200
          and "sentinel_monitor" not in settings.maintenance_intervals)

    # ═══ 10. Every applied action journaled under its agent_run_id ═════════
    print("── 10. Audit trail ──")
    applied_runs = [run0, run3, run4, run7]
    ok = True
    for rid in applied_runs:
        evs = agent_events(rid)
        if not evs:
            ok = False
        for r in evs:
            if r.payload.get("source") != "maintenance_agent":
                ok = False
    check("every applied run has CodexEvents keyed by agent_run_id, "
          "all tagged source=maintenance_agent", ok)

finally:
    ma.DETECTORS = orig_detectors
    ma.DUP_PAIRS_PER_RUN = orig_dup_cap
    db.rollback()
    try:
        # FK-safe order; marker/id-keyed — NEVER truncate.
        if ent_ids:
            db.execute(text("DELETE FROM codex_events WHERE entity_id = ANY(:ids)"),
                       {"ids": ent_ids})
            db.execute(text("DELETE FROM codex_edges WHERE source_id = ANY(:ids) "
                            "OR target_id = ANY(:ids)"), {"ids": ent_ids})
        if run_ids:
            db.execute(text("DELETE FROM codex_events WHERE batch_source = ANY(:b)"),
                       {"b": run_ids})
            db.execute(text("DELETE FROM review_queue "
                            "WHERE item_content->>'agent_run_id' = ANY(:r)"),
                       {"r": [str(r) for r in run_ids]})
        if rq_ids:
            db.execute(text("DELETE FROM review_queue WHERE id = ANY(:ids)"),
                       {"ids": rq_ids})
        db.execute(text("DELETE FROM review_queue WHERE item_content::text LIKE :m"),
                   {"m": f"%{MARK}%"})
        if ent_ids:
            db.execute(text("DELETE FROM codex_entities WHERE id = ANY(:ids)"),
                       {"ids": ent_ids})
        if slot_snapshot == "absent":
            pass
        elif slot_snapshot is None:
            db.execute(text("DELETE FROM memory_slots WHERE slot_name = 'pending_items' "
                            "AND content LIKE :m"), {"m": f"%{MARK}%"})
        else:
            sid, scontent, supdated, sby = slot_snapshot
            db.execute(text("UPDATE memory_slots SET content = :c, last_updated = :u, "
                            "updated_by = :b WHERE id = :i"),
                       {"c": scontent, "u": supdated, "b": sby, "i": sid})
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"CLEANUP FAILED (marker {MARK}): {exc}")
    db.close()

print(f"\n{'=' * 50}\n{_passed} passed, {_failed} failed  (marker {MARK})")
sys.exit(1 if _failed else 0)
