"""C5 behavioral test: clustering v5 against the live DB.

Covers: wait-for-a-friend creation, exclusive assignment, session-affinity
bonus, aged-singleton creation, singleton re-absorption, pairwise merge, and
the orchestrator's adaptive cluster band. LLM naming is stubbed (like the
A1/A2/A6 tests — real bg-model naming pends Z1); NER runs for real.

Inserts uniquely-marked rows in its own conversations and deletes them after —
never truncates. Run: uv run python tests/test_clustering_v5.py
"""
import os
import sys
import uuid
import math
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, ContextCluster, EpisodicClusterLink

import src.workers.clustering as cw

# Stub the LLM naming (bg model not running; naming quality is Z1's check).
cw._generate_cluster_name = lambda turns, recurring_entities=None: "StubCluster"
cw._generate_cluster_description = (
    lambda turns, recurring_entities=None:
    "DOMAIN: test\nCONTENT_TYPE: facts\nRECURRING_ENTITIES: n/a\nSETTING_OR_CONTEXT: n/a"
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


def unit(theta_deg):
    """384-dim unit vector at angle theta from e1 in the (e1,e2) plane."""
    v = [0.0] * 384
    v[0] = math.cos(math.radians(theta_deg))
    v[1] = math.sin(math.radians(theta_deg))
    return v


db = SessionLocal()
NOW = datetime.now(timezone.utc)
convs = {k: Conversation(memory_scope_type="auto") for k in
         ("friend", "excl", "session", "session_ctl", "aged", "absorb", "merge", "band")}
db.add_all(convs.values())
db.commit()
_turn_ids, _cluster_ids = [], []


def mk_turn(conv, text, emb, ts=None, session_id=None, tags=None):
    t = EpisodicMemory(
        conversation_id=conv.id, batch_id=uuid.uuid4(),
        timestamp=ts or NOW, session_id=session_id,
        topic_tags=tags or [], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", raw_text=text, embedding=emb,
        decay_score=1.0, idempotency_key=f"test-c5-{uuid.uuid4()}",
    )
    db.add(t)
    db.commit()
    _turn_ids.append(t.id)
    return t


def mk_cluster(conv, centroid, member_turns, name="Seed"):
    cl = ContextCluster(
        name=name, description="DOMAIN: t\nRECURRING_ENTITIES: n/a",
        conversation_id=str(conv.id), created_at=NOW, updated_at=NOW,
        embedding=centroid, tags=[],
    )
    db.add(cl)
    db.flush()
    for t in member_turns:
        db.add(EpisodicClusterLink(episodic_id=t.id, cluster_id=cl.id))
    db.commit()
    _cluster_ids.append(cl.id)
    return cl


def clusters_of(conv):
    return db.query(ContextCluster).filter_by(conversation_id=str(conv.id)).all()


def links_of(turn):
    return db.query(EpisodicClusterLink).filter_by(episodic_id=turn.id).all()


try:
    print("── wait-for-a-friend creation ──")
    c = convs["friend"]
    a1 = mk_turn(c, "first note about the alpha subject matter", unit(0))
    a2 = mk_turn(c, "second note about the alpha subject matter", unit(0),
                 ts=NOW + timedelta(minutes=1))
    lone = mk_turn(c, "completely different beta topic here", unit(90),
                   ts=NOW + timedelta(minutes=2))
    stats = cw.run_cluster_assignment(db, conversation_ids=[c.id])
    cls = clusters_of(c)
    check("two similar turns → ONE cluster created", stats["created"] == 1 and len(cls) == 1)
    check("both friends are members",
          {l.episodic_id for l in db.query(EpisodicClusterLink).filter_by(cluster_id=cls[0].id)}
          >= {a1.id, a2.id})
    check("lone dissimilar turn WAITS (no instant singleton)",
          stats["waiting"] == 1 and len(links_of(lone)) == 0)

    print("── exclusive assignment (no multi-membership convergence) ──")
    c = convs["excl"]
    seedA1, seedA2 = (mk_turn(c, "seed A one", unit(0)), mk_turn(c, "seed A two", unit(0)))
    seedB1, seedB2 = (mk_turn(c, "seed B one", unit(37)), mk_turn(c, "seed B two", unit(37)))
    clA = mk_cluster(c, unit(0), [seedA1, seedA2], "A")
    clB = mk_cluster(c, unit(37), [seedB1, seedB2], "B")
    t = mk_turn(c, "a new turn near both clusters", unit(5), ts=NOW + timedelta(minutes=3))
    cw.run_cluster_assignment(db, conversation_ids=[c.id])
    lk = links_of(t)
    check("turn above threshold for BOTH joins exactly ONE cluster", len(lk) == 1)
    check("and it is the best-scoring cluster (A)", lk and lk[0].cluster_id == clA.id)

    print("── session-affinity bonus (C6 payoff) ──")
    # Mirrored setups (assignment moves the centroid, so the control needs
    # clean geometry in its own conversation): raw sim cos(57°)≈0.545 < 0.6;
    # only the +0.10 same-session bonus crosses the threshold.
    c = convs["session"]
    sess = uuid.uuid4()
    m1 = mk_turn(c, "session seed one", unit(0), session_id=sess)
    m2 = mk_turn(c, "session seed two", unit(0), session_id=sess)
    mk_cluster(c, unit(0), [m1, m2], "S")
    borderline_in = mk_turn(c, "borderline same sitting", unit(57),
                            ts=NOW + timedelta(minutes=4), session_id=sess)
    cw.run_cluster_assignment(db, conversation_ids=[c.id])
    check("borderline turn in SAME session → assigned", len(links_of(borderline_in)) == 1)

    c2 = convs["session_ctl"]
    sess2 = uuid.uuid4()
    n1 = mk_turn(c2, "session seed one", unit(0), session_id=sess2)
    n2 = mk_turn(c2, "session seed two", unit(0), session_id=sess2)
    mk_cluster(c2, unit(0), [n1, n2], "S2")
    borderline_out = mk_turn(c2, "borderline other sitting", unit(57),
                             ts=NOW + timedelta(minutes=5), session_id=uuid.uuid4())
    cw.run_cluster_assignment(db, conversation_ids=[c2.id])
    check("identical turn in OTHER session → waits", len(links_of(borderline_out)) == 0)

    print("── aged lone turn becomes a singleton after the wait ──")
    c = convs["aged"]
    old = mk_turn(c, "an old one-off topic", unit(0), ts=NOW - timedelta(hours=30))
    stats = cw.run_cluster_assignment(db, conversation_ids=[c.id])
    check("aged (>24h) lone turn → singleton cluster",
          stats["singletons"] == 1 and len(links_of(old)) == 1)

    print("── merge pass: singleton re-absorption (repair path) ──")
    c = convs["absorb"]
    b1, b2 = mk_turn(c, "big cluster t1", unit(0)), mk_turn(c, "big cluster t2", unit(0))
    big = mk_cluster(c, unit(0), [b1, b2], "Big")
    s1 = mk_turn(c, "stray singleton close to big", unit(20))
    stray = mk_cluster(c, unit(20), [s1], "Stray")   # cos20°≈0.94 ≥ 0.6
    stats = cw.run_cluster_merge(db, conversation_ids=[str(c.id)])
    check("singleton absorbed into fitting sibling", stats["absorbed"] == 1)
    check("stray cluster deleted; turn now in Big",
          db.query(ContextCluster).filter_by(id=stray.id).first() is None
          and links_of(s1) and links_of(s1)[0].cluster_id == big.id)

    print("── merge pass: converged centroids merge (conservative gates) ──")
    c = convs["merge"]
    p1, p2 = mk_turn(c, "p one", unit(0)), mk_turn(c, "p two", unit(0))
    q1, q2 = mk_turn(c, "q one", unit(10)), mk_turn(c, "q two", unit(10))
    clP = mk_cluster(c, unit(0), [p1, p2], "P")
    clQ = mk_cluster(c, unit(10), [q1, q2], "Q")     # cos10°≈0.985 ≥ 0.90
    far1, far2 = mk_turn(c, "far one", unit(80)), mk_turn(c, "far two", unit(80))
    clFar = mk_cluster(c, unit(80), [far1, far2], "Far")
    stats = cw.run_cluster_merge(db, conversation_ids=[str(c.id)])
    remaining = {cl.name for cl in clusters_of(c)}
    check("near-identical clusters merged", stats["merged"] == 1 and "Q" not in remaining)
    check("dissimilar cluster untouched", "Far" in remaining)

    print("── orchestrator: adaptive cluster band ──")
    c = convs["band"]
    n1, n2 = mk_turn(c, "n one", unit(0)), mk_turn(c, "n two", unit(0))
    w1, w2 = mk_turn(c, "w one", unit(75)), mk_turn(c, "w two", unit(75))
    near = mk_cluster(c, unit(0), [n1, n2], "Near")
    weak = mk_cluster(c, unit(75), [w1, w2], "Weak")  # cos75°≈0.26 → outside 0.8×best
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    orch = HybridRetrievalOrchestrator(db, cw.cluster_embedder)
    ids = orch._relevant_cluster_ids(unit(0), classification=None,
                                     conversation_id=str(c.id), top_k=10)
    check("strong cluster in scope, weak tail dropped (not flat top-10)",
          str(near.id) in ids and str(weak.id) not in ids)

finally:
    db.rollback()
    all_conv_ids = [cv.id for cv in convs.values()]
    all_conv_strs = [str(i) for i in all_conv_ids]
    # links → clusters → turns → conversations (only rows this test created)
    db.query(EpisodicClusterLink).filter(
        EpisodicClusterLink.episodic_id.in_(
            db.query(EpisodicMemory.id).filter(EpisodicMemory.conversation_id.in_(all_conv_ids))
        )
    ).delete(synchronize_session=False)
    db.query(ContextCluster).filter(ContextCluster.conversation_id.in_(all_conv_strs)).delete(
        synchronize_session=False)
    db.query(EpisodicMemory).filter(EpisodicMemory.conversation_id.in_(all_conv_ids)).delete(
        synchronize_session=False)
    db.query(Conversation).filter(Conversation.id.in_(all_conv_ids)).delete(
        synchronize_session=False)
    db.commit()
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
