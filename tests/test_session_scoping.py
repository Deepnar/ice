"""C6/G16 behavioral test: session_id assignment + incognito privacy invariant.

Runs against the live Postgres (docker up). Inserts its own uniquely-marked
rows and deletes them afterwards — never truncates (the dev DB holds real data).

Run: uv run python tests/test_session_scoping.py
"""
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal
from src.api.config import settings
from src.memory.models import (
    ContextCluster, Conversation, EpisodicClusterLink, EpisodicMemory,
    ProceduralMemory,
)
from src.memory.session import resolve_session_id
from src.classifier.schemas import ClassificationResult

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


MARK_PUB = "zanzibar quokka microfiche public"
MARK_PRIV = "xylophone begonia stratagem private"
EMB = [0.05] * 1024

db = SessionLocal()
conv_pub = Conversation(memory_scope_type="auto")
conv_priv = Conversation(memory_scope_type="none")
conv_sess = Conversation(memory_scope_type="auto")
db.add_all([conv_pub, conv_priv, conv_sess])
db.commit()
now = datetime.now(timezone.utc)


def mk_turn(conv, text, ts, session_id=None, is_private=False):
    t = EpisodicMemory(
        conversation_id=conv.id, batch_id=uuid.uuid4(), timestamp=ts,
        session_id=session_id, is_private=is_private,
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", raw_text=text, embedding=EMB,
        decay_score=1.0, idempotency_key=f"test-c6-{uuid.uuid4()}",
    )
    db.add(t)
    db.commit()
    return t


try:
    print("── session_id assignment (30-min gap) ──")
    t0 = now - timedelta(hours=3)
    sid1, new1, gap1 = resolve_session_id(db, conv_sess.id, t0, settings.session_gap_minutes)
    check("empty conversation → new session", new1 is True and gap1 is None)
    mk_turn(conv_sess, "session turn one", t0, session_id=sid1)

    sid2, new2, gap2 = resolve_session_id(db, conv_sess.id, t0 + timedelta(minutes=5),
                                    settings.session_gap_minutes)
    check("5-min gap → same session", sid2 == sid1 and new2 is False and gap2 is not None)
    mk_turn(conv_sess, "session turn two", t0 + timedelta(minutes=5), session_id=sid2)

    sid3, new3, gap3 = resolve_session_id(db, conv_sess.id, t0 + timedelta(minutes=55),
                                    settings.session_gap_minutes)
    check("50-min gap → NEW session", sid3 != sid1 and new3 is True and round(gap3 / 60) == 50)

    print("── privacy invariant on the episodic legs ──")
    mk_turn(conv_pub, f"User: tell me about {MARK_PUB}\n\nAssistant: sure", now)
    mk_turn(conv_priv, f"User: tell me about {MARK_PRIV}\n\nAssistant: sure", now,
            is_private=True)

    from src.memory.embedder import get_embedder
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    embedder = get_embedder()
    orch = HybridRetrievalOrchestrator(db, embedder)

    def clf(prompt):
        return ClassificationResult(
            topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
            context_reliance="Long_Term_Memory", raw_probs=[0.0] * 25,
            max_confidence=0.9, prompt=prompt,
        )

    # BM25 — global search must not see the private turn
    frags = orch._bm25_episodic(clf(MARK_PRIV), scope=None, conv_id=None)
    check("bm25 global: private turn invisible",
          not any(MARK_PRIV in f.text for f in frags))
    frags = orch._bm25_episodic(clf(MARK_PUB), scope=None, conv_id=None)
    check("bm25 global: public turn found",
          any(MARK_PUB in f.text for f in frags))
    frags = orch._bm25_episodic(clf(MARK_PRIV), scope={"conversation_id": str(conv_priv.id)},
                                conv_id=str(conv_priv.id))
    check("bm25 own-conversation: private turn readable by itself",
          any(MARK_PRIV in f.text for f in frags))

    # Vector — same invariant
    frags = orch._vector_episodic(EMB, clf(MARK_PRIV), scope=None, conv_id=None)
    check("vector global: private turn invisible",
          not any(MARK_PRIV in f.text for f in frags))
    frags = orch._vector_episodic(EMB, clf(MARK_PRIV), scope=None, conv_id=str(conv_priv.id))
    check("vector own-conversation: private turn readable",
          any(MARK_PRIV in f.text for f in frags))

    # Wide net — must honor scope now (was a leak: searched everything)
    incog_scope = {"conversation_id": str(conv_priv.id), "isolated": True, "incognito": True}
    frags = orch._wide_net_fallback(clf(MARK_PRIV), EMB, str(conv_priv.id), incog_scope)
    check("wide-net incognito: only own conversation's turns",
          all(getattr(f, 'source_type', '') != 'episodic' or MARK_PRIV in f.text or 'session turn' not in f.text
              for f in frags)
          and any(MARK_PRIV in f.text for f in frags))
    frags = orch._wide_net_fallback(clf(MARK_PRIV), EMB, str(conv_pub.id), None)
    check("wide-net global: private turn invisible",
          not any(MARK_PRIV in f.text for f in frags))

    print("── incognito leg gating flags ──")
    check("scope carries isolated for codex (A5 empty-set)",
          incog_scope.get("isolated") is True)

    print("── scope-change propagation (none → auto → none) ──")
    from src.api.routers.user_control import set_conversation_scope, ScopeUpdate
    set_conversation_scope(str(conv_priv.id),
                           ScopeUpdate(memory_scope_type="auto",
                                       cluster_ids=None), db)
    flag = db.query(EpisodicMemory).filter_by(conversation_id=conv_priv.id).first().is_private
    check("none→auto clears is_private on rows", flag is False)
    set_conversation_scope(str(conv_priv.id),
                           ScopeUpdate(memory_scope_type="none",
                                       cluster_ids=None), db)
    db.expire_all()
    flag = db.query(EpisodicMemory).filter_by(conversation_id=conv_priv.id).first().is_private
    check("auto→none re-sets is_private on rows", flag is True)

    # ══ C6: scope vocabulary, manual mode, exclusion ═════════════════════
    # Every scope check below is TWO-SIDED on purpose: it asserts the
    # in-scope row IS returned as well as the out-of-scope row is NOT. A
    # one-sided "the excluded text is absent" passes just as happily on a
    # broken filter as on a working one — an empty result satisfies it.
    print("── C6: scope vocabulary is closed ──")
    from src.services import scoping as scoping_svc
    from src.services.errors import ValidationError

    try:
        scoping_svc.set_scope(db, str(conv_pub.id), "porject")   # typo
        rejected = False
    except ValidationError:
        rejected = True
    db.rollback()
    db.expire_all()
    check("set_scope rejects an unknown mode", rejected)
    check("...and the stored mode is untouched by the rejected write",
          db.query(Conversation).get(conv_pub.id).memory_scope_type == "auto")
    scoping_svc.set_scope(db, str(conv_pub.id), "manual")
    check("set_scope accepts 'manual'",
          db.query(Conversation).get(conv_pub.id).memory_scope_type == "manual")

    print("── C6: manual = the user's pick, as a CLOSED set ──")
    conv_in = Conversation(memory_scope_type="auto")     # ticked into scope
    conv_out = Conversation(memory_scope_type="auto")    # deliberately not
    db.add_all([conv_in, conv_out])
    db.commit()
    MARK_IN = "harmonica trilobite included manual"
    MARK_OUT = "cassowary meridian excluded manual"
    mk_turn(conv_in, f"User: about {MARK_IN}\n\nAssistant: sure", now)
    mk_turn(conv_out, f"User: about {MARK_OUT}\n\nAssistant: sure", now)

    scoping_svc.set_scope(db, str(conv_pub.id), "manual",
                          included_conversation_ids=[str(conv_in.id)])
    db.expire_all()
    manual_scope = scoping_svc.resolve_retrieval_scope(
        db, db.query(Conversation).get(conv_pub.id))
    check("manual resolves to self + the picked conversation",
          set(manual_scope["conversation_ids"]) == {str(conv_pub.id), str(conv_in.id)})
    check("...and an unpicked conversation is NOT in the set",
          str(conv_out.id) not in manual_scope["conversation_ids"])

    frags = orch._bm25_episodic(clf(MARK_IN), scope=manual_scope, conv_id=None)
    check("manual scope RETURNS the picked conversation's turn",
          any(MARK_IN in f.text for f in frags))
    frags = orch._bm25_episodic(clf(MARK_OUT), scope=manual_scope, conv_id=None)
    check("manual scope does NOT return an unpicked conversation's turn",
          not any(MARK_OUT in f.text for f in frags))

    print("── C6: a bare /scope no longer wipes the pick ──")
    scoping_svc.set_scope(db, str(conv_pub.id), "manual")   # no id sets passed
    db.expire_all()
    check("set_scope with id sets omitted leaves them unchanged",
          [str(c) for c in
           (db.query(Conversation).get(conv_pub.id).included_conversation_ids or [])]
          == [str(conv_in.id)])
    scoping_svc.set_scope(db, str(conv_pub.id), "manual",
                          included_conversation_ids=[])
    db.expire_all()
    check("...and an explicit [] clears them",
          not db.query(Conversation).get(conv_pub.id).included_conversation_ids)
    scoping_svc.set_scope(db, str(conv_pub.id), "manual",
                          included_conversation_ids=[str(conv_in.id)])

    print("── C6: an explicit cluster choice survives the auto picker ──")
    # conversation_id is set so _relevant_cluster_ids can actually FIND these
    # under the scope below — otherwise the "picker would have overwritten"
    # half of the check passes for the wrong reason (picker returned nothing).
    clus_keep = ContextCluster(name="c6-keep", tags=["Software_&_Tech"],
                               embedding=EMB, conversation_id=conv_in.id)
    clus_drop = ContextCluster(name="c6-drop", tags=["Software_&_Tech"],
                               embedding=EMB, conversation_id=conv_in.id)
    db.add_all([clus_keep, clus_drop])
    db.commit()
    MARK_CK = "petrichor abacus clusterkeep"
    MARK_CD = "zeppelin marmalade clusterdrop"
    turn_ck = mk_turn(conv_in, f"User: about {MARK_CK}\n\nAssistant: sure", now)
    turn_cd = mk_turn(conv_in, f"User: about {MARK_CD}\n\nAssistant: sure", now)
    db.add_all([EpisodicClusterLink(episodic_id=turn_ck.id, cluster_id=clus_keep.id),
                EpisodicClusterLink(episodic_id=turn_cd.id, cluster_id=clus_drop.id)])
    db.commit()

    explicit_scope = {"conversation_ids": [str(conv_in.id)],
                      "cluster_ids": [str(clus_keep.id)],
                      "cluster_ids_explicit": True}
    frags = orch._bm25_episodic(clf(MARK_CK), scope=explicit_scope, conv_id=None)
    check("explicit cluster scope RETURNS a turn in the chosen cluster",
          any(MARK_CK in f.text for f in frags))
    frags = orch._bm25_episodic(clf(MARK_CD), scope=explicit_scope, conv_id=None)
    check("explicit cluster scope does NOT return a turn in another cluster",
          not any(MARK_CD in f.text for f in frags))

    # The regression the audit named: retrieve() overwrote scope["cluster_ids"]
    # with the automatic picker's output, so a hand-picked set only survived
    # when the picker happened to return nothing.
    orch.retrieve(classification=clf(MARK_CK), conversation_id=str(conv_in.id),
                  prompt_embedding=EMB, scope=explicit_scope)
    check("retrieve() leaves an explicit cluster choice intact",
          explicit_scope["cluster_ids"] == [str(clus_keep.id)])
    auto_scope = {"conversation_ids": [str(conv_in.id)]}
    orch.retrieve(classification=clf(MARK_CK), conversation_id=str(conv_in.id),
                  prompt_embedding=EMB, scope=auto_scope)
    # The other side: the picker DOES run here and DOES produce a different
    # set (it reaches clus_drop, which the explicit choice left out) — so the
    # check above is the flag working, not the picker being idle.
    check("...while an unmarked scope still gets the automatic picker's",
          str(clus_drop.id) in (auto_scope.get("cluster_ids") or []))

    print("── G29: the wide net's silently-absent cluster filter ──")
    # The wide net widens RANKING, not visibility — but it was the one
    # episodic leg with no cluster predicate, so a cluster-scoped
    # conversation that tripped it saw every cluster.
    wide_scope = {"conversation_ids": [str(conv_in.id)],
                  "cluster_ids": [str(clus_keep.id)],
                  "cluster_ids_explicit": True}
    frags = orch._wide_net_fallback(clf(MARK_CK), EMB, str(conv_in.id), wide_scope)
    check("wide net RETURNS a turn in the scoped cluster",
          any(MARK_CK in f.text for f in frags))
    check("wide net does NOT return a turn from an unscoped cluster",
          not any(MARK_CD in f.text for f in frags))

    print("── C6: exclusion — keep the memory, stop reading it ──")
    # Under `auto` (unscoped), which is the case an allow-list cannot express.
    scoping_svc.set_scope(db, str(conv_sess.id), "auto",
                          excluded_conversation_ids=[str(conv_out.id)])
    db.expire_all()
    excl_scope = scoping_svc.resolve_retrieval_scope(
        db, db.query(Conversation).get(conv_sess.id))
    check("exclusion resolves into the scope under auto",
          excl_scope.get("exclude_conversation_ids") == [str(conv_out.id)]
          and "conversation_ids" not in excl_scope)
    frags = orch._bm25_episodic(clf(MARK_OUT), scope=excl_scope, conv_id=None)
    check("excluded conversation's turn is NOT retrieved",
          not any(MARK_OUT in f.text for f in frags))
    frags = orch._bm25_episodic(clf(MARK_IN), scope=excl_scope, conv_id=None)
    check("...while a non-excluded conversation's turn still IS",
          any(MARK_IN in f.text for f in frags))
    # And the same rows are still there — excluded, not deleted.
    check("the excluded turn is still in the store (not deleted)",
          db.query(EpisodicMemory).filter_by(conversation_id=conv_out.id).count() == 1)
    frags = orch._vector_episodic(EMB, clf(MARK_OUT), scope=excl_scope, conv_id=None)
    check("the vector leg honors conversation exclusion too",
          not any(MARK_OUT in f.text for f in frags))

    scoping_svc.set_scope(db, str(conv_sess.id), "auto",
                          excluded_conversation_ids=[],
                          excluded_cluster_ids=[str(clus_drop.id)])
    db.expire_all()
    clus_excl_scope = scoping_svc.resolve_retrieval_scope(
        db, db.query(Conversation).get(conv_sess.id))
    frags = orch._bm25_episodic(clf(MARK_CD), scope=clus_excl_scope, conv_id=None)
    check("a turn in an excluded CLUSTER is NOT retrieved",
          not any(MARK_CD in f.text for f in frags))
    frags = orch._bm25_episodic(clf(MARK_CK), scope=clus_excl_scope, conv_id=None)
    check("...while a turn in a non-excluded cluster still IS",
          any(MARK_CK in f.text for f in frags))

    print("── C6/G29: the procedural leg's fourth leak site ──")
    batch_in = db.query(EpisodicMemory).filter_by(
        conversation_id=conv_in.id).first().batch_id
    batch_out = db.query(EpisodicMemory).filter_by(
        conversation_id=conv_out.id).first().batch_id
    pat_in = ProceduralMemory(
        pattern_name="c6-in", pattern_description="c6 pattern from the picked chat",
        topic_tags=["Software_&_Tech"], trigger_conditions={},
        confidence_score=0.9, is_active=True, embedding=EMB,
        source_batch_ids=[batch_in])
    pat_out = ProceduralMemory(
        pattern_name="c6-out", pattern_description="c6 pattern from the unpicked chat",
        topic_tags=["Software_&_Tech"], trigger_conditions={},
        confidence_score=0.9, is_active=True, embedding=EMB,
        source_batch_ids=[batch_out])
    db.add_all([pat_in, pat_out])
    db.commit()

    # A project/manual scope populates conversation_ids and NO conversation_id
    # — the exact shape the old hand-rolled lookup could not see, so it ran
    # against every pattern in the store.
    proc_scope = {"conversation_ids": [str(conv_in.id)]}
    pats = orch._procedural_lookup(EMB, clf("c6 pattern"), scope=proc_scope)
    descs = [p.text for p in pats]
    check("procedural under conversation_ids scope RETURNS the in-scope pattern",
          any("picked chat" in d for d in descs))
    check("...and does NOT return the out-of-scope one (the G29 leak)",
          not any("unpicked chat" in d for d in descs))

    excl_proc_scope = {"exclude_conversation_ids": [str(conv_out.id)]}
    pats = orch._procedural_lookup(EMB, clf("c6 pattern"), scope=excl_proc_scope)
    descs = [p.text for p in pats]
    check("procedural honors exclusion: excluded pattern absent",
          not any("unpicked chat" in d for d in descs))
    check("...while the non-excluded pattern is still returned",
          any("picked chat" in d for d in descs))

    print("── C6: one resolver, both request paths ──")
    # The MCP pull path used to rebuild scope itself and only reproduced the
    # project arm, so an ice_context pull inside an incognito conversation
    # missed these flags and ran the RAG/procedural legs against global memory.
    incog_resolved = scoping_svc.resolve_retrieval_scope(
        db, db.query(Conversation).get(conv_priv.id))
    check("resolver marks incognito for BOTH paths",
          incog_resolved.get("isolated") is True
          and incog_resolved.get("incognito") is True
          and incog_resolved.get("conversation_id") == str(conv_priv.id))

finally:
    db.rollback()
    convs = [c for c in (conv_pub, conv_priv, conv_sess,
                         globals().get("conv_in"), globals().get("conv_out"))
             if c is not None]
    # FK order: links → turns → procedural/clusters → conversations.
    turn_ids = [t.id for c in convs for t in
                db.query(EpisodicMemory).filter_by(conversation_id=c.id).all()]
    if turn_ids:
        db.query(EpisodicClusterLink).filter(
            EpisodicClusterLink.episodic_id.in_(turn_ids)).delete(
                synchronize_session=False)
    for c in convs:
        db.query(EpisodicMemory).filter_by(conversation_id=c.id).delete()
    for pat in (globals().get("pat_in"), globals().get("pat_out")):
        if pat is not None:
            db.query(ProceduralMemory).filter_by(id=pat.id).delete()
    for clus in (globals().get("clus_keep"), globals().get("clus_drop")):
        if clus is not None:
            db.query(ContextCluster).filter_by(id=clus.id).delete()
    db.commit()
    for c in convs:
        db.query(Conversation).filter_by(id=c.id).delete()
    db.commit()
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
