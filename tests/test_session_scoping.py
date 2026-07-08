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
from src.memory.models import Conversation, EpisodicMemory
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
EMB = [0.05] * 384

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
    sid1, new1 = resolve_session_id(db, conv_sess.id, t0, settings.session_gap_minutes)
    check("empty conversation → new session", new1 is True)
    mk_turn(conv_sess, "session turn one", t0, session_id=sid1)

    sid2, new2 = resolve_session_id(db, conv_sess.id, t0 + timedelta(minutes=5),
                                    settings.session_gap_minutes)
    check("5-min gap → same session", sid2 == sid1 and new2 is False)
    mk_turn(conv_sess, "session turn two", t0 + timedelta(minutes=5), session_id=sid2)

    sid3, new3 = resolve_session_id(db, conv_sess.id, t0 + timedelta(minutes=55),
                                    settings.session_gap_minutes)
    check("50-min gap → NEW session", sid3 != sid1 and new3 is True)

    print("── privacy invariant on the episodic legs ──")
    mk_turn(conv_pub, f"User: tell me about {MARK_PUB}\n\nAssistant: sure", now)
    mk_turn(conv_priv, f"User: tell me about {MARK_PRIV}\n\nAssistant: sure", now,
            is_private=True)

    from sentence_transformers import SentenceTransformer
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", truncate_dim=384)
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
                           ScopeUpdate(memory_scope_type="auto", cluster_ids=None,
                                       custom_filter=None), db)
    flag = db.query(EpisodicMemory).filter_by(conversation_id=conv_priv.id).first().is_private
    check("none→auto clears is_private on rows", flag is False)
    set_conversation_scope(str(conv_priv.id),
                           ScopeUpdate(memory_scope_type="none", cluster_ids=None,
                                       custom_filter=None), db)
    db.expire_all()
    flag = db.query(EpisodicMemory).filter_by(conversation_id=conv_priv.id).first().is_private
    check("auto→none re-sets is_private on rows", flag is True)

finally:
    db.rollback()
    for c in (conv_pub, conv_priv, conv_sess):
        db.query(EpisodicMemory).filter_by(conversation_id=c.id).delete()
        db.query(Conversation).filter_by(id=c.id).delete()
    db.commit()
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
