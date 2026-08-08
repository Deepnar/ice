"""C8 + C15 behavioral test: in-score recency weighting and the wide-net
trigger/budget rework. Live DB for the SQL paths; own rows only.

Run: uv run python tests/test_c8_c15.py
"""
import os
import sys
import uuid
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


from src.api.config import settings

print("── C15: per-head confidence trigger logic (pure) ──")
from src.retrieval.orchestrator import _head_confidences

# ML result: topic peaked, intent weak — old max-over-25 would NOT fire; the
# honest check requires BOTH weak, so a peaked topic still blocks firing.
probs = [0.05] * 25
probs[2] = 0.95          # strong topic
probs[13] = 0.40         # weak intent
clf = SimpleNamespace(raw_probs=probs, max_confidence=0.95)
tc, ic = _head_confidences(clf)
check("per-head split (topic 0.95 / intent 0.40)", tc == 0.95 and ic == 0.40)
check("peaked topic alone blocks wide net (needs BOTH weak)",
      not (tc < 0.75 and ic < 0.75))

probs2 = [0.05] * 25
probs2[3] = 0.5
probs2[14] = 0.45
clf2 = SimpleNamespace(raw_probs=probs2, max_confidence=0.5)
tc2, ic2 = _head_confidences(clf2)
check("both heads weak → wide net fires", tc2 < 0.75 and ic2 < 0.75)

blank = SimpleNamespace(raw_probs=[0.0] * 25, max_confidence=0.9)
tc3, ic3 = _head_confidences(blank)
check("all-zero probs → falls back to max_confidence",
      tc3 == 0.9 and ic3 == 0.9)

print("── C15: dynamic wide-net budget ──")
check("fraction of model-aware budget",
      max(settings.retrieval_wide_net_budget_floor, int(20000 * settings.retrieval_wide_net_budget_fraction)) == 6000)
check("floor holds for small budgets",
      max(settings.retrieval_wide_net_budget_floor, int(3000 * settings.retrieval_wide_net_budget_fraction)) == 1500)

print("── C8: in-score recency (live DB) ──")
from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

db = SessionLocal()
conv = Conversation(memory_scope_type="auto")
db.add(conv)
db.commit()
EMB = [0.03] * 1024
NOW = datetime.now(timezone.utc)


def mk(text, ts):
    t = EpisodicMemory(
        conversation_id=conv.id, batch_id=uuid.uuid4(), timestamp=ts,
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", raw_text=text, embedding=EMB,
        decay_score=1.0, idempotency_key=f"test-c8-{uuid.uuid4()}",
    )
    db.add(t)
    db.commit()
    return t


try:
    old_t = mk("identical content old", NOW - timedelta(days=90))
    new_t = mk("identical content new", NOW - timedelta(hours=1))
    orch = HybridRetrievalOrchestrator(db, None)

    clf_tech = SimpleNamespace(topic_tags=["Software_&_Tech"],
                               intent_tags=["Factual_Retrieval"],
                               prompt="identical content", max_confidence=0.9,
                               raw_probs=[0.0] * 25)
    frags = orch._vector_episodic(EMB, clf_tech, scope=None, conv_id=str(conv.id))
    s_new = next(f.score for f in frags if f.source_batch_id == str(new_t.id))
    s_old = next(f.score for f in frags if f.source_batch_id == str(old_t.id))
    check(f"identical embeddings: recent outranks old ({s_new:.3f} > {s_old:.3f})",
          s_new > s_old)

    clf_creative = SimpleNamespace(topic_tags=["Creative_&_Media"],
                                   intent_tags=["Generation"],
                                   prompt="identical content", max_confidence=0.9,
                                   raw_probs=[0.0] * 25)
    frags_c = orch._vector_episodic(EMB, clf_creative, scope=None, conv_id=str(conv.id))
    # in-score weight off for creative → base scores equal; only the small
    # post-hoc tiebreakers in _rows_to_fragments differ, which are < the C8
    # boost. Verify the gap shrinks vs the tech case.
    sc_new = next(f.score for f in frags_c if f.source_batch_id == str(new_t.id))
    sc_old = next(f.score for f in frags_c if f.source_batch_id == str(old_t.id))
    check(f"creative: recency gap much smaller ({(sc_new-sc_old):.3f} < {(s_new-s_old):.3f})",
          (sc_new - sc_old) < (s_new - s_old))
finally:
    db.rollback()
    db.query(EpisodicMemory).filter_by(conversation_id=conv.id).delete()
    db.query(Conversation).filter_by(id=conv.id).delete()
    db.commit()
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
