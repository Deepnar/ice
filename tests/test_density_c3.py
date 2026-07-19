"""C3 behavioral test: all-turns chunking, hierarchy (raw→summary→abstract),
heading-aware chunk boundaries, sentence-boundary truncation, chunk/parent
dedupe in the vector leg.

Run: uv run python tests/test_density_c3.py
"""
import os
import sys
import uuid
from types import SimpleNamespace
from datetime import datetime, timezone

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


print("── heading-aware chunking (section-level decomposition) ──")
from src.memory.chunking import chunk_text
doc = ("Intro sentence about the project. " * 30 +
       "\n## Deployment\n" + "Deployment detail sentence here. " * 30 +
       "\n## Rollback\n" + "Rollback procedure sentence here. " * 30)
chunks = chunk_text(doc, max_tokens=200)
check("chunks produced", len(chunks) >= 3)
check("no chunk spans a section boundary (heading always starts its chunk)",
      all(("## " not in c) or c.index("## ") == 0 or c.count("## ") == 1 and c.strip().startswith("##") or "## " not in c[1:]
          for c in chunks) and any(c.strip().startswith("## Deployment") for c in chunks)
      and any(c.strip().startswith("## Rollback") for c in chunks))

print("── abstract parsing (same LLM call) ──")
import src.workers.post_flight as pf
content = ("Orien defeated Kazama.\nKey terms: Orien, Kazama\n"
           "Abstract: A duel between Orien and Kazama is recounted.")
body, abstract = pf._split_abstract(content)
check("abstract line extracted", abstract == "A duel between Orien and Kazama is recounted.")
check("abstract removed from summary body", "Abstract:" not in body and "Orien defeated" in body)
body2, abstract2 = pf._split_abstract("No abstract line here.\nKey terms: x")
check("no abstract → None, body intact", abstract2 is None and body2.startswith("No abstract"))

print("── sentence-boundary truncation ──")
from src.retrieval.orchestrator import _truncate_at_sentence
txt = ("Alpha beta gamma delta. " * 40)   # 4-word sentences
out = _truncate_at_sentence(txt, 50)
check("truncated under cap", len(out.split()) <= 52)
check("cut lands on a sentence boundary", out.rstrip("… ").endswith("."))
check("short text untouched", _truncate_at_sentence("Tiny.", 50) == "Tiny.")

print("── degrade chain: raw → summary → abstract ──")
from src.retrieval.orchestrator import HybridRetrievalOrchestrator, ContextFragment
orch = HybridRetrievalOrchestrator(None, None)
big_raw = "word " * 600
med = "medium " * 80          # ~106 tokens — too big for a 40-token budget
tiny = "one line abstract"    # ~4 tokens
f = ContextFragment(text=big_raw, source_type="episodic", score=2.0,
                    token_count=int(600 * 1.33), degrade_text=med, abstract_text=tiny)
kept = orch._enforce_token_budget([f], max_tokens=40)
check("summary too big too → degrades to the ABSTRACT", len(kept) == 1 and kept[0].text == tiny)
f2 = ContextFragment(text=big_raw, source_type="episodic", score=2.0,
                     token_count=int(600 * 1.33), degrade_text=med, abstract_text=tiny)
kept = orch._enforce_token_budget([f2], max_tokens=150)
check("summary fits → degrades to the SUMMARY (not abstract)",
      len(kept) == 1 and kept[0].text == med)

print("── chooser attaches abstract under trust rules ──")
def row(**kw):
    base = dict(raw_text="raw zephyr text", summary_text="summary text",
                summary_coverage=0.9, inject_raw=True, abstract_text="abs line")
    base.update(kw)
    return SimpleNamespace(**base)
clf = SimpleNamespace(intent_tags=["Casual_Banter"])
t, dg, ab = orch._choose_representation(row(), clf, set())
check("trusted → abstract attached", ab == "abs line")
t, dg, ab = orch._choose_representation(row(summary_coverage=0.3), clf, set())
check("untrusted summary → no abstract either", ab is None)
t, dg, ab = orch._choose_representation(row(), clf, {"zephyr"})
check("keyword-protected → no degradation levels at all", dg is None and ab is None)

print("── live DB: long NON-document turn gets chunked + dedupe ──")
from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, EpisodicChunk
import src.workers.document_chunker as dc

db = SessionLocal()
conv = Conversation(memory_scope_type="auto")
db.add(conv)
db.commit()
long_turn = EpisodicMemory(
    conversation_id=conv.id, batch_id=uuid.uuid4(), timestamp=datetime.now(timezone.utc),
    topic_tags=["Creative_&_Media"], intent_tags=["Generation"],
    context_reliance="Long_Term_Memory",
    raw_text="User: continue\n\nAssistant: " + ("The chapter unfolds with new events. " * 200),
    decay_score=1.0, embedding=[0.02] * 1024,
    idempotency_key=f"test-c3-{uuid.uuid4()}",
)
db.add(long_turn)
db.commit()
try:
    healed = dc.run_pending_documents(db, limit=10)
    n = db.query(EpisodicChunk).filter_by(turn_id=long_turn.id).count()
    check("catch-up chunks a long non-document turn", n > 1)

    orch2 = HybridRetrievalOrchestrator(db, dc.shared_embedder)
    q = dc.shared_embedder.encode("the chapter unfolds", convert_to_tensor=False).tolist()
    clf2 = SimpleNamespace(intent_tags=["Factual_Retrieval"], topic_tags=[],
                           prompt="the chapter unfolds", max_confidence=0.9)
    frags = orch2._vector_episodic(q, clf2, scope=None, conv_id=str(conv.id))
    parents = [f for f in frags if f.source_batch_id == str(long_turn.id)]
    full_texts = [f.text for f in parents]
    check("parent turn retrieved (non-doc turns stay in turn-level search)",
          len(parents) >= 1)
    check("DEDUPE: no chunk fragment when its parent is already in results",
          sum(1 for f in parents) == 1)
finally:
    db.rollback()
    db.query(EpisodicChunk).filter_by(turn_id=long_turn.id).delete()
    db.query(EpisodicMemory).filter_by(conversation_id=conv.id).delete()
    db.query(Conversation).filter_by(id=conv.id).delete()
    db.commit()
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
