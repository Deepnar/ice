"""Does the procedural leg return a fragment REGARDLESS of the question?

The typed scorer credits `procedural` probes on "did any procedural fragment
come back at all". Across 372 probes it returned exactly 1 every time, in all
five classes. If it also fires on questions the corpus cannot answer, the
1.000 is a constant, not a measurement.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
from sqlalchemy import text
from src.api.config import settings
from src.api.db import SessionLocal
from src.api.context_ledger import effective_memory_budget
from src.api.memory_decision import (decide_memory_retrieval, derive_total_budget,
                                     estimate_recent_window_tokens)
from src.classifier.classifier import PyTorchClassifier
from src.memory.embedder import get_embedder
from src.memory.models import EpisodicMemory
from src.memory.tokens import estimate_from_chars
from src.model_registry.registry import find_best_model, get_model_context_window
from src.model_registry.runtime_probe import serving_window
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from sqlalchemy import func

db = SessionLocal()
settings.codex_reinforce_increment = 0.0
settings.decay_strengthen_amount = 0.0
settings.retrieval_strengthen_writes = False

conv_of = {}
for cid, key in db.execute(text(
        "select conversation_id, idempotency_key from episodic_memory "
        "where idempotency_key like 'z1seed-%'")):
    _, slug, _turn = key.split("-", 2)
    conv_of[slug] = str(cid)

embedder = get_embedder()
clf = PyTorchClassifier(model_path=settings.classifier_model_path,
                        schema_path=settings.label_schema_path)
orch = HybridRetrievalOrchestrator(db, embedder)

slug = sorted(conv_of)[0]
cid = conv_of[slug]
tc = db.query(EpisodicMemory).filter_by(conversation_id=cid).count()
tt = estimate_from_chars(db.query(func.coalesce(
    func.sum(func.length(EpisodicMemory.raw_text)), 0)
).filter_by(conversation_id=cid).scalar() or 0)


def retrieve(q):
    c = clf.classify(q[:2000])
    emb = embedder.encode(q, convert_to_tensor=False).tolist()
    m, _ = find_best_model(c.topic_tags, c.intent_tags)
    rw = get_model_context_window(m)
    win = serving_window(m, rw) if settings.context_use_serving_window else rw
    tb = effective_memory_budget(derive_total_budget(win, settings), q,
                                 generation_reserve=settings.context_generation_reserve,
                                 floor=settings.context_budget_floor)
    d = decide_memory_retrieval(c, turn_count=tc, total_tokens=tt, settings=settings,
                                recent_window_tokens=estimate_recent_window_tokens(tc, tb),
                                timescope_mode="current", coding_scope=False)
    if not d.retrieve:
        return None
    c.context_reliance = "Long_Term_Memory"
    orch.set_budget_from_turn_count(tc, total_tokens=tt, classification=c, total_budget=tb)
    return orch.retrieve(classification=c, conversation_id=cid,
                         prompt_embedding=emb, scope=None)


NONSENSE = [
    "what is the airspeed velocity of an unladen swallow in Patagonia",
    "describe the mating rituals of the Antarctic tube worm",
    "how many bricks are in the Great Wall of China exactly",
    "what did Napoleon eat for breakfast on 3 March 1806",
    "explain the tax code of the Duchy of Grand Fenwick",
    "zzzq wumpus glorbnax fleeb",
]
print("\n" + "=" * 70)
print("PROCEDURAL LEG — does it fire on questions the corpus cannot answer?")
print("=" * 70)
for q in NONSENSE:
    frags = retrieve(q)
    if frags is None:
        print(f"  DECLINED           | {q[:52]}")
        continue
    n_proc = sum(1 for f in frags if f.source_type == "procedural")
    texts = [f.text for f in frags if f.source_type == "procedural"]
    print(f"  procedural={n_proc}  total={len(frags):<3} | {q[:52]}")
    for t in texts:
        print(f"       -> {(t or '')[:150]}")
db.close()
