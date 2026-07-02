#!/usr/bin/env python3
"""Test retrieval for FACT-01 exactly as the experiment does."""
import sys, os, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.memory.models import MemorySlot
from src.api.prompt_assembler import assemble_prompt

CID = "633e26f8-5889-5c21-8c70-f4d7ab22cb00"
PROMPT = "wait, what was that thing about the grandfathers? did they actually get along?"

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v3_qwen_ft3.pt",
    schema_path="data/labeled/label_schema.json",
)
embedder = classifier.embedder

db = SessionLocal()
orchestrator = HybridRetrievalOrchestrator(db, embedder)
orchestrator.set_budget_from_turn_count(23, 20921)

classification = classifier.classify(PROMPT, conversation_id=CID)
emb = embedder.encode(PROMPT, convert_to_tensor=False).tolist()

# ---- Run each leg individually ----
print("=== BM25 ===")
bm25 = orchestrator._bm25_episodic(classification, None, CID, PROMPT)
for f in bm25:
    print(f"score={f.score:.4f}  id={f.source_batch_id}  has 'grandfather': {'grandfather' in f.text.lower()}  text={f.text[:120]}...")

print("\n=== VECTOR ===")
vec = orchestrator._vector_episodic(emb, classification, None, CID)
for f in vec:
    print(f"score={f.score:.4f}  id={f.source_batch_id}  has 'grandfather': {'grandfather' in f.text.lower()}  text={f.text[:120]}...")

# ---- Full retrieval ----
print("\n=== FUSED (full ICE) ===")
fused = orchestrator.retrieve(
    classification=classification,
    conversation_id=CID,
    prompt_embedding=emb,
    scope={"conversation_id": CID},
)
print(f"Total fragments: {len(fused)}")
for f in fused:
    has = 'grandfather' in f.text.lower()
    print(f"[{f.source_type}] score={f.score:.4f}  has_grandfather={has}  len={len(f.text.split())}w  {f.text[:150]}...")

# ---- Check if any fragment contains the answer ----
print("\n=== ANSWER CHECK ===")
answer_found = any("yoshiji" in f.text.lower() and "ginnosuke" in f.text.lower() for f in fused)
print(f"Grandfather names present: {answer_found}")

db.close()