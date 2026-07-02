#!/usr/bin/env python3
"""Test the claw‑machine probe at turn 55 with the current DB state."""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.prompt_assembler import assemble_prompt
from src.memory.models import MemorySlot, EpisodicMemory

CID  = "633e26f8-5889-5c21-8c70-f4d7ab22cb00"
PROBE = "did shinchan actually win the claw machine game?"
OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "gemma4:26b-a4b-it-q4_K_M"

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v3_qwen_ft3.pt",
    schema_path="data/labeled/label_schema.json",
)
embedder = classifier.embedder
client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")

db = SessionLocal()

# 1. Classify
classification = classifier.classify(PROBE, conversation_id=CID)
print(f"Classification: {classification.topic_tags} | {classification.intent_tags} | {classification.context_reliance}")

# 2. Retrieve (same budget as experiment at turn 55)
orchestrator = HybridRetrievalOrchestrator(db, embedder)
orchestrator.set_budget_from_turn_count(55, 51345)   # 55 turns, ~51k tokens
emb = embedder.encode(PROBE, convert_to_tensor=False).tolist()
fragments = orchestrator.retrieve(
    classification=classification,
    conversation_id=CID,
    prompt_embedding=emb,
    scope={"conversation_id": CID},
)
# Debug: check the raw BM25 leg and the fused list
bm25_debug = orchestrator._bm25_episodic(classification, None, CID, PROBE)
if bm25_debug:
    top_bm25 = bm25_debug[0]
    print(f"  DEBUG top BM25 text: {top_bm25.text[:120]}...")
    print(f"  DEBUG top BM25 score: {top_bm25.score}, token_count: {top_bm25.token_count}")
    in_fused = any(top_bm25.text == f.text for f in fragments)
    print(f"  DEBUG top BM25 in fused (by text equality): {in_fused}")
else:
    print("  DEBUG BM25 returned empty!")
print(f"\nRetrieved {len(fragments)} fragments ({sum(f.token_count for f in fragments)} tokens)")

# 3. Check if the correct fragment is present
found = False
for f in fragments:
    if "claw machine" in f.text.lower() or "samurai keychain" in f.text.lower():
        found = True
        print(f"  ✅ Found relevant fragment: score={f.score:.4f}, type={f.source_type}, {f.text[:120]}...")
        break
if not found:
    print("  ❌ No fragment mentions claw machine or samurai keychain")
    # Also check if it's in the top BM25/vector legs
    bm25 = orchestrator._bm25_episodic(classification, None, CID, PROBE)
    for f in bm25:
        if "claw" in f.text.lower():
            print(f"  BM25 has it (score={f.score:.4f}) but it was dropped from fused list")
            break

# 4. Assemble prompt and generate answer
memory_slots = db.query(MemorySlot).filter_by(is_active=True).all()
messages = assemble_prompt(
    memory_slots=memory_slots,
    retrieved_fragments=fragments,
    user_message=PROBE,
    db_session=db,
    conversation_id=CID,
    classification=classification,
)
print("\nSystem prompt (first 400 chars):")
print(messages[0]["content"][:400] + "...\n")

start = time.time()
resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.0,
    max_tokens=4096,
    timeout=120.0,
)
answer = resp.choices[0].message.content
print(f"Answer (latency {time.time()-start:.1f}s):\n{answer}")

db.close()