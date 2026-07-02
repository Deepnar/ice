#!/usr/bin/env python3
"""End-to-end test of ICE Generalist on a single probe."""
import sys, os, time, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.prompt_assembler import assemble_prompt
from src.memory.models import MemorySlot

# Config
CID = "633e26f8-5889-5c21-8c70-f4d7ab22cb00"   # Shinchan
PROBE = "so do you remember the fake wedding why did that happen??"
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
print(f"Classification: {classification.topic_tags} | {classification.intent_tags} | {classification.context_reliance}\n")

# 2. Retrieve
orchestrator = HybridRetrievalOrchestrator(db, embedder)
orchestrator.set_budget_from_turn_count(290, 236757)   # match the checkpoint params
emb = embedder.encode(PROBE, convert_to_tensor=False).tolist()
fragments = orchestrator.retrieve(
    classification=classification,
    conversation_id=CID,
    prompt_embedding=emb,
    scope={"conversation_id": CID},
)
print(f"Retrieved {len(fragments)} fragments ({sum(f.token_count for f in fragments)} tokens)")

# 3. Assemble prompt
memory_slots = db.query(MemorySlot).filter_by(is_active=True).all()
messages = assemble_prompt(
    memory_slots=memory_slots,
    retrieved_fragments=fragments,
    user_message=PROBE,
    db_session=db,
    conversation_id=CID,
    classification=classification,
)

# 4. Generate
print("System prompt (first 500 chars):")
print(messages[0]["content"][:500] + "...\n")

start = time.time()
resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.0,
    max_tokens=6000,
    timeout=120.0,
)
answer = resp.choices[0].message.content
latency = time.time() - start

print(f"Answer (latency {latency:.1f}s):")
print(answer)

db.close()