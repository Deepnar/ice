#!/usr/bin/env python3
"""Quickly test a single probe against an already-prepared database."""
import sys, os, json, time, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from src.api.db import SessionLocal
from src.memory.models import MemorySlot
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.prompt_assembler import assemble_prompt

OLLAMA_URL = "http://localhost:11434/v1"
SINGLE_MODEL = "gemma4:26b-a4b-it-q4_K_M"

# ── Config ──
CONVERSATION_ID = "633e26f8-5889-5c21-8c70-f4d7ab22cb00"   # Shinchan conversation
PROBE_PROMPT = "who all are in the inital friend grp of shin chan, and who all got added later??"

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json",
)
embedder = classifier.embedder
client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")

# Classify
classification = classifier.classify(PROBE_PROMPT)
print(f"Classification: {classification.topic_tags} | {classification.intent_tags} | {classification.context_reliance}")

# Retrieve
orchestrator = HybridRetrievalOrchestrator(SessionLocal(), embedder)
orchestrator.max_retrieval_tokens = 5000   # same as your Phase 2 setting
emb = embedder.encode(PROBE_PROMPT, convert_to_tensor=False).tolist()
fragments = orchestrator.retrieve(
    classification=classification,
    conversation_id=CONVERSATION_ID,
    prompt_embedding=emb,
    scope={"conversation_id": CONVERSATION_ID},
)
tokens_injected = sum(f.token_count for f in fragments)
print(f"\nTokens injected: {tokens_injected}")
print(f"\nRetrieved {len(fragments)} fragments:")
for i, f in enumerate(fragments[:10]):
    print(f"  [{i}] [{f.source_type}] score={f.score:.4f}  {f.text[:100]}...")

# Assemble prompt
memory_slots = SessionLocal().query(MemorySlot).filter_by(is_active=True).all()
messages = assemble_prompt(
    memory_slots=memory_slots,
    retrieved_fragments=fragments,
    user_message=PROBE_PROMPT,
    db_session=SessionLocal(),
    conversation_id=CONVERSATION_ID,
    classification=classification,
)

# Generate answer
resp = client.chat.completions.create(
    model=SINGLE_MODEL,
    messages=messages,
    temperature=0.0,
    max_tokens=4096,
    timeout=120.0,
)
answer = resp.choices[0].message.content
print(f"\nAnswer:\n{answer}")