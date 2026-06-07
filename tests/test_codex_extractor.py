#!/usr/bin/env python3
"""Integration test for the Codex Extractor – uses a high‑value turn with many entities."""

import sys
import os
import uuid
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, Conversation, CodexEntity, CodexEdge, CodexEvent
from src.workers.post_flight import evaluate_turn

# 1. Create a fresh test conversation and a fresh fake turn
db = SessionLocal()
conv = Conversation(id=uuid.uuid4(), memory_scope_type="auto")
db.add(conv)
db.commit()

conv_id = conv.id
batch_id = uuid.uuid4()

# High‑level prompt and a comprehensive assistant response
user_prompt = (
    "Explain how the ICE system uses PostgreSQL with pgvector, Redis, Celery, and a 1.5B model "
    "to evaluate and store conversational memory. Include code examples for the Post‑Flight Evaluator "
    "and the Codex Extractor."
)

assistant_response = (
    "ICE is a middleware that intercepts chat requests and stores every turn in the episodic_memory table. "
    "PostgreSQL with the pgvector extension handles vector embeddings for similarity search. "
    "Redis is used as the message broker for Celery, which runs the Post‑Flight Evaluator. "
    "The evaluator checks if the response contains code blocks or proper nouns and marks the turn as lossless. "
    "If the turn is lossless, it triggers the Codex Extractor, which calls a 1.5B model (Qwen2.5‑1.5B‑Instruct‑AWQ) "
    "to extract subject‑relation‑object triplets.\n\n"
    "Here is an example of the Post‑Flight Evaluator task:\n\n"
    "```python\n"
    "@app.task(bind=True, max_retries=5)\n"
    "def evaluate_turn(self, batch_id, prompt, response, conversation_id):\n"
    "    if is_lossless(response):\n"
    "        extract_codex.delay(batch_id)\n"
    "```\n\n"
    "And the Codex Extractor calls the 1.5B model like this:\n\n"
    "```python\n"
    "from openai import OpenAI\n"
    "client = OpenAI(base_url='http://localhost:8002/v1', api_key='dummy')\n"
    "completion = client.chat.completions.create(\n"
    "    model='Qwen/Qwen2.5-1.5B-Instruct-AWQ',\n"
    "    messages=[{'role': 'system', 'content': 'Extract triplets'}, {'role': 'user', 'content': text}]\n"
    ")\n"
    "```\n\n"
    "The extracted triplets are stored in the codex_entities and codex_edges tables, "
    "which together form the knowledge graph of the system."
)

turn = EpisodicMemory(
    conversation_id=conv_id,
    batch_id=batch_id,
    timestamp=datetime.now(timezone.utc),
    topic_tags=["Software_&_Tech"],
    intent_tags=["Generation"],
    context_reliance="Long_Term_Memory",
    raw_text=f"User: {user_prompt}\n\nAssistant: {assistant_response}",
    idempotency_key=str(uuid.uuid4()),
    embedding=[0.0] * 384
)
db.add(turn)
db.commit()
db.close()

print(f"✅ Inserted test turn with batch_id {batch_id}")

# 2. Fire the Post‑Flight Evaluator (will trigger Codex Extractor if lossless)
evaluate_turn.delay(
    batch_id=str(batch_id),
    prompt=user_prompt,
    response=assistant_response,
    conversation_id=str(conv_id)
)
print("✅ Enqueued evaluate_turn – will run Post‑Flight then Codex Extractor")

# 3. Wait for Celery to process
time.sleep(10)

# 4. Check the Codex tables
db = SessionLocal()
entities = db.query(CodexEntity).all()
edges = db.query(CodexEdge).all()
events = db.query(CodexEvent).all()
print(f"\n📊 Codex Entities: {len(entities)}")
for e in entities:
    print(f"   - {e.canonical_name}")
print(f"📊 Codex Edges: {len(edges)}")
for e in edges:
    print(f"   - {e.relation} (confidence: {e.confidence})")
print(f"📊 Codex Events: {len(events)}")
for e in events:
    print(f"   - {e.event_type}: {e.payload}")
db.close()