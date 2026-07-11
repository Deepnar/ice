#!/usr/bin/env python3
"""
Full pipeline integration test.
Tests:  Classifier → (simulated) storage → Post‑Flight Evaluator → Codex Extractor.

Requires:
- Background 1.5B model running (vllm-bg on port 8002)
- Celery worker running
- Database up (docker compose)

Before running, truncate existing data:
  docker exec -i ice_postgres psql -U ice -d ice_db -c "TRUNCATE episodic_memory, conversations, codex_entities, codex_edges, codex_events, codex_snapshots, idempotency_keys RESTART IDENTITY CASCADE;"
"""

import sys
import os
import uuid
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, Conversation, CodexEntity, CodexEdge, CodexEvent
from src.classifier.classifier import PyTorchClassifier
from src.workers.post_flight import evaluate_turn

# ---------------------------------------------------------------------
# 1. Load the classifier
# ---------------------------------------------------------------------
classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json"
)
print("✅ Classifier loaded")

# ---------------------------------------------------------------------
# 2. A realistic, high‑value prompt (will produce code blocks in response)
# ---------------------------------------------------------------------
user_prompt = (
    "Explain how ICE uses PostgreSQL, Redis, and Celery to provide memory for conversations. "
    "Include a code example for the Post‑Flight Evaluator."
)

assistant_response = (
    "ICE stores every turn in the episodic_memory table in PostgreSQL, using the pgvector extension "
    "for vector similarity search. Redis acts as the message broker for Celery, which runs the "
    "Post‑Flight Evaluator. The evaluator checks the response for code blocks or proper nouns and "
    "marks the turn as lossless if it finds them. Lossless turns trigger the Codex Extractor, "
    "which calls a 1.5B model to extract knowledge triplets.\n\n"
    "Here is an example of the evaluator task:\n\n"
    "```python\n"
    "@app.task(bind=True, max_retries=5)\n"
    "def evaluate_turn(self, batch_id, prompt, response, conversation_id):\n"
    "    lossless = is_lossless(response)\n"
    "    if lossless:\n"
    "        extract_codex.delay(batch_id)\n"
    "```"
)

# ---------------------------------------------------------------------
# 3. Pre‑flight classification (simulating what the proxy does)
# ---------------------------------------------------------------------
result = classifier.classify(user_prompt)
print(f"🔍 Classifier output:")
print(f"   topics: {result.topic_tags}")
print(f"   intents: {result.intent_tags}")
print(f"   context_reliance: {result.context_reliance}")
print(f"   max_confidence: {result.max_confidence:.4f}")

# ---------------------------------------------------------------------
# 4. Create a fake conversation and an episodic memory row
# ---------------------------------------------------------------------
db = SessionLocal()
conv = Conversation(id=uuid.uuid4(), memory_scope_type="auto")
db.add(conv)
db.commit()

conv_id = conv.id
batch_id = uuid.uuid4()

turn = EpisodicMemory(
    conversation_id=conv_id,
    batch_id=batch_id,
    timestamp=datetime.now(timezone.utc),
    topic_tags=result.topic_tags,
    intent_tags=result.intent_tags,
    context_reliance=result.context_reliance,
    raw_text=f"User: {user_prompt}\n\nAssistant: {assistant_response}",
    idempotency_key=str(uuid.uuid4()),
    embedding=[0.0] * 384
)
db.add(turn)
db.commit()
db.close()

print(f"✅ Inserted turn with batch_id {batch_id}")
print(f"   lossless_flag initially: NULL (will be set by Post‑Flight Evaluator)")

# ---------------------------------------------------------------------
# 5. Fire the Post‑Flight Evaluator (simulating the proxy's background call)
# ---------------------------------------------------------------------
evaluate_turn(
    batch_id=str(batch_id),
    prompt=user_prompt,
    response=assistant_response,
    conversation_id=str(conv_id)
)
print("✅ Ran evaluate_turn (direct call, C7)")
time.sleep(10)

# ---------------------------------------------------------------------
# 6. Verify the Post‑Flight Evaluator updated the turn
# ---------------------------------------------------------------------
db = SessionLocal()
turn_after = db.query(EpisodicMemory).filter_by(batch_id=batch_id).first()
if turn_after:
    print(f"\n🔎 Post‑Flight result:")
    print(f"   lossless_flag: {turn_after.lossless_flag}")
    print(f"   summary_text:  {turn_after.summary_text or 'None (lossless turns skip summarisation)'}")
    if turn_after.lossless_flag:
        print("   ✅ Turn correctly marked as lossless (code block detected).")
else:
    print("❌ Turn not found – something went wrong.")

# ---------------------------------------------------------------------
# 7. Check the Codex tables (extractor should have run)
# ---------------------------------------------------------------------
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

print("\n✅ Full pipeline test complete.")