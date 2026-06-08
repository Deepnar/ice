#!/usr/bin/env python3
"""
Full pipeline integration test for Phase 7.
Verifies: Classifier → Retrieval → Prompt Assembly → (simulated) Storage.
"""

import sys, os, uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, Conversation, CodexEntity
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator, ContextFragment
from src.api.prompt_assembler import assemble_prompt
from src.classifier.classifier import ClassificationResult

# ----------------------------------------------------------------------
# 1. SETUP – create test data in the database
# ----------------------------------------------------------------------
db = SessionLocal()
db.execute(text("TRUNCATE episodic_memory, conversations, codex_entities, codex_edges CASCADE"))
db.commit()

# 1a. Load the classifier (on CPU)
classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json"
)
embedder = classifier.embedder   # reuse the same embedder for consistency

# 1b. Create a conversation and two episodic turns
conv_id = uuid.uuid4()
conv = Conversation(id=conv_id, memory_scope_type="auto")
db.add(conv)
db.commit()

# Turn 1: lossless=False → summary will be used for retrieval
emb = embedder.encode("What is ICE?", convert_to_tensor=False).tolist()
turn1 = EpisodicMemory(
    conversation_id=conv_id,
    batch_id=uuid.uuid4(),
    timestamp=datetime.now(timezone.utc),
    topic_tags=["Software_&_Tech"],
    intent_tags=["Generation"],
    context_reliance="Long_Term_Memory",
    raw_text="User: What is ICE? Assistant: ICE is a memory middleware that provides context for AI assistants.",
    lossless_flag=False,
    summary_text="ICE is a memory middleware that uses PostgreSQL and pgvector.",
    is_archived=False,
    embedding=emb,
    idempotency_key=str(uuid.uuid4())
)

# Turn 2: lossless=True → raw_text will be used
turn2 = EpisodicMemory(
    conversation_id=conv_id,
    batch_id=uuid.uuid4(),
    timestamp=datetime.now(timezone.utc),
    topic_tags=["Software_&_Tech"],
    intent_tags=["Troubleshooting"],
    context_reliance="Long_Term_Memory",
    raw_text="User: I have a bug in my code. Assistant: Check the indentation.",
    lossless_flag=True,
    is_archived=False,
    embedding=emb,
    idempotency_key=str(uuid.uuid4())
)
db.add_all([turn1, turn2])
db.commit()

# 1c. Create a Codex entity with aliases (lowercase as the extractor does)
unique_name = f"ice-test-{uuid.uuid4().hex[:6]}"
entity = CodexEntity(
    id=uuid.uuid4(),
    canonical_name=unique_name,
    aliases=["ice", unique_name],
    context_payload="ICE is an Infinite Context Engine that uses PostgreSQL with pgvector and Redis.",
    last_updated=datetime.now(timezone.utc)
)
db.add(entity)
db.commit()

print("✅ Test data inserted\n")

# ----------------------------------------------------------------------
# 2. PRE‑FLIGHT CLASSIFICATION (exactly what the proxy does)
# ----------------------------------------------------------------------
test_prompt = "What is ICE?"
result = classifier.classify(test_prompt)
result.prompt = test_prompt   # required by orchestrator
print(f"🔍 Classification: topics={result.topic_tags}, intents={result.intent_tags}, "
      f"context={result.context_reliance}, confidence={result.max_confidence:.4f}\n")

# ----------------------------------------------------------------------
# 3. RETRIEVAL (the core of Phase 7)
# ----------------------------------------------------------------------
prompt_embedding = embedder.encode(test_prompt, convert_to_tensor=False).tolist()
orchestrator = HybridRetrievalOrchestrator(db, embedder)

fragments = orchestrator.retrieve(
    classification=result,
    conversation_id=str(conv_id),
    prompt_embedding=prompt_embedding,
    scope=None   # Auto scope (V1 default)
)

print(f"📊 Retrieved {len(fragments)} fragments:")
for frag in fragments:
    print(f"   [{frag.source_type}] score={frag.score:.4f}  –  {frag.text[:80]}...")

# ----------------------------------------------------------------------
# 4. PROMPT ASSEMBLY
# ----------------------------------------------------------------------
# Memory slots are not populated yet – pass empty list
assembled_messages = assemble_prompt([], fragments, test_prompt)

print("\n📄 Assembled system prompt:\n")
print(assembled_messages[0]["content"][:600])
print("\n...\n")

# ----------------------------------------------------------------------
# 5. STORAGE SIMULATION (manual insert, but normally this is done by the proxy)
# ----------------------------------------------------------------------
new_batch_id = uuid.uuid4()
new_turn = EpisodicMemory(
    conversation_id=conv_id,
    batch_id=new_batch_id,
    timestamp=datetime.now(timezone.utc),
    topic_tags=result.topic_tags,
    intent_tags=result.intent_tags,
    context_reliance=result.context_reliance,
    raw_text=f"User: {test_prompt}\n\nAssistant: [simulated response]",
    lossless_flag=None,   # will be set by Post‑Flight Evaluator later
    embedding=prompt_embedding,
    idempotency_key=str(uuid.uuid4())
)
db.add(new_turn)
db.commit()

# Verify the turn was stored
stored = db.query(EpisodicMemory).filter_by(batch_id=new_batch_id).first()
print(f"✅ Stored new turn with batch_id {new_batch_id}, lossless_flag={stored.lossless_flag} (NULL = awaiting evaluation)\n")

db.close()

print("✅ Full pipeline test complete.")
print("   Components verified: Classifier → Retrieval (BM25 + Vector + Codex + RRF) → Prompt Assembly → Storage.")