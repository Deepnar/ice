#!/usr/bin/env python3
import sys, os, uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, Conversation, CodexEntity
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.classifier.classifier import ClassificationResult

embedder = SentenceTransformer("all-MiniLM-L6-v2")
test_embedding = embedder.encode("What is ICE?", convert_to_tensor=False).tolist()

db = SessionLocal()
# Clear everything
db.execute(text("TRUNCATE episodic_memory, conversations, codex_entities, codex_edges CASCADE"))
db.commit()

conv = Conversation(id=uuid.uuid4(), memory_scope_type="auto")
db.add(conv)
db.commit()

# Turn 1 – lossless false, summary provided
turn1 = EpisodicMemory(
    conversation_id=conv.id,
    batch_id=uuid.uuid4(),
    timestamp=datetime.now(timezone.utc),
    topic_tags=["Software_&_Tech"],
    intent_tags=["Generation"],
    context_reliance="Long_Term_Memory",
    raw_text="User: What is ICE? Assistant: ICE is a memory middleware.",
    lossless_flag=False,
    summary_text="ICE is a memory middleware that provides context for AI assistants.",
    is_archived=False,
    embedding=test_embedding,
    idempotency_key=str(uuid.uuid4())
)
# Turn 2 – lossless true, no summary
turn2 = EpisodicMemory(
    conversation_id=conv.id,
    batch_id=uuid.uuid4(),
    timestamp=datetime.now(timezone.utc),
    topic_tags=["Software_&_Tech"],
    intent_tags=["Troubleshooting"],
    context_reliance="Long_Term_Memory",
    raw_text="User: I have a bug in my code. Assistant: Check the indentation.",
    lossless_flag=True,
    is_archived=False,
    embedding=test_embedding,
    idempotency_key=str(uuid.uuid4())
)
db.add_all([turn1, turn2])
db.commit()

# Codex entity with a name that will match our prompt
unique_name = f"ice-test-{uuid.uuid4().hex[:6]}"
entity = CodexEntity(
    id=uuid.uuid4(),
    canonical_name=unique_name,
    aliases=["ice", unique_name.lower()],      # include both uppercase and lowercase
    context_payload="ICE is an Infinite Context Engine that uses PostgreSQL and pgvector.",
    last_updated=datetime.now(timezone.utc)
)
db.add(entity)
db.commit()

# Run retrieval
orchestrator = HybridRetrievalOrchestrator(db, embedder)
result = ClassificationResult(
    topic_tags=["Software_&_Tech"],
    intent_tags=["Generation"],
    context_reliance="Long_Term_Memory",
    raw_probs=[0.9]*25,
    max_confidence=0.9,
    prompt="What is ICE?"
)
prompt_embedding = embedder.encode("What is ICE?", convert_to_tensor=False).tolist()

fragments = orchestrator.retrieve(
    classification=result,
    conversation_id=str(conv.id),
    prompt_embedding=prompt_embedding,
    scope=None
)

print(f"\nRetrieved {len(fragments)} fragments:")
for f in fragments:
    print(f"  [{f.source_type}] score={f.score:.4f}, text={f.text[:80]}...")

from src.api.prompt_assembler import assemble_prompt
messages = assemble_prompt([], fragments, "What is ICE?")
print("\nAssembled system prompt (first 500 chars):")
print(messages[0]["content"][:500])

db.close()