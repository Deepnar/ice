#!/usr/bin/env python3
"""Test the retrieval fixes on a previously-failed probe."""
import sys, os, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.prompt_assembler import assemble_prompt
from src.memory.models import MemorySlot
from openai import OpenAI

cid = "633e26f8-5889-5c21-8c70-f4d7ab22cb00"  # Shinchan
probe = "so do you remember the fake wedding why did that happen??"

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v3_qwen_ft3.pt",
    schema_path="data/labeled/label_schema.json",
)
embedder = classifier.embedder

db = SessionLocal()
orchestrator = HybridRetrievalOrchestrator(db, embedder)
orchestrator.set_budget_from_turn_count(290, 236757)  # the turn & tokens from the checkpoint

classification = classifier.classify(probe, conversation_id=cid)
emb = embedder.encode(probe, convert_to_tensor=False).tolist()

fragments = orchestrator.retrieve(
    classification=classification,
    conversation_id=cid,
    prompt_embedding=emb,
    scope={"conversation_id": cid},
)

print(f"Retrieved {len(fragments)} fragments ({sum(f.token_count for f in fragments)} tokens):")
for f in fragments[:10]:
    print(f"  [{f.source_type}] score={f.score:.4f}  {f.text[:120]}...")

# Check if any fragment mentions "wedding"
mentions_wedding = any("wedding" in f.text.lower() for f in fragments)
print(f"\nMentions 'wedding': {mentions_wedding}")

db.close()