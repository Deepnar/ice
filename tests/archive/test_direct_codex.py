#!/usr/bin/env python3
"""Direct Codex extraction test – no Celery, just the logic."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, CodexEntity, CodexEdge, CodexEvent
from src.workers.codex_extractor import extract_triplets, handle_triplet
import uuid

# Use the batch_id from the last test turn (the one with lossless_flag=True)
batch_id = "cebd7fe5-0a24-41e4-83c7-80afb8e480f3"  # <-- adjust if needed

db = SessionLocal()
turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
if not turn:
    print("❌ Turn not found – check batch_id")
    db.close()
    exit()

print(f"📝 Raw text:\n{turn.raw_text[:200]}...\n")

triplets = extract_triplets(turn.raw_text)
print(f"🔍 Extracted {len(triplets)} triplets:")
for t in triplets:
    print(f"   {t}")

if not triplets:
    db.close()
    exit()

# Apply triplets to the database
for triplet in triplets:
    s = triplet.get("subject", "").strip()
    r = triplet.get("relation", "").strip()
    o = triplet.get("object", "").strip()
    if s and r and o:
        handle_triplet(db, s, r, o, batch_id)

db.commit()
print("✅ Triplets committed to database.")

# Show results
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
db.close()