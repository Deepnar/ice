#!/usr/bin/env python3
"""Simulation Harness – replays historical conversations for evaluation."""

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone

import sys
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, Conversation
from src.classifier.classifier import PyTorchClassifier
from src.workers.post_flight import is_lossless, generate_summary
from src.workers.codex_extractor import extract_triplets, handle_triplet

parser = argparse.ArgumentParser(description="Run longitudinal simulation for ICE.")
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--input', type=str, default='data/simulation_input.jsonl')
parser.add_argument('--speed', type=float, default=1.0, help='Simulation speed multiplier')
args = parser.parse_args()

# Reproducibility
import random, numpy as np, torch
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json"
)
embedder = classifier.embedder

db = SessionLocal()

with open(args.input, 'r') as f:
    lines = [json.loads(line) for line in f if line.strip()]

# Sort by original timestamp
lines.sort(key=lambda x: x.get('timestamp', ''))

sim_start = datetime.now(timezone.utc)
for i, entry in enumerate(lines):
    prompt = entry['prompt']
    response = entry.get('response', '')

    # Pre‑flight classification
    result = classifier.classify(prompt)
    result.prompt = prompt

    # Create synthetic conversation (committed once)
    conv = Conversation(id=uuid.uuid4(), memory_scope_type='auto')
    db.add(conv)
    db.flush()

    # Compute embedding
    embedding = embedder.encode(prompt, convert_to_tensor=False).tolist()

    # Insert turn (only flush to keep the object active)
    batch_id = uuid.uuid4()
    turn = EpisodicMemory(
        conversation_id=conv.id,
        batch_id=batch_id,
        timestamp=sim_start,
        topic_tags=result.topic_tags,
        intent_tags=result.intent_tags,
        context_reliance=result.context_reliance,
        raw_text=f"User: {prompt}\n\nAssistant: {response}",
        embedding=embedding,
        idempotency_key=str(uuid.uuid4())
    )
    db.add(turn)
    db.flush()  # assigns ID without expiring the object

    # Post‑flight evaluation (synchronous)
    lossless = is_lossless(response)
    summary = None if lossless else generate_summary(prompt, response)
    turn.lossless_flag = lossless
    turn.summary_text = summary

    # Codex extraction if lossless
    if lossless:
        triplets = extract_triplets(turn.raw_text)
        for t in triplets:
            s = t.get("subject", "").strip()
            r = t.get("relation", "").strip()
            o = t.get("object", "").strip()
            if s and r and o:
                handle_triplet(db, s, r, o, str(batch_id))

    # Single commit per turn avoids DetachedInstanceError
    db.commit()

    if i % 10 == 0:
        print(f"Processed {i+1}/{len(lines)} turns...")

    time.sleep(0.01 / args.speed)

db.close()
print(f"Simulation complete. {len(lines)} turns processed.")
print(f"Run ID: {uuid.uuid4()} (seed={args.seed})")