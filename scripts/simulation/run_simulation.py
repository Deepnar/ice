#!/usr/bin/env python3
"""Simulation harness that respects conversation_id. Reuses existing Conversation rows."""

import argparse, json, os, uuid, time
from datetime import datetime, timezone
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import random as rn, numpy as np

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, Conversation
from src.classifier.classifier import PyTorchClassifier
from src.workers.post_flight import is_lossless, generate_summary
from src.workers.codex_extractor import extract_triplets, handle_triplet

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--speed', type=float, default=1.0)
args = parser.parse_args()

# Reproducibility
rn.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json",
)
embedder = classifier.embedder

db = SessionLocal()

with open(args.input, 'r', encoding='utf-8') as f:
    turns = [json.loads(line) for line in f if line.strip()]

turns.sort(key=lambda x: x.get("timestamp", ""))

conv_cache = {}   # conversation_id -> Conversation row

for i, entry in enumerate(turns):
    prompt = entry["prompt"]
    response = entry.get("response", "")
    ts_str = entry.get("timestamp")
    cid = entry.get("conversation_id", "default")

    # Get or create Conversation object
        # Safely convert conversation_id to UUID
    cid_raw = entry.get("conversation_id", "")
    try:
        cid_uuid = uuid.UUID(cid_raw) if cid_raw else uuid.uuid4()
    except (ValueError, AttributeError):
        # Fallback: generate a deterministic UUID from the string
        cid_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, str(cid_raw))

    if cid_uuid not in conv_cache:
        conv = db.query(Conversation).filter_by(id=cid_uuid).first()
        if not conv:
            conv = Conversation(id=cid_uuid, memory_scope_type="auto")
            db.add(conv)
            db.flush()
        conv_cache[cid_uuid] = conv
    else:
        conv = conv_cache[cid_uuid]

    # Classify
    result = classifier.classify(prompt)
    result.prompt = prompt

    # Compute embedding
    emb = embedder.encode(prompt, convert_to_tensor=False).tolist()

    batch_id = uuid.uuid4()
    ts_str_clean = ts_str.replace("Z", "")
    timestamp = datetime.fromisoformat(ts_str_clean)

    turn = EpisodicMemory(
        conversation_id=conv.id,
        batch_id=batch_id,
        timestamp=timestamp,
        topic_tags=result.topic_tags,
        intent_tags=result.intent_tags,
        context_reliance=result.context_reliance,
        raw_text=f"User: {prompt}\n\nAssistant: {response}",
        embedding=emb,
        idempotency_key=str(uuid.uuid4()),
    )
    db.add(turn)
    db.flush()

    # Post‑flight evaluation
    lossless = is_lossless(response)
    summary = None if lossless else generate_summary(prompt, response)
    turn.lossless_flag = lossless
    turn.summary_text = summary

    # Codex extraction if lossless
    if lossless:
        triplets = extract_triplets(turn.raw_text)
        for t in triplets:
            if isinstance(t, dict) and "subject" in t and "relation" in t and "object" in t:
                s, r, o = t["subject"].strip(), t["relation"].strip(), t["object"].strip()
                if s and r and o:
                    handle_triplet(db, s, r, o, str(batch_id))

    db.commit()
    if i % 10 == 0:
        print(f"Processed {i+1}/{len(turns)} turns...")
    time.sleep(0.01 / args.speed)

db.close()
print(f"Simulation complete. {len(turns)} turns processed.")