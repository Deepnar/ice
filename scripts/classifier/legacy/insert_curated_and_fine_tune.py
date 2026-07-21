#!/usr/bin/env python3
import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uuid
from src.api.db import SessionLocal
from src.memory.models import CuratedLabel
from src.workers.fine_tune import fine_tune_classifier

PROBES_FIXED = "data/labeled/probes_labeled_ltm.jsonl"

db = SessionLocal()
count = 0
with open(PROBES_FIXED, "r") as f:
    for line in f:
        item = json.loads(line)
        label = item["label"]
        db.add(CuratedLabel(
            batch_id=uuid.uuid4(),
            prompt=item["prompt"],
            corrected_topic_labels=label.get("topic_labels", []),
            corrected_intent_labels=label.get("intent_labels", []),
            corrected_context_reliance="Long_Term_Memory",
        ))
        count += 1
db.commit()
db.close()
print(f"Inserted {count} curated labels.")

# Trigger fine‑tuning
fine_tune_classifier()
print("Fine‑tuning ran inline (direct call, C7).")