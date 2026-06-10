#!/usr/bin/env python3
"""Scoped retrieval evaluation using exact ground‑truth turn matching (timestamp‑normalised)."""

import json, os, sys, uuid, csv
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

HELD_OUT = "data/simulation/held_out_set.jsonl"
OUTPUT_CSV = "experiments/scoped_eval_results.csv"

# ------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------
db = SessionLocal()
classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json",
)
orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)

# ------------------------------------------------------------------
# 2. Load held‑out set
# ------------------------------------------------------------------
with open(HELD_OUT, "r", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f if line.strip()]

# ------------------------------------------------------------------
# 3. Build a timestamp‑to‑IDs map (both sides normalised)
# ------------------------------------------------------------------
ts_to_ids = defaultdict(list)
rows = db.execute(text("SELECT id, timestamp FROM episodic_memory")).fetchall()
for row in rows:
    # PostgreSQL datetime with tzinfo -> ISO string without trailing "Z"
    ts_norm = row.timestamp.isoformat()
    ts_to_ids[ts_norm].append(str(row.id))

# ------------------------------------------------------------------
# 4. Evaluate each prompt
# ------------------------------------------------------------------
results = []
for entry in eval_set:
    prompt = entry["test_prompt"]
    target_cid = entry["conversation_id"]
    timestamps = entry.get("original_timestamps", [])

    # Build the set of relevant IDs by normalising the held‑out timestamps
    relevant_ids = set()
    for ts in timestamps:
        try:
            ts_clean = ts.replace("Z", "")
            dt = datetime.fromisoformat(ts_clean)
            ts_norm = dt.isoformat()
            relevant_ids.update(ts_to_ids.get(ts_norm, []))
        except Exception:
            pass

    classification = classifier.classify(prompt)
    embedding = classifier.embedder.encode(prompt, convert_to_tensor=False).tolist()

    # Scoped retrieval – only inside the target conversation
    fragments = orchestrator.retrieve(
        classification=classification,
        conversation_id=str(uuid.uuid4()),
        prompt_embedding=embedding,
        scope={"conversation_id": target_cid},
    )

    # Precision@5: how many of the top‑5 are in the ground‑truth set?
    hits = 0
    for frag in fragments[:5]:
        if frag.source_batch_id in relevant_ids:
            hits += 1
    precision = hits / 5.0 if fragments else 0.0

    results.append({
        "prompt": prompt[:120],
        "precision@5": precision,
        "tokens_fetched": sum(f.token_count for f in fragments),
        "context_reliance": classification.context_reliance,
    })

# ------------------------------------------------------------------
# 5. Aggregate and save
# ------------------------------------------------------------------
avg_precision = sum(r["precision@5"] for r in results) / len(results) if results else 0
total_tokens = sum(r["tokens_fetched"] for r in results)
zero_shot = sum(1 for r in results if r["context_reliance"] == "Zero_Shot")

print(f"Scoped Evaluation ({len(results)} prompts):")
print(f"  Avg Precision@5 : {avg_precision:.4f}")
print(f"  Total tokens    : {total_tokens}")
print(f"  Zero‑Shot gated : {zero_shot}/{len(results)}")

with open(OUTPUT_CSV, "w", newline="") as cf:
    writer = csv.DictWriter(cf, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"  Per‑prompt results → {OUTPUT_CSV}")
db.close()