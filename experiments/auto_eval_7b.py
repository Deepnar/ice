#!/usr/bin/env python3
"""Automatic evaluation using the 7B model as a relevance judge (all prompts)."""

import json, os, sys, uuid, csv
from collections import defaultdict
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

HELD_OUT = "data/simulation/held_out_set.jsonl"
OUTPUT_CSV = "experiments/auto_eval_7b_results.csv"
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
JUDGE_API = "http://localhost:8001/v1"

# ------------------------------------------------------------------
# 1. Setup ICE retrieval (no background model needed)
# ------------------------------------------------------------------
db = SessionLocal()
classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json",
)
orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)

# ------------------------------------------------------------------
# 2. Setup the 7B judge
# ------------------------------------------------------------------
judge = OpenAI(base_url=JUDGE_API, api_key="dummy")

# ------------------------------------------------------------------
# 3. Load held‑out set
# ------------------------------------------------------------------
with open(HELD_OUT, "r", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f if line.strip()]

print(f"Evaluating {len(eval_set)} prompts with 7B judge...")

# ------------------------------------------------------------------
# 4. Evaluate every prompt
# ------------------------------------------------------------------
results = []
for idx, entry in enumerate(eval_set):
    prompt = entry["test_prompt"]
    target_cid = entry["conversation_id"]

    classification = classifier.classify(prompt)
    embedding = classifier.embedder.encode(prompt, convert_to_tensor=False).tolist()

    fragments = orchestrator.retrieve(
        classification=classification,
        conversation_id=str(uuid.uuid4()),
        prompt_embedding=embedding,
        scope={"conversation_id": target_cid},
    )[:5]

    for i, frag in enumerate(fragments):
        # Ask the 7B judge: is this fragment relevant?
        try:
            resp = judge.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are an evaluation assistant. You will be given a user question and a fragment "
                        "from a past conversation. Determine if the fragment contains information that would "
                        "help an AI answer the question. Answer ONLY 'YES' or 'NO'."
                    )},
                    {"role": "user", "content": (
                        f"Question:\n{prompt}\n\n"
                        f"Fragment:\n{frag.text}\n\n"
                        "Is this fragment relevant? YES or NO:"
                    )},
                ],
                temperature=0.0,
                max_tokens=3,
                timeout=30.0,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            relevant = 1 if verdict == "YES" else 0
        except Exception as e:
            print(f"  Judge error for prompt {idx} frag {i}: {e}")
            relevant = 0

        results.append({
            "prompt": prompt,
            "fragment_rank": i + 1,
            "fragment_text": frag.text,
            "relevant": relevant,
            "source_type": frag.source_type,
            "context_reliance": classification.context_reliance,
            "max_confidence": classification.max_confidence,
        })

    if (idx + 1) % 20 == 0:
        print(f"  Judged {idx + 1}/{len(eval_set)} prompts…")

# ------------------------------------------------------------------
# 5. Compute aggregate metrics
# ------------------------------------------------------------------
total_relevant = sum(r["relevant"] for r in results)
num_prompts = len(eval_set)
precision_at_5 = total_relevant / (num_prompts * 5) if num_prompts else 0
total_tokens_fetched = sum(len(f.text.split()) * 1.33 for f in fragments if hasattr(f, 'text'))  # rough estimate

print(f"\nAutomatic Evaluation (7B judge) – {num_prompts} prompts:")
print(f"  Precision@5 : {precision_at_5:.4f}")
print(f"  Zero‑Shot gated       : {sum(1 for r in results if r['context_reliance'] == 'Zero_Shot')}/{num_prompts}")

# ------------------------------------------------------------------
# 6. Save detailed CSV
# ------------------------------------------------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as cf:
    writer = csv.DictWriter(cf, fieldnames=[
        "prompt", "fragment_rank", "fragment_text", "relevant",
        "source_type", "context_reliance", "max_confidence"
    ])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r[k] for k in writer.fieldnames})

print(f"  Detailed results → {OUTPUT_CSV}")
db.close()