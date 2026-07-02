#!/usr/bin/env python3
"""
Test the VECTOR BASELINE with thinking mode against the current database,
exactly as run_mature_experiment.py does it.
"""

import sys, os, time, json, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
from sqlalchemy import bindparam, text
from pgvector.sqlalchemy import Vector as PgVector
from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier

# ── Config ──
OLLAMA_URL = "http://localhost:11434/v1"
GENERALIST_MODEL = "gemma4:26b-a4b-it-q4_K_M"

PROBES = [
    {
        "conversation_id": "633e26f8-5889-5c21-8c70-f4d7ab22cb00",
        "question": "hey, what roles did ai-chan pick out for everyone for the student council?",
        "label": "Student Council Roles"
    },
    {
        "conversation_id": "633e26f8-5889-5c21-8c70-f4d7ab22cb00",
        "question": "was there some reason why shinchan's grades suddenly got so much better after ai-chan came back?",
        "label": "Shinchan Grades"
    },
]

def estimate_tokens(text):
    return int(len(text.split()) * 1.33)

def main():
    print("Loading classifier for embedding…")
    classifier = PyTorchClassifier(
        model_path="models/classifier/ice_classifier_v3_qwen_ft3.pt",
        schema_path="data/labeled/label_schema.json",
    )
    embedder = classifier.embedder
    client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")

    for probe_cfg in PROBES:
        cid = probe_cfg["conversation_id"]
        question = probe_cfg["question"]
        label = probe_cfg["label"]

        print(f"\n{'='*70}")
        print(f"  VECTOR BASELINE: {label}")
        print(f"  Question: {question}")
        print(f"{'='*70}")

        db = SessionLocal()
        emb = embedder.encode(question, convert_to_tensor=False).tolist()

        # Exact same SQL as run_mature_experiment.py (vector condition)
        query = text("""
            SELECT raw_text, summary_text, lossless_flag, inject_raw,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM episodic_memory
            WHERE embedding IS NOT NULL AND is_archived = false
              AND conversation_id = :conv_id
            ORDER BY score DESC LIMIT 30
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        rows = db.execute(query, {"prompt_embedding": emb, "conv_id": cid}).fetchall()

        fragments = []
        for r in rows:
            text_val = r.raw_text if r.lossless_flag else (r.summary_text or r.raw_text[:300])
            fragments.append(text_val)

        context = "\n\n".join(fragments)
        tokens_injected = sum(estimate_tokens(t) for t in fragments)

        # Exact same messages as run_mature_experiment.py (vector condition)
        messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]

        print(f"  Fragments: {len(fragments)} ({tokens_injected} tokens)")

        print(f"\n  Generating with {GENERALIST_MODEL} + thinking…")
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=GENERALIST_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=6144,
                timeout=180.0,
                extra_body={"num_ctx": 64000, "think": True},
            )
            raw = resp.choices[0].message.content or ""
            answer = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
            if not answer:
                answer = raw
            print(f"  Answer ({time.time()-start:.1f}s, {len(answer.split())} words):")
            print(f"  {answer}")
            if len(answer) > 800:
                print(f"  … (truncated)")
        except Exception as e:
            print(f"  ERROR: {e}")

        db.close()

if __name__ == "__main__":
    main()