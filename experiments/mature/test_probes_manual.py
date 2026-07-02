#!/usr/bin/env python3
"""
Test specific early probes against the current database state with thinking mode.
Replicates the exact retrieval + assembly logic from run_mature_experiment.py.
"""

import sys, os, time, json, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
from src.api.db import SessionLocal
from src.memory.models import MemorySlot, EpisodicMemory
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.prompt_assembler import assemble_prompt

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

def main():
    print("Loading classifier…")
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
        print(f"  PROBE: {label}")
        print(f"  Question: {question}")
        print(f"{'='*70}")

        db = SessionLocal()

        turn_count = db.query(EpisodicMemory).filter_by(conversation_id=cid).count()
        total_tokens = 0
        for t in db.query(EpisodicMemory).filter_by(conversation_id=cid).all():
            total_tokens += int(len(t.raw_text.split()) * 1.33)

        classification = classifier.classify(question, conversation_id=cid)
        emb = embedder.encode(question, convert_to_tensor=False).tolist()

        print(f"  Classification: {classification.topic_tags} | {classification.intent_tags}")
        print(f"  Turn count: {turn_count}, Total tokens: ~{total_tokens}")

        orchestrator = HybridRetrievalOrchestrator(db, embedder)
        orchestrator.set_budget_from_turn_count(turn_count, total_tokens, classification=classification)

        scope = {"conversation_id": cid}
        cluster_ids = orchestrator._relevant_cluster_ids(emb, classification=classification, conversation_id=cid)
        if cluster_ids:
            scope["cluster_ids"] = cluster_ids
            print(f"  Clusters: {cluster_ids}")

        fragments = orchestrator.retrieve(
            classification=classification,
            conversation_id=cid,
            prompt_embedding=emb,
            scope=scope,
        )
        print(f"  Fragments: {len(fragments)} ({sum(f.token_count for f in fragments)} tokens)")

        memory_slots = db.query(MemorySlot).filter_by(is_active=True).all()
        messages = assemble_prompt(
            memory_slots=memory_slots,
            retrieved_fragments=fragments,
            user_message=question,
            db_session=db,
            conversation_id=cid,
            classification=classification,
            max_recent_tokens=getattr(orchestrator, 'recent_token_budget', 4000),
            scope=scope,
        )

        total_input = sum(int(len(m.get("content", "").split()) * 1.33) for m in messages)
        print(f"  Input tokens: ~{total_input}")

        print(f"\n  Generating with {GENERALIST_MODEL} + thinking…")
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=GENERALIST_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=10000,
                timeout=180.0,
                extra_body={"num_ctx": 64000, "think": True},
            )
            raw = resp.choices[0].message.content or ""
            answer = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
            if not answer:
                answer = raw
            print(f"  Answer ({time.time()-start:.1f}s, {len(answer.split())} words):")
            print(f"  {answer}")
            if len(answer) > 600:
                print(f"  … (truncated)")
        except Exception as e:
            print(f"  ERROR: {e}")

        db.close()

if __name__ == "__main__":
    main()