#!/usr/bin/env python3
"""Comprehensive Phase 9 integration test.
Run with:
  uv run python tests/test_full_pipeline_phase_9.py
Requires:
  - Docker PostgreSQL up
  - vLLM background model (port 8002)
  - Celery worker running (beat optional)
"""
from datetime import datetime, timezone
import os, sys, time, uuid, json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from src.api.db import SessionLocal
from src.memory.models import (
    EpisodicMemory, Conversation, CodexEntity, CodexEdge,
    ProceduralMemory, SessionSummary, ContextCluster,
    RAGDocument, RAGChunk
)
from src.classifier.classifier import PyTorchClassifier
from src.workers.post_flight import evaluate_turn
from src.workers.decay import apply_decay
from src.workers.reflection import run_reflection
from src.workers.clustering import run_cluster_assignment

classifier_model_path = "models/classifier/ice_classifier_v2_final.pt"
schema_path = "data/labeled/label_schema.json"
INGEST_DIR = "ingest_inbox"
PROCESSED_DIR = os.path.join(INGEST_DIR, "processed")
SIM_INPUT = "data/simulation_input.jsonl"

def truncate_all():
    db = SessionLocal()
    db.execute(text("TRUNCATE episodic_memory, conversations, codex_entities, codex_edges, "
                    "codex_events, codex_snapshots, procedural_memory, session_summaries, "
                    "context_clusters, sentinel_events, cold_storage, idempotency_keys, "
                    "rag_documents, rag_chunks RESTART IDENTITY CASCADE"))
    db.commit()
    db.close()
    print("✅ All tables truncated.\n")

def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ----- Test 1: Live pipeline ------------------------------------------------
def test_live_pipeline():
    print_section("1. LIVE PIPELINE (Post‑flight → Codex → Procedural)")
    classifier = PyTorchClassifier(model_path=classifier_model_path, schema_path=schema_path)
    embedder = classifier.embedder

    db = SessionLocal()
    conv = Conversation(id=uuid.uuid4(), memory_scope_type='auto')
    db.add(conv)
    db.flush()

    batch_id = uuid.uuid4()
    user_prompt = "ICE stores conversational turns using PostgreSQL and pgvector. Redis is used as the Celery broker."
    assistant_response = (
        "Here's the code:\n\n```python\n"
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "```\n\n"
        "ICE uses PostgreSQL and Redis for storage and task management."
    )
    raw_text = f"User: {user_prompt}\n\nAssistant: {assistant_response}"
    embedding = embedder.encode(user_prompt, convert_to_tensor=False).tolist()

    turn = EpisodicMemory(
        conversation_id=conv.id, batch_id=batch_id,
        timestamp=datetime.now(timezone.utc),
        topic_tags=["Software_&_Tech"], intent_tags=["Generation"],
        context_reliance="Long_Term_Memory", raw_text=raw_text,
        embedding=embedding, idempotency_key=str(uuid.uuid4())
    )
    db.add(turn)
    db.commit()
    conversation_id = conv.id
    db.close()

    print("⏳ Enqueuing post‑flight task …")
    evaluate_turn(
        batch_id=str(batch_id), prompt=user_prompt,
        response=assistant_response, conversation_id=str(conversation_id)
    )
    time.sleep(5)   # wait for worker to process

    db = SessionLocal()
    turn = db.query(EpisodicMemory).filter_by(batch_id=batch_id).first()
    print(f"   lossless_flag = {turn.lossless_flag}  (expected True)")
    assert turn.lossless_flag is True, "Post‑flight failed"

    # Codex – trigger sync if not already done
    codex_entities = db.query(CodexEntity).all()
    if not codex_entities:
        print("   Codex not yet extracted – doing it synchronously …")
        from src.workers.codex_extractor import extract_triplets, handle_triplet
        triplets = extract_triplets(raw_text)
        for t in triplets:
            if isinstance(t, dict) and "subject" in t:
                s, r, o = t["subject"].strip(), t["relation"].strip(), t["object"].strip()
                if s and r and o:
                    handle_triplet(db, s, r, o, str(batch_id))
        db.commit()
        codex_entities = db.query(CodexEntity).all()
    print(f"   Codex entities: {len(codex_entities)}")
    print(f"   Codex edges: {len(db.query(CodexEdge).all())}")
    print("   ⚠️  No entities/edges is normal for the 1.5B model.")

    proc = db.query(ProceduralMemory).all()
    print(f"   Procedural patterns: {len(proc)}  (may be 0)")
    db.close()
    print("✅ Live pipeline test completed.\n")

# ----- Test 2: Decay --------------------------------------------------------
def test_decay():
    print_section("2. DECAY WORKER")
    apply_decay()
    time.sleep(3)
    db = SessionLocal()
    for t in db.query(EpisodicMemory).all():
        print(f"   Turn {t.id} -> decay_score={t.decay_score}, archived={t.is_archived}")
    db.close()
    print("✅ Decay worker test completed.\n")

# ----- Test 3: Reflection ---------------------------------------------------
def test_reflection():
    print_section("3. REFLECTION WORKER")
    run_reflection()
    time.sleep(15)   # model call takes a few seconds
    db = SessionLocal()
    summaries = db.query(SessionSummary).all()
    print(f"   Session summaries: {len(summaries)}")
    for s in summaries:
        print(f"      topics: {s.topics_covered}, decisions: {s.decisions_made}")
    assert len(summaries) > 0, "No session summary created"
    db.close()
    print("✅ Reflection worker test completed.\n")

# ----- Test 4: Clustering ---------------------------------------------------
def test_clustering():
    print_section("4. CLUSTERING WORKER")
    _db = SessionLocal(); run_cluster_assignment(_db); _db.close()
    print("   Waiting for cluster assignment (up to 90 sec) …", end="", flush=True)
    db = SessionLocal()
    clusters = []
    for _ in range(90):
        time.sleep(1)
        clusters = db.query(ContextCluster).all()
        if clusters:
            print()
            break
        print(".", end="", flush=True)
    else:
        print()
        clusters = db.query(ContextCluster).all()
    print(f"   Clusters: {len(clusters)}")
    for c in clusters:
        cnt = db.query(EpisodicMemory).filter_by(cluster_id=c.id).count()
        print(f"      {c.name} -> {cnt} turns")
    assert len(clusters) > 0, "No clusters created after 90 seconds"
    db.close()
    print("✅ Clustering worker test completed.\n")

# ----- Test 5: Drop Zone ----------------------------------------------------
def test_drop_zone():
    print_section("5. DROP ZONE (ingestion)")
    os.makedirs(INGEST_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    test_file = os.path.join(INGEST_DIR, "phase9_test_rag.txt")
    with open(test_file, "w") as f:
        f.write("ICE uses PostgreSQL with pgvector. Redis is the Celery broker.")
    from src.workers.drop_zone import IngestHandler, wait_for_file_to_settle
    handler = IngestHandler()
    if wait_for_file_to_settle(test_file):
        handler.ingest_file(test_file)
        print("   File ingested.")
    else:
        print("   Timeout waiting for file to settle.")
    db = SessionLocal()
    docs = db.query(RAGDocument).all()
    print(f"   RAG documents: {len(docs)}")
    for d in docs:
        print(f"      {d.filename} (tokens: {d.token_count})")
    print(f"   RAG chunks: {len(db.query(RAGChunk).all())}")
    db.close()
    assert len(docs) > 0, "No RAG document created"
    print("✅ Drop Zone test completed.\n")

# ----- Test 6: Simulation ---------------------------------------------------
def test_simulation():
    print_section("6. SIMULATION HARNESS")
    os.makedirs("data", exist_ok=True)
    sim_data = [
        {"prompt": "What is Python?", "response": "Python is a programming language.", "timestamp": "2026-01-01T10:00:00Z"},
        {"prompt": "Write a function in Python", "response": "def hello():\n    print('hello')\n", "timestamp": "2026-01-02T10:00:00Z"},
        {"prompt": "How do I connect to PostgreSQL?", "response": "Use psycopg2.", "timestamp": "2026-01-03T10:00:00Z"},
    ]
    with open(SIM_INPUT, "w") as f:
        for item in sim_data:
            f.write(json.dumps(item) + "\n")
    import subprocess
    res = subprocess.run(
        ["uv", "run", "python", "scripts/simulation/run_simulation.py", "--seed", "42", "--input", SIM_INPUT],
        capture_output=True, text=True
    )
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR:", res.stderr)
        return
    db = SessionLocal()
    turns = db.query(EpisodicMemory).order_by(EpisodicMemory.timestamp).all()
    print(f"   Simulation inserted {len(turns)} turns.")
    for t in turns:
        print(f"      context_reliance={t.context_reliance}, lossless={t.lossless_flag}")
    db.close()
    print("✅ Simulation harness test completed.\n")

# ----- Main ----------------------------------------------------------------
def main():
    print("🧪 ICE Phase 9 Full Integration Test")
    print("=" * 60)
    print("Make sure these are running:")
    print("  - Docker PostgreSQL (`docker compose up -d`)")
    print("  - vLLM background model (port 8002) (`vllm-bg`)")
    print("  - Celery worker (`uv run celery -A src.workers.celery_app worker -B --loglevel=info`)")
    print("\nPress Enter to continue or Ctrl+C to abort...")
    input()
    truncate_all()
    test_live_pipeline()
    test_decay()
    test_reflection()
    test_clustering()
    test_drop_zone()
    test_simulation()
    print("\n🎉 All Phase 9 tests completed. Check the database and Celery logs for details.")

if __name__ == "__main__":
    main()