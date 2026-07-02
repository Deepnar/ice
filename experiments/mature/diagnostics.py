#!/usr/bin/env python3
"""ICE-Mature System Diagnostics — run before the experiment to catch retrieval failures."""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.memory.models import EpisodicMemory, MemorySlot
from src.api.prompt_assembler import assemble_prompt
from openai import OpenAI

OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "gemma4:26b-a4b-it-q4_K_M"
CID = "633e26f8-5889-5c21-8c70-f4d7ab22cb00"  # Shinchan — change to test others

# Probes with expected keywords that MUST appear in retrieved fragments
PROBES = [
    ("did shinchan actually win the claw machine game?", ["claw", "samurai", "keychain"]),
    ("wait, what was that thing about the grandfathers?", ["yoshiji", "ginnosuke", "grandfather"]),
    ("what happened when shinchan and ai-chan were alone at the end of the day?", ["toothpaste", "blush", "whisper"]),
    ("so who's gonna be the treasurer and the secretary?", ["treasurer", "secretary"]),
]

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v3_qwen_ft3.pt",
    schema_path="data/labeled/label_schema.json",
)
embedder = classifier.embedder
db = SessionLocal()
orchestrator = HybridRetrievalOrchestrator(db, embedder)
client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")
memory_slots = db.query(MemorySlot).filter_by(is_active=True).all()

# Get total turns and tokens
total_turns = db.query(EpisodicMemory).filter_by(conversation_id=CID).count()
total_tokens = 0
for turn in db.query(EpisodicMemory).filter_by(conversation_id=CID).all():
    total_tokens += int(len(turn.raw_text.split()) * 1.33)
print(f"Database state: {total_turns} turns, ~{total_tokens} tokens\n")

passed = 0
failed = 0

for probe, keywords in PROBES:
    print(f"{'─'*70}")
    print(f"PROBE: {probe}")
    classification = classifier.classify(probe, conversation_id=CID)
    emb = embedder.encode(probe, convert_to_tensor=False).tolist()
    orchestrator.set_budget_from_turn_count(total_turns, total_tokens)

    # ── 1. BM25 leg ──
    bm25 = orchestrator._bm25_episodic(classification, None, CID, probe)
    bm25_hit = any(any(kw in f.text.lower() for kw in keywords) for f in bm25)
    print(f"  BM25  : {'✅ HIT' if bm25_hit else '❌ MISS'}  (top-3 scores: {[round(f.score,3) for f in bm25[:3]]})")

    # ── 2. Vector leg ──
    vec = orchestrator._vector_episodic(emb, classification, None, CID)
    vec_hit = any(any(kw in f.text.lower() for kw in keywords) for f in vec)
    print(f"  Vector: {'✅ HIT' if vec_hit else '❌ MISS'}  (top-3 scores: {[round(f.score,3) for f in vec[:3]]})")

    # ── 3. Fused (full retrieval) ──
    fused = orchestrator.retrieve(
        classification=classification,
        conversation_id=CID,
        prompt_embedding=emb,
        scope={"conversation_id": CID},
    )
    fused_hit = any(any(kw in f.text.lower() for kw in keywords) for f in fused)
    print(f"  Fused : {'✅ HIT' if fused_hit else '❌ MISS'}  ({len(fused)} fragments, {sum(f.token_count for f in fused)} tokens)")

    # ── 4. Sliding window ──
    from src.api.prompt_assembler import get_recent_turns
    recent = get_recent_turns(db, CID, n=20)
    recent_hit = any(any(kw in t.lower() for kw in keywords) for t in recent)
    print(f"  Window: {'✅ HIT' if recent_hit else '❌ MISS'}  ({len(recent)} turns)")

    # ── 5. Answer ──
    messages = assemble_prompt(memory_slots, fused, probe, db_session=db, conversation_id=CID, classification=classification)
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.0, max_tokens=4096, timeout=120.0)
    answer = resp.choices[0].message.content
    answer_hit = any(kw in answer.lower() for kw in keywords)
    print(f"  Answer: {'✅ CORRECT' if answer_hit else '❌ WRONG'}  ({len(answer.split())} words)")
    if not answer_hit:
        print(f"    → {answer[:200]}...")

    if answer_hit:
        passed += 1
    else:
        failed += 1
        print(f"  ⚠️  FAILED: fragments present but answer was WRONG — model quality issue")
    print()

print(f"{'='*70}")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(PROBES)} probes")
if failed > 0:
    print("⚠️  System has retrieval gaps — fix them before running the experiment.")
else:
    print("✅ All probes pass — system is ready for the experiment.")
db.close()