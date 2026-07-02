#!/usr/bin/env python3
"""Quick Codex diagnostic – run while the experiment is running."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sentence_transformers import SentenceTransformer
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.db import SessionLocal
from src.memory.models import CodexEntity, CodexEdge
from src.classifier.classifier import PyTorchClassifier
from src.classifier.schemas import ClassificationResult

embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", truncate_dim=384)
db = SessionLocal()

# 1. Check Codex size
entities = db.query(CodexEntity).count()
edges = db.query(CodexEdge).filter(CodexEdge.valid_until == None).count()
print(f"Codex entities: {entities}, active edges: {edges}")

# 2. Show a few entities with properties
print("\nSample entities:")
for e in db.query(CodexEntity).filter(CodexEntity.properties != None).limit(10).all():
    print(f"  {e.canonical_name}: {e.properties}")

# 3. Test NER + Codex on a probe
classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v3_qwen_ft3.pt",
    schema_path="data/labeled/label_schema.json",
)
orchestrator = HybridRetrievalOrchestrator.__new__(HybridRetrievalOrchestrator)
orchestrator.embedder = embedder
orchestrator.db = db
orchestrator.ner_model = orchestrator._load_ner_model()
orchestrator.ner_tokenizer = orchestrator._load_ner_tokenizer()

probes = [
    "who is the rival of shinchan from kendo?",
    "why does ai return back after all this year?",
    "who all went for college abroad?",
]
for prompt in probes:
    print(f"\n--- Probe: {prompt}")
    entities = orchestrator._extract_entities_with_ner(prompt)
    print(f"  NER extracted: {entities}")
    if entities:
        matched = orchestrator._match_entities_by_similarity(entities)
        print(f"  Matched: {[e.canonical_name for e in matched]}")
    else:
        print("  No entities found by NER")

    # Run full Codex graph
    classification = ClassificationResult(
        topic_tags=["Creative_&_Media"],
        intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory",
        raw_probs=[0.9]*25,
        max_confidence=0.9,
        prompt=prompt,
    )
    fragments = orchestrator._codex_graph(classification)
    print(f"  Codex fragments: {len(fragments)}")
    if fragments:
        print(f"    {fragments[0].text[:200]}...")

db.close()