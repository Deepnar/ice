#!/usr/bin/env python3
"""Test the orchestrator's NER method directly."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sentence_transformers import SentenceTransformer
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", truncate_dim=384)
orch = HybridRetrievalOrchestrator.__new__(HybridRetrievalOrchestrator)
orch.embedder = embedder
orch.ner_model = orch._load_ner_model()
orch.ner_tokenizer = orch._load_ner_tokenizer()

tests = [
    "What is the goo blade?",
    "Kael and Lethe are fighting.",
    "The Binary Universe Theory explains everything.",
    "My project ICE uses Celery.",
    "Keal raised the goo blade.",
]
for t in tests:
    print(f"{t}  →  {orch._extract_entities_with_ner(t)}")