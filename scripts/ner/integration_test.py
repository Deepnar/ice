#!/usr/bin/env python3
"""Integration test for micro‑NER inside the Codex retriever."""

import sys
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Ensure the project root is on sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.classifier.ner_model import MicroNER

MODEL_PATH = "models/ner/ner_model.pt"
EMBEDDER_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPTS = [
    ("What is the goo blade?", ["goo blade"]),
    ("Keal is the main character.", ["Keal"]),
    ("The Binary Universe Theory explains everything.", ["Binary Universe Theory"]),
    ("Kael and Lethe are fighting.", ["Kael", "Lethe"]),
    ("When does the Goo Blade appear?", ["Goo Blade"]),
    ("I'm using FastAPI with SQLAlchemy.", ["FastAPI", "SQLAlchemy"]),
    ("Call fetch_data() to get the results.", ["fetch_data()"]),
    ("The class DataLoader handles batching.", ["DataLoader"]),
    ("Import torch and numpy.", ["torch", "numpy"]),
    ("My project ICE uses Celery.", ["ICE", "Celery"]),
    ("In the story, Kael wields the goo blade against the Order.", ["Kael", "goo blade", "Order"]),
    ("The function get_user() returns a User object.", ["get_user()", "User"]),
    ("We deployed the FastAPI app with Redis caching.", ["FastAPI", "Redis"]),
    ("Keal raised the goo blade.", ["Keal", "goo blade"]),
    ("The Binary Univerce Theory is complex.", ["Binary Univerce Theory"]),
    ("Lethe and Kael are friends.", ["Lethe", "Kael"]),
    ("The Infinite Context Engine is amazing.", ["Infinite Context Engine"]),
    ("He used the Infinity Blade to win.", ["Infinity Blade"]),
    ("The Lord of the Rings is a classic.", ["Lord of the Rings"]),
    ("Hello, how are you?", []),
    ("This is a generic sentence.", []),
    ("I feel happy today.", []),
    ("Set DEBUG_MODE to True.", ["DEBUG_MODE"]),
    ("The API_KEY is stored in .env.", ["API_KEY"]),
    ("Use config.yaml for settings.", ["config.yaml"]),
    ("The second protagonist sacrificed himself.", ["second protagonist"]),
    ("The final antagonist was revealed.", ["final antagonist"]),
    ("The mysterious figure appeared.", ["mysterious figure"]),
    ("Version 2.0 was released.", ["2.0"]),
    ("The Googolplex is a huge number.", ["Googolplex"]),
    ("so the plot twist is that the protagonist of the second series is the decendent of the protagonist of the first series.", ["second series", "first series"]),
    ("it has 2 sagas. 1st set up a protagonist lost in time, he meet new people and friends and realises his connection to an entity which creates life and universes.", ["entity", "life", "universes"]),
]

def test_ner_integration():
    passed = 0
    total = len(TEST_PROMPTS)

    embedder = SentenceTransformer(EMBEDDER_NAME, device=DEVICE, truncate_dim=384)

    # Load the correct model architecture
    model = MicroNER()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDER_NAME)

    for prompt, expected in TEST_PROMPTS:
        formatted_prompt = f"User: {prompt}"
        encoding = tokenizer(formatted_prompt, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        token_strs = tokenizer.convert_ids_to_tokens(token_ids)
        offsets = encoding["offset_mapping"]

        embeddings = embedder.encode(token_strs, convert_to_tensor=True).float().to(DEVICE)
        with torch.no_grad():
            logits = model(embeddings.unsqueeze(0))
            preds = torch.argmax(logits, dim=-1).squeeze(0)

        # Reconstruct entities from BIO tags
        entities = []
        start_idx = None
        for i, p in enumerate(preds.tolist()):
            if p == 0:  # B-ENT
                if start_idx is not None:
                    entities.append(formatted_prompt[start_idx:end_idx])
                start_idx = offsets[i][0]
                end_idx = offsets[i][1]
            elif p == 1 and start_idx is not None:  # I-ENT
                end_idx = offsets[i][1]
            else:
                if start_idx is not None:
                    entities.append(formatted_prompt[start_idx:end_idx])
                    start_idx = None
        if start_idx is not None:
            entities.append(formatted_prompt[start_idx:end_idx])

        user_prefix_len = len("User: ")
        entities = [e for e in entities if formatted_prompt.find(e) >= user_prefix_len]
        entities = [e.strip() for e in entities if e.strip()]

        result_norm = sorted(entities)
        expected_norm = sorted(expected)
        if result_norm == expected_norm:
            print(f"✅ {prompt}  →  {entities}")
            passed += 1
        else:
            print(f"❌ {prompt}  expected: {expected}  got: {entities}")

    print(f"\n{passed}/{total} tests passed.")

if __name__ == "__main__":
    test_ner_integration()