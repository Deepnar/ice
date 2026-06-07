#!/usr/bin/env python3
"""Test triplet extraction directly – prints raw model output."""

import sys
import os
import re, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workers.codex_extractor import bg_client  # use the pre‑configured client

text = """
User: Write a Python function using FastAPI and SQLAlchemy.
Assistant: Here's the function:
```python
from fastapi import FastAPI
from sqlalchemy import create_engine

app = FastAPI()
engine = create_engine('postgresql://...')
This function connects FastAPI to PostgreSQL using SQLAlchemy.
"""
prompt = (
        "Extract text elements as subject-relation-object triplets. "
        "Output exclusively a valid JSON array of objects with keys: \"subject\", \"relation\", \"object\". "
        "Do not include extra explanations or text padding."
    )

completion = bg_client.chat.completions.create(
model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
messages=[
{"role": "system", "content": "You are an isotropic semantic extraction engine."},
{"role": "user", "content": f"Text:\n{text}\n\n{prompt}"}
],
temperature=0.0,
max_tokens=500,
timeout=30.0
)

raw = completion.choices[0].message.content.strip()
print("===== RAW MODEL OUTPUT =====")
print(raw)
print("============================")

decoder = json.JSONDecoder()
try:
    triplets, _ = decoder.raw_decode(raw)
    print(f"\nExtracted triplets: {len(triplets)}")
    for t in triplets:
        print(f"  {t}")
except json.JSONDecodeError:
    print("\nCould not parse JSON from model output.")