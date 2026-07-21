#!/usr/bin/env python3
"""Quick test: can Qwen3-14B-AWQ produce structured JSON for background tasks?"""

import json
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")
MODEL = "Qwen/Qwen3-14B-AWQ"

def call(prompt, max_tokens=150, temperature=0.0):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30.0,
    )
    elapsed = time.time() - t0
    raw = resp.choices[0].message.content.strip()
    return raw, elapsed

def test_clustering():
    """Simulates cluster_turns – expects JSON array of strings."""
    prompt = (
        "The following are topic tags and brief snippets from several conversation turns. "
        "Suggest 1-3 cluster names that group related topics. "
        "Output ONLY a valid JSON array of strings, e.g. [\"Database Systems\", \"Creative Writing\"]. "
        "Each string must be a short, descriptive phrase. "
        "Do NOT output anything else.\n\n"
        "[Creative_&_Media] The story begins with Kael discovering a hidden power within the goo blade\n"
        "[Creative_&_Media] Orien arrives at the Binary Universe and meets the Council of Elders\n"
        "[Creative_&_Media] The goo blade has been passed down through generations of the royal family\n"
        "[Software_&_Tech] The orchestrator uses RRF fusion to combine multiple retrieval legs\n"
        "[Software_&_Tech] PostgreSQL with pgvector handles vector similarity search for episodic memory\n"
    )
    raw, elapsed = call(prompt)
    print(f"\n=== CLUSTERING (elapsed {elapsed:.1f}s) ===")
    print(f"RAW: {raw[:200]}")
    try:
        parsed = json.loads(raw)
        print(f"✅ Valid JSON: {parsed}")
    except Exception as e:
        print(f"❌ Not valid JSON: {e}")

def test_codex_extraction():
    """Simulates codex extraction – expects JSON array of triplets."""
    prompt = (
        "Extract text elements as subject-relation-object triplets. "
        "Output exclusively a valid JSON array of objects with keys: \"subject\", \"relation\", \"object\". "
        "Do not include extra explanations or text padding.\n\n"
        "Text:\n"
        "ICE uses PostgreSQL for memory and Redis for tasks. The system depends on Celery "
        "and is part of the Infinite Context Engine project. "
        "My character Kael is a fire mage from the northern kingdom.\n"
    )
    raw, elapsed = call(prompt, max_tokens=200)
    print(f"\n=== CODEX EXTRACTION (elapsed {elapsed:.1f}s) ===")
    print(f"RAW: {raw[:200]}")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(
            isinstance(t, dict) and "subject" in t and "relation" in t and "object" in t
            for t in parsed
        ):
            print(f"✅ Valid triplet list ({len(parsed)} triplets): {parsed}")
        else:
            print(f"⚠️  JSON is valid but not a triplet list: {parsed}")
    except Exception as e:
        print(f"❌ Not valid JSON: {e}")

def test_reflection():
    """Simulates session synthesis – expects JSON object with specific keys."""
    prompt = (
        "Generate a structured session summary from the following conversation turns.\n"
        "Output ONLY a valid JSON object with these keys:\n"
        "  - \"topics_covered\": a list of strings\n"
        "  - \"decisions_made\": a string\n"
        "  - \"unresolved_items\": a string\n"
        "If a field has no content, use an empty list [] or empty string \"\".\n"
        "Do NOT include markdown or additional text.\n\n"
        "User: I think we should use PostgreSQL for the memory store.\n"
        "Assistant: That makes sense – pgvector gives us native vector search.\n"
        "User: What about Redis?\n"
        "Assistant: Redis works well as a Celery broker.\n"
    )
    raw, elapsed = call(prompt, max_tokens=300)
    print(f"\n=== REFLECTION (elapsed {elapsed:.1f}s) ===")
    print(f"RAW: {raw[:200]}")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "topics_covered" in parsed:
            print(f"✅ Valid summary JSON: {parsed}")
        else:
            print(f"⚠️  JSON valid but missing expected keys: {parsed}")
    except Exception as e:
        print(f"❌ Not valid JSON: {e}")

if __name__ == "__main__":
    print(f"Testing model {MODEL} on http://localhost:8001/v1\n")
    test_clustering()
    test_codex_extraction()
    test_reflection()