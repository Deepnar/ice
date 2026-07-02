#!/usr/bin/env python3
"""Test background-model JSON extraction with thinking DISABLED."""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name

client = get_bg_client()
MODEL = get_bg_model_name()

def call(prompt, max_tokens=150):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=30.0,
    )
    elapsed = time.time() - t0
    raw = resp.choices[0].message.content.strip()
    # Double‑check no thinking block leaked through
    if raw.startswith("<think>"):
        print(f"⚠️  Thinking block still present! Stripping it…")
        raw = raw.split("</think>")[-1].strip()
    return raw, elapsed

# ── Clustering test ──
raw, elapsed = call(
    "[Creative_&_Media] Kael discovers the goo blade…\n"
    "[Creative_&_Media] Orien arrives at the Binary Universe…\n"
    "[Software_&_Tech] PostgreSQL with pgvector handles vector search…\n"
    "Suggest 1‑3 cluster names. Output ONLY a valid JSON array of strings."
)
print(f"CLUSTERING ({elapsed:.1f}s):")
try:
    print(f"  ✅ {json.loads(raw)}")
except Exception as e:
    print(f"  ❌ Not valid JSON: {e} | RAW: {raw[:100]}")

# ── Codex extraction test ──
raw, elapsed = call(
    "Extract subject‑relation‑object triplets as a JSON array. "
    "Output ONLY the JSON.\n\n"
    "Text:\nICE uses PostgreSQL and Redis. Kael is a fire mage from the northern kingdom.\n"
)
print(f"CODEX ({elapsed:.1f}s):")
try:
    parsed = json.loads(raw)
    if isinstance(parsed, list) and all("subject" in t for t in parsed):
        print(f"  ✅ {len(parsed)} triplets")
    else:
        print(f"  ⚠️  Not a triplet list: {parsed}")
except Exception as e:
    print(f"  ❌ Not valid JSON: {e} | RAW: {raw[:100]}")

# ── Reflection test ──
raw, elapsed = call(
    "Generate a structured session summary as JSON with keys: topics_covered (list), "
    "decisions_made (string), unresolved_items (string). Output ONLY the JSON.\n\n"
    "User: I think we should use PostgreSQL for the memory store.\n"
    "Assistant: That makes sense — pgvector gives us native vector search.\n",
    max_tokens=300
)
print(f"REFLECTION ({elapsed:.1f}s):")
try:
    parsed = json.loads(raw)
    if "topics_covered" in parsed:
        print(f"  ✅ {parsed}")
    else:
        print(f"  ⚠️  Missing keys: {parsed}")
except Exception as e:
    print(f"  ❌ Not valid JSON: {e} | RAW: {raw[:100]}")