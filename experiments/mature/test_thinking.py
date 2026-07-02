#!/usr/bin/env python3
"""
Test thinking mode in the Ollama OpenAI-compatible endpoint.

This script replicates the exact call structure used in
run_mature_experiment.py but with verbose logging to show:
  - Raw response before any stripping
  - Whether <think> tags are present
  - What the final cleaned answer looks like
  - Which model is used and whether the extra_body options are passed correctly.

Usage:
  uv run python experiments/mature/test_thinking.py
"""

import json
import re
from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIG – match your run_mature_experiment.py settings
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/v1"
MODEL_NAME = "gemma4:26b-a4b-it-q4_K_M"   # change to whichever model you're testing
API_KEY = "dummy"

# A prompt that should trigger a thoughtful answer
PROMPT = "Explain the difference between a list and a tuple in Python, and when to use each."

# ---------------------------------------------------------------------------
# CLIENT – same as in run_mature_experiment.py
# ---------------------------------------------------------------------------
client = OpenAI(base_url=OLLAMA_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# TEST 1: Direct call with extra_body options
# ---------------------------------------------------------------------------
print("=" * 70)
print("TEST 1: OpenAI-compatible call with extra_body={'options': {...}}")
print("=" * 70)

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_tokens=2048,
        extra_body={
            "options": {
                "num_ctx": 64000,
                "think": True,
            }
        }
    )
    raw_content = response.choices[0].message.content or ""

    print(f"\n✅ Response received (length: {len(raw_content)} chars)")
    print("\n--- RAW CONTENT (first 1000 chars) ---")
    print(raw_content[:1000] + ("..." if len(raw_content) > 1000 else ""))
    print("----------------------------------------\n")

    # Check for thinking tags
    has_think = "<think>" in raw_content
    has_gemma_think = "<|channel|>" in raw_content or "<|think|>" in raw_content

    if has_think or has_gemma_think:
        print("✅ THINKING DETECTED in raw output!")
        if has_think:
            # Extract thinking block
            match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
            if match:
                print("\n--- EXTRACTED THINKING BLOCK ---")
                print(match.group(1).strip()[:500] + "...")
                print("----------------------------------")
        if has_gemma_think:
            match = re.search(r"<\|channel>thought\n(.*?)(?:<channel\|>|$)", raw_content, flags=re.DOTALL)
            if match:
                print("\n--- EXTRACTED GEMMA THINKING BLOCK ---")
                print(match.group(1).strip()[:500] + "...")
                print("----------------------------------------")
    else:
        print("❌ No thinking tags found in raw output.")

    # Simulate the stripping logic used in run_mature_experiment.py
    clean = raw_content
    for pattern in [
        r"<think>.*?</think>\s*",
        r"<\|channel>.*?<channel\|>\s*",   # corrected
        r"<\|think\|>.*?\n",
    ]:
        clean = re.sub(pattern, "", clean, flags=re.DOTALL).strip()

    print("\n--- CLEANED ANSWER (after removing thinking) ---")
    print(clean[:500] + ("..." if len(clean) > 500 else ""))
    print("------------------------------------------------\n")

except Exception as e:
    print(f"❌ Error: {e}")

# ---------------------------------------------------------------------------
# TEST 2: Native Ollama client (if installed) – optional comparison
# ---------------------------------------------------------------------------
try:
    import ollama
    print("=" * 70)
    print("TEST 2: Native Ollama client (for comparison)")
    print("=" * 70)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT}],
        options={
            "temperature": 0.7,
            "num_ctx": 64000,
            "think": True,
        }
    )
    raw = response.get("message", {}).get("content", "")
    thinking = response.get("message", {}).get("thinking", "")

    print(f"\n✅ Response received (length: {len(raw)} chars)")
    if thinking:
        print("\n--- NATIVE THINKING FIELD ---")
        print(thinking[:500] + ("..." if len(thinking) > 500 else ""))
        print("----------------------------")
    else:
        print("❌ No thinking field in native response.")
    print("\n--- NATIVE RAW CONTENT (first 500 chars) ---")
    print(raw[:500] + ("..." if len(raw) > 500 else ""))
    print("--------------------------------------------")

except ImportError:
    print("\n⚠️  Ollama native SDK not installed – skipping Test 2.")
except Exception as e:
    print(f"❌ Native client error: {e}")

# ---------------------------------------------------------------------------
# TEST 3: Simulate the full pipeline's call structure (messages assembly)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 3: Full pipeline simulation (with system message)")
print("=" * 70)

# Replicate the messages structure from assemble_prompt (simplified)
system_msg = {
    "role": "system",
    "content": (
        "<|think|> You have access to the user's conversation history below. "
        "Think step-by-step through the ALL the fact and relevant facts before answering. "
        "When facts have changed over time ... "
    )
}
user_msg = {"role": "user", "content": PROMPT}
messages = [system_msg, user_msg]

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        extra_body={
            "options": {
                "num_ctx": 64000,
                "think": True,
            }
        }
    )
    raw_content = response.choices[0].message.content or ""
    print(f"\n✅ Response received (length: {len(raw_content)} chars)")

    # Check for thinking tags again
    if "<think>" in raw_content:
        print("✅ THINKING TAGS PRESENT in pipeline call.")
    else:
        print("❌ No thinking tags in pipeline call.")

    print("\n--- RAW PIPELINE OUTPUT (first 800 chars) ---")
    print(raw_content[:800] + ("..." if len(raw_content) > 800 else ""))
    print("---------------------------------------------")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("DEBUG SUMMARY")
print("=" * 70)
print("1. Check if raw output contains <think> tags.")
print("2. If not, verify that your model actually supports the 'think' option.")
print("   - For Qwen models, it should work with extra_body={'options': {'think': True}}.")
print("   - For Gemma models, they do not produce <think> tags; they have a different mechanism.")
print("3. If you see thinking tags but they are stripped, ensure your cleaning regex is correct.")
print("4. If using the OpenAI wrapper, the extra_body must be nested exactly as shown.")
print("   - Incorrect: extra_body={'think': True}   (this will be ignored)")
print("   - Correct:   extra_body={'options': {'think': True}}")
print("=" * 70)