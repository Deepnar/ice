#!/usr/bin/env python3
"""
ICE Classifier - Human Prompt Extraction Script (v2)
=====================================================
Extracts human-authored prompts from unstructured chat logs using the
Amnesia Method — feeding raw text to a local LLM with no assumed structure.

Features:
- Checkpointed: saves every chunk, safe to Ctrl+C and resume anytime
- Resume support: skips already-processed chunks on restart
- Larger chunks: 12k chars = ~370 LLM calls instead of 1628
- Normalized deduplication: catches near-duplicates differing only in whitespace
- Robust JSON recovery: handles markdown fences, trailing commas, partial arrays
- Confidence scoring: filters low-confidence extractions
- Modern ollama client (v0.6.x compatible)

Usage:
    python extract_prompts.py --input-file raw_chats.txt --output-file personal_prompts.jsonl

Resume after interruption (just run the same command again):
    python extract_prompts.py --input-file raw_chats.txt --output-file personal_prompts.jsonl
"""

import argparse
import json
import re
import os
import hashlib
import time
from typing import List
from ollama import Client

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_MODEL   = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_HOST    = "http://localhost:11434"

CHUNK_SIZE     = 12000   # chars per chunk — ~3000 tokens, well inside num_ctx
OVERLAP_SIZE   = 1000    # overlap to avoid cutting prompts at boundaries
MIN_PROMPT_LEN = 8       # discard extractions shorter than this
CONFIDENCE_MIN = 0.85    # discard extractions below this confidence — raised from 0.70
                         # the 0.70-0.85 band was full of AI-text false positives
NUM_CTX        = 8192    # context window — keeps entire workload on GPU VRAM
NUM_PREDICT    = 2048    # max output tokens (enough for dense JSON)

PROGRESS_FILE  = "extraction_progress.json"

# ── System Prompt ──────────────────────────────────────────────────────────────
# Engineered specifically for Deepesh's use case:
# - Massive multi-page prompts (architecture specs, code dumps, story briefs)
# - Short one-liners ("how to do on workbench", "what about postgres?")
# - No structural separators in the file whatsoever
# - AI responses packed with emojis, numbered lists, markdown, tutorials

SYSTEM_PROMPT = """
You are a forensic conversation reconstruction engine with a STRICT precision bias.

Your only task: extract text AUTHORED BY THE HUMAN USER from an unstructured chat transcript.

════════════════════════════════════════
PRECISION OVER RECALL — CRITICAL RULE
════════════════════════════════════════

You MUST prefer returning [] over returning wrong entries.

A FALSE POSITIVE (extracting AI text as human) is far worse than a FALSE NEGATIVE
(missing a real human prompt). When in doubt: return [].

Only assign confidence ≥ 0.85 when you are highly certain. If you cannot reach 0.85
confidence that something is human-authored, do NOT include it.

════════════════════════════════════════
CRITICAL: LENGTH IS NOT A SIGNAL
════════════════════════════════════════

Human messages range from:
  - Single words: "ok", "why", "wdym"
  - Short informal questions: "how to do on workbench", "so what happens?"
  - MASSIVE multi-page prompts: architecture specs, source code, log dumps,
    story briefs, project requirements, copied documentation, entire file contents

DO NOT assume human messages are short.
DO NOT split a large human prompt into fragments — preserve the full block.

════════════════════════════════════════
HOW TO IDENTIFY HUMAN TEXT
════════════════════════════════════════

STRONG signals it is HUMAN:
  ✓ Typos, slang, abbreviations ("idk", "pls", "bc", "tbh", "wtf")
  ✓ Informal first-person ("i cant get this", "ok so i tried", "we are making")
  ✓ Abrupt topic shift with no polished lead-in
  ✓ Frustration or impatience ("still not working", "tf is this", "DONT CHANGE ANYTHING")
  ✓ Raw pasted content with no explanation: code, error logs, terminal output
  ✓ Questions that end with "??" or are phrased as "so like how do i..."
  ✓ Incomplete sentences that trail off
  ✓ ALL CAPS commands ("GIVE ME THE ENTIRE THING", "DONT CHANGE ANYTING")

STRONG signals it is AI:
  ✗ Emoji section markers: ✅, 💡, 1️⃣, 2️⃣, 3️⃣, ❌, ⚠️, 🔥, 🧠, 🛠️
  ✗ Structured headers followed by content: "Why it's good", "Steps:", "Example:"
  ✗ Numbered step lists: "1. Do X\n2. Do Y\n3. Do Z"
  ✗ Offers at the end: "If you want, I can also...", "Let me know if...", "Would you like..."
  ✗ Polished grammar with no typos across multiple paragraphs
  ✗ Phrases like "Here's how:", "Here is a breakdown:", "Below is the..."
  ✗ Bullet summaries: "What this version includes:\n✔ ...\n✔ ..."
  ✗ Conversational AI openers: "Good catch —", "Great question —", "Exactly —"
  ✗ Closing offers: "Which would turn this into...", "These make the report look..."

════════════════════════════════════════
HARD RULES — NEVER VIOLATE THESE
════════════════════════════════════════

RULE 1: Never extract text that begins with AI opener phrases.
  Examples of AI openers (DO NOT extract):
  "Good catch —", "Great question —", "Exactly —", "Perfect —", "Alright —",
  "Sure!", "Got it —", "Below is", "Here is", "Here's", "Now we're at",
  "Short answer:", "Let me make this", "I'll give you", "What you wrote is"

RULE 2: Never extract a mid-sentence fragment.
  If the text starts mid-word or mid-sentence (e.g. "ister college\n    Define..."),
  it is a chunk boundary artifact. Return [] for it.

RULE 3: Never extract text that contains ✅, 1️⃣, 2️⃣, 💡 as structural markers.
  These are AI formatting patterns. Even if the surrounding text looks human,
  a block that uses these as section headers was written by the AI.

RULE 4: A chunk that is MOSTLY AI content with one ambiguous line → return [].
  Do not mine for the "least AI-looking" fragment in a chunk of AI output.
  Only extract if you can clearly identify a human speaker boundary.

RULE 5: Never extract the extraction prompt itself.
  If the chunk contains "Analyze this raw chat excerpt" or "CALIBRATION EXAMPLES"
  or "Extract ALL text authored by the human user", those are instructions,
  not human prompts. Return [] for any chunk containing this text.

════════════════════════════════════════
TRICKY CASES
════════════════════════════════════════

CASE 1 — Short human prompt buried in AI content:
  The AI writes a long tutorial. Then suddenly: "how to do on workbench"
  That abrupt unpolished line IS human. Extract it with high confidence.

CASE 2 — Massive human specification:
  A wall of unformatted text describing a project, feature list, or architecture
  with obvious typos and informal language — this is human. Preserve fully.

CASE 3 — AI response that ends with a short fragment:
  "If you want, I can also show you t"
  This is a TRUNCATED AI sentence (chunk boundary cut it). Do NOT extract.

CASE 4 — AI response quoted back by human:
  If human is copy-pasting an AI response to ask about it, the quote itself
  was AI-authored. Extract ONLY the human's surrounding question, not the quote.

════════════════════════════════════════
OUTPUT FORMAT — STRICT
════════════════════════════════════════

Return ONLY valid JSON. No markdown. No explanation. No preamble.

[
  {"text": "the human prompt here", "confidence": 0.95},
  {"text": "another human prompt", "confidence": 0.87}
]

confidence = certainty that this is genuinely human-authored (0.0 to 1.0)
Only include entries where confidence ≥ 0.85.
If zero qualifying human text found: []
"""

# ── User Prompt Template ───────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """
Analyze this raw chat excerpt. Extract ONLY text that was authored by the HUMAN USER.
Prefer precision over recall. When uncertain, return [].

══════════════════════════════════════════════════════
CALIBRATION EXAMPLES
══════════════════════════════════════════════════════

EXAMPLE A — short human prompt buried in AI content:

RAW INPUT:
  Use two tools:
  1️⃣ dbdiagram.io → design schema
  2️⃣ DBeaver → view real database later
  how to do on workbench
  To create an ER (EER) Diagram in MySQL Workbench, follow this simple workflow.

CORRECT OUTPUT:
  [{{"text": "how to do on workbench", "confidence": 0.96}}]

WHY: "how to do on workbench" is abrupt, unformatted, and surrounded by AI emoji-list content.
     The AI content before and after it does NOT get extracted.

──────────────────────────────────────────────────────

EXAMPLE B — massive human specification:

RAW INPUT:
  Here is the full pipeline architecture I want:
  
  1. Prompt → Classifier (11x11x3 PyTorch model)
  2. Classifier → MoE Router
  3. Router → best local model via Ollama
  
  The classifier needs to run in under 50ms.
  Can you help me design the FastAPI middleware layer?
  
  Sure! Here's a FastAPI middleware approach that fits your pipeline:

CORRECT OUTPUT:
  [{{"text": "Here is the full pipeline architecture I want:\\n\\n1. Prompt → Classifier (11x11x3 PyTorch model)\\n2. Classifier → MoE Router\\n3. Router → best local model via Ollama\\n\\nThe classifier needs to run in under 50ms.\\nCan you help me design the FastAPI middleware layer?", "confidence": 0.97}}]

WHY: The human wrote everything up to "Can you help...". The AI response starts at "Sure!".
     The AI's "Sure! Here's a FastAPI..." does NOT get extracted.

──────────────────────────────────────────────────────

EXAMPLE C — pure AI content, nothing to extract:

RAW INPUT:
  Good catch — two fixes are needed:
  1. Remove the date from the title page
  2. Fix the pgfplots overlap issue

  ✅ My recommendation for your ERP project
  Use two tools:
  1️⃣ dbdiagram.io → design schema
  2️⃣ DBeaver → view real database later
  This matches your workflow: Idea → ER Diagram → Database Tables → Backend API
  
  Very important question — and I'm glad you asked it now, not 3 months later.

CORRECT OUTPUT:
  []

WHY: This chunk is entirely AI. "Good catch —", "✅ My recommendation", "Very important question —"
     are all AI openers. There is no human speaker in this chunk. Return [].

══════════════════════════════════════════════════════

NOW PROCESS THIS CHUNK:

{chunk_text}

Return ONLY the JSON array. No other text.
"""

# ── Ollama Client ──────────────────────────────────────────────────────────────

def get_client() -> Client:
    return Client(host=OLLAMA_HOST)


def extract_from_chunk(client: Client, chunk_text: str) -> List[dict]:
    """Call the LLM and return list of {text, confidence} dicts."""
    user_prompt = USER_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    try:
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=user_prompt,
            system=SYSTEM_PROMPT,
            stream=False,
            options={
                "temperature":    0.0,
                "num_ctx":        NUM_CTX,
                "num_predict":    NUM_PREDICT,
                "repeat_penalty": 1.0,
            }
        )
        raw = response["response"].strip()
        return parse_llm_json(raw)

    except Exception as e:
        print(f"    ⚠ Ollama error: {e}")
        return []


# ── JSON Recovery ──────────────────────────────────────────────────────────────

def parse_llm_json(text: str) -> List[dict]:
    """Robustly parse JSON from LLM output, handling common failure modes."""
    text = text.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Fix trailing commas before ] or }
    text = re.sub(r",\s*\]", "]", text)
    text = re.sub(r",\s*\}", "}", text)

    # Attempt 1: direct parse
    try:
        result = json.loads(text)
        return normalize_entries(result)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the array substring
    try:
        start = text.index("[")
        end   = text.rindex("]") + 1
        result = json.loads(text[start:end])
        return normalize_entries(result)
    except (ValueError, json.JSONDecodeError):
        pass

    # Attempt 3: pull out individual JSON objects
    entries = []
    for match in re.finditer(r'\{[^}]+\}', text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if "text" in obj:
                entries.append(obj)
        except json.JSONDecodeError:
            continue

    if entries:
        return normalize_entries(entries)

    print(f"    ⚠ Could not parse JSON from response (len={len(text)})")
    return []


def normalize_entries(raw: list) -> List[dict]:
    """Ensure every entry is {text: str, confidence: float}."""
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append({"text": item, "confidence": 0.80})
        elif isinstance(item, dict) and "text" in item:
            result.append({
                "text":       str(item["text"]),
                "confidence": float(item.get("confidence", 0.80))
            })
    return result


# ── Checkpoint / Resume ────────────────────────────────────────────────────────

def save_progress(chunk_idx: int):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_chunk": chunk_idx}, f)


def load_progress() -> int:
    if not os.path.exists(PROGRESS_FILE):
        return 0
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)["last_chunk"] + 1
    except Exception:
        return 0


def clear_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


# ── Output ─────────────────────────────────────────────────────────────────────

def append_prompts(prompts: List[dict], output_file: str, seen_hashes: set):
    """Append new unique prompts to the JSONL file immediately."""

    # Patterns that definitively identify AI-authored text
    # If an extracted entry matches any of these, it's a false positive
    AI_OPENER_PATTERNS = [
        r"^good catch\s*[—–-]",
        r"^great question\s*[—–-]",
        r"^perfect\s*[—–-]",
        r"^exactly\s*[—–-]",
        r"^alright\s*[—–-]",
        r"^sure[!,.]",
        r"^got it\s*[—–-]",
        r"^below is ",
        r"^here is ",
        r"^here's ",
        r"^now we're at ",
        r"^short answer:",
        r"^very important question",
        r"^i'll give you",
        r"^let me make this",
        r"^what you wrote is",
        r"^you are not ",
        r"^this is a ",
        r"if you want.*i can also",
        r"analyze this raw chat excerpt",    # leaked extraction prompt
        r"calibration examples",             # leaked extraction prompt
        r"extract all text authored by",     # leaked extraction prompt
    ]
    ai_pattern = re.compile(
        "|".join(AI_OPENER_PATTERNS),
        re.IGNORECASE
    )

    # Mid-sentence fragment detector: starts with lowercase mid-word or whitespace
    fragment_pattern = re.compile(r"^\s*[a-z]{1,4}\s+[a-z]")  # e.g. "ister college..."

    written = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for item in prompts:
            text = item["text"].strip()
            confidence = item["confidence"]

            if len(text) < MIN_PROMPT_LEN:
                continue
            if confidence < CONFIDENCE_MIN:
                continue

            # Hard filter: AI opener patterns
            if ai_pattern.search(text[:120]):
                continue

            # Hard filter: mid-sentence fragment (chunk boundary artifact)
            # Only flag if very short lead-in before a lowercase continuation
            first_line = text.split("\n")[0].strip()
            if len(first_line) < 20 and fragment_pattern.match(first_line):
                continue

            # Normalized hash for deduplication
            normalized = re.sub(r"\s+", " ", text.lower())
            h = hashlib.md5(normalized.encode()).hexdigest()
            if h in seen_hashes:
                continue

            seen_hashes.add(h)
            # Collapse internal newlines to single spaces for clean single-line prompts
            text = re.sub(r"\n+", " ", text).strip()
            text = re.sub(r" {2,}", " ", text)  # also collapse any double spaces left behind
            f.write(json.dumps({"prompt": text, "confidence": confidence}, ensure_ascii=False) + "\n")
            written += 1

    return written


def load_existing_hashes(output_file: str) -> set:
    """On resume, rebuild the seen-hash set from what's already saved."""
    seen = set()
    if not os.path.exists(output_file):
        return seen
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("prompt", "").strip()
                normalized = re.sub(r"\s+", " ", text.lower())
                seen.add(hashlib.md5(normalized.encode()).hexdigest())
            except Exception:
                continue
    return seen


# ── Chunking ───────────────────────────────────────────────────────────────────

def make_chunks(content: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunks.append(content[start:end])
        start += chunk_size - overlap
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract human prompts from raw chat logs (v2 — resumable)")
    parser.add_argument("--input-file",   required=True,  help="Path to raw chat text file")
    parser.add_argument("--output-file",  required=True,  help="Path to output JSONL file")
    parser.add_argument("--chunk-size",   type=int, default=CHUNK_SIZE,   help="Chars per chunk")
    parser.add_argument("--overlap-size", type=int, default=OVERLAP_SIZE, help="Overlap between chunks")
    parser.add_argument("--reset",        action="store_true", help="Ignore checkpoint and start fresh")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: input file not found: {args.input_file}")
        return

    # Load content
    with open(args.input_file, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = make_chunks(content, args.chunk_size, args.overlap_size)
    total  = len(chunks)

    print(f"Input:  {args.input_file} ({len(content):,} chars)")
    print(f"Output: {args.output_file}")
    print(f"Model:  {OLLAMA_MODEL}")
    print(f"Chunks: {total} × {args.chunk_size} chars (overlap {args.overlap_size})")
    print()

    # Resume logic
    if args.reset:
        clear_progress()
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
        start_idx = 0
        seen_hashes: set = set()
        print("Reset: starting from scratch.")
    else:
        start_idx   = load_progress()
        seen_hashes = load_existing_hashes(args.output_file)
        if start_idx > 0:
            print(f"Resuming from chunk {start_idx + 1}/{total}  "
                  f"({len(seen_hashes)} prompts already saved)")
        else:
            print("Starting fresh extraction.")

    print()

    client = get_client()
    total_written = len(seen_hashes)

    for i in range(start_idx, total):
        chunk = chunks[i]
        t0    = time.time()

        print(f"[{i+1:>4}/{total}] processing... ", end="", flush=True)

        entries = extract_from_chunk(client, chunk)
        written = append_prompts(entries, args.output_file, seen_hashes)
        total_written += written

        elapsed = time.time() - t0
        print(f"extracted {len(entries):>2} → kept {written:>2}  "
              f"({elapsed:.1f}s)  total saved: {total_written}")

        save_progress(i)

    clear_progress()
    print()
    print("══════════════════════════════════════")
    print(f"Done. Total unique prompts saved: {total_written}")
    print(f"Output: {args.output_file}")
    print("══════════════════════════════════════")


if __name__ == "__main__":
    main()