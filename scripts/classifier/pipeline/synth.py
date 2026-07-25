#!/usr/bin/env python3
"""Stage 3 — generate synthetic prompts for MEASURED label gaps.

Rewrites ``legacy/promt_labeling/synthetic_data.py``, keeping its best property:
prompts are generated in a **casual, human voice**, because a corpus of tidy
"Please explain the difference between X and Y" sentences teaches a classifier to
recognise a register real users don't write in.

Two deliberate differences from v1:

1. **Gaps are measured, not assumed.** v1 generated from a fixed grid
   (``synth_promt_gen_number.csv``). This stage reads the actual per-label counts
   from the labeled corpus and only generates for labels under the floor. Before
   any labeling has run it falls back to the schema's known-empty labels — the
   four that start at zero positives (the coding intents, High_Complexity, and
   Temporal_Recall's non-detectable half).

2. **Synthetic rows carry NO labels.** v1 stamped the intended label onto the
   generated row. That is fabrication: the model was asked for a Code_Change
   prompt, so the row is *assumed* to be one. Here the intended label is recorded
   as ``meta.target_label`` — a generation hint, provenance only — and the row
   goes through the same two-labeler pass as everything else. If the labelers
   disagree with the intent it was generated for, that is a bad generation and it
   should be labeled as what it actually is.

    Rows for the four labels seeded here are the ones most at risk of the §4
    "<150 positives ⇒ drop the label" rule, so it matters that their positives are
    real rather than self-certified.

Usage:
    uv run python scripts/classifier/pipeline/synth.py --per-label 300
"""

import argparse
import asyncio
import json
import random
from collections import Counter

from common import (CORPUS_SYNTH, LABELS_FINAL, JsonlAppender, completed_ids,
                    ensure_data_dir, is_usable_prompt, read_jsonl, stable_id)
from serving import LocalServer, is_up, served_model

from src.classifier.schema import (CONTEXT_RELIANCE, HIGH_COMPLEXITY, INTENT,
                                   TEMPORAL_RECALL, load_schema)

SEED = 42
FLOOR = 300
BATCH = 10          # prompts per generation call

GENERATION_SYSTEM_PROMPT = """You are a synthetic training-data generator for a \
personal AI memory classifier. You produce realistic, human-written user prompts \
— the kind a real person actually types into a chat box.

VOICE RULES (these matter more than correctness):
- Casual and conversational. Write like someone texting, not writing an essay.
- Typos, lowercase starts, missing punctuation, and trailing "?" are all realistic.
- Vary length hard: some prompts are 4 words, some are three sentences.
- No "Please explain the difference between X and Y" textbook phrasing.
- No numbering, no preamble, no quotes around prompts.
- Never mention that these are examples or training data.

Return ONLY a JSON array of strings. No other text."""

# Per-label generation briefs. Each says what the label MEANS in behavioural terms
# and, critically, what its near-miss looks like — the confusable neighbour is what
# makes a generated set useful rather than a pile of obvious positives.
BRIEFS = {
    "Codebase_Query": (
        "Someone asking to UNDERSTAND code that already exists in their project: "
        "where something lives, how a piece works, what calls what, why a file is "
        "structured a certain way. They want knowledge, not changes. Mix in vague "
        "ones ('where does the retry thing happen again') and precise ones."),
    "Code_Change": (
        "Someone asking for code in their existing project to be written or changed: "
        "add a feature, refactor, migrate, wire something up, delete dead code. The "
        "codebase should be DIFFERENT afterwards. Not standalone snippets."),
    HIGH_COMPLEXITY: (
        "Requests where the strongest available model would genuinely answer better: "
        "multi-step reasoning, cross-domain synthesis, designing something with real "
        "constraints, weighing subtle trade-offs. Keep the casual voice — a hard "
        "question can still be typed lazily. NOT just long, and NOT trivia that "
        "happens to be obscure."),
    TEMPORAL_RECALL: (
        "Questions about the user's OWN past with a time dimension, phrased WITHOUT "
        "an explicit parseable date (those are already caught by a deterministic "
        "detector, so they add nothing here): 'what was I leaning towards back when "
        "we started this', 'how did my thinking on the schema change', 'what did I "
        "say about it before the rewrite'. The time reference must be fuzzy and "
        "relational, not a calendar date."),
    "Needs_Live_Info": (
        "Questions needing information newer than any training data: current prices, "
        "today's news, live scores, latest release versions, whether something is "
        "still maintained. Include some that ALSO lean on personal context ('is the "
        "gpu i wanted still that price') — those overlaps are exactly what the old "
        "single-label taxonomy could not express."),
}


def measure_gaps(schema, floor: int) -> dict:
    """Which labels are short, and by how much."""
    counts = Counter()
    total = 0
    for entry in read_jsonl(LABELS_FINAL):
        total += 1
        for head_name in (INTENT, CONTEXT_RELIANCE):
            for label in entry.get("labels", {}).get(head_name, []):
                counts[label] += 1

    if not total:
        print("[synth] no labeled corpus yet — seeding the known-empty labels "
              "(the ones with zero v1 positives by construction)")
        return {label: floor for label in BRIEFS}

    gaps = {}
    for head_name in (INTENT, CONTEXT_RELIANCE):
        for label in schema.labels(head_name):
            have = counts[label]
            if have < floor and label in BRIEFS:
                gaps[label] = floor - have
    print(f"[synth] measured over {total} labeled rows; gaps: {gaps}")
    return gaps


async def _generate(client, model_id, label, brief, need, out, existing):
    made = 0
    rng = random.Random(f"{SEED}{label}")
    # Rotating flavours keep 300 prompts for one label from being 300 rephrasings
    # of the same sentence — temperature alone does not buy this much spread.
    flavours = ["frustrated and terse", "curious and rambling", "in a hurry",
                "thinking out loud", "half-remembering something",
                "polite and precise", "typing on a phone with typos",
                "picking up a thread after a break"]
    while made < need:
        flavour = rng.choice(flavours)
        user = (f"Generate {BATCH} distinct user prompts that would be labeled "
                f"{label}.\n\n{brief}\n\nWrite them as someone who is {flavour}. "
                f"Return a JSON array of {BATCH} strings.")
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0.95, top_p=0.95, max_tokens=1200)
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else parts[0]
                text = text[4:] if text.startswith("json") else text
            decoder = json.JSONDecoder()
            start = text.find("[")
            prompts, _ = decoder.raw_decode(text[start:] if start >= 0 else text)
        except Exception as exc:
            print(f"  ! {label}: {str(exc)[:120]}")
            break

        for prompt in prompts:
            if not isinstance(prompt, str) or not is_usable_prompt(prompt):
                continue
            key = prompt.strip().lower()
            if key in existing:
                continue
            existing.add(key)
            out.write({
                "id": stable_id("synth", prompt),
                "source": "synth",
                "provider": None,
                "text": prompt.strip(),
                "context_text": None,
                "conversation_id": None,
                "turn_index": None,
                "ts": None,
                # A HINT, not a label. This row still gets labeled by both
                # labelers like every other row — see the module docstring.
                "meta": {"target_label": label, "flavour": flavour,
                         "generated_by": model_id},
            })
            made += 1
        print(f"  {label}: {made}/{need}", flush=True)


async def run(gaps, base_url, model_id):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=base_url, api_key="local", timeout=300.0)
    existing = {row["text"].strip().lower() for row in read_jsonl(CORPUS_SYNTH)}
    with JsonlAppender(CORPUS_SYNTH) as out:
        for label, need in gaps.items():
            await _generate(client, model_id, label, BRIEFS[label], need, out, existing)


def main():
    ap = argparse.ArgumentParser(description="B1 stage 3: synthesise gap-filling prompts")
    ap.add_argument("--model", default="gemma-4-26b-a4b")
    ap.add_argument("--backend", default="vllm", choices=["sglang", "vllm"])
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--per-label", type=int, default=FLOOR)
    ap.add_argument("--attach", action="store_true")
    args = ap.parse_args()

    ensure_data_dir()
    schema = load_schema()
    gaps = measure_gaps(schema, args.per_label)
    if not gaps:
        print("[synth] every label is above the floor — nothing to generate")
        return
    already = len(completed_ids(CORPUS_SYNTH))
    if already:
        print(f"[synth] {already} synthetic rows already on disk (resuming)")

    server = LocalServer(args.model, backend=args.backend, port=args.port)
    if args.attach:
        if not is_up(server.base_url):
            raise SystemExit(f"nothing serving at {server.base_url}")
        asyncio.run(run(gaps, server.base_url,
                        served_model(server.base_url) or server.model))
        return
    with server:
        asyncio.run(run(gaps, server.base_url,
                        served_model(server.base_url) or server.model))


if __name__ == "__main__":
    main()
