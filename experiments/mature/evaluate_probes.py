#!/usr/bin/env python3
"""
ICE-Mature Judge Evaluation Pipeline
====================================
Loads Experiment 2 master_results.json, runs four forensic judge passes
per probe, and writes evaluation_raw.json.

Key change from Experiment 1: the absolute scoring prompt is now
TEMPORALLY‑AWARE.  The ground truth dossier accumulates facts across
checkpoints; the AI answer is only expected to know facts up to the
current turn.  The rubric distinguishes "correct but outdated" (score 3)
from "correct and current" (score 5).
"""

import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from tqdm.asyncio import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MATURE_DIR = Path(__file__).parent
MASTER_RESULTS = MATURE_DIR / "intermediates" / "master_results.json"
OUTPUT_FILE = MATURE_DIR / "intermediates" / "evaluation_raw.json"

JUDGE_URL = "http://localhost:8003/v1/chat/completions"
JUDGE_MODEL = "mattbucci/gemma-4-12B-AWQ"
JUDGE_TEMPERATURE = 0.0

API_CONCURRENCY = 6
PROBE_CONCURRENCY = 2

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

async def call_judge(session: aiohttp.ClientSession, messages: list,
                     sem: asyncio.Semaphore, max_tokens: int = 2048) -> Optional[str]:
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with sem:
        for attempt in range(2):
            try:
                async with session.post(JUDGE_URL, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Judge error {resp.status}: {text}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    if not content and attempt == 0:
                        await asyncio.sleep(3)
                        continue
                    return content
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(3)
                else:
                    print(f"    Judge error: {e}")
                    return None

def load_completed_probes(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    try:
        with open(output_path, "r") as f:
            data = json.load(f)
        return set((item["conversation_id"], item["probe_id"], item["checkpoint_id"])
                   for item in data)
    except (json.JSONDecodeError, KeyError):
        return set()

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_master_entries() -> List[dict]:
    """Experiment 2 master_results.json → list of probe entries."""
    with open(MASTER_RESULTS) as f:
        data = json.load(f)
    return data["evaluation_run_results"]

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

ABSOLUTE_SCORE_PROMPT = """You are an impartial judge evaluating an AI assistant's answer.
The 'Ground Truth Dossier' contains ALL facts known up to the CURRENT point in the
conversation (turn {turn_index}).  Some facts may have been updated by later events; when
a fact changed, both the earlier and later versions are present, each labelled with its
approximate turn range. But there can also be some unnecessary information that has inflated
the ground truth, therefore, make the judgement based on both what the question is asking, 
and how much of the ground truth is actually necessary, and answer that question itself 
and compare based on that.

The AI is answering at turn {turn_index}.  It ONLY has access to information up to that turn.

### SCORING RUBRIC (1–5)

**5 – Complete, Current, and Specific**
The answer is fully correct for turn {turn_index}.  It includes the MOST RECENT state of
every relevant fact.  If a fact evolved over time, it mentions both the earlier version AND
what it changed to (or at minimum gives the current state).  Specific anchors (names,
numbers, chapter references) are present.


**4 – Mostly Complete**
The answer is correct and gives the current state of most facts, but is missing a minor
detail that was knowable at this turn.  No outdated or hallucinated information.

**3 – Partially Correct / Outdated**
The answer contains correct facts, but:
- It reports an EARLIER version of a fact without acknowledging a later update that was
  already known by this turn (e.g., gives Chapter 23 roles at turn 55 when Chapter 35
  roles had already been assigned), OR
- It omits a major current fact while including other correct facts.
Not hallucinating, just temporally incomplete or lagging.

**2 – Mixed Correct & Incorrect**
Some correct facts are present, but they are mixed with hallucinations, significant
errors, or the answer is so generic that it fails to use the specific details available
at this turn.

**1 – Wrong / No Answer**
The answer is almost entirely wrong, hallucinated, contradicts the dossier, or the model
refused to answer.

★ Inference Bonus: If the answer adds CORRECT, SPECIFIC details and information that are not 
explicitly in the dossier but are logically consistent with it (e.g., a correct character trait, 
a specific library name, a decision, or information), treat this as a strength, not a
hallucination.  The score should reflect that the system surfaced and used high‑quality detail 
and be increased by 1, only for score which are 4 and below.

### OUTPUT FORMAT
Return ONLY a JSON object:
{{
  "score": <integer 1-5>,
  "reasoning": "<one sentence explaining the score, referencing temporal correctness,
                missing updates, or the presence of correct specific inferences>"
}}

Question: {question}
Current turn: {turn_index}
Ground Truth Dossier: {ground_truth}
AI Answer: {answer}

Rating:"""


TOURNAMENT_PROMPT = """You are an expert judge comparing four AI answers to the same question.
The 'Ground Truth Dossier' contains all facts known up to the current turn ({turn_index}).
Some facts may have been updated; the most recent version of each fact is the authoritative
one for this turn.

Rank the answers from 1 (best) to 4 (worst) based on:
- ACCURACY: faithfulness to the dossier
- TEMPORAL CORRECTNESS: does the answer reflect the CURRENT state of facts, or does it
  report outdated information that was superseded by this turn?
- COMPLETENESS: does it cover the key facts knowable at this turn?
- SPECIFICITY: does it use concrete details (names, numbers, chapter references) from the
  dossier, or is it vague?

Output ONLY a JSON object:
{{
  "rankings": ["A", "B", "C", "D"],
  "best_reason": "<one sentence why the best is best>",
  "worst_reason": "<one sentence why the worst is worst>"
}}

Question: {question}
Current turn: {turn_index}

Ground Truth Dossier:
{ground_truth}

Answers:
A: {answer_A}
B: {answer_B}
C: {answer_C}
D: {answer_D}

Ranking:"""


HALLUCINATION_AUDIT_PROMPT = """You are a fact-checking auditor.
The 'Ground Truth Dossier' contains all facts known up to turn {turn_index}.
But there can also be some unnecessary information that has inflated
the ground truth, therefore, make the judgement based on both what the question is asking, 
and how much of the ground truth is actually necessary, and answer that question itself 
and compare based on that.
Identify statements in the AI's answer that are:
- Not present in the dossier (hallucinations)
- Contradictory to the CURRENT state of facts in the dossier (errors)

CRITICAL RULES (apply in this order):

1. ★ INFERENCE SAFE‑ZONE (MOST IMPORTANT — apply FIRST): If the AI adds a CORRECT, 
SPECIFIC detail that is not explicitly listed in the dossier but is logically consistent 
with it, do NOT flag it as a hallucination. This applies ESPECIALLY to:
  - Specific creative works (anime, manga, songs, games) the user mentioned as inspiration
  - Character names, project names, tool names, library names, or other proper nouns
  - Numbers, dates, or statistics that are consistent with what the dossier describes
  - Emotional states, preferences, or personal history that the dossier implies but doesn't spell out
  If you are unsure whether a detail is a hallucination or a correct inference, 
  err on the side of NOT flagging it.

2. TEMPORAL INCOMPLETENESS IS NOT HALLUCINATION: If the dossier contains both an earlier 
and later version of a fact, the later version is authoritative for this turn. Reporting 
the EARLIER version without the update is NOT a hallucination (it's a temporal 
incompleteness). Only flag things that were NEVER true.

3. GENERAL KNOWLEDGE EXEMPTION: Ignore general external knowledge (e.g. definitions of 
programming terms, public facts). Only flag hallucinations regarding specific personal 
lore, story events, or project decisions contained in the dossier.

Output ONLY a JSON object:
{{
  "hallucination_count": <integer>,
  "error_count": <integer>,
  "details": "<list of specific statements that are unsupported or wrong>"
}}

Question: {question}
Current turn: {turn_index}
Ground Truth Dossier:
{ground_truth}
AI Answer:
{answer}

Audit:"""

FRAGMENT_NOISE_PROMPT = """You are a context quality auditor evaluating retrieved fragments
at turn {turn_index}. Rate the noise level (1-10) and relevance (0-100%).

UTILITY RULE:
Consider 'Structural Relevance.' A fragment might not contain the exact keyword of the
question but may provide necessary background context known at this turn. Treat foundational
context as RELEVANT, not noise.
- Fragments containing information from AFTER this turn are NOT valid context and should
  be considered noise (high noise score, low relevance).
- ★ Inference Safe‑Zone: A fragment that contains correct, specific details that the
  question implies or that logically connect to the ground truth should be treated as
  RELEVANT (high relevance, low noise), even if those exact details aren't in the
  ground truth dossier. Do not penalise the fragment for being more specific than the
  dossier requires.

Output ONLY a JSON object:
{{
  "noise_score": <integer 1-10>,
  "relevance_percentage": <integer 0-100>,
  "explanation": "<one sentence justifying the relevance>"
}}

Question: {question}
Current turn: {turn_index}
Retrieved Fragments: {fragments}

Assessment:"""

# ---------------------------------------------------------------------------
# PER-PROBE EVALUATION
# ---------------------------------------------------------------------------
async def evaluate_probe(session: aiohttp.ClientSession, entry: dict,
                         sem: asyncio.Semaphore) -> dict:
    cid = entry["conversation_id"]
    checkpoint_id = entry["checkpoint_id"]
    probe_id = entry["probe_id"]
    question = entry["question"]
    ground_truth = entry.get("ground_truth", "")
    turn_index = entry.get("turn_index", 0)
    conditions = entry.get("conditions", {})

    probe_eval = {
        "conversation_id": cid,
        "checkpoint_id": checkpoint_id,
        "probe_id": probe_id,
        "question": question,
        "ground_truth": ground_truth,
        "turn_index": turn_index,
        "absolute_scores": {},
        "tournament": None,
        "hallucination": {},
        "fragment_analysis": {},
    }

    # 1. Absolute Scoring
    for cond_name, cond_data in conditions.items():
        answer = cond_data.get("answer", "")
        if not answer or answer.startswith("ERROR"):
            probe_eval["absolute_scores"][cond_name] = None
            continue
        prompt = ABSOLUTE_SCORE_PROMPT.format(
            turn_index=turn_index,
            question=question,
            ground_truth=ground_truth,
            answer=answer,
        )
        messages = [{"role": "user", "content": prompt}]
        resp = await call_judge(session, messages, sem, max_tokens=4096)
        if resp:
            parsed = extract_json(resp)
            if parsed and "score" in parsed:
                probe_eval["absolute_scores"][cond_name] = {
                    "score": parsed.get("score"),
                    "reasoning": parsed.get("reasoning", ""),
                }
            else:
                probe_eval["absolute_scores"][cond_name] = None
        else:
            probe_eval["absolute_scores"][cond_name] = None

    # 2. Tournament (4 conditions)
    cond_names = list(conditions.keys())
    if len(cond_names) == 4:
        shuffled = cond_names.copy()
        random.shuffle(shuffled)
        answer_map = {}
        for i, cn in enumerate(shuffled):
            label = chr(65 + i)
            answer_map[label] = conditions[cn].get("answer", "")
        prompt = TOURNAMENT_PROMPT.format(
            turn_index=turn_index,
            question=question,
            ground_truth=ground_truth,
            answer_A=answer_map["A"],
            answer_B=answer_map["B"],
            answer_C=answer_map["C"],
            answer_D=answer_map["D"],
        )
        messages = [{"role": "user", "content": prompt}]
        resp = await call_judge(session, messages, sem, max_tokens=8192)
        if resp:
            parsed = extract_json(resp)
            if parsed and "rankings" in parsed:
                label_to_cond = {chr(65 + i): cn for i, cn in enumerate(shuffled)}
                rankings = [label_to_cond.get(r, r) for r in parsed["rankings"]]
                probe_eval["tournament"] = {
                    "rankings": rankings,
                    "best_reason": parsed.get("best_reason", ""),
                    "worst_reason": parsed.get("worst_reason", ""),
                }

    # 3. Hallucination Audit
    for cond_name, cond_data in conditions.items():
        answer = cond_data.get("answer", "")
        if not answer or answer.startswith("ERROR"):
            continue
        prompt = HALLUCINATION_AUDIT_PROMPT.format(
            turn_index=turn_index,
            question=question,
            ground_truth=ground_truth,
            answer=answer,
        )
        messages = [{"role": "user", "content": prompt}]
        resp = await call_judge(session, messages, sem, max_tokens=4096)
        if resp:
            parsed = extract_json(resp)
            if parsed:
                probe_eval["hallucination"][cond_name] = {
                    "hallucination_count": parsed.get("hallucination_count", 0),
                    "error_count": parsed.get("error_count", 0),
                    "details": parsed.get("details", []),
                }

    # 4. Fragment Noise (full_ice only — fragments are stored in fragments.jsonl,
    #    but we can access the fragment_ids and reconstruct from there if needed.
    #    For simplicity, we'll skip fragment noise in this script for now, or
    #    load fragments.jsonl and match.  We'll add a basic implementation.)
    #    Since the fragment texts are not in the master_results (only IDs), we
    #    load them from fragments.jsonl.
    frag_file = MATURE_DIR / "intermediates" / "fragments.jsonl"
    if frag_file.exists():
        # Build lookup: (conv_id, probe_id, checkpoint_id, condition) → [texts]
        frag_lookup = {}
        with open(frag_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frag = json.loads(line)
                key = (frag["conversation_id"], frag["probe_id"],
                       frag["checkpoint_id"], frag["condition"])
                frag_lookup.setdefault(key, []).append(frag["text"])

        for cond_name in conditions:
            key = (cid, probe_id, checkpoint_id, cond_name)
            texts = frag_lookup.get(key, [])
            if texts:
                frag_text = "\n---\n".join(t[:500] for t in texts[:10])
                prompt = FRAGMENT_NOISE_PROMPT.format(
                    turn_index=turn_index,
                    question=question,
                    fragments=frag_text,
                )
                messages = [{"role": "user", "content": prompt}]
                resp = await call_judge(session, messages, sem, max_tokens=1024)
                if resp:
                    parsed = extract_json(resp)
                    if parsed:
                        probe_eval["fragment_analysis"][cond_name] = {
                            "noise_score": parsed.get("noise_score"),
                            "relevance_percentage": parsed.get("relevance_percentage"),
                            "explanation": parsed.get("explanation"),
                        }

    return probe_eval


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main():
    print("Loading master results...")
    entries = load_master_entries()
    print(f"Loaded {len(entries)} probe results.")

    completed = load_completed_probes(OUTPUT_FILE)
    if completed:
        pending = [e for e in entries
                   if (e["conversation_id"], e["probe_id"], e["checkpoint_id"]) not in completed]
        print(f"Resuming: {len(pending)} remaining out of {len(entries)}")
    else:
        pending = entries
        print("Starting from scratch.")

    sem = asyncio.Semaphore(API_CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=API_CONCURRENCY * 2)

    if os.path.exists(OUTPUT_FILE):
        try:
            eval_results = load_json(OUTPUT_FILE)
        except Exception:
            eval_results = []
    else:
        eval_results = []

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [evaluate_probe(session, e, sem) for e in pending]
        sem_probes = asyncio.Semaphore(PROBE_CONCURRENCY)

        async def bounded(task):
            async with sem_probes:
                return await task

        processed = 0
        for coro in tqdm(asyncio.as_completed([bounded(t) for t in tasks]),
                         total=len(tasks), desc="Evaluating probes"):
            result = await coro
            eval_results.append(result)
            save_json(eval_results, OUTPUT_FILE)
            processed += 1
            score_ice = result.get("absolute_scores", {}).get("full_ice_generalist", {})
            s = score_ice.get("score", "?") if isinstance(score_ice, dict) else "?"
            print(f"[{processed}/{len(pending)}] {result['checkpoint_id']} {result['probe_id']} ICE score: {s}")

    print(f"\nDone. {len(eval_results)} probes in {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())