#!/usr/bin/env python3
"""
ICE Flaw Buildup Evaluation (Single‑Pass)
=========================================
Runs the judge pipeline on all buildup conditions.
Each probe appears once against the fully‑mature database (turn 1119).

Reads:  experiments/flaw_ablation/buildup/master_results.json
Writes: experiments/flaw_ablation/buildup/evaluation_raw.json
"""

import asyncio, json, os, re
from typing import Optional
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MASTER_RESULTS = "experiments/flaw_ablation/buildup/master_results.json"
OUTPUT_FILE    = "experiments/flaw_ablation/buildup/evaluation_raw.json"
JUDGE_URL      = "http://localhost:8003/v1/chat/completions"
JUDGE_MODEL    = "mattbucci/gemma-4-12B-AWQ"
JUDGE_TEMP     = 0.0
API_CONCURRENCY = 6
PROBE_CONCURRENCY = 2

# ── Temporally‑aware judge prompts (same as subtraction eval) ─────────
ABSOLUTE_SCORE_PROMPT = """You are an impartial judge evaluating an AI assistant's answer.
The 'Ground Truth Dossier' contains ALL facts known up to the CURRENT point in the
conversation (turn {turn_index}).  Some facts may have been updated by later events; when
a fact changed, both the earlier and later versions are present, each labelled with its
approximate turn range. But there can also be some unnecessary information that has inflated
the ground truth or the ground truth may be wrong, therefore, make the judgement based on both 
what the question is asking, and how much of the ground truth is actually necessary, and answer 
that question itself and compare based on that.

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
  err on the side of NOT flagging it. There will be times where somthing might look like a 
  hallucination with a lot more info in the AI answer, but those are not hallucinations at all.
  So be lenient as possible.

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
  be considered noise.
- ★ Inference Safe-Zone: A fragment with correct, specific details logically connected to
  the ground truth should be treated as RELEVANT. Do not penalise the fragment for being
  more specific than the dossier requires.

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
# HELPERS
# ---------------------------------------------------------------------------
def load_json(path): 
    with open(path, "r") as f: return json.load(f)
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2)
def extract_json(text):
    if not text: return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    try: return json.loads(text)
    except: return None

def load_completed_probes(output_path):
    if not os.path.exists(output_path): return set()
    try:
        with open(output_path, "r") as f: data = json.load(f)
        return set(item["probe_id"] for item in data)
    except: return set()

async def call_judge(session, messages, sem, max_tokens=2048):
    payload = {"model": JUDGE_MODEL, "messages": messages, "temperature": JUDGE_TEMP,
               "max_tokens": max_tokens, "stream": False}
    async with sem:
        for attempt in range(2):
            try:
                async with session.post(JUDGE_URL, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=180)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Judge error {resp.status}: {text}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    if not content and attempt == 0: await asyncio.sleep(3); continue
                    return content
            except Exception:
                if attempt == 0: await asyncio.sleep(3)
    return None

def _load_fragment_texts(probe_result, cond_name):
    fragments_file = "experiments/flaw_ablation/buildup/fragments.jsonl"
    if not os.path.exists(fragments_file): return []
    pid = probe_result["probe_id"]
    texts = []
    with open(fragments_file, "r") as f:
        for line in f:
            try:
                frag = json.loads(line.strip())
                if frag.get("probe_id") == pid and frag.get("condition") == cond_name:
                    texts.append(frag.get("text", "")[:500])
            except: continue
    return texts

# ---------------------------------------------------------------------------
# PER‑PROBE EVALUATION
# ---------------------------------------------------------------------------
async def evaluate_probe(session, probe_result, sem):
    pid = probe_result["probe_id"]
    question = probe_result["question"]
    ground_truth = probe_result["ground_truth"]
    turn_index = probe_result["turn_index"]
    conditions = list(probe_result["conditions"].keys())

    probe_eval = {"probe_id": pid, "question": question,
                  "ground_truth": ground_truth, "turn_index": turn_index,
                  "absolute_scores": {}, "hallucination": {}, "fragment_analysis": {}}

    for cond in conditions:
        answer = probe_result["conditions"][cond].get("answer", "")
        if not answer or answer.startswith("ERROR"):
            probe_eval["absolute_scores"][cond] = None; continue
        prompt = ABSOLUTE_SCORE_PROMPT.format(question=question, turn_index=turn_index,
                                               ground_truth=ground_truth, answer=answer)
        resp = await call_judge(session, [{"role": "user", "content": prompt}], sem, max_tokens=4096)
        if resp:
            parsed = extract_json(resp)
            probe_eval["absolute_scores"][cond] = {"score": parsed.get("score"), "reasoning": parsed.get("reasoning", "")} if parsed and "score" in parsed else None
        else: probe_eval["absolute_scores"][cond] = None

    for cond in conditions:
        answer = probe_result["conditions"][cond].get("answer", "")
        if not answer or answer.startswith("ERROR"): continue
        prompt = HALLUCINATION_AUDIT_PROMPT.format(question=question, turn_index=turn_index,
                                                    ground_truth=ground_truth, answer=answer)
        resp = await call_judge(session, [{"role": "user", "content": prompt}], sem, max_tokens=4096)
        if resp:
            parsed = extract_json(resp)
            if parsed:
                probe_eval["hallucination"][cond] = {"hallucination_count": parsed.get("hallucination_count", 0),
                                                      "error_count": parsed.get("error_count", 0),
                                                      "details": parsed.get("details", [])}

    # Fragment noise: full_ice and bare_vector
    if "full_ice" in conditions:
        frag_texts = _load_fragment_texts(probe_result, "full_ice")
        if frag_texts:
            prompt = FRAGMENT_NOISE_PROMPT.format(question=question, turn_index=turn_index,
                                                   fragments="\n---\n".join(frag_texts[:10]))
            resp = await call_judge(session, [{"role": "user", "content": prompt}], sem, max_tokens=1024)
            if resp:
                parsed = extract_json(resp)
                if parsed:
                    probe_eval["fragment_analysis"]["full_ice"] = {
                        "noise_score": parsed.get("noise_score"),
                        "relevance_percentage": parsed.get("relevance_percentage"),
                        "explanation": parsed.get("explanation")}
    if "vector_baseline" in conditions:
        frag_texts = _load_fragment_texts(probe_result, "vector_baseline")
        if frag_texts:
            prompt = FRAGMENT_NOISE_PROMPT.format(question=question, turn_index=turn_index,
                                                   fragments="\n---\n".join(frag_texts[:10]))
            resp = await call_judge(session, [{"role": "user", "content": prompt}], sem, max_tokens=1024)
            if resp:
                parsed = extract_json(resp)
                if parsed:
                    probe_eval["fragment_analysis"]["vector_rag"] = {
                        "noise_score": parsed.get("noise_score"),
                        "relevance_percentage": parsed.get("relevance_percentage"),
                        "explanation": parsed.get("explanation")}
    return probe_eval

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main():
    print("Loading master results...")
    master = load_json(MASTER_RESULTS)["evaluation_run_results"]
    print(f"Loaded {len(master)} probe results.")
    completed = load_completed_probes(OUTPUT_FILE)
    pending = [e for e in master if e["probe_id"] not in completed] if completed else master
    print(f"Pending: {len(pending)}")
    sem_judge = asyncio.Semaphore(API_CONCURRENCY)
    eval_results = load_json(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else []

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=API_CONCURRENCY*2)) as session:
        sem_probes = asyncio.Semaphore(PROBE_CONCURRENCY)
        async def bounded(entry):
            async with sem_probes: return await evaluate_probe(session, entry, sem_judge)
        tasks = [bounded(entry) for entry in pending]
        processed = 0
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating probes"):
            result = await coro
            eval_results.append(result)
            save_json(eval_results, OUTPUT_FILE)
            processed += 1
            print(f"\n[{processed}/{len(pending)}] {result['probe_id']}")
    print(f"\nEvaluation complete. {len(eval_results)} probes. Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())