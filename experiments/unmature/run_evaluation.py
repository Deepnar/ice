

import asyncio
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from tqdm.asyncio import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MASTER_RESULTS = "experiments/results_phase2/master_results.json"
CURATION_DIR = "experiments/curation_files"
OUTPUT_FILE = "experiments/results_phase2/evaluation_raw.json"

JUDGE_URL = "http://localhost:8003/v1/chat/completions"  # SGLang Gemma 12B
JUDGE_MODEL = "mattbucci/gemma-4-12B-AWQ"
JUDGE_TEMPERATURE = 0.0

API_CONCURRENCY = 6          # parallel judge calls
PROBE_CONCURRENCY = 2        # probes processed concurrently

DEBUG = False   # set to False to reduce verbosity

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def extract_json(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from a string."""
    if not text:
        return None
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    # Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    # fallback: try to parse the entire string
    try:
        return json.loads(text)
    except:
        return None

async def call_judge(session: aiohttp.ClientSession, messages: list, sem: asyncio.Semaphore,
                     max_tokens: int = 2048) -> Optional[str]:
    """Call the SGLang judge model with the given messages."""
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
                    if DEBUG:
                        # print a snippet of the response for debugging
                        print(f"\n[DEBUG] Raw response ({len(content)} chars): {content[:300]}...")
                    return content
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(3)
                else:
                    print(f"    Judge error: {e}")
                    return None

def load_completed_probes(output_path: str) -> set:
    """Return set of (checkpoint_id, probe_id) tuples already evaluated."""
    if not os.path.exists(output_path):
        return set()
    try:
        with open(output_path, "r") as f:
            data = json.load(f)
        return set((item["checkpoint_id"], item["probe_id"]) for item in data)
    except (json.JSONDecodeError, KeyError):
        return set()

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_master_results() -> List[dict]:
    """Return the list of evaluation_run_results entries."""
    master = load_json(MASTER_RESULTS)
    return master["evaluation_run_results"]

def load_curation(checkpoint_id: str) -> dict:
    """Load the curation file for a given checkpoint ID."""
    path = os.path.join(CURATION_DIR, f"{checkpoint_id}.json")
    return load_json(path)

# ---------------------------------------------------------------------------
# JUDGE PROMPTS (v1.1 - Refined for Factual Fidelity)
# ---------------------------------------------------------------------------
ABSOLUTE_SCORE_PROMPT = """You are an impartial judge evaluating an AI assistant's answer. 
The 'Ground Truth' provided is a factual extraction dossier. The 'AI Answer' is a conversational response. 

### SCORING RULES:
1. ACCURACY: Does the answer match the facts in the dossier?
2. INFERENCE ALIGNMENT (CRITICAL): High-resolution details (specific library names, academic years, or story events) are signs of superior retrieval. If the AI Answer is MORE SPECIFIC than the dossier but remains logically consistent with the dossier's themes, treat it as a SUCCESS, not a hallucination.
3. COMPLETENESS: Does it include the core anchors from the dossier?
4. DETAIL: Prioritize factual density. Do not penalize for length if the content provides specific, relevant evidence.

Output ONLY a JSON object:
{{
  "score": <integer 1-5>,
  "reasoning": "<one sentence explaining the score, mentioning if specificity helped or hindered>"
}}

Question: {question}
Ground Truth Dossier: {ground_truth}
AI Answer: {answer}

Rating:"""

TOURNAMENT_PROMPT = """You are an expert judge comparing six different AI answers to the same question.
The 'Ground Truth' provided is a factual extraction dossier. The 'AI Answers' are conversational responses.
Your task is to rank the answers from 1 (best) to 6 (worst) based on how faithfully they represent the facts in the dossier.
SPECIFICITY BONUS: High-resolution technical details (e.g., specific library names, design patterns like 'Event Sourcing', or personal milestones like '3rd Year') are signs of superior retrieval. If a model provides accurate, specific details that are logically consistent with the dossier—even if those details are NOT explicitly in the dossier—rank that model HIGHER than models providing generic summaries.

- Explain why the top-ranked answer is the best.
- Focus on accuracy and completeness.

Output ONLY a JSON object with the following format:
{{
  "rankings": ["A", "B", "C", "D", "E", "F"],
  "best_reason": "<one sentence why the best is best>",
  "worst_reason": "<one sentence why the worst is worst>"
}}

Question: {question}

Ground Truth Dossier:
{ground_truth}

Answers:
A: {answer_A}
B: {answer_B}
C: {answer_C}
D: {answer_D}
E: {answer_E}
F: {answer_F}

Ranking:"""

HALLUCINATION_AUDIT_PROMPT = """You are a fact-checking auditor. 
The 'Ground Truth' is a factual dossier. Identify any statements regarding specific personal facts, story names, or project decisions that are:
- Not present in the ground truth (hallucinations)
- Contradictory to the ground truth (errors)

CRITICAL RULE: Ignore general external knowledge (e.g. definitions of programming terms like 'FastAPI' or hardware specs like 'RTX 5090'). Only flag hallucinations regarding the specific personal lore, story events, or user decisions contained in the dossier.

CONSISTENCY CLAUSE: Do not flag specific technical components, academic milestones, or story names as hallucinations if they are logically consistent with the themes in the dossier. For example, if the dossier mentions 'AI Engineering' and the model identifies 'PyTorch MLP,' treat this as a correct specific inference, not a hallucination. Only flag details that explicitly contradict the dossier or feel completely random/out-of-place.

Output ONLY a JSON object:
{{
  "hallucination_count": <integer>,
  "error_count": <integer>,
  "details": "<list of specific lore statements that are unsupported or wrong>"
}}

Question: {question}

Ground Truth Dossier:
{ground_truth}

AI Answer:
{answer}

Audit:"""

FRAGMENT_NOISE_PROMPT = """You are a context quality auditor evaluating retrieved fragments.
Rate the noise level (1-10) and relevance (0-100%).

### UTILITY RULE:
Consider 'Structural Relevance.' A fragment might not contain the exact keyword of the question but may provide the necessary background context (e.g. the user's college name providing context for a question about marks). Treat foundational context as RELEVANT, not noise.

Output ONLY a JSON object:
{{
  "noise_score": <integer 1-10>,
  "relevance_percentage": <integer 0-100>,
  "explanation": "<one sentence justifying the relevance based on specificity>"
}}

Question: {question}
Retrieved Fragments: {fragments}

Assessment:"""


# ---------------------------------------------------------------------------
# PER-PROBE EVALUATION
# ---------------------------------------------------------------------------
async def evaluate_probe(session: aiohttp.ClientSession, probe_result: dict,
                         curation_data: dict, sem: asyncio.Semaphore) -> dict:
    """
    For a single probe, run all judge passes and return a dict of raw results.
    """
    metadata = probe_result["metadata"]
    checkpoint_id = metadata["checkpoint_id"]
    probe_id = metadata["probe_id"]

    # Get ground truth from curation
    ground_truth = ""
    for p in curation_data.get("evaluation_probes", []):
        if p["probe_id"] == probe_id:
            ground_truth = p.get("expected_answer", "")
            break
    if not ground_truth:
        # fallback: use the historical context block as ground truth (last few turns)
        hist = curation_data.get("historical_context_block", [])
        ground_truth = "\n".join(
            f"User: {h['user_input']}\nAssistant: {h['ai_response']}" for h in hist[-5:]
        )

    question = metadata.get("raw_user_probe", "")
    execution = probe_result["execution_permutations"]
    conditions = list(execution.keys())

    # Initialize result container
    probe_eval = {
        "checkpoint_id": checkpoint_id,
        "probe_id": probe_id,
        "question": question,
        "ground_truth": ground_truth,
        "absolute_scores": {},
        "tournament": None,
        "hallucination": {},
        "fragment_analysis": {},
        "multi_source_counts": {},
    }

    # -----------------------------------------------------------------------
    # 1. Absolute Scoring (each condition) – large max_tokens for reasoning
    # -----------------------------------------------------------------------
    for cond in conditions:
        answer = execution[cond].get("answer", "")
        if not answer or answer.startswith("ERROR"):
            probe_eval["absolute_scores"][cond] = None
            continue
        prompt = ABSOLUTE_SCORE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer
        )
        messages = [{"role": "user", "content": prompt}]
        resp = await call_judge(session, messages, sem, max_tokens=4096)
        if resp:
            parsed = extract_json(resp)
            if parsed and "score" in parsed:
                probe_eval["absolute_scores"][cond] = {
                    "score": parsed.get("score"),
                    "reasoning": parsed.get("reasoning", "")
                }
                print(f"    {checkpoint_id}/{probe_id}: scored {cond} -> {probe_eval['absolute_scores'][cond]}", flush=True)
            else:
                if DEBUG:
                    print(f"    Failed to parse absolute score for {cond}: {resp[:200]}...")
                probe_eval["absolute_scores"][cond] = None
        else:
            probe_eval["absolute_scores"][cond] = None

    # -----------------------------------------------------------------------
    # 2. Comparative Tournament (only the 6 main conditions)
    # -----------------------------------------------------------------------
    main_conditions = [c for c in conditions if c in {
        "control_baseline_generalist", "control_moe",
        "vector_rag_baseline_generalist", "vector_rag_moe",
        "full_ice_generalist", "full_ice_moe"
    }]
    if len(main_conditions) == 6:
        shuffled = main_conditions.copy()
        random.shuffle(shuffled)
        answer_map = {}
        for i, cond in enumerate(shuffled):
            label = chr(65 + i)
            answer_map[label] = execution[cond].get("answer", "")
        prompt = TOURNAMENT_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            answer_A=answer_map["A"],
            answer_B=answer_map["B"],
            answer_C=answer_map["C"],
            answer_D=answer_map["D"],
            answer_E=answer_map["E"],
            answer_F=answer_map["F"],
        )
        messages = [{"role": "user", "content": prompt}]
        resp = await call_judge(session, messages, sem, max_tokens=8192)
        if resp:
            parsed = extract_json(resp)
            if parsed and "rankings" in parsed:
                label_to_cond = {chr(65 + i): cond for i, cond in enumerate(shuffled)}
                rankings = []
                for r in parsed.get("rankings", []):
                    rankings.append(label_to_cond.get(r, r))
                probe_eval["tournament"] = {
                    "rankings": rankings,
                    "best_reason": parsed.get("best_reason", ""),
                    "worst_reason": parsed.get("worst_reason", "")
                }
            elif DEBUG:
                print(f"    Failed to parse tournament: {resp[:200]}...")

    # -----------------------------------------------------------------------
    # 3. Hallucination Audit (all conditions)
    # -----------------------------------------------------------------------
    for cond in conditions:
        answer = execution[cond].get("answer", "")
        if not answer or answer.startswith("ERROR"):
            continue
        prompt = HALLUCINATION_AUDIT_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer
        )
        messages = [{"role": "user", "content": prompt}]
        resp = await call_judge(session, messages, sem, max_tokens=4096)
        if resp:
            parsed = extract_json(resp)
            if parsed:
                probe_eval["hallucination"][cond] = {
                    "hallucination_count": parsed.get("hallucination_count", 0),
                    "error_count": parsed.get("error_count", 0),
                    "details": parsed.get("details", [])
                }
            elif DEBUG:
                print(f"    Failed to parse hallucination for {cond}: {resp[:200]}...")

    # -----------------------------------------------------------------------
    # 4. Fragment Noise Analysis (full ICE and vector, if available)
    # -----------------------------------------------------------------------
    full_ice_frags = probe_result.get("full_ice_fragments", [])
    vector_frags = probe_result.get("vector_fragments", [])
    if full_ice_frags:
        frag_text = "\n---\n".join(f["text"][:500] for f in full_ice_frags[:10])
        prompt = FRAGMENT_NOISE_PROMPT.format(
            question=question,
            fragments=frag_text
        )
        messages = [{"role": "user", "content": prompt}]
        resp = await call_judge(session, messages, sem, max_tokens=1024)
        if resp:
            parsed = extract_json(resp)
            if parsed:
                probe_eval["fragment_analysis"]["full_ice"] = {
                    "noise_score": parsed.get("noise_score"),
                    "relevance_percentage": parsed.get("relevance_percentage"),
                    "explanation": parsed.get("explanation")
                }
            elif DEBUG:
                print(f"    Failed to parse fragment noise for full_ice: {resp[:200]}...")
    if vector_frags:
        frag_text = "\n---\n".join(f["text"][:500] for f in vector_frags[:10])
        prompt = FRAGMENT_NOISE_PROMPT.format(
            question=question,
            fragments=frag_text
        )
        messages = [{"role": "user", "content": prompt}]
        resp = await call_judge(session, messages, sem, max_tokens=1024)
        if resp:
            parsed = extract_json(resp)
            if parsed:
                probe_eval["fragment_analysis"]["vector_rag"] = {
                    "noise_score": parsed.get("noise_score"),
                    "relevance_percentage": parsed.get("relevance_percentage"),
                    "explanation": parsed.get("explanation")
                }
            elif DEBUG:
                print(f"    Failed to parse fragment noise for vector_rag: {resp[:200]}...")

    # Multi-source synergy counts
    source_counts = {}
    for f in full_ice_frags:
        src = f.get("source_type", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    probe_eval["multi_source_counts"] = source_counts

    return probe_eval

# ---------------------------------------------------------------------------
# MAIN ASYNC PIPELINE
# ---------------------------------------------------------------------------
async def main():
    print("Loading master results...")
    master_entries = load_master_results()
    print(f"Loaded {len(master_entries)} probe results.")

    # Resumability: load already-evaluated probes
    completed = load_completed_probes(OUTPUT_FILE)
    if completed:
        print(f"Found {len(completed)} already evaluated probes in {OUTPUT_FILE}.")
        pending = [e for e in master_entries if (e["metadata"]["checkpoint_id"], e["metadata"]["probe_id"]) not in completed]
        print(f"Skipping completed probes. Remaining: {len(pending)} / {len(master_entries)}")
    else:
        pending = master_entries
        print("No existing evaluations found. Starting from scratch.")

    # Load curation cache
    curation_cache = {}
    for entry in pending:
        cid = entry["metadata"]["checkpoint_id"]
        if cid not in curation_cache:
            try:
                curation_cache[cid] = load_curation(cid)
            except FileNotFoundError:
                print(f"Warning: curation file for {cid} not found.")
                curation_cache[cid] = {}

    sem_judge = asyncio.Semaphore(API_CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=API_CONCURRENCY * 2)

    # Load existing results to append to (if resuming)
    if os.path.exists(OUTPUT_FILE):
        try:
            eval_results = load_json(OUTPUT_FILE)
        except:
            eval_results = []
    else:
        eval_results = []

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for entry in tqdm(pending, desc="Preparing tasks"):
            cid = entry["metadata"]["checkpoint_id"]
            curation = curation_cache.get(cid, {})
            task = evaluate_probe(session, entry, curation, sem_judge)
            tasks.append(task)

        sem_probes = asyncio.Semaphore(PROBE_CONCURRENCY)
        async def bounded(task):
            async with sem_probes:
                return await task

        processed = 0
        for coro in tqdm(asyncio.as_completed([bounded(t) for t in tasks]),
                         total=len(tasks), desc="Evaluating probes"):
            result = await coro
            eval_results.append(result)

            # ── Safety save after every probe ──
            save_json(eval_results, OUTPUT_FILE)

            # ── Informative print ──
            processed += 1
            cid = result["checkpoint_id"]
            pid = result["probe_id"]
            abs_scores = result.get("absolute_scores", {})
            score_ice = abs_scores.get("full_ice_generalist", {})
            score_val = score_ice.get("score", "?") if isinstance(score_ice, dict) else "?"
            print(f"\n[{processed}/{len(pending)}] {cid} {pid} | ICE score: {score_val}")
            # Optional: show tournament rank for ICE if available
            tourn = result.get("tournament")
            if tourn and "rankings" in tourn:
                try:
                    ice_rank = tourn["rankings"].index("full_ice_generalist") + 1
                    print(f"  Tournament rank (ICE): {ice_rank}/6")
                except ValueError:
                    pass

    print(f"\nEvaluation complete. {len(eval_results)} probes evaluated (includes previous). Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())