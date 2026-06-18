#!/usr/bin/env python3
"""Generate ground‑truth expected answers for every probe using vector_contexts.json.
   Writes the answers directly into the curation files (expected_answer field)."""

import asyncio, json, os, time
import aiohttp
from tqdm.asyncio import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VECTOR_CONTEXTS = "experiments/results_phase2/vector_contexts.json"
CURATION_DIR = "experiments/curation_files"
PROGRESS_FILE = "experiments/results_phase2/ground_truth_progress.json"

SGLANG_URL = "http://localhost:8003/v1/chat/completions"# rename variable to VLLM_URL or keep OLLAMA_URL – up to you
MODEL_NAME = "gemma4:12b-256k"

API_CONCURRENCY = 10
ENTRY_CONCURRENCY = 4       # 4 probes in parallel – keep GPU busy but not overloaded
MAX_TOKENS = 6000           # enough for a thorough answer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "r") as f: return json.load(f)
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2)
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return set()
                return set(json.loads(content))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()
def save_progress(checkpoint_id, probe_id):
    prog = load_progress()
    prog.add(f"{checkpoint_id}|{probe_id}")
    save_json(sorted(prog), PROGRESS_FILE)

# ---------------------------------------------------------------------------
# Prompt for ground‑truth generation
# ---------------------------------------------------------------------------
GROUND_TRUTH_PROMPT = """You are a neutral evidence compiler. Below are excerpts from a conversation.
Answer the user's question using ONLY facts explicitly stated in the excerpts.

RULES (no exceptions):
1. Report every relevant detail, name, number, and decision verbatim or in close paraphrase.
2. If the question asks for a list (e.g. "what are the X"), scan all excerpts and list every distinct item found.
3. Attribute all opinions, interpretations, and evaluations to their source (e.g. "the assistant argued that…", "the user said…").
4. Never speak in second person ("you are…", "you deserve…"). Always use third-person or source-attributed form.
5. Do NOT add conclusions, reassurance, or analysis not present in the excerpts.
6. If multiple versions of a fact exist, state the most recent version and note earlier ones briefly.
7. Use dense, bullet‑point style. No narrative framing.

The output will be used as ground truth for retrieval evaluation, so factual completeness and neutrality are critical.

Conversation excerpts:
{context}

User question: {question}

Ground‑truth answer:"""



# ---------------------------------------------------------------------------
# Judge call (native API, reasoning ON for better quality)
# ---------------------------------------------------------------------------
async def call_generate(session, prompt_text, api_sem, max_tokens=2048):
    messages = [
        {"role": "system", "content": "You are a meticulous assistant. Provide thorough, accurate answers based only on the given context."},
        {"role": "user", "content": prompt_text}
    ]
    payload = {
        "model": "mattbucci/gemma-4-12B-AWQ",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False
    }
    async with api_sem:
        for attempt in range(2):
            try:
                async with session.post(SGLANG_URL, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"SGLang error {resp.status}: {text}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    if not content and attempt == 0:
                        await asyncio.sleep(3)
                        continue
                    return content if content else None
            except Exception as e:
                if attempt == 0: await asyncio.sleep(3)
                else: print(f"    Generate error: {e}"); return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    contexts = load_json(VECTOR_CONTEXTS)
    processed = load_progress()

    conn = aiohttp.TCPConnector(limit=API_CONCURRENCY * 2)
    api_sem = asyncio.Semaphore(API_CONCURRENCY)

    async with aiohttp.ClientSession(connector=conn) as session:
        probes_to_process = []
        for cid, cprobes in contexts.items():
            for pid, data in cprobes.items():
                key = f"{cid}|{pid}"
                if key not in processed:
                    probes_to_process.append((cid, pid, data))

        pbar = tqdm(total=len(probes_to_process), desc="Ground‑truth generation")

        async def generate_one(cid, pid, data):
            # Build context text from retrieved turns
            context_text = "\n\n".join(t["text"] for t in data["retrieved_turns"])
            prompt = GROUND_TRUTH_PROMPT.format(context=context_text, question=data["question"])
            answer = await call_generate(session, prompt, api_sem, max_tokens=MAX_TOKENS)
            if answer:
                # Locate curation file and update expected_answer
                curation_file = os.path.join(CURATION_DIR, f"{cid}.json")
                if os.path.exists(curation_file):
                    curation = load_json(curation_file)
                    for probe in curation.get("evaluation_probes", []):
                        if probe["probe_id"] == pid:
                            probe["expected_answer"] = answer
                            break
                    save_json(curation, curation_file)
                save_progress(cid, pid)
            else:
                print(f"\n  Failed to generate for {cid}/{pid}")
            pbar.update(1)

        # Process concurrently with limited concurrency
        sem_probes = asyncio.Semaphore(ENTRY_CONCURRENCY)
        async def bounded(cid, pid, data):
            async with sem_probes:
                await generate_one(cid, pid, data)

        tasks = [bounded(cid, pid, data) for cid, pid, data in probes_to_process]
        await asyncio.gather(*tasks)
        pbar.close()

    print("Ground‑truth generation complete. All curation files updated.")

if __name__ == "__main__":
    asyncio.run(main())