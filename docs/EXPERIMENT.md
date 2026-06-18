Here’s the complete, step‑by‑step plan — from cleaning the database through running the full LSREP pipeline. Every command is listed. No steps are skipped.

---

## 0. Services you need running

| Service | How to start | Needed when |
|---------|-------------|-------------|
| PostgreSQL + Redis | `docker compose -f docker/docker-compose.yml up -d` | Always (database) |
| Ollama | Already running on your machine | Always (inference & background workers) |
| vLLM‑bg | **Do NOT start** | Not needed; we run everything in shared mode |
| Celery worker | **Do NOT start** | Not needed; we run background tasks synchronously in‑process |
| ICE proxy | **Do NOT start** | Phase 2 calls the orchestrator directly, no proxy |

**Set shared mode** – add this to your `.env` (or change it if already present):

```
BACKGROUND_MODEL_MODE=shared
```

---

## 1. Preprocessing – fix Flaw timestamps (once)

The Flaw conversation (`bb558b5f-…`) currently has timestamps starting in 2024. Shift them to 2025 so decay can differentiate old vs. recent.

```bash
uv run python experiments/fix_flaw_timestamps.py
```
```python
#!/usr/bin/env python3
"""Shift the Flaw conversation (bb558b5f-5365-5bac-9ed0-07219025b5f2) to start in 2025."""

import json, re
from datetime import datetime, timezone, timedelta

INPUT = "data/simulation/simulation_full.jsonl"
OUTPUT = "data/simulation/simulation_full.jsonl"
FLAW_CID = "bb558b5f-5365-5bac-9ed0-07219025b5f2"

def parse_timestamp(ts_str: str) -> datetime:
    """
    Parse any ISO‑8601 timestamp string that may have:
      - trailing 'Z'
      - a numeric offset like +00:00 or -05:00
      - milliseconds or microseconds
    Returns a timezone‑aware datetime.
    """
    ts = ts_str.strip()
    # Remove trailing 'Z' → replace with +00:00 only if no offset already present
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    # If the string already contains an offset (e.g., ends with +HH:MM or -HH:MM),
    # Python 3.11's fromisoformat can handle it.
    return datetime.fromisoformat(ts)

def main():
    # Read all turns
    with open(INPUT, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Find Flaw turns
    flaw_turns = [t for t in lines if t.get("conversation_id") == FLAW_CID]
    if not flaw_turns:
        print("No Flaw turns found.")
        return

    # Parse the earliest Flaw timestamp
    first_ts = parse_timestamp(flaw_turns[0]["timestamp"])
    # Target: January 1, 2025, same time of day as the original first turn
    new_start = first_ts.replace(year=2025, month=1, day=1)
    delta = new_start - first_ts

    # Shift all Flaw turns by the same delta
    for turn in lines:
        if turn.get("conversation_id") == FLAW_CID:
            ts = parse_timestamp(turn["timestamp"])
            new_ts = ts + delta
            turn["timestamp"] = new_ts.isoformat()

    # Write back
    with open(OUTPUT, "w") as f:
        for turn in lines:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    print(f"Updated Flaw timestamps. File saved to {OUTPUT}")

if __name__ == "__main__":
    main()

```
After this, `data/simulation/simulation_full.jsonl` is ready.

---

## 2. Re‑create the merged simulation input (only if you don’t have it already)

If you still have the three raw JSONL files (`gpt.jsonl`, `claude.jsonl`, `deepseek.jsonl`), you don’t need to re‑extract. The merge script should already exist.

```bash
# Only run these if simulation_full.jsonl doesn’t exist or is outdated
uv run python scripts/data/extract_gpt.py
uv run python scripts/data/extract_claude.py
uv run python scripts/data/extract_deepseek.py
uv run python scripts/data/merge.py
```

Then run the Flaw timestamp fix again to be safe.

---

## 3. Truncate the database (start fresh)

```bash
uv run python -c "
from src.api.db import SessionLocal
from sqlalchemy import text
db = SessionLocal()
db.execute(text('TRUNCATE episodic_memory, conversations, codex_entities, codex_edges, codex_events, codex_snapshots, procedural_memory, session_summaries, context_clusters, sentinel_events, cold_storage, idempotency_keys, rag_documents, rag_chunks RESTART IDENTITY CASCADE'))
db.commit()
db.close()
print('Truncated.')
"
```

---

## 4. Model pool – pull & register non‑overlapping MoE models

We’ll use a set of models where **every label is assigned to exactly one model** (no overlap). Labels not explicitly assigned will fall back to the generalist.

### 4.1 Model pool definition

| Model (Ollama name) | Size | Role | Topic Tags | Intent Tags |
|---------------------|------|------|-----------|-------------|
| `gemma4:26b-a4b-it-q4_K_M` | 26 B | **Generalist** (default) | `General_Reference_&_Trivia` | `Factual_Retrieval`, `Casual_Banter` |
| `qwen3-coder:30b-a3b-q4_K_M` | 30 B MoE | **Coding & STEM** | `Software_&_Tech`, `STEM_&_Academics` | `Generation`, `Troubleshooting` |
| `qwen2.5:32b-instruct-q4_K_M` | 32 B | **Business & Planning** | `Business_&_Finance`, `Admin_&_Productivity` | `Strategic_Planning`, `Decision_Making`, `Analysis_&_Summarization` |
| `llama3:8b` | 8 B | **Creative & Emotional** | `Creative_&_Media`, `Social_&_Relationships`, `Lifestyle_&_Health` | `Ideation`, `Open_Exploration`, `Emotional_Processing` |
| `qwen2.5:7b` | 7 B | **Utility & Meta** | `Meta_AI`, `World_&_Current_Events` | `Utility_Formatting` |
| `tinyllama:latest` | 1.1 B | **Null / Trivial** | `Null_Noise` | (none) |

- The topic labels `Software_&_Tech`, `STEM_&_Academics` are **only** on the coder; `Business_&_Finance`, `Admin_&_Productivity` only on the planner; `Creative_&_Media`, `Social_&_Relationships`, `Lifestyle_&_Health` only on the creative model; `Meta_AI`, `World_&_Current_Events` only on the utility model; `Null_Noise` only on tinyllama. `General_Reference_&_Trivia` only on gemma.
- The intent labels are similarly partitioned with no overlap.

### 4.2 Pull the models

```bash
ollama pull gemma4:26b-a4b-it-q4_K_M
ollama pull qwen3-coder:30b-a3b-q4_K_M
ollama pull qwen2.5:32b-instruct-q4_K_M
ollama pull llama3:8b
ollama pull qwen2.5:7b
ollama pull tinyllama:latest
```

(If any of these are already pulled, Ollama will just verify.)

### 4.3 Populate the ICE model registry

First, refresh the registry from Ollama:

```bash
curl -X POST http://localhost:8000/user-control/model-registry/refresh
```

Then confirm and assign tags to each model **exactly as shown above** (the tags will be used by the Mini‑MoE routing in Phase 2). Use the following `curl` commands:

```bash
# Generalist
curl -X PUT http://localhost:8000/user-control/model-registry/gemma4:26b-a4b-it-q4_K_M \
  -H "Content-Type: application/json" \
  -d '{"confirmed":true, "topic_tags":["General_Reference_&_Trivia"], "intent_tags":["Factual_Retrieval","Casual_Banter"]}'

# Coder
curl -X PUT http://localhost:8000/user-control/model-registry/qwen3-coder:30b-a3b-q4_K_M \
  -H "Content-Type: application/json" \
  -d '{"confirmed":true, "topic_tags":["Software_&_Tech","STEM_&_Academics"], "intent_tags":["Generation","Troubleshooting"]}'

# Planner
curl -X PUT http://localhost:8000/user-control/model-registry/qwen2.5:32b-instruct-q4_K_M \
  -H "Content-Type: application/json" \
  -d '{"confirmed":true, "topic_tags":["Business_&_Finance","Admin_&_Productivity"], "intent_tags":["Strategic_Planning","Decision_Making","Analysis_&_Summarization"]}'

# Creative
curl -X PUT http://localhost:8000/user-control/model-registry/llama3:8b \
  -H "Content-Type: application/json" \
  -d '{"confirmed":true, "topic_tags":["Creative_&_Media","Social_&_Relationships","Lifestyle_&_Health"], "intent_tags":["Ideation","Open_Exploration","Emotional_Processing"]}'

# Utility
curl -X PUT http://localhost:8000/user-control/model-registry/qwen2.5:7b \
  -H "Content-Type: application/json" \
  -d '{"confirmed":true, "topic_tags":["Meta_AI","World_&_Current_Events"], "intent_tags":["Utility_Formatting"]}'

# Trivial
curl -X PUT http://localhost:8000/user-control/model-registry/tinyllama:latest \
  -H "Content-Type: application/json" \
  -d '{"confirmed":true, "topic_tags":["Null_Noise"], "intent_tags":[]}'
```

Now the registry is fully populated and non‑overlapping.

---

## 5. Phase 1 – generate curation files (no simulation)

Delete any old curation files:

```bash
rm -rf experiments/curation_files
```
*Updated Phase 1 script:** `experiments/phase1_generate_curation_files.py`

```python
#!/usr/bin/env python3
"""
Phase 1: Generate curation files with empty probe slots.
Reads simulation_full.jsonl, splits conversations at random points,
writes one JSON file per checkpoint. Shows ALL history turns (not just last N).
No simulation or database interaction.
"""

import json, os, sys, random
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SIMULATION_INPUT = "data/simulation/simulation_full.jsonl"
CURATION_DIR = "experiments/curation_files"
SPLITS_PER_CONV = 3
FUTURE_WINDOW = 10          # how many future turns to show in the reference block
SEED = 42

def main():
    random.seed(SEED)
    os.makedirs(CURATION_DIR, exist_ok=True)

    with open(SIMULATION_INPUT, "r") as f:
        all_turns = [json.loads(line) for line in f if line.strip()]

    convs = defaultdict(list)
    for turn in all_turns:
        cid = turn.get("conversation_id", "unknown")
        convs[cid].append(turn)

    for cid, turns in convs.items():
        L = len(turns)
        if L < 10:
            continue

        min_idx = max(1, int(L * 0.25))
        max_idx = min(L - 1, int(L * 0.75))
        possible = list(range(min_idx, max_idx + 1))
        if len(possible) < SPLITS_PER_CONV:
            split_indices = possible[:SPLITS_PER_CONV] if possible else [int(L * 0.5)]
        else:
            split_indices = sorted(random.sample(possible, SPLITS_PER_CONV))

        for split_n in split_indices:
            history = turns[:split_n]
            future = turns[split_n:split_n + FUTURE_WINDOW]

            # Build FULL history block (every turn before the split)
            history_block = []
            for i, t in enumerate(history, start=1):
                history_block.append({
                    "turn_number": i,
                    "user_input": t["prompt"][:300],      # truncated for readability
                    "ai_response": t["response"][:300],
                })

            future_block = []
            for i, t in enumerate(future, start=split_n + 1):
                future_block.append({
                    "turn_number": i,
                    "user_input": t["prompt"][:300],
                    "ai_response": t["response"][:300],
                })

            probes = []
            for i in range(8):
                probes.append({
                    "probe_id": f"P-{i+1:02d}",
                    "probe_type": "ENTER_TYPE",
                    "user_injected_prompt": "ENTER_PROBE_HERE",
                    "expected_answer": "ENTER_EXPECTED_ANSWER_OR_BLANK",
                    "ground_truth_expected_fragments": []
                })

            checkpoint_id = f"EC-{cid[:8]}-TURN{split_n}"
            curation = {
                "evaluation_checkpoint_id": checkpoint_id,
                "original_conversation_id": cid,
                "simulated_present_timestamp": history[-1]["timestamp"] if history else "",
                "split_turn_index": split_n,
                "total_turns": L,
                "historical_context_block": history_block,
                "future_reference_block": future_block,
                "evaluation_probes": probes
            }

            out_path = os.path.join(CURATION_DIR, f"{checkpoint_id}.json")
            with open(out_path, "w") as f:
                json.dump(curation, f, indent=2)

    print(f"Phase 1 complete. Curation files saved to {CURATION_DIR}")

if __name__ == "__main__":
    main()
```
Run the Phase 1 script (the updated version that shows **all** history turns):

```bash
uv run python experiments/phase1_generate_curation_files.py
```

This creates JSON files in `experiments/curation_files/`. Each file contains:

- The full history (every turn up to the split).
- The next 10 future turns (so you can see what actually happened next).
- 8 empty probe slots.

### 5.1 Manual editing

Open each JSON file and fill in the `"user_injected_prompt"` for the probes you want to test. Also fill `"probe_type"` (e.g., `"recent_recall"`, `"deep_historical_recall"`, `"procedural_format"`) and optionally `"expected_answer"`.  
Delete any unused probe slots (remove the entire object from the `evaluation_probes` array). Save the files.

---

## 6. Phase 2 – automatic matrix execution

Make sure PostgreSQL and Ollama are running.  
No vLLM‑bg, no Celery, no proxy.

### 6.1 Start (or resume) the evaluation

**Updated Phase 2 script:** `experiments/phase2_run_evaluation_matrix.py`

```python
#!/usr/bin/env python3
"""
Phase 2: Reads curated probe files, reconstructs state, runs the 6‑condition matrix
for each probe, captures all outputs, and runs an AI judge.
Supports progress bars and checkpoint resume.
"""

import json, os, sys, time, uuid, glob, random
from datetime import datetime, timezone
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, numpy as np
from sqlalchemy import text
from pgvector.sqlalchemy import Vector as PgVector

from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, MemorySlot
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers.post_flight import is_lossless, generate_summary
from src.workers.codex_extractor import extract_triplets, handle_triplet
from src.workers.reflection import run_reflection
from src.workers.decay import apply_decay
from src.workers.clustering import cluster_turns
from src.workers.sentinel_monitor import monitor_sentinels
from src.workers.codex_decay import decay_codex_edges
from src.workers.procedural_decay import decay_procedural_patterns
from src.api.prompt_assembler import assemble_prompt
from src.model_registry.registry import find_best_model

CURATION_DIR = "experiments/curation_files"
RESULTS_DIR = "experiments/results_phase2"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
SINGLE_MODEL = "gemma4:26b-a4b-it-q4_K_M"
JUDGE_MODEL = "gemma4:26b-a4b-it-q4_K_M"
SEED = 42
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "_completed.txt")

def load_completed():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_completed(checkpoint_id, probe_id):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{checkpoint_id}|{probe_id}\n")

def replay_simulation(conv_id, history_turns_data, classifier, embedder):
    db = SessionLocal()
    conv = Conversation(id=uuid.UUID(conv_id), memory_scope_type="auto")
    db.add(conv)
    db.flush()
    for entry in history_turns_data:
        prompt = entry["prompt"]
        response = entry.get("response", "")
        ts_str = entry.get("timestamp")
        result = classifier.classify(prompt)
        result.prompt = prompt
        emb = embedder.encode(prompt, convert_to_tensor=False).tolist()
        batch_id = uuid.uuid4()
        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        turn = EpisodicMemory(
            conversation_id=conv.id,
            batch_id=batch_id,
            timestamp=timestamp,
            topic_tags=result.topic_tags,
            intent_tags=result.intent_tags,
            context_reliance=result.context_reliance,
            raw_text=f"User: {prompt}\n\nAssistant: {response}",
            embedding=emb,
            idempotency_key=str(uuid.uuid4()),
        )
        db.add(turn)
        db.flush()
        lossless = is_lossless(response)
        full_text = f"User: {prompt}\nAssistant: {response}"
        word_count = len(full_text.split())
        has_code = "```" in response
        inject_raw = True
        if lossless and word_count > 500 and not has_code:
            summary = generate_summary(prompt, response)
            turn.summary_text = summary
            inject_raw = False
        elif not lossless:
            summary = generate_summary(prompt, response)
            turn.summary_text = summary
            inject_raw = False
        turn.lossless_flag = lossless
        turn.inject_raw = inject_raw
        if lossless:
            triplets = extract_triplets(turn.raw_text)
            for t in triplets:
                if isinstance(t, dict) and "subject" in t and "relation" in t and "object" in t:
                    s, r, o = t["subject"].strip(), t["relation"].strip(), t["object"].strip()
                    if s and r and o:
                        handle_triplet(db, s, r, o, str(batch_id))
        db.commit()
    db.close()

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed = load_completed()
    client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")

    classifier = PyTorchClassifier(
        model_path="models/classifier/ice_classifier_v2_final.pt",
        schema_path="data/labeled/label_schema.json",
    )
    embedder = classifier.embedder

    curation_files = sorted(glob.glob(os.path.join(CURATION_DIR, "*.json")))
    if not curation_files:
        print("No curation files found.")
        return

    all_results = []

    # Outer progress bar over curation files
    for cf_path in tqdm(curation_files, desc="Checkpoints"):
        with open(cf_path, "r") as f:
            curation = json.load(f)

        conv_id = curation["original_conversation_id"]
        split_n = curation["split_turn_index"]

        # Load the full conversation from simulation_full.jsonl
        with open("data/simulation/simulation_full.jsonl", "r") as f:
            all_turns = [json.loads(line) for line in f if line.strip()]
        conv_turns = [t for t in all_turns if t.get("conversation_id") == conv_id]
        history_turns_data = conv_turns[:split_n]

        # Truncate DB and replay
        db = SessionLocal()
        db.execute(text("TRUNCATE episodic_memory, conversations, codex_entities, codex_edges, codex_events, codex_snapshots, procedural_memory, session_summaries, context_clusters, sentinel_events, cold_storage, idempotency_keys, rag_documents, rag_chunks RESTART IDENTITY CASCADE"))
        db.commit()
        db.close()

        replay_simulation(conv_id, history_turns_data, classifier, embedder)

        # Run background workers to maturity
        apply_decay()
        cluster_turns()
        run_reflection()
        monitor_sentinels()
        decay_codex_edges()
        decay_procedural_patterns()

        # Probes
        probes = [p for p in curation.get("evaluation_probes", [])
                  if p.get("user_injected_prompt") and p["user_injected_prompt"] != "ENTER_PROBE_HERE"]

        for probe in probes:
            probe_id = probe["probe_id"]
            key = f"{curation['evaluation_checkpoint_id']}|{probe_id}"
            if key in completed:
                continue   # already processed

            prompt = probe["user_injected_prompt"]
            expected = probe.get("expected_answer", "")

            # ADD THIS: Wipe out the placeholder if you forgot to delete it!
            if expected == "ENTER_EXPECTED_ANSWER_OR_BLANK":
                expected = ""

            classification = classifier.classify(prompt)
            embedding = embedder.encode(prompt, convert_to_tensor=False).tolist()

            # ---- Retrieve once for full ICE ----
            orchestrator = HybridRetrievalOrchestrator(SessionLocal(), embedder)
            scope = {"conversation_id": conv_id} if conv_id else None
            full_ice_fragments = orchestrator.retrieve(
                classification=classification,
                conversation_id=conv_id,
                prompt_embedding=embedding,
                scope=scope,
            )

            memory_slots = SessionLocal().query(MemorySlot).filter_by(is_active=True).all()
            full_ice_prompt_payload = assemble_prompt(
                memory_slots=memory_slots,
                retrieved_fragments=full_ice_fragments,
                user_message=prompt,
                db_session=SessionLocal(),
                conversation_id=conv_id,
            )

            # ---- Naive context (raw history) ----
            naive_context_raw = "\n\n".join(
                [f"User: {t['prompt']}\nAssistant: {t['response']}" for t in history_turns_data]
            )

            # ---- Vector-only context (no decay/graph/HyDE) ----
            db_vec = SessionLocal()
            q = text("""
                SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id, is_bookmarked,
                       (1 - (embedding <=> :prompt_embedding)) * COALESCE(decay_score, 1.0) as score
                FROM episodic_memory
                WHERE embedding IS NOT NULL AND is_archived = false
                ORDER BY score DESC LIMIT 10
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = db_vec.execute(q, {"prompt_embedding": embedding}).fetchall()
            vector_fragments = []
            for r in rows:
                text = r.raw_text if r.lossless_flag else (r.summary_text or r.raw_text[:300])
                words = text.split()
                if len(words) > 500:
                    text = " ".join(words[:500]) + "…"
                vector_fragments.append(text)
            db_vec.close()

            # ---- 6 conditions ----
            conditions = {
                "control_baseline_generalist": ("naive", SINGLE_MODEL),
                "control_moe": ("naive", "moe"),
                "vector_rag_baseline_generalist": ("vector_only", SINGLE_MODEL),
                "vector_rag_moe": ("vector_only", "moe"),
                "full_ice_generalist": ("full_ice", SINGLE_MODEL),
                "full_ice_moe": ("full_ice", "moe"),
            }

            record_for_probe = {}
            for cond_name, (retrieval_mode, model_choice) in conditions.items():
                # Choose model
                if model_choice == "moe":
                    best_model, _ = find_best_model(classification.topic_tags, classification.intent_tags)
                    model_to_use = best_model if best_model else SINGLE_MODEL
                else:
                    model_to_use = SINGLE_MODEL

                # Build final messages
                if retrieval_mode == "naive":
                    system_content = naive_context_raw
                    final_messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ]
                    # crude token trim
                    total_words = len(system_content.split()) + len(prompt.split())
                    if total_words > 6000:
                        keep = 6000 - len(prompt.split())
                        words = system_content.split()
                        system_content = " ".join(words[-keep:])
                        final_messages[0]["content"] = system_content
                elif retrieval_mode == "vector_only":
                    final_messages = [
                        {"role": "system", "content": "\n\n".join(vector_fragments)},
                        {"role": "user", "content": prompt}
                    ]
                else:   # full_ice
                    final_messages = full_ice_prompt_payload

                # Call LLM
                start = time.time()
                try:
                    resp = client.chat.completions.create(
                        model=model_to_use,
                        messages=final_messages,
                        temperature=0.0,
                        max_tokens=500,
                        timeout=120.0,
                    )
                    answer = resp.choices[0].message.content.strip()
                    latency = time.time() - start
                except Exception as e:
                    answer = f"ERROR: {str(e)}"
                    latency = -1

                # AI judge
                judge_score = None
                if expected:
                    try:
                        jresp = client.chat.completions.create(
                            model=JUDGE_MODEL,
                            messages=[
                                {"role": "user", "content": (
                                    f"Question: {prompt}\n"
                                    f"Expected answer: {expected}\n"
                                    f"Actual answer: {answer}\n"
                                    "Rate the relevance of the actual answer to the expected answer on a scale of 1 to 5. Output only the number."
                                )}
                            ],
                            temperature=0.0, max_tokens=3, timeout=30.0,
                        )
                        judge_score = int(jresp.choices[0].message.content.strip())
                    except:
                        judge_score = -1

                record_for_probe[cond_name] = {
                    "model_used": model_to_use,
                    "answer": answer,
                    "latency_seconds": round(latency, 3),
                    "judge_score": judge_score,
                }

            # Store result
            result = {
                "metadata": {
                    "checkpoint_id": curation["evaluation_checkpoint_id"],
                    "probe_id": probe_id,
                    "probe_type": probe.get("probe_type", ""),
                    "raw_user_probe": prompt,
                },
                "execution_permutations": record_for_probe
            }
            all_results.append(result)
            save_completed(curation["evaluation_checkpoint_id"], probe_id)

            # Save incremental results after each probe (safety)
            master = {
                "experiment_session_timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluation_run_results": all_results,
            }
            with open(os.path.join(RESULTS_DIR, "master_results.json"), "w") as f:
                json.dump(master, f, indent=2)

    print(f"Phase 2 complete. Results saved to {RESULTS_DIR}/master_results.json")

if __name__ == "__main__":
    main()
```

```bash
uv run python experiments/phase2_run_evaluation_matrix.py
```

- A progress bar will show each checkpoint as it’s processed.
- Results are written to `experiments/results_phase2/master_results.json`.
- If you interrupt the script (Ctrl + C), simply run it again – it will resume from where it left off (it reads `_completed.txt`).

### 6.2 After completion

You’ll have a single JSON file with every condition, every answer, and every judge score, ready for analysis.

---

## 7. Summary of the entire workflow

1. **Services** → PostgreSQL + Redis (Docker), Ollama (already running).  
2. **Fix timestamps** → `experiments/fix_flaw_timestamps.py`  
3. **(Optional) Re‑merge** → extraction & merge scripts if needed.  
4. **Truncate DB** → one‑liner above.  
5. **Pull & register models** → step 4.2 & 4.3.  
6. **Phase 1** → `phase1_generate_curation_files.py`  
7. **Manual probe entry** → edit JSON files.  
8. **Phase 2** → `phase2_run_evaluation_matrix.py`  
9. **Analyse** → read `master_results.json`

The entire evaluation now respects conversation state, memory lifecycle, scoping, sliding window, and dynamic routing – exactly as ICE is designed to be used.


1. Blind‑judge scoring (automatic, required)

What: Run experiments/blind_judge.py on master_results.json to fill every judge_score field with a 1‑5 rating.
Why: You didn’t write expected answers, so this is the only way to get quantitative scores.
Status: Script ready to be written; run it once after Phase 2 finishes.
2. Longitudinal improvement analysis (already happening)

What: The same probes across splits of the same conversation (e.g., P‑01 at TURN30, TURN65, TURN87) will have different judge scores.
How: After blind‑judge scoring, group master_results.json by probe_id and plot the average judge score over split_turn_index.
Why: Proves ICE gets better the longer it’s used (G9).
Status: Data is being collected right now; just needs a small analysis script.
3. Fine‑tuning the classifier on your probes

What: Label all your probes (topic, intent, context‑reliance) using the Gemma 26B model, spot‑check ~30, insert into curated_labels, run fine‑tuning, then re‑run Phase 2 on a subset of checkpoints (e.g., 10‑15).
Why: Shows the classifier adapts to your personal language, reducing Zero‑Shot gating and improving MoE routing.
Status: Infrastructure ready; needs the labeling step and one additional Phase 2 run.
4. Scope‑mode comparison (Auto vs. Project vs. None)

What: Re‑run a subset of probes under three different scope settings.
Why: Proves conversation scoping improves retrieval precision by reducing cross‑topic noise.
Status: Phase 2 probe loop can be easily extended with a scope toggle; standalone experiment.
5. Truth‑quorum ablation (Codex accuracy)

What: Modify the Codex Extractor to promote all edges to active immediately (skip the quorum), replay a conversation, and manually sample triplets for factual correctness. Compare with the standard quorum‑enabled run.
Why: Proves the truth quorum reduces hallucination in the knowledge graph.
Status: Requires a small Codex‑extractor patch and a manual accuracy check on ~50 triplets.
6. HyDE ablation (Full ICE with vs. without HyDE)

What: Set orchestrator._hyde_rewrite = lambda *a: None before calling retrieve() for a condition, and compare judge scores.
Why: Measures HyDE’s contribution to retrieval quality, especially for vague prompts.
Status: Can be added as a new condition in the Phase 2 matrix; minor change.
7. Sliding‑window ablation (Full ICE with vs. without)

What: In the full_ice condition, pass conversation_id=None to assemble_prompt to suppress the sliding window, then compare with the full version.
Why: Shows the sliding window’s value for recent‑context prompts.
Status: Same as above – new condition in the probe loop.
8. Procedural memory toggle

What: Temporarily set PROCEDURAL_ALWAYS_INJECT = False (or skip the procedural leg) and compare scores.
Why: Measures whether procedural memory actually helps or adds noise.
Status: Trivial to add as a new condition.
9. Bookmark boost verification (qualitative)

What: Manually bookmark a few key turns, then verify they appear in the top‑3 of Full ICE for related probes.
Why: Proves bookmarked memories are prioritised.
Status: Can be done manually with a few curl commands; no script needed.
10. Decay‑reinforcement demonstration (quantitative)

What: Run a small experiment on one conversation:

    Replay history, run apply_decay() once, then run a batch of probes.

    Run apply_decay() multiple times (simulating weeks), then run the same probes again.
    Why: Shows that turns with higher access_count decay slower and remain retrievable longer.
    Status: Script already exists; just needs to be called in a loop before probing.

11. Codex graph growth / case study (qualitative)

What: After the incremental run, query how many entities/edges were created for the Flaw conversation at each split.
Why: Demonstrates the knowledge graph accumulates valid facts over time.
Status: Simple SQL queries after the run.
