#!/usr/bin/env python3
"""
ICE-Mature Experiment Runner
=============================
Replays four long conversations with the full ICE‑Mature stack,
pauses at randomly‑spaced checkpoints, runs background workers
synchronously (simulating realistic decay), and collects
six‑condition probe responses with full metadata.

Resumable: saves completed probes line‑by‑line; Ctrl‑C safe.
"""

from typing import List
import json, os, sys, uuid, time, random, re
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy import bindparam
import torch
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, MemorySlot
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.prompt_assembler import assemble_prompt
from src.model_registry.registry import find_best_model, get_fallback_model

# Background workers (imported for synchronous execution)
from src.workers.decay import apply_decay
from src.workers.reflection import run_reflection
from src.workers.clustering import cluster_turns, merge_similar_clusters
from src.workers.sentinel_monitor import monitor_sentinels
from src.workers.codex_decay import decay_codex_edges
from src.workers.procedural_decay import decay_procedural_patterns

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SEED = 42
OLLAMA_URL = "http://localhost:11434/v1"
SINGLE_MODEL = "gemma4:26b-a4b-it-q4_K_M"
MATURE_DIR = Path(__file__).parent
RESULTS_DIR = MATURE_DIR / "results"
MASTER_RESULTS_FILE = RESULTS_DIR / "master_results.json"
FRAGMENTS_FILE = RESULTS_DIR / "fragments.jsonl"
COMPLETED_FILE = MATURE_DIR / "_completed.txt"
LAST_TURN_FILE = MATURE_DIR / "_last_turn.txt"
LOG_FILE = MATURE_DIR / "run.log"

# Load generated probes
GENERATED_PROBES_FILE = MATURE_DIR / "generated_probes.json"
generated_probes = {}
if GENERATED_PROBES_FILE.exists():
    with open(GENERATED_PROBES_FILE) as f:
        generated_probes = json.load(f)
    print(f"📂 Loaded generated probes for {len(generated_probes)} conversations.")

# ── Load corrected ground truths, produced by correct_ground_truths.py ──
# Structure on disk: corrected[cid][probe_id][str(checkpoint_turn)] -> answer text
# Previously this file was NEVER read anywhere in this script — every probe's
# ground_truth was pulled straight from the ORIGINAL, static expected_answer
# in generated_probes.json regardless of checkpoint, which made all forensic
# ground-truth correction work have zero effect on the actual experiment.
CORRECTED_GT_FILE = RESULTS_DIR / "corrected_ground_truths.json"
corrected_ground_truths = {}
if CORRECTED_GT_FILE.exists():
    with open(CORRECTED_GT_FILE) as f:
        corrected_ground_truths = json.load(f)
    print(f"📂 Loaded corrected ground truths for {len(corrected_ground_truths)} conversations.")
else:
    print("⚠️  No corrected_ground_truths.json found — falling back to static "
          "expected_answer from generated_probes.json for all checkpoints. "
          "Run experiments/mature/correct_ground_truths.py first for "
          "checkpoint-accurate ground truth.")


def get_ground_truth_for_checkpoint(cid: str, probe: dict, checkpoint_turn: int) -> str:
    """Resolve the correct ground truth for a probe AT A SPECIFIC CHECKPOINT.

    Falls back through three tiers, from most to least accurate:
      1. The exact checkpoint's corrected ground truth, if present.
      2. The LATEST corrected ground truth at or before this checkpoint
         (a probe's forward-pass walk in correct_ground_truths.py only writes
         entries from its origin_split onward — earlier checkpoints for the
         same probe_id simply don't exist yet, so this finds the closest
         available corrected answer rather than skipping straight to raw).
      3. The static expected_answer from generated_probes.json (legacy
         behavior, used only if no corrected data exists at all for this
         probe — e.g. correct_ground_truths.py hasn't been run yet).
    """
    pid = probe["probe_id"]
    per_probe = corrected_ground_truths.get(cid, {}).get(pid, {})

    if not per_probe:
        return probe.get("expected_answer", "")

    exact = per_probe.get(str(checkpoint_turn))
    if exact is not None:
        return exact

    # No exact match – fall back to the latest checkpoint ≤ this one
    candidates = [int(k) for k in per_probe.keys() if int(k) <= checkpoint_turn]
    if candidates:
        return per_probe[str(max(candidates))]

    # ── Flaw‑specific nearest‑neighbour fallback ──
    # Flaw's corrected truths were generated with an older, different set of
    # checkpoints.  When the experiment uses the new split from generated_probes.json
    # and no exact corrected entry exists, we pick the closest available checkpoint
    # within ±30 turns instead of falling back to the raw expected_answer.
    # ── Flaw‑specific nearest‑neighbour fallback ──
    # Flaw's corrected truths were generated with an older, different set of
    # checkpoints.  When the experiment uses the new split and no exact corrected
    # entry exists, the closest old checkpoint's truth is ALWAYS better than the
    # uncorrected static expected_answer, so we use it unconditionally.
    FLAW_ID = "bb558b5f-5365-5bac-9ed0-07219025b5f2"
    if cid == FLAW_ID:
        all_cps = [int(k) for k in per_probe.keys()]
        if all_cps:
            closest = min(all_cps, key=lambda x: abs(x - checkpoint_turn))
            return per_probe[str(closest)]

    # Final fallback – raw answer from generated_probes.json
    return probe.get("expected_answer", "")


CURATION_DIR = Path("experiments/curation_files")
SIMULATION_INPUT = Path("data/simulation/simulation_full.jsonl")

# The four target conversations and their curation file checkpoints
TARGET_CONVERSATIONS = [
    {
        "conversation_id": "633e26f8-5889-5c21-8c70-f4d7ab22cb00",
        "curation_checkpoint": "EC-633e26f8-TURN275",
        "label": "Shinchan"
    },
    {
        "conversation_id": "bb558b5f-5365-5bac-9ed0-07219025b5f2",
        "curation_checkpoint": "EC-961862eb-FULL",
        "label": "Flaw"
    },
    {
        "conversation_id": "a77c15cf-2078-4279-aeaa-8c3a6d58a972",
        "curation_checkpoint": "EC-a77c15cf-TURN238",
        "label": "ICE-Dev"
    },
    {
        "conversation_id": "ecc64aab-1979-5586-b0d8-c53448c0882e",
        "curation_checkpoint": "EC-ecc64aab-TURN145",
        "label": "Masters"
    },
]

# Six retrieval conditions
CONDITIONS = [
    "vector_rag_baseline_generalist",
    "full_ice_generalist",
    "vector_rag_moe",
    "full_ice_moe",
]
# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_completed():
    """Return a set of (conversation_id, probe_id, checkpoint_id) tuples."""
    if not COMPLETED_FILE.exists():
        return set()
    completed = set()
    with open(COMPLETED_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("|")
                if len(parts) == 3:
                    completed.add((parts[0], parts[1], parts[2]))
    return completed

def save_completed(conversation_id, probe_id, checkpoint_id):
    """Append a completed probe line to the file (crash‑safe)."""
    with open(COMPLETED_FILE, "a") as f:
        f.write(f"{conversation_id}|{probe_id}|{checkpoint_id}\n")
        f.flush()
        os.fsync(f.fileno())

def load_last_turns():
    if not LAST_TURN_FILE.exists():
        return {}
    turns = {}
    with open(LAST_TURN_FILE, "r") as f:
        for line in f:
            cid, turn = line.strip().split("|")
            turns[cid] = int(turn)
    return turns

def save_last_turn(conversation_id, turn_index):
    turns = load_last_turns()
    turns[conversation_id] = turn_index
    tmp = LAST_TURN_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for c, t in turns.items():
            f.write(f"{c}|{t}\n")
    os.replace(tmp, LAST_TURN_FILE)

def estimate_tokens(text):
    return int(len(text.split()) * 1.33)

def save_fragments(fragments_list, conv_id, probe_id, checkpoint_id, cond_name):
    """Save fragment texts to fragments.jsonl and return a list of fragment IDs."""
    fragment_ids = []
    for frag in fragments_list:
        fid = str(uuid.uuid4())[:8]
        append_jsonl(FRAGMENTS_FILE, {
            "fragment_id": fid,
            "conversation_id": conv_id,
            "probe_id": probe_id,
            "checkpoint_id": checkpoint_id,
            "condition": cond_name,
            "text": frag.get("text", ""),
            "source_type": frag.get("source_type", frag.get("source", "vector")),
            "score": frag.get("score", None)
        })
        fragment_ids.append(fid)
    return fragment_ids

def save_master_results(path, entry):
    """Append a result to a JSON list stored in *path*, keeping the file readable."""
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {
            "experiment_session_timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_run_results": []
        }
    data["evaluation_run_results"].append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# CHECKPOINT GENERATION
# ---------------------------------------------------------------------------
def generate_checkpoints(total_turns: int) -> List[int]:
    """Generate reasonably spaced checkpoint turn indices, ending at total_turns."""
    if total_turns <= 0:
        return []

    # Determine number of checkpoints based on conversation length
    if total_turns >= 1000:
        n = random.randint(15, 20)
    elif total_turns >= 500:
        n = random.randint(12, 16)
    elif total_turns >= 200:
        n = random.randint(10, 14)
    else:
        n = random.randint(8, 12)
    n = min(n, total_turns)

    # Evenly spaced base points
    step = total_turns / n
    checkpoints = []
    for i in range(1, n):
        # Random jitter of ±20% around the even step
        jitter = random.uniform(-0.2, 0.2) * step
        cp = int(step * i + jitter)
        cp = max(1, min(cp, total_turns - 1))  # keep in bounds, leave last for final
        checkpoints.append(cp)

    # Remove duplicates and sort
    checkpoints = sorted(set(checkpoints))

    # Always end with total_turns
    if checkpoints and checkpoints[-1] >= total_turns:
        checkpoints[-1] = total_turns
    else:
        checkpoints.append(total_turns)

    # Ensure minimum gap of 5 turns between checkpoints
    cleaned = []
    for cp in checkpoints:
        if not cleaned or cp - cleaned[-1] >= 5:
            cleaned.append(cp)
        else:
            # Merge with previous by skipping
            cleaned[-1] = max(cleaned[-1], cp)
    if cleaned[-1] != total_turns:
        cleaned.append(total_turns)

    return cleaned

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Force fresh start (truncate DB)")
    args = parser.parse_args()

    os.makedirs(MATURE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Conditional one‑time database reset ──
    if args.fresh or not COMPLETED_FILE.exists():
        db = SessionLocal()
        from sqlalchemy import text
        db.execute(text(
            "TRUNCATE episodic_memory, conversations, codex_entities, codex_edges, "
            "codex_events, codex_snapshots, procedural_memory, session_summaries, "
            "context_clusters, sentinel_events, cold_storage, idempotency_keys, "
            "rag_documents, rag_chunks, batch_summaries, episodic_cluster_links "
            "RESTART IDENTITY CASCADE"
        ))
        db.commit()
        db.close()
        print("✅ Database truncated – starting fresh.\n")
    else:
        print("📂 Resuming from existing state – database left untouched.\n")

    # Initialise classifier and embedder
    classifier = PyTorchClassifier(
        model_path=settings.classifier_model_path,
        schema_path=settings.label_schema_path,
    )
    embedder = classifier.embedder
    ollama_client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")

    # Load simulation data and group by conversation
    with open(SIMULATION_INPUT, "r") as f:
        all_turns = [json.loads(line) for line in f if line.strip()]
    conv_turns = defaultdict(list)
    for t in all_turns:
        conv_turns[t.get("conversation_id")].append(t)

    completed = load_completed()
    last_turns = load_last_turns()

    for conv_cfg in TARGET_CONVERSATIONS:
        cid = conv_cfg["conversation_id"]
        label = conv_cfg["label"]
        curation_path = CURATION_DIR / f"{conv_cfg['curation_checkpoint']}.json"
        if not curation_path.exists():
            print(f"⚠️  Curation file not found for {label}: {curation_path}")
            continue

        curation = json.loads(curation_path.read_text())
        # Generated probes (accumulating)
        gen_for_conv = generated_probes.get(cid, {})
        # Manual probes from curation (held until final checkpoint)
        manual_probes = [p for p in curation.get("evaluation_probes", [])
                         if p.get("user_injected_prompt") and p["user_injected_prompt"] != "ENTER_PROBE_HERE"]

        turns = conv_turns.get(cid, [])
        if not turns:
            print(f"⚠️  No simulation data for conversation {cid}")
            continue

        # Sort turns by timestamp
        turns.sort(key=lambda x: x.get("timestamp", ""))

        start_idx = last_turns.get(cid, 0)
        # Use the same splits that were used to generate probes
        gen_for_conv = generated_probes.get(cid, {})
        if not gen_for_conv:
            print(f"⚠️  No generated probes for {label} — skipping.")
            continue
        checkpoints = sorted(int(k) for k in gen_for_conv.keys())
        # Ensure conversation exists in DB
        db = SessionLocal()
        conv = db.query(Conversation).filter_by(id=cid).first()
        if not conv:
            conv = Conversation(id=cid, memory_scope_type="auto")
            db.add(conv)
            db.commit()
        db.close()

        # Calculate total tokens so far
        def compute_total_tokens(up_to_idx):
            total = 0
            for t in turns[:up_to_idx]:
                total += estimate_tokens(t.get("prompt", "") + " " + t.get("response", ""))
            return total

        # Find the next checkpoint that is greater than start_idx
        cp_idx = 0
        while cp_idx < len(checkpoints) and checkpoints[cp_idx] <= start_idx:
            cp_idx += 1

        # Replay loop
        while cp_idx < len(checkpoints):
            target_turn = checkpoints[cp_idx]
            checkpoint_id = f"MAT-{cid[:8]}-TURN{target_turn}"

            # ── Guard: skip replay + workers if this checkpoint is already ingested ──
            last_turn_of_checkpoint = turns[target_turn - 1]
            last_ts = last_turn_of_checkpoint.get("timestamp", "")
            last_batch_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{cid}:{last_ts}")
            db_check = SessionLocal()
            already_inserted = db_check.query(EpisodicMemory).filter_by(
                batch_id=last_batch_id
            ).first() is not None
            db_check.close()
            total_tokens_now = compute_total_tokens(target_turn)
            days_simulated = max(1, target_turn // 12)
            if already_inserted:
                print(f"  ⏭️  Checkpoint turn {target_turn} already in DB — skipping replay & workers.")
                start_idx = target_turn
            else:
                # Replay new turns
                new_turns = turns[start_idx:target_turn]
                db = SessionLocal()
                for entry in tqdm(new_turns, desc=f"Replaying {label} → turn {target_turn}"):
                    prompt = entry["prompt"]
                    response = entry.get("response", "")
                    ts = entry.get("timestamp", "")
                    # NOTE: classified WITHOUT conversation_id here, while the
                    # probe classification below uses conversation_id=cid.
                    # This asymmetry is a known, separately-tracked issue
                    # (context-aware classification at probe time vs
                    # context-free classification at ingest time) — left
                    # unchanged here since it's outside the scope of the
                    # ground-truth-correction fix this file is making.
                    classification = classifier.classify(prompt)
                    emb = embedder.encode(prompt, convert_to_tensor=False).tolist()
                    batch_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{cid}:{ts}")
                    existing = db.query(EpisodicMemory).filter_by(batch_id=batch_id).first()
                    if existing:
                        continue
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)

                    turn = EpisodicMemory(
                        conversation_id=cid,
                        batch_id=batch_id,
                        timestamp=timestamp,
                        topic_tags=classification.topic_tags,
                        intent_tags=classification.intent_tags,
                        context_reliance=classification.context_reliance,
                        raw_text=f"User: {prompt}\n\nAssistant: {response}",
                        embedding=emb,
                        idempotency_key=str(uuid.uuid4()),
                    )
                    db.add(turn)
                    db.flush()

                    # Post‑flight evaluation
                    from src.workers.post_flight import is_lossless, generate_summary
                    lossless = is_lossless(response)
                    force_lossless = (
                        turn.topic_tags and
                        ("Creative_&_Media" in turn.topic_tags or "Emotional_Processing" in turn.intent_tags)
                    )
                    if force_lossless:
                        lossless = True
                        inject_raw = True
                        summary = None
                    else:
                        full_text = f"User: {prompt}\nAssistant: {response}"
                        word_count = len(full_text.split())
                        has_code = "```" in response
                        inject_raw = True
                        if lossless and word_count > 500 and not has_code:
                            summary = generate_summary(prompt, response)
                            inject_raw = False
                        elif not lossless:
                            summary = generate_summary(prompt, response)
                            inject_raw = False
                        else:
                            summary = None
                    turn.lossless_flag = lossless
                    turn.summary_text = summary
                    turn.inject_raw = inject_raw

                    # Codex extraction
                    if lossless:
                        from src.workers.codex_extractor import extract_triplets, handle_triplet
                        triplets = extract_triplets(turn.raw_text, topic_tags=turn.topic_tags)
                        for triplet in triplets:
                            if isinstance(triplet, dict):
                                s, r, o = triplet.get("subject"), triplet.get("relation"), triplet.get("object")
                                if isinstance(s, str) and isinstance(r, str) and isinstance(o, str):
                                    s, r, o = s.strip(), r.strip(), o.strip()
                                    if s and r and o:
                                        handle_triplet(db, s, r, o, str(batch_id))
                    db.commit()
                db.close()
                start_idx = target_turn

                # Run background workers (only when new data was inserted)

                print(f"  Simulating {days_simulated} days of decay at turn {start_idx}...")
                for d in range(days_simulated):
                    apply_decay()
                    decay_codex_edges()
                    decay_procedural_patterns()
                    if d % 5 == 0:
                        cluster_turns()
                        run_reflection()
                        monitor_sentinels()
                        merge_similar_clusters()

            # ── Probes (always run, governed by _completed.txt) ──
            # Build probe list for this checkpoint
            checkpoint_probes = []
            for split_str, probes_list in gen_for_conv.items():
                split_turn = int(split_str)
                if split_turn <= start_idx:
                    checkpoint_probes.extend(probes_list)
            if cp_idx == len(checkpoints) - 1:
                checkpoint_probes.extend(manual_probes)

            print(f"  Probing {len(checkpoint_probes)} questions...")
            for probe in tqdm(checkpoint_probes, desc=f"Probes @ turn {start_idx}"):
                pid = probe["probe_id"]
                if (cid, pid, checkpoint_id) in completed:
                    continue

                prompt = probe["user_injected_prompt"]
                # ── Ground truth now resolved PER CHECKPOINT via corrected
                # ground truths (see get_ground_truth_for_checkpoint above),
                # instead of always using the probe's original static
                # expected_answer regardless of how far the conversation has
                # moved on. This is what actually makes correct_ground_truths.py's
                # output have any effect on the experiment.
                ground_truth = get_ground_truth_for_checkpoint(cid, probe, start_idx)

                classification = classifier.classify(prompt, conversation_id=cid)
                emb = embedder.encode(prompt, convert_to_tensor=False).tolist()

                result_entry = {
                    "conversation_id": cid,
                    "checkpoint_id": checkpoint_id,
                    "probe_id": pid,
                    "question": prompt,
                    "ground_truth": ground_truth,
                    "turn_index": start_idx,
                    "simulated_days": days_simulated,
                    "decay_cycles_run": days_simulated,
                    "total_tokens_in_conversation": total_tokens_now,
                    "conditions": {}
                }

                memory_slots = SessionLocal().query(MemorySlot).filter_by(is_active=True).all()

                for cond_name in CONDITIONS:
                    is_vector = cond_name.startswith("vector_rag")
                    is_moe = "moe" in cond_name

                    if is_moe:
                        best_model, _ = find_best_model(classification.topic_tags, classification.intent_tags)
                        model_to_use = best_model if best_model else SINGLE_MODEL
                    else:
                        model_to_use = SINGLE_MODEL

                    orch_db = SessionLocal()
                    orchestrator = HybridRetrievalOrchestrator(orch_db, embedder)
                    orchestrator.set_budget_from_turn_count(start_idx, total_tokens_now, classification=classification)
                    if is_vector:
                        from sqlalchemy import text
                        from pgvector.sqlalchemy import Vector as PgVector
                        query = text("""
                            SELECT raw_text, summary_text, lossless_flag, inject_raw,
                                1 - (embedding <=> :prompt_embedding) as score
                            FROM episodic_memory
                            WHERE embedding IS NOT NULL AND is_archived = false
                            AND conversation_id = :conv_id
                            ORDER BY score DESC LIMIT 30
                        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
                        rows = orch_db.execute(query, {
                            "prompt_embedding": emb,
                            "conv_id": cid,
                        }).fetchall()
                        fragments = []
                        for r in rows:
                            text = r.raw_text if r.lossless_flag else (r.summary_text or r.raw_text[:300])
                            fragments.append(text)
                        context = "\n\n".join(fragments)
                        messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}]
                        tokens_injected = sum(estimate_tokens(t) for t in fragments)
                        retrieved_fragments = [{"text": t, "source": "vector"} for t in fragments]
                        hyde_used = False
                        hyde_rewritten = None
                    else:
                        scope = {"conversation_id": cid}
                        retrieved = orchestrator.retrieve(
                            classification=classification,
                            conversation_id=cid,
                            prompt_embedding=emb,
                            scope=scope,
                        )
                        hyde_used = getattr(orchestrator, "_hyde_used", False)
                        hyde_rewritten = getattr(orchestrator, "_last_hyde_query", None)
                        retrieved_fragments = [{"text": f.text, "source_type": f.source_type, "score": f.score} for f in retrieved]
                        messages = assemble_prompt(
                            memory_slots=memory_slots,
                            retrieved_fragments=retrieved,
                            user_message=prompt,
                            db_session=orch_db,
                            conversation_id=cid,
                            classification=classification,
                            scope=scope,
                            max_recent_tokens=orchestrator.recent_token_budget,
                        )
                    # Use the actual assembled prompt word count, not just fragments
                    # Count ALL message content being sent to the model
                    total_text = " ".join(m["content"] for m in messages if m.get("content"))
                    tokens_injected = int(len(total_text.split()) * 1.33)
                    start_time = time.time()
                    answer = ""
                    # Generation with thinking mode (universal – Gemma 4 & Qwen 3)
                    for attempt in range(2):
                        try:
                            resp = ollama_client.chat.completions.create(
                                model=model_to_use,
                                messages=messages,
                                temperature=0.7,
                                max_tokens=20000,
                                timeout=180.0,
                                extra_body={
                                    "num_ctx": 64000,
                                    "think": True,
                                },
                            )
                            raw = resp.choices[0].message.content or ""

                            # Strip thinking blocks for ALL model families
                            # Qwen: <think>…</think>   Gemma 4: <|channel|>…<|channel|>
                            clean = raw
                            for pattern in [
                                r"<think>.*?</think>\s*",
                                r"<\|channel>.*?<channel\|>\s*",  # Fixed Gemma 4 closing tag structure
                                r"<\|think\|>.*?\n",
                            ]:
                                clean = re.sub(pattern, "", clean, flags=re.DOTALL).strip()

                            answer = clean if clean else raw   # fallback to raw if stripping wiped everything

                            if answer:
                                break   # success – exit retry loop
                        except Exception as e:
                            answer = ""   # keep empty so the retry condition fires correctly
                            if attempt == 0:
                                time.sleep(0.5)

                    if not answer:
                        answer = "ERROR: Model failed to generate a response after 2 attempts."

                    latency = time.time() - start_time

                    fids = save_fragments(
                        retrieved_fragments, cid, pid,
                        result_entry["checkpoint_id"], cond_name
                    )

                    result_entry["conditions"][cond_name] = {
                        "model_used": model_to_use,
                        "answer": answer,
                        "tokens_injected": tokens_injected,
                        "latency_seconds": round(latency, 3),
                        "classification": {
                            "topic_tags": classification.topic_tags,
                            "intent_tags": classification.intent_tags,
                            "context_reliance": classification.context_reliance,
                            "max_confidence": classification.max_confidence,
                        },
                        "hyde_used": hyde_used,
                        "hyde_rewritten_query": hyde_rewritten,
                        "fragment_ids": fids,
                    }
                    orch_db.close()

                save_master_results(MASTER_RESULTS_FILE, result_entry)
                save_completed(cid, pid, checkpoint_id)
                completed.add((cid, pid, checkpoint_id))
            # Move to next checkpoint
            save_last_turn(cid, start_idx)
            cp_idx += 1

        # Mark conversation as fully done
        print(f"✅ {label} complete.")

    print(f"\n🎉 Experiment finished. Responses saved to {MASTER_RESULTS_FILE}")

if __name__ == "__main__":
    main()