#!/usr/bin/env python3
"""
ICE Flaw Subtraction Experiment Runner
=======================================
Replays the Flaw conversation (bb558b5f) with the full ICE‑Mature stack,
pauses at 11 kept checkpoints, runs background workers, then probes with
the baseline (all features ON) and each feature turned OFF individually.

Resumable: saves completed probes per condition; Ctrl‑C safe.
Uses SGLang backend (mattbucci/Qwen3.6-27B-AWQ) with async concurrency.

Output: experiments/flaw_ablation/subtraction/master_results.json
"""

from typing import List
import json, os, sys, uuid, time, random, re, asyncio
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import aiohttp
from openai import OpenAI
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# Ensure project root is on sys.path
# To this:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import torch
import numpy as np
from sqlalchemy import text, bindparam
from pgvector.sqlalchemy import Vector as PgVector

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, MemorySlot, ContextCluster
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.configurable_orchestrator import ConfigurableOrchestrator
from src.api.prompt_assembler import assemble_prompt

# Background workers
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
SGLANG_URL = "http://localhost:8001/v1"
SGLANG_MODEL = "Qwen/Qwen3-14B-AWQ"
SGLANG_TEMPERATURE = 0.7
SGLANG_TOP_P = 0.95
SGLANG_TOP_K = 20
SGLANG_MAX_TOKENS = 4096
CONCURRENCY = 4

FLAW_CID = "bb558b5f-5365-5bac-9ed0-07219025b5f2"
FLAW_LABEL = "Flaw"

ABLATION_DIR = Path("experiments/flaw_ablation")
SUBDIR = ABLATION_DIR / "subtraction"
RESULTS_DIR = SUBDIR
MASTER_RESULTS_FILE = RESULTS_DIR / "master_results.json"
FRAGMENTS_FILE = RESULTS_DIR / "fragments.jsonl"
COMPLETED_FILE = SUBDIR / "_completed.txt"
LAST_TURN_FILE = SUBDIR / "_last_turn.txt"

# Kept checkpoints: every other from generated_probes.json plus final
# Full list: 51,115,170,216,285,336,397,448,492,555,604,681,735,790,834,885,959,1017,1053,1119
KEPT_CHECKPOINTS = [51, 170, 285, 397, 492, 604, 735, 834, 959, 1053, 1119]

# Load generated probes (for all conversations)
GENERATED_PROBES_FILE = Path("experiments/mature/generated_probes.json")
generated_probes = {}
if GENERATED_PROBES_FILE.exists():
    with open(GENERATED_PROBES_FILE) as f:
        generated_probes = json.load(f)

# Load corrected ground truths
CORRECTED_GT_FILE = Path("experiments/mature/results/corrected_ground_truths.json")
corrected_ground_truths = {}
if CORRECTED_GT_FILE.exists():
    with open(CORRECTED_GT_FILE) as f:
        corrected_ground_truths = json.load(f)

# Curation file for manual probes
CURATION_PATH = Path("experiments/curation_files/EC-961862eb-FULL.json")
SIMULATION_INPUT = Path("data/simulation/simulation_full.jsonl")

# All conditions: baseline (all on) + one feature off at a time + vector baseline
FEATURE_FLAGS = {
    "baseline_all_on": {},
    "no_vector": {"vector": False},
    "no_bm25": {"bm25": False},
    "no_rrf": {"rrf": False},
    "hyde_on": {"hyde": True},                 # HyDE ON (normally off)
    "no_cluster_restrict": {"cluster_restrict": False},
    "no_session_diversify": {"session_diversify": False},
    "no_codex": {"codex": False},
    "no_mera": {"mera": False},
    "no_fuzzy_match": {"fuzzy_match": False},
    "no_procedural": {"procedural": False},
    "no_batch_summary": {"batch_summary": False},
    "static_budget": {"dynamic_budget": False},
    "no_sliding_window": {"sliding_window": False},
    "no_keyword_boost": {"keyword_boost": False},
    "no_recency_boost": {"recency_boost": False},
}
VECTOR_BASELINE_NAME = "vector_baseline"

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

def estimate_tokens(text):
    return int(len(text.split()) * 1.33)

def load_completed():
    if not COMPLETED_FILE.exists():
        return set()
    completed = set()
    with open(COMPLETED_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("|")
                if len(parts) == 4:
                    completed.add((parts[0], parts[1], parts[2], parts[3]))
    return completed

def save_completed(cid, pid, checkpoint_id, cond_name):
    with open(COMPLETED_FILE, "a") as f:
        f.write(f"{cid}|{pid}|{checkpoint_id}|{cond_name}\n")
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

def save_last_turn(cid, turn_index):
    turns = load_last_turns()
    turns[cid] = turn_index
    tmp = LAST_TURN_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for c, t in turns.items():
            f.write(f"{c}|{t}\n")
    os.replace(tmp, LAST_TURN_FILE)

def save_fragments(fragments_list, conv_id, probe_id, checkpoint_id, cond_name):
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

def get_ground_truth_for_checkpoint(cid, probe, checkpoint_turn):
    pid = probe["probe_id"]
    per_probe = corrected_ground_truths.get(cid, {}).get(pid, {})
    if not per_probe:
        return probe.get("expected_answer", "")
    exact = per_probe.get(str(checkpoint_turn))
    if exact is not None:
        return exact
    candidates = [int(k) for k in per_probe.keys() if int(k) <= checkpoint_turn]
    if candidates:
        return per_probe[str(max(candidates))]
    # Nearest neighbor fallback for Flaw
    if cid == FLAW_CID:
        all_cps = [int(k) for k in per_probe.keys()]
        if all_cps:
            closest = min(all_cps, key=lambda x: abs(x - checkpoint_turn))
            return per_probe[str(closest)]
    return probe.get("expected_answer", "")

# ---------------------------------------------------------------------------
# SGLANG GENERATION HELPER (async)
# ---------------------------------------------------------------------------
async def generate_sglang(session, sem, messages, max_tokens=SGLANG_MAX_TOKENS):
    """Call SGLang and return cleaned answer text (thinking stripped)."""
    async with sem:
        payload = {
            "model": SGLANG_MODEL,
            "messages": messages,
            "temperature": SGLANG_TEMPERATURE,
            "top_p": SGLANG_TOP_P,
            "max_tokens": max_tokens,
            "stream": False,
            "extra_body": {"top_k": SGLANG_TOP_K},
        }
        for attempt in range(2):
            try:
                async with session.post(
                    f"{SGLANG_URL}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"SGLang error {resp.status}: {text}")
                    data = await resp.json()
                    raw = data["choices"][0]["message"]["content"] or ""
                    # Strip Qwen thinking blocks
                    clean = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
                    return clean if clean else raw
            except Exception:
                if attempt == 0:
                    await asyncio.sleep(0.5)
        return "ERROR: SGLang generation failed after 2 attempts."

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Force fresh start (truncate DB)")
    args = parser.parse_args()

    os.makedirs(SUBDIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Conditional one‑time database reset
    # Conditional one‑time database reset
    db_check = SessionLocal()
    turns_exist = db_check.query(EpisodicMemory).filter_by(conversation_id=FLAW_CID).first() is not None
    db_check.close()

    if args.fresh or (not turns_exist and not COMPLETED_FILE.exists()):
        db = SessionLocal()
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
        print("📂 Database already has Flaw turns – skipping truncate and resume.\n")

    # Initialise classifier and embedder
    classifier = PyTorchClassifier(
        model_path=settings.classifier_model_path,
        schema_path=settings.label_schema_path,
    )
    embedder = classifier.embedder

    # Load simulation data
    with open(SIMULATION_INPUT, "r") as f:
        all_turns = [json.loads(line) for line in f if line.strip()]
    conv_turns = defaultdict(list)
    for t in all_turns:
        conv_turns[t.get("conversation_id")].append(t)

    completed = load_completed()
    last_turns = load_last_turns()

    cid = FLAW_CID
    label = FLAW_LABEL
    turns = conv_turns.get(cid, [])
    if not turns:
        print(f"❌ No simulation data for Flaw conversation {cid}")
        return

    turns.sort(key=lambda x: x.get("timestamp", ""))

    start_idx = last_turns.get(cid, 0)
    # If DB already has turns but _last_turn.txt doesn't, recover start_idx from DB
    if start_idx == 0:
        db_temp = SessionLocal()
        existing = db_temp.query(EpisodicMemory).filter_by(conversation_id=cid).order_by(EpisodicMemory.timestamp.desc()).first()
        if existing:
            start_idx = db_temp.query(EpisodicMemory).filter_by(conversation_id=cid).count()
            print(f"  📂 Recovered start_idx={start_idx} from existing DB turns.")
        db_temp.close()
    gen_for_conv = generated_probes.get(cid, {})
    if not gen_for_conv:
        print("❌ No generated probes for Flaw — cannot proceed.")
        return

    # Filter to kept checkpoints
    kept_splits = sorted([k for k in KEPT_CHECKPOINTS if str(k) in gen_for_conv])

    # Load manual probes
    manual_probes = []
    if CURATION_PATH.exists():
        curation = json.loads(CURATION_PATH.read_text())
        manual_probes = [p for p in curation.get("evaluation_probes", [])
                         if p.get("user_injected_prompt") and p["user_injected_prompt"] != "ENTER_PROBE_HERE"]

    # Ensure conversation exists in DB
    db = SessionLocal()
    conv = db.query(Conversation).filter_by(id=cid).first()
    if not conv:
        conv = Conversation(id=cid, memory_scope_type="auto")
        db.add(conv)
        db.commit()
    db.close()

    def compute_total_tokens(up_to_idx):
        total = 0
        for t in turns[:up_to_idx]:
            total += estimate_tokens(t.get("prompt", "") + " " + t.get("response", ""))
        return total

    # Find next checkpoint to process
    cp_idx = 0
    while cp_idx < len(kept_splits) and kept_splits[cp_idx] <= start_idx:
        cp_idx += 1

    # ── REPLAY LOOP ──────────────────────────────────────────────────────
    while cp_idx < len(kept_splits):
        target_turn = kept_splits[cp_idx]
        checkpoint_id = f"MAT-{cid[:8]}-TURN{target_turn}"

        # Guard: skip replay if already ingested
        last_turn_of_checkpoint = turns[target_turn - 1]
        last_ts = last_turn_of_checkpoint.get("timestamp", "")
        last_batch_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{cid}:{last_ts}")
        db_check = SessionLocal()
        already_inserted = db_check.query(EpisodicMemory).filter_by(batch_id=last_batch_id).first() is not None
        db_check.close()
        total_tokens_now = compute_total_tokens(target_turn)
        days_simulated = max(1, target_turn // 12)

        if already_inserted:
            print(f"  ⏭️  Checkpoint turn {target_turn} already in DB — skipping replay & workers.")
            start_idx = target_turn
        else:
            new_turns = turns[start_idx:target_turn]
            db = SessionLocal()
            for entry in tqdm(new_turns, desc=f"Replaying {label} → turn {target_turn}"):
                prompt = entry["prompt"]
                response = entry.get("response", "")
                ts = entry.get("timestamp", "")
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

            print(f"  Simulating {days_simulated} days of decay at turn {start_idx}...")
            import src.workers.gpu_check as _gpu
            _original_is_gpu_busy = _gpu.is_gpu_busy
            _gpu.is_gpu_busy = lambda: False
            try:
                for d in range(days_simulated):
                    print(f"    Day {d+1}/{days_simulated}: decay...", flush=True)
                    apply_decay()
                    decay_codex_edges()
                    decay_procedural_patterns()
                    if d % 5 == 0:
                        print(f"    Day {d+1}: clustering...", flush=True)
                        cluster_turns()
                        print(f"    Day {d+1}: reflection...", flush=True)
                        run_reflection()
                        print(f"    Day {d+1}: sentinel...", flush=True)
                        monitor_sentinels()
                        print(f"    Day {d+1}: merge...", flush=True)
                        merge_similar_clusters()
                        print(f"    Day {d+1}: workers done.", flush=True)
            finally:
                _gpu.is_gpu_busy = _original_is_gpu_busy

        # ── PROBES ───────────────────────────────────────────────────────
        # Accumulate probes from all kept splits ≤ current checkpoint
        checkpoint_probes = []
        for split_str, probes_list in gen_for_conv.items():
            split_turn = int(split_str)
            if split_turn in KEPT_CHECKPOINTS and split_turn <= start_idx:
                checkpoint_probes.extend(probes_list)
        if cp_idx == len(kept_splits) - 1:
            checkpoint_probes.extend(manual_probes)

        print(f"  Probing {len(checkpoint_probes)} questions...")

        # ── Define async probe worker ─────────────────────────────────
        async def probe_one(probe, session, sem):
            pid = probe["probe_id"]
            prompt = probe["user_injected_prompt"]
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

            # ── All ICE feature‑flag conditions ──────────────────────────
            for cond_name, overrides in FEATURE_FLAGS.items():
                if (cid, pid, checkpoint_id, cond_name) in completed:
                    continue

                orch_db = SessionLocal()
                orchestrator = ConfigurableOrchestrator(orch_db, embedder, overrides=overrides)
                orchestrator.set_budget_from_turn_count(start_idx, total_tokens_now, classification=classification)

                # HyDE handling
                if overrides.get("hyde"):
                    orchestrator._force_hyde = True

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

                max_recent = orchestrator.recent_token_budget if overrides.get("sliding_window", True) else 0
                messages = assemble_prompt(
                    memory_slots=memory_slots,
                    retrieved_fragments=retrieved,
                    user_message=prompt,
                    db_session=orch_db,
                    conversation_id=cid,
                    classification=classification,
                    scope=scope,
                    max_recent_tokens=max_recent,
                )

                total_text = " ".join(m["content"] for m in messages if m.get("content"))
                tokens_injected = int(len(total_text.split()) * 1.33)

                answer = await generate_sglang(session, sem, messages)
                fids = save_fragments(retrieved_fragments, cid, pid, checkpoint_id, cond_name)

                result_entry["conditions"][cond_name] = {
                    "model_used": SGLANG_MODEL,
                    "answer": answer,
                    "tokens_injected": tokens_injected,
                    "latency_seconds": 0,
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
                save_completed(cid, pid, checkpoint_id, cond_name)

            # ── Vector baseline condition ──────────────────────────────────
            vec_cond = VECTOR_BASELINE_NAME
            if (cid, pid, checkpoint_id, vec_cond) not in completed:
                orch_db = SessionLocal()
                query = text("""
                    SELECT raw_text, summary_text, lossless_flag, inject_raw,
                           1 - (embedding <=> :prompt_embedding) as score
                    FROM episodic_memory
                    WHERE embedding IS NOT NULL AND is_archived = false
                    AND conversation_id = :conv_id
                    ORDER BY score DESC LIMIT 30
                """).bindparams(bindparam("prompt_embedding", type_=PgVector))
                rows = orch_db.execute(query, {"prompt_embedding": emb, "conv_id": cid}).fetchall()
                fragments = []
                for r in rows:
                    text_val = r.raw_text if r.lossless_flag else (r.summary_text or r.raw_text[:300])
                    words = text_val.split()
                    if len(words) > 500:
                        text_val = " ".join(words[:500]) + "…"
                    fragments.append(text_val)
                context = "\n\n".join(fragments)
                vec_messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}]
                tokens_injected = sum(estimate_tokens(t) for t in fragments)
                answer = await generate_sglang(session, sem, vec_messages)
                retrieved_frags = [{"text": t, "source": "vector"} for t in fragments]
                fids = save_fragments(retrieved_frags, cid, pid, checkpoint_id, vec_cond)

                result_entry["conditions"][vec_cond] = {
                    "model_used": SGLANG_MODEL,
                    "answer": answer,
                    "tokens_injected": int(tokens_injected),
                    "latency_seconds": 0,
                    "classification": {
                        "topic_tags": classification.topic_tags,
                        "intent_tags": classification.intent_tags,
                        "context_reliance": classification.context_reliance,
                        "max_confidence": classification.max_confidence,
                    },
                    "hyde_used": False,
                    "hyde_rewritten_query": None,
                    "fragment_ids": fids,
                }
                orch_db.close()
                save_completed(cid, pid, checkpoint_id, vec_cond)

            save_master_results(MASTER_RESULTS_FILE, result_entry)
            # Update in-memory completed set
            for c in FEATURE_FLAGS:
                completed.add((cid, pid, checkpoint_id, c))
            completed.add((cid, pid, checkpoint_id, vec_cond))

        # ── Run all probes concurrently via asyncio.run() ─────────────────
        sem = asyncio.Semaphore(CONCURRENCY)
        
        async def run_all_probes():
            async with aiohttp.ClientSession() as session:
                tasks = [probe_one(probe, session, sem) for probe in checkpoint_probes]
                for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Probes @ turn {start_idx}"):
                    await coro

        asyncio.run(run_all_probes())

        save_last_turn(cid, start_idx)
        cp_idx += 1

    print(f"\n✅ Subtraction experiment complete. Results: {MASTER_RESULTS_FILE}")

if __name__ == "__main__":
    main()