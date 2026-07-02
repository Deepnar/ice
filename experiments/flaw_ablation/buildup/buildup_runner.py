#!/usr/bin/env python3
"""
ICE Flaw Buildup Experiment Runner (Single‑Pass)
=================================================
Uses the EXISTING mature database from the subtraction run.
Collects all unique probes from all kept splits + manual probes,
and evaluates each once against the fully‑mature state with
features added cumulatively (bare vector → full ICE).

Longitudinal measurement comes from grouping probes by their
origin split — earlier splits test recall of old facts buried
under the full conversation history.

Output: experiments/flaw_ablation/buildup/master_results.json
"""

import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import aiohttp
import torch
import numpy as np
from sqlalchemy import text, bindparam
from pgvector.sqlalchemy import Vector as PgVector
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.classifier.schemas import ClassificationResult
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, MemorySlot
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.configurable_orchestrator import ConfigurableOrchestrator
from src.api.prompt_assembler import assemble_prompt

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

ABLATION_DIR = Path("experiments/flaw_ablation")
SUBDIR = ABLATION_DIR / "buildup"
RESULTS_DIR = SUBDIR
MASTER_RESULTS_FILE = RESULTS_DIR / "master_results.json"
FRAGMENTS_FILE = RESULTS_DIR / "fragments.jsonl"
COMPLETED_FILE = SUBDIR / "_completed.txt"

KEPT_CHECKPOINTS = [51, 170, 285, 397, 492, 604, 735, 834, 959, 1053, 1119]
VECTOR_BASELINE_NAME = "vector_baseline"
FINAL_TURN = 1119

GENERATED_PROBES_FILE = Path("experiments/mature/generated_probes.json")
CORRECTED_GT_FILE = Path("experiments/mature/results/corrected_ground_truths.json")
CURATION_PATH = Path("experiments/curation_files/EC-961862eb-FULL.json")

# ── Cumulative condition definitions ─────────────────────────────────
BUILDUP_CONDITIONS = [
    ("bare_vector", {"vector": True, "bm25": False, "rrf": False, "hyde": False,
     "cluster_restrict": False, "session_diversify": False, "codex": False,
     "mera": False, "fuzzy_match": False, "procedural": False,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_bm25", {"vector": True, "bm25": True, "rrf": False, "hyde": False,
     "cluster_restrict": False, "session_diversify": False, "codex": False,
     "mera": False, "fuzzy_match": False, "procedural": False,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_rrf", {"vector": True, "bm25": True, "rrf": True, "hyde": False,
     "cluster_restrict": False, "session_diversify": False, "codex": False,
     "mera": False, "fuzzy_match": False, "procedural": False,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_hyde", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": False, "session_diversify": False, "codex": False,
     "mera": False, "fuzzy_match": False, "procedural": False,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_cluster_restrict", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": False, "codex": False,
     "mera": False, "fuzzy_match": False, "procedural": False,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_session_diversify", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": False,
     "mera": False, "fuzzy_match": False, "procedural": False,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_codex", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": True,
     "mera": False, "fuzzy_match": True, "procedural": False,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_mera", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": True,
     "mera": True, "fuzzy_match": True, "procedural": False,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_procedural", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": True,
     "mera": True, "fuzzy_match": True, "procedural": True,
     "batch_summary": False, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_batch_summary", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": True,
     "mera": True, "fuzzy_match": True, "procedural": True,
     "batch_summary": True, "dynamic_budget": False, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_dynamic_budget", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": True,
     "mera": True, "fuzzy_match": True, "procedural": True,
     "batch_summary": True, "dynamic_budget": True, "sliding_window": False,
     "keyword_boost": False, "recency_boost": False}),
    ("add_sliding_window", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": True,
     "mera": True, "fuzzy_match": True, "procedural": True,
     "batch_summary": True, "dynamic_budget": True, "sliding_window": True,
     "keyword_boost": False, "recency_boost": False}),
    ("add_keyword_boost", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": True,
     "mera": True, "fuzzy_match": True, "procedural": True,
     "batch_summary": True, "dynamic_budget": True, "sliding_window": True,
     "keyword_boost": True, "recency_boost": False}),
    ("full_ice", {"vector": True, "bm25": True, "rrf": True, "hyde": True,
     "cluster_restrict": True, "session_diversify": True, "codex": True,
     "mera": True, "fuzzy_match": True, "procedural": True,
     "batch_summary": True, "dynamic_budget": True, "sliding_window": True,
     "keyword_boost": True, "recency_boost": True}),
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_json(path):
    with open(path, "r") as f: return json.load(f)

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
            parts = line.strip().split("|")
            if len(parts) == 3:
                completed.add((parts[0], parts[1], parts[2]))
    return completed

def save_completed(cid, pid, cond_name):
    with open(COMPLETED_FILE, "a") as f:
        f.write(f"{cid}|{pid}|{cond_name}\n")
        f.flush()
        os.fsync(f.fileno())

def save_fragments(fragments_list, conv_id, probe_id, cond_name):
    fragment_ids = []
    for frag in fragments_list:
        fid = str(uuid.uuid4())[:8]
        append_jsonl(FRAGMENTS_FILE, {
            "fragment_id": fid, "conversation_id": conv_id,
            "probe_id": probe_id, "condition": cond_name,
            "text": frag.get("text", ""),
            "source_type": frag.get("source_type", frag.get("source", "vector")),
            "score": frag.get("score", None)
        })
        fragment_ids.append(fid)
    return fragment_ids

def load_master_results(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {
        "experiment_session_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_run_results": [],
    }

def save_condition_to_master(path, entry):
    """Add or update *entry* (keyed by probe_id) and write back atomically."""
    data = load_master_results(path)
    results = data["evaluation_run_results"]

    existing_idx = None
    for i, e in enumerate(results):
        if e.get("probe_id") == entry["probe_id"]:
            existing_idx = i
            break

    if existing_idx is not None:
        results[existing_idx] = entry
    else:
        results.append(entry)

    data["experiment_session_timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
def get_ground_truth_for_probe(cid, probe):
    """Return the LATEST (turn 1119) corrected ground truth for a probe."""
    corrected_ground_truths = {}
    if CORRECTED_GT_FILE.exists():
        with open(CORRECTED_GT_FILE) as f:
            corrected_ground_truths = json.load(f)
    pid = probe["probe_id"]
    per_probe = corrected_ground_truths.get(cid, {}).get(pid, {})
    if not per_probe:
        return probe.get("expected_answer", "")
    # Use the latest available checkpoint
    all_cps = sorted([int(k) for k in per_probe.keys()])
    if all_cps:
        return per_probe[str(all_cps[-1])]
    return probe.get("expected_answer", "")

async def generate_sglang(session, sem, messages, max_tokens=SGLANG_MAX_TOKENS):
    """Call SGLang and return cleaned answer text (thinking stripped)."""
    payload = {
        "model": SGLANG_MODEL, "messages": messages,
        "temperature": SGLANG_TEMPERATURE, "top_p": SGLANG_TOP_P,
        "max_tokens": max_tokens, "stream": False,
        "extra_body": {"top_k": SGLANG_TOP_K},
        }
    for attempt in range(2):
        try:
            async with session.post(
                f"{SGLANG_URL}/chat/completions", json=payload,
                timeout=aiohttp.ClientTimeout(total=180)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"SGLang error {resp.status}: {text}")
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"] or ""
                clean = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
                return clean if clean else raw
        except Exception:
            if attempt == 0: await asyncio.sleep(0.5)
    return "ERROR: SGLang generation failed after 2 attempts."

def extract_origin_split(probe_id):
    """Extract the origin split turn from a probe ID like '51-GEN-01'."""
    match = re.match(r"(\d+)-GEN-", probe_id)
    return int(match.group(1)) if match else 0


def load_stored_classifications(path):
    """Return dict: probe_id -> classification dict from previously saved results."""
    stored = {}
    if not Path(path).exists():
        return stored
    data = load_json(path)
    for entry in data.get("evaluation_run_results", []):
        pid = entry.get("probe_id")
        if not pid or pid in stored:
            continue
        # grab classification from the first condition that has one
        for cond_name, cond_data in entry.get("conditions", {}).items():
            cls = cond_data.get("classification")
            if cls:
                stored[pid] = cls
                break
    return stored

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SUBDIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Verify database has data
    db_check = SessionLocal()
    turn_count = db_check.query(EpisodicMemory).filter_by(conversation_id=FLAW_CID).count()
    db_check.close()
    if turn_count == 0:
        print("❌ Database is empty. Run subtraction_runner.py first to build mature state.")
        return
    print(f"✅ Database ready: {turn_count} turns for Flaw conversation.\n")

    # Load classifier
    classifier = PyTorchClassifier(model_path=settings.classifier_model_path,
                                   schema_path=settings.label_schema_path)
    embedder = classifier.embedder

    # ── Collect all unique probes ─────────────────────────────────────
    generated_probes = {}
    if GENERATED_PROBES_FILE.exists():
        with open(GENERATED_PROBES_FILE) as f:
            generated_probes = json.load(f)
    gen_for_conv = generated_probes.get(FLAW_CID, {})

    all_probes = []   # list of (probe_dict, origin_split)
    seen_ids = set()

    for split_str, probes_list in gen_for_conv.items():
        split_turn = int(split_str)
        if split_turn not in KEPT_CHECKPOINTS:
            continue
        for probe in probes_list:
            pid = probe.get("probe_id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_probes.append((probe, split_turn))

    # Manual probes
    if CURATION_PATH.exists():
        curation = json.loads(CURATION_PATH.read_text())
        for probe in curation.get("evaluation_probes", []):
            if probe.get("user_injected_prompt") and probe["user_injected_prompt"] != "ENTER_PROBE_HERE":
                pid = probe.get("probe_id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_probes.append((probe, FINAL_TURN))

    print(f"📊 {len(all_probes)} unique probes to evaluate.")
    print(f"   Origin splits: {sorted(set(s for _, s in all_probes))}")

    # ── Compute total tokens in conversation ──────────────────────────
    db = SessionLocal()
    total_tokens_now = 0
    for t in db.query(EpisodicMemory).filter_by(conversation_id=FLAW_CID).all():
        total_tokens_now += estimate_tokens(t.raw_text or "")
    db.close()
    print(f"   Total tokens in conversation: ~{total_tokens_now}")

    completed = load_completed()
    stored_classifications = load_stored_classifications(MASTER_RESULTS_FILE)
    # ── Run probes ────────────────────────────────────────────────────
    async def probe_one(probe, origin_split, session, sem):
        async with sem:          # <--- gate EVERYTHING, not just SGLang
            pid = probe["probe_id"]
            prompt = probe["user_injected_prompt"]
            ground_truth = get_ground_truth_for_probe(FLAW_CID, probe)

            # Reuse stored classification if available, else compute and store
            stored_cls = stored_classifications.get(pid)
            if stored_cls:
                classification = ClassificationResult(
                    topic_tags=stored_cls.get("topic_tags", []),
                    intent_tags=stored_cls.get("intent_tags", []),
                    context_reliance=stored_cls.get("context_reliance", ""),
                    raw_probs=[],
                    max_confidence=stored_cls.get("max_confidence", 0.0),
                    prompt=prompt,
                )
            else:
                classification = classifier.classify(prompt, conversation_id=FLAW_CID)
                stored_classifications[pid] = {
                    "topic_tags": classification.topic_tags,
                    "intent_tags": classification.intent_tags,
                    "context_reliance": classification.context_reliance,
                    "max_confidence": classification.max_confidence,
                }
            emb = embedder.encode(prompt, convert_to_tensor=False).tolist()

            result_entry = {
                "conversation_id": FLAW_CID,
                "probe_id": pid,
                "question": prompt,
                "ground_truth": ground_truth,
                "turn_index": FINAL_TURN,
                "origin_split": origin_split,
                "total_tokens_in_conversation": total_tokens_now,
                "conditions": {}
            }
                        # Load previously saved conditions so we don't overwrite them
            existing_data = load_master_results(MASTER_RESULTS_FILE)
            for saved_entry in existing_data.get("evaluation_run_results", []):
                if saved_entry.get("probe_id") == pid:
                    for cond_name, cond_data in saved_entry.get("conditions", {}).items():
                        if cond_name not in result_entry["conditions"]:
                            result_entry["conditions"][cond_name] = cond_data
                    break

            slot_db = SessionLocal()
            memory_slots = slot_db.query(MemorySlot).filter_by(is_active=True).all()
            slot_db.close()
            for cond_name, overrides in BUILDUP_CONDITIONS:
                if (FLAW_CID, pid, cond_name) in completed:
                    continue

                orch_db = SessionLocal()
                orchestrator = ConfigurableOrchestrator(orch_db, embedder, overrides=overrides)
                orchestrator.set_budget_from_turn_count(FINAL_TURN, total_tokens_now, classification=classification)
                if overrides.get("hyde"):
                    orchestrator._force_hyde = True

                scope = {"conversation_id": FLAW_CID}
                retrieved = await asyncio.to_thread(
                    orchestrator.retrieve,
                    classification=classification, conversation_id=FLAW_CID,
                    prompt_embedding=emb, scope=scope,
                )
                hyde_used = getattr(orchestrator, "_hyde_used", False)
                hyde_rewritten = getattr(orchestrator, "_last_hyde_query", None)
                retrieved_fragments = [{"text": f.text, "source_type": f.source_type, "score": f.score} for f in retrieved]

                max_recent = orchestrator.recent_token_budget if overrides.get("sliding_window", True) else 0
                messages = assemble_prompt(
                    memory_slots=memory_slots, retrieved_fragments=retrieved,
                    user_message=prompt, db_session=orch_db, conversation_id=FLAW_CID,
                    classification=classification, scope=scope, max_recent_tokens=max_recent,
                )
                total_text = " ".join(m["content"] for m in messages if m.get("content"))
                tokens_injected = int(len(total_text.split()) * 1.33)
                answer = await generate_sglang(session, sem, messages)
                fids = save_fragments(retrieved_fragments, FLAW_CID, pid, cond_name)

                result_entry["conditions"][cond_name] = {
                    "model_used": SGLANG_MODEL, "answer": answer,
                    "tokens_injected": tokens_injected, "latency_seconds": 0,
                    "classification": {
                        "topic_tags": classification.topic_tags,
                        "intent_tags": classification.intent_tags,
                        "context_reliance": classification.context_reliance,
                        "max_confidence": classification.max_confidence,
                    },
                    "hyde_used": hyde_used, "hyde_rewritten_query": hyde_rewritten,
                    "fragment_ids": fids,
                }
                orch_db.close()
                save_completed(FLAW_CID, pid, cond_name)
                save_condition_to_master(MASTER_RESULTS_FILE, result_entry)

            # Vector baseline
            vec_cond = VECTOR_BASELINE_NAME
            if (FLAW_CID, pid, vec_cond) not in completed:
                orch_db = SessionLocal()
                query = text("""
                    SELECT raw_text, summary_text, lossless_flag, inject_raw,
                           1 - (embedding <=> :prompt_embedding) as score
                    FROM episodic_memory
                    WHERE embedding IS NOT NULL AND is_archived = false
                    AND conversation_id = :conv_id
                    ORDER BY score DESC LIMIT 30
                """).bindparams(bindparam("prompt_embedding", type_=PgVector))
                rows = (await asyncio.to_thread(
                    orch_db.execute, query, {"prompt_embedding": emb, "conv_id": FLAW_CID}
                )).fetchall()
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
                fids = save_fragments(retrieved_frags, FLAW_CID, pid, vec_cond)

                result_entry["conditions"][vec_cond] = {
                    "model_used": SGLANG_MODEL, "answer": answer,
                    "tokens_injected": int(tokens_injected), "latency_seconds": 0,
                    "classification": {
                        "topic_tags": classification.topic_tags,
                        "intent_tags": classification.intent_tags,
                        "context_reliance": classification.context_reliance,
                        "max_confidence": classification.max_confidence,
                    },
                    "hyde_used": False, "hyde_rewritten_query": None,
                    "fragment_ids": fids,
                }
                orch_db.close()
                save_completed(FLAW_CID, pid, vec_cond)
                save_condition_to_master(MASTER_RESULTS_FILE, result_entry)

            for c, _ in BUILDUP_CONDITIONS:
                completed.add((FLAW_CID, pid, c))
            completed.add((FLAW_CID, pid, vec_cond))

    # ── Run all probes concurrently ───────────────────────────────────
    sem = asyncio.Semaphore(CONCURRENCY)
    async def run_all_probes():
        async with aiohttp.ClientSession() as session:
            tasks = [probe_one(probe, split, session, sem) for probe, split in all_probes]
            for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Probes"):
                await coro

    asyncio.run(run_all_probes())
    print(f"\n✅ Buildup experiment complete. Results: {MASTER_RESULTS_FILE}")

if __name__ == "__main__":
    main()