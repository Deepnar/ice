#!/usr/bin/env python3
"""
Phase 2 – Incremental replay version.
Processes conversation checkpoints in chronological order without
re‑replaying the whole history each time.
"""

import json, os, sys, time, uuid, glob, random
from datetime import datetime, timezone
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, numpy as np
from sqlalchemy import bindparam, text
from pgvector.sqlalchemy import Vector as PgVector
from src.api.config import settings
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
OLLAMA_URL = "http://localhost:11434/v1"
SINGLE_MODEL = "gemma4:26b-a4b-it-q4_K_M"
JUDGE_MODEL = "gemma4:26b-a4b-it-q4_K_M"
SEED = 42
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "_completed.txt")

# ── Optional extra experiments (set to True to enable) ──
RUN_SCOPE_EXPERIMENT      = True
RUN_HYDE_ABLATION         = True
RUN_SLIDING_WINDOW_ABLATION = True
RUN_PROCEDURAL_TOGGLE     = True
# To keep runtime sane, you can limit extra experiments to specific checkpoints:
EXTRA_EXPERIMENTS_CHECKPOINTS = None  # e.g. ["EC-bb558b5f-TURN336", ...] or None to run on all

# ------------------------------------------------------------------
# Resume support
# ------------------------------------------------------------------
def load_completed():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_completed(checkpoint_id, probe_id):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{checkpoint_id}|{probe_id}\n")

def load_existing_results():
    results_path = os.path.join(RESULTS_DIR, "master_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            data = json.load(f)
            return data.get("evaluation_run_results", [])
    return []

# ------------------------------------------------------------------
# Simulation replay (same as before)
# ------------------------------------------------------------------
def replay_simulation(conv_id, history_turns_data, classifier, embedder):
    db = SessionLocal()
    conv = db.query(Conversation).filter_by(id=uuid.UUID(conv_id)).first()
    if not conv:
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

        # Force lossless AND inject_raw for personal / creative content
        force_lossless = (
            turn.topic_tags and
            ("Creative_&_Media" in turn.topic_tags or "Emotional_Processing" in turn.intent_tags)
        )
        if force_lossless:
            lossless = True
            inject_raw = True          # <-- keep the full raw text
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
            turn.summary_text = summary

        turn.lossless_flag = lossless
        turn.inject_raw = inject_raw

        if lossless:
            triplets = extract_triplets(turn.raw_text)
            for t in triplets:
                if isinstance(t, dict):
                    s, r, o = t.get("subject"), t.get("relation"), t.get("object")
                    if isinstance(s, str) and isinstance(r, str) and isinstance(o, str):
                        s, r, o = s.strip(), r.strip(), o.strip()
                        if s and r and o:
                            handle_triplet(db, s, r, o, str(batch_id))
        db.commit()
    db.close()

# ------------------------------------------------------------------
# Background workers (run after each incremental addition)
# ------------------------------------------------------------------
def run_all_background_workers():
    apply_decay.apply()
    cluster_turns.apply()
    run_reflection.apply()
    monitor_sentinels.apply()
    decay_codex_edges.apply()
    decay_procedural_patterns.apply()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed = load_completed()
    client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")

    classifier = PyTorchClassifier(
        model_path=settings.classifier_model_path,
        schema_path=settings.label_schema_path,
    )
    embedder = classifier.embedder

    # ---- Group curation files by conversation_id ----
    curation_files = sorted(glob.glob(os.path.join(CURATION_DIR, "*.json")))
    if not curation_files:
        print("No curation files found.")
        return

    conv_groups = defaultdict(list)
    for cf_path in curation_files:
        with open(cf_path, "r") as f:
            cdata = json.load(f)
        conv_groups[cdata["original_conversation_id"]].append((cf_path, cdata))

    # ═══ Resume fix: load existing results ═══
    all_results = load_existing_results()

    # ---- Process each conversation ----
    for conv_id, group in tqdm(conv_groups.items(), desc="Conversations"):
        # Sort splits by turn number
        group.sort(key=lambda x: x[1]["split_turn_index"])

        # Load the full conversation ONCE
        with open("data/simulation/simulation_full.jsonl", "r") as f:
            all_turns = [json.loads(line) for line in f if line.strip()]
        conv_turns_all = [t for t in all_turns if t.get("conversation_id") == conv_id]

        previous_n = 0   # how many turns are already in the DB

        for cf_path, curation in group:
            checkpoint_id = curation["evaluation_checkpoint_id"]
            split_n = curation["split_turn_index"]

            # ---- Incremental replay ----
            new_turns = conv_turns_all[previous_n:split_n]
            if previous_n == 0:
                db = SessionLocal()
                # Kill any lingering connections to this database (except ours)
                db.execute(text("""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = current_database()
                      AND pid <> pg_backend_pid();
                """))
                db.execute(text("SET LOCAL lock_timeout = '30s';"))
                db.execute(text("TRUNCATE episodic_memory, conversations, codex_entities, codex_edges, codex_events, codex_snapshots, procedural_memory, session_summaries, context_clusters, sentinel_events, cold_storage, idempotency_keys, rag_documents, rag_chunks RESTART IDENTITY CASCADE"))
                db.commit()
                db.close()
                print(f"  Truncated DB – replaying turns 1–{split_n}")
            else:
                print(f"  Adding turns {previous_n+1}–{split_n} (no truncate)")

            if new_turns:
                replay_simulation(conv_id, new_turns, classifier, embedder)

            # Run background workers to maturity
            run_all_background_workers()
            # ---- Automatic extra experiments for the FLAW conversation ----
            FLAW_CONV_ID = "bb558b5f-5365-5bac-9ed0-07219025b5f2"
            is_flaw = (conv_id == FLAW_CONV_ID)
            if is_flaw:
                run_extras = True
            else:
                run_extras = False
            # ---- Probes ----
            probes = [p for p in curation.get("evaluation_probes", [])
                      if p.get("user_injected_prompt") and p["user_injected_prompt"] != "ENTER_PROBE_HERE"]

            pending_probes = [p for p in probes
                              if f"{checkpoint_id}|{p['probe_id']}" not in completed]

            for probe in tqdm(pending_probes, desc=f"  {checkpoint_id}", leave=False):
                probe_id = probe["probe_id"]

                prompt = probe["user_injected_prompt"]
                expected = probe.get("expected_answer", "")
                if expected == "ENTER_EXPECTED_ANSWER_OR_BLANK":
                    expected = ""

                classification = classifier.classify(prompt)
                embedding = embedder.encode(prompt, convert_to_tensor=False).tolist()
                # CL2: LTM Bias – force memory retrieval for long conversations or uncertain classification
                if classification.context_reliance == "Zero_Shot":
                    if split_n > 10 or classification.max_confidence < 0.95:
                        classification.context_reliance = "Long_Term_Memory"

                # ---- Retrieve once for full ICE (standard) ----
                orchestrator = HybridRetrievalOrchestrator(SessionLocal(), embedder)
                orchestrator.set_budget_from_turn_count(split_n)
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
                    classification=classification,
                )

                # ---- Naive context (raw history) ----
                # all turns up to split_n that are now in the DB
                history_turns_data = conv_turns_all[:split_n]
                naive_context_raw = "\n\n".join(
                    [f"User: {t['prompt']}\nAssistant: {t['response']}" for t in history_turns_data]
                )

                # ---- Vector-only context ----
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
                    vec_text = r.raw_text if r.lossless_flag else (r.summary_text or r.raw_text[:300])
                    words = vec_text.split()
                    if len(words) > 500:
                        vec_text = " ".join(words[:500]) + "…"
                    vector_fragments.append(vec_text)
                db_vec.close()

                # ---- 6 main conditions ----
                conditions = {
                    "control_baseline_generalist": ("naive", SINGLE_MODEL),
                    "control_moe": ("naive", "moe"),
                    "vector_rag_baseline_generalist": ("vector_only", SINGLE_MODEL),
                    "vector_rag_moe": ("vector_only", "moe"),
                    "full_ice_generalist": ("full_ice", SINGLE_MODEL),
                    "full_ice_moe": ("full_ice", "moe"),
                }

                # Optional extra experiments – only added if enabled and (optionally) this checkpoint matches filter
                if run_extras:
                    # Scope modes: each generates new full_ice context with different scope
                    if RUN_SCOPE_EXPERIMENT:
                        for scope_mode, scope_val in [("auto", {"conversation_id": conv_id}),
                                                      ("project", {"conversation_id": conv_id, "cluster_ids": [str(conv_id)]}),
                                                      ("none", {})]:
                            extra_orchestrator = HybridRetrievalOrchestrator(SessionLocal(), embedder)
                            extra_orchestrator.max_retrieval_tokens = 6000
                            extra_fragments = extra_orchestrator.retrieve(
                                classification=classification,
                                conversation_id=conv_id,
                                prompt_embedding=embedding,
                                scope=scope_val,
                            )
                            extra_payload = assemble_prompt(
                                memory_slots=memory_slots,
                                retrieved_fragments=extra_fragments,
                                user_message=prompt,
                                db_session=SessionLocal(),
                                conversation_id=conv_id if scope_mode != "none" else None,
                                classification=classification,
                            )
                            conditions[f"full_ice_scope_{scope_mode}_generalist"] = ("full_ice", SINGLE_MODEL, extra_payload)
                            conditions[f"full_ice_scope_{scope_mode}_moe"] = ("full_ice", "moe", extra_payload)

                    if RUN_HYDE_ABLATION:
                        # Full ICE without HyDE: temporarily disable HyDE rewriting
                        hyde_orchestrator = HybridRetrievalOrchestrator(SessionLocal(), embedder)
                        hyde_orchestrator.max_retrieval_tokens = 6000
                        original_hyde = hyde_orchestrator._hyde_rewrite
                        hyde_orchestrator._hyde_rewrite = lambda *a, **kw: None
                        hyde_fragments = hyde_orchestrator.retrieve(
                            classification=classification,
                            conversation_id=conv_id,
                            prompt_embedding=embedding,
                            scope=scope,
                        )
                        hyde_orchestrator._hyde_rewrite = original_hyde
                        hyde_payload = assemble_prompt(
                            memory_slots=memory_slots,
                            retrieved_fragments=hyde_fragments,
                            user_message=prompt,
                            db_session=SessionLocal(),
                            conversation_id=conv_id,
                            classification=classification,
                        )
                        conditions["full_ice_no_hyde_generalist"] = ("full_ice", SINGLE_MODEL, hyde_payload)
                        conditions["full_ice_no_hyde_moe"] = ("full_ice", "moe", hyde_payload)

                    if RUN_SLIDING_WINDOW_ABLATION:
                        # Full ICE without sliding window (pass conversation_id=None)
                        sw_payload = assemble_prompt(
                            memory_slots=memory_slots,
                            retrieved_fragments=full_ice_fragments,
                            user_message=prompt,
                            db_session=SessionLocal(),
                            conversation_id=None,   # no sliding window
                            classification=classification,
                        )
                        conditions["full_ice_no_sliding_window_generalist"] = ("full_ice", SINGLE_MODEL, sw_payload)
                        conditions["full_ice_no_sliding_window_moe"] = ("full_ice", "moe", sw_payload)

                    if RUN_PROCEDURAL_TOGGLE:
                        # Full ICE without procedural memory – filter out procedural fragments
                        proc_fragments = [f for f in full_ice_fragments if f.source_type != "procedural"]
                        proc_payload = assemble_prompt(
                            memory_slots=memory_slots,
                            retrieved_fragments=proc_fragments,
                            user_message=prompt,
                            db_session=SessionLocal(),
                            conversation_id=conv_id,
                            classification=classification,
                        )
                        conditions["full_ice_no_procedural_generalist"] = ("full_ice", SINGLE_MODEL, proc_payload)
                        conditions["full_ice_no_procedural_moe"] = ("full_ice", "moe", proc_payload)

                record_for_probe = {}
                for cond_name, cond_data in conditions.items():
                    if len(cond_data) == 3:
                        retrieval_mode, model_choice, prebuilt_payload = cond_data
                        final_messages = prebuilt_payload
                    else:
                        retrieval_mode, model_choice = cond_data
                        final_messages = None

                    if model_choice == "moe":
                        best_model, _ = find_best_model(classification.topic_tags, classification.intent_tags)
                        model_to_use = best_model if best_model else SINGLE_MODEL
                    else:
                        model_to_use = SINGLE_MODEL

                    if final_messages is None:
                        if retrieval_mode == "naive":
                            sys_words = naive_context_raw.split()
                            # Keep the LAST 3000 words instead of the first
                            if len(sys_words) > 3000:
                                context = " ".join(sys_words[-3000:]) + "…"
                            else:
                                context = naive_context_raw
                            final_messages = [
                                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                            ]
                        elif retrieval_mode == "vector_only":
                            context = "\n\n".join(vector_fragments)
                            final_messages = [
                                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                            ]
                        else:  # full_ice
                            final_messages = full_ice_prompt_payload

                    start = time.time()
                    try:
                        if retrieval_mode == "full_ice":
                            gen_tokens = 4096
                        else:
                            gen_tokens = 2048

                        resp = client.chat.completions.create(
                            model=model_to_use,
                            messages=final_messages,
                            temperature=0.0,
                            max_tokens=gen_tokens,
                            timeout=120.0,
                        )
                        answer = ""
                        if resp.choices:
                            msg = resp.choices[0].message
                            answer = msg.content or ""
                            if not answer and hasattr(msg, 'reasoning') and msg.reasoning:
                                answer = msg.reasoning.strip()
                        latency = time.time() - start
                    except Exception as e:
                        answer = f"ERROR: {str(e)}"
                        latency = -1

                    # Token count calculation
                    tokens_injected = 0
                    if retrieval_mode == "full_ice":
                        tokens_injected = sum(f.token_count for f in full_ice_fragments)
                    elif retrieval_mode == "vector_only":
                        tokens_injected = sum(len(f.split()) * 1.33 for f in vector_fragments)
                    elif retrieval_mode == "naive":
                        tokens_injected = len(naive_context_raw.split()) * 1.33

                    record_for_probe[cond_name] = {
                        "model_used": model_to_use,
                        "answer": answer,
                        "latency_seconds": round(latency, 3),
                        "judge_score": None,  # will be filled later by blind judge
                        "tokens_injected": round(tokens_injected),
                        "classification": {
                            "topic_tags": classification.topic_tags,
                            "intent_tags": classification.intent_tags,
                            "context_reliance": classification.context_reliance,
                            "max_confidence": classification.max_confidence,
                        },
                    }

                # Store retrieved fragments for full ICE and vector-only
                full_ice_fragment_texts = [{"text": f.text, "source_type": f.source_type, "score": f.score} for f in full_ice_fragments]
                vector_fragment_texts = [{"text": f, "source": "vector"} for f in vector_fragments]

                result = {
                    "metadata": {
                        "checkpoint_id": checkpoint_id,
                        "probe_id": probe_id,
                        "probe_type": probe.get("probe_type", ""),
                        "raw_user_probe": prompt,
                    },
                    "execution_permutations": record_for_probe,
                    "full_ice_fragments": full_ice_fragment_texts,
                    "vector_fragments": vector_fragment_texts,
                }
                all_results.append(result)
                save_completed(checkpoint_id, probe_id)

                # Save incrementally
                master = {
                    "experiment_session_timestamp": datetime.now(timezone.utc).isoformat(),
                    "evaluation_run_results": all_results,
                }
                with open(os.path.join(RESULTS_DIR, "master_results.json"), "w") as f:
                    json.dump(master, f, indent=2)

            # Update the counter for the next split
            previous_n = split_n

    print(f"Phase 2 complete. Results saved to {RESULTS_DIR}/master_results.json")

if __name__ == "__main__":
    main()