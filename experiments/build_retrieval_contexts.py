#!/usr/bin/env python3
"""Build retrieval contexts with dynamic TOP_K based on total history size."""

import json, os, re
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SIMULATION_INPUT = "data/simulation/simulation_full.jsonl"
CURATION_DIR = "experiments/curation_files"
MASTER_RESULTS = "experiments/results_phase2/master_results.json"
OUTPUT_FILE = "experiments/results_phase2/vector_contexts.json"

MAX_TOKENS = 30_000                  # hard cap on combined retrieved context
TOP_K_SMALL = 20
TOP_K_LARGE = 40
LARGE_HISTORY_THRESHOLD = 200_000    # tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "r") as f: return json.load(f)

def parse_checkpoint(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"EC-([a-f0-9]+)-TURN(\d+)", name)
    if m: return m.group(1), int(m.group(2))
    m = re.match(r"EC-([a-f0-9]+)-FULL", name)
    if m: return m.group(1), None
    return None, None

def token_count(text):
    return int(len(text.split()) * 1.33)

# ---------------------------------------------------------------------------
# Load conversation data
# ---------------------------------------------------------------------------
print("Loading conversation turns …")
conv_turns = defaultdict(list)
with open(SIMULATION_INPUT, "r") as f:
    for line in f:
        if not line.strip(): continue
        obj = json.loads(line)
        cid = obj["conversation_id"]
        conv_turns[cid].append((obj["prompt"], obj.get("response", "")))
prefix_to_full = {cid[:8]: cid for cid in conv_turns}
print(f"  {len(conv_turns)} conversations loaded.")

# ---------------------------------------------------------------------------
# Build history-size lookup from master_results.json
# ---------------------------------------------------------------------------
print("Loading history sizes from master_results.json …")
master = load_json(MASTER_RESULTS)
checkpoint_history_tokens = {}
for entry in master["evaluation_run_results"]:
    meta = entry["metadata"]
    cid = meta["checkpoint_id"]
    perms = entry["execution_permutations"]
    naive_cond = "control_baseline_generalist"
    if naive_cond not in perms: naive_cond = "control_moe"
    tokens = perms.get(naive_cond, {}).get("tokens_injected", 0)
    checkpoint_history_tokens[cid] = tokens

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ---------------------------------------------------------------------------
# Process curation files
# ---------------------------------------------------------------------------
curation_files = sorted(
    f for f in os.listdir(CURATION_DIR) if f.endswith(".json") and f.startswith("EC-")
)
all_contexts = {}

for cf_name in tqdm(curation_files, desc="Processing curation files"):
    cf_path = os.path.join(CURATION_DIR, cf_name)
    try:
        curation = load_json(cf_path)
    except Exception as e:
        print(f"  Skipping {cf_name}: {e}")
        continue

    checkpoint_id = curation.get("evaluation_checkpoint_id", cf_name.replace(".json", ""))
    conv_prefix, split_n = parse_checkpoint(cf_name)
    if conv_prefix is None: continue

    cid_full = prefix_to_full.get(conv_prefix)
    if cid_full is None: continue

    turns = conv_turns[cid_full]
    if not turns: continue

    if split_n is None or split_n > len(turns):
        history_turns = turns
    else:
        history_turns = turns[:split_n]

    if not history_turns: continue

    # Determine TOP_K from total history size
    total_tokens = checkpoint_history_tokens.get(checkpoint_id, 0)
    top_k = TOP_K_LARGE if total_tokens >= LARGE_HISTORY_THRESHOLD else TOP_K_SMALL
    print(f"  {checkpoint_id}: history={total_tokens:,} tokens → TOP_K={top_k}")

    # Encode
    turn_texts = [f"User: {p}\nAssistant: {r}" for p, r in history_turns]
    turn_embeddings = embedder.encode(turn_texts, convert_to_tensor=True, show_progress_bar=False)

    probes = curation.get("evaluation_probes", [])
    if not probes: continue

    checkpoint_contexts = {}
    for probe in probes:
        probe_id = probe.get("probe_id")
        prompt = probe.get("user_injected_prompt", "").strip()
        if not prompt or prompt == "ENTER_PROBE_HERE": continue

        prompt_emb = embedder.encode(prompt, convert_to_tensor=True)
        similarities = np.dot(turn_embeddings.cpu().numpy(), prompt_emb.cpu().numpy())
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved = []
        total_tokens = 0
        for idx in top_indices:
            text = turn_texts[idx]
            tokens = token_count(text)
            if total_tokens + tokens > MAX_TOKENS:
                break
            retrieved.append({
                "turn_number": int(idx) + 1,
                "text": text,
                "score": float(similarities[idx])
            })
            total_tokens += tokens

        checkpoint_contexts[probe_id] = {
            "question": prompt,
            "retrieved_turns": retrieved,
            "total_tokens": total_tokens,
        }

    all_contexts[checkpoint_id] = checkpoint_contexts

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_contexts, f, indent=2)

print(f"\nContexts saved to {OUTPUT_FILE}")
print(f"  {len(all_contexts)} checkpoints, {sum(len(c) for c in all_contexts.values())} probes.")