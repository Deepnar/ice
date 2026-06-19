<!-- **ICE‑Mature Blueprint — Complete Implementation Plan**

This is the master plan for turning ICE‑Minimal into ICE‑Mature. We'll build everything you listed, step by step, and then run Experiment 2 on the four long conversations with all features active. Let's go.

---

## Scope

### ✅ Included (P0 + P1 + your hand‑picked P2)

| # | Feature | Subsystem |
|---|---------|-----------|
| CL1 | Embedder → Qwen3‑Embedding‑0.6B | Classifier |
| CL2 | LTM bias (turn‑count + confidence override) | Classifier |
| CL4 | Dynamic token budget (turn‑scaled) | Retrieval |
| CL5 | Creative & Software hard LTM overrides | Classifier |
| CL6 | Hybrid rule engine **before** ML classifier | Classifier |
| CL7 | Context‑aware classification (last 3 turns) | Classifier |
| CX1 | Tiny NER MLP (BIO tagger) | Codex |
| CX2 | Vector‑based Codex entity matching (fuzzy) | Codex |
| CX4 | Contradiction detection + new‑fact boosting | Codex |
| CX5 | Codex extraction for code entities | Codex |
| CU1 | Cluster limit → 30 turns | Clustering |
| CU2 | Vector‑based cluster merging | Clustering |
| ML1 | Decay‑stress experiment loop | Memory Lifecycle |
| ML2 | Bookmark boost (verify 1.5× in RRF) | Retrieval |
| ML3 | Bookmark → immediate Codex extraction | Memory Lifecycle |
| ML4 | Per‑turn cap override for document turns | Retrieval |
| ML5 | Conversation‑level batch summarisation | Memory Lifecycle |
| UC1 | Thumbs up/down logging | User Control |
| UC3 | Conversation scoping isolation (None = incognito1) | User Control |

---

## 🔨 Implementation Order

Some features depend on others; we'll group them to minimise re‑training and re‑testing.

**Phase A: Classifier core improvements (CL1, CL2, CL4, CL5, CL6, CL7)**  
**Phase B: Codex / Entity extraction (CX1, CX2, CX4, CX5)**  
**Phase C: Memory lifecycle & Retrieval (ML1–ML5, CU1, CU2)**  
**Phase D: User control & Scoping (UC1, UC3)**  
**Phase E: Experiment 2 script (integrates all of the above)**

---

## Phase A: Classifier & Embedding Overhaul

### CL1 – Swap embedder to Qwen3‑Embedding‑0.6B

**Files to create/modify:**  
- `src/classifier/classifier.py`  
- `src/classifier/dataset.py`  
- `src/retrieval/orchestrator.py`  
- `scripts/training/train_classifier.py`  
- `scripts/training/fine_tune.py`

**Steps:**

1. **Install the model**  
   ```bash
   uv pip install sentence-transformers
   # Model will be downloaded automatically on first use
   ```

2. **Change the embedding model in `PyTorchClassifier`**  
   In `src/classifier/classifier.py`:
   ```python
   self.embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
   ```

3. **Add instruction prefix for classification**  
   Still in `classify()`:
   ```python
   prefixed_prompt = f"Given a user prompt, classify its intent: {prompt}"
   embedding = self.embedder.encode(prefixed_prompt, convert_to_tensor=True).unsqueeze(0)
   ```

4. **Re‑build training data** with the new embedder  
   In `scripts/training/build_training_data.py`, just run it again — the dataset class will use the new embedder automatically because it imports `SentenceTransformer`. Actually, `ICEClassifierDataset` uses its own embedder internally; change it to also use Qwen3.
   ```python
   # In src/classifier/dataset.py
   model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
   ```
   Then re‑run:
   ```bash
   uv run python scripts/training/build_training_data.py
   ```

5. **Re‑train the MLP head**  
   ```bash
   uv run python scripts/training/train_classifier.py --seed 42 --epochs 30 --model_path models/classifier/ice_classifier_v3_qwen.pt
   ```
   (You can keep the old MiniLM model as a fallback.)

6. **Update fine‑tuning script** to use the new embedder  
   In `scripts/training/fine_tune.py`, change:
   ```python
   embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
   ```

7. **Update retrieval orchestrator embedder**  
   The orchestrator already receives `embedder` from the classifier, so it's automatically updated. But if you instantiate a separate embedder anywhere (like in tests), change it there too.

8. **Re‑run the probe labeling** (if needed) – but for Experiment 2 we'll use the existing probes and judge, so no need to re‑label.

**Verification:**  
Run `scripts/training/test_classifier.py` with the new model. Watch confidence scores; they should generally be higher and more stable. AND fine tune it if needed by generating synthetic promts if need or which ever methos feels correct.

---

### CL5 – Creative & Software hard LTM overrides (already partially done)

In `src/classifier/classifier.py`, the `classify()` method already has:
```python
if "Creative_&_Media" in topic_tags:
    context_reliance = "Long_Term_Memory"
```

We need to add a similar rule for `Software_&_Tech` when the prompt contains personal possessives (`"my"`, `"our"`, `"this"`).  
Add this after the Creative override:
```python
if "Software_&_Tech" in topic_tags:
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ("my ", "our ", "this ")):
        context_reliance = "Long_Term_Memory"
```
This is quick and effective.

---

### CL2 – LTM bias via turn‑count + confidence override

**Where:** `src/api/main.py` (the FastAPI proxy) and the Phase 2 evaluation script (we'll need a utility for experiment use).

Since the proxy already has a `conversation_id`, we can implement the logic directly inside the retrieval decision.

In `src/api/main.py`, after classification:
```python
if result.context_reliance == "Zero_Shot":
    turn_count = db.query(EpisodicMemory).filter_by(
        conversation_id=conversation_id
    ).count()
    if turn_count > 10 or result.max_confidence < 0.95:
        result.context_reliance = "Long_Term_Memory"
```
For the evaluation script (`phase2_run_evaluation_matrix.py`), we'll add the same logic before calling `orchestrator.retrieve()`. Since the simulation uses a synthetic conversation and has the full turn list, we can compute `turn_count = split_n` (the current turn index).

**Verification:** Test with a short conversation — a Zero_Shot prompt should remain Zero_Shot. In a long conversation, it should be overridden to LTM.

---

### CL4 – Dynamic token budget (turn‑scaled)

**Where:** `src/retrieval/orchestrator.py` and the evaluation script.

In `HybridRetrievalOrchestrator`, add a method:
```python
def set_budget_from_turn_count(self, turn_count: int):
    self.max_retrieval_tokens = min(10000, max(3000, turn_count * 50))
```
In the proxy `main.py`, after determining retrieval is needed, call:
```python
orchestrator.set_budget_from_turn_count(turn_count)
```
In the evaluation script, we can also compute `turn_count = split_n` and set it before retrieval.

**Verification:** Check logs to see that token counts increase for later checkpoints of long conversations.

---

### CL6 – Hybrid rule engine before ML classifier

**Purpose:** Catch obvious patterns with deterministic rules, bypassing the MLP for those cases. This improves accuracy and speed for clear‑cut prompts.

**Where:** `src/classifier/classifier.py`, inside `classify()`.

**Rules to implement:**

| Rule | Condition | Output |
|------|-----------|--------|
| Code block | contains ` ``` ` | topic=`Software_&_Tech`, intent=`Generation`, context=`Zero_Shot` (or LTM if scoped) |
| Emotional | starts with "I feel", "I'm feeling", etc. | intent=`Emotional_Processing` |
| Null noise | length < 5 chars or only punctuation | topic=`Null_Noise`, intent=`Casual_Banter`, context=`Zero_Shot` |
| Meta AI | "how do I prompt you", "what model are you", etc. (regex) | topic=`Meta_AI`, intent=`Factual_Retrieval`, context=`Zero_Shot` |
| Document paste | prompt length > 2000 words and low dialogue density | flag as potential document, not a rule per se, but let's not override classifier; just set a flag for later use (ML4). We'll implement ML4 separately. |

**Implementation:**  
Before the ML inference, check these rules. If a rule matches with high confidence, directly return a `ClassificationResult` without running the MLP.

**Verification:** Test a few obvious prompts; they should bypass the model and return instantly.

---

### CL7 – Context‑aware classification (last 3 turns)

**Purpose:** Feed the last 3 user+assistant turns along with the current prompt into the embedder, so the classifier can disambiguate.

**Where:** `src/classifier/classifier.py` and `src/api/main.py`.

**Implementation:**
- Modify `classify()` to accept an optional `context_text: str` parameter.
- If provided, prefix it to the prompt: `combined = f"{context_text}\nCurrent: {prompt}"`.
- In the proxy `main.py`, fetch the last 3 turns from `episodic_memory` and pass them.

For the evaluation script, we have `history_turns_data` available; we can form the context.

**Verification:** An ambiguous prompt like "I hate this" should now be correctly routed if preceded by a technical turn.

---

## Phase B: Codex & Entity Extraction

### CX1 – Tiny NER MLP (BIO tagger)

**Purpose:** Replace regex entity extraction with a learned model that catches lowercase, multi‑word, and misspelled entities.

**Training data:** We'll use **weak supervision** from existing Codex triplets. For every episodic turn where the Codex Extractor produced triplets, we search for the subject/object strings in the prompt, label tokens with B‑ENT/I‑ENT.

**Files to create:**
- `scripts/training/build_ner_data.py`
- `src/ner/model.py` (BIO tagger architecture)
- `src/ner/train.py`
- `src/ner/inference.py`

**Model:** A single linear layer on top of the frozen Qwen3‑Embedding token embeddings (same as intent classifier but token‑level). Output: 3 logits (B, I, O).

**Training:** CrossEntropyLoss over tokens, with a mask for non‑entity tokens (weighted). Train for a few epochs.

**Integration:** In `HybridRetrievalOrchestrator._codex_graph`, replace the `re.findall(r'\b[A-Z]...')` with a call to the NER model. The NER model returns entity spans, which we then look up in `codex_entities` (using fuzzy matching from CX2).

**Verification:** Test with a prompt like "what is the goo blade?" and ensure "goo blade" is extracted.

---

### CX2 – Vector‑based Codex entity matching (fuzzy)

**Purpose:** Even with the NER model, entity names may be misspelled or aliased. We'll use cosine similarity over entity embeddings to find the best match.

**Implementation:**  
- Add an `embedding` column to `codex_entities` (Vector 384).  
- When a new entity is created, embed its `canonical_name` using the Qwen3 embedder.  
- In `_codex_graph`, for each extracted span, embed it and query `codex_entities` for the nearest neighbor (cosine similarity > 0.85 threshold). If a match is found, use that entity; otherwise, treat it as a new unknown entity (but don't create one automatically during retrieval).

**Verification:** After misspelling "Kael" as "Keal", the system should still find the entity.

ALSO we had also promts in my probes like asking what the names of characters in the 1000 turn convo were, but we got wrong answers as neither did the codex work, nor the system was able to understand the update of info, whihc was basicaly the job of codex, right? but if we look at the promt again, and also how we are toggling the codex retrieval, its based on if some named things comes up right, but that in way still doesnt help with the name change update, as if in the probe askig to name the characters, the codex wont be toggled as there are not names to be retreived, same for if we enable this for coding too, we have to think of a way for this.

---

### CX4 – Contradiction detection + new‑fact boosting

**Purpose:** When a fact contradicts an existing edge, expire the old one but **boost** the strength of the new fact so it becomes active faster (overriding decay).

**Current code** already handles contradiction (sets `valid_until`). We need to add the boost:
- In `handle_triplet`, when creating a new contradictory edge, set its `strength` to `3.0` (instead of 1.0), making it immediately `active` if your threshold is 2.0. This is a one‑line change.

**Verification:** Simulate a fact update; the new edge should appear with high strength.

---

### CX5 – Codex extraction for code entities

**Purpose:** Currently the extractor prompt is generic; we want it to also extract code‑specific entities (function names, class names, libraries) and create relationships like "uses", "imports", "extends".

**Implementation:**  
- Extend the `extract_triplets` system prompt with examples of code entities.  
- Optionally, use a separate extraction model (or the same 1.5B model with a code‑specific prompt) when the conversation topic is `Software_&_Tech`.  
- The extracted triplets will flow through the same `handle_triplet` logic.

**Verification:** After processing a code‑heavy turn, check `codex_entities` for function names.

---

## Phase C: Memory Lifecycle & Retrieval

### ML1 – Decay‑stress experiment loop

**Purpose:** Prove that decay and reinforcement work correctly. In Experiment 2, after each turn insertion, run the decay worker multiple times and then run probes.

**Implementation in evaluation script:**  
After replaying a batch of turns and running `post_simulation`, execute a loop:
```python
for _ in range(5):   # simulate 5 decay cycles
    apply_decay.apply()
    time.sleep(2)
```
Then run the probes. This will show whether frequently retrieved memories survive decay.

**Verification:** Probes targeting old but frequently accessed info should retain high scores; rarely accessed info should degrade.

---

### ML2 – Bookmark boost (verify)

**Where:** `src/retrieval/orchestrator.py` in `_rows_to_fragments`.

The code already has:
```python
if getattr(row, "is_bookmarked", False):
    score_val *= 1.5
```
This means the 1.5× multiplier is already implemented. No change needed, just confirm it works during testing.

---

### ML3 – Bookmark triggers immediate Codex extraction

**Where:** `src/api/routers/user_control.py` in the `bookmark_turn` endpoint.

Currently, it calls:
```python
extract_codex.delay(batch_id=str(turn.batch_id))
```
That's correct. The only missing piece is that the extraction should bypass GPU idle checks (i.e., high priority). We can modify `extract_codex` to accept a `priority` flag that skips the `is_gpu_busy()` check. This is a small change in `codex_extractor.py`.

**Verification:** Bookmark a turn; check that Codex entities appear quickly.

---

### ML4 – Per‑turn cap override for document turns

**Purpose:** When a turn is a massive pasted document, we need to inject its full text without the 500‑word cap.

**Implementation steps:**

1. **Flag document turns in post‑flight.**  
   In `src/workers/post_flight.py`, inside `evaluate_turn`, after determining lossless:
   ```python
   # If the raw_text is very long and has low dialogue density, mark as document
   word_count = len(turn.raw_text.split())
   if word_count > 2000 and turn.raw_text.count("Assistant:") < 3:
       turn.is_document = True
   ```
   You'll need to add an `is_document` column to `episodic_memory` (via Alembic migration).  
   (`sa.Column('is_document', sa.Boolean, default=False)`)

2. **Override cap in orchestrator.**  
   In `_rows_to_fragments`, check `if row.is_document: text = row.raw_text` (no truncation).  
   But still respect the overall token budget.

**Verification:** Simulate a pasted article; retrieved fragments should include the full article text.

---

### ML5 – Conversation‑level batch summarisation

**Purpose:** For very old turns, merge summaries of consecutive turns into a single batch summary, reducing noise.

**Implementation:**

1. **Create a Celery task `batch_summarise`** that runs periodically.  
2. **Group old turns** (decay_score < 0.3, not lossless) by conversation and time window.  
3. **Call the background model** with a prompt: "Summarise the following conversation block into 2‑3 paragraphs, preserving all names, numbers, and decisions."  
4. **Store the batch summary** in a new table `batch_summaries` (with `conversation_id`, `start_turn_id`, `end_turn_id`, `summary_text`, `embedding`).  
5. **Modify retrieval** to also query `batch_summaries` when an episodic fragment would be too noisy; replace individual summaries with the batch summary if it's more compact.

**Verification:** After running, check that old sections of the Flaw conversation are represented by clean block summaries.

---

### CU1 – Cluster limit increase

**Where:** `src/workers/clustering.py`

Change the limit from `LIMIT 30` (already there? Actually currently it's 30). The architecture says 10 turns per cluster; we'll increase it to 50. But more importantly, the clustering worker should assign **up to 50 turns per cluster** without creating too many tiny clusters. We'll also allow clusters to grow beyond 50 if they're the same topic.

**Implementation:**
```python
# In cluster_turns()
unassigned = db.query(EpisodicMemory).filter_by(cluster_id=None).limit(100).all()
# Then when creating clusters, don't impose a hard size limit; just distribute.
```

We'll also modify the clustering prompt to not enforce a strict maximum.

Also we had to reduce the clustering due to llm input token limitations, so think a soln for that.

**Verification:** After clustering, check that large conversations have few, meaningful clusters, not dozens. 

---

### CU2 – Vector‑based cluster merging

**Purpose:** Avoid duplicate clusters like "AI-driven OS" and "OS based on AI".

**Implementation:**

1. **Add an `embedding` column** to `context_clusters` (vector 384).  
2. **When a cluster is created**, embed its name and description (using Qwen3 embedder) and store it.  
3. **Periodic merge task**: for each cluster, find its nearest neighbor by cosine similarity. If similarity > 0.85, merge them (reassign turns from one to the other, delete the empty cluster).  
4. **Merge logic**: choose the cluster with more turns as the surviving one; update its embedding to the average of the two.

**Verification:** Create two similar clusters manually, run the merge task, and verify they become one.

---

## Phase D: User Control & Scoping

### UC1 – Thumbs up/down logging

**Purpose:** Collect user feedback for future fine‑tuning.

**Where:**  
- New Alembic migration to add `thumbs` column to `episodic_memory` (values: `up`, `down`, `neutral`, nullable).  
- API endpoint `POST /user-control/turns/{turn_id}/thumbs` with body `{"value": "up"}`.  
- Store the feedback.

**Implementation:** Already partially designed in `user_control.py`. Add the column and endpoint.

**Verification:** Send a thumbs up via API; verify the DB record.

---

### UC3 – Conversation scoping isolation (None = incognito)

**Purpose:** When `memory_scope_type = "none"`, the conversation should be fully isolated: no retrieval from other conversations, and its own turns are not retrievable by any other conversation.

**Implementation:**

In `src/api/main.py`, when `memory_scope_type == "none"`, the orchestrator should return an empty list. Also, when storing new turns, ensure they are not assigned to any cluster, and in retrieval, the scope filter should exclude them from global searches.

**Changes:**
- In `chat_completions`, if `conv_row.memory_scope_type == "none"`, skip retrieval entirely.
- In `store_turn_async`, if scope is "none", set `cluster_id = None` (already done by not assigning a cluster).
- In `HybridRetrievalOrchestrator`, add a filter: for "auto" scope, exclude turns from conversations with `memory_scope_type = "none"` (add a subquery to check). This ensures incognito conversations don't leak.

**Verification:** Create two conversations, one incognito. Search from the other; incognito turns should not appear.

---

## Phase E: Experiment 2 Script

**New script:** `experiments/phase2b_mature_experiment.py`

This script will:
1. **Load only the four long conversations** (Flaw GPT, Flaw Claude merged, Shinchan, Laptop decision — or whichever you pick).
2. **Replay them chronologically** with all new features enabled.
3. **Run decay loops** at each checkpoint (ML1).
4. **Use the enhanced classifier** (CL1, CL2, CL4, CL5, CL6, CL7).
5. **Use the NER model** for Codex search.
6. **Record all metrics** (like Phase 2 but with additional logging of feature toggles).

We'll adapt the existing `phase2_run_evaluation_matrix.py`. The structure is already there; we just need to:
- Import the new classifier and NER.
- Set feature flags.
- Change the retrieval budget dynamically.
- Add decay stress.

I'll produce this script once all features are implemented.

---

## 🗓️ Suggested Work Sequence

**Day 1:** CL1, CL2, CL4, CL5, CL6, CL7 — get the improved classifier running. Re‑train the MLP.  
**Day 2:** CX1 (NER model) — build training data, train, integrate.  
**Day 3:** CX2, CX4, CX5 — fuzzy matching and code extraction.  
**Day 4:** ML1, ML2, ML3, ML4, ML5 — memory lifecycle improvements.  
**Day 5:** CU1, CU2 — clustering fixes.  
**Day 6:** UC1, UC3 — user control endpoints.  
**Day 7:** Assemble Experiment 2 script, test with one conversation.  
**Day 8:** Run full Experiment 2, collect results.

---

I'm ready to provide the exact code for any of these steps whenever you are. Let's start with **Phase A, CL1**.


 -->



This "ICE-Mature" blueprint serves as the core plan for Experiment 2, integrating all planned improvements. I have thoroughly rewritten it to ensure it functions as a robust design document that carefully addresses all your previous concerns.

This comprehensive blueprint stands as a complete design document. It begins with a clear summary, delves into each section, and provides solutions for all the open questions you raised.

---

## 📋 **ICE-Mature Blueprint — Experiment 2 Implementation Plan**

This is the master plan for evolving ICE-Minimal into ICE-Mature. We will build out the listed features step-by-step, then run Experiment 2 across four long-form conversations with all new features enabled.

---

### 📌 Scope & Core Conflict Resolution: Dimensionality Alignment

Before implementing any features, we must resolve the most critical architectural conflict: **embedding vector dimensionality**.

-   **Issue**: `Qwen3-Embedding-0.6B` outputs 1024-dimensional vectors by default, whereas the entire ICE system (database, classifier, retrieval) is built around 384 dimensions.
-   **Solution**: Utilize MRL (Matryoshka Representation Learning) to **truncate the Qwen3 embedder's output to 384 dimensions** (`truncate_dim=384`). This allows us to benefit from the new model's superior quality while maintaining compatibility with the existing database.

#### 🗺️ Experiment 2 Data Strategy: Ensuring Comparability

*   **Experiment 1 Data (Baseline)**: Remains unchanged; uses MiniLM vectors.
*   **Experiment 2 Data (Validation of Improvements)**: **Fully re-embed** all relevant data—including `episodic_memory`, `procedural_memory`, and `rag_chunks`—using Qwen3 vectors.
*   **Paper Narrative**: Frame Experiment 2 as a standalone "in-depth validation experiment." It employs a semantically richer embedding model to test the system's peak performance—following comprehensive upgrades—against the most challenging (ultra-long) subset of conversations.

---

### 🔨 **Phase A: Classifier & Embedding Layer Overhaul**

The goal of these improvements is to make the classifier smarter, more accurate, and capable of understanding conversational context. #### **CL1 – Replacing the Embedder (Qwen3-Embedding-0.6B)**

1.  **Implementation Steps**:
*   **Model Loading**: In `PyTorchClassifier`, replace the embedder with `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", truncate_dim=384)`.
*   **Instruction Prefix**: You **must** use the officially recommended query instruction prefix to maximize embedding quality. Example: `prefixed_prompt = f"Given a user prompt, classify its intent: {prompt}"`. 
*   **Training Data Reconstruction**: You **must** regenerate the `training_data.jsonl` file—used to train the classifier's MLP head—using the new Qwen3 embedder. 
*   **MLP Head Retraining**: Train the classifier's MLP head from scratch using the reconstructed data (`train_classifier.py`). 
*   **Fine-tuning Script Update**: Ensure the `fine_tune.py` script also utilizes the new Qwen3 embedder.
2.  **Dataset and Fine-tuning**:
*   **Dataset Splitting**: Divide the 20k annotated dataset into training, validation, and test sets to ensure robust evaluation. 
*   **Fine-tuning Strategy**: The initial training (Experiment 2) utilizes the full 20k dataset. Post-deployment, fine-tuning via `fine_tune.py` relies primarily on high-quality, human-corrected data collected from the `curated_labels` table, enabling the model to rapidly adapt to user-specific patterns.
3.  **Validation**:
*   **Accuracy**: Run `test_classifier.py` to verify whether the new model's accuracy, precision, and recall on the test set are significantly superior to those of the old model. 
*   **Confidence**: Check whether the output `max_confidence` is generally higher and more stable.

#### **CL5 – Mandatory LTM for "Creative" and "Software" Categories**

1.  **Issue**: Prompts related to creative work or programming—even when referring to ongoing projects—may be misclassified as `Zero_Shot`, preventing the system from retrieving relevant memories. 2.  **Solutions**:
*   **`Creative_&_Media` LTM Enforcement**: Remains unchanged; for any prompt classified as "Creative & Media," the `context_reliance` is **unconditionally** forced to `Long_Term_Memory`. 
*   **`Software_&_Tech` LTM Enforcement**: When a prompt contains personalized or referential terms (e.g., "my," "our," "this," "the"), the `context_reliance` is **forced** to `Long_Term_Memory`. This ensures memory retrieval for queries such as "this project" or "our code."

#### **CL2 – LTM Bias Based on Turn Count and Confidence**

1.  **Issue**: In long conversations, prompts are sometimes misclassified as `Zero_Shot`, causing queries that should have triggered memory retrieval to miss critical context.
2.  **Solutions**:
*   **Turn Count Override**: If the conversation turn count exceeds a threshold (e.g., > 10 turns) and the classifier outputs `Zero_Shot`, **override** the classification to `Long_Term_Memory`. In long conversations, the prior probability of relying on historical context is far higher than that of a zero-shot scenario. 
*   **Low Confidence Override**: If the classifier's `max_confidence` falls below a threshold (e.g., < 0.95), override `Zero_Shot` to `Long_Term_Memory` regardless of the turn count. This captures ambiguous queries that are neither clearly `Zero_Shot` nor clearly `Long_Term_Memory`.
3.  **Implementation Location**: Implement this logic at the API layer (`src/api/main.py`) and in experimental scripts, prior to invoking the retriever.

#### **CL4 – Dynamic Token Budget**

1.  **Issue**: A fixed budget (e.g., 5,000 tokens) is excessive for short conversations yet insufficient for long ones.
2.  **Solution**: Dynamically adjust the retrieval budget based on conversation depth (turn count). *   **Budget Formula**: `budget = min(10000, max(3000, turn_count * 50))`. 
*   **Strategy**: Long conversations receive more context (up to 10,000 tokens), while short conversations (< 60 turns) receive less (3,000 tokens). This improves the TUR for short conversations and ensures that critical information in long conversations is not truncated.
3.  **Implementation Location**: Add the `set_budget_from_turn_count` method to the retriever and call it from the API layer and experiment scripts.

Below is the complete implementation specification for **DI3 (Dynamic Intent Inferencer)**, presented in a structured format suitable for handing off to another AI for coding.

---

## 📋 DI3 (Dynamic Intent Inferencer) – Complete Implementation Specification

### 1. Overview

| Attribute | Value |
|------|------|
| **Component Name** | DI3 – Dynamic Intent Inferencer |
| **File Location** | `src/classifier/di3.py` |
| **Responsibility** | Acts as a fast, signal-based pre-classifier before invoking the ML classifier |
| **Input** | `prompt: str`, `conversation_history: List[str]`, `conversation_length: int` |
| **Output** | `ClassificationResult` or `None` (if DI3 cannot make a determination) |
| **Priority** | P1 |
| **Status** | Pending Implementation |

---

### 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         classify(prompt)                        │
│                              │                                  │
│                              ▼                                  │
│                  ┌─────────────────────┐                        │
│                  │   DI3 (Pre‑flight)   │                        │
│                  │  Signal Extraction   │                        │
│                  │  & Density Analysis  │                        │
│                  └──────────┬──────────┘                        │
│                             │                                   │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                    │
│              ▼                             ▼                    │
│       High Confidence                  Low Confidence           │
│              │                             │                    │
│              ▼                             ▼                    │
│   Return ClassificationResult      Pass to ML Classifier        │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3. Signal Extraction and Density Calculation

DI3 scans the prompt and calculates five independent density signals.

#### 3.1 Code Density

**Purpose**: Detects whether the prompt contains code or code-related content. **Implementation Method**:

| Feature | Weight |
|------|------|
| Contains ` ``` ` | +0.4 |
| Contains `=`, `==`, `!=`, `>`, `<` | +0.1 |
| Contains `def`, `class`, `function`, `import` | +0.1 |
| Contains `{` or `}` | +0.1 |
| Contains `;` | +0.1 |
| Contains `print`, `return`, `if`, `else`, `for`, `while` | +0.05 |

**Code Density Threshold**: `0.3`

#### 3.2 Sentiment Density

**Purpose**: To detect whether the prompt expresses an emotional state.

**Implementation Method**: Scan a predefined list of sentiment-related words.

- **Sentiment Word List**: `feel`, `felt`, `feeling`, `frustrated`, `upset`, `angry`, `happy`, `sad`, `love`, `hate`, `excited`, `worried`, `scared`, `tired`, `overwhelmed`, `depressed`, `anxious`

| Feature | Weight |
|------|------|
| Contains any sentiment word | +0.1 per word |
| Contains "I feel" | +0.2 |
| Contains "I’m" + sentiment word | +0.15 |

**Sentiment Density Threshold**: `0.4`

#### 3.3 Meta-Density

**Purpose**: To detect whether the prompt asks questions about the AI ​​itself. **Implementation Method**:

| Feature | Weight |
|------|------|
| Contains `you`, `your`, `model` | +0.1 |
| Contains `prompt`, `prompting` | +0.15 |
| Contains `how do I` + `prompt`/`use` | +0.2 |
| Contains `what model`, `which model` | +0.2 |

**Meta-density Threshold**: `0.2`

#### 3.4 Noise Density

**Purpose**: To detect whether the prompt is disorganized or constitutes a "test" message.

**Implementation Method**:

| Feature | Weight |
|------|------|
| Contains only punctuation marks | +0.6 |
| Contains repeated characters (`aaaa`, `bbbb`) | +0.2 |
| Length < 5 characters | +0.2 |
| Contains random key sequences (`asdf`, `qwerty`) | +0.3 |

**Noise Density Threshold**: `0.8`

#### 3.5 Reference Density

**Purpose**: To detect whether the prompt contains references to the subject currently under discussion (which increases the need for memory retrieval). **Implementation Method**:

| Feature | Weight |
|------|------|
| Contains `this` | +0.15 |
| Contains `that` | +0.1 |
| Contains `it` | +0.05 |
| Contains `these`, `those` | +0.1 |
| Contains `the` (if conversational context is clear) | +0.05 |

**Reference Density Threshold**: `0.2` (in long conversations, this threshold drops to `0.1`, and LTM is enforced)

---

### 4. Decision Logic

Once signals are extracted, DI3 executes the following decision process:

```python
def infer(prompt: str, conversation_length: int, history: List[str]) -> Optional[ClassificationResult]:
signals = extract_signals(prompt)

# Signal 1: Meaningless/Noise
if signals['noise_density'] > 0.8:
return ClassificationResult(
topic_tags=["Null_Noise"],
intent_tags=["Casual_Banter"],
context_reliance="Zero_Shot",
confidence=0.95
)

# Signal 2: Code (returns classification result if code density is high)
if signals['code_density'] > 0.3:
return ClassificationResult(
topic_tags=["Software_&_Tech"],
intent_tags=["Generation"],
context_reliance="Long_Term_Memory" if conversation_length > 0 else "Zero_Shot",
confidence=0.9
)

# Signal 3: Sentiment (returns classification result if sentiment density is high)
if signals['sentiment_density'] > 0.4:
return ClassificationResult(
topic_tags=["Lifestyle_&_Health", "Social_&_Relationships"],
intent_tags=["Emotional_Processing"],
context_reliance="Long_Term_Memory" if conversation_length > 5 else "Zero_Shot",
confidence=0.85
)

# Signal
4: Meta (if meta density is high, return classification result)
if signals['meta_density'] > 0.2:
return ClassificationResult(
topic_tags=["Meta_AI"],
intent_tags=["Factual_Retrieval"],
context_reliance="Zero_Shot",
confidence=0.9
)

# Signal 5: Reference (if reference density is high, force LTM)
if signals['reference_density'] > 0.2:
return ClassificationResult(
topic_tags=[],  # Let the ML classifier determine the topic, but force LTM
intent_tags=[],
context_reliance="Long_Term_Memory",
confidence=0.7  # Low confidence, but force LTM
)

# No decisive signal detected → Pass to ML classifier
return None
```

---

### 5. Integration with the Classifier

In `src/classifier/classifier.py`:

```python
def classify(self, prompt: str, conversation_history: List[str] = None, conversation_length: int = 0) -> ClassificationResult:
# Step 1: Run DI3 pre-classifier
di3_result = run_di3(prompt, conversation_length, conversation_history)

if di3_result is not None:
# DI3 made a decision → return immediately
return di3_result

# Step 2: DI3 undecided → run ML classifier
return self._run_ml_classifier(prompt)
```

---

### 6. Configuration File (`config.py`)

Add the following configuration items:

```python
# DI3 Configuration
DI3_ENABLED = True                # Enable/Disable DI3
DI3_CODE_DENSITY_THRESHOLD = 0.3
DI3_SENTIMENT_DENSITY_THRESHOLD = 0.4
DI3_META_DENSITY_THRESHOLD = 0.2
DI3_NOISE_DENSITY_THRESHOLD = 0.8
DI3_REFERENCE_DENSITY_THRESHOLD = 0.2
DI3_LTM_REFERENCE_DENSITY_THRESHOLD = 0.1
```

---

### 7. Logging

DI3 should log the following for debugging and analysis:

- `{"event": "di3_decided", "signal": "code", "confidence": 0.9}`
- `{"event": "di3_passed_to_ml", "signal_scores": {...}}`

---

### 8. Test Cases

| Prompt | Expected Behavior |
|--------|----------|
| `def hello():\n    print("world")` | DI3 → `Software_&_Tech`, `Generation` |
| `I feel really frustrated and overwhelmed.` | DI3 → `Emotional_Processing` |
| `How do I prompt you better?` | DI3 → `Meta_AI` |
| `asdfghjkl` | DI3 → `Null_Noise` |
| `What is that?` | DI3 → Fallback (Force ML) | LTM, passed to ML classifier |
| `Can you fix this code?` | DI3 → Undecided, passed to ML classifier |

---

### 9. File Structure

```
src/
└── classifier/
├── classifier.py      # Modified: DI3 integration
├── di3.py             # New: DI3 implementation
├── di3_signals.py     # New: Signal extractor
├── di3_config.py      # New: Configuration loader
└── di3_logger.py      # New: DI3 event logger
```

---

### 10. Implementation Order

| Step | File | Content |
|------|------|------|
| 1 | `di3_signals.py` | Implement signal extraction functions (code, sentiment, meta, noise, anaphora) |
| 2 | `di3.py` | Implement decision tree logic |
| 3 | `di3_config.py` | Load configuration from `config.py` |
| 4 | `di3_logger.py` | Set up structured logging |
| 5 | `classifier.py` | Integrate DI3 into `classify()` |
| 6 | `test_di3.py` | Run unit tests |

---

### 11. Considerations

1.  **DI3 is a pre-classifier, not a replacement.** When uncertain, it should hand over control to the ML classifier rather than forcing an incorrect result.
2.  **Signal density is a continuous value, not binary.** Signal density should be calculated as a value between 0 and 1 and evaluated against thresholds.
3.  **Contextual thresholds may vary:** Adjust signal thresholds based on conversation length (e.g., "I feel" carries more contextual significance in long conversations).
4.  **Use a feedback loop:** Track DI3's decisions versus the ML classifier's decisions. Periodically review instances where DI3's decisions proved incorrect and adjust signal thresholds accordingly. ---

### 12. Sample Code (AI-Generated Template)

```python

# src/classifier/di3_signals.py

def compute_code_density(text: str) -> float:

""Returns a score between 0.0 and 1.0, representing the similarity between the text and the code."""

features = {

'```': 0.4,

'=': 0.05,

'def': 0.1,

'class': 0.1,

'function': 0.1,

'import': 0.1,

'if': 0.05,

'else': 0.05,

'for': 0.05,

'while': 0.05,

'return': 0.05,

'{': 0.1,

'}': 0.1,

';': 0.1,

}
score = 0.0

for token, weight in features.items():

if token in text:

score += weight

return min(score, 1.0)

# Similarly, implement compute_sentiment_density, compute_meta_density, compute_noise_density, compute_reference_density

```

---

**Please let me know if you need to adjust any thresholds or add new signals to DI3.**

#### **CL7 – Context-Aware Classification (Last 3 Turns)**

1.  **Problem**: Prompts in isolation can be ambiguous. For example, "I hate this" could refer to a code error or a plot point in a story. 2.  **Solution**: Use the last three dialogue turns (user + assistant) as context and input them into the embedder **along with** the current prompt. 
*   **Code Modification**: Modify the `classify()` method to accept an optional `context_text` parameter. If provided, concatenate the context with the current prompt before encoding. 
*   **API Integration**: In `main.py`, retrieve the last three dialogue turns from the database as context before calling the classifier. 
*   **Experiment Integration**: In the experiment script, construct the context from `history_turns_data`.
3.  **Validation**: Test using ambiguous prompts like "I hate this." Without context, it might be classified as `Emotional_Processing`; with technical context, it should be classified as `Troubleshooting`.

---

### 🔧 **Phase B: Codex & Entity Extraction Overhaul**

These improvements aim to fix two fundamental flaws in the Codex Knowledge Graph (KG): **missing entities** and **entity confusion**.

#### **CX1 – Micro NER MLP (BIO Tagger)**

1.  **Problem**: The current regex-based entity extractor only matches capitalized, single-word terms and fails on misspellings. Consequently, entities like "the goo blade" and "Keal" (intended to be "Kael") cannot be extracted.
2.  **Solution**: Train a micro NER model using the **BIO tagging scheme** to predict entity spans within the prompt; this scheme handles lowercase, multi-word, and misspelled entities. 
*   **Model Architecture**: Add a linear classification layer (384 → 3) on top of the **token-level** output of the frozen Qwen3 embedder, outputting probabilities for each token belonging to **B** (Beginning of entity), **I** (Inside entity), or **O** (Outside entity). 
*   **Training Data**: Use a **weak supervision** approach to automatically generate training data leveraging existing Codex triples. For every `episodic_memory` turn that generated a triple, perform a fuzzy search for the triple's subject or object within the prompt and automatically generate BIO tags. No manual labeling required! 
*   **Integration**: Replace regex matching with NER model inference results within `HybridRetrievalOrchestrator._codex_graph`.
3.  **Validation**: Test using prompts containing lowercase, multi-word, or misspelled entities (e.g., "What is a goo blade?"). Ensure the NER system extracts them correctly.

#### **CX2 – Vector-based Fuzzy Matching for Codex Entities**

1.  **Problem**: Even if the NER system extracts "Keal," a lookup in `codex_entities` fails to find "Kael."
2.  **Solution**:
*   Add an `embedding` column to the `codex_entities` table to store Qwen3 embedding vectors for the entities' "canonical names." 
*   Whenever an entity is extracted via NER, generate its embedding and perform a cosine similarity search against the `codex_entities` table. 
*   If the similarity score exceeds a threshold (e.g., `0.85`), consider it a match. This resolves issues related to misspellings and aliases. 
*   If no match is found, the entity is treated as unknown (however, a new entity is **not** automatically created during retrieval to prevent polluting the knowledge graph).
3.  **Verification**: Insert an entity named "Kael" into the database. Then, perform a retrieval using a prompt containing "Keal" to verify that the system successfully finds "Kael."

#### **CX4 – Contradiction Detection and New Fact Augmentation**

1.  **Issue**: When a new fact contradicts an existing one, the existing edge is correctly marked as expired (via `valid_until`), but the new edge requires multiple "confirmations" before being promoted to `active`. In the short term, this causes the system to return outdated information or respond with "I don't know."
2.  **Solution**:
*   In `handle_triplet`, when a contradiction is detected (causing the old edge's `valid_until` to be set) and a new edge is created, set the new edge's `strength` to `3.0` (instead of `1.0`). 
*   Since the threshold for `active` status is `2.0`, the new edge becomes `active` immediately, taking effect right away.
3.  **Verification**: Simulate an update to a fact (e.g., a character changing from "good" to "bad"). Query the new fact to verify that the system immediately returns the updated information.

#### **CX5 – Codex Extraction of Code Entities**

1.  **Issue**: Existing extraction prompts are generic; they fail to capture code-specific entities (such as function names, class names, or library names) and cannot establish relationships like "uses," "imports," or "extends."
2.  **Solution**:
*   **Prompt Augmentation**: Expand the system prompt for `extract_triplets` to include examples of code-specific entities (e.g., "Function X `uses` Library Y"). 
*   **Conditional Extraction**: When a conversation is classified as `Software_&_Tech`, invoke a more robust code-extraction model (or add code-specific instructions to the prompt) to capture function names, class names, and technical dependencies.
3.  **Verification**: Process a turn containing a code snippet. Check whether `codex_entities` contains function or class names and ensure they are linked via appropriate relationships.

#### New Feature: MERA (Meta-Enumeration Retrieval Agent)

**Category**: Retrieval / Codex

**Priority**: P1 (High)

**Description**: MERA is an enhancement to Codex that enables the system to answer "list all X" type queries that do not contain specific named entities. This resolves the "recursive" problem (i.e., needing to know entity names to retrieve them, yet needing to retrieve them to know their names).

**Current Issue**:
- A query like "Who are the characters in the story?" does not contain named entities such as "Kael" or "Aroh." - Standard Codex retrieval is not triggered because the candidate extraction step returns no results.
- The system replies "I don't know," even though Codex contains all the characters.

**How ​​MER Works**:

MERA is triggered when the following conditions are met:

1.  The `Factual_Retrieval` intent is identified by the classifier.
2.  No candidate entities are extracted.
3.  The prompt contains a category descriptor (e.g., "character," "role," "dependency," "function").

MERA employs a three-stage process:

1.  **Category Extraction**: A lightweight LLM maps category terms in the query to Codex tags.
2.  **Candidate Collection**: Codex filters by tag, performs keyword searches across relationships, and applies window-based retrieval.
3.  **Ranking**: Candidates are ranked based on weighted factors: mention frequency, recency of mention, and tag match quality.

**Output**: The ranked list is injected as structured context. **It uses no special formatting**—it simply appears as a standard context snippet. The model can then answer in natural language: "According to Codex, the main characters include: Kael, Aroh, Lethe..."

**Why this is better than hard-coded SQL**:
- It handles arbitrary categories (not just "characters" or "dependencies").
- It does not rely on pre-mapped tags.
- It ranks lists based on frequency and recency.
- It is a "learning" system—it does not rely on hard-coded rules.
- It works across domains (e.g., "Which characters appear in the coding conversation?").

**Advantages**: It solves the recursion problem, avoids reliance on SQL, and the ranked list helps the model prioritize information.

**Implementation Time**: Approximately 2–3 hours, utilizing existing LLM infrastructure.


#### New Feature: Collaborative Entity Retrieval with NER and MERA

**Category:** Retrieval / Codex

**Priority:** P0 (Critical)

**Description:** The NER model and MERA component are integrated into a unified retrieval pipeline. When a user query contains explicit named entities, NER extracts these entities and passes them to the Codex matcher. When the query is a "list" type (e.g., "What characters are in the story?"), the MERA component enumerates all known entities in Codex that match that category. These two components together ensure the system can handle both "explicit search" and "knowledge enumeration" scenarios, solving the "recursive retrieval" problem that plagued Experiment 1.

---

### 🧠 **Phase C: Memory Lifecycle & Retrieval Augmentation**

These improvements aim to ensure the system can extract, retain, and prioritize the most critical information.

#### **ML4 – Per-Turn Limit Override for Documents**

1.  **Issue**: Long pasted text (e.g., exceeding 2,000 words) is truncated to 500 words, resulting in information loss.
2.  **Solution**:
*   **Flagging**: During post-processing in `evaluate_turn`, if a turn's raw text exceeds 2,000 words and has low conversational density (e.g., `"Assistant:"` appears fewer than 3 times), mark it as `is_document = True`. 
*   **Limit Override**: In `_rows_to_fragments`, if `row.is_document` is `True`, bypass the 500-word truncation limit and inject the full `raw_text`. 
*   **Budget Constraints**: Even when injecting full text, adhere to dynamic token budgets. If the document is too large, inject it in chunks.

#### **ML2 & ML3 – Bookmark Enhancement & Immediate Extraction**

1.  **ML2 – Bookmark Enhancement**: Verify that retrieval scores for bookmarked turns are boosted (`score_val *= 1.5`), as implemented in `_rows_to_fragments`.
2.  **ML3 – Bookmark-Triggered Immediate Codex Extraction**:
*   **Issue**: Bookmarked turns require high-priority processing, but current Codex extraction may be delayed if the GPU is busy. 
*   **Solution**: In the `bookmark_turn` endpoint, call `extract_codex.delay(batch_id=...)`. Modify the `extract_codex` task to **skip** the `is_gpu_busy()` check when processing bookmarks, ensuring immediate extraction. #### **ML1 – Decay Stress Test Cycle**

1.  **Problem**: Initial experiments performed only three decay cycles, failing to realistically simulate a multi-month timeframe; consequently, the impact of decay on long-term memory could not be effectively tested.
2.  **Solution**: In Experiment 2, after injecting history and running background workers at each checkpoint, execute a **decay stress cycle**:
```python
for _ in range(30):  # Simulate 30 days of decay
apply_decay.apply()
time.sleep(0.5)
```
Only then are probe queries executed. This allows for verifying:
*   Whether frequently retrieved memories resist decay and maintain high scores. 
*   Whether memories untouched for a long period decay gracefully.
3.  **Verification**: Probe for old but frequently referenced information to observe if scores remain stable (or increase). Probe for information that has never been referenced to observe if scores decrease.

#### **ML5 – Conversation-Level Batch Summarization**

1.  **Problem**: In long conversations, summarizing individual turns accumulates a large number of fragmented segments, creating noise rather than providing a coherent, high-level overview.
2.  **Solution**:
*   **Batch Summarization Task**: Create a new background Celery task that runs periodically (e.g., daily). 
*   **Grouping Logic**: Group old turns (where `decay_score` < 0.3) by `conversation_id` and batch them according to time windows (e.g., batches of 50 turns). 
*   **Summary Generation**: For each batch, invoke the background model with the prompt: "Summarize the following conversation in 2–3 paragraphs, preserving all names, numbers, and decisions."
*   **Storage**: Store the generated summaries in a `batch_summaries` table (containing `conversation_id`, turn range, summary text, and embeddings). 
*   **Retrieval Integration**: Query `batch_summaries` alongside individual turn summaries during retrieval. Prioritize returning the batch summary if it is more concise and informative than the individual turn summaries. 3.  **Validation**: Run this task on the "Flaw" conversation, then inspect the retrieved context regarding the "major revelation" to verify if it is more coherent.

#### **CU1 – Increasing Cluster Limits**

1.  **Issue**: Limiting each cluster to only 10 turns leads to fragmentation and restricts the utility of the conversation scope.
2.  **Solution**: Increase the minimum number of turns per cluster to 30–50.
3.  **Token Limit Solution**: Cluster limits stem from LLM input size constraints. While batch processing multiple clusters is possible, it results in excessively large prompts. A better approach is to **use a summary of cluster descriptive statistics**—specifically the cluster name, top 5 tags, and 5 representative turn samples—to identify and merge similar clusters.

#### **CU2 – Vector-based Cluster Merging**

1.  **Issue**: Similar cluster names, such as "AI-driven OS" and "OS based on AI," currently exist as separate clusters.
2.  **Solution**:
*   **Cluster Embeddings**: Add an `embedding` column to the `context_clusters` table. When creating a new cluster, use the Qwen3 embedder to encode and store its name and description. 
*   **Periodic Merge Task**: Create a periodic task that scans all cluster embeddings to identify nearest-neighbor pairs with cosine similarity > 0.85. If found, merge them (reassign all turns from one cluster to the other, delete the empty cluster, and update the embedding to the average of the two).
3.  **Validation**: Manually create two similar clusters. Run the merge task and verify that they are combined into a single cluster.

---

### 🎮 **Phase D: User Control and Scope**

These features enhance system usability and privacy while laying the groundwork for future data collection and fine-tuning.

#### **UC1 – Like/Dislike Logging**

1.  **Purpose**: Collect implicit user feedback for future model fine-tuning. 2.  **Implementation**:
*   **Database Migration**: Add a `thumbs` column to the `episodic_memory` table (values: `up`, `down`, `neutral`). 
*   **API Endpoint**: Implement an endpoint `POST /user-control/turns/{turn_id}/thumbs` that accepts a request body of `{"value": "up"}` and stores the feedback.
3.  **Verification**: Send a "thumbs-up" via the API and verify that it is stored correctly.

#### **UC3 – Conversation Scope Isolation ("none" = Incognito)**

1.  **Objective**: When `memory_scope_type = "none"`, conversations should be completely isolated: no information is retrieved from other conversations, and the conversation's own turns are not retrieved by others, achieving a true "incognito" mode.
2.  **Implementation**:
*   **API Layer**: In `chat_completions`, if `conv_row.memory_scope_type == "none"`, **skip** retrieval entirely. 
*   **Storage Layer**: In `store_turn_async`, if the scope is "none", ensure `cluster_id` is set to `None` (as it should be). 
*   **Retrieval Layer**: In `HybridRetrievalOrchestrator`, for the "auto" scope, add a filter condition to exclude conversation turns where `memory_scope_type = "none"`, ensuring incognito conversations do not leak.
3.  **Verification**: Create one incognito conversation and one standard conversation. Perform a search in the standard conversation and verify that turns from the incognito conversation do not appear.

---

### 🧪 **Phase E: Experiment 2 Script**

**New Script**: `experiments/phase2b_mature_experiment.py`

This script will be used to run the final validation experiment. It will:

1.  **Load four long conversations**: Extract `bb558b5f`, `cca73c87`, `633e26f8`, and `a77c15cf` from `simulation_full.jsonl`. 2.  **Replay chronologically** and activate all new features.
3.  **Run stress-test cycles** (with load shedding) at each checkpoint.
4.  **Use the enhanced classifier**.
5.  **Use the NER model** for Codex searches.
6.  **Log all metrics**, including feature flag states, to facilitate experimental analysis.

---

### 📊 **Final Analysis**

Upon completion of Experiment 2, you will have a comprehensive set of data and conclusions:

*   **Summary of Improvements**: Clearly present the incremental gains from each improvement (e.g., "The Qwen3 embedder increased classifier accuracy by X%").
*   **Performance Report**: Demonstrate the new system's performance—regarding hallucination rates, entity recall, and overall TUR—on the most challenging long-form conversations.
*   **Case Studies**: Showcase the improvements by highlighting specific probes (e.g., cases where gating previously failed), describing the prior failure modes and explaining why they succeed under the new system.

This constitutes the finalized design document.







lets do this, with making sure doing all the above. and any other changes that is to be done in the files if not mentioned above, and also changes which are wrong and needs to be corrected
FOR THE ABOVE I WANT YOU TO DIRTECR ME EXTREMELY PRECISELY WITH WHAT TO WRITE, WHERE TO WRTIE, AND HOW TO WRITE, LIKE FOR NEW FILES JUST GIVE ME THE FILE PATH AS WELL AS THE ENTIRE CODE, BUT FOR PARTS WHERE WE HAVE TO CHANGE EXSISTING CODE, WHAT I WANT YOU TO DO IS GIVE ME DIRECTIONS LIKE THIS, EXAMPLE MENTION WHICH FILE THE CHANGE IT, MENTION ITS PATH, ALSO THE WHICH FUNCTION THE CHANGE IS IN, AND LIKE LETS SAY WE ARE CHANGING OR ADDING CODE TO LINE 10-12, WHAT I WANT YOU TO DO IS GIVE ME A BEFORE AND AFTER, THE BEFORE WILL HAVE WHAT THE CODE CURRENTLY LOOKS LIKE, WITH LIKE N-2 AND N+2 FROM THE AREA OR PLACE WE ARE CHANGING LIKE IF CHANGE AT 10 THE BEFORE CODE BLOCK YOU SHOWCASE WILL BE FROM 8 AND GO TILL 12, AND THEN YOU SHOW ME AN AFTER HOW THE CODE WILL LOOK AFTER DOING THE CHANGES. ALSO AFTER ALL THE STEPS OF THIS GIVE ME A GIT COMMIT MSG