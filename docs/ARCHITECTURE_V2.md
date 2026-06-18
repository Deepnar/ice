# Infinite Context Engine (ICE) — Architecture Document
## Architecture Specification — v2.0


> This document is the authoritative specification of the Infinite Context Engine as built and deployed. It is written as a direct implementation reference for agentic AI systems and developers. Every section reflects the actual state of the codebase.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Classifier Subsystem](#2-classifier-subsystem)
3. [Memory Architecture](#3-memory-architecture)
4. [Background Worker Cluster](#4-background-worker-cluster)
5. [Hybrid Retrieval Orchestrator](#5-hybrid-retrieval-orchestrator)
6. [Context Assembly and Prompt Construction](#6-context-assembly-and-prompt-construction)
7. [Dynamic Model Registry and MoE Routing](#7-dynamic-model-registry-and-moe-routing)
8. [User Control and Oversight Layer](#8-user-control-and-oversight-layer)
9. [Operational Infrastructure](#9-operational-infrastructure)
10. [Research Framework and Evaluation Protocol](#10-research-framework-and-evaluation-protocol)

---

# 1. System Overview

## 1.1 Purpose and Architecture Philosophy

The Infinite Context Engine (ICE) is a self-hosted AI memory middleware system designed to give local language models persistent, semantically structured memory across arbitrarily long interaction histories.

ICE is deployed as a **FastAPI proxy** that sits between a chat frontend (Open WebUI) and a local inference backend (Ollama). Every chat request passes through ICE, which intercepts the prompt, classifies it, retrieves relevant memory, assembles a context-enriched prompt, routes to the most appropriate model, streams the response, and asynchronously processes the completed exchange into the memory store.

The system is not a wrapper. It is a full cognitive middleware layer with its own database schema, background processing cluster, retrieval engine, and knowledge graph. The fundamental design principle is:

> Context that is retrieved surgically is worth more than context that is stuffed blindly.

ICE is built and runs on a Lenovo Legion Pro 7i with an RTX 5090 (24GB VRAM), on CachyOS/Arch Linux. The entire stack — Ollama, vLLM, PostgreSQL with pgvector, Redis, Celery, and the FastAPI proxy — runs locally on this machine.

---

## 1.2 Request Lifecycle

A complete request flows through ICE in two phases.

### Pre-Flight Phase

1. Open WebUI sends an OpenAI-compatible `POST /v1/chat/completions` to ICE.
2. ICE extracts the latest user message and passes it to the **PyTorch Classifier** for synchronous classification.
3. The classifier returns topic tags, intent tags, a context reliance label, and a confidence score.
4. If context reliance is `Long_Term_Memory` (or confidence falls below a fallback threshold), the **Hybrid Retrieval Orchestrator** is invoked.
5. Retrieved context fragments are assembled into a structured system prompt by the **Prompt Assembler**.
6. The **Model Registry** selects the most appropriate Ollama model based on the classified tags. Session stickiness logic prevents thrashing.
7. The assembled prompt is forwarded to Ollama, and the response is streamed back to the client.

### Post-Flight Phase

8. After the stream completes, a `BackgroundTask` commits the raw turn to episodic memory, attaches embeddings, and publishes a `CHAT_COMPLETED` event to Redis.
9. The **Post-Flight Evaluator** (Celery task) evaluates the exchange for information density, sets `lossless_flag` and `inject_raw`, and optionally generates a summary using the background model.
10. If lossless, the **Codex Extractor** runs and mines the exchange for structured entity triplets.
11. The **Procedural Extractor** independently scans for recurring behavioral patterns.

---

## 1.3 Infrastructure Stack

| Component | Technology |
|---|---|
| Proxy + API | FastAPI (async, Python 3.12) |
| Inference Backend | Ollama (primary), vLLM (background, port 8002) |
| Database | PostgreSQL 16 + pgvector extension |
| ORM + Migrations | SQLAlchemy (synchronous) + Alembic |
| Task Queue | Celery + Redis |
| Embedder | `all-MiniLM-L6-v2` (SentenceTransformers, CPU-resident) |
| Terminal UI | Textual (multi-tab TUI) |
| Configuration | Pydantic Settings + `.env` file |

---

# 2. Classifier Subsystem

## 2.1 Architecture

The ICE Classifier is a lightweight PyTorch MLP that sits on top of a frozen `all-MiniLM-L6-v2` sentence transformer encoder. It does not fine-tune the encoder — it learns only the MLP head. This design keeps the classifier fast and CPU-resident with negligible inference overhead.

### Model Architecture

```
Input: 384-dimensional sentence embedding (all-MiniLM-L6-v2)
  → Linear(384 → 128)
  → ReLU
  → Dropout(0.3)
  → Linear(128 → 25)
Output: 25 raw logits
```

The 25 output logits map directly to the three classification heads:

- **Positions 0–10**: 11 Topic Labels (BCEWithLogitsLoss, multi-label sigmoid)
- **Positions 11–21**: 11 Intent Labels (BCEWithLogitsLoss, multi-label sigmoid)
- **Positions 22–24**: 3 Context Reliance Labels (CrossEntropyLoss, single-class softmax)

---

## 2.2 Label Schema

### Topic Labels (11)

```
Software_&_Tech
STEM_&_Academics
Business_&_Finance
Creative_&_Media
Admin_&_Productivity
Lifestyle_&_Health
Social_&_Relationships
World_&_Current_Events
Meta_AI
Null_Noise
General_Reference_&_Trivia
```

### Intent Labels (11)

```
Factual_Retrieval
Troubleshooting
Generation
Ideation
Analysis_&_Summarization
Strategic_Planning
Decision_Making
Emotional_Processing
Utility_Formatting
Casual_Banter
Open_Exploration
```

### Context Reliance Labels (3)

```
Zero_Shot        → No memory needed; generic query
Long_Term_Memory → Retrieve from episodic/semantic/procedural stores
Real_Time_Search → Requires live web data (not yet routed)
```

---

## 2.3 Classification Output

```python
@dataclass
class ClassificationResult:
    topic_tags: List[str]       # all topics with sigmoid > 0.3
    intent_tags: List[str]      # all intents with sigmoid > 0.3
    context_reliance: str       # argmax of softmax over 3 classes
    raw_probs: List[float]      # all 25 raw probabilities
    max_confidence: float       # highest probability across all 25 outputs
    prompt: str                 # original prompt text
```

At inference, topic and intent tags are threshold-filtered at `0.3`. If no tag passes the threshold, the argmax label is used as a fallback to guarantee at least one classification per head.

---

## 2.4 Special Classification Overrides

Two hard overrides are applied after the standard classification pass:

**Creative Safety Override**: Any prompt classified under `Creative_&_Media` is unconditionally forced to `Long_Term_Memory` regardless of the model's context reliance prediction. Lore and narrative continuity require memory even when the classifier is uncertain.

**Conversation Scope Override**: At the orchestrator level, any `Zero_Shot` classification is overridden to `Long_Term_Memory` when an active conversation ID is present. The system treats a known conversation as implicit evidence that prior context may be relevant.

---

## 2.5 Confidence Fallback Threshold

When `max_confidence < 0.75` (configurable), the classifier result is treated as unreliable. The orchestrator responds by triggering the **Wide-Net Fallback** — a broad full-vector retrieval sweep that bypasses intent-gating and retrieves the top 10 most semantically similar turns globally. This prevents silent failures when the classifier is uncertain about an unusual prompt.

---

## 2.6 Training and Evolution Pipeline

The ICE Classifier is developed through a multi-stage pipeline designed to transform unstructured historical logs into a high-precision intent-routing engine.

### 2.6.1 Data Harvesting: The Amnesia Method

Training data is primarily sourced from authentic user interaction history. To isolate human-authored prompts from interleaved AI responses in raw exports (Claude, ChatGPT, DeepSeek), the system employs the **Amnesia Method**:

*   **Linear Slicing:** Raw logs are split into 3,000-character chunks with a 500-character overlap to prevent prompt truncation at boundaries.
*   **Stateless Extraction:** Each chunk is processed by a 1.5B/3B background model. The session is reset between chunks to ensure the model has no prior context, forcing it to identify human text based solely on structural and linguistic markers (slang, typos, informal first-person) rather than conversation flow.
*   **Deduplication:** Extracted prompts are normalized and deduplicated via SHA-256 content hashing.

### 2.6.2 High-Fidelity Labeling Architecture

Ground-truth labels are generated using a **vLLM-backed inference server** running `Qwen2.5-7B-Instruct-AWQ` (or equivalent).

*   **Structured Prompting:** Labels are assigned via an exhaustive system prompt that enforces four logical checks (Source Calibration, Immunity Traps, Signal Detection, and Final Decision) before outputting the 25-dimensional label vector.
*   **Asynchronous Labeling:** Utilizing the OpenAI-compatible vLLM endpoint allows for high-concurrency labeling ($C \geq 20$), processing the 20,000-prompt dataset in a fraction of the time required by standard sequential providers.

### 2.6.3 Dataset Balancing via Synthetic Injection

To mitigate class imbalance (e.g., underrepresented labels like `Real_Time_Search` or `Strategic_Planning`), the pipeline includes a **Synthetic Data Generator**.

*   The generator uses high-reasoning models to produce source-authentic, "noisy" human prompts for specific label combinations.
*   These prompts are blended with the authentic harvested data to ensure the MLP head achieves high F1-scores across the entire 25-label spectrum.

### 2.6.4 Iterative Training and Fine-Tuning Logic

The classifier follows a "Frozen Encoder, Trainable Head" architecture.

**Initial Training:**
The MLP head (128-neuron hidden layer) is trained on the full 20k blended dataset. Loss is calculated as a composite of Binary Cross Entropy (for multi-label Topic/Intent heads) and CrossEntropy (for the single-class Context Reliance head).

**Curated Fine-Tuning (The Feedback Loop):**
After deployment, the system evolves through targeted fine-tuning passes using the `fine_tune.py` infrastructure:

*   **Encoder Freezing:** The `all-MiniLM-L6-v2` encoder remains frozen to preserve semantic stability.
*   **Curated Reinforcement:** The worker loads manual corrections from the `curated_labels` table. These "Gold Standard" fixes are repeated ($N=50$) within the training loop to ensure the model prioritizes user-specific corrections.
*   **Weighted Objective:** A `curated_weight` multiplier (default: 10.0) is applied to the loss of curated examples, forcing the MLP head to adapt to the user's specific linguistic patterns and project-specific nomenclature (e.g., distinguishing "FLAW" as a creative work rather than a general noun).

### 2.6.5 Checkpoint Management

Fine-tuning generates time-stamped PyTorch weights (`.pt` files). The operational environment utilizes the `classifier_model_path` configuration to switch between the baseline `v2_final` model and recent fine-tuned increments, allowing for safe regression testing and rollback if model performance drifts.

---

# 3. Memory Architecture

ICE maintains four structurally distinct memory stores. Each is optimized for a different type of knowledge and operates at a different retrieval latency.

---

## 3.1 Episodic Memory

**Table**: `episodic_memory`

Episodic memory is the primary accumulation layer. Every completed turn — user prompt plus assistant response — is stored as a single episodic record.

### Key Fields

| Field | Type | Description |
|---|---|---|
| `id` | UUID | Primary key |
| `conversation_id` | UUID | Foreign key → `conversations` |
| `batch_id` | UUID | Processing unit identifier for workers |
| `timestamp` | DateTime(tz) | Exact creation time |
| `topic_tags` | Array[Text] | Classifier topic output |
| `intent_tags` | Array[Text] | Classifier intent output |
| `context_reliance` | Text | `Zero_Shot`, `Long_Term_Memory`, `Real_Time_Search` |
| `raw_text` | Text | Full verbatim turn: `"User: ...\n\nAssistant: ..."` |
| `summary_text` | Text | Background-generated summary (nullable) |
| `embedding` | Vector(384) | Sentence embedding for vector search |
| `decay_score` | Float | Starts at 1.0; decreases over time |
| `access_count` | Integer | Incremented each retrieval |
| `lossless_flag` | Boolean | Set by Post-Flight; NULL = unprocessed |
| `inject_raw` | Boolean | If true, raw_text is injected; else summary_text |
| `is_bookmarked` | Boolean | User-pinned; decay_immune = true |
| `decay_immune` | Boolean | Exempt from decay regardless of score |
| `is_archived` | Boolean | Archived from active retrieval |
| `idempotency_key` | Text | SHA-256 hash preventing duplicate commits |

### Decay Mechanics

Episodic memory uses access-weighted exponential decay. Turns are evaluated daily by the Decay Worker:

- Turns **never accessed** (`access_count == 0`) decay at `0.95x` per day (5% reduction).
- Turns **previously accessed** decay at `0.98x` per day (2% reduction).
- Retrieval **strengthens** turns: each retrieval adds `+0.15` to `decay_score` (capped at 1.0) and increments `access_count`.
- Turns below `decay_score = 0.1` are archived (`is_archived = True`).
- Turns below `decay_score = 0.05` are migrated to cold storage and deleted from the active table.

Bookmarked turns and decay-immune turns bypass all decay operations entirely.

---

## 3.2 Codex — Semantic Knowledge Graph

**Tables**: `codex_entities`, `codex_edges`, `codex_events`, `codex_snapshots`

The Codex is ICE's structured semantic memory layer. It models the world as a typed graph: entities connected by directed, time-stamped relationships. Unlike episodic memory (which stores raw interaction text), the Codex stores distilled factual assertions extracted from interactions.

### Entity Schema

```
CodexEntity
  canonical_name   → normalized lowercase identifier
  aliases          → array of known surface forms
  tags             → classification labels
  properties       → JSONB key-value store
  context_payload  → free-text description, enriched by Reflection Worker
```

Entities are identified by **UUIDv5** derived from their canonical name under a fixed namespace. This guarantees deterministic, collision-free identity across extraction passes.

### Edge Schema

```
CodexEdge
  source_id     → CodexEntity
  target_id     → CodexEntity
  relation      → verb phrase (e.g., "uses", "is located at", "replaced by")
  strength      → float, starts at 1.0, reinforced by repeated corroboration
  confidence    → "pending" | "active"
  valid_from    → when the edge was first asserted
  valid_until   → when the edge was superseded (NULL = currently true)
```

Edges begin as `pending`. They are promoted to `active` when `strength >= 2.0` (i.e., the relationship has been corroborated by at least two independent extractions). If a new extraction asserts a contradictory relation between the same source and target, the existing edge's `valid_until` is set to the current timestamp, and a new pending edge is created. This implements a **bi-temporal validity model** that preserves historical state while tracking transitions.

### Event Log

Every mutation to the graph is recorded as a `CodexEvent` with a typed payload (`edge_added`, `edge_expired`, `edge_strengthened`, `context_appended`). This append-only log enables full audit trails and retrospective state reconstruction.

### Compaction

When an entity accumulates 100 or more uncompacted events, the Compaction Worker generates a `CodexSnapshot` — a materialized view of the entity's current state — and marks all processed events as `compacted = True`. This bounds query latency for heavily-written entities.

---

## 3.3 Procedural Memory

**Table**: `procedural_memory`

Procedural memory captures recurring behavioral patterns and workflows. It answers the question: *what does this user consistently do when facing a certain type of problem?*

### Schema

```
ProceduralMemory
  pattern_name          → short identifier (≤80 chars)
  pattern_description   → one-sentence behavioral description
  topic_tags            → associated topic labels
  trigger_conditions    → JSONB conditions for retrieval activation
  reinforcement_count   → how many times this pattern has been observed
  confidence_score      → float; promoted to active at ≥ 0.8 after 3+ reinforcements
  is_active             → false until reinforcement threshold met
  source_batch_ids      → array of originating batch IDs
  embedding             → Vector(384) for similarity matching
```

Patterns are extracted by the **Procedural Extractor** from each post-flight turn. New patterns are compared against existing embeddings at a `0.85` cosine similarity threshold — above the threshold, the existing pattern is reinforced; below it, a new candidate is created. Patterns require a minimum of three reinforcements before becoming active and eligible for retrieval injection.

### Decay

A separate `procedural_decay` worker runs daily and deactivates patterns that have not been observed in 180 days and have fewer than 3 total reinforcements. Confident, well-established patterns are never decayed.

---

## 3.4 RAG Document Store

**Tables**: `rag_documents`, `rag_chunks`

The RAG store holds user-uploaded reference documents ingested via the **Drop Zone**. Documents are split into 512-word chunks, embedded with `all-MiniLM-L6-v2`, and stored as `RAGChunk` records with vector embeddings.

The Drop Zone is a `watchdog`-based filesystem monitor targeting `./ingest_inbox/`. When a `.txt`, `.jsonl`, or `.md` file is dropped there, it is automatically chunked, embedded, committed to the database, and moved to `./ingest_inbox/processed/`. A file-settle check prevents race conditions from partial writes.

RAG retrieval is gated by a strict intent filter: it only activates for `Factual_Retrieval` or `Analysis_&_Summarization` intents AND requires the prompt to contain document-reference keywords (`document`, `pdf`, `reference`, `manual`, `guide`). This prevents RAG noise from contaminating general-purpose conversational responses.

---

## 3.5 Memory Slots

**Table**: `memory_slots`

Memory slots are user-managed persistent text blocks that are injected verbatim into every retrieval-enabled prompt, regardless of the query. They represent stable, high-priority identity information that should always be present in context.

Seven named slots are defined:

| Slot Name | Purpose |
|---|---|
| `persona` | How the AI should present itself |
| `user_preferences` | Communication and interaction preferences |
| `tool_guidelines` | Specific instructions for tools and workflows |
| `project_context` | Current active project description |
| `guidance` | Hard behavioral rules and constraints |
| `pending_items` | Open questions and unresolved tasks |
| `session_patterns` | Observed behavioral patterns from the Reflection Worker |

Slots can be updated by the user through the TUI or the REST API. The Reflection Worker can also propose updates via the `review_queue`, which requires user approval before applying.

---

## 3.6 Context Clusters

**Table**: `context_clusters`

Clusters are named groupings of episodic turns. They enable **Cross-Chat Memory Scoping**: a user can configure a conversation to restrict retrieval to turns belonging to a specific cluster or set of clusters, preventing cross-project context contamination.

Clusters are created manually by the user via the TUI, or proposed automatically by the **Clustering Worker** and **Motif Detector** (part of the Reflection Worker). All automated cluster proposals enter the `review_queue` and require user approval before being applied.

---

# 4. Background Worker Cluster

ICE runs a persistent Celery worker fleet backed by Redis. Workers operate asynchronously and independently of the main FastAPI request cycle. All workers implement GPU resource gating and idempotency.

---

## 4.1 GPU Resource Gate

All background workers call `is_gpu_busy()` before executing any LLM inference. This function queries `nvidia-smi` and returns `True` if any GPU exceeds a 20% utilization threshold. In `dedicated` background model mode, GPU saturation causes the task to retry after 30–60 seconds. In `shared` mode, workers additionally yield when a user has chatted within the last 10 seconds (checked via the `ice:last_chat_completed` Redis key).

---

## 4.2 Background Model Architecture

ICE supports two background model modes:

**Dedicated Mode**: A separate vLLM instance runs `Qwen/Qwen2.5-3B-Instruct-AWQ` on port 8002. This model is always available for background tasks independent of the main Ollama instance.

**Shared Mode**: Background tasks use the main Ollama instance (default: `qwen2.5:7b`). User-activity gating ensures background tasks yield when the user is actively chatting.

The mode is configured via `BACKGROUND_MODEL_MODE` in `.env` and can be toggled live from the TUI Settings tab (requires service restart).

---

## 4.3 Post-Flight Evaluator

**Task**: `evaluate_turn`
**Trigger**: Immediately after each turn, via `BackgroundTask`
**Retry**: 5 attempts, 15s delay

The Post-Flight Evaluator is the primary data quality gate for the episodic store. It resolves the **Asymmetrical Value Problem** — the fact that not all stored text is equally worth retrieving.

### Lossless Classification

The evaluator applies `is_lossless()` to determine whether the full raw text should be injected or replaced by a summary:

A turn is classified as **lossless** (raw text preserved, no summarization) if:

- The response contains code blocks (backticks).
- The response exceeds 500 words.
- After stripping standard sentence boundaries, three or more true proper nouns remain.

A forced lossless override applies to all `Creative_&_Media` or `Emotional_Processing` tagged turns — these always retain raw text regardless of density analysis.

For turns classified as **lossy**, the background model generates a 2–3 sentence summary that preserves all named entities, numbers, code descriptions, and specific decisions. The summary replaces raw text at injection time.

After evaluation, the worker:
1. Dispatches `extract_codex.delay()` if the turn is lossless.
2. Dispatches `extract_procedural.delay()` unconditionally.

---

## 4.4 Codex Extractor

**Task**: `extract_codex`
**Trigger**: Post-flight, on lossless turns only
**Retry**: 3 attempts, 30s delay

The Codex Extractor submits the raw turn text to the background model with a structured extraction prompt requesting subject-relation-object triplets in JSON format. The parser implements multi-layer JSON extraction resilience: it attempts direct parsing, then strips markdown fences, then applies a fallback regex pattern for individual triplet objects.

Each valid triplet is processed through `handle_triplet()`, which:

1. Resolves or creates the source and target entities via UUIDv5 identity derivation.
2. Checks for an existing active edge between the entities.
3. If the edge exists with the same relation: reinforces it (strength +1.0, promote to active at strength ≥ 2.0).
4. If the edge exists with a different relation: expires the old edge (`valid_until = now`), creates a new pending edge.
5. If no edge exists: creates a new pending edge.
6. Records a `CodexEvent` for every mutation.

All extraction passes are idempotency-keyed with SHA-256 hashes of `"codex:{batch_id}"`.

---

## 4.5 Procedural Extractor

**Task**: `extract_procedural`
**Trigger**: Post-flight, every turn
**Retry**: 3 attempts, 30s delay

The Procedural Extractor asks the background model to identify one recurring behavioral pattern or workflow observable in the exchange, expressed as a single sentence. If the model outputs `NONE`, the worker exits cleanly.

If a pattern is identified, it is encoded via `all-MiniLM-L6-v2` and compared against existing procedural memories using cosine similarity. Matches above `0.85` trigger reinforcement. Misses create new candidate patterns with `is_active = False` and `confidence_score = 0.3`.

---

## 4.6 Reflection Worker

**Task**: `run_reflection`
**Schedule**: Daily at 05:00 UTC

The Reflection Worker is the highest-level background intelligence pass. It operates over the 200 most recent episodic turns and executes five sequential sub-operations:

**Session Synthesis**: Generates a structured `SessionSummary` JSON containing topics covered, decisions made, unresolved items, entities updated, and patterns observed. Unresolved items are appended to the `pending_items` memory slot.

**Pattern Crystallization**: Scans all recent turns for recurring workflows and reinforces or creates procedural memory entries accordingly. This is a higher-level pass than the per-turn Procedural Extractor, looking for cross-session patterns.

**Memory Slot Evolution**: Proposes updates to `project_context`, `user_preferences`, and `guidance` slots based on recent content. All proposals are routed to the `review_queue` and require user approval before being applied.

**Codex Enrichment**: Identifies entities with thin or empty `context_payload` values and enriches them using recent episodic passages that reference those entities. Each enrichment is logged as a `context_appended` CodexEvent.

**Motif Detection**: Scans recent sessions for recurring thematic patterns that do not yet correspond to named clusters. Detected motifs are proposed as new cluster names via the `review_queue`.

---

## 4.7 Decay Worker

**Task**: `apply_decay`
**Schedule**: Daily at 03:00 UTC

Applies access-weighted exponential decay to all non-immune, non-bookmarked episodic turns older than 7 days. Archives turns below `decay_score = 0.1`. Migrates turns below `decay_score = 0.05` to `cold_storage` and removes them from the active episodic table.

---

## 4.8 Codex Decay Worker

**Task**: `decay_codex_edges`
**Schedule**: Daily at 03:30 UTC

Applies 1% decay per run to all active Codex edges. Edges whose strength falls below `0.3` are demoted back to `pending`. This ensures that stale, unconfirmed relationships are gradually deprioritized without permanent deletion.

---

## 4.9 Procedural Decay Worker

**Task**: `decay_procedural_patterns`
**Schedule**: Daily at 04:30 UTC

Deactivates procedural patterns that have not been observed in 180 days and have fewer than 3 reinforcements. Confident, heavily-reinforced patterns are never deactivated.

---

## 4.10 Clustering Worker

**Task**: `cluster_turns`
**Schedule**: Daily at 04:00 UTC

Scans episodic turns with no assigned cluster. Groups up to 30 unassigned turns and asks the background model to propose 1–3 cluster names based on topic tags and snippet content. Assigns turns to existing or newly-created clusters proportionally. Proposals bypass the review queue — clusters are created directly; turn assignment is reversible by the user.

---

## 4.11 Sentinel Monitor

**Task**: `monitor_sentinels`
**Schedule**: Every 30 minutes

Evaluates all active `SentinelRule` records against current database state. Supported rule types:

- **Threshold**: Evaluates quantitative conditions (e.g., entities with pending edge accumulation above a threshold).
- **Absence**: Fires when a resource has not been updated within a configured number of days (e.g., a stale `pending_items` slot).

When a rule fires, a `SentinelEvent` is logged and the rule's action is executed. Supported actions: `log_event`, `notify`, `schedule_worker`, `create_review_item`. A per-rule cooldown prevents repeated firing within a configured window.

---

## 4.12 Fine-Tune Worker

**Task**: `fine_tune_classifier`
**Schedule**: Weekly, Monday at 04:00 UTC

Retrains the classifier MLP head on accumulated `CuratedLabel` records from the database. Uses the frozen `all-MiniLM-L6-v2` encoder and runs 10 training epochs. Saves the new checkpoint to `models/classifier/` with a datestamp suffix. The new checkpoint is not automatically activated — the operator must update `classifier_model_path` in `.env` and restart the proxy.

---

## 4.13 Codex Inject Watcher

A separate `watchdog`-based process monitors the `./codex_inject/` directory for YAML or JSON entity definition files. When a file is detected, it is parsed and injected directly into the Codex as `active` edges with `strength = 2.0`, bypassing LLM extraction entirely. This provides a manual override pathway for seeding the knowledge graph with high-confidence facts. Processed files are moved to `./codex_inject/processed/`.

---

# 5. Hybrid Retrieval Orchestrator

## 5.1 Entry Point and Safety Overrides

The `HybridRetrievalOrchestrator.retrieve()` method is the single entry point for all memory lookup. Before executing any retrieval leg, two overrides are applied unconditionally:

1. `Zero_Shot` classification is promoted to `Long_Term_Memory` whenever a `conversation_id` is active.
2. `Creative_&_Media` topic tags force `Long_Term_Memory` regardless of classifier output.

If context reliance remains `Zero_Shot` or is `Real_Time_Search`, an empty list is returned immediately.

If `max_confidence < 0.75`, the **Wide-Net Fallback** is invoked instead of the standard multi-leg pipeline.

---

## 5.2 HyDE Query Rewriting

Before executing retrieval legs, the orchestrator rewrites the user query using **Hypothetical Document Embeddings (HyDE)**. The background model is prompted to rewrite the question as a dense, factual search query that would retrieve the relevant past conversation — stripping politeness, expanding implied context using the last 5 turns, and surfacing key entities.

If the HyDE rewrite succeeds, the query embedding is re-computed from the rewritten text. If HyDE fails (timeout, model unavailable), the original query embedding is used transparently.

---

## 5.3 Retrieval Legs

The orchestrator executes five concurrent retrieval legs and fuses their results.

### BM25 Episodic (Full-Text Search)

Executes a PostgreSQL full-text search using `to_tsquery` over the concatenation of `raw_text` and `summary_text`. The query is constructed from the first 30 significant words of the (HyDE-rewritten) prompt after punctuation removal. Results are filtered by `decay_score > 0.2` and `is_archived = False`. Returns up to 20 candidates.

### Vector Episodic (pgvector Cosine Similarity)

Executes a pgvector approximate nearest-neighbor search over `embedding` using the `<=>` cosine distance operator. The final score multiplies cosine similarity by `decay_score`, ensuring decayed memories are naturally deprioritized even if semantically similar. Returns up to 20 candidates.

Both episodic legs support **conversation-scope filtering**: if a `scope` dict containing `conversation_id` is provided, both SQL queries append `AND conversation_id = :conv_id`, restricting retrieval to the specified conversation.

### Codex Graph Traversal

Extracts candidate entity names from the prompt by scanning for capitalized words matching `\b[A-Z][a-zA-Z0-9_]+\b`. Resolves candidates against `codex_entities` by canonical name and alias matches. Performs a depth-limited graph traversal (max depth 2) collecting `context_payload` from each visited entity and following active edges (where `valid_until IS NULL`) to their targets.

The resulting entity context is returned as a single concatenated `ContextFragment` with `score = 1.0`. Codex retrieval supports conversation-scope filtering by cross-referencing `codex_events.batch_source` against episodic `batch_ids` for the target conversation.

### Procedural Lookup

Executes a cosine similarity search over `procedural_memory` embeddings. Results are filtered by the **trigger condition evaluator** — a procedural pattern is only injected if the current classification's topic and intent tags satisfy the pattern's `trigger_conditions` JSONB. Retrieval is further restricted to patterns that originated from the target conversation's batch IDs. Procedural lookup is only activated for `Strategic_Planning`, `Generation`, or `Open_Exploration` intents.

### RAG Lookup

Executes a cosine similarity search over `rag_chunks`. Only activated for `Factual_Retrieval` or `Analysis_&_Summarization` intents AND when the prompt contains explicit document-reference keywords. Returns up to 5 chunks.

---

## 5.4 Dynamic Leg Weighting and RRF Fusion

Rather than applying fixed weights across retrieval legs, the orchestrator derives a **blended weight vector** from the active intent classifications.

Five intent profiles define override weight maps:

| Profile | Intents | Vector Bias |
|---|---|---|
| Factual | `Factual_Retrieval`, `Utility_Formatting` | vector ↑, bm25 ↑, codex ↓, procedural ↓ |
| Troubleshooting | `Troubleshooting`, `Strategic_Planning` | vector ↑, procedural ↑↑ |
| Creative | `Generation`, `Ideation`, `Open_Exploration` | codex ↑↑, vector ↓, bm25 ↓ |
| Analytical | `Emotional_Processing`, `Analysis_&_Summarization`, `Decision_Making` | vector ↑, codex ↑, bm25 ↓ |
| Casual | `Casual_Banter`, `Null_Noise` | all weights reduced |

When multiple intents are active, their profile weights are averaged proportionally. Topic overrides apply cumulatively on top: `Creative_&_Media` adds +0.3 to the Codex weight; `Software_&_Tech` adds +0.4 to the Procedural weight.

The final blended weights are passed to the **Reciprocal Rank Fusion (RRF)** aggregator as `alpha_map`. RRF scores each leg's result list independently by rank position and sums the weighted `alpha / (k + rank)` scores across legs. The default constant `k = 60` prevents top-ranked items from dominating when they appear in only one leg.

---

## 5.5 Post-Fusion Processing

After fusion, three additional passes clean and bound the result set:

**Session Diversification**: Caps fragments from any single non-current conversation at 3. Fragments from the current conversation are admitted without restriction. This prevents one historically-verbose conversation from dominating the context window.

**Deduplication**: SHA-256 hash deduplication removes identical text fragments that may have appeared in multiple legs.

**Token Budget Enforcement**: Fragments are greedily accepted until the total token count (estimated as `words * 1.33`) exceeds 5,000 tokens. Fragments beyond the budget are discarded in order.

**Retrieval Strengthening**: All retrieved episodic fragments have their `access_count` incremented and `decay_score` boosted by `+0.15` to reflect their demonstrated relevance.

---

## 5.6 Wide-Net Fallback

When classifier confidence falls below the threshold, the orchestrator abandons intent-gated retrieval and executes a raw full-vector sweep over all non-archived episodic memory with `decay_score > 0.2`. The top 10 results by cosine similarity are merged with any Codex and RAG results, fused via RRF, and token-budget-capped at 2,000 tokens. This fallback trades precision for recall, ensuring that uncertain prompts still receive potentially relevant memory rather than empty context.

---

# 6. Context Assembly and Prompt Construction

## 6.1 Prompt Assembler

The `assemble_prompt()` function in `src/api/prompt_assembler.py` constructs the final `[system, user]` message list from the retrieved fragments and persistent memory state.

### Standard Assembly Order

Context is assembled in a fixed priority order, from highest to lowest:

1. **System Rules**: A static instruction block defining the AI's behavior toward memory.
2. **Bookmarked Memories**: Explicitly pinned turns surface first.
3. **Persistent Memory Slots**: All active slots injected verbatim (persona, preferences, guidelines, project context, etc.).
4. **Recent Context (Sliding Window)**: The last 10 turns from the current conversation in chronological order.
5. **Codex Knowledge Graph Assertions**: Structured entity facts.
6. **Retrieved Episodic Interactions**: Past turns surfaced by the Hybrid Retrieval Orchestrator.
7. **Procedural Execution Patterns**: Behavioral workflow patterns.
8. **Reference Material (RAG)**: Document chunks from the RAG store.

Each section is labeled with a clear header: `=== PERSISTENT CORE PREFERENCES ===`, `=== CODEX KNOWLEDGE GRAPH ASSERTIONS ===`, etc.

### Emotional and Social Query Bypass

When the classification contains `Emotional_Processing` intent, `Social_&_Relationships` topic, or `Creative_&_Media` topic, the standard structured assembly is bypassed in favor of a plain-context format that injects retrieved fragments as a flat block. The system prompt voice shifts accordingly: `Factual_Retrieval` intents receive an exhaustive, list-everything tone; all others receive a warm, personally-resonant tone. This prevents clinical structure from undermining emotionally-sensitive interactions.

---

## 6.2 Token Budget Management

After assembly, the proxy estimates total prompt token count using a `words * 1.33` heuristic targeting 90% of a 4,096-token context window. If the assembled prompt exceeds the budget, the lowest-priority fragments (procedural, then episodic, in reverse order) are iteratively removed and the prompt is reassembled until it fits. This ensures that core memory slots and recent context always survive to the final prompt.

---

# 7. Dynamic Model Registry and MoE Routing

## 7.1 Model Registry

ICE maintains a dynamic registry of available Ollama models at `models/model_registry.json`. The registry is populated by querying the Ollama API for all available models and attempting to enrich each model's tags by querying the Hugging Face model card API.

Each registry entry tracks:

```json
{
  "model_name": {
    "topic_tags": ["Software_&_Tech"],
    "intent_tags": ["Generation", "Troubleshooting"],
    "confirmed": true,
    "base_url": null
  }
}
```

Entries with `confirmed: false` are excluded from MoE routing until the user confirms them via the TUI Models tab.

---

## 7.2 MoE Routing

When a request arrives with `model: "ice-proxy"` in the body, the registry's `find_best_model()` function selects the highest-scoring confirmed model by computing tag overlap against the current classification. The selected model is cached for the conversation with **session stickiness**: the same model is reused for up to 2 consecutive turns with zero topic/intent overlap before re-routing is triggered. This prevents thrashing when a single off-topic message would otherwise cause a model switch.

The fallback model (`qwen2.5:7b` by default) is always available and requires no confirmation.

---

## 7.3 SSE Metadata Events

The streaming response emits four structured SSE events before token generation begins:

```
event: classified      → {topic_tags, intent_tags, context_reliance, max_confidence}
event: retrieval       → {active_legs, hyde_used, tokens_injected}
event: context_ready   → {fragments_count, sources: {codex, episodic, procedural, rag}, total_tokens}
event: generating      → {model}
```

The TUI's Context tab parses and displays these events in real time, providing live visibility into every retrieval and routing decision.

---

# 8. User Control and Oversight Layer

ICE exposes a REST API and a full-featured Textual TUI for user-directed memory management. All automated memory changes that involve persistent state modifications (slot updates, new clusters, sentinel flags) are gated through the `review_queue` before being applied.

---

## 8.1 Bookmarking

`POST /user-control/turns/{turn_id}/bookmark`

Bookmarked turns become decay-immune (`decay_immune = True`), are forced lossless, and are elevated to the top of the context injection order. Bookmarking also triggers an immediate Codex extraction job on the turn. The TUI exposes a `Ctrl+B` binding to bookmark the most recent turn without leaving the chat interface.

---

## 8.2 Manual Label Correction

`POST /user-control/batch/override-tags`

Allows the user to manually correct the classifier's topic, intent, and context reliance labels for a given batch. Corrections are stored in `CuratedLabel` and consumed by the weekly fine-tuning cycle. This is the primary mechanism for improving classifier accuracy on the user's specific vocabulary and conversational patterns.

---

## 8.3 Conversation Scoping

`PUT /user-control/conversations/{conv_id}/scope`

Three scoping modes control retrieval boundaries:

- **`auto`**: Retrieval searches globally across all conversations. Default.
- **`project`**: Retrieval is restricted to the current conversation and any associated clusters. Prevents cross-project contamination.
- **`none`**: Retrieval is fully disabled. The proxy behaves as a transparent pass-through.

Scoping mode is set per-conversation and persists until changed. The TUI Settings tab exposes a live scope selector.

---

## 8.4 Cluster Management

Users can create named clusters, assign specific turns to them, and scope conversations to retrieve only from those clusters. This implements the **Cross-Chat Memory Scoping** pattern, enabling project-level memory organization that mirrors how humans mentally compartmentalize knowledge domains.

---

## 8.5 Review Queue

All automated proposals — Reflection Worker slot updates, Motif Detector cluster proposals, Sentinel review items — land in the `review_queue` as `pending` items. The user reviews and approves or rejects them via `GET /user-control/review-queue` and `POST /user-control/review-queue/{item_id}/approve`. The TUI does not yet have a dedicated review panel; interaction is currently via API.

---

# 9. Operational Infrastructure

## 9.1 Service Startup

ICE is started via the `ice` shell script, which launches the following services in a coordinated sequence:

1. **vLLM background model** (port 8002) — serves the dedicated 3B background model.
2. **Celery worker** — runs all background tasks.
3. **Celery Beat** — scheduler for all periodic tasks.
4. **FastAPI proxy** (port 8000) — the main ICE middleware server.

All services log to `$LOG_DIR/` and the startup script tails unified logs after launch.

---

## 9.2 Database Configuration

PostgreSQL is configured at `postgresql+psycopg://ice:ice_local_dev@localhost:5432/ice_db`. The connection pool is set to `pool_size=20` with `max_overflow=0` and `pool_pre_ping=True`.

Alembic manages schema migrations. All tables are defined in `src/memory/models.py` via SQLAlchemy declarative base. The `pgvector` extension must be enabled before migrations run.

---

## 9.3 Idempotency Architecture

Every worker task that mutates the database is gated by the `idempotency_keys` table. Each task derives a unique key (SHA-256 of `"{task_type}:{batch_id}"`) and checks for its existence before executing. Processed keys are committed atomically with the task's mutations. This prevents duplicate processing from retry storms, celery beat overlaps, or race conditions during high-load periods.

The main API request path uses a different idempotency mechanism: a SHA-256 hash of `"{correlation_id}:{user_message}"` is computed and stored on the `EpisodicMemory` row as `idempotency_key`. Duplicate request submissions are rejected at the database constraint level.

---

## 9.4 Redis Integration

Redis serves as both the Celery broker/backend and a real-time coordination bus. The following Redis operations are in active use:

- **Celery broker**: All task dispatch and result storage.
- **`chat:completed` pub/sub channel**: Published after every completed turn with `correlation_id`, `conversation_id`, and `batch_id`. Workers can subscribe for reactive processing.
- **`ice:last_chat_completed` key**: Updated after each turn with the current UTC timestamp. Workers check this to implement user-activity gating in shared background model mode.

---

# 10. Research Framework and Evaluation Protocol

## 10.1 Research Context and Motivation

The Infinite Context Engine is not only a deployed system — it is the subject of an original research program examining the empirical value of structured, cognitively-layered memory retrieval over naive context management in local AI deployments.

The research program produces **two papers**, reflecting a deliberate structural decision made during system development:

**Paper 1** covers the architecture and system design of ICE, including the classifier pipeline, memory hierarchy, retrieval orchestration, and evaluation methodology. It is a systems paper that establishes ICE as a complete, reproducible research artifact.

**Paper 2** covers the longitudinal behavioral findings: what ICE's evaluation protocol actually measures over months of real interaction data, how retrieval quality evolves as memory matures, and what the empirical gap is between cognitive layering and naive context stuffing.

This two-paper structure reflects how the work developed in practice: the system design and evaluation methodology were co-designed, making them inseparable as a first contribution, while the longitudinal findings constitute a second, temporally distinct contribution that can only be produced after sufficient operational history.

---

## 10.2 Protocol Objective: Longitudinal State-Replay Evaluation (LSREP)

ICE is fundamentally a **stateful cognitive architecture**. Unlike conventional Retrieval-Augmented Generation (RAG) systems, whose performance can be measured using static benchmark datasets, ICE continuously evolves through memory formation, procedural abstraction, factual consolidation, and temporal decay.

Evaluating ICE using traditional benchmark methodologies would fail to measure its primary objective:

> Maintaining factual consistency, identity persistence, and knowledge coherence across long-horizon interactions spanning months of accumulated experience.

To address this challenge, ICE introduces the **Longitudinal State-Replay Evaluation Protocol (LSREP)**.

LSREP reconstructs the cognitive state of the system at arbitrary historical coordinates and evaluates retrieval behavior under realistic temporal conditions. Rather than treating memory as a static database, LSREP treats the system as a **temporal organism** that must be grown through experience, allowed to mature through background processing, audited through controlled experimentation, and forensically analyzed against objective evidence.

The protocol reconstructs approximately **3.2 million characters of historical interaction data distributed across 58 independent conversation streams**, representing roughly seven months of operational history.

---

## 10.3 Simulation Harness and Temporal Reconstruction

The Simulation Harness (`scripts/simulation/run_simulation.py`) serves as the core investigative infrastructure. Its purpose is to reconstruct historical cognitive states and generate reproducible experimental conditions.

Unlike conventional benchmark runners that load preprocessed datasets directly into memory, the harness performs a complete chronological replay of historical interactions through the live ICE middleware. This ensures that all embeddings, memory transformations, procedural abstractions, and factual extractions emerge naturally from the architecture itself.

---

### 10.3.1 Incremental State Reconstruction

Evaluation begins with a completely empty system state. Historical data is replayed chronologically through a three-stage reconstruction cycle.

**Stage 1 — Chronological Injection**

The harness iterates through historical turns in timestamp order. Each turn is processed by the full production pipeline: classification, embedding generation, temporal tagging, episodic storage, and post-flight evaluation (including lossless detection, summarization, and Codex extraction). Conversation identity is preserved — turns are associated with their original `conversation_id` via deterministic UUIDv5 derivation where necessary.

**Stage 2 — Background Maturation**

Memory insertion alone is insufficient because ICE derives higher-order knowledge through asynchronous processing. After each temporal checkpoint, the harness runs `post_simulation.py`, which blocks until all pending Celery tasks complete: clustering (polling until no unassigned turns remain), reflection, decay, and sentinel monitoring. This ensures that reconstructed state faithfully represents a cognitively mature snapshot rather than a raw turn dump.

**Stage 3 — Maturation Gating**

Evaluation probes are never executed against freshly inserted memory. A checkpoint is only considered valid when all background workers have reached a stable state. This prevents the benchmark from measuring raw storage capacity and instead evaluates fully synthesized, background-processed knowledge.

---

### 10.3.2 Temporal Checkpoint Coordinates

To evaluate memory behavior across different stages of historical development, each conversation is sampled at three proportional positions:

| Checkpoint | Position | Captures |
|---|---|---|
| Early State | 0.25L | Initial memory formation |
| Mid State | 0.60L | Active consolidation and abstraction |
| Late State | 0.95L | Long-term memory saturation and decay |

Where `L = total conversation length`.

This slicing strategy allows the protocol to measure how retrieval quality evolves as memory density increases and how decay affects older information over time.

---

## 10.4 Experimental Evaluation Matrix

Once a mature checkpoint has been reconstructed, every evaluation probe is executed through a six-way experimental matrix. The purpose is to isolate the contribution of retrieval architecture from the contribution of inference architecture.

### T1 — Naive Chronological Control

**Objective**: Establish the Context Collapse baseline.

A literal extraction of conversation history immediately preceding the evaluation probe is performed. A strict sliding window of approximately 3,000 words (4,000 tokens) is enforced. No vector search, no Codex, no procedural memory, no decay weighting, no intent filtering. This represents the behavior of conventional chat interfaces operating under finite context windows.

### T2 — Unfiltered Semantic Baseline

**Objective**: Establish the Standard RAG baseline.

A raw Top-K cosine similarity search is executed against the episodic memory store using `all-MiniLM-L6-v2`, with no intent gating, no decay thresholds, no Codex joins, no procedural injections, and no hybrid fusion. This measures the effectiveness of pure vector retrieval without cognitive layering.

### T3 — Full ICE Cognitive Stack

**Objective**: Evaluate the target architecture.

The complete Hybrid Retrieval Orchestrator is activated. Retrieval combines BM25, vector, Codex, and procedural legs fused via Reciprocal Rank Fusion with dynamic intent-weighted alpha maps. Session diversification, decay-aware prioritization, relationship traversal, HyDE rewriting, and procedural enrichment are all active. This measures the cumulative benefit of the complete architecture.

### Routing Permutations

Each retrieval condition is evaluated under two inference configurations:

**G1 — Static Generalist**: A single base model processes all prompts. Measures retrieval quality independently of routing specialization.

**G2 — Dynamic MoE Selector**: The Model Registry routes each probe to the highest-scoring specialist model for its classification tags. Measures the interaction effect between retrieval quality and specialized inference.

### Final Matrix

| Retrieval Layer | Generalist | MoE |
|---|---|---|
| T1 Naive Control | T1-G1 | T1-G2 |
| T2 Vector RAG | T2-G1 | T2-G2 |
| T3 Full ICE Stack | T3-G1 | T3-G2 |

**Total Conditions**: 6

---

## 10.5 Ground-Truth Distillation Framework

### Evidence Harvester

A high-recall retrieval pass identifies the top 40 historical fragments from the entire replayed corpus for each evaluation probe. These fragments are assembled into an Evidence Dossier of approximately 30,000 tokens.

### Surgical Extraction Protocol

The Evidence Dossier is processed by a high-reasoning validator model to produce the official **Ground-Truth Dossier**. The extraction process enforces four mandatory constraints:

**Neutrality**: Third-person reporting only. "The user selected..." not "You selected..."

**Verbatim Anchors**: Numbers, names, identifiers, and specifications must remain exact. Hardware specs, GPA values, model names, and version strings are not paraphrased.

**Temporal Dominance**: The newest valid fact supersedes all earlier versions. Previous values are preserved as legacy records rather than discarded.

**Exhaustive Enumeration**: No summarization is permitted. Every relevant fact must be included, regardless of apparent redundancy.

---

## 10.6 Four-Pass Forensic Audit System

All six matrix outputs are evaluated through a centralized forensic audit pipeline. A high-capacity validator model performs four independent passes.

### Pass 1 — Absolute Factual Fidelity

Measures alignment against the Ground-Truth Dossier on a 1–5 scale. Evaluation is based on successful recovery of factual anchors: names, dates, specifications, decisions, numerical values. This is the primary accuracy metric.

### Pass 2 — Blind Comparative Tournament

All six responses are shuffled and presented anonymously as A through F. The judge ranks responses 1 (best) to 6 (worst) based on accuracy, density, completeness, and efficiency. For ranks 4–6, a mandatory failure diagnosis is produced identifying the specific failure mode (context dilution, hallucinated entity, missing anchor fact, redundant verbosity).

### Pass 3 — Temporal Consistency Oracle

The judge receives future conversation history that occurred after the evaluation checkpoint. Each response is evaluated for temporal contradictions: does it predict events that didn't happen, contradict known future history, or correctly anticipate the trajectory of decisions? This audit detects hidden inconsistencies that raw factual scoring alone cannot surface — particularly important for multi-month conversations where decisions are revised and projects evolve.

Outcomes are classified as **Clean** (consistent with future events), **Hallucinated** (contradicts known future history), or **Predictive Alignment** (correctly anticipates future trajectory).

### Pass 4 — Retrieval Ingredient Audit

This pass ignores the final model response entirely and evaluates only the retrieved fragments injected into the prompt.

**Relevance Density Percentage (RDP)**: The fraction of retrieved tokens that are genuinely relevant to the probe query.

```
RDP = Relevant Tokens / Total Tokens
```

**Missing Link Analysis**: Identifies relevant facts that were present in the corpus but failed to surface through retrieval.

**Noise Attribution**: Quantifies irrelevant context injected into the prompt — the primary diagnostic for architectural weaknesses.

---

## 10.7 Primary Research Metrics

The evaluation consolidates all audit outputs into three architecture-level metrics.

### Token Utility Ratio (TUR)

Measures intelligence generated per unit of context consumed.

```
TUR = Mean Fidelity Score / (Tokens Injected / 1000)
```

Higher TUR indicates superior computational efficiency. This is the primary economic argument for surgical retrieval over context stuffing.

### Noise-to-Signal Ratio (NSR)

Derived from Pass 4 Retrieval Ingredient Audit.

```
NSR = Irrelevant Tokens / Total Tokens
```

Lower values indicate cleaner retrieval. NSR is the primary diagnostic for false-positive retrievals and cross-topic context contamination.

### Identity Persistence Delta (IPD)

Measures the system's ability to correctly overwrite outdated information without contamination from legacy memories. Evaluated specifically on temporal sequences where old project decisions were replaced by newer ones, hardware was upgraded, or named entities were renamed. A low delta indicates stable, temporally-coherent memory evolution.

---

## 10.8 Reproducibility Standards

All evaluations conform to fixed execution parameters to ensure scientific validity.

**Concurrency Control**:
- 128k-context tasks: `C=1` (prevents VRAM instability)
- Standard tasks: `C=10`

**Deterministic Generation**:
- `temperature = 0.0`
- `repetition_penalty = 1.15`
- Fixed random seed

**Telemetry Preservation**: Every evaluation record stores `full_ice_fragments`, `vector_fragments`, `judge_scores`, `ranking_results`, `hallucination_flags`, and `token_counts` in a structured results database for complete retrospective auditing and independent verification.

---

## 10.9 Central Research Hypothesis

The central hypothesis of ICE, tested by Paper 2, is:

> Surgical retrieval outperforms naive context stuffing by maximizing factual fidelity, minimizing noise, preserving identity consistency, and delivering greater intelligence per token consumed.

LSREP exists to test that claim under realistic long-horizon conditions and quantify the exact computational return on memory-aware cognition.

The two-paper structure ensures that the system's architecture and its evaluation methodology receive rigorous treatment in the first contribution, while the longitudinal behavioral findings — which require months of real operational data to produce — constitute a temporally distinct and independently meaningful second contribution.
