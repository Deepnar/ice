## Architecture Specification — v1.0

---

## Abstract

The Infinite Context Engine (ICE) is a middleware system that wraps local LLM inference to provide persistent, structured, multi-tier memory and long-horizon cognitive continuity for human-AI conversational interfaces. It is a conversational-use system, not an agentic code-execution system. Its target use cases are those where a human engages repeatedly over weeks or months on evolving topics: long-form creative writing, deep technical projects, academic work, and personal knowledge management.

The core problem ICE addresses is **context collapse**: the stateless nature of standard chat interfaces forces users to re-explain architectural decisions, character lore, personal preferences, and project context at the start of every session. ICE eliminates this by intercepting every conversation turn, classifying its intent and topic, routing it to the appropriate local model, managing structured retrieval from four distinct memory stores, maintaining a persistent knowledge graph that evolves over time, and accumulating behavioral and procedural knowledge through reflection passes.

ICE is more than a memory middleware layer. It is a **long-horizon conversational cognition system**: it adapts to the user's patterns, reinforces recurring workflows, watches for anomalies and contradictions, and surfaces structured knowledge at the right moment. Memory evolves. Patterns crystallize. The system gets better the longer it is used.

ICE runs entirely self-hosted on consumer-grade hardware with local inference as the default backend. All data remains on the user's machine. PostgreSQL is the unified data store. FastAPI is the middleware backbone. Celery over Redis handles all asynchronous background cognition.

---

## Design Goals

**G1 — Zero re-explanation.** Information established in any prior session is retrievable without user action, provided it was tagged as high-value at storage time.

**G2 — Sub-100ms routing latency.** The pre-flight classification pass completes before the user perceives delay. The PyTorch classifier is small enough to run on CPU.

**G3 — Retrieval precision over recall.** Fewer, highly relevant context fragments are preferred over a broad dump. Classifier-gated, session-diversified hybrid retrieval is the enforcement mechanism.

**G4 — Hardware budget discipline.** All background processing yields to active inference. The VRAM budget for the primary model is never compromised by background workers.

**G5 — User authority over memory.** Classifier outputs are probabilistic. User-level controls — scope selectors, bookmarks, manual Codex injection, memory slot edits — override classifier decisions unconditionally. Memory is collaborative, not fully autonomous.

**G6 — Research reproducibility.** All evaluation experiments are reproducible from a fixed seed and logged configuration.

**G7 — Raw data immutability.** No user-authored content is ever deleted. Compression, summarization, decay, and archival operate on derived representations only.

**G8 — Incremental deployability.** Each major subsystem is independently functional. The system reaches a useful state with only the classifier, episodic store, memory slots, and FastAPI proxy in place.

**G9 — Cognitive evolution.** The system grows more accurate, more personalized, and more contextually aware the longer it operates. Reflection passes, procedural extraction, and sentinel monitoring are the mechanisms of this evolution.

---

## System Overview

ICE inserts itself as an OpenAI-compatible HTTP proxy between the frontend (Open WebUI) and local inference (Ollama). Every request passes through a classification stage before any memory is accessed or any model is selected. The result of classification determines retrieval strategy, model selection, and prompt assembly.

```
Open WebUI Frontend
        |
        |  POST /chat/stream (SSE)
        v
ICE FastAPI Middleware
  [Pre-Flight Classifier]
        |
        v
  [Retrieval Orchestrator]  <-- Episodic + Codex + Procedural + RAG + Memory Slots
        |
        v
  [Prompt Assembler + KV Cache Manager]
        |
        v
  [Model Router + Registry]
        |
        v
  Ollama / llama.cpp (Local Inference)
        |
        v (SSE stream)
  Open WebUI Frontend

Background Plane (async, GPU-idle only, Celery + Redis):
  - Post-Flight Evaluator
  - Codex Extractor
  - Procedural Extractor
  - Reflection / Consolidation Worker
  - Compaction Worker
  - Clustering Worker
  - Sentinel Monitor

Storage Plane (PostgreSQL + pgvector):
  - episodic_memory
  - memory_slots
  - codex_entities + codex_edges + codex_events + codex_snapshots
  - procedural_memory
  - rag_documents + rag_chunks
  - context_clusters
  - sentinel_rules + sentinel_events
  - session_replays
  - idempotency_keys
```

---

## Architectural Invariants

These are hard constraints. No subsystem implementation, optimization, or future extension may violate them.

**INV-1 — Raw text is never deleted.** The `raw_text` column in `episodic_memory` is write-once. Summarization, compression, lossless flag changes, and archival all operate on derived fields only.

**INV-2 — The LLM never directly writes to the Codex or Procedural store.** All mutations are mediated by the Codex Extractor, Procedural Extractor, or Reflection Worker. LLM output is input to extraction pipelines; it is never executed as a direct write.

**INV-3 — All retrieval passes through the classifier gate.** No memory store is queried without a classification context (topic tags, intent tags, context reliance class) that constrains which stores are accessed and with what filters.

**INV-4 — Only currently-valid Codex edges participate in retrieval.** Edges with `valid_until IS NOT NULL` are excluded from all retrieval queries. They are preserved for auditability and explicit historical queries only.

**INV-5 — Background workers yield to active inference.** No background Celery task is acquired while GPU utilization exceeds a configurable threshold (default: 20%). Workers poll utilization before acquiring tasks.

**INV-6 — Idempotency is enforced at the worker boundary.** Every event consumed from the Redis message bus carries a content-derived idempotency key. Workers check the `idempotency_keys` table before processing; duplicate events are silently dropped.

**INV-7 — All Codex mutations are transactional.** Snapshot creation and compaction markers are written within a single database transaction. Partial writes are rolled back.

**INV-8 — The conversation scope filter is never widened by the retrieval engine.** If a conversation is scoped to a specific cluster set, the retrieval engine may only narrow that set (by classifier tags), never expand it.

**INV-9 — Pre-flight classification is stateless.** The pre-flight classifier receives only the current prompt. It has no access to conversation history, VRAM state, or retrieval results. All context-dependent decisions belong to the post-flight phase.

**INV-10 — Cloud routing is a failover path, not a primary path.** Cloud endpoints are only selected when local capacity is explicitly insufficient.

**INV-11 — Memory slots are always injected at session start.** Active memory slots are included in every prompt payload regardless of classifier output. They are not retrieval targets; they are persistent working memory.

**INV-12 — Bookmarked turns are immune to decay and archival.** Bookmarked turns are permanently lossless and are never moved to cold storage regardless of age or access frequency.

**INV-13 — Sentinel rules are declarative and version-controlled.** Sentinel rules are not procedural scripts. They declare trigger conditions and response actions. The Sentinel Monitor interprets them; they do not execute directly.


---

### Deferred Systems

The following systems are architecturally designed for but explicitly out of scope for V1. The schema supports them. The interfaces are designed to not foreclose them. They are not built prematurely.

**Conversation Branching UI:** The `parent_message_id` column exists in `episodic_memory` and defaults to NULL for all linear conversations. When non-NULL, it points to the turn before a branch point, enabling retrieval to follow branch-pointer chains rather than timestamp ordering. Full implementation requires a custom chat frontend — Open WebUI does not support branching natively. The retrieval logic switch (timestamp order → parent-pointer traversal when `parent_message_id IS NOT NULL`) requires no schema migration when eventually implemented. Deferred until custom frontend work begins.

**Multi-user / Team Memory:** The schema uses `conversation_id`, `cluster_id`, and is designed for `USER_ID` namespacing. The memory scope system (None / Auto / Project / Manual) already handles isolation between conversations. Multi-user routing and shared-vs-private memory regions are structurally prepared. Not built for V1.

**P2P Mesh Sync:** Multiple ICE instances on different machines synchronizing shared memory clusters. The cluster and conversation_id structure supports this. Not built for V1.

**Full Custom Frontend:** Open WebUI is the interim frontend. A custom frontend with branching support, cluster navigation, Codex graph browser, Bookmark index, and full memory health dashboard is a longer-term deliverable. All backend endpoints are designed to support this frontend when it exists.

**LLM Classifier (for research comparison only):** The `IntentClassifier` abstract interface supports an `LLMClassifier` implementation that routes classification to an LLM rather than the PyTorch MLP. This exists solely as a research comparison target for Paper 1. It is never the production path due to latency cost.

---

## Architectural Note: The Asymmetrical Value Problem

This is one of the deepest architectural insights underpinning the design of ICE and must be understood to reason correctly about the pre-flight/post-flight split and the lossless classification system.

The problem: **a short, low-information prompt may elicit an extremely high-value response.** A prompt like "write the FastAPI route" is three words. The model's response may be 200 lines of production-quality code that the user will reuse across the project. Conversely, a long, emotionally rich prompt like "I feel terrible today, I've been thinking about everything going wrong" will elicit an empathetic response that is high-value emotionally but low-value informationally for long-term memory.

The consequence: **memory valuation cannot occur pre-flight.** If the system decides how to store a turn based only on the user's prompt, it will systematically mis-classify code-generation turns (short prompts, high-value responses) and over-retain emotional venting (long prompts, low-value for future retrieval). The lossless flag cannot be set until the response is seen.

This single insight drives several design decisions:
- The pre-flight/post-flight classification split is not optional overhead. It is architecturally necessary.
- The Post-Flight Evaluator, not the pre-flight classifier, sets the lossless flag.
- Post-flight evaluation has access to the full exchange and examines the response, not just the prompt.
- All summarization and Codex extraction happen post-flight, after the value of the turn is known.

Any refactoring that moves lossless classification to pre-flight violates this principle and will produce systematically incorrect memory hierarchies.

---

## Major Subsystems

---

### 1. Classification Engine

#### 1.1 Pre-Flight Classifier

A feed-forward multi-layer perceptron (MLP) that maps a user prompt to a 25-dimensional probability vector across three independent label dimensions: 11 Topic labels, 11 Intent labels, and 3 Context Reliance classes.

Architecture: A frozen SentenceTransformer encoder (all-MiniLM-L6-v2) converts the prompt to a 384-dimensional dense vector. Two linear layers (384 → 128 → 25) with ReLU activation and dropout produce raw logits. Topic and Intent labels are independently sigmoid-activated (multi-label). Context Reliance is softmax-activated (exactly one active class).

**Topic Labels:** Software_&_Tech, STEM_&_Academics, Business_&_Finance, Creative_&_Media, Admin_&_Productivity, Lifestyle_&_Health, Social_&_Relationships, World_&_Current_Events, Meta_AI, Null_Noise, General_Reference_&_Trivia

**Intent Labels:** Factual_Retrieval, Troubleshooting, Generation, Ideation, Analysis_&_Summarization, Strategic_Planning, Decision_Making, Emotional_Processing, Utility_Formatting, Casual_Banter, Open_Exploration

**Context Reliance Classes:** Zero_Shot (no retrieval), Long_Term_Memory (query memory stores), Real_Time_Search (web search pipeline)

Any label exceeding the active threshold (default: 0.50) is considered active. Multiple topics and intents may be simultaneously active. Exactly one Context Reliance class is active.

Training: 20,000-prompt blended dataset (5k personal, 5k WildChat, 5k LMSYS Chatbot Arena, 5k ShareGPT). Labels generated by a 70B instruct model with chain-of-thought prompting and strict JSON schema. Loss: BCEWithLogitsLoss per label. Model size: approximately 5MB weights.

The classifier is exposed through a dependency-injectable abstract interface (IntentClassifier) with a concrete PyTorchClassifier implementation. An LLMClassifier implementation is architecturally supported for research comparison without changing middleware code.

#### 1.2 Confidence Threshold Safety Net

After the pre-flight pass, the middleware extracts the maximum probability across all 25 labels. If this value falls below a configurable threshold (default: 0.75), the system declares an uncertain state and falls back to wide-net retrieval: query the last 20 Episodic turns, pull the top Codex nodes by keyword overlap, run vector similarity over the RAG store, merge and deduplicate by content hash, and route to the generalist model with an uncertainty annotation in the system prompt.

Special case: if the dominant label is Null_Noise or Casual_Banter, even at low confidence, the system routes to a small fast model without retrieval.

#### 1.3 Pre-Flight / Post-Flight Classification Split

Classification is divided into two structurally distinct phases.

**Pre-flight:** Executes within 10–50ms of user input. Input: user prompt only. Responsible for routing, retrieval selection, and model selection. Stateless. Runs synchronously in the FastAPI middleware.

**Post-flight:** Executes seconds to minutes after the LLM response completes. Input: full exchange (prompt + response + tool outputs). Responsible for: setting the lossless flag, generating summary text, extracting Codex entities, extracting procedural patterns, correcting pre-flight tag errors, and triggering reflection workers. Runs asynchronously in the Celery background plane.

The split exists because pre-flight has only the prompt, which is frequently ambiguous. The post-flight phase has the response, which reveals the true value of the exchange. This is a direct consequence of the Asymmetrical Value Problem.

#### 1.4 Manual Label Correction and Fine-Tuning Loop

A FastAPI endpoint accepts a batch ID and corrected topic/intent tags. Corrected labels are stored in a curated_labels table. On a configurable schedule (default: weekly), the system runs a fine-tuning pass: all layers except the final classification head are frozen, and the model trains on curated examples for a small number of epochs at a very low learning rate. This incrementally improves classifier accuracy on the user's specific prompt patterns without full retraining.

#### 1.5 Training Pipeline — Data Refinery

The classifier is trained on a blended 20,000-prompt dataset built through a structured extraction and labeling pipeline before any model training occurs.

**Stage 1 — Raw Log Extraction (Amnesia Method):**
Personal prompt data is extracted from unstructured historical chat log files where human and AI turns are interleaved with no consistent delimiter. The extraction strategy uses overlapping chunks: 3,000-character chunks with 500-character overlap (chunk N covers chars 0–3,000; chunk N+1 covers chars 2,500–5,500). Each chunk is sent independently to the 1.5B background model with the instruction: "Identify all text written by the human user in this excerpt. Output as a JSON list of strings. If nothing was written by a human user, output an empty list."

Between each chunk, the model connection is closed and reopened — the Amnesia Method. This enforces stateless processing: the model has no memory of previous chunks. VRAM usage remains constant regardless of log file size, and there is no risk of cross-chunk contamination in extractions. After all chunks are processed, results are loaded into a Pandas DataFrame and deduplicated by content hash. Output: a clean JSONL file of human-authored prompts, one per line.

**Stage 2 — Public Dataset Integration:**
WildChat (5k samples), LMSYS Chatbot Arena (5k samples), and ShareGPT (5k samples) are downloaded and filtered to English-only, human-authored turns. These are sampled to complement the personal prompt distribution, not to dominate it. Combined with the 5k personal prompts, the blended dataset reaches 20,000 entries.

**Stage 3 — LLM Labeling:**
Every prompt is labeled by a locally-run 70B instruct model (Llama-3-70B or Qwen2.5-72B) using chain-of-thought prompting with a strict JSON output schema. The labeling prompt instructs the model to: (1) reason step-by-step about the prompt's topic and intent, (2) output a JSON object containing `topic_labels` (list of active topics), `intent_labels` (list of active intents), and `context_reliance` (exactly one of: Zero_Shot, Long_Term_Memory, Real_Time_Search). Temperature is set to 0.0 for labeling to ensure deterministic output. Labels are converted to multi-hot encoded binary vectors of dimension 25.

**Stage 4 — Training:**
A PyTorch `Dataset` subclass reads the labeled JSONL, encodes each prompt through the frozen SentenceTransformer, and returns `(embedding_tensor, label_tensor)` pairs. A `DataLoader` wraps this with batch size 32, shuffling enabled, and multi-worker loading. Training loop: forward pass → BCEWithLogitsLoss (Topic + Intent heads) + CrossEntropyLoss (Context Reliance head) → backward → Adam optimizer (lr=1e-3). 20–50 epochs with early stopping on validation loss. Final model weights (~5MB) saved to disk.

**Reproducibility:** All training runs require a `--seed` argument that sets the random state for PyTorch initialization, the 70B labeling model temperature (enforced at 0.0), the SentenceTransformer embedding calls, and dataset shuffle order. Runs without a seed are invalid for paper methodology. Every training run logs: seed, dataset path, epoch count, final validation loss, and model checkpoint path to a `training_runs` log file.

---

### 2. Persistent Memory Slots

Memory Slots are a dedicated subsystem distinct from all other memory stores. They represent **persistent structured working memory**: facts, preferences, guidelines, and context that should be injected into every session without requiring retrieval. They are not Codex nodes. They are not episodic turns. They exist above the retrieval plane.

#### 2.1 Slot Types

| Slot Name | Content | Example |
|---|---|---|
| `persona` | How the AI should present itself in this user's sessions | "You are a systems architect who challenges assumptions" |
| `user_preferences` | Stable user preferences about style, format, interaction | "I prefer dense technical explanations with no bullet points" |
| `tool_guidelines` | Which tools to prefer, which to avoid, and why | "Always use pgvector for vector search, never ChromaDB" |
| `project_context` | The user's active projects, their current state, and open threads | "Currently building ICE. FastAPI backend. Legion GPU." |
| `guidance` | Behavioral rules the AI should follow | "Always ask before deleting anything. Prefer reversible operations." |
| `pending_items` | Outstanding tasks, unresolved questions, deferred decisions | "Still need to design the Sentinel schema" |
| `session_patterns` | Recurring patterns the system has learned about the user's behavior | "User codes in long bursts, then asks for architectural review" |

#### 2.2 Properties

Memory slots are:
- Persistent across all sessions by default
- Size-limited structured storage (each slot has a configurable maximum token budget, default: 300 tokens per slot)
- Injected at session start as a dedicated `[PERSISTENT CONTEXT]` block, before Codex and Episodic content
- Directly editable by the user through the UI or through the FastAPI admin endpoint
- Updatable by the Reflection Worker, which may propose slot updates after detecting stable patterns (with user confirmation before write)
- Version-tracked: every slot write is logged with a timestamp and source (user or reflection)

#### 2.3 Injection Position

Memory slots occupy the position immediately after the system rules block and before all retrieved memory in the prompt payload:

```
[1] SYSTEM RULES
[2] PERSISTENT CONTEXT (Memory Slots)   <-- always present
[3] CODEX: ABSOLUTE FACTS               <-- retrieval-conditional
[4] EPISODIC CONTEXT                    <-- retrieval-conditional
[5] PROCEDURAL CONTEXT                  <-- retrieval-conditional
[6] USER INPUT
```

This ordering ensures that stable user identity, preferences, and project context are always available to the model before any retrieved material. Memory slots are part of the stable prefix for KV cache purposes.

#### 2.4 Slot Update Authority

Users have unconditional write authority over their own memory slots. The Reflection Worker may propose updates (especially to `session_patterns` and `project_context`) but writes only after explicit user confirmation. The `pending_items` slot is the one exception: the Reflection Worker is permitted to append to it automatically, since appending an unresolved question is low-risk; deletions from `pending_items` require user confirmation.

---

### 3. Memory Systems

ICE maintains four structurally distinct memory stores. They serve different retrieval tasks, are built by different processes, and are queried independently based on classifier output.

#### 3.1 Episodic Memory

Episodic memory stores every conversational turn as a row in PostgreSQL.

**Core schema columns:** id (UUID), conversation_id, cluster_id, parent_message_id (for branching, nullable), batch_id, timestamp, topic_tags, intent_tags, context_reliance, entropy_score (float, post-flight), lossless_flag (boolean, post-flight), raw_text (write-once), summary_text (nullable, post-flight), embedding (vector(384)), decay_score (float, starts at 1.0, decreases over time without access), access_count (incremented on retrieval), is_archived (boolean), idempotency_key.

**Lossless flag:** Set by the Post-Flight Evaluator. Criteria for lossless=true: presence of code blocks, structured data, specific named entities, or information density above threshold. This is a post-flight decision because of the Asymmetrical Value Problem.

**Retrieval:** When context_reliance is Long_Term_Memory, the retrieval engine executes a time-weighted cosine similarity search via pgvector, filtered by topic_tags and cluster_id (when scoped), excluding archived rows with decay_score below the archival threshold.

**Injection:** For lossless=true rows, raw_text is injected. For lossless=false rows, summary_text is injected. This distinction is the primary token-budget control mechanism.

**Sliding window:** The active context window includes the last N turns by chronological or branch-pointer position (default: 10). Older turns remain in the database and are retrievable by semantic search but are not blindly included in every call.

**Conversation branching (schema-ready, deferred):** Every episodic turn carries a `parent_message_id` column (UUID, nullable FK self-referencing `episodic_memory.id`). In all current linear conversations this column is NULL. When a user edits a past message and resubmits, the original turn and its descendants remain untouched; a new turn is created with `parent_message_id` pointing to the turn immediately before the edit point, creating a new branch. Retrieval reconstructs context by following `parent_message_id` pointers backward from the current turn to the root, then reversing — naturally following whichever branch is active without mixing contexts from parallel branches. The column exists now so that no schema migration is required when branching is implemented. See Deferred Systems for full implementation notes.

#### 3.2 Semantic Memory — The Codex

The Codex is a knowledge graph stored in PostgreSQL as codex_entities (nodes) and codex_edges (directed, typed relationships). It is the factual, relationship-aware, temporally versioned long-term memory of the system.

**Entity node:** id (UUIDv5 from canonical name), canonical_name, aliases (for fuzzy matching), tags, properties (JSONB), context_payload (Markdown, injected verbatim), last_updated.

**Edge:** source_id, target_id, relation (verb phrase), strength (float, incremented on corroboration), source_batch, confidence (pending | active), valid_from, valid_until (NULL = currently true).

**Temporal versioning:** When the extractor identifies a contradicting fact, the old edge's valid_until is set and a new edge is inserted with valid_until=NULL. Retrieval unconditionally filters WHERE valid_until IS NULL. Historical facts are preserved and queryable explicitly.

**Truth quorum:** Edges from a single source batch are assigned confidence=pending and are not used in retrieval. Promotion to active requires corroboration from a second independent batch or confirmation by the background validation pass.

**Entity identity:** UUIDv5 derived from canonical name. Alias lookup merges variant names into existing entities to prevent fragmentation.

**Graph construction:** The Codex Extractor worker scans new episodic entries, prompts the 1.5B background model for subject-relation-object triplets, resolves entity IDs, checks for contradictions, and writes all changes as codex_events within a transaction.

**Retrieval:** NER pass on the prompt extracts candidate entity names. Candidates are looked up via canonical name and alias match. The graph is traversed 1–2 hops outward. Collected entity context_payloads are concatenated and injected as a CODEX: ABSOLUTE FACTS block.

**Manual injection:** Users may write entity YAML files to a /codex_inject directory. A watcher process applies them as explicit Codex events, bypassing LLM extraction. This is the highest-authority write path in the system.

#### 3.3 Procedural Memory

Procedural memory is a first-class memory tier, distinct from episodic and semantic memory. It stores **recurring workflows, coding patterns, user habits, reusable solution strategies, and behavioral sequences** that have been observed across multiple sessions.

Where episodic memory records what happened and the Codex records what is true, procedural memory records **how the user consistently does things.**

**Examples of procedural memory entries:**
- "When starting a new FastAPI route, the user always creates a Pydantic model first, then the endpoint, then the test."
- "The user prefers to review database schema before writing any ORM models."
- "When debugging, the user asks for the error message first, then the surrounding code, then the fix."
- "The user's preferred project structure for Python backends: src/, tests/, alembic/, docker/."

**Schema (procedural_memory table):** id (UUID), pattern_name (short descriptor), pattern_description (full natural language description), topic_tags, trigger_conditions (JSONB: when to surface this pattern), reinforcement_count (incremented each time the pattern is observed), confidence_score (float), first_observed (timestamp), last_observed (timestamp), is_active (boolean), source_batch_ids (array of batches where pattern was observed).

**Extraction:** The Procedural Extractor worker scans episodic batches after post-flight evaluation. It prompts the 1.5B model: "Identify any recurring workflows, decision sequences, or behavioral patterns in this exchange that represent how the user approaches problems. If no recurring pattern is evident, output empty." Extracted patterns are inserted as new procedural entries or merged (reinforcement_count incremented) with existing matching entries. Matching uses embedding similarity on the pattern_description field.

**Reinforcement:** Each time the Procedural Extractor observes a pattern already in the store, it increments reinforcement_count and updates last_observed. Patterns are promoted to high confidence only after being observed a configurable number of times (default: 3). Low-count patterns are stored but not injected unless explicitly retrieved.

**Retrieval:** When the classifier detects intents involving Strategic_Planning, Generation (especially project scaffolding), or Open_Exploration, the retrieval engine queries procedural_memory for patterns whose topic_tags overlap with the active topic and whose trigger_conditions match the current context. Retrieved patterns are injected as a PROCEDURAL CONTEXT block in the prompt.

**Decay:** Procedural patterns decay slowly (much slower than episodic turns) because user habits are stable over long periods. A pattern is considered stale only if it has not been observed in a configurable window (default: 6 months) and its reinforcement_count is below a threshold.

#### 3.4 Vector RAG Store

Standard vector embeddings of chunked document content, stored via pgvector. Used exclusively for static reference material: uploaded PDFs, documentation, textbooks. No structural awareness of relationships, chronology, or entity boundaries.

**Activation:** Only when context_reliance=Long_Term_Memory AND intent is Factual_Retrieval or Analysis_&_Summarization AND the prompt contains explicit reference language. In all other cases, the RAG store is bypassed.

**Why separate:** Vector RAG shreds documents into fixed-size chunks and loses structure. The Codex preserves structure but requires extraction effort. The Episodic store preserves chronology but lacks factual compression. Procedural memory records behaviors. Each is optimal for a different retrieval task. They are not fallbacks for each other.

#### 3.5 Bulk Ingestion Pipeline — Drop Zone

The Drop Zone is a watched `/ingest_inbox` directory that accepts external documents and raw conversation exports for integration into the memory system. Files dropped here are automatically processed through a four-stage pipeline: `Extractor → Chunker → Tagger → Ingester`.

**Why this exists:** Open WebUI's default RAG pipeline performs naive character-count chunking with no structural awareness and no metadata tagging. It cannot distinguish a human prompt from an AI response, cannot assign classifier tags, and cannot respect conversational boundaries. The Drop Zone pipeline is conversational-structure-aware: it understands turn boundaries, extracts only human-authored content where applicable, applies classifier metadata, and produces records that are queryable by topic, intent, and cluster — not just by vector similarity.

**Pipeline stages:**

*Extractor:* Format-specific extraction logic. For raw chat log files (plain text), the Amnesia Method (§1.5) is applied to isolate human-authored content. For PDFs and EPUB files, text is extracted and split at paragraph or section boundaries. For structured exports (JSONL conversation exports from other platforms), a dedicated parser reads the turn structure directly. Each `Extractor` implementation exposes a single `extract(file_path) → List[str]` interface. Adding support for a new format requires only a new Extractor class; the rest of the pipeline is unchanged.

*Chunker:* Splits extracted text into appropriately-sized units. For conversational content, chunk boundaries respect turn boundaries — an AI response is never split mid-sentence to satisfy a token limit. For document content, chunks are 512 tokens with 64-token overlap.

*Tagger:* Runs the pre-flight classifier over each chunk to assign topic_tags, intent_tags, and context_reliance. Also computes entropy_score. This metadata makes ingested content queryable by the same classifier-gated retrieval pipeline as live conversation turns.

*Ingester:* Deduplicates by content hash against existing rows in `episodic_memory` and `rag_chunks`. Non-duplicate chunks are inserted with appropriate lossless flags (set by entropy_score threshold), assigned a synthetic batch_id, and queued for the Codex Extractor worker. Documents with no structural relationship to conversational memory are inserted into `rag_chunks` instead.

**Deduplication:** Every ingested chunk generates a SHA256 content hash before insert. The hash is checked against existing rows. Duplicate hashes are silently skipped. This prevents re-ingesting the same source file from polluting the database with identical records.

**Initialization use case:** Before the first ICE session, the user may drop pre-written Codex entity files (YAML format, processed via the `/codex_inject` watcher), example conversation transcripts, and reference documents into the appropriate ingest directories. The system builds an initial knowledge graph and memory base from these, so the first live session begins with a populated Codex rather than complete amnesia.

#### 3.6 Memory Tier Summary

| Dimension | Episodic | Codex (Semantic) | Procedural | Vector RAG |
|---|---|---|---|---|
| Storage unit | Conversational turn | Named entity + typed edges | Recurring workflow / pattern | Document chunk |
| Structure | Temporal sequence | Directed graph | Pattern + trigger conditions | Flat embedding space |
| Built by | Middleware (automatic) | Codex Extractor (NLP) | Procedural Extractor (NLP) | User upload pipeline |
| Retrieval method | pgvector cosine + time-weight | Graph traversal from NER | pgvector cosine + trigger match | pgvector cosine |
| Temporal versioning | Yes (timestamps, branching) | Yes (valid_until on edges) | Yes (reinforcement count, last_observed) | No |
| Decay | Yes (decay_score) | Slow (strength decay) | Very slow (reinforcement-weighted) | No |
| Activated by | Long_Term_Memory + episodic intents | Long_Term_Memory + entity/lore intents | Planning, generation, open exploration | Long_Term_Memory + reference language |

---

### 4. Memory Lifecycle and Decay

Memory in ICE is not static. Entries are created, strengthened, weakened, archived, and structurally frozen through a lifecycle that runs continuously in the background.

#### 4.1 Lifecycle States

```
Created (raw_text written, lossless_flag NULL)
    |
    v [Post-Flight Evaluator]
Evaluated (lossless_flag set, summary_text generated if needed)
    |
    v [Codex / Procedural Extractor]
Processed (entities and patterns extracted)
    |
    v [Clustering Worker]
Clustered (cluster_id assigned)
    |
    |-- [Bookmark action] --> FROZEN (lossless=true, decay_immune, Codex linked)
    |
    v [Decay Worker, periodic]
Decaying (decay_score decreasing over time without access)
    |
    |-- [Retrieval access] --> decay_score partially restored (strengthened)
    |
    v [Archival threshold crossed]
Archived (is_archived=true; excluded from default retrieval; raw_text preserved)
    |
    v [Manual or time threshold]
Cold Storage (moved to cold_storage table; queryable only on explicit request)
```

#### 4.2 Decay Mechanics

Every episodic turn has a `decay_score` initialized at 1.0. The Decay Worker (periodic Celery task) applies a decay function to all non-bookmarked, non-lossless turns older than a configurable minimum age (default: 7 days). The decay function is time-weighted and access-weighted: turns that have never been retrieved decay faster; turns that have been retrieved recently decay more slowly.

Decay does not delete. It lowers the `decay_score`. Retrieval queries apply a minimum decay_score filter (default: 0.2) to exclude very stale turns from default retrieval results. Turns with decay_score below the archival threshold (default: 0.1) are flagged as `is_archived=true` and excluded from all retrieval unless the user explicitly queries archived content.

**Strengthening:** When a turn is retrieved and injected into a prompt, its `access_count` is incremented and its `decay_score` is partially restored (by a configurable amount, default: +0.15, capped at 1.0). Frequently retrieved turns remain perpetually active regardless of age.

#### 4.3 Cold Storage

Archived turns (is_archived=true, decay_score below cold threshold) are moved periodically to a `cold_storage` table. Cold storage is a separate table optimized for sequential scans rather than vector similarity. Turns in cold storage are invisible to the default retrieval pipeline but are queryable explicitly through a dedicated FastAPI endpoint or through the Session Replay interface.

**Raw text is never deleted, even from cold storage.** Cold storage is the archive, not the trash.

#### 4.4 Codex Decay

Codex edge strength values decay slowly when the edges are not referenced in retrieval sessions. An edge whose strength falls below a minimum threshold is not deleted — it is demoted to confidence=pending, making it invisible to active retrieval but preserving the historical record. Strength is restored when the edge is referenced again or when corroborating evidence is extracted.

---

### 5. Sentinel System

The Sentinel System is a **reactive cognition infrastructure** that monitors the memory ecosystem and triggers automatic background actions when declarative conditions are met. Sentinels are the mechanism by which ICE detects systemic problems, memory health degradation, and opportunities for proactive knowledge maintenance.

Sentinels do not contain procedural logic. They declare: *when these conditions are true, take this action.* The Sentinel Monitor interprets them.

#### 5.1 Sentinel Rule Schema

Each sentinel rule is stored in the `sentinel_rules` table with the following structure:

**Fields:** id, name (human-readable descriptor), description, is_active, trigger_type (threshold | frequency | absence | contradiction | composite), trigger_conditions (JSONB: the declarative condition expression), action_type (notify | schedule_worker | create_review_item | log_event | propose_memory_update), action_payload (JSONB: parameters for the action), cooldown_seconds (minimum time between fires for the same rule), last_fired_at, created_at.

#### 5.2 Trigger Types

**Threshold:** A numeric measure crosses a defined boundary. Examples: a Codex entity's edge count exceeds 50 (potential concept fragmentation), a cluster's episodic turn count exceeds 500 (cluster may need splitting), the unresolved contradiction count in codex_edges exceeds 10.

**Frequency:** A topic, entity, or pattern appears with unusual frequency across recent sessions. Example: the same Codex entity has been referenced in retrieval across 5 consecutive sessions (likely warrants context_payload review and enrichment).

**Absence:** Something expected is missing. Example: a pending_items slot entry that has not been referenced or resolved in 14 days (may be stale or forgotten), a cluster that has had no new turns in 30 days (potentially dormant project).

**Contradiction:** A structural inconsistency is detected. Example: two active codex_edges exist between the same source and target with conflicting relation values and no valid_until set on either (unresolved graph contradiction), a procedural pattern's trigger_conditions conflict with an existing memory slot guideline.

**Composite:** A logical combination of the above trigger types, expressed as a JSON expression tree.

#### 5.3 Action Types

**notify:** Surfaces an alert in the SSE telemetry stream during the next session, visible in the observability panel. Does not block the pipeline.

**schedule_worker:** Enqueues a specific Celery task (e.g., schedule the Reflection Worker to run a focused consolidation pass on a specific cluster, or schedule the Codex Extractor to re-process a set of batches).

**create_review_item:** Appends an entry to a `review_queue` table, visible in the admin UI, requesting the user to manually review a specific Codex entity, memory slot, or cluster.

**log_event:** Records a sentinel firing event in `sentinel_events` table for observability and research measurement purposes.

**propose_memory_update:** Constructs a proposed change to a memory slot or Codex entity and presents it to the user for confirmation in the next session's SSE telemetry.

#### 5.4 Example Sentinel Rules

- **Stale pending items:** If any entry in the pending_items memory slot has been present for more than 14 days and has not been referenced in any recent retrieval, fire action=notify to surface it.

- **High-contradiction entity:** If a Codex entity has more than 3 edges in pending state and more than 2 active edges with overlapping relation types on the same target, fire action=create_review_item.

- **Retrieval health degradation:** If the hybrid retrieval pipeline returns zero results from any store for more than 5 consecutive Long_Term_Memory classified turns in the same conversation, fire action=schedule_worker for the Clustering Worker (cluster assignment may have drifted).

- **Topic persistence signal:** If the same topic_tag has been the dominant classifier output for more than 10 consecutive sessions, fire action=propose_memory_update to add or update the project_context memory slot.

- **Codex entity staleness:** If a Codex entity's context_payload has not been updated in more than 60 days but continues to appear in retrieval results, fire action=create_review_item for manual review of whether the entity's facts are still current.

- **Procedural pattern emergence:** If a procedural pattern's reinforcement_count reaches the promotion threshold for the first time, fire action=notify to inform the user that a behavioral pattern has been crystallized and ask if they want to review it.

#### 5.5 Sentinel Monitor

The Sentinel Monitor is a Celery beat task (scheduled periodic task) that runs on a configurable interval (default: every 30 minutes during active use, every 6 hours during idle periods). It loads all active sentinel rules from the database, evaluates each rule's trigger_conditions against the current system state, and fires the appropriate action for any rule that triggers and has exceeded its cooldown window.

The monitor does not fire rules during active inference. It checks GPU utilization (respecting INV-5) before beginning evaluation.

---

### 6. Reflection and Consolidation System

The Reflection Worker executes periodic post-session and post-batch consolidation passes that produce higher-order knowledge from accumulated episodic content. This is the mechanism of cognitive evolution in ICE: raw conversational turns are distilled into stable behavioral patterns, preference updates, procedural workflows, and project context summaries.

#### 6.1 Trigger Conditions

The Reflection Worker is triggered by:
- Session end event (CHAT_SESSION_ENDED emitted to Redis when conversation closes)
- Scheduled periodic consolidation (configurable, default: daily)
- Explicit Sentinel action (schedule_worker pointing to reflection)
- Manual trigger via FastAPI admin endpoint

#### 6.2 Reflection Tasks

**Session synthesis:** After a session ends, the worker reads all episodic turns from that session and generates a structured session summary: what topics were covered, what decisions were made, what was left unresolved, and what new entities or patterns appeared. This summary is written to a `session_summaries` table and optionally used to update the pending_items memory slot.

**Pattern crystallization:** The worker scans episodic turns from the last N sessions (configurable, default: the last 5 sessions or 30 days, whichever is shorter) and prompts the 1.5B model to identify recurring behavioral sequences not already captured in procedural_memory. Novel patterns are sent to the Procedural Extractor for evaluation. Strengthened patterns (appearing again) trigger reinforcement_count increments.

**Memory slot evolution:** The worker analyzes recent episodic content for evidence that the user's stated preferences (in user_preferences), active project context (in project_context), or behavioral guidelines (in guidance) have evolved or been contradicted by recent behavior. Proposed slot updates are presented to the user for confirmation at the start of the next session.

**Codex enrichment:** The worker identifies Codex entities that appear frequently in recent retrieval but whose context_payload is thin (below a length threshold or lacking certain expected fields like aliases or properties). It constructs enrichment proposals: passages from the episodic store that could be appended to the entity's context_payload. Enrichment writes require no user confirmation; they append, never replace.

**Motif detection:** The worker looks for recurring thematic motifs across sessions that do not yet correspond to any named cluster or procedural pattern. Motifs are surfaced as cluster creation proposals for the Clustering Worker.

#### 6.3 Reflection Output Destinations

| Reflection task | Output destination |
|---|---|
| Session synthesis | session_summaries table; optional pending_items update |
| Pattern crystallization | Procedural Extractor pipeline |
| Memory slot evolution | Review queue (user confirmation required before write) |
| Codex enrichment | codex_events (append type, no user confirmation) |
| Motif detection | Clustering Worker queue |

---

### 7. Bookmarking System

Bookmarking is the primary mechanism for **human-guided memory reinforcement**. It allows the user to directly signal that a specific conversational turn is high-value and should be permanently prioritized in the memory hierarchy, regardless of what the classifier or Post-Flight Evaluator would have decided automatically.

#### 7.1 Mechanics

When the user clicks the bookmark control on any AI response:

1. The turn's lossless_flag is immediately set to true, overriding any post-flight evaluation (INV-12).
2. A priority processing task is enqueued for the Codex Extractor, bypassing the GPU-idle scheduler. The extractor runs on this turn immediately, regardless of current GPU utilization.
3. A structured summary is generated in meeting-minutes format: what was discussed, what decisions were made, what action items were identified, what is still unresolved.
4. A Bookmark node is created in the Codex, linked to all entities mentioned in the turn via REFERENCES edges.
5. The turn is marked decay_immune: it will never have its decay_score reduced, and it will never be archived or moved to cold storage.
6. An entry is appended to the pending_items memory slot if any unresolved action items were identified in the meeting-minutes summary.

#### 7.2 Retrieval Priority

Bookmarked turns receive a retrieval priority boost in the hybrid retrieval pipeline. When the retrieval orchestrator merges results from multiple stores, bookmarked entries are scored higher than equivalent-similarity non-bookmarked entries by a configurable multiplier (default: 1.5×).

#### 7.3 Bookmark Index

The bookmarks are exposed through a dedicated `/bookmarks` endpoint and in the session viewer UI as a filterable, searchable index. The user can view all bookmarks, navigate to the original conversation context, and promote bookmarked content into explicit Codex entities if they are not already there.

---

### 8. Cross-Chat Memory Scoping

Memory scope is a per-conversation configuration that determines which memory stores are accessible during retrieval for that conversation. It is set when a conversation is created and is stored in the `conversations` table (`memory_scope_type`, `cluster_ids`, `custom_filter` columns). The retrieval orchestrator enforces it on every retrieval query within that conversation.

#### 8.1 Scope Types

**None:** No shared memory. The conversation is fully private. Episodic turns created in this conversation are stored with their `conversation_id` but are excluded from all cluster assignments and are not retrievable by any other conversation. Turns may be manually promoted to a cluster via the admin endpoint. The `conversations` row has `memory_scope_type=none`, `cluster_ids=[]`.

**Auto (default):** Classifier-driven retrieval with no cluster filter. The retrieval orchestrator queries all clusters accessible to the user. This is the standard operating mode for general use.

**Project:** Hard-scoped retrieval to one or more specific clusters. The retrieval orchestrator appends `AND cluster_id = ANY(:cluster_ids)` to all Episodic queries. Codex graph traversal is also restricted: only entities that appear in turns belonging to the selected cluster(s) are considered. This prevents a coding conversation from pulling in story lore, and prevents a creative session from pulling in unrelated technical decisions. `conversations` row has `memory_scope_type=project`, `cluster_ids=[uuid1, uuid2, ...]`.

**Manual:** The user supplies a custom SQL filter expression (validated against an allowlist of safe column references before execution). Example: `topic_tags @> ARRAY['Software_&_Tech'] AND timestamp > '2025-01-01'`. The retrieval orchestrator appends this expression as a WHERE clause. `conversations` row has `memory_scope_type=manual`, `custom_filter='...'`.

#### 8.2 Scope Enforcement

Scope enforcement is the retrieval orchestrator's responsibility, not the classifier's. The classifier produces labels that narrow the scope further (e.g., a Project-scoped conversation also filtered by `topic_tags`). The scope filter is always applied first as the outer constraint; classifier filters are applied within that constraint. INV-8 states that the scope filter is never widened by the retrieval engine — it may only be narrowed.

#### 8.3 Scope Display and Modification

The active scope is surfaced in the session header visible to the user. Example: `Memory: FastAPI Backend + ICE`. Scope changes mid-conversation are permitted but are logged as system events in `session_replays`. The user is notified via the SSE telemetry stream when scope changes take effect.

#### 8.4 Scope and Codex Traversal

When a conversation is Project-scoped, Codex entity lookup is filtered to entities whose `source_batch` references are associated with turns in the selected cluster(s). Entities that exist in the Codex but were extracted exclusively from turns outside the selected clusters are excluded from graph traversal. This is a hard boundary that prevents semantic bleed between domains. The user may explicitly bypass this for a specific session by requesting Auto scope.10
---

### 9. Orchestration Layer

#### 9.1 FastAPI Middleware

ICE exposes an OpenAI-compatible API surface. The primary endpoint is POST /chat/stream, which returns a StreamingResponse with media_type=text/event-stream. The middleware executes the full pipeline and interleaves SSE status events with LLM token events.

**Request lifecycle:**
1. Receive POST /chat/stream
2. Generate correlation_id (UUID); attach to all downstream log statements
3. Inject persistent memory slots into prompt context (unconditional; INV-11)
4. Pre-flight: classify(prompt) → ClassificationResult
5. Check confidence threshold → normal routing or wide-net fallback
6. Apply conversation scope filter (cluster_ids from session metadata)
7. Execute Hybrid Retrieval Orchestrator (§8.3)
8. HyDE query rewriting if applicable (§8.4)
9. Assemble prompt: SYSTEM + SLOTS + CODEX + EPISODIC + PROCEDURAL + RAG + USER
10. Validate assembled token count against model context_window; trim if needed
11. KV cache prefix validation
12. Model selection via registry
13. Dual-agent routing check (if conflicting intents)
14. Stream inference from Ollama; yield tokens as SSE events
15. Emit CHAT_COMPLETED event to Redis (with idempotency_key)
16. Close SSE connection

All log output is structured JSON with correlation_id, conversation_id, classifier_result, model_selected, retrieval_sources, and tokens_fetched fields on every request.

#### 9.2 Dynamic Model Registry

The model registry is a JSON file mapping each locally-installed Ollama model to a topic/intent profile, a priority score, and a context window limit.

**Selection logic:** Scan for best topic AND intent overlap. Among all matching entries, select highest priority. If no exact match, attempt topic-only match. If that fails, use default generalist model. The selected model must have a context_window large enough to hold the assembled prompt.

**Session stickiness:** A loaded model is not evicted unless the classifier detects a sustained hard topic shift across 3 consecutive turns. A single off-topic question does not trigger eviction.

**Auto-population:** At startup or on model pull, a discovery script queries Hugging Face model card tags and maps them to registry topics/intents. Models without clear tags are analyzed by the 1.5B background model. All auto-discovered entries are marked confirmed=false and require user confirmation before first routing.

**Auto-discovery at startup:** When Ollama pulls a new model or the system restarts, a discovery script queries the Hugging Face model card API for the model's `tags` and `pipeline_tag` fields. Tags such as `code`, `python`, and `debugging` map to `Software_&_Tech` / `Troubleshooting`. Tags such as `roleplay` and `creative-writing` map to `Creative_&_Media` / `Generation`. For models with no clear public tags, the 1.5B background model reads the model's README and generates suggested topic/intent mappings via a structured prompt. All auto-discovered entries are written to the registry with `confirmed: false`. Before the first routing decision that would select an unconfirmed model, the SSE telemetry stream surfaces a confirmation request: the user is shown the proposed tags and must confirm or edit them before the entry is committed as active. Auto-confirmed entries that are never confirmed remain in the registry but are skipped by the selection logic until confirmed.

#### 9.3 Hybrid Retrieval Orchestrator

The Hybrid Retrieval Orchestrator is a dedicated pipeline that executes multi-source retrieval, fuses results, applies session diversification, and returns a ranked, deduplicated context payload. It is not scattered across the middleware; it is a named subsystem with defined inputs and outputs.

**Inputs:** ClassificationResult (topic_tags, intent_tags, context_reliance), conversation_id, cluster_scope (from session metadata), prompt embedding, optionally HyDE-rewritten query embedding.

**Retrieval legs (executed in parallel where safe):**

**BM25 Retrieval:** Keyword-based full-text search over raw_text (or summary_text for lossless=false turns) in episodic_memory. BM25 handles exact matches — specific variable names, character names, error codes — that vector similarity misses due to embedding space broadness. PostgreSQL's full-text search (tsvector/tsquery) is used as the BM25 implementation.

**Vector Similarity:** pgvector cosine similarity against the stored embeddings in episodic_memory, using the prompt or HyDE-rewritten query embedding. Filtered by topic_tags and cluster_id before vector comparison.

**Graph Traversal:** NER pass on the prompt, followed by entity lookup in codex_entities (canonical name and alias), followed by 1–2 hop graph traversal in codex_edges (WHERE valid_until IS NULL), collecting entity context_payloads.

**Procedural Lookup:** Embedding similarity against procedural_memory entries, filtered by topic_tags and trigger_conditions match. Activated selectively based on intent (Strategic_Planning, Generation, Open_Exploration).

**RAG Lookup:** Vector similarity against rag_chunks, activated only when classifier detects reference language (as defined in §3.4).

**Fusion:** All retrieval legs produce ranked result lists. These lists are merged using Reciprocal Rank Fusion (RRF): each result's fused score is the sum of 1/(rank_in_list + k) across all lists where it appears (k=60 is the standard RRF constant). This gives higher scores to results that appear consistently across multiple retrieval legs, rather than ranking highly in only one.

**Session Diversification:** After RRF fusion, a diversification filter is applied. No single conversation_id (session) may contribute more than N results to the final payload (default: N=3). This prevents one dominant historical conversation from monopolizing the context window at the expense of more diverse, potentially more relevant results from other sessions.

**Deduplication:** Results with identical content hashes are collapsed to a single entry (the highest-fused-score instance is retained).

**Output:** A ranked list of context fragments, each tagged with source type (episodic | codex | procedural | rag), source batch ID, confidence score, and token count. The orchestrator respects the assembled token budget: lower-ranked results are dropped when the budget is exhausted.

#### 9.4 HyDE Query Rewriting

When context_reliance=Long_Term_Memory and the prompt's entropy_score is below threshold (indicating vague or anaphoric language), the retrieval pipeline applies HyDE rewriting before executing vector search legs.

The 1.5B background model receives the raw prompt plus the last 5 turns and is instructed to rewrite it into a highly specific, self-contained search query. Only the search uses the rewritten query; the original prompt is used for LLM generation. The SSE stream surfaces this step. Latency cost: 200–400ms.

Bypass condition: if the prompt already has high-specificity technical language (entropy_score above threshold), HyDE is skipped.

#### 9.5 Dual-Agent Protocol

When the pre-flight classifier detects simultaneous high-confidence scores on a technical intent and Emotional_Processing, a multi-model execution path is triggered.

**Conflict score:** absolute difference between top two intent probabilities.

| Conflict Score | Strategy |
|---|---|
| < 0.15 (both nearly equal) | Full dual-agent: coding model generates response (captured, not shown); empathetic model generates final response incorporating the technical solution |
| 0.15 – 0.30 | Prompt chaining: primary model generates response; 1.5B worker rewrites only the opening and closing sentences for tone |
| ≥ 0.30 (one dominant) | Single model; secondary intent injected as tone directive in system prompt |

VRAM conflict: if both models cannot coexist in VRAM, execution is serialized: load coder → generate → unload → load empathetic → generate. SSE stream surfaces both stages.

#### 9.6 Graceful Degradation

Every external dependency has an explicit fallback chain. The system must never surface a raw stack trace to the user.

**Pre-flight classifier failure:** Fall back to wide-net retrieval with generalist model. Log and alert via Sentinel.

**pgvector query failure:** Serve Zero_Shot response. Surface structured error message in SSE stream explaining the retrieval failure.

**Redis / Celery unavailable:** API continues serving. Post-flight events are buffered to a local JSONL file. The buffer is replayed when Redis recovers. In-flight requests are unaffected.

**Codex Extractor crash:** Episodic memory continues storing turns. Codex state is frozen at last successful extraction. Worker is retried with Celery exponential backoff. A Sentinel rule fires to notify the user of the degraded state.

**Ollama primary model timeout (>3 seconds):** Retry once. If still failing, traverse the registry fallback chain: next-best topic match → generalist model → graceful structured error message.

**HyDE rewrite timeout:** Bypass HyDE; use raw prompt embedding. Log the timeout.

**Dual-agent VRAM conflict:** Serialize execution (above). User informed via SSE.

**Compaction Worker crash mid-run:** Transaction rollback guarantees no partial snapshot is written. The next run reprocesses from the last valid snapshot.

**All degraded states are surfaced in the SSE telemetry stream** as distinct event types so that the user and any monitoring system can observe exactly which subsystem degraded and what fallback was applied.

---

### 9.01. Simulation Harness

The Simulation Harness is the primary evaluation infrastructure for Paper 1 and Paper 2. It replays historical conversation data through the full ICE pipeline in accelerated time, accumulating a realistic memory state that would otherwise require months of live usage to produce.

#### 9.01.1 Purpose

The core evaluation challenge for a longitudinal memory system is that real quality is only observable after many sessions — a single-session retrieval test does not measure whether the system gets better over time, whether decay and reinforcement work correctly, or whether the Codex graph accumulates accurate knowledge. The Simulation Harness compresses this timeline: it takes stored `(prompt, response)` pairs from historical chat logs and replays them in chronological order with artificial but correctly-spaced timestamps, producing a fully populated memory state in hours rather than months.

#### 9.01.2 Operation

**Input:** A JSONL file of `(prompt, response, original_timestamp)` tuples, sourced from historical chat exports processed through the Drop Zone extraction pipeline.

**Execution:** For each tuple in chronological order:
1. Assign a synthetic timestamp that preserves the original inter-turn spacing (scaled to simulation speed).
2. Run the full pre-flight classification pipeline on the prompt.
3. Insert the turn into `episodic_memory` with the synthetic timestamp.
4. Trigger the post-flight evaluation pipeline (synchronously in simulation mode, not deferred).
5. Trigger the Codex Extractor and Procedural Extractor pipelines.
6. Optionally log retrieval accuracy: before inserting turn N, run the retrieval pipeline as if turn N were a live prompt, record which turns and entities were retrieved, and evaluate relevance against a ground-truth label set.

**Output:** A fully populated ICE database (episodic turns, Codex graph, procedural patterns, cluster assignments) representing the accumulated memory state after the simulated period. This state is used as the baseline for retrieval quality experiments.

#### 9.01.3 Reproducibility Requirements

Every simulation run requires a `--seed` argument. The seed controls: PyTorch random state, SentenceTransformer embedding randomness (if any), 1.5B model sampling temperature (set to 0.0 for extraction tasks), timestamp scaling jitter, and shuffle order for any batched operations. Runs without an explicit seed are invalid for paper methodology and will be rejected by the harness CLI with an error.

Every run logs to a `simulation_runs` table: seed value, input file path, start and end timestamps, turn count processed, final Codex entity and edge counts, and a run_id UUID. Experiments reference this run_id to link results to a specific simulation state.

#### 9.01.4 Evaluation Metrics

**Retrieval precision@k:** For a set of held-out prompts with ground-truth relevant turns labeled, measure what fraction of the top-k retrieved results are relevant. Compare across: BM25-only, vector-only, graph-only, and full RRF fusion — this is the core Paper 1 result.

**Codex accuracy:** After simulation, sample Codex entity-relation-entity triplets and evaluate them for factual correctness against the source conversations. Measure precision (fraction of extracted triplets that are correct) and recall (fraction of ground-truth facts that were extracted).

**Longitudinal improvement:** Run retrieval precision@k measurements at simulation steps representing 1 session, 10 sessions, 30 sessions, and 60 sessions. If the system is working correctly, precision should improve over time as the Codex and procedural stores accumulate. This is the core Paper 2 result.

**Decay/reinforcement validation:** Verify that turns with high `access_count` have higher `decay_score` than equally-aged turns with low `access_count`. Verify that bookmarked turns never fall below `decay_immune` threshold.

---

### 10. KV Cache Optimization

KV cache management is a first-class architectural concern, not an implementation detail. Incorrect prompt ordering causes unnecessary cache invalidation that adds measurable latency on every turn.1
#### 10.1 Stable-Prefix Ordering

The prompt payload is assembled in a fixed structural order designed to maximize prefix cache hits:

```
[1] SYSTEM RULES          — rarely changes; highest cache reuse
[2] PERSISTENT CONTEXT    — changes only when memory slots are updated; high reuse
[3] CODEX: FACTS          — changes on topic shift; medium reuse
[4] EPISODIC CONTEXT      — changes on new retrieval; lower reuse
[5] PROCEDURAL CONTEXT    — changes on intent shift; lower reuse
[6] RAG CHUNKS            — changes on document reference; occasional
[7] USER INPUT            — changes every turn; no reuse
```

The inference engine (Ollama/llama.cpp) tokenizes from the top and compares against the KV cache from the previous turn. Tokens that match the previous payload exactly are served from cache without recomputation. Only divergent tokens — typically the tail of EPISODIC CONTEXT onward — are recomputed.

#### 10.2 Cache Invalidation Strategy

When the classifier detects a hard topic shift (e.g., from Software_&_Tech to Creative_&_Media), the Codex content in block 3 changes substantially. At the exact character position where the new payload diverges from the previous, a cache miss occurs. The engine discards the stale portion and recomputes from that point. This intentional cache break serves as a cognitive reset and is architecturally correct behavior, not a failure.

#### 10.3 TurboQuant

Google's TurboQuant algorithm (released March 2026, polar coordinate quantization) operates at the inference engine level and reduces KV cache VRAM footprint by approximately 6× with negligible accuracy loss. ICE does not directly invoke TurboQuant; it structures payloads for optimal prefix matching and relies on the Ollama/llama.cpp backend to apply compression. The effect is that a 24GB VRAM GPU can maintain context windows that would otherwise require approximately 40GB, making the 10-turn sliding window viable on consumer hardware.

#### 10.4 Sliding Window FIFO

When turn N+1 is assembled, turn 1 is dropped from the prompt payload. Its content remains in episodic_memory and is retrievable by semantic search. The sliding window applies only to the active context payload, not to stored history. Default window size: 10 turns. Bounded by the selected model's context_window from the registry.

---

### 11. Storage Design

#### 11.1 Unified PostgreSQL Store

All persistent state lives in a single PostgreSQL instance with the pgvector extension. A multi-database design (e.g., SQLite + ChromaDB + JSON files) is explicitly rejected. The unified store enables atomic cross-table queries, simplifies backup, and avoids consistency hazards under concurrent writes.

**pgvector filtered queries:** The classifier always narrows the search space by tag before executing vector similarity. A query like WHERE topic_tags @> ARRAY['Software_&_Tech'] AND cluster_id = $1 ORDER BY embedding <=> $2 is dramatically faster than querying a standalone vector database and merging with a relational filter in application code.

**Schema management:** Alembic manages all schema migrations. No create_all() calls in production code. Every schema change is a versioned migration script.

#### 11.2 Event Sourcing (Codex Mutations)

All Codex mutations are recorded as append-only events in codex_events before being applied to the materialized entity and edge tables.

**Event schema:** id, entity_id, event_type (edge_added | edge_expired | property_updated | context_appended | snapshot), payload (JSONB), timestamp, batch_source, compacted (boolean).

**Current state derivation:** Load the most recent snapshot from codex_snapshots for the entity, then replay all codex_events where id > snapshot.last_event_id AND compacted=false.

**Rollback:** Revert entity state to any prior point by replaying events up to that timestamp. No database restore required.

#### 11.3 Event Log Compaction — Detailed Mechanics

Without compaction, the event sourcing model degrades over time. Replaying 5,000 events to compute current entity state is unacceptable at retrieval time.

**Compaction trigger:** After the Codex Extractor processes a batch, it checks the uncompacted event count for each affected entity. If the count exceeds the threshold (default: 100 events), the Compaction Worker is enqueued.

**Compaction process (atomic transaction):**
1. Load all events for the entity, sorted chronologically.
2. Apply each event in sequence to an in-memory state representation, tracking: the set of active edges, the current context_payload, the current properties JSONB, and the current aliases array.
3. Write a single snapshot row to codex_snapshots: entity_id, snapshot_ts (now), last_event_id (the highest event ID processed), full_state (JSONB of the computed state).
4. Set compacted=true on all events with id ≤ last_event_id.
5. Commit both the snapshot write and the compaction marks in a single transaction.

**Entity state reconstruction at retrieval time:**
1. Query codex_snapshots for the most recent snapshot where entity_id = target.
2. If a snapshot exists, deserialize full_state into memory. Then query codex_events for any events where entity_id = target AND id > snapshot.last_event_id AND compacted=false, ordered chronologically. Apply these events to the snapshot state.
3. If no snapshot exists, replay all uncompacted events from scratch.

**What compaction does not do:** It does not delete events. It sets compacted=true, which excludes them from standard reconstruction queries. The complete event history remains for auditability, research measurement, and explicit historical queries.

---

### 12. Background Worker Subsystem

All background work executes in Celery workers consuming tasks from Redis. All workers respect INV-5 (yield to active inference).

#### 12.1 Post-Flight Evaluator

**Trigger:** CHAT_COMPLETED Redis event (emitted after each response).

**Work:** Set lossless_flag (examine response for code blocks, structured data, entity density, information score). Generate summary_text if lossless=false. Correct pre-flight tag errors. Emit BATCH_PROCESSED event.

#### 12.2 Codex Extractor

**Trigger:** BATCH_PROCESSED event.

**Work:** Extract subject-relation-object triplets using 1.5B model. Resolve entity IDs. Check for contradictions (set valid_until on old edge). Check for corroboration (increment strength, promote pending edges). Write all changes as codex_events in a transaction. Emit ENTITY_UPDATED events.

#### 12.3 Procedural Extractor

**Trigger:** BATCH_PROCESSED event (runs after Codex Extractor).

**Work:** Prompt 1.5B model to identify recurring behavioral patterns in the exchange. Match against existing procedural_memory entries by embedding similarity. Increment reinforcement_count on matches. Insert new entries with reinforcement_count=1 and confidence=pending for novel patterns. Promote pending patterns to active when reinforcement_count reaches the threshold.

#### 12.4 Reflection Worker

**Trigger:** CHAT_SESSION_ENDED event, or scheduled periodic task, or Sentinel action.

**Work:** Session synthesis, pattern crystallization, memory slot evolution proposals, Codex enrichment, motif detection. See §6 for full detail.

#### 12.5 Compaction Worker

**Trigger:** ENTITY_UPDATED event; also runs on configurable schedule.

**Work:** Execute compaction protocol (§10.3) for entities whose uncompacted event count exceeds threshold.

#### 12.6 Clustering Worker

**Trigger:** Periodic schedule, or Sentinel action, or manual trigger.

**Work:** Scan unassigned episodic turns. Group into named clusters using the 1.5B model. Update context_clusters table and episodic_memory.cluster_id. Merge overlapping clusters. Split overly broad clusters.

#### 12.7 Decay Worker

**Trigger:** Periodic schedule (default: daily).

**Work:** Apply decay function to non-bookmarked, non-lossless turns older than minimum age. Restore decay_score partially for recently accessed turns. Flag sub-threshold turns as is_archived=true. Migrate sub-archival-threshold archived turns to cold_storage.

#### 12.8 Sentinel Monitor

**Trigger:** Periodic schedule (default: every 30 minutes during active use, every 6 hours during idle).

**Work:** Load all active sentinel_rules. Evaluate each rule's trigger_conditions against current system state. Fire action for any rule that triggers and has exceeded its cooldown. Record each firing in sentinel_events.

#### 12.9 Background Model (The Silent Janitor)

All background workers above use the same 1.5B model (Qwen2.5-1.5B, quantized Q8_0, approximately 1.6GB VRAM) for all NLP tasks: summarization, triplet extraction, procedural pattern detection, session synthesis, HyDE rewriting, and registry inference. It is never user-facing. The 1.5B size is sufficient for all these bounded tasks; loading a larger model wastes VRAM without measurable quality improvement for these specific operations.

---

### 13. Ingestion Pipeline

#### 13.1 Extraction Engine

The Extraction Engine recovers human-authored prompts from unstructured raw chat exports using an overlapping-chunk strategy with the Amnesia Method.

The file is read in 3,000-character chunks with 500-character overlap between consecutive chunks. Each chunk is processed as a completely independent, stateless model call. The context window never grows beyond 3,000 characters; VRAM usage is constant regardless of file size. After all chunks are processed, results are deduplicated by content hash.

#### 13.2 Simulation Harness

The Simulation Harness replays captured historical conversations chronologically into a fresh system instance, simulating months of usage in hours. It is the primary evaluation tool for retrieval accuracy experiments.

**Reproducibility requirement (architectural, not optional):** Every simulation run accepts a --seed parameter that seeds all stochastic processes (PyTorch RNG, LLM temperature set to 0.0, embedding model RNG, timestamp assignment). The seed and all configuration parameters are logged at run start. Any reviewer must be able to reproduce exact results from the seed alone.

#### 13.3 Drop Zone Ingestion

Users place documents into /ingest_inbox. A file watcher triggers the ingestion pipeline: extract human-authored content, classify each turn, store in Episodic with appropriate flags, trigger Codex and Procedural Extractors, skip duplicates by content hash. Fully asynchronous.

---

### 14. Session Replay and Observability

#### 14.1 Session Replay

Every session ICE records is replayable. Sessions are stored in the `session_replays` table as ordered event sequences: prompts, SSE status events, tool calls, tool results, token streams, and background worker outcomes. Each event carries a timestamp, event_type, and payload.

**Replay capabilities:**
- Navigate a session's timeline as a sequence of discrete events
- View exactly which memory stores were queried at each turn, what was retrieved, and what was injected
- Observe the classifier's output at each turn alongside the model's response
- Identify turns where the classifier misfired (detectable by comparing pre-flight tags with post-flight corrected tags)
- Import historical conversation exports (e.g., Claude Code JSONL transcripts) for replay and retrospective analysis

**Research value:** Session replay enables memory tracing — following a specific entity or pattern from its first extraction through its evolution across multiple sessions. This is a direct tool for evaluating the quality of the Codex Extractor and Procedural Extractor over time.

#### 14.2 Audit Trail

Every write to any memory store (episodic_memory, codex_events, procedural_memory, memory_slots) carries a source annotation: user_input | post_flight | codex_extractor | procedural_extractor | reflection_worker | manual_injection | sentinel_action | bookmark. This makes the provenance of every memory entry fully traceable and supports rollback of any specific source's contributions.

---

### 15. SSE Telemetry and Observability

#### 15.1 Core Principle

Invisible infrastructure is perceived as broken infrastructure. The SSE telemetry layer exists to make ICE's internal cognition observable to the user in real time. Every major pipeline stage emits a structured SSE event that is rendered in the UI as an expandable status panel.

#### 15.2 SSE Event Taxonomy

**Routing events:**
- Stage: classifying → Stage: classified with dominant topic, confidence score, and context reliance class
- Confidence color: green (≥ 0.75), yellow (0.50–0.74), red (< 0.50)
- Dual-agent trigger visible: HIGH CONFLICT DETECTED — dual-model protocol active

**Retrieval events:**
- Stage: expanding_query with original and rewritten text (when HyDE fires)
- Stage: retrieving with list of active retrieval legs (episodic, codex, procedural, rag)
- Stage: context_ready with: tokens fetched, sources breakdown (N codex nodes, N episodic turns, N procedural patterns, N RAG chunks)

**Execution events:**
- Stage: generating with model name and context window usage
- GPU state: VRAM used / VRAM total, updated after context injection

**Degradation events:**
- Stage: degraded with subsystem name, failure reason, and fallback applied
- Sentinel alerts surfaced here when action_type = notify

**Background events (optional, for power users):**
- Post-flight completion: lossless flag decision, summary generated
- Codex update: N entities updated, N edges added
- Sentinel fires: which rule triggered, what action was taken

#### 15.3 Toggleability

A user preference setting controls visibility. Power users who want full telemetry keep the panel expanded. Users who prefer a clean chat experience collapse it permanently. The preference is stored in the user_preferences memory slot.

---

### 16. Conversation Scoping and Clustering

#### 16.1 Automatic Clustering

A background Clustering Worker periodically scans unassigned Episodic turns and groups them into named clusters using the 1.5B model. Clusters are not static; the worker merges overlapping clusters and splits overly broad ones on subsequent runs. Results are written to context_clusters and episodic_memory.cluster_id.

#### 16.2 Conversation Scope Selector

Every conversation is assigned a memory_scope_type at creation:

| Scope | Behavior |
|---|---|
| None (default) | No existing memory fetched; new turns private to this conversation |
| Auto | Classifier-driven retrieval across all accessible clusters |
| Project: X | Hard WHERE cluster_id IN (...) on all Episodic queries |
| Manual | User-supplied filter expression (topic, date range, etc.) |

Scope is stored in conversations metadata. Scope changes mid-conversation are logged as system events.

**Isolation under None scope:** New turns are stored with a conversation_id that marks them private. They are invisible to all other conversations. Promotion to a shared cluster requires explicit user action.

**Codex traversal under scoped retrieval:** The Codex graph is not scoped at the storage level. Under scoped retrieval, graph traversal only starts from entities that appear in Episodic turns belonging to the selected clusters. This prevents cross-project entity bleed without requiring per-conversation Codex copies (INV-8).

---

### 17. Human-Guided Reinforcement

ICE's memory is collaborative, not fully autonomous. The user has explicit mechanisms to guide, correct, correct, and override every layer of the memory system.

**Bookmarking** (§7): user signals high-value turns; immediate lossless, decay immunity, priority extraction.

**Memory slot editing:** user directly edits persona, preferences, project_context, guidance, and pending_items at any time through the admin UI or API. User writes take immediate effect.

**Manual Codex injection:** user writes YAML entity files to /codex_inject; watcher applies them as the highest-authority write path, bypassing LLM extraction entirely.

**Manual label correction:** user corrects classifier mistakes via the /batch/override-tags endpoint; corrections feed the fine-tuning loop.

**Memory scope selection:** user selects which clusters a conversation can access; classifier is overridden by this selection (INV-8).

**Sentinel review queue:** user reviews and resolves items surfaced by the Sentinel System; decisions are logged for research measurement.

**Memory slot update confirmation:** Reflection Worker proposals for slot updates require user confirmation before write. The user is the final authority on what constitutes a stable preference.

**Explicit cluster creation:** user can manually name and define a cluster, overriding the automatic clustering worker's assignments.

The design principle: the classifier and background workers make intelligent probabilistic decisions. The user makes deterministic final decisions. ICE never silently overrides a user's explicit choice.

---

### 18. Retrieval Flow (End-to-End)

```
User prompt received
    |
    v
Memory slots assembled (unconditional; INV-11)
    |
    v
Pre-flight classifier → ClassificationResult
    |
    |-- context_reliance = Zero_Shot → skip retrieval, proceed to model selection
    |-- context_reliance = Real_Time_Search → web search pipeline, skip memory stores
    |-- context_reliance = Long_Term_Memory → enter Hybrid Retrieval Orchestrator
           |
           v
    Confidence check → wide-net fallback if max_prob < threshold
           |
           v
    HyDE query rewriting (if entropy_score < threshold)
           |
           v
    Apply conversation scope filter (cluster_ids from session metadata)
           |
           v
    Hybrid Retrieval Orchestrator:
      - BM25 retrieval (episodic full-text)
      - Vector similarity (episodic pgvector, scope-filtered)
      - Graph traversal (codex, NER-seeded, valid_until IS NULL)
      - Procedural lookup (topic + trigger match)
      - RAG lookup (reference language detected)
           |
           v
    RRF fusion → unified ranked result list
           |
           v
    Session diversification filter (max N results per conversation_id)
           |
           v
    Deduplication by content hash
           |
           v
    Token budget enforcement (trim lower-ranked results)
           |
           v
    Assemble prompt:
      SYSTEM + SLOTS + CODEX + EPISODIC + PROCEDURAL + RAG + USER
           |
           v
    Check assembled token count vs model context_window
    (trim EPISODIC and PROCEDURAL first; never trim CODEX or SYSTEM or SLOTS)
           |
           v
    Route to selected model
```

---

### 19. Memory Lifecycle (Full View)

```
Turn Created (raw_text stored; lossless_flag NULL; decay_score=1.0)
    |
    v [Post-Flight Evaluator, minutes after]
Evaluated:
  lossless=true  → raw_text injected at retrieval; no summary; decay slower
  lossless=false → summary_text generated; summary injected; standard decay
    |
    v [Codex Extractor]
Entities extracted → codex_events appended; edges created as pending
    |
    v [Procedural Extractor]
Patterns extracted → procedural_memory updated; reinforcement_count incremented
    |
    v [Clustering Worker, periodic]
cluster_id assigned → turn is now scopeable
    |
    v [Reflection Worker, post-session]
Session synthesis → session_summaries; pending_items updated; enrichment proposals
    |
    |-- [Bookmark] → FROZEN:
    |     lossless forced true
    |     decay_immune = true
    |     is_archived = false (permanently)
    |     Bookmark Codex node created
    |     Priority indexing in hybrid retrieval
    |
    v [Decay Worker, daily]
decay_score decreasing over time
    |
    |-- [Retrieval access] → decay_score partially restored
    |
    v [decay_score < archival_threshold]
is_archived = true → excluded from default retrieval; raw_text preserved
    |
    v [decay_score < cold_threshold]
Moved to cold_storage → queryable only on explicit request or session replay
```

---

### 20. Operational Model

**Infrastructure stack:** PostgreSQL + pgvector (single instance), Redis (message broker + Celery backend), Celery (worker pool), FastAPI (middleware process), Ollama (inference daemon, host-native or Docker with GPU passthrough).

**Containerization:** All components except Ollama run in Docker Compose. This decouples the application from the host OS's rolling-release update cycle.

**Worker concurrency:** A single Celery worker is sufficient for single-user deployment. The worker pool is designed to scale horizontally if multi-user deployment is needed, but this is outside the current scope.

**GPU utilization policy:** All workers poll GPU utilization before acquiring tasks (INV-5). The threshold is configurable. Workers that acquire tasks during low-utilization windows and then observe utilization rise may either complete their current task and pause further acquisition, or checkpoint and yield, depending on task type. Long-running Reflection passes checkpoint at defined boundaries.

---

### 21. Deferred Systems

These systems are architecturally planned. Schemas are designed to accommodate them at zero cost. Implementation is explicitly deferred.

#### 21.1 Conversation Branching (Thread Lineage)

A non-linear conversation model where each turn carries a parent_message_id foreign key, allowing the turn graph to be a tree. When a user edits and resubmits a prior turn, a new branch is created without overwriting the original. The parent_message_id column is included in the initial schema (defaulting to NULL for linear conversations). The retrieval switch from ORDER BY timestamp to parent-pointer traversal is straightforward. What is deferred is the frontend requirement: a UI that exposes branch points and supports navigation between timelines.

#### 21.2 Local-to-Cloud Hybrid Routing

A Cloud_Heavy context reliance class that routes the compressed context payload to a cloud inference endpoint when local capacity is insufficient. Deferred pending authenticated cloud endpoint configuration and cost tracking. INV-10 governs this: cloud routing is a failover path, not a primary path.

#### 21.3 Multi-Agent Shared Memory

Extension of the conversation scoping model to multiple concurrent AI agent instances sharing the same memory server, with lease-based write coordination to prevent Codex mutation conflicts. Deferred; the current design is single-user.

---

### 22. Research Hooks

ICE is designed to produce three research contributions in sequence. Each paper depends on the system being further developed than the previous. Paper 1 is producible once the classifier and simulation harness are operational. Paper 2 requires several months of accumulated memory state and a working full pipeline. Paper 3 requires the agentic extension of ICE's architecture.

All experiments run through the Simulation Harness with a fixed seed. All claims are quantitative. No paper is submitted without reproducible numbers from the harness.

---

#### Paper 1 — Intent-Driven Context Compression and Retrieval Quality

**Venue target:** ACL, EMNLP, or NAACL (NLP systems track). Alternatively a top ML systems workshop.

**Primary claim:** Classifier-gated hybrid retrieval (BM25 + vector + graph, gated by a 25-label intent classifier) reduces token consumption per request by X% versus naive context injection while maintaining retrieval precision at or above Y%, measured over a 500-prompt held-out evaluation set.

**Secondary claim:** HyDE query rewriting on vague/anaphoric prompts improves retrieval precision by Z% versus embedding the raw prompt directly, with no measurable degradation on high-specificity prompts where HyDE is bypassed.

**Measurement points:**
- `tokens_fetched` per request (logged on every retrieval call)
- `retrieval_precision@k` (k=5) against a labeled held-out set
- `classifier_confidence` distribution across the 500-prompt test set
- Token savings breakdown by Context Reliance class (Zero_Shot saves the most; Long_Term_Memory with classifier gating saves versus wide-net-always)

**Experimental design:**
- Condition A: Full ICE with classifier gating (production configuration)
- Condition B: ICE with classifier replaced by wide-net-always (retrieval runs unconditionally on every prompt)
- Condition C: ICE with classifier gating but without HyDE rewriting (isolates HyDE contribution)
- Condition D: Baseline — raw prompt passed to LLM with no retrieval (measures ceiling token cost)

All four conditions run on the same held-out prompt set against the same pre-populated simulation database. The classifier interface abstraction (§1.1) makes swapping Condition B in with a one-line config change.

**HyDE sub-experiment:** 500 prompts selected specifically for high anaphor density ("fix the thing from yesterday", "continue where we left off", "the approach we discussed"). Retrieval precision compared with and without HyDE rewriting. Latency cost of HyDE (200–400ms) is reported alongside precision gain to allow cost-benefit analysis.

**Novelty position:** Prior work on RAG retrieval quality (REALM, RAG, FiD) operates on document retrieval for factual QA. ICE's retrieval is over personal conversational history — a fundamentally different distribution. The classifier is trained on conversational intent, not document relevance. This is a different problem with a different solution and a different evaluation methodology. The paper argues that conversational retrieval requires intent-awareness that document retrieval does not.

---

#### Paper 2 — Longitudinal Memory Health and User-Scoped Retrieval in Personal AI Systems

**Venue target:** CHI or CSCW (human-computer interaction with memory systems angle), or ACL (long-horizon NLP systems).

**Primary claim:** User-scoped retrieval (Project mode) achieves higher retrieval precision than automatic retrieval (Auto mode) on domain-specific prompts, with the hybrid model (Auto with user override via scope selector) achieving precision within 5% of manual scoping at substantially lower user effort. Measured over a simulated 7-month conversation history.

**Secondary claim:** The truth quorum mechanism reduces Codex hallucination rate (fraction of active edges that are factually incorrect) by W% compared to single-source extraction with no validation gate.

**Tertiary claim:** Memory decay and reinforcement mechanics produce measurably better retrieval quality at session 50 than at session 1, demonstrating longitudinal improvement — the defining property that separates a cognition system from a retrieval system.

**Measurement points:**
- `retrieval_precision@5` across three scope modes (None / Auto / Project) on domain-specific vs. cross-domain prompt sets
- Entity persistence rate: fraction of Codex entities created in month 1 that are still active (not decayed to pending) at month 7
- Codex edge accuracy: manual evaluation of sampled triplets against source conversations (precision and recall)
- Hallucination rate: fraction of active Codex edges not corroborated by source content, with and without truth quorum
- Longitudinal precision curve: `retrieval_precision@5` plotted at simulation steps representing sessions 1, 10, 30, 60, 120

**Experimental design:**
- Simulation Harness replays 7 months of chronological conversation data (personal + supplementary synthetic sessions)
- Retrieval quality measured at each checkpoint using the same 200-prompt evaluation set
- Truth quorum ablation: Condition A (quorum enabled, default) vs. Condition B (all extracted edges immediately promoted to active regardless of source count)
- Scope mode comparison: same prompts evaluated against each of the three scope configurations

**Sentinel sub-section:** Sentinel firing event logs from `sentinel_events` table provide quantitative data on memory health degradation patterns — how often entity staleness, contradiction accumulation, and retrieval health degradation occur naturally in long-running personal memory systems. This data is a secondary contribution: the first characterization of memory health failure modes in personal AI systems at this timescale.

**Novelty position:** No prior work characterizes personal AI memory quality longitudinally. MemGPT, mem0, and agentmemory are all evaluated in single-session or very short multi-session settings. ICE's simulation harness enables evaluation at 7-month scale for the first time. The paper's methodological contribution — using a simulation harness to compress real usage patterns into reproducible experiments — is itself novel.

---

#### Paper 3 — Structured State Memory for Autonomous Agents: From Conversational Cognition to Agentic Continuity

**Venue target:** NeurIPS (ML systems), ICLR, or ICML. Higher bar, higher impact.

**Contingency:** Requires ICE V1 to be complete and a working agentic extension (ICE for agents) to be built and evaluated.

**Core argument:** Current agentic memory systems (mem0, agentmemory, MemGPT) treat agent memory as a retrieval problem — what did the agent observe before, and how do we surface it. This framing is insufficient. It addresses episodic history but ignores three other memory requirements that are critical for long-running autonomous agents: (1) a structured world model that the agent maintains and updates as it learns new facts — equivalent to ICE's Codex, (2) procedural knowledge of what approaches work and what approaches fail — equivalent to ICE's Procedural Memory, and (3) reactive monitoring of memory state consistency — equivalent to ICE's Sentinel System. The paper introduces the structured state memory framework, demonstrates its implementation via ICE's agentic extension, and evaluates it against retrieval-only baselines.

**Primary claim:** Agents equipped with structured state memory (Codex + Procedural + Episodic + Sentinel) outperform agents with retrieval-only memory on multi-run task performance, measured by: task success rate on tasks requiring knowledge from prior runs, decision consistency rate (agent does not contradict decisions made in prior runs), and hallucination rate on facts established in prior runs.

**Secondary claim:** The four-file external state structure (World Model / Task State / Decision Log / Behavioral Rules — equivalent to the four ICE planning files) provides a human-readable, inspectable, correctable interface to agent state that retrieval-only systems cannot provide. This enables human oversight of agent reasoning in a way that is impossible when agent memory is a black-box vector database.

**The branching contribution (integrated, not separate):** Autonomous agents backtrack. When an agent tries an approach, fails, and tries a different one, this is structurally identical to conversation branching: two branches diverge from a common decision point. The agent must not confuse "what I tried in the failed branch" with "what I know is true about the world." ICE's `parent_message_id` branching mechanism — extended to agent action sequences — provides exactly this separation. This makes the branching system a subsection of Paper 3's architecture, not a standalone paper, and gives it a significantly stronger motivation: not just UI polish but correctness guarantees for multi-path agent execution.

**The structured external state contribution:** The observation that the four-file planning system used to develop ICE is itself an instance of structured agent memory — and that it works demonstrably well for planning across many sessions — is a genuine empirical data point. Paper 3 formalizes this pattern: World Model (stable facts about the environment), Task State (current execution position and history), Decision Log (resolved choices and their rationale), and Behavioral Rules (constraints the agent operates under). These map exactly to `01_VISION.md`, `03_WORKFLOW.md`, `DECISIONS.md`, and `04_CONTEXT.md` in the ICE development workflow. The paper argues this structure is not specific to ICE development — it is the correct general architecture for agent state, and it should be a first-class abstraction in agentic frameworks.

**Experimental design:**
- Agent task suite: multi-run tasks where run N requires knowledge from runs N-1 and N-2 (database schema established in run 1 must be respected in run 3; decision made in run 2 must not be contradicted in run 4)
- Condition A: ICE agentic extension (full structured state memory)
- Condition B: retrieval-only memory (vector similarity over past action logs, no Codex, no procedural, no sentinel)
- Condition C: no memory (stateless baseline)
- Condition D: file-based structured state only (the four-file pattern without vector retrieval) — isolates the structured state contribution from the retrieval contribution
- Metrics: task success rate, decision consistency rate, hallucination rate on established facts, human oversight effort (time to inspect and correct agent state)

**Novelty position:** This is the first paper to apply a full cognitive memory architecture (episodic + semantic + procedural + reactive monitoring) to autonomous agents. It is also the first paper to formally characterize the structured external state pattern as a general solution to agent state management, with empirical evidence from both the ICE development workflow (the four-file system) and the agentic task suite experiments.

---

#### Research Checkpoints — Build Gates

These are hard stops in the implementation sequence. The Brain must not propose features 
beyond a checkpoint's feature set until that checkpoint's experiments have been run and 
logged. The workflow file (03_WORKFLOW.md) is the authoritative record of which 
checkpoints have been cleared.

---

**CHECKPOINT 1 — Paper 1 Gate**

Experiments must be run before any of the following are built:
- Codex (entities, edges, event log, extractor worker)
- Procedural memory
- Memory lifecycle (decay worker, archival, cold storage)
- Reflection worker
- Sentinel system
- Clustering worker
- Memory slots (beyond schema definition)
- Cross-chat scoping enforcement

Required to be complete and stable before running experiments:
- PyTorch classifier — trained, deployed, <50ms inference
- Training pipeline — 20k dataset labeled and validated
- Episodic memory — schema, inserts, vector column, BM25 + vector retrieval legs
- RRF fusion of BM25 + vector
- HyDE query rewriting
- Confidence threshold safety net (wide-net fallback)
- Simulation Harness V1 — can replay chronological data and log retrieval metrics
- FastAPI middleware — end to end, prompt in, tokens_fetched logged, response out
- 500-prompt held-out evaluation set with ground truth labels

Experiment to run: four conditions (full ICE classifier / wide-net-always / no HyDE / 
no retrieval) against the held-out set. Log retrieval_precision@5 and tokens_fetched 
per condition. Log results to simulation_runs table. Record results in 03_WORKFLOW.md 
under Research Checkpoint Status.

Checkpoint is cleared when: experiments are run, numbers are recorded in 03_WORKFLOW.md, 
and the human confirms in writing that Paper 1 experiments are complete.

---

**CHECKPOINT 2 — Paper 2 Gate**

Experiments must be run before any of the following are built:
- Agentic extension of ICE
- Custom frontend
- Conversation branching UI
- Cloud routing
- P2P mesh sync
- Multi-user namespacing

Required to be complete and stable before running experiments:
- Everything from Checkpoint 1
- Codex — full schema, extractor, truth quorum, temporal versioning, graph traversal leg
- Procedural memory — schema, extractor, reinforcement mechanics
- Memory lifecycle — decay worker, archival, cold storage, strengthening on access
- Memory slots — all seven types, injection at session start
- Clustering worker
- Cross-chat scoping — all four modes enforced at retrieval
- Bookmarking system — lossless override, decay_immune, Codex Bookmark node
- Reflection worker — session synthesis, pattern crystallization, slot evolution proposals
- Sentinel system — enough rules to populate sentinel_events meaningfully
- Simulation Harness V2 — longitudinal checkpoints at sessions 1, 10, 30, 60, 120

Experiment to run: 7-month simulation with precision logged at each checkpoint. 
Truth quorum ablation (quorum on vs off). Three scope modes compared on domain-specific 
vs cross-domain prompt sets. Sentinel_events table analyzed for memory health patterns.

Checkpoint is cleared when: experiments are run, numbers recorded in 03_WORKFLOW.md, 
human confirms Paper 2 experiments are complete.

---

**Brain Instruction on Checkpoints:**

Before proposing any feature, check 03_WORKFLOW.md Research Checkpoint Status.
If Checkpoint 1 is not cleared: do not propose any feature in Checkpoint 1's 
blocked list, even if it appears in 02_ARCHITECTURE.md as a described subsystem.
If a feature from that list seems like the logical next step, say so explicitly 
and tell the human that Checkpoint 1 experiments must be run first.
Never silently proceed past a checkpoint gate.

---

### 23. Architectural Risks and Open Questions

#### Risks

**R1 — Classifier cold-start on specialized domains.** The 20k blended dataset is biased toward English, developer prompts, and public internet queries. Classification quality on specialized personal domains (extended story lore, domain-specific academic topics) may degrade before the fine-tuning loop accumulates sufficient personal data.

**R2 — Codex extraction accuracy at 1.5B scale.** Triplet extraction is a structured output task that small models handle inconsistently. Malformed JSON, hallucinated entities, and incorrect relation types are real failure modes. The truth quorum provides a backstop but cannot catch systematically biased extractions. Extraction prompt quality is load-bearing.

**R3 — pgvector performance at scale.** Without explicit ivfflat or hnsw index configuration, pgvector's exact cosine similarity degrades at very large table sizes. No index strategy or performance migration path is currently specified.

**R4 — Post-flight queue lag under rapid conversation pace.** If the user sends turns faster than the Post-Flight Evaluator processes them, retrieval operates on turns with NULL lossless flags. The retrieval engine needs a defined behavior for this state (treat as lossless=false by default).

**R5 — GPU utilization polling accuracy.** Polling via nvidia-smi or equivalent has latency and may not accurately reflect momentary VRAM pressure during model loading/unloading. Workers could wake up during a model swap and cause OOM pressure.

**R6 — Sentinel rule evaluation cost at scale.** As the number of active sentinel rules grows, the Sentinel Monitor's per-run evaluation cost grows linearly. No indexing or caching strategy is specified for sentinel trigger_conditions evaluation.

**R7 — Procedural extractor false positive rate.** The Procedural Extractor will generate spurious pattern entries from one-off behaviors that happen to look like patterns to a 1.5B model. The reinforcement threshold (default: 3 observations) provides a backstop, but the rate of spurious pending entries accumulating in procedural_memory is uncharacterized.

#### Open Questions

**Q1 — What is the correct cluster granularity?** The automatic clustering worker groups turns using an LLM. No principled selection criterion for cluster size or splitting threshold is defined. This directly affects scoped retrieval quality.

**Q2 — How are Codex alias conflicts resolved between domains?** If the same name refers to two genuinely distinct entities (e.g., "Blade" as a story character and "Blade" as a Python library), the alias disambiguation criterion is unspecified.

**Q3 — What is the post-flight latency budget?** If the user starts a new turn before the previous turn's post-flight pass completes, the new turn's retrieval may see a stale lossless flag. Is this acceptable, or does the API need to enforce a minimum inter-turn delay?

**Q4 — HyDE and post-flight interaction.** HyDE rewrites the query using the last 5 turns as context. If those turns have not yet completed their post-flight passes, HyDE's context may be incomplete. The interaction between HyDE rewriting and post-flight processing lag is uncharacterized.

**Q5 — LLM classifier vs. PyTorch classifier: decision criterion.** The interface supports both, but no criterion is defined for when an LLM classifier outperforms the MLP sufficiently to justify the latency cost. This should be resolved by Paper 1's evaluation before a production recommendation is made.

**Q6 — Memory slot token budget conflicts.** Memory slots have per-slot token budgets and are always injected. If multiple slots are populated and the selected model has a small context window, the combined slot payload may consume a disproportionate fraction of the available budget. No resolution strategy (truncation priority, slot weight) is specified.

**Q7 — Procedural memory retrieval triggering.** The current design activates procedural retrieval on Strategic_Planning, Generation, and Open_Exploration intents. The correct set of activating intents, and whether procedural retrieval should ever be activated unconditionally (like memory slots), is underspecified.

---

## Appendix A — Core Schema Reference

**episodic_memory:** id, conversation_id (FK), cluster_id (FK nullable), parent_message_id (FK nullable, self-referencing), batch_id, timestamp, topic_tags (TEXT[]), intent_tags (TEXT[]), context_reliance (TEXT), entropy_score (FLOAT), lossless_flag (BOOLEAN nullable), raw_text (TEXT, write-once), summary_text (TEXT nullable), embedding (VECTOR(384)), decay_score (FLOAT default 1.0), access_count (INTEGER default 0), is_archived (BOOLEAN default false), is_bookmarked (BOOLEAN default false), decay_immune (BOOLEAN default false), idempotency_key (TEXT UNIQUE)

**memory_slots:** id, slot_name (TEXT, one of the seven defined types), content (TEXT), token_count (INTEGER), version (INTEGER), last_updated (TIMESTAMPTZ), updated_by (TEXT: user | reflection_worker), is_active (BOOLEAN)

**codex_entities:** id (UUIDv5), canonical_name (TEXT), aliases (TEXT[]), tags (TEXT[]), properties (JSONB), context_payload (TEXT), last_updated (TIMESTAMPTZ)

**codex_edges:** id, source_id (FK), target_id (FK), relation (TEXT), strength (FLOAT), source_batch (UUID), confidence (TEXT: pending | active), valid_from (TIMESTAMPTZ), valid_until (TIMESTAMPTZ nullable)

**codex_events:** id, entity_id (FK), event_type (TEXT), payload (JSONB), timestamp (TIMESTAMPTZ), batch_source (UUID), compacted (BOOLEAN default false)

**codex_snapshots:** id, entity_id (FK), snapshot_ts (TIMESTAMPTZ), last_event_id (UUID), full_state (JSONB)

**procedural_memory:** id, pattern_name (TEXT), pattern_description (TEXT), topic_tags (TEXT[]), trigger_conditions (JSONB), reinforcement_count (INTEGER default 1), confidence_score (FLOAT), first_observed (TIMESTAMPTZ), last_observed (TIMESTAMPTZ), is_active (BOOLEAN), source_batch_ids (UUID[]), embedding (VECTOR(384))

**rag_documents:** id, filename, file_type, uploaded_at, token_count

**rag_chunks:** id, document_id (FK), chunk_index, chunk_text, embedding (VECTOR(384))

**context_clusters:** id, name (TEXT), description (TEXT), created_at, updated_at

**sentinel_rules:** id, name, description, is_active (BOOLEAN), trigger_type (TEXT), trigger_conditions (JSONB), action_type (TEXT), action_payload (JSONB), cooldown_seconds (INTEGER), last_fired_at (TIMESTAMPTZ), created_at

**sentinel_events:** id, rule_id (FK), fired_at (TIMESTAMPTZ), trigger_state (JSONB), action_taken (TEXT)

**session_replays:** id, conversation_id (FK), event_sequence (JSONB[]), created_at

**session_summaries:** id, conversation_id (FK), session_date, topics_covered (TEXT[]), decisions_made (TEXT), unresolved_items (TEXT), entities_updated (UUID[]), patterns_observed (UUID[])

**conversations:** id, created_at, memory_scope_type (TEXT), cluster_ids (UUID[]), custom_filter (TEXT)

**idempotency_keys:** key (TEXT PK), processed_at (TIMESTAMPTZ)

**cold_storage:** id (original episodic turn id), archived_at (TIMESTAMPTZ), raw_text (TEXT), summary_text (TEXT nullable), topic_tags (TEXT[]), timestamp (TIMESTAMPTZ)

---

## Appendix B — Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Storage | PostgreSQL + pgvector | Unified relational + vector; atomic filtered vector queries; no multi-DB consistency hazards |
| Schema migrations | Alembic | Versioned scripts; safe for long-running simulation |
| Embeddings | all-MiniLM-L6-v2 (local) | Free, no API key, 384 dimensions, fast CPU inference, same model in classifier |
| Inference | Ollama / llama.cpp | GPU passthrough; GGUF quantization; KV cache control |
| Background tasks | Celery + Redis | Retry semantics; worker isolation; event-driven decoupling from API path |
| Graph visualization | NetworkX + PyVis | Self-contained HTML output; useful for Codex debugging and demos |
| Classifier interface | Python ABC | Enables PyTorch ↔ LLM classifier swap for research A/B |
| Logging | structlog | Structured JSON; correlation_id propagation |
| Containerization | Docker Compose | Decouples application from rolling-release host OS |
| Background model | Qwen2.5-1.5B Q8_0 | ~1.6GB VRAM; sufficient for all bounded background NLP tasks |

---

## Appendix C — Cognitive Systems Positioning

ICE is described throughout this document as a memory middleware system, which is accurate at the implementation level. At the architectural level, it is more precisely a **long-horizon conversational cognition system**.

The distinction matters for design decisions. A memory middleware system stores and retrieves. A long-horizon cognition system also:
- Accumulates structured behavioral knowledge over time (Procedural Memory)
- Evolves its understanding of the user's preferences through reflection (Reflection Worker)
- Detects and responds to systemic anomalies reactively (Sentinel System)
- Reinforces high-value knowledge and allows low-value knowledge to decay gracefully (Memory Lifecycle)
- Becomes more accurate, more personalized, and more contextually relevant the longer it operates

None of this requires AGI-adjacent claims. These are concrete, measurable engineering properties. The system improves because reinforcement_count increases, because decay_score separates signal from noise, because Sentinel rules fire and trigger review, and because the Reflection Worker surfaces crystallized patterns.

The framing matters because it sets the correct design constraint: **ICE must be evaluated not at session level but at longitudinal level.** A system that performs well in session 1 but does not improve by session 50 has failed as a cognition system, even if it functions correctly as a retrieval system. This constraint drives the Simulation Harness, the memory lifecycle mechanics, and the procedural and reflection subsystems.
