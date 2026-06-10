
---

## Phase A — Core Retrieval Fixes (highest impact on evaluation)

These directly affect the precision/recall numbers and must be done before any paper experiments.

| # | Feature | Architecture ref | Current state | What to build | Rough effort |
|---|---------|-----------------|---------------|---------------|-------------|
| A1 | **Decay‑score filtering in retrieval** | §4.2 | Retrieval only checks `is_archived`, not `decay_score`. | Add `AND decay_score > 0.2` (or the configured threshold) to all episodic retrieval queries. | 30 min |
| A2 | **Access‑weighted decay + retrieval strengthening** | §4.2 | Decay applies a flat 3% rate; `access_count` is never incremented. | 1) In the orchestrator, after injecting fragments, increment `access_count` for each retrieved episodic turn. 2) Modify the Decay Worker: decay rate should be a function of days since last access, not just age. 3) Add a “strengthening” step: when a turn is retrieved, its `decay_score` is increased by +0.15 (capped at 1.0). | 2 h |
| A3 | **Wide‑net fallback uses full vector search** | §9.3 / §1.2 | Fallback only returns last 20 turns + global Codex + RAG. | Replace the fallback with: 1) Run the full vector similarity leg (unfiltered by topic tags) over episodic memory. 2) Still include Codex and RAG. 3) Fuse with RRF. | 1 h |
| A4 | **Codex scoping to conversation/cluster** | §8.4 | Codex traversal is always global. | When a `conversation_id` or `cluster_ids` scope is active, filter Codex entities to those that appear in episodic turns belonging to the selected conversation/cluster. Requires a subquery or a join with `episodic_memory`. | 3 h |
| A5 | **Procedural memory scoping to conversation/cluster** | §8 (implied) | Procedural lookup is global. | Same as A4: filter procedural patterns to those whose `source_batch_ids` belong to the scoped conversation/cluster. | 1 h |
| A6 | **Procedural trigger conditions evaluation** | §3.3 | `trigger_conditions` JSONB is never evaluated. | Before injecting a procedural pattern, check if the current prompt’s topic/intent tags satisfy the pattern’s `trigger_conditions`. Only inject if they match. | 1 h |
| A7 | **HyDE query rewriting** | §9.4 | Stub only. | 1) In the orchestrator, when `context_reliance == Long_Term_Memory` and `entropy_score` is below threshold, call the background model to rewrite the prompt into a dense search query. 2) Use the rewritten query for BM25 and vector legs. 3) Add a bypass flag for experiments. | 3 h |
| A8 | **Sliding window – always inject last 10 turns of current conversation** | §3.1 / §10.4 | Not implemented. | In the proxy, after retrieval, fetch the last 10 episodic turns for the current `conversation_id` and prepend them to the assembled prompt as a separate `[RECENT CONTEXT]` block (before the retrieved blocks). | 1 h |
| A9 | **Session diversification before dedup is fine – but missing bookmarked‑boost** | §7.2 | Bookmarked turns are not boosted. | After the bookmarking backend is built (Phase D), multiply the score of bookmarked fragments by 1.5× before RRF fusion. | 30 min |
| A10 | **Classifier fine‑tuning loop** | §1.4 | Not implemented. | 1) Create a Celery beat task that loads `curated_labels`, freezes the SentenceTransformer, retrains the MLP head for 5 epochs, and saves a new checkpoint. 2) Expose an endpoint to trigger it manually. | 2 h |

---

## Phase B — Memory Lifecycle & Cognition Completion

These turn ICE into a true long‑horizon cognition system (G9) and provide data for the paper’s longitudinal claims.

| # | Feature | Architecture ref | Current state | What to build | Rough effort |
|---|---------|-----------------|---------------|---------------|-------------|
| B1 | **Retrieval strengthening (part of A2)** | §4.2 | Already covered above. | — | — |
| B2 | **Codex edge decay** | §4.4 | Not implemented. | 1) Add a periodic task that decays `strength` for edges not referenced in recent retrieval. 2) When strength falls below threshold, demote to `pending`. | 2 h |
| B3 | **Procedural pattern decay** | §3.3 | Not implemented. | 1) Add a periodic task that marks patterns as inactive if not observed in 6 months and reinforcement_count is low. | 1 h |
| B4 | **Cold storage periodic migration** | §4.3 | Exists but only manually triggered. | 1) Ensure the Decay Worker moves sub‑cold‑threshold archived turns to `cold_storage` on each daily run. 2) Verify it works end‑to‑end. | 1 h |
| B5 | **Reflection Worker – full implementation** | §6.2 | Only session synthesis. | 1) Pattern crystallization: scan recent sessions, feed novel patterns to Procedural Extractor. 2) Memory slot evolution: propose updates to `project_context`, `user_preferences`, `guidance`. 3) Codex enrichment: append episodic passages to thin entities. 4) Motif detection: propose new clusters. | 8 h |
| B6 | **Sentinel Monitor – real rule evaluation** | §5 | Placeholder only. | 1) Implement evaluation for at least 3 rule types (threshold, frequency, absence). 2) Populate a few default rules (e.g., staleness, contradiction). 3) Connect actions: `log_event` (already works), `notify` (write to a notifications table), `schedule_worker` (enqueue Celery task). | 6 h |

---

## Phase C — User Guidance & Control (Human‑Guided Reinforcement)

These are required by the architecture’s design goals (G5) and provide the manual evaluation hooks for the paper.

| # | Feature | Architecture ref | Current state | What to build | Rough effort |
|---|---------|-----------------|---------------|---------------|-------------|
| C1 | **Bookmarking backend** | §7 | None. | 1) `POST /turns/{id}/bookmark` – sets `is_bookmarked=true`, `lossless_flag=true`, `decay_immune=true`, triggers priority Codex extraction. 2) `GET /bookmarks` with filter/sort. 3) When assembling the prompt, inject a `[BOOKMARKED]` block with the bookmarked turns (scoped to the conversation). | 4 h |
| C2 | **Manual Codex injection** | §3.2 | Not built. | 1) Create a `/codex_inject` directory. 2) Add a file watcher (like Drop Zone) that parses YAML/JSON entity files and writes them directly as Codex events. | 3 h |
| C3 | **Manual label correction endpoint** | §1.4 | Table exists, no endpoint. | 1) `POST /batch/override-tags` – accepts batch_id and corrected tags, writes to `curated_labels`. | 1 h |
| C4 | **Conversation scoping endpoints** | §8 | Partially done. | 1) `PUT /conversations/{id}/scope` – sets `memory_scope_type` and `cluster_ids`. 2) Ensure the orchestrator respects these fields when a request comes from that conversation. | 2 h |
| C5 | **Explicit cluster creation API** | §17 | None. | 1) `POST /clusters` – manually create a named cluster. 2) `PUT /clusters/{id}/assign` – assign turns to a cluster manually. | 2 h |
| C6 | **Memory slot update confirmation flow** | §2.4 | Reflection proposes updates without user confirmation. | 1) When Reflection proposes a slot update, write it to a `review_queue` table instead of applying immediately. 2) Add `GET /review-queue` and `POST /review-queue/{id}/approve` endpoints. | 3 h |

---

## Phase D — Orchestration Layer Completion

These make the proxy and background plane match the full request lifecycle (§9.1).

| # | Feature | Architecture ref | Current state | What to build | Rough effort |
|---|---------|-----------------|---------------|---------------|-------------|
| D1 | **CHAT_COMPLETED event emission** | §9.1 step 15 | Not done. | After the SSE stream closes, publish a `CHAT_COMPLETED` event to Redis with the idempotency key. | 30 min |
| D2 | **KV cache prefix validation / token count check** | §9.1 steps 10‑11 | Not done. | 1) After prompt assembly, count the actual tokens (using the background model’s tokenizer or a rough heuristic). 2) If the count exceeds the model’s context window, trim EPISODIC and PROCEDURAL blocks first, never CODEX/SYSTEM/SLOTS. | 2 h |
| D3 | **Graceful degradation – Redis/Celery unavailable** | §9.6 | Not done. | 1) In the proxy, if the Celery task queue is unreachable, buffer the post‑flight event to a local JSONL file. 2) A recovery script replays the buffer when Redis comes back. | 3 h |
| D4 | **Graceful degradation – Ollama timeout → registry fallback** | §9.6 | Not done. | After the Model Registry is built, if the primary model times out, route to the next‑best model from the registry. | 1 h |
| D5 | **Graceful degradation – HyDE timeout** | §9.6 | Not done. | If the HyDE rewrite request times out, skip HyDE and use the raw prompt embedding. | 30 min |
| D6 | **SSE telemetry events** | §15 | None. | 1) Define the SSE event types (`classifying`, `classified`, `expanding_query`, `retrieving`, `context_ready`, `generating`, `degraded`). 2) Emit these events interleaved with the LLM token stream. 3) The frontend can parse them to show the telemetry panel. | 6 h |

---

## Phase E — Operations & Packaging (from Post‑V1 Roadmap §24)

These make ICE deployable, shareable, and usable by others.

| # | Feature | Roadmap ref | What to build | Rough effort |
|---|---------|-------------|---------------|-------------|
| E1 | **Single‑command startup** | §24.5.1 | 1) Create an `ice start` shell script that starts PostgreSQL, Redis, vLLM‑bg, Celery worker+beat, and the FastAPI proxy, with a unified log output. 2) Optionally provide a `docker compose up` variant. | 2 h |
| E2 | **Shared background model option** | §24.5.3 | 1) Add a config flag `BACKGROUND_MODEL_MODE`. 2) In `shared` mode, background workers route their LLM calls to the same Ollama/vLLM endpoint as the proxy, with a low‑priority queue. 3) Adjust GPU checks to allow background work only when the user is idle. | 3 h |
| E3 | **Terminal frontend (TUI)** | §24.1.2 | 1) Build a simple TUI using `textual` or `rich` that provides: chat input/output, display of injected context, classifier tags, scope selector, memory slot editor, bookmark toggle. 2) This becomes the primary demo interface. | 12 h |
| E4 | **One‑click installer** | §24.5.2 | 1) Write a `setup.sh` that installs system dependencies (PostgreSQL, Redis, Python), creates the venv, runs Alembic migrations, and pulls the background model. 2) Optionally build a Docker image. | 4 h |
| E5 | **Model Registry backend** | §24.2.1 | 1) Create a JSON registry file populated at startup from Ollama’s `/api/tags`. 2) For unknown models, use the background model to suggest tags. 3) Expose a `/model-registry` endpoint. | 4 h |
| E6 | **Classifier‑driven model selection** | §24.2.2 | 1) In the proxy, after classification, score registry entries by tag overlap and select the best model that fits the context window. 2) Fall back to default generalist. | 2 h |

---

## Phase F — Remaining Missing Items (lower priority, can be post‑paper)

| # | Feature | Architecture ref | Notes |
|---|---------|-----------------|-------|
| F1 | **Drop Zone full pipeline** | §3.5 | The four‑stage pipeline is not implemented; current Drop Zone is a simple text‑to‑RAG ingester. This can be built later as it doesn’t affect evaluation. |
| F2 | **Session Replay** | §14 | The `session_replays` table is empty; no code writes to it. Needed for the custom frontend, not for the paper. |
| F3 | **Audit trail** | §14.2 | Source annotations are not recorded on writes. Important for transparency but not evaluation‑critical. |
| F4 | **Conversation branching retrieval logic** | §21.1 | Deferred until custom frontend exists. |
| F5 | **Custom web frontend** | §24.1.1 | Huge effort; out of scope for the paper. The TUI (E3) is a better V1 demo. |
| F6 | **Null_Noise / Casual_Banter special routing** | §1.2 | Minor; the classifier rarely outputs these labels with high confidence for real prompts. Can be added later. |
| F7 | **Memory slot token budget enforcement** | §2.2 | Add truncation when slots exceed 300 tokens. |
| F8 | **Simulation Harness – procedural extraction + logging** | §9.01 | Add procedural extraction to the simulation loop; log run info to a `simulation_runs` table for reproducibility. |
| F9 | **Time‑weighting in episodic retrieval** | §3.1 | The architecture specifies time‑weighted cosine similarity; currently it’s plain cosine. Adding a decay‑based weight to the vector score would improve relevance. |
| F10 | **Trigger conditions for procedural memory** | §3.3 | Already covered in A6. |
| F11 | **Conversation scoping isolation (None scope)** | §8.1 | Ensure None‑scoped conversations are invisible to all other retrieval. |
| F12 | **RAG store activation rules** | §3.4 | Already implemented correctly. |
| F13 | **Manual Codex injection watcher** | §3.2 | Covered in C2. |
| F14 | **Session replay & audit trail** | §14 | Covered in F2/F3. |

---

## Execution Order (Rough Timeline)

1. **Phase A (A1–A10) → 2‑3 days** – retrieval quality fixes; will directly raise Precision@5.
2. **Phase B (B1–B6) → 2‑3 days** – memory lifecycle; enables longitudinal claims.
3. **Phase C (C1–C6) → 2 days** – user control endpoints; needed for manual evaluation.
4. **Phase D (D1–D6) → 2 days** – proxy completeness and observability.
5. **Phase E (E1–E6) → 3‑4 days** – packaging, TUI, model registry; makes ICE demoable.
6. **Phase F → after paper submission** – remaining polish.

We can start with A1 tomorrow and work straight through. Each item is self‑contained, so you’ll see steady progress. After Phase A is done, we can re‑run the automatic evaluation and you’ll see the precision number climb. Then we’ll keep building.