# Feature inventory — everything ICE actually has

**Why this file exists.** By FINAL, nobody remembers the whole system. The
failure this prevents has already happened twice: a capability gets rebuilt
because nobody knew it was there (the coding core, built and forgotten inside
45 days), and a capability gets *assumed active* in a write-up when its setting
defaults to off. Both produce a false claim about the system.

**Built 2026-08-15** by reading `src/` end to end — three passes, one per area,
each entry carrying a verified `file:line`. **~433 features.**

## How to use it

* **The `On by default?` column is the point of the file.** A feature that is
  OFF is not a feature the system has; it is a feature the system *could* have.
  Never describe one as active without checking the setting.
* **`DEAD OR INERT` sections are the second point.** They list what is
  implemented and cannot currently fire. Read them before attributing a
  measurement to a subsystem — several of today's wrong conclusions came from
  assuming an inert path was running.
* **It rots.** Code wins over this file, exactly as with every other doc here.
  Re-derive an entry before depending on it; the `file:line` is there so that
  costs seconds.

> ⚠ This is an inventory, **not** a design reference and **not** a queue.
> [ICE_Architecture.md](ICE_Architecture.md) owns how things work,
> [ROADMAP.md](ROADMAP.md) owns what is left to do.

---

## Request path — api, retrieval, classifier, model_registry

Scope read in full: `src/api/` (main.py, config.py, prompt_assembler.py, memory_decision.py, context_ledger.py, core.py, db.py, chat_commands.py, routers/{adapter,memory_slots,user_control}.py), `src/retrieval/` (orchestrator.py, leg_weights.py, coverage.py, timescope.py, evolution.py, ner_utils.py, configurable_orchestrator.py), `src/classifier/` (classifier.py, schema.py, schemas.py, model.py, templates.py, dataset.py, promotion.py, ner_model.py), `src/model_registry/` (registry.py, runtime_probe.py).

Defaults are the code defaults in `src/api/config.py`. `.env` overrides them at runtime and is gitignored — every "Default" below is what a stranger's install gets.

---

### 1. Request lifecycle — `src/api/main.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| OpenAI-compatible chat proxy | `src/api/main.py:366` | — | `POST /v1/chat/completions` is the only chat entry point. Classifies, decides, retrieves, assembles, routes, streams, then stores the finished turn. | — | — | YES |
| Health endpoint | `src/api/main.py:82` | — | `GET /health` returns `{"status":"ok"}`. No auth, no DB touch. | — | — | YES |
| Conversation id via header | `src/api/main.py:400` | G26 | Client sends `X-ICE-Conversation-ID` to continue a conversation; absent, ICE mints a new one and returns it in the response header. | — | — | YES |
| Boot-time model load | `src/api/main.py:56`, `src/api/core.py:62` | E0/G13/C7 | One classifier+embedder per process, loaded eagerly at startup so the first request is not slow. | — | — | YES |
| Embedding-identity boot guard | `src/api/core.py:78` | G23/C17/D1 | Refuses to start if the store was written with a different embedding model or width than configured — prevents silently comparing incompatible vectors. | `embedding_model_name`, `embedding_dim` | `Qwen/Qwen3-Embedding-0.6B`, `1024` | YES |
| Runtime lease / standby | `src/api/core.py:88` | E7/D6 | If another ICE process already owns background maintenance, this one runs event jobs only and promotes itself when the owner exits. | `runtime_lease_ttl_seconds` | `180.0` | YES |
| User-activity signal | `src/api/main.py:380` | C7/D7 | Every chat request marks the user as active, which is what stops background jobs from stealing the GPU mid-conversation. | `user_active_threshold_seconds`, `job_yield_grace_seconds` | `90`, `10` | YES |
| Generation in-flight gate | `src/api/main.py:744`, `:817` | C7/D7 | While a reply is streaming, no GPU background job dispatches. Fires even if the client disconnects mid-stream. | — | — | YES |
| Mini-MoE model routing | `src/api/main.py:512` | C16/G20 | Picks the best-matching model from the registry for this turn's topic/intent — **only when the client sends `model: "ice-proxy"`**. Any other model name is passed straight through to Ollama. | — | — | NO (client must ask for `ice-proxy`) |
| Session-sticky model | `src/api/main.py:513` | C7/D9/G8 | Once routed, a conversation keeps the same model until 3 consecutive turns share no topic/intent with the previous turn. State lives on the conversation row. | — | hardcoded `3` shifts | YES (within `ice-proxy` routing) |
| Topic/intent shift counter | `src/api/main.py:478` | G8 | Compares this turn's tags against the last stored turn's; no overlap increments the shift counter, any overlap resets it. | — | — | YES |
| Low-confidence log | `src/api/main.py:499` | — | Emits `low_confidence_fallback` when the classifier's peak probability is under threshold. Log only — the retrieval consequence is the wide net (below). | `confidence_fallback_threshold` | `0.75` | YES |
| Streaming SSE relay | `src/api/main.py:739` | — | Streams the model's tokens straight through, unmodified. | — | — | YES |
| ICE telemetry SSE events | `src/api/main.py:748`–`774` | F5 | Emits `classified`, `retrieval`, `context_ready`, `generating` (and `degraded`) events before the model tokens, so a frontend can show what ICE did. | — | — | YES |
| Primary-model timeout fallback | `src/api/main.py:792` | G5 | If the primary model does not respond within 10s, emits `degraded` and re-runs on the registry fallback model with a 30s timeout. The user sees both models' output spliced. | — | hardcoded 10s / 30s | YES |
| Per-stream segment isolation | `src/api/main.py:736`, `:241` | G5 | Primary and fallback streams are kept in separate buffers so a truncated primary line cannot corrupt the fallback's first line. | — | — | YES |
| SSE salvage of truncated JSON | `src/api/main.py:112` | G5 | Recovers `delta.content` from a stream line whose JSON object never closed, instead of dropping the token silently. | — | — | YES |
| SSE damage reporting | `src/api/main.py:265` | G5 | Logs `sse_stream_damaged` at WARNING with counts every time a line was dropped or salvaged. Never first-occurrence-only. | — | — | YES |
| Multi-model turn warning | `src/api/main.py:275` | G5 | Warns when one stored turn's answer came from two models (timeout + fallback) — the turn is not clean extraction/training signal. | — | — | YES |
| Token-prediction reconciliation | `src/api/main.py:99`, `:247` | C16 | Asks the server for its own prompt token count (`stream_options.include_usage`) and logs predicted-vs-actual on every turn. Logged, never enforced. | `token_usage_reconciliation` | `True` | YES |
| `num_ctx` request | `src/api/main.py:105` | C16 | Tells Ollama exactly how large a context window this prompt needs, instead of letting the server apply its own default and truncate silently. Costs KV-cache VRAM, so it is opt-in. | `ollama_send_num_ctx`, `ollama_num_ctx_max` | `False`, `32768` | **NO** |
| Post-flight turn storage | `src/api/main.py:213`, `:821` | — | After the response finishes, embeds the prompt and writes the turn to `episodic_memory` with an idempotency key. Runs as a FastAPI BackgroundTask. | — | — | YES |
| Session ("sitting") boundary | `src/api/main.py:294` | C6 | Silence longer than the gap opens a new session id on the turn; clustering, the recent window and maintenance all key off it. | `session_gap_minutes` | `30` | YES |
| Post-flight job enqueue | `src/api/main.py:330` | C7 | Hands the finished turn to the in-process maintenance runtime (codex extraction, procedural extraction, summarisation). | — | — | YES |
| Session-gap work unit | `src/api/main.py:342` | C7 | A new sitting triggers a cluster-freshening + overdue-maintenance burst. | — | — | YES |
| Private-conversation storage flag | `src/api/main.py:422`, `:831` | G16 | A conversation scoped `none` stores every turn with `is_private=True`, which every retrieval leg excludes from global search. | — | — | YES (when scope = `none`) |

---

### 2. Memory-retrieval decision (B2) — `src/api/memory_decision.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Log-odds retrieval decision | `src/api/memory_decision.py:166` | B2 | One posterior decides whether to search long-term memory at all. Replaces the old hard overrides; prefers memory but never forces it. | `ltm_decision_threshold`, `ltm_prior_bias` | `0.5`, `0.4` | YES |
| Classifier prior | `src/api/memory_decision.py:190` | B2 | The starting point is the classifier's own `Needs_Memory` probability, tilted toward retrieving. | `ltm_prior_bias` | `0.4` | YES |
| Memory-pressure length prior | `src/api/memory_decision.py:137` | B2 | Adds push toward retrieval as conversation history accumulates beyond what the recent-turns window can show. One-sided: never penalises a short conversation. | `ltm_length_weight`, `ltm_pressure_midpoint_tokens`, `ltm_pressure_scale_tokens` | `0.8`, `2000`, `4000` | YES |
| Creative-topic bump | `src/api/memory_decision.py:204` | B2 | A `Creative_&_Media` prompt leans toward retrieving. | `ltm_bump_creative` | `0.7` | YES |
| Referential-word bump | `src/api/memory_decision.py:64`, `:206` | B2/D8 | Presence of "my/that/last/again/…" in the prompt leans toward retrieving. A word-presence test, not a density heuristic. | `ltm_bump_referential` | `0.5` | YES |
| Low-confidence safety net | `src/api/memory_decision.py:208` | B2 | When the classifier is unsure, lean toward retrieving rather than answering blind. | `ltm_bump_low_confidence`, `confidence_fallback_threshold` | `0.8`, `0.75` | YES |
| Temporal bump (detector OR label) | `src/api/memory_decision.py:239` | T2/B1-D7/E12 | A parsed time expression, or a high `Temporal_Recall` probability, is a large push toward retrieving. Measured inert (see DEAD section). | `ltm_bump_timescope`, `temporal_label_threshold` | `3.0`, `0.85` | YES (but see DEAD) |
| Coding-scope bump | `src/api/memory_decision.py:243` | E1/D11 | A conversation attached to a project almost always wants context, so it leans toward retrieving. | `ltm_bump_coding` | `0.7` | YES |
| Decision breakdown telemetry | `src/api/memory_decision.py:252` | B2/F5 | Every input to the decision is logged as `memory_decision` — p_ltm, pressure, each bump, the final probability. | — | — | YES |
| Model-derived total budget | `src/api/memory_decision.py:92` | C16 | Derives the context budget from the routed model's real window instead of a fixed number, clamped so it can never exceed the window minus room for the answer. | `context_input_fraction`, `context_budget_min/max/fallback`, `context_generation_reserve` | `0.75`, `4000`/`40000`/`23000`, `2048` | YES |

---

### 3. Context accounting & eviction (C16) — `src/api/context_ledger.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| One-unit prompt ledger | `src/api/prompt_budget.py::assemble_budgeted_prompt`, `src/api/prompt_assembler.py::assemble_prompt` | C16 | v3 reports actual structured block costs including message envelopes and evidence acknowledgement; remeasures each bounded reassembly. Unknown window warns that fit is unmeasured. | `token_count_safety_margin` | `1.20` | YES |
| Question-aware memory budget | `src/api/context_ledger.py:163`, `src/api/main.py:544` | C16 | Subtracts the user's own message from the memory allowance, so a 10,000-word paste does not get a full memory block on top of it. Memory shrinks but never below the floor. | `context_budget_floor`, `context_generation_reserve` | `1500`, `2048` | YES |
| Eviction plan | `src/api/prompt_budget.py::assemble_budgeted_prompt` | C16 | v3 removes optional session status, slots, bookmarks, summary, recent turns, then evidence. Required project constraints/question/essential system survive; final counts and usage describe survivors. | — | order is code-fixed | YES; all six blocks reachable |
| Extreme-pressure warning | `src/api/main.py::chat_completions` | C16 | v3 refuses known required-context overflow with HTTP400 context_length_exceeded before generation or graph usage credit. Answer reserve is never silently reduced. | `context_generation_reserve` | `2048` | YES |
| Serving-window truth | `src/api/main.py:536`, `src/model_registry/runtime_probe.py:122` | C16 | Budgets against what Ollama actually allocated (`/api/ps`), clamped by the model's architectural ceiling (`/api/show`), instead of the registry's claim. | `context_use_serving_window` | `True` | YES |
| Window-truth log line | `src/model_registry/runtime_probe.py:90` | C16 | Emits registry / runtime / GGUF-max / derived-budget on every request, and warns when the budget exceeds any visible ceiling. | `runtime_probe_timeout`, `runtime_probe_cache_ttl_seconds` | `2.0`, `300` | YES |

---

### 4. Prompt assembly — `src/api/prompt_assembler.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Multi-message prompt | `src/api/prompt_assembler.py:230` | — | Builds a real user/assistant message sequence rather than one fused string, with an explicit "the last message is the live question" instruction. | — | — | YES |
| Current datetime anchor | `src/api/prompt_assembler.py::assemble_prompt` | T1 | Gives UTC date and clock time; explains that source recording, learning and event time differ. | — | — | YES |
| Memory slots, three tiers | `src/api/prompt_assembler.py:280`–`296` | C9/D6 | Injects standing user memory under one header: global slots always, project slots only for the attached project, conversation slots only for this conversation. | `slot_token_cap` | `300` per slot | YES |
| Bookmark injection | `src/api/prompt_assembler.py::bookmarked_turn_texts` | C16/C1 | v3 latest five pinned turns use the same source-supported representation selector as other turn readers. Complete evidence and source time; budget removes whole blocks rather than source prefixes. | — | 5 turns | YES |
| Project session-start block | `src/services/projects.py::chat_session_start`, `::chat_constraints` | E4/D6 | v3 status/diff/tasks appear at sitting start; up to five active project constraints load separately every request and cannot be evicted. MCP keeps complete session-start default. | `session_gap_minutes` | `30` | YES (project-scoped conversations only) |
| Evolving conversation summary | `src/api/prompt_assembler.py:49`, `:306` | C4/D3a | Once the conversation outgrows the recent-turns window, its rolling summary is injected for global shape, stamped with how many turns behind it runs. | `conversation_summary_max_words` | `250` | YES (once history > window) |
| Session-bounded recent window | `src/api/prompt_assembler.py:112`, `:158` | C16/C6 | The recent-turns window is bounded by the current sitting, not a turn count, plus the last turn of the previous sitting so a coffee break does not cut the thread. | `recent_window_scope`, `recent_window_bridge_turns`, `recent_window_max_turns` | `"session"`, `1`, `40` | YES |
| Recent-window filters | `src/api/prompt_assembler.py:132` | C16 | Private turns, document sections and promoted pastes are excluded from the continuity window — matching every retrieval leg's filters. | — | — | YES |
| Per-turn window cap | `src/api/prompt_assembler.py:193` | C16 | One turn may not eat more than a third of the recent window; an over-large turn degrades to an eligible summary, then a independently source-supported abstract, rather than being cut mid-sentence. | `recent_window_max_turn_frac` | `0.35` | YES |
| Newest-first budget spend | `src/api/prompt_assembler.py:196`–`226` | C16 | The budget is spent on the newest turns; the oldest fall off. (It used to discard the newest.) | — | — | YES |
| Cluster names in the header | `src/api/prompt_assembler.py:334` | C5 | When retrieval was cluster-scoped, the retrieved-context header names the clusters. | — | — | YES |

---

### 5. Chat commands (C11) — `src/api/chat_commands.py`

All eight are always on; a message whose first line starts with `/` never reaches the classifier, retrieval or the model, and is not stored to episodic memory.

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Slash-command interception | `src/api/main.py:430`, `src/api/chat_commands.py:84` | C11/D3 | Parses the first line only; unknown `/x` returns a help hint rather than leaking into chat as a prompt. Handled commands stream a normal SSE completion. | — | — | YES |
| `/remember` | `src/api/chat_commands.py:160` | C11 | Appends text to a memory slot; `in <slot>` picks the slot, `@project`/`@conversation` picks the tier. Default slot `pending_items`, global tier. | `slot_token_cap` | `300` | YES |
| `/slots` | `src/api/chat_commands.py:183` | C11 | Shows the persistent slots for one or all tiers, with a 160-char preview each. | — | — | YES |
| `/bookmark` | `src/api/chat_commands.py:216` | C11/C1 | Bookmarks the last stored turn — makes it lossless and decay-immune. | — | — | YES |
| `/search` | `src/api/chat_commands.py:227` | C11 | Searches memory directly and prints up to 12 dated fragments. No model call. Uses the same scope resolver as chat. | — | 12 shown | YES |
| `/scope` | `src/api/chat_commands.py:248` | C11/C6 | Sets the conversation's memory scope to `auto`, `project`, `none` (incognito) or `manual`. A bare `/scope` changes only the mode. | — | — | YES |
| `/forget` | `src/api/chat_commands.py:274` | C11/D1 | Queues a forget proposal for review. Nothing is deleted until approved. | — | — | YES |
| `/delete-conversation` | `src/api/chat_commands.py:321` | C11/C10 | Prints a full deletion manifest (turns, chunks, clusters, summaries, codex edges, slots, decisions) as a dry run; `confirm` within the TTL performs it. | `chat_confirm_ttl_seconds` | `600` | YES |
| `/help` | `src/api/chat_commands.py:109` | C11 | Lists the commands. | — | — | YES |
| Command journalling | `src/api/chat_commands.py:135` | C11/F5 | One `chat_command` structlog event per handled command with its outcome (applied / rendered / refused / unknown / error). | — | — | YES |

---

### 6. REST control surface — `src/api/routers/`

Thin adapters over `src/services/`. All routes are unauthenticated and mounted unconditionally.

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Memory-slot CRUD | `src/api/routers/memory_slots.py:43`,`54`,`66`,`81` | C9/E0/G14 | List / get / update slots per tier, plus `/initialize` to create the seven default global slots. | `slot_token_cap` | `300` | YES |
| Bookmark endpoints | `src/api/routers/user_control.py:97`,`102` | C1 | Bookmark a turn; list bookmarks (optionally per conversation). | — | — | YES |
| Manual label correction | `src/api/routers/user_control.py:110` | C3 | Overrides a stored batch's topic/intent/context-reliance labels — the curated-label source the fine-tune reads. | — | — | YES |
| Conversation deletion | `src/api/routers/user_control.py:121` | C10 | Manifest-first deletion; `?dry_run=true` previews. | — | — | YES |
| Scope set/get | `src/api/routers/user_control.py:132`,`154` | C4/C6/E1 | Sets scope mode, cluster ids, project attachment, and the included/excluded conversation and cluster sets. Every id set is `None`=unchanged, `[]`=clear. | — | — | YES |
| Commit notification | `src/api/routers/user_control.py:146` | E3 | The post-commit hook's target — triggers code-graph reconciliation for a project. | `maintenance_intervals["project_poll"]` | `600`s poll fallback | YES |
| Cluster create/assign | `src/api/routers/user_control.py:163`,`167` | C5 | Explicit user-created clusters and manual turn assignment. | — | — | YES |
| Review queue | `src/api/routers/user_control.py:176`,`180` | C6/D1 | Lists pending proposals (forget, dedupe, fine-tune, tier-2 agent actions) and approves them. | `agent_max_tier2_proposals` | `5`/run | YES |
| Conversation import | `src/api/routers/user_control.py:190`,`202`,`207` | F10/F14 | Replays a server-side transcript file into memory with a decay policy; status endpoints for progress. | `import_immune_window_days`, `import_recent_days` | `14`, `30` | YES |
| Document ingest + library | `src/api/routers/user_control.py:217`–`254` | C12/D10 | Add (path or blob), list, status, per-chat enable toggle, delete. Documents become their own conversations. | `document_extraction_engine` | `"builtin"` (scanned PDFs refused, never silently ingested empty) | YES |
| Model-registry CRUD | `src/api/routers/user_control.py:269`–`285` | — | Get / refresh / update / delete registry entries (tags, priority, context_window, base_url, confirmed). | — | — | YES |
| Domain-error translation | `src/api/routers/adapter.py:15` | E0 | Maps service errors to HTTP 404/400/409 so routers stay parse→service→format. | — | — | YES |
| DB session pool | `src/api/db.py:12` | — | Sync SQLAlchemy engine, pool 50 + 20 overflow, pre-ping, 1h recycle. | `database_url` | `postgresql+psycopg://ice:ice_local_dev@localhost:5432/ice_db` | YES |

---

### 7. Retrieval legs — `src/retrieval/orchestrator.py`

Eight mechanisms report as five `source_type`s. All legs run on every retrieving turn unless the noted condition fails.

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| BM25 episodic (full-text) | `src/retrieval/orchestrator.py:782` | — | Postgres `ts_rank` full-text search (historical BM25 name): OR of full-query lexemes normalized by the same `english` configuration as stored text. Digits, Unicode and late terms retained; not true BM25 scoring. | `retrieval_bm25_candidate_limit` | `100` | YES |
| BM25 AND-fallback | `src/retrieval/orchestrator.py:857` | G36 | If the OR query fails, retries with `plainto_tsquery` (AND) before giving up. | — | — | YES |
| Vector episodic (pgvector) | `src/retrieval/orchestrator.py:891` | C8/C2 | Cosine similarity over turn embeddings, multiplied by decay and an in-score recency weight. Document turns are excluded (their chunks compete instead). | `retrieval_vector_candidate_limit` | `100` | YES |
| In-score recency weight | `src/retrieval/orchestrator.py:916` | C8 | Recency is folded into the vector score itself, because the candidate set is cut by score before post-fusion bonuses can rescue anything recent. Skipped for creative topics. | `retrieval_episodic_recency_boost`, `retrieval_episodic_recency_tau_days` | `0.25`, `30.0` | YES |
| Chunk-level vector search | `src/retrieval/orchestrator.py:986` | C2 | Searches document chunks directly; visibility (privacy, decay, archive, cluster, conversation) is enforced through the parent turn. Max 3 chunks per document. | `retrieval_chunk_candidate_limit` | `30` | YES |
| Codex graph leg | `src/retrieval/orchestrator.py:1670` | A3/A4/A7.2/A10 | Extracts entities from the prompt, matches them to graph nodes, and emits one fragment per anchor with its note, trust-gated neighbour previews and its relation facts. | `codex_max_depth`, `codex_direct_trust_floor` | `3`, `0.5` | YES |
| Procedural leg | `src/retrieval/orchestrator.py:1997` | C9/D4 | Surfaces learned user habits ("restates the plan before proceeding") ranked by embedding similarity, gated by a confidence floor and trigger-condition match. The old 3-intent whitelist is gone. | `procedural_min_conf`, `retrieval_procedural_limit` | `0.3`, `5` | YES |
| Batch-summary leg (own) | `src/retrieval/orchestrator.py:2107` | — | This conversation's own batch summaries, ranked by similarity, prefixed `[summary, <date>]`. | `retrieval_batch_summary_limit` | `3` | YES |
| Cross-conversation summary leg | `src/retrieval/orchestrator.py:2148` | C4/D3b | Other conversations' evolving whole-conversation summaries as overview hits. Private conversations never leave their scope; the active conversation's own summary is excluded (the assembler injects it). | `retrieval_conversation_summary_limit` | `2` | YES (off under incognito) |
| Cold-storage leg | `src/retrieval/orchestrator.py:2179` | T3 | Searches archived turns, but **only for time-scoped queries with a resolved window**. Ranks by meaning when the cold row has a vector, keyword patterns only for pre-embedding rows. | `timescope_cold_limit` | `5` | YES (fires only when a time window was parsed) |
| Cold resurrection (probation) | `src/retrieval/orchestrator.py:2269` | T3/D-U1 | A cold memory selected by retrieval budgeting moves back into live memory with its original source spans and timestamp provenance with a decay score just above the archive line. Unengaged, it re-archives within days. | `timescope_probation_score` | `0.12` | YES (when a cold hit survives the budget) |
| Timeline leg | `src/retrieval/orchestrator.py:1769`, `src/retrieval/evolution.py:74` | T4/D-U2 | When a matched entity carries real supersession history, attaches a dated "[Timeline: X]" block showing how the fact changed over time. Fuses and budgets as its own leg. | `timeline_max_fragments`, `timeline_max_transitions`, `timeline_max_tokens` | `2`, `8`, `300` | YES |
| Evolution timeline widening | `src/retrieval/orchestrator.py:1721` | T4 | Under an "how did X evolve" query, more timeline fragments are allowed and the vector leg's candidate pool widens before era stratification. | `timeline_max_fragments_evolution`, `retrieval_vector_candidate_limit_evolution` | `4`, `300` | YES (evolution mode only) |
| Era-stratified sampling | `src/retrieval/orchestrator.py:967` | T4 | For evolution queries, cuts candidates into equal-count time buckets and keeps each bucket's best, so the idea's early life is represented and not drowned by the recent, densest era. | `evolution_era_buckets`, `evolution_per_era` | `4`, `3` | YES (evolution mode only) |
| Wide-net fallback | `src/retrieval/orchestrator.py:2691` | C15 | When **both** the topic and intent heads are genuinely unsure, replaces all legs with one broad vector sweep plus the codex leg. Widens ranking, not visibility — every scope, privacy and time filter still applies. | `confidence_fallback_threshold`, `retrieval_wide_net_candidate_limit`, `retrieval_wide_net_budget_fraction`, `retrieval_wide_net_budget_floor` | `0.75`, `100`, `0.3`, `1500` | YES (fires only on double-head uncertainty) |
| Leg-failure reporting | `src/retrieval/orchestrator.py:190` | G36 | Every leg that swallows an error logs `retrieval_leg_failed` with the leg name, every time. DB errors roll back; Python errors do not (rolling back would expire the request's identity map). | — | — | YES |
| Honest empty-window note | `src/retrieval/orchestrator.py:2322` | T3 | A windowed query with no episodic matches injects an explicit "no memories between X and Y" note, plus the nearest eras that do have matches. Never silently widens. | — | — | YES (windowed queries only) |
| Retrieval strengthening (write-on-read) | `src/retrieval/orchestrator.py:2665` | G38/Z1 | Selected episodic row IDs gain access count/decay once per retrieval, even for multiple chunks; one atomic SQL update. The same write switch gates cold restoration. This is retrieval selection, not final-prompt exposure or answer-use proof. | `retrieval_strengthen_writes`, `decay_strengthen_amount` | `True`, `0.15` | YES |
| Codex evidence promotion on read | Removed from `src/retrieval/orchestrator.py` | G38/G70 | Candidate lookup never strengthens/promotes graph facts; selected-context usage affects retention only, never source support. Four obsolete read-promotion settings removed. | — | — | NO — removed |

---

### 8. Codex graph internals — `src/retrieval/orchestrator.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Micro-NER prompt entity extraction | `src/retrieval/ner_utils.py:266`, called `orchestrator.py:1677` | A9b | Pulls entity strings out of the live prompt to look up in the graph. Uses the already-loaded encoder, so it costs no extra VRAM on the request path. | — | `tier="preflight"` | YES |
| NER regex fallback (loud) | `src/retrieval/ner_utils.py:311` | G31 | If the micro-NER `.pt` file is missing, falls back to a capitalised-word regex and **warns with the cwd** — this silently halved entity counts before it logged. | — | — | YES (as a guard) |
| Entity matching by vector | `src/retrieval/orchestrator.py:1064` | A4/G41 | Nearest graph entity by cosine, ranked by Postgres rather than a Python loop over the whole graph. | `codex_entity_match_threshold` | `0.85` | YES |
| Entity matching, exact | `src/retrieval/orchestrator.py:1426` | A4 | Canonical-name / alias exact match. Production fallback and the ablation `fuzzy_match=False` path. | — | — | YES (as fallback) |
| Entity matching by payload | `src/retrieval/orchestrator.py:1442` | A4 | Descriptor fallback — "main fortress" finds the entity whose payload mentions "fortress". | `codex_entity_payload_match_limit` | `20` | YES (when the first two fail) |
| **Extraction direction rule** | `src/workers/codex_extractor.py:874` (prompt rule 8 + a passive worked example) | G59 | Tells the extractor which argument goes in `subject`: the active case, the `_by` mirror case, that a passive sentence does not force a passive relation, and a read-back check. Added because ~25–28% of stored triplets are REVERSED and the rate did NOT move when 1,611 direction-changing canonicalisation merges were blocked — so reversals are GENERATED, not merged in, and the prompt had no direction instruction at all while its three examples were all active SVO. ⛔ **EFFECT MEASURED 2026-08-23: NULL.** 4 arms, 2 seeds each, full turn coverage — correct **15.8% ON vs 15.5% OFF** (delta +0.28 pts, z=0.09); `reversed`, the label it was built to move, **32.6% vs 32.6%** (delta +0.01). The earlier "+9.4 pts" is **WITHDRAWN** — it compared a flat-sampler control against per-turn treatments; the same control store re-judged per-turn is 16.7%, not 11.0%. Smallest detectable effect at this design ±6.4 pts. **Still ON by default and still a flag** — it makes the prompt correct on its own terms and the A/B seam is worth keeping; removing it is a user decision, not implied by this null. | `codex_extraction_direction_rule` | `True` | **YES — measured null, kept deliberately** |
| **Summariser generation ceiling** | `src/workers/post_flight.py:91` | G57 | Was a literal `300` and it BOUND: 84 of 212 summaries (40%) came back truncated, and 47 have no `abstract_text` because the Abstract line the prompt asks for sits after the Key-terms block and a 300-token cut removes it. Now a setting; raising it took truncations 138 → 73 on a full seed. | `turn_summary_max_tokens` | `900` | YES |
| **Seeder model warm-up** | `scripts/z1/seed_store.py:341` | Z1 | One throwaway extraction before the first real turn. The FIRST call after Ollama loads a model returns different text from every call after it (633 chars cold vs 650 warm, identical input, temperature 0), and that one call cascades — canonicalisation feeds accepted relations back into the vocabulary. Fixing it took the two-seed edge gap from −17% to −2.9%. Announces loudly on failure and marks the run non-comparable. ⚠ Does NOT achieve full reproducibility — see [TRAPS #47](TRAPS.md), the model varies across processes. | — | always on (seeder only) | YES |
| Grounded query expansion | `src/retrieval/orchestrator.py:1650`, `:530` | A4 | Adds matched entities' canonical names and aliases to the BM25 search prompt, so lexical search finds turns using the full name. Nothing is generated — this replaced HyDE. **⚠ THE "IT CHANGES NOTHING" VERDICT IS SUSPENDED (2026-08-22), NOT CONFIRMED AND NOT REFUTED.** The 2026-08-20 ablation (`episodic_lookup` 0.664/0.664, `temporal` 0.536/0.536, and +0.033 on `summary_synthesis` from REMOVING expansion) was taken on instruments since found defective in ways that hit both of its conditions equally: the harness classified without the conversation, moving the RRF blend weights on **65%** of probes ([G54](ROADMAP.md#g54)), and the summary leg was off entirely ([G53](ROADMAP.md#g53)), which is what the `summary_synthesis` half of that verdict was reading. A null produced by two identically-mis-called conditions is the exact shape [TRAPS #43](TRAPS.md) describes — the arms agreed with each other and neither was production. ⇒ **Re-measure under the [G52](ROADMAP.md#g52) re-run before crediting or deleting this.** Marking a live path inert is the mirror of the failure this column exists to prevent, and it is the more expensive direction: a feature deleted on a bad null does not come back. | `codex_expansion_max_terms` | `8` | **YES — effect unmeasured pending re-run** |
| Relation-fit ranking | `src/retrieval/orchestrator.py:1491`, `:1534` | G34 | Ranks an anchor's own edges by how well each relation answers the question, and scores the anchor by how sharply one relation stood out. A contentless prompt earns zero by construction — no threshold needed. | `codex_relation_fit_weight`, `codex_relation_pool_multiplier`, `codex_entity_edge_limit` | `0.25` (unmeasured), `5`, `10` | YES |
| Bounded graph fan-out | `src/retrieval/orchestrator.py:1903`–`1954` | G35 | Caps how many neighbours each node expands, ranked by relation fit then trust, so a hub entity's fragment size is set by the question rather than by the corpus. | `codex_max_fanout` | `12` (explicitly unmeasured) | YES |
| Depth-graded rendering | `src/retrieval/orchestrator.py:1815` | A7.2 | The anchor injects its full note; deeper neighbours get a one-line preview (name + type + snippet). | `codex_entity_edge_limit` | `10` | YES |
| Bidirectional traversal | `src/retrieval/orchestrator.py:1871` | A7.2 | Walks outgoing links and incoming backlinks alike, so the graph is navigable both ways. | — | — | YES |
| Negated-edge handling | `src/retrieval/orchestrator.py:1486`, `:1566`, `:1909` | A8 | "X does NOT use Y" renders as `NOT`, never ranks first on a question about using, and is never walked as a navigable link. | — | — | YES |
| Edge trust = strength × confidence × recency | `src/retrieval/orchestrator.py:1792` | A3/A11 | An edge's weight combines how often it has been used, how well it was extracted, and how recently it was asserted. Recency only rewards; it never penalises age. | `codex_recency_boost`, `codex_recency_tau_days`, `codex_deep_strength_floor` | `0.3`, `30.0`, `1.0` | YES |
| Relation-vocabulary embedding cache | `src/retrieval/orchestrator.py:1138` | G45/G41 | Embeds the union of the seed relation list **and every relation the graph actually uses**, once per process — so relation fit keeps working for relations invented after the code was written. | `codex_relation_open_vocabulary` | `True` (write side) | YES |
| Graph enumeration ("list all the X") | `src/retrieval/orchestrator.py:1588` | A4 (re-homed MERA) | Answers entity-less category queries from the graph itself when the prompt carries an explicit enumeration cue AND a grounded tag/relation signal. No LLM. | `codex_enum_entity_limit`, `codex_enum_edge_limit` | `8`, `15` | YES (cue-gated) |
| Relation detection (enumeration only) | `src/retrieval/orchestrator.py:1198` | A4/G34 | Two channels — a relation's content word appearing in the prompt, and top-k gloss similarity. **Now feeds enumeration only**; the anchor path uses relation-fit instead. | `codex_relation_detection_enabled`, `codex_relation_top_k`, `codex_relation_sim_floor` | `True`, `5`, `0.45` | YES (but only reachable via enumeration) |
| Bi-temporal `valid_at(T)` reads | `src/retrieval/orchestrator.py:1467` | T3/D4 | Under a time window, an edge counts if it was established by the window's end and had not expired then — "what was true back then", not "what is true now". | — | — | YES (windowed queries only) |

---

### 9. Fusion, selection and budget — `src/retrieval/orchestrator.py`, `leg_weights.py`, `coverage.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Local relevance reranker | `src/retrieval/reranker.py`, `orchestrator.py::retrieve`, `_wide_net_fallback` | v3 repair / G64/G66 | Scores rendered candidates and alternatives before caps/collapse. Pinned local Qwen3-Reranker-0.6B; weights return to CPU after scoring. Warning + original fusion fallback on failure. | `retrieval_rerank_enabled`, `retrieval_rerank_model`, `retrieval_rerank_revision`, `retrieval_rerank_device` | `True`, `Qwen/Qwen3-Reranker-0.6B`, `e61197ed45024b0ed8a2d74b80b4d909f1255473`, `auto` | YES, requires cached weights |
| Reranker bounds | `src/retrieval/reranker.py` | v3 repair | Caps candidates, model batch and full templated input length; overlength fails visibly rather than scoring a truncated prefix. | `retrieval_rerank_candidates`, `retrieval_rerank_batch_size`, `retrieval_rerank_max_tokens` | `64`, `4`, `4096` | YES |
| Reranker rejection floor | `src/retrieval/reranker.py` | v3 repair | Optional yes/no logit floor. Zero admitted 4/45 synthetic distractors, so rejection remains unqualified. | `retrieval_rerank_min_score` | `None` | NO |
| Relevance-ordered packing | `orchestrator.py::_enforce_token_budget` | v3 repair | Successful reranking uses score order without source-type quotas; alternatives were scored independently. Oversized candidates do not hide later small ones. | `retrieval_rerank_enabled` | `True` | YES on successful rerank |
| Reciprocal-rank fusion | `src/retrieval/orchestrator.py:2369` | — | Merges the legs by rank rather than by raw score, weighted per leg. | `retrieval_rrf_k` | `60` | YES |
| Intent-dependent leg weights | `src/retrieval/leg_weights.py:101` | G9/D9 | Each active intent contributes its opinion about how much each leg should count; topic overrides add on top. E.g. `Code_Change` pushes procedural and codex up, `Casual_Banter` pushes everything down. | `retrieval_leg_base_weights`, `retrieval_leg_profiles`, `retrieval_leg_topic_overrides` | 13 intent rows + 2 topic rows | YES |
| Leg-name validation (raises) | `src/retrieval/leg_weights.py:63` | G9 | A weight on a non-existent leg is refused outright, because fusion would silently ignore it. Validated at **read** time, so a runtime sweep cannot bypass it. | — | — | YES |
| Unknown-label warning (warn-once) | `src/retrieval/leg_weights.py:73` | G9 | A profile row naming a label the live checkpoint does not have warns once and is ignored — a v1 rollback drops `Code_Change` and must not take retrieval down. | — | — | YES |
| All-zero weight guard | `src/retrieval/leg_weights.py:165` | G9 | If every rankable leg collapses to ≤0, warns loudly and falls back — an all-zero set admits every candidate at score 0 (recency decides, relevance discarded), which is unrankable rather than restrictive. | — | — | YES |
| Post-fusion bonuses | `src/retrieval/orchestrator.py:276` | — | Adjusts each fragment's score by keyword match, length brackets, recency band and a meta-discussion downweight, clamped to a total multiplier. | `retrieval_bonus_*`, `retrieval_penalty_short`, `retrieval_max_total_bonus_multiplier`, `retrieval_min_total_bonus` | keyword `1.0`, long `1.5`, substantial `0.5`, short `-0.7`, bookmarked `0.5`, recent top-10% `1.0` / top-30% `0.5`; clamp `[-0.9, 4.0]` | YES |
| Meta-discussion downweight | `src/retrieval/orchestrator.py:307` | — | When the query wants narrative fact, turns whose stored intent leans analytical are multiplied down. Classifier-driven, not string-matched. | `retrieval_meta_downweight_factor` | `0.55` | YES |
| Recency-band bonus | `src/retrieval/orchestrator.py:317` | — | The most recent 10% / 30% of a conversation's turns get a bonus — skipped entirely below a minimum turn count, where "the recent 10%" is one turn and means nothing. | `retrieval_recent_top_pct`, `retrieval_recent_mid_pct`, `retrieval_recency_min_turns` | `0.10`, `0.30`, `20` | YES |
| Cluster-scoped retrieval | `src/retrieval/orchestrator.py:219` | C5/G29 | Picks the clusters most relevant to the prompt (embedding similarity + topic-tag overlap) and scopes the episodic legs to them, keeping an adaptive band rather than a fixed top-10. Unclustered turns always stay visible. | `retrieval_cluster_top_k`, `retrieval_cluster_candidate_multiplier` | `10`, `3` | YES |
| Explicit cluster pick wins | `src/retrieval/orchestrator.py:499` | C6 | A hand-picked cluster set is never overwritten by the automatic picker. | — | — | YES |
| Session diversification | `src/retrieval/orchestrator.py:2529` | — | Caps how many fragments one *foreign* conversation may contribute; the active conversation is uncapped. | `retrieval_max_per_conversation` | `3` | YES |
| Exact-text dedupe | `src/retrieval/orchestrator.py:2546` | — | Drops byte-identical fragments. | — | — | YES |
| Provenance collapse | `src/retrieval/orchestrator.py:2406` | C16 | A chunk and its parent turn are the same memory — an ID join, not a similarity check — so the same text is not injected twice by two different legs. | `retrieval_collapse_enabled`, `retrieval_max_frags_per_turn` | `True`, `2` | YES |
| Coverage-based stopping | `src/retrieval/orchestrator.py:2484`, `src/retrieval/coverage.py:108` | C16/Z2 | Asks "do I have enough to answer this yet?" against the *question* rather than against the other fragments: repeatedly takes the candidate covering most of what is left, and stops when nothing adds anything. Catches redundancy without looking at words. | `retrieval_coverage_enabled`, `coverage_alpha`, `coverage_min_gain`, `coverage_min_keep`, `coverage_max_keep` | **`False`**, `0.7`, `0.02`, `2`, `40` | **NO** |
| Coverage knee detection | `src/retrieval/coverage.py:79` | C16 | Cuts where the cumulative-coverage curve bends (Kneedle), with a prominence floor so a curve with no knee is not randomly truncated. | `coverage_knee_enabled`, `coverage_knee_min_prominence` | `True`, `0.08` | NO in practice (only runs when coverage is on) |
| Set-level quality floor | `src/retrieval/orchestrator.py:2492`, `src/retrieval/coverage.py:65` | C16 | The only arm that can say "there is nothing here" — if the best candidate is far below what a real match looks like, injects nothing rather than the best of a bad set. | `retrieval_set_floor_enabled`, `retrieval_set_floor` | **`False`**, `0.25` | **NO** |
| Unplaced-fragment admission | `src/retrieval/coverage.py:125` | C16 | A fragment ICE cannot place in the vector space (a rendered timeline, a codex payload with no embedding) is admitted without competing, and the count is logged rather than inferred. | — | — | NO in practice (coverage off) |
| Fragment vectors fetched, not encoded | `src/retrieval/orchestrator.py:2442` | C16 | Coverage reuses stored embeddings rather than encoding 100 fragments (~380 ms) on the request path. | — | — | NO in practice (coverage off) |
| Fusion-fallback leg-diversity guarantee | `src/retrieval/orchestrator.py:2561` | A10 | Each leg's single best fragment is admitted first, so no leg is completely crowded out. Keys on `source_type` deliberately, not on `leg`. | `retrieval_leg_guarantee_enabled` | `True` — **but the setting has no reader; see DEAD** | YES (unconditionally) |
| Fusion-fallback round-robin budget fairness | `src/retrieval/orchestrator.py:2592` | A10 | Each round, every leg contributes its next-best fragment, highest-scoring leg first — so episodic (dozens of fragments) cannot soak the whole remainder. Exhausted legs drop out and their share goes to the rest. | — | — | YES |
| Degrade-before-drop | `src/retrieval/orchestrator.py:2569` | C1/C3 | A fragment too big for the remaining budget is swapped for its trusted summary, then a independently source-supported abstract, before being dropped entirely. | `turn_summary_coverage_threshold` | `0.7` (term-retention prerequisite) | YES |
| Leg budget-share telemetry | `src/retrieval/orchestrator.py:2628` | G35 | Logs what share of the injected tokens each leg actually took, on every retrieval, not sampled. Reveals a leg crowding others out *inside* the budget — which a total-tokens check cannot show. | — | — | YES |
| Dynamic budget split | `src/retrieval/orchestrator.py:372`, `:407` | CL4/G9/C16 | Splits the context budget between the recent-turns window and retrieval, by conversation length, average turn size and active labels. Leftover budget is deliberately left **unspent** — that is what makes ICE token-efficient. | `context_growth_cap_ladder`, `context_recent_fraction_ladder`, `context_recent_density_ladder`, `context_recent_fraction_groups`, `context_overhead_reserve` | 3-row cap ladder; 4-row fraction ladder + default `0.15`; 3-row density ladder; 5 label groups; `1800` | YES |

---

### 10. Scoping, privacy and exclusion — `src/retrieval/orchestrator.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Scope resolution (shared) | `src/api/main.py:421`, `src/services/scoping.py:114` | C6 | One resolver turns a conversation row into a retrieval scope. Modes: `none` (incognito), `manual` (user's closed pick), project-attached, `project`, `auto` (unscoped). | — | `auto` for a new conversation | YES |
| Private-turn invariant | `src/retrieval/orchestrator.py:808`, `:899`, `:1003`, `:2210`, `:2707` | G16 | Global search never sees private turns. Explicit conversation scoping is the only door to them, and every episodic leg carries the same filter. | — | — | YES |
| Incognito isolation | `src/retrieval/orchestrator.py:539`, `:1274` | G16/C6 | A `none`-scoped conversation reads only itself; codex resolves the empty set; procedural and cross-conversation summaries are skipped entirely. | — | — | YES (when scope = `none`) |
| Conversation-set filter | `src/retrieval/orchestrator.py:639` | D11/C6/G29 | One shared filter for all episodic legs — a single conversation or a list. An **emptied** set matches nothing rather than falling through to global. | — | — | YES |
| Cluster filter (shared) | `src/retrieval/orchestrator.py:659` | C5/G29 | One cluster-scope SQL fragment, used by all four episodic legs including the wide net (which had none until C6). Turns no clustering pass has reached yet stay visible. | — | — | YES |
| Exclusion sets ("keep it, stop reading it") | `src/retrieval/orchestrator.py:693`, `:1361` | C6 | Excluded conversations and clusters are never retrieved, in **every** scope mode. An entity survives unless *every* piece of evidence for it comes from an excluded conversation. | — | — | YES (when exclusions are set) |
| Document knowledge-vs-text split | `src/retrieval/orchestrator.py:1390` | C12 | A document switched off in a chat stops contributing its text but its extracted facts stay in the graph. Text visibility and knowledge visibility are different questions. | — | — | YES |
| Fail-closed scope resolution | `src/retrieval/orchestrator.py:1276`, `:1347` | G36 | If the scope or exclusion resolver errors, the codex and procedural legs match **nothing** instead of everything. Failing open would widen visibility silently. | — | — | YES |
| Empty conversation set = nothing | `src/retrieval/orchestrator.py:1311`–`1326` | G37 | A named conversation set that resolves to no batches reads nothing, not everything — which is the first turn of every manually-scoped conversation. | — | — | YES |
| Project derived-entity visibility | `src/retrieval/orchestrator.py:618`, `:630` | E1b/D3 | Code-graph and project-fact entities are invisible outside their project, even when reachable through a conversation edge. | — | — | YES |
| Project code-graph allowance | `src/retrieval/orchestrator.py:1335` | E1b/D11 | An attached project adds its derived entities and the static-edge batch to the allowed set, and a project scope never falls back to unscoped. | — | — | YES (project conversations) |
| Project conventions in procedural | `src/retrieval/orchestrator.py:2055` | E1/D1 | Project-scoped conventions are invisible outside their project and pass regardless of batch scope inside it. | — | — | YES |
| Working-tree freshening on read | `src/services/retrieval_svc.py:131` (via `settings`) | E11/D5 | A project-scoped pull runs `git status` first (rate-limited) so retrieved code pointers match the tree being edited right now. | `reconcile_on_read`, `reconcile_on_read_min_interval_seconds` | `True`, `2.0` | YES |

---

### 11. Temporal retrieval (Track T) — `src/retrieval/timescope.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| TimeScope kill switch | `src/retrieval/timescope.py:428`, `orchestrator.py:601` | T2 | Off makes every query behave as "current", byte-identical to pre-T behaviour. | `timescope_enabled` | `True` | YES |
| Deterministic time parsing | `src/retrieval/timescope.py:258` | T2 | Parses ISO dates, month+year, month+day, "5th of march", quarters, halves, early/mid/late, seasons, anchored bare years and months, "N days/weeks/months/years ago", "yesterday", "the other day", "last week/month/year". Regex only, no LLM, <1 ms. | — | — | YES |
| Joint gate | `src/retrieval/timescope.py:446`–`465` | T2/A4 | A time expression alone never flips the mode — it must co-occur with a recall shape (question mark, interrogative opener, or high `Needs_Memory`). "Yesterday" inside a statement stays "current". | — | — | YES |
| Code stripping | `src/retrieval/timescope.py:433` | T2 | Temporal words inside fenced or inline code are content, not intent, and are removed before scanning. | — | — | YES |
| Vague pasts not resolved | `src/retrieval/timescope.py` (by omission, module docstring) | T2 | "A while back" / "long ago" are deliberately NOT turned into a window — an invented window silently hides everything outside it. | — | — | YES |
| Window padding | `src/retrieval/timescope.py:189`–`222` | T2 | Every resolved window is padded so a memory written a few days either side of the named date still matches, scaled by granularity. | `timescope_pad_min_days`, `pad_month_days`, `pad_span_days`, `pad_year_days`, `rel_short_frac/cap`, `rel_long_frac/cap` | `3`, `14`, `21`, `30`, `0.4`/`45`, `0.15`/`120` | YES |
| Evolution cues | `src/retrieval/timescope.py:153` | T4 | "How did X evolve", "evolution of", "over time", "originally", "at the start" put the query into evolution mode. A compare cue counts only when two expressions resolved. | — | — | YES |
| Future guard | `src/retrieval/timescope.py:498` | T2 | A window entirely in the future is scheduling, not memory — falls back to "current". Windows never extend past now. | — | — | YES |
| Mode-aware time filters | `src/retrieval/orchestrator.py:720` | T3 | Under a window, archived turns become visible and the decay floor drops to 0 (archived rows sit below 0.1 by construction). "Current" keeps the 0.2 floor and hides archived. | — | floor `0.2` current / `0.0` windowed | YES |
| Mode-aware recency origin | `src/retrieval/orchestrator.py:739` | T3/D9 | "As of" re-anchors the recency boost to the window's midpoint (proximity to the target, not freshness); "range"/"evolution" flatten it entirely — the user asked for a period. | `retrieval_episodic_recency_boost/tau` | `0.25`, `30.0` | YES |
| Summary leg skipped under a window | `src/retrieval/orchestrator.py:2112` | T3/D14 | A summary's creation date is long after the period it compresses, so serving it under a time window would mislead. | — | — | YES |
| Procedural span overlap | `src/retrieval/orchestrator.py:2030` | T3 | Under a window, a habit is relevant only if its observation span overlaps it — a habit first seen after the window did not exist then. | — | — | YES |

---

### 12. Fragment representation — `src/retrieval/orchestrator.py`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Read-time raw-vs-summary choice | `src/memory/representation.py:19` | C1/G75 | Shared by retrieval, chat recent history and explicit recent reads. Finite coverage AND current independent NLI support required. Unknown roles, changed text/model and missing verdicts retain raw. No foreground inference or hidden source prefix. | `turn_summary_coverage_threshold` | `0.7` | YES |
| Keyword protection | `src/memory/representation.py:19` | C1/G75 | Each matched query term must survive a compressed representation; preserving one name cannot excuse dropping another. | — | — | YES |
| Date/time stamping | `src/memory/time_format.py`, `orchestrator.py`, `prompt_assembler.py` | T1 | Episodic alternatives, chunks and recent history retain source date/time/timezone. Synthetic import and unknown provenance are explicit. Summary creation/update timestamps are labeled and budgeted. Cold provenance is currently unknown. | — | — | YES |
| Fact dating | `orchestrator.py::_prime_edge_times`, `_fact_line` | T1 | Batched source-batch lookup separates source-recorded time from learned/recorded-validity time. It does not invent event dates from import clocks. | — | — | YES, explicit fact lines |
| Keyword-aware word cap | `src/retrieval/orchestrator.py:2861` | C2 | A fragment matching a prompt keyword is allowed 1500 words instead of 500. Documents are never injected whole — only their keyword-relevant chunks (max 2). | — | `500` / `1500` / 2 chunks | YES |
| Sentence-boundary truncation | `src/retrieval/orchestrator.py:105` | C3 | Cuts on the last sentence boundary inside the cap (if past 60% of it) so fragments stop mid-thought less often. | — | — | YES |
| Bookmark score multiplier | `src/retrieval/orchestrator.py:2882`, `:1047` | C1 | A bookmarked turn's retrieval score is multiplied up. | `retrieval_bonus_bookmarked` | `0.5` (i.e. ×1.5) | YES |

---

### 13. Classifier — `src/classifier/`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Pre-flight prompt classification | `src/classifier/classifier.py:129` | B1/D8 | The only classifier. MLP head over a frozen Qwen3-Embedding-0.6B encoder. v2 schema: 27 logits — 11 topic + 12 intent (multi-label) + 4 independent context sigmoids. | `classifier_model_path`, `label_schema_path` | `models/classifier/ice_classifier_v4_schema2.pt`, `data/labeled/label_schema.json` | YES |
| Schema-driven head layout | `src/classifier/schema.py:90` | B1/D1 | Head widths, offsets and label names all come from the schema JSON — no magic slices anywhere in the tree. | — | — | YES |
| Two-generation checkpoint loading | `src/classifier/model.py:179` | B1/D5 | Loads v1 or v2 checkpoints; the file declares which it is. A rollback is a file swap, not a code change. | — | — | YES |
| Checkpoint-stamped decision threshold | `src/classifier/classifier.py:68`, `:202` | B1 | The tag threshold travels with the weights (v1 calibrated 0.3, v2 sweeps to 0.65) — promoting a model promotes its calibration. The setting is only the fallback for checkpoints predating the stamp. | `classifier_threshold` | `0.3` (**fallback only** — the live checkpoint's own stamp wins) | YES |
| Argmax floor | `src/classifier/classifier.py:223` | — | If no label clears the threshold, the single highest label is emitted, so a turn is never untagged. | — | — | YES |
| CL7 prior-turn context prefix | `src/classifier/classifier.py:87`, `:162` | CL7/G26 | The last 3 turns (summary-preferred, word-capped, budget-truncated) are prefixed to the prompt before encoding, improving accuracy. Emits `cl7_context_prefix` so it is visibly live. | `classifier_context_turns`, `classifier_context_max_words`, `classifier_turn_word_cap` | `3`, `500`, `150` | YES |
| Shared train/inference templates | `src/classifier/templates.py:103` | B1/D3 | Both training and serving render the encoder input through one function, closing the v1 train/inference mismatch by construction. Templates are versioned and frozen. | — | template v2 | YES |
| Offline context twin | `src/classifier/templates.py:128` | B1/D3 | The pipeline builds the same-shaped context prefix from a message list that the live path builds from the DB. | `classifier_context_turns`, `classifier_context_max_words` | `3`, `500` | YES (pipeline only) |
| Derived 3-way context reliance | `src/classifier/schema.py:258`, `:285` | B1/D6 | The old `Zero_Shot` / `Long_Term_Memory` / `Real_Time_Search` string is derived from v2's independent sigmoids, so every pre-B1 consumer keeps working unchanged. | — | — | YES |
| `p_temporal` / `p_complex` exposure | `src/classifier/schema.py:325` | B1 | Two signals the 3-way could never express: a memory query with a time dimension, and "the strongest model would answer this materially better". | `temporal_label_threshold` | `0.85` | YES for `p_temporal`; `p_complex` has **no consumer** |
| Per-head confidences | `src/classifier/classifier.py:197` | C15/B3 | Publishes each head's peak probability separately, so the wide net fires on honest per-head uncertainty rather than a max over all 27 logits. | — | — | YES |
| Native 1024-dim input | `src/classifier/model.py:51`, `classifier.py:146` | C17/A9a | v2 heads consume the full embedding; a rolled-back v1 head automatically gets the 384-dim MRL prefix of the same encode. | `embedding_dim` | `1024` | YES |
| Encoder device placement | `src/classifier/classifier.py:135` | C16 | The encoder runs on the GPU, the ~25k-param head on CPU; one 1024-float copy per classification bridges them. GPU encode measured 21 ms vs 321 ms on CPU. | `embedding_device` | `"auto"` | YES |
| Checkpoint promotion (backup + atomic) | `src/classifier/promotion.py:56` | B1/B4 | Copies the outgoing checkpoint to a timestamped backup, then atomically replaces the live path. One implementation, two callers (fine-tune and the full retrain pipeline). | `classifier_model_path` | — | YES (when a promotion runs) |
| Unattended fine-tune gate | (setting) `src/api/config.py:84` | D6/H5 | The classifier is never retrained unattended: enough curated labels + a session end produces a **review-queue proposal**, not a promotion. | `auto_finetune`, `finetune_min_curated`, `finetune_val_fraction` | **`False`**, `20`, `0.2` | **NO** |
| Embedding cache for training | `src/classifier/dataset.py:123` | Z1-prep | Caches encoded rows next to the JSONL, keyed by row count + template version, so sweeps do not re-encode 25k rows. | — | `cache=True` | YES (pipeline only) |
| OOM-backoff encoding | `src/classifier/dataset.py:48` | B1 | Halves the batch size on CUDA OOM rather than capping sequence length — capping would truncate in training where inference does not. | — | batch `128` | YES (pipeline only) |
| Capped positive weights | `src/classifier/model.py:254` | B1 | Rare labels are heard without shouting: `neg/pos` ratios capped, because uncapped a 1%-prevalence label makes "say yes" the cheapest policy. | — | cap `3.0` | YES (training only) |
| Background NER tier | `src/retrieval/ner_utils.py:161`, `:266` | A9b | `tier="background"` uses NuNER Zero (entity types for free, 4.7× faster, 28× better conversation separation) for post-flight consumers. Empty model name disables it and everything falls back to the micro-NER. | `background_ner_model`, `background_ner_device`, `background_ner_threshold`, `background_ner_chunk_words` | `numind/NuNER_Zero`, `"auto"`, `0.5`, `250` | YES (background callers only — never the request path) |

---

### 14. Model registry — `src/model_registry/`

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Registry file | `src/model_registry/registry.py:26`, `:65` | G31 | `models/model_registry.json`, anchored to the install rather than the cwd — read from the wrong directory it silently returned an empty registry and handed a whole eval run the wrong model. | — | — | YES |
| Ollama discovery | `src/model_registry/registry.py:257` | — | Enumerates locally installed Ollama models and adds unseen ones with priority 5 and an 8192 context window. | `ollama_base_url` | `http://localhost:11434` | YES (on refresh) |
| Hugging Face auto-tagging | `src/model_registry/registry.py:135` | — | Maps HF model-card tags onto ICE topic/intent labels. An HF-tagged model is auto-confirmed. | — | — | YES |
| Background-model auto-tagging | `src/model_registry/registry.py:176` | G29 | If HF has nothing, asks the background LLM to tag the model against the label list, with a JSON schema constraint. Failure now logs `model_autotag_failed` (it had silently 404'd forever in the shipped config). | `background_model_mode`, `background_model_name` | `"shared"`, `None` | YES (fallback path) |
| Best-model selection | `src/model_registry/registry.py:295` | — | Scores confirmed models by topic overlap + intent overlap + priority. Falls back to the first confirmed model, then the configured default. | `default_fallback_model` | `qwen2.5:7b` | YES (only reachable via `model: "ice-proxy"`) |
| Confirmed-only gate | `src/model_registry/registry.py:303` | — | An unconfirmed model is never routed to. A user must confirm it via `PUT /user-control/model-registry/{name}`. | — | — | YES |
| Context-window lookup | `src/model_registry/registry.py:320` | C16 | Provides the model's declared window to the budget derivation. Returns None for an unknown model, which falls to `context_budget_fallback`. | `context_budget_fallback` | `23000` | YES |
| Runtime window probe | `src/model_registry/runtime_probe.py:44` | C16 | Reads the live runner's allocated window (`/api/ps`) and the GGUF architectural max (`/api/show`), cached per model. | `runtime_probe_timeout`, `runtime_probe_cache_ttl_seconds` | `2.0`, `300` | YES |
| Over-window warning | `src/model_registry/runtime_probe.py:112` | C16 | Warns when the derived budget exceeds **any** visible ceiling, not just the live runner's — tinyllama registers 8192 against a 2048 GGUF ceiling. | — | — | YES |

---

### 15. Ablation harness — `src/retrieval/configurable_orchestrator.py`

Not on the live request path (`main.py:594` constructs `HybridRetrievalOrchestrator` directly). Listed because the final experiment will reach for it.

| Feature | Where | Roadmap id | What it does (plain) | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Per-leg ablation flags | `src/retrieval/configurable_orchestrator.py:30` | G19/G36 | Toggles `vector`, `bm25`, `rrf`, `cluster_restrict`, `session_diversify`, `codex`, `mera`, `fuzzy_match`, `procedural`, `batch_summary`, `dynamic_budget`, `keyword_boost`, `recency_boost`, `timescope`. Every flag defaults ON. | — | all ON | N-A (experiment only) |
| Real recency-boost override | `src/retrieval/configurable_orchestrator.py:146` | G19 | Saves and zeroes the settings values around the call. The old version rebound a *module-local copy* while scoring read the parent's — so `recency_boost: False` did nothing, for as long as the flag existed, making Exp 3's `add_keyword_boost` and `full_ice` arms the same configuration. | `retrieval_bonus_recent_top_10pct`, `..._top_30pct` | `1.0`, `0.5` | N-A |
| Simple merge (no RRF) | `src/retrieval/configurable_orchestrator.py:109` | — | Concatenates legs and sorts by raw score, for the `rrf: False` arm. | — | — | N-A |
| Fixed-budget arm | `src/retrieval/configurable_orchestrator.py:162` | — | `dynamic_budget: False` pins retrieval to 8000 tokens and the recent window to 4000. | — | hardcoded | N-A |
| Timescope ablation seam | `src/retrieval/configurable_orchestrator.py:45` | T2/T3 | `timescope: False` forces every query to CURRENT in the parent, collapsing all temporal branches without touching the legs. | — | — | N-A |

---

## DEAD OR INERT

Ordered by how likely each is to be assumed working.

1. **RESOLVED v3 2026-09-19: optional prompt blocks were buried in the essential system block.**
   Structured assembly now reports all six evictable blocks independently. The
   bounded reassembly consumer removes static context before evidence, preserves
   separately fetched project constraints, and refuses known required overflow.
   Regression checks exercise the actual assembler, SQL bookmark reader and chat
   route; historical claims that this already worked were incorrect.

2. **Per-leg attribution (`ContextFragment.leg`) never reaches any output on the default config.**
   `leg` is stamped in `_apply_rrf` (`orchestrator.py:2400`) and read in exactly one place — `_apply_coverage`'s audit record (`:2522`–`:2523`) — which only runs when `retrieval_coverage_enabled` is True, and it is `False`. Everything user-visible keys on `source_type` instead: the `retrieval` SSE event (`main.py:757`), `_log_leg_budget_share` (`:2646`), and the budget's round-robin lanes (`:2600`). C16 added this field precisely because "no experiment ICE has run could say whether BM25, vector search or chunk retrieval did the episodic work" — on the default config, that is still true.

3. **`settings.retrieval_leg_guarantee_enabled` (default `True`) has no reader.**
   `grep -rn retrieval_leg_guarantee_enabled src/` matches only `config.py:234`. Phase 1 of `_enforce_token_budget` (`orchestrator.py:2561`–`2567`) runs unconditionally. Setting it `False` changes nothing. (`docs/CLEANUP.md:479` records keeping it deliberately, so this is a known state, not a discovery — but the knob does not work.)

4. **`settings.ollama_num_ctx_mode` (default `"fit"`) has no reader anywhere in the repo.**
   `grep -rn ollama_num_ctx_mode .` matches only `config.py:771`. `_ollama_body` (`main.py:105`–`108`) hardcodes the "fit" computation (`prompt_tokens + context_generation_reserve`, clamped by `ollama_num_ctx_max`). There is no other mode to select.

5. **`release_background_ner()` has no caller.**
   Defined at `ner_utils.py:108`; `grep -rn release_background_ner .` finds only that definition and a docs reference. The stated design — "loaded for a drain and released again rather than sitting resident" (~1.7 GB, `config.py:183`–`185`) — does not happen: once `_load_background_ner()` runs, the model stays in VRAM for the process's lifetime.

6. **`ClassificationResult.p_complex` is computed and read by nothing.**
   Set at `schema.py:326`. `grep -rn p_complex src/` finds only the assignment and docstrings. The field's own comment says "B3 consumes it; nothing routes on it yet" — confirmed.

7. **`ClassificationResult.p_rts` has no live consumer either.**
   Set at `schema.py:315`/`:323`. Its only indirect path is `context_reliance == "Real_Time_Search"`, checked at `orchestrator.py:468` — but both callers overwrite the label to `Long_Term_Memory` *before* calling `retrieve()` (`main.py:587`, `retrieval_svc.py:108`), so that branch and the `Zero_Shot` branch at `:466` are unreachable from every production path. The code comment at `:456` already says they are "purely a defensive guard".

8. **`ctx_confidence` is telemetry only.**
   Computed at `schema.py:317`/`:323`, and the only reader outside the classifier is the log breakdown (`memory_decision.py:254`). The low-confidence bump uses `result.max_confidence` (`:202`), not this. So the head-margin signal does not participate in any decision.

9. **The temporal arm of the B2 decision is measured inert.**
   `memory_decision.py:237`–`240` adds `ltm_bump_timescope` (3.0). The comment at `:217`–`:236` records the E12 measurement: over 9,441 held-out rows, disabling the whole arm moves **one** decision, because Temporal_Recall rows carry mean `p_ltm` 0.931 and 98.3% already retrieve. This is a subset signal added to its own superset's decision. Sweeping `temporal_label_threshold` (`0.85`) here will not move anything; the earned consumers (T5) are unwired.

10. **`memory_decision.estimate_recent_window_tokens` duplicates the recent-fraction ladder in code, not from settings.**
    `memory_decision.py:122`–`134` hardcodes `<10 → 0.3`, `<200 → 0.2`, `else 0.15` plus `_TOTAL_CONTEXT_BUDGET = 23_000` and `_OVERHEAD_RESERVE = 1_800` (`:75`–`:76`). The orchestrator reads `settings.context_recent_fraction_ladder` / `context_overhead_reserve` for the same quantity (`orchestrator.py:407`, `:369`). They agree numerically today. A Z1 sweep of the ladder moves the orchestrator's split and leaves B2's memory-pressure window estimate frozen at the old values — the exact "second copy of a value" failure `config.py:245`–`249` warns about.

11. **`find_best_model(required_tokens=…)` is never exercised.**
    `registry.py:295` filters out models whose context window is too small — but the only caller (`main.py:518`) never passes the argument, so `required_tokens` is always 0 and the filter never applies.

12. **`ConfigurableOrchestrator` is unreachable from the request path.**
    `main.py:594` and `retrieval_svc.py:147` both construct `HybridRetrievalOrchestrator`. The subclass exists for `experiments/flaw_ablation/*` and two smoke tests only.

13. **`schema.empty_probs()` and `_LABEL_ONLY_PRIORS` have no producer.**
    `schema.py:131`, `:278`. DI3 was the only thing that emitted a probability-free result, and it was deleted in D8. Both survive as guards for hand-built `ClassificationResult` objects (tests, scratch scripts) — correct to keep, but nothing in `src/` reaches them.

14. **`_codex_enumeration`'s relation channel is doubly gated.**
    `codex_relation_detection_enabled` (default `True`) now governs only `_detect_relations` (`orchestrator.py:1215`), whose only caller is the enumeration path (`:1701`), which itself first requires a literal `_ENUM_CUES` substring ("list", "all", "who are", …) at `:1595`. Its scope shrank on 2026-08-11 and the comment at `config.py:681`–`686` says so; worth knowing before treating it as a general codex ablation seam.

15. **`LegacyICEClassifierV1` and `load_v1_schema` are live only for rollback/gate.**
    `model.py:136`, `schema.py:241`. Not dead — `_by_width()` (`schema.py:246`) and the D5 non-regression gate need them — but no production request loads them while `classifier_model_path` points at the v2 checkpoint.

---

## SURPRISES

1. **Mini-MoE routing is off unless the client asks for it by name.**
   `main.py:512` gates the entire registry-based model selection (and therefore session stickiness, and therefore the whole `find_best_model` scoring path) on `body["model"] == "ice-proxy"`. Any other model string is forwarded verbatim to Ollama. An experiment harness that sends a real model name gets *no* routing at all — while still getting the full retrieval and assembly path. This is easy to configure wrongly by accident and produces a plausible-looking run.

2. **Two of C16's three "is this enough" arms are off by default, and they are the only arms that can inject nothing.**
   `retrieval_coverage_enabled = False` and `retrieval_set_floor_enabled = False`. On the default config there is no mechanism anywhere in retrieval that says "there is nothing worth injecting" — the budget always fills to its cap with the best of whatever came back. The `coverage.py` module docstring is explicit that coverage measures direction and never quality, and that the set floor is the only quality gate; both are inert today.

3. **`ollama_send_num_ctx = False` means the model server picks its own window on every request.**
   `main.py:105`. Everything upstream — `derive_total_budget`, `serving_window`, the ledger, the safety margin — is careful arithmetic against a window ICE never actually requests. `runtime_probe.py`'s module docstring says the truncation casualty is the retrieved-memory block ICE just spent a request building. This is the single most consequential OFF flag in the request path.

4. **`_match_entities_by_similarity` does not apply the project-visibility filter that its two sibling matchers do.**
   `_match_entities_exact` (`orchestrator.py:1433`) and `_match_entities_by_payload` (`:1455`) both pass `*self._entity_source_filters()`. The vector matcher's SQL (`:1094`–`:1106`) queries `codex_entities` with no source or project predicate. Under `auto` scope `allowed_entity_ids` is `None`, so the anchor guard at `:1725` does not fire either — meaning a project-derived (code-graph / project-fact) entity can become a retrieval anchor in an unscoped conversation. Traversal *neighbours* are still filtered by `_entity_visible` (`:1958`), so the leak is bounded to the anchor's own note. Worth a query against a populated store before the final run.

5. **Retrieval calls no LLM at all — and that is now enforced by deletion, not by policy.**
   `_hyde_rewrite` and the per-orchestrator background client were both removed (`orchestrator.py:134`, `:758`). Query expansion is grounded in the graph (`_expansion_terms`, `:1650`). This matters for the experiment: the pre-flight path's only model forward passes are one classifier encode, one prompt encode, and the micro-NER — there is no generation-side variance in retrieval.

6. **Cold storage and cold resurrection are far more complete than "archived" suggests.**
   The cold leg (`:2179`) ranks archived rows by *meaning* when they carry a vector, and a cold hit that survives the token budget is physically re-inserted into `episodic_memory` at its original timestamp with `decay_score = 0.12` and the cold row deleted (`:2269`–`:2320`). That is a write to the live store performed on the read path. It fires only under a parsed time window, so a probe set with no temporal prompts will never exercise it — and one with temporal prompts will mutate the store between runs.

7. **`_strengthen_retrieved` is a write on every retrieving turn, and the reproducibility cost is measured.**
   `retrieval_strengthen_writes` is `True` in production; `config.py:549`–`559` records 26/40 identical result sets with it on against 40/40 with it off. Any A/B comparison that leaves it on is competing against its own drift.

8. **`config.py` is not purely a settings file — it carries measurements.**
   Several defaults are annotated with the run that set them (`codex_relation_canonical_threshold` calibrated on 30 pairs, `token_count_safety_margin` from a live tinyllama reconciliation) and several are explicitly flagged unmeasured (`codex_max_fanout = 12`, `codex_relation_fit_weight = 0.25`, `procedural_similarity_threshold = 0.85` "PROBABLY WRONG"). Any headline number quoting these should carry the same flag.

9. **`probe_api_key` / `probe_api_base_url` / `probe_model` are runtime settings for a tooling-only path.**
   `config.py:522`–`532`. They exist in the production `Settings` class solely because pydantic forbids extra keys and bare `.env` lines took the whole application down. `probe_api_key` is a credential living in the same object the proxy loads — never log a settings dump.

10. **The RAG leg is gone, not disabled.**
    `orchestrator.py:2098`–`2105`. Documents now enter as their own conversations and are retrieved by the ordinary episodic/codex/cluster legs. `ConfigurableOrchestrator` no longer has a `rag` flag, so "ablate documents" now means ablating vector+bm25+codex — there is nothing separate left to switch off. Any experiment design carried over from Exp 1/2 that names a RAG arm is describing a system that no longer exists.

11. **Retrieval deliberately leaves budget unspent.**
    `orchestrator.py:398`: "NO reallocation — leftover stays unused. This is what makes ICE token-efficient compared to the vector baseline." A token-efficiency claim therefore depends on the growth-cap ladder, and `config.py:618`–`623` records that `dynamic_budget` scored **worst** in the flaw ablation (−0.11 while adding tokens and doubling fragment count). The ladders are now data (`context_growth_cap_ladder` etc.) and unswept.

12. **Eight legs, five `source_type`s, and the budget's fairness lanes key on `source_type`.**
    `orchestrator.py:2600`. bm25, vector, chunks, cold and the wide net all report `"episodic"`, so they share one round-robin lane while codex, procedural, batch_summary and timeline each get their own. The RRF weights, by contrast, key on the leg name (`_apply_rrf:2386`) — so `cold` and `batch_summary` carry distinct fusion weights but no distinct budget share. This asymmetry is documented as intentional (`ContextFragment.leg` comment, `:61`–`:66`) and is easy to misread as a bug.

---

## Storage and background jobs — memory, workers

Scope: every file in `src/memory/` and `src/workers/`. Read from source on 2026-08-15; live-DB queries used where a claim about behaviour needed one (marked **[queried]**).

---

### 0. The job table itself — `src/workers/runtime.py` `JOBS` (C7)

Every entry, with its cadence from `settings.maintenance_intervals`. **A job with no interval is event-only and NEVER runs on a cadence.**

| Job name | Callable | Lane | Where | Interval (`settings.maintenance_intervals`) | Trigger | Cadence-run by default? |
|---|---|---|---|---|---|---|
| `post_flight` | `post_flight:evaluate_turn` | gpu | `src/workers/runtime.py:75` | — (absent) | event: `main.py:331` after a turn is stored | NO (event-only) |
| `codex_extract` | `codex_extractor:extract_codex` | gpu | `src/workers/runtime.py:78` | — | event: `src/services/bookmarks.py:41` (bookmark endpoint) only; the normal path is a direct call inside post_flight | NO (event-only) |
| `chunk_pending_documents` | `document_chunker:run_pending_documents` | cpu | `src/workers/runtime.py:79` | 7200 s (2 h) | overdue | YES |
| `cluster_assignment` | `clustering:run_cluster_assignment` | cpu | `src/workers/runtime.py:80` | 1800 s (30 min) | overdue + `session_gap` work-unit | YES |
| `cluster_merge` | `clustering:run_cluster_merge` | cpu | `src/workers/runtime.py:81` | 10800 s (3 h) | overdue | YES |
| `compaction` | `compaction:compact_entities` | cpu | `src/workers/runtime.py:82` | 86400 s (24 h) | overdue | YES |
| `decay_episodic` | `decay:apply_decay` | cpu, `pass_cycles` | `src/workers/runtime.py:83` | 5400 s (1.5 h) | overdue | YES |
| `decay_codex` | `codex_decay:decay_codex_edges` | cpu, `pass_cycles` | `src/workers/runtime.py:84` | 5400 s | overdue | YES |
| `decay_procedural` | `procedural_decay:decay_procedural_patterns` | cpu, `pass_cycles` | `src/workers/runtime.py:85` | 5400 s | overdue | YES |
| `reflection` | `reflection:run_reflection` | gpu | `src/workers/runtime.py:86` | 7200 s | overdue + session-end burst | YES |
| `batch_summarize` | `batch_summarizer:batch_summarize` | gpu | `src/workers/runtime.py:87` | 7200 s | overdue + session-end burst | YES |
| `conversation_summary` | `conversation_summary:run_conversation_summaries` | gpu, needs_db | `src/workers/runtime.py:90` | 7200 s | overdue + session-end burst | YES |
| `maintenance_agent` | `maintenance_agent:run_maintenance_agent` | gpu, needs_db | `src/workers/runtime.py:93` | 43200 s (12 h) | overdue + session-end burst | YES |
| `fine_tune` | `fine_tune:fine_tune_classifier` | gpu | `src/workers/runtime.py:94` | — (deliberately absent) | session-end burst **only if** `auto_finetune` | **NO** |
| `project_reconcile` | `src.coding.reconciler:reconcile_project` | cpu, needs_db | `src/workers/runtime.py:99` | — | event: `commit` work-unit handler (`src/api/core.py:95`) | NO (event-only) |
| `project_poll` | `src.coding.reconciler:poll_projects` | cpu, needs_db | `src/workers/runtime.py:100` | 600 s (10 min) | overdue | YES |
| `decision_extract` | `decision_extractor:run_decision_extraction` | gpu, needs_db | `src/workers/runtime.py:101` | — | event: `src/coding/reconciler.py:196`; also a direct call in post_flight for project-attached turns | NO (event-only) |
| `import_replay` | `src.ingestion.importer:run_import_replay` | gpu, needs_db | `src/workers/runtime.py:107` | — | event, self-re-enqueueing slices (`src/services/ingestion.py:114`) | NO (event-only) |
| `ingest_document` | `src.services.documents:run_document_ingest` | gpu, needs_db | `src/workers/runtime.py:111` | — | event: upload (`src/services/documents.py:184`) | NO (event-only) |
| `ingest_folder` | `src.ingestion.documents.watch_folder:run_watch_folder` | cpu, needs_db | `src/workers/runtime.py:114` | 900 s (15 min) | overdue | YES |

---

### 1. Maintenance runtime — scheduling, gating, leases (C7 / D1–D10, G4, G24)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| In-process maintenance runtime (replaces Celery+Redis) | `src/workers/runtime.py:188` | C7 | One asyncio tick task inside the proxy owns all background work. No separate worker process. | — | — | YES |
| Two lane semaphores | `src/workers/runtime.py:206` | C7 | GPU-touching jobs run one at a time; DB-only jobs two at a time, so background work never stacks against itself on the card. | — (hardcoded 1 / 2) | 1 / 2 | YES |
| ~60 s jittered tick | `src/workers/runtime.py:162`, `:391` | C7 | The scheduler wakes every 60 s + 0–15 s of jitter to check what is overdue. | — (hardcoded) | 60 / 15 | YES |
| Idle gate for overdue dispatch | `src/workers/runtime.py:302` | C7 D7 | Periodic jobs only start when no generation is in flight and the user has been quiet. | `settings.user_active_threshold_seconds` | 90 | YES |
| Idle-burst gate for queued gpu events | `src/workers/runtime.py:324` | C7 D7 | A queued post-flight waits for this much quiet before it starts; cpu-lane events go immediately. | `settings.idle_burst_seconds` | 120 | YES |
| Mid-flight yield (a running job stands down) | `src/workers/runtime.py:308`, `:762` | G4(a) | If the user comes back while a job is running, the job abandons at its next chunk/turn boundary and is requeued — no retry burned, ledger row marked `yielded`. | `settings.job_yield_grace_seconds` | 10 | YES |
| Yield call sites | `codex_extractor.py:748`, `batch_summarizer.py:68`, `conversation_summary.py:162` | G4(a) | The three places a long job can cheaply abandon: per extraction chunk, per conversation summarised, per conversation folded. | — | — | YES |
| GPU-utilisation gate (nvidia-smi) | `src/workers/gpu_check.py:27` | C7 D7 / G4 | Dedicated mode only: refuse to start bg work while the card is above the threshold. **Always returns False in shared mode.** | `settings.gpu_util_threshold`, `gpu_check_cache_seconds`, `background_model_mode` | 70, 10.0, `"shared"` | NO (inert in the default shared mode) |
| Event queue with single-flight dedupe | `src/workers/runtime.py:339` | C7 D4 | Identical (job, kwargs) already queued/running is dropped rather than double-run. | — | — | YES |
| Queue-depth warning | `src/workers/runtime.py:356` | C7 §4 | Logs a WARNING when more than 20 events are backed up. | — (hardcoded `QUEUE_DEPTH_WARN`) | 20 | YES |
| Exponential retry with backoff | `src/workers/runtime.py:519` | C7 | A failed job retries at 30 s / 120 s / 480 s, then logs `maintenance_job_failed`. | — (hardcoded `RETRY_DELAYS`) | (30,120,480) | YES |
| Ledger (`maintenance_ledger`) schedule state | `src/workers/runtime.py:662`, `models.py:484` | C7 D3 | Per-job last_started/last_finished/status/error/runs, surviving restarts; feeds overdue catch-up. | — | — | YES |
| Optimistic ledger claim (lease = 2× interval) | `src/workers/runtime.py:687` | C7 D3 | Stops a second app instance double-running periodic jobs (double decay destroys memory). Event runs stamp unconditionally. | — | — | YES |
| Overdue computation, incl. crashed-run grace | `src/workers/runtime.py:167` | C7 | Never-run ⇒ due now; started-but-unfinished ⇒ due after 2× interval. | — | — | YES |
| Closed-form decay catch-up | `src/workers/runtime.py:153`, `:488` | C7 D5 | After downtime, decay applies `rate ** cycles` in one UPDATE instead of N runs; cycles clamped. | `settings.runtime_cycles_cap` | 96 | YES |
| Runtime lease / STANDBY mode | `src/workers/runtime.py:130`, `:630`, `:646` | E7 D6 | A second core process (app + `ice-mcp`) starts in standby: its own event jobs run, but periodic dispatch, the burst and lease stamping stay with the owner. Promotes itself when the lease goes stale. | `settings.runtime_lease_ttl_seconds` | 180.0 | YES |
| Session-gap work unit → cluster this sitting now | `src/workers/runtime.py:372`; fired at `src/api/main.py:342` | C7 / C6 | A >30-min silence opens a new sitting; the runtime immediately freshens that conversation's clusters and forces an overdue pass. | `settings.session_gap_minutes` | 30 | YES |
| Commit work unit (coding) | registered `src/api/core.py:95`; fired `src/services/projects.py:205` | E3 | A git commit triggers `project_reconcile`. | — | — | YES (when a project is registered) |
| `task_done` work unit | `src/workers/runtime.py:225` (comment) | E4 | Reserved name; no handler registered anywhere. | — | — | **NO — unregistered** |
| Session-end burst (the quartet) | `src/workers/runtime.py:563` | D6/D8 + C4 | When a sitting has been over for `session_gap_minutes`, run reflection + batch_summarize + maintenance_agent + conversation_summary once for that sitting. | `settings.session_gap_minutes` | 30 | YES |
| Consent-gated fine-tune proposal | `src/workers/runtime.py:595` | D6/H5 | Enough curated labels + session end ⇒ either run the fine-tune (if `auto_finetune`) or leave ONE pending `finetune_proposal` in the review queue. | `settings.auto_finetune` / `settings.finetune_min_curated` | **False** / 20 | **NO** (proposal only) |
| Destructive-op guard (`generation_in_flight`) | `src/workers/runtime.py:290` | C10 | Conversation deletion refuses to run while a chat stream is live (post-flight would FK-fail into a deleted conversation). | — | — | YES |

---

### 2. Post-flight — density, representation, and the chained pipelines

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Post-flight turn evaluation | `src/workers/post_flight.py:136` | C1 / C7 | On every stored turn: measure density, decide raw vs summary, then chain chunking → codex → procedural → decisions. | — | — | YES |
| Key-term extraction (MUST-PRESERVE vocabulary) | `src/workers/turn_density.py:72` | C1 / A9b | Named entities (background NER) + figures + identifiers become the terms a summary must keep. | `settings.turn_max_must_terms` | 25 | YES |
| Entropy score (facts-per-token) | `src/workers/turn_density.py:108` | C1 | Blends entity density, figure density, code presence and lexical diversity into 0–1, stored on the turn. | — | — | YES |
| v3 Codex extraction eligibility | `src/workers/post_flight.py::evaluate_turn`, `src/workers/codex_extractor.py::extract_codex` | G57 | Every non-private turn reaches extraction, including short corrections. Lossless controls representation and batch summarization, not graph eligibility. Standalone extraction also checks privacy. | — | all non-private turns | YES |
| v3 extraction completion contract | `src/workers/extraction_result.py`, `codex_extractor.py::extract_codex` | G61 | Valid empty JSON completes; malformed, missing or truncated output raises for runtime retry. All chunks must succeed before graph and completion marker commit together. Emotion values and reflexive relations are no longer silently filtered. | — | strict complete output | YES |
| Representation decision matrix | `src/workers/turn_density.py:130` | C1 / ML4 | Documents and short turns skip the summary LLM call entirely; code and creative turns keep raw as the hint but store a summary; long turns let measured coverage decide. | `settings.turn_raw_keep_max_words` | 350 | YES |
| Grounded summarisation + one retry | `src/workers/post_flight.py:105`, `turn_density.py:183` | C1 / C3 / G29 | The must-preserve terms are injected into the prompt; if the summary drops terms, one retry names them, and the better-scoring result wins. | — | — | YES |
| Coverage trust gate | `src/workers/post_flight.py:228` | C1 | A summary is only allowed to be the injected form if it measurably preserved the key terms; otherwise raw wins and the summary stays as metadata. | `settings.turn_summary_coverage_threshold` | 0.7 | YES |
| One-line abstract (hierarchy level 3) | `src/workers/post_flight.py`, `src/memory/representation.py` | C3/G75 | Generated and stored; read-side budget use requires a verbatim source span and preserved query terms until independent verification exists. Summary coverage cannot qualify it. | — | — | YES, source-extractive only |
| Document-section handling | `src/workers/post_flight.py::evaluate_turn` | C12 | Document sections retain lossless representation eligibility and use a document-specific summary prompt. Graph extraction no longer depends on density. | — | — | YES |
| Incognito skip of derivative pipelines | `src/workers/post_flight.py:267` | G16 | A private turn keeps its own summary/lossless evaluation but never feeds codex, procedural, clustering, batch summary or reflection. | — (per-conversation `memory_scope_type='none'`) | — | YES |
| Long-turn chunking trigger | `src/workers/post_flight.py:260` | C2/C3 | A document turn or a turn over the word threshold is chunked for chunk-level retrieval. | `settings.turn_long_turn_chunk_words` | 600 | YES |
| Project-turn decision extraction | `src/workers/post_flight.py:281` | E8 | Project-attached turns also run cue-gated decision extraction inline. | — | — | YES (only for project-attached conversations) |
| Namespaced job idempotency keys | `src/workers/idempotency.py:24` | G29 | One shared `idempotency_keys` table; keys are `sha256("<job>:<batch>")` so one job's marker cannot make another skip work it never did. | — | — | YES |

---

### 3. Codex extraction — the knowledge graph write path

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Triplet extraction over chunks | `src/workers/codex_extractor.py:654` | A1 | Splits the turn into ~550-token sentence/code-aware chunks and asks the bg model for subject-relation-object triplets per chunk. | `settings.chunk_tokens` / `chunk_overlap_words` | 550 / 50 | YES |
| Constrained decoding — JSON shape | `src/workers/codex_extractor.py:773` | G32/a1 | Forces the reply into the triplet array schema; removes malformed-by-confusion output. | `settings.codex_constrain_shape` | True | YES |
| Constrained decoding — relation ENUM | `src/workers/codex_extractor.py:775` | G32/a1, Z2 | Would additionally force `relation` onto the 197-word list. Measured: keeps 100% of triplets but ~78% of rescued ones are wrong. | `settings.codex_constrain_relation_enum` | **False** | **NO** |
| Extraction output budget | `src/api/config.py` `codex_extraction_max_tokens` | G32/a1 → **G68** | Raised 500 → 1200 because truncation silently lost whole turns' extraction, then **1200 → 3000 on 2026-08-25 because it was still happening**: template mode lost **30 of 60 turns** at 1200 (3 at 3000), and `bg_model_output_truncated content_chars=3648` was logged on **qwen** in the same sweep. ⚠ Couples to the timeout (`base × clamp(max_tokens/500,1,6)`): per-call ceiling moved 72s → 180s. ⚠ Every published ICE graph number was produced at 1200 — reproduce with `CODEX_EXTRACTION_MAX_TOKENS=1200`. | `settings.codex_extraction_max_tokens` | **3000** | YES |
| **Extraction prompt shape** | `src/workers/codex_extractor.py` `extract_triplets()` | **G63/P1** | `instruct` sends the nine numbered rules + worked examples (built for a generalist that does not know the task); `template` sends a JSON schema to FILL and nothing else (what NuExtract-class specialists are trained on), strips `</think>`, accepts a `{"facts": […]}` envelope, and skips the JSON-schema constraint which fights a template model. **Measured: qwen+instruct 15% correct / 30% reversed; NuExtract3+template 63% / 0%; each model is garbage in the other's shape.** | `codex_extraction_mode` | **`instruct`** | **NO — `template` is opt-in.** Default preserves reproducibility of every published number |
| **Codex extraction model (its own pin)** | `src/api/config.py`, `src/workers/codex_extractor.py:959` | **G63** | Extraction resolves `model_override or settings.codex_extraction_model or get_bg_model_name()`. Exists because ONE background pin served **ten** jobs and only extraction wants a specialist — pinning the shared setting to NuExtract3 would hand a JSON-template filler to the summariser and the cluster namer, which fails silently. ⚠ **NOT the codex conflict reconciler** (`make_llm_reconciler`), which reasons and stays on the general model. ⚠ Moves with `codex_extraction_mode` — each model is usable only in its own prompt shape. | `codex_extraction_model` | `hf.co/numind/NuExtract3-GGUF:Q8_0` | YES |
| **Background model release after drain** | `src/workers/bg_client_factory.py` `release_bg_model()`, `src/workers/runtime.py` `_release_bg_model_once()` | **G32(a), small half** | ICE unloads its own background model (`POST /api/generate {"keep_alive": 0}`) once the maintenance queue is genuinely empty — nothing overdue, running or queued. Guarded by a flag so it fires once per idle stretch, cleared the moment any job starts. **Because `/v1/chat/completions` silently drops `keep_alive`**, ICE otherwise inherits the host's `OLLAMA_KEEP_ALIVE`; measured 2026-08-27, three background models sat resident at ~14 GB of a 24 GB card, `UNTIL: Forever`, for jobs finished hours earlier. ⚠ Per-REQUEST keep_alive still needs the native chat endpoint — not done. Verified live: `bg_model_released`, model gone from `ollama ps`. | `bg_release_after_drain`, `bg_release_timeout_seconds` | `True`, `10.0` | YES |
| **Faithfulness-tuned summariser prompt** | `src/workers/post_flight.py` `_summary_llm_call()` | **BG_LAYER_FIXES §1.1** | The must-preserve block and its retry twin ask the model to preserve terms *the turn supports* and omit the rest, instead of ordering every term verbatim. Measured on `gemma4:e4b`, n=70: fabricated **15.7% → 2.9%**, faithful **82.9% → 95.7%**, incomplete unchanged. ⚠ **MODEL-COUPLED** — the identical change on `qwen3:4b-instruct` was a LOSS (54% → 34% faithful). Changing the background model means re-measuring this prompt. ⚠ Expect lower `summary_coverage` (0.784 → 0.640); that is the metric noticing the model stopped padding, not a regression. | *(no setting — prompt text)* | — | YES |
| **Adaptive extraction chunking** | `src/workers/codex_extractor.py` `extract_triplets()` | **G68/P3** | Sizes the extraction chunk from `serving_window()` (the same probe the request path budgets against) minus prompt and output budget, clamped by a ceiling; falls back to `chunk_tokens` **with a WARNING** when the probe fails. Replaces a fixed 550 that split a 1,178-token turn into three. ⚠ **The ceiling is deliberate, not a formality:** every NuExtract3 measurement is on 150–1,500-token turns and **8k+ is untested**. Measured over 71,256 turns: at 550 only **23% of tokens** fit whole; at 4096, 66%; at 8192, 78%. | `codex_extraction_chunk_adaptive`, `codex_extraction_chunk_max` | `True`, **4096** | YES |
| Open relation vocabulary | `src/workers/codex_extractor.py:566` | G45 | A relation not on the 197-word list is no longer discarded — it is reused if close enough to a relation the graph already holds, else accepted as new. | `settings.codex_relation_open_vocabulary` | True | YES |
| Relation canonicalisation threshold | `src/workers/codex_extractor.py` `canonical_relation()` | G45 | Cosine above which an incoming relation is folded onto an existing one. Calibrated on 30 hand-built pairs; **now also measured on the real 700-relation vocabulary (2026-08-25)**: 76 pairs at or above 0.90, **62 blocked** by `_is_inverse_pair`, 14 allowed, **3 unsafe (~4%)**. The uncovered classes are **modals** (`prioritises`/`can_prioritize` 0.9053) and **comparatives** (`good`/`higher` 0.9043) — structural gaps like the polarity rule, **not a threshold to retune**: all three sit among legitimate merges in 0.904–0.912. | `settings.codex_relation_canonical_threshold` | **0.90** | YES |
| Converse/passive merge guard | `src/workers/codex_extractor.py:463`, `:492`, `:1229` | G45 | Deterministically refuses to merge `parent_of`/`child_of`, `before`/`after`, `built`/`built_by` — similarity cannot tell them apart and a merge writes the fact backwards. | — (curated `_ANTONYM_PAIRS`) | — | YES |
| Harvested relation seed | `src/workers/codex_extractor.py:502` | G45 | Loads 111 starter relations from `data/relation_seed.json` so a fresh store has something to converge onto. A seed, never a gate. | — | — | YES |
| Clausal-relation demotion | `src/workers/codex_extractor.py:549`, `:879` | G49 | A relation longer than 5 underscore-words is a sentence fragment, not a predicate; the edge is written at rejected confidence rather than dropped. | `settings.codex_relation_max_words` | 5 | YES |
| Relation-gap ledger | `src/workers/codex_extractor.py:1660`, `models.py:643` | G32/a1 | Triplets whose relation still cannot be mapped are persisted with subject/object so the missing vocabulary can be ranked and later replayed with no new LLM call. | — | — | YES |
| NER grounding → extraction confidence | `src/workers/codex_extractor.py:330`, `:856` | A2/A3 | Triplets naming NER-confirmed entities are trusted high; ungrounded ones enter the graph at low confidence instead of being deleted. | `settings.codex_conf_grounded` / `_ungrounded` / `_rejected` | 0.9 / 0.7 / 0.35 | YES |
| Property-value source check | `src/workers/codex_extractor.py:373` | G43 | A property relation's object is a value, not an entity — it must literally occur in the source turn, or it is marked rejected. Closes the hole that let `november --eye_color--> golden black` in. | — | — | YES |
| Unusable-entity-name refusal | `src/workers/codex_extractor.py:935`, `:1409` | G44 | Refuses to make a graph node out of a pronoun, article or pure punctuation. Numbers are explicitly allowed again (the earlier letter-requirement deleted 94 true numeric subjects in one seed). | — (closed `_NON_REFERRING` set) | — | YES |
| Node promotion (generic → specific) | `src/workers/codex_extractor.py:999` | G44 | When a more specific name arrives for a stored generic stub, the generic becomes an alias of the specific one. Only zero-edge stubs are eligible. | `settings.codex_node_promotion` / `codex_node_promotion_max_degree` | **False** / 0 | **NO** |
| **Write-time `merge_key` tier** | `src/workers/codex_extractor.py:986` | **G50** | Third resolution tier after exact-name and alias: a name whose normalisation key matches a live entity resolves to it instead of minting a row. Stops `gemma-4-e4b q4` / `gemma-4-e4b-q4` becoming two entities. Key stamped at creation and on promoted nodes; migration `a1c4e7b90d22` backfilled 8,280 rows + partial index. | — (no flag) | — | **YES — always on** |
| **Typed rejection of merge candidates** | `src/workers/maintenance_agent.py:275` | **G50** | Every duplicate candidate is typed by `difference_kind()` before queueing. A pair differing by a digit, spelled-out number, gender word, tense word or token permutation is dropped with a journalled `merge_rejected` event and never shown to a model or a human. Measured on the live store: 51 merge / 59 defer / 46 rejected. | — (no flag) | — | **YES — always on** |
| **⚠ Timeline leg depends on the CODEX leg** | `src/retrieval/orchestrator.py` (timeline leg), via `_codex_graph` | — | Undocumented coupling found 2026-08-20 by ablation: with codex FRAGMENTS dropped, timeline fragments still supply **30 of 75** codex-probe anchors; with the codex LEG disabled (`ConfigurableOrchestrator(codex=False)`), that falls to **0**. Timeline cannot function without the codex graph, so any decision to retire or gate codex silently takes timeline with it. | — | — | YES (and load-bearing for timeline) |
| **Codex-grounding NER tier** | `src/workers/codex_extractor.py` (NER call in `extract_triplets`) | A9b → **G63/P4** | Picks which tagger confirms the entities extraction may relate: `preflight` (micro-NER) or `background` (NuNER Zero). ⚑ **FLIPPED TO `background` 2026-08-25 — this is the settlement A9b deferred.** Over 60 turns, both tiers on the same turns: **junk entity names 8.7% (NuNER) vs 19.5% (micro)**, capitalised subjects 54.0% vs 68.7%. micro produced more facts (838 vs 748) and "grounded" more (53.7% vs 38.2%) — but **grounding is scored against the tagger's own list, so a looser tagger flatters itself**; junk-name rate is the tier-independent number. Also aligns the default with reality: every measured arm set `background` by env var. ⚠ **Decided on SHAPE — no blind round has compared the tiers for TRUTH.** ⚠ The tiers are not interchangeable: the whitelist is a constraint, so the narrower tagger moves triplets from `codex_conf_grounded` to `codex_conf_rejected` rather than deleting them. Old behaviour: `CODEX_EXTRACTION_NER_TIER=preflight`. | `codex_extraction_ner_tier` | **`background`** | YES |
| **Extraction entity-shape prompt rule** | `src/workers/codex_extractor.py:705` | — | Adds a rule telling the extractor a subject/object must be a noun phrase, never a clause or verb phrase. **Measured 2026-08-17 and it does not work**: superset rate 0.0858 → 0.1166, triplets −16%. Kept behind the flag as the record of a tested negative. | `codex_extraction_entity_shape_rule` | **False** | **NO — DEAD/INERT by default, deliberately** |
| **Token-budgeted summary batching** | `src/workers/batch_summarizer.py:31`, `:49` | — | Batches turns by a TOKEN BUDGET (`ollama_num_ctx_max` − `batch_summary_max_tokens` − prompt overhead, ÷ safety margin ≈ **26,200 content tokens**) instead of a fixed 50 turns. The fixed count sent **37,359 tokens at a 32,768 window** on 2026-08-17 — a hard 400. Always advances ≥1 turn, so an oversized single turn becomes a batch of one (then skipped by the `<5` floor) rather than an infinite loop. | `ollama_num_ctx_max` / `batch_summary_max_tokens` / `token_count_safety_margin` | 32768 / 1200 / 1.20 | **YES — always on** |
| **Per-batch summariser isolation** | `src/workers/batch_summarizer.py:126` | — | One batch's failure is rolled back, logged at **WARNING with its turn and token count, every time**, and the loop continues. Previously a single `try` wrapped the whole pass and re-raised, so the first oversized batch killed two later batches that would have fitted — the arm finished with **0** summaries and 40 `summary_synthesis` probes were unscoreable. The outer `try` still re-raises genuinely fatal errors. | — (no flag) | — | **YES — always on** |
| **Batch-summary output ceiling** | `src/workers/batch_summarizer.py:92` | — | Max tokens for one batch summary. Was hardcoded 500, which truncated 2 of 3 summaries mid-sentence while `bg_model_output_truncated` logged it and nothing acted. | `batch_summary_max_tokens` | **1200** | YES |
| **CoE gateway credentials** | `src/api/config.py:557` | — | Endpoint/key/model for the TCET campus Qwen3.6-35B-A3B gateway. Declared so `.env` validates; **no code path reads them yet** — reserved for model-judgement calls that would otherwise need a local model spun up. See [MODELS.md](MODELS.md) §3. | `coe_api_key` / `coe_api_base_url` / `coe_model` | `""` / `""` / `qwen3.6` | **NO — declared, unused** |
| In-flight-entity protection during promotion | `src/workers/codex_extractor.py:1423` | G44 | The subject of a triplet being written is exempt from being promoted away mid-write — otherwise the edge points at a deleted row and the whole turn's extraction is lost on the FK. | — | — | YES (when promotion is on) |
| Self-referential + suspicious-object filters | `src/workers/codex_extractor.py:840`, `:847` | A1 | Drops `fastapi uses fastapi` and objects that are bare verbs (`blush`, `laugh`). | — (hardcoded set) | — | YES |
| Property relations write JSONB + expire old edge | `src/workers/codex_extractor.py:1465` | — | `role`, `age`, `email` etc. update the entity's `properties` and supersede the previous value's edge. | — | — | YES |
| Uniform assertion writer | `codex_extractor.handle_triplet` | G76 | Property, ordinary and negative assertions share source reconciliation; category cannot expire an old fact. New edges pending/strength1; distinct batches promote independently of reads. | — | — | YES |
| A8 negation | `src/workers/codex_extractor.py:1430` | A8 | "X no longer uses Y" retracts the positive edge and stores a negated edge — a stored negative fact, not a navigable link. | — | — | YES |
| Conflict candidates | `codex_extractor.conflict_candidates` | G76 | All same-relation target/polarity differences plus known oppositions; no correction lexicon or arbitrary first candidate. Candidates alone authorize no expiry. | — | — | YES |
| Source authority for reconciliation | `_reconciliation_evidence` | G76 | Current source hashes, same conversation/role, original timestamps strictly ordered; complete role units supplied. Unknown or older input remains reviewable evidence. | — | — | YES |
| Bounded LLM reconciler for ambiguous supersession | `src/workers/codex_extractor.py:1365` | A6 | One-word verdict (`expire_old`/`keep_both`/`reject_new`); anything else queues for human review rather than guessing. | — | — | YES |
| Bidirectional context payload regeneration | `src/workers/codex_extractor.py:1154`, `:1656` | A7/G33 | Rebuilds an entity's Obsidian-style note (description + properties + Links + Backlinks + Negations) on BOTH ends of every edge write. The `db.flush()` is load-bearing — without it every payload was one write behind. | — | — | YES |
| Entity-type inference with direction | `src/workers/codex_extractor.py:1111` | A7/G33 | Types an entity from its OUTGOING relations; incoming edges vote only for symmetric relations, so `fire mage` is no longer typed `person`. | — | — | YES |
| Deterministic UUIDv5 entity ids | `src/workers/codex_extractor.py:285` | — | Same canonical name always gets the same id across machines/imports. | — | — | YES |
| Bi-temporal mirror listener | `src/memory/models.py:708` | G45 §5 | Any ORM path that sets `valid_until` automatically sets `unlearned_at`, so "when did the fact stop being true" and "when did ICE stop believing it" can never diverge by omission. **Bypassed by bulk `query.update()` / raw SQL.** | — | — | YES |
| Entity merge (graph surgery) | `src/workers/codex_ops.py:38` | D5 | Re-points every edge (live and expired), unions aliases/tags/properties, keeps the longer description, moves the journal, and leaves the absorbed row as a renamed husk. Nothing is hard-deleted. | — | — | YES (called by agent Tier 0 and by review approval) |

---

### 4. Procedural memory

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Session-scoped habit extraction | `src/workers/procedural_extractor.py:61` | G42 | Reads a whole sitting of the USER's own prompts (not one turn, not the assistant's words) and asks for one repeated habit. | `settings.procedural_min_session_turns` | 3 | YES |
| Re-extract in steps as a session grows | `src/workers/procedural_extractor.py:117` | G42 | Buckets by `n_turns // step` so a 20-turn session costs 5 model calls, not 20. | `settings.procedural_session_step` | 5 | YES |
| Prompt-block cap, trimmed from the front | `src/workers/procedural_extractor.py:137` | G42 | The most recent behaviour survives the character cap. | `settings.procedural_max_prompt_chars` | 12000 | YES |
| Two-message evidence requirement | `src/workers/procedural_extractor.py:187` | G42 | A pattern citing fewer than two numbered messages is rejected outright — enforced in code, not asked for in the prompt. | — | — | YES |
| Cross-session reinforcement only | `src/workers/procedural_extractor.py:216` | G42 | Re-reading the same sitting cannot reinforce a pattern against itself; `source_batch_ids` must be disjoint. | — | — | YES |
| Pattern dedupe by embedding similarity | `src/workers/procedural_extractor.py:207` | G42 | Above the threshold a new extraction counts as the same habit. Measured 2026-08-13: two phrasings of the same habit score 0.708, so the default probably misses real repeats. | `settings.procedural_similarity_threshold` | 0.85 | YES |
| Promotion to active | `src/workers/procedural_extractor.py:219` | — | 3 cross-session reinforcements ⇒ confidence 0.8 and `is_active=True` (the state the retrieval leg requires). | — (hardcoded 3 / 0.8) | — | YES |
| Project-scoped patterns | `src/workers/procedural_extractor.py:123` | E1 D1 | A habit observed in a project-attached conversation is stored with `project_id` — coding conventions are not a fourth store. | — | — | YES |
| Staleness retirement | `src/workers/procedural_decay.py:14` | — | A pattern unobserved for 180 days with fewer than 3 reinforcements is deactivated. | `settings.procedural_stale_days` / `procedural_min_reinforcement` | 180 / 3 | YES |

---

### 5. Decay, archival and cold storage

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Access-weighted episodic decay | `src/workers/decay.py:97`–`132` | — | Three UPDATE classes: unaccessed, accessed, creative. Turns younger than 7 days are never touched. | `decay_daily_unaccessed` / `_accessed` / `_creative` | 0.95 / 0.98 / 0.99 per day | YES |
| Per-cycle rate DERIVED from the cadence | `src/workers/decay.py:32`, `:56` | G9 | The daily target is the setting; the per-cycle multiplier is computed as `daily ** (1/(86400/cadence))`, so changing the cadence preserves the target. Deliberately not a knob. | (derived from `maintenance_intervals`) | — | YES |
| Creative floor | `src/workers/decay.py:135` | — | Creative turns never decay below this — they are re-read in bursts months apart. | `settings.decay_creative_floor` | 0.3 | YES |
| Bookmark / permanent immunity | `src/workers/decay.py:100` etc. | — | `decay_immune=TRUE` or `is_bookmarked=TRUE` skips all three decay UPDATEs. | — | — | YES |
| Self-expiring import immunity window | `src/workers/decay.py:102`; column `models.py:116` | F10 | `decay_immune_until` protects imported history for a window and needs no sweeper — the filter re-admits the row when the window lapses. | `import_immune_window_days` | 14 | YES |
| Archive at threshold | `src/workers/decay.py:153` | — | Below 0.1 a turn is archived (excluded from normal retrieval). | `settings.decay_archive_threshold` | 0.1 | YES |
| Un-archive on recovery | `src/workers/decay.py:146` | T3 D11 | A row whose score recovered above the archive line comes back — symmetric with archiving, and the automatic probation reversal for resurrected rows. | — | — | YES |
| Archived rows keep decaying | `src/workers/decay.py:86` (comment) | T3 D11 | The three UPDATEs no longer filter `is_archived=FALSE`; before the fix an archived score froze at ~0.1 and cold storage was unreachable. | — | — | YES |
| Move to cold storage | `src/workers/decay.py:163` | T3 D12 | Below 0.05 the row is copied to `cold_storage` (with conversation_id, is_private, batch_id, embedding, source_spans and ts_provenance) and deleted from `episodic_memory`. | `settings.decay_cold_threshold` | 0.05 | YES |
| Codex retention aging | `src/workers/codex_decay.py:28` | A3 / E1b D3 | Live conversation edges decay to a retention floor; nonuse never demotes confidence or expires validity. Derived edges remain exempt. | `codex_decay_daily` / `codex_retention_floor` | 0.99 / 0.1 | YES |
| Journal compaction (snapshots) | `src/workers/compaction.py:15` | G10 | An entity with ≥100 uncompacted events gets a state snapshot; events are marked `compacted`, never deleted. | `settings.compaction_event_threshold` | 100 | YES |

---

### 6. Clustering (C5 v5)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Exclusive assignment | `src/workers/clustering.py:494` | C5 v5 | A turn joins only its single best-scoring cluster — multi-assignment plus mean centroids was the mega-cluster feedback loop. | `settings.cluster_similarity_threshold` | 0.6 | YES |
| Wait-for-a-friend cluster creation | `src/workers/clustering.py:514` | C5 v5 | An unmatched turn waits; clusters are born only from ≥2 mutually similar waiting turns (union-find), so a new cluster starts with two turns of evidence. | — | — | YES |
| Singleton age-out | `src/workers/clustering.py:541` | C5 v5 | A turn that waits longer than 24 h gets its own cluster — one-off topics are real. | `settings.cluster_singleton_age_hours` | 24.0 | YES |
| Session-affinity bonus | `src/workers/clustering.py:322` | C6 payoff | Sharing a sitting with a cluster's members adds a bonus. | `settings.cluster_session_affinity_bonus` | 0.10 | YES |
| Entity-overlap bonus (capped) | `src/workers/clustering.py:320` | C5 | Shared NER entities bridge low bulk-text cosine between scenes of one story. | `cluster_entity_overlap_bonus_per_shared` / `_cap` | 0.08 / 0.30 | YES |
| Tag-overlap bonus (capped) | `src/workers/clustering.py:318` | C5 v5 | Capped because within one conversation tags are near-uniform and uncapped this was a constant offset. | `cluster_tag_overlap_bonus` / `_cap` | 0.05 / 0.10 | YES |
| Background-tier NER for cluster entities | `src/workers/clustering.py:105` | A9b | Uses the background NER model (28× better conversation separation than the micro-NER, which tags `and`/`but` as entities). | `background_ner_model` / `cluster_entity_extraction_max_chars` | `numind/NuNER_Zero` / 2000 | YES |
| NER-grounded cluster naming | `src/workers/clustering.py:152`, `:197` | C5 v5 | Name/description prompts receive the entities that actually recur across ≥2 member turns. | — | — | YES |
| Name/description regeneration cadence | `src/workers/clustering.py:401` | C5 | Regenerated every 5th member, not on every assignment. | `settings.cluster_name_regen_interval` | 5 | YES |
| Centroid renormalisation | `src/workers/clustering.py:250` | C5 (v4 carry) | An average of unit vectors is not unit length; without this, centroid scores deflate as clusters grow. | — | — | YES |
| Deterministic oldest-first queue | `src/workers/clustering.py:451` | C5 v5 | The unassigned query had LIMIT without ORDER BY; oldest-first is also what makes the age-out work. | `settings.cluster_max_turns_per_run` | 25 | YES |
| Singleton re-absorption (repair pass) | `src/workers/clustering.py:590` | C5 v5 / G10 | A one-turn cluster whose member scores above the assignment bar against a sibling is folded in and deleted — heals singletons already in the DB. | `cluster_similarity_threshold` | 0.6 | YES |
| Pairwise cluster merge | `src/workers/clustering.py:641` | C5 | Two gates: raw cosine floor, then adjusted score with a tight entity-bonus cap. Merging is permanent, hence the higher bar. | `cluster_merge_min_raw_sim` / `cluster_merge_similarity_threshold` / `cluster_merge_entity_bonus_cap` | 0.82 / 0.90 / 0.10 | YES |
| Private-turn exclusion | `src/workers/clustering.py:446` | G16 | Incognito turns are never clustered. | — | — | YES |

---

### 7. Summarisation layers

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Batch summarisation of old turns | `src/workers/batch_summarizer.py:24` | G11 | Compresses 5–50 old turns per conversation into one summary row with its own embedding. | — | — | YES |
| Age-OR-decay selection | `src/workers/batch_summarizer.py:53` | G11 | A turn qualifies at `decay_score < 0.3` OR simply being older than the age cutoff; decay alone left old-but-accessed turns in long conversations uncompressed forever. | `settings.batch_summary_age_days` (and hardcoded 0.3) | 30 | YES |
| Coverage stamp (`batch_summary_id`) | `src/workers/batch_summarizer.py:115` | G11 | Marks exactly the turns a summary covers, in the same transaction. Without it every cadence pass re-summarised the same turns and injected duplicates. | — | — | YES |
| Lossless/document/private exclusions | `src/workers/batch_summarizer.py:51`–`56` | G11/G16 | Lossless turns, documents and incognito turns are never batch-summarised. | — | — | YES |
| Evolving whole-conversation summary | `src/workers/conversation_summary.py:130` | C4 | ONE summary per conversation, folded incrementally from the new turns since `covers_through`. Quiet conversations are a no-op. | — | — | YES |
| Window gate on creation | `src/workers/conversation_summary.py:171` | C4 D3a | A summary row is created only once the conversation outgrows the recent-turns window — a 2-turn chat never earns one. | (via `memory_decision.estimate_recent_window_tokens`) | — | YES |
| Bounded folding chunks | `src/workers/conversation_summary.py:195` | C4 | New turns are folded in ~3500-word bites, each turn's contribution capped so one huge turn cannot dominate. | `conversation_summary_chunk_words` / `_per_turn_words` / `_max_words` | 3500 / 400 / 250 | YES |
| Never half-advance | `src/workers/conversation_summary.py:206` | C4 | A failed LLM call rolls back that conversation and leaves the old row intact for the next burst. | — | — | YES |
| Grounded fold with coverage retry | `src/workers/conversation_summary.py:110` | C4/G29 | Same must-preserve-terms + one-retry machinery as post_flight, shared via `turn_density.retry_on_coverage_miss`. | `settings.turn_summary_coverage_threshold` | 0.7 | YES |
| Incognito conversations DO get summaries | `src/workers/conversation_summary.py` module docstring; query at `:146` has no privacy filter | C4/G16 | Deliberate: their own context. The retrieval consumer's join is the privacy shield. | — | — | YES |

---

### 8. Reflection (the five-part pass)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Reflection driver | `src/workers/reflection.py:98` | — | For each of up to 200 recent conversations with ≥10 non-private turns: synthesis, crystallisation, slot evolution, motif detection. Then one global codex enrichment pass. | — (hardcoded 200 / 10) | — | YES |
| Session synthesis | `src/workers/reflection.py:139` | — | Writes a `SessionSummary` row (topics, decisions, unresolved items, entities, patterns) via a schema-constrained JSON call. | — | — | YES |
| Pending-items slot append | `src/workers/reflection.py:176` | — | Unresolved items are appended to the `pending_items` memory slot. **[queried] `memory_slots` is empty, so this is a no-op today.** | — | — | YES (but inert, see DEAD) |
| Pattern crystallisation | `src/workers/reflection.py:190` | — | A SECOND writer of `procedural_memory`, from conversation snippets, with a **hardcoded 0.85** similarity threshold and no evidence requirement. | — (hardcoded, ignores `procedural_similarity_threshold`) | 0.85 | YES |
| Memory-slot evolution proposals | `src/workers/reflection.py:246` | — | Proposes updates to `project_context` / `user_preferences` / `guidance` into the review queue — never writes a slot directly. | — | — | YES |
| Codex entity enrichment | `src/workers/reflection.py:281` | A7.3 | Fills empty entity `description`s first (richest-mentioned first), then refreshes stale well-mentioned ones, from the turns that mention them. | `reflection_enrich_limit` / `reflection_enrich_refresh_days` | 25 / 14 | YES |
| Description-update journalling | `src/workers/reflection.py:358` | T4 D13 | Every overwrite of the one mutable-in-place field leaves a `description_updated` event. | — | — | YES |
| Motif detection | `src/workers/reflection.py:366` | — | Proposes new cluster names into the review queue as `new_cluster_proposal`. | — | — | YES |

---

### 9. Maintenance agent — every detector, tier, cap and zero-condition

Driver: `run_maintenance_agent(db, llm_decider)` at `src/workers/maintenance_agent.py:720`. Detectors run in registry order (`:336`) under a SHARED scan budget: `detect_all` (`:347`) gives each detector `agent_max_scanned - len(items_so_far)` and **breaks entirely when that hits zero**.

**Global caps**

| Cap | Setting | Default | Effect |
|---|---|---|---|
| Items scanned per run (all detectors combined) | `settings.agent_max_scanned` | 50 | `detect_all` truncates to 50 and stops calling later detectors |
| LLM decisions per run | `settings.agent_max_llm_decisions` | 25 | Items needing a verdict past this return `skipped_cap` |
| Tier-2 proposals per run | `settings.agent_max_tier2_proposals` | 5 | A flooded review queue is worse than no agent |
| Auto-applications per run | `settings.agent_max_applications` | 10 | Bounds graph writes |
| Retry attempts per review item | `settings.agent_max_attempts` | 2 | An item the LLM was unsure about twice is never retried |

**Detector 1 — `reconciliation_leftover` (Tier 1, LLM)**

| Aspect | Detail |
|---|---|
| Where | `src/workers/maintenance_agent.py:59` |
| Finds | `review_queue` rows of type `codex_reconciliation`, status `pending`, with `agent_attempts < 2`, oldest first, LIMIT = remaining budget |
| Decides | Same source eligibility and complete two-source context as inline reconciliation; bounded input, exact enum. Missing authority returns unsure without a model call. |
| Applies | Rechecks source evidence before expiry, refreshes property projections and endpoint notes, marks resolved; unsure increments attempts. Manual review requires explicit keep_edge_ids. |
| Zero when | No pending `codex_reconciliation` rows, or all of them already have ≥2 agent attempts, or `llm_decider is None` (returns `skipped_no_llm`) |
| **[queried]** | 19 pending + 50 resolved rows on the live store. **This detector alone returned 50 items on the last run and consumed the entire `agent_max_scanned` budget — detectors 2–6 never ran.** That is the direct cause of "scanned 50, zero entity merges proposed". |

**Detector 2 — `pending_pileup` (Tier 1, deterministic)**

| Aspect | Detail |
|---|---|
| Where | `src/workers/maintenance_agent.py:76` |
| Finds | Entities with **>3** live pending edges AND **>2** live active edges that share the SAME `(source, target, relation)` |
| Applies | `_apply_pileup` (`:569`) expires each pending edge that exactly duplicates a live active one; the active edge keeps the max extraction_confidence |
| Thresholds | `agent_pileup_min_pending` = 3, `agent_pileup_min_active_overlap` = 2 |
| Zero when | No entity has both a pending AND an active live edge for the same triple. **`handle_triplet` reinforces an existing edge for a given (source,target) rather than writing a second one, so the extraction path essentially cannot create this state.** |
| **[queried]** | 4,662 live pending edges, **0 entities match**. This detector is structurally near-dead on conversation-extracted graphs. |

**Detector 3 — `duplicate_entities` (Tier 0 auto-merge OR Tier 2 proposal)**

| Aspect | Detail |
|---|---|
| Where | `src/workers/maintenance_agent.py:104` |
| Cap | `min(remaining_budget, agent_dup_pairs_per_run=10)` |
| Channel A (Tier 0, auto-merge, no LLM) | Two DISTINCT entities sharing a casefolded canonical name or alias |
| Channel B (Tier 2, LLM verdict required) | pgvector cosine ≥ `agent_dup_cosine_threshold` on the entity-NAME embedding, same `entity_type`, both `source='conversation'`, neither already merged, ORDER BY cosine DESC LIMIT cap |
| Merge order | `_merge_order` (`:179`) keeps the entity with more `codex_events`; ties break on the older first event, then id |
| Skip-forever list | Every `review_queue` row of type `entity_merge` carrying a `pair_key`, **regardless of status** — a pair proposed, approved, rejected or auto-dismissed is never re-detected |
| Threshold | `agent_dup_cosine_threshold` = **0.90**, `agent_dup_pairs_per_run` = 10 |
| Zero when | (a) the scan budget was consumed by detectors 1–2; (b) no name/alias collisions; (c) no pair ≥0.90; (d) every candidate pair already has a `pair_key` row; (e) `llm_decider is None`; (f) the LLM answers `different`/`unsure` — only `same` proposes |
| **[queried]** | 7,949 entities, all with embeddings, all `source='conversation'`. **Name-overlap channel: exactly 0 colliding names.** `canonical_name` is UNIQUE and `get_or_create_entity` sets `aliases=[name]` where `name.lower() == canonical_name`, so two distinct rows can essentially never share a normalised name — **Tier 0 auto-merge is dead on conversation-extracted entities.** Cosine channel: **≥200 pairs above 0.90 exist** — candidates are plentiful; the detector simply never got to run. |

**Detector 4 — `contradiction` (Tier 2, source-required proposal)**

| Aspect | Detail |
|---|---|
| Where | `src/workers/maintenance_agent.py:197` |
| Arm (a) polarity | A live positive edge coexisting with a live NEGATED edge for the same `(source, relation, target)` — A8 residue the in-line retraction missed across batches |
| Arm (b) antonym | Two live edges for one entity pair whose relations are an opposition candidate pair (`_OPPOSITION_PAIRS`); converses excluded |
| Applies | `_apply_contradiction` records one pending `codex_contradiction` proposal per edge pair. No graph expiry; existing Tier-2 proposal cap applies. |
| Zero when | No cross-batch polarity residue and no antonym pair coexists live. The in-line A8 path expires the positive at write time, so arm (a) only ever catches cross-batch leakage |
| **[queried]** | Only 2 live negated edges in the whole graph — arm (a) has almost nothing to find |

**Detector 5 — `stale_slot` (Tier 2, LLM)**

| Aspect | Detail |
|---|---|
| Where | `src/workers/maintenance_agent.py:257` |
| Finds | The `pending_items` memory slot (a fixed slot NAME, not a knob) with non-empty content untouched for ≥14 days |
| Decides | LLM proposes updated slot content from the 5 most recent non-private turns; empty suggestion ⇒ nothing proposed |
| Blockers | Skipped while a `memory_slot_update` proposal for that slot is pending, or the user rejected one since the slot last changed |
| Threshold | `agent_stale_slot_days` = 14 |
| Zero when | The slot does not exist, is empty, was touched within 14 days, a proposal is pending, or a rejection postdates the slot's last change |
| **[queried]** | `memory_slots` has **0 rows** — this detector returns zero on the live store, always |

**Detector 6 — `stale_work` (Tier 2, deterministic — no LLM)**

| Aspect | Detail |
|---|---|
| Where | `src/workers/maintenance_agent.py:291` |
| Finds | `tasks` rows in status `pending`/`active` untouched for ≥14 days, with no pending/recently-rejected `stale_work` proposal |
| Applies | The payload IS the proposal (self-describing for the review UI) — branch + goal included so drift is visible |
| Threshold | `agent_stale_task_days` = 14 |
| Zero when | No registered projects/tasks. Also silently zero on any DB error — the whole query is wrapped in `try/except: rollback; return []` (`:320`) |
| **[queried]** | 0 projects, 0 tasks |

**Agent infrastructure**

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Bounded JSON decider | `src/workers/maintenance_agent.py:370` | D1 | One temperature-0 completion on the bg model, JSON-schema constrained, short output; any error or unparseable reply reads as `unsure` and nothing is written. | — | — | YES |
| Forced-enum answers with an escape | `src/workers/maintenance_agent.py:422` | D1/D2 | The enum always contains `unsure`, so a model with no good answer has a legal way to decline — the difference from the codex relation vocabulary, where forcing produced ~78% wrong facts. | — | — | YES |
| Graceful LLM-free degradation | `src/workers/maintenance_agent.py:724`, `:765` | D2 §4 | Passing `llm_decider=None` still runs Tier 0 and the deterministic Tier-1 items. | — | — | N-A (call-site) |
| Per-run journalling | `src/workers/maintenance_agent.py:728`, codex_ops | G17 / D4 | Every graph write carries `source: "maintenance_agent"` and `batch_source = agent_run_id`. | — | — | YES |
| LLM budget counts only real calls | `src/workers/maintenance_agent.py:735` | D8 | An item that resolves without the model does not consume the decision budget. | — | — | YES |
| Dismissed-pair record | `src/workers/maintenance_agent.py:689` | D2 | An LLM "different" verdict writes a `resolved` review row so the pair is never re-proposed — without user noise. | — | — | YES |

---

### 10. Fine-tuning

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Curated-label fine-tune | `src/workers/fine_tune.py:141` | B4/G1 | Retrains the MLP head from the CURRENT live checkpoint on user-curated labels (10 Adam epochs, lr 1e-4). | — | — | **NO (never cadence-run)** |
| Validated promotion | `src/workers/fine_tune.py:229` | B4/G1 | The candidate replaces the live checkpoint only if it beats it on a deterministic held-out split; otherwise the artifact is kept unpromoted. | `settings.finetune_val_fraction` | 0.2 | YES (when the job runs) |
| Row-count gate | `src/workers/fine_tune.py:176` | G9 | Below this many curated rows there is no held-out check and no promotion. One setting, one gate. | `settings.finetune_min_curated` | 20 | YES |
| Schema-driven, template-faithful encoding | `src/workers/fine_tune.py:94` | B1 D3 | Prompts are rendered through the SAME templates the live classifier serves with; a bare-text fine-tune would reintroduce the train/inference mismatch. | — | — | YES |
| v1-row context masking | `src/workers/fine_tune.py:74` | B1 | Rows labelled under schema v1 train the topic/intent heads and are masked out of the context head rather than fabricating a v2 target. | — | — | YES |
| Timestamped artifact always kept | `src/workers/fine_tune.py:217` | G31 | Every run writes `models/classifier/ice_classifier_finetuned_<ts>.pt`, anchored to the install root. | — | — | YES |

---

### 11. Decision memory (E8)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Deterministic cue gate | `src/workers/decision_extractor.py:64` | E8/A6 | No cue, no LLM call. Priority: constraint > incident > decision. Code blocks are stripped first. | — (hardcoded cue lists) | — | YES |
| Bounded JSON extraction | `src/workers/decision_extractor.py:81` | E8 | One call extracts decision / rationale / alternatives / files / type, with an explicit "reply empty if there is no real decision". | — | — | YES |
| Duplicate skip | `src/workers/decision_extractor.py:184` | E8 D7 | Same type + same files + very high similarity ⇒ nothing written. | `settings.decision_duplicate_threshold` | 0.95 | YES |
| Auto-supersede (Tier 1) | `src/workers/decision_extractor.py:191` | E8 D7 | Similar + overlapping files ⇒ the old row gets `valid_until` + `superseded_by`. Bi-temporal, no event journal. | `settings.decision_conflict_threshold` | 0.85 | YES |
| Conflict proposal (Tier 2) | `src/workers/decision_extractor.py:203` | E8 D7 | Similar but different files ⇒ new row plus a `decision_supersession` review item. | `decision_conflict_threshold` | 0.85 | YES |
| Single write door | `src/workers/decision_extractor.py:150` | G29 | `reconcile_and_insert` is the ONLY way a Decision row is written — the manual `/decision` path used to bypass it and write unconditional duplicates. | — | — | YES |

---

### 12. Background-model plumbing

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Shared-mode background client | `src/workers/bg_client_factory.py:153` | C7 D7 | Background work reuses the main Ollama server (no second model in VRAM). Dedicated mode points at `localhost:8002`. | `settings.background_model_mode` | `"shared"` | YES |
| Background model pin | `src/workers/bg_client_factory.py:169` | G27 | Unset ⇒ shared mode falls through to the first `confirmed` registry entry, which is NOT the chat model. **Warns on every call** because a 100%-firing fallback is an outage in costume. | `settings.background_model_name` | **None (unpinned)** | NO (unpinned by default) |
| Foreground/background identity separation | `src/workers/post_flight.py::generate_summary`, `src/workers/procedural_extractor.py::extract_procedural` | G27 | v3 foreground model_used remains provenance; turn summary and habit extraction use the background factory even when a cloud model answered. | `background_model_name` | None; existing unpinned warning remains | YES |
| Complete background text required | `src/workers/completion_text.py::complete_text` | G32/G73 | v3 turn/conversation/batch summaries and procedural extraction reject partial, empty or missing-finish output. No batch coverage/checkpoint advances from a length-limited prefix; turn summary keeps raw. | — | finish_reason=stop | YES |
| Reasoning suppression | `src/workers/bg_client_factory.py:23` | G32 | Adds `reasoning_effort="none"` to every bg completion. Without it a reasoning model spends the whole budget in a thinking block Ollama strips, and `content` comes back EMPTY — the entire background layer producing nothing. | `settings.bg_disable_reasoning` | True | YES |
| Empty-content and truncation warnings | `src/workers/bg_client_factory.py:69`, `:79` | G32(a) | Logs when a bg call returns empty content or `finish_reason == "length"` — the two silent ways background output dies. | — | — | YES |
| Schema-refusal retry, loudly | `src/workers/bg_client_factory.py:55`, `:120` | G32(a) | A server that rejects constrained decoding gets one unconstrained retry, at WARNING; timeouts/OOM are re-raised, never downgraded. | — | — | YES |
| JSON-schema constructor | `src/workers/bg_client_factory.py:136` | G32 | One constructor for `response_format`. Measured: `{"type":"json_object"}` scored 0/8 (honoured by nobody, still HTTP 200) while `json_schema` scored 8/8 on the same endpoint. | — | — | YES |
| Output-scaled timeouts | `src/workers/bg_client_factory.py:202` | G12 | `base × clamp(max_tokens/500, 1, 6)`; prefill-heavy callers add their own 60 s floor. | `settings.bg_timeout_base_seconds` | 30.0 | YES |
| Loud JSON salvage | `src/workers/llm_json.py:83`, `:88` | G29 | One place turns a model reply into JSON, returning None (never a silent `{}`/`[]`) and logging which failure mode hit. | — | — | YES |

---

### 13. Storage primitives — `src/memory/`

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| One ORM file, all models | `src/memory/models.py` | — | 30 tables incl. episodic + chunks, codex entities/edges/events/snapshots/relation-gaps, procedural, clusters, slots, documents, imports, cold storage, review queue, ledger, store_meta. | — | — | N-A |
| Process-singleton embedder | `src/memory/embedder.py:66` | G23 D6 | ONE `SentenceTransformer` shared by every writer and retrieval path (was five copies at truncate_dim 384). | `settings.embedding_model_name` / `embedding_dim` | `Qwen/Qwen3-Embedding-0.6B` / 1024 | YES |
| Embedder device selection | `src/memory/embedder.py:41` | C16 | `auto` puts the encoder on the GPU when one exists — measured 321 ms/encode on CPU vs 21 ms on GPU, paid on every chat turn. Costs ~1.2 GB VRAM. | `settings.embedding_device` | `"auto"` | YES |
| GPU-load fallback to CPU | `src/memory/embedder.py:90` | C16 | If the card is full the embedder loads on CPU and says so at WARNING rather than refusing to boot. | — | — | YES |
| MRL 384-prefix narrowing | `src/memory/embedder.py:108`, `:114` | A9a | The one legal narrowing, for the micro-NER and the v1 rollback checkpoint; any other width raises rather than feeding a head garbage. | — | — | YES |
| Shared chunker | `src/memory/chunking.py:119` | A1/C2 | Sentence/code-line atomic units, greedy packing to a token budget, overlap carried forward, headings as hard section boundaries, char-level hard split for minified blobs. | `chunk_tokens` / `chunk_overlap_words` | 550 / 50 | YES |
| Real tokenizer counting | `src/memory/tokens.py:83` | C16 | The one answer to "how many tokens is this?", using the embedder's own already-loaded tokenizer. Replaced four incompatible estimates, one of which undercounted ICE's own stamped format by 1.68×. | — | — | YES |
| Per-message envelope accounting | `src/memory/tokens.py:102` | C16 | Counts messages, not their concatenation — chat templates add ~4 tokens per message. | — (hardcoded 4) | — | YES |
| Cross-family safety margin | `src/memory/tokens.py:145` | C16 | Applied only where UNDER-counting costs correctness. Measured, not guessed (first live reconciliation: predicted 283 vs server 331). | `settings.token_count_safety_margin` | 1.20 | YES |
| Session (sitting) resolution | `src/memory/session.py:15` | C6 | A turn joins the previous sitting if the silence is ≤30 min, else mints a new `session_id`. Feeds clustering, the maintenance trigger, and procedural extraction. | `settings.session_gap_minutes` | 30 | YES |
| Embedding-identity stamp + boot guard | `src/memory/store_meta.py:98` | G23 D1 | `create_core()` refuses to boot when settings' embedder disagrees with what the store carries — cosine-comparing mixed-model vectors is the existential failure. Bootstraps a stamp only on a provably empty store. | `embedding_model_name` / `embedding_dim` | as above | YES |
| Pending-re-embed warning | `src/memory/store_meta.py:128` | G23 D1 | A matching stamp with unfinished re-embed tables boots WITH a loud warning — degraded recall, never a crash. | — | — | YES |
| Vector-column discovery from the catalog | `src/memory/store_meta.py:53` | G23 D4 | Never a hardcoded list; a new vector column announces itself. | — | — | YES |
| Resumable re-embed runner | `src/memory/reembed.py:232` | G23 D4 | Re-encodes every registered vector column in dependency order, per-table progress stamps, safe to kill and rerun. `context_clusters` centroids are recomputed last from fresh member vectors. | (`batch_size` arg) | 256 | N-A (manual/import-triggered) |
| Unregistered-vector hard error | `src/memory/reembed.py:126` | G23 D4 | A vector column with no source-text rule is a HARD ERROR naming the spec — a future table must state how its text is derived before it can ship. | — | — | YES |
| Husk-skipping re-embed | `src/memory/reembed.py:99` | G23/D5 | Merge husks, deletion husks and non-conversation entities are skipped — re-encoding a husk would resurrect a deleted entity into vector matching. | — | — | YES |
| Portable JSONL export | `src/memory/portability.py:72` | G23 D2 | One JSONL per table + manifest (alembic head, embedding identity, counts). Vectors excluded by default; `maintenance_ledger` always excluded (a foreign runtime lease could stall the target machine). | `with_vectors` arg | False | N-A (manual) |
| Staged state-copy import | `src/memory/portability.py:143` | G23 D2 | Manifest validation (alembic heads must MATCH exactly) → id-preserving inserts → re-embed pass (forced: inherited stamps describe the source store) → codex payload regeneration sweep. | `merge` arg | False | N-A (manual) |
| Self-referential FK backfill | `src/memory/portability.py:131` | G23 D2 | `sorted_tables` cannot order rows WITHIN a table, so self-FKs are stripped on insert and backfilled in a second pass. | — | — | YES |
| Full backup (pg_dump + models + config) | `src/memory/backup.py:132` | G23 D3 | One tar.gz with the dump, `models/`, `env.snapshot`, `backup_info.json` (alembic head, git commit, db size) and a RESTORE.md. Falls back to pg_dump inside the container when the host lacks it. | `--out-dir` / `--tag` | `backups/` | N-A (manual) |
| Implausibly-small-dump refusal | `src/memory/backup.py:76` | G23 D3 | A dump under 4 KB raises rather than being called a backup. | — | — | YES |

---

### DEAD OR INERT

1. **`src/workers/codex_inject_watcher.py` — entirely dead.** A watchdog-based `/codex_inject` YAML/JSON entity injector with its own `main()`. Evidence: `grep -rn codex_inject_watcher src scripts tests` returns **zero** references outside the file itself; it is not in `JOBS`, not imported anywhere, and nothing starts it. Same pattern as the v1 drop-zone watchdog C12 replaced with a cadence scan. It also writes entity ids with `uuid.uuid5(NAMESPACE_DNS, …)` while the extractor uses `CODEX_NAMESPACE` — the two would mint different ids for the same name.

2. **Maintenance-agent Tier 0 (auto-merge of normalisation-equal entities) cannot fire on conversation-extracted entities.** `codex_entities.canonical_name` is UNIQUE and stored lowercase, and `get_or_create_entity` (`codex_extractor.py:1049`) sets `aliases=[name]` where `name.lower()` IS the canonical name — so two distinct rows cannot share a normalised name unless something outside extraction writes an alias. **[queried]** 0 colliding names across 7,949 entities. Every duplicate therefore has to go through the Tier-2 LLM channel.

3. **`pending_pileup` (detector 2) is structurally near-dead.** It requires an entity with >3 live pending AND >2 live active edges that share `(source, target, relation)`. `handle_triplet` reinforces the existing edge for a given (source,target) pair instead of writing a duplicate, so extraction cannot produce that state. **[queried]** 4,662 live pending edges, 0 matching entities.

4. **`stale_slot` (detector 5) and reflection's `pending_items` append are both inert on the live store.** **[queried]** `memory_slots` has 0 rows. `reflection.py:179` looks the slot up and silently does nothing when it is absent; `_detect_stale_slot` returns `[]`.

5. **`stale_work` (detector 6) is inert and fails silently.** **[queried]** 0 projects, 0 tasks. Worse, the whole query is wrapped in `except Exception: db.rollback(); return []` (`maintenance_agent.py:320`) — a missing table or a schema error is indistinguishable from "nothing stale".

6. **`ProceduralMemory.trigger_conditions` is written as `{}` by both writers** (`procedural_extractor.py:236`, `reflection.py:229`) **and its only reader returns True when empty** (`orchestrator.py:2086`). The "trigger match" precision mechanism named in the C9 setting comment is therefore a no-op today.

7. **`CodexSnapshot` rows are write-only.** Written by `compaction.py:62`; `grep -rn CodexSnapshot src` finds no reader anywhere. Compaction marks events `compacted` (which nothing filters on either) and builds a state snapshot nothing consumes.

8. **`SessionReplay` has no writer in `src/`.** The model exists, `services/conversations.py` counts and deletes rows, tests create them — but nothing in production ever inserts one.

9. **`BatchSummary.start_turn_index` / `end_turn_index` are write-only by design** (documented at `models.py:559`): positions in a run-local filtered list, not turn identity. Coverage is `episodic_memory.batch_summary_id`.

10. **`TableRule.post_sql` has no user.** Documented at `reembed.py:74` — C12 dropped `rag_chunks`, the only NOT NULL vector column. Kept as a two-line seam.

11. **`is_gpu_busy()` always returns False in the default configuration** (`gpu_check.py:31`) because `background_model_mode` defaults to `"shared"`. `gpu_util_threshold` (70) and `gpu_check_cache_seconds` (10.0) are dead settings unless dedicated mode is enabled.
    - **⚑ WHY `dedicated` SURVIVES ANYWAY — settled 2026-08-17, do not re-litigate as dead code.** The mode conflates two independent things: *which model* does background work, and *which server* serves it. **`background_model_name` already decouples them** — pinning `qwen3:4b-instruct` gives a separate bg model on plain Ollama with no second server, and `config.py:43` says so outright ("That replaces the old vLLM side-server for most setups"). So `dedicated`'s only unique contribution is **a second process on :8002**, whose purpose — running bg work *alongside* live serving — is **contradicted by `yield_if_user_active()`** (`runtime.py:762`), which makes bg jobs *abandon* when the user returns. Two opposite strategies for one problem; the parallelism is never exercised.
      ⇒ **It is not dead code to delete — it is the Track F packaging seam.** A shipped app may have to bring its own model server rather than assume Ollama is installed, and this is that seam, already built. **The model half is live today** (`--bg-model` is what the two-arm seed varies); **the server half is a productisation decision** and belongs to Track F, not to any experiment. Both the app and every experiment currently reach the model through in-process function calls, so neither path has a live consumer for the distinction. Deleting it is USER-GATED and a deliberate Track F call.

12. **The `task_done` work-unit kind is reserved but unregistered** (`runtime.py:225`); calling `notify_work_unit("task_done", …)` logs `work_unit_unhandled` and does nothing. `commit` IS registered (`api/core.py:95`).

13. **`codex_node_promotion` is OFF by default** and, when on, only stub nodes with degree ≤ 0 are eligible — so the G44 "second half" ships inert.

14. **`codex_constrain_relation_enum` is OFF by default**; the enum wiring exists solely so Z2 can flip it and measure.

15. **`auto_finetune` is OFF by default** — the fine-tune job has no cadence entry at all, so the only path to it is a session-end burst with `auto_finetune=True`. Otherwise it leaves one pending review-queue proposal, forever (a second one is never created while the first is pending).

16. **`decay_strengthen_amount` (0.15) and `retrieval_strengthen_writes` are read on the retrieval side, not here** — but note `access_count` IS read by decay (`decay.py:104`, `:117`) to split the access classes, so it is not the pure dead column the config comment's "no reader in the retrieval path" wording might suggest.

---

### SURPRISES

**More complete than expected**

- **The runtime's yield mechanism is real and wired into three long jobs.** `JobYielded` is a distinct exception type specifically so a caller's `except Exception` cannot swallow it, a yield does not burn a retry, and the ledger records `yielded` as a non-failure status. **[queried]** the live ledger shows `conversation_summary` at status `yielded` — it has actually fired in production.
- **Runtime lease / standby arbitration across processes** is fully implemented, including self-promotion when the owner's lease goes stale and lease release on clean shutdown — a two-process (app + `ice-mcp`) story most projects would have left as a TODO.
- **The codex write path has SEVEN independent quality gates** stacked before an edge lands: relation canonicalisation, converse guard, clause-shape demotion, unusable-node refusal, NER grounding, property-value source check, and A6 conflict reconciliation. Each was added against a measured failure and each logs its own rate.
- **Grounded summarisation is measured, not hoped for.** Coverage is computed, one retry names the dropped terms, and the retry is kept ONLY if it scores better — and the shared helper exists because the two copies had already drifted.
- **The re-embed runner refuses to run on an unknown vector column** rather than skipping it. That is the rare case of a maintenance tool being fail-loud about its own coverage.
- **Portability's import refuses a mismatched alembic head outright** instead of "trying its best" — with the exact commands to fix it in the error message.

**Less complete than expected**

- **The maintenance agent's scan budget is shared and ordered, with no fairness.** `detect_all` hands each detector the *remaining* budget and breaks at zero, so a backlog in detector 1 starves detectors 2–6 completely. That is exactly what happened on the live store: 50 reconciliation leftovers ate the entire budget and the entity-merge detector — which had ≥200 valid candidates waiting — never executed. Nothing in the run summary says a detector was skipped; `outcomes` simply has no key for it.
- **Reflection is a SECOND, unregulated writer of `procedural_memory`** (`reflection.py:190`). It uses a hardcoded 0.85 similarity (ignoring `settings.procedural_similarity_threshold`), has no evidence requirement, and derives patterns from truncated 200-char snippets of the last 200 turns — i.e. it re-creates precisely the single-source fabrication G42 removed from `procedural_extractor`. It can also reinforce a pattern to `is_active=True` with no cross-session check.
- **`_enrich_codex_entities` selects by `codex_events` count with no privacy filter** — it reads `episodic_memory` by `batch_source` and does not exclude `is_private` turns, unlike every other reflection sub-pass (which filter `is_private = FALSE` at the query). Worth confirming before the final experiment.
- **The relation vocabulary is still 197 hardcoded words in the extractor's prompt** (`codex_extractor.py:78`–`263`) even though G45 made the vocabulary open — the prompt still leads with the closed list and the model is only *invited* to invent. The gate moved; the bias did not.
- **`extract_codex` still hashes its own idempotency key by hand** (`codex_extractor.py:1709`: `sha256(f"codex:{batch_id}")`) instead of using `idempotency.job_key` — the G29 consolidation reached post_flight and procedural_extractor but not this one. The value happens to match, so it is a latent inconsistency rather than a live bug.
- **`compaction`'s event-reconstruction only understands `edge_added` / `edge_expired`.** `edge_strengthened`, `property_updated`, `entity_merged` and `description_updated` events are marked `compacted` and contribute nothing to the snapshot.
- **Batch summarisation's `decay_score < 0.3` trigger is a hardcoded literal** (`batch_summarizer.py:53`) sitting beside the settings-driven age cutoff — G9 swept the age but not the decay half of the same OR.
- **`decide_representation` returns `inject_raw: True` on every branch** (`turn_density.py:130`–159); only the coverage gate in post_flight can flip it. So "store both, choose at read time" is real, but the write-time default is always raw.

---

## Interfaces and coding core — services, mcp, coding

Scope read in full: `src/services/` (13 modules), `src/mcp/server.py`, `src/coding/`
(3 modules), `src/ingestion/` (3 modules + `documents/` 6 modules), `src/paths.py`.
`ls src/` = api, classifier, coding, ingestion, mcp, memory, model_registry,
paths.py, retrieval, services, workers — every non-excluded directory is covered.
Defaults cross-checked against `src/api/config.py`.

---

### A. `src/services/retrieval_svc.py` — explicit retrieval pulls (E0/D4)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Explicit context pull (`context_for`) | `src/services/retrieval_svc.py:85` | E0 D4, C11, F1 | Runs the real chat stages — classify → memory-decision → orchestrate — for an arbitrary task string and returns structured fragments (not a rendered prompt). The engine behind MCP `ice_context`, the chat `/search` command, and F1's preview. | `settings.context_budget_fallback` (when no `budget` passed) | `23_000` | YES |
| Explicit pull ALWAYS orchestrates | `src/services/retrieval_svc.py:104-108` | B2, spec rev 5 | The B2 retrieve/don't-retrieve decision is computed and *reported* in the result but never used to answer empty-handed — `context_reliance` is forced to `Long_Term_Memory`. | — | — | YES |
| Timescope detection on the pull | `src/services/retrieval_svc.py:99-112` | T/timescope | `detect_timescope` runs on the task text and its scope dict is merged into the retrieval scope ("last week", "in June"). | — | — | YES |
| C6 scope parity for pulls | `src/services/retrieval_svc.py:119-129` | C6 | If the caller passes only a `conversation_id`, the conversation row is resolved through the ONE scope resolver, so an MCP pull inside an incognito/manual conversation gets the same flags the chat path gets. Caller keys win. | — | — | YES |
| Project pull freshens the working tree | `src/services/retrieval_svc.py:131-142` | E11 | A project-scoped pull calls `freshen_working_tree` first so retrieved code pointers match the tree being edited now. Failure logs a warning and the read continues commit-fresh. | `settings.reconcile_on_read` | `True` | YES |
| Constraint injection ("do not touch") | `src/services/retrieval_svc.py:36` | E8 (D7) | Active `constraint` decisions whose `files_affected` appear in the task text are prepended to the fragment list with score 10.0, ahead of every retrieved fragment. Path-ish tokens are regex-extracted from the task. | — (cap param `cap=5`) | 5 | YES (only on the explicit-pull path) |
| Recent turns | `src/services/retrieval_svc.py:186` | G16, E1 | Most recent stored turns; unscoped calls honour the privacy invariant (`is_private = FALSE`), `project=` returns all of a project's conversations' turns. | — (`limit=10`) | 10 | YES |
| Conventions listing | `src/services/retrieval_svc.py:220` | — | Active procedural patterns, strongest confidence first. | — (`limit=15`) | 15 | YES |
| Session-start block data | `src/services/retrieval_svc.py:236` | E4, C9 | Global-tier slots + the most recent `SessionSummary`. Backs `ice://session-start`. Global tier only by design. | — | — | YES |

### B. `src/services/slots.py` — memory slots (E0/C9/G14/C16)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Three-tier slot vocabulary | `src/services/slots.py:30` | C9 (D5) | ONE constant defines the legal slot names per tier: global (7: persona, user_preferences, tool_guidelines, project_context, guidance, pending_items, session_patterns), project (4), conversation (2). Every adapter inherits it. | — | — | YES |
| Hard per-slot token cap | `src/services/slots.py:55` | G14, C16 | Content over the cap is hard-truncated on every write, using REAL tokens (not word×1.33), and the response carries `truncated: true` + a warning + a WARNING log line. | `settings.slot_token_cap` | `300` | YES |
| Tier validation with named missing anchor | `src/services/slots.py:97` | C9 (D6/D7) | A project slot without a project_id, or a conversation slot without a conversation_id, raises a ValidationError naming the missing attachment. | — | — | YES |
| Versioned update / create-on-write | `src/services/slots.py:154` | C9 | Update bumps `version`, stamps `updated_by` and `last_updated`, reactivates an inactive slot; missing slots are created at version 1. | — | — | YES |
| Append (never silent overwrite) | `src/services/slots.py:191` | C9 | `ice_remember`'s slot branch — newline-appends through the same versioned path. | — | — | YES |
| Initialize the seven global slots | `src/services/slots.py:207` | C9 | Creates the seven default global slots empty; skips existing; the unique index guards the concurrent-init race (→ ConflictError). | — | — | N-A (explicit call; REST-only) |

### C. `src/services/graph.py` — codex graph read/write (E0 D3 / T4 / A7 / F3)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Entity resolution by UUID / canonical / alias | `src/services/graph.py:26` | E0 D3 | One resolver used by every graph call. | — | — | YES |
| Entity view (note + links + backlinks) | `src/services/graph.py:58` | F3, E0 D3 | Description, derived note (`context_payload`), properties, up to 20 out-edges and 20 in-edges each with relation, direction, negation, strength, confidence, validity window. | — | — | YES |
| Entity description edit (journaled) | `src/services/graph.py:81` | A7/F3, D13, G17 | Manual description edits are journaled via `log_description_update` and the derived note is REGENERATED — `context_payload` is never writable directly. MCP writes carry `source="mcp_edit"`. | — | — | YES |
| Edge listing (± expired) | `src/services/graph.py:102` | T4 | All edges touching an entity, optionally including expired ones (history). | — | — | YES |
| Entity timeline (`log` porcelain) | `src/services/graph.py:113` | T4 | Event-backed supersession history; returns `None` when the entity has only current facts. | `settings.timeline_max_transitions` | `8` | YES |
| Entity diff over `[t0, t1)` | `src/services/graph.py:136` | T4 | What changed about an entity between two ISO datetimes, JSON-safe. | — | — | YES |
| Symbol lookup with codex fallback | `src/services/graph.py:181` | E1b, E11 | `ice_where`'s engine: code graph first (file:line, signature, docstring), then codex name/alias resolution for non-code names, then an honest "no match" object. | `settings.reconcile_on_read` | `True` | YES |
| Freshen-target resolution for a symbol read | `src/services/graph.py:146` | E11 D3 | Explicit project ref → the symbol's `slug:` prefix → the sole registered project → None (bare cross-project lookup stays commit-fresh). | — | — | YES |
| Rendered architecture document | `src/services/graph.py:204` | E8 (D8), E11 | Builds an architecture markdown doc live over the stores: module tree with docstring one-liners, key decisions with rationale, constraints, project conventions, plus the last 15 git commits read live via subprocess. Never a maintained file. Flags an unreachable root inline. | — | — | YES (MCP `arch_doc` only) |

### D. `src/services/scoping.py` — conversation scope + curation (C3/C4/C6/C12/G16)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Closed scope vocabulary | `src/services/scoping.py:22` | C6 | Only `none`/`auto`/`project`/`manual` accepted; an invented mode is rejected rather than silently retrieved as `auto`. | — | — | YES |
| Set scope (+ project attach/detach) | `src/services/scoping.py:35` | C4, E1 D11 | Sets the mode, optionally attaches/detaches a project by slug/name/id (`""` detaches). An incognito conversation REFUSES project attachment. | — | — | YES |
| Cross-chat inclusion set (manual mode) | `src/services/scoping.py:38,152` | C6 | `included_conversation_ids` is the user's own pick for a `manual` conversation; kept (not wiped) across mode flips. `None` = unchanged, `[]` = clear. | — | — | YES |
| Exclusions in EVERY mode | `src/services/scoping.py:39-40,176` | C6 | `excluded_conversation_ids` / `excluded_cluster_ids` are never retrieved in any mode — "keep the memory, stop reading it". Also subtracted from a closed set so an excluded conversation cannot be re-admitted. | — | — | YES |
| Privacy re-sync on scope flip | `src/services/scoping.py:84-88` | G16 | Crossing the `none` boundary in either direction rewrites `is_private` on every episodic row of that conversation (the denormalised retrieval-time invariant). | — | — | YES |
| The ONE scope resolver | `src/services/scoping.py:114` | C6 | Conversation row → retrieval scope dict, shared by the chat path and the service/MCP path. Precedence none → manual → attached → project → auto. Marks hand-picked clusters `cluster_ids_explicit` so the automatic picker cannot overwrite them. | — | — | YES |
| Document visibility injection | `src/services/scoping.py:194` | C12, D3/D4/D5 | Documents are OPT-IN everywhere: enabled document conversations get ADDED to closed-set modes; in `auto` every non-enabled document conversation is added to the exclusion set. | — | — | YES |
| Knowledge-vs-text split for documents | `src/services/scoping.py:246-251` | C12 D4 | A document whose `knowledge_shared` latch has tripped keeps its codex/procedural facts visible everywhere even where its TEXT is hidden — a separate `exclude_knowledge_conversation_ids` deny set, read by `orchestrator.py:1390`. | — | — | YES |
| Manual label correction | `src/services/scoping.py:254` | C3 | Records a `CuratedLabel` row (corrected topic/intent/context_reliance) feeding the consent-gated fine-tune's curated set. | — | — | YES (REST-only surface) |

### E. `src/services/conversations.py` — deletion & /forget (C10/C11/G25/G32)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Manifest-first conversation deletion | `src/services/conversations.py:124` | C10 | Phase A is pure reads producing a per-store manifest and exact work lists; phase B applies them FK-safely in one transaction, one commit. | — | — | YES |
| Dry-run preview identical to the real run | `src/services/conversations.py:336` | C10 | `dry_run=True` runs phase A only and returns the same manifest the real run would. | — (`dry_run=False`) | `False` | NO (must be asked for) |
| Refuse deletion mid-generation | `src/services/conversations.py:139-144` | C10 | Raises ConflictError while `runtime.generation_in_flight` — a live stream's post-flight would write into the void. | — | — | YES |
| Corroborated codex edges survive | `src/services/conversations.py:64` | C10 D2 (spec rev 2) | An edge is kept if some `edge_added` event outside the deleted conversation's batches supports it; double-extraction inside the deleted conversation cannot immunize its own edge. | — | — | YES |
| Sole-support edges expired, never hard-deleted | `src/services/conversations.py:344-346` | C10 | Journaled `edge_expired`, reason `source_deleted`, excluded from T4 timelines (deletion is not evolution). | — | — | YES |
| User-authored entity descriptions protected | `src/services/conversations.py:83` | C10 | An entity whose description was ever edited via `mcp_edit`/`manual_edit` is never husked by a cascade. | — | — | YES |
| Entity husking (no liveness column) | `src/services/conversations.py:99` | C10, D1/D2 | Orphaned conversation-source entities get an unmatchable canonical name, emptied aliases, NULL embedding, `[deleted]` payload, journaled `entity_expired`. Row kept for the audit trail. | — | — | YES |
| Relation-gap rows in the cascade | `src/services/conversations.py:263-266,388` | G32/a1 | The gap ledger holds raw subject/object text from the deleted turns; it has no FK and was silently left behind until this filter was added. Deleted (not expired) — a gap is not a fact with history. | — | — | YES |
| Cluster emptiness / detach handling | `src/services/conversations.py:200-228` | C10, C5 | Clusters with no remaining members (via both FK paths) are deleted; surviving born-here clusters get their conversation anchor NULLed. | — | — | YES |
| Procedural pruning + deactivation | `src/services/conversations.py:232-243,364` | C10 | The deleted batches are removed from `source_batch_ids`; a pattern with no evidence left is deactivated. | — | — | YES |
| Decision expiry + review staling | `src/services/conversations.py:268-303` | C10 (spec rev 6) | Live decisions from the deleted batches expire bi-temporally; pending review items of three types referencing the content go `status="stale"`. | — | — | YES |
| Honest logs caveat | `src/services/conversations.py:57` | G25 | Every manifest restates that log files are NOT redacted by deletion. | — | — | YES |
| `/forget` proposal (fuzzy ⇒ queue) | `src/services/conversations.py:425` | C10/C11 D4 | Embedding-matches turns (visibility-guarded) + name-contained live edges and files a `forget_request` review item. NEVER applies anything. Zero matches ⇒ nothing queued. | — (`limit=5`) | 5 | YES |
| `/forget` apply arm | `src/services/conversations.py:513` | C10/C11 | On review approval: delete the listed turns (+ links, chunks by CASCADE, curated labels, cold rows) and expire the listed edges journaled with reason `user_forget`; tolerates rows that vanished since the proposal. | — | — | N-A (approval-gated) |

### F. `src/services/projects.py` — the project/coding control plane (E1/E3/E4/E8)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Project resolution by UUID/slug/name | `src/services/projects.py:41` | E1 | Every project-taking call accepts any of the three. | — | — | YES |
| Register + bootstrap a repo | `src/services/projects.py:83` | E1, E1b, E9 | Validates the root, slugifies, creates `Project` + `ProjectState` (branch + HEAD), then runs a FULL code-graph parse and the six fact parsers, returning a report with entity/edge/fact counts. | — | — | N-A (USER-REQUIRED entry point) |
| Consent-installed post-commit git hook | `src/services/projects.py:58,122` | E3 D5 | Writes a `post-commit` hook that POSTs the new hash to `http://localhost:8000/user-control/projects/{id}/commit`, falling back to a `$GIT_DIR/ice_pending_commit` marker file if the app is down. Always `exit 0`. Never overwrites a hook ICE did not write. | `install_git_hook` param | `False` | **NO** |
| Git-log replay flag | `src/services/projects.py:86,108` | F10 (deferred) | Accepted, stored into `project.settings["replay_git_log"]`, echoed in the report — and read by nothing. See DEAD OR INERT. | `replay_git_log` param | `False` | **NO** (and inert) |
| Project listing / status counts | `src/services/projects.py:142,156` | E1 | Roots, settings, goal, branch, last reconciled commit, unreachable flag, plus counts of code entities, derived facts, active decisions, open tasks. | — | — | YES |
| Project goal | `src/services/projects.py:185` | E1/E4 | Sets `ProjectState.goal`, surfaced in the session-start block. | — | — | YES |
| Commit notification routing | `src/services/projects.py:197` | E3 | Turns a commit signal into the runtime's `commit` work unit; with no runtime (headless/tests) it reconciles INLINE instead. | — | — | YES |
| Task CRUD + status machine | `src/services/projects.py:215,231,244` | E1/E3 | pending/active on create; pending/active/done/dropped on transition. Tasks accumulate commit hashes and changed files via the reconciler. | — | — | YES |
| Decisions list (± superseded) | `src/services/projects.py:259` | E8 | Active decisions by recency, optionally by type, optionally including superseded ones with their `superseded_by` pointer. | — (`limit=25`) | 25 | YES |
| Manual decision entry through E8's reconciler | `src/services/projects.py:277` | E8 D8 | A hand-written decision goes through `reconcile_and_insert`, so it dedupes and supersedes exactly like an extracted one. Returns E8's real status vocabulary — `recorded`/`duplicate`/`superseded`/`conflict_queued` — never a bare "ok". Types: decision \| constraint \| incident. | `settings.decision_duplicate_threshold` / `decision_conflict_threshold` | `0.95` / `0.85` | YES |
| Session-start ("where was I") data | `src/services/projects.py:320` | E4 D6 | Project state + a reflog-anchored `git diff --stat` since the last sitting + open constraints + ≤3 idle tasks + ≤3 freshest decisions. Advances `last_session_at` only when it was itself stale past the session gap. | `settings.session_gap_minutes` | `30` | YES |
| Session-start markdown renderer | `src/services/projects.py:373` | E4 | One renderer shared by the chat assembler and the MCP `ice://session-start` resource. | — | — | YES |
| Chat-path session-start entry | `src/services/projects.py:399` | E4 | The assembler's one-call entry (used by `main.py:644`) — rendered block or None. | — | — | YES |

### G. `src/services/documents.py` — document registry (C12)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Add a document (path or blob) | `src/services/documents.py:88` | C12 | Registers + ingests exactly one of `path`/`blob`. Origins: upload \| paste \| watch_folder \| project. `conversation_id` is optional — a watch-folder drop belongs to no conversation and lands enabled NOWHERE. | `origin` param | `"upload"` | YES |
| SHA-256 de-duplication → enable instead of re-ingest | `src/services/documents.py:122-132` | C12 | Same bytes = same document: it is enabled in the current conversation and the response says `deduplicated: true`. | — | — | YES |
| Document-vs-transcript routing | `src/services/documents.py:136-156` | C12 D6 | Structured formats (csv/xlsx/pptx/docx) are documents by construction (and refuse `kind=transcript`); everything else is classified by `detect_blob_kind`. Explicit `doc_kind` always wins. | — | — | YES |
| Transcript ingest → ghost conversation | `src/services/documents.py:193` | C12b, F14 | A pasted chat log goes through F14's raw slicer and lands as a real conversation with user/assistant turns whose id IS the document's conversation — so scoping, sharing and deletion treat it exactly like a PDF. | — | — | YES |
| Dry-run estimate | `src/services/documents.py:164-173` | C12 D5 | Section count, page count, and a time estimate with a human string, plus the note that extraction runs per section. Writes nothing. | `settings.document_seconds_per_section` | `6.0` | NO (must be asked for) |
| Background ingest via the runtime | `src/services/documents.py:181-186` | C12 | With a runtime present the document is marked `ingesting` and the `ingest_document` gpu-lane job is enqueued; without one it ingests inline. | — | — | YES |
| Kill-safe resumable ingest job | `src/services/documents.py:321` | C12 | Re-reads the source (file OR the stored `source_text` — blobs never had a file) and replays; per-section idempotency keys skip what already landed. | — | — | YES |
| No-silent-failure statuses | `src/services/documents.py:296-315` | C12 | Every-section-failed is `failed` with a reason, not `ready` with 0 sections; partial success is `ready` plus a visible `error` string and a WARNING. | — | — | YES |
| Library listing with `enabled_here` | `src/services/documents.py:359` | C12 | Every document newest-first; with a conversation id each carries whether it is enabled there. | — | — | YES |
| Live enable/disable toggle | `src/services/documents.py:372` | C12 D3 | A document is readable only where it is currently enabled. A link row is NEVER deleted on disable — `first_enabled_at` is the promotion latch's evidence. | — | — | YES |
| Knowledge-promotion latch | `src/services/documents.py:396-404` | C12 D4 | The moment a SECOND distinct conversation enables a document, its extracted knowledge joins the graph permanently (`knowledge_shared`); the text stays opt-in forever. Irreversible, logged. | — (link count ≥ 2) | 2 | YES |
| Delete document = delete its conversation | `src/services/documents.py:411` | C12, C10 | Routes through C10's single cascade rather than being a second deletion path; also how "un-sharing" a promoted document works. | `dry_run` param | `False` | NO for dry-run |

### H. `src/services/ingestion.py` — conversation import lifecycle (F10/F14)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Start an import | `src/services/ingestion.py:62` | F10 | Validates the decay policy and path, detects the format, parses to count work, prints a cost estimate, creates the `ImportRun` and enqueues the sliced replay job. | `policy` param | `"hybrid"` | YES |
| Dry-run preview (no run row) | `src/services/ingestion.py:96` | F10 D5 | Format, conversation/turn counts, seconds estimate + human string, and the "keep the machine on / imports arrive auto-scoped" note. Creates NO ImportRun. | — | `False` | NO |
| One import at a time | `src/services/ingestion.py:100-102` | F10 rev 10 | A live `running` row raises ConflictError. | — | — | YES |
| Crashed-import reaping | `src/services/ingestion.py:47` | F10 rev 10 | A `running` row whose heartbeat is older than the window is marked `aborted` so a fresh start is not blocked; resume rides the hash ledger, not the row. | `settings.import_stale_run_seconds` | `600.0` | YES |
| Inline replay with no runtime | `src/services/ingestion.py:117-126` | F10 | Headless/test path: replay synchronously then finalize the run row. | — | — | N-A |
| Import status (one or latest + recent) | `src/services/ingestion.py:130` | F10 | Full progress dict: totals, done, skipped, failed turns, timestamps, error, report; plus the 10 most recent runs. | — | — | YES |

### I. `src/services/bookmarks.py` — turn-level control (C1/C7)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Bookmark a turn | `src/services/bookmarks.py:47` | C1 | Flags a turn bookmarked + `lossless_flag` + `decay_immune`, then enqueues `codex_extract` for its batch (priority). | — | — | YES |
| Extraction enqueue with loud failure | `src/services/bookmarks.py:38` | C7 | No runtime ⇒ an ERROR log line, never a silent skip. | — | — | YES |
| List bookmarks | `src/services/bookmarks.py:63` | C1 | All bookmarked turns, optionally conversation-scoped. | — | — | YES |
| Latest turn id | `src/services/bookmarks.py:71` | — | TUI/chat-command helper: the most recent turn id of a conversation. | — | — | YES |
| Free-text note capture (`remember_note`) | `src/services/bookmarks.py:81` | E7 spec rev 6 | Stores arbitrary text as a bookmarked, decay-immune, lossless, `inject_raw` turn in ONE dedicated deterministic-id conversation (`uuid5(NAMESPACE_URL, "ice://mcp-notes")`, scope `auto`) and enqueues codex extraction. | — | — | YES |

### J. `src/services/review.py`, `clusters.py`, `registry_svc.py`, `errors.py`

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Review queue listing by status | `src/services/review.py:35` | D8 | Statuses in play: pending, approved, rejected, plus `resolved` (agent-settled) and `stale` (C10). | — (`status="pending"`) | pending | YES |
| Approval APPLIES the item | `src/services/review.py:44` | D1/D2 D6 | Five live arms: `memory_slot_update` → slots service (proposer recorded as author); `new_cluster_proposal` → creates the cluster; `entity_merge` → `codex_ops.merge_entities`; `codex_reconciliation` → expires the old edge journaled as `supersession`; `forget_request` → `apply_forget`. | — | — | YES |
| Reject (feeds the never-re-propose check) | `src/services/review.py:100` | D1 | Flips to rejected without applying; D1's detectors query rejected rows so a proposal is not re-raised. | — | — | YES |
| Explicit cluster creation | `src/services/clusters.py:12` | C5 | Creates a named `ContextCluster` by hand. | — | — | YES (REST-only) |
| Explicit turn→cluster assignment | `src/services/clusters.py:21` | C5 | Sets `cluster_id` on the given turns. Note: writes the legacy `cluster_id` column, not `EpisodicClusterLink`. | — | — | YES (REST-only) |
| fcntl-locked model registry | `src/services/registry_svc.py:19` | E7 spec §4 | The registry JSON is shared mutable state between the app and `ice-mcp`; every load-modify-save (and every read) runs inside one exclusive sidecar-lockfile lock so no reader sees a half-written file. | — | — | YES |
| Registry read / refresh / update / delete | `src/services/registry_svc.py:31,36,41,54` | — | Refresh re-populates from Ollama; updates are restricted to four fields (`topic_tags`, `intent_tags`, `confirmed`, `base_url`). | — | — | YES |
| Domain-error vocabulary | `src/services/errors.py:9-22` | E0 | `NotFoundError`/`ValidationError`/`ConflictError`; each adapter maps them (REST 404/400/409, MCP tool error, chat inline reply). Services know nothing about HTTP. | — | — | YES |

### K. `src/mcp/server.py` — ICE-as-MCP (E7)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| `ice-mcp` entrypoint, stdio or HTTP | `src/mcp/server.py:499` | E7 | `pyproject [project.scripts]` console script. stdio by default; `--http` serves streamable-http. Logging is forced to stderr because stdio transport owns stdout. | `--http` flag | stdio | NO (stdio default) |
| Postgres attach-or-boot | `src/mcp/server.py:67` | E7 D6 | If postgres answers, attach; else `docker compose up -d postgres` and wait 30 s; on failure exit nonzero printing the exact command. | — | — | YES |
| Docker linger on shutdown | `src/mcp/server.py:106-111` | E7 D6 | The core stops but docker stays up so the next session attaches instantly; `./stop_ice` remains the all-the-way-down path. | — | — | YES |
| Lease-checked runtime (standby) | `src/mcp/server.py:101`, `src/api/core.py:88` | E7 D6 | `create_core()` starts the maintenance runtime in STANDBY if the app already holds a fresh `runtime_lease` — event jobs still run, periodic dispatch stays with the owner. Never a second dispatcher. | `settings.runtime_lease_ttl_seconds` | `180.0` | YES |
| Per-call telemetry journal | `src/mcp/server.py:140` | D8/E5 | Every tool call (reads included) emits an `mcp_tool_call` structlog event — the pull-discipline measurement stream. | — | — | YES |
| Guided self-correction on bad action | `src/mcp/server.py:152` | E7 spec §4 | An unknown action raises a ValueError listing the valid actions. | — | — | YES |
| Server instructions steer pull-first behaviour | `src/mcp/server.py:116-123` | E7 | The FastMCP `instructions` string tells the calling model to call `ice_context` BEFORE grepping the codebase or asking the user. | — | — | YES |
| `ice://session-start` resource | `src/mcp/server.py:479` | E4, E7 | Renders slots + last session summary + up to 5 registered projects' where-was-I blocks. A broken repo warns and is skipped rather than killing the resource. | — (limit 5 projects) | 5 | YES |
| 10 tools (6 composite + 4 action-multiplexed) | `src/mcp/server.py:161-451` | E7 | See the MCP COVERAGE section below for the per-tool mapping. | — | — | YES |

### L. `src/coding/code_graph.py` — E1b static code graph

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Python AST extraction into the ONE codex | `src/coding/code_graph.py:188` | E1b (D2/D3) | Modules, classes and functions become `CodexEntity` rows with `source='static_analysis'`, canonical `"{slug}:{module}.{qualname}"` casefolded, `file_path`, `line_start/end`, `signature`, docstring FIRST LINE. Pointers, never bodies. | — | — | YES |
| Derived-memory hygiene | `src/coding/code_graph.py:1-24,288-295` | E1b | Code rows are decay-exempt, journal-free (no CodexEvents — a re-parse would bury T-track signal), keep empty aliases so chat mentions can't attach, carry NO embedding, and are excluded from conversational retrieval unless the query's project matches. | — | — | YES |
| Deterministic per-project batch id | `src/coding/code_graph.py:51` | E1b rev 5 | `uuid5("ice:code-graph:{project_id}")` — this is the "code-graph allowance" the orchestrator's `_codex_scope_sets` admits for project-scoped retrieval. | — | — | YES |
| `imports` edges | `src/coding/code_graph.py:482-486` | E1b | module → module, project-internal only. | `settings.code_graph_resolved_confidence` | `1.0` | YES |
| `calls` edges, best-effort static resolution | `src/coding/code_graph.py:403` | E1b (D2) | Resolves same-module names, `self.` methods, imported names, `module.attr` on imported project modules, and a unique-name-anywhere heuristic. No type inference. | `settings.code_graph_heuristic_confidence` | `0.6` | YES |
| Heuristic-ratio deferral signal | `src/coding/code_graph.py:387-392` | E1b rule 2b | Bootstrap logs the heuristic fraction; the ">60% ⇒ tighten to imports-only" rule is LOGGED, NOT ENFORCED. | — | — | N-A (logged only) |
| Ignore set + file cap | `src/coding/code_graph.py:44,208` | E1b spec §4 | Default ignores (.git, .venv, node_modules, models, data, logs, …) plus per-project `settings["ignore"]`; the walk refuses past the cap with a WARNING. | `settings.code_graph_max_files` | `20_000` | YES |
| Incremental delete-and-recreate sync | `src/coding/code_graph.py:317` | E1b D5 | Vanished symbols and files are hard-deleted with their edges (derived memory is regenerable); surviving canonicals update in place so ids and inbound cross-file edges stay stable. Entities flush BEFORE edges. | — | — | YES |
| Cross-process advisory lock | `src/coding/code_graph.py:330` | E11 D4 | `pg_advisory_xact_lock(hashtext(project_id))` — at most one reparse of a project at a time across app + `ice-mcp`, since the deterministic entity ids would collide. | — | — | YES |
| Syntax-error tolerance (keep the last good map) | `src/coding/code_graph.py:342-356` | E1b rev 12 | A broken file is usually mid-edit: no symbol deletions, no edge rebuild; the module entity is stamped with `parse_error` and a WARNING is logged. | — | — | YES |
| Full sync + stale sweep | `src/coding/code_graph.py:372` | E1b | Bootstrap/force-full: sync every discoverable file then drop derived rows whose file no longer exists. | — | — | YES |
| Symbol lookup rows | `src/coding/code_graph.py:498` | E1b spec §4 | Exact canonical, then `.suffix` and `:suffix` LIKE matches (escaped), shortest-name first, ≤10 rows; an attached project's rows rank first. | — | — | YES |

### M. `src/coding/reconciler.py` — E3 reconcile-on-commit + E11 reconcile-on-read

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Commit work-unit handler | `src/coding/reconciler.py:68`, registered at `src/api/core.py:94-95` | E3, C7 seam | The `commit` work unit becomes a `project_reconcile` job — nothing runs inline. Registered in `create_core()`, so BOTH boot paths (app and headless `ice-mcp`) get reconcile-on-commit. | — | — | YES |
| One reconcile pass | `src/coding/reconciler.py:99` | E3 D5 | Per commit range since `last_reconciled_commit`: changed-file incremental re-parse → hash-gated fact re-derive → task linking → cue-gated decision extraction → advance the state row. | — | — | YES |
| Unknown-base full re-parse fallback | `src/coding/reconciler.py:141-154` | E3 §4 | A force-push/history rewrite (or no anchor) falls back to a full sync + full fact derive. | — | — | YES |
| Root reachability tracking | `src/coding/reconciler.py:79` | E3 §4 | Sets/clears `project.settings["unreachable"]` so tools can say "stale" instead of silently serving an old graph. | — | — | YES |
| Commit→task linking | `src/coding/reconciler.py:167-183` | E3 D5 step 3 | The range's commits attach to the active task (or the most recent pending one), merging changed files and stamping `state.last_task_id`. | — | — | YES |
| Cue-gated decision extraction from commit messages | `src/coding/reconciler.py:187-200` | E8 | Each commit message is checked by `decision_cue`; a hit enqueues the gpu-lane `decision_extract` job. The LLM never runs in the reconcile job itself. | `settings.reconcile_max_commits_scanned` | `20` | YES |
| Poll fallback (HEAD drift + marker file) | `src/coding/reconciler.py:216` | E3 D5 | Every registered project is checked; reconciles when the hook's `$GIT_DIR/ice_pending_commit` marker exists or HEAD drifted. The ≤10-min lag path when the hook was declined. | `maintenance_intervals["project_poll"]` | `600` s | YES |
| Reconcile-on-read (working-tree freshener) | `src/coding/reconciler.py:279` | E11 | Makes the code graph match the working tree at read time — the case neither commits nor polling covers (long in-session editing with no commits). Called by the SERVICE layer only; episodic reads never trigger it. Never advances `last_reconciled_commit`. | `settings.reconcile_on_read` | `True` | **YES** |
| Freshen throttle | `src/coding/reconciler.py:294-298` | E11 | Per-process, per-project: at most one `git status` per interval under a burst of agent reads. | `settings.reconcile_on_read_min_interval_seconds` | `2.0` | YES |
| Dirty-set signature short-circuit | `src/coding/reconciler.py:266,309-312` | E11 | A sha1 over (path, mtime_ns, size) — an unchanged digest means the same edits are already in the graph, so the reparse is skipped. | — | — | YES |
| Large-dirty-set warning (not a cap) | `src/coding/reconciler.py:313-315` | E11 D5 | Warns past the threshold; deliberately never caps — a Z1 signal. | `settings.reconcile_large_dirty_set` | `50` | YES |
| Rename-aware dirty detection | `src/coding/reconciler.py:243` | E11 | `git status --porcelain` parsed so a rename contributes BOTH sides (old path deletes, new path creates); ignore-filtered, `.py` only. | — | — | YES |

### N. `src/coding/project_facts.py` — E9 derived project facts

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Six deterministic fact parsers, no LLM | `src/coding/project_facts.py:277` | E9 (D9), E6 | Each yields ONE `codex_entities` row, `entity_type='project_fact'`, `source='derived'`, structured `properties` + human `description` + file pointers. | — | — | YES |
| dependencies | `src/coding/project_facts.py:48` | E9 | `pyproject.toml` declared deps + resolved versions from `uv.lock`, plus `package.json` deps. | — | — | YES |
| db_schema | `src/coding/project_facts.py:95` | E9 | Walks every .py for `__tablename__` classes → tables, models, columns, file; plus the computed alembic head. | — | — | YES |
| commands | `src/coding/project_facts.py:150` | E9 | Makefile targets, npm scripts, pyproject entry points. | — | — | YES |
| config_surface (KEY NAMES ONLY) | `src/coding/project_facts.py:183` | E9 D9 | `BaseSettings` field names + `.env`/`.env.example` key names. Values are NEVER stored. | — | — | YES |
| infrastructure | `src/coding/project_facts.py:217` | E9 | Regex-parses docker-compose (no yaml dependency) into services with images and ports. | — | — | YES |
| data_shapes | `src/coding/project_facts.py:255` | E9 | Top-level keys of up to 10 `*schema*.json` files under 200 kB. | — | — | YES |
| Content-hash gating + changed-file skip | `src/coding/project_facts.py:287,297` | E9, E3 D5 step 2 | On the reconcile path, parsers whose source files did not change AND whose unit exists are skipped; the sha256 over the source files gates the write either way. A parser that now returns nothing DELETES its stale unit. | — | — | YES |
| Parser failure isolation | `src/coding/project_facts.py:315-320` | E9 | One throwing parser logs a WARNING and the other five still run. | — | — | YES |

### O. `src/ingestion/` — conversation import (F10) + raw slicing (F14)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Replay engine (history lives through the pipeline) | `src/ingestion/importer.py:142` | F10 D1 | Per turn pair: store the episodic turn with its ORIGINAL timestamp + gap-derived session, run the REAL `evaluate_turn` post-flight chain (density → summary → codex → procedural → decision), then cluster the conversation and summarise at the end. Mature memory, not an archive. | — | — | YES |
| Four decay policies | `src/ingestion/importer.py:62` | F10 | `hybrid` (default: ≤30 d preserved + 14 d immunity, older fast-forwarded with aging counted from the threshold so there is no cliff), `preserve`, `fast_forward`, `fresh`. Fast-forwarded scores floor at the cold threshold — the importer never deletes a row it just created. | `settings.import_recent_days` / `import_immune_window_days` / `decay_daily_unaccessed` / `decay_daily_creative` / `decay_cold_threshold` / `decay_creative_floor` | 30 / 14 / 0.95 / 0.99 / 0.05 / 0.3 | YES |
| Two-level idempotency (resume) | `src/ingestion/importer.py:132,180-190,258` | F10 D4/D8 | A fully-replayed conversation is recorded by content hash and skipped; a mid-conversation kill re-runs it and per-turn keys dedupe. Deterministic conversation ids via `uuid5(NS_ICE_IMPORT, provider:source_id)`. | — | — | YES |
| Message merge + user→assistant pairing | `src/ingestion/importer.py:101` | F10 rev 13 | Consecutive same-role messages merge with `\n\n`; a trailing user turn pairs with an empty assistant half; a leading assistant turn is skipped and COUNTED in the report. | — | — | YES |
| Yield-to-live-chat gate | `src/ingestion/importer.py:219-223` | F10 | The engine sleeps between turns while `runtime.generation_in_flight` — never replays into the pipeline behind a live stream. | — | — | YES |
| Self-re-enqueueing sliced job | `src/ingestion/importer.py:369` | F10 rev 9 | Each `import_replay` dispatch replays one slice then re-enqueues the next, so an hours-long import never starves live chat. End-of-run summaries fire only on the final slice. | `settings.import_slice_budget_seconds` | `600.0` | YES |
| Per-turn failure isolation | `src/ingestion/importer.py:241-245` | F10 | One bad turn rolls back and increments `failed_turns`; the run continues. | — | — | YES |
| Ghost-conversation override | `src/ingestion/importer.py:196-198` | C12b | `conversation_id_override` replays a single source into an already-created conversation — how a pasted chat log becomes a document-owned ghost conversation. | — | — | YES |
| ChatGPT export adapter | `src/ingestion/formats.py:148` | F10 D2 | Mapping tree walked parent-ward from `current_node`; other branches counted (`branch_messages_skipped`), not imported; hidden messages dropped. | — | — | YES |
| Claude export adapter | `src/ingestion/formats.py:215` | F10 rev 6 | `chat_messages` includes abandoned edit branches; the live path is the parent chain from the latest-created message. Flat `text` preferred over block join. | — | — | YES |
| DeepSeek export adapter | `src/ingestion/formats.py:291` | F10 rev 5 | Mapping tree with NO `current_node` — at a branch, follow the child whose subtree holds the latest `inserted_at`. REQUEST/RESPONSE fragments only; THINK/SEARCH/TOOL_* dropped. | — | — | YES |
| Generic JSONL adapter (two shapes) | `src/ingestion/formats.py:360` | F10 rev 7 | `{role, content, timestamp}` per line, OR the pair shape `{prompt, response, timestamp}` — the shape ICE's own historical exports use. Groups by `conversation`/`conversation_id`. | — | — | YES |
| Format sniffing | `src/ingestion/formats.py:430` | F10 D2 | Extension + first-byte + structure detection; unknown structure raises an error NAMING every supported shape. `.txt`/non-JSON routes to the raw slicer. | — | — | YES |
| Timestamp synthesis with honesty flags | `src/ingestion/formats.py:102` | F10 §3, T-honesty | Forward/back-fills missing times from anchors, end-anchors when there are none at all, clamps inverted times monotonically, and flags `ts_synthesized` for the report. | — | — | YES |
| Raw-dump slicing v2 (seam reconciliation) | `src/ingestion/raw_slicer.py:164` | F14 D6 | Overlapping word-window slices, one bg-model turn extraction per slice, then ONE open seam call per adjacent pair that sees A's tail + B's head + the raw overlap and returns the corrected boundary. Replaces the v1 "amnesia method" cold guessing. | `settings.raw_slice_tokens` / `raw_slice_overlap_words` / `raw_slice_seam_turns` | `2667` / `200` / `3` | YES |
| Cross-dump dedup + synthetic times ending at mtime | `src/ingestion/raw_slicer.py:138,152` | F14 | Normalized-alnum dedup sweep; timestamps spaced 1/min backwards from the file mtime, every turn flagged `ts_synthetic` so T-timelines caveat those dates. | — | — | YES |
| Loud unparseable-slice fallback | `src/ingestion/raw_slicer.py:84` | F14, G29 | On any parse failure the whole slice is stored as ONE user turn — and says so at WARNING, because a silent failure turns a conversation into an undifferentiated blob. Uses G29's shared `parse_array` salvage. | — | — | YES |

### P. `src/ingestion/documents/` — document ingestion (C12)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| A document IS a conversation | `src/ingestion/documents/ingest.py:84` | C12 | Sections become turns of the document's own conversation, so C6 scoping, C4 summaries, C2/C3 chunk retrieval, C5 clustering and the codex/procedural chain all apply for free. No new retrieval leg. | — | — | YES |
| Section building via the shared chunker | `src/ingestion/documents/ingest.py:47` | C12, C2/C3 | Blocks → sections with the shared sentence/code-aware greedy packer. Two blocks are NEVER merged, so a section never straddles a page/slide/heading/sheet. | — | — | YES |
| Provenance header inside `raw_text` | `src/ingestion/documents/ingest.py:64` | C12 | `[title · p.7 · section 3/40]` rides inside the stored text so BM25 finds the filename, the assembler renders it unchanged, and re-embed reproduces it deterministically. | — | — | YES |
| Per-section idempotency + slice deadline | `src/ingestion/documents/ingest.py:78,112` | C12 | A crash costs only uncommitted sections; a deadline ends a slice with `complete: False` for the job to re-enqueue. | — | — | YES |
| Sections deliberately NOT flagged `is_document` | `src/ingestion/documents/ingest.py:167-179` | C12 trap 1 | The vector leg excludes `is_document` rows expecting chunks instead — but a section is already chunk-sized so no chunks exist, and the row would be represented nowhere. | — | — | YES |
| One-sitting section timestamps | `src/ingestion/documents/ingest.py:123` | C12, C5 | Sections stamped seconds apart so `resolve_session_id` keeps them in one session and C5's session affinity clusters them together. | `settings.document_section_seconds` | `1` | YES |
| Model-based document/transcript detection | `src/ingestion/documents/kind.py:91` | C12 D6, G28 | Head+middle+tail sample judged by the background model (STRUCTURE of the exchange, not speaker labels) — explicitly replacing `raw_text.count("Assistant:") >= 3`. Enum-constrained response, bare top-level enum (an object wrapper truncated every answer). | `settings.document_blob_kind_sample_chars` | `1500` | YES |
| Asymmetric failure default | `src/ingestion/documents/kind.py:91-125` | C12 D6 | Any model failure ⇒ DOCUMENT, deliberately: a document run through the slicer is MANGLED (invented turns stored as memory), a transcript ingested as a document only loses role structure. | — | DOCUMENT | YES |
| Swappable extraction engine | `src/ingestion/documents/extract.py:31` | C12 D10, Track F | `builtin` = the pure-python parsers. `tika`/`docling` are the reserved OCR-container branches. | `settings.document_extraction_engine` | `"builtin"` | YES (builtin); tika/docling **NO** |
| Format parsers | `src/ingestion/documents/parsers.py:120-346` | C12 D9 | pdf (pypdf, per-page blocks), docx (heading-grouped + tables), pptx (per slide incl. speaker notes), xlsx (per sheet), csv/tsv (delimiter-sniffed), html (bs4), plain text + ~45 source extensions. | — | — | YES |
| Scanned PDFs REFUSED, not silently empty | `src/ingestion/documents/parsers.py:151-156` | C12 D9 | 0 extracted characters raises with the page count and points at the Track F OCR item. | — | — | YES |
| Explicit unsupported-format refusals | `src/ingestion/documents/parsers.py:68,101` | C12 D9 | .doc/.rtf/.odt/.epub/images refuse with a message naming what IS supported. | — | — | YES |
| Tables DESCRIBED, not dumped | `src/ingestion/documents/parsers.py:245,271` | C12 D11 | Schema + per-column stats (min/max/mean, date range, top distinct values, null count) plus a deterministic head+spread row sample as markdown — deterministic so re-ingest keeps the same idempotency keys. | `settings.document_table_sample_rows` / `document_table_max_render_cols` / `document_table_top_values` | `50` / `30` / `5` | YES |
| Watch folder (`ingest_inbox/`) | `src/ingestion/documents/watch_folder.py:63` | C12, G13 | A cadence runtime job (the honest replacement for the never-started `drop_zone.py` watchdog, which also loaded a SECOND classifier). Scans, ingests through the document service, moves to `processed/` or `failed/`. Dropped files are enabled in NO conversation. | `maintenance_intervals["ingest_folder"]` / `settings.document_watch_files_per_run` | `900` s / `1` file | YES |
| Settling check before ingesting | `src/ingestion/documents/watch_folder.py:41` | C12 | A file whose mtime is younger than 5 s is still being written and is skipped. | — (`min_age_seconds=5.0`) | 5.0 s | YES |

### Q. `src/paths.py` — install-root anchoring (G31)

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| One install root for ICE's own artifacts | `src/paths.py:76` | G31, TRAPS #10 | `REPO_ROOT` resolved once at import from `src/paths.py`'s parents — closes the class of bug where `.env`, the model registry, the NER checkpoint, the classifier checkpoint and the label schema resolved against the CWD and silently fell back (a 52.7→34.1 entity-count drop and a `qwen2.5:7b` word-salad fallback both traced here). | — | — | YES |
| `ICE_HOME` override | `src/paths.py:61` | G31, E7, Track F | The ONE override, resolved before import, for Track F's packaged app and the headless `ice-mcp` boot. | `ICE_HOME` env var | unset | **NO** |
| The rule: ICE artifacts anchored, USER material not | `src/paths.py:35-39` | G31 | `coding/project_facts.py` and `coding/code_graph.py` walk a repo the user points ICE at — anchoring those would be a bug, not a fix. | — | — | YES |

---

## 1. MCP COVERAGE

### Per-tool mapping (tool → service function → capabilities reached)

| MCP tool / action | Calls | Capabilities reached |
|---|---|---|
| `ice_context(task, conversation_id, budget)` | `retrieval_svc.context_for` (`server.py:172`) | The whole retrieval stack: live classifier (27-logit head), timescope detection, B2 memory decision (reported, not obeyed), C6 scope resolution from the conversation row, **E11 working-tree freshen when project-scoped**, the full hybrid orchestrator with RRF fusion and budgeting, and E8 constraint injection. |
| `ice_why(name)` | `graph_svc.entity_view` + `graph_svc.entity_timeline` (`server.py:182-183`) | Codex entity note, aliases, properties, ≤20 links + ≤20 backlinks, and T4's supersession timeline. |
| `ice_recent(conversation_id, limit, project)` | `retrieval_svc.recent_turns` (`server.py:196`) | Recent episodic turns, G16 privacy-guarded; project mode spans all of a project's conversations. |
| `ice_conventions()` | `retrieval_svc.conventions` (`server.py:206`) | Active procedural patterns with confidence + reinforcement counts. |
| `ice_where(symbol, project)` | `graph_svc.where_symbol` → `code_graph.where_symbol_rows` (`server.py:219`) | E1b code graph (file:line, signature, docstring), codex name/alias fallback, **and an E11 working-tree reparse of the resolved project (a WRITE)**. |
| `ice_remember(text, target)` | `bookmarks_svc.remember_note` OR `slots_svc.append_to_slot` (`server.py:232-233`) | Bookmarked/decay-immune/lossless note in the MCP-notes conversation **plus a `codex_extract` job enqueue**; or a versioned, G14-capped slot append recorded as `mcp_edit`. |
| `ice_slots(list/get/set)` | `slots_svc.list_slots` / `get_slot` / `update_slot` (`server.py:258-268`) | All three C9 tiers, with `projects_svc.resolve_project` for the project tier. **No `initialize` action.** |
| `ice_graph(view/edit/edges/timeline/diff)` | `graph_svc.entity_view` / `entity_edit` / `edges_list` / `entity_timeline` / `entity_diff` (`server.py:284-292`) | Full codex read surface + the journaled description write (`source="mcp_edit"`, derived payload regenerated). |
| `ice_bookmarks(list/add)` | `bookmarks_svc.list_bookmarks` / `bookmark_turn` (`server.py:448-450`) | Bookmark listing; bookmarking promotes to lossless + decay-immune **and enqueues `codex_extract`**. |
| `ice_control` — 26 actions (`server.py:344-426`) | see below | The control plane. |

`ice_control` actions and their targets: `scope_get`/`scope_set` → `scoping_svc`;
`review_list`/`review_approve`/`review_reject` → `review_svc` (approve APPLIES:
slot updates, cluster creation, entity merges, edge supersession, forget);
`registry_view`/`registry_edit` → `registry_svc`; `project_register`,
`project_list`, `project_status`, `project_goal` → `projects_svc`;
`project_reconcile` → `coding.reconciler.reconcile_project` directly; `arch_doc`
→ `graph_svc.render_architecture_doc`; `decisions_list`/`decisions_add`,
`task_add`/`task_list`/`task_status` → `projects_svc`; `delete_conversation`,
`forget_propose` → `conversations_svc`; `import_status` → `ingestion_svc`;
`document_list`/`document_status`/`document_add`/`document_enable`/`document_disable`
→ `documents_svc`.

### Reachable ONLY through the HTTP proxy / REST / CLI — NOT through MCP

1. **The entire chat pipeline.** `POST /v1/chat/completions` is the only path that
   classifies a live turn, assembles a prompt (`prompt_assembler.py`), routes to a
   model, streams, stores the turn (`main.py:213 store_turn_async`) and enqueues
   `post_flight` (`main.py:331`). MCP has no equivalent — see the write-path answer below.
2. **C11 chat commands** (`/remember`, `/slots`, `/bookmark`, `/search`, `/scope`,
   `/forget`, `/delete-conversation`) — `src/api/chat_commands.py:45`. They reach the same
   services, but only from inside a chat message.
3. **Starting an import.** `POST /user-control/import` (`user_control.py:190`) and
   `scripts/ice_replay_import.py`. MCP exposes `import_status` only — its own docstring
   says so (`server.py:330`).
4. **Deleting a document.** `DELETE /user-control/documents/{id}` (`user_control.py:250`).
   No MCP action.
5. **Slot initialization.** `POST /memory-slots/initialize` (`memory_slots.py:81`).
6. **Manual label correction (C3).** `POST /user-control/batch/override-tags`
   (`user_control.py:110`) — the curated-label feed for fine-tuning.
7. **Explicit cluster creation and turn assignment (C5).**
   `POST /user-control/clusters`, `PUT /user-control/clusters/{id}/assign`
   (`user_control.py:163,167`).
8. **Model registry refresh and delete.** `POST /user-control/model-registry/refresh`,
   `DELETE /user-control/model-registry/{name}` (`user_control.py:273,282`). MCP has view
   and edit only.
9. **The commit-notification endpoint.** `POST /user-control/projects/{ref}/commit`
   (`user_control.py:146`) — the git hook's target.
10. **Latest-turn lookup.** `GET /user-control/conversations/{id}/latest-turn`
    (`user_control.py:260`).
11. **CLI-only front doors:** `scripts/register_project.py` (which, unlike MCP, *requires*
    an explicit `--hook`/`--no-hook` choice), `scripts/ice_add_document.py`,
    `scripts/ice_replay_import.py`.

### Reachable ONLY through MCP — NOT through the HTTP proxy

1. **The whole codex-graph read/write surface.** `ice_graph` view/edit/edges/timeline/diff
   and `ice_why`. There is NO REST router for `services/graph.py` — `entity_view`,
   `entity_edit`, `edges_list`, `entity_timeline`, `entity_diff` are referenced only from
   `src/mcp/server.py`.
2. **Symbol lookup (`ice_where`).** `where_symbol` has exactly one caller: `server.py:219`.
3. **The rendered architecture document.** `render_architecture_doc` has exactly one
   caller: `server.py:380`.
4. **Everything in the project/coding control plane except the commit hook endpoint:**
   `project_register`, `project_list`, `project_status`, `project_goal`,
   `project_reconcile`, `decisions_list`, `decisions_add`, `task_add`, `task_list`,
   `task_status`. No REST equivalents exist (only `scripts/register_project.py` covers
   registration).
5. **Review REJECT.** REST has `POST /review-queue/{id}/approve` but no reject endpoint;
   `review_svc.reject` is reachable only via `ice_control action="review_reject"`.
6. **The session-start welcome-back block over ALL projects.** `ice://session-start`
   (`server.py:479`) is the only caller of `retrieval_svc.session_start_block`. (The chat
   path injects only the *current* project's block via `chat_session_start`.)
7. **`ice_remember` note capture.** `bookmarks_svc.remember_note` has one caller,
   `server.py:232` — REST can bookmark an existing *turn* but cannot store a free-text note.
8. **`ice_context` / `ice_recent` / `ice_conventions` as APIs.** `context_for` is otherwise
   reachable only from inside a chat message (`/search`); `recent_turns` and `conventions`
   have no REST endpoint at all.

### Does MCP trigger the write / extraction path?

**No, it is not read-mostly — it writes a great deal, but it never runs the
chat-turn post-flight.** Concretely:

*Extraction/pipeline work MCP DOES trigger:*
- `ice_bookmarks action="add"` → `bookmark_turn` → `runtime.enqueue("codex_extract", priority=True)` (`bookmarks.py:41`) — the standalone codex extractor.
- `ice_remember target="bookmark"` → `remember_note` → writes a new `EpisodicMemory` row and enqueues the same `codex_extract` job (`bookmarks.py:118`).
- `ice_control action="document_add"` → `documents_svc.add_document` → runtime job `ingest_document` → `ingest_document()` → **`post_flight.evaluate_turn` per section** (`ingest.py:129`), i.e. the FULL chain: density → summary → codex → procedural → decision, plus clustering and the conversation summary. A `transcript` blob additionally runs the F14 bg-model slicer and `importer.import_conversations`, which also calls `evaluate_turn` per turn.
- `ice_control action="project_register"` / `project_reconcile` → full code-graph write, fact derivation, and `runtime.enqueue("decision_extract", ...)` — an LLM stage.
- `ice_context` (project-scoped) and `ice_where` → `freshen_working_tree` → `CodeExtractor.sync_files`, which **writes and deletes codex entities and edges under an advisory lock**. Two nominally "read" tools mutate the graph.

*Other direct writes:* `ice_slots set`, `ice_remember` slot branch, `ice_graph edit`
(journaled CodexEvent, `source="mcp_edit"`), `ice_control scope_set` (including the G16
`is_private` re-sync across every turn of a conversation), `review_approve` (merges
entities, expires edges, applies `/forget`, writes slots, creates clusters),
`delete_conversation` (destructive, dry-run available), `forget_propose` (queues),
`document_enable` (can permanently trip the irreversible knowledge-promotion latch),
`registry_edit`, all task/decision/goal writes.

*What MCP CANNOT do:* store a live chat turn. There is no MCP tool that takes a
prompt+response pair, stores an `EpisodicMemory` row for it, and enqueues the
`post_flight` job — that path exists only in `main.py:213/331`. So the answer is:
**MCP reaches the extraction machinery through three side doors (bookmark, note,
document/transcript ingest) and through the coding core, but the conversational
write path itself is proxy-only.**

---

## 2. THE CODING CORE (`src/coding/`, ~1,200 lines)

**What it is.** A per-project static map of the user's own repositories, stored in
the SAME codex tables as conversational memory but namespaced and marked
`source='static_analysis'` (code symbols) or `source='derived'` (project facts).
Three files:

- **`code_graph.py` (537 lines, E1b).** A Python `ast` extractor. Modules, classes,
  functions become `CodexEntity` rows keyed `"{project_slug}:{module}.{qualname}"`
  (casefolded), carrying `file_path`, `line_start/end`, `signature`, and the docstring's
  first line — **pointers, never bodies**. Two edge types: `imports` (module→module,
  project-internal) and `calls` (resolved statically: same-module names, `self.` methods,
  imported names, `module.attr` on imported project modules, plus a unique-name-anywhere
  heuristic), tagged `extraction_confidence` 1.0 or 0.6. Rows are decay-exempt,
  journal-free (no CodexEvents), embedding-free, alias-free, and share one deterministic
  per-project batch id that is exactly what the retrieval orchestrator's
  `_codex_scope_sets` admits for project-scoped queries.
- **`project_facts.py` (352 lines, E9).** Six deterministic no-LLM parsers producing one
  `project_fact` entity each: dependencies (+resolved versions from `uv.lock`), db_schema
  (ORM tables/columns + alembic head), commands (make targets, npm scripts, entry points),
  config_surface (**key names only — values are never stored**), infrastructure (compose
  services/images/ports), data_shapes (`*schema*.json` top-level keys). Writes are
  content-hash gated; a parser that stops returning anything deletes its stale unit.
- **`reconciler.py` (320 lines, E3 + E11).** Keeps the map in step with reality via three
  triggers (below).

**How it is triggered — all three paths are wired and live:**

1. **On commit (the design).** A consent-installed `post-commit` hook curls
   `POST /user-control/projects/{id}/commit` → `projects_svc.notify_commit` →
   `runtime.notify_work_unit("commit", …)` → the handler registered at
   `src/api/core.py:94-95` → the `project_reconcile` cpu-lane job. If the POST fails the
   hook writes `$GIT_DIR/ice_pending_commit`. **The hook is OFF by default** —
   `install_git_hook=False` (`projects.py:86`); the MCP tool description explicitly says
   to ask the user first.
2. **On a poll (the fallback).** `project_poll` runs every 600 s
   (`config.py:107`), reconciling any project with a marker file or HEAD drift.
   This is on by default and is what makes the core work with no hook.
3. **On a read (E11, on by default).** `freshen_working_tree` is called from the SERVICE
   layer — `graph.where_symbol` (`graph.py:189-192`), `retrieval_svc.context_for` under a
   project scope (`retrieval_svc.py:131-142`), and the arch-doc render (`graph.py:217`) —
   so every adapter inherits it and episodic reads never pay for it. Cheap gate
   (`git status --porcelain`) → 2 s throttle → mtime/size signature short-circuit → the
   same incremental `sync_files`. It never advances `last_reconciled_commit`.

**What a reconcile pass does** (`reconciler.py:99`): changed-file incremental re-parse →
hash-gated fact re-derive → link the range's commits to the active task (falling back to
the newest pending one) → run `decision_cue` over up to 20 commit messages and enqueue
the gpu-lane `decision_extract` for hits → advance `ProjectState`. Unknown base
(force-push/history rewrite) falls back to a full re-parse. An unreachable root sets
`project.settings["unreachable"]`, which `project_status` and the arch-doc render both
surface rather than silently serving a stale graph.

**What is reachable and how:** everything above is reachable through
`ice_control` (`project_register`, `project_reconcile`, `project_status`, `arch_doc`,
`decisions_*`, `task_*`) and through `ice_where`/`ice_context`; registration is also
available as `scripts/register_project.py`. The commit endpoint is the one REST surface.
The payoff surfaces are: `ice_where` (file:line answers), E8 constraint injection into
explicit pulls (`retrieval_svc.py:36`), the E4 where-was-I block on `ice://session-start`,
and the rendered architecture doc.

**What is dormant:**
- `replay_git_log` — accepted, stored, echoed, read by nothing (see DEAD OR INERT).
- The heuristic-ratio tightening rule (`code_graph.py:387-392`) is logged, never enforced.
- Only Python is implemented; the `CodeExtractor` class is explicitly the language seam
  for a future tree-sitter extractor (`code_graph.py:22-24`).
- `origin="project"` is a valid document origin (`documents.py:38`) that no caller ever
  passes — the project↔document link is declared but unused.

---

## DEAD OR INERT

| Thing | Where | Evidence |
|---|---|---|
| `replay_git_log` git-log replay | `src/services/projects.py:86,108,135`; `src/mcp/server.py:368`; `scripts/register_project.py:42` | The flag is written into `project.settings["replay_git_log"]` and echoed in the report. `grep -rn replay_git_log src/ scripts/` returns only the writes — **nothing ever reads the key**. Its own docstring says "honored when F10's importer lands". |
| `tika` and `docling` extraction engines | `src/ingestion/documents/extract.py:23,40-45` | `extract()` raises `NotImplementedError` for any engine that is not `builtin`. Setting `document_extraction_engine` to either makes every document ingest fail. |
| Heuristic-ratio "tighten to imports-only" rule | `src/coding/code_graph.py:387-392` | The ratio is computed and logged; the comment says "Logged, not enforced". No branch consumes it. |
| `origin="project"` document origin | `src/services/documents.py:38` | Present in the `ORIGINS` tuple. Grep across `src/` and `scripts/` finds only `origin="watch_folder"` and the default `"upload"` ever passed. `"paste"` likewise has no caller. |
| `ICE_HOME` override | `src/paths.py:61-63` | Read once at import; unset in this repo. Live only for Track F's packaged app / a relocated install. |
| Non-Python languages in the code graph | `src/coding/code_graph.py:22-24,208` | `iter_python_files` globs `*.py` only; the tree-sitter seam is a comment. |
| `--http` MCP transport | `src/mcp/server.py:503,511` | Implemented, but stdio is the default and the harness path; nothing else in the repo uses streamable-http. |
| Review-queue REJECT over REST | `src/api/routers/user_control.py:176-183` | `approve` has an endpoint, `reject` does not. `review_svc.reject` is MCP-only. |
| `clusters_svc.assign_turns` writes the legacy column | `src/services/clusters.py:28` | Sets `EpisodicMemory.cluster_id`, not `EpisodicClusterLink` rows — the same split C10's deletion code has to handle on both FK paths (`conversations.py:200-208`). A hand-assigned turn may not be visible to link-based cluster logic. |
| `install_git_hook` | `src/services/projects.py:86,122` | Default `False`; the MCP description tells the model to ask first. The 10-min poll is what actually runs unless the user opts in. |

---

## SURPRISES

**More complete than expected**

- **The C10 deletion cascade is exhaustive and honest.** 15 stores enumerated, a manifest
  identical between dry-run and real runs, corroboration analysis so a fact supported by
  another conversation survives, user-authored notes protected from cascade, husking rather
  than hard-deleting entities so the audit trail lives, and a hard-coded caveat string
  admitting logs are not redacted. `conversations.py` is 561 lines for one delete.
- **The document subsystem is a full second ingestion pipeline**, not a file store: 6
  format families, described-not-dumped tables with column statistics, scanned-PDF
  refusal, a model-based document/transcript decision that explicitly obeys the G28
  invariance rule, a live per-conversation enable toggle, an irreversible
  knowledge-promotion latch, and a watch folder wired into the runtime cadence.
- **E11 reconcile-on-read is ON by default and mutates the graph from read paths.**
  `ice_where` and a project-scoped `ice_context` reparse the working tree under a Postgres
  advisory lock before answering. Cheap-gated and throttled, but it means two "read" tools
  are writers — worth knowing for any experiment that counts writes.
- **Three format adapters were reverse-engineered from real exports**, including Claude's
  edit-branch structure and DeepSeek's `current_node`-less tree, each with a
  `branch_messages_skipped` count so nothing silently disappears.
- **`ice-mcp` boots the whole stack itself** (docker compose up, lease-checked runtime,
  standby coordination with the app) — it is a real second entrypoint, not a shim.

**Less complete than expected**

- **REST and MCP have diverged badly.** The graph surface, symbol lookup, arch doc, all
  project/task/decision operations, review-reject, session-start and free-text notes are
  MCP-ONLY; import-start, document-delete, slot-initialize, label correction, cluster CRUD
  and registry refresh/delete are REST-ONLY. Neither adapter is a superset. A Track F UI
  built on REST would be missing the entire coding core.
- **`constraints_for_task` (E8's do-not-touch payoff) fires only on EXPLICIT pulls.**
  Its single caller is `context_for` (`retrieval_svc.py:160`), and the live chat path in
  `main.py` calls the orchestrator directly. A constraint therefore reaches a coding agent
  via `ice_context` or `/search`, but never an ordinary chat turn.
- **`services/documents.py` hosts a runtime job.** `JOBS["ingest_document"]` points at
  `src.services.documents:run_document_ingest` (`runtime.py:111`) — the only place the
  supposedly HTTP-free service layer is also a worker entry point.
- **`clusters.py` is 30 lines** — the thinnest service by an order of magnitude, and it
  writes the legacy cluster column (see DEAD OR INERT).
- **The `full_sync` path re-walks the entire tree twice on registration**
  (`code_graph.py:372` then `project_facts.derive_project_facts`, which does its own
  `rglob("*.py")` twice more in `_parse_db_schema` and `_parse_config_surface`) — four
  full tree walks to register one project.

---

## Corrections — 2026-08-16

The inventory was built 2026-08-15 and this session changed or disproved several
entries. **Where this section conflicts with the tables above, this section
wins; where it conflicts with the code, the code wins.**

| Entry above | Correction |
|---|---|
| Per-leg attribution readable only inside the coverage path | **Fixed.** `producing_legs` now rides on `leg_budget_share`, which fires on every retrieval. Also completes G46's second half — `_apply_rrf` stamped only the FIRST leg, so `bm25` (first in the dict) masked vector; legs now join as `bm25+vector`. What read as `{'bm25': 4}` is really `{'bm25+vector': 4}`. |
| Maintenance agent Tier-0 name-collision channel is structurally impossible | **Fixed.** Replaced with `merge_key()` — casefold, split digit/letter runs, collapse separator punctuation, **token order preserved**. 20 real groups on the seeded arm. Sorting tokens adds 11 groups of which two are converses. |
| `codex_node_promotion` default OFF | Still OFF, and now known **inert**: 0 `entity_merged` events across a 586-turn run. Every promotion candidate was an in-flight endpoint; the guard fired 98 times. |
| `PROPERTY_RELATIONS` (51 words) described as a gate | **Not a gate.** 50 edges use property relations against 9,602 open-vocabulary ones, and open-vocab facts render identically in the context payload. It is a storage *style*. The earlier claim that it destroyed facts is withdrawn. |
| Relation supersession | **Changed.** `handle_triplet` superseded whenever a relation was absent from `MULTI_VALUED_RELATIONS`, so every open-vocabulary relation retired its predecessor. Now supersedes only for relations *known* single-valued. |
| `codex_relation_canonical_threshold` 0.82 | **Now 0.90**, and `known_relations()` is fed forward within a turn. |
| Procedural `is_active` requires `reinforcement_count >= 3` | **Second path added:** `procedural_min_cited_turns` (10). 29 of 61 patterns on the existing arm would activate, against 1 today. |
| `is_unusable_entity_name` refuses non-alphabetic names | **Changed.** Now refuses only names with no letter **and** no digit. It was deleting 94 numeric entities per run — years, GPAs, scores. |

### New in this session

| Feature | Where | Roadmap id | What it does | Setting | Default | On by default? |
|---|---|---|---|---|---|---|
| Fragment origin provenance | `src/retrieval/orchestrator.py:71` | G48b | Carries the turns a codex/procedural fragment was derived from, so recall can credit legs other than episodic. Timeline still unwired. | — | — | YES |
| Clause-shaped relation demotion | `src/workers/codex_extractor.py` | G49 | A relation longer than N words is demoted, never dropped. | `codex_relation_max_words` | `5` | YES |
| Deterministic entity merge key | `src/workers/maintenance_agent.py` | G50 | Tier-0 auto-merge of spelling variants with no model and no review. | — | — | YES |
| In-flight endpoint protection | `src/workers/codex_extractor.py` | G44 | Stops promotion deleting the endpoint of the triplet being written. Fired 98 times in 586 turns. | — | — | YES |
| Z2-mini answer harness | `scripts/z1/answer_probes.py` | Z2 | Retrieve → assemble → generate an answer. The half no recall number measures. | — | — | N-A |
| Paired blind judge | `scripts/z1/judge_answers.py` | Z2 | Head-to-head arm comparison with a fixed reason taxonomy; writes partial state after every probe. | — | — | N-A |

| Bounded reconciliation source | `src/workers/codex_extractor.py::make_llm_reconciler` | G62 | Complete source input or explicit review; exact complete decision parsing. | `codex_reconcile_input_tokens` | `8192` | YES |

| Explicit conflict resolution | `src/services/review.py::approve`, REST review approval, MCP `ice_control` | G62 | `keep_edge_ids` explicitly retains both/one/neither; validates pair, journals expiries, refreshes payloads. | — | — | YES |

| Writer-supplied speaker boundaries | `src/memory/source.py`, API/import/document/note writers; recent prompt reader | G76 | Raw hash plus role offsets prevent quoted role markers from changing authorship. Legacy/malformed metadata yields unknown. Preserved through cold lifecycle. Attributed sentence extraction consumes these boundaries. | — | — | YES for new writes/raw recent reads |

### v3 selected-context retention (2026-09-13)

| Feature | Implementation | Control/default | Active |
|---|---|---|---|
| Exact fact exposure | `memory/usage.py`, chat after evidence eviction and explicit context service; unique rendered `origin_edge_ids` increment usage and last-access time | `retrieval_strengthen_writes=True`, `codex_retention_increment=0.15`, `codex_retention_cap=10` | YES; prepared/returned context, not answer use; cached note-only edges not attributed yet |
| Bounded retention ranking | `_edge_trust`: extraction quality gates entry, then retention/recency ranks; quiet supported edges remain eligible | `codex_retention_rank_weight=0.5` | YES; not calibrated truth |
| Distinct source observation | `_observe_edge` ignores repeated original/observed batch; another batch may promote pending status | `observed_batches`, separate from usage | YES; source-role independence still requires claim repair |

### v3 attributed sentence claims (2026-09-14)

| Feature | Implementation | Control/default | Active |
|---|---|---|---|
| Source sentences independent of entity recognition | `CodexClaim`, `CodexClaimLink`; extractor template includes exact `source_sentence`, role units processed independently, paragraph context retained; native1024 embedding + lexical index | `codex_sentence_claims=True` | YES for new extraction; no automatic legacy backfill |
| Direct claim search | `orchestrator._codex_claims` in normal and wide-net paths, before global reranker/packing; source/hash/privacy/scope checked | `codex_claim_candidate_limit=64`; existing RRF constant | YES for warm/cold sources; cold cluster-scoped claims withheld until archive membership is preserved |
| Source-support compression | `memory/support.py`, `claims.store_claims` / `claim_representation`; source and claim hashes + pinned verifier; uncertain source stays whole | `source_support_threshold=0.95`, max tokens512, device auto | YES for source-sentence shortening; not yet summary or conflict verification |
| Claim deletion | Conversation FK cascade; explicit turn-forget by stable episodic ID; absent/edited sources never render | shared conversation/forget services | YES; archive retains claims and reader resolves warm state first |
| Foreground/extraction model separation | `extract_codex` does not pass foreground `model_used` as extraction override; explicit arm overrides remain in `extract_triplets` | existing specialist setting | YES |

| Graph source rendering | `orchestrator._fact_line`, `_render_codex_entity`, tag enumeration; linked attributed evidence replaces cached relationships; exact rendered edge IDs; negative facts remain non-navigable | Existing graph scope/time/trust controls | YES; legacy relations/notes explicitly unverified |
| Cold restoration preserves evidence | `_resurrect_cold_hits`; archived vector reused, NULL vector retained with warning; unknown timestamp provenance remains unknown | `retrieval_strengthen_writes=True` | YES for selected cold hits with conversation identity |

### v3 turn representation support (2026-09-19)

| Feature | Implementation | Setting/default | On by default? |
|---|---|---|---|
| Independent summary and abstract verification | Post-flight stores `representation_verification`; shared chat/MCP/retrieval readers check canonical role source hash, candidate hash, model version and score | Existing `source_support_threshold=0.95`, complete pair max512 tokens | YES for newly evaluated turns; legacy missing verdicts use raw |
| NLI uncertainty preserves evidence | Overlength, missing authority, model error or unsupported summary cannot replace original; generated metadata remains stored | Shared support verifier | YES; long-source/fold compression remains unfinished |
