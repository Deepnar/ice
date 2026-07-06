# ICE Post-Paper Roadmap

> Living checklist of everything queued after the paper. Distilled from [rough_post_paper_work.md](rough_post_paper_work.md) (raw notes — keep that file as the source of nuance), the experiment reports in `experiments/*/results*`, and the paper's limitations section.
>
> **How to use this file:**
> - Each entry is *intent + rationale*, **not a spec**. When an item's turn comes, the concrete design gets discussed in-session first; only then is it implemented.
> - Check off `[x]` when done. On completion, update [ICE_Architecture.md](ICE_Architecture.md): new systems get a new section, reworked systems get their existing section edited.
> - Phase 0 decisions may kill, reshape, or re-home items below (some conversational features may move to the Coding side). That is expected.

**Why-tags used below:** `(planned)` designed but never implemented · `(rework)` exists but the current version is bad/underperformed in experiments · `(bug)` broken behavior found in audit/experiments · `(new)` new direction from post-paper thinking · `(open)` limitation with no settled solution yet.

---

## Phase 0 — Ablation review & pruning (gate for everything else)

- [ ] **P0.1 Review ablation results and sentence every feature** `(rework)` — Go through `experiments/flaw_ablation/buildup/paper_summary.md` + subtraction results and classify each feature: genuinely useful (keep/polish), neutral (simplify or remove), harmful (cut or rework), or better suited to the Coding side (move). Notable inputs: HyDE scored *worse* with than without (4.03 vs 4.20) and is already commented out in production; MERA's −0.21 is downstream of NER failure, not MERA itself; Codex contributed only 3.3% of fragments but under multiple simultaneous handicaps (floor, not ceiling).

---

## Track A — Codex overhaul (highest-leverage rework)

The experiments showed Codex is the most ambitious *and* most handicapped subsystem. These items are roughly ordered.

- [ ] **A1 Extraction chunking fix** `(rework)` — Replace 6,000-token extraction chunks with ~512–800-token chunks (~50-token overlap, dedup across overlaps). Why: the "chunking paradox" — a 3B/4B extractor's attention dilutes past ~1k tokens, causing entity dropping and hallucinated triplets (`fastapi uses fastapi`).
- [ ] **A2 NER-grounded extraction (hybrid CPU/GPU pipeline)** `(rework)` — CPU micro-NER (512-token chunks, 20–50 overlap) produces the confirmed entity list; the GPU LLM only maps relations *between those entities*; relations naming entities NER didn't confirm are rejected as hallucinations. Also fix the NER model not reliably loading (the regex fallback misses lowercase/multi-word entities — a root cause of Codex's weak numbers).
- [ ] **A3 Confidence-calibrated edges (fix the corroboration trap)** `(rework)` — Replace binary `pending`/`active` with stored extraction confidence + dynamic thresholds at retrieval. Why: extraction runs ~once per fact, so edges rarely get the 2 corroborations needed to go `active`; the graph starves itself of true knowledge (seen in the 1,119-turn Flaw run).
- [ ] **A4 Relation-aware retrieval + NER/relations retrain** `(rework)` — Search by relation as well as entity ("who inspired X", "name"-relation queries), boosting where relation hits and entity hits overlap. Closes the semantic-vs-lexical gap ("main fortress" never resolves to "The Obsidian Citadel" today).
- [ ] **A5 Codex conversation-scoping division** `(bug)` — Massive technical conversations (ice_dev) pollute the shared graph; extraction/retrieval need per-conversation/project scoping inside Codex.
- [ ] **A6 Self-correcting graph (reconciliation loop)** `(new)` — Before saving, `check_conflict()` against existing edges; on conflict, re-prompt the extractor to reconcile (merge/expire/update) instead of blind-saving. Turns the extractor from passive triplet collector into an active state reconciler. (Pairs with Track D tooling.)
- [ ] **A7 Codex V3 — one unified, code-aware graph** `(new)` — Add a deterministic static-analysis layer: AST-level entities (deterministic IDs from definition sites), deterministic edges (`imports`, `calls`, `inherits`, `defined_in`, `tested_by`), and Graph-RAG-style community summarisation (Louvain/Leiden over the import/call graph, LLM summary per community). **Constraint from the notes: there is ONE Codex** serving both conversation and coding modes — not two graphs. OKF is adapted here as design philosophy (typed knowledge units in our tables, not markdown files).
- [ ] **A8 Expand codex relations** `(new/open)` — The current set arent that diverse, for example there is not negetive version of exisiting relations so the ai chooses the positive instead, also we cant just make it wayy bigger as that also counts into what is being sent to the model.

## Track B — Classifier & routing

- [ ] **B1 Classifier retrain: multi-label context-reliance + cloud toggle** `(rework)` — Real_Time_Search is orthogonal (a Zero_Shot or LTM prompt can *also* need real-time), so context-reliance shouldn't be single-label 3-way. Add a toggleable "cloud" routing signal for users with API models. The Coding side may need new intent labels too.
- [ ] **B2 Replace the hard LTM override with a principled combination** `(rework)` — The >10-turns / low-confidence ⇒ LTM override is a loose fix that masks the classifier (system is "safe, not smart"). Treat classifier confidence as a prior combined with a conversation-length prior (Bayesian-ish) instead of a hard threshold.
- [ ] **B3 Learned MoE routing** `(rework)` — Current router is a hardcoded overlap scorer; empirically neutral-to-harmful (≈−0.02 Exp2, −0.04 Exp1) and blind to confidence and context-reliance. Wanted: confidence-weighted routing, Zero_Shot→small-model / LTM→big-model policies, a lightweight bandit over empirical per-topic performance, and model-load integration (preload/keep-warm to kill the 5–15 s Ollama swap spike). LSREP replays can train the policy offline.
- [ ] **B4 Feedback loop + fine-tune promotion (close the broken loop)** `(bug/planned)` — Thumbs-up/down per response: up ⇒ auto-add to curated set; at ~100 downs, offer to auto-label with a strong model, then user reviews via toggle-button UI; feature OFF by default for inexperienced users. Critically, fix promotion: today `fine_tune.py` loads hardcoded `v2_final.pt`, writes `finetuned_{ts}.pt`, and the live path is `v3_qwen_ft3.pt` — the weekly fine-tune changes nothing. Auto-promote the new checkpoint to `settings.classifier_model_path` (with restart/reload).
- [ ] **B5 Ensemble classifiers** `(open — parked)` — Soft-voting over 2–3 heterogeneous classifiers. Explicitly an *extreme-scenario-only* option if retraining can't hit accuracy targets.

## Track C — Retrieval & memory quality

- [ ] **C1 Lossless / inject_raw rework + better summarisation** `(rework)` — Currently effectively everything lossless injects raw; need a real dynamic decider between raw vs summary, and higher-quality summaries.
- [ ] **C2 Big-input handling** `(rework)` — Massive pasted inputs currently go in whole (is_document bypasses the word cap). Chop at ingestion; inject only relevant chunks, never the entire doc.
- [ ] **C3 Extreme-density retrieval (ICE-Dev follow-up)** `(new)` — Chunk-aware retrieval (retrieve chunks, not turns), hierarchical summaries (abstract → medium → full), section-level document decomposition, smarter truncation. Why: the vector baseline collapsed to 94.2% failure on ICE-Dev because giant turns blew the context window; ICE won there (4.33 vs 1.23) but the same failure class threatens ICE at higher density.
- [ ] **C4 Whole-conversation batch summary** `(planned)` — A conversation-level summary object so a model answering in isolation still gets global conversation info — prerequisite if the sliding window is ever removed/shrunk.
- [ ] **C5 Clustering fix** `(bug)` — Exp2 showed over-merging (2–3 mega-clusters + dozens of singletons, sometimes one giant cluster) and top-30→top-10 cluster scoping that defeats the purpose. Fix merge thresholds, cluster count, and name/description quality (needed for cross-chat UX in C6).
- [ ] **C6 Session IDs + scoping semantics rework** `(rework)` — Introduce a session_id per sitting (turns in one session are likely same-topic ⇒ better clustering; also the hook for decay catch-up, C7). Rethink scopes: `none` should behave like true incognito; `manual` becomes a mode *inside* auto/project rather than a peer; user can toggle extra conversations into scope from the sidebar (cross-chat retrieval over N ticked convos) and `@`-mention a specific conversation/turn for one-off context.
- [ ] **C7 Decay/worker scheduling rework** `(bug/open)` — All decay depends on the app staying open; a user who closes it gets no maintenance. Run catch-up at session start based on gap since last session (msgs + elapsed time + calendar spread of mini-sessions). Decay is cheap enough for session-start; **open question:** where to put fine-tuning, which is expensive even on strong hardware.
- [ ] **C8 Time-weighted episodic retrieval** `(planned — F9)` — Architecture specifies time-weighted cosine; implement the recency weight in the vector score itself.
- [ ] **C9 Procedural memory + slots rework** `(rework)` — Procedural is gated to 3 intents and contributed ~nothing (+0.02); widen/rethink so people actually feel it. Slots: split into global slots vs per-conversation slots; allow agentic slot updates from chat (ties to D1/C11). (Also noted: the probes may simply not have exercised procedural — an eval problem, but the feature still needs to earn its place.)
- [ ] **C10 Deletion feature** `(planned)` — Conversation deletion with correct cascade semantics: episodic rows deleted; Codex edges solely supported by that convo demoted/expired (corroborated ones stay); procedural `source_batch_ids` pruned (deactivate if orphaned); batch summaries + session replays deleted.
- [ ] **C11 User memory control through chat** `(planned)` — "add X to pending items", "search specifically for Y" — chat-level commands that write slots / steer retrieval directly.
- [ ] **C12 RAG leg completion** `(planned)` — Proper document pipeline (PDF, CSV, …), per-conversation uploads, and a sidebar way to add an existing document into another conversation's scope.
- [ ] **C13 Caching strategy** `(open)` — Never designed at all. Decide what to cache (embeddings, cluster scores, classifier outputs, hot fragments) and whether Redis or in-process.
- [ ] **C14 KV-cache persistence & cache-aware retrieval** `(open — long-term)` — Stable-prefix ordering is right but empirically only ~10–15% of the prompt is cacheable today. Investigate persistent prefix KV storage, cache-aware fragment selection, incremental prefix updates — depends on backend (Ollama/vLLM/SGLang) cache APIs.
- [ ] **C15 Wide-net fallback budget** `(bug)` — The 2,000-token hardcoded ceiling ignores conversation length/available budget; make it dynamic.

## Track D — Active ICE (agentic background maintenance)

- [ ] **D1 Memory Maintenance Agent** `(new)` — Replace the fixed extract→save pipeline with a small tool-using agent (3B/4B, guided decoding) running in idle GPU time. Toolbox: `update_entity_relation`, `merge_conflicting_entities`, `reconcile_graph_state`, `flag_for_review`, `run_cluster_consolidation`. Agent decides *whether/how* to modify memory; review queue stays as the safety net for high-uncertainty actions. NER (A2) is the grounding anchor.
- [ ] **D2 Sentinel completion + agent integration** `(planned/bug)` — Implement the declared-but-missing `frequency`, `contradiction`, `composite` triggers and the `propose_memory_update` action; have the agent subscribe to Sentinel events and *resolve* them instead of just logging.
- [ ] **D3 Agentic telemetry** `(new)` — Expose the agent's decisions/reasoning through SSE so its actions are visible (feeds Track F telemetry).

## Track E — Coding Mode / Project State Engine

The second operational profile: same frontend, proxy, classifier, registry, and worker infra; different stores and pipeline. Gated by `memory_scope_type == "coding"` on the conversation.

- [ ] **E1 Project State Engine schema** `(new)` — Tables: `project_state` (goal, branch, last task/session), `architecture_clusters` (feature-based file groupings), `decisions` (temporal versioning à la codex_edges), `tasks` (with commit hashes + files_changed), `development_patterns`, `daily_checklist`. Design against Codex V3 (A7) so code knowledge lands in the *one* unified graph — the rough notes explicitly say to rethink these tables in cohesion with the revamped Codex/OKF philosophy, not to blindly build them as drafted.
- [ ] **E2 Coding-mode routing** `(new)` — Branch in `chat_completions` on the coding scope; coding-specific intent labels for the classifier (ties to B1).
- [ ] **E3 State-reconciliation agents** `(new)` — Coding counterparts of the workers: State Reconciler (post-session git-diff scan → clusters/decisions/tasks), Dependency Tracker (import graph, circular-dep flags), Stale Work Detector, Pattern Extractor.
- [ ] **E4 Session-start checklist flow** `(new)` — On coding-session start: restore project_state, `git diff --stat` since last session, surface pending/stale decisions+tasks, assemble the "welcome back" context block.
- [ ] **E5 Executor decision: own harness vs Aider** `(open — decide before building)` — The notes lean toward building our own executor designed for our system rather than adapting to Aider's; either way, hide it behind a thin adapter (`plan(prompt, files) → PlanResult`, `execute(plan, files) → DiffResult`) so the executor is swappable (same pattern as `bg_client_factory`). Plan/Act split is a requirement regardless.
- [ ] **E6 OKF-inspired knowledge units** `(new)` — Adapt OKF's typed-knowledge-document philosophy (type/title/description/resource/tags) *into our tables/Codex*, not as markdown files. Applies to both coding tables and Codex entity payloads.

## Track F — Frontend & UX (custom frontend era)

- [ ] **F1 Custom web frontend foundation** `(planned — F5 in the old missing-items table)` — The big one; all user-control APIs already exist, the frontend surfaces them. Everything below rides on it.
- [ ] **F2 Review-queue panel** `(bug-adjacent)` — Reflection's slot/cluster proposals currently rot in `review_queue` forever because nothing renders it. Highest-value early frontend piece.
- [ ] **F3 Graph view of Codex** `(planned)` — Obsidian-style interactive graph; user can inspect, edit, and delete entities/edges manually.
- [ ] **F4 Full settings exposure** `(planned)` — Every knob editable in-UI: dynamic budget max, max input/output tokens, temp, top-p/k, thresholds — no file editing or raw API calls needed.
- [ ] **F5 Telemetry & forensics layer** `(planned)` — Expanded SSE event set (system has outgrown the current 5), real-time retrieval attribution (which leg, which bonuses, why), memory provenance (source_batch_id back-links), model thinking visibility (opt-in), and a subtle background-workers activity indicator.
- [ ] **F6 Select-text → add-to-context** `(planned)` — Select part of an answer and pin it as context for exactly the next prompt.
- [ ] **F7 Real-time search integration** `(planned — likely adapt existing tooling)` — Web search so the Real_Time_Search label actually routes somewhere.
- [ ] **F8 Deep research mode** `(planned — likely adapt existing tooling)` — Deep-research flow whose outputs are ingested through our own stores.
- [ ] **F9 Feedback UI** `(planned)` — Thumbs up/down + label-correction toggles (frontend half of B4).
- [ ] **F10 Conversation import (LSREP as migration tool)** `(new — flagship)` — Expose the replay pipeline as user-facing ingestion: drop exported ChatGPT/Claude/DeepSeek logs, system *lives through* them (post-flight, Codex, procedural, clustering, reflection) producing a mature memory state, not a searchable archive. Decay policy choices at import: preserve (decay_immune window), simulate natural decay (fast-forward), or start fresh.
- [ ] **F11 Cloud API models** `(planned)` — Let users register cloud API models alongside local ones (pairs with B1's cloud toggle; local-first stays the default posture).
- [ ] **F12 Multi-model responses** `(new)` — When a prompt is genuinely dual-natured (e.g. emotional + coding), consult two specialists and compose.
- [ ] **F13 Session replay + conversation branching** `(planned — F2/F4 old table)` — `session_replays` table is written by nothing today; branching retrieval logic deferred until the frontend exists.
- [ ] **F14 Raw log extraction v2** `(rework)` — For unformatted text dumps: word-boundary slicing with an in-session overlap-resolution pass (both slices + overlap asked about in one open session), replacing the amnesia-method cold chunks; dedup at the end.

## Track G — Ops, bugs & hardening (the "forgotten problems")

- [ ] **G1 Classifier model-path mismatch** `(bug — critical)` — `fine_tune.py` and `drop_zone.py` hardcode `v2_final.pt` while the live path is `v3_qwen_ft3.pt`; fine-tune output goes nowhere. Single source of truth via settings + a promotion step (subsumed by B4, but the path unification can land first).
- [ ] **G2 Background-model client is dead-code dedicated** `(bug)` — `get_bg_client()` hardcodes SGLang :8001 (the 14B user-facing model); the dedicated :8002 path is commented out. Background work chews user VRAM, undermining the GPU-gating philosophy. Restore mode selection via `background_model_mode`.
- [ ] **G3 Runtime shared↔dedicated switching** `(planned)` — Switching background-model mode currently requires a full restart; make it hot-switchable.
- [ ] **G4 GPU gating fixes** `(bug)` — `GPU_UTIL_THRESHOLD = 20` is far too aggressive (LLM boxes idle at 10–30%) → workers starve and queues build; raise to ~60–75 and cache the nvidia-smi poll instead of spawning a subprocess per task.
- [ ] **G5 SSE stream resiliency** `(bug)` — Partial/unclosed JSON chunks from Ollama are silently dropped in the stream parser → truncated `raw_text` stored → degraded extraction downstream. Guard the parse / use a partial-JSON parser.
- [ ] **G6 DB indexes via migrations** `(bug)` — `scripts/database/create_indexes.sql` isn't applied by Alembic; `batch_id` lookups full-scan episodic_memory at scale. Fold indexes into a migration.
- [ ] **G7 Idempotency enforcement** `(bug)` — `episodic_memory.idempotency_key` has no unique constraint (informational only); enforce it.
- [ ] **G8 Sticky-state persistence** `(bug)` — `SESSION_STATE` is an in-process dict; model stickiness resets on restart. Back it with Redis.
- [ ] **G9 Constants → configuration** `(rework)` — Decay rates, RRF k, bonus multipliers, token budgets etc. are module-level constants; tuning requires code edits. Move the tunable ones into settings.
- [ ] **G10 Compaction scheduling** `(bug)` — Beautifully written, never runs: `compact_entities` isn't beat-scheduled, so `codex_events` grows unbounded. Schedule it (beat or sentinel-triggered).
- [ ] **G11 Batch-summariser coverage** `(bug)` — Only decayed turns are summarised; very old but undecayed turns in long conversations never compress.
- [ ] **G12 Dynamic LLM timeouts** `(bug)` — Hardcoded 30 s timeouts cascade into retry storms in shared mode; scale timeout with max_tokens / load.
- [ ] **G13 Drop-zone duplicate classifier** `(bug)` — `drop_zone.py` instantiates its own `PyTorchClassifier` instead of sharing the embedder — doubles memory during ingestion.
- [ ] **G14 Memory-slot token budget enforcement** `(planned — F7 old table)` — Truncate slots that exceed ~300 tokens.
- [ ] **G15 Null_Noise / Casual_Banter routing** `(planned — F6 old table)` — Minor special-casing for noise labels.
- [ ] **G16 None-scope isolation** `(planned — F11 old table)` — Guarantee None-scoped conversations are invisible to all other retrieval (prerequisite for C6's "incognito").
- [ ] **G17 Audit trail** `(planned — F3 old table)` — Annotate every memory write with its source (user / post_flight / codex_extractor / reflection / manual_injection / sentinel / bookmark); queryable + exportable. Feeds F5 forensics.
- [ ] **G18 Celery observability** `(rework)` — Result backend is configured but unused; no way to inspect task outcomes outside the DB. Decide: use it properly or drop it and log outcomes.
- [ ] **G19 Simulation-harness upkeep** `(planned — F8 old table)` — Add procedural extraction to the replay loop and log runs to a `simulation_runs` table for reproducibility.

## Track H — Research follow-ups & open questions

- [ ] **H1 Cross-conversation retrieval evaluation** `(open)` — All 1,211 probes were within-conversation; the scoping/cross-convo machinery (and C6) has never been measured.
- [ ] **H2 Multi-user evaluation** `(open)` — All benchmark conversations are one user's; generalisation across users is unvalidated.
- [ ] **H3 Year-scale memory studies** `(open)` — 93 days max simulated so far. Saturation, retrieval drift, decay convergence/cold-start (does everything but bookmarks decay to zero?), compaction cadence.
- [ ] **H4 Probe realism** `(open)` — LLM-generated probes under-represent anaphoric/ambiguous human questions; grow the manually-authored probe set.
- [ ] **H5 Fine-tune scheduling on user machines** `(open)` — No good answer yet for when to run expensive fine-tunes for users who close the app (tail of C7).
