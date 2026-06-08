# ICE — Project Progress
> Last updated: 2026-06-07

---

## Current Phase & Step
Phase 8 — Memory Slots
---

## Last Completed
- `src/retrieval/orchestrator.py` — Hybrid Retrieval Orchestrator implementing BM25 full‑text retrieval, pgvector cosine lookup, Codex graph traversal, Procedural pattern lookup, RAG chunk retrieval, and true Reciprocal Rank Fusion (RRF) with session diversification, deduplication, and token‑budget enforcement — does NOT include web search integration, production tuning, or multi‑tenant rate‑limiting.

---

## File Inventory
*(Append-only — never delete entries)*
- `scripts/classifier/promt_labeling/validate_promt.py` — validates labeled prompts by analyzing label frequencies, co-occurrences, and data completeness — complete
- `data/unlabled/personal_promts.jsonl` — extracted personal prompts from raw chat logs — raw, no labels
- `data/unlabled/wildchat_promts.jsonl` — 5k WildChat first-turn prompts — raw, no labels
- `data/unlabled/lmsys_promts.jsonl` — 5k LMSYS first-turn prompts — raw, no labels
- `data/unlabled/sharegpt_promts.jsonl` — 5k ShareGPT human turns — raw, no labels
- `data/unlabled/dataset_unlabeled.jsonl` — merged and deduplicated, ~19,710 rows — no labels
- `data/unlabled/dataset_cleaned_filtered.jsonl` — cleaned version after filtering edge cases — no labels
- `data/labled/labeled_prompts.jsonl` — 19,710 labeled rows with topic, intent, context_reliance fields — complete for Paper 1 training
- `data/labled/failed_prompts.jsonl` — rows that failed labeling — not used for training
- `scripts/classifier/promt_extraction/extract_promts.py` — Amnesia Method extraction from raw chat logs — complete
- `scripts/classifier/promt_extraction/wildchat_extractor.py` — WildChat source extractor — complete
- `scripts/classifier/promt_extraction/lmsys_extractor.py` — LMSYS source extractor — complete
- `scripts/classifier/promt_extraction/sharegpt_extractor.py` — ShareGPT source extractor — complete
- `scripts/classifier/promt_extraction/combine_dataset.py` — merges four sources into one JSONL — complete
- `scripts/classifier/promt_extraction/clean_dataset.py` — deduplication and edge case filtering — complete
- `scripts/classifier/promt_labeling/VLLM_label_dataset.py` — async parallel labeling via vLLM + instructor — complete, using 7B not 70B
- `scripts/classifier/promt_labeling/prune_failed_promts.py` — removes failed rows from output — complete
- `scripts/classifier/promt_labeling/compare_labeling.py` — compares labeling outputs for quality checks — complete
- `docs/BLUEPRINT.md` — step-by-step build guide — reference only, not a contract
- `docs/ARCHITECTURE.md` — full system design and invariants — authoritative
- `scripts/classifier/promt_labeling/generate_synthetic_data.py` — generates synthetic prompts based on provided labels — complete
- `scripts/classifier/promt_labeling/synth_promt_gen_number.csv` — defines the number of synthetic prompts to generate for each label combination — complete
- `scripts/classifier/promt_labeling/synth_promt_renumber.py` — renumbers synthetic prompt IDs in a JSONL file — complete
- `scripts/classifier/promt_labeling/validate_promt.py` — validates labeled prompts by analyzing label frequencies, co-occurrences, and data completeness — complete
- `data/curated_fixes.jsonl` — contains curated fixes for training — complete
- `scripts/training/build_training_data.py` — builds training data from labeled prompts — complete
- `scripts/training/fine_tune.py` — fine-tunes the classifier model on curated fixes — complete
- `scripts/training/test_classifier.py` — tests the classifier with various prompts — complete
- `scripts/training/train_classifier.py` — trains the classifier model — complete
- `models/classifier/ice_classifier_v2_final.pt` — final trained PyTorch classifier weights (saved artifact from latest run) — does NOT include TorchScript optimization or deployment wrapper
- `models/classifier/training_runs.jsonl` — JSONL log of training runs, seeds, and validation losses — does NOT provide full experiment metadata or remote tracking
- `scripts/training/train_classifier.py` — training script for the ICE classifier — does NOT handle multi‑GPU/distributed training or CI integration
- `scripts/training/fine_tune.py` — fine‑tuning script that applies curated fixes to a checkpoint and saves an updated state dict — does NOT implement large‑scale distributed fine‑tuning or third‑party experiment tracking
- `scripts/training/build_training_data.py` — converts labeled JSONL into vectorized training examples and writes the label schema — does NOT perform label cleaning or automated repairs
- `scripts/training/test_classifier.py` — smoke/integration tests that run the model on hard prompts and print predictions — does NOT run as a unit/CI test suite
- `src/classifier/model.py` — PyTorch MLP architecture (`ICEClassifier`) used for training and inference — does NOT include model export utilities (TorchScript/ONNX)
- `src/classifier/dataset.py` — dataset loader that precomputes `all‑MiniLM‑L6‑v2` embeddings for training_data.jsonl — does NOT persist embeddings to disk or support streaming very large datasets
- `src/classifier/classifier.py` — `PyTorchClassifier` inference wrapper and `ClassificationResult` dataclass — does NOT provide a network API (FastAPI) or batch inference endpoint
- `scripts/classifier/promt_labeling/synthetic_data.py` — async synthetic prompt generator (uses local model endpoint / OpenAI‑compatible client) — does NOT include robust retry/backoff or production authentication handling
- `scripts/classifier/promt_labeling/synth_promt_gen_number.csv` — CSV defining how many synthetic prompts to generate per label combination — does NOT validate or normalize counts
- `scripts/classifier/promt_labeling/synth_promt_renumber.py` — renumbers `synth_` IDs in a JSONL file — does NOT merge conflicting IDs safely (simple overwrite approach)
- `scripts/classifier/promt_labeling/validate_promt.py` — dataset validation and label frequency/co‑occurrence reporting — does NOT auto‑fix detected data issues
- `data/curated_fixes.jsonl` — curated fixes used for fine‑tuning (sampled lines shown in repo) — does NOT claim to be exhaustive; used as high‑signal correction set
- `data/labeled/labeled_prompts.jsonl` — labeled prompt dataset (first lines inspected) — does NOT include the full file here due to size; canonical labeled dataset lives in `data/labeled/`
-- `alembic.ini` — Alembic configuration (local dev `sqlalchemy.url` set to the ICE dev DB) — does NOT include production credentials and requires environment-specific values.
- `alembic/README` — brief note about the single-database Alembic configuration — informational only.
- `alembic/env.py` — Alembic environment script wired to `src/memory/models.py` `Base.metadata` for autogenerate support — does NOT perform runtime migration steps or manage connection secrets.
- `alembic/script.py.mako` — Alembic migration template used to generate revision files — template only; does NOT change migration semantics.
- `alembic/versions/675b74e56988_initial_schema.py` — autogenerated Alembic revision implementing the initial schema (see Last Completed) — does NOT run without invoking Alembic migration commands.
- `scripts/database/create_indexes.sql` — SQL script to create HNSW pgvector indexes on embedding columns (`episodic_memory.embedding`, `procedural_memory.embedding`, `rag_chunks.embedding`) — does NOT run automatically; intended for post-migration execution.
- `docker/docker-compose.yml` — Compose file to run `pgvector` Postgres and `redis` for development — expects environment variables and does NOT include production hardening.
- `src/memory/models.py` — SQLAlchemy declarative models matching the DB schema (EpisodicMemory, CodexEntity, CodexEdge, ProceduralMemory, RAG models, MemorySlot, etc.) — does NOT include DB session wiring, migration orchestration, or admin utilities.
- `pyproject.toml` — project manifest updated with runtime dependencies required by ICE (alembic, pgvector, sentence-transformers, vllm, etc.) — does NOT lock installation environment; use `uv.lock` for resolved versions.
- `uv.lock` — lockfile for the `uv` package manager listing resolved package versions — does NOT replace environment reproducibility guarantees beyond the lock semantics.
- `.gitignore` — updated ignore rules to exclude local env, data dumps, and model artifacts — does NOT affect already-tracked files.
- `src/api/main.py` — FastAPI middleware proxy that loads the `PyTorchClassifier`, exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`), streams responses from Ollama, and schedules background storage of episodic turns — does NOT implement authentication, background worker orchestration, or production-grade error handling.
- `src/api/db.py` — SQLAlchemy engine and `SessionLocal` session factory plus `get_db` FastAPI dependency used by the proxy — does NOT provide async DB support, migration orchestration, or production connection tuning.
- `src/api/config.py` — `pydantic_settings`-backed `Settings` (database_url, redis_url, ollama_base_url, classifier thresholds, model paths); reads `.env` — does NOT manage secrets or multi-environment profiles.
- `src/classifier/classifier.py` — updated `PyTorchClassifier` wrapper that loads `ICEClassifier` weights, initializes `SentenceTransformer` embedder, and returns `ClassificationResult` (topic/intents/context + confidence) — does NOT provide batch inference, GPU offload, or a network inference endpoint.
- `pyproject.toml` — project manifest updated to include FastAPI and runtime dependencies (fastapi, httpx, uvicorn, structlog, pydantic-settings, sse-starlette) required by the proxy — does NOT pin platform-specific extras or CI/test deps.
- `uv.lock` — updated `uv` lockfile recording resolved dependency versions after adding proxy dependencies — does NOT guarantee cross-platform reproduction of binary wheels.
- `src/workers/compaction.py` — compresses append-only `CodexEvent` ledgers into `CodexSnapshot` snapshots and marks events compacted; does NOT handle cross-shard compaction, long-running maintenance windows, or advanced conflict resolution.
- `src/workers/codex_extractor.py` — Celery task that extracts subject‑relation‑object triplets using a background LLM, materialises `CodexEntity`/`CodexEdge`/`CodexEvent` rows, and writes `IdempotencyKey` entries to enforce worker idempotency — does NOT manage model hosting, rate‑limiting, or multi-turn reconciliation logic.
- `tests/test_full_pipeline.py` — integration test exercising classifier → storage → Post‑Flight → Codex Extractor flow; requires DB, Celery worker, and background model to run — does NOT run in CI without environment orchestration.
- `tests/test_direct_codex.py` — direct Codex extractor smoke test that runs extraction and applies triplets to the DB — does NOT mock external model responses.
- `tests/test_triplet.py` — unit test for triplet extraction parsing and JSON fallback handling; exercises raw model output parsing — does NOT assert DB side effects.
- `tests/test_codex_extractor.py` — integration test inserting a high‑value turn and enqueuing `evaluate_turn` to validate Codex extraction behavior — does NOT run headlessly without Celery and DB.
- `scripts/classifier/promt_labeling/OLAMA_BAD_label_dataset.py` — auxiliary async labeling script with strict Pydantic schema and an Ollama/OpenAI async client (instructor wrapper) for structured labeling — does NOT include production-grade retry/backoff, secrets management, or orchestration.
- `src/retrieval/orchestrator.py` — Hybrid Retrieval Orchestrator implementing BM25, vector, codex, procedural and RAG legs plus true RRF fusion, session diversification, deduplication, and token budgeting — does NOT include web search integration or production tuning.
- `src/api/prompt_assembler.py` — Context Structural Assembly Plane: composes the final system prompt from active `MemorySlot`s and retrieved `ContextFragment`s (codex/episodic/procedural/rag) in a stable prefix order — does NOT perform automatic token‑truncation policies or slot selection heuristics.
- `src/workers/post_flight.py` — Post‑Flight Evaluator Celery task: computes `lossless_flag`, generates summaries for non‑lossless turns, enforces idempotency keys, commits state, and triggers `extract_codex` for lossless turns — does NOT manage external model availability, distributed retries beyond Celery, or GPU orchestration.
- `src/workers/gpu_check.py` — GPU gating utility that queries `nvidia‑smi` to decide whether background tasks should yield under high GPU utilization (INV‑5) — does NOT support non‑NVIDIA GPUs or containerized cgroup metrics.
- `src/workers/celery_app.py` — Celery application factory and task registration for background workers; wired to Redis broker/backend per `settings.redis_url` — does NOT include autoscaling, worker pools tuning, or broker HA configuration.
- `tests/test_full_pipeline_phase_7.py` — integration test: Classifier → Retrieval → Prompt Assembly → Storage; inserts test turns and verifies retrieval legs and assembled prompt — requires DB and embedder, not CI‑ready.
- `tests/test_full_pipeline_phase_6.py` — integration test: Classifier → Storage → Post‑Flight → Codex Extractor; inserts a high‑value turn and enqueues `evaluate_turn` to validate post‑flight and extractor behaviour — requires Celery, DB, and background model.
- `tests/test_retrieval.py` — unit/integration test for retrieval legs (BM25, vector, codex) and prompt assembly; verifies fragment scoring and assembled system prompt — does NOT mock external services.
- `docs/DAILY_COMMANDS.md` — developer runbook for starting Docker services, the background model, Celery workers, and the proxy — does NOT include platform‑specific troubleshooting steps.
- `docs/VISION.md` — project vision and long‑term goals document explaining ICE's rationale, differentiation, and design principles — documentation only, not a technical spec.
- `main.py` — tiny top‑level script (`print("Hello from ice!")`) used for quick smoke checks — does NOT run the full application.

---

## Deviations from Blueprint
*(Append-only)*
- [2025-XX-XX] Used uv instead of pip for all package management — Reason: project standard, cleaner dependency tracking
- [2025-XX-XX] Used vLLM (Qwen2.5-7B-AWQ) instead of Ollama (70B) for labeling — Reason: vLLM already set up, faster throughput, structured output via instructor
- [2025-XX-XX] Used instructor library for structured output instead of raw JSON parsing — Reason: more robust schema enforcement and retry logic
- [2025-XX-XX] Folder structure differs from BLUEPRINT.md — using scripts/classifier/ subdirectory instead of flat scripts/ — Reason: better organization
- [2026-05-06] Added `scripts/training/build_training_data.py`, `scripts/training/fine_tune.py`, `scripts/training/test_classifier.py`, and `scripts/training/train_classifier.py` — Reason: to fine-tune and test the classifier model
- [2025-XX-XX] Used uv instead of pip for all package management — Reason: project standard, cleaner dependency tracking
- [2025-XX-XX] Used vLLM (Qwen2.5-7B-AWQ) instead of Ollama (70B) for labeling — Reason: vLLM already set up, faster throughput, structured output via instructor
- [2025-XX-XX] Used instructor library for structured output instead of raw JSON parsing — Reason: more robust schema enforcement and retry logic
- [2025-XX-XX] Folder structure differs from BLUEPRINT.md — using scripts/classifier/ subdirectory instead of flat scripts/ — Reason: better organization

---

## Active Blockers
None.

---

## Next Step

PHASE 8