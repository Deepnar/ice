# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ICE is

The **Infinite Context Engine (ICE)** is a local-first AI memory middleware. It runs as a **FastAPI proxy** (`src/api/main.py`) that sits between a chat frontend (Open WebUI) and a local inference backend (**Ollama**, port 11434). Every OpenAI-compatible `POST /v1/chat/completions` passes through ICE, which classifies the prompt, retrieves relevant memory, assembles a context-enriched prompt, routes to a model, streams the response, and asynchronously processes the completed turn into a structured memory store.

The authoritative design reference is [docs/ICE_Architecture.md](docs/ICE_Architecture.md) (July 2026, derived from the source tree — where any doc conflicts with code, **code is authoritative**). `docs/ARCHITECTURE.md` (v1) and `docs/ARCHITECTURE_V2.md` are legacy versions kept for history. `docs/VISION.md` explains intent — conversational ICE is memory for human–AI thinking sessions; a separate Coding Mode is planned post-paper (see the roadmap). The `docs/` folder in general holds most project knowledge, and code comments throughout `src/` often explain *why* something is the way it is — but neither is guaranteed current; verify against the code.

## Project status and research context

ICE is also a research project: v2 of the system is finished, the experiments are complete, and a paper has been written for later posting on arXiv. The project is now **past that version threshold and into the post-paper update cycle** — the experiments exposed gaps (not everything works as intended), and closing them is the current work.

Experiment results live in `experiments/`, each with a results folder containing `.md` summary reports (the quick way to understand what worked and what didn't):

- `experiments/mature/results/*.md` — Experiment 2, mature-memory benchmark (1,211 probes; full ICE vs vector-RAG: +0.4 score, ~25% fewer tokens, but e.g. Codex contributed only 3.3% of fragments and MoE routing was ≈neutral)
- `experiments/unmature/results_phase2/paper_summary.md` — Experiment 1, unmature-memory phase
- `experiments/flaw_ablation/buildup/paper_summary.md` — Experiment 3, cumulative feature build-up ablation (plus `subtraction/`)

Raw metrics JSON sits alongside each report. `docs/paper_rough_notes.md` and `docs/related_work_notes.md` are paper supporting material.

## Post-paper workflow (current phase)

The queue of upcoming work is **[docs/ROADMAP.md](docs/ROADMAP.md)** — a living checklist distilled from `docs/rough_post_paper_work.md` (planned-but-never-built features, reworks where the current version underperformed in the experiments, known bugs, and open questions with no settled solution).

Rules for working it:

- **Roadmap entries are intent + rationale, not specs.** When a feature's turn comes, discuss the concrete implementation with the user first — never build straight from the entry.
- **No first versions.** Build the robust, thought-through version of each feature (proper algorithms, edge cases, end-state in mind), not a throwaway MVP. If too big for one pass, split into robust sub-items rather than shipping a knowingly-temporary version.
- **Look ahead before building.** Before implementing any item, scan the roadmap for later items in the same subsystem. Design the current work to be forward-compatible with where those are heading (build on the primitive they'll need), or if they genuinely conflict, decide explicitly (do the later one first, or record the exact seam). Don't implement anything a known-future item will have to tear out; note the look-ahead result in the completion entry.
- **Earn the checkmark.** Only mark an item done after its full original scope is implemented *and* behaviorally validated (a real run/test, not just a syntax check) — audit against the entry text, not memory of it. Always before implementaions check what how it actually is in realtion to what we want to do and roughly check thru the previous implemented if they are actually done or not.
- **It doubles as the progress tracker.** Check items off in `docs/ROADMAP.md` as they're completed.
- **Keep the architecture doc in sync.** When a brand-new system/feature is finished, add a section for it to `docs/ICE_Architecture.md`; when an existing subsystem is reworked, search out its existing section there and update it to match the new behavior. The architecture doc must keep reflecting the system as built.

## Commands

Package/deps are managed with **uv** (Python 3.11.9, pinned in `.python-version`). Always run project code through `uv run`.

```bash
./ice          # start everything: docker (postgres+redis), vLLM bg model, celery worker+beat, uvicorn proxy; tails logs, BUT only when testing the whole service, but even this has to change later
./stop_ice     # stop all services
./setup.sh     # first-time install (Arch/CachyOS): pacman deps, pyenv, uv sync, docker up, alembic upgrade, model pull

# Individual services (what ./ice runs under the hood)
docker compose -f docker/docker-compose.yml up -d
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
uv run celery -A src.workers.celery_app worker -B --loglevel=info   # -B also runs beat

# Database migrations (SQLAlchemy sync + Alembic; DB is postgres+psycopg on localhost:5432/ice_db)
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "message"
```

Logs go to `logs/{proxy,celery,vllm_bg}.log`.

### Tests

Tests in `tests/` are **standalone scripts, not a pytest suite** — they `sys.path.insert` the repo root and run directly against a live Postgres (several `TRUNCATE` the tables). Run one with:

```bash
uv run python tests/test_retrieval.py
```

They require the docker services (postgres/redis) up and, for pipeline tests, Ollama/vLLM running. There is no lint config; match existing style.

## Architecture

### Request lifecycle (the core flow in `src/api/main.py`)

**Pre-flight (synchronous, latency-sensitive):**
1. Extract latest user message → **DI3** (`src/classifier/di3.py`), a heuristic "Dynamic Intent Inferencer" that tries to classify from signals and returns `None` to fall back to the ML model.
2. **PyTorch classifier** (`src/classifier/classifier.py`) — an MLP head over a frozen `all-MiniLM-L6-v2` encoder. Outputs 25 logits → 11 topic + 11 intent (multi-label sigmoid) + 3 context-reliance (softmax) labels. Schema in `data/labeled/label_schema.json`.
3. If context reliance is `Long_Term_Memory` (or confidence < fallback threshold), invoke the **Hybrid Retrieval Orchestrator** (`src/retrieval/orchestrator.py`), which blends BM25 + vector + codex-graph + procedural sources with **intent-dependent weights** (see the weight tables around `orchestrator.py:339`).
4. **Prompt Assembler** (`src/api/prompt_assembler.py`) builds the system prompt from retrieved `ContextFragment`s.
5. **Model Registry** (`src/model_registry/registry.py`, `find_best_model`) picks the Ollama model from tags, with session stickiness in `SESSION_STATE`.
6. Forward to Ollama, stream back as SSE.

**Post-flight (async, accuracy-sensitive):** After the stream, a FastAPI `BackgroundTask` stores the raw turn to episodic memory + embeddings and publishes a `CHAT_COMPLETED` event to Redis. Celery tasks then take over (see below).

Guiding principle in the code: **"memory is earned"** — a turn is stored losslessly only if dense enough (`lossless_flag` / `inject_raw` set by the post-flight evaluator); otherwise it's summarized by the background model.

### Background workers (`src/workers/`, Celery on Redis)

The task graph is wired in `src/workers/celery_app.py` — the `include=[...]` list is the source of truth for active workers, and `beat_schedule` defines the cron cadence. Key stages: `post_flight` (density eval + summary) → `codex_extractor` (entity triplets → knowledge graph) and `procedural_extractor` (recurring behavior patterns). Periodic: `clustering`, `sentinel_monitor`, `decay`/`codex_decay`/`procedural_decay`, `reflection`, `batch_summarizer`, and a weekly `fine_tune` of the classifier. **When adding a worker, register it in both the `include` list and (if scheduled) `beat_schedule`.**

### Memory schema (`src/memory/models.py`)

All ORM models live in one file. The memory stores form the system's substance: `EpisodicMemory` (raw turns + pgvector embeddings), `Codex*` (entities/edges/events/snapshots — the knowledge graph), `ProceduralMemory`, `ContextCluster` + `EpisodicClusterLink` (user-scoped memory clusters), `MemorySlot`, `RAGDocument`/`RAGChunk`, `Sentinel*`, plus operational tables (`IdempotencyKey`, `ColdStorage`, `ReviewQueue`, session replay/summaries). DB uses the **pgvector** extension (`pgvector/pgvector:pg16` image).

### API routers (`src/api/routers/`)

Beyond `/v1/chat/completions` and `/health`: `memory_slots.py` (`/memory-slots`) and `user_control.py` (`/user-control`) — the human-oversight layer for bookmarks, tag overrides, conversation scope assignment, cluster management, the review queue, and model-registry editing.

## Config and conventions

- Runtime config is **Pydantic Settings** in `src/api/config.py`, loaded from `.env`. Model paths (e.g. `classifier_model_path` currently `models/classifier/ice_classifier_v3_qwen_ft3.pt`), thresholds, `ollama_base_url`, and `background_model_mode` (`dedicated` runs a separate vLLM on port 8002; `shared` reuses the main LLM) all live there. The DB URL is duplicated in `alembic.ini`.
- Logging is **structlog** (`ice.api`, `ice.*` loggers), consistent with the "no silent failure" principle — surface what the system decided, don't swallow it.
- Trained artifacts (`models/`), datasets (`data/`), and one-off `scripts/` are versioned; classifier checkpoints are suffixed by version/fine-tune generation. Update `config.py` when promoting a new checkpoint.
