# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ICE is

The **Infinite Context Engine (ICE)** is a local-first AI memory middleware. It runs as a **FastAPI proxy** (`src/api/main.py`) that sits between a chat frontend (Open WebUI) and a local inference backend (**Ollama**, port 11434). Every OpenAI-compatible `POST /v1/chat/completions` passes through ICE, which classifies the prompt, retrieves relevant memory, assembles a context-enriched prompt, routes to a model, streams the response, and asynchronously processes the completed turn into a structured memory store.

The authoritative design reference is [docs/ICE_Architecture.md](docs/ICE_Architecture.md) (July 2026, derived from the source tree — where any doc conflicts with code, **code is authoritative**). `docs/ICE_Architecture[real_v2].md` is the frozen technical report for the system **as evaluated in the paper** (git tag `v2-paper-eval`) — the paper cites it; never update it to match current code (it has minor known inaccuracies, e.g. its NER framing, accepted as part of the historical record). Superseded docs live in `docs/outdated/` (v1/intermediate architecture docs, paper-era notes) — kept to show what the system was, never edited. `docs/VISION.md` explains intent — conversational ICE is memory for human–AI thinking sessions; a separate Coding Mode is planned post-paper (see the roadmap). The `docs/` folder in general holds most project knowledge, and code comments throughout `src/` often explain *why* something is the way it is — but neither is guaranteed current; verify against the code.

> **Maintainer note:** some working files at the repo root and under `docs/` are
> **gitignored on purpose** (personal planning, publishing strategy). If present locally,
> read them for context — but never commit them, and never quote their contents into
> tracked files.

## Project status and research context

ICE is also a research project: v2 of the system is finished, the experiments are complete, and a paper has been written for later posting on arXiv. The project is now **past that version threshold and into the post-paper update cycle** — the experiments exposed gaps (not everything works as intended), and closing them is the current work.

Experiment results live in `experiments/`, each with a results folder containing `.md` summary reports (the quick way to understand what worked and what didn't):

- `experiments/mature/results/*.md` — Experiment 2, mature-memory benchmark (1,211 probes; full ICE vs vector-RAG: +0.4 score, ~25% fewer tokens, but e.g. Codex contributed only 3.3% of fragments and MoE routing was ≈neutral)
- `experiments/unmature/results_phase2/paper_summary.md` — Experiment 1, unmature-memory phase
- `experiments/flaw_ablation/buildup/paper_summary.md` — Experiment 3, cumulative feature build-up ablation (plus `subtraction/`)

Raw metrics JSON sits alongside each report. Paper supporting notes (`paper_rough_notes.md`, `related_work_notes.md`) now live in `docs/outdated/`.

## Post-paper workflow (current phase)

The queue of upcoming work is **[docs/ROADMAP.md](docs/ROADMAP.md)** — the authoritative living checklist (planned-but-never-built features, reworks where the current version underperformed in the experiments, known bugs, and open questions with no settled solution). It was originally distilled from raw post-paper notes; those have been fully mined and archived to `docs/outdated/rough_post_paper_work.md`, so **the roadmap is the queue — do not go looking for work anywhere else.**

**Starting a session:** read this file and `docs/ROADMAP.md` — its **top section ("HOW TO EXECUTE THIS ROADMAP") is the implementation order**; follow it. **Since S1 (2026-07-10), every design-heavy item has a decision-complete spec in `docs/specs/`** — read the item's spec AND its `Assumes decided specs:` chain before coding, and obey `docs/specs/README.md` rules 11–12 (USER-REQUIRED steps; the divergence protocol: code↔spec mismatch ⇒ stop, re-ground, fix the spec first, never improvise past it). Items without a spec carry an explicit no-spec note. Completed items carry detailed notes (what shipped, why, validation) so you can pick up cold.

Rules for working it:

- **Roadmap entries are intent + rationale, not specs.** When a feature's turn comes, discuss the concrete implementation with the user first — never build straight from the entry.
- **No first versions.** Build the robust, thought-through version of each feature (proper algorithms, edge cases, end-state in mind), not a throwaway MVP. If too big for one pass, split into robust sub-items rather than shipping a knowingly-temporary version.
- **Look ahead before building.** Before implementing any item, scan the roadmap for later items in the same subsystem. Design the current work to be forward-compatible with where those are heading (build on the primitive they'll need), or if they genuinely conflict, decide explicitly (do the later one first, or record the exact seam). Don't implement anything a known-future item will have to tear out; note the look-ahead result in the completion entry.
- **Earn the checkmark.** Only mark an item done after its full original scope is implemented *and* behaviorally validated (a real run/test, not just a syntax check) — audit against the entry text, not memory of it. Always before implementaions check what how it actually is in realtion to what we want to do and roughly check thru the previous implemented if they are actually done or not.
- **Propagate on completion — refresh downstream stale references.** The mirror of look-ahead, done *after* finishing an item: scan the still-unchecked roadmap items for any that describe the *old* behavior of what you just changed, and update them to the new reality (and the new dependency). A shipped feature that leaves later items describing the pre-change world silently misleads the next session. Do this pass before checking the item off.
- **It doubles as the progress tracker.** Check items off in `docs/ROADMAP.md` as they're completed.
- **Keep the architecture doc in sync.** When a brand-new system/feature is finished, add a section for it to `docs/ICE_Architecture.md`; when an existing subsystem is reworked, search out its existing section there and update it to match the new behavior. The architecture doc must keep reflecting the system as built.

- **⚑ DELETION SWEEP — the rule that stops zombie docs (standing rule, 2026-08-01).** Adding
  a section is the easy half. The half that keeps getting missed is **removing a thing**: a
  deleted component keeps living in the *overview* prose, the *diagrams*, and the *settings
  list* long after its own section says "DELETED". This has actually happened — DI3 was
  deleted in D8 and its own §2.2 said so, while §1.1 still called the classifier a
  "two-stage pipeline (DI3 → 25-way MLP)", the component-map diagram still had a "DI3 + MLP"
  box, and §10 still documented seven `DI3_*` settings that no longer exist in `config.py`.
  So whenever a component is **deleted, replaced, renamed, or changes shape** (label counts,
  leg counts, dimensions, cadences), `grep -rin '<old name>' docs/ README.md` and fix **every**
  hit, classifying each as either:
  - **live claim** → must be corrected, or
  - **deliberate history** ("X was replaced by Y in D8") → keep, it's the record.

  Check these four places specifically, because they are the ones that rot:
  **(1)** §1 System Overview prose · **(2)** the ASCII diagrams · **(3)** §10's configuration
  /settings lists · **(4)** `README.md`.

- **⚑ The README links the VENUE-AGNOSTIC paper, never a venue submission (standing rule,
  2026-08-01).** `experiments/paper/` holds one canonical paper (`ICE_paper_v2.tex`, generic
  `article` class) plus venue twins built from it (`_tmlr`, `_tist`, …). **Only the canonical
  one may be linked from `README.md` or offered publicly.** A venue twin carries that venue's
  furniture — ACM/TMLR branding, line numbers from `review` mode, placeholder volume/article
  numbers and a dummy DOI (`10.1145/nnnnnnn.nnnnnnn`). Correct for a submission; on a public
  repo it reads as *"published in that venue"*, which is a false claim about work that is
  merely submitted. This has already happened once and was reverted. When a venue twin's
  content improves on the canonical paper, **back-port it into the canonical file** (keep the
  generic preamble, take the body) rather than repointing the link — the twin is a build
  target, never the source of truth.

- **README and the architecture doc follow the same contract.** Both describe **`main` as it
  is now**, and both are updated *in the same session* as the change that invalidates them.
  The split: `README.md` is the outside view (what ICE is, what it does, headline numbers,
  how to run it) and stays short; `ICE_Architecture.md` is the inside view (how each
  subsystem actually works) and carries the detail. A number that appears in both — label
  counts, leg counts, embedding dimensions, headline results — must be changed in both or in
  neither. Where either doc quotes evaluation results, say which snapshot they describe
  (the paper's numbers are the `v2-paper-eval` tag, not `main`).

## Check where a signal LANDS, not just that it exists (standing rule, 2026-07-27)

ICE computes a lot of signals. A signal can be trained, accurate, stored on every
result — and still change nothing, because it is wired to a decision that was
already made. Three were found in this state on the same day (`p_complex`: zero
readers; `Codebase_Query`: dropped; `Temporal_Recall`: see below). **When a new
classifier label or score is added, trace it to the decision it changes and
measure the delta with it on vs off.** "It's wired up" is not the test; "turning
it off changes N decisions" is.

The worked example, because it generalises. `Temporal_Recall` was wired to the
retrieve-or-not gate — and a question about the past *needs memory by
definition*, so 78% of its rows were already `Needs_Memory`, sitting at mean
`p_ltm` 0.931. Turning the whole arm off moved **1 decision in 9,441**. The label
was fine; it was answering a question something else had already answered. Its
real information — *only 20% of memory queries are time-shaped* — belongs where
time matters (ranking old-vs-recent, gating a time window), which is roadmap
**T5**. Beware this shape generally: **a signal that is a subset of another
signal cannot improve that signal's own decision.**

Corollary on deletions: when a measurement says a designed hook or seam is not
paying off, **that is evidence, not permission — ask the user first.** A null
result on one wiring does not mean the signal is homeless, and the seam records
design intent the measurement does not contain.

## No decision may depend on HOW a thing is written (standing rule, 2026-07-28)

The user's diagnosis, and the measured reason Codex underperformed across a month
of experiments: a rule keyed on punctuation, word order, or a fixed vocabulary is
a **bet on writing convention**. ICE was validated on LMSYS/ShareGPT/WildChat —
all "people typing at a chatbot", all sharing conventions — so the bets never
looked broken.

**The goal is INVARIANCE, not personalization.** The maintainer writes
out-of-convention and is a useful canary, but they are an *existence proof*, not
a target: tuning toward their corpus is the same mistake with a different corpus.
**We have no idea how any given user writes and no corpus can tell us** — a
writing style is not a distribution you can sample your way out of. The
requirement is that the same intent, written any way, yields the same decision.

Two measurements to keep in mind. (1) T2's gate accepts a time expression if the
prompt *starts with* one of twelve interrogatives: lmsys 25%, wildchat 14%,
sharegpt 13%, personal 2% — a 10× swing, and 2× **between the public corpora
alone**. (2) **The trained head is not automatically the fix**: firing-rate spread
across those sources is 1.5× for `has "?"` but 16.2× for `p_ltm ≥ 0.5`. That is
confounded (personal rows genuinely need memory more), and *that confound is the
lesson* — **firing-rate-by-source cannot separate "different people" from
"different meanings", so it is a smell detector, never an acceptance test.**

The test that works holds meaning fixed and varies only form: write one prompt
several ways (± question mark, "ok so"/"like" prefixes, interrogative buried,
lowercase, typos, terse vs rambling) and measure the **decision-flip rate**. A
rule that flips is measuring typography. **Apply it to the replacement too.**
Parsers that RESOLVE a value (a date → a datetime) may stay; lexicons that INFER
INTENT must pass invariance. Roadmap **G28** is the systematic sweep and owns the
style-variant probe set; D8 is the worked deletion protocol.

## A silent fallback hides an outage (standing rule, 2026-08-03)

The measured case: **every background LLM call in ICE was returning nothing**,
and the system looked fine. Reasoning models spend the whole `max_tokens`
budget inside a hidden thinking block, so Ollama returns `content=""` — and the
callers had defaults. `clustering._generate_cluster_name` returned
`"Unnamed Cluster"`. `detect_blob_kind` returned `document` via its
`blob_kind_unparsed` branch. `post_flight` turned the empty summary into `None`
and let raw text win. Each of those defaults is individually *correct*
engineering. Together they made a dead subsystem indistinguishable from a
working one, for an unknown length of time.

**So: a fallback must be observable.** When a component substitutes a default
for a real answer, it emits at WARNING with the reason — every time, not on the
first occurrence. A fallback that fires on 100% of calls is not resilience, it
is an outage wearing resilience as a costume.

Two corollaries, both earned the same day. **(1)** Check the *rate*, not the
existence: `ner_utils` has a regex fallback whose own docstring says it is
"log-worthy if this fires in normal operation" — and there is no log line, so
nobody could have known it fires whenever the process starts outside the repo
root. **(2)** When a subsystem produces plausible-but-thin output, verify the
model was actually called before tuning anything about it — the first
explanation is usually "it never ran".

## Boy-scout cleanup (standing rule, 2026-07-10)

Every implementation session leaves the files it touches cleaner than found: imports sorted/grouped + unused dropped (`ruff` on touched files only — never a repo-wide reformat), dead code and lying comments fixed in place, one-off scripts moved (never deleted) to `scripts/oneoff/` with their paths fixed. **No barrel re-exports in `__init__.py`** (import-time side effects, hidden provenance, heavy transitive imports; the lazy in-function imports that break circular deps stay, commented). The pre-FINAL `experiments/*` folders are a **frozen historical record** — never reorganize them. Log every move/rename in [docs/CLEANUP.md](docs/CLEANUP.md).

## Provenance ledger (standing rule, 2026-07-26)

When a run produces an artifact anything downstream depends on — a corpus, a labeled set, a checkpoint, an experiment result — record what produced it in **[docs/PROVENANCE.md](docs/PROVENANCE.md)** *in the same session*. Model repo **and revision SHA**, quantization, serving engine + version, key parameters, row counts, and the decisions taken.

Two reasons this is a rule and not a nicety. **Community model quantizations are not stable references** — they get re-uploaded, revised, or deleted, so `org/model-AWQ` is unverifiable a year later while `org/model-AWQ @ <sha>` is; the SHA is free to read (`ls ~/.cache/huggingface/hub/models--*/snapshots`). And **the paper gets written months after the runs**: reconstructing which weights or which corpus produced a number is archaeology, and the details that matter most are the ones that decay fastest. Record rejected candidates too, with the symptom — that is what stops the next session re-testing a model that was already found broken.

`ICE_Architecture.md` describes the system as it *is*; PROVENANCE.md records what was *done*. They are not substitutes.

## Git & pushing (standing rule, 2026-07-14)

The repo has a **private** GitHub remote (`origin` → `github.com/Deepnar/ice`). **Pushing during normal development is pre-authorized and encouraged** — it's a private backup, so push your work at natural points (end of a session, after a meaningful milestone); you do **not** need to ask each time. Commit habits are unchanged (plain messages, **no AI attribution**; branch off `main` first if the default-branch rule applies).

**Commit message style (standing rule, 2026-08-01):** messages are **impersonal and
factual** — describe *what changed and why*, in the repository's voice. **Never narrate the
session**: no "the user asked…", "as requested…", "we decided…", "per our discussion". A
reader six months from now cares about the change, not the conversation that produced it.
Subject line ≤ ~70 chars, imperative or noun-phrase; body explains the *why* when it isn't
obvious. This repo is destined to be public — the log is part of the artifact.

**⚑ THIS REPO IS GOING PUBLIC (standing rule, 2026-08-01).** It is private today, but it
will be flipped to public as the portfolio/paper artifact. Every session must work as if the
tree, the docs, **and the commit log** will be read by strangers — recruiters, admissions
readers, reviewers. Consequences:
- **Never commit personal content.** Private planning, career notes, conversation corpora,
  third-party email addresses, credentials, endorsement codes. If it names a real person who
  didn't consent, or reveals the maintainer's private life, it does not belong in the tree.
  (Git history is forever: a file committed once and gitignored later is *still public* —
  the labeled-prompt corpus had to be purged with `git-filter-repo` for exactly this reason.)
- **README.md is a first-class deliverable, not an afterthought.** It is the first and often
  only thing a visitor reads. **Whenever a session changes what the project *is* or how it is
  run — a new subsystem, a changed entry point, new setup steps, a new headline result — update
  `README.md` in the same session**, the same way `ICE_Architecture.md` is kept in sync.
- The public release is gated on a genuinely good README. Until then the repo stays private.

**Commit granularity (standing rule, 2026-08-01):** **many small, focused commits — never
one massive end-of-session commit.** Work as normal, but when committing, split by *concern*
rather than by session: one logical change per commit (e.g. the migration, then the worker,
then the docs, then the tests). This keeps each message short and specific, creates multiple
restore points, and keeps `git bisect`/`git revert` usable. If a commit message needs
bullet points to list unrelated changes, it should have been several commits. **Exception — freeze at the experiment phase:** once **SEMIFINAL (Z1)** or **FINAL** begins (the last / second-last roadmap phases — they generate personal data/results and are the pre-public-release cutoff), **stop pushing** until the user explicitly says otherwise. Until then, pushing is not a problem.

## Commands

Package/deps are managed with **uv** (Python 3.11.9, pinned in `.python-version`). Always run project code through `uv run`.

```bash
./ice          # start everything: docker (postgres+redis), vLLM bg model, celery worker+beat, uvicorn proxy; tails logs — use only when testing the whole service. NOTE: ./ice/stop_ice/setup.sh are dev scaffolding with a decided fate (ROADMAP Track-F end-state): replaced by one packaged app, PLUS a separate headless boot path for ICE-as-MCP (E7); the vLLM bg server also leaves the default stack (shared-first decision, C7). Keep changes to these scripts thin.
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
2. **PyTorch classifier** (`src/classifier/classifier.py`) — an MLP head over a frozen `Qwen/Qwen3-Embedding-0.6B` encoder (truncated to 384 dim; live checkpoint `models/classifier/ice_classifier_v3_qwen_ft3.pt`). Outputs 25 logits → 11 topic + 11 intent (multi-label sigmoid) + 3 context-reliance (softmax) labels. Schema in `data/labeled/label_schema.json`.
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
