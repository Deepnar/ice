<div align="center">

# ICE — Infinite Context Engine

**A local-first memory layer for conversational AI.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Status: research](https://img.shields.io/badge/status-research%20project-orange.svg)](#status)

</div>

ICE sits between a chat client and any OpenAI-compatible model and gives that model a
persistent, structured memory of everything it has discussed — entirely on your own hardware.

```
client  ──▶  ICE proxy  ──▶  local model (Ollama / SGLang)
               │
               └── PostgreSQL + pgvector
                   episodic · knowledge graph · procedural · documents
```

---

## Contents

- [Motivation](#motivation)
- [How it works](#how-it-works)
- [Evaluation](#evaluation)
- [Repository layout](#repository-layout)
- [Running ICE](#running-ice)
- [Status](#status)
- [Citation](#citation)
- [License](#license)

## Motivation

Every chat session starts from zero. The user re-explains who they are, what they are
building, and which decisions were already ruled out. Larger context windows have not solved
this: a window is a *buffer*, not a memory. Once a conversation outgrows it the model
silently loses the thread, and the usual workarounds — exporting transcripts, pasting them
into a fresh session — degrade within a dozen turns.

The instinctive fix is to retrieve *more*: fill the window with everything plausibly related.
The central finding of this project is that this instinct is wrong. Retrieval quality is
governed by what you leave out.

## How it works

Each turn traverses a synchronous **pre-flight** and an asynchronous **post-flight** phase.

**Pre-flight.** The prompt is classified by a two-stage pipeline — a rule-based pre-pass,
falling through to a 25-way PyTorch head over a frozen `Qwen3-Embedding-0.6B` encoder —
producing topic tags, intent tags, and a context-reliance label. A single calibrated decision
combines that reliance signal with a memory-pressure prior to decide whether long-term
retrieval fires at all. The hybrid orchestrator then runs its retrieval legs in parallel,
fuses them with weighted Reciprocal Rank Fusion, and post-processes the fused list with
keyword/recency/length bonuses, session diversification, deduplication, and a per-query token
budget. A prompt assembler lays the result out under a stable prefix to maximise KV-cache
reuse, and a mixture-of-experts router picks the best locally-served model.

**Post-flight.** Once the response has streamed, the turn is evaluated for information
density, summarised if it does not earn lossless storage, mined for behavioural patterns, and
— when dense enough — passed to the knowledge-graph extractor. An in-process maintenance
runtime then decays, clusters, reflects on, and compacts the stores on ledger-driven
cadences.

The organising principle in the code is that **memory is earned**: a turn is preserved
losslessly only if it is dense enough to deserve it. Everything else is compressed, decayed,
and eventually archived to cold storage.

### Memory stores

| Store | Contents | Retrieval |
| :--- | :--- | :--- |
| **Episodic** | every turn, with decay scores, summaries, and access counts | BM25 and decay-weighted vector search |
| **Codex** | a temporally-versioned knowledge graph: entities, typed edges carrying `valid_from`/`valid_until`, an append-only event log | graph traversal from resolved entities |
| **Procedural** | recurring behavioural patterns mined from interaction history | vector match behind trigger conditions |
| **Documents** | ingested files, chunked and embedded | chunk-level vector search |

These are complemented by persistent **memory slots**, topical **context clusters**, **batch
summaries**, **cold storage**, and a **timeline** leg serving temporal queries.

## Evaluation

ICE is evaluated with **LSREP** (Longitudinal State-Replay Evaluation Protocol), a benchmark
developed for this work. LSREP replays real, long-running conversations turn by turn,
reconstructs the complete memory state at each of 50 checkpoints, and scores answers against
a ground truth that *evolves* as facts are superseded — measuring what static benchmarks
cannot: memory that accumulates, decays, and is revised over months. The reported study
covers 1,985 turns and 1,211 probes, judged by an independent model.

Compared against a strong vector-RAG baseline sharing the same embedder, database, and
budget logic:

| Result | Finding |
| :--- | :--- |
| **Answer quality** | Statistical tie — paired difference +0.00 (95% CI [−0.07, +0.07]) |
| **Context efficiency** | **32% fewer fragments** injected for that same quality |
| **Head-to-head preference** | **30.6%** vs 21.2% of blind tournament wins, non-overlapping CIs |
| **Fragment quality** | ICE's fragments correlate **positively** with answer quality (r = +0.19); the baseline's correlate negatively (r = −0.02) |
| **Robustness** | On 8,000+ token turns the unbudgeted baseline fails **94.2%** of probes; ICE holds at a mean score of 4.33 |

A cumulative ablation isolates rank fusion as the mechanism that makes multi-signal retrieval
*safe*: adding an unfused lexical leg is actively harmful (−0.74, 95% CI [−1.14, −0.36]), and
fusion recovers it (+0.82, [+0.39, +1.24]).

**Read plainly:** ICE does not beat a well-built vector-RAG baseline on raw answer quality.
It matches it on a third less context, is preferred in blind comparison, and survives
conditions under which the baseline collapses. A component-level fidelity audit published
alongside the paper documents precisely which parts of the system were active during that
evaluation and which were not — curation, fusion, and the token budget carried the result.

- 📄 Paper — [`experiments/paper/ICE_paper_v2.pdf`](experiments/paper/ICE_paper_v2.pdf)
- 🔍 Fidelity audit — [`experiments/paper/notes/FIDELITY_AUDIT.md`](experiments/paper/notes/FIDELITY_AUDIT.md)
- 🏷 Evaluated snapshot — git tag `v2-paper-eval`

## Repository layout

```
src/api/          FastAPI proxy, prompt assembly, configuration, routers
src/classifier/   intent / topic / context-reliance classifier and rule-based pre-pass
src/retrieval/    hybrid orchestrator — legs, RRF fusion, budgeting, post-processing
src/memory/       ORM models, shared embedder, backup / export / re-embed tooling
src/workers/      in-process maintenance runtime and the individual jobs
src/coding/       project-state engine for code-aware memory
src/ingestion/    conversation import (ChatGPT / Claude / DeepSeek exports)
src/mcp/          ICE as an MCP server, for headless use by agents
src/services/     HTTP-free service layer shared by the API and MCP surfaces
docs/             architecture reference, roadmap, provenance, cleanup ledger
experiments/      the three experiments, their harnesses, results, and the paper
tests/            standalone integration scripts plus a fast pytest smoke suite
```

[`docs/ICE_Architecture.md`](docs/ICE_Architecture.md) is the authoritative description of the
system as built and the best entry point for reading the code.

## Running ICE

> [!IMPORTANT]
> **ICE is not packaged or distributable software.** It is a research system developed on and
> for a single Arch Linux workstation with an NVIDIA GPU. `setup.sh` is a personal bootstrap
> script, not an installer — it invokes `pacman` directly, assumes `pyenv`, and does not
> provision a model server or configuration file. Expect to adapt it. A packaged application
> is a roadmap item, not a current capability.

**Environment:** Linux, Docker, [uv](https://docs.astral.sh/uv/), Python 3.11.9+, PostgreSQL
with pgvector (supplied via Docker), a running [Ollama](https://ollama.com) instance with at
least one pulled model, and an NVIDIA GPU for background extraction work.

Bring the stack up manually:

```bash
docker compose -f docker/docker-compose.yml up -d    # PostgreSQL + pgvector
uv sync                                              # Python dependencies
uv run alembic upgrade head                          # database schema
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Configuration is read from a `.env` file at the repository root through Pydantic Settings
(`src/api/config.py` documents every field and its default, including `ollama_base_url` and
the model paths).

Point an OpenAI-compatible client at `http://localhost:8000/v1` and address the synthetic
model name `ice-proxy`. Send an `X-ICE-Conversation-ID` header to scope memory to a
conversation.

Maintenance — decay, clustering, reflection, extraction, compaction — runs **in-process** on
an async scheduler. There is no broker and no worker fleet; PostgreSQL is the only external
service.

```bash
uv run pytest tests/smoke -q                         # fast sanity suite
```

## Status

An active research project by a single author. It is not a product, and the following limits
are deliberate and documented rather than incidental:

- **The evaluation is single-user.** Every benchmark conversation was written by the author.
  The results demonstrate effectiveness across conversation *types*, not across a population
  of users. The corpora themselves are not released, because they are personal; the protocol,
  harness, and metrics are.
- **The published numbers describe a tag, not `main`.** They were produced at
  `v2-paper-eval`. Since then the Celery worker fleet was replaced by an in-process runtime,
  embeddings moved to 1024 dimensions, the RAG leg was replaced by a document store, and
  temporal retrieval, a coding mode, an MCP surface, and conversation import were added.
- **Several components remain immature.** The knowledge graph under-contributes relative to
  its design, and some mechanisms have not yet been measured in a fully working state. The
  paper and the fidelity audit identify exactly which, and why.
- Single-machine, single-user, and not hardened for deployment.

Planned work is tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md).

### Design stance

Memory is the most intimate thing a user can hand to an AI system. ICE keeps it on the user's
own hardware, in a database they can inspect, edit, export, and delete, and requires explicit
approval before high-stakes memory updates are applied. A system that faithfully models
someone's beliefs and history is dual-use by nature; keeping it local, inspectable, and under
the user's control is treated here as part of the contribution rather than a property to be
traded away.

## Citation

```bibtex
@misc{sonar2026ice,
  author = {Sonar, Deepesh},
  title  = {Curation Over Collection: A Local-First Conversational Memory
            System and a Longitudinal Protocol for Evaluating It},
  year   = {2026},
  note   = {\url{https://github.com/Deepnar/ice}}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.
