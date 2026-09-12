<div align="center">

# ICE — Infinite Context Engine

**A local-first memory layer for conversational AI.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Status: research](https://img.shields.io/badge/status-research%20project-orange.svg)](#status)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21759702.svg)](https://doi.org/10.5281/zenodo.21759702)
</div>

ICE sits between a chat client and any OpenAI-compatible model and gives that model a
persistent, structured memory of everything it has discussed — entirely on your own hardware.

```
client  ──▶  ICE proxy  ──▶  local model (Ollama, or vLLM)
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

**Pre-flight.** The prompt is classified by a small PyTorch head over a frozen
`Qwen3-Embedding-0.6B` encoder: 27 all-sigmoid logits across three heads — 11 topic labels,
12 intent labels, and 4 independent context-reliance signals (`Needs_Memory`,
`Temporal_Recall`, `Needs_Live_Info`, `High_Complexity`). The reliance signals are deliberately
independent rather than a single choice, because a prompt can need stored memory *and* live
information at once; a fully self-contained prompt is the derived state where all four stay
low. A calibrated decision then combines the memory signal with a memory-pressure prior to
decide whether long-term retrieval fires at all. The hybrid orchestrator runs its legs in
parallel,
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

### Beyond retrieval

The system has grown several capabilities that sit alongside the core loop:

- **Temporal retrieval.** Memory is queryable along the time axis, not just by similarity:
  *as-of* ("what did I think about this in March?"), *range*, and *evolution* ("how did my
  design for X change?") — backed by the knowledge graph's `valid_from`/`valid_until` edges,
  so a superseded fact stays retrievable as history rather than being overwritten.
- **Agentic maintenance.** A maintenance agent reconciles graph state during idle GPU time —
  merging duplicate entities, resolving contradictions, and escalating anything ambiguous to
  a review queue rather than guessing. Deterministic checks run first; the model is consulted
  only for genuinely ambiguous supersessions.
- **Conversation import.** Exported histories from ChatGPT, Claude, and DeepSeek can be
  replayed through the full pipeline, reconstructing episodic, graph, procedural, and cluster
  state as though ICE had been present all along — rather than dumping them into a searchable
  archive. This is the same replay machinery the evaluation protocol uses.
- **Coding mode.** A project-state engine tracks code structure, decisions, and git history
  as first-class memory, so retrieval can surface an architectural decision alongside the
  file it applies to.
- **MCP server.** ICE exposes itself over the Model Context Protocol for headless use by
  agents, booting the stack itself without the HTTP proxy.
- **User control.** In-chat commands (`/remember`, `/forget`, `/bookmark`, `/scope`,
  `/search`, `/slots`, `/delete-conversation`), per-conversation and project scoping, an
  incognito mode that writes nothing, and a real deletion cascade — memory the user can
  inspect, correct, and remove rather than merely accumulate.

## Evaluation

The paper evaluates **frozen ICE v2** at `v2-paper-eval`, not current v3 on
`main`. **LSREP** reconstructs evolving memory during ordered conversational
replay, with repeated probes, lifecycle schedules and changing reference answers.
The private single-user study covers 1,985 turns, 219 distinct probes, 1,211
probe–checkpoint observations and 52 checkpoints. Manual records are merged
into the historical scores; sensitivity analyses disclose their effect.

| Frozen-v2 result | Finding |
| :--- | :--- |
| Ordinary-density LSREP | Mean ICE–vector difference +0.002; probe-cluster 95% CI [−0.148,+0.158]. No detected difference, not equivalence. |
| Context use | 32% fewer selected fragments, but 6.6% more estimated prompt tokens. |
| Tournament preference | ICE 30.6%, vector 21.2% first places in four-condition tournaments; not head-to-head win rates. |
| Density stress | ICE mean score 4.33 versus vector 1.23; the unbudgeted vector arm has 94.2% score-1 observations. This is a full-system reliability contrast. |
| Matched LongMemEval oracle | ICE 50.8%, pure vector-RAG 72.8%; paired gap −22.0 points [−26.6,−17.4]. |
| Matched LongMemEval full-S | ICE 43.0%, vector 69.5%; paired gap −26.5 [−31.3,−21.8], n=499. Missing-judgement bounds preserve the ordering. |

Both public-benchmark arms use gpt-5.6-luna and Muse Spark 1.3 Contributor
judging. ICE v2 loses decisively overall, with descriptive conservative abstention
and severe multi-session and temporal failures. It supplies less context there
while answering less accurately: a quality–cost trade-off, not superior efficiency.
These are within-study comparisons, not official GPT-4o-judged leaderboard scores.

The fidelity audit finds defective procedural retrieval, an additionally defective
and unused document leg, unexercised paths, and unconfirmed graph utility.
An unfused lexical leg harms the ablation score (−0.74 [−1.14,−0.36]); RRF
recovers it (+0.82 [+0.39,+1.24]) in that buildup, without a general safety claim.

- 📄 Canonical paper — [`ICE_paper_v2.pdf`](experiments/paper/ICE_paper_v2.pdf)
- 🔍 Fidelity audit — [`FIDELITY_AUDIT.md`](experiments/paper/notes/FIDELITY_AUDIT.md)
- 📊 Analyses and release scope — [`ARTIFACTS.md`](experiments/paper/ARTIFACTS.md)
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
the model paths). That file — and every model and data path ICE loads — is resolved against
the installation directory rather than the shell's working directory, so ICE behaves the same
whatever you launch it from; set `ICE_HOME` to override where it looks.

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
@software{sonar2026ice_software,
  author = {Sonar, Deepesh},
  title  = {{ICE}: Infinite Context Engine},
  year   = {2026},
  doi    = {10.5281/zenodo.21759702},
  url    = {https://github.com/Deepnar/ice}
}
```

```bibtex
@unpublished{sonar2026ice_paper,
  author = {Sonar, Deepesh},
  title  = {{LSREP}: A Longitudinal State-Replay Protocol for Evaluating
            Conversational Memory, with {ICE v2} as an Audited Local-First Architecture},
  year   = {2026},
  note   = {Unpublished manuscript}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.
