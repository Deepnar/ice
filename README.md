<div align="center">

# ICE — Infinite Context Engine

**A local-first memory layer for conversational AI.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Status: research](https://img.shields.io/badge/status-research%20project-orange.svg)](#status)
[![arXiv](https://img.shields.io/badge/arXiv-2609.16730-b31b1b.svg)](https://arxiv.org/abs/2609.16730)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21759702.svg)](https://doi.org/10.5281/zenodo.21759702)
</div>

ICE sits between a chat client and any OpenAI-compatible model and gives that model a
persistent, structured memory of everything it has discussed — entirely on your own hardware.

```
client  ──▶  ICE proxy  ──▶  local model (Ollama, or vLLM)
               │
               └── PostgreSQL + pgvector
                   episodic · codex facts · procedural · documents
```

---

## Contents

- [Motivation](#motivation)
- [How it works](#how-it-works)
- [Evaluation](#evaluation): [paper 1, ICE v2](#paper-1-frozen-ice-v2) · [ICE v3, under evaluation](#ice-v3-current-main-frozen-and-under-evaluation)
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
decide whether long-term retrieval fires at all; a strong reranker match can still admit
memory the decision declined. The hybrid orchestrator runs its legs in parallel, fuses them
with weighted Reciprocal Rank Fusion, adjusts scores with keyword/recency/length bonuses,
reranks the candidates with `Qwen3-Reranker-0.6B`, then diversifies by session,
deduplicates, and fits a per-query token budget. A prompt assembler lays the result out
under a stable prefix to maximise KV-cache reuse, and a mixture-of-experts router picks the
best locally-served model.

**Post-flight.** Once the response has streamed, v3 evaluates the turn's information
density and stores a grounded summary alongside its original text when eligible.
Every non-private turn reaches the knowledge-graph and behavioural-pattern extractors.
Their separate completion keys let a retry finish failed work without repeating a
successful stage. An in-process maintenance runtime then decays, clusters, reflects
on, and compacts the stores on ledger-driven cadences.

The organising principle is that **memory is earned**: evidence and relevance determine
which stored information deserves space in the answer prompt. V3 retains originals
alongside compressed representations and chooses between them at retrieval time.

### Memory stores

| Store | Contents | Retrieval |
| :--- | :--- | :--- |
| **Episodic** | every turn, with decay scores, summaries, and access counts | BM25 and decay-weighted vector search |
| **Codex** | verified, self-contained fact sentences (each with its quote, speaker and source turns) linked to entities; a fact carries `valid_from`/`valid_until` and the fact that replaced it, plus an append-only event log | keyword, meaning and named-entity search over the facts |
| **Procedural** | recurring behavioural patterns mined from interaction history | vector match behind trigger conditions |
| **Documents** | ingested files, chunked and embedded | chunk-level vector search |

These are complemented by persistent **memory slots**, topical **context clusters**, **batch
summaries** and **cold storage**. Archived and cold-stored originals stay
searchable under the same budgets; decay lowers their ranking rather than
hiding them.

### Beyond retrieval

The system has grown several capabilities that sit alongside the core loop:

- **Temporal retrieval.** Memory is queryable along the time axis, not just by similarity:
  *as-of* ("what did I think about this in March?"), *range*, and *evolution* ("how did my
  design for X change?") — backed by the codex facts' `valid_from`/`valid_until`, so a
  replaced fact stays retrievable as history rather than being overwritten.
- **Agentic maintenance.** A maintenance agent tidies memory during idle GPU time — merging
  duplicate entities (journaled and undoable) and flagging stale slots and work. Whether a
  new fact repeats, replaces or sits beside an older one is decided when the fact is written. Deterministic checks run first; the model is consulted
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

### Paper 1: frozen ICE v2

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

- 📄 Paper — [arXiv:2609.16730](https://arxiv.org/abs/2609.16730)
- 📄 Repository copy — [`ICE_paper_v2.pdf`](experiments/paper/ICE_paper_v2.pdf)
- 🔍 Fidelity audit — [`FIDELITY_AUDIT.md`](experiments/paper/notes/FIDELITY_AUDIT.md)
- 📊 Analyses and release scope — [`ARTIFACTS.md`](experiments/paper/ARTIFACTS.md)
- 🏷 Evaluated snapshot — git tag `v2-paper-eval`
- 🤗 Frozen **ICE v2** paper models — [classifier](https://huggingface.co/Deepnar/ice-v2-classifier), [MicroNER](https://huggingface.co/Deepnar/ice-v2-microner), and [reproducibility collection](https://huggingface.co/collections/Deepnar/ice-lsrep-reproducibility-6ac6210257541dd6f8594e14). Original `.pt` files, SHA-256 checksums and frozen preprocessing are included; these are separate from current ICE v3 models.

### ICE v3: current `main`, frozen and under evaluation

v3 is v2 rebuilt after the paper's audit, and it is what `main` holds. It is now
**frozen**: nothing that affects an answer changes until its final evaluation
is complete. The main differences from v2:

- **A codex instead of a triplet graph.** Facts are short verified sentences,
  each carrying its quote, speaker and source turns, with a validity window so
  a replaced fact stays retrievable as history.
- **Grounded writes.** Summaries, facts, behavioural patterns and entity notes
  are kept only when the cited source supports them. A question or a
  hypothetical stays evidence; it does not become an established fact.
- **Originals kept.** Turns keep their original text beside the summary, and
  retrieval chooses which one the answer prompt gets.
- **One ranked pool.** Every memory family competes in a single reranked
  candidate pool (`Qwen3-Reranker-0.6B`) under the same token budget.
- **Defaults chosen by measurement.** Bounded tuning on a development history
  found no setting better than the defaults beyond answer noise, so the
  defaults were frozen unchanged
  ([results](experiments/v3_repair/results/tuning-freeze-2026-10-15.md)).

The final evaluation is **pre-registered**: its claim, measures, tests and
handling rules were fixed before any answer was graded
([pre-registration](docs/specs/FINAL_PREREGISTRATION.md)). It covers:

- **LSREP**, on nine synthetic histories and five long private ones, scored on
  whether an answer is correct *as of the moment it was asked*, uses no
  outdated value and invents nothing, and reported by how far back the
  evidence lies.
- **Public benchmarks**, each in its own published scoring: BEAM,
  LongMemEval-S and LoCoMo.
- **Comparisons** against plain vector search under ICE's own token budget, a
  256K-token long-context reader, a recent-turns-only floor and, on LoCoMo,
  three other memory systems (Graphiti, Mem0, A-Mem).
- **Ablations**, switching each ICE part off in turn on a history built to
  exercise it.

There are no v3 results yet; they will be published in a follow-up paper.
Until then, no number in this repository describes v3's quality.

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
experiments/      evaluation harnesses, benchmark adapters, results, and the paper
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

Maintenance — decay, clustering, reflection, summaries, entity merges — runs **in-process** on
an async scheduler. There is no broker and no worker fleet; PostgreSQL is the only external
service.

```bash
uv run pytest tests/smoke -q -m quick                # fast checks, no database needed
```

## Status

An active research project by a single author. It is not a product, and the following limits
are deliberate and documented rather than incidental:

- **The private evaluation is single-user.** Every private history was written by the
  author, so it speaks to conversation *types*, not to a population of users; public benchmarks and synthetic histories cover other writing. The private corpora
  are not released, because they are personal; the protocol, harness, and metrics are.
- **The published numbers describe a tag, not `main`.** They were produced at
  `v2-paper-eval`. Since then the Celery worker fleet was replaced by an in-process runtime,
  embeddings moved to 1024 dimensions, the RAG leg was replaced by a document store, the
  graph became a fact codex, a reranker was added, and temporal retrieval, a coding mode,
  an MCP surface, and conversation import were added. v3 is measured by its own
  pre-registered evaluation, not by the paper.
- **Whether each v3 component earns its place is still open.** The ablations in v3's
  evaluation answer that; until they report, no component is claimed to help. The paper
  and the fidelity audit identify which v2 components were immature, and why.
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

If you use **LSREP, its evaluation methodology, or the results reported for ICE v2**, please cite the paper:

```bibtex
@misc{sonar2026lsrep,
  author        = {Sonar, Deepesh},
  title         = {{LSREP}: A Longitudinal State-Replay Protocol for Evaluating Conversational Memory, with {ICE v2} as an Audited Local-First Architecture},
  year          = {2026},
  eprint        = {2609.16730},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.16730}
}
```

If you use the **ICE software or frozen evaluation artifact**, please cite the archived software release:

```bibtex
@software{sonar2026ice,
  author  = {Sonar, Deepesh},
  title   = {{ICE}: Infinite Context Engine},
  year    = {2026},
  version = {v2-paper-eval},
  doi     = {10.5281/zenodo.21759702},
  url     = {https://doi.org/10.5281/zenodo.21759702}
}
```

The paper evaluates the frozen `v2-paper-eval` snapshot. Development on `main` has continued beyond the evaluated version.


## License

Licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.
