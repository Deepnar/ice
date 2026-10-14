# v3 Z2 readiness — ablation switches, runners and compute (2026-10-14)

Read-only audit of what Z2 (semi-LSREP + ablations + LongMemEval oracle — not
FINAL, the paper experiment) needs (spec §2.4–2.6) against what `main` has. No code changed; `src/` and the
pinned collector files were frozen while the tuning collection ran.

## Ablation switches for the twelve groups (spec §2.5)

"Read" = can be removed on an already-written store. "Write" = removal changes
what gets stored, so the condition needs its own chronological replay.

| # | Group | Read-side off switch | Write-side off switch | Needs own replay |
|---|---|---|---|---|
| 1 | BM25/hybrid search | yes — `bm25` leg (orchestrator method / `ConfigurableOrchestrator`) | not needed | no |
| 2 | Reranker | yes — `retrieval_rerank_enabled` | not needed | no |
| 3 | Sentence-claim index | yes — `codex_claims` leg | yes — `codex_sentence_claims` | yes |
| 4 | Graph identity/relations | yes — `codex_graph` leg | **no** — `extract_codex` always runs in post-flight | yes |
| 5 | Supported turn summaries/abstracts | **no** declared reader flag | **no** — post-flight always summarises | yes |
| 6 | Rolling/batch summaries | batch leg yes; conversation fold **no** flag | yes — drop `conversation_summary`/`batch_summarize` from `maintenance_intervals` | yes |
| 7 | Procedural memory | yes — `procedural` leg | **no** — `extract_procedural` always runs | yes |
| 8 | Retrieval gating | source gate yes (`memory_source_gate_enabled`); learned B2 gate only via thresholds, no declared switch | gate changes exposure writes | yes (exposure) |
| 9 | Clusters | scope/ranking effect has no flag | yes — drop `cluster_assignment`/`cluster_merge` | yes |
| 10 | Temporal interpretation | yes — `timescope_enabled` / `timescope` | not needed | no |
| 11 | Decay/cold lifecycle | cold leg yes | yes — drop the three `decay_*` jobs | yes |
| 12 | Semantic maintenance | — | yes — drop `reflection`/`maintenance_agent`/`compaction` | yes |

Missing switches (production code, maintainer-gated): graph writer (4), turn
summary writer and reader (5), conversation-fold reader (6), procedural writer
(7), declared learned-gate bypass (8), cluster scoping reader (9). Each is a
small guarded flag defaulting to today's behaviour.

## Runners

- **Chronological replay** (`experiments/v3_repair/chronological_replay.py`)
  records `condition="full"` only; it needs a per-condition manifest that applies
  settings, removed jobs and read switches, with its own store per condition.
- **Bare-vector control**: not built. Independent role-tagged chunk index with
  the same embedder, cosine top-k, deterministic dedup and budget packing; k and
  chunk size tuned on development histories only.
- **LongMemEval oracle**: dataset present and its 500 keys frozen
  (`prepare_inputs.py::oracle_keys`). The existing `experiments/lme/lme_run.py`
  is pinned to the v2 tree; a v3 per-instance adapter is still needed.
- **Second judge**: not part of Z2 — used only in FINAL (the paper experiment),
  untuned, as a second opinion; model identity not yet chosen.

## Compute — the binding constraint for Z2

The repaired 251-turn development replay took 5.5 h with no pauses (about 65 s
per turn of memory writing plus about 1 h of question preparation). At that
rate one write-changing condition on B + C (1,444 turns) is roughly **26 h**.

Groups 3–9 and 11–12 change writing. Leave-one-out needs about nine such
replays; the additive ladder adds about ten more distinct write configurations
after BM25/reranker. Deduplicated, that is roughly **20 replays ≈ 500+ hours
(about three weeks) of continuous local compute**, before thermal throttling.
Read-only groups (1, 2, 10) and both vector controls reuse the full store.

This is an estimate from one 251-turn run, not a measured Z2 profile.
Z2 is expected to run more than once (roadmap Z2). It means the roster, the histories each arm runs on, or the replay speed has to
change before launch; that is a maintainer decision.

## Decision after this audit (maintainer, 2026-10-14)

Ablations become read-time switches on the saved checkpoint stores of one
full semi-LSREP replay per history (spec §2.5 Z2 design decision). Compute drops
from about 20 replays to 2 replays plus read passes (about 13 s per question per
condition). The missing switches shrink to read switches: graph, turn
summaries, conversation fold, procedures, learned gate, cluster scoping, a
neutral-decay treatment and maintenance outputs.
