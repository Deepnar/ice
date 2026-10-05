# Handoff — ICE v3, 2026-10-05 21:33 IST

**State, not a queue.** ROADMAP owns work; paper v2 remains frozen. Current
inspection, correction and continuation concern v3.

## Told → did

**Told:** the maintainer's r4 stopped after starting the second history. Diagnose
and repair it without losing the long run's progress. Check whether concurrent
model calls/persistent residency could improve speed. Full campaign remains
maintainer-run; no push or new task branch.

**Did:** read actual failure and checkpoint receipts. The conversation-summary
worker hit `IncompleteCompletion(length)` on three successive scheduled calls:
first-history turns 1070/1095, then second-history turn 1. It asks for up to
250 words under 400 output tokens. The outage guard mistakenly counted returned
truncation as transport unavailability. It now counts recognized transport/
temporary server failures only; responsive content failures reset that streak
but remain degraded. Legacy streak reclassification preserves every failed
attempt and degraded source. Memory writers/output budgets are unchanged.

The archived original code reproduces the old recovery hash. Only the two
reviewed recovery tools differ; a closed exact old/new hash registry prevents
arbitrary code compatibility. Explicit preparation archives the immutable
checkpoint files and independently copies the committed trace prefix without
editing the old store, trace, pointer or manifest. Normal resume requires
identical settings/models/inputs/database/plan, verifies SQL restore, and
records the instrument boundary in trace/report. No general ignore-hash flag.
The spec narrowly supersedes fresh-start requirements for this instrument-only
correction; writer/configuration changes remain outside its authority.

## Current campaign state

- **R4:** `logs/z1-v3-manual-2026-10-04-r4`, paused with **1120 durable originals**
  (1119 in first history plus one in second), **82 frozen probes**, and **34
  degraded workers** (29 post-flight, five summary). Most post-flight failures
  were extraction completion length; one missing source sentence. No original
  or failure receipt was erased or turned into successful processing.
- **Continuation is prepared and verified:** private receipt at
  `seed.recovery/instrument-continuation.json`; original archive beneath that
  recovery root. Current actual resume-time configuration matches its identity;
  all eight input file digests and installed local writer manifests match.
  Original checkpoint identity is unchanged. No DB restore, full experiment
  or cloud call was launched by the repair. Resume starts second-history turn 2.
- **All writers/inputs/settings/models remain frozen.** No concurrency,
  residency, output budget, production code, database schema or AGENTS change.
  Do not edit pinned code while the maintainer's campaign runs.
- **Scope unchanged:** three histories of 1119/251/101 pairs, 1471 total,
  39 checkpoint cutoffs, 259 prompts (255 base plus four development repeats),
  117 admitted questions, four answer arms, three both-order judge contrasts.
  If all succeed: 468 cloud answers/702 judge-order requests before retries.
  Earlier failed r3/r2 checkpoints remain preserved.
- **Ground truth:** all 255 base reviews complete, zero pending; 113 valid,
  102 invalid, 40 uncertain. Two dependent repeat families (51→115, 115→216)
  are diagnostics, not full semi-LSREP. No valid multi-hop, one abstention and
  six admitted third-history questions remain coverage limits.
- **Models:** local gemma4:e4b general background and NuExtract3-Q8_0 extraction;
  cloud gpt-6-luna answers and deepseek-v4-flash judge. Cloud answers have not
  begun; current seed uses existing replies and saves as-of prompt contexts.
  Judge qualification remains pending and score_of_record=false.

## Validation and limits

Final disposable smoke **578 passed in 37.66 s**, five existing warnings;
focused **36** controls overlap. Tests cover local output versus transport
failure, legacy streaks, exact registered tool compatibility, identity/writer/
archive/prefix refusals and trace-bound transitions. Actual four-turn local
transport fault/resume exercised preflight, post-flight, originals/probe and
35-table restoration: three originals persisted before outage; only turn 4
processed on resume. Initial attempt was invalidated by editing pinned code
during the disposable test; strict identity correctly refused and it was not
credited. Final fixture ran with code frozen; databases were removed.

Actual r4 archive/identity and saved-streak controls preserve 1120 originals,
82 probes and all 34 faults. Runtime readiness checks configuration, source and
model identity without model generation or DB mutation. No full-campaign,
independent answer-quality, final throughput or concurrency result is claimed.
Private receipts and exact limits are recorded in PROVENANCE and the current
fault audit. Truncated summaries/extractions remain real quality evidence for
post-seed targeted repair/tuning; this guard fix does not cure their budgets.

## Operator entry point and research position

The receipt/archive is already prepared. From the repo root, without a venv:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-04-r4 --run
```

This retains the 1120 saved turns; **do not initialize a new bundle**. Source
history one is finished, second resumes after turn 1. Normal restore may take
time before counters advance. Ctrl+C then the same command resumes under
unchanged identity. Real transport/access/integrity failures still pause safely.
One command continues seed → snapshot → answers → judge → report.

Concurrency inspection: cached weights, GPU residency and simultaneous calls
are separate. Encoder is shared; classifier/MicroNER are small heads, while
NLI/reranker/NuNER are substantial transformers. NLI/reranker retain CPU caches
with guarded inference; background LLMs retain existing warm/idle policies.
Ollama concurrent requests need additional context/cache memory. Earlier live
GPU usage was 96%, not proof that overlap speeds the workload. Historical turns
must remain sequential; bounded independent work within a turn is a future
measured option under a shared residency budget. No speed change was made now.

Roadmap counts unchanged: 160 anchors, 93 checked/67 open, partitioned into
29 evidence-dependent, 35 later and three gates. The agreed progression remains
complete seed → output/quality review and targeted repair/tuning → freeze →
**LME oracle and semi-LSREP only**. Final resumable runners and independent
all-originals vector-baseline contracts remain separate work. No full LME-S.
Current requested false-pause repair and progress-preserving preparation are done.

## Git and propagation

Main-local commits: `f2da6cd` responsive-versus-transport guard, `9b6d708` closed
registered continuation/archive, `fe71d92` durable audit/launch docs, and runtime
readiness provenance. No push: earlier one-time authorization is consumed;
experiment publication freeze remains active. Specs, architecture, inventory,
roadmap note, provenance, traps, README/manual guides reconciled. MODELS/CLEANUP
n/a (no assignment or move); AGENTS unchanged. This handoff is written/committed
last. Private SESSION is archived under logs then reset after propagation.
