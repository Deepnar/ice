# Handoff — ICE v3, 2026-10-04 12:30 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) owns the work. Current repairs
and checks concern v3; the paper's v2 tag/report remain frozen.

## Told → did

**Told:** finish repairing the repeated turn-80 crash, cap expensive parent
attempts at two, normalize copied source quotes where safe, and tolerate isolated
model failures in production and large experiments without hiding missing memory
work. Preserve progress and resume safety. The maintainer runs the full campaign;
do not launch it or live cloud stages. Continue the existing memory-quality
research plan rather than expanding product edge cases.

**Did:** actual local role-unit capture established a first-letter capitalization
copying error, not an outage or paraphrase. Shared Codex source selection now
resolves unique case/whitespace copies to original bytes. Complete typed rows
with unsupported quotes are withheld from graph writes while their original
chunk remains searchable alongside valid rows. Ambiguous chunk mapping retains
the original role unit. Malformed/truncated executions remain failures; no
unsupported proposal becomes a graph assertion or corroboration.

New manual bundles use explicit worker/cloud continue policies. Post-flight has
two idempotent attempts; exhausted isolated failures preserve originals and
successful independent derivatives with degraded receipts. Periodic jobs get
one attempt per due cadence, inspect committed partial changes and keep failed
attempt time separate from success. Three successive degraded invocations of
one worker checkpoint the current turn before an outage pause. Production's
existing background retry/runtime remains alive; an already sent answer is not
undone. Shared quote repair applies to production, while serial continue policy
belongs to the research instrument.

Cloud answers and each judge display order have two attempts, reserved durably
before sending, including interrupted requests with unconfirmed outcomes.
Isolated exhausted errors remain ungraded terminal rows and later questions
continue. Failed-answer pairs use no judge call. Repeated outages, access,
integrity and unknown failures pause visibly. Errors remain in planned
counts and clean/degraded-gold/other strata; fault metadata never enters answer
or judge prompts. Successful outputs are separate from processing completion.
Parent recovery now permits two total child attempts, with one 2-second backoff.
New bundles checkpoint every completed turn rather than replaying ten turns.

Mechanisms and limits: [fault audit](reviews/2026-10-03-v3-campaign-fault-audit.md),
[manual guide](reviews/2026-10-01-v3-manual-campaign.md),
[PROVENANCE.md](PROVENANCE.md), and the dated repair/reseed specs.

## Current execution state

- **Fresh v3 r4:** ignored `logs/z1-v3-manual-2026-10-04-r4`, initialized and
  validated only. Five reviewed packets are byte-identical to r3. No seed trace,
  stage state or campaign database exists; no full seed or cloud call was made.
  Do not initialize again. New config uses per-turn checkpoints and both
  continue policies. Readiness admits 117 questions; configured endpoints and
  credentials were checked without provider requests.
- **R3:** failed twice at turn 80, restoring its 70-turn checkpoint each time.
  Its verified campaign-only third attempt was stopped gracefully by SIGINT;
  interruption receipt, store, manifests and failed tails remain unchanged.
  Changed writer/policy identity requires fresh r4, not editing old hashes.
- **Earlier failures:** r2's turn-54/51-durable checkpoint and the first nullable
  turn-4 attempt remain preserved. Previous source-only/null-field, whitespace,
  native-model residency and stream-cancellation repairs remain in place.
- **Ground truth:** all 255 base source/answer dispositions reviewed, zero
  pending. Source-linked 124: 77 valid/25 invalid/22 uncertain. Native 131:
  36 valid/77 invalid/18 uncertain. Total 113 valid/102 invalid/40 uncertain;
  excluded cases were not invented into valid memory questions. The 248 reviewed
  key overrides preserve catalog originals and labels stop at their own cutoff.
- **Plan:** histories of 1119/251/101 recorded pairs, 1471 total; 39 checkpoint
  cutoffs; 259 prompts (255 base plus four development repeats). 117 admissions
  (113 base plus four repeats), four arms, 468 cloud answers and 702 judge-order
  requests before retries if all answers succeed. Two dependent repeat families
  compare recent→old at 51→115 and 115→216, not full semi-LSREP retention.
- **Models:** local gemma4:e4b general background and NuExtract3-Q8_0 extraction;
  cloud gpt-6-luna answering and deepseek-v4-flash judging. No model assignment
  changed. Reranker/NLI weights return to CPU between calls; owned background
  models retain existing guarded idle release and serial-exit cleanup. No
  maximum concurrent VRAM or all-model residency guarantee.
- **No new quality result:** independent judge qualification and output-specific
  truth reviews remain pending; `score_of_record=false`. Before a complete
  trace, `cloud_answers_ready=false` is expected, not missing source review.

## Validation and limits

Final disposable smoke: **564 passed in 38.06 s**, five existing warnings.
Seven progress controls pass. One preceding fixture incorrectly expected a
committed failure to disappear on restore (1 failed/563 passed); the corrected
control retains committed faults and discards unfinished ones. Focused and
repeated suites overlap and are not independent sample counts.

Actual saved turn-80 completions through real parser/writer/reader/NLI stored
33 original claims, wrote 20 source-supported edges and withheld two unknown
relations in 20.458 s. Actual capture took 29.467 s. Neither is an independently
graded graph truth or answer-quality result.

Actual four-turn synthetic local fault control exercised preflight, writing,
post-flight, a read-only historical probe and 35-table restore. Three injected
Codex failures exhausted six calls; originals/probe were checkpointed before
outage, and resume processed only turn 4. Prefix health excluded future faults.
Its final changed-code rerun passed; periodic cadence was disabled in this
fixture. Separate actual SQL control captured partial committed maintenance
changes and failed cadence without immediate replay or false success. Cloud
controls used controlled transports/outputs, not paid provider requests.

A copied actual 70-turn store plus synthetic next-turn interruption verified
35 tables, a 17.45 MB dump, turn-71 restoration, retained failed tail and two
rolling generations. Captures took 4.214/4.125 s; this adds snapshot overhead
that grows with store size, not a final-store speed qualification. Three setup
fixtures were uncredited and fixed without weakening old manifests. Earlier
49 runtime and six actual route/transport controls remain in provenance, not
rerun or counted as independent new observations. Full-campaign robustness,
whole-system throughput and supported answers per token remain unmeasured.

## Operator entry point and research position

From the repository root, without activating a venv:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-04-r4 --run
```

PostgreSQL and Ollama must be available. One command runs seed → snapshot →
answers → judge → report with stage/arm progress, processed/durable counts,
probe counts and persistent degraded/ungraded counters. This fresh r4 starts
from turn 1. Later Ctrl+C and the same r4 command resume with unchanged
code/settings/models/corpus/labels. Critical failures leave an explicit pause
reason; bounded recovery is not a guarantee that every possible error continues.

Roadmap remains 160 anchors, 93 checked/67 open: 29 evidence-dependent,
35 later product/research and three gates. The
[29-item evidence map](reviews/2026-09-29-v3-reseed-harness-audit.md#open-item-evidence-map)
distinguishes captured data from truth reviews, additional style/leg/model/
resource controls and tuning. Coverage limits remain no valid multi-hop cases,
one abstention and six admitted third-history questions. The agreed progression
is seed-output review → targeted repair/tuning on as-of snapshots → freeze →
**LME oracle and semi-LSREP only**. No full LME-S campaign is authorized.
Final resumable runners and independent all-originals vector-baseline contracts
are not supplied by this development command. The requested crash/recovery
repair and prelaunch checks are complete; the maintainer can start r4.

## Git and propagation

Main only. Local commits: `22eac74` original-source repair, `095a5e7` worker
recovery, `6d1a5eb` cloud reservations/continue policy, `8499d11` durable audit
and launch docs. No push: the earlier one-time authorization through `5672f45`
is consumed and the experiment publication freeze remains active. No schema
migration, new model, tuning-default change or AGENTS edit. Specs, roadmap note,
provenance, traps, inventory, architecture, README and manual guides reconciled;
MODELS/CLEANUP n/a (no assignment change or move/rename). This handoff is written
and committed last. Private session notes are archived under ignored logs and
SESSION is reset after propagation.
