# Handoff — ICE v3, 2026-10-03

**State, not a queue.** [ROADMAP.md](ROADMAP.md) owns the work. This replaces
the August pre-reseed position; historical measurements remain in provenance
and Git history, not current acceptance evidence.

## Told → did

**Told:** finish every source/expected-answer review, then sweep the complete
current harness against the repair plan, roadmap and session requirements.
Repair recording gaps, verify resumability, and provide one manual command.
The maintainer will launch the full experiment; the agent must not launch it.

**Did:** finished all 255 base source/answer dispositions and the harness sweep.
No source review remains pending. Native 131: 36 valid, 77 invalid, 18 uncertain;
linked 124: 77 valid, 25 invalid, 22 uncertain. Total 113 valid, 102 invalid,
40 uncertain. Read all imported originals (1119/251/101 turns); every label
stops at its own cutoff, with no future evidence. 248 reviewed key overrides
preserve the catalog originals. All valid rows pass real catalog/source/input/
bounds validation. Invalid and uncertain rows stay excluded; they were not
invented into a valid memory denominator. The private immutable final r3
packet/receipt and hashes are recorded in [PROVENANCE.md](PROVENANCE.md).

Fixed stale status language and added private initial summary/abstract/source
hash capture plus exact changed semantic rows around maintenance jobs. The
watched table list and separate observer/job time are explicit. Aggregate
counts cannot substitute for inspecting rewritten text. No production `src/`,
model assignment, memory policy, cadence or AGENTS rule changed in this sweep.
Spec, architecture, inventory, roadmap, provenance and execution docs are current.

## Current execution state

- **Current ICE v3 bundle:** ignored `logs/z1-v3-manual-2026-10-01`, already
  initialized and source-reviewed. Do not initialize it again.
- **Actual v3 plan:** 1471 recorded turns across three complete histories,
  39 question checkpoints and 259 scheduled prompts (255 base plus four reviewed
  development repeat occurrences). 113 base admissions plus four repeats plan
  468 cloud answers and 702 both-order judge requests before retries.
- **Models:** local gemma4:e4b and NuExtract3-Q8_0 for memory work; configured
  cloud gpt-6-luna answerer and deepseek-v4-flash judge. Configuration was
  checked; the full cloud stages have not run.
- **No full v3 campaign was launched:** no campaign database, seed trace or
  completed stage exists in this bundle. Pre-seed `cloud_answers_ready=false`
  is expected; labels are complete, but a complete trace is not present yet.
- **Verified v3 mechanics:** 499 disposable smoke checks; actual 35-table schema
  bootstrap/reattachment controls; actual SQL unchanged-count summary rewrite
  and real procedural decay; actual local-model two-turn interruption/recovery
  before and after telemetry. All ten memory jobs were exercised. The final
  fixture retained 79 changed semantic rows and preserved first-turn identity;
  unfinished second write was restored/replayed once and its failed tail kept.
  Disposable test databases were removed. These controls do not measure final
  memory quality, supported-answer gain, full-run latency or peak VRAM.
- **Git:** main only; local checkpoints `6fb6eaa` (output inspection) and
  `75e767d` (source readiness). Nothing pushed; the experiment push freeze
  remains active. An explicit request to push authorizes that push only.

## Meaning of readiness and next entry point

The current **development campaign** is manually runnable under its declared
scope. The [manual guide](reviews/2026-10-01-v3-manual-campaign.md) owns the
command, recovery limits, per-history checkpoints, coverage matrix and final
sweep. From the repository root:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-01 --run
```

One command runs/resumes seed → snapshot → answers → judge → report. PostgreSQL
and Ollama must be available. Keep code/settings/models/corpus/labels unchanged
after starting; rerun the same command after interruption. Exact frozen prompts
drive the cloud answerer; expected answers and gold sources enter judging only.
Small recent→old repeat comparisons are included; full semi-LSREP is later.

Readiness does not close all research questions. The v3 roadmap remains 160
anchors: 93 checked/67 open, partitioned into 29 evidence-dependent, 35 later
product/research and three stage gates. The [29-item evidence map](reviews/2026-09-29-v3-reseed-harness-audit.md#open-item-evidence-map)
and manual guide distinguish raw evidence captured here from output-specific
truth reviews, additional style/leg/model/resource controls and tuning. There
are no valid multi-hop cases, one abstention case and six admitted third-history
questions. Independent judge qualification remains pending, and campaign
reports retain `score_of_record=false`. Current source reviews are adjudication,
not independently qualified answer-quality measurements.

The agreed progression remains seeded output review → targeted repair/tuning
on as-of snapshots → freeze → **LME oracle and semi-LSREP only**. No full LME-S
campaign is authorized by this entry point. The maintainer runs the experiment;
the agent's requested prelaunch task is complete. Completed private session
notes are preserved under ignored logs; the live session file was cleared after
propagation.
