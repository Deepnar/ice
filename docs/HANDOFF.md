# Handoff — ICE v3, 2026-10-03

**State, not a queue.** [ROADMAP.md](ROADMAP.md) owns the work. This replaces
the August pre-reseed position; historical measurements remain in provenance
and Git history, not current acceptance evidence.

## Told → did

**Told:** repair the maintainer's manual v3 seed crash, add live terminal progress
and check the later stages for similar failures before giving a restart command.
Preserve the failed attempt. The maintainer runs the full experiment; the agent
must not launch it or live cloud stages.

**Did:** reproduced the actual null-relation/object output with background NER.
Generalized the existing exact attributed source-sentence path without allowing
partial graph assertions. Added a parent-only artifact progress observer, clear
failure/log notices and stage-failure state. Audited answering/judging and fixed
acceptance of explicit provider noncompletion despite usable text or JSON.
522 disposable smoke checks, focused failure controls and actual four-turn
local replay/resume passed. Prepared a fresh reviewed bundle without launching
it; the old attempt/store/identity are preserved. Details and private artifact
hashes live in [PROVENANCE.md](PROVENANCE.md).

**Preserved prelaunch source work:** all 255 base source/answer dispositions done.
No source review remains pending. Native 131: 36 valid, 77 invalid, 18 uncertain;
linked 124: 77 valid, 25 invalid, 22 uncertain. Total 113 valid, 102 invalid,
40 uncertain. Read all imported originals (1119/251/101 turns); every label
stops at its own cutoff, with no future evidence. 248 reviewed key overrides
preserve the catalog originals. All valid rows pass real catalog/source/input/
bounds validation. Invalid and uncertain rows stay excluded; they were not
invented into a valid memory denominator. The private immutable final r3
packet/receipt and hashes are recorded in [PROVENANCE.md](PROVENANCE.md).

The earlier private initial summary/abstract/source hash and exact maintenance
row-change recording remain. Aggregate counts cannot substitute for inspecting
rewritten text. This repair changes the production extraction output contract
and cloud completion validation; no model assignment, cadence, schema, tuning
default or AGENTS rule changed. Spec, architecture, inventory, roadmap,
provenance and execution docs are current.

## Current execution state

- **Current ICE v3 bundle:** ignored `logs/z1-v3-manual-2026-10-03-r2`, already
  initialized with byte-identical reviewed label/repeat packets and unchanged
  replay policies. No seed trace, stage state or owned database exists yet.
  Do not initialize it again.
- **Failed v3 attempt:** `logs/z1-v3-manual-2026-10-01` stopped in turn4
  post-flight. Three turns processed, zero durable completed turns; its store
  retains four rows including the unfinished write. Original files/packets are
  unchanged. The repaired writer has a different code hash, so do not edit its
  checkpoint or resume it with changed source. The fresh bundle replaces it.
- **Actual v3 plan:** 1471 recorded turns across three complete histories,
  39 question checkpoints and 259 scheduled prompts (255 base plus four reviewed
  development repeat occurrences). 113 base admissions plus four repeats plan
  468 cloud answers and 702 both-order judge requests before retries.
- **Models:** local gemma4:e4b and NuExtract3-Q8_0 for memory work; configured
  cloud gpt-6-luna answerer and deepseek-v4-flash judge. Configuration was
  checked; the full cloud stages have not run.
- **No full v3 campaign is complete:** the maintainer's first attempt failed;
  the repaired bundle has not been launched. No live cloud calls were made by
  this audit. Pre-seed `cloud_answers_ready=false` is expected; labels are
  complete but a complete trace does not exist yet.
- **Verified v3 mechanics:** 522 disposable smoke checks; 79 focused provider/
  answer/judge/coordinator checks (overlap), seven claim-writer/reader controls
  and six standalone extraction completion/retry checks. All five parent-stage
  failures, provider truncation/empty/quota/transport errors, second-order judge
  resume, checkpoint rollback/identity refusal and real TTY/child progress are
  covered. Actual four-turn local-model replay of the failing history finished
  all ten jobs and the same nullable turn4, saved durable4 and resumed with
  identical written rows/store fingerprints. Test databases were removed.
  Earlier actual 35-table schema
  bootstrap/reattachment controls; actual SQL unchanged-count summary rewrite
  and real procedural decay; actual local-model two-turn interruption/recovery
  before and after telemetry. All ten memory jobs were exercised. The final
  fixture retained 79 changed semantic rows and preserved first-turn identity;
  unfinished second write was restored/replayed once and its failed tail kept.
  Disposable test databases were removed. These controls do not measure final
  memory quality, supported-answer gain, full-run latency or peak VRAM.
- **Git:** main only. The earlier explicit push published through `5672f45`;
  that one-time authorization is consumed. New parser repair `5954e94`, cloud
  noncompletion checks `775b496` and this progress/docs closeout are local.
  The experiment push freeze remains; no new push was made.

## Meaning of readiness and next entry point

The current **development campaign** is manually runnable under its declared
scope. The [manual guide](reviews/2026-10-01-v3-manual-campaign.md) owns the
command, recovery limits, per-history checkpoints, coverage matrix and final
sweep. From the repository root:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-03-r2 --run
```

One command runs/resumes seed → snapshot → answers → judge → report, with live
stage/arm bars, processed/durable turns, checkpoint probes and saved answer/
judge-order counters. PostgreSQL and Ollama must be available. No venv activation
is needed; `uv run` selects the project environment. Keep code/settings/models/corpus/labels unchanged
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
the agent's requested crash/progress/failure-audit task is complete. Completed private session
notes are preserved under ignored logs; the live session file was cleared after
propagation.
