# Handoff — ICE v3, 2026-10-04 01:39 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) owns the work. Historical v2
measurements remain frozen; current repairs and checks below concern v3.

## Told → did

**Told:** continue repairing the manual seed after its second crash; audit the
whole campaign and shared production error boundaries, preserve resumability,
and explain/check slow local processing and model reloads. The maintainer runs
the full campaign. Do not launch the experiment or live cloud stages.

**Did:** reproduced the actual turn 54 background-NER/NuExtract response. Its
source-only quote changed only whitespace. Quotes now resolve to unique original
source bytes; changed words, values and ambiguous originals remain failures.
Codex and procedural stages can progress independently before failure propagates.
The coordinator now gives fresh persisted transient/local-output failures two
bounded retries, restoring seed checkpoints and preserving successful cloud
calls/judge orders. Unknown, identity, resource, quota/authentication and invalid
cloud output failures pause visibly; no work or grade is fabricated. Attempt
receipts and pause reasons are durable. Actual urllib judge failure metadata
also reaches recovery; stale errors cannot authorize a new retry.

Native source judges no longer request unload after every call: they retain the
owned model, use existing idle release, and log useful duration metadata. Serial
seed exit releases its owned models on completion/failure/interruption. An actual
route cancellation also exposed a stuck generation flag; it now starts directly
before its protected try/finally. Cancellation before or during generation,
success and upstream failure were checked. These repairs include shared v3
production behavior; bounded parent retries belong to the manual instrument.
See the [fault audit](reviews/2026-10-03-v3-campaign-fault-audit.md) and
[PROVENANCE.md](PROVENANCE.md) for mechanisms, receipts and limits.

## Current execution state

- **Current v3 bundle:** ignored `logs/z1-v3-manual-2026-10-03-r3`, initialized
  only with byte-identical reviewed source/repeat packet copies. No seed trace,
  stage state or campaign-created database yet. Do not initialize it again.
- **Failed r2:** ignored `logs/z1-v3-manual-2026-10-03-r2` failed in turn 54
  post-flight, after 53 processed turns, 51 durable turns and 5 saved checkpoint
  probes. Files, store and recovery generations remain unchanged. The agent did
  not stop it. Corrected writer/recovery bytes change identity; never alter its
  manifest or resume this old snapshot with new code. r3 starts from turn 1.
- **First failed attempt:** `logs/z1-v3-manual-2026-10-01` stopped in turn 4
  with nullable relation/object output; its original trace/store/checkpoint are
  also preserved. Earlier source-only/null-field repair remains in place.
- **Ground truth remains complete:** all 255 base source/answer dispositions
  reviewed, 0 pending. Native 131: 36 valid/77 invalid/18 uncertain; linked 124:
  77 valid/25 invalid/22 uncertain. Total 113 valid/102 invalid/40 uncertain.
  All original histories were read, labels stop at their own cutoff, and 248
  reviewed key overrides preserve catalog originals. Invalid/uncertain rows
  remain excluded, not invented into valid memory cases.
- **Plan unchanged:** three full histories of 1119/251/101 recorded pairs,
  1471 total; 39 checkpoint cutoffs; 259 prompts (255 base + four development
  repeats). 117 admissions (113 base + four repeats), four arms,468 cloud
  answers and702 both-order judge requests before retries. Two repeat families
  compare recent→old at 51→115 and 115→216; these are dependent diagnostics,
  not the full semi-LSREP retention schedule.
- **Models unchanged:** local gemma4:e4b general background, NuExtract3-Q8_0
  extraction, cloud gpt-6-luna probe answering and deepseek-v4-flash judging.
  Cloud answering/judging needs no local answer model. Reranker/NLI weights
  return to CPU between calls; shared encoder/background NER retain existing
  policies. Keeping the owned background LLM warm is not a whole-GPU budget
  or proof that all six substantial components fit concurrently.
- **No complete campaign or new quality result:** r3 has not run; no live cloud
  call was made by this repair audit. Status confirms 0 source reviews pending;
  `cloud_answers_ready=false` is expected before a complete trace.

## Validation and its limits

Final disposable smoke: 546 passed in 34.80 s, five existing SQLAlchemy warnings.
Final disposable extraction/claim checks: 12 passed; final parser fixture
check: 27 passed. Live-runtime controls:
49 passed, including retry exhaustion without killing the scheduler. Focused
coordinator tests execute real failing Python children, successful resume and
three-attempt exhaustion. Actual route/transport controls passed 6/6 after the
before-repair cancellation failure. Actual local-model replay of the failing
source stored/retrieved three original claims, zero edges and a completion key;
the repeat took 14.728 s. Earlier focused 101/76 controls overlap these results;
do not add them into an independent sample count. Test databases were removed.
These checks establish tested recovery paths, not exhaustive full-run robustness
or supported-answer gain.

Actual repeated native frame calls under keep-alive 0, 0,-1,-1 took
3.543/6.771/6.409/3.213s; provider load 0.329/3.522/3.197/0.307s. This isolates
roughly 3 seconds of avoidable reload overhead, not a whole-run speedup.
The screenshot's 28:03/51 was about 33 seconds per fully processed recorded pair,
including preflight, summary/extraction/NLI, maintenance and checkpoint probes.
Maintenance time 148.03 s and semantic-observer overhead 0.81 s through turn 53.
Competing laptop activity, peak residency and full-run latency were not measured.
Initial summaries/abstracts/source hashes and exact maintenance row changes
remain recorded by the instrument; aggregate row counts never replace content
inspection.

## Operator entry point and research position

From the repository root, without activating a venv:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-03-r3 --run
```

One command runs/resumes seed → snapshot → answers → judge → report, with live
stage/arm progress, processed/durable turns, checkpoint probes and saved cloud
answer/judge-order counts. PostgreSQL and Ollama must be available. Ctrl+C then
rerun the same r3 command with unchanged code/settings/models/corpus/labels.
Temporary failures get 2/8-second backoff and at most three attempts. Persistent
failures leave `campaign-pause.json`; resolve the reason before rerunning.
The [manual guide](reviews/2026-10-01-v3-manual-campaign.md) owns the coverage
matrix, commands and recovery limits. Reviewed answers/gold sources enter judging,
never the answering prompt.

Roadmap unchanged: 160 anchors, 93 checked/67 open, partitioned into 29
evidence-dependent, 35 later product/research and three gates. The
[29-item evidence map](reviews/2026-09-29-v3-reseed-harness-audit.md#open-item-evidence-map)
distinguishes captured evidence from output truth reviews, additional
style/leg/model/resource controls and tuning. No valid multi-hop cases, one
abstention case and six admitted third-history questions remain coverage limits.
Independent judge qualification is pending; `score_of_record=false` remains.
The agreed progression remains seed-output review → targeted repair/tuning on
as-of snapshots → freeze → **LME oracle and semi-LSREP only**. No full LME-S
campaign is authorized. This requested crash/recovery/speed-audit repair is
complete; the maintainer can launch r3.

## Git and propagation

Main only. The earlier one-time push published through `5672f45`; its
authorization is consumed and the experiment push freeze remains active.
Earlier nullable-output/cloud-completion/progress commits and new repairs remain
local. New commits: `abda95e` source alignment, `d12d53e` residency telemetry,
`be6e71e` stream cancellation, `0121fee` campaign recovery and `1a75e06` durable
audit docs. No push, schema migration, new model, tuning-default change or
AGENTS edit was made. Specs, roadmap note, provenance, traps, inventory,
architecture, model and README/manual docs are reconciled. Cleanup ledger n/a:
no move/rename. This handoff is written/committed last. Completed private session
notes are archived under ignored logs; SESSION is reset after propagation.
