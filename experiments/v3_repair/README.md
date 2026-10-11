# ICE v3 post-reseed experiment preparation

This directory implements offline input preparation for the current
[repair, tuning and Z2 contract](../../docs/specs/V3_POST_RESEED_REPAIR.md).
It now includes the selected repaired development replay adapter and offline
diagnostics/candidate planning and a resumable local read-coordinate collector.
It does **not** yet implement the full tuning
executor or final experiment runner. The old
`scripts/z1/run_v3_campaign.py` command runs the earlier development campaign,
not this final protocol. Long/cloud runs remain manually launched.

## Selected inputs

| Lane | Original pairs | Questions/references | Current qualification |
|---|---:|---|---|
| Two other r4 development histories | 352 | 73 reviewed keys: 67 tuning, 6 validation | Complete-history partition; previous development exposure, not new users |
| Paper history B | 1,119 | 91 legacy families, 638 observations at 20 historical cutoffs | Final source/correction/repeat review pending |
| Paper history C | 325 | 37 legacy families, 154 observations at 10 historical cutoffs | Final source/correction/repeat review pending |
| LongMemEval oracle | Separate public sessions | 500 official keys: 468 strings, 32 integers | New adapter/judge execution not yet qualified |

Reviewed task labels are scorer inputs, separate from catalog origin/target
types. Development keys now include explicit metric contracts and per-partition
counts: only5 tuning and2 validation occurrences qualify for precise single-source
rank diagnostics. Synthesis, multi-hop, temporal/update, procedures, negative/
abstention and public/mixed questions cannot silently enter that rank cohort;
their content/support/answer outcomes stay separately qualified. These7
occurrences are not7 certified independent families or a powered optimization
result. See the [consumer audit](../../docs/reviews/2026-10-10-v3-task-metric-consumers.md).

B and C have isolated state, the same complete additive and independent
leave-one-out ablation roster, full lifecycle/decay and repeated probes. Both
warm ICE-vector and independent all-originals bare-vector controls are required.
Report each history before aggregating. Both histories share one author; prior
v2 use and B's v3 development use are disclosed. Both are excluded from new
tuning. There is no full LongMemEval-S haystack condition in this scope.

The planner preserves selected original role text, recorded replies, source-line
identity, hashes and original/constructed timestamp provenance. Constructed
source dates do not prove authentic elapsed gaps; the later aging schedule must
declare simulated time separately. Probe answers are scorer keys, never evidence
in the answering input. Legacy references remain `legacy_unqualified`, with
no reviewed source IDs or correction cutoff invented from the old score.

The source archive audit records B's one disagreement and 81 rows beyond its
paper endpoint; C's archived prefixes cover only 248 of 325 selected pairs.
Selected originals and archives remain unchanged. Full historical byte identity
is not claimed from matching conversation IDs/counts.

## Offline command and recovery

From the repository root, using the existing project environment:

```bash
uv run --no-sync python experiments/v3_repair/prepare_inputs.py --out logs/v3-input-plan
```

This requires the existing private corpus, completed r4 source reviews and
historical paper artifacts, plus the local oracle release. Those private inputs
are not published by this repository. The command makes no DB/model/API call.
It writes only below `logs/`, which stays private and gitignored.

Rerun the same command to verify/resume preparation. Complete products are
byte-stable; interruption before the final manifest seal preserves existing
products and writes only missing ones. A nonblocking directory lock prevents
two preparations from writing together. Every product uses atomic replace and
fsync. Existing corruption or input/code/policy/settings drift is refused before
any product is overwritten. Use a new output directory for a new revision;
there is no ignore-hash option and no need to erase a previous bundle.
The reviewed metric helper is also bound by the manifest; upgrading the
instrument requires a new bundle even when the original source bytes match.
The identity also binds declared local model roles/pins, classifier/schema paths
and NLI revision. Changed model pins cannot silently resume the same plan.
Unset pins are unresolved; downloaded weights, runtime capacity and endpoint/
decoding remain freeze gates.

The outputs are `development-originals.json`, `development-keys.json`, separate
`final-originals-B.json`/`final-originals-C.json`, `final-legacy-references.json`,
`lme-oracle-keys.json`, `settings-candidates.json` and the final `manifest.json`.
The manifest binds source reviews to unchanged corpus/catalog inputs and binds
production/helper code and all product hashes. Both `replay_ready` and
`score_ready` remain false. Mutable future review answers belong beside the
immutable bundle, not inside products overwritten on resume.

## Repaired development replay and audit

The new adapter uses the existing production preparation, original recorded
writer, post-flight and ten-job schedule, with source/trace checkpoints. It
selects only the tuning history (251 turns/67 reviewed questions), with a
separate owned database and progress display. Planning the101-turn/6-question
validation history is allowed; executing it is reserved until tuning/freeze.
Its outcomes must not choose knobs. No final B/C history enters this replay.

From the repository root, first plan without calls:

```bash
uv run --no-sync python experiments/v3_repair/run_development.py --inputs logs/v3-offline-input-plan-2026-10-10-r4 --run-dir logs/v3-repaired-dev-tuning-2026-10-10-r1
```

Add `--run` to execute when manually ready. Rerun the identical command with
`--run` after Ctrl+C or an interruption; the adapter automatically resumes its
owned trace/checkpoint. Completed unchanged replay preserves trace bytes. It
records degradation and pauses on persistent faults without an outer blind
retry loop. Do not delete its directory/database, change its inputs/settings or
reuse the old r4 writer state. Details go to `development.log`, with the same
artifact-based progress display. No cloud answers/judgments are made here.
The adapter pins installed writer manifests at execution; offline plans do not
certify installed bytes, runtime capacity or memory quality.

After complete replay, produce separate offline diagnostics and a pending
content-review template:

```bash
uv run --no-sync python experiments/v3_repair/audit_development.py --run-dir logs/v3-repaired-dev-tuning-2026-10-10-r1 --out logs/v3-repaired-dev-audit-2026-10-10-r1
```

The report retains actual source/task identity, separate lookup ranks and other
source-coverage diagnostics. Blank required-unit/semantic reviews remain pending;
correct source IDs cannot earn content or answer grades. Copy the review template
to a separate mutable review file, fill exact original role/hash/offset units and
through-cutoff reasons, then use the existing `scripts/z1/evidence_content.py`
auditor. Never edit the sealed template or guess units from answer-text overlap.

The manual251-turn/67-probe replay is now complete; the
[current audit](../../docs/reviews/2026-10-10-v3-repaired-development-audit.md)
retains its writer failures and semantic-review limits. Subsequent exact cited-
span verification repairs do not rewrite this baseline or its existing proofs.
Old producer/tuning execution fingerprints must still refuse changed source.
An explicit new read/derived qualification identity is required before adopting
repairs; do not bypass guards or rerun the old command as a migration. No new
final/tuning launcher is implied by the completed audit.

The bounded cheap read-side candidate seal is also offline:

```bash
uv run --no-sync python experiments/v3_repair/tuning_plan.py --inputs logs/v3-offline-input-plan-2026-10-10-r4 --out logs/v3-bounded-tuning-plan-2026-10-10-r1
```

Candidates have no measured delta/cost/verdict. The source-truth, gate/style and
aging-dependent stages remain explicitly blocked; this is not a tuning launcher
or promotion. Identical plans/reports verify/resume byte-stably; changed bytes
refuse before replacement. Use a new directory for each new revision.

After the repaired development replay completes, plan one local coordinate:

```bash
uv run --no-sync python experiments/v3_repair/run_tuning_contexts.py --development-run logs/v3-repaired-dev-tuning-2026-10-10-r1 --tuning-plan logs/v3-bounded-tuning-plan-2026-10-10-r1 --coordinate baseline --out logs/v3-read-baseline-2026-10-10-r1
```

Default performs complete-source/coordinate/code/settings checks without a model
or DB call. Add `--run` only when ready to collect that coordinate. Other IDs
come from the sealed `coordinates.json`; use a separate output directory for
each coordinate. This is local retrieval/prompt collection, not answering,
grading or automatic selection of a winning setting.

The collector locks the producer bundle against concurrent writer/read operators,
restores each exact historical prefix once for its unanswered question group,
and uses real shared preamble/final preparation under a read-only transaction.
One successful question atomically saves one content-hashed unit. Identical
reruns reuse these units; changed content, code, producer, source keys or policy
refuses reuse. Source/expected-answer metadata stays scorer-only. Explicit caps
reach both actual budget consumers; an unknown/infeasible serving window pauses
with prior units retained, never becomes a successful prompt or blind retry.

Details/progress go to `collector.log`. Ctrl+C preserves saved units; rerun the
same command to finish missing questions. Even a complete cache must restore
and verify the producer's final store before reporting success. A failed restore
remains an error with evidence retained. `contexts.json` contains the complete
four-arm traces and marks answer quality unmeasured. No writer replay or cloud
calls occur here; independent content review and later bounded Luna/Muse calls
remain required. Validation is still reserved until qualified tuning/freeze.

The completed replay predates the approved cited-span repair. Its ordinary
collector command intentionally refuses changed code. Preserve that frozen
writer and seal a separate current-reader profile, with no replay/model/DB call:

```bash
uv run --no-sync python experiments/v3_repair/reader_adoption.py --development-run logs/v3-repaired-dev-tuning-2026-10-10-r1 --tuning-plan logs/v3-bounded-tuning-plan-2026-10-10-r1 --out logs/v3-reader-adoption-2026-10-11-r2/profile.json
```

When reader qualification is ready, pass that exact path as `--reader-adoption`
to each collector coordinate and use new output bundles. Only the exact
registered summary, graph-argument, matching extraction-transport and optional
summary-retry byte transitions are accepted (five exact paths); earlier profiles remain retained
and refuse under the later source bytes. Any other change, changed prefix/dump,
settings/model drift or altered profile refuses. Identical sealed exports are
byte-stable. Existing completed units remain tied to their own reader identity.
The original writer/proofs never become new-writer outcomes; generated-note
policy can affect fallback under the current reader. This profile is a recovery
and identity instrument, not permission to launch tuning or a qualified result.
The earlier summary-only profile remains an immutable historical artifact and
cannot be reused after the separately approved graph-argument changes.

Compare a complete baseline collection with one complete candidate offline:

```bash
uv run --no-sync python experiments/v3_repair/compare_tuning_contexts.py --development-run logs/v3-repaired-dev-tuning-2026-10-10-r1 --baseline logs/v3-read-baseline-2026-10-10-r1 --candidate logs/v3-read-CANDIDATE --out logs/v3-comparisons/CANDIDATE.json
```

Replace `CANDIDATE` with the explicit coordinate's output directory/name. This
command checks all saved-unit bodies and aggregate equality, then reports actual
per-arm token/input/gate/source changes on the same producer. Only reviewed
eligible lookups enter rank comparisons; other task contracts remain separate.
It does not make DB/model/API calls or select a winner.

Optional `--source-units PATH` supplies a separate, through-cutoff source-unit
packet bound to the producer trace. Required role/hash/offset alternatives use
the existing evidence-content auditor. Such a shared packet declares literal
requirements only; content-bound semantic receipts are refused because changed
candidate context needs its own review. Missing labels remain unmeasured, and
retained text in a negation/unrelated statement earns no positive grade. A source
ID hit with an omitted value remains visible. Identical derived reports resume
byte-stably; changed inputs/evaluator refuse overwrite. Independent answer
quality, calibrated support and promotion still require later qualified steps.

Manually collect diagnostic Luna answers for a source-reviewed shortlist:

```bash
uv run --no-sync python experiments/v3_repair/run_tuning_answers.py --development-run logs/v3-repaired-dev-tuning-2026-10-10-r1 --baseline logs/v3-read-baseline-2026-10-10-r1 --candidate logs/v3-read-CANDIDATE --out logs/v3-tuning-answers/CANDIDATE
```

Default validates complete collections and seals the exact question/call plan
without credentials or an SDK client. Add `--run` only for manual cloud execution.
Use at most two `--candidate` arguments from one cheap read stage. This does not
select survivors, grade answers or promote a setting. Four arms reuse the existing
GPT-6-Luna Responses profile,1500 output-token limit and provider-default
temperature. Only the already saved final messages go to the answerer; source
references and expected answers remain scorer-only. Sessions are namespaced by
coordinate. Cloud attempt receipts are saved before requests; successful answers
are reused, two exhausted isolated faults stay ungraded and authentication/quota
or repeated outages pause with saved progress. Ctrl+C and the identical command
retain progress; changed source/code/profile/cached metadata refuses reuse.

Each coordinate directory contains immutable `projected.jsonl`, native
`answers-ARM.json` caches and distinct `grading-ARM.json` derived inputs for the
existing independent grader. Raw cache tags/bytes remain unchanged. Planning and
controlled SDK tests do not qualify native answer quality or the judge. Validation
and final campaigns remain locked behind their own qualification/freeze gates.
Full cloud-call faults are retained in private `faults-ARM.jsonl`, including the
exact frozen occurrence/coordinate/model/attempt/input identity and sanitized
stack/chain. Journal hashes are checked on resume even with a complete answer
cache; previous answers remain unchanged. Final writer/judge/internal caught-error
hooks remain separate work.

After the complete repaired replay, export partial per-source reader frequency:

```bash
uv run --no-sync python experiments/v3_repair/report_reader_usage.py --development-run logs/v3-repaired-dev-tuning-2026-10-10-r1 --out logs/v3-reader-audit-2026-10-10/report.json
```

Optional `--collection` uses a complete matched coordinate. Counts stay separate
by arm and distinguish final selected-text visibility, source lineage, unknown
eligibility and missing derived artifact IDs. This is not answer-use attribution
or complete recent/slot/artifact accounting; final granular capture is pending.

Inspect every sealed tuning field and its disposition without running trials:

```bash
uv run --no-sync python experiments/v3_repair/export_tuning_ledger.py --tuning-plan logs/v3-bounded-tuning-plan-2026-10-10-r1 --out logs/v3-tuning-ledger-2026-10-10/ledger.json
```

Current30 coordinates touch13 settings plus a hard input cap;29 conditional fields
await source/gate/aging qualification,130 stay held and one retired threshold is
excluded. Sixteen fixed maintenance controls are explicit too. Current-profile
drift refuses reuse. The [full observation/automation/tuning audit](../../docs/reviews/2026-10-10-v3-observation-and-tuning-audit.md)
lists exact ranges, human gates and future capture requirements. Private complete
fault observer qualification does not mean it is wired into the running producer
or final dispatcher; those integrations still need their actual-path controls.

Inspect chronological decay preparation without running jobs:

```bash
uv run --no-sync python experiments/v3_repair/lifecycle_plan.py --start 2020-01-01T00:00:00+00:00 --end 2020-01-31T00:00:00+00:00 --clock-basis simulated --out logs/v3-lifecycle-plan-2026-10-10/plan.json
```

This declares an example simulated interval, not source calendar evidence.
The report binds safe resolved decay/exposure/cap fields and actual consumers,
emits three-job chronological cadence events, and reports what capped endpoint
dispatch omits. Original clocks must be independently verified before declaring
`--clock-basis original`. Exact output reuse is byte-stable; changed inputs/code
refuse overwrite. No models, worker calls or database mutation occur.

The current96-cycle cap represents six days at90-minute cadence. Raising it
alone cannot reproduce seven-day eligibility and first cold transitions. The
separate ten-job gap adapter is implemented; final source-event/parent integration
remains pending. This is not a launch
command or a tuned configuration. Four exposure/cap/ranking fields missing from
the first conditional ledger are included in revision2. Export to a new path if
an old ledger exists; preserve that original. Half-life/increment/cap sensitivity
values prepare comparisons and do not change defaults or qualify memory gains.

`chronological_maintenance.drain` is the new parent's execution seam, with no
standalone CLI. It requires a restored isolated native checkpoint, matching
code/settings/registry/clock-basis identity and complete per-job ledgers. It
merges exact cadence events, runs the ten existing job functions and preserves
native bounded degradation/pause policy. Research checkpoint_every_groups
defaults1; a parent may explicitly freeze1..32 in its version2 identity. This
batches checkpoint publication only; no job, cadence or observer is skipped.
End-of-drain and outage always publish fully observed groups. An interruption
must restore the previous native checkpoint and replay the bounded uncommitted
suffix; durable groups do not rerun. Disabled jobs retain their ledger and execute nothing. Exact serial
cadence is declared research policy, not measured production idle/tick latency.

Native SQL controls preserve original source through cold movement, age graph
retention without confidence promotion, retire an explicitly stale procedure,
and restore interrupted writes to the same tables as uninterrupted execution.
Full private uncaught-job/observer/checkpoint/pause receipts are retained. Other
semantic jobs are no-op/gated on that fixture; they are not quality-qualified.
Source events/preflight/exposure/post-flight/probes, per-condition freeze/model
pins and final manual orchestration still require the new parent. The current
selected replay and its command do not adopt this adapter automatically.

## Tuning and final readiness still to earn

The static inventory records relevant numeric/boolean/weight settings, defaults,
resolved values, candidate read sites and dynamic-read gaps. It does not prove
that changing a setting changes a live decision. All live deltas/costs/verdicts
remain unmeasured. The retired procedural similarity threshold is excluded from
the tuning protocol. The former unread fusion-diversity switch now has qualified
opposing main/wide caller controls, retaining default-on behavior. Successful
reranking ignores that fallback strategy flag; a corpus that always reranks
successfully cannot credit it with an observed effect. Consumer reachability is
not answer benefit. Use a new preparation directory after this production-code
revision; earlier input bundles remain immutable.

Next prepare fresh repaired development state, qualify live consumer deltas and
implement the bounded tuning instrument. Final readiness then requires reviewed
through-cutoff source/answer/repeat keys, declared time advancement, verified
enabled/disabled writer/read paths, independent baselines, durable per-unit
recovery, short actual-path controls and the frozen call/cost table. Keep Muse
Spark 1.3 Contributor as the primary judge and qualify a second independent
family; each grades every successful final answer independently. No final score,
source-review completion, tuning result or final launch command exists yet.
The [remaining-work ledger](../../docs/specs/V3_POST_RESEED_REPAIR.md#52-remaining-work-by-next-evidence-stage--rechecked2026-10-10)
maps all29 evidence-dependent owners to7 prerequisite,16 development/tuning,
4 frozen-comparison and2 conditional stages. The eleven repair dispositions do
not close these owners or the35 later items. Complete relevant development and
launch-readiness work before moving to final experiments.
