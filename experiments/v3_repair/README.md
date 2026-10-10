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
