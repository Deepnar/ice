# ICE v3 post-reseed experiment preparation

This directory implements offline input preparation for the current
[repair, tuning and Z2 contract](../../docs/specs/V3_POST_RESEED_REPAIR.md).
It does **not** yet implement the tuning or final experiment runner. The old
`scripts/z1/run_v3_campaign.py` command runs the earlier development campaign,
not this final protocol. Long/cloud runs remain manually launched.

## Selected inputs

| Lane | Original pairs | Questions/references | Current qualification |
|---|---:|---|---|
| Two other r4 development histories | 352 | 73 reviewed keys: 67 tuning, 6 validation | Complete-history partition; previous development exposure, not new users |
| Paper history B | 1,119 | 91 legacy families, 638 observations at 20 historical cutoffs | Final source/correction/repeat review pending |
| Paper history C | 325 | 37 legacy families, 154 observations at 10 historical cutoffs | Final source/correction/repeat review pending |
| LongMemEval oracle | Separate public sessions | 500 official keys: 468 strings, 32 integers | New adapter/judge execution not yet qualified |

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

The outputs are `development-originals.json`, `development-keys.json`, separate
`final-originals-B.json`/`final-originals-C.json`, `final-legacy-references.json`,
`lme-oracle-keys.json`, `settings-candidates.json` and the final `manifest.json`.
The manifest binds source reviews to unchanged corpus/catalog inputs and binds
production/helper code and all product hashes. Both `replay_ready` and
`score_ready` remain false. Mutable future review answers belong beside the
immutable bundle, not inside products overwritten on resume.

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
