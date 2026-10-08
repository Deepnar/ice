# ICE v3 — repairs after the first complete reseed, tuning, and Z2

Assumes decided specs: `V3_REPAIR.md` (existing whole-system repair authority
and source preservation), `RESEED_PLAN.md` (r4 identities and chronological
replay), `G63_extractor_decision.md` (NuExtract3 extraction),
`BG_LAYER_FIXES.md` (separate E4B background generation), `T_temporal.md`
(source time and lifecycle semantics). The dated decisions here supersede the
older execution order and experiment roster in `Z1_tuning_coverage.md` and
`FINAL_experiments.md` for this v3 phase; their historical measurements remain.

**Status: decided 2026-10-07, finalized 2026-10-08; not implemented or measured.**
ROADMAP is the queue. The numbered packages below decompose existing entries,
not a new G-track or claims of eleven newly discovered defects. The completed
r4 evidence is in [PROVENANCE](../PROVENANCE.md#2026-10-07--v3-completed-r4-development-campaign).
This is the next plan referenced by the retained, private `docs/SESSION.md`.

## 1. Decisions

### 1.1 Scope, sequence and completion

The maintainer requests evidence-led core memory repair, including bounded
design improvements where the likely cause is not yet settled. Execute:

1. Correct the measurement interpretation and freeze r4 as the development
   baseline; reuse its sources, outputs and 39 as-of stores.
2. Repair long-source compression, answer-bearing retrieval, failed writes,
   graph identity/update decisions, procedural usefulness and memory gates.
   Qualify the judge and review semantic outputs alongside these repairs.
3. Run the bounded hyperparameter protocol in §2.3, then freeze configuration,
   code, model/endpoint identities, labels and experiment manifests.
4. Run **Z2: one long-conversation semi-LSREP with full memory lifecycle and
   additive/subtractive ablations, plus LongMemEval oracle**. “LME-S oracle”
   here means the oracle evidence-session condition, not a full S haystack run.
5. Diagnose final failures without silently tuning the same final results.
   A further repair creates a new revision; preserve and report the first run.

Quality means supported answers at a given total prompt budget, or fewer tokens
at preserved quality. More nodes, denser graphs, equal leg shares, fewer logged
errors, or 100% benchmark accuracy are not substitute objectives. Do not keep
changing a final benchmark until it becomes perfect. Repair an identified
mechanism now; do not wait for Z2 to rediscover r4's confirmed defects.

This planning/closeout pass changes no runtime. Subsequent implementation is
within the existing authorized v3 repair scope. No new model is required by
this plan. Keep the local/cloud switch; local E4B and NuExtract3 write memory,
cloud GPT-6-Luna answers, Muse Spark 1.3 Contributor Responses remains the
selected judge subject to qualification. No silent judge/provider substitution.
Keep all originals and source-linked sentences; a failed verifier is neither
deletion authority nor proof the source fact is false.

Product packaging, MCP/REST expansion, deletion edge cases, unrelated routing
features, broad model bake-offs and the frozen v2 paper are outside this repair
pass. Session instructions explicitly retain SESSION until the maintainer asks
to clear it. Publication authorization covers this closeout/merge/push only.

### 1.2 Evidence ledger and repair packages

**Confirmed** means the recorded output and traced contract identify the
mechanism. **Hypothesis** means a defect/weak outcome is observed but its remedy
needs a bounded comparison. **Unexercised** means r4 cannot establish behavior.
Each package ends with a decision and production-path validation; no endless
candidate search. Ground-check code and prior experiments before implementing.

| Package | Evidence and diagnosis | Decided next action; acceptance |
|---|---|---|
| P0 — trustworthy measurements | Confirmed: `report_v3_replay.py` checks whether `summary_support` is absent, not nested summary/abstract statuses. It reports zero unknowns despite 986 unknown summaries. A source-ID hit also survived after the required answer content disappeared. | Add status/reason/denominator counts and semantic evidence stages. Keep raw reports immutable; write corrected derived reports with input hashes. Test absent, empty, unknown, contradicted and supported independently. A deliberately missing answer span must fail semantic support while still passing source-ID recall. |
| P1 — useful long-source summaries and folds | Confirmed: all 986 nonempty summaries hit `SupportInputError`; the whole attributed turn plus summary must fit 512 NLI tokens. Almost every conversation/batch note therefore renders originals. This is input-contract failure, not 986 NLI contradiction decisions. | Implement source-linked units and bounded claim verification (§2.1), then reuse the same contract for turn summaries, abstracts and rolling/batch notes. Preserve complete-source coverage manifests, exact-value alternatives and later corrections. Verify actual substitutions and quality/token changes, not the mere existence of a summary. |
| P2 — answer-bearing source selection | Confirmed: original evidence can exist and its parent ID rank first while every supplied excerpt omits the required value. Other stable misses have no candidate or lose candidates before final assembly. Exact root differs by case. | Trace each stable miss to index/eligibility, candidate generation, rerank, chunk selection, representation, budget or answer use. Rank relevant child windows with original offsets; offer adjacent/parent evidence when a fragment lacks its referent or requested code block. Reuse the current embedder/reranker, deduplicate overlap, and repack under the same total budget. No gold text or expected answer may guide runtime selection. |
| P3 — failed or incomplete local writes | Confirmed: 58 terminal extraction-output failures, 3 terminal post-flight incomplete completions, 20 incomplete conversation folds; 14 admitted occurrences have a degraded gold source. Exact remaining parser/output causes require the saved response. | Bucket the saved receipts, reproduce each distinct contract failure, and repair template/chunk/output handling at its existing seam. Typed unsupported proposals remain quarantined; truncated structured output remains incomplete. Size requests using actual complete input plus output reserve. Validate repaired failed sources and matched successful controls, then replay affected chronological suffixes only in new development stores. |
| P4 — graph identity, relation precision and confidence | Observed: 2,409 components, 82.2% degree-one nodes, alias-like duplicates, 80% of edges at confidence 0.35 below the 0.5 direct-read floor. All edges have supported internal links; independent precision is unknown. Sparse topology alone is not a defect. | Review role/direction/polarity/modality and every confidence tier, including excluded facts. Repair source-backed alias resolution and unsafe relation canonicalization where review confirms errors. Preserve separate entities when identity is ambiguous. Treat confidence as extraction evidence, access as exposure. Decide threshold changes from calibrated source truth and answer effects; never connect nodes or lower trust just to improve counts. |
| P5 — temporal updates and maintenance reachability | Confirmed coverage restriction: 508 distinct reconciliation items touched across 1,000 inspections all have synthetic-time sources, so the original-timestamp guard prevents a decision call. The priority scanner spends its entire cap on that detector. This does not measure the model's reconciliation accuracy. | Record authority/eligibility/defer reasons per detector. Add verified within-conversation source order distinct from calendar authority; permit an explicitly supported later correction to compare ordered sources without inventing a date. Never infer cross-conversation order from import order. Prevent ineligible backlog from consuming the whole decision budget; fairly schedule eligible core detectors. Test genuine dates, ordinal-only corrections, coexistence, reported/hypothetical claims and historical reads. |
| P6 — procedural memory that contributes | Observed: 95 patterns, 2 active, zero selected procedural fragments on 117 answered occurrences. The activation/relevance/packing bottleneck is not yet isolated. | Trace cited distinct support → activation → eligible candidates → selected text → answer use on actual procedural questions. Repair matching/eligibility only where source-backed repeated behavior supports it; distinguish explicit instructions from inferred habits. Never reactivate the former reflection shortcut or count repeated reads as supporting turns. Keep inactive material stored; require relevant-answer gain or context savings to change activation policy. |
| P7 — gate, entity activation and style | Coverage gap: all 259 probes retrieve; a memory-question panel cannot measure unnecessary retrieval. Historical gating skipped some requests, but source-matched false-positive/false-negative labels are absent. | Reuse the existing source-need/style fixtures plus reviewed r4 misses. Hold meaning fixed across terse, rambling, lowercase, typo and punctuation variants; include current-supplied facts, banter and historical requests. Trace MicroNER/preflight activation and rescue to final evidence and answers. Keep the best complete existing source-proof mechanism unless a bounded replacement improves both misses and false suppression. |
| P8 — judge reliability and question intent | Confirmed: the same saved full answer receives different final absolute grades across contrasts on 36/111 fully graded cases. Some general questions are judged against an unstated historical-answer requirement. This is not proof that all disputed model grades are wrong. | Review disputed keys and saved answers; grade each unique answer independently once, cache by answer+question+sources+rubric+judge identity, then compare preferences separately. Qualify on correct/partial/incorrect/uncertain real examples and opposing controls (§2.2). Preserve old grades and label revisions; do not regenerate good answers just to repair the judge. |
| P9 — completion and resource costs | Confirmed: 12/468 answer records ended `incomplete`; receipts do not preserve a precise provider reason. Whole-stack latency/VRAM and real-time concurrency were not measured. Output-cap causality is a hypothesis. | Preserve completion details, reasoning/output usage and transport versus content failure. Reproduce a small failed-input set at the existing profile before changing token reserves. Profile actual local-job stages and resident/peak memory; use bounded batching and existing ownership/scheduling, not unrestricted local concurrency or more resident models. Retain per-request durable attempts and byte-stable completed artifacts. |
| P10 — semantic maintenance outputs and untested lifecycle | Observed: 21 clusters, 1,750 enriched entity descriptions, 109 session summaries, 8 batch summaries; these are output counts, not quality. No cold rows or cold retrieval occurred. | Review actual member/source-backed names, enriched descriptions and notes for attribution/completeness. Apply the P1 evidence contract to generated memory that reaches readers. Run a short controlled time-advance/restore path to exercise decay, compaction, warm→cold→read and all scheduled core jobs before Z2. Do not claim the synthetic dates establish natural calendar retention. |

P0–P10 are **11 execution packages**, not eleven roadmap checkmarks. Fixes may
share one mechanism (for example P1 supports P10). Complete the core repairs
before tuning; judge review can proceed alongside local repair. For a hypothesis,
use the retained failure and an opposing control, implement one reasoned
candidate, and allow at most one evidence-led revision before deciding keep,
replace, or retain the baseline with a precise limitation. A newly demonstrated
integrity defect overrides that search limit and must be repaired. Focused
experiments are allowed; repeated broad reseeds are not the default tool.

### 1.3 Removal and small decision-model policy

No component is authorized for deletion merely because r4 selected zero outputs.
Procedural memory may be starved; cold memory had no eligible population;
reconciliation was blocked before the model; summaries failed length admission.
First remove the cause, then measure utility. Defaults may be simplified only
after matched-budget omission preserves reviewed support/answers and reduces
cost across the relevant task slice. Retain the ablation switch and original
data. Structural deletion requires the existing explicit deletion decision and
consumer/doc sweep; this plan grants no blanket deletion permission.

Laya-like open decision models remain a bounded option for entity identity,
relation equivalence, reconciliation or candidate utility **after** eligible
source-backed tasks exist. Their prior source-need and generic-relation tests
do not qualify these jobs. Compare against deterministic resolution and the
existing E4B on the same opposing-source controls. A candidate must improve
the relevant decision/answer measure without new false merge/expiry controls,
and fit the measured whole-stack memory/latency budget. No extra permanently
resident model, hosted JEV dependency, or classifier/NLI replacement is implied.

## 2. Algorithm and data model

### 2.1 Long-source support and evidence selection

1. Enumerate **all** original role units with immutable source ID/hash and
   offsets. Build token-bounded windows with overlap and exact offsets using
   the deployed tokenizer, reserving space for hypothesis, role and provenance.
   Keep paragraph/context relationships; punctuation alone cannot decide truth.
2. Request source-linked summary assertions from the existing local background
   model. Every assertion includes its proposed source spans and qualifiers.
   Resolve spans to exact original text; unresolved spans are unknown. No
   generated assertion can become its own verification premise.
3. Check each complete assertion against its complete cited window(s), within
   the verifier's real bound. Retain speaker, event, time, modality and negation.
   If required context spans too many windows, keep the source alternative;
   do not truncate or OR together unrelated entailment scores. Check all
   assertions in a composed summary; unsupported additions fail that summary.
4. Store a versioned verification manifest: source hashes/offsets, assertion
   text/hash, verifier revision, method revision, actual token counts, status,
   reason, candidate hash and coverage over original units. Changes to policy
   invalidate old derived approval without deleting originals. Extend existing
   `representation_verification`/note manifests before adding duplicate stores.
5. **Faithfulness and completeness are separate.** An entailed short note may
   omit the needed answer. Keep original exact strings and query-selectable
   child windows; term coverage remains only a retention signal. Code/value,
   update and synthesis controls must verify required information survives
   final assembly. Do not turn term lists into a truth metric again.
6. Teach `choose_representation` and existing note readers to use only current
   manifests. Search/query-pack original excerpts when the compact view cannot
   cover the question. Count supported summary substitutions, source fallbacks,
   missing facts, prompt tokens and inference cost by input length.

This adapts the granularity insight in
[SummaC](https://aclanthology.org/2022.tacl-1.10/): sentence-sized NLI inputs
with document-level aggregation. That paper does not qualify ICE's model or
our authority/completeness checks. The previously tested ModernBERT and BGE
long-input candidates remain unpromoted; their failed controls are in MODELS.
Do not redo that model search before repairing how inputs are constructed.

P2 evidence receipts add exact rendered source spans, available answer-support
units for **evaluation only**, and distinct stages: indexed/eligible/generated/
ranked/budgeted/rendered/answer-supported. Evaluate required claims and literals
against reviewed alternatives and original context; substring checks suffice
only for exact-value questions. A semantic decision may be uncertain and must
not silently become recall=1. Gold annotations never enter query expansion,
candidate generation, reranking, or the answering prompt.

### 2.2 Absolute scoring and qualification

Reuse all saved successful r4 answers. Build one blinded packet of the 36
cross-contrast disagreements plus 24 deterministically stratified agreement
controls (across grades, scopes, lengths and task types; backfill only if a
cell is empty). Show original question, complete dated sources and reviewed
requirements, but hide arm/model labels. Distinguish historical assistant
advice from world truth and requested current knowledge. Rewrite an ambiguous
question only in a new packet revision; its old answer cannot score the new
question without a new answer call.

**USER-REQUIRED:** review the prepared 60-answer packet in batches, approximately
45–90 minutes total, with correct/partial/incorrect/uncertain and a brief source
reason. The agent prepares all evidence first and continues independent repairs
while labels are pending. Done means each selected row has a disposition;
uncertain rows remain uncertain rather than being forced into a binary label.

Qualify the actual independent-answer rubric against these labels, including
planted role/date/negation errors and irrelevant same-speaker sources. Require
at least 90% agreement on definitive reviewed labels and no systematic
false-accept pattern on the opposing controls. Report the confusion matrix,
denominators and uncertainty; this is a development admission rule, not a
population guarantee. Before reading the labels, reserve 20 of the 60 cases by
stratified fixed-seed selection for qualification; the other 40 support rubric
development. Do not tune the rubric on the reserved subset. Failed qualification
permits one rubric repair and fresh controls, then a separately qualified second judge profile;
model choice never rewrites old verdicts.

Absolute grades are keyed by immutable input/output identities and are reused
across contrasts. Preferences still use both A/B orders with disagreement
reported. Use a second qualified model family for disputed final cases and a
frozen random audit; human review resolves remaining disagreements, otherwise
report uncertainty. Historical v2 judge calibration does not qualify v3.

### 2.3 Hyperparameter tuning before Z2

**Inputs:** immutable r4 baseline, versioned repaired development stores and
reviewed semantic keys. All r4 questions/results are development material.
Partition new controls by conversation/source event/question family, never by
paraphrase row; reserve an untouched validation packet before looking at new
results. With only three histories, use leave-one-history-out diagnostics and
report heterogeneity; do not claim a random population or user-generalization
interval. Final semi-LSREP families/conversation and oracle instances do not
select hyperparameters.

**Knob inventory first:** enumerate current Settings and every actual consumer.
For each relevant knob record default/frozen value, eligible population, an
opposing-value decision delta, cost, and verdict: load-bearing, plateau,
unexercised, or dead consumer. A declared but unread knob is repaired before
sweeping. Historical duplicate constants/defaults are not assumed current.

**Stage order and bounded candidate sets:**

| Stage | Sweep | What stays fixed / keep rule |
|---|---|---|
| Budget and representation | Total prompt caps 4k, 8k, 16k, 32k, plus unchanged r4 policy as reference; then recent/evidence allocation at 0.75×, 1×, 1.25× current value within valid bounds | Same answering model, output reserve and current input question. Include all system/recent/memory tokens in cap. Refuse infeasible caps explicitly. Compare quality at equal cap and actual cost; preserve exact facts. |
| Candidate selection | Current warm-vector/BM25/claim/chunk limits at 0.5×, 1×, 2×, rounded/clamped to valid bounds; reranker shortlist 32, 64, 128 | Freeze selected representation contract; inspect missing answer spans and candidate redundancy. No writer replay for a read-only limit change. |
| Fusion and ranking | Current RRF k at 30, 60, 90; each evidenced leg weight at 0.5×, 1×, 1.5× current; reranker on/off as a distinct mechanism arm | One coordinate changes at a time; log real decision deltas, task outcomes and latency. Zero-weight is not automatically a complete ablation. |
| Trust and activation | Only source-reviewed graph tiers and procedural evidence thresholds; candidate range derived from labeled support/false-action cases, not arbitrary global lowering | Freeze independent source-truth constraints. Reads cannot create corroboration. If no eligible examples exist, mark unexercised and retain default. |
| Gates and temporal policy | Existing gate thresholds/bumps and temporal-label consumer on meaning-preserving memory/no-memory controls | Recall and false-suppression guardrails; temporal changes need original-date and ordinal-only strata. T5 temporal-label effects are tested only after a frozen baseline. |
| Lifecycle/resource | Solve declared half-life targets, then compare baseline and 0.5×/2× duration sensitivity on a time-advanced development history; batch sizes only within measured residency headroom | No future-time jumps merely to produce desired cold counts. Test reinforcement/decay interaction and real job ordering, not algebra alone. Local foreground residency is a separately labeled optional profile. |

Run cheap exact/source-stage checks first. Spend new cloud answers only on the
baseline and at most two surviving candidates per stage. A candidate must pass
all source/authority controls and either improve reviewed supported-answer
count at equal budget or keep it equal while reducing actual median prompt
tokens by at least 10%. Any new failure on a previously correct required fact
is inspected before promotion; a net score alone cannot excuse false expiry
or source corruption. Ties keep the current setting. These are operational
development rules, not statistical proof of superiority.

Finish with up to 12 frozen-seed interaction checks around the three most
load-bearing selected knobs (valid ±20% settings), plus an all-originals
answer control on the targeted losses. Reuse cached answers only when complete
assembled input and provider identity match. Publish the knob verdict ledger,
quality/token curves, per-task losses, settings and a final freeze manifest.
No exhaustive combination search, no tuning against final scores.

### 2.4 Z2 semi-LSREP: recorded replies, complete memory effects

**Fixture selection:** one complete long conversation with ordered originals,
updates, recurring topics/procedures and interpretable sessions. Prefer a
different history from r4 to reduce development leakage. Inspect the existing
private mature/stitched and public multi-session assets before choosing; freeze
path/hash, source dates, role map, turn count, sessions, gaps, selected families
and source/answer reviews in a private manifest. Do not silently concatenate
different users into one person. If only a development history is usable,
label the experiment a development longitudinal replication, not held-out.
No corpus is claimed selected merely because a historic spec names it.

For **every recorded original prompt** in chronological order:

```text
restore/validate the condition's own durable prefix
set source clock; drain memory jobs due before this event
run the real shared preflight and final prompt assembly
apply the normal selected-evidence exposure/retention effects once
store the recorded original assistant reply (no regenerated replay reply)
run real post-flight, extraction, update and scheduled memory work
save the complete state, job cadence and append-only receipt
at scheduled checkpoints: observe and answer probes without memory writes
```

This measures the memory mechanism under **fixed recorded responses**. It does
not show what an alternate model would have said or how a user would then
respond. All conditions receive the same recorded turns. Only diagnostic probe
answers are generated by the cloud answerer; their questions/replies never enter
future state. Whole mechanisms disabled by a condition must also be absent
from historical preflight/exposure and writer/maintenance effects.

Use the actual runtime's ten periodic memory jobs and a declared synchronous
turn-boundary schedule, including time gaps, decay, compaction, clustering,
summary/reflection, maintenance and warm/cold transitions. Drain gap-due jobs
without ingesting future evidence. Preserve timestamps' origin: authentic
calendar gaps are one stratum; declared simulated aging is another. A model's
wall-clock generation time is not conversation time. Record job eligibility,
calls, changes, failures and source effects; no cold events means no claim about
cold quality, even when the function was called.

**Repeated probes:** freeze at least 20 distinct reviewed fact/task families if
the selected history supports them, covering exact recall, synthesis, updates,
procedures, temporal questions and abstention. Include each eligible family at
first answerability, first cutoff after the actual recent window, a later
section (target at least 100 intervening turns), and final checkpoint. Add a
pre/post-correction pair for update families and pre/post-cold pair only where
an authentic or explicitly simulated lifecycle transition exists. Review the
answer key through **each cutoff**, not only the first. Use the whole qualified
schedule, not a few favorable families; label unsupported category/age cells.

Measure first answerability, retained-correct fraction conditional on earlier
correctness, recovery, correct→incorrect and stale→updated transitions, evidence
survival and tokens/latency at each age. Keep original-state questions separate
from current-value questions. A family with five checkpoints is one dependent
family trajectory, not five independent users. A run with one conversation
supports a case study; block resampling by family is descriptive within that
conversation, not a population confidence claim.

### 2.5 Z2 baselines and additive/subtractive ablations

**Shared answer envelope:** same current question, source/query-time formatting,
recent window, system instruction, answering provider/decoding/output reserve
and total token cap for all memory conditions. Expected answers never enter
that envelope. Include recent-only to expose answerability without retrieval.

**Two vector controls are required:**

- **Warm vector:** the current ICE warm-vector leg with its shared gate,
  reranker and representation policy. Exclude graph/claim/procedure/summary
  legs, source notes, slots/bookmarks, timeline helpers and query expansion.
  Its same-store form measures direct read-path contribution; label that shared
  state inheritance. A lifecycle comparison uses its declared condition state.
- **Bare vector:** independent append-only index of every original role-tagged
  history chunk available at the cutoff, using the same embedding model,
  deterministic chunking/overlap, raw text and cosine top-k. No learned memory
  gate, reranker, BM25, graph/claims, summaries, procedures, importance/recency
  boosts, consolidation, decay, cold exclusion or memory-write strengthening.
  Deterministic overlap deduplication and budget-fitting are allowed and declared.
  Record parent/source spans. Tune its k/chunk-size on development data only;
  keep the frozen best baseline, not a deliberately weak choice.

Bare vector still needs embeddings, an index, source metadata and prompt
packing; “nothing” means none of ICE's additional memory mechanisms. Do not
make this control weak by giving it fewer original sources or a smaller cap.

Before launch, enumerate components from **actual reachable consumers**, with
switches for writers, readers, helpers, prompts, retention and scheduled jobs.
The initial mechanism groups (split only if effects are independently viable):

1. BM25/hybrid candidate search.
2. Reranker.
3. Attributed sentence-claim index.
4. Graph identity/relations/traversal and grounded query expansion (requires
   source claims; disabling graph may keep claims, disabling claims forces
   graph off unless a separately qualified source-only dependency exists).
5. Supported turn summaries/abstracts.
6. Rolling conversation and batch summary memory (requires supported-note
   contract; include source-mode notes as part of this mechanism).
7. Procedural extraction/activation/retrieval.
8. Learned/source-proof retrieval gating.
9. Cluster organization and scope/ranking effects.
10. Temporal query filtering/timeline interpretation (source dates remain in
    all conditions; disabling this is not permission for future evidence).
11. Decay/retention/cold lifecycle.
12. Semantic maintenance: reconciliation, enrichment/reflection and compaction
    effects not already owned by another group. Record its dependency overlap;
    split update versus reflection only when the switch truly isolates them.

**Additive:** start from the independently specified raw vector condition and
add groups in the above dependency-respecting order, one at a time; also report
the warm-vector and recent controls. Freeze the order before seeing Z2 answers.
Order-dependent gains are conditional contributions, not intrinsic values.
**Subtractive:** start from the same complete full-ICE configuration each time,
disable one group, rerun that condition, then restore full configuration before
removing another. Never cumulatively subtract. If a parent switch necessarily
disables dependents, name the compound removal and do not call it a singleton.

Run all requested additive and leave-one-out conditions on the frozen
semi-LSREP panel. Do not quietly select only cheap or winning arms. With K
independent groups, the naive layout has K+1 additive states and K full-minus
states plus controls; deduplicate identical endpoint configurations by complete
manifest hash. Produce the exact arm/call/cost count before operator launch.
Full-ICE and both vector baselines also run the oracle condition. Additional
oracle ablations are diagnostics only if explicitly included before its freeze;
the comprehensive mechanism ablations belong to semi-LSREP.

**State rules:** read-only omissions can reuse the identical frozen as-of store
only for labeled read-side contrasts. Writer, entity identity, summary,
procedural, gate-exposure, maintenance or decay changes require their own
chronological replay. Reuse source bytes, embeddings and response caches only
when all relevant prompt/model/config identities are identical; never reuse
full-system derived state as though an absent mechanism had not created it.
Separate immutable r4 originals from these new condition stores.

### 2.6 Z2 LongMemEval oracle

Pin the downloaded official oracle dataset file/revision and hash; inspect
all schema/count/type coverage before running. The
[official LongMemEval definition](https://github.com/xiaowu0162/LongMemEval)
uses evidence sessions only and contains 500 instances. Oracle session arrays
are not guaranteed chronologically sorted: sort linked ID/date/session tuples
stably, preserve every role turn, and verify question-time eligibility. Do not
substitute the full S haystack or the separately named LongMemEval-V2 dataset.

Per instance, isolate state and replay the supplied history through the repaired
memory path using recorded replies. Answer at `question_date` with full ICE,
warm vector and bare vector. Also send the complete supplied evidence sessions
directly to the **same answerer** as an all-evidence oracle control when they
fit its verified context window. Record any capacity failure instead of
truncating the reference. This separates memory loss from answerer/key limits.
Preserve upstream question types, source session/turn labels and abstentions;
do not infer new answers from answer-session IDs alone.

Target excellent oracle performance and inspect every miss, but do not require
manufactured 100% correctness to let the result exist. Freeze before the first
complete run. Any subsequent repairs evaluated on those failures are labeled
oracle development revisions; a future generalization claim needs untouched
data. Report the upstream-compatible score separately from richer source-based
partial/uncertain grades and completeness/errors.

## 3. Files and integration points

Use existing mechanisms; these are seams verified in current main, not promised
new independent implementations:

- P0: `scripts/z1/report_v3_replay.py::summarize`, `seed_v3.py` trace,
  `answer_as_of.py`, `judge_answers.py`; preserve existing artifact identities.
- P1/P10: `src/memory/support.py::score_pairs/verify_support/supported_current`,
  `representation.py::verify_representations/choose_representation`,
  `summary_snapshot.py`; `post_flight.py`, `conversation_summary.py`,
  `batch_summarizer.py`, `reflection.py` and their real reader consumers.
- P2/P4: `src/retrieval/orchestrator.py` existing chunk, claim, graph and budget
  paths; `reranker.py`; `src/memory/claims.py`, `relation_support.py`,
  `src/workers/codex_extractor.py` and source-backed entity operations.
- P3: `extraction_result.py`, `codex_extractor.py`, `post_flight.py`, existing
  bounded worker recovery; real rejected responses remain private fixtures.
- P5: `_reconciliation_evidence`, maintenance `_decide_reconciliation` and
  `detect_all`; extend source-order provenance through actual import/store/
  archive/claim consumers if needed, with a migration only for new fields.
- P6: `procedural_extractor.py`, `reflection.py`, orchestrator
  `_procedural_lookup`, existing support identities and temporal fields.
- P7/P9: `memory_decision.py`, `source_need.py`, `source_proof.py`, classifier/
  NER consumers, `bg_client_factory.py`, runtime scheduling; cloud completion
  receipts in `experiments/lme/cloud_provider.py` and existing z1 runners.
- Tuning: extend `scripts/z1` current as-of/control/config seams and add a
  versioned private sweep manifest; document verdicts in `docs/tuning_report.md`
  when measurements exist, not invented results now.
- Z2: a new current-v3 coordinator under `experiments/v3_repair/` reuses
  proven checkpoint/cloud-recovery/clock modules; inspect existing
  `experiments/mature/` and `experiments/lme/` for reusable mechanisms. Keep
  frozen v2 experiment code unchanged. Document the actual CLI after it exists.

## 4. Failure modes and resumability

One operator entry point owns each campaign and can run all declared stages or
a named remaining stage. New Z2 runner design must retain r4's durable unit
semantics: full state+trace checkpoints, per-answer/per-order atomic writes,
reserved cloud attempts, model/endpoint/settings/source identity, active-run
locks, status and useful progress per condition. No manual sequence of loosely
coupled scripts with untracked dependencies. The runner is **not implemented
yet**; do not present a speculative command as runnable.

Ordinary isolated model failures retain originals, bounded attempts and explicit
degraded/ungraded records. Transport outages, auth/quota problems and identity
failures pause for the operator without deleting progress. Completed upstream
stages are byte-stable. Never append new model/settings results to a frozen r4
file or bypass changed-hash guards to save work. New repair children bind their
baseline snapshot and changed configuration explicitly.

Exercise Ctrl-C/restart during a local write, snapshot boundary, gap job,
cloud reservation, first judge order and report. Compare resumed versus
uninterrupted state fingerprints, scheduled-unit identities and costs. An
unconfirmed remote request may have been billed: do not promise exactly-once
billing. Progress separates planned, processed, successfully saved, degraded,
excluded and pending. Every final denominator reloads durable artifacts.

## 5. Validation and exit checklist

Each package gets a positive/opposing source control, one isolated check and
one actual production-path check. Use disposable DBs with current schema; keep
private counterexamples out of Git. Cheap semantic checks and saved-output
review precede costly cloud calls. No new test merely restates an implementation.

**Repair exit:** all P0–P10 have a written implemented/validated disposition or a
bounded empirical decision with the affected behavior explicitly retained and
its coverage limit named. Confirmed false evidence, false expiry, lost originals,
wrong chronological state or silent measurement errors cannot be waived as
“low benefit.” A hypothesis failing to improve can retain the measured best
baseline; do not claim that component has been fixed or qualified.

**Tuning exit:** live knob-consumer checks, matched-budget curves, source/answer
validation, resource profile and interaction checks complete; final manifests
frozen. The small source-support/negative/style panels remain mandatory even
if aggregate quality improves. Broad product coverage is not added to this gate.

**Z2 readiness:** each condition's real enabled/disabled paths differ as intended;
no unintended shared writer state, future evidence or observer side effect;
truth labels exist for every admitted claim; the exact arm/question/call table
and cost estimate are available; short full-cycle resume/control tests pass.

**USER-REQUIRED:** after implementation, inspect the short readiness summary,
provide the chosen cloud credentials locally if missing, and manually launch
the documented command in tmux. The agent does not start the long experiments.
Duration is estimated from the verified short path and frozen arm count, not
the historical four-night guess. Done means every planned unit has a durable
result/error/exclusion and final reports bind those identities. Review disputed
final semantic cases in prepared batches; no approval is needed for routine
independent repair work already authorized.

## 6. Look-ahead and roadmap closure

Existing ownership, with concrete acceptance examples:

| Existing item | This plan supplies / what still earns completion |
|---|---|
| G40 — task labels, e.g. synthesis needs several facts | P0/P8 reviewed task/required-fact labels; current 255 dispositions are complete, older taxonomy consumers still need audit. |
| G55 — complete meaningful scoring, e.g. a gold-ID hit loses its value | P0/P2 full semantic funnel; the 1471-turn seed is complete, source IDs alone remain insufficient. |
| G57 — write representation, e.g. a long source never compresses | P1/P3 actual supported substitution and loss audit. |
| G63 — extractor truth, e.g. a linked relation reverses its source | P3/P4 independent source/role/direction review. |
| G68 — input/output ceilings, e.g. a long turn produces incomplete JSON | P1/P3 length-stratified coverage and failure-rate repair. |
| G69 — relation canonicalization, e.g. “can” becomes “does” | P4 opposing modal/directional controls and real relation review. |
| G71 — old findings, e.g. an old extractor percentage reused as v3 | Map each remeasure claim to r4/new evidence or inapplicable; no borrowed scores. |
| G72 — fresh state and confidence tiers, e.g. low-tier facts excluded from direct reads | Fresh seed/snapshots done; P4 independent tier truth and P0 identity retention remain. |
| G73 — conversation folds, e.g. a later correction disappears | P1/P10 support, completeness and actual prompt-cost evaluation. |
| G74 — cluster names, e.g. a label unsupported by its actual members | P10 blind member-grounded specificity review. |
| G60 — supersession, e.g. two coexisting values wrongly treated as replacement | P5 eligible source-order/date controls and historical reads. |
| G67 — maintenance, e.g. every candidate is rejected before model inference | P5/P10 eligibility, fair scans, real decisions and semantic output review. |
| A9 — entity activation, e.g. a paraphrase fails to open graph search | P7 same-meaning preflight/source-arrival controls. |
| A12 — local model quality/cost, e.g. E4B and NuExtract residency | P3/P9 deployed-pair quality/latency/VRAM; old full model-comparison scope remains separate. |
| G4 — GPU budget, e.g. overlapping encoder and local writer peaks | P9 measured complete-stack profile, not sum of isolated model sizes. |
| G15 — noise gate, e.g. banter unnecessarily retrieves | P7 memory/no-memory opposing cases and token effects. |
| G27 — background defaults, e.g. unpinned fallback changes writer | P9 actual resolved identities and explicit default decision; no silent pin edit. |
| G28 — style invariance, e.g. punctuation changes source access | P7 source/gate/answer changes with intent held fixed. |
| G29 — duplicate consumers, e.g. tuning a setting leaves a literal active | Tune live consumers and audit the entry's named remaining copies; unrelated product work stays later. |
| G30 — reader quality coverage, e.g. smoke passes but a useful span is absent | P0/P2/P10 actual-model source-to-answer checks, with product-only readers deferred. |
| G49 — procedures, e.g. repeated supported instructions never activate | P6 and final procedural ablation; exercise learned/unlearned time consumers explicitly. |
| G51 — semantic links, e.g. related names incorrectly imply identity | P4 only source-proven links with answer benefit; no topology target. |
| G70 — write/read/use, e.g. exposure counted as corroboration | P0 per-representation semantic use, exposure and source support reported separately. |
| G75 — summary trust, e.g. invented or overlength summaries | P1 source-faithful, complete-enough compression at measured cost. |
| G65 — judge reliability, e.g. one answer changes grade by comparator | P8 independent absolute grades and human qualification; graph rubric separately qualified. |
| G66 — memory improves answers, e.g. full vs vector at the same budget | Tuning and both Z2 controls/answer comparisons; no source-presence surrogate. |
| G48 — leg utility, e.g. non-episodic evidence gets zero credit | P0 semantic credit plus full additive/subtractive Z2 conditions. |
| G76 — attributed claims, e.g. assistant speculation becomes user fact | P4 source linkage/role/precision and downstream answer review. |
| T5 — temporal-label effect, e.g. old-state ranking changes | Freeze baseline first, then the dedicated temporal-consumer comparison. |

The formal 29 dependency-sensitive items are not 29 mandatory reseeds and not
29 guaranteed closures. The first campaign completed their shared data
collection prerequisite and exposed repairs. ROADMAP checkmarks require each
original item's actual acceptance scope, not simply reaching this document's end.

## 7. Traps

- R4 is a development baseline, not a clean healthy-system estimate: local
  degraded work and ungraded cloud answers remain part of its result.
- A good source quote does not establish the proposed relation or cover every
  question qualifier. A good graph shape does not establish fact precision.
- Adding a decision model cannot fix a path that never admits a source pair.
- Full/no-Codex context comparison is not full graph deletion; warm vector is
  not independent bare vector; read-only probes are not ordinary remembered turns.
- Reusing 39 as-of snapshots saves read-side work. It cannot make a new writer
  or new decay history retroactively exist in an old store.
- One history, overlapping task labels, repeats and display orders do not
  multiply independent evidence. Correct-answer quality and provider failures
  use separate denominators; excluded/uncertain labels remain visible.
- Keep measured v3 results separate from the frozen v2 JMLR manuscript and
  from prior failed-instrument numbers. Repairing documentation changes no score.
