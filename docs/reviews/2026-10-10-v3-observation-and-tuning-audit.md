# ICE v3 — reader counts, complete faults, automation and tuning

This audits current code and the sealed development policy. It extends the
[post-reseed repair contract](../specs/V3_POST_RESEED_REPAIR.md#23i-reader-frequency-and-full-fault-receipts--grounded2026-10-10),
not the roadmap owner count. Core memory quality remains the goal. No production
default, running writer, original r4 artifact or final experiment is changed.

## What the reader counts now and what final integration still needs

`seed_v3.stage_record` captures generated legs, post-fusion/rerank candidates,
budgeted fragments, final selected fragments, source-note lineage, evictions,
block counts and exact final messages. The selected list already comes from
`memory_preparation.evidence_after_eviction`; do not rebuild that selection.
`report_v3_replay` aggregates fragment counts by leg, but has no per-item
frequency or exact unique derived-artifact denominator.

The new offline `report_reader_usage.py` validates the complete repaired producer
and optional complete matched coordinate before counting. Its per-question/arm
records and repeated counters include:

- Known stored original turns at the cutoff, with row/batch aliases coalesced.
- Generated, ranked, budgeted and final selected source-lineage hits separately.
- Selected fragments whose exact saved text is visible in final messages.
- Number of different originals credited in final evidence, and how many sampled
  questions each reaches. Multiple aliases/fragments in one question count once.
- Zero-evidence-read originals with at least one observed question opportunity;
  originals introduced after the last question have no observed opportunity.
- Fragment-version frequencies and credited graph-edge lineage frequencies,
  kept separate from stable artifact IDs. Each arm has its own counters.
- Unknown aliases, unconfirmed selected text, unknown eligible inventories and
  unmeasured answer use, rather than guessed positive scores.

**This is a partial source-exposure instrument, not the complete final counter.**
Source batch IDs cannot identify an exact procedure, claim, chunk, slot or note
version. A graph fragment's edge list is lineage for aggregated text; it cannot
prove every edge's fact was physically stated. A visible source can omit a
required value or appear in negated text. Exposure does not prove answer use.

Before semi-LSREP/ablation launch, final capture must carry stable artifact and
version IDs across stored-at-cutoff → eligible → candidate → selected → final
rendered blocks. Include recent history, slots, bookmarks, conversation/batch/
entity notes, procedures, claims and chunks; total writes are distinct from live
rows and source turns. Bind final block receipts after eviction to exact final
messages, never credit a selected candidate that was dropped. Report unique
items, versions, exposure count, exposure/opportunity distribution and token
cost per representation/arm/time cohort. Eligibility needs the actual scope/
privacy/activation predicate; all stored items are not necessarily eligible.
Preserve unexercised branches and dependency between repeated questions.

Oracle inventory is supplied evidence, not a learned-store writer denominator.
Observe what its adapter passes, without inventing sessions/lifecycle effects.
Keep this instrumentation read-only and prevent exposure from becoming factual
corroboration. Required-answer support and qualified full/control answer deltas
remain separate measurements.

## Error receipts and continuation

`WorkerRecovery.call` currently records type/status/context but truncates the
message to500 characters and omits exception chains/tracebacks. Cloud outputs
already retain bounded text-free provider receipts and durable attempt/cache
state; those are useful but are not a full exception journal. Several retrieval
and per-item maintenance handlers catch errors internally and log only `str(exc)`.
The outer dispatcher cannot recover a stack that was already swallowed.

New `fault_receipts.FaultRecorder` is a private observer seam. It stores complete
sanitized messages, exception/cause/group chains, stack, wall time, available
status/request/provider receipt and declared turn/job/model/profile/window/
attempt/artifact metadata. It never dumps arbitrary locals, requests or headers.
Files are private, append/fsync protected and hash-chained; changed chains or
interrupted tails refuse further append with earlier evidence preserved.
Credential values, bearer tokens and URL credentials are redacted. Known safe
provider usage counters remain, generated text/headers do not.

Wrap an actual call **inside** the existing recovery boundary. The recorder
re-raises the same exception; existing bounded retries/degradation/pause behavior
decides what happens. A successful call creates no error. Two isolated failures
still leave originals and ungraded work; repeated outages, identity/storage/access
failures still pause. Failed error capture cannot pretend it succeeded.

Controlled tests exercise actual WorkerRecovery with two long chained failures,
per-attempt turn/job context, credential redaction, durable resume and a structural
fault that is recorded but still propagates. Final-run dispatcher/provider and
internal caught-error seams still need integration and duplicate receipt linking.
This helper is not retrofitted into the maintainer's running producer, and cannot
reconstruct old missing stack/message detail. Report capture completeness too.

The tuning-answer bridge now wraps its **actual SDK adapter call** with this
observer. Exact native session/probe, source clock/cutoff, coordinate/arm/model/
profile/window and content-hashed durable attempt reservation are checked before
the request. Its implementation is bound in the answer plan. Each arm has a
private `faults-ARM.jsonl`; complete-cache resume verifies the journal without
constructing an SDK client. Controlled interrupt/quota/two-exhausted-request
checks preserve full metadata/stacks and saved successful answers; altered
journal content refuses even a complete answer cache. This is the development
answer hook, not all final writer/judge/internal catches.
Credential redaction refreshes at record time because the native runner loads
its selected environment after the observer is constructed. An actual controlled
SDK error containing an unlabelled late-loaded key confirms the sidecar redacts
it; this does not reconstruct or rewrite older native error logs.

## Automation: keep automatic memory work; do not auto-approve everything

The current replay uses the existing ten core jobs on its source-time schedule.
The production runtime additionally gates dispatch on activity/inflight/lease/
resource state and has a session-end burst; replay is a serial research schedule,
not a measurement of interactive scheduling. Explicit aging/catch-up and both
clock provenances still need the final semi-LSREP protocol.

| Core job/action | Current execution and human gate | Fixed research decision |
|---|---|---|
| Per-turn post-flight/extraction/chunk handling | Automatic after storing the original reply; source-support/length/completion guards decide admission. | Run in full arm; record supported/withheld/error outputs and actual representations. |
| Cluster assignment | Automatic,30-minute cadence. | Run; review assignments/source-specific quality, not just cluster count. |
| Cluster merge | Automatic,3-hour cadence and similarity/structural conditions. | Run under current caps/guards; qualify actual merged members/specificity. |
| Conversation summaries | Automatic,2-hour cadence plus production session-end burst. | Run; observe eligibility, source proof, substitutions and detail loss. |
| Batch summaries | Automatic,2-hour cadence plus session-end burst. | Run; observe exact processed originals, eligible/withheld summaries and reuse. |
| Reflection session synthesis | Automatic for conversations with at least10 non-private turns;2-hour cadence plus session-end burst. | Run; session pattern observations are not procedural reinforcement. |
| Reflection entity notes | Automatic but source/citation/NLI qualified; unknown/failure keeps previous state. | Run; report selected/attempted/support/unknown/error/written receipts. |
| Reflection slot evolution and motifs | Pending review proposals; no automatic slot replacement or motif cluster creation. | Keep pending without human intervention; count proposed/withheld/pending separately. |
| Reflection `pending_items` append | Direct automatic append if that slot exists; separate from review-gated slot evolution. | Qualify its source/freshness/repetition branch before a seeded slot profile. Do not describe all reflection slot writes as human-gated. |
| Maintenance exact entity merges / pending duplicate cleanup | Capped automatic deterministic operations. | Run with existing type/source guards and journals; measure actual changes. |
| Maintenance ordered reconciliation | Automatic only when source authority/complete decision permits; unsure remains unresolved. | Run; qualify false expiry/coexistence/updates before changing guards. |
| Maintenance fuzzy entity merge / stale-slot update / ambiguous conflict | Review proposals, not automatic graph/slot writes. | Keep pending; no mass approval to make the job seem useful. |
| Episodic, graph and procedural decay | Automatic,90-minute cadence; procedural retirement uses absolute staleness, not multiplicative cycles. | Run against declared lifecycle clock/targets; count moves/retirements/remaining recall. |
| Compaction | Automatic,24-hour cadence and event threshold. | Run; distinguish eligible snapshots from executed no-op jobs. |
| Classifier fine-tuning | Event-only and `auto_finetune=False`; normally a review proposal when enough curated labels exist. | Freeze/off for comparisons; no mid-experiment model changes. |
| Project polling/decision extraction/import/document product jobs | Different project/ingestion eligibility, outside the ten conversation jobs. | Outside this research fixture; disclose rather than count as exercised. |

**Decision:** keep the existing automatic core path, preserve source guards and
pending-review semantics. Human work is needed for label/judge qualification and
any explicit review-enabled condition, not to babysit each background job. The
main research arm receives no hidden human corrections between checkpoints.
If a later assisted condition is desired, freeze its review budget/policy and
report it separately. Additive/leave-one-out arms must verify the named mechanism
actually changes execution/state/context; disabled versus ineligible differs.
Oracle evidence-only scoring does not credit conversational maintenance.

## Which exact parameters are in the current sealed grid

There are **30 coordinates = baseline +29 alternatives**, touching **13 Settings
fields plus one research hard input cap**. Equal-to-baseline trials are omitted;
weights change one leg at a time, not all combinations. Resolved values below
are from the sealed private development policy, not assumptions about defaults.

| Field(s) | Sealed baseline | Actual alternatives | Consumer / dependency |
|---|---|---|---|
| `context_budget_min`, `context_budget_max`, `context_budget_fallback` and separate `input_prompt_cap` | 4000/40000/23000; no research hard cap | Set the three budget fields together to4000,8000,16000,32000; also enforce that hard input cap. | Memory budget setter **and final prompt ledger**. Infeasible inputs refuse; output reserve remains. |
| `context_recent_fraction_default` |0.15 |0.1125,0.1875 | Shared recent-window allocation. |
| `context_recent_fraction_ladder` |10→0.30,50→0.20,200→0.20,500→0.15 | Scale the entire ladder by0.75 or1.25 within existing min/max. | Recent-history/memory split, tested at real callers. |
| `retrieval_bm25_candidate_limit` |100 |50,200 | Lexical candidate query. |
| `retrieval_vector_candidate_limit` |100 |50,200 | Warm vector candidate query. |
| `retrieval_chunk_candidate_limit` |30 |15,60 | Relevant turn/document chunk candidate query. |
| `codex_claim_candidate_limit` |64 |32,128 | Sentence-backed graph claim candidate query. |
| `retrieval_rerank_candidates` |64 |32,128 | Existing reranker shortlist; no new resident model. |
| `retrieval_rrf_k` |60 |30,90 | Reciprocal-rank fusion. |
| `retrieval_leg_base_weights` |BM25 .8,vector1,Codex .5,procedure .2,timeline .6 | Each of five weights independently×0.5 or×1.5; ten alternatives. | Shared topic/intent-adjusted leg weighting; changing a base does not prove useful final selection. |
| `retrieval_rerank_enabled` |True |False | Mechanism toggle, with existing fallback-fusion behavior retained. |

Counts by stage: budget4, recent allocation2, candidate limits8, reranker
shortlist2, fusion2, leg weights10, reranker toggle1, baseline1.
Cheap matched contexts reveal actual prompt/gate/source deltas first. Only
baseline and at most two source-qualified survivors per stage receive Luna
answers. Independent qualified support/answer/token comparisons are still
required before selecting a winner. Planned capped settings are not results.

## Subsequent parameters and explicit exclusions

These **30 named conditional fields** await qualified stages. Their
current/default values and actual static read sites are in the private exported
ledger; changing a source-sensitive threshold without its negative controls
would maximize activity rather than correctness.

- Relevance admission,1 field: `retrieval_rerank_min_score`. Existing ordering
  does not reject scored distractors while the floor isNone. Five authored native
  before/older/corrected/historical/quiet pairs preserve required original content
  and save up to2619 prompt tokens with diagnostic -5; a passive zero-floor
  counterexample loses required historical content. Require content/noise/rejection
  opponents and supported-answer qualification before promotion; no trial range
  or default change is implied. Revision3 binds this dependency and refuses old
  seals lacking it. The stage is read-only, not another writer replay.

- Trust/activation,7 fields: `codex_direct_trust_floor`,
  `codex_deep_strength_floor`, `codex_entity_match_threshold`,
  `codex_relation_fit_weight`, `procedural_min_conf`,
  `procedural_min_cited_turns`, `turn_summary_coverage_threshold`.
  Require independently reviewed source/truth/false-admission and content-loss
  panels. The last two also change writer state/proofs and require fresh replay.
- Gates/style,6 fields: `ltm_decision_threshold`, `ltm_prior_bias`,
  `ltm_length_weight`, `ltm_pressure_midpoint_tokens`,
  `ltm_pressure_scale_tokens`, `confidence_fallback_threshold`.
  Require memory-needed versus current-supplied/banter opponents and meaning-
  preserving style variants. Correctness and unnecessary token cost both matter.
- Lifecycle,16 fields: `decay_daily_unaccessed`, `decay_daily_accessed`,
  `decay_daily_creative`, `decay_archive_threshold`, `decay_cold_threshold`,
  `decay_creative_floor`, `decay_strengthen_amount`, `codex_decay_daily`,
  `codex_retention_floor`, `codex_retention_increment`, `codex_retention_cap`,
  `codex_retention_rank_weight`,
  `procedural_stale_days`, `procedural_min_reinforcement`,
  `cluster_similarity_threshold`, `cluster_merge_similarity_threshold`.
  Require declared age targets/schedule and source/correction/false-expiry/member
  quality panels. Use separate chronological writer/read histories; read-only
  overrides cannot retroactively measure different writes, merges or decay.
  Graph rank weight alone is read-only, but exposure/cap coupling needs those
  histories. Four fields were omitted from the first conditional list and are
  added in ledger revision2; old exports remain immutable.

The complete **173-field static inventory** now exports an explicit disposition
for every field:13 planned read-grid,30 pending qualified stage,129 held current
policy and1 retired procedural cosine threshold. The count is an inventory,
not173 executed trials. Hard NLI/source ordering/privacy/completion guards stay
fixed; admission safety is not tuned against the final answer score. Other sizing,
allocation and heuristic candidates stay held until an observed bottleneck merits
a new grounded coordinate. Static missing-read fields remain unresolved, not
automatically deleted. Model identities remain fixed, without a tournament.

Earlier revision2 exports retain29 pending/130 held. They are historical sealed
artifacts, not missing-field defects to overwrite. The original floor existed
there under held policy; the new disposition follows measured context waste,
not a rebuilt feature. Actual native/preserved-state evidence and its limits are
in [the v3 window/relevance provenance](../PROVENANCE.md#2026-10-12--v3-chronological-windowcorrection-and-relevance-admission).

Sixteen additional explicit fixed-profile controls cover `maintenance_intervals`,
`runtime_cycles_cap`, `auto_finetune`, the eleven named `agent_*` caps/detector ages/
similarities and `reflection_enrich_limit`/`reflection_enrich_refresh_days`.
They are absent from the original numeric prefix inventory but have real
consumers. Export them separately; neither absence nor fixed status means inert.
Current caps are scan50/model decisions25/proposals5/applications10/attempts2,
entity enrichment25 with14-day refresh, runtime catch-up cap96. Actual resolved
values are bound to the seal and refuse drift.

Before freeze: at most12 interaction checks on three demonstrated load-bearing
knobs, then reserved validation. B/C and official oracle labels stay out of
configuration selection. A tuned policy needs live deltas, source/support/answer
quality and cost plus its held assumptions; no parent roadmap closure follows
from this exported ledger. These29 fields are knobs, not the29 roadmap owners.

## Retention amounts and limits before tuning

Daily multipliers are not per-dispatch percentages:0.95/day means5% daily loss
while eligible, derived into16 cycles/day at the current90-minute cadence.
From score1 with no exposure, unaccessed half-life is13.51 eligible days,
accessed34.31 and creative68.97. The seven-day strict eligibility boundary is
additional. Archive/cold thresholds0.1/0.05 and creative floor0.3 have separate
effects; current creative aging never crosses them. These are analytical
horizons, not empirically chosen memory-quality targets.

Current native catch-up is bounded to96 cycles, six days, and successful replay
dispatch advances its ledger to the endpoint. A30-day gap therefore omits384
numeric cycles per worker. The new `lifecycle_plan.py` emits1440 chronological
events for that declared interval across the three decay jobs, with safe current
policy/consumer hashes and explicit original versus simulated clock provenance.
It runs no jobs and cannot infer trustworthy source calendars. The separate
chronological adapter now executes all ten native jobs with checkpointed time
groups; full final source-event/writer/read/probe parent integration remains.

Actual native ten-day controls distinguish the policies:

| Source/graph control | Current capped endpoint | Chronological cadence | Larger endpoint batch |
|---|---|---|---|
| Old untouched source, initially0.08 |0.058807; warm archived |0.049937; cold at first crossing |0.047899; cold at endpoint |
| Source one day old at start, initially1 |0.735092 |0.814506; four eligible days |0.598737; incorrectly aged for all ten days |
| Graph strength initially1 |0.941480 |0.904382 |0.904382 |

All source originals/IDs/clocks/receipts survive, graph confidence/validity stays
fixed, protected memory stays1 and creative floor stays0.3. Actual replay
dispatcher records96 cycles and updates all three last-run clocks to the
endpoint; the next interval sees only one due cycle. Procedural age is absolute
and must not be interpreted as missed numeric aging. These controls explain why
raising the runtime limit alone cannot qualify full decay: endpoint eligibility
and first cold transitions also matter. The cap remains fixed in the running
producer; final research scheduling must preserve chronological boundaries.

Sensitivity preparation now derives0.5x/2x half-life daily multipliers, exposure
increments0/0.5x/1x/2x, and graph caps0.5x/1x/2x current, not a blind search.
At unchanged graph strength1 and rank weight0.5, caps5/10/20 give bonus factors
1.1/1.05/1.025. More cap can mean less ranking influence. Episodic score ceiling1
remains normalized; zero increment still changes access count/rate class.
Actual deduplicated exposure/master-off/weak-source controls verify those paths.
Choose changes only after source-qualified chronological recall, correction,
false-retirement and supported-answer/context comparisons. No value is promoted
from arithmetic or these synthetic SQL controls.

The new `chronological_maintenance.drain` is an ideal-cadence research execution
seam, not a deployment scheduling change. It merges per-job known anchors and
failed attempts with each actual interval, passes native signatures/one-cycle
arguments, and retains remainders. Do not use the old strictly-overdue dispatcher
at exact cadence equality. Code/settings/job/provenance, trace/state and full
table fingerprints must match a parent-owned restored native checkpoint before
work. Unknown/duplicate jobs, missing/future clocks, wrong factories and excessive
intervals refuse. Completed groups publish native store/trace/state checkpoints;
interrupted groups must restore first. Disabled jobs retain their last-run clock.

Actual twelve-hour SQL controls configure hourly fixture cadences and a private
original to keep seven semantic jobs ineligible. All ten functions run: cold transition, graph
retention and absolute-age procedural retirement change state, while seven other
semantic jobs have no eligible population. No model HTTP requests occur in those
controls. Mid-group interrupt after committed native SQL, actual dump restore,
and missing-only continuation match the uninterrupted table fingerprints.
Cached complete groups are byte-stable/no-call. Three transport failures save
the third group's state before native outage pause. Complete fault journals
retain the three attempts and derived pause separately; observer/storage failures
propagate without accepting them as degraded successful groups. Job-boundary
model identity is unknown (`model=None`); a job may involve multiple models, and
the parent must bind their actual identities/provider receipts rather than claim
one background model from the job's GPU lane. Internal swallowed faults and
semantic quality remain unqualified. Exact parent source/condition/model/freeze
and full replay/answer orchestration are still required before final launch.
