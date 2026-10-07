# The reseed — plan, and the clean break

**Written 2026-08-25. The dated execution claim below is historical.**

**v3 correction, 2026-09-29:** the old `seed_store.py` still selects 293
curated turns and skips per-turn preflight. The replacement `seed_v3.py` now
replays the agreed 1,471 turns in their original order with recorded replies,
and `snapshot.py` covers all current ORM tables. The complete run and answer
judgment have **not** happened. A small disposable replay verifies the shared
path; source labels, historical-clock limits, and judge calibration remain
before a trusted quality score. [The current harness audit](../reviews/2026-09-29-v3-reseed-harness-audit.md)
records those gates.
Local `gemma4:e4b` remains the general background pin, NuExtract3 the separate
Codex extractor, cloud `gpt-6-luna` the default *probe* answerer, and a cloud
answer judge needs current answer-pair calibration. The August tier/delete
decision and 1,000–1,500-turn sizing notes below are historical proposals,
not permission to delete v3 facts or call a partial seed complete.

**v3 execution contract, 2026-09-29:** replay the three full conversations
in source order with their original timestamps. Before each recorded reply is
stored, prepare that user's prompt through shared v3 preflight; then write the
existing reply with source-role spans and run the real post-flight chain. Log
candidate, selected and final-prompt evidence separately, including source
IDs, legs, token costs, gate decisions and failures. Periodic jobs need an
explicit recorded cadence and failure ledger, not a silent end-of-run catch-up.
The unified 618-probe catalog carries `split_turn` for the 444 typed probes,
although `typed_probes.json` does not. **The original typed cutoff equals the
latest gold turn on all 444 rows.** Those immediate prompts are diagnostics,
not a long-term-memory quality test: their gold can still be in the 40-turn
recent window. An optional delayed typed panel asks at original cutoff + the
live recent-window maximum; 378 fit the selected histories and 66 do not. The
default panel uses existing questions at later section checkpoints. Capture
each prompt before any later turn is inserted, then validate its expected
answer against source and intervening turns before cloud scoring. Thirty-two
non-typed curated rows have a split beyond the selected full conversation;
quarantine them until their cutoff is corrected. A final-store scorer is
end-of-history only, never a chronological score. Save a complete,
identity-checked snapshot including
claims, notes and all source links before tuning; keep answer generation and
cloud judgment separate from recorded-response reconstruction.

**Existing long-term checkpoint pool, recovered on 2026-09-29:** the unified
catalog also holds 93 mature and 81 curated questions for the selected three
histories. Their `split_turn` is a designed checkpoint, unlike the typed file's
latest-gold proxy. The older `derived_gt.json` supplies source-turn mappings for
a subset: 11 questions already have an in-corpus source at least 40 turns
before their checkpoint (six curated, five mature). They are scheduled at
their native cutoffs. The remaining
checkpoint questions lack reliable gold mapping or chronology and need source
labeling; do not silently convert an expected answer into a gold turn. The 11
derived mappings themselves still require source-support review.

**Source-first checkpoint expansion:** `generated_probes.json` has 592 older
questions with a known original gold turn and evidence quote. After remap to
the selected histories, exact quote verification, a corrected question-only
ambiguity screen, and placement at the first real checkpoint at least 40 turns
later, 113 are candidates. The ambiguity weights are fitted only on history
available at each checkpoint. Together with the 11 native mapped questions,
these 124 form the source-linked candidate panel. The default replay also
captures the 131 unlabeled native questions at their real section times,
for **255 frozen as-of prompts**. Those 131 have no gold-fragment credit or
answer score until a reviewed source packet supplies older necessary turns;
the frozen prompts can then be reused without a second full seed. The trace
pins the resolved recent-history window, and a native source packet
must match it before its reviewed labels can join the trace. The 124
source-linked candidates span 11 checkpoint times; the 1,119-turn conversation
has only 14 candidates, all at its final
checkpoint, so this panel cannot establish retention across its earlier
sections. The native catalog spans 39 in-history checkpoint times; 131
additional questions need source-turn review, and 32 curated cutoffs exceed
the selected history. The private source-review packet ranks possible turns
for navigation, not as labels. Catalog IDs repeat across files; the replay
uses unique stable IDs and preserves each old ID as metadata.

**Implementation checkpoint (not a scored run):** `scripts/z1/seed_v3.py`
implements the sequential replay and captures full, vector-only and
recent-history prompt arms at each selected **long-term checkpoint** cutoff;
`report_v3_replay.py` reports
fragment-stage gold coverage, provenance and failure stages. The new snapshot
covers all ORM tables. A 3-turn disposable replay without periodic jobs passed;
one-turn disposable replay with the earlier six due periodic jobs and matched
arms also passed. The 2026-10-01 instrument expands this to ten registered
memory jobs and adds historical writer clocks, read-only probe isolation and
seed-to-snapshot fingerprints; see the contracts below and current provenance
for their path checks. `build_longterm_label_review.py` now creates a private
gold-plus-intervening-turn packet balanced by conversation and checkpoint;
its current 30-candidate sample has no verdicts yet.
`build_checkpoint_source_review.py` prepares the 131 unlabeled native
questions for source-turn review. Its reviewed `valid` rows may supply gold
to the frozen prompts through `answer_as_of.py --validated-native-sources`;
source turns must all be outside recent history. `answer_as_of.py` can plan
cloud Luna answers from frozen prompts and requires reviewed rows before a
full cloud campaign. No full seed, new cloud answers, blind judge or quality
claim has been made. The old
`seed_store.py`/final-store answer path cannot substitute for chronological
as-of scoring.

**Rank, evidence and answer contract:** each source-linked probe has an
expected answer requiring review and original gold turn(s). The v3 trace
measures whether a gold source ID appeared in a generated leg, its first
position among ordered post-rerank/pre-budget candidate **fragments**
(rank@5 and rank@10), plus its rank after collapsing repeated fragments from
the same source turn, whether it survived retrieval budgeting, and whether
it reached the final prompt as a selected fragment or a complete original
in a source-mode conversation note. Several fragments from one turn occupy
several fragment-rank slots. Distinct-source-turn rank collapses those repeats;
multi-source fragments give their sources a tied position and fragments
without source IDs have no turn rank. A source hit
does not prove that a compressed fragment retains the answer-bearing words.
The frozen full, no-Codex-evidence, vector-only and recent-only prompts go to one cloud answering
profile; vector-only excludes non-vector legs, slots, bookmarks and
conversation notes. `no_codex` suppresses graph, claim and graph-timeline
fragments but keeps the other legs and standing context. Its query expansion
and writer-created state remain the full system's, so it measures the direct
contribution of Codex evidence, not removal of the entire graph subsystem.
`vector_only` is the ICE warm-vector-leg control with the shared gate,
reranker and representation policy; it is not a standalone vector-memory
baseline and excludes the time-gated cold leg. The later vector-baseline
comparison must keep all original sources available and specify its own
selection policy. Each checkpoint freezes warm/cold/archive locations of
original sources, including unlabeled native questions, so later source
review can distinguish absent indexing/eligibility from poor ranking.
**Control-integrity correction, v3 2026-10-06:** disabling retrieval legs is
insufficient: the temporal empty-window helper directly queries nearest source
eras and emits a memory fragment; the low-confidence wide-net branch has its
own SQL and labels its fragments `fallback`. For `recent_only`, retain the
shared budget setter and recent-history assembly but bypass the entire search,
including temporal metadata and wide-net helpers. Record final retrieval as
disabled, keep the original B2 prior, and keep the zero-fragment assertion.
For `vector_only`, use the normal warm-vector leg even under low confidence,
suppress the empty-window/nearest-era note, and keep the vector-only assertion.
Full and no-Codex arms retain their production temporal and fallback behavior.
These are isolated control corrections, not production writer changes.
The blind paired judge also grades **each** answer
against the reviewed expected answer and complete original source as correct,
partial, incorrect or uncertain. Supply the historical question time and each
gold turn's recorded timestamp to the judge, so relative-date answers are
evaluated at the checkpoint rather than the date the cloud call runs.
Estimated prompt tokens are reported
separately from rank. The judge rubric still requires calibration on current
human-reviewed answer pairs before its grades become results of record.

**Answer-judge calibration contract (v3, 2026-10-01):**
**Judge-profile correction, v3 2026-10-06:** use the existing
`opencode-muse13` Responses profile for new manual campaigns and the current
r4 judge phase. The older DeepSeek pin was not a measured win over Muse:
Muse was selected for v2 LongMemEval and had stronger small human-label
calibration, which still does not qualify v3 paired judgments. Store an explicit
`judge_profile` in the private campaign bundle and supply `ICE_JUDGE_PROFILE`
only to its judge child. Do not edit global PROBE_MODEL/.env: all Settings are
hashed by the frozen seed. Existing bundles without a profile retain their
legacy explicit PROBE_MODEL route; never silently migrate existing judgments.
DeepSeek-v4.1 Flash is an explicit available alternative profile, never an
automatic fallback. Reuse the shared cloud TextGenerator with endpoint/status/
usage checks, stable session and ICE user-agent, no hidden SDK retries, the
unchanged full-source rubric and both orders. Bind model/profile/endpoint/
decoding/transport implementation to judge and calibration resume identity.
Preserve bounded caller retries and access/outage/error decisions. Independently
qualify v3 judgments before any score-of-record claim; endpoint controls are not
a judge-accuracy comparison. This changes no local writer or answering profile.

`calibrate_answer_judge.py` uses the same `judge_one` request and absolute
rubric as the answer campaign. Each complete case has a question/time, dated
source, expected answer, two anonymous answers and independently assigned
grades/preference with a review reason. Run both display orders, map grades
and winners back to the original answer identities, and report errors,
per-grade confusion, preference agreement and order consistency with explicit
denominators. Validate the entire packet before any cloud call; save progress
after each order. Bind outputs to packet SHA256, judge model, system rubric
and judge implementation SHA256. No source/answer truncation is allowed.
Preserve the original error reason; stop after three consecutive judge errors
and retain the incomplete artifact rather than exhausting calls on an outage.
The shared judge request must identify ICE honestly and send OpenCode Go's
required stable `x-opencode-session` header. Campaign calls use trace plus
conversation identity; both calibration orders use the same case identity.
Keep provider routing metadata outside the rubric and record the HTTP status
and provider error type without exposing credentials.

Keep authored controls separate from human-reviewed real answer pairs.
Authored cases cover exact/paraphrased answers, partial multi-source answers,
wrong values, both failures, equivalent correct answers, unsupported claims,
assistant suggestions, temporal updates and evidence near the end of a long
source. They expose a broken rubric, not human agreement on real conversations.
Neither packet kind automatically qualifies a judge or closes the roadmap;
current real-pair human agreement and swapped-order behavior must be inspected
before a score-of-record decision. Output remains diagnostic by default.
Distinguish an answer contradicted by the supplied source from a claim the
source cannot verify. The latter is uncertain, not known false. Explicit
source statements such as "I have not chosen" can contradict a claimed choice;
mere silence cannot. Two unverified claims do not earn `both_failed`.

**Campaign order handling, 2026-10-01:** the three real maintainer-reviewed
pairs had stable factual grades but one preference changed with display order.
V3 campaigns therefore judge each pair in both orders through the same request
and session identity. Preserve both raw verdicts and map them to arm identity.
If preferences disagree, report `UNCERTAIN / order_unstable`, not a winner or
an equivalent tie. If an arm's absolute grades disagree, report uncertain for
that arm and retain both grades. Either order's API/schema error keeps the
pair erroneous and the campaign incomplete. Save the first order before the
second call so an interruption does not erase work. Report order agreement
and uncertain counts. Absolute factual grades are primary; stylistic paired
preference is secondary and does not prove memory use. Legacy single-order
artifacts remain historical. No automatic score qualification follows from
the three-pair all-correct pilot.

**Completeness gate:** a campaign trace must contain one ordered preflight and
one completed write for every selected historical turn, and every scheduled
probe exactly once at its declared cutoff. The final completion flag alone
is insufficient. The answer loader and campaign snapshot use the same event
validator. Before any judge call, each arm must contain every declared probe
exactly once with matching labels, complete sources and answering model;
missing expected answers and malformed or contradictory judge verdicts are
errors, never equivalent ties or successful completion.

**Observer isolation:** historical turns keep the configured production
episodic/cold exposure behavior and record graph exposure after final prompt
selection, as chat does. Diagnostic questions and all matched controls run
inside a read-only PostgreSQL transaction with retrieval exposure writes
disabled, then roll back. They do not reinforce or resurrect memory and
cannot change the next historical turn's state. Freeze their prompts, not
their recorded historical answers; cloud answer generation happens later.

**Historical-clock instrument:** the isolated runner supplies the original
corpus timestamp to Python clocks in the memory path, including ORM defaults,
graph/procedural writes and periodic writers. Its dedicated SQLAlchemy engine
also binds explicit SQL `NOW()` reads to that timestamp. This is scoped to
each replay turn and restores real clocks afterward; network timeouts and
model runtimes retain real elapsed time. Verify the overrides with actual
SQL and ORM writes, then through recorded-turn post-flight before a full seed.
The replay drives the ten chat-memory periodic jobs from the production job
registry, cadence, overdue ordering and capped missed-cycle calculation:
cluster assignment/merge, conversation/batch notes, reflection, maintenance,
episodic/graph/procedural decay and graph-event compaction. Jobs run at source
turn boundaries, serially; idle-time asynchronous ordering, GPU deferral,
leases, retries and session-end bursts are not simulated. Trace per-job cycles,
results and before/after memory state. Cold moves preserve the original
turn's identity; cutoff state counts include both warm and cold originals.
This covers the memory writer callables, not concurrency or project/document
workflows. Validate both the schedule in isolation and actual jobs through
the replay before calling the campaign ready.

**Output inspection contract, 2026-10-03:** aggregate row counts and strength
sums cannot show a bad rewrite that leaves counts unchanged. Freeze each
turn's initial summary/abstract and original source hash after post-flight.
For each periodic job, record exact added/removed/changed semantic rows with
before/after text and provenance from explicitly listed observed tables.
Exclude vector bytes and repeated original turn bodies from this observer;
the corpus and complete store checkpoints preserve those. Capture clustering
membership, notes/manifests, graph/entity/procedural state, slots, session
summaries and review proposals; decay/compaction can use narrower relevant
tables. Emit the observed table list even when no row changes, and measure
observer time separately from job time using a real monotonic clock.
These are private inspection artifacts, not semantic correctness grades or
an exhaustive transaction log. Fail the replay on an observation error rather
than silently claiming full inspection. Verify unchanged-count text rewrites
and composite source keys in isolation, then actual SQL capture and a real
maintenance callable through the replay's job boundary. This instrumentation
must not add memory writes or change job eligibility/cadence.

**Timestamp-origin correction, 2026-10-01:** two selected histories were
prepared from text exports with a constructed five-minute clock. Preserve
their corpus dates/order, but write `synthetic_raw_import`, not `original`;
existing readers then caveat imported dates and do not treat them as original
calendar evidence. The third history's101 source pairs/times match the
provider export exactly and retain `original`. Pin this per-history provenance
in the run, every write and question/gold source; carry it into dated judge
evidence. Old traces lacking this declaration cannot enter a complete answer
campaign. These two synthetic schedules test the declared simulated cadence
and turn ordering, not authentic session gaps, calendar dates or elapsed-time
retention. Absolute calendar probes require original dates or explicit
date-bearing source text. Do not infer timestamp authenticity from timezone
awareness, monotonic order or faithful reproduction of the corpus file.

**Snapshot identity:** the seed's completion event records every table's row
fingerprint as well as counts. A campaign snapshot must match both, so a
changed graph fact with unchanged row counts cannot pass as the seed's state.

**Manual campaign and recovery contract, 2026-10-01:** the maintainer will run
the full campaign manually; development checks must not launch it. Provide one
entry point for status/plan/run, orchestrating replay, complete snapshot,
review-gated matched cloud answers, paired judges and aggregate reporting.
These remain distinct stages with saved artifacts. Missing reviewed labels or
judge qualification must be explicit, never silently bypassed to finish a run.

Replay needs durable recovery before that manual run. Checkpoint the complete
isolated store and trace prefix every ten completed turns by default and at
question checkpoints. Pin code, resolved settings, corpus/catalog/model inputs,
run arguments and isolated database. Publish the checkpoint pointer only after
snapshot and trace identity are durable; keep the previous recovery generation.
Resume verifies identity and the intact trace prefix, restores all ORM tables,
archives any unfinished trace tail, and rewinds to the committed prefix. Restore
the conversation IDs, source-ID maps, completion counts and maintenance cadence,
then rerun only the uncheckpointed tail. A failed/partial writer or job must not
be treated as a completed turn. Lock the run against simultaneous operators.
Retain a separate immutable store snapshot at every question checkpoint,
bound to that trace prefix. Later retrieval/weight/budget tuning must query
that as-of state, never the final store containing future turns. Hard-link
owned snapshot files so rolling recovery cleanup cannot delete the evaluation
snapshot or needlessly copy its bytes. Publish those snapshots atomically;
recovery repairs an interrupted publication from the committed generation.
Declare the bounded redo cost; this is checkpoint recovery, not exactly-once
execution of unfinished model calls. No normal user database may be restored.

Cloud answer resume must preserve the exact model/API profile, actual decoding
policy, selected probe IDs, label/trace hashes and exact frozen input receipt.
Only error-free answers count as done. Paired judge resume must pin input files,
model and rubric, retain each completed order, resume the missing order, and
preserve failed attempts. API/schema failures stop the stage with saved progress.
No automatic retry loop may consume quota while an outage persists. Reports
must distinguish partial completion, missing ground truth and unqualified
judgments from clean scores. Validate isolated interruption/recovery and the
actual shared preparation/transport path before documenting the manual command
as ready.

**Completed cloud-input repair, v3 2026-10-07:** a successful retry must clear
temporary request error/type/status/disposition metadata. Only an actually
failed unconfirmed second request can become a terminal degraded answer.
Completed answer resume validates all frozen receipts and leaves artifact bytes
unchanged; progress counts actual failed rows, never a stale disposition on a
success. Register the exact old/new answer-runner hash transition for read-only
completed outputs only. All other identity fields and adapter/recovery hashes
must match; old partial outputs or changed models/prompts remain refused.

The observed old resume bug added false `degraded_final` fields to successful
second-attempt answers, invalidating the judge's input-file hashes. A separate
private repair may remove only those false fields when reserialization exactly
reproduces BOTH original judge-pinned file hashes. Verify unchanged judge
code/rubric/provider/transport/recovery identity, hold all answer/judge locks,
archive original inputs and judge state with a fsynced receipt before writing,
and verify restored bytes. Do not edit judge identities or saved orders. A
changed answer/source or any other unmatched bytes must refuse before mutation.
Interrupted repair can be rerun with already-restored files. Validate this
through answer completion, interrupted judging, full answer-stage resume, and
judge continuation, including successful retries and true terminal errors.

**Terminal progress and failure display, 2026-10-03:** the manual entry point
shows its current stage and arm, processed/total turns, durable checkpoint turns,
completed checkpoint probes, saved successful answers, and saved judge orders.
Count replay turns only after their ending clock event (post-flight, maintenance
and probes finished), not when a turn was merely written. Distinguish processed
from checkpointed progress. On resume, read actual existing artifacts and reset
counts to a restored trace rather than accumulate duplicate turn events.
For judging, one successful persisted first order counts even if its second
order failed; failed orders do not count. Snapshot/report use an explicitly
indeterminate display. Show elapsed time and active work with a terminal bar,
plain changed-state updates when redirected, and a log path on stage failure.
The display is a parent-process read-only observer: it must not alter prompts,
decoding, source labels, run identity, child arguments or persistence. Do not
add a new model or dependency. Error/interruption stops the campaign and retains
stage state. A corrected writer cannot resume a differently pinned campaign;
preserve it and prepare a fresh, validated bundle instead of editing manifests.

**v3 later-stage fault audit, 2026-10-03:** a nonempty provider reply is not
necessarily complete. The cloud answer adapter must reject an explicit
chat-completion finish reason other than `stop`, or Responses status other
than `completed`, before treating text as a successful saved answer. Likewise,
the judge must reject an explicit non-`stop` finish reason even if its prefix
contains parseable verdict JSON. Persist these as failures via the existing
answer/judge paths, retaining successful earlier calls/orders for resume. Do
not automatically retry quota or turn truncated replies into partial grades.
Compatibility responses without completion metadata remain unconfirmed; record
the available answer completion marker instead of inventing one. No generation
limit, model, prompt or decoding change is implied by this check.

**Bounded campaign recovery, 2026-10-03:** live v3 workers already retry failed
jobs; the serial seeder bypasses that scheduler and initially stopped on the
first exception. The coordinator must read structured persisted failure data
and retry only known local extraction/completion, connection/timeout and
temporary provider/server failures (including urllib judge connection errors),
at most two additional attempts with
2/8-second backoff. Seed retry uses its verified checkpoint restore, not a
second execution against a partially changed store. Answer and judge retry
reuse their persisted successful calls/orders. Cloud quota/rate-limit,
authentication, validation/identity, unsupported inputs, OOM and unknown bugs
pause with saved progress and an explicit reason; they are not silently skipped
or retried forever. Every attempt has a durable parent policy/attempt receipt;
the recovery classifier's source is pinned with replay code. All stages use
the same clean pause boundary; no later stage runs after an unresolved failure.
Do not count a failed writer/job as a completed turn or convert an API failure
into an answer/grade. Ctrl+C still returns130 and keeps recoverable artifacts.
The serial seeder has no live idle scheduler: release only its owned Ollama
models on normal exit, failure or Ctrl+C through the existing guarded release
manager. A cleanup error is warned and must not replace the underlying failure.

The writer/source-alignment and residency changes alter pinned code. Preserve
the failed r2 bundle and its51-turn recovery snapshot; prepare a fresh reviewed
campaign for the corrected writer, without editing old manifests or launching
the full run. Earlier controls do not establish exhaustive robustness. Add
failure-injection through actual child execution and actual source-quote writer
checks, plus provider duration observations; reports retain the limitations.

**Operator retry cap, 2026-10-04:** supersede the two-additional-retry policy
above with TWO total attempts (one initial attempt plus one retry after2s).
The maintainer observed the same turn80 extraction failure twice; a third
prefix replay wasted time and was stopped gracefully for diagnosis. Persist
the two-attempt policy and derive displayed attempt totals from it. A second
failure pauses for investigation; the coordinator does not edit production
code or silently keep retrying. Source-proposal errors are repaired under
V3_REPAIR's isolation contract, not by weakening checkpoint identity. Changed
writer/policy requires a fresh bundle; preserve r3's70-turn checkpoint and both
failure tails. Do not launch the full experiment during this repair.

The manual coordinator's new bundles checkpoint after every completed turn
(`checkpoint_every=1`), retaining only current/prior rolling generations plus
the independent question snapshots. This limits successful-turn replay to the
failed/in-flight turn, without marking a partial write complete. Snapshot
cost is separate from model latency and grows with store size; measure an actual
saved store, preserve fsync/identity checks, and never weaken rollback for speed.
Standalone seeder callers keep their explicit/default checkpoint setting.

> **⚑ EVERYTHING BEFORE THIS RESEED IS DEAD DATA (maintainer, 2026-08-25).**
> *"lets just call ALL from before as we have no data, we are restarting ALL
> again."* Every store, every arm, every graph number in `PROVENANCE.md` before
> this date was produced by **`qwen3:4b-instruct` on the instruct prompt** — the
> configuration [G63](../ROADMAP.md#g63) measured at **15% correct with 30%
> reversed** and replaced. Nothing downstream depends on those arms any more.
> **Do not re-derive from them, do not compare against them, do not cite them
> except as method lessons.** [G71](../ROADMAP.md#g71) lists the specific
> findings that die with them.

---

## 1. The configuration being seeded

**v3 configuration correction, 2026-09-12:** template extraction is now the
source default. The August 27 handoff separates the general background model
from the extraction specialist; the reseed must preserve that separation.
The general background selection below is the decided run pin, not a claim
that `background_model_name` has a non-null source default.

| setting | value | why |
|---|---|---|
| `codex_extraction_mode` | **`template`** (source default) | specialist template path |
| general background model | **`gemma4:e4b`** (decided run pin) | summaries and other general background jobs; August 27 handoff |
| `codex_extraction_model` | **`hf.co/numind/NuExtract3-GGUF:Q8_0`** | dedicated extraction pin, e.g. template-filling facts rather than summarizing |
| `codex_extraction_ner_tier` | `background` (**NuNER**) | junk names 8.7% vs 19.5% |
| `codex_extraction_max_tokens` | `3000` | 1200 lost 30 of 60 turns |
| `codex_extraction_chunk_adaptive` | `True` | 550 split a 1,178-token turn into three |
| `codex_extraction_chunk_max` | `4096` | ⚠ ceiling unresolved — [G68](../ROADMAP.md#g68) |
| relation vocabulary in prompt | **NEVER** | 22% vs 60% correct |

**Verified end-to-end 2026-08-25** on 5 turns through `extract_codex`: 94 edges,
76 entities, **zero expiries**, canonicalisation fired (`has`→`have`,
`became`→`becomes`), merge key fired (`manhattan 5 lb book` →
`manhattan 5lb book`), entity gate refused `you` 9 times. The write path handles
this model.

## 2. Size — seed 1,000–1,500 DENSE turns, not 180

The corpus holds **71,256 turns**. The current arms hold 180–293.

⚠ **The median turn is 24 tokens.** Turn count is a misleading unit: the
**14,665 turns at or above 100 tokens carry 91% of all content**. Seed from
that population, not from a flat sample, or most of the run is spent extracting
from "ok" and "yes".

**Why bigger matters, and it is not vanity:** [G66](../ROADMAP.md#g66) asks
whether memory improves the ANSWER. At 180 turns a probe often has no answer in
the store at all — memory would look useless for reasons that have nothing to do
with its quality. The store has to be big enough that the right answer is
*there* before "did retrieval find it" is a fair question.

**Cost at ~13 s/turn:** 300 ≈ 1 h · 1,000 ≈ 3.5 h · 3,000 ≈ 11 h. Overnight is
affordable; start with 1,000 and extend if the first hour looks right.

## 3. ⚑ INSTRUMENT THE REJECT-BUT-KEEP QUESTION — it must be VISIBLE in the output

**The maintainer's standing irritation, and it deserves a real answer rather
than another year of "keep both".**

**Today:** grounded `0.9` · ungrounded `0.7` · rejected `0.35`, and **all three
are stored**. In the 2026-08-25 write-path test **79% landed at 0.35.**

**The fact that decides it:** `extraction_confidence` was measured
**uninformative about truth** — two full-coverage runs disagreed on which
direction it leaned. **If grounding does not predict truth, deleting rejected
facts removes true and false ones at the same rate** — that is not a filter,
it is shrinking the graph at random.

⚠ **But that was measured on qwen's output and is provisional ([G71](../ROADMAP.md#g71)).**

**So the reseed answers it:**

- [ ] Record the tier on every edge (already stored) **and surface the split in
      the run's own output** — grounded / ungrounded / rejected counts per turn
      and in total, printed, not buried in a table nobody queries.
- [ ] After the reseed, judge a **tier-stratified** sample: equal facts drawn
      from grounded and from rejected, blind, arm hidden.
- [ ] **Then pick a side and delete the loser:**
  - grounded measurably truer ⇒ grounding earns its place; **reject means
    delete**, and the 0.35 tier goes.
  - no difference ⇒ **the tier is theatre**; collapse to one confidence, and
    strip the retrieval trust floor that keys on it.

⚠ **The trust floor is why this is not cosmetic** — retrieval gates on
`extraction_confidence` today, so a meaningless tier is actively steering what
gets surfaced.

## 4. The standing rule this reseed adopts

> **⚑ PICK A SIDE ON EVIDENCE, THEN DELETE THE LOSER.**
> *(maintainer, 2026-08-25 — "lets start hacking and slashing away and picking
> sides, like how we have for nu thing")*

Applies twice over: to what a **feature** does — stop building one that caters
to every case — and to what **we** do — stop keeping both paths alive because
neither has been measured.

**[G63](../ROADMAP.md#g63) is the model for it.** Two models, two prompt shapes,
a blind round, a decision, and the loser is gone rather than parked behind a
flag.

⚠ **The one correction, and it is not a hedge — it is the failure this rule must
avoid.** The old open-relation default, later diagnosed under
[G51](../ROADMAP.md#g51), picked supersession without a measurement and silently
retired real multi-valued facts. The retained [PROVENANCE](../PROVENANCE.md)
documents eight components with seven retired; a separate “667 true facts”
number was repeated without a retained measurement and is withdrawn. ⇒ **the
rule is "commit after measuring", never "commit instead of measuring".** A side picked from a preference is that failure wearing this
rule's clothes.

## 5. Order of work

1. **Seed** 1,000–1,500 dense turns on the §1 config. Snapshot it.
2. **Judge** a blind sample — the new baseline, and the first number of the
   post-qwen era.
3. **Judge tier-stratified** (§3) and settle reject-but-keep.
4. **[G66](../ROADMAP.md#g66)** — codex leg ON vs OFF, answers judged blind.
   The reason the store had to be this size.
5. **[G71](../ROADMAP.md#g71) — re-measure the last two weeks.** Scope widened
   2026-08-25 by the maintainer: **not just the graph findings, everything
   accepted between 2026-08-11 and 2026-08-24.** G71 carries the register — 16
   PROVENANCE entries triaged: **3 survive, 3 already dead, 10 to re-measure or
   re-record.**
   - ⚠ **The retrieval half (08-13 → 08-20) cannot be redone before step 1.** A
     retrieval number measured over a 15%-correct graph is measuring the graph,
     not retrieval. That is why the reseed comes first.
   - **Drop rather than redo where the new extractor makes a finding moot** —
     e.g. the direction rule ([G59](../ROADMAP.md#g59)) targeted a defect
     NuExtract3 does not produce. Re-measuring it out of completeness is waste.

## Native labels and final repeat timing — 2026-10-02

The full retention schedule belongs to final semi-LSREP. On 2026-10-02 the
maintainer explicitly authorized a SMALL development repeat pass plus a
before/after scoring comparison. It augments the existing 255-prompt panel;
it does not replace source adjudication or become the final semi-LSREP run.

Use one private, source-reviewed `development-repeat-review.json` packet, at
most four question families. Each family copies one native catalog question
exactly, at its original recent cutoff and ONE existing later section cutoff.
Both occurrences have their own stable IDs and full through-cutoff review.
The before source must still be inside the resolved recent window; the after
source must be outside it and recent context alone must not answer. Restrict
this small pass to unchanged facts: identical gold turns and reviewed expected
answer in both phases, explicitly affirmed after correction review. Require
reviewer, reason, knowledge scope and semantic tasks per phase. Reject
unreviewed, malformed, changed-answer or non-checkpoint pairs before any replay.
The recent occurrence is an explicitly labeled control, never an old-memory
admission. Neither expected answers nor review metadata enter answer prompts.

Pin the packet bytes in replay/recovery identity and require that same packet
for answering/resume. Both occurrences traverse the existing read-only observer
and four prompt arms, without storing diagnostic questions or answers. The
manual entry point automatically detects the reviewed packet before seeding;
adding/changing a packet after replay starts fails rather than mixing schedules.
Before/after reporting joins only declared family and phase, retaining missing
phases, uncertainty and errors. Report each arm's grades, correct-to-not-correct
and inverse transitions, ordinal grade movement, estimated prompt-token delta,
and selected source evidence delta. List family/cutoff/source age. These are
dependent development observations with intervening-content/time confounding,
not independent samples or proof of a component's causal effect. Keep recent
controls separate from old-memory totals. Persist partial/final/manual summaries
and verify the actual plan→freeze→answer→judge→report path plus interruption
resume without launching the full campaign.

For semi-LSREP, stable question-family identity must connect original and later
occurrences. Every later cutoff requires its own source/correction/recent-history
review; repeated exposure cannot reinforce the replayed store. Retention
observations are dependent, not extra independent questions. Its executable
schedule remains final-condition work under the existing roadmap entry.
Existing native checkpoint questions can already revisit the same fact. Source
review may assign `question_family_id` to meaning-equivalent occurrences; keep
their probe IDs, cutoffs and labels distinct. Carry that declaration into answer
and judge receipts and report occurrences per declared family. Unassigned
equivalence remains unknown; family counts are not a claim of independent
samples. Natural-repeat declarations alone do not change the replay schedule.

Catalog expected answers can contain overprecision or unsupported assistant
interpretations. Either source-review packet may supply a nonempty
`reviewed_expected_answer` with a reason and cited originals while retaining
immutable `expected_answer` as catalog provenance. Use the reviewed key for
judging and save both; never put either oracle key into the answering prompt.
This also records corrections on excluded recent controls for later reuse.
Separate required answer facts from optional context in that key: an answer
must not fail solely for omitting an unrelated descriptive detail. Record the
reviewer and review scope; a Codex source review is not an independent human
qualification of the answer judge.
Source/answer validity and long-term eligibility are different: an accurate
source mapping within recent history remains excluded from long-term scoring.

## Manual campaign and measurement admission — 2026-10-01

The entry point runs replay, snapshot, frozen answers, both-order judging and
reporting as distinct resumable stages. Default invocation is read-only status;
`--init` prepares private review packets without a database or API call. The
user runs the full campaign manually. Stage completion remains diagnostic and
never closes all roadmap entries or qualifies a judge automatically.
Status must distinguish unset verdicts from absent source/answer reviews.
A structural recent-window exclusion does not complete a source mapping or
answer-key review. Report explicit invalid/uncertain dispositions and recorded
source-review scope/reviewer separately; these declarations are not independent
verification. Keep uncertain delayed references out of answer scoring until the
question uniquely identifies its target and source support is established.
Before a combined manual run starts expensive replay, validate review packet
corpus/catalog digests, question/cutoff identities, native source bounds and
admission fields without requiring a seeded store. Repeat trace/label validation
before cloud calls. Mis-edited labels must fail before1471 writer turns, not
only at the answer stage.

Valid long-term labels must explicitly record review through the exact query
cutoff, whether recent history alone answers the question (must be false), a
knowledge scope (private history, assistant history, public knowledge or mixed),
and semantic task labels. Gold-turn age alone cannot establish this: a later
turn can repeat the same fact. Public questions may remain useful controls, but
their correct answers do not establish memory gain. Unreviewed task labels are
not inferred from source role, punctuation or word overlap.

**Reviewed outcome strata, 2026-10-02:** the paired judge must consume those
reviewed labels in its reports, not merely save them on each answer. Report
absolute grade counts, paired correct/incorrect outcomes, judge errors and
paired estimated prompt costs separately by knowledge scope and semantic task.
Keep public-knowledge controls separate from private-history results: a public
answer may be correct without any memory benefit. Show every allowed scope and
task, including zero-case cells, and an explicit unreviewed bucket for missing
labels. Task labels are multi-label and their groups overlap; do not sum them
into an independent sample count. Every row remains a question occurrence at
one cutoff, with declared question families reported within each group. Unknown
or error grades stay unresolved, not incorrect or a tie. Persist these summaries
with partial and final judge receipts and expose them through the manual report;
completed resume must reproduce them without additional judge calls.

Pin the actual answer adapter and provider identity. For Luna's Responses
profile, temperature is omitted by the adapter; report `provider_default`,
never a claim of deterministic temperature-zero sampling. Store the exact
frozen messages/hash and evidence receipt beside each answer. Prompt source
presence and answer correctness remain distinct from semantic evidence support
and causal memory use, which require their own review/controls.

The replay intentionally uses controlled per-prompt automatic routing, without
session model stickiness. Report the routed model, serving window, B2 probability
and classifier probabilities; derive warm-store pressure with the foreground's
shared helper. This exercises the memory pipeline for the declared routing
policy, not the entire HTTP/streaming endpoint. No foreground production change
is authorised by this harness clarification.

**Schema divergence, 2026-10-01:** ORM creation omits migration-owned indexes;
the historical fresh Alembic chain also fails while dropping a missing index.
Do not repair installation migrations inside this measurement task. The manual
campaign freezes a private schema-only template of the current production ORM
tables (constraints/defaults/indexes, no data), verifies every current table is
present, and installs it transactionally into its owned empty database. Preserve
the template/hash across interruptions. This measures the current memory schema;
it does not certify fresh installation or the migration chain. Pin local writer
Ollama manifest digests as well as their mutable tag names before replay/resume.

## 6. Open before step 1

### Isolated worker failures — 2026-10-04

**Outage classification correction, 2026-10-05:** r4 reached 1120 durable
originals, then paused because three scheduled conversation-summary calls hit
their output limit. A returned malformed/truncated output is evidence that the
model responded, not a transport outage. Continue recording these content
failures as degraded work. Increment outage streaks only for recognized
connection/timeouts/temporary server failures; a responsive content failure or
successful call resets the transport streak. Critical failures still pause.
Legacy checkpoint streaks may be reclassified from their bounded consecutive
failure receipts; never erase failed attempts or degradation history.

This changes only the instrument's pause decision, not any memory writer,
model, output budget, source, label, setting, database or plan. An explicit
instrument-continuation receipt may allow this exact compatibility boundary:
verify the previous code hash from its Git revision, verify all writer and
other pinned files unchanged, allow only `worker_recovery.py` and
`replay_checkpoint.py` with an exact old/new digest pair in the closed reviewed
`instrument_repairs.json` registry, and bind the new code hash, complete original identity,
committed prefix/hash and archived original manifest. Preserve the frozen
checkpoint with hard links and a separate copy of the committed trace prefix.
Do not edit an original manifest/hash or permit a general ignore-identity flag.
Restore still verifies the full store; append the instrument-change receipt
before later work. Resume under changed settings/models/inputs or any unapproved
code still refuses. Reports expose the boundary; this is a development
instrument correction, never a new clean quality score. Summary-budget and
extraction-output failures remain visible evidence for subsequent quality work.

This narrowly supersedes the fresh-bundle requirement above for this verified
instrument-only correction. Any writer/configuration/label change still needs
its own decided contract; this receipt cannot authorize one.

**Second control-only continuation, v3 2026-10-06:** preserve the1434-turn,
213-probe committed prefix. Earlier recent-only controls have no fragments;
the offending turn65 probe batch is unfinished and must be rebuilt. Register
the exact `seed_v3.py`/`replay_checkpoint.py` transition separately from the
first recovery-tool transition. Verify its baseline using the current snapshot's
Git provenance and code digest, rather than the original run header's older
revision. Keep all earlier receipts/archives; each boundary gets a distinct
receipt and independent prefix copy. Completed-trace validation must accept
only an ordered chain of exact registered code transitions, at complete-turn
boundaries with correct counts and identical non-code identities. Reject
duplicate/disconnected/reverted transitions, writer changes and prefix changes.
Do not relabel old controls, erase old faults or change memory writers. The
finished development report must expose both instrument boundaries.

The maintainer authorizes continuing after isolated recoverable model failures.
New manual bundles select `--worker-failure-policy continue`; standalone seed
defaults to `strict`. Keep both policies in run identity. In continue mode,
idempotent post-flight gets at most two local attempts (one two-second backoff),
then records a degraded turn and retains its already committed original and
any successful independent derivatives. Periodic jobs get one attempt per due
cadence: partial job commits are not safe to retry immediately. Persist a
separate last-attempt clock for failed jobs, without claiming successful cadence.

Only recognized model connection/server/completion/extraction failures qualify.
Authentication/quota, database, source/provenance, schema/identity, resource,
unknown errors and failed preflight/probes still cause a controlled durable
pause. Three successive degraded invocations of the same worker indicate an
outage: finish/checkpoint the current turn, then pause with a distinct reason.
A successful invocation resets that worker's streak; checkpoints retain streaks,
failed attempts, degradation and failed maintenance cadence across resume.

Record every attempt and the exhausted worker, exception class and original
source IDs. Never insert derivative completion keys or substitute fabricated
facts. Validate fault events against run policy and actual historical writes;
completion binds exact degraded/attempt counts. `complete_selected_corpus` means
all original turns and planned observers were processed, not all workers
succeeded. A separate memory-processing status remains `degraded` if any failed.
Keep degraded probes in denominators. Each probe/answer receipt carries prior
worker failures and affected gold-turn identities; this audit metadata must not
enter answer or judge prompts. The replay and campaign reports distinguish
clean/degraded processing and enumerate faults. This seed policy does not change
the production runtime's job retry ledger or silently enable cloud-error grades:
cloud stages also select `--failure-policy continue` in new bundles: each answer
or judge order gets at most two attempts. Isolated exhausted completion/format/
transport/server failures remain terminal ungraded ERROR rows, not ties or
incorrect answers, and the next question proceeds. Failed-answer pairs consume
no judge call; keep them in planned denominators as ungraded.
`processing_complete` means every question was attempted; `complete` requires
all answers/judgments to succeed. Resume reuses terminal errors only under the
pinned continue policy. Persist each call's failure before retry and retain
successful first-order receipts. Three successive exhausted questions pause as
an outage; quota/authentication/unknown failures pause immediately. Report the
planned, successful and error counts; standalone and legacy remain strict.

- **The judge.** muse-spark (73%) has been down three days. ox-alpha is 67% but
  **37 s/call**; deepseek-v4-flash 60%. For G66's coarser question — "is answer A
  better than B" — 60–67% may be enough; for triplet-level truth it is not.
  [G65](../ROADMAP.md#g65).
- **The chunk ceiling** — 4096 leaves 34% of content split. Sweep deferred to
  the next test round ([G68](../ROADMAP.md#g68)).
