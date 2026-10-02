# v3 reseed and combined-Z harness audit — 2026-09-29

This began as a **code-and-document audit**, not an answer result. It
supersedes the executable order in the August reseed notes. A new v3 runner,
reporter and complete snapshot have now passed small isolated checks, but no
complete seed or scored answer run exists yet. The table below describes the
historical scripts; `scripts/z1/README.md` points to the new path.

## Whole-process v3 scoring audit

**Current source-review checkpoint, 2026-10-02:**31 source/answer reviews:
23 native mappings (six admitted,17 recent controls) and eight linked candidates
(six admitted,two recent or updated exclusions). Pending:108 native plus116
linked reviews,224 total. Ten additional early-cutoff exclusions are structural,
not source reviews. Complete original history review reaches turn216 in one
conversation and64 in another; all31 reviewed mappings have through-cutoff
scope. Twelve admissions span two histories, with the new linked cases primarily
covering earlier assistant advice rather than private user facts. Task and third
conversation coverage remain incomplete. Corrected keys preserve proposals, unknown details and
source attribution; unseen attachment contents and assistant interpretations
are not independently verified facts.
Corrected keys preserve the frozen catalog version; optional context cannot
become a mandatory answer fact. One natural repeated fact now has a reviewed
question family across distinct cutoffs, without changing replay scheduling.
The full retention schedule remains semi-LSREP work. These are Codex reviews,
not independent human qualification or a complete measurement panel.

**Small development repeats, authorized 2026-10-02:** two reviewed
unchanged-fact families add four recent/old occurrences to the base255. One
manual entry point passes the pinned private packet through seed and answers.
Per-arm before/after accuracy, prompt-token and source-presence changes are
reported with missing/error phases and dependent-family limits. Recent controls
stay separate from old-memory totals. The full semi-LSREP schedule remains final.

**Reviewed outcome reporting, 2026-10-02:** the judge now uses reviewed
knowledge scope and semantic tasks to stratify grades, paired correctness and
estimated prompt cost, preserving zero-case/unreviewed cells and errors. An
authored actual-coordinator/stubbed-judge control demonstrates that4/5 pooled
correct can conceal0/1 private recall and higher private cost; it is not an
observed memory rate. Partial/final/manual receipts retain the strata, and
completed resume adds no calls. Focus33 and disposable smoke484 checks pass;
these are instrument controls, not source truth or broad judge qualification.

**Current manual/readiness follow-through, 2026-10-01:** the [manual guide and
metric/ground-truth matrix](2026-10-01-v3-manual-campaign.md) provide one
resumable entry point, exact answer-input receipts, stronger through-cutoff and
recent-only label gates, explicit controlled-routing coverage, a frozen
schema-only production memory template, and separate retained as-of stores for
tuning. These are instrument repairs. Source/answer labels, output-specific
graph/summary/update/procedure truth and broader judge qualification remain
unmeasured; the 29-item evidence map below is unchanged. No full campaign is
launched by these checks.

The default replay captures 255 questions at 39 real section checkpoints.
Every question has a catalog expected answer. Of these, 124 have candidate
gold source turns old enough to be outside the 40-turn recent window; 131
native section questions initially had **no confirmed gold turns** and cannot enter a
retrieval denominator or cloud answer score until reviewed. Source-linked
coverage spans only 11 checkpoint times; the 1,119-turn conversation has
source-linked questions only at its final section. Of 113 source-first
questions, 101 cite assistant text, eight user text and four both, so exact
quotation alone does not validate factual support.

The private label packets pair each expected answer with original user and
assistant source, source timestamp, gold-turn age and intervening turns. The
131-question native packet contains the complete selected histories. Its two
lexical top-five lists are **navigation suggestions**, not ICE retrieval or
gold labels. Review must establish source support, later corrections and
answerability at that section; invalid or uncertain questions stay outside
scored quality denominators. Answer files pin catalog/review digests and keep
the expected answer with complete gold source.

For a reviewed question, the chronological trace records the gate decision,
candidate counts by leg, first matching gold-source **fragment** rank after
fusion/reranking and before budgeting, fragment and distinct-source-turn
rank@5/@10, budget survival, and final
prompt survival. The latter has separate channels for selected retrieved
fragments and complete original source-mode conversation notes. Rank@5 is
among five fragments, not five distinct turns: several excerpts from one
turn can occupy several slots. The distinct-source-turn rank collapses those
repeats; sources in a multi-source fragment tie, and fragments without
source IDs have no turn rank. Both any-gold and all-gold probe counts use
explicit gold-turn denominators. These measurements locate retrieval loss;
they do not certify that answer-bearing words survived a compressed fragment
or were used by the answerer.

The frozen full, no-Codex-evidence, vector-only and recent-only prompts use one cloud answerer
after label review. Arms share a serving-window budget rule but can spend
different actual tokens, recorded per pair. Vector-only has no graph,
lexical, procedural, note, slot or bookmark channel; recent-only contains
no retrieved fragment. The no-Codex arm removes graph/claim/timeline evidence,
keeping other legs and standing context; it isolates direct graph evidence,
not all effects of graph writing or query expansion. `vector_only` is the
ICE warm-vector-leg control with its shared gate/reranker/representation
policy, not a standalone vector-memory baseline; it omits the time-gated cold
leg. The later independent vector baseline must preserve access to all
original sources. Checkpoints also freeze warm/cold/archive source locations,
including currently unlabeled native questions, so later review does not
use final-store state to explain earlier misses. The blind judge compares paired answers and grades
each against the reviewed expected answer and complete original source
(`correct`, `partial`, `incorrect`, `uncertain`). `both_failed`, judge errors
and the historical question/source timestamps remain explicit; relative-date
answers are judged at their recorded checkpoint.
Incomplete answers remain explicit; estimated prompt-token medians and
per-pair differences accompany answer grades. The new absolute rubric has
schema checks and the small calibration pilots below, but no broad current
real-pair qualification. No full-campaign answer-quality number is yet valid.
Judge outputs explicitly carry `qualification_pending` and are not scores of
record. Passing JSON schema checks is not judge calibration.

**2026-10-01 authored judge controls:** the new calibration command calls the
same campaign request in both orders and reports grade/preference confusion,
order agreement and errors. Actual calls first exposed missing provider
session metadata, then a rubric ambiguity between unsupported and
contradicted facts. With routing repaired and wording clarified, the same
twelve authored cases matched all reference grades and preferences in both
orders. These are known development controls, not independently reviewed real
pairs or a memory-quality result. Outputs remain `score_of_record=false`.

**2026-10-01 maintainer-reviewed real pairs:** all six saved Luna answers
were graded correct; preferences were B, tie, and tentatively B. The actual
judge matched 12/12 factual grades across both orders,3/6 preferences, and
2/3 order-consistent preferences, with zero errors. Pair 1 changed tie/B;
Pair 3's extra context was optional and its human preference was tentative.
This all-correct three-pair pilot cannot qualify partial/incorrect/uncertain
grades or establish general reliability. V3 campaigns now judge both orders,
save the first before the second call, retain raw verdicts, and report disputed
preferences as `UNCERTAIN / order_unstable`. Grade disagreement becomes
uncertain for the affected arm; either call's error keeps the pair incomplete.
Factual grades remain primary and preference secondary. An actual one-pair
campaign-path check retained correct/correct and an uncertain preference;
458 disposable smoke checks passed. These are instrument checks, not memory gains.

Five complete-gold source inspections also distinguish ambiguous general
advice, public facts and assistant reports about an artifact from user facts.
They are source-only notes, with no full intervening/recent-history review or
admitted labels. General-knowledge answerability must be recorded separately
from historical source support; a correct public answer cannot by itself
establish memory gain. Assistant text can be valid memory for a question about
what the assistant said, without becoming a user choice. These distinctions
agree with LongMemEval's separate assistant-recall, knowledge-update and
abstention tasks ([official benchmark](https://github.com/xiaowu0162/LongMemEval)).
Order swaps address a documented judge bias, but do not replace real human
agreement ([MT-Bench judge study](https://arxiv.org/abs/2306.05685)).

The complete seed trace records resolved settings, candidate/final stages,
job results and before/after state, failures and every table's row fingerprint.
The shared streaming validator checks each ordered preflight/write, unique
source identity, exact historical query time/cutoff and complete scheduled
probe manifest. A completion flag cannot hide a missing replay turn. A
campaign snapshot must match the seed's counts **and** fingerprints; restore
verifies every ORM table again. An unchanged row count cannot conceal a
modified graph fact. This checks stored state, not live HTTP streaming.

**2026-10-01 clock/maintenance repair:** the isolated replay substitutes source
time in memory-path Python clocks, ORM defaults and explicit SQL `NOW()` calls.
Real model/network elapsed timers remain unchanged. It drives all ten chat
memory periodic jobs from the production registry, cadence, overdue ordering
and missed-cycle cap, including the previously omitted three decay jobs and
compaction. Per-job before/after state includes warm/cold movement, retention,
active patterns, notes, cluster links and pending proposals. Original-turn
counts span both warm and cold storage. Actual ORM/SQL, stale-slot, decay/cold
and clock-restoration controls passed in a disposable store. **Clock-origin
correction:** two selected text-export histories use constructed five-minute
timestamps, not authentic recorded dates. Their rows and frozen source/query
metadata now carry `synthetic_raw_import`; prompts caveat unknown source time.
All 101 turns in the third history exactly match complete provider-export
user/assistant/time tuples and retain `original`. Synthetic schedules test
the declared replay cadence, not real calendar retention or timeline truth.
An actual synthetic-time replay and snapshot check passed with all 12 frozen
prompt arms showing the source-time caveat. Jobs run serially
at source-turn boundaries; asynchronous idle scheduling, GPU deferral,
leases/retries, session-end bursts and project/document workflows remain
outside this replay's coverage.

**2026-10-01 observer repair:** historical prompts retain production exposure
writes and record graph access after final selection. As-of questions and
their full/vector/recent arms disable exposure writes and use a read-only
PostgreSQL transaction, rolled back afterward. Otherwise the question could
reinforce a fact before the next historical turn, or the full arm could change
what its vector control sees. Actual SQL rejection and unchanged exposure
counts are tested; graph access changes retention, not confidence/support.

Before calling the judge, each arm must contain every declared probe exactly
once, and all labels/sources/models must match. V3 rejects missing expected
answers, invalid reason codes and inconsistent absolute/paired verdicts;
judge errors produce incomplete output and a failing exit code. The full
1,471-turn seed, source-label adjudication, full cloud answer run and broad
judge qualification still have not happened. These instrument checks close no quality item.

## Historical scripts audited before replacement

**Next gate after instrument validation:** adjudicate the existing private
30-candidate source/expected-answer packet, resolve native section source
labels with explicit unscored coverage, and calibrate the current answer-pair
rubric on independently reviewed pairs in both orders. Review must distinguish
assistant suggestions from user decisions, later corrections, public facts
that require no memory, insufficient evidence and complete multi-source
answers. Freeze reviewed probe IDs and the declared replay protocol before
the full 1,471-turn seed and matched cloud-answer campaign. Hyperparameter and
leg-ablation results belong to the combined development Z phase; only LME
oracle and semi-LSREP follow as the final campaigns. This order does not require
a broad LME-S run, nor does it promise that all 29 evidence-dependent entries
will pass without further repair.

| Stage | Current path | What it establishes |
|---|---|---|
| Corpus | `seed_store.py` calls `derive_retrieval_gt.load_conversations()` | The current selection is 293 turns from three curated checkpoints. `gt_feasibility.load_full_conversations()` can read the agreed 1,471-turn, three-conversation corpus, but the seeder does not call it. The full corpus has recorded timestamps. |
| Ingest | `seed_store.py` creates rows directly, classifies `user[:2000]` without the conversation ID, assigns synthetic sitting timestamps, and calls `evaluate_turn()` after each historical answer | Post-flight density/representation, chunking, Codex extraction, and procedural extraction run. The classifier's live context prefix, preflight memory decision/retrieval/prompt assembly, and writer-supplied source spans do not run through this path. Synthetic timestamps inherit the row's default `ts_provenance="original"`, although the full corpus has recorded timestamps. |
| Periodic work | Seeder calls cluster assignment/merge and batch summarisation once after ingest | Rolling conversation notes, reflection, decay/archive, and the maintenance agent are not exercised by this seed path. The corpus is ordinary chat, so document/project-specific paths also need separately chosen fixtures if they are in the claimed coverage. |
| Snapshot | `snapshot.py` dumps a hardcoded list of 14 store tables despite describing it as ORM-derived | Current v3 tables including `codex_claims`, `codex_claim_links`, `batch_notes`, and `conversation_notes` are absent. Restore can therefore erase or omit source-backed graph/summary state. Row counts over the selected tables do not prove a full restore. Do not tune from this snapshot until a save→restore identity check covers the tables each scored reader uses. |
| Retrieval | `score_typed.py` uses shared `production_parity.build()` and final `prepare()`; `score_retrieval.py` measures raw candidates | Final-evidence and candidate metrics are intentionally different. The typed metrics measure source presence/coverage and rank, not whether an excerpt actually supports the answer. Missing Codex anchors and temporal successor labels remain unscored for those metrics. |
| Answers | `answer_probes.py` assembles the final prompt, defaults to cloud `gpt-6-luna`, and can use local Ollama | It does not drive the HTTP streaming route or generate the historical response used to build the store. It routes a local-registry model for the budget/window, then sends that prompt directly to Luna. This is a useful answer probe, but not yet a cloud-serving production route or a matched-budget vector/recent-history comparison. |
| Query time | The 444 `typed_probes.json` rows have no query-turn or checkpoint field | They can be treated as end-of-history questions against a completed store, but cannot establish earlier-turn sequential quality. A replay question about turn 100 must read only state available before that question; semi-LSREP and any chronological development probes need explicit as-of/cutoff metadata and a leak check. |

**Correction found during implementation, 2026-09-29:** the *older typed file*
lacks a cutoff, but `unified_probes.json` already supplies `split_turn` for all
444 typed probes and remaps the duplicated `cca73c87` tail into the full
`bb558b5f` conversation. Join by the unique question and validate the
conversation/gold mapping before tracing. **A second check found all 444
`split_turn` values equal the latest gold turn.** Asking there leaves the gold
eligible for the 40-turn recent window, so these are immediate diagnostics,
not a long-term-memory test. The new runner defaults to first eligible delayed
cutoffs (`gold + 40`): 378 fit, 66 do not. Its expected answers must be checked
against the intervening turns before scoring. Do not present a final-store
query or an unreviewed delayed query as a valid chronological answer result. Among the
other 174 unified probes, 32 curated rows specify a split past the selected
conversation length; quarantine those rather than silently treating them as
end-of-history.

**Existing long-term probes found after the maintainer's prompt:** the same
catalog has 93 mature and 81 curated questions at actual section checkpoints
within the selected histories. The older `derived_gt.json` supplies source-turn
labels for only part of this pool. After checking conversation remaps, full
history bounds, nonfuture gold and a ≥40-turn age, **11 questions** qualify as
already source-mapped native long-term probes (five mature, six curated). Their
expected answers and model-derived gold still need source review. The other
checkpoint questions must not be silently credited to guessed source turns.
The v3 runner includes the 11 at their original checkpoints. A separate
existing source-first pool yields 113 more candidates after exact source-quote
verification, as-of question-only ambiguity screening and placement at
the first real checkpoint at least 40 turns after gold. The default panel is
124 unreviewed source-linked candidates, at 11 checkpoint times. The replay
also freezes the 131 unlabeled native questions at their original section
times, for 255 as-of prompts; they receive no gold or answer credit until a
source review joins complete older turns to the frozen trace. The 1,119-turn
history has only 14 source-linked questions, all at the final checkpoint. The
optional delayed typed panel adds 378 candidates (502 source-linked plus 131
unscored captured), all requiring 40 intervening turns of answer review.
Across the original native catalog, 39 in-history
checkpoint times hold 142 questions: 11 source-mapped and 131 requiring
source-turn review. Thirty-two other curated cutoffs exceed the selected
history. A private source packet makes the 131 reviewable; lexical turn ranks
are suggestions, never gold. Duplicate original catalog IDs were also found
(including one typed duplicate); the runner now derives unique stable IDs
from the full source row before tracing or pairing. No quality result follows
from this catalog work.
| Judging | `judge_answers.py` pairs probe identities, checks complete gold source and equal answer model, randomises A/B, and distinguishes `both_failed` | The configured cloud judge is `deepseek-v4-flash`. Its historical graph-truth calibration is not a calibration for this answer-pair rubric. There is no demonstrated human-agreement, order-swap, or second-family check for current v3 answer pairs. A blind verdict is diagnostic until that calibration passes. |

## Required order before trusting a number

1. **Repair the fixture and verify a dry run.** Load the complete agreed corpus
   exactly once, preserve ordered recorded timestamps and user/assistant source
   attribution, and replay each prompt through v3 preflight before storing its
   existing historical answer. Use local background workers for the post-flight
   state. No historical answer needs to be regenerated. A small diagnostic
   trace already demonstrates this ordering in `trace_v3_memory.py`; it is not
   yet the full seed runner. Fail the run on missing turns or failed jobs.
2. **Freeze one versioned store.** Run the relevant maintenance passes, inspect
   counts and sampled source→claim/summary chains, then save and restore the
   complete state into an isolated store. Compare table counts, source links,
   selected evidence, settings, code revision, corpus digest, and model pins
   before treating it as the common starting point for tuning arms. Keep
   retrieval write-strengthening off in scored probes or restore between arms.
3. **Validate labels before scoring.** Complete missing graph anchors and
   temporal pairs. Check expected answers against complete gold turns; an
   unanswerable or unsupported probe must not become a retrieval failure. The
   typed presence metrics stay per type; inspect answer support and failures
   separately. Pin each question to an answerable time; reject future-source
   leakage. Add no-memory, absent-fact, update, style-variant, and
   multi-source controls where the corpus supports them.
4. **Tune and inspect in one combined Z1/Z2 phase.** On development probes,
   compare full ICE, matched-budget vector, recent-history, and targeted leg
   ablations using the same frozen cloud answerer and complete final prompts.
   Record supported-answer quality **and prompt tokens**, both per probe and
   per class. Calibrate a cloud judge against human-reviewed answer pairs,
   include swapped A/B controls, retain `both_failed` and errors, and visually
   inspect source→store→selected evidence→prompt→answer. Sweep only parameters
   that can change these decisions, with a small interaction check. Freeze the
   configuration before end-stage data.
5. **End-stage conditions only:** LME oracle and semi-LSREP on a substantial
   recorded conversation. For each historical turn, prepare ICE first and
   store the already-recorded assistant reply second. Use cloud answering and
   calibrated cloud judging for separate probes; background work remains local.
   Do not turn the held-out oracle into a tuning loop or run full LME-S.

This audit does not reopen a general model search or change production memory
behavior. A failed seed or combined-Z case should reopen a specific repair with
its observed effect on supported answers per prompt token.

## Corpus split, ablations, and the decision rule

The **development seed is three complete conversations (1,471 ordered turns),**
not one conversation and not the current 293-turn checkpoint subset. Their
histories and lengths differ; probe strata must verify, rather than assume,
that they cover different distractor and writing-style cases. The later
semi-LSREP confirmation uses one substantial recorded conversation outside the
development tuning set. LME oracle is a separate end-stage diagnostic. Reusing
one development conversation for all decisions would make a clean result too
dependent on its writing style and facts.

Run *small, paired* ablations during the combined Z phase, after the seed and
instruments pass their gates: full ICE vs matched-budget vector-only, recent
history, no memory, and each leg removed in turn. Use the same questions,
answerer, source-time cutoffs, and prompt-token budget for each pair. Record
the actual selected evidence and answer support. A leg can earn a check only
when it either improves supported answers at comparable prompt cost or
establishes a distinct supported-answer class that the simpler arm misses.
Repeat only the consequential frozen contrast on semi-LSREP; do not tune on
that conversation. This makes ablation a diagnostic *now* and an independent
confirmation *later*.

For every outcome, distinguish **stored**, **retrieved candidate**, **selected
for the final prompt**, **cited or used in the answer**, and **supported by the
original source**. These are different events. Stratify by fact update,
temporal reference, procedural preference, graph relation, summary/long turn,
negative or absent fact, and style variant. Report both errors and token cost,
including a failure to retrieve, an unsupported selected sentence, a correct
source crowded out by budget, and a correct source ignored by the answerer.
Judge questions against complete original turns; generated summaries and graph
claims cannot be the gold source. Calibrate the answer-pair judge on a small
human-reviewed set with both arm orders and a second judge family where the
human label is disputed. Preserve ties, both-wrong, and judge failures rather
than forcing a winner. This is the acceptance design, not a claim that those
tests or the judge calibration have already run.

## Open-item evidence map

This maps the **29 unchecked, seed/tuning-dependent roadmap entries** on
2026-09-29. The stage is where the main decision is earned, *not* an automatic
checkmark. A failed observation leads to a specific repair; a clean result
still needs the item's original acceptance scope. The 35 later/product/research
items and the three Z gates are outside this count. In particular, G19's
whole product item is later, although its ablation switch must work in combined
Z. The roadmap remains the source of each item's full scope.

| Stage | Item and concrete unresolved question | Required observation / closure evidence |
|---|---|---|
| Seed (12) | **G40 — typed probe labels:** most old probes still say `ENTER_TYPE`. | Complete and audit graph, procedural, summary, temporal and negative labels before a stratified score; show label coverage and disagreements. |
| Seed | **G55 — typed scoring and full corpus:** the scorer must not call mere source presence graph or summary quality. | Full 1,471-turn fixture, source-complete labels, final-prompt per-type credit, separate semantic-support audits, and no missing-anchor denominator hidden. |
| Seed | **G57 — write representation:** long turns, summary retention and graph eligibility used to discard information. | Per-turn source→representation audit, length/compression and support judgments, extraction eligibility/counts on all turn types; inspect failures, not just averages. |
| Seed | **G63 — NuExtract3 graph writer:** the selected extractor replaced the old one, but the shipped combination needs a new-store outcome. | Sample source-attributed claims and graph relations on a NuExtract3 seed, with blind truth/role/negation judgments and truncation/failure counts. |
| Seed | **G68 — chunk and output ceilings:** a long turn may lose its tail or truncated JSON. | Length-stratified source→claim recall and truncation rate; targeted small same-source chunk/output-budget comparison, then verify writer path. |
| Seed | **G69 — relation vocabulary:** post-extraction canonicalisation may merge modal or comparative relations incorrectly. | Seeded relation-frequency and unsafe-merge sample, including `can` vs `does` and `higher` vs `good`; decide threshold or representation from judged errors. |
| Seed | **G71 — superseded graph findings:** old qwen-store conclusions may no longer hold with NuExtract3. | Recheck its named remeasure register on the new seed, mark each old conclusion confirmed, overturned or inapplicable; do not cite old percentages as v3. |
| Seed | **G72 — clean break:** the old store and its rejected-but-kept confidence population cannot be a baseline. | Fresh full-corpus seed with recorded order and source dates, explicit counts of accepted/rejected-but-kept facts, completeness and snapshot/restore identity. |
| Seed | **G73 — conversation fold:** a short note may fabricate or omit a later correction. | Blind support, contradiction and completeness judgment against full originals on real generated folds, plus compression cost and source fallback. |
| Seed | **G74 — cluster naming:** old aptitude test supplied arbitrary turns, not production cluster members. | Run the real membership/naming path on the seed; blind judge specificity and factual support against each cluster's actual members, with repeat-noise check. |
| Seed | **G60 — supersession semantics:** an unknown relation can retire a true earlier object. | Curate same-subject old/new source pairs; score coexist/supersede/contradict/unknown, surviving historical retrieval and false-retirement rate. |
| Seed | **G67 — maintenance decisions:** seeded graphs must actually pass decay/reflection/maintenance paths. | Drive due work on a copy, count eligible/proposed/applied actions, judge changed entity profiles and retained original support; compare before/after graph quality. |
| Combined Z (16) | **A9 — preflight entity activation:** a true earlier entity can fail to open Codex search. | Same-meaning query variants through actual preflight; count missed graph activation, final source arrival, latency and VRAM; repair only if the miss matters downstream. |
| Combined Z | **A12 — two local background models:** E4B and NuExtract3 may be too costly resident together. | Measure current-pair per-job quality/latency and same-machine residency under real idle/generation schedule. The original all-candidate profile remains open unless that larger scope is explicitly decided or completed; do not call pair profiling a full A12 checkmark. |
| Combined Z | **G4 — GPU budget:** a model load can OOM or delay live chat despite idle release. | Peak and steady VRAM, yield/release and cold-load timings for answer + local background combination, with explicit headroom. |
| Combined Z | **G15 — noise/banter routing:** `Null_Noise` is a topic, so the old intent rule is half dead. | Noise and useful-memory controls through B2; compare retrieval decisions, supported answers and saved prompt tokens before setting a negative bump. |
| Combined Z | **G27 — background default:** an unpinned shared client may use registry fallback instead of the routed answer model. | Record exact model used for every background job and compare pinned-vs-unpinned semantics; decide the default with quality and residency data. |
| Combined Z | **G28 — style invariance:** punctuation or phrasing can flip a memory decision. | Meaning-preserving variants through classify→gate→final prompt→answer; decision-flip, source-arrival and answer-change rates, not classifier accuracy alone. |
| Combined Z | **G29 — drifted copies:** duplicate policy paths can disagree even when one path works. | Audit the entry's remaining named copies, then assert identical decisions on same inputs through the exercised writer/reader routes. |
| Combined Z | **G30 — quality blind spots:** smoke connectivity does not prove useful retrieved context or correct answers. | Exercise actual models and readers on sampled full traces; inspect source→prompt→answer alongside unit controls and report what the suite cannot see. |
| Combined Z | **G49 — procedural activation / inert time fields:** one activated nonsense pattern once scored perfectly. | Seeded activation distribution plus query-specific relevance and final-prompt/answer gain; decide an actual consumer for `learned_at`/`unlearned_at` or stop crediting them. |
| Combined Z | **G51 — containment links:** a name pair like `emotional validation`/`validation` must not authorise a false edge. | Only reopen source-backed candidate writer if opposing-source controls pass and it gains supported answers per token beyond searchable original claims. |
| Combined Z | **G70 — memory read/use:** 560 candidate claims do not show that answers used them. | Per representation, count writes→candidates→selected prompt→answer citations/support and token cost on real sequential replay; distinguish exposure from corroboration. |
| Combined Z | **G75 — summary trust:** an invented compressed note can replace a complete original. | Measure actual substitution in final prompts, blind source support and supported-answer effect at equal budget; decide retain/fallback/delete gate from that result. |
| Combined Z | **G65 — judge selection:** graph-truth judgment is unstable and its 15-label calibration does not validate answer pairs. | Recheck graph/source judgments against the existing human labels; separately calibrate answer-pair judgments on human-labeled pairs with swapped order and second-family disagreement audit before using either as a score of record. |
| Combined Z | **G66 — better facts → better answers:** truer claims may never change replies. | Paired full/no-Codex/vector/recent answer probes at equal budget and one cloud answerer; judge support and inspect graph evidence actually used. |
| Combined Z | **G48 — leg credit and tuning:** an episodic-only gold ID can make every other leg appear useless. | Source-turn credit for every leg at final-prompt rank, then matched-budget leg ablations and per-class answer effects; do not tune weights from presence alone. |
| Combined Z | **G76 — attributed claims:** a supported source sentence can still yield an unsupported relation. | Blind edge precision, original-source linkage and source-scope tests on seed plus selected-context and answer gain; report false assertions separately from recall. |
| Post-Z (1) | **T5 — temporal label consumer:** `Temporal_Recall` mostly duplicates `Needs_Memory` at B2 yet could improve time-window ranking. | After frozen Z baseline, test the label in Track T's gate and recency-flattening with dated and undated temporal prompts; compare old/new fact answers and false time windows. |

The twelve seed-stage rows are **questions the reseed can expose and help
decide**, not twelve guaranteed closures. The sixteen combined-Z rows need
selection, quality, cost or answer evidence; a seed count alone cannot close
them. T5 explicitly follows the frozen Z baseline. The three Z gates themselves
require their own completion records.
