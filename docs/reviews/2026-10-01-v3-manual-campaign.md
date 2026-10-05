# ICE v3 manual campaign and measurement readiness

This is the current execution guide. The [whole-process audit and 29-item
evidence map](2026-09-29-v3-reseed-harness-audit.md) owns the remaining research
questions; [reseed spec](../specs/RESEED_PLAN.md) owns the execution contract.
The first manual seed attempt stopped on turn4; r2 then stopped on turn54.
R3 twice failed at turn80 because a copied clause began with a capital letter;
its durable70 checkpoint and failed tails are preserved. Its third automatic
attempt was stopped gracefully for repair. No full campaign is complete.
The r4 restart bundle reuses identical reviewed labels. Source matching now
normalizes case/whitespace for comparison and keeps original bytes. Unresolved
quotes are withheld from graph writes while original source evidence survives.
The [fault audit](2026-10-03-v3-campaign-fault-audit.md) records the repairs and
their validation. The runner is a
development instrument, and completion does not prove all memory-quality work
is done.

## One entry point, five saved stages

### Current ICE v3 corpus and question schedule — 2026-10-03

Counts below come from the actual current `load_plan`, not the archived 618-probe
catalog. A recorded turn is one original user prompt plus its assistant reply;
1471 turns therefore contain 2942 such messages, excluding absent attachments.
The replay retains full histories. Checkpoints pause to observe accumulated
state; they do not reset memory or partition the database into isolated splits.

| History | Recorded turns | Checkpoints | Base questions | Added repeat occurrences | Source/answer reviewed | Source/answer unreviewed |
|---|---:|---:|---:|---:|---:|---:|
| `bb558b5f` |1119|22|78|4|78|0|
| `ecc64aab` |251|14|121|0|121|0|
| `355a5709` |101|3|56|0|56|0|
| Total |1471|39|255|4|255|0|

Checkpoint turn numbers:

- `bb558b5f`: 51, 115, 170, 216, 285, 336, 397, 425, 448, 492, 555, 604, 681, 735, 790, 834, 885, 959, 1017, 1053, 1067, 1119.
- `ecc64aab`: 18, 40, 64, 82, 106, 128, 132, 144, 145, 163, 184, 209, 234, 251.
- `355a5709`: 30, 65, 87. Its remaining originals 88–101 still replay; no extra question checkpoint is currently scheduled there.

| Question source | First history | Second history | Third history | Total |
|---|---:|---:|---:|---:|
| Native questions originally lacking source mapping |64|34|33|131|
| Existing native source-linked candidates |0|11|0|11|
| Generated source-first candidates delayed to checkpoints |14|76|23|113|
| Development repeats, separate recent/old controls |4|0|0|4|
| Frozen occurrences |82|121|56|259|

The native-unmapped row describes the original panel category; every row now
has a reviewed source/answer disposition, including exclusions and uncertainty.
All 255 base questions have catalog expected answers. Candidate
gold turns existed for 124 initially; an initial candidate is not a reviewed
denominator. The two text-export histories have synthetic five-minute clocks;
the third has provider-original dates. Order/turn-age retention is measurable;
the synthetic histories do not establish authentic calendar retention.

Each admitted occurrence uses four frozen arms: full ICE, direct Codex evidence
off, ICE warm-vector-only, and recent history only. The comparison arm is not
an independent vector database baseline. Each occurrence has three full-versus-
control contrasts, each judged in both display orders. Current admission is 113
old-source questions plus four development occurrences: 468 cloud answers and
702 successful judge-order requests if all are executed, before any retries.
This is a current-plan count, not completed requests or the final campaign size.
Question families and multiple arms/orders are dependent observations.

### Question counts at every checkpoint

These are scheduled base questions, not reviewed-valid answer denominators.
The added repeat controls have their own column. “Reviewed” refers to base
source/answer reviews; none of these counts is an observed correctness rate.

**History `bb558b5f`**

| Checkpoint turn | Base questions | Added repeats | Base source/answer reviewed |
|---:|---:|---:|---:|
|51|4|1|4|
|115|4|2|4|
|170|2|0|2|
|216|4|1|4|
|285|2|0|2|
|336|5|0|5|
|397|2|0|2|
|425|1|0|1|
|448|2|0|2|
|492|4|0|4|
|555|2|0|2|
|604|3|0|3|
|681|3|0|3|
|735|2|0|2|
|790|2|0|2|
|834|4|0|4|
|885|3|0|3|
|959|4|0|4|
|1017|2|0|2|
|1053|4|0|4|
|1067|1|0|1|
|1119|18|0|18|

**History `ecc64aab`**

| Checkpoint turn | Base questions | Added repeats | Base source/answer reviewed |
|---:|---:|---:|---:|
|18|3|0|3|
|40|4|0|4|
|64|10|0|10|
|82|15|0|15|
|106|24|0|24|
|128|18|0|18|
|132|5|0|5|
|144|8|0|8|
|145|6|0|6|
|163|6|0|6|
|184|13|0|13|
|209|4|0|4|
|234|3|0|3|
|251|2|0|2|

**History `355a5709`**

| Checkpoint turn | Base questions | Added repeats | Base source/answer reviewed |
|---:|---:|---:|---:|
|30|10|0|10|
|65|29|0|29|
|87|17|0|17|

Run from the repository root using `uv run`. PostgreSQL in `ice_postgres` and
Ollama must already be available; the unmaintained `./ice` scripts are not used.
No virtual-environment activation is needed: `uv run` uses the project's `.venv`.
The runner creates a persistent, randomly named campaign database. It does not
seed or restore the normal user database. It freezes the current memory-table
schema, including production indexes/defaults/constraints, **without data**.
The historical fresh migration chain is unqualified: a development check fails
while dropping an absent old vector index. This is an installation question,
not a reason to run the measurement against a different schema.

The active private bundle is `logs/z1-v3-manual-2026-10-04-r4`, already initialized
with byte-identical reviewed labels and repeat controls from the original
`logs/z1-v3-manual-2026-10-01`. It contains all 124 source-linked review candidates
and 131 native source-mapping reviews. Do not initialize it again.

**v3 crash repair, 2026-10-03:** the original attempt processed three turns and
failed in turn4 extraction, with zero completed durable turns. NuExtract emitted
an exact source sentence with null relation/object fields; the parser's earlier
source-only exception covered null objects only. The generalized template path
retains verified original sentences without graph assertions. The old trace,
store, checkpoint identity and source packets are preserved. The changed writer
must use the new bundle; never edit the old checkpoint's code hash to bypass
identity validation. The new bundle has no seed trace, stage state or database
yet. Private `restart-preparation.json` pins the old-file and copied-packet hashes.

**Current source-review checkpoint, 2026-10-03 (final r3):** all 255 base
questions have complete through-cutoff source/answer reviews; 0 remain pending.
Native 131: 36 valid, 77 invalid, 18 uncertain. Linked 124: 77 valid, 25 invalid,
22 uncertain. Total: 113 valid, 102 invalid, 40 uncertain. All three imported
original histories were read completely (1119/251/101 turns); each label uses
only sources through its own cutoff. No structural-only exclusions remain.
113 old-source admissions comprise 31 private, 26 mixed, 33 assistant-history and
23 public controls, reported separately. 248 reviewed answer keys correct or
qualify catalog keys; the other seven retain their reviewed originals.
Invalid/uncertain cases remain excluded, not silently converted into successful
memory questions. Recent supporting facts stay shared context and cannot count
as old-source hits. These are Codex source reviews, not independent human judge
qualification or measured answer gains. The immutable private r3 packet and
receipt pin all dispositions; earlier receipts remain historical artifacts.
No completed full replay or cloud campaign exists. Status separates
missing reviews from invalid/uncertain verdicts and window-only exclusions.
The current bundle is already reviewed: do not initialize it again or rewrite
its labels before launching. The source-review instructions below apply to new
bundles or separately versioned studies.
For a new bundle, initialize once:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/my-v3-run --init
```

Status is the default and makes no cloud request:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-04-r4
```

The existing bundle has passed label admission; cloud configuration is present.
Status still reports `cloud_answers_ready=false` before a complete replay trace
exists; this is expected and does not mean source reviews remain pending.
Execute or resume everything:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-04-r4 --run
```

The command runs **seed → snapshot → answers → judge → report**. It invokes
separate scripts internally; the operator need not run each script. To inspect
writer outcomes before answering, use the same entry point with `--run --stage
seed`. Individual `snapshot`, `answers`, `judge`, and `report` stages are also
available. Stage logs append under the bundle; the command prints each log path.
Live terminal progress now shows the stage/arm, processed turns out of 1471,
durable turns, checkpoint probes out of 259, successful answers out of 117 per
arm, and successful judge orders out of 234 per contrast. Processed turns finish
post-flight, maintenance and probes; durable turns have a complete saved store
checkpoint. Snapshot/report display elapsed work without inventing a percentage.
The phase is the last saved observable event, so a model call can run while the
counter stays still. Detailed logs remain separate; redirected output receives
plain changed-state/heartbeat lines. Optional detailed log view:
`tail -f logs/z1-v3-manual-2026-10-04-r4/seed_v3.log`.
On failure or Ctrl+C, the parent retains state and prints the relevant log path
or interruption notice. With unchanged inputs/code/settings, rerun the same
command; the observer reads retained/restored progress without double-counting.
Explicit truncated/noncompleted provider responses are failures even if their
prefix looks plausible. New bundles use `worker_failure_policy=continue` and
`cloud_failure_policy=continue`; standalone commands remain strict by default.
Idempotent post-flight gets two local attempts, then retains the original and
successful derivatives with a degraded receipt and proceeds. Periodic jobs get
one attempt per due cadence; committed changes are recorded and failed cadence
does not claim success. Three consecutive transport failures of one worker
checkpoint the current turn, then pause as an outage. Responsive local output
errors stay degraded and do not imply a provider outage. Original-turn completion
does not claim complete derivative processing.
Cloud answering and each judge order get two attempts, reserved durably before
requesting so interruption cannot silently reset the cap. Isolated exhausted
errors remain terminal ungraded rows; later questions proceed. Failed-answer
pairs need no judge call. Three successive failed cloud questions pause; quota/
authentication/unknown errors pause immediately. `processing_complete` means all
questions attempted; `complete` requires all successful. Errors remain in
planned denominators, never ties or incorrect grades. Reports separate clean,
degraded-gold and other degraded prefixes, with metadata kept out of prompts.
The coordinator itself permits TWO total child attempts for eligible fresh
failures; seed retry restores its verified checkpoint before replay. This is
bounded recovery, not a promise that a long run cannot fail.
New bundles save after every completed turn, retaining current/prior recovery
generations plus question snapshots. A failed turn therefore does not require
repeating ten successful turns. This adds store-dependent snapshot overhead;
the standalone seeder's default remains ten unless explicitly configured.
Actual copied70-turn store checkpoints took4.21/4.12s at17.45MB, verifying every
table/hash and restoring the synthetic71st turn. Snapshot cost grows with the
store; this is not final-store performance qualification.
Every attempt is saved in `campaign-attempts.jsonl`. Persistent failures leave
`campaign-pause.json`, with no later stage running. Cloud quota/authentication,
identity/configuration, resource and unknown failures require review instead
of automatic repeated calls. A pause does not mark incomplete memory work done.
The seeder releases its owned models when it exits; native proof/need calls keep
their model between calls instead of unloading it every time. Provider load and
evaluation durations and background-call wall times are logged.

**2026-10-05 instrument repair:** r4 has 1120 durable originals and 82 captured
probes. It paused because three summary completions hit their 400-token limit;
this was a responsive-output failure, incorrectly counted as a transport outage.
The exact reviewed guard/checkpoint-tool changes are registered and a private
continuation archive/receipt is prepared. Memory writers, budgets, labels and
settings are unchanged. Resume the same r4 command above: the first history
remains complete, and the second continues after its saved first turn. All 34
degraded workers stay in the evidence. The old manifest is never edited and
any unrelated change still refuses resume. For another eligible paused bundle,
preparation is explicit, never automatic:

```bash
uv run python scripts/z1/prepare_worker_continuation.py --run-dir logs/z1-v3-manual-2026-10-04-r4
```

This preparation does not restore the DB or launch the campaign. Summary and
extraction output budgets remain quality findings for post-seed review/tuning.
Do not change inference parallelism or residency during this frozen run.

The 2026-10-04 corrected writer changed replay identity. Its fresh r4
restart began from turn 1; do not resume r3 with altered code or edit its
70-turn manifest. The old checkpoint remains usable with its original frozen
writer, but cannot certify the corrected system. Subsequent interruptions of r4
use the same r4 command with unchanged code/settings/models/inputs.
Combined execution checks packet corpus/catalog identity and source bounds
before starting replay, then checks the complete trace and labels again before
cloud calls. Marking a malformed review `valid` does not make it ready to run.

**USER-REQUIRED — source review for a new bundle:** edit `labels-source-linked.json` and
`labels-native.json` under the private bundle. Each admitted row needs a
`valid` verdict, a concrete reason, `reviewed_through_turn` equal to its cutoff,
`recent_only_answerable: false`, an allowed `knowledge_scope`, and nonempty
`task_types`. Native rows additionally need every necessary old source in
`reviewed_gold_turns`. Read complete user/assistant sources and intervening
turns through that cutoff. Lexical suggestions are navigation only. Mark
unsupported, ambiguous or superseded answers `invalid` or `uncertain`; do not
manufacture a valid denominator. Public facts are controls, not evidence of
private-memory gain. The packets list allowed categories. Review duration
depends on source length; there is no credible blanket minutes estimate.
Done means the intended coverage cells have reviewed valid cases and all
excluded/unreviewed counts remain visible. A partial panel stays partial.
Either packet can supply `reviewed_expected_answer` when the catalog key is
unsupported or overprecise. Keep the original `expected_answer` unchanged;
both are saved beside the answer, and neither enters its input. Distinguish
required facts from optional elaboration, user statements from assistant
interpretations, and approximate recollections from verified exact values.
Record reviewer/scope honestly: the current Codex source reviews do not qualify
the cloud judge against independent human judgments.
A catalog-only bound check found 17 of the 131 unmapped native questions at
cutoffs no later than turn 40. Under the current 40-turn window, they have no
possible older source; keep them as immediate-history controls, not failed
long-term retrievals. The other 114 are only structurally eligible for old-source
mapping. None of this validates their expected answers. The private
`native-window-eligibility.json` receipt pins the packet and exact question IDs.

The native catalog already contains some natural revisits: the current review
verified one fact asked at an immediate checkpoint and again at a later
checkpoint where recent history cannot answer it. Reviewed meaning-equivalent
occurrences share `question_family_id`, while probe IDs/cutoffs and each
source/correction review remain separate. Answer plans and judge reports count
occurrences per declared family; undeclared equivalence stays unreviewed and
no independent-sample count is inferred. The full repeated retention schedule
belongs to final semi-LSREP. The base255 panel stays intact; the authorized
small development pass below adds its own reviewed before/after controls.

**USER-REQUIRED — manual execution:** keep the machine running for the chosen
stage and rerun the same command after interruption. Set the existing private
`PROBE_API_BASE_URL`, `PROBE_API_KEY`, and `PROBE_MODEL` configuration before
cloud stages. The answerer is `opencode-luna6` / `gpt-6-luna`; the judge uses
`PROBE_MODEL`; local background models are E4B and NuExtract3. The runner checks
configuration without spending a completion. For N admitted questions the
planned answer/judge budget is 4N answers plus 6N judge orders across three
contrasts; interrupted unpersisted requests can be repeated. Runtime is not
yet measured for the full corpus. Done means every stage passed its artifact
validation, not merely that `stage-status.json` says complete.

## Delayed reference and source limits

22 reviewed linked questions and 18 native questions remain uncertain. Reasons
include delayed references such as: generic
references such as “that answer” or “this club” no longer identify a unique
original topic, an advertised opportunity is mistaken for an actual outcome,
or an assistant assertion supplies an unconfirmed personal fact.
A gold-turn label cannot supply missing context to the answering model. Keep
such rows out of scored answers; a separately reviewed explicit question or
independently supported outcome would be needed. Do not grade plausible alternate
interpretations as wrong. Corrected historical-advice keys also distinguish
“not mandatory” from “has no value,” and “not automatic” from “never possible.”

## Small development repeat comparison

The current bundle contains a reviewed `development-repeat-review.json` with
two unchanged-fact families: recent→old cutoffs51→115 and115→216. It adds four
separate prompts to the base255 (259 total). The same `--run` command detects
this packet automatically for seed and all four answer arms. No separate full
replay or operator repeat loop is needed. Standalone scripts accept
`--development-repeat-review logs/<bundle>/development-repeat-review.json`.

Each phase carries its own source/answer review. The first is an explicitly
recent-history control and is excluded from old-memory scope/task totals.
`retention_comparison` in partial/final judge files and the campaign report
shows each family's cutoff/source age and each arm's grades, ordinal movement,
correct→not-correct and reverse transitions, estimated after-minus-before prompt
tokens, and selected source-presence counts/deltas. Missing phases and judge
errors/uncertainty remain unresolved. Source presence does not prove that the
answer-bearing words survived. Families are dependent observations and time
plus intervening content confound causal explanations; this is a small
**development** diagnostic, not the final semi-LSREP schedule.

The packet is pinned before seeding. Do not add or edit it after starting a
replay/answer pass; a changed schedule/review needs a new campaign bundle.
Full campaign and broader judge qualification remain pending; the current
bundle's source-review dispositions are complete.

## Recovery guarantees and limits

- Replay saves all 35 ORM tables, a flushed/fsynced trace prefix, conversation
  and source IDs, completion counters and job cadence every 10 completed turns,
  plus every query checkpoint and conversation end. Resume verifies code,
  settings, corpus/catalog/classifier inputs and local writer manifest digests.
  It restores the saved state, preserves the failed tail privately and reruns
  only work after the last checkpoint. Up to 10 unfinished turns can repeat;
  this is not exactly-once background generation.
- Two rolling recovery generations are retained. Query-time snapshots remain
  independently available under `seed.recovery/evaluation/`; hard links keep
  them through recovery cleanup. Later tuning must use those as-of states,
  not the final store containing future turns.
- A completed seed resumes without rewriting its trace if the live store still
  matches its final counts and row fingerprints. Changed code/settings/inputs,
  schema, mutable model tag contents or trace bytes fail closed. A changed
  study needs a new bundle; do not edit manifests to force resume.
- Answers save after each completed request. They pin the selected IDs,
  labels, model/API adapter/endpoint and actual decoding policy, and preserve
  the exact frozen input. Failed attempts are retained and retried only by an
  explicit resume. Luna omits temperature: the effective policy is provider
  default, not deterministic temperature zero.
- Judges save after each display order. Resume reuses a saved first order or
  completed pair and performs only missing work. Each v3 order makes one API
  attempt; outages/schema errors stop the stage. Both raw orders, disagreements
  and failed attempts remain visible. A response returned just before a process
  dies but before durable saving can require one repeat request.
- Operator locks prevent two copies racing the same artifacts. Bootstrap
  installs DDL transactionally and rechecks schema identity on reattachment.
  An unmarked database is never adopted. Interruption in the narrow database
  creation/ownership-comment gap can require manual ownership inspection;
  the runner refuses attachment rather than guessing. Recovery files need
  intact local disk; interruption recovery is not protection from disk loss.

## What the measurements can establish

| Measurement | Ground truth now | Signal and valid inference | Still needed |
|---|---|---|---|
| Replay order/completeness | Pinned1471 originals across three histories | Every preflight precedes its recorded reply; no missing turn or future evidence | Full manual replay; this does not score new answers |
| Expected answers/source turns | 255 source/answer reviews complete; 113 old-source admissions including 23 public controls; 102 invalid and 40 uncertain excluded | After review, old source identity at exact query time is a usable retrieval denominator | Remaining private/task/conversation coverage and ambiguous cases; no valid multi-hop cases, only one abstention case and six admitted third-history questions; historical assistant claims do not establish public truth |
| Retrieval rank@5/@10 | Reviewed source IDs | First five/ten pre-budget fragments; distinct-source-turn ranks separately collapse repeats | A source ID is not proof of relevant words surviving compression |
| Final prompt evidence | Exact frozen messages and source receipts | Candidate→budget→selected/source-note funnel locates where a source disappears | Semantic support review against exact visible text |
| Answer correctness | Reviewed expected answer and complete dated originals | Per-arm correct/partial/incorrect/uncertain, both-failed and errors | Broad real-pair human qualification; all-correct three-pair review is insufficient |
| Scope/task outcome strata | Reviewed knowledge scope and multi-label semantic tasks | Per-scope/task grades, paired correctness and paired estimated token cost; empty/unreviewed groups and errors are visible | Fill task/conversation coverage; public controls do not establish private memory gain; overlapping task/family observations are dependent |
| Memory contribution | Four paired prompt arms and reviewed knowledge scope | Direct graph evidence, warm-vector and recent-history contrasts estimate answer effects | Do not infer causal use from correctness/citations; every-leg ablations and independent vector baseline remain combined-Z work |
| Prompt cost | Same tokenizer estimate and saved provider usage | Paired medians/differences and errors, alongside answer quality | Estimates are not provider tokenizer counts; no matched-quality cost curve or proven token saving yet |
| Graph/claim truth and entity merges | Complete originals and source links; earlier labels may describe superseded extractor | Output-specific source/role/negation precision and recall review | Seeded blind graph review; graph density is not correctness |
| Summary/fold quality | Originals, initial summaries/abstracts and exact maintenance output changes retained privately | Judge support, contradiction, required-detail omission and fallback actually selected | Output-specific review; NLI verdict/source freshness is not independent ground truth |
| Temporal/update behavior | Source order; only one history has authentic dates | Old/new fact pairs can test coexistence, correction and historical answers | Reviewed successor pairs; synthetic dates cannot validate authentic calendar retention |
| Procedural relevance | Original cited user sources and stored activations | Query-specific usefulness and supported answer effects | Reviewed procedure/task labels; activation count or session length is not quality |
| Classifier/style invariance | Existing meaning-preserving controls; source-review task taxonomy | Gate flips, source arrival and answer changes on same-meaning variants | Combined-Z variants with final answers, not just a classifier score |
| Maintenance/model resources | Ten registry jobs, source-clock events, aggregate/semantic diffs, separate job/observer timings and model manifest pins | Eligible/proposed/applied changes and declared serial cadence | Output-quality reviews, peak VRAM/latency and actual asynchronous scheduling remain unmeasured |

The replay report's source funnel uses **unreviewed screened candidates** and is
marked accordingly. The campaign report separately counts source coverage on
admitted answer labels. An unlabelled source is neither a successful retrieval
nor a failed one. `qualification_pending` and `score_of_record=false` remain
explicit. Regrading the same full answer in three comparisons does not create
three independent questions; do not pool those repeated grades as sample size.

## Pipeline scope and remaining order

The replay uses conversation-aware classification, shared foreground warm-store
pressure, B2, timescope, retrieval, final source refinement, prompt budgeting,
source spans, post-flight and ten real memory jobs. It supplies the existing
reply only after preflight. Query observers are read-only, disable exposure
writes and roll back between arms. Source-time clocks cover memory decisions;
model/network elapsed timers remain real. Two histories use declared simulated
five-minute dates; the third matches original provider timestamps.

Routing is deliberately per-prompt registry selection without session
stickiness. The routed local model determines the budget; cloud Luna answers
the frozen messages. This is memory-path evaluation under a declared routing
policy, not a complete HTTP/streaming, asynchronous-runtime, project/document,
GPU-deferral or multi-user product test. The historical parity test duplicates
an older preamble and cannot certify the current whole endpoint; use current
shared-preparation and foreground route controls for their actual scope.

Cross-checking ROADMAP, SESSION and V3_REPAIR found no separately demonstrated
pre-tuning **core-memory** code blocker. Counts remain 160 anchors:93 checked,
67 open, partitioned into 29 evidence-dependent (12 seed/write-side, 16 combined
coverage/tuning/answer-side, 1 post-Z), 35 later/product/research, and 3 Z gates.
The [existing evidence map](2026-09-29-v3-reseed-harness-audit.md#open-item-evidence-map)
states each item's required observation; no automatic new checkmarks follow
from harness tests. Source review and instrument mechanics are now verified
for the current development bundle. Broader judge qualification remains needed
before treating its diagnostic grades as scores of record. Seeded output reviews then drive
specific repairs and targeted tuning, with stage-wise sensitivity and an
interaction check on development cases. Freeze defaults before the two agreed
final conditions: **LME oracle and semi-LSREP only**, not full LME-S. Those final
condition runners and an independent vector baseline are not supplied by this
development campaign command and need their own frozen/resumable contracts.

## Final prelaunch sweep — 2026-10-03

The current source-review work is complete. All 255 base questions have an
as-of expected-answer/source disposition, including honest exclusions. This
does not mean all 255 can be answered or belong in an old-memory denominator.
The admitted base panel has 80 episodic, 40 procedural, 24 summary, 21 relation,
16 temporal, 12 update, 16 negative and one abstention task labels. These overlap;
there are no valid multi-hop cases. By history, valid base cases are 40/67/6.
An empty or sparse cell is a coverage limit, not a perfect score.

The private seed trace now preserves initial `summary_text`, `abstract_text`
and `source_raw_sha256` for every completed post-flight. Each maintenance event
preserves aggregate state plus `semantic_observed_tables` and exact
`semantic_changes` with before/after values and stable (including composite)
keys. Text-producing jobs observe notes/manifests, entity/edge/procedural state,
clusters and memberships, slots, session summaries and review proposals;
decay/compaction observe narrower relevant tables. Empty changes explicitly
mean no differences in those observed tables. Vectors and repeated original
bodies are excluded from this observer; full snapshots preserve them. This
is an inspection record, not an exhaustive transaction log or a faithfulness
grade. `observer_elapsed_ms` is separate from `job_elapsed_ms`; serial job
timings cannot establish real interactive latency or peak VRAM.

For the 29 evidence-dependent roadmap questions, use the existing detailed
evidence map together with this division:

| Question group | Saved by this campaign | What happens after the run |
|---|---|---|
| Twelve seed/write questions: source representation, graph extraction/canonicalisation, long-source tails, folds, clusters, updates and maintenance | Full original corpus, initial representations, source-linked claims/edges, all-table as-of snapshots, exact maintenance output changes, failures and model/settings receipts | Inspect actual outputs against originals; label accepted/rejected relations, omissions, unsafe merges, successor pairs and cluster names. Job completion or an NLI pass alone does not close these questions. |
| Retrieval, representation use and answer effects within the sixteen combined-stage questions | Per-leg candidates, fragment and distinct-source ranks@5/@10, budget/selected-source receipts, exact final messages, four paired answers, both-order grades and costs, two repeated recent→old families | Locate losses from source→write→candidate→prompt→answer; review semantic evidence survival. Use frozen checkpoint stores for targeted weight/budget tuning and needed leg controls, rather than replaying the entire history for each setting. |
| Remaining combined-stage controls: same-meaning styles, every-leg ablations, alternative background/default models, independent vector baseline and judge qualification | Frozen input/state and current model identities provide reusable inputs; current controls remove direct graph evidence or restrict ICE to its warm vector leg | Run the specific variants/controls and independent human output reviews that their original scopes require. The four-arm campaign does not silently perform these additional experiments. |
| One post-baseline temporal consumer question | Source dates/order and as-of state retained | Test the consumer only after freezing the combined-stage baseline; synthetic histories cannot prove authentic calendar retention. |

The campaign therefore gathers the reusable system evidence needed to diagnose
its observed failures. It does not promise coverage of every unobserved failure,
automatic semantic ground truth for every generated output, or closure of all
29 items from one command. Its source review is Codex adjudication, and broad
independent judge calibration remains separate. Run readiness means the
development instrument can execute/resume safely under its declared scope;
the aggregate report deliberately keeps `score_of_record=false`.

Prelaunch verification includes 499 disposable-database smoke checks, actual
schema bootstrap/reattachment controls, an actual SQL unchanged-count summary
rewrite plus real procedural decay through the job boundary, and an actual
two-turn local-model replay interrupted after its second committed write.
Recovery preserves the first turn's identity, restores/replays the unfinished
second turn once, and retains the failed trace tail. These are harness controls,
not a completed 1471-turn seed or a memory-quality result. No full campaign or
cloud answer/judge request was launched during this sweep.

**v3 post-crash verification, 2026-10-03:** 522 disposable smoke checks pass,
plus the six standalone extraction-completion checks and 79 focused provider/
answer/judge/coordinator checks (overlap, not independent totals). Nullable
source fields pass through the real claim writer/retriever without an edge;
missing keys, wrong types, fabricated quotes and truncated extraction still
fail. A real local-model four-turn replay of the failing history completed all
ten maintenance jobs, observed the same null-relation shape on turn4, saved a
four-turn durable checkpoint and resumed with identical written rows. A real
child-write control verifies progress updates while its subprocess runs. Fault
controls cover all five parent stages, quota/transport/empty/incomplete answers,
second-order judge interruption/outage and checkpoint write/identity failures.
No full campaign or live cloud request was launched by these checks. The first
four-turn fixture's byte-preserving assertion was wrong for a partial replay;
it was corrected and rerun. One smoke fixture also needed to mock the new plan
reader alongside its already mocked label validator; actual validation stayed
strict. Passing controls do not establish memory gains or exhaust every future
model response.
