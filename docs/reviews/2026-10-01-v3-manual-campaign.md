# ICE v3 manual campaign and measurement readiness

This is the current execution guide. The [whole-process audit and 29-item
evidence map](2026-09-29-v3-reseed-harness-audit.md) owns the remaining research
questions; [reseed spec](../specs/RESEED_PLAN.md) owns the execution contract.
The full campaign has **not** been run. The runner is a development instrument,
and its completion is not a claim that all memory-quality work is done.

## One entry point, five saved stages

Run from the repository root using `uv run`. PostgreSQL in `ice_postgres` and
Ollama must already be available; the unmaintained `./ice` scripts are not used.
The runner creates a persistent, randomly named campaign database. It does not
seed or restore the normal user database. It freezes the current memory-table
schema, including production indexes/defaults/constraints, **without data**.
The historical fresh migration chain is unqualified: a development check fails
while dropping an absent old vector index. This is an installation question,
not a reason to run the measurement against a different schema.

The private bundle `logs/z1-v3-manual-2026-10-01` is already initialized. It
contains all 124 source-linked review candidates and 131 native source-mapping
reviews. The third 2026-10-02 receipt records **31 source/answer reviews**:
23 native mappings (six admitted,17 recent controls) and eight source-linked
candidates (six admitted,two recent or updated exclusions). Another ten early
native exclusions remain structural only. Native verdicts are six valid,
27 excluded and98 unset; linked verdicts are six valid,two excluded and116 unset.
**108 native mappings/keys plus116 linked candidates remain unreviewed:224 total.**
No campaign database, full replay or cloud campaign exists. Twelve old-source
admissions span two histories; the new linked admissions primarily recall past
assistant claims/recommendations, not verified public facts or private user
decisions. They do not fill the remaining private/task coverage gaps. Twenty-four
reviewed keys preserve attribution, uncertainty and required-versus-optional
facts while retaining immutable catalog answers. Complete original user and
assistant review reaches turn216 in one history and64 in another. All31 current
source/answer reviews now have full through-cutoff scope; the third history
remains unread for source adjudication.
For a new bundle, initialize once:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/my-v3-run --init
```

Status is the default and makes no cloud request:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-01
```

After label admission and provider setup, execute or resume everything:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-01 --run
```

The command runs **seed → snapshot → answers → judge → report**. It invokes
separate scripts internally; the operator need not run each script. To inspect
writer outcomes before answering, use the same entry point with `--run --stage
seed`. Individual `snapshot`, `answers`, `judge`, and `report` stages are also
available. Stage logs append under the bundle; the command prints each log path.
Watch replay progress with `tail -f logs/z1-v3-manual-2026-10-01/seed_v3.log`.
Combined execution checks packet corpus/catalog identity and source bounds
before starting replay, then checks the complete trace and labels again before
cloud calls. Marking a malformed review `valid` does not make it ready to run.

**USER-REQUIRED — source review:** edit `labels-source-linked.json` and
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
Full campaign and broader source/judge readiness remain pending.

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
| Expected answers/source turns |255 catalog answers;31 source/answer reviews,12 old-source admissions;108 native and116 linked reviews remain | After review, old source identity at exact query time is a usable retrieval denominator | Complete-source, correction and recent-only review;224 pending reviews and remaining private/task/conversation coverage; historical assistant claims do not establish public truth |
| Retrieval rank@5/@10 | Reviewed source IDs | First five/ten pre-budget fragments; distinct-source-turn ranks separately collapse repeats | A source ID is not proof of relevant words surviving compression |
| Final prompt evidence | Exact frozen messages and source receipts | Candidate→budget→selected/source-note funnel locates where a source disappears | Semantic support review against exact visible text |
| Answer correctness | Reviewed expected answer and complete dated originals | Per-arm correct/partial/incorrect/uncertain, both-failed and errors | Broad real-pair human qualification; all-correct three-pair review is insufficient |
| Scope/task outcome strata | Reviewed knowledge scope and multi-label semantic tasks | Per-scope/task grades, paired correctness and paired estimated token cost; empty/unreviewed groups and errors are visible | Fill task/conversation coverage; public controls do not establish private memory gain; overlapping task/family observations are dependent |
| Memory contribution | Four paired prompt arms and reviewed knowledge scope | Direct graph evidence, warm-vector and recent-history contrasts estimate answer effects | Do not infer causal use from correctness/citations; every-leg ablations and independent vector baseline remain combined-Z work |
| Prompt cost | Same tokenizer estimate and saved provider usage | Paired medians/differences and errors, alongside answer quality | Estimates are not provider tokenizer counts; no matched-quality cost curve or proven token saving yet |
| Graph/claim truth and entity merges | Complete originals and source links; earlier labels may describe superseded extractor | Output-specific source/role/negation precision and recall review | Seeded blind graph review; graph density is not correctness |
| Summary/fold quality | Originals and generated notes retained in store | Judge support, contradiction, required-detail omission and fallback actually selected | Output-specific review; NLI verdict/source freshness is not independent ground truth |
| Temporal/update behavior | Source order; only one history has authentic dates | Old/new fact pairs can test coexistence, correction and historical answers | Reviewed successor pairs; synthetic dates cannot validate authentic calendar retention |
| Procedural relevance | Original cited user sources and stored activations | Query-specific usefulness and supported answer effects | Reviewed procedure/task labels; activation count or session length is not quality |
| Classifier/style invariance | Existing meaning-preserving controls; source-review task taxonomy | Gate flips, source arrival and answer changes on same-meaning variants | Combined-Z variants with final answers, not just a classifier score |
| Maintenance/model resources | Ten registry jobs, source-clock events, state diffs and model manifest pins | Eligible/proposed/applied changes and declared serial cadence | Output-quality reviews, peak VRAM/latency and actual asynchronous scheduling remain unmeasured |

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
from harness tests. Immediate prerequisites are trustworthy labels, instrument
mechanics and broader judge qualification. Seeded output reviews then drive
specific repairs and targeted tuning, with stage-wise sensitivity and an
interaction check on development cases. Freeze defaults before the two agreed
final conditions: **LME oracle and semi-LSREP only**, not full LME-S. Those final
condition runners and an independent vector baseline are not supplied by this
development campaign command and need their own frozen/resumable contracts.
