# Z1 — the experiment index

**V3 manual entry point, 2026-10-01:** use `run_v3_campaign.py --run-dir
logs/<bundle>` for status, `--init` once for complete private review packets,
and `--run` to execute/resume seed → snapshot → four cloud answer arms →
three both-order judge contrasts → report. `--stage` selects one stage. The
maintainer runs the full campaign manually. The runner uses an owned persistent
database with a frozen schema-only production template; source data stays out
of bootstrap. It saves replay store/trace checkpoints and independent as-of
snapshots, plus each cloud answer and judge order. Exact input/model/label
identity is required for resume. Status does not call cloud models. Labels
require complete through-cutoff/recent review and explicit task/knowledge scope;
the sixth 2026-10-02 receipt records 88 source/answer reviews: 28 native
(six admitted, 22 recent controls) and 60 linked (39 admitted, 11 recent exclusions,
ten uncertain). Ten early exclusions remain structural only. Another 103 native
and 64 linked reviews remain, 167 total; ten reviewed uncertain cases also remain
unusable. Forty-five old-source admissions comprise 11 private, seven mixed, 16
assistant-history and 11 public controls; generic answers do not establish
private-memory gain. Eighty-one reviewed keys preserve corrections,
attribution and uncertainty. Complete source review reaches 216/128 in two
histories; third-history coverage and broader judge qualification are pending.
Status separates missing source reviews from unset/invalid/uncertain verdicts.
Reviewed key corrections preserve catalog provenance and never enter answering
inputs. Natural revisits can share reviewed question families while retaining
cutoffs; the full repeat schedule belongs to final semi-LSREP. Reports remain diagnostic
while judge qualification and output-specific truth audits are pending.
Judge receipts and the manual report separate absolute grades, paired
correctness and estimated prompt costs by reviewed knowledge scope and semantic
task. Public controls stay separate from private recall; missing and zero-case
groups remain visible, and multi-label task groups/families cannot be summed as
independent samples. Partial and resumed reports keep the same summaries.
A private `development-repeat-review.json` now adds a small recent-to-old pass
using the same manual command. Current reviewed selection: two unchanged-fact
families/four occurrences, 259 frozen prompts including the base255. Both
cutoffs keep separate source/key review and IDs; repeat queries do not earn
memory writes. The report compares each arm before/after with accuracy movement,
estimated prompt costs, source presence, missing phases and errors. The recent
phase is a separate control; this does not run full semi-LSREP.
See the [manual guide and complete measurement matrix](../../docs/reviews/2026-10-01-v3-manual-campaign.md).

**v3 reseed harness in verification, 2026-09-29:** use `seed_v3.py`, not the
historical 293-turn `seed_store.py`, for the new 1,471-turn run. Run it only in
a dedicated disposable database through `tests/support/disposable_database.py`
until the short live gates and the full-run design review pass. `--check`
validates the full source and probe mappings without loading models. All 444
typed catalog cutoffs equal their latest gold turn, so those questions cannot
establish long-term-memory quality at their original cutoffs. The default
`--probe-panel existing` schedules 124 source-linked candidates: 11 source-mapped
mature/curated questions at their real section checkpoint, plus 113 existing
source-first questions at the first later section checkpoint outside the
40-turn recent window. The 113 pass an exact-quote and question-only ambiguity
screen using only history before that checkpoint, not a semantic label review.
These 124 candidates cover 11 checkpoint times, but the 1,119-turn history is
represented only at its final section.
The same replay also freezes **131 unscored native questions** at their
original section times, for 255 as-of prompts in total. This preserves their
real query-time state before source labeling; a zero-gold trace is never
credited as a successful retrieval. The larger native catalog spans 39
in-history section checkpoints, and 32 curated cutoffs lie beyond the
selected history. `build_checkpoint_source_review.py` prepares the private
source-label packet under `logs/`; its lexical
suggestions are navigation aids, never gold. `--probe-panel all` adds 378
delayed typed candidates (502 source-linked and 131 unscored, 633 captured);
`typed_delayed` and `immediate` are diagnostic alternatives. Every answer label
needs source and intervening-turn
review. Catalog IDs repeat, so scheduled probes use unique stable identities
and retain the original ID as metadata. A short immediate development run uses
`--conversation 355a5709 --limit 1 --probe-panel immediate`; its
private JSONL trace belongs under `logs/`. `report_v3_replay.py <trace>` prints
aggregate generated → ranked candidate (gold fragment and distinct source-turn
rank@5/@10) → budgeted → final
fragment/source-note credit, write counts,
maintenance calls, lineage and failures. `snapshot.py` now covers every ORM
table and verifies restored row identities. The replay stores recorded answers,
so it does **not** provide an answer-quality number. Its as-of records include
full/no-Codex-evidence/vector-only/recent-only final prompts. The no-Codex
control removes graph/claim/timeline fragments while preserving other context.
It isolates direct graph evidence, not the full graph writer/query-expansion
effect. The vector control retains ICE's shared gate/reranker and warm vector
leg; it is not an independent all-originals vector-memory baseline.
Source-linked records also carry
complete gold sources for a later paired cloud-answer pass.
`build_longterm_label_review.py` creates a private packet under `logs/`
balanced by conversation and checkpoint, with complete gold and all intervening
turns; a reviewed `valid`
verdict needs a concrete reason, exact through-cutoff review, confirmation that
recent history alone cannot answer, knowledge scope and semantic task labels.
`answer_as_of.py --plan` validates saved prompts and expected answers; its full cloud run
accepts only valid rows from that matching packet. A reviewed native source
packet can be supplied with `--validated-native-sources` to attach complete
gold to its frozen checkpoint prompt without reseeding. `judge_answers.py`
pairs generated arm files by stable probe and historical cutoff, returns a
blind relative verdict and an absolute correct/partial/incorrect/uncertain
grade for each arm using the reviewed expected answer. It reports estimated
prompt-token medians and checks both display orders. Raw verdicts stay saved;
disputed preferences become `UNCERTAIN / order_unstable`, disputed grades
become uncertain for that arm, and either call's error keeps output incomplete.
The first order is saved before the second call. It explicitly marks its v3
rubric `qualification_pending` and not
a score of record until the current answer-pair calibration is resolved. Use
`calibrate_answer_judge.py --write-controls logs/<packet>.json` to prepare
12 authored diagnostic cases, then `--packet logs/<packet>.json --out
logs/<results>.json --plan` to validate without cloud calls. The actual run
uses the same campaign request/rubric in both answer orders, preserves every
result, supports matching `--resume`, and stops on three consecutive errors.
Real human-reviewed answer pairs use a separate packet kind; authored success
does not qualify the judge. The 2026-10-01 controls initially exposed a missing
OpenCode session header and an insufficient-evidence grade ambiguity. After
repair/clarification the same 12 authored cases passed both orders; this is
development on known controls, not independent human agreement.
Three maintainer-reviewed real pairs matched 12/12 factual grades across both
orders, but only 3/6 preferences and 2/3 order consistency. All six human grades
were correct, so this pilot does not qualify the other grade classes. The third
human preference was tentative. Keep factual correctness primary; do not
tune the rubric to force agreement on optional extra context.
Judging uses a stable OpenCode session header and ICE's own User-Agent;
historical question time and complete dated gold turns reach the judge.
Use
`snapshot.py save --arm <name> --trace logs/<complete-v3-trace>` to bind a
campaign snapshot to the complete unchanged replay; without `--trace` the
snapshot is development-only.
The complete trace is checked against its ordered preflight/write events,
source identities and exact scheduled probe manifest. Snapshot binding checks
both seed counts and row fingerprints. The isolated replay supplies source
time to memory-path Python/ORM clocks and explicit SQL `NOW()` calls, and
drives ten periodic memory jobs from the real registry/cadence/cycle cap.
Job effects include physical warm/cold movement. Each checkpoint freezes
source storage locations for later label joins and diagnosis. Two selected
text-export histories have constructed five-minute timestamps and retain
`synthetic_raw_import`; only the provider-export history has verified original
times. Source/query provenance reaches prompts and judges. Simulated elapsed
time does not prove authentic calendar retention. Historical turns retain
normal exposure writes; diagnostic questions and all matched arms use
read-only transactions with exposure disabled, then roll back. The trace
declares its serial turn-boundary schedule; it does not simulate asynchronous
GPU deferral, leases/retries or session-end bursts. Run
`tests/test_z1_historical_clock.py` and `tests/test_z1_replay_path.py --out logs/<fresh-name>`
through the disposable-database wrapper for the clock/observer and full
replay-to-snapshot path controls.
The full seed, full answer pass, broad real-pair judge qualification
and combined Z have **not** run yet; the old scripts and results below remain
historical.

**⚑ READ THIS BEFORE RUNNING ANY Z1 EXPERIMENT.** It exists for one reason: to
stop a session re-running something already settled, or re-deriving a number
already recorded. Both have happened repeatedly — [TRAPS #42](../../docs/TRAPS.md)
is the entry for it, and it names two cases from a single afternoon.

**v3 instrument update, 2026-09-28:** `score_typed.py` measures final
source-gated prompt evidence, while `score_retrieval.py` measures only raw
retrieval candidates; do not compare their recall numbers as the same metric.
`answer_probes.py` uses the final prompt and defaults to cloud gpt-6-luna,
with a local answer switch. The paired judge requires complete gold source and
matching answer models. The 180-turn working seed is incomplete, and the
29 anchorless Codex probes plus 15 temporal probes without successor pairs
remain unscored for their intended metrics. The earlier runs indexed below
were made with older instruments and are historical.

**v3 entry checkpoint, 2026-09-29 — supersedes §2's old Step 0 order:**
the independent repair work has reached the combined Z1/Z2 phase. The
10-item pre-reseed list below records its 2026-08 state and must not be run
as a fresh queue. `test_retrieval_quality.py` now guards the current reader
with 30 competing synthetic turns and 15 single-/two-source questions; it
does not score answers or Codex value. **Use the repaired v3 harness:** the
historical `seed_store.py` selects only 293 curated turns and skips per-turn
preflight. The current `seed_v3.py` supplies the full chronological path, and
the current snapshot includes all 35 ORM tables. [Exact audit and acceptance
order](../../docs/reviews/2026-09-29-v3-reseed-harness-audit.md). Then make a
versioned complete v3 seed and validated gold/source mapping, complete the
missing typed anchor and temporal-pair labels, then run production-parity
retrieval and gpt-6-luna answer probes with matched-budget vector/recent-
history controls. Read individual outputs while tuning on development data;
keep the final LME oracle and semi-LSREP conversations separate. The roadmap
owns remaining items and the current count.

**This file is an INDEX, not a record.** One line per thing, with a pointer.
Nothing is explained here; everything lives in its normal home:

| what | where it lives |
|---|---|
| what was run, with what numbers, and what it does NOT show | [`docs/PROVENANCE.md`](../../docs/PROVENANCE.md) |
| the queue, and what is blocked on what | [`docs/ROADMAP.md`](../../docs/ROADMAP.md) |
| how a measurement lied, and how it was caught | [`docs/TRAPS.md`](../../docs/TRAPS.md) |
| which feature is on, off, or inert | [`docs/FEATURE_INVENTORY.md`](../../docs/FEATURE_INVENTORY.md) |
| which model did which job, and what was tested | [`docs/MODELS.md`](../../docs/MODELS.md) §5 |

**Z1/Z2 are one combined phase** for the current v3 work. This index includes
the retrieval instrument and points to the roadmap for answer-level checks.

---

## 0. ⛔⛔ EVERYTHING BELOW PREDATES THE 2026-08-25 CLEAN BREAK

**Read this before using ANY number in this file.**

Every Z1 arm — `ner-a-micro`, `ner-b-nuner`, `ner-b-postfix`, all four `dir-*` —
was built by **`qwen3:4b-instruct` on the instruct prompt**. That configuration
was measured on 2026-08-24 at **15% of stored triplets true, 30% reversed**, and
has been **replaced** by `NuExtract3-Q8_0` on a template prompt, which measures
**63% true, 0 reversals in 20 judged facts**.

⇒ **The maintainer has declared the prior arms dead data** (2026-08-25,
[G72](../../docs/ROADMAP.md#g72)): *"lets just call ALL from before as we have
no data, we are restarting ALL again."*

**What that means for this file:**

| | |
|---|---|
| §1 SETTLED | ⚠ **the answers hold; the NUMBERS do not.** "Does a bigger model fix extraction?" is still No — but all eight models were **generalists**, and a specialist was never tested ([MODELS.md:328](../../docs/MODELS.md)) |
| §3 measurement floor | ✅ **survives** — sampling, judge self-consistency and cross-process nondeterminism are model-independent |
| §3b falsification table | ✅ **survives, and the whole table is now historical** |
| §6 ARMS | ⛔ **all dead.** Kept for reproducing history only |

**The re-measurement register is [G71](../../docs/ROADMAP.md#g71)** — 16
PROVENANCE entries from 2026-08-11 to 08-24, triaged: 3 survive, 3 already dead,
**10 to re-measure**. The new baseline comes from
[docs/specs/RESEED_PLAN.md](../../docs/specs/RESEED_PLAN.md).

⚠ **The judge changed too.** `muse-spark-1.2-contributor` (73% against the
maintainer's labels) has 500'd on every call since 2026-08-23 — both keys, both
routes, all sizes, server-side. Calibrate any replacement with
`scripts/oneoff/calibrate_judge.py` **before** adopting it: `mimo-v2.5` is fast,
free, available, and scored **27%**, missing 6 of 6 reversals.

---

## 1. SETTLED — do not re-run these

| question | answer | where |
|---|---|---|
| Does a bigger background model fix extraction? | **No.** 8 models 4B→26B, defects in all 7 viable arms ⇒ prompt/design, not capacity | PROVENANCE `A12` (2026-08-12) |
| Does NuNER beat micro-NER for codex grounding? | **Form, not outcome.** malformed 25.0→12.5%, non-entity nodes 58.7→40.7%; correctness 18→20% (inside 1 SE) | PROVENANCE 2026-08-20 |
| Is `extraction_confidence` a usable truth signal? | **No — it is UNINFORMATIVE.** ⚠ The earlier "INVERTED" claim is **WITHDRAWN**: it rested on a 40-triplet subgroup quoted without an interval. Re-measured over 124 turns on two seeds, the runs disagree on the DIRECTION (grounded 22.3% vs 12.2%; rejected 15.0% both). Do not use it as a truth prior — because it says nothing, not because it is backwards | PROVENANCE 2026-08-23 |
| Do the six write-path mechanism fixes make the graph more TRUE? | **No — a null.** Every mechanism moved; correctness did not (z ≤ 1.14) | PROVENANCE 2026-08-22 |
| Are reversals created by relation canonicalisation? | **No.** Blocking 1,611 direction/polarity merges left the reversal rate flat ⇒ generated, not merged | PROVENANCE 2026-08-22 |
| Can a closed 197-word relation vocabulary work? | **No.** 67.8% of real relations are out-of-vocabulary; the repair ladder recovers 5.7% | PROVENANCE 2026-08-03/04 |
| Is the procedural leg's 1.000 a real score? | **No — a tautology.** It returns its limit (5 fragments) on every query, nonsense included | TRAPS #37 |
| Does the **direction rule (G59)** make the graph more true? | **No — a null.** 4 arms, 2 seeds each, full coverage: correct **15.8%** ON vs **15.5%** OFF (delta +0.28, z=0.09); `reversed` — the label it targets — **32.6% vs 32.6%** (delta +0.01). The earlier **+9.4 pts** is **WITHDRAWN**: it came from comparing a flat-sampler control (11.0%) against per-turn treatments. Same store re-judged per-turn is 16.7% | PROVENANCE 2026-08-23 |

## 2. Historical 2026-08 reseed order — superseded by the v3 entry checkpoint

**Everything below is blocked on one thing: a store big enough to measure.**
That is why the order matters more than the list. Full plan:
[`docs/specs/RESEED_PLAN.md`](../../docs/specs/RESEED_PLAN.md); the background
layer's own fixes: [`docs/specs/BG_LAYER_FIXES.md`](../../docs/specs/BG_LAYER_FIXES.md).

### ⚑ WHY THIS IS ORDERED THIS WAY

**The post-reseed list is enormous, and Z2 is bigger still.** So the rule is:
**anything that does NOT need a big store gets done BEFORE the reseed.** Not
because it is more important — because parking it behind a multi-hour seed and
a queue of blind judging is how work disappears for three sessions.

Two things make a piece of work "post-reseed", and nothing else does:
- it needs **more content than 180 turns** can provide (batch summaries,
  retrieval at realistic scale, answer probes with distractors), or
- it needs the **graph to be correct** (anything reading codex, since every
  pre-2026-08-26 graph was built by the 15%-correct extractor).

Everything else can run now, on turns read straight from
`simulation_full.jsonl`.

### Historical Step 0 — PRE-RESEED (2026-08; do not rerun as a queue)

| # | work | needs a store? | instrument |
|---|---|---|---|
| 0.1 | **[G32(a)](../../docs/ROADMAP.md#g32) native endpoint + per-request `keep_alive`** ⚠ do first — seeding is hours of background work and would otherwise inherit the host's residency policy. Same pass: `maintenance_agent`'s `json_object` (measured **0/8**) → `json_schema` (**8/8**) | no | — |
| 0.2 | **Gold turns for the 174 anchorless probes** — safe now the model is settled; feasibility proven at **343/368 = 93%** | no | `derive_retrieval_gt.py` |
| 0.3 | **[G73](../../docs/ROADMAP.md#g73) conversation fold** — 33-67% fabricated. A/B: fold against ORIGINAL turns · cap depth · apply the §1.1 softening | no | `judge_bg_quality.py --jobs conv_fold` |
| 0.4 | **[G74](../../docs/ROADMAP.md#g74) cluster naming** — prior quality numbers RETRACTED; inspect real similarity-grouped members with recurring-entity hints | no | repair input parity before `--jobs cluster_name` |
| 0.5 | **[G61](../../docs/ROADMAP.md#g61)** — four silent drops left in `extract_triplets` (1 of 6 fixed 2026-08-27) | no | code + logs |
| 0.6 | **Reconciler `reject_new`** — 4 uses in 30 chances. ⚠ **build a real gold set first**; the current one is n=9 hand-written | no | `bg_model_bakeoff.py --jobs reconcile` |
| 0.7 | **[G62](../../docs/ROADMAP.md#g62)** antonym branch — never fired (0 of 106) | no | — |
| 0.8 | **[G28](../../docs/ROADMAP.md#g28)** style invariance sweep — owns its own probe set | no | style-variant probes |
| 0.9 | **[G29](../../docs/ROADMAP.md#g29)** drift audit · **[G30](../../docs/ROADMAP.md#g30)** test blind spots | no | code reading |
| 0.10 | **[G69](../../docs/ROADMAP.md#g69)** relation-vocabulary growth loop (built, inert) · **[G64](../../docs/ROADMAP.md#g64)** fact-as-sentence (design) | no | — |

⇒ **Historical 2026-08 count: ten items, none then blocked.** Their current
status and remaining conditions are in the roadmap's 2026-09-29 checkpoint.

### What the old (pre-2026-08-25) queue said, re-homed — nothing here is lost

| old item | where it went |
|---|---|
| #1 "decide what the 14-17% ceiling is" | ✅ **answered — it was the model.** 15% → 63% ([G63](../../docs/ROADMAP.md#g63)) |
| #2 small-model sweep | ⛔ dropped — its premise was the broken prompt, and a specialist beat all eight generalists |
| #3 non-lossless turns ([G57](../../docs/ROADMAP.md#g57)) | still open; its gate *"sequence after something moves correctness"* is now **passed** |
| #4 stop using `extraction_confidence` as a truth prior | folded into reject-but-keep, step 2.2 below |
| #5-#7 [G61](../../docs/ROADMAP.md#g61) · [G62](../../docs/ROADMAP.md#g62) · [G60](../../docs/ROADMAP.md#g60) | now steps 0.5 / 0.7 / post-reseed |
| #8 per-leg ablations | post-reseed — ⚠ and `target_leg` exists on **93 of 618** probes only |
| #9 NuNER vs micro | ✅ settled — NuNER, junk names 8.7% vs 19.5% |
| #10 Z2 / answer layer | now [G66](../../docs/ROADMAP.md#g66); the reseed exists to feed it |

⚠ **Every NUMBER from that era still needs its status checked before use — that
is §3b, the falsification table, which is not superseded.**

### Step 1 — Seed

**1,471 turns**, three conversations: `bb558b5f` full (1,119) + `ecc64aab`
(251) + `355a5709` (101). ⛔ **`cca73c87` is NOT seeded — it IS `bb558b5f`'s
tail** (turns 1039-1119, verified 81/81; offset **+1038**). Source is
`data/simulation/simulation_full.jsonl`, the only file with both sides of every
turn. Config: `gemma4:e4b` background, `NuExtract3-Q8_0` extraction, `template`
mode, NuNER tier. ⚠ Print the grounded/ungrounded/rejected tier split in the
run's own output — [RESEED_PLAN §3](../../docs/specs/RESEED_PLAN.md) requires it
VISIBLE.

### Step 2 — What the store unblocks, and what each answers

| order | item | the question | needs |
|---|---|---|---|
| 2.1 | **graph baseline** | is the NuExtract3 graph TRUE at scale? First number of the post-qwen era | judge + blind sample |
| 2.2 | **reject-but-keep** ([G72](../../docs/ROADMAP.md#g72) §3) | 79% of facts land at `0.35`. Does grounding predict truth? | tier-stratified blind judging |
| 2.3 | **[G70](../../docs/ROADMAP.md#g70) read-side instrumentation** | is stored memory ever READ? **Plus the substitution counter** [G75](../../docs/ROADMAP.md#g75) needs | a counter at `_choose_representation` |
| 2.4 | **[G66](../../docs/ROADMAP.md#g66) the ablation** | do better facts produce better ANSWERS? codex leg ON vs OFF | 618 unified probes, blind answer judging |
| 2.5 | **[G75](../../docs/ROADMAP.md#g75) the trust gate** | ⚠ **decision, not a measurement** — resolved BY 2.3 + 2.4 | rate LOW ⇒ delete substitution · HIGH + harmful ⇒ build the recheck · HIGH + harmless ⇒ record coverage as decorative |
| 2.6 | **[G71](../../docs/ROADMAP.md#g71) the register** | re-measure everything accepted 2026-08-11 → 08-24 | 16 entries: 3 survive, 3 dead, **10 to redo** |

### Step 3 — *(merged into Step 0)*

The background-layer defects used to be listed here as "post-reseed adjacent".
They are not: none of them needs a store, so they are **steps 0.3-0.7 above**.
Keeping two lists of the same work is how one of them goes stale.

### ⚠ Three things that will trip the next session

1. **`summary_coverage` will read ~0.64, not ~0.78.** The summariser prompt is
   now faithfulness-tuned; the drop is the metric noticing the model stopped
   padding. **Not a regression.**
2. **The background prompts are COUPLED to `gemma4:e4b`.** The identical
   softening measured as a LOSS on `qwen3:4b-instruct` (54% → 34% faithful).
   Changing the background model means re-measuring the prompts.
3. **`dir-false-run1` no longer matches its `.counts` file** — the bake-off's
   store-backed jobs cleared `procedural_memory`, `batch_summaries` and the
   procedural idempotency keys. Backed up; deliberately not restored.

## 3. ⚑ THE MEASUREMENT FLOOR — read before quoting any number

- **Judge self-consistency: ±2.5 pts.** Same judge, same store, same seed, same
  200 triplets, run twice → 87.1% identical verdicts.
- **Sampling was the big term, and it is FIXED.** 200 triplets came from only
  ~76 turns, and triplets in a turn are correlated, so the independent unit is
  the TURN. `--per-turn 3` now covers **124 turns**, taking the interval from
  ±8 to **±6.2–6.6**.
- **Run-to-run variance in the RATE is small — measured 2026-08-23.** Two
  independent seeds of one configuration, full coverage: **17.1%** and
  **14.4%**, 2.7 points apart and inside both intervals. ⇒ **no heavy bootstrap
  is needed**; one run per arm suffices **provided it reports its interval**.
- ⚠ **The model is nondeterministic ACROSS PROCESSES** (temperature 0 fixes
  sampling, not logits). Two identical seeds share only 6 of 9 raw responses.
  That changes WHICH triplets exist; it does not move the aggregate rate much.
- ⇒ **Current honest figure: 14–17% correct, ±6 pts.** Every earlier number
  (20%, 15.8%, 11.0%, 20.4%, 10.0%) was one measurement under-sampled.

## 3b. ⛔ EVERY Z1 NUMBER, WITH ITS STATUS — the falsification table

**Read this before citing ANY figure from this project.** Nothing below is
deleted; each row says what the number is worth now. Added 2026-08-23 after a
session in which three separate "settled" findings were withdrawn — two of them
produced by the session doing the withdrawing.

| number | status | why |
|---|---|---|
| **recall@10 = 0.508** (2026-08-13) | ⛔ **DEAD, three ways** | episodic-only crediting (TRAPS #32) · 64% of that probe set is contaminated · the scorer classified without the conversation. Cite the *method* lesson, never the number |
| the 0.250 legacy control | ⛔ dead | same three defects; it is the other end of the same run |
| **20% triplet correctness** (2026-08-20) | ⚠ **superseded** | measured over 76 turns with no interval. Current figure is **14–17% ±6** over 124 turns |
| **25% reversed** (2026-08-20) | ⚠ superseded | same sampling; reversal now 21–37% depending on tier and run |
| **`extraction_confidence` INVERTED** | ⛔ **WITHDRAWN** | a 40-triplet subgroup. Two full-coverage runs disagree on the direction. The field is **uninformative**, not inverted |
| **`summary_synthesis` 0.303 / 0.322 / 0.324** | ⛔ dead | the metric was `max(coverage, presence)` — any summary fragment scored 1.0. Also measured with the summary leg switched off |
| **`procedural` 1.000** | ⛔ dead | presence test; the leg returns its limit on every query, nonsense included. Grounded score is **0.000** |
| **`episodic_lookup` 0.698 → 0.754** | ⛔ dead | 0.754 credited a 33-turn summary as "this turn came back". Corrected figure **0.599 ±0.032** |
| **`codex_multihop` 0.531** | ⚠ superseded | averaged two populations scored by different formulas. Split: anchored 0.46, unanchored 0.172. Corrected total **0.380** |
| **`temporal` 0.500 vs 0.643** | ⛔ never a result | n=28, ~1.9 SE apart. Was read as a difference; it is not one |
| **A4 grounded expansion "changes nothing"** | ⚠ **suspended** | a null from two conditions that were BOTH mis-called (65% of RRF weights wrong). Re-measure before deleting the feature |
| **`legoff-batch_summary` ablation** | ⛔ dead | ablated a leg that was already off. Four of five scores byte-identical to baseline |
| **the five per-leg ablation deltas** | ⛔ dead | same broken metric + no clean same-commit baseline |
| **codex ablation 25-14 / 36-17** | ⚠ ordering only | all three conditions shared the defect; the *rank order* survives, the absolutes do not |
| **answer verdicts 31-30, 58%/31% both_failed** | ⛔ dead | the judge truncated BOTH answers at 2,500 chars against a ~3,250 median |
| **A12's eight-model ranking** | ⚠ **weak** | scored on `summary_coverage`, which TRAPS #45 shows is circular; the ranking itself came from ONE subagent read. Its *conclusion* (defects in all 7 arms ⇒ prompt/design not capacity) stands |
| **direction rule +9.4 pts** | ⛔ **WITHDRAWN — resolved as a NULL** | the controls were re-judged at full coverage 2026-08-23. Pooled: 15.8% ON vs 15.5% OFF, z=0.09; `reversed` 32.6% vs 32.6%. The "+9.4" was a flat-sampler control (11.0%) against per-turn treatments — the same store per-turn is **16.7%** |
| **the six write-path fixes AND the direction rule, together** | ✅ **a trustworthy pair of nulls** | two independent prompt/mechanism interventions, each verifiably changing the mechanism, neither moving truth. ⇒ the 14–17% ceiling is not located at prompt-level instruction |
| **the six write-path mechanism fixes** | ✅ **null, and trustworthy** | every mechanism verifiably moved; correctness did not (z ≤ 1.14) |
| **store-level counts** (entities, edges, degree-1 %, expiry rate, `in` = 1 edge) | ✅ **TRUSTED** | direct SQL counts, no sampling, no judge, no retrieval |
| **NuNER halves malformed / non-entity nodes** | ✅ trusted | two independent instruments agree, store-level |
| **run-to-run rate variance is small** | ✅ trusted | two seeds, full coverage, 17.1% vs 14.4% |

**The pattern worth carrying:** everything measured by **counting the store
directly** survived. Everything that went through **a sampler, a metric or a
judge** needed correcting. That is where to be suspicious first.

## 4. SCRIPTS — what each one is for

**Seeding / arms**
| script | purpose |
|---|---|
| `seed_store.py` | build a store from the corpus (`--clean`, `--limit`, `--bg-model`) |
| `ner_arm_seed.sh` | the A9b two-arm NER comparison (micro vs NuNER) |
| `two_arm_seed.sh` | earlier two-arm background-model seed |
| `reseed_postfix.sh` | one arm, arm-B config, varying ONLY the code |
| `prompt_ab_noisefloor.sh` | 4 arms: each prompt twice → effect + noise floor |
| `check_reproducible.sh` | **detector**: same seed twice, compare. No fix in it, deliberately |
| `snapshot.py` | `save` / `restore` / `list` / `counts` an arm |
| `drain_batch_summaries.py` | force + ASSERT batch summaries (a swallowed failure once cost 3 days) |

**Probes / ground truth**
| script | purpose |
|---|---|
| `generate_typed_probes.py` | build the typed probe set |
| `generate_probes.py` | the older flat probe set |
| `derive_retrieval_gt.py` | ground truth for retrieval |
| `verify_gold.py` / `check_gold_consistency.py` | validate the gold before trusting a score |

**Scoring / judging**
| script | purpose |
|---|---|
| `score_typed.py` | 5 per-type metrics. **Never averaged into one number** |
| `score_retrieval.py` | recall@k / MRR on the flat probe set |
| `answer_probes.py` | retrieve → assemble the REAL prompt → generate an answer |
| `judge_answers.py` | paired A/B verdicts on answers |
| `judge_codex.py` | **is the stored graph TRUE?** the only outcome metric here |
| `judge_summaries.py` | summary faithfulness. ⚑ **Gates the judge first** — plants fabrications AND feeds it a verbatim copy of the source (which cannot fabricate). Without the verbatim arm, a judge that flags everything scores 100% detection and looks excellent |
| `judge_bg_quality.py` | the quality half for batch summary · conversation fold · cluster naming · procedural. Each job's defect is planted in ITS OWN shape — splice for the summarisation-shaped, **swap** (output belonging to different source material) for naming and procedural. VOIDs rather than ranks when the judge misses >40% |
| `bg_model_bakeoff.py` | 11 models × 9 background jobs, calling the PRODUCTION functions. Models swap via `settings.background_model_name`, resolved per call |
| `bakeoff_report.py` | merges every bake-off run. **Disqualifiers, not a composite score** — weighting 9 jobs into one number lets invented weights pick the winner silently |
| `prompt_ab_fabrication.py` | prompt / must-term A/Bs. Rewrites ONE substring in flight so arm B differs by exactly that, and **aborts** rather than reporting a null if the rewrite failed to bite |
| `probe_census.py` · `unify_probes.py` · `gt_feasibility.py` | probes across all 3 sources deduped (777 raw → **618 distinct**), unified to one schema, and the gold-turn feasibility check |
| `compare_judgements.py` | two judgement runs, per TRIPLET not per rate |
| `dump_graph_fingerprint.py` | compare two stores triplet-by-triplet; localises divergence |
| `harvest_probe_context.py` | reading instrument, computes no score |
| `run_leg_ablations.sh` | the five per-leg ablations |
| `store_report.py` | store statistics |
| `production_parity.py` | **the ONE reproduction of `main.py`'s pre-retrieval path** |
| `run_meta.py` | provenance block stamped into every artifact |

⚠ **Two known gaps in `judge_codex.py`, both deliberate-to-leave, not bugs to
trip over:**
1. **Verdicts record `edge_id` but NOT the source turn**, so a judgement
   artifact cannot be re-clustered by turn afterwards — `scripts/oneoff/g59_compare.py`
   has to hardcode turn counts read from the run logs. One line in the writer.
2. **stdout is neither flushed nor written incrementally**, so redirecting to a
   log shows *nothing* until the run ends (~30 min). That looks exactly like a
   hang. Check the process instead: an established socket to the judge API and
   zero CPU time means it is working, not stuck.

## 5. ARTIFACTS — where output lands

⚠ **All of `experiments/curation_files/` is GITIGNORED** (it carries the
corpus). A number that matters must be copied into PROVENANCE in the same
session — [TRAPS #40](../../docs/TRAPS.md).

| path | holds |
|---|---|
| `snapshots/<arm>.sql` + `.counts` | a full store, restorable |
| `score_runs/*.json` | typed / retrieval scores. ⚠ some are stamped `INVALIDATED` — read `meta` |
| `judgements/codex_quality_*.json` | per-triplet graph-truth verdicts |
| `probe_answers/*.json` | assembled prompts + generated answers |
| `fingerprints/*.json` | graph fingerprints for reproducibility checks |
| `typed_probes.json` | **the live probe set** (444). Others are older generations |
| `logs/` (repo root) | run logs. Gitignored but survive reboot — never write to `/tmp` |

## 6. ARMS — which store is which

| snapshot | NER | code | prompt | note |
|---|---|---|---|---|
| `ner-a-micro` | micro | pre-08-22 | old | A9b arm A |
| `ner-b-nuner` | NuNER | pre-08-22 | old | **A9b arm B — the baseline most numbers cite** |
| `ner-b-postfix` | NuNER | 08-22 fixes | old | the six write-path fixes, 293 turns |
| `dir-false-run1/2` | NuNER | 08-22 | old | direction-rule control, 180 turns |
| `dir-true-run1/2` | NuNER | 08-22 | **new** | direction rule ON |
| `fixed-*`, `gemma4-26b`, `pre-*`, `arm1-post-g50` | — | older | — | superseded; kept for history |

## 7. RULES THIS FILE EXISTS TO ENFORCE

1. **Check section 1 before proposing an experiment.** Grep PROVENANCE for the
   *subject*, not the item id you have in mind.
2. **Quote the interval, never the bare percentage.** §3 is why.
3. **One variable per arm.** Two changes and one number answers neither.
4. **Same judge, same seed, AND SAME SAMPLER, or the comparison is not a
   comparison.** The sampler clause was added 2026-08-23 after the "+9.4 pts"
   direction-rule effect turned out to be a flat-sampler control subtracted from
   per-turn treatments — same judge, same seed, still meaningless.
   [TRAPS #46](../../docs/TRAPS.md), fifth instance.
5. **A number that matters goes into PROVENANCE the same session.** The folder
   it was written to is gitignored.
6. **Update this file when an experiment lands.** One line. If it takes more
   than a line it belongs in PROVENANCE and this gets the pointer.
