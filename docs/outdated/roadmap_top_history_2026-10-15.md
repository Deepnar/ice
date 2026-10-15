# ROADMAP top-of-file status, moved 2026-10-15

Verbatim copy of the dated phase/status notes that sat between "HOW TO EXECUTE" and
"THE SPLIT" in `docs/ROADMAP.md` (2026-08-10 through 2026-10-15), replaced there by a
single current-position block. Historical record only; never a work source.
Relative links below were written for `docs/ROADMAP.md` and resolve from `docs/`.

---

### Active v3 repair phase — 2026-09-12

The whole-system repair is authorized. Implement existing defects and the
[reviewed design](reviews/2026-09-12-v3-system-review.md) with focused validation
as changes land; broad benchmarks, recorded-response replay comparisons and
multi-source LME/LSREP dataset work follow the repairs. Earlier per-change
production gates and pre-reseed sequencing are superseded within this scope;
privacy, frozen v2 history and the public-push freeze remain. No new G-number
for the phase. [Execution contract](specs/V3_REPAIR.md).

**Current phase, 2026-10-15:** bounded read tuning is closed — no survivor beat the
±3/67 answer-noise floor, so settings are frozen unchanged ([results](../experiments/v3_repair/results/tuning-freeze-2026-10-15.md)).
The development set was near ceiling and showed no answer benefit from ICE's extra
legs over its vector leg. Next: build **Z2** per the approved
[Z2_EVALUATION](specs/Z2_EVALUATION.md) — speed profile first, then the cited
question machinery (dry run on the development history), read switches, replay,
background audit, read passes, oracle. Z2 is not FINAL. The 2026-10-07 text below
is history.

**Current phase, 2026-10-07:** r4 completed all five stages diagnostically.
[PROVENANCE](PROVENANCE.md#2026-10-07--v3-completed-r4-development-campaign)
records the actual results and their limits. The next work is
[V3_POST_RESEED_REPAIR](specs/V3_POST_RESEED_REPAIR.md): eleven grouped core
repair packages → bounded hyperparameter tuning → freeze → Z2 semi-LSREP
with complete lifecycle and additive/leave-one-out ablations plus LongMemEval
oracle. Both ICE warm-vector and independent bare-vector controls are required.
No full LME-S haystack campaign or product expansion is added. The old merged-Z
phrasing below describes the development history; this is the current execution
order. The source corpus, completed answers and39 as-of stores are preserved.

**v3 repair implementation boundary,2026-10-09:** all11 grouped packages have
named mechanism validation or a bounded keep decision in the
[disposition audit](specs/V3_POST_RESEED_REPAIR.md#51-repair-disposition-audit--implementation-boundary-2026-10-09).
Next prepare versioned repaired development state/keys and the live-consumer,
resumable tuning instrument. Broad semantic/answer/context/resource qualification
and the final runner still earn readiness. This does not close29 quality owners;
formal160/93checked/67open partition remains unchanged. No final/tuning score.

**v3 fixture decision and offline preparation,2026-10-09:** final semi-LSREP
uses original paper histories B and C separately (1119+325 pairs), with the same
complete ablations/lifecycle/control roster. Both are excluded from new tuning;
the other two development histories provide352 pairs/73 reviewed keys. The
[offline planner](../experiments/v3_repair/README.md) freezes originals, scorer
keys, unqualified legacy references, archive coverage and static knob candidates.
It does not seed, tune, qualify final sources, schedule aging or launch Z2.

**v3 remaining-work audit,2026-10-10:** the29 evidence-dependent owners are
partitioned by next work:7 prerequisite audits/scoring/instruments,16 repaired
development/tuning qualifications,4 frozen contribution comparisons and2
conditional/after-baseline decisions. Thus23 owners have relevant unfinished
prerequisite/development work before final launch, alongside the final labels,
clock/baseline/ablation/dual-judge/recovery readiness. This is not23 new defects
or automatic parent closure. Wider original scopes and35 later items remain.
The [item-by-item readiness ledger](specs/V3_POST_RESEED_REPAIR.md#52-remaining-work-by-next-evidence-stage--rechecked2026-10-10)
owns the stage explanation; complete those shared prerequisites before tuning
promotions/freeze. Selecting B/C is preparation, not a passed readiness gate.

**Focused v3 lifecycle repair, 2026-10-09:** ordinary cold-only recall was
blocked by the older dated-only contract. Restored 0.12 probation sources were
also blocked by the 0.2 warm floor. Archived/probation originals now remain
bounded warm candidates, retaining decay ranking. Bounded cold candidates compete
in normal and wide-net retrieval without a recording date; exposure remains
separate from source corroboration. G30 — reader quality, e.g. an old recovery
phrase must reach the final prompt — has focused SQL/final-preparation controls,
not a closed quality claim. The bounded ten-job/cold-read/resource pass now
completes, preserving42 originals; broader semantic and worst-case capacity
qualification remain. [The repair decision](specs/V3_POST_RESEED_REPAIR.md#21o-undated-cold-recall--grounded-decision-2026-10-09)
records the deliberate scope change. Roadmap owner counts are unchanged.

**Default-rate v3 quiet/cold qualification,2026-10-11:** two authored recorded
pairs and an explicitly simulated67-day quiet query exercise3216 actual calls
of the three decay jobs. Both originals remain intact and retrievable cold;
completed resume is byte-stable, with zero failed/degraded workers. This is a
core retention mechanism check, not all-ten-job semantic quality, >40-turn
retention, cloud answer gains or an independent vector comparison. The native
source→claim→decay→query check separately exposes a stale cluster restriction:
preserved cold membership still fails to admit sentence evidence. G76 —
attributed evidence, e.g. retaining a source's sandbox-only qualification — owns
the [membership repair](specs/V3_POST_RESEED_REPAIR.md#cold-sentence-cluster-membership--grounded2026-10-11).
No whole owner closes; formal counts and tuning/final gates remain unchanged.

**Final judging, reaffirmed 2026-10-09:** Muse Spark1.3 Contributor remains
primary. A separately selected second family independently grades every final
answer, with per-judge scores and disagreements reported separately. G65 —
judge reliability, e.g. the same saved answer changes grade when its comparator
changes — checks the current scoring protocol against references, not a new
model tournament. Two agreeing judges are not independent ground truth. The
remaining human batches stay parked; independent memory repairs continue.

**USER-REQUIRED for this phase:** the independent answer-judge packet is ready.
Review batch1 now; batches2–6 are parked for late evaluation qualification
(about45–90 minutes for all60 answers; source evidence supplied). Ten labels
give an initial diagnostic reference, not full judge qualification. Independent
core repairs and experiment preparation continue meanwhile. Inspect the frozen
arm/cost/readiness summary and manually launch the
future documented tuning/Z2 command when ready. The new runner does not exist
yet; the old r4 command is not a tuning or Z2 launcher. The agent continues
independent repairs while review is pending. SESSION is retained until the
maintainer explicitly requests clearing it.

**Current count, rechecked 2026-10-10:** 160 anchored top-level entries, 93 checked and
67 unchecked. The 58/51 counts in the dated 2026-08-10 position below are
historical, not the current queue size. Fourteen open Track F entries are
product work, four Track H entries are longer-term research, and Z1/Z2/Z3 are
gates; the other open entries mix active core work, parked work and partial
repairs. See the entries themselves for ownership; this count is not a list
of 73 independent memory defects.

### 📍 CURRENT POSITION — 2026-08-10

**⚑ Z1 AND Z2 HAVE MERGED IN PRACTICE, AND THAT IS CORRECT (user, 2026-08-12).**
Z1's instruments turned out to measure Z2's questions: seeding the tuning corpus
exercises every background job, so extraction quality, summary faithfulness and
codex correctness were all measured *during* Z1-prep — the very things Z2's
entry lists as "the places the automated suite provably cannot see". The user
predicted this before it happened. ⇒ **treat them as one phase**: measure the
whole pipeline on real data, read the output every time, cross-check often, and
do not rush a sweep to completion. The eyeball pass is not a later stage; it is
what caught every finding a metric missed on 2026-08-12.

**⚑ [Z1](#z1) HAS STARTED — declared by the user 2026-08-10. Z1-prep is not a
separate phase; it is Z1's first work.** [G9](#g9) · [G36](#g36) · [G37](#g37) done,
[G30](#g30)'s real-reader quality fixture now done and its actual-model/answer
coverage inside combined Z, [G28](#g28)'s style variants inside that coverage
matrix as always.

**v3 pre-tuning checkpoint, 2026-09-29:** the independent core-repair and
instrument work has reached the combined Z1/Z2 entry. This is a queue boundary,
not a claim that ICE now beats vector or that all memory-quality defects are
solved. Of **160 anchored entries, 93 checked / 67 open**: the earlier 10
"pre-Z candidates" are now placed by their actual remaining work. Nine belong
inside reseeding, configuration/VRAM tuning, or the combined Z coverage/answer
review: [A9](#a9) (current NER activation on representative queries),
[G4](#g4) (numerical GPU budget), [G27](#g27) (background default),
[G28](#g28) (style variants), [G29](#g29) (deferred drift questions),
[G30](#g30) (actual-model/answer coverage), [G48](#g48) (leg value),
[G55](#g55) (typed labels/complete seed), and [G76](#g76) (graph precision,
source use and answer gain). [G32](#g32)'s remaining native background
transport is conditional later work under its revised spec: schema output and
idle model release already work through the compatible path. The resulting
open partition is **29 Z/reseed/tuning-dependent, 35 later/product/research,
and 3 Z gates**, with no separately demonstrated pre-Z **core memory** code
blocker. A 2026-09-29 harness audit found immediate *evaluation-infrastructure*
work before the complete seed: the older seeder selects only 293 historical
turns and skips preflight; its former snapshot omitted claim/note tables.
The replacement runner and 35-table snapshot now pass short checks, including
source-clock/job/observer/row-identity validation on 2026-10-01. These belong to
the existing [Z1](#z1)/[G55](#g55) instrument and complete-seed work, not a
new open item; [audit and corrected order](reviews/2026-09-29-v3-reseed-harness-audit.md).
That audit also maps each of the 29 seed/tuning-dependent entries to the
observation needed for a decision: 12 primarily seed/write-side, 16 combined-Z
selection/answer-side, and T5 after the frozen Z baseline. These are not 29
guaranteed checkmarks from one run.
**Manual campaign/GT gate, 2026-10-01:** the [manual entry point and measurement
matrix](reviews/2026-10-01-v3-manual-campaign.md) keep replay, snapshot, cloud
answers, both-order judges and reporting in one resumable private bundle. The
maintainer runs the full campaign manually; this session does not launch it.
The instrument belongs to **Z1 — live memory coverage and tuning**, for example
preserving turn100's store before tuning retrieval at that cutoff, and
**G55 — meaningful source/task scoring**, for example refusing to count a gold
source ID as a faithful compressed answer. All 255 questions have catalog
answers; 124 originally had screened source candidates and 131 needed mappings,
now all 255 have reviewed dispositions; no label is
admitted until complete-source/correction/recent-history review and explicit
task/knowledge scope. Replay completion does not establish graph truth,
summary faithfulness or answer use. Broader real-pair judge qualification and
output-specific reviews remain prerequisites to scores of record. Current
schema-only cloning preserves memory-table indexes without personal rows;
the failed historical fresh-migration check is installation work, not a new
core-memory repair item or a measured quality failure. The 29 partition remains
unchanged. Freeze reviewed development labels before cloud scoring; then use
seeded output reviews, targeted tuning and interaction checks before the two
agreed final conditions.
A Z failure can and should reopen a concrete repair. After combined Z, the agreed
end-stage runs are LME oracle and semi-LSREP only, not a full LME-S campaign.
**V3 source/repeat checkpoint, 2026-10-03 (final r3):** all 255 base
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
Natural repeated facts
can share a reviewed family while keeping occurrence/cutoff identity. The
authorized small development pass adds two reviewed families/four occurrences
(259 prompts with base255), with per-arm before/after accuracy/token/source
comparisons and recent controls separate from old-memory totals. The full
repeated retention schedule stays in final semi-LSREP. No new roadmap entry or
quality checkmark follows from these instrument/source reviews.

**⚑ THE PUSH FREEZE IS ON.** Per CLAUDE.md's experiment-phase rule: commit
locally, **do not push to the public remote** until the user lifts it. This is
not a documentation detail — a half-run experiment published mid-flight is a
result nobody chose to publish.

**⚠ THE PROBE BUDGET IS THE BINDING CONSTRAINT, AND IT IS SMALLER THAN IT LOOKS
(measured 2026-08-10).** `experiments/curation_files/` holds **58 files / 681 raw
probes**, but probes repeat across a conversation's three checkpoints: **259
unique by text, over 19 conversations.** *(FINAL's spec §3b records 649/252 —
drifted, re-count before use.)* That is the entire hand-written evidence base,
and **[FINAL](specs/FINAL_experiments.md) §3b already claims it** ("reusable, do
NOT ask the user to rewrite them"). The contamination risk is real and worth
naming: tuning and reporting on the same 259 questions would make FINAL's
numbers a measurement of the tuning, and it is invisible afterwards.

**✅ RESOLVED BY USER DECISION 2026-08-10 — Z1 MAY tune against these probes.**
The user will **regenerate FINAL's probes** (covering the classes the current
set never measured), so FINAL does not inherit the set Z1 tuned on and the two
corpora are separated by construction rather than by abstinence. The sealed-set
rule this block previously stated is **superseded** — do not re-impose it.
What survives is the *principle*: whatever FINAL finally reports on must not be
what Z1 swept. See [G30](#g30) for the instrument, and [G40](#g40) for the
`probe_type` gap the regeneration should close while it is there.

- **Inventory: 58 open items** *(51 + G42–G47, opened 2026-08-12 from the whole-pipeline measurement)*, all of them top-level `- [ ]` lines that you can
  count. The split into PRE-FINAL research work and POST-FINAL productization is
  below and is the triage that matters — the raw count reads as far more
  remaining work than actually gates the experiments.
  - ⚠ **The count moved 52 → 51, and NOT because work was finished.** The
    2026-08-10 cleanup found **three items that were counted but never written**:
    **[G34](#g34)** and **[G35](#g35)** (opened 2026-08-08 as two summary lines in a
    preamble note — G34's measurements were even filed in PROVENANCE) and
    **[G27](#g27)** (announced 2026-07-25 as "new G27 (shared-mode bg model bug)",
    with a decision-complete description sitting in
    [specs/G_mechanical.md](specs/G_mechanical.md) the whole time). A session told to
    do any of the three would have found nothing. All three now have entries, and the
    counted set and the written set agree — which is the number to keep true.
- **Finished items live in [ROADMAP_DONE.md](ROADMAP_DONE.md)** — 70 of them,
  verbatim, each reachable from a one-line stub here that keeps its anchor. The
  completion notes are where look-ahead, propagation and validation are
  recorded, so they are kept in full and never compressed.
- **Alembic head `d5c81a37e9b2`** — unchanged since G11.
- **Where the last session left off, and what it was told to do:
  [HANDOFF.md](HANDOFF.md)**, rewritten every session. Read it before this file.

> **📕 The dated session notes that used to live here are in
> [outdated/roadmap_session_log.md](outdated/roadmap_session_log.md)** (archived
> 2026-08-10): seventeen "DOCS ARE IN SYNC" / "ADDENDUM" paragraphs and four
> stacked inventories, ~7,100 words, none of it queue. They accumulated because
> there was nowhere else to put a session note; [HANDOFF.md](HANDOFF.md) owns
> that now. Nothing was deleted — read them for historical phrasing only.

> ---
>
