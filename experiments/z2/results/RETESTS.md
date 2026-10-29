# Z2 retests — what must run before any fix or finding becomes a number (v3)

Every fix from [FINDINGS.md](FINDINGS.md) that changed behaviour, and every switch built for one,
is **unvalidated until the retests below say otherwise**. Run from the fix-pass worktree
(`cd ~/Programs/ice-worktrees/fix-pass`). **GPU** items wait until Z2's S12 is done; **CLOUD**
items wait for the Go allowance (weekly reset ~Nov 2). Long runs are started by the maintainer.

**⚑ Grading rule (maintainer, 2026-10-29):** Z2 is **not** re-graded. Its grading finishes under
grade-1 on `main` (R6) so B, C and S12 stay complete and comparable; the new judge (grade-2) is
measured on a 250-answer sample graded under both rubrics (R5). Corrected Z2 comparisons use the
per-point rescoring of grade-1's own verdicts — if R5 shows that rescoring agrees with grade-2.
Retests that compare settings (R3, R4, R7) grade both sides with grade-2. FINAL uses grade-2 only.

## What happens, in order (written 2026-10-29)

**Why no Z2 re-run is needed.** Every fix and switch so far acts at read time (what reaches the
prompt) or at grading. The expensive part of Z2 — replaying B and C turn by turn with the
background model (~11 h and ~14 h of GPU) — builds the stores, and none of the fixes changes what
is stored. So each retest re-reads the same saved stores for the same questions, answers and
grades only those, and compares settings question by question (paired, so a few points show).
A change to what gets *stored* (e.g. not writing some summaries) would need a new replay; none is
built.

1. **S12 finishes** (~3 h from 2026-10-29 afternoon). Nothing else needs the GPU until then.
2. **GPU day (~10–12 GPU hours):** R1 database tests + the full disposable rails; R8 fix-pass tests
   and the three classifier variants (B7); R2 candidate sweep on B (coverage + seconds per read →
   picks the candidate setting); the reads of R3 and R4 on B and C; R9 live trials.
3. **Cloud from the Go reset (~Nov 2), three phases, one after another, never mixed (maintainer,
   2026-10-29):**
   - **Phase 1 — Z2's own cloud run** (R6): `main`'s Z2 runner as it stopped — grade-1 grading of B's
     remaining phases, C, and S12's answers and grades. Run to the end.
   - **Phase 2 — Z3 judging** (`experiments/z2/z3_judging.py`, its own runner; it waits while any Z2
     run is active): R5 judge panel and judge checks on B and C, the R3/R4 retest answers and their
     grade-2 verdicts, and anything Z2 or the GPU reads turn up (R7 is added once R2–R4 choose).
   - **Phase 3 — cloud prep** (`experiments/final/cloud_prep.sh`): the synthetic conversations and
     A's private ground truth, then A's review.
   Budget at measured prices: phase 1 ~$11.5, phase 2 ~$9–10 (answers ~$6.3, verdicts ~$1.5,
   judge checks ~$1), phase 3 ~$9 — ~$30, the month's whole remainder; phase 3 may spill onto Go
   Plus, which FINAL's cloud phase needs anyway. Phase 3 last does not delay FINAL: its local
   phase starts after the Z3 freeze.
4. **R7 (~Nov 8–10):** the chosen settings together on B and C — the fix pass's number. Needs a few
   dollars beyond Go's month if the budget is spent; else waits for Go Plus.
5. **Z3 freeze (~Nov 10):** set the switch defaults from R2–R4/R7, D9 simplification, merge fix-pass
   into `main` (only after R6), record the freeze commit.
5½. **Research briefing (~1 week, maintainer 2026-10-29) — gate before FINAL's local run:** a for external review (ICE v3 architecture, Z2→Z3 findings, the FINAL plan; a 4–6 page PDF, appendices if useful; task in the gitignored docs/PUBLISHING.md [PRIVATE:research-briefing]). Made from the frozen configuration and the retest results, sent for critical feedback while the
   design can still change; feedback goes into the pre-registration as a revision. No code, config or runs change.
6. **FINAL local (~Nov 17 → ~Nov 28, after the briefing):** 2 lanes, ~9–11 GPU days.
7. **FINAL cloud (Go Plus, ~$92):** answering can start per history as its local reads finish
   (~Nov 22 onward); at Go Plus's weekly cap (~$60) it takes two weeks or more → ~early December.

**The GPU day is one command** (`experiments/z2/gpu_day.py`, Z2's runner — meter, resume, lock):
gates (DB tests, full rails) → R8 (G60, G75, B7's four trainings; run even if a gate fails) → R2
sweep (7 settings on B, one stage each for timing) → coverage report → R3/R4 reads on B and C →
R9 LoCoMo slice. A failed gate holds every read and the slice; it waits by itself while S12 uses
the GPU. LongMemEval-S full-context and BEAM trials are not in it (short, run by hand after).

```bash
cd ~/Programs/ice-worktrees/fix-pass
uv run --no-sync python experiments/z2/gpu_day.py run
uv run --no-sync python experiments/z2/gpu_day.py status --watch
```

**Decisions wait for C (maintainer, 2026-10-29).** The GPU day measures; it decides nothing. C's
grade-1 results (R6) can arrive during the cloud week and may disagree with B — on the summary
policy, on fold notes, on any part. Every setting is chosen at R7 from B **and** C together (code
checks plus grade-2 answers), after R5; where C disagrees with B, that part's default waits or is
re-read, never set from B alone.

| | What | Needs | Status |
|---|---|---|---|
| R1 | F1 fix — database tests of the source check | GPU/DB | ☐ |
| R2 | F2 — how many candidates to rank, and keep the rest | GPU | ✅ measured (pick pending) |
| R3 | F10 — conversation order for "which came first" | GPU + CLOUD | ☐ |
| R4 | F6/F13/F14 — turn summaries in reads: off, or a margin of 1–2 | GPU + CLOUD | ☐ |
| R5 | F12 — judge the judge: 250 answers under both rubrics, planted panel, invariance | CLOUD (~$0.6) | ☐ |
| R6 | Z2's grade-1 grading finishes on `main` (B's remaining phases, C, S12) | CLOUD (~$11) | ☐ |
| R7 | All fixes together on B and C, answered and graded | GPU + CLOUD | ☐ |
| R8 | fix-pass ⏳ items (G79, G60, G75, B7 classifier variants) | GPU/DB | ☐ |
| R9 | Live trials stopped earlier (LoCoMo slice, LongMemEval-S full context) | GPU | ☐ |
| R10 | F4 — B2 "no memory" on a third of LongMemEval | after S12 graded | ☐ |
| R11 | F18 — clusters dropped from FINAL's parts test (audit A12); decay tested on S4 + S9 | design | ✅ |

**Cloud budget (Go, ~$30 usable Nov 2–27; measured per-call prices):** Z2 grade-1 finishing — B's
remaining claim checks and diagnostics ~$4, C ~$4, S12 answers + grades ~$3.5 — ~$11.5; finish cloud
prep ~$9; R5 ~$0.6; retest answers + grade-2 grades (6 settings × 174) ~$5.7 — **~$27 in all**, ~$3
slack. Order: R6 (Z2's cloud run) → Z3 judging (R5, retest answers and grades) → cloud prep. FINAL's own cloud
(~$92) and any private-question top-up need Go Plus.

## R1 — F1 fix, database tests (GPU/DB, ~20 min)

```bash
uv run --no-sync python tests/support/disposable_database.py -m pytest tests/smoke/test_source_need_route.py tests/smoke/test_temporal_axis_path.py -q
uv run --no-sync python tests/support/disposable_database.py -m pytest tests/smoke -q
```
Pass: all green; `positive` route test shows memory retrieved with no source-judge call.
Then R7 shows the effect on answers (expected from Z2's `loo_gate`: +47 points on B's 18 skipped).

## R2 — candidates ranked, keep the rest (GPU, reads only, ~1–2 h)

Each read of B's 174 questions under one setting; full ICE as it runs on fix-pass (F1 fixed), so
`retest_cand64` is the new baseline. Time each condition separately (rank time per read is the
cost of a larger pool).

```bash
for c in retest_cand64 retest_cand128 retest_cand256 retest_cand512 retest_cand64_keep retest_cand128_keep retest_cand256_batch16; do
  /usr/bin/env time -f "$c %e s" uv run --no-sync python experiments/z2/read_passes.py --history B --database ice_z2_main_b \
    --replay logs/z2-run-main/s7-B --asks logs/z2-questions-final/B/z2-asks.json --out logs/z2-retest-B --conditions $c
done
uv run --no-sync python experiments/z2/retest_coverage.py --history B --units logs/z2-retest-B/units \
  --reference logs/z2-run-main/s9-B/units --occurrences logs/z2-questions-final/B/occurrences.json
```
Measures: cited-turn coverage (Z2 reference: full ICE 71.3%, plain search 87.5%), pieces, memory
tokens, seconds per read. **Pick** the smallest setting within 1 point of the best coverage whose
read time is at most ~3 s above `retest_cand64`; batch 16 is kept if it is faster at equal
coverage. Coverage is presence, not quality — the chosen setting goes on to R7 for answers.

**R2 RESULT (GPU day, 2026-10-29; B, 159 questions with cited turns; fix-pass with F1 fixed):**
cited-turn coverage / seconds per read — 64 candidates 81.1% / 7.3 s; 128: 85.5% / 10.7 s; **256: 90.8% /
15.2 s**; 512: 90.8% / 15.2 s (the pool is ~250); 64 + keep the rest 82.2% / 7.4 s (fills the budget,
+1 point); 128 + keep 85.9%; 256 with batch 16 90.8% / 22.1 s (slower, rejected). Reference: plain vector
search 87.5%, Z2's ICE as run 71.3% (the F1 fix accounts for 71→81). 256 is the first setting above plain
search, at +7.9 s per message — over this plan's own +3 s rule, so the pick is the maintainer's (128 is the
middle ground); answers decide at R7.

## R3 — conversation order (GPU read + CLOUD answers/grade)

```bash
uv run --no-sync python experiments/z2/read_passes.py --history B --database ice_z2_main_b \
  --replay logs/z2-run-main/s7-B --asks logs/z2-questions-final/B/z2-asks.json --out logs/z2-retest-B \
  --conditions retest_cand64,retest_conversation_order
uv run --no-sync python experiments/z2/answers.py --units logs/z2-retest-B --out logs/z2-retest-B-answers --answerer muse12
```
Then grade both settings with fix-pass's `experiments/z2/grade.py` (grade-2) into a new `--out`,
with the same arguments as Z2's B grading (`--units logs/z2-retest-B --answers logs/z2-retest-B-answers`). Pass: temporal questions improve (ICE reversed 7
of 21) and no other type falls by more than its interval. Same for C when C is graded.

## R4 — turn summaries in reads (GPU read + CLOUD)

Z2 already shows (grade-1, all-points): adding summaries −5.8 (±6.0), summaries off in full ICE
+4.6 (±5.5); where ICE was partial and plain search correct, ICE carried a median 59% of the cited
turns' words (96% when both correct). Three settings against `retest_cand64` (preset `retest_summaries`): `retest_no_summaries` (turn
summaries off in reads; stored ones untouched), `retest_original_margin1` and `…margin2` (F14: a
summary replaces the turn's text only when it out-scores it by 1 or 2 logits). Plus F17's two: `retest_policy_text` (the reranker
keeps choose_representation's pick) and `retest_originals_first` (that, with the summary-first
intents off — summaries only as the budget fallback; the allowance B's evidence supports). Answer
and grade as R3; run `retest_coverage.py`, and `summary_allowance.py` on the retest units to see
the policy land. Decide with R7 on B and C.
**A v3 change that removes something v2 shipped is a maintainer decision** ("memory is earned").

**R3/R4 CODE CHECKS (GPU day; B reads done, C running):** conversation order — the same pieces, now
100% in turn order (0% before), coverage unchanged (81.1%). Summary share of single-turn pieces in the
prompt: shipped 50%, margin 1 44%, margin 2 37%, policy text 26%, originals first 10%, summaries off 0%;
coverage 79.5–81.1% throughout (summaries change detail, not which turns). Memory tokens rise as
originals replace summaries (20.7k → 28.6k). Answers and grade-2 decide (Z3 judging).

## R5 — judge the judge (CLOUD, ~$0.6)

Light, but every part runs: the largest single loss Z2 found was the judge (F12).
1. **250 answers graded under grade-2**, already graded under grade-1 — about 60% from the strata
   F12 found mis-graded and 40% controls, spread across ICE and plain-search settings, B and C:
   - issue strata (~150): grade-1 *partial* with every required point covered (60), grade-1
     *incorrect* with every point covered (40), downgrades whose reason cites dates/timestamps (30),
     C answers of the same kinds once C's grade-1 exists (20);
   - controls (~100): grade-1 *correct* (35), *incorrect* with points missing (35), *partial* with
     points missing (30) — grade-2 must not inflate these.
   Reported per stratum: how many verdicts change, and **how often grade-2 agrees with the
   per-point rescoring of grade-1** (the number that decides whether Z2's rescored comparisons
   stand in for a re-grade). The answers that change go into a blind sheet (answer, REQUIRED,
   SOURCES, both verdicts; condition hidden) for the maintainer to read; a change the maintainer
   rejects is a rubric defect, fixed before FINAL grades anything.
2. **Planted panel** — `experiments/z2/qualify.py` under grade-2 beside its grade-1 results: planted
   wrong stays wrong, planted correct stays correct.
3. **Invariance** — 20 answers graded as given and with one true extra detail from the
   conversation appended: the grade must not change.
4. FINAL's cross-family judge sample (pre-registration §4, ~450 answers) stays as planned.

Script to build before the cloud returns: `experiments/z2/judge_check.py` (strata draw from the
saved grade-1 verdicts, grade-2 calls through `grade.judge_caller`, report + blind sheet).

## R6 — Z2's grade-1 grading finishes (CLOUD, ~$11.5)

`main`'s Z2 cloud lane, restarted as it stopped (B's claim checks, ranking, controls, context
precision; C; then S12's answers and grades) — grade-1, unchanged. **It must finish before
fix-pass merges into `main`**, which brings grade-2.

## R7 — everything together (GPU + CLOUD)

Full ICE with F1 fixed + R2's chosen candidate setting + R3/R4 if they passed, read, answered and
graded under grade-2 on B and C, against plain search and long context. This is the number the
fix pass reports; FINAL's design freeze waits for it.

## R8 — fix-pass ⏳ items (GPU/DB)

As listed in the ROADMAP's FIX-PASS checklist: G79 (replay writes batch notes), G60
(`logs/fix-pass/reconcile_eval.py`, then `tests/test_codex_facts.py`), G75
(`logs/fix-pass/claim_check_eval.py`), **B7** classifier variants — shipped, standardize,
layernorm, kaiming (`uv run --no-sync python /home/deepnar/Programs/ice/logs/fix-pass/b7_variants.py`;
keep a variant only if it beats shipped on both F1 and the style flip rate).

## R9 — live trials (GPU)

A LoCoMo replay slice (sessions ending with an empty assistant side — never exercised), one
LongMemEval-S full-context read, short BEAM / LongMemEval-S / baseline trials.

## R10 — F4 (after S12 is graded)

Of LongMemEval's questions where B2 said no memory (97 of 298), check whether the answer was
already in the recent window (oracle histories are short) before calling any of it a miss.
