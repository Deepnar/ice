# Z2 retests — what must run before any fix or finding becomes a number (v3)

Every fix from [FINDINGS.md](FINDINGS.md) that changed behaviour, and every switch built for one,
is **unvalidated until the retests below say otherwise**. Run from the fix-pass worktree
(`cd ~/Programs/ice-worktrees/fix-pass`). **GPU** items wait until Z2's S12 is done; **CLOUD**
items wait for the Go allowance (weekly reset ~Nov 2). Long runs are started by the maintainer.

**⚑ One rule for every number below:** after the judge change (R5), **all of Z2's grades are
redone under grade-2** (R6) — B, C and S12 — and every comparison (ICE vs plain search, each
fix vs without it) is read on the grade-2 numbers. grade-1 numbers in FINDINGS stay as the
record of what the first judge said; they are never mixed with grade-2 ones.

| | What | Needs | Status |
|---|---|---|---|
| R1 | F1 fix — database tests of the source check | GPU/DB | ☐ |
| R2 | F2 — how many candidates to rank, and keep the rest | GPU | ☐ |
| R3 | F10 — conversation order for "which came first" | GPU + CLOUD | ☐ |
| R4 | F6/F13/F14 — turn summaries in reads: off, or a margin of 1–2 | GPU + CLOUD | ☐ |
| R5 | F12 — judge the judge: planted panel, paired B and C samples, invariance | CLOUD (~$0.5) | ☐ |
| R6 | F12 — full re-grade of Z2 under grade-2 | CLOUD (~$12) | ☐ |
| R7 | All fixes together on B and C, answered and graded | GPU + CLOUD | ☐ |
| R8 | fix-pass ⏳ items (G79, G60, G75, B7 classifier variants) | GPU/DB | ☐ |
| R9 | Live trials stopped earlier (LoCoMo slice, LongMemEval-S full context) | GPU | ☐ |
| R10 | F4 — B2 "no memory" on a third of LongMemEval | after S12 graded | ☐ |

**Cloud budget (Go, ~$30 usable Nov 2–27; measured per-call prices):** re-grade B $9.8 (or $7.8
without ranking/controls/semantic diagnostics), grade C $4.1, S12 answers + grades $3.5, finish
cloud prep ~$9, R5 $0.5, retest answers + grades (6 settings × 174) $5.7 — **~$32.6 in all**. To fit:
order R5 → R6 B/C (diagnostics last) → S12 → cloud prep → retests; drop the diagnostics if needed
(−$3), or Extra Usage for the last ~$3. FINAL's own cloud (~$92) needs Go Plus regardless.

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

## R3 — conversation order (GPU read + CLOUD answers/grade)

```bash
uv run --no-sync python experiments/z2/read_passes.py --history B --database ice_z2_main_b \
  --replay logs/z2-run-main/s7-B --asks logs/z2-questions-final/B/z2-asks.json --out logs/z2-retest-B \
  --conditions retest_cand64,retest_conversation_order
uv run --no-sync python experiments/z2/answers.py --units logs/z2-retest-B --out logs/z2-retest-B-answers --answerer muse12
```
Then grade-2 (R6's command on these folders). Pass: temporal questions improve (ICE reversed 7
of 21) and no other type falls by more than its interval. Same for C when C is graded.

## R4 — turn summaries in reads (GPU read + CLOUD)

Z2 already shows (grade-1, all-points): adding summaries −5.8 (±6.0), summaries off in full ICE
+4.6 (±5.5); where ICE was partial and plain search correct, ICE carried a median 59% of the cited
turns' words (96% when both correct). Three settings against `retest_cand64` (preset `retest_summaries`): `retest_no_summaries` (turn
summaries off in reads; stored ones untouched), `retest_original_margin1` and `…margin2` (F14: a
summary replaces the turn's text only when it out-scores it by 1 or 2 logits). Answer and grade
as R3; also run `retest_coverage.py` and the verbatim-share check from F14. Decide with R7 on B and C.
**A v3 change that removes something v2 shipped is a maintainer decision** ("memory is earned").

## R5 — judge the judge (CLOUD, ~$0.5) — before R6 spends anything

Light, but every part runs, because the largest single loss Z2 found was the judge (F12):
1. **Planted panel** — `experiments/z2/qualify.py` (planted correct / wrong / partial / abstain
   answers) under grade-2, beside its grade-1 results. Pass: planted wrong stays wrong (no
   inflation), planted correct stays correct.
2. **Paired B sample** — 150 B answers, stratified: grade-1 partial with every point covered,
   grade-1 incorrect with every point covered, grade-1 correct, grade-1 incorrect with points
   missing. Grade under grade-2; count flips per stratum. The ~40 that flip go into a blind
   sheet (answer, REQUIRED, SOURCES, both verdicts, condition hidden) for the maintainer to read:
   a flip the maintainer disagrees with is a rubric defect, fixed before R6.
3. **Paired C sample** — 150 C answers graded under BOTH rubrics (C was never graded under
   grade-1, so this is how much the judge change moves C).
4. **Invariance** — 20 answers graded twice: as given, and with one true extra detail from the
   conversation appended. Pass: the grade does not change (the defect F12 found).
5. The cross-family judge sample (pre-registration §4, ~450 answers) stays as planned for FINAL.

## R6 — full re-grade under grade-2 (CLOUD, ~$12)

```bash
uv run --no-sync python experiments/z2/grade.py --history B --units logs/z2-run-main/s9-B \
  --answers logs/z2-run-main/s10-B --questions logs/z2-questions-final \
  --ledger logs/z2-ledger-detail-2026-10-18-r2/B/ledger.json --changes logs/z2-questions-final/B/changes.json \
  --audit logs/z2-run-main/s8-B/bg-audit.json --replay logs/z2-run-main/s7-B/progress.jsonl \
  --out logs/z2-regrade-B
```
Same for C and S12 (their run_z2 grade commands with a new `--out`). Rough cost at measured
prices: B ~4,300 grade + ~3,100 claim calls ≈ $8; C and S12 ≈ $4. The claim check (unsupported
claims verified against the whole conversation) completes the clean yes/no score: correct under
grade-2 **and** no claim confirmed invented (M2).

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
