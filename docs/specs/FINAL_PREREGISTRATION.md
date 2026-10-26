# FINAL — pre-registration (DRAFT, 2026-10-26, for the maintainer's approval)

**Status: draft.** Fixed before FINAL runs; nothing here changes after the first FINAL answer is
graded. A change after that is a new revision, reported beside the first (Z3 D11 item 3).
Roster and machinery: [FINAL_experiments.md](FINAL_experiments.md) §2.1, [Z3_DECISIONS.md](Z3_DECISIONS.md).

## 1. The system and the data are frozen
- ICE at the commit recorded at freeze (after Z2's fix pass and the D9 simplification pass).
- Datasets, each reported **separately, never pooled**: private B (sealed families; labelled
  development), A, D (labelled development), LAP, SW; synthetic S1–S9; BEAM 128K/500K/1M
  (40 conversations, MANIFEST seed 20261026); LongMemEval-S (100 instances, final_slice.json).
- Answerer local qwen3.8:27b, thinking on (Z2 D6c). Judge Muse 1.3; a second judge if chosen
  before grading (D7). No tuning on any FINAL data.

## 2. Primary question and metric
**Does full ICE answer better than simple ways of giving a model memory, at the same prompt
budget?**
- **Metric:** the judge's 1–5 score per answered question.
- **Primary comparisons (per dataset):** `full` vs `bare_vector` (budget-matched retrieval)
  and `full` vs `long_context` (the newest history packed into the same prompt ceiling).
- **Statistic:** mean paired difference by question family, 95% bootstrap interval over
  families (10,000 resamples, seed 0).
- **A win** in a dataset: the interval excludes 0 in ICE's favour after Holm–Bonferroni across
  that dataset's two primary comparisons. A loss is reported as plainly as a win.
- BEAM uses its own metric (nugget mean, Kendall tau-b for ordering) in the same way;
  LongMemEval-S its upstream yes/no accuracy beside the 1–5 grade.

## 3. Hallucination and honesty — registered measures
Per dataset and condition, reported for every condition:
1. **Invented claims:** share of answers containing a claim the conversation never supports
   (Z2's claim chain: `confirmed_invented`).
2. **Never-said questions:** share answered as "not mentioned" (abstention accuracy on the
   negatives); a confident answer to a never-said question counts as a hallucination.
3. **Outdated values:** share of answers that use a superseded value as current.
4. **Attribution mix-ups:** score on attribution questions (an assistant suggestion reported as
   the user's decision, a mused option reported as decided).

**Registered expectation:** ICE is **not worse** than `bare_vector` on 1–3 — non-inferiority
margin 2 percentage points (the upper interval bound of ICE minus baseline below +2 pp).

## 4. Secondary measures (reported, no win/loss claim)
- Score per question type; per-1,000-prompt-token score (quality at budget).
- **Retention (S4):** score against distance since the fact was said (100 / 300 / 600 / 1,000 turns).
- **Style invariance:** S1 vs S8 (same plan, different writing) — the paired difference and its
  interval; a difference is reported as sensitivity to writing style.
- **Ablations:** leave-one-out per component group on S9 and sealed B; the additive build-up on S9
  (D10, decided yes) in Z2's frozen order; a group "earns its place" when switching it off lowers the mean score with an
  interval excluding 0 on the questions planned for it.
- Memory-system baseline (D11) on LongMemEval-S; the cloud-answerer subset; run-to-run variance
  (S1 replayed twice); cost and speed (prompt tokens, answer seconds, write time per turn).
- Judge validity: agreement with the maintainer's blind sample (exact, Cohen's kappa).

## 5. Handling rules
- An answer cut at its output cap is **ungraded**, never wrong; the ungraded rate is reported per
  condition, and a condition with >5% ungraded is flagged.
- Questions a history's review dropped stay dropped. Controls (no-memory questions) are reported
  apart.
- If time runs short, the drop order is fixed in advance: BEAM 1M 10 → 5 conversations;
  LongMemEval-S 100 → 50; the S9 build-up; BEAM conditions 6 → 4. Never dropped: the primary
  comparisons, the hallucination measures, the private and synthetic main runs.

## 6. Open before approval
- Second judge (D7, benched); which memory system (D11, decided at the time).
