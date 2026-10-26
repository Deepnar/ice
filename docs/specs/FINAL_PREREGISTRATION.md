# FINAL — pre-registration, revision 2 (APPROVED by the maintainer, 2026-10-26)

**Status: revision 2 approved 2026-10-26** (revision 1 used the judge's 1–5 score as the headline;
replaced before anything ran — 1–5 saturates: most answers in every condition score 4–5).
Amended before anything ran, 2026-10-27: models (§1, Z3 D13) and paper 1's measures (§4).
Fixed before FINAL runs; nothing here changes after the first FINAL answer is graded. A change
after that is a new revision, reported beside the first (Z3 D11 item 3).
Roster and machinery: [FINAL_experiments.md](FINAL_experiments.md) §2.1, [Z3_DECISIONS.md](Z3_DECISIONS.md).

## 1. The system and the data are frozen
- ICE at the commit recorded at freeze (after Z2's fix pass and the D9 simplification pass).
- Datasets, each reported **separately, never pooled**: private B (sealed families; labelled
  development), A, D (labelled development), LAP, SW; synthetic S1–S9; BEAM 128K/500K/1M
  (40 conversations, MANIFEST seed 20261026); LongMemEval-S (100 instances, final_slice.json);
  LoCoMo (10 conversations, extra table).
- Answerer Muse Spark 1.2, judge Muse Spark 1.3 (Z2 D6d; Z3 D13) — or the closest available
  models at freeze if either is retired, recorded here before the first answer. The shared
  family is measured, not assumed away: a cross-family judge re-grades ~450 answers (§4) and the
  blind human sample is reported per condition; Muse writes none of the ground truth.
  No tuning on any FINAL data.
- The long-context window is fixed at freeze from the chosen answerer's window and the
  measured cost per call, and named in every table ("long context, N K window").

## 2. The claim, and the three headline measures

**The claim.** ICE gives the answering model a small, curated working context so it behaves as if
it had the whole history: answers stay **correct, current and honest** as the conversation grows
far past what any window holds — at a fraction of the tokens of stuffing the window.

Three headline measures, each covering what the others miss. The judge (Muse, fixed prompt,
published) gives one verdict per answer: correct / partial / incorrect, plus whether the answer
used an outdated value, made an unsupported claim, or abstained.

**M1 — Accuracy (comparable with the field).** An answer counts when the judge calls it
*correct* (partial counts as wrong; the judge's *uncertain* is never credited and its rate is
reported), as LongMemEval, LoCoMo-J, Zep and Mem0 report it; per dataset
and question type. Benchmarks also keep their own native score beside it: BEAM's nugget mean
(Kendall tau-b for ordering), LongMemEval's upstream yes/no, LoCoMo's judge. *Misses:* whether a
correct answer also carried an invented or outdated claim, and how hard the question was.

**M2 — Supported current accuracy against distance (the paper's measure; LSREP- and
ICE-native).** An answer counts only if it is **correct as of the moment it was asked** (LSREP's
evolving answer key), uses **no outdated value**, makes **no invented claim** (an unsupported
claim the claim chain finds in neither the ledger nor the originals), and **declines** when the
thing was never said. Reported as a curve over **how far the evidence lies from the question** —
the history's tokens between the farthest turn holding the *current* evidence and the question (a
superseded value's turn is not evidence): < 32K · 32K–128K · 128K–512K · > 512K — and split
**in-window / out-of-window**: exactly, from the turns `long_context`'s prompt held; where no
`long_context` read exists, distance > 37,500 tokens (its prompt budget). The split is a property
of the question, applied to every condition. *This replaces Z2's 1–5 scoring for FINAL.*
*Misses:* cost. Implementation: `experiments/z2/scoring.py`.

**M3 — Cost per correct answer.** Prompt tokens per answer and per M1-correct answer, beside
write-time cost per turn (replay wall seconds per turn; model calls per turn are not in the replay
trace and are reported as not measured). A system can buy accuracy by
filling the window; M3 shows the price. *Misses:* quality — it is read beside M1/M2, never alone.

## 3. Primary tests (decided before any result)
- **P1 (the thesis):** on **out-of-window** questions, M2 of `full` vs `long_context`, and M2 of
  `full` vs `bare_vector` (budget-matched retrieval).
- **P2:** M1 of `full` vs the same two, on all questions.
- Per dataset, never pooled. Each question is a paired yes/no outcome across conditions:
  McNemar's exact test and a bootstrap interval over question families (10,000 resamples, seed 0);
  Holm–Bonferroni across a dataset's primary tests. A win = interval excludes 0 in ICE's favour
  after correction; a loss is reported as plainly.
- **Detectable difference:** each dataset states, before running, the smallest difference its
  question count can detect (≈ 6–8 points at ~300 paired questions); smaller differences are
  reported as "no detectable difference", never as a tie.
- **Hallucination non-inferiority:** ICE's rates of (i) unsupported claims, (ii) confident answers
  to never-said questions, (iii) outdated values, (iv) attribution mix-ups are not worse than
  `bare_vector`'s by more than 2 points (upper interval bound).

## 4. Comparisons and diagnostics (reported, no win/loss claim beyond §3)
- **Upper bound — full history:** on BEAM's 128K tier (conversations ~100K–225K tokens) a cloud
  model with a **≥ 256K window** answers from the **entire** conversation; ICE's own contexts are
  answered by the **same cloud model**, so ICE vs full history compares memory, not answerers.
- **Other memory systems,** run by us in the same setup (same answerer, judge, data):
  Graphiti (Zep), Mem0, A-Mem — on the LongMemEval-S slice; also LoCoMo (below).
- **LoCoMo** as a small, clearly labelled extra table (its known answer-key errors and lenient
  common judge stated); it is the common ground of most memory papers.
- **Evidence recall (turn-cited):** is every current cited turn a prompt fragment's source or
  inside the recent window? Separates "memory did not find it" from "the model misread it".
  Literal by design — a summary or fact carrying the content without citing every turn is not
  credited here (TRAPS #32); context precision, read by the judge, credits it.
- **Paper 1's measures, kept only where they add something (2026-10-27):**
  *update accuracy* — M2 on the questions whose answer changed, asked after the change
  (LSREP's evolving key: did the system follow the update?); *context precision* — the share
  of prompt fragments that carry a required item, and the share that carry a wrong or outdated
  one, for `full` and `bare_vector` (paper 1's SPF, made independent of fragment length: it
  separates "memory served stale evidence" from "the model chose the old value"); M3 is
  reported as **TUR** — correct answers per 1,000 prompt tokens (paper 1's ratio, now on a
  yes/no outcome, so it is a real rate); the **fidelity audit** on B and S9 — per component:
  did it run, is its output faithful (background-content audit), does it reach the prompt,
  does switching it off lower M2 → *defective / unexercised / contributing / no measurable
  effect*; the judge's 1–5 score and per-type tables as descriptive columns.
  *Dropped:* the multi-answer ranking (win rate) — M1/M2 already compare conditions question
  by question, and a preference ranking is where a same-family judge is most exposed;
  recency Δ (the distance curve replaces it); ordinal net superiority (McNemar replaces it).
  Human grades never enter an aggregate (paper 1 merged 72 of them); they are agreement only.
- **Retention (S4)** by distance; **style invariance** S1 vs S8; **ablations** — leave-one-out on
  S9 and sealed B, build-up on S9 (a group earns its place when switching it off lowers M2 on the
  questions planned for it, interval excluding 0); **run-to-run variance** (S1 replayed twice);
  **answerer sensitivity** — the cloud-answerer subset: 25 asks from each of S9, B, S5 under
  `full`, `recent_only`, `bare_vector` (225 contexts) answered by a non-Meta model and graded the
  same way: does ICE keep its margin?; the **cross-family judge check** — GLM-5.3-Flash gives
  verdicts on those 225 contexts' Muse 1.2 answers and on the subset's answers (~450 calls); the
  family effect is Muse 1.3's leniency against GLM on Muse 1.2's answers minus its leniency on the
  non-Meta answers, with a 95% interval over question families (reported, with agreement and
  kappa on each side); **judge vs the maintainer** (exact, kappa).

## 5. Handling rules
- An answer cut at its output cap is **ungraded**, never wrong; the ungraded rate is reported per
  condition, and a condition with >5% ungraded is flagged.
- Questions a history's review dropped stay dropped. Controls (no-memory questions) are reported
  apart.
- If time runs short, the drop order is fixed in advance: LoCoMo; A-Mem; BEAM 1M 10 → 5
  conversations; LongMemEval-S 100 → 50; the S9 build-up; BEAM conditions 6 → 4. Never dropped: the
  primary tests, the hallucination measures, the full-history upper bound, the private and
  synthetic main runs.

## 6. Left open by design (approved as such)
- The cross-family judge (GLM-5.3-Flash as of 2026-10-27, Z3 D13) qualifies on the planted
  controls before use; if it fails, another non-Meta family replaces it, recorded here.
- The cloud-answerer subset's model, picked at the time from a non-Meta family (default
  LongCat-2.0); recorded here before use.
- The cloud model for the full-history upper bound (≥ 256K window), picked at the time after the
  per-call cost check; recorded here before it answers.
