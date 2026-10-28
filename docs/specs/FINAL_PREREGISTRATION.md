# FINAL — pre-registration, revision 2 (APPROVED by the maintainer, 2026-10-26)

**Status: revision 2 approved 2026-10-26** (revision 1 used the judge's 1–5 score as the headline;
replaced before anything ran — 1–5 saturates: most answers in every condition score 4–5).
Amended before anything ran, 2026-10-27: models (§1, Z3 D13) and paper 1's measures (§4).
Fixed before FINAL runs; nothing here changes after the first FINAL answer is graded. A change
after that is a new revision, reported beside the first (Z3 D11 item 3).
Roster and machinery: [FINAL_experiments.md](FINAL_experiments.md) §2.1, [Z3_DECISIONS.md](Z3_DECISIONS.md).

**Revision 3 — PROPOSED 2026-10-29 from Z2's evidence, before any FINAL answer; maintainer to
approve** ([Z2 FINDINGS](../../experiments/z2/results/FINDINGS.md) F12, [RETESTS](../../experiments/z2/results/RETESTS.md)):
- **Judge rubric grade-2.** M1's definition is unchanged — the judge's *correct* — but the
  rubric now defines it: every REQUIRED item present with SOURCES' values and no superseded value
  as current; details beyond REQUIRED never lower the grade (they go to the claim check, which
  feeds M2). SOURCES carry each passage's date. Z2 showed grade-1 marking 65% of partials and
  206 "incorrect" answers down with every required point covered — over half of ICE's gap to
  plain search on B.
- **A judge check gates FINAL's grading** (RETESTS R5): planted panel, paired samples, invariance
  to a true extra detail; a rubric defect found there is fixed before anything is graded.
- **The freeze (§1) waits for RETESTS R7** — the fix pass measured on B and C under grade-2 —
  and records the defaults of the switches built from Z2's findings (`retrieval_rerank_candidates`
  / `retrieval_rerank_keep_unranked`, `retrieval_evidence_order`, `retrieval_rerank_original_margin`).

## 1. The system and the data are frozen
- ICE at the commit recorded at freeze (after Z2's fix pass and the D9 simplification pass).
- Datasets (how they are reported: §3, revised 2026-10-28): LSREP private — A, SW, LAP (held out)
  and B — stated plainly: Z2 used B to test the system and find bugs, and those findings shaped
  fix-pass changes, but no hyper-parameter sweep ran on it, and its FINAL questions are sealed
  families never shown or used; D, the v3 hyper-parameter tuning history, is dropped from FINAL
  (maintainer, 2026-10-28); LSREP synthetic S1–S9; public benchmarks, each
  in its own scoring — BEAM 128K tier (20) + 500K tier (5 of the 10 drawn; the 1M tier dropped
  2026-10-28; MANIFEST seed 20261026, trim seed 20261028), LongMemEval-S (100 instances,
  final_slice.json), LoCoMo (10 conversations, extra table).
- Answerer Muse Spark 1.2, judge Muse Spark 1.3 (Z2 D6d; Z3 D13) — or the closest available
  models at freeze if either is retired, recorded here before the first answer. The shared
  family is measured, not assumed away: a cross-family judge re-grades ~450 answers (§4) and the
  blind human sample is reported per condition; Muse writes none of the ground truth.
  No tuning on any FINAL data.
- **What runs where (maintainer, 2026-10-28) — our data may combine anything; each public
  benchmark runs the way its own field runs it, in its own scoring, never mixed with ours (BEAM,
  built for long conversations, is the one place long context and full history meet ICE):**

  | data | conditions answered | scored by |
  |---|---|---|
  | **our data** — synthetic S1–S9, private B, A, SW, LAP | ICE frozen as it runs (`full_default`, every part on, decay live); plain vector search (`bare_vector`, ICE's own token budget); long context (`long_context`, the newest **256K** tokens — a window many providers offer, a quarter of the answerer's 1M — read **in time order**: older history, then the recent turns, then the question; every question at a checkpoint starts at the same turn, so the provider serves that start from its cache (2026-10-28)); recent turns only (the no-memory floor) | the judge's verdict (§2) |
  | **S9 only** — the parts test | ICE with decay neutral (`full`), each part switched off one at a time, parts added one at a time | the judge's verdict, on the questions built for each part |
  | **BEAM** — 128K tier (20) + 500K tier (5) | ICE, plain vector search, long context (256K); full history on the 500K tier (§4) | BEAM's own nugget score |
  | **LongMemEval-S** — 100 instances | ICE, plain vector search, full context (every haystack session, ~115K tokens), LongMemEval's own oracle setting (only the answer sessions) | LongMemEval's own judge (its prompts, yes/no) — nothing else |
  | **LoCoMo** — 10 conversations | ICE, ICE at a 2K-token memory budget, plain vector search, full context (each conversation fits 256K whole), Graphiti, Mem0, A-Mem | Mem0's protocol, as most memory papers report it: J (LLM judge, its prompt verbatim), F1, BLEU-1; categories 1–4 (category 5 excluded) |

  Warm vector (ICE's own vector search alone) appears only inside S9's parts test. On the public
  benchmarks `recent_only` is read — it is the prompt frame plain vector search and long context
  are built in — but never answered.

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
and question type. The **public benchmarks are headlined in their own scoring** — BEAM's nugget
mean (Kendall tau-b for ordering), LongMemEval's own judge, LoCoMo by Mem0's protocol (J, F1, BLEU-1) — because each
was built to show its own point of view, and memory papers report them that way (§1's table;
2026-10-28: nothing of ours is added to them). ICE's own claim (M2 by distance, P1) is tested on
LSREP's suites. *Misses:* whether a
correct answer also carried an invented or outdated claim, and how hard the question was.

**M2 — Supported current accuracy against distance (the paper's measure; LSREP- and
ICE-native).** An answer counts only if it is **correct as of the moment it was asked** (LSREP's
evolving answer key), uses **no outdated value**, makes **no invented claim** (an unsupported
claim the claim chain finds in neither the ledger nor the originals), and **declines** when the
thing was never said. Reported as a curve over **how far the evidence lies from the question** —
the history's tokens between the farthest turn holding the *current* evidence and the question (a
superseded value's turn is not evidence): < 32K · 32K–128K · 128K–512K · > 512K — and split
**in-window / out-of-window**: exactly, from the turns `long_context`'s prompt held; where no
`long_context` read exists, distance > 256,000 tokens (its prompt budget). The split is a property
of the question, applied to every condition. *This replaces Z2's 1–5 scoring for FINAL.*
*Misses:* cost. Implementation: `experiments/z2/scoring.py`.

**M3 — Cost per correct answer.** Prompt tokens per answer and per M1-correct answer, beside
write-time cost per turn (replay wall seconds per turn; model calls per turn are not in the replay
trace and are reported as not measured). A system can buy accuracy by
filling the window; M3 shows the price. *Misses:* quality — it is read beside M1/M2, never alone.

**The named measures (2026-10-28)** — the names the paper uses, so each can be cited; two keep paper
1's names with new, sounder definitions (paper 1's were ratios of a 1–5 score):

| name | what it counts | lineage |
|---|---|---|
| **ACC** — accuracy | share of answers the judge calls fully correct (= M1) | the field's standard |
| **SCA** — supported current accuracy | correct, no outdated value, nothing invented, declines when never said (= M2) | new: LSREP's headline |
| **SCA-BW** — SCA beyond the window | SCA on questions whose evidence lies beyond the long-context window (P1) | new |
| **SCA@d** — SCA by evidence distance | SCA in bins of < 32K · 32K–128K · 128K–512K · > 512K tokens back | replaces paper 1's Recency Δ |
| **UA** — update accuracy | SCA on questions whose answer changed | new: LSREP's evolving key |
| **TUR** — token utility ratio | correct answers per 1,000 prompt tokens (M3) | paper 1's TUR, now on a yes/no outcome |
| **SPF** — supporting-fragment precision | share of prompt fragments that carry a needed fact | paper 1's SPF, made length-free |
| **SFR** — stale-fragment rate | share of prompt fragments that carry a wrong or outdated one | new |
| **ER** — evidence recall | share of questions whose cited evidence reached the prompt | new |
| **ICR · NSR · OVR · AER** — the hallucination profile | invented-claim rate · never-said answer rate (confident answers about things never said) · outdated-value rate · attribution-error rate (who said what) | paper 1's Hallucination %, split in four |
| **MFA** — mechanism fidelity audit | per part: did it run, is its output faithful, does it reach a prompt, does switching it off lower SCA (S9) | paper 1's fidelity audit |

Dropped from paper 1: Win rate (the ranking), Recency Δ (SCA@d replaces it), the 1–5 score as a
headline (kept as a descriptive column).

## 3. Primary tests (decided before any result)
- **The tests (revised 2026-10-28):** ICE as it runs (`full_default`) against `long_context` and
  `bare_vector` (budget-matched retrieval): **M1 on all questions** and **M2 on all questions** —
  with a 256K window the long-context model often reads the whole conversation, so the question
  is whether ICE matches it, stays current (that model sees every old value beside the new one)
  and invents nothing, at a fraction of the tokens (M3 beside) — and **P1, M2 on out-of-window
  questions**, tested only where a suite has at least 100 paired questions whose evidence lies
  beyond 256K (B, the longest synthetic chats); otherwise reported as too few, with the count.
- **Per suite, pooled (revised 2026-10-28):** the headline is each suite — **LSREP synthetic
  carries the claims** (S1–S9; S8, S1's style twin, counted once through S1; plans, questions,
  answer keys and conversations released, so anyone can inspect them); **LSREP private is the
  real-world check** (A, SW, LAP and B, with B's role disclosed as above; not releasable — it
  supports the claims, never carries one alone) — with its histories' questions pooled
  (`scoring.pool`; families resampled within their history). Each question is a paired yes/no
  outcome across conditions: McNemar's exact test and a bootstrap interval over question families
  (10,000 resamples, seed 0); Holm–Bonferroni across a suite's primary tests. A win = interval
  excludes 0 in ICE's favour after correction; a loss is reported as plainly. **No per-conversation
  verdicts:** each conversation is then a deep dive under the aspect it was built to test (S4
  retention by distance, S8 style invariance, S9 ablations, the laptop decision thread, …).
- **Detectable difference:** each dataset states, before running, the smallest difference its
  question count can detect (≈ 6–8 points at ~300 paired questions); smaller differences are
  reported as "no detectable difference", never as a tie. Stated 2026-10-27 (McNemar, 80% power,
  5% two-sided, 20% of pairs assumed to disagree; `scoring.detectable`), in points:

  | suite (the verdicts) | questions | all (P2) | Holm-worst (4 tests) | out-of-window, if 60% (P1) |
  |---|---|---|---|---|
  | **LSREP private (A, SW, LAP, B)** | ≈381 | 6.4 | 7.6 | 8.2 |
  | **LSREP synthetic (S1–S9, S8 once)** | 1,044 | 3.9 | 4.6 | 5.0 |
  | *per conversation (deep dives; no verdict)* | | | | |
  | B (sealed: 64 families; seal not opened, ≈2.3 asks each) | ≈147 | 10.2 | 12.2 | 13.2 |
  | A (planned) | ≈131 | 10.8 | 12.9 | 14.0 |
  | SW / LAP (planned) | ≈58 / ≈45 | 16.1 / 18.2 | 19.1 / >20 | >20 / >20 |
  | synthetic S1–S9 (S8 is S1's twin, not counted) | 1,044 | 3.9 | 4.6 | 5.0 |
  | BEAM (128K 400 · 500K 100; own scoring) | 500 | 5.6 | 6.7 | 7.2 |
  | LongMemEval-S | 100 | 12.4 | 14.7 | 15.9 |
  | LoCoMo, categories 1–4 (no out-of-window questions) | 1,540 | 3.2 | 3.8 | — |

  Recomputed from the real counts (and the real out-of-window share) before the first answer;
  private counts follow their question chain. (The 2026-10-27 proposal — no verdict above 15
  points — is moot: verdicts are per suite, and every suite detects ≤ 8.2 points.)
- **Hallucination non-inferiority:** ICE's rates of (i) unsupported claims, (ii) confident answers
  to never-said questions, (iii) outdated values, (iv) attribution mix-ups are not worse than
  `bare_vector`'s by more than 2 points (upper interval bound).

## 4. Comparisons and diagnostics (reported, no win/loss claim beyond §3)
- **Full history vs long context vs ICE (revised 2026-10-28):** on BEAM's 500K tier (5
  conversations, ~320K–857K tokens) the answerer itself, Muse 1.2, answers from the **entire**
  conversation within its own 1,048,576-token window; ICE and long context (256K) answer the same
  questions with the same model, so the three compare memory, not answerers, in BEAM's own
  scoring. On the 128K tier (≤ 225K tokens) long context already holds the whole conversation, so
  there it *is* full history. 100 questions: reported, no verdict. Full history is laid out like long
  context, in time order from the first turn, so a conversation's questions share one cached start.
- **Other memory systems,** run by us in the same setup (same answerer, judge, data):
  Graphiti (Zep), Mem0, A-Mem — **on LoCoMo only**, the benchmark all three report on (maintainer,
  2026-10-28). Each ingests through its own write path with FINAL's background model (gemma4:e4b)
  at temperature 0 and its own documented embedder, is searched with the raw question, and its top
  10 results sit in exactly bare_vector's evidence position. LoCoMo's two speakers reach each
  system as its own LoCoMo evaluation fed them: Graphiti and A-Mem read every message with its
  speaker's name; Mem0 keeps one memory per speaker, that speaker as its "user", and searches both
  (each memory labelled with whose it is). A-Mem is the memory layer its paper evaluated (pinned in
  `experiments/final/baselines/AMEM.json`) — its library package skips note construction; its
  fallback rate (a failed model call becoming an empty note analysis) is reported.
- **LoCoMo** as a small, clearly labelled extra table (its known answer-key errors — ~6% — and
  lenient common judge stated; every conversation fits a 256K window, so it cannot show
  out-of-window behaviour), conditions and scoring as in §1 — Mem0's protocol (J, F1, BLEU-1) on
  categories 1–4, category 5 excluded, as memory papers do; J is judged by FINAL's model, so it is
  labelled non-official. **Each system's prompt size and its share of the conversation are
  shown beside its score** — a short conversation can be "remembered" by reading most of it —
  and ICE also answers at a 2K-token memory budget, the size the memory systems hand the model.
  LoCoMo's first speaker is always ICE's user side and the second its assistant side (124 of its
  272 sessions open with the second speaker; pairing by position would swap them). Such an
  opening line rides the user side with the first speaker's reply, names kept: ICE embeds a turn's
  user half, and an empty one could not be found (156 questions cite such a line). It is the
  common ground of most memory papers.
- **Evidence recall (turn-cited):** is every current cited turn a prompt fragment's source or
  inside the recent window? Separates "memory did not find it" from "the model misread it".
  Literal by design — a summary or fact carrying the content without citing every turn is not
  credited here (TRAPS #32); context precision, read by the judge, credits it.
- **Paper 1's measures, kept only where they add something (2026-10-27):**
  *update accuracy* — M2 on the questions whose answer changed, asked after the change
  (LSREP's evolving key: did the system follow the update?); *context precision* — the share
  of prompt fragments that carry a required item, and the share that carry a wrong or outdated
  one, for ICE and `bare_vector` (paper 1's SPF, made independent of fragment length: it
  separates "memory served stale evidence" from "the model chose the old value"); M3 is
  reported as **TUR** — correct answers per 1,000 prompt tokens (paper 1's ratio, now on a
  yes/no outcome, so it is a real rate); the **fidelity audit** — per component: did it run, is
  its output faithful (background-content audit, on B and S9), does it reach the prompt, and (S9
  only) does switching it off lower M2 → *defective / unexercised / contributing / no measurable
  effect*; the judge's 1–5 score and per-type tables as descriptive columns.
  *Dropped:* the multi-answer ranking (win rate) — M1/M2 already compare conditions question
  by question, and a preference ranking is where a same-family judge is most exposed;
  recency Δ (the distance curve replaces it); ordinal net superiority (McNemar replaces it).
  Human grades never enter an aggregate (paper 1 merged 72 of them); they are agreement only.
- **Retention (S4)** by distance; **style invariance** S1 vs S8; **the parts test** — on S9 only
  (B's run dropped 2026-10-28): each part switched off one at a time, parts added one at a time (a
  group earns its place when switching it off lowers M2 on the
  questions planned for it, interval excluding 0); **run-to-run variance** (S1 replayed twice);
  **answerer sensitivity** — the cloud-answerer subset: 25 asks from each of S9, B, S5 under ICE,
  plain vector search and long context (225 contexts) answered by a non-Meta model and graded the
  same way: does ICE keep its margin?; the **cross-family judge check** — a fresh non-Meta model picked at the time — local or a cheap cloud model — run as the last pass — gives
  verdicts on those 225 contexts' Muse 1.2 answers and on the subset's answers (~450 calls); the
  family effect is Muse 1.3's leniency against that judge on Muse 1.2's answers minus its leniency
  on the non-Meta answers, with a 95% interval over question families (reported, with agreement and
  kappa on each side); **judge vs the maintainer** (exact, kappa).

## 5. Handling rules
- An answer cut at its output cap is **ungraded**, never wrong; the ungraded rate is reported per
  condition, and a condition with >5% ungraded is flagged.
- Questions a history's review dropped stay dropped. Controls (no-memory questions) are reported
  apart.
- If time runs short, the drop order is fixed in advance: LoCoMo; A-Mem; LongMemEval-S 100 → 50; the S9 build-up. Never dropped: the
  primary tests, the hallucination measures, the full-history upper bound, the private and
  synthetic main runs.

## 6. Left open by design (approved as such)
- The cross-family judge: a fresh non-Meta model picked at the time — a local model or a cheap
  cloud one — qualified on the planted controls before use, run as the last pass; recorded here
  (and in `CROSS_JUDGE`, run_final.py, whose stages wait until it is set).
- The cloud-answerer subset's model, picked at the time from a non-Meta family (default
  LongCat-2.0); recorded here before use.
- ~~The full-history model, picked at the time~~ — **recorded 2026-10-28:** Muse 1.2 itself
  (models.dev: a 1,048,576-token window, $0.10/M input, $0.002/M cached — list price matched the
  measured allowance), within its own window (`UPPER_BOUND` in experiments/final/run_final.py); a
  prompt that would not fit is reported, never cut. The long-context window is fixed: 256K (§1).
