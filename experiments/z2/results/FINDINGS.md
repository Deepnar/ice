# Z2 findings — what the run showed about the system (v3)

**Version: v3** (`main` as Z2 runs it). None of these numbers are comparable with the
paper's v2 numbers. Every finding names its evidence, its size with an uncertainty, and
its status: **TO FIX** (cause measured, fix not yet chosen), **OPEN** (seen, cause or size
not settled), **FIXED** (on `fix-pass`, see the ROADMAP's FIX-PASS checklist), **NO FIX**
(measured and left, with the reason).

Scoring used below: each answer graded once by the judge (Muse Spark 1.3), correct = 1,
partial = 0.5, wrong = 0; differences are **paired** (same questions, two settings), with a
95% interval. Evidence lives under the gitignored `logs/` (B is a private history).

## F1 ⚑ FIXED on fix-pass (2026-10-29, fix (a); DB tests pending) — the source check cancelled memory the main decision asked for

**What happens.** After ICE's main retrieval decision (B2, `src/api/memory_decision.py`)
says "retrieve", the source refinement (`src/api/memory_preparation.py` →
`source_need.judge_source_need` → `source_action`) asks `gemma4:e4b` whether the current
message already holds everything needed. On a `visible_evidence` verdict whose quote is a
substring of the message (and passes source proof), the retrieval is **skipped** and the
prompt gets no memory at all. It can only ever remove memory a positive decision granted.
The code settles the mechanism: the positive path called `prove_source_fact(message, message)`
— the current message proved against itself, so a question naming what it asks about (X and Y)
proves itself "supported". **Fix (a):** a positive is never put to the source judge (no model
call on those turns either); the negative rescue is unchanged.

**B, Z2 questions (174).** Skipped on **18**; every one needs older memory (12 are
"which came first, X or Y" questions, which name X and Y — the likely quote). On those 18
the main decision was ~0.998 sure memory was needed. Plain vector search minus full ICE on
them: **+69 points (±16)**; ICE lost 16, tied 2, won 0. The same 17–18 are skipped in every
ICE setting that has the check (`full`, `full_b*`, `warm_vector`, `add_7_gate`), and adding
the gate group costs **−4.3 (±2.9)**, changing 19 of 208 prompts (worse 11, better 1).

**Why the headline comparison looked bad.** Full ICE vs plain vector search on B is +11.8
(±6.5) for plain search over all questions — but **+69 on the 18 skipped and +5.2 (±6.2,
not significant) on the other 155**. On those 156, ICE is *ahead* at fixed budgets: plain −
ICE −9.3 (±6.2) at 2k tokens, −5.4 (±6.8) at 4k, −8.7 (±6.8) at 8k.

**Not only a benchmark effect.** In B's turn-by-turn replay of the real conversation (S7)
the skip fired on **175 of 1,119 real turns (16%)**; C's replay 47 skips. A sample read:
most are new details or reactions where the main decision wanted context, and some are
explicit requests to look something up in earlier material. C's questions: 3 of 75
skipped (plus 1 small-talk control, the intended use). LongMemEval oracle (S12): 0 of 298.

**History.** Development qualification (PROVENANCE 2026-09-27) measured a related arm
falsely suppressing an existing positive on 7 of 22 unrelated blocks, "including
question-as-evidence quotes", and kept it off; the shipped arm was enabled the same day
on a 24-question development set (22/24 vs 17/24). Z2 is the first measurement at scale.

**Benefit side, measured.** Skips that were right: 1 control in B, 1 in C. Skips that were
wrong on questions: 18 in B, 3 in C.

**Fix candidates (maintainer decides; `src/` is gated):** (a) never let the check cancel a
positive B2 decision — keep only its rescue role for negatives (`source_action` returns
`keep` whenever `base_retrieve`); (b) switch the whole check off
(`memory_source_gate_enabled=False`); (c) keep it but refuse quotes that come from the
question itself. (a) removes every measured harm and keeps the only path that can *add*
memory. Confirm the quote mechanism first on the GPU: re-ask `gemma4:e4b` on the 18.

## F2 SWITCH BUILT (`retrieval_rerank_keep_unranked`, candidates up to 512; off) — extra memory budget helps plain search, not ICE — retest [R2](RETESTS.md)

At its own budget (median ~29k tokens) ICE gains nothing over its 8k selection (−1.6,
±4.3) while plain search gains **+12.3 (±6.1)**. Cited-turn coverage (questions not
skipped, 141): ICE 76% at 8k → 80% at its own budget; plain search 79% → 86%.

**Causes measured.** (1) ICE chooses from **64 ranked candidates**
(`retrieval_rerank_candidates = 64`) and uses 54–57 of them at its own budget — the extra
space has almost nothing left to draw from; past 8k it added a new cited turn on 7 of 141
questions. (2) ICE's pieces are whole exchanges (~370 tokens); plain search packs single
messages (~110), so it covers ~5× more distinct turns (231 vs 46 at the median). The answer
difference on these questions is within noise, so this is the smaller problem.

**Next.** GPU: re-read B's questions with 128 and 256 candidates and compare cited-turn
coverage and read time — no answering or grading needed to see whether coverage moves.

**Measured from the read logs (B, 5,088 rerank calls):** ICE's legs find a median **250**
candidates (quartiles 195–269); the cap keeps 64 and **drops the rest outright** — a median 186,
on 100% of reads — before the token budget sees them. Reranking costs 39 ms per scored text
(median 3.3 s per read for ~93 texts), so reranking all 250 would add ~10 s per turn. The
reranker also picks each piece's version (original / shorter / abstract) by the highest score,
which may be what shortens copies (F6). **Proposal (maintainer's, 2026-10-29):** rerank the top
64 as now, keep the rest behind them in fusion order instead of dropping them, and let the token
budget be the only bound — no extra rerank cost. Test on the GPU: B's questions re-read, cited-
turn coverage and read time compared.

## F3 OPEN — three subsystems change nothing on B

Prompts compared directly (208 per setting): adding clusters changed **0**; adding the
temporal leg changed **2**; leaving out the maintenance agent changed **0**. Their grades
are identical to the setting before them. "Inert on B" is the right wording until C and
FINAL's histories are checked — Z3 D9 (post-Z2 simplification) is where it is decided.

## F4 OPEN — the main decision says "no memory" on a third of LongMemEval

S12 (LongMemEval oracle, partial): B2 said no retrieval on **97 of 298** questions; the
rescue path added memory on 14, withheld it on 32, and 51 kept "no memory". In the oracle
setting the whole evidence may already sit in the recent window, so some of these may be
right — decided when S12 is graded.

## F5 OPEN — where ICE still misses (questions it retrieved for)

Of the 56 B questions plain search won: 25 had no piece from the cited turns (16 of them
the F1 skip, 9 retrieved but missed), 11 had only facts/summaries/notes of those turns,
19 had the original words present. After F1, the 9 misses and 11 processed-only cases are
the next look; the 19 suggest answer-side noise rather than retrieval.

## F1 addendum — the rescue half of the same check, and how much the skip explains

**Rescues** (B2 said no, the check proved older memory was needed and admitted one verified
original): B and C — **0**, in the question reads and in both replays of the real
conversations (1,570 turns). LongMemEval (S12, partial): 31 in the log; of the 15 in saved
units, the reference answer reached a prompt that otherwise had no memory on **8**; held back
on 15 where plain search had the answer. The rescue does real work, only on LongMemEval —
so the fix that keeps it (candidate (a)) is preferred over switching the check off.

**Measured counterfactual, no rerun:** `loo_gate` is full ICE with this check off and memory
always retrieved — on B's questions the main decision already said retrieve every time, so it
*is* ICE without the skip. On the 18: +47 (±12.5) over ICE as run, of the 67-point gap; plain
search still leads there by +19 (±16) — mostly ordering questions ICE now answers partly. On
the other 155 it changes nothing (−0.3 ±0.6). Overall (partial = half) it recovers +4.6 of
plain search's +11.3.

## F6 OPEN — ICE's copy of the right turn is often shortened — retest [R4](RETESTS.md)

Where ICE (without the skip) was partial and plain search correct (22 questions), ICE's prompt
still drew on 94% of the cited turns, but carried a **median 59% of those turns' original
words** (bottom quarter ~3%: a summary or abstract stood in for the original); where both were
correct, 96%. The missed required points' 3-word phrases are in ICE's prompt at 16% vs plain
search's 40% (control: points ICE covered, 28% vs 28%) — only 9 testable points, so a lead,
not proof. The budget had room for the originals (F7). Proof needs one small targeted read:
these 22 questions with originals preferred over summaries.

## F7 — the budget is not too tight; ICE cannot use it

Same per-question budget for ICE and plain search (median ~29k memory tokens). ICE fills ~95%
of it but chooses from 64 candidates (F2), packs whole exchanges, and on F6's questions sends
shortened copies. Earlier smaller tests read this as "budget too tight"; at Z2's scale the
limiter is what ICE puts in the budget, not its size.

## F8 ⚑ MEASUREMENT — "partial" mostly means "all required points, plus extra details"

The judge chooses correct / partial / incorrect; FINAL's M1 counts partial as wrong. On B,
**1,199 of 1,852 partial answers (65%) covered every required point**; 1,196 of them were
flagged for "unsupported claims" (median 8 each) — details beyond the cited turns, which the
judge cannot see. The rubric never defines the correct/partial line. ICE's answers carry more
such claims (5.4 per answer vs plain search's 4.3), 8–11% of them its own date/order labels
echoed by the answerer (plain search 1%). So judge-correct penalises ICE's richer prompts.

Re-scored from the judge's own per-point coverage (correct = every required point covered,
no outdated value; no new calls): ICE as run 24.7% → 52.3%, plain search 40.5% → 60.7%.
Plain search minus ICE: as run +8.7 (±7.5); **without the skip +4.6 (±7.3, no difference)**;
at 8k −2.3 (±7.8). Judge-correct gives +16.2 / +11.0 / +5.7 for the same pairs.

**Not yet counted:** whether those extra claims are true. The claim-check phase (phase 2,
paused at the cloud cap) checks each against the whole conversation; with it, "all required
points + no confirmed-invented claim" is the clean binary score. **Decision before FINAL
(pre-registration revision, maintainer):** define M1 from per-point coverage plus the claim
check, or tighten the rubric's partial definition, or keep judge-correct and report this.

## F9 — where ICE is better, and where it is clearly worse (B, all-points rule)

| | ICE without the skip | plain search |
|---|---|---|
| correct (all required points, no outdated value) | 56.1% | 60.7% (+4.6 ±7.3, no difference) |
| prompt tokens per answer (median) | **26,723** | 35,816 (+34%) |
| correct per 10k prompt tokens | **0.21** | 0.19 |
| outdated value presented as current | **2.3%** (update questions 23.5%) | 3.5% (29.4%) |
| declined although the answer existed | **0** | 4 |
| never-said questions declined correctly | 9 / 15 | 8 / 15 |
| unsupported claims per answer | 5.3 | **4.3** |

At equal budgets ICE is level or slightly ahead (2k +1.7, 4k +0.6, 8k +2.3, each ±7–8). By
question type (ICE − plain): procedure/advice +14 (±20), multi-hop +5 (±28), update 0,
fact recall −9 (±17), messy rewordings −15 (±21), **temporal ordering −24 (±19)** — the one
clear weakness, and the same "which came first" shape F1 skipped. Every advantage above is
within one history's noise except the token count; FINAL's pooled suites are where they can
be claimed.

## F10 SWITCH BUILT (`retrieval_evidence_order`; off) — "which came first": ICE reverses the order — retest [R3](RETESTS.md)

Temporal questions (21), ICE without the skip vs plain search: order reversed **7 vs 4**;
order right but graded partial 6 vs 4 (an omitted side detail, or extra timestamps flagged as
unsupported — F8). ICE's pieces carry correct dates and conversation order; both cited turns
are usually present. Neither system orders evidence by time — both use relevance — but ICE's
labels are long (`[imported; source time unknown: …] [Verified conversation order: N; …]`) and
its instructions say to order events by these dates, while plain search shows `[User, turn N]`.
Example read: the later turn at position 1, the earlier at position 8, answer reversed.
**Proposal (maintainer's, 2026-10-29):** after reranking and the budget cut, put the selected
pieces in conversation order. Test: re-read + answer B's questions both ways.

## F11 — ICE's own vector leg is where the gap sits

`warm_vector` (ICE's vector leg alone, with its reranker, 64-candidate cap and version choice)
scores 48.6% all-points vs plain vector search's 60.7%; full ICE without the skip adds +7.6
(±5.6) over its own vector leg. So the other legs help; the shortfall against plain search is
inside the vector pipeline — candidate cap (F2), whole exchanges vs single messages, version
choice (F6). At 8k, warm 52.3% vs plain 49.4%: the gap again opens only with more budget.

**Token counts** (F9) are the answering provider's own input-token count for the whole prompt
(system, recent turns, memory, question). Not included: ICE's background model work when
storing turns, and output tokens.

## F12 ⚑ JUDGE FIXED on fix-pass (grade-2; re-grade pending, [R5/R6](RETESTS.md)) — how much the judge skews

Audit of B's 5,983 grade-1 verdicts against the judge's own per-point coverage:
- **206 answers graded *incorrect* although every required point was covered** — reasons like
  "invents extensive dated lore … not in SOURCES": true details from elsewhere in the
  conversation read as invented, because SOURCES show only the cited quotes.
- **1,023 partials on one-point questions whose one point was covered**; 1,199 of 1,852 partials
  overall covered every point (F8).
- **SOURCES carried no dates** (turn, role, quote only), so every dated statement was
  unverifiable. Not-correct answers whose reason mentions dates/timestamps/turns: ICE 61–69,
  plain search 40 — ICE's prompts carry dates, so it was hit harder.
- The rubric never defined the correct/partial line.

**Size:** ICE (no skip) vs plain search is −11.0 (±7.3) by judge-correct but −4.6 (±7.3) by
required points covered: **over half the apparent gap is the judge**. Both systems are judged
alike, so this is a bias against richer answers, not a coin flip.

**Fix (grade-2, `experiments/z2/grade.py`):** the grade is decided by REQUIRED and its values
only; extra details go to `unsupported_claims` (verified by the claim check against the whole
conversation) and never lower the grade unless they contradict SOURCES; each source passage
carries the date the system saw (replay clock — matches ICE's labels exactly); new policy
`ice-z2-grade-2`, so no cached grade-1 verdict is reused. Clean yes/no score = grade-2 correct
and no claim confirmed invented (M2). **FINAL's pre-registration names M1 as the judge's
"correct": with grade-2 that definition stays and its meaning sharpens — record the rubric change
as pre-registration r3 (maintainer).**

## F13 — the build-up curve: what each part of ICE adds on B (grade-1, all-points)

| step | all-points | vs plain search | step change |
|---|---|---|---|
| ICE's vector leg alone | 48.6% | −12.2 ±7.4 | |
| + keyword search (BM25) | 59.2% | −1.7 ±7.6 | **+10.4 ±6.6** |
| + reranker | 64.2% | +2.9 ±6.6 | **+5.2 ±5.4** |
| + facts (codex) | 64.4% | +3.5 ±7.0 | 0.0 ±4.5 |
| + turn summaries | 59.0% | −2.3 ±6.5 | **−5.8 ±6.0** |
| + fold notes | 56.1% | −4.7 ±7.4 | −2.3 ±5.1 |
| + procedures | 58.0% | −2.9 ±7.1 | +1.7 ±4.9 |
| + gate (F1 skip, now fixed) | 53.4% | −7.5 ±7.5 | **−4.6 ±3.1** |
| + clusters, + temporal | 53.4% | −7.5 ±7.5 | 0.0, 0.0 (F3) |
| full (+ maintenance) | 52.0% | −8.7 ±7.8 | −1.2 ±3.9 |

Plain search: 60.7%. **ICE peaks above plain search after its first three parts**; what follows
— summaries most of all, then the gate's skip — takes it below. Additive order is fixed and each
step is one history, so this is the map for R4/R7, not a verdict. Removing or changing summaries
in reads is a v3 design decision for the maintainer.

## F14 SWITCH BUILT (`retrieval_rerank_original_margin`; 0) — summaries stand in for half the turns

In full ICE's prompts on B (174 questions, 6,161 single-turn episodic pieces): **51% of pieces are
reworded** (summary or abstract; 16% of episodic tokens), 45% the whole original (79% of tokens),
5% verbatim excerpts. S8's audit: turn summaries faithful 95% (B) / 85% (C) but **complete on
only 63% / 62%**; fold notes complete 30% / 52%; session summaries faithful 18% / 25% (read only
by MCP). The reranker makes a shorter form the piece's text whenever it scores higher (scores are
logits, −14 to +7.5). Three Z2 measurements agree that less summary helps on B: adding summaries
−5.8 (±6.0), summaries off in full ICE +4.6 (±5.5), the older stricter summary guard +7.4 (±8.3,
81 questions). Switch: a margin the shorter form must beat. Retest [R4](RETESTS.md).

## F15 — the other scores Z2 already holds (B, grade-1, all-points)

Leave-one-out (full ICE minus one part; + = removing helps): gate +4.0 (±2.9, F1), summaries
+4.6 (±5.5), notes +1.2, maintenance +1.2, codex +1.7, rerank +1.7, procedural −0.6, bm25 0.0,
clusters 0.0, temporal 0.0 — in full ICE the legs cover for each other, so no single leg's
removal costs anything measurable. Decay (live vs neutral): 0.0 (±3.6). `no_codex` vs full: +1.7
(±4.1). Budget curves: F9.

## F16 — every part, both ablations, every measure (B, grade-1) — and a verdict per part

Effect of a part = with it − without it (+ helps). ADD = the build-up step; REMOVE = full ICE
vs full ICE without it. Tokens = median change in prompt tokens per answer.

| part | ADD all-points | ADD tokens | REMOVE all-points | REMOVE tokens | outdated values | verdict (B; C pending) |
|---|---|---|---|---|---|---|
| BM25 | **+10.4 ±6.6** | +3,531 | 0.0 ±4.9 | +3,506 | −0.6 | keep — biggest single gain; redundant only once the rest is in |
| reranker | **+5.2 ±5.4** | −116 | −1.7 ±4.9 | −308 | −1.2 | keep — fix its candidate cap (F2) and summary swap (F14) |
| codex facts | 0.0 ±4.5 | −214 | −1.7 ±4.1 | −531 | −1.2 | keep — neutral on B, cheap; judge on C/FINAL |
| turn summaries | **−5.8 ±6.0** | −1,039 | −4.6 ±5.5 | −581 | −1.2 | **reduce** — R4 picks off vs margin |
| fold notes | −2.3 ±5.1 | +564 | −1.2 ±4.0 | +563 | −1.2 | candidate to drop from reads (complete 30% B) — needs C |
| procedures | +1.7 ±4.9 | +8 | +0.6 ±3.8 | 0 | −1.2 | keep — neutral |
| gate (source check) | **−4.6 ±3.1** | 0 | **−4.0 ±2.9** | 0 | 0.0 | fixed (F1) |
| clusters | 0.0 | 0 | 0.0 | 0 | 0.0 | inert on B — D9 decides after C |
| temporal leg | 0.0 | 0 | 0.0 | 0 | 0.0 | inert on B — D9 decides after C |
| maintenance | −1.2 ±3.9 | +69 | −1.2 ±3.9 | +69 | −1.7 | keep — store hygiene acts over months, not one history |
| decay (live vs neutral) | — | — | 0.0 ±3.6 | — | — | keep — designed for long horizons; no effect here |

**Turn summaries in detail.** By question type, adding them: temporal **−24 (±19)**, fact recall
−11 (±11), cumulative −20 (±26); possibly helpful for update (+12 on removal side, n.s.),
preference +8, messy rewordings +15 (n.s.). With summaries a prompt spans 46 distinct turns (55
pieces); without, 34 (35 pieces) — summaries buy breadth and cost detail, and on B detail
mattered more. Removing them from reads costs ~+580 prompt tokens per answer (~2%), leaves
outdated values and extra claims within noise, and does not save background cost (they are
still written; they also serve budget degradation and the classifier prefix).

**Ceiling.** 83% of B's questions (144/174) were answered with every required point by at least
one of the 34 answered settings (78% by ICE settings alone); 30 by none — full ICE covered a median
0% of their points; mostly update, multi-hop, cumulative and never-said questions, two with nine
required points. Those 30 are the first place to look for answer-key defects (SW/LAP's review
dropped 23–42%). 64% is the best single fixed setting, not a ceiling: the headroom is choosing per
question what the best settings already found (summaries vs originals, breadth vs detail, order).

## F17 ⚑ SWITCHES BUILT — the summary policy exists, and is overridden; on B its premises don't hold

`choose_representation` already decides per question and per turn: the original for fact /
troubleshooting intents, the summary for analysis / planning / ideation intents, otherwise the
turn's lossless flag (`inject_raw`, post-flight's "memory is earned"). Measured on B
(`experiments/z2/summary_allowance.py`):
- **It does not land.** ~50% of pieces in full ICE's prompts are summaries whether the policy
  said "prefer original" (49%), "prefer summary" (50%) or "the turn's flag" (54%): the reranker
  sends whichever version scores highest (F14).
- **The lossless flag cannot discriminate:** inject_raw is true on 1,104 of 1,119 turns (99%);
  among audited summaries it does not separate complete from incomplete; turn length does not
  either (complete 64% at 1.5–4k chars, 62% above).
- **The intent rule is wrong on B:** summaries hurt the "prefer summary" intents too (−9.1 ±8.2,
  88 questions); "prefer original" −4.8 to −7.1; time score ≥0.5 −16.7 (±15.2). No group where
  summaries clearly help (update +12, preference +8, messy +15 — each n.s., 12–20 questions).

**The allowance B supports:** originals first, a summary only when the original cannot fit the
budget (its degrade-before-drop job). Switches: `retrieval_rerank_representation="policy"` (the
reranker keeps the policy's pick, one pair per candidate) and `representation_summary_first_intents=False`
(original first for every intent). Retest R4 (`retest_policy_text`, `retest_originals_first`, beside
summaries off and the margins); on C and in FINAL before any default changes. Storage side: keep
writing summaries (budget fallback, classifier prefix), but the lossless flag needs a criterion
that actually varies — none measured yet predicts an incomplete summary.

## F18 ⚑ Can the test see the part? — why four parts read "no effect" on B and will on C

| part | what it needs to act | on B | on C | FINAL (S9 parts test) |
|---|---|---|---|---|
| clusters | automatic cluster scoping — **off since 2026-10-15 after audit A12 measured it scoping out 19 of 108 gold turns** (`retrieval_auto_cluster_scope=False`), or an explicit user scope | never acts (0/208 prompts) | never acts | **cannot see it** — 23 ordinary questions, scoping off |
| time-based retrieval | a question naming a time window ("in March") | fired on 2/174 | 0/75 | **can see it** — all 24 questions name a window |
| maintenance (slots, merges) | slots written, entities merged, questions about them | 33 slots, 49 merges; few questions | 3 slots, 23 merges | likely — alias questions ("what do my customers call…") |
| decay | months of history and questions where recency should win | acted live (22,800 row updates over 90 days) but gently — no answer changed | 33 days | **cannot see it** — no planted decay questions; 8 months of gentle decay |

So B and C are the wrong instruments for these four, and a "no effect" on C will be predicted, not
learned. **For FINAL (Z3 D14 item 5, decided 2026-10-29):** the cluster group is dropped from the parts
test — automatic scoping was measured harmful (audit A12, 19/108 gold turns scoped out) and ships
off, so there is nothing to switch; decay is tested live vs neutral on S4 (353 days) and S9 (250 days)
with `decay_by_age.py`. Decay needs planted questions (an old value vs a recent one, a long-untouched detail) on
a history spanning many months, or it is reported as "not measurable in FINAL" in its own words.
Today's database for B/C shows decay 1.000 everywhere — the read passes' neutral-decay copy, not
the live state (the replay's events hold that).

## F19 — the same question asked again later (retention as the history grows)

B asks 123 of its 179 question families two or three times, a median 150 turns apart (50–1,069).
Later ask minus first ask (per-point, paired by family): plain search **−13.6 (±12.0)**, ICE without
the skip **−11.9 (±9.6)**, ICE as run −3.3 (±9.3; its first asks were already hurt by F1),
recent turns only 0.0 — both systems lose about an eighth of their accuracy over ~150 more turns.
FINAL measures the same thing as M2 by evidence distance (SCA@d bins) and the retention report
(pre-registration §2, §4); these repeats are the paired version of it.

## F20 ⚑ SWITCH BUILT (`codex_fact_reconcile_design`, default `pre_g60`) — fact replacement needs the source passages

G60 (fix-pass 1b4b79b, a third "same specific item?" question) failed its own acceptance on the
GPU day: of Z2 S8's 99 judged replacements it kept 9 of 46 correct and 7 of 53 wrong — it said
"keep both" to nearly everything. Designs compared on the same pairs (`supersession_designs.py`,
answers saved): with **the two sentences alone no design separated correct from wrong** — pre-G60
kept 33/46 correct and 44/53 wrong (more wrong than right), Graphiti's explicit-contradiction rule
20/46 and 18/53, an attribute decomposition 1/46 and 7/53. The audit's judge had decided from the
source turns. Giving the model each fact's source passage (D4) separated them for the first time
(29 vs 19); writing the reason before the answer (D5) improved it (31 vs 16).

Through the production path (`experiments/z2/reconcile_eval.py`: batch screen + single-pair, each
fact's cited quote and date): pre_g60 33/46 kept, 44/53 wrong kept, real 87/99, planted 10/12;
g60 10, 8, 91, 8; **context 33, 15, 81, 10** — the fewest errors overall (48 vs 71 / 56), every
correct replacement pre-G60 keeps, two thirds fewer wrong ones; weaker on the qualification set's
ordinary pairs (81 vs 87). No design reaches the bar (≥38/46, ≤10/53, ≥89/99): the S8 labels are
partly inconsistent (a later prediction run of the same prompt "replaces"; a later run of the same
test "does not"; two training runs to different files labelled both ways). Default stays
`pre_g60` (what Z2 measured); the GPU day reruns the eval with the qualification misses broken down
(harmless merges vs wrong replacements), then the maintainer picks.

## Where the scores stand, B only (best estimate; grade-1 rescored per point)

| | judge-correct (grade-1) | all required points |
|---|---|---|
| plain vector search | 40.5% | 60.7% |
| ICE as Z2 ran it | 24.7% | 52.3% |
| ICE without the skip (`loo_gate`) | 29.5% | 56.1% |
| ICE's first three parts only (`add_3_codex`) | 36.2% | 64.4% |

With the skip fixed and summaries reduced, ICE is plausibly **level with plain search on B,
perhaps a few points ahead, at ~25% fewer prompt tokens** — every gap here is within ±7, so this
is a direction, not a result. R5 (does per-point rescoring match grade-2?) and R7 on B and C are what make it one.

## Already found and fixed or decided (fix-pass)

From Z2's earlier stages, each in the ROADMAP's FIX-PASS checklist with its numbers:
G79 batch summaries never written (FIXED, test pending); G60 fact replacements 53/99 wrong
(FIXED, test pending); S8 part-wise judge failure counted as unfaithful (FIXED); G75 turn
summaries 5–15% unfaithful — second reading, default off (FIXED, test pending); entity links
~20% wrong (NO FIX — no structural rule separates them); codex fact recall 0.42–0.51 (NO FIX
unless answers show it); session summaries unverified and read only by MCP (maintainer
decision); B7 classifier init/input variants (built, test pending).
