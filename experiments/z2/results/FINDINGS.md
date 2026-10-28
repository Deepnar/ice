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

## F2 OPEN (small) — extra memory budget helps plain search, not ICE

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

## F6 OPEN — ICE's copy of the right turn is often shortened

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

## F10 OPEN — "which came first": ICE reverses the order

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

## Already found and fixed or decided (fix-pass)

From Z2's earlier stages, each in the ROADMAP's FIX-PASS checklist with its numbers:
G79 batch summaries never written (FIXED, test pending); G60 fact replacements 53/99 wrong
(FIXED, test pending); S8 part-wise judge failure counted as unfaithful (FIXED); G75 turn
summaries 5–15% unfaithful — second reading, default off (FIXED, test pending); entity links
~20% wrong (NO FIX — no structural rule separates them); codex fact recall 0.42–0.51 (NO FIX
unless answers show it); session summaries unverified and read only by MCP (maintainer
decision); B7 classifier init/input variants (built, test pending).
