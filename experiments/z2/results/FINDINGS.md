# Z2 findings — what the run showed about the system (v3)

**Version: v3** (`main` as Z2 runs it). None of these numbers are comparable with the
paper's v2 numbers. Every finding names its evidence, its size with an uncertainty, and
its status: **TO FIX** (cause measured, fix not yet chosen), **OPEN** (seen, cause or size
not settled), **FIXED** (on `fix-pass`, see the ROADMAP's FIX-PASS checklist), **NO FIX**
(measured and left, with the reason).

Scoring used below: each answer graded once by the judge (Muse Spark 1.3), correct = 1,
partial = 0.5, wrong = 0; differences are **paired** (same questions, two settings), with a
95% interval. Evidence lives under the gitignored `logs/` (B is a private history).

## F1 ⚑ TO FIX — the source check cancels memory the main decision asked for

**What happens.** After ICE's main retrieval decision (B2, `src/api/memory_decision.py`)
says "retrieve", the source refinement (`src/api/memory_preparation.py` →
`source_need.judge_source_need` → `source_action`) asks `gemma4:e4b` whether the current
message already holds everything needed. On a `visible_evidence` verdict whose quote is a
substring of the message (and passes source proof), the retrieval is **skipped** and the
prompt gets no memory at all. It can only ever remove memory a positive decision granted.

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

## Already found and fixed or decided (fix-pass)

From Z2's earlier stages, each in the ROADMAP's FIX-PASS checklist with its numbers:
G79 batch summaries never written (FIXED, test pending); G60 fact replacements 53/99 wrong
(FIXED, test pending); S8 part-wise judge failure counted as unfaithful (FIXED); G75 turn
summaries 5–15% unfaithful — second reading, default off (FIXED, test pending); entity links
~20% wrong (NO FIX — no structural rule separates them); codex fact recall 0.42–0.51 (NO FIX
unless answers show it); session summaries unverified and read only by MCP (maintainer
decision); B7 classifier init/input variants (built, test pending).
