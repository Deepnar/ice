# Z3 — FINAL's decisions (decision sheet, DRAFT 2026-10-26)

**Status: proposal for the maintainer — nothing here is decided until marked so.** Z3 is
the roadmap item that fixes FINAL's roster before FINAL runs ([ROADMAP Z3](../ROADMAP.md),
[FINAL_experiments.md](FINAL_experiments.md) §2.1, [Z2_EVALUATION.md](Z2_EVALUATION.md) D13).
FINAL reuses Z2's machinery unchanged (ledger, questions, replay, read passes, local
answering, grading); every Z2 hard-coding of B/C becomes a parameter.

Why a sheet: the 2026-10-15 roster (FINAL_experiments.md §2.1) is larger than one laptop
can run. Rough costs measured in Z2 (v3):

| work | measured rate | source |
|---|---|---|
| replay (semi-LSREP) | B ~35 s/turn, C ~160 s/turn (long pastes); LME oracle ~14 s/turn est. | Z2 runner log |
| read passes | B ~4 s/context, C ~16 s | Z2 S9 |
| local answers (qwen3.8:27b, thinking) | ~50–110 s/answer | Z2 S10, D6c |
| ledger + questions (cloud writers) | the largest cloud spend of Z2; plus the maintainer's review | Z2 S1–S6 |

Each decision: options, the pick, and what it costs. **⚑ = needs the maintainer.**

## D1 ⚑ Private conversations

**Locked 2026-10-26 (maintainer):** B (re-replayed), A, D (development), the long laptop-decision
thread, the software project — each replayed alone. The stitched ICE chat is pending the BEAM
check (D3). Linux and short chats dropped. Written into FINAL_experiments.md §2.1. The text
below is the original proposal.

Candidates (sizes in FINAL_experiments.md §2.1; ids in `docs/PRIVATE_CONTEXT.md`
[PRIVATE:final-private-roster]):

| Group | Size | Overlaps Z2? |
|---|---|---|
| ICE project, stitched (six chats) | 1,737 user turns, ~5.9M tokens | **yes** — its first segment is Z2's C |
| Creative saga + review chat | ~1.07M tokens | **yes** — the saga is Z2's B |
| Fan-fiction history | 290 pairs, ~267K tokens | no — untouched by tuning and Z2 |
| Hardware/laptop decision (two chats) | 122 pairs, ~144K | no |
| Linux setup (three chats) | 113 pairs, ~370K | no |
| Software project (one chat) | 128 pairs, ~96K | no |

**Pick:** the FINAL-sealed B/C families (already sealed at S6) **plus the four untouched
groups** (~650 pairs). Drop the two overlapping groups: their new material would need a
fresh ledger over millions of tokens, and their histories are already Z2 development data.
Cost: ledger + questions for ~650 pairs (about a fifth of Z2's B+C prep); replay ~1 day.

## D2 ⚑ Synthetic conversations

A synthetic set is the reproducible, publishable part with a perfect answer key: the facts
are planted, so ground truth is the plan itself.

**Pick:** 6 conversations = 3 lengths (~300, ~800, ~1,500 user turns) × 2 writing styles,
each style different from the people-typing-at-a-chatbot convention ICE was tuned on
(e.g. terse lowercase with typos; long rambling run-ons; formal and structured — choose
two), across different domains (a project, life planning, a hobby). Each has a planted-fact
plan of ~150 items with a schedule: introduced at turn t, changed at t′, corrected,
preferences, plans, numbers, dated events, cross-session references, and never-stated
negatives. Questions come from the plan through Z2's question machinery.

**How:** seeds (personas, arcs, fact plans with schedules — structured JSON) by a fast
model; the maintainer suggested a Claude Haiku agent for this step. The conversation text
itself by a cheap cloud model playing the user from the persona and schedule, and the
assistant side by another family. Rough cost: ~5,200 turns × 2 sides; ~60M input tokens
on a Flash-class model (a few dollars at list price), or 60–115 GPU hours locally. Check
the per-call cost on 20 calls before sizing (TRAPS 148).

## D3 ⚑ Public data

**Decided 2026-10-26 (maintainer):** BEAM natively — 128K all 20, 500K 10, 1M 10, no 10M,
seeded and category-balanced; the stitched ICE chat is dropped for it (see FINAL_experiments.md
§2.1). The LongMemEval-S slice size is still open.

**BEAM, checked 2026-10-26** ([paper, ICLR 2026](https://arxiv.org/abs/2510.27246);
[dataset](https://huggingface.co/datasets/Mohammadta/BEAM), CC-BY-SA-4.0, not gated;
10M tier in `Mohammadta/BEAM-10M`): 100 synthetic conversations — 128K tier 20, 500K 35,
1M 35, 10M 10 — in sessions with time anchors; general, coding and math domains (19 in
all). Per conversation, on average: 128K ~144 user messages, 500K ~544, 1M ~1,067 (about
B's length), 10M ~10,435. 20 human-validated probing questions per conversation over ten
abilities (abstention, contradiction resolution, event ordering, information extraction,
instruction following, knowledge update, multi-session reasoning, preference following,
summarization, temporal reasoning), asked **after** the whole conversation. Scoring:
nugget-based (an LLM judge scores each reference nugget 0 / 0.5 / 1; Kendall tau-b for
event ordering; judge model unnamed). Published 1M averages: vanilla long context
0.19–0.26, RAG top-5 0.27–0.31, the authors' LIGHT 0.28–0.34. The user side is an LLM
role-player. Replay estimate at B's rate: ~12 h per 1M conversation, ~6 h per 500K, ~1.5 h
per 128K; no ledger or question writing needed.

| option | cost | pick |
|---|---|---|
| LongMemEval-S, all 500 | 122,506 user turns ≈ ~20 days of replay | no |
| **LongMemEval-S, 100-instance slice, stratified by question type** | ~24k turns ≈ ~4 days | **yes** |
| LongMemEval oracle replay (as in Z2) | ~2 days | **no** — Z2 ran it (pre-fix, development); in FINAL the oracle evidence is kept as a read-time control inside the S slice (`all_evidence`: answer from the evidence sessions only), which needs no replay |
| BEAM | 128K–1M-token conversations; replay scales with turns | no (or one 128K conversation if a venue asks) |
| MemoryAgentBench | agent-memory benchmark, chat path only | no |
| LoCoMo | dropped 2026-10-15 (short; ~6.4% wrong key; lenient judge) | no |

LongMemEval-S is already downloaded (`experiments/lme/data/longmemeval_s`, pinned in
`MANIFEST.json`). Report the slice as a slice; never compare to published full-set scores.

## D4 ⚑ Conditions and baselines

**Pick:** `recent_only`, `bare_vector` (budget-matched), `warm_vector`, `full` (neutral
decay) and `full_default`, plus **a long-context baseline**: the newest history packed into
the answerer's whole window (49,152 tokens), no memory system — the "just use a long
context" answer to the v2 reviewers' straw-man-baseline criticism (FINAL_experiments.md
criticism 1), which the current roster had lost. Optional: a frontier long-context model
on a ~50-question sample if a venue requires it (cost per call ~the whole history).

## D5 ⚑ Ablations in FINAL

**Decided 2026-10-26:** leave-one-out on S9 (a purpose-built ablation conversation) plus sealed B;
see FINAL_experiments.md §2.1. The text below is the original proposal.

Z2's ablations are development evidence (fixes were made against them), so FINAL repeats
them on the frozen post-fix system (FINAL_experiments.md §2.7). **Pick:** the 10
leave-one-out conditions only, on sealed B/C and the synthetic set; the 9-step additive
ladder is reported from Z2, labelled pre-fix development evidence. Read-time only — no new
replays; cost = reads + local answers (GPU days) + grading.

## D6 Answerer

**Pick:** the same as Z2 D6c — local `qwen3.8:27b`, thinking on — so FINAL's answers cost
GPU days, not plan money. Plus a ~200-answer cloud-answerer subset (a different family) to
show whether the conclusions depend on the answerer.

## D7 ⚑ Judges

Muse 1.3 (as Z2) plus a **second, untuned judge** (Z2 D13). It must share a family with
neither the answerer (Qwen), the writers (DeepSeek, MiMo) nor Muse (Meta). **Pick:** Claude
Haiku 5.5 on the plan (3,850 per 5 h by the table — check the real per-call cost first).
Report agreement; never average the two.

## D8 Time and money, rough

**Superseded 2026-10-26** by the build list and run estimate in FINAL_experiments.md §2.1
(≈ 26 GPU days). MemoryAgentBench benched; LongMemEval-S unchanged.

GPU: replays ~1 (private) + ~1 (B/C again, post-fix) + ~2–3 (synthetic) + ~4 (LME-S)
days; reads ~2 days; local answers ~5–8 days. ≈ 2.5–3 weeks of GPU, resumable.
Cloud: ledger + questions for ~650 private pairs and the synthetic set (the planted plan
replaces a ledger there), two judges over FINAL's answers. One renewed plan period, sized
by measured per-call cost before launch.

## Not decided here

The fold staleness question (G73) and the entity-link precision are fix-pass items
decided from Z2's S11 evidence, before FINAL freezes the system.

## D9 ⚑ Post-Z2 simplification pass (maintainer, 2026-10-26)

**Decision:** once Z2's results are in (S11 grades, the leave-one-out and build-up ablations,
S8's quality audit, reach into prompts) and after a short re-investigation, the maintainer
intends to **remove** parts that do not earn their place — named so far: batch summaries,
session summaries, "and things alike". Done **before FINAL freezes the system**, so FINAL
measures the simpler system.

**Rules for each removal:**
1. **Evidence per part, not a hunch:** its ablation effect in Z2 (does switching it off change
   answers beyond noise?), its S8 quality, and how often it reaches a prompt. A part that
   costs work, is often wrong and changes no answer goes.
2. **A part Z2 could not see is not judged by Z2.** Batch summaries never ran in Z2 (G79 bug:
   0 rows in both replays), so Z2 says nothing about them. Either test them after the G79 fix,
   or remove them as a *design* decision recorded as such — never as "Z2 showed they don't help".
3. **Recoverable and recorded:** each removal is logged in docs/CLEANUP.md (what, why, the
   evidence, the commit), FEATURE_INVENTORY rows updated, code removed in its own commit so a
   revert brings it back.
4. **Ablation groups follow:** a removed part leaves FINAL's leave-one-out/build-up set.

**Candidates and what is known now (decided only after Z2):**
| Part | Known now |
|---|---|
| Batch summaries | never ran in Z2 (bug); fixed on branch, untested |
| Session summaries (reflection) | faithful B 8/44, C 3/12; written unverified; their original consumer (procedural crystallization) was removed; only reader is MCP `ice://session-start` |
| Procedural memory | B 26% unfaithful; ~0.7 fragments per prompt |
| Entity mention links | 19% of mention links wrong; no structural fix separates them |
| Conversation fold (notes) | in every prompt; ~1/3 stale, C ~17% unfaithful; `loo_notes` decides |
| Cluster scoping, others | read from the ablations |

**View recorded with the decision:** this is the right direction — a smaller system is easier
to defend, cheaper to run, and Z2 is exactly the evidence to decide with. The two cautions are
rule 2 (batch summaries) and keeping each removal reversible.

## D10 ⚑ Build-up ablation on S9 (open)

Leave-one-out (decided) switches off one part at a time with all others on. Its blind spot:
two parts that cover for each other each look useless alone. A **build-up** starts from the
bare system and adds one part at a time in Z2's frozen order, showing what each adds on top
of the earlier ones (order-dependent: the first parts get the easy credit). Cost on S9:
9 more conditions × 260 asks ≈ 2,300 local answers ≈ 2 GPU days. Recommended: yes.

## D11 Research-paper gaps to close before or with FINAL (proposed 2026-10-26)

What a reviewer will look for that the current plan does not yet give:
1. **Other memory systems as baselines.** FINAL compares ICE with our own baselines (bare
   vector, warm vector, recent-only, long context). Reviewers expect at least one published
   memory system run on the same data (e.g. Mem0 or Letta/MemGPT, run locally with the same
   answerer). Without it the claim is "better than simple retrieval", not "competitive".
2. **Judge validity against people.** Planted controls prove the judge rejects wrong answers;
   a human-graded sample (~100 answers graded blind by the maintainer, agreement reported)
   shows its grades match a person's.
3. **A pre-registration note** written before FINAL runs: primary metric, the comparisons that
   count, the keep rule, how many comparisons (corrected), what counts as a win.
4. **Run-to-run variance.** Answers are deterministic, but replay is not (background models).
   Replay one history twice and report how much the scores move.
5. **Cost and speed.** ICE's claim is quality at a fixed budget: report prompt tokens per
   answer, write-path time per turn and background model calls per turn beside the scores.
6. **Error analysis.** A small taxonomy of why answers failed (not retrieved, retrieved but
   misread, stale value used, attribution confused), from the graded answers.
7. **Threats to validity, stated plainly:** synthetic users written by an LLM; private data is
   one person's; LLM judges; the answerer is one local model (the cloud-answerer subset
   addresses part of it); possible benchmark contamination of public sets.
8. **Reproducibility package:** code, synthetic plans and conversations, BEAM/LongMemEval
   selections and seeds, model ids and settings — everything except the private histories.
