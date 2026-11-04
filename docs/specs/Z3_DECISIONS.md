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

**Locked 2026-10-26 (maintainer; D dropped 2026-10-28 — the v3 hyper-parameter tuning history):** B (re-replayed; development), A, ~~D (development)~~, LAP (the
101-pair laptop-decision chat — the DeepSeek thread first chosen is an excluded conversation and is
never used), SW — each replayed alone. The stitched ICE chat is **dropped** for BEAM (D3); one of its chats, ICE2
(the 331-pair chat after C), was **added 2026-10-31** (pre-registration revision 4). Linux and
short chats dropped. Written into FINAL_experiments.md §2.1. The text below is the original proposal.

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

## D2 Synthetic conversations — **decided (maintainer, 2026-10-26)**

S1–S8 plus S9 (the ablation conversation), plans by Claude Haiku 5.5 agents, the simulated
assistant working from a local plan-built briefing (never the probes); see FINAL_experiments.md §2.1.
The text below is the original proposal.

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

## D4 Conditions and baselines — **built as proposed** (`final_main`: full_default, full, warm_vector,
recent_only, bare_vector, long_context); not changed by the maintainer, not explicitly confirmed

**Superseded 2026-10-28 (maintainer):** our data runs ICE as it runs, plain vector, long context
(256K) and recent only; `full` and warm vector live in S9's parts test only; each public benchmark
runs its own way — the table in FINAL_PREREGISTRATION.md §1.

**Pick:** `recent_only`, `bare_vector` (budget-matched), `warm_vector`, `full` (neutral
decay) and `full_default`, plus **a long-context baseline**: the newest history packed into
the answerer's window (49,152 tokens for the local answerer; for FINAL fixed at freeze, D13), no memory system — the "just use a long
context" answer to the v2 reviewers' straw-man-baseline criticism (FINAL_experiments.md
criticism 1), which the current roster had lost. Optional: a frontier long-context model
on a ~50-question sample if a venue requires it (cost per call ~the whole history).

## D5 ⚑ Ablations in FINAL

**Decided 2026-10-26:** leave-one-out on S9 (a purpose-built ablation conversation) plus sealed B;
see FINAL_experiments.md §2.1. The text below is the original proposal.
**Revised 2026-10-28 (maintainer):** S9 only — B's questions were not written per part, and Z2 already
ran B's switched-off conditions; B keeps its background-content audit.

Z2's ablations are development evidence (fixes were made against them), so FINAL repeats
them on the frozen post-fix system (FINAL_experiments.md §2.7). **Pick:** the 10
leave-one-out conditions only, on sealed B/C and the synthetic set; the 9-step additive
ladder is reported from Z2, labelled pre-fix development evidence. Read-time only — no new
replays; cost = reads + local answers (GPU days) + grading.

## D6 Answerer — as Z2 D6c (local qwen3.8:27b, thinking on); cloud-answerer subset built

**Superseded 2026-10-27 by D13:** answers on Muse 1.2 (cloud); the laptop's daily hours made
local answering too slow. The cloud-answerer subset stays (a non-Meta family).

**Pick:** the same as Z2 D6c — local `qwen3.8:27b`, thinking on — so FINAL's answers cost
GPU days, not plan money. Plus a ~200-answer cloud-answerer subset (a different family) to
show whether the conclusions depend on the answerer.

## D7 ⚑ Judges

**Superseded 2026-10-27 by D13:** judge Muse 1.3; no second full judge — a cross-family
judge — a fresh non-Meta model picked at the time, local or cheap cloud, the last pass —
re-grades ~450 answers. The Haiku pick below is dropped.

**Benched 2026-10-26 (maintainer; confirmed the same day — benched until a cheap model is available
at the time):** no second judge is fixed yet — no cheap model at hand
(Claude Haiku on OpenCode costs about as much as Luna; no Anthropic API key). The machinery is
ready (`grade.py --judge`, `qualify.py --judge`). The "writers never judge" rule may be relaxed
for the second judge if that is what the limits allow — e.g. DeepSeek, which co-wrote the
private ledgers — with the overlap stated in the paper; the synthetic answer keys come from
the plans, not from DeepSeek. Decided before FINAL's grading. The text below is the original
proposal.

Muse 1.3 (as Z2) plus a **second, untuned judge** (Z2 D13). It must share a family with
neither the answerer (Qwen), the writers (DeepSeek, MiMo) nor Muse (Meta). **Pick:** Claude
Haiku 5.5 on the plan (3,850 per 5 h by the table — check the real per-call cost first).
Report agreement; never average the two.

## D8 Time and money, rough

**Superseded:** estimates in D12 and D13.

**Superseded 2026-10-26** by the build list and run estimate in FINAL_experiments.md §2.1
(≈ 30–32 GPU days with the S9 build-up and the memory-system baselines). MemoryAgentBench benched;
LongMemEval-S unchanged.

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

**⚑ Amended by the maintainer, 2026-11-01: switch off before FINAL, delete after.** A part that does
not earn its place is switched OFF at the freeze — its default changes, its code stays — and is
deleted only after FINAL, so a deletion cannot introduce bugs the experiment never measures. FINAL
measures the simpler system either way (the part is off). The freeze record is FEATURE_INVENTORY's
"Z3 FREEZE" section; rule 3's removal commit and CLEANUP entry happen after FINAL.

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

**Decided 2026-11-04 (maintainer), from Z2's ablations and S8 quality:** fold notes OFF (B +1.1 / C −1.3
judge-correct, ~560 tokens a prompt, ~⅓ stale); batch summaries OFF as a design decision (never written
in Z2 — rule 2); procedures OFF (0 to ±1.7, 26% unfaithful); session summaries already off (`2e9f4ed`);
temporal retrieval and maintenance ON (S9 tests them); entity links unchanged (no switch). R7 reads
them off by Z2's own ablation switches; each gets a production off switch at the freeze, checked by
query; rule 4 applied (they leave S9's parts test). Record: FEATURE_INVENTORY "Z3 FREEZE".

## D10 Build-up ablation on S9 — **decided yes (maintainer, 2026-10-26)**

Leave-one-out (decided) switches off one part at a time with all others on. Its blind spot:
two parts that cover for each other each look useless alone. A **build-up** starts from the
bare system and adds one part at a time in Z2's frozen order, showing what each adds on top
of the earlier ones (order-dependent: the first parts get the easy credit). Cost on S9:
9 more conditions × 260 asks ≈ 2,300 local answers ≈ 2 GPU days. Recommended: yes. (With answers on
the cloud, D13: ~0.3 GPU days of reads plus ~2,300 cloud answers.)

## D11 Research-paper gaps — when each is done (maintainer, 2026-10-26)

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
   one person's; LLM judges; the answerer is one cloud model, Muse 1.2 (the cloud-answerer subset
   addresses part of it); possible benchmark contamination of public sets.
8. **Reproducibility package:** code, synthetic plans and conversations, BEAM/LongMemEval
   selections and seeds, model ids and settings — everything except the private histories.

**When each item happens:**
| Item | When |
|---|---|
| 3 Pre-registration note | **before FINAL runs** — and it includes the hallucination measures: invented claims per answer (Z2's claim chain: confirmed_invented), wrong answers to never-said questions (abstention accuracy on negatives), use of a superseded value, attribution mix-ups (assistant suggestion reported as the user's decision) |
| 1 A published memory system as a baseline | **in FINAL**, ~~on the LongMemEval-S slice~~ **on LoCoMo only** (2026-10-28: the benchmark Graphiti, Mem0 and A-Mem all report on); Graphiti, Mem0, A-Mem |
| 4 Run-to-run variance | **in FINAL**: replay one history twice (S1, ~4 h) |
| 5 Cost and speed | **in FINAL**: logged per run, reported beside scores |
| 2 Judge vs a human | **during FINAL grading**: ~100 answers blind-graded by the maintainer |
| 6 Error analysis | **after FINAL**, from the graded answers |
| 7 Threats to validity, 8 reproducibility package | **after FINAL**, with the paper |

**Memory-system baseline — candidates (no independent leaderboard exists; every headline score
is self-reported under different reader/judge models):** Graphiti (Zep's open-source temporal
knowledge graph — the closest design to ICE: bi-temporal facts that supersede each other, an
entity graph; Zep's paper reports 71.2% on LongMemEval with GPT-4o) and Mem0 (the most-cited;
its open-source SDK, not its managed platform). Both must run entirely in our setup: same
LongMemEval-S slice, same answerer, same judge. Cost: their ingestion makes several model
calls per session — on the 100-instance slice roughly 2–4 GPU days each with a local model.

**Requirements (maintainer, 2026-10-26):** no paid service — the system's code must be open
source and run locally, through our own adapter. Its LLM roles use FINAL's models in place of
any API model — its background/extraction work on gemma4:e4b; its contexts answered by FINAL's
answerer (Muse 1.2, D13) and graded by FINAL's judge, like every condition; its small
parts (embedder, reranker) stay as the system itself chose them, run locally. Choose by three
tests together: it runs locally; it has its own LongMemEval score at a good level; users
regard its real-world output as good. Candidates: Graphiti (Zep), Mem0 (open-source SDK),
Hindsight (Vectorize, MIT); **which one (or two) is decided at the time, by how things stand then.**

## D12 Scoring and comparison set — pre-registration r2 (maintainer, 2026-10-26)

Replaces the judge's 1–5 score as FINAL's headline (it saturates: most answers in every
condition score 4–5, so it cannot separate systems). Three measures, each covering what the
others miss — full text in [FINAL_PREREGISTRATION.md](FINAL_PREREGISTRATION.md) §2–4:
- **M1 accuracy** (correct / not, partial = not) — what LongMemEval, LoCoMo, Zep and Mem0
  report, so ICE is comparable; benchmarks keep their native score beside it.
- **M2 supported current accuracy against evidence distance** — correct as of the moment
  asked, no outdated value, no invented claim, declines on never-said; plotted by how far
  the evidence lies (< 32K · 32K–128K · 128K–512K · > 512K tokens) and split in-window /
  out-of-window. LSREP's thesis measure; replaces Z2's paired 1–5 scoring.
- **M3 cost per correct answer** — prompt tokens per answer and write-time cost per turn.

Comparison set (superseded in detail by FINAL_PREREGISTRATION.md §1, 2026-10-28): own baselines (bare_vector, recent_only, long_context; warm_vector only in S9's parts test);
**Graphiti, Mem0 and A-Mem** run locally in the same setup; a **full-history upper bound** on
BEAM 128K (cloud model, ≥ 256K window — the tier runs 100K–225K tokens, 15 of 20 over 128K)
with ICE's contexts answered by the same model; **LoCoMo** as an extra table. Drop order and
the never-dropped list are in the pre-registration §5. Estimate ≈ 35–38 GPU days with local
answers (≈ 23–27 after D13). Which of paper 1's measures come back, and which are dropped:
pre-registration §4.

## D13 Models and run order for FINAL (maintainer, 2026-10-27)

**Every role, as decided now** (any model may be swapped for the closest available one at
the time, recorded before use — never silently):

| Where | Role | Model |
|---|---|---|
| ICE itself (local, under test) | background model (summaries, facts, checks) | gemma4:e4b |
| | encoder (classifier + embeddings) | Qwen3-Embedding-0.6B |
| Memory-system baselines (local) | their background/extraction work | gemma4:e4b |
| | their answers (cloud) | Muse Spark 1.2, like every condition |
| | their embedder | nomic-embed-text |
| Synthetic conversations (cloud, 2026-10-28) | user side / assistant side | MiMo-V2.6-Flash / DeepSeek V4.1 Flash, plain text, thinking off (was local ministral-3:8b / mistral-nemo); picked by `experiments/synth/speaker_trial.py` |
| Ground truth for private histories (cloud) | ledger + questions: writer / checker | DeepSeek V4.1 Flash / MiMo V2.6 Flash — **no Muse**; disputes dropped or sent to the maintainer's review |
| | mechanical jobs (quote finding, dedupe, format) | a small local model |
| Answers (cloud) | answerer, every condition | Muse Spark 1.2 |
| Grading (cloud) | judge | Muse Spark 1.3 |
| | cross-family judge, ~450 answers | a fresh non-Meta model picked at the time — local or cheap cloud — qualified on the planted controls, the last pass (revised 2026-10-27; not GLM by default) |
| | background-content audit second opinion | DeepSeek V4.1 Flash, 20% sample |
| Comparisons (cloud) | cloud-answerer subset (~200) | a non-Meta family, picked at the time |
| | full history on BEAM's 500K tier (5) | Muse Spark 1.2 itself, in its 1M window (2026-10-28) |
| | long context (every dataset) | the newest 256K tokens, answered by Muse 1.2 (2026-10-28) |
| People | blind check, ~100 answers | the maintainer |

Already done, not re-run: synthetic plans and questions (Claude Haiku 5.5 agents).

**Run order — all local, then all cloud.** Replay checkpoints at every session end, not at
question points, so only reads need the questions.
*Amended 2026-10-31 (maintainer):* the memory systems' ingest + search (no ICE in it) run early, in
the GPU-free weeks before the freeze; only their placement in ICE's prompt frame waits for FINAL's
LoCoMo read (RETESTS timetable).
1. *Cloud, inside Z2's cloud block (same plan period):* the private ground truth for A, SW,
   LAP (ledger, questions), then the maintainer's review (D dropped 2026-10-28).
2. *Local:* fix-pass GPU tests → freeze → synthetic generation → every replay (private,
   synthetic, BEAM, LongMemEval-S, LoCoMo, the three memory systems' ingestion) → every
   read pass.
3. *Cloud, one block:* every answer, grading, the cross-family judge, the cloud-answerer
   subset, the full-history upper bound, the background-content audit.
4. *People:* the blind human check.

Planning rate for the local phase: ~10–12 GPU hours a day, up to ~15–16. At ~11 h a day the
local phase (≈ 18–22 GPU days after the 2026-10-28 BEAM trim; ≈ 23–27 before it) takes ≈ 5½–7
weeks; the cloud block ≈ 1–2 plan periods of allowance, sized from
real calls by `experiments/z2/allowance.py` (TRAPS 148).

**Data use — accepted by the maintainer, 2026-10-27 (Z2 D6d):** Muse Contributor models may
train on prompts and are not zero-retention (Z2 spec, Data-use terms); Muse 1.2 Contributor is
accepted as the answerer on those terms, for every full memory prompt of the private
histories (B, A, SW, LAP; ICE2 from 2026-10-31, with its addition).

**Built 2026-10-27 (fix-pass):** run_final.py runs every prep-/local- stage before any cloud stage
(answers `cloud-answers-*`, audits, grading, subset, cross-judge); B and S9 freeze their audit
inventory locally before any read pass (the read passes restore checkpoints into the replay's
database, so a later inventory would see a checkpoint, not the final store); the private ground
truth runs with `--tie-breaker none` and `--verifier mimo`; the cross-family judge and the
cloud-answerer subset are runner stages (judge picked at the time, `CROSS_JUDGE`; subset default LongCat-2.0). Also built the same
day: A-Mem (the memory layer its paper evaluated — the library skips note construction), the LoCoMo
adapter, and the full-history upper-bound runner (FINAL_experiments.md §2.1 items 15–17). The
upper-bound model is the one choice left open (pre-registration §6).

**Revised 2026-10-28 (maintainer):** reporting — LSREP suites pooled first (private held out;
private development, B's sealed questions + D; synthetic, S8 once), public benchmarks each in its
own scoring, then every conversation as a deep dive under its built-for aspect; no per-conversation
verdicts. BEAM trimmed to the 128K tier (20) + 5 of the 500K tier; the 1M tier dropped.
**Revised 2026-10-28 (maintainer), main experiment:** ICE frozen as it runs (`full_default`) against
plain vector, recent only and long context — now the newest 256K tokens; decay-off (`full`), warm
vector and every switched-off variant only in S9's parts test. Tests: M1 and M2 on all
questions, P1 out-of-window where a suite has ≥ 100 pairs. The synthetic suite carries the claims
(releasable); private is the real-world check (not releasable). Full history on BEAM's 500K tier, in
Muse 1.2's own window, compared with ICE and long context on the same questions.

**Made clean 2026-10-28 (maintainer: "if it's ours we can do anything; public benchmarks run their
own way, BEAM the exception"):** memory systems on LoCoMo only; LongMemEval-S as its field runs it
(ICE, plain vector, full context, its oracle setting); LoCoMo as its field runs it (ICE, ICE at a 2K
budget, plain vector, full context, the three systems; categories 1–4 headline; prompt size shown);
LoCoMo speakers on fixed sides; Mem0 per speaker; B's parts run dropped. Table:
FINAL_PREREGISTRATION.md §1.

**Scoring names and norms, 2026-10-28 (maintainer):** LSREP's measures get citable names (ACC,
SCA, SCA-BW, SCA@d, UA, TUR, SPF, SFR, ER, ICR/NSR/OVR/AER, MFA — table in
FINAL_PREREGISTRATION.md §2; TUR and SPF keep paper 1's names with sounder definitions). Public
benchmarks use their own scoring and the field's norms only: LoCoMo by Mem0's protocol (J, F1,
BLEU-1; categories 1–4), LongMemEval-S by its own judge, BEAM by its nuggets — no LSREP measure
is added to them.

## D14 ⚑ Z2's findings carried into FINAL — **approved by the maintainer 2026-10-29**

Z2's grades on B (findings F1–F15 in `experiments/z2/results/FINDINGS.md`) changed four things
FINAL depends on. Each is built on fix-pass; nothing is a number until its retest
(`experiments/z2/results/RETESTS.md`) runs.
1. **Retrieval:** the source check no longer cancels a positive decision (F1, shipped on fix-pass).
   Three switches, off until measured: candidates ranked + keep the rest (F2), conversation
   order (F10), original-text margin for summaries (F14). Their defaults are set from R2–R4 and
   recorded at freeze.
2. **Judging:** grade-2 rubric and dated SOURCES (F12) — the same judge model, the same M1/M2,
   a defined rubric. **Z2 is NOT re-graded (maintainer, 2026-10-29):** its grading finishes under
   grade-1 on `main` (B's remaining phases, C, S12 — complete and comparable with B), and the
   judge change is measured on a 250-answer sample graded under both rubrics (R5). Corrected Z2
   comparisons use the per-point rescoring of grade-1's own verdicts, if R5 shows it agrees with
   grade-2. FINAL grades only under grade-2, after R5. Z2's grade-1 grading must finish on
   `main` before fix-pass merges into it (the merge brings grade-2).
3. **Design decisions the evidence raises** (maintainer's): turn summaries in reads (F13/F14 —
   less summary helps on B), and whether clusters / temporal / maintenance stay (inert on B, F3;
   D9's simplification pass).
4. **Order:** S12 local → GPU retests (R1, R2, R8, R9 and the reads of R3/R4) → cloud: Z2's
   grade-1 grading and S12 to completion (R6) → Z3 judging (R5, retest answers + grade-2) → cloud
   prep (synthetic, A) — three phases, never mixed (maintainer, 2026-10-29) → R7 → merge →
   freeze → **research briefing for external review (ICE v3 architecture, Z2→Z3 findings, the FINAL plan; a 4–6 page PDF, appendices if useful; task in the gitignored docs/PUBLISHING.md [PRIVATE:research-briefing]) — the gate before FINAL starts; ~1 week; feedback becomes a
   pre-registration revision** → FINAL local → FINAL cloud (Go Plus). A later private-question top-up (more questions
   for A/SW/LAP) is its own cloud block.
5. **Every part FINAL tests must be able to act (Z2 F18).** B and C cannot exercise clusters
   (automatic scoping off), time windows (no question names one), decay (gentle over 33–90 days,
   no planted questions) or much of maintenance. S9 covers time windows and aliases. **Decided
   2026-10-29:** **clusters** leave the parts test — audit A12 (2026-10-15) measured automatic
   scoping cutting 19 of 108 gold turns, so it ships off and its switch changes nothing; reported
   as "off by default after A12, not tested". **Decay** is tested: live vs neutral decay on S4
   (353 days, `final_decay` preset) and S9 (250 days), split by evidence age and update questions
   (`experiments/z2/decay_by_age.py`); B's questions use evidence under 30 days old (133/158).

