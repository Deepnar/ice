# Z2 — question machinery, semi-LSREP, background-content quality, LME oracle

**Version: v3.** Decided with the maintainer 2026-10-14/15. **Approved by the maintainer 2026-10-15**; build starts in the next session, after tuning closes. **Z2 is not FINAL.**
FINAL is only the paper experiment that follows Z2 → fix pass → Z3 (roadmap
gates). Z2 builds the evaluation machinery FINAL will reuse, then measures the
frozen, tuned v3 system once. Z2 reruns only for parts still broken after its
first run.

**Controls over:** the Z2 sections of
[V3_POST_RESEED_REPAIR](V3_POST_RESEED_REPAIR.md) (§2.4–2.6 and the §2.5/§5.2
maintainer decisions), which remain the historical record and the source of the
oracle, bare-vector and replay contracts this spec cites. Where they conflict,
this spec wins. [FINAL_experiments](FINAL_experiments.md) §2.4/§2.5 (probe
taxonomy, citation-verified ground truth) are adopted and extended here.

Assumes decided specs: `V3_POST_RESEED_REPAIR.md` (repairs, tuning, frozen
configuration), `T_temporal.md` (timescope modes).

## 1. Decisions

**D1 — What Z2 is.** Seven parts, in this order:

1. **Speed measurement and fixes** for the replay write path (before any long run).
2. **Question machinery** on histories B and C: an evolving fact ledger, new
   questions with checkpoint-correct ("evolving") ground truth, and type quotas.
3. **One full-system semi-LSREP replay** per history, live decay, every checkpoint
   store saved, every strengthening/decay event logged.
4. **Background-content quality audit** of everything the background models wrote
   into the full-system store (full ICE only, no ablation).
5. **Read passes** on the saved checkpoint stores: full ICE, recent-only, ICE
   warm-vector, bare vector, and the read-time additive and leave-one-out
   ablations.
6. **Answers and grading** for every read pass, plus fair retrieval diagnostics.
7. **LongMemEval oracle** (500 instances).

**D1a — Why this is a BETTER semi-LSREP than v2's (consolidated 2026-10-17; each row
points at the decision that does it).** v2's LSREP is the paper's method; Z2 keeps its
shape (replay a real history, ask questions at checkpoints, score and rank answers,
token efficiency) and fixes what was weak in it:

| v2 LSREP | Z2 | Where |
|---|---|---|
| ground truth written freely by a model (how v2's question set went wrong) | computed from a cited, two-family fact ledger, audited by v2's own forward walk | D4 |
| one fixed answer per question | evolving ground truth: the answer that was true at each checkpoint | D4 |
| questions of whatever kind the writer produced | type quotas (updates, temporal, aggregation, …) and difficulty by construction | D4, D11a |
| no check that memory is not over-used | no-memory control prompts at every checkpoint | D4b |
| hallucination percentage | invented-content and superseded-value flags anchored to required items | D11 |
| four answers ranked 1–4 once | blind ranking in two randomised orders, ties allowed, separate from the absolute grade | D11 |
| a source-ID or verbatim hit counted as recall | semantic credit — compressed memory judged on meaning, and wrong content flagged | D11 |
| pooled headline numbers, occurrences as samples | per history and question type; bootstrap intervals over question families | D11 |
| weak or absent baselines | bare vector and warm vector at the matched memory budget, plus recent-only | D10 |
| component value inferred from one configuration | additive and leave-one-out read-time ablations; pre-registered keep rule | D10, D11c |
| what the background models wrote never judged | background-content quality audit of every artifact kind | D9 |
| answering/judging models could overlap with writers | writers never answer or judge; every prompt qualified before use | D6, D6a |
| ad-hoc clock | declared session clock, live decay, every checkpoint store saved | D3, D8 |
| — | one pre-launch flow audit of the whole pipeline before the replay | D7b |

**D2 — Histories.** B (1,119 original pairs, long-range entity/revision memory)
and C (325 pairs, dense long-input/compression), frozen in
`logs/v3-offline-input-plan-2026-10-10-r4/final-originals-{B,C}.json`. Both are
excluded from tuning — **one recorded exception** (maintainer, 2026-10-17): the G77
conversation-slot prompt was qualified on dev + B + C, so Z2's numbers for those slots are
partly on data the prompt was tuned on (PROVENANCE 2026-10-17). C is only the first segment (ICE-1, 325 pairs, ~1.1M tokens)
of the stitched ICE project conversation (1,737 user turns, ~5.9M tokens, real
timestamps); Z2 keeps C as frozen for replay cost, and FINAL uses the full
stitched conversation. Their legacy v2 probes (B: 91 families/638 observations;
C: 37/154) are **replaced**, kept only as a historical reference: they were mostly
single-message lookups with end-state answers.

**D3 — Clock.** B's and C's timestamps are synthetic imports compressed into four
days each, so decay cannot act plausibly on them. Z2 replays on a **declared
clock**: sessions of 25 consecutive pairs; 3 minutes between turns inside a
session; between sessions a deterministic seeded gap of 1, 2 or 3 days. B is then
about 45 sessions over roughly three months, C about 13 sessions over roughly a
month. Real, authentic timestamps are used unchanged wherever a history has them.
The clock table (session → start time) is written before the replay and is a
USER-REQUIRED review. **Approved by the maintainer, 2026-10-18** (`experiments/z2/clock.py`,
policy `ice-z2-declared-clock-1`): any split is fine as long as the gaps stay even — never
small gaps then one very large one. B: 45 sessions over 90.2 days, C: 13 over 32.7 days; every
gap is 1, 2 or 3 days.

**D4 — Question machinery (built before Z2, reused by FINAL).**

- *Ledger first, questions second.* Two writer families extract, window by window,
  every durable fact, preference, decision, plan, piece of advice/procedure and
  every change to one (correction, reversal, update), each with exact citations:
  (turn, role, start, end, source sha256). Entries carry `valid_from` and, when
  superseded, `superseded_by` (the turn and entry that replaced it). Agreement
  between the two families marks an entry verified. A disagreement goes to a
  third-family tie-break, then to the review queue.
  *How the check works (2026-10-18, after planted controls):* the checker is shown an
  entry's subject and sentence only, never its speaker or modality, and reports what the
  turns show — who says it (user / assistant / nobody), how (asserted / suggested /
  hypothetical / not stated), and any contradicted detail; code compares (TRAPS 136; a
  yes/no "supported?" made MiMo follow whether it had a similar entry of its own).
  *Nothing is deleted (maintainer, 2026-10-18).* Every entry keeps a status: `verified`
  (the other family's observation matches), `tie_break` (kept by the tie-breaker only),
  `relabeled` (checker and tie-breaker both see the item, same speaker, every detail
  right, and agree on another modality — the written modality is kept beside it), or
  `unconfirmed` with its reason (`speaker`, `not_stated`, `wrong_detail`, `modality`,
  unanswered). Only `verified`, `tie_break` and `relabeled` entries build questions and
  ground truth; `unconfirmed` ones mark content that has *some* source, so an answer
  repeating one is flagged as such, never as invented (D9, D11).
- *Evolving ground truth: ledger first, v2's forward walk as the audit.* Changes
  are captured when they are extracted (a correction or reversal links to the
  entry it supersedes). Then v2's method runs as an independent check: for each
  family, a verifier reads the new turns between consecutive checkpoints and
  answers "does anything here change this answer? cite it", with v2's
  temporal-anchor guard so a later step cannot swap in a different event. A change
  the walk finds but the ledger missed (or the reverse) goes to the review queue.
- *Ground truth is computed, not written.* A question's answer at checkpoint t is
  derived from the ledger state at t: the entries valid at t, plus the history
  of superseded ones for "what did it used to be" questions. Each occurrence
  stores its reference answer, its required-content list (meaning, not wording,
  as in the 2026-10-13 content contracts) and its citations.
- *Types and quotas* (share of question families per history):

  | Type | Share | What it asks |
  |---|---:|---|
  | Fact recall (user- or assistant-stated) | 15% | a stated fact or decision |
  | Knowledge update / correction | 15% | the *current* value after a change |
  | Temporal | 12% | when, in what order, or as of a date |
  | **Cumulative mega-span** | 10% | reconstruct a whole thread from many entries spread over ≥ 100 turns: "list all the characters and their powers", "re-explain the whole plan", "the full timeline so far" (the shape of the maintainer's handwritten B questions); the answer grows and changes over time |
  | Aggregation across sessions | 8% | counts and narrow lists spread over time |
  | Procedure / advice recall | 10% | steps or advice given earlier |
  | Negative / abstention | 10% | never stated by t: "don't know" is correct |
  | Preference | 5% | stable likes, dislikes, constraints |
  | Multi-hop | 5% | two cited entries combined |
  | Messy-phrasing variants | 10% | an existing family re-asked with typos, lowercase, terse or rambling wording (style invariance) |

- *Fact-based ground truth.* An occurrence's reference is a set of required ledger
  items plus their qualifiers, not a model-written model answer. An answer is
  correct when it carries every required item's meaning (any wording) with its
  qualifiers and contradicts none; cumulative and aggregation questions are also
  scored by item coverage (how many of the required items, and any wrong extras).
- *How questions are asked.* The question writer phrases each question the way the
  user actually writes in that history (casual, lowercase, "do you remember…",
  "can you list down…"), never leaks the answer, and asks something answerable
  from the cited entries alone and unambiguous at its checkpoint. A verifier
  rejects questions that fail any of these.

- *Cutoffs and repeats.* A family is first asked only after its supporting turns
  have left the recent window (40 turns). Every fact-like family is asked at
  least twice (first eligible checkpoint, and again ≥ 150 turns later); 30% of
  families get a third ask after the longest available gap. Update families are
  asked before **and** after the change, so the right answer differs.
- *Checkpoints* are session boundaries on the declared clock (D3). A question
  never sees turns after its checkpoint, and its ground truth never uses them.
- *Volume targets:* B about 300 occurrences, C about 120, after the split (D5).

**D4b — No-memory control prompts (maintainer, 2026-10-15).** Besides the memory
question types, each checkpoint gets prompts that need **no memory at all**:
general knowledge, self-contained tasks, small talk and follow-ups answerable from
the recent window. They are not abstention items (those ask about the history and
expect "don't know"); they test the retrieve-or-not decision (B2 + source-need gate;
audit A11 found base_retrieve True on 518/519 development prompts before its fix)
and whether injected memory costs tokens or harms an answer. Reported per system:
retrieve rate, memory tokens injected, and answer quality against recent-only on the
same prompts. Their share and generation are decided at the Z2 build; they sit
outside the D4 quota table so the memory types keep their shares.

**D5 — Keeping FINAL clean.** Questions are generated over B and C and split **by
question family**: 60% to Z2, 40% sealed for FINAL (hash-recorded, never read,
answered or graded before FINAL). FINAL additionally covers the maintainer's other
v1 conversations with the same machinery. Whatever Z2 measures will be fixed
against, so Z2 families are development material for FINAL purposes.

**D6 — Model roles** (high-limit opencode profiles, maintainer-approved):

| Role | Model |
|---|---|
| Ledger/question writers (two families) | DeepSeek V4.1 Flash; MiMo V2.5/V2.6 Flash |
| Ledger verifier / tie-break | the non-writing family of the two, then LongCat-2.0 |
| Answerer | ~~GPT 6 Luna (`opencode-luna6`)~~ → ~~local `qwen3.8:27b`~~ (D6c) → **Muse Spark 1.2 Contributor (`muse12`)** (D6d, 2026-10-27) |
| Answer judge, background-content judge | Muse Spark 1.3 Contributor (`opencode-muse13`) |
| Second-opinion sample (background audit) | DeepSeek V4.1 Flash on 20% of judged items |
| Second answer judge | **FINAL only**, untuned, reported beside Muse |

Writers never answer or judge. If a profile becomes unavailable, the replacement
is recorded before use; no silent substitution.

**D6c — Local answerer; every cloud call at the end (maintainer, 2026-10-26).** The
opencode plan's limits are spend-based: each response reports an `allowance` cost, and the
per-model request table assumes a typical request. Z2 answers send a median ~25k input
tokens, so one Luna answer cost ≈ $0.0137 of allowance, ~5× a typical request; the
5-hour window (≈ $12) filled after 973 answers in 49 minutes and the monthly limit
(≈ $60, already 78% used by S1–S8) was reached after 1,493 of B's 4,864 distinct prompts.
Decision:
- **Answerer = `qwen3.8:27b`, Ollama (Q4_K_M GGUF, 17 GB), thinking ON, temperature 0,
  `num_ctx` 49,152, output cap = what the window leaves after the prompt, up to 16,000
  and never below 8,000 tokens (thinking + answer), `keep_alive` held for the stage and
  released after.** Every Z2 answer (S10 B/C, S12) uses it; the 1,493 Luna
  answers are moved aside (`s10-B.luna-answers/`), never mixed with local answers.
- **Fit, measured:** the largest of all 6,830 distinct B/C prompts is 37,539 tokens
  (ICE's context budget caps it; C ≤ 19,749), the largest oracle evidence set 22,228.
  The largest prompt answered at 19.0 GB peak VRAM. A prompt that cannot fit
  `num_ctx` with the output cap is recorded `too_long` — never truncated.
- **Speed, measured (laptop RTX 5090, 24 GB):** 25k-token prompt, thinking off 39 s;
  37.5k-token prompt, thinking off 58 s, thinking on ~100 s (~2,200 thinking tokens).
  Estimate for all ~8,800 Z2 answers with thinking on: ~7–9 days of GPU (≈ 4 days
  with it off), plus the ~2-day S12 replay — inside the ~13 days before the plan's
  monthly reset.
- **Why not vLLM/SGLang:** Qwen's official builds are BF16 (~54 GB) and FP8 (~28 GB);
  the 4-bit vLLM builds (NVIDIA NVFP4, AWQ-INT4) are ~21–22 GB because the 248k-vocab
  embedding/output layers stay full precision — with a 41k-token cache they exceed
  the 23.5 GB ceiling. vLLM's main gain (batching concurrent requests) needs cache
  room this card does not have.
- **Why thinking on:** Luna answered with reasoning (~1,000–2,000 reasoning tokens);
  turning Qwen's off would make the answerer change twice. It costs GPU days only,
  and the cloud calls wait for the reset either way.
- **Order:** every local stage runs first (S10 B, S10 C, S12 replay + reads, S12
  answers); every cloud stage (S11 B, S11 C, S12 grading) runs last, on one renewed
  plan period. D7's 4,000 cap was Luna's (reasoning included). Qwen thinks longest
  where memory is thinnest (a recent-only read: 2.2k-token prompt, 6.3k thinking-and-
  answer tokens), so a fixed 8,000 cap would turn exactly those failures into
  `incomplete` (ungraded, D11) and flatter the weakest conditions; the window-sized cap
  avoids that, and the report states the incomplete rate per condition.
- **What this changes:** Z2's answers measure ICE with a local 27B answerer, which is
  the configuration ICE itself runs (local-first). Within-study comparisons hold —
  every condition shares the answerer. Absolute scores are not comparable with any
  Luna-answered number.

**D6d — Answerer Muse 1.2; local stages first, then every cloud stage (maintainer,
2026-10-27).** Supersedes D6c's answerer. The laptop can run only some hours a day
(thermals), so D6c's ~7–9 GPU days of local answering would stretch into weeks; S10-B had
257 of 4,864 local answers when stopped.
- **Answerer = `muse-spark-1.2-contributor`** (confirmed in the provider's model list
  2026-10-27), Responses API, temperature 0, output cap 16,000 (reasoning included — D6c's
  reason against a tight cap holds: it would turn the weakest conditions' long answers into
  ungraded `incomplete`), four workers, no request pacing: the spend limit is met by
  waiting (TRAPS 148). The 257 Qwen answers are moved aside (`s10-B.qwen-answers/`);
  `answers.py` refuses to add answers from a different model to a saved file.
- **Judge unchanged:** Muse 1.3. Answerer and judge share a family (Meta). Accepted: the
  judge checks required items against quoted sources, not taste, and every condition
  shares the answerer, so any family leniency cancels in paired comparisons; it can move
  absolute levels only, which Z2 never compares outside the study. FINAL measures it
  (cross-family judge sample, blind human sample) and keeps Muse out of its ground truth.
- **Order:** local — S12 replay + reads; then cloud, on one renewed plan — S10 B, S10 C,
  S12 answers, S11 B, S11 C, S12 grading. Before the cloud stages: read the allowance cost
  of ~20 real Muse 1.2 answers and size the stage from it (TRAPS 148) —
  `experiments/z2/allowance.py` projects every answer stage from the saved answers'
  allowance (on Luna's 2,669 kept answers: median $0.0114, all Z2 answers ≈ $101).
- **Data use — accepted by the maintainer, 2026-10-27:** Muse Contributor models may
  train on prompts and completions and are not zero-retention (Data-use terms below). The
  2026-10-17 acceptance covered the judge, which sees cited source text; the answerer sees
  every full memory prompt (median ~25k tokens of B/C). **Muse 1.2 Contributor is accepted
  as the answerer on these terms, for Z2's B/C prompts and FINAL's private prompts.**
- **What this changes:** Z2's answers come from a cloud model again; nothing answered by
  Luna or Qwen is mixed in, and no Qwen- or Luna-answered number is compared with them.

**D6e — Z2 reports FINAL's scoring as its headline (maintainer, 2026-10-27).** Nothing is
graded yet, and FINAL's measures are counts over the same judge call, so Z2 becomes a full
rehearsal of FINAL's grading and report. `experiments/z2/scoring.py` (FINAL_PREREGISTRATION r2
§2–§4), as `report.json → scoring`:
- **M1** accuracy (correct; partial, incorrect and uncertain are not), **M2** supported current
  accuracy (correct, no outdated value, no confirmed-invented claim), by question type, by
  evidence distance (<32K · 32K–128K · 128K–512K · >512K tokens, `ledger.pair_tokens`) and by
  window side — Z2 reads no `long_context`, so out of window = distance > 37,500 tokens;
- **M3**: prompt tokens per answer and per correct answer, TUR (correct per 1,000 tokens);
  replay wall seconds per turn (model calls per turn are not in the trace — said so);
- **update accuracy** (questions with superseded values), **turn-cited evidence recall**
  (literal; compressed memory is credited by context precision), **context precision** (share
  of fragments carrying a required item / a wrong one; semantic credit now also reads
  `bare_vector`), the **primary tests** (`full` vs `bare_vector`: P1 M2 out-of-window, P2 M1;
  McNemar, family bootstrap 10,000, Holm, detectable difference), **hallucination
  non-inferiority**, and the **fidelity audit** per component (S8's items and faithfulness,
  prompt reach, prompts changed when switched off, M2 change → unexercised / contributing /
  defective / no measurable effect; over all questions — Z2's asks name no target component).
- Kept as planned, below it: the 1–5 score, paired 1–5 differences, ranking win rates,
  retention transitions, controls. Bare-vector units were invisible to the old token and
  fragment columns (TRAPS 149); `unit_index` now reads them.

**Data-use terms (read 2026-10-17, opencode.ai/docs/go).** MiMo, DeepSeek and LongCat on
Go: no training, 0-day retention (DeepSeek's zero-retention agreement is renewed monthly;
the one read was valid through 2026-10-31 — re-check before each stage that sends B/C
text). GPT 6 Luna: no training, abuse-monitoring logs kept up to 30 days. **Muse Spark 1.3
Contributor: prompts and completions MAY be used to train future Meta models, and it is not
zero-retention** — that is the price of its discount, and the judge is sent cited B/C
source text. **Accepted by the maintainer, 2026-10-17: Muse 1.3 Contributor stays the Z2
judge on these terms.** Each run manifest records the terms as read.

**D6b — Free tier first (maintainer, 2026-10-16) — NOT AVAILABLE TO THE HARNESS (measured
2026-10-17).** With a valid key, `deepseek-v4.1-flash-free` and `mimo-v2.6-flash-free` on Zen
both return `403 FreeTierError` ("OpenCode's free tier can only be used from within
OpenCode"); the Go `deepseek-v4.1-flash` and `mimo-v2.6-flash` answer. So every Z2 cloud call
runs on Go, and the free-vs-paid quality comparison the maintainer asked for cannot be run
through the harness. (Zen's terms also said MiMo-V2.6-Flash Free may use inputs to improve
the model.) The original rule is kept below as the record. OpenCode Zen
(`https://opencode.ai/zen/v1`) serves free variants of some selected models; use
them first and move to the OpenCode Go subscription (`https://opencode.ai/zen/go/v1`)
only when the free quota is exhausted. Catalog checked 2026-10-16:
`deepseek-v4.1-flash-free` and `mimo-v2.6-flash-free` exist and are used first.
**Muse Spark 1.3 Contributor (the judge) always runs on Go**, never its free variant —
the Go quota for it is generous, and one judge identity across all of Z2 keeps grades
comparable. LongCat 2.0 and GPT 6 Luna have no free variant (`longcat-2.5-preview-free`
is a different model, not a substitute), so those run on Go. Rules: (1) the free variant is
qualified under D6a like any profile — same model name on another endpoint is not
assumed identical (TRAPS 59); (2) every call records endpoint + model id; (3) a stage
(one writer pass, one checkpoint's judging) runs on one identity — if the quota runs
out mid-stage, the remainder runs on Go and a 30-item overlap is run on both to show
agreement, reported with the stage; (4) check the free tier's data-use terms before
sending B/C text, record them in the run manifest.

**D6a — Every prompt is qualified before use.** Each writer, verifier,
question-phrasing and judge prompt is run on the development history first and
passes measured checks before it touches B or C: citation pass rate, two-family
agreement, question rejection rate, planted-control accuracy for judges (a
deliberately wrong but on-topic answer must not pass), plus a maintainer-reviewed
sample of its outputs. A prompt that fails is revised and re-qualified; the
qualification record is kept with the prompt's hash.

**D7 — Answer output cap.** Raise the answer output reserve from 1,500 to 4,000
tokens for Z2. Evidence: on the tuning answers, one question failed in 6 of 9
arms, and its successes used 1,427–1,492 tokens, about 1,030 of them reasoning.
This is a new frozen answer identity; tuning answers are not mixed with Z2 answers.

**D7b — Pre-launch flow audit (maintainer, 2026-10-17).** When the Z2 machinery is
built and before the S7 replay starts, one small audit traces a prompt through
every section once, on the frozen build: chat-command check → classification →
memory decision → each retrieval leg → fusion, reranker, D7 bonus, budget →
prompt assembly (recent window, slots, notes, facts skip) → the answer call →
post-flight (density, summary/claim pass, facts stage, procedural, decisions) →
the maintenance jobs on their cadence. At each hop: what goes in, what comes out,
and does the next hop receive exactly that (no silent default, no dropped field,
no inert switch). Read the actual prompts and rows, not the code. Findings are
fixed (reviewed production change) or recorded in PROVENANCE before launch.

**D8 — Replay.** One full-system semi-LSREP replay per history with the frozen
tuned configuration (spec §5.2 decisions): recorded original replies, full ICE
preflight and post-flight per historical prompt, all nine periodic maintenance jobs
(`seed_v3.MEMORY_JOBS`; ten before CODEX_V4 deleted `compaction`) on the declared clock, **live default decay** (option A). The replay logs every
strengthening (row, source time, amount, turn or probe) and every decay pass.
**Every history turn goes through the whole answering pipeline (maintainer,
2026-10-18 — this is what makes Z2's semi-LSREP an improvement over v2's).** For each
historical user prompt: classification → memory decision → source gate → every
retrieval leg → fusion, reranker, bonuses, budget → the final prompt fragments, with
retrieval exposure and strengthening **written** exactly as in live use; the pipeline
stops before the answer model, and the recorded original reply is stored and
post-flighted. So decay, strengthening, usage and everything downstream of them evolve
from the conversation itself, not from the evaluation's questions (in v2 only the
probes touched retrieval state). Verified on the Z2 S0 profile (B turns 1–150, no
questions asked): 133/150 turns accessed (1,396 accesses), 165 facts with usage, all 150
history turns with exposure writes on (`scripts/z1/seed_v3.py`).
**Questions never write:** each question retrieves like real use but in a read-only
transaction that is rolled back (exposure logged and tagged, never applied), so asking
cannot change the memory it measures (TRAPS 80). Every checkpoint store is saved. Speed fixes (D1.1) come first, measured on a short
slice of B, and never change the stored outputs' identity silently.

**D9 — Background-content quality audit** (full-system store only). Every kind of
artifact the background models write is inventoried from the actual writers
before judging. At minimum: turn summaries and abstracts; conversation folds;
batch summaries; **codex facts** (the sentence, speaker, modality and the
source's subject/relation/object — CODEX_V4) with their entity links; entity
resolution and automatic entity merges (D12); same/replaces/both-true decisions
and expiries (D6); cluster names and memberships; procedural patterns; session
summaries. **Since G77 (2026-10-17), also:** reflection's conversation-tier
`conversation_focus` / `pending_items` slots (quoted, NLI-checked claims — judged
faithful/attributed/current like facts), the maintenance agent's self-applied
stale-slot removals, and tier-2 coding-decision verdicts (superseded vs both-stand —
D6's framing, never qualified on decisions). All are `applied` rows in
`review_queue`, so the inventory reads them there. (Since CODEX_V4, 2026-10-16, there are no conversation edges, entity
notes or compaction snapshots to audit.) For each kind, a stratified sample (all items if fewer than
100, else 100 per kind per history) is judged with its cited sources supplied:

- **faithful** — supported by its sources; nothing invented or contradicted (the
  hallucination check on background output; content matching only an `unconfirmed`
  ledger entry is flagged *unconfirmed source*, not invented — D11);
- **attributed** — the right speaker; an assertion kept separate from a suggestion;
- **current** — not presenting a superseded fact as current (checked against the ledger);
- **complete** — required detail kept (summaries, folds, notes);
- **decision valid** — merges, expiries, supersessions match the ledger.

Codex facts and ledger entries are also matched directly, giving fact precision
and recall against the verified ledger (the ledger's `asserted|suggested|hypothetical`
maps to the fact's modality). Each artifact kind also reports how often
it reached a prompt and the graded answers it appeared in. Muse judges; DeepSeek
V4.1 Flash re-judges a 20% sample for agreement; disagreements are listed for the
maintainer. This settles the open quality owners G49, G57, G60, G63, G67, G69,
G73, G74, G75 and G76 at Z2 scale.

**D10 — Read passes and ablations.** On every saved checkpoint store, with that
checkpoint's questions:

- *Controls:* recent-only; ICE warm-vector; bare vector (whole role-tagged
  messages filling the matched memory budget — frozen development choice,
  PROVENANCE 2026-10-14).
- *Ablations:* the twelve groups of V3_POST_RESEED_REPAIR §2.5 as **read-time
  switches**, re-pointed for CODEX_V4 (2026-10-17): groups 3 + 4 become **one codex
  group** — the facts leg (keyword, meaning and entity channels, the predecessor
  clause, the D7 strength bonus, grounded query expansion); group 10 keeps temporal
  filtering and the facts' as-of validity (the timeline leg is gone); group 12 is
  reflection and entity merges (enrichment and compaction are gone; D6 reconciliation
  is a write-time decision — report it through D9, it has no read-time switch). Additive from bare vector up to full ICE in the frozen order;
  leave-one-out from full ICE, restoring between removals. Semantic maintenance is
  its own condition. All ablation conditions read with the same **neutral decay
  overlay**; decay is reported separately as its own condition and as settings
  sensitivity via overlays (D8 event log). Label: read-time ablation on a fully
  seeded store; indirect write-time effects are not removed.
- *The groups, as they stand after CODEX_V4 (copied here from V3_POST_RESEED_REPAIR
  §2.5 so this spec is complete; that section stays the historical record),* in the
  frozen additive order:
  1. BM25 / hybrid candidate search.
  2. Reranker.
  3. **Codex facts** (old groups 3 + 4): the facts leg's keyword, meaning and entity
     channels, the predecessor clause, the D7 strength bonus, grounded query expansion.
  4. Supported turn summaries and abstracts — including the summary guard
     (`representation_guard_terms`).
  5. Rolling conversation and batch summary memory, with source-mode notes.
  6. Procedural extraction, activation and retrieval.
  7. Learned / source-proof retrieval gating.
  8. Cluster organisation and scope/ranking effects.
  9. Temporal query filtering and the facts' as-of validity (source dates stay in every
     condition; switching this off never admits future evidence).
  10. Decay / retention / cold lifecycle (read with the neutral overlay in every other
      condition; reported as its own condition).
  11. Semantic maintenance: reflection (G77 conversation slots) and entity merges; D6
      reconciliation is write-time and reported through D9.
- *Rules (from §2.5):* every condition shares one answer envelope (same question,
  formatting, recent window, system instruction, answerer, decoding, output reserve and
  total token cap; expected answers never enter it). Additive starts from bare vector
  and adds one group at a time in the order above, frozen before any Z2 answer is seen;
  order-dependent gains are conditional contributions. Leave-one-out starts from full
  ICE each time, removes one group, then restores — never cumulative; a parent switch
  that necessarily disables dependents is named as a compound removal. Components are
  enumerated from actual reachable consumers before launch, each switch is proved to
  change only its own component (§5 item 6), identical configurations are deduplicated
  by manifest hash, and the exact arm/call/cost count is produced before launch. All
  requested conditions run; none is dropped for being cheap or losing.
- *Warm vector* = ICE's own vector leg with its gate, reranker and representation
  policy, no other legs. *Bare vector* = an independent append-only index of every
  original role-tagged chunk at the cutoff, same embedder, cosine top-k, no ICE
  mechanism; tuned on development data only (PROVENANCE 2026-10-14), never weakened.
- *Summary guard pair (maintainer, 2026-10-18):* the default reads with
  `representation_guard_terms = specific` (a summary stands in for its raw turn only if it
  keeps the question's names and numbers). A stratified sample of 60-100 questions across
  all checkpoints and question types is also read with `every_word` (the old guard, which
  demanded every question word) and both are answered and graded. Pre-registered: the
  default stays unless `every_word` wins beyond the noise floor on some question type.
- Identical assembled prompts reuse one cached answer.

**D11 — Scoring.**

- *Answers:* Muse absolute grades (correct, partial, incorrect, uncertain) against
  the checkpoint's required items, qualifiers and cited sources; the arm is never
  disclosed.
- *LSREP-native scores (v2, reformed):*
  - **1-5 score** per answer, anchored to required items: 5 all items and
    qualifiers, no errors; 4 one minor omission; 3 partial; 2 mostly missing or
    stale; 1 wrong or invented. Separate flags: used a superseded value; invented
    content (replaces v2's hallucination percentage); abstained.
    **Invented** means found in neither the checkpoint's originals nor any ledger entry
    of any status; content matching only an `unconfirmed` entry gets its own flag,
    *unconfirmed source*, and is not counted as a hallucination (maintainer,
    2026-10-18). The same rule applies to D9's *faithful*.
    **A ledger miss is not a hallucination (maintainer, 2026-10-18).** Content that
    matches no ledger entry is checked against the originals before it counts as
    invented, the same way abstention ground truth is checked (§2): an embedding search
    and an exact keyword search over the originals up to the checkpoint, then the
    verifier reads what they return. Found → *ledger miss* (counted against the ledger's
    recall and reported, not against the system under test); not found → *confirmed
    invented*. Applies to answers (D11) and to background output (D9).
  - **Ranking** of the conditions per question (v2 ranked four answers 1-4):
    blind, order randomised and repeated in two orders, ties allowed, judged
    separately from the absolute score; reported as win rates against bare vector
    and warm vector.
  - **Token efficiency** (v2): TUR = 1-5 score per 1,000 injected tokens, SPF =
    score per retrieved fragment, plus quality at matched budget; full prompts
    counted on both sides. Reported beside quality, never instead of it.
  - **Item coverage** for cumulative and aggregation questions.
- *Retrieval diagnostics:* source-ID credit (existing); verbatim span presence
  (existing); and **semantic credit**: every graph fact, claim or summary in the
  prompt that cites a gold turn is judged for "carries requirement X (meaning,
  not wording)" and for wrong content. This lets the diagnostic see compressed
  memory instead of rewarding only raw text.
- *Reporting:* per history and per question type; never one pooled headline.
  Paired comparisons against bare vector and warm vector; bootstrap intervals over
  **question families**, not occurrences; repeats reported as retention
  transitions. Incomplete answers are ungraded, never counted as wrong.

**D11a — Difficulty by construction (maintainer, 2026-10-15).** Tuning's
development set was near ceiling (full ICE 61/67, ICE vector-only 63/67), so it
could not show what ICE's machinery adds. Z2 must be hard enough to separate
systems, and the difficulty must come from how questions are built, never from
selecting questions a competitor happens to fail:

- weighted toward what plain retrieval cannot do from one message: updates and
  corrections (a stale value is retrievable and wrong), cumulative mega-span and
  aggregation (more items than one budget of raw messages holds), temporal order,
  abstention, and long-gap retention repeats;
- questions asked late in B and C (730K-1.2M tokens), so the history far exceeds
  any memory budget;
- **ceiling check on the development dry run:** a question type on which full ICE
  scores above 85% is made harder for every system (longer gaps, more required
  items, later cutoffs) before B/C generation. The check uses full ICE's score,
  never a competitor's failure.

**D11b — Token evidence: budget curves.** ICE may help by needing fewer tokens
for the same quality. Full ICE, ICE warm vector and bare vector are each read and
answered at four memory budgets (2k, 4k, 8k tokens and the derived default) on
every Z2 question. Report quality per budget and the smallest budget at which
each system reaches its own default-budget quality. Read passes are local and
identical prompts reuse answers.

**D11c — When a component earns its place (pre-registered, maintainer
2026-10-15).** A component keeps its place only if, in the leave-one-out
ablations, removing it costs answers beyond the noise floor (repeated identical
prompts, measured in Z2 as in tuning) on at least one question type, **or**
keeping it saves ≥ 10% of tokens at equal quality (D11b). Otherwise it is
reported as "no measured benefit", and the default becomes simplification, which
is a maintainer decision. ICE as a whole is judged the same way against bare
vector and warm vector: better answers beyond noise on some type, or equal
answers at ≥ 10% fewer tokens.

**D12 — LongMemEval oracle.** As V3_POST_RESEED_REPAIR §2.6: pinned release, all
500 instances, sessions sorted and checked, isolated per-instance replay; full
ICE, ICE warm vector, bare vector, and the all-evidence oracle control with the
same answerer. Muse grades; results are labelled **within-study, non-official**
(the original GPT-4o judge is not used). Native question types are kept.

**D13 — FINAL handoff (direction, specified at Z3).** FINAL reuses the question
machinery, replay runner, read-pass runner, scoring and background audit without
redesign. FINAL's inputs: the sealed B/C families; the maintainer's other v1
conversations (private, reported separately, never published raw); public and
synthetic data as the reproducible part; the second judge. FINAL's data roster
and public-dataset adapters are decided in Z3, not here.
*Direction for FINAL (maintainer, 2026-10-18):* the ledger tie-breaker should **re-extract**
a disputed item from the cited source itself — an independent third reading — rather than
only choose between the two writers' versions.

## 2. Algorithm and data model

**Stages and artifacts** (each stage resumable, each artifact immutable and
hash-bound to its inputs; private content stays under `logs/`):

| Stage | Output | Calls |
|---|---|---|
| S0 speed profile and fixes on a 50-pair slice of B | profile report; fixes as reviewed production changes | local only |
| S1 ledger extraction, both writer families, 25-pair windows with 5-pair overlap | `ledger-raw-{writer}.jsonl` | ~2 × (45 + 13) windows |
| S2 ledger merge, citation check, tie-break, review queue | `ledger.json`, `review-queue.json` | disagreements only |
| S3 question families and cutoffs from the ledger by quota | `families.json` | ~1 per family |
| S4 occurrence ground truth computed from ledger state | `occurrences.json` | none |
| S5 maintainer review (USER-REQUIRED) | `review-decisions.json` | — |
| S6 Z2/FINAL split by family, FINAL part sealed | `split.json` + sealed hash | none |
| S6b pre-launch flow audit (D7b): one prompt traced through every section | audit note in PROVENANCE; fixes as reviewed changes | local |
| S7 replay B and C on the declared clock | checkpoint stores, event log | local |
| S8 background-content audit | `bg-audit.json` | Muse + sample |
| S9 read passes (controls, ablations, decay overlays) | per-condition contexts | local |
| S10 answers | per-condition answers | Muse 1.2 (D6d) — cloud, after every local stage |
| S11 grading and retrieval diagnostics | grades, semantic credits, report | Muse — runs last (D6c) |
| S12 LME oracle | per-instance contexts, answers, grades | replay + reads local; answers Muse 1.2, grades Muse 1.3 (D6d) |

**Ledger entry:** `{id, history, kind (fact|preference|decision|plan|advice|procedure|event), subject, value, speaker, asserted|suggested|hypothetical, citations[], valid_from_turn, superseded_by?, writers[], verified}`.

**Question family:** `{id, history, type, ledger_entries[], phrasing_variants[], split}`.

**Occurrence:** `{family, checkpoint_turn, clock_time, question, reference_answer, required_content[], qualifications[], citations[], expected_abstention, repeat_index}`.

**Citation check (mechanical, every entry and occurrence):** the cited span exists
byte-for-byte in the original role text; its turn is at or before the checkpoint;
the source sha256 matches. A failure removes the item; it is never repaired by
hand-editing the model output.

**Abstention ground truth:** an occurrence is a valid negative only if neither
writer's ledger holds a matching entry up to the checkpoint, **and** an embedding
search over all originals up to the checkpoint returns no passage the verifier
judges as answering it.

**Call budget** (per 5-hour limits, maintainer's opencode plan): S1–S3 about
2,000–3,000 writer calls; S8 about 2,000 judge calls; S10/S11 about 10,000 answers
and 10,000 grades across conditions (identical prompts deduplicated); S12 about
2,000 answers and grades. Writers and Muse have five-digit limits; Luna (4,230 per
5 hours) is the pacing limit, so answering runs in resumable batches.
*(Superseded 2026-10-26 by D6c: the limits are spend-based, Luna reached the monthly
limit at 1,493 answers, and answering moved to the local model.)*

**Compute** (from the 5.5 h, 251-pair development replay, before speed fixes):
replay B + C ≈ 26 h; LME oracle replay ≈ 4 days (≈ 11,000 messages); read passes
≈ 1 day. Speed fixes (S0) are measured before committing to these.

## 3. Files and integration points

New package `experiments/z2/` (research code; production stays in `src/`):

| File | Holds |
|---|---|
| `clock.py` | declared session clock (D3) and its table |
| `ledger.py` | window extraction prompts, merge, citation checks, tie-break |
| `questions.py` | family generation by quota, cutoffs, repeats, phrasing variants |
| `ground_truth.py` | occurrence answers from ledger state; abstention checks |
| `split.py` | Z2/FINAL family split and seal |
| `replay.py` | full-system replay on the declared clock; event log (wraps `chronological_replay.py` and the decay/strengthen writers) |
| `decay_overlay.py` | per-profile decay from the event log, applied to restored copies |
| `bg_audit.py` | artifact inventory, sampling, judging, ledger matching |
| `read_passes.py` | conditions on checkpoint stores (reuses `run_tuning_contexts`/`run_full_arm_coordinates` patterns and `bare_vector.py`) |
| `score.py` | answers/grades via the existing Luna/Muse paths; semantic credit |
| `oracle.py` | LongMemEval oracle adapter (V3_POST_RESEED_REPAIR §2.6) |

Production changes (maintainer-approved 2026-10-14, each gated and reviewed):
read switches for graph reading, turn summaries, conversation fold, procedures,
learned gate, cluster scoping, maintenance outputs and decay neutrality; the
answer output cap (D7); speed fixes from S0.

## 4. Failure modes and resumability

**Who runs what (maintainer, 2026-10-18).** The long runs — S7 replay onward (S8, S9,
S10, S11, S12) — are launched by the maintainer, not by a session, through one command:
`experiments/z2/run_z2.py run` runs every built stage in order (or `--only` one stage),
resumes after any break, crash or stop, and shows the work and time left per stage and
overall (`run_z2.py status --watch` from any terminal). A production run meets failures a
test run never does, so the runner never just stops: missing prerequisites (Postgres,
Ollama, disk) are waited for; a stage process that dies is resumed after a back-off;
only three failures in a row with no progress mark a stage stuck — its dependents are
blocked, independent stages still run, and the error is kept in the run log. Every new
long stage is built as resumable units behind this runner. FINAL reuses it, so starting
FINAL is a command, not a build.

- A writer returns uncited or wrongly cited entries: dropped by the mechanical
  check, counted, never silently fixed.
- Two writers agree on a wrong entry: the S5 review sample and the background
  audit's ledger matching are the catch; report the measured disagreement rate.
- A question leaks future information: the cutoff check refuses it.
- Phrasing variants change the answer: kept, and reported as a style-invariance
  failure (G28), not removed.
- Decay overlay cannot reproduce a cold move: that profile is refused for that
  row, never approximated silently.
- A profile hits its 5-hour limit: the run pauses with saved units; resuming
  the identical command continues.
- Every stage writes durable, content-hashed units; an identical rerun resumes or
  verifies; changed inputs require a new output directory.

## 5. Validation checklist

1. **Machinery dry run on the 251-pair development history first.** Its ledger and
   questions are compared with the 67 reviewed content contracts: how many reviewed
   facts the ledger finds, and how many generated ground truths a reviewer rejects.
   Development data never enters Z2 or FINAL results.
2. Citation checker: planted wrong offset, wrong turn, future turn and paraphrased
   span all refuse.
2a. Prompt qualification (D6a) passed and recorded for every generation, verifier
   and judge prompt, including the ranking and 1-5 rubrics.
3. Ground truth: a planted correction yields different answers before and after it.
4. Split: sealed FINAL families are unreadable by every Z2 stage (test asserts the
   Z2 code paths never open them).
5. Replay: a short B slice reproduces the declared clock, decay passes and event
   log; decay overlay on a restored copy reproduces the live scores exactly for
   the default profile.
6. Read switches: each turns off only its own component (prompt diff on a fixed
   checkpoint), and neutral decay changes only decay-dependent ranking.
7. Background audit: the Muse/DeepSeek agreement rate is reported before any
   quality number is quoted.
8. Full smoke suite (disposable database) before the long runs.

**USER-REQUIRED:** approve this spec; review the S5 queue; approve the clock
table (D3); launch the long replay, answer and grading runs; confirm profile
availability.

## 6. Look-ahead

- FINAL reuses every stage unchanged except data roster, sealed families and the
  second judge (D13). Anything Z2 hard-codes to B/C is a parameter.
- The same machinery can later replay other memory systems through the same
  questions; nothing in the stages assumes ICE internals except S7–S9.
- Public datasets (FINAL) need their own adapters; the ledger format is chosen so
  a synthetic planted-fact ledger can be loaded directly.

## 7. Traps

- **TRAPS 32:** a metric that cannot see a subsystem credits nothing to it.
  Semantic credit (D11) exists for this reason.
- **TRAPS 128:** confirm a switch reaches a decision before measuring it
  (validation 6).
- A source-ID hit is not the required content; verbatim presence is not meaning.
- A model grading its own writing: writers never answer or judge (D6).
- Ground truth written freely by a model is how v2's question set went wrong;
  here it is computed from a cited, two-family ledger.
- More checkpoint observations are not more independent samples: intervals are
  over question families.
