# Handoff — 2026-08-21, ~16:45 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Overwritten every session,
committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

> **⚑ READ [G52](ROADMAP.md#g52) AND [TRAPS #43](TRAPS.md) BEFORE TRUSTING ANY
> RETRIEVAL NUMBER FROM THIS CYCLE.** Both evaluation harnesses call
> `orchestrator.retrieve()` with `scope=None`; production passes a populated
> scope. That switched the batch-summary leg **off** in every measurement and
> changed what every other leg returned. **The fix is small. The re-run is the
> work, and it comes first.**

---

## TOLD → DID

**Told:** run the two-arm NER re-seed (A9b), answer the probes, re-score and
re-judge.

**Did:** all of it, plus far more measurement than was asked for, plus one
retraction that invalidates the retrieval half. Both arms seeded, drained,
snapshotted, scored, answered and judged. A codex truth-judge and a codex
ablation were built and run because the arm comparison kept raising questions the
existing instruments could not answer. **Then the harness defect was found, and
it retracts every retrieval table.** The store-level findings survive.

## ⚑ START HERE — THE STATE OF THE WORLD

| | |
|---|---|
| **Store** | arm B (`ner-b-nuner`) live — 293 turns / 6,271 entities / 10,228 edges / 65 procedural / 3 summaries. |
| **Snapshots** | `ner-a-micro` (arm A, WITH summaries) · `ner-b-nuner` · `ner-a-micro-presummary` · `arm1-post-g50` (the "before"). |
| **Git** | Clean. **Local only — the Z1 freeze holds.** 17 commits this session. |
| **Tests** | smoke 183/183 · `test_codex_write_path` 32/32. |
| **CLAUDE.md** | 408 → 340 lines. Git rules and session-closeout rules moved verbatim into `.claude/skills/ice-git/` and `.claude/skills/ice-session-closeout/`. |

## 1. ⚑ WHAT IS TRUE, AND WHAT WAS RETRACTED

Full evidence and confidence levels: **[PROVENANCE.md](PROVENANCE.md)**, entries
2026-08-20 and 2026-08-21.

**SURVIVES — measured against the store directly, not through the orchestrator:**

- **Only ~20% of stored triplets are TRUE; ~25% are merely REVERSED** (right
  entities, right relation, backwards). n=200 per arm, judged against each
  triplet's own source turn.
- ~~**`extraction_confidence` is INVERTED against truth**~~ ⛔ **WITHDRAWN 2026-08-23** — a 40-triplet subgroup quoted without an interval. Re-measured over 124 turns on two seeds, the two runs disagree on the DIRECTION (grounded 22.3% vs 12.2%; rejected 15.0% both). The field is UNINFORMATIVE, not inverted. Original text follows:
- **`extraction_confidence` is INVERTED against truth** — grounded 0.9 is 14–15%
  correct, rejected 0.35 is 22–25%. Independently in both arms.
- **A9b settled: NuNER improves FORM, not OUTCOME.** Malformed triplets halved
  (25.0→12.5%), non-entity nodes 58.7→40.7%, ~1,640 junk nodes vs ~3,400. But
  correctness 18.0→20.0% (inside 1 SE) and answers 31–30 — both real nulls.
- **64–68% of entities carry exactly one edge**, and 41–59% of those are not
  entities at all.
- **A12 already ruled out "use a bigger model"**: eight models, 4B→26B, the 26B
  ranked third, defects present in all seven arms ⇒ prompt/design, not capacity.

**RETRACTED — measured off the production path (G52):**

- ~~batch summaries never reach the prompt~~ — the harness disabled the leg.
- ~~a partnerless leg cannot win RRF~~ — a mechanism invented to explain that zero.
- ~~`summary_synthesis` = 0.303~~ — measured with the summary leg off.
- **The codex ablation's absolute verdict.** Its internal ordering stands (all
  three conditions shared the defect): dropping codex won 25–14, disabling it
  won 36–17. **Whether that survives a correct scope is unknown.**

## 2. WHAT WAS FIXED AND SHIPPED

| what | commit |
|---|---|
| batch summariser — token-budgeted batching + per-batch isolation (it was sending 37,359 tokens at a 32,768 window and one 400 killed the whole pass) | `11fc18f` |
| codex-grounding NER is a selectable tier | `8babddc` |
| per-consumer NER labels (`concept`/`object` restored for codex only) | `e027956` |
| the answer judge saw a **median 12.7%** of the gold — caps removed, source rebuilt from the store | `d2c7944` |
| `finish_arm.sh` read a pipeline's exit code, so its abort could never fire | `1dea4e5` |
| CLAUDE.md → skills | `236c396` |

## 3. NEXT — IN THIS ORDER

1. **[G52](ROADMAP.md#g52) — fix the harness scope and RE-RUN.** Decide first
   whether to fix the harnesses or make `retrieve()` fall back to its own
   `conversation_id` parameter — **they are not equivalent; the second changes
   production and is USER-GATED.** Then re-run: both arms' typed scores, the
   codex ablation, the per-leg ablation. **Everything else waits on this.**
2. **Then re-read the codex verdict.** If codex still loses under a correct
   scope, the graph-repair queue is not the priority. If it wins, it is.
3. **The graph-repair queue** (only after 2): direction check (~25% of edges) →
   stop using `extraction_confidence` as a truth prior → junk-entity filter →
   malformed shape gate → G51 linking (**prune before linking**).
4. **Cheap and unanswered:** split `wrong` (37–38%) into hallucination vs
   provenance drift — 14.8% of edges have a subject absent from their source
   turn. Sets the real cleanup ceiling.
5. **Old-pipeline comparison** — `score_typed` on `arm1-post-g50`, ~15 min, no
   LLM. ⚠ Report with the `merge_key` payload asymmetry (~9 redundant tokens per
   injected entity) and note it conflates the six fixes, G50 and the NER.
6. **A product idea worth an entry (user, 2026-08-21):** gate legs on whether
   their data exists yet. A new user's store has no summaries, no clusters, no
   decay and a thin graph — if those legs cost budget and return nothing, every
   new user gets worse answers than plain search would give.

## 4. WHAT IS STILL UNMEASURED — do not infer these from anything above

- **The whole write path.** The earned-lossless density decision (164 of 293
  turns lossless, 129 summarised), the B2 retrieve/don't-retrieve gate, the
  context ledger. Never under test.
- **Decay, reflection, the maintenance agent.** None run during a seed —
  `decay_score < 1.0` is **0 rows**. Every number describes a store that has
  never aged.
- **Graph enumeration.** Cue-gated; no probe in the 444 triggers it. Its 0.000
  delta is *untested*, not a null — see the two-kinds-of-zero note in PROVENANCE.

## 5. THE INSTRUMENT RECORD, WHICH BOUNDS EVERYTHING

**Eight measurement defects in one cycle, none of which raised an exception** —
[TRAPS #41](TRAPS.md), [#42](TRAPS.md), [#43](TRAPS.md). Three interpretations
were written and withdrawn. The answer comparison alone reversed twice.

**The two rules that actually caught things:** write the assert that would FAIL,
and prove the instrument DISCRIMINATES before spending the run. **The one that
would have caught G52 and did not exist:** diff the harness's call against
production's **argument by argument**, then run one request each way and diff the
returned `source_type` counts.

**⚑ Do not write this file, or close a session, without the user saying so.**
