# Handoff — 2026-08-27

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Overwritten every session,
committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

> **⚑ START AT [`scripts/z1/README.md`](../scripts/z1/README.md) §2.** The
> reseed is NOT next. **Ten pre-reseed items are, and none of them is blocked.**

---

## TOLD → DID

**Told:** start at `docs/specs/RESEED_PLAN.md` — seed 1,000–1,500 dense turns
and judge the new baseline.

**Did:** did not seed. The reseed's *inputs* turned out to be wrong or unproven,
so the session became: fix what the reseed depends on, then decide the
background model it would run under. **The reseed is now correctly specified and
starts from a clean base — which it would not have done yesterday.**

## 1. ⚑ WHAT IS TRUE NOW

| | |
|---|---|
| **General background model** | ⚑ **`gemma4:e4b`** — decided on measurement, 11 candidates × 9 jobs. Wins or ties **every** quality-judged job |
| **Codex extractor** | `NuExtract3-Q8_0`, unchanged (G63) — now with its **own** setting, `codex_extraction_model` |
| **Turn summary faithfulness** | **95.7%** (gemma, softened prompt) · 82.9% before · `qwen3:4b-instruct` 54% |
| **Reseed corpus** | **1,471 turns** — `bb558b5f` (1,119) + `ecc64aab` (251) + `355a5709` (101). ⛔ `cca73c87` dropped: it **IS** `bb558b5f`'s tail |
| **Probes** | **618 distinct**, one schema, 26 cross-conversation duplicates removed |
| **Store** | ⚠ `dir-false-run1` no longer matches its `.counts` — see §5 |
| **Tests** | smoke 183/183 · settings freeze + dynamics invariants 164/164 |
| **Git** | clean tree except this session's work, local only |

## 2. SHIPPED (`src/`, 6 files)

1. **`codex_extraction_model`** — extraction gets its own pin. One background
   setting served **ten** jobs; only extraction wants a specialist. Pointing the
   shared pin at NuExtract3 would hand a JSON-template filler to the summariser.
2. **`codex_extraction_mode` default → `template`** — the reproducibility
   argument for `instruct` expired with G72.
3. **The summariser prompt is faithfulness-tuned** — must-preserve block **and
   its retry twin**. gemma fabrication 15.7% → 2.9%. ⚠ **model-coupled.**
4. **`release_bg_model()` + runtime hook** — ICE unloads its own background
   model when the queue empties. Verified live.
5. **Procedural's silent idempotency return now logs** (G61's sixth path).
6. **Two lying comments fixed** — `extract_key_terms` claimed MicroNER (it has
   been NuNER since A9b); `.env`/scripts claimed the 26B pin.

## 3. ⚑ THE THREE FINDINGS THAT MATTER

**a. The summary trust gate is INVERTED.** `summary_coverage` decides whether a
summary REPLACES the raw turn. Judged with a gated judge: **13 of 13 fabricated
summaries cleared the gate; only 23 of 29 faithful ones did.** Fabricated
summaries score *higher* coverage inside every model. Moving the threshold
cannot fix it. ✅ Raw text is never lost — this corrupts one assembled prompt,
not the store. **Deferred to the reseed on purpose** ([G75](ROADMAP.md#g75)),
with the decision rule pre-agreed.

**b. The verbatim instruction MANUFACTURED the fabrication.** Ordering a model
to include terms it cannot ground makes it invent context to carry them — and
coverage rewards that. Fixed for gemma. ⚠ The same fix is a **loss** on qwen
(54% → 34% faithful): the background prompts are now **coupled to the model**,
exactly like extraction is to NuExtract3.

**c. The conversation fold is the worst thing in the layer** —
[G73](ROADMAP.md#g73), 33–67% fabricated, n=12/model. `qwen3:4b-instruct`
produced **0 faithful folds of 12**. It **compounds by design**: each step
re-summarises the previous summary. Runs every 2 h. Nothing had ever looked at
it.

## 4. ⛔ RETRACTED THIS SESSION — read before citing anything

- **Cluster naming (62–76% wrong).** The harness fed the namer *five arbitrary
  consecutive turns* with `recurring_entities=None`; production passes
  similarity-grouped **members** plus a recurring-entity hint. It measured the
  harness. The rate, the model ordering **and** the prompt A/B null are all
  void ([G74](ROADMAP.md#g74)).
- **"171 of 174 probes have no lexical signal"** — an invented threshold.
  Calibrated against known-good probes, it rejected most of those too
  ([TRAPS #51](TRAPS.md)).
- **A judge calibration** — `judge_summaries.py` truncated the source at 4,000
  chars, a bug **this repo had already fixed and documented** in
  `judge_answers.py` ([TRAPS #41](TRAPS.md) recurrence).
- **Two prompt A/Bs are nulls** — must-term count (premise wrong: the cap never
  binds, median demand is 11 not 25) and the naming contradiction.

⚑ **The noise floor, measured twice by accident: ±4–5 pts at n=70, ±9.5 at
n=21.** Identical arms re-run scored 37.1%/32.9% and 0.286/0.381. **Both nulls
above are exactly that size.** Quote run-to-run variance before quoting a delta.

## 5. STORE + ENVIRONMENT STATE

- `dir-false-run1`: `procedural_memory` 43→0, `batch_summaries` 2→0,
  `batch_summary_id` cleared, procedural idempotency keys deleted — the
  store-backed bake-off jobs reset themselves so each model started identical.
  Episodic/codex/chunks **untouched**. Backup
  `bakeoff/PRE_RESET_BACKUP.json`; restore `snapshots/dir-false-run1.sql`.
  **Deliberately not restored** — G72 wipes it.
- ⚠ **`ollama list` sizes are DISK, not VRAM.** `gemma4:e4b` is 9.6 G on disk
  and **3.4 G resident** — *smaller* than `qwen3:4b-instruct`'s 4.1 G. A VRAM
  argument was made backwards on this today.

## 6. NEXT — ⚑ NOT THE RESEED

**[`scripts/z1/README.md`](../scripts/z1/README.md) §2 Step 0 — ten items, none
blocked**, because the post-reseed list is enormous and Z2 is bigger again.
Highest value first: **G32(a) native endpoint** (seeding is hours of background
work and would otherwise inherit the host's residency policy) · gold turns for
the 174 anchorless probes (safe now the model is settled; feasibility proven at
343/368) · the **fold** A/B (G73) · **re-do cluster naming against real
members** (G74) · G61's four remaining silent drops · a real reconciler gold set.

**Then** the reseed, then what the store unblocks: baseline → reject-but-keep →
G70 read-side → G66 ablation → G75's decision → the G71 register.

**⚑ Do not write this file, or close a session, without the user saying so.**
