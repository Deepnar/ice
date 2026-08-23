# Handoff — 2026-08-23, ~17:00 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Overwritten every session,
committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

> **⚑ START AT [`scripts/z1/README.md`](../scripts/z1/README.md).** New this
> session. It indexes every Z1 experiment — what is settled, what is open in
> priority order, the measurement floor, and **§3b: every number this project
> has produced with its status** (dead / superseded / suspended / trusted).
> It exists because three "settled" findings were withdrawn this session.

---

## TOLD → DID

**Told:** the previous handoff said every retrieval number was measured off the
production path (G52), the fix was small, and *"the re-run is the work."*

**Did:** found that diagnosis **wrong**, then found ten more measurement
defects, fixed eight production bugs, and re-measured. **Three findings were
withdrawn during the session, two of them produced by this session.** The
re-run happened and its numbers are lower than everything they replaced.

## 1. ⚑ WHAT IS TRUE NOW

| | |
|---|---|
| **Graph correctness** | **14–17%, ±6 pts** — measured over 124 turns on two seeds. Every earlier figure (20%, 15.8%, 11.0%, 20.4%, 10.0%) was one under-sampled measurement |
| **Retrieval** (corrected metrics) | episodic **0.599** ±0.032 · codex **0.380** ±0.026 · summary **0.145** ±0.027 · procedural **0.000** · temporal 0.536 ±0.094 |
| **Store** | `ner-b-postfix` restored — 293 turns / 4,768 entities / 7,413 edges / 3 summaries |
| **Git** | clean, **39 commits**, local only — the Z1 freeze holds |
| **Tests** | smoke 183/183 · codex write path 32/32 · harness parity 4/4 · settings freeze 147/147 |

## 2. WHAT WAS WITHDRAWN, AND BY WHOM

- **G52's diagnosis** — it blamed `scope=None`, citing the MCP pull as
  "production". The chat path resolves an `auto` conversation to `{}`, so
  `conv_id` is None there too. The real invalidator was **G54**: the harnesses
  classified without the conversation, moving RRF blend weights on **65%** of
  probes.
- **"`extraction_confidence` is INVERTED"** — a **40-triplet** subgroup quoted
  without an interval, written into three tracked docs. Two full-coverage runs
  disagree on the direction. It is **uninformative**, not inverted.
- **"the 24% triplet drop was my code"** and **"the noise floor is 1.1%"** —
  both mine, both stated before the supporting measurement existed.

⇒ [TRAPS #46](TRAPS.md) is the shape: *a floor measured for one source, applied
to a comparison it does not cover.* Four instances in one session.

## 3. THE FINDINGS THAT SURVIVED

- **The six write-path mechanism fixes are a NULL on truth.** Every mechanism
  verifiably moved — expiries 7.99%→1.93%, negations 0.11%→6.04%, relation `in`
  from the #2 slot to **one edge** — and correctness did not (z ≤ 1.14).
- **Reversals are GENERATED, not merged in.** Blocking 1,611 direction- and
  polarity-changing merges left the reversal rate flat. That pointed at the
  prompt, which had **no direction instruction at all** — now [G59](ROADMAP.md#g59).
- **ICE cannot reproduce itself, and the last cause is physics.** Three
  order-dependencies fixed; the fourth is the model being nondeterministic
  **across processes** (temperature 0 fixes sampling, not logits).
  [TRAPS #47](TRAPS.md).
- **But the aggregate RATE is stable** — 17.1% vs 14.4% across two seeds. So
  **no heavy bootstrap**: one run per arm, provided it reports its interval and
  samples turns rather than triplets.
- **Everything counted directly against the store survived scrutiny. Everything
  that passed through a sampler, a metric or a judge needed correcting.**

## 4. WHAT SHIPPED

**Production (8):** the G51 supersession branch (667 true facts were being
deleted) · the G45 merge guard (`is`→`in` ×942) · A8 negation routing (686
edges) · the G50 `merge_key` leak (6,271 payloads) · G53 `conv_id` split (the
summary leg was dead) · G48 summary provenance · G57 coverage de-circularised +
token ceiling 300→900 · G59 the direction rule.

**Instruments (~11):** one shared production-parity path for all four harnesses
plus a test that fails on drift · four metrics that were presence tests ·
judge answer-truncation removed · turn-stratified judge sampling that reports
its own interval · ablation-flag validation · the two-id-space fix in
`score_retrieval`.

**New tools:** `check_reproducible.sh`, `dump_graph_fingerprint.py`,
`compare_judgements.py`, `production_parity.py`, and the Z1 index.

## 5. ⚑ DECISIONS MADE (do not re-litigate)

- **Stay on NuNER** until the extraction side lands. Micro is the shipped
  default, but nothing ships yet and re-opening it adds a second variable.
- **Never a large model for background** — laptop card, beside the user's chat
  model. The 26B is out of the candidate set on principle. [G58](ROADMAP.md#g58).
- **Non-lossless turns SHOULD feed the graph** — a code change waiting, not a
  question. [G57](ROADMAP.md#g57).
- **`CLAUDE.md` line 32 is knowingly stale** ("dense enough" — density decides
  18%). Fix the behaviour first, correct the line in the same change. **Do not
  edit it before then.**

## 6. WHAT IS STILL UNMEASURED

The whole answer layer (Z2) · decay, reflection and the maintenance agent
(never run; `decay_score < 1.0` is 0 rows) · the B2 gate (never once observed
declining) · the context ledger (0 `context_evicting` events ever) · whether
the direction rule actually works.

## 7. NEXT

**[`scripts/z1/README.md`](../scripts/z1/README.md) §2 is the ordered plan.**
It starts with re-judging three arms that already exist — ~30 minutes, no
re-seed — because the direction rule is the only intervention that has ever
moved correctness and its result is still unresolved.

**⚑ Do not write this file, or close a session, without the user saying so.**
