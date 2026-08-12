# Handoff — 2026-08-12, ~21:00 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Everything else here is a
pointer — if a doc already records it, it does not get repeated. Overwritten
every session, committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

---

## TOLD → DID

**Told:** verify the derived ground-truth key, then build Z1's retrieval scorer.

**Did:** the key failed hand-verification **twice**, was rebuilt in the opposite
direction, and the resulting instruments were used to measure the whole pipeline
end to end — seeding, extraction, summarisation, clustering, retrieval, and
eight candidate background models. **Six new roadmap items came out of it
([G42](ROADMAP.md#g42)–[G47](ROADMAP.md#g47)) and three TRAPS entries (#20–22).**

**⚑ Z1 AND Z2 HAVE MERGED, and the user called it in advance.** Seeding the
tuning corpus runs every background job, so extraction quality, summary
faithfulness and codex correctness were measured *during* Z1-prep — exactly what
Z2's entry lists as what the automated suite cannot see. The roadmap's CURRENT
POSITION now says so. **Treat them as one phase.** Standing instruction from the
user: *measure with real use every time, cross-check often, do not rush.*

## ⚑ THE PUSH FREEZE IS ON

Everything since `3fd49b5` is local and unpushed. Do not push until the user
lifts it.

## WHERE THINGS STAND

- **Alembic head `f2a7c9e04b31`** (G6's btree indexes — moved this cycle).
- **Store: the full 293-turn gemma4-26b seed, restored.** Snapshot at
  `experiments/curation_files/snapshots/gemma4-26b.sql` (71.8 MB), restore
  verified against recorded row counts.
- **⚠ The probe→row map was REBUILT BY HAND** from `idempotency_key` after the
  arm runs desynchronised it. It is correct now; the tooling still has the bug
  ([G46](ROADMAP.md#g46)).
- **592 generated probes**, key correct by construction, in
  `experiments/curation_files/generated_probes.json`.
- **Eight arm reports + samples** in `store_reports/` and `arm_stores/`.
- **3,244 vocabulary candidates** with proposed opposites in
  `VOCAB_CANDIDATES.md`.
- Background model pinned to `gemma4:26b-a4b-it-q4_K_M` in `.env`.

## THE NUMBERS, AND WHAT EACH IS WORTH

| measured | value | trust |
|---|---|---|
| retrieval recall (592 probes) | **0.250** | ⚠ off-production — the scorer skips the budget setter |
| fragments actually returned | **2–5** | solid, and it reframes the above |
| graph: degree-1 entities | **65%** at 293 turns | solid; did not densify from 60 turns (67%) |
| relations destroyed | **14,091** across arms | solid |
| model ranking | **gemma4:e4b > qwen3:4b-instruct > gemma4:26b** | by reading; the metric ranking was overturned |
| summary coverage | 0.607–0.979 | ⚠ **cannot see an empty summary** (TRAPS #21) |

## NEXT — the user's instruction, in order

**Measure and test everything first. This is a whole-pipeline test, not a tuning
sweep.** Nothing below is a code change until the user approves it.

1. **Confirm [G47](ROADMAP.md#g47)'s zero-fragment hypothesis** — print the token
   count of the top fragment for one of the 15 zero probes. If it exceeds the
   budget, the enforcer stops instead of skipping and that is the bug. Ten
   minutes, and it gates everything downstream.
2. **Does the degradation chain fire?** raw → trusted summary → abstract. If it
   does not, ten summaries would fit where three raw turns do now — a bigger
   lever than any leg weight.
3. **Fix the instruments** ([G46](ROADMAP.md#g46)) — call the budget setter,
   derive the map from the store, score the ambiguity guard on the question
   alone, record all producing legs. **Until these land every number is partly
   about the harness.**
4. **Measure the noise floor.** Nothing is provably an improvement without it.
5. **Then** the fixes, in leverage order: [G43](ROADMAP.md#g43) grounding
   (highest — one change hits fabrication, wrong-turn facts and junk subjects),
   [G42](ROADMAP.md#g42) procedural, [G44](ROADMAP.md#g44) subjects,
   [G45](ROADMAP.md#g45) open vocabulary **(spec before code — it supersedes a
   decision Z2 owns)**.
6. **Then** re-seed and re-run the whole comparison on a pipeline that works.
   Today's ranking partly measures which model best survives our defects.

## WHAT WOULD OTHERWISE BE LOST

- **Everything measured is in [PROVENANCE.md](PROVENANCE.md)** under the
  2026-08-12 A12 entry, and the defects are one roadmap item each. Read those,
  not a summary.
- **⚠ FIVE CONCLUSIONS WERE DRAWN AND THEN CORRECTED IN THE SAME SESSION.** They
  are listed because the pattern matters more than any one of them: *"the vector
  leg is dead"* (it returns 87–89 candidates alone — attribution artifact);
  *"`max_per_conversation=3` caps recall"* (diversify passes 90 of 90; the
  **budget** caps it); *"probe ambiguity explains the low score"* (0.284 vs
  0.255 — it does not); *"the pgvector query has no index"* (it exists and is
  migration-covered — written from reading a diff instead of the database);
  *"procedural is 0 for every arm"* (wrong dict key in my own report script).
- **The eyeball found what no metric did**, twice: the user's verification killed
  a key that looked fine, and the agent's read overturned an eight-model ranking
  and found invented facts. Every finding that survived scrutiny came from
  reading output.
- **Nine bugs in the session's own instruments** are catalogued in
  [G46](ROADMAP.md#g46) and TRAPS #20/#22.

## OPEN, AND ONLY HERE

- `experiments/curation_files/` is gitignored and holds personal conversation
  text — probes, samples, snapshots, arm output. **Never commit it.** The user
  authorised cloud-agent review of this content on 2026-08-12.
- `granite4:small-h` (32.2 B, 19 GB) was pulled by mistake — the shortlist asked
  for the 3 B (`granite4:micro`). Excluded from the comparison by user decision.
- `qwen3:4b-instruct-bg` exists locally; the registry never routed to it. Shared
  mode resolved to the 26B all along ([G27](ROADMAP.md#g27)).
- The scripts built this cycle live in `scripts/z1/`: `derive_retrieval_gt`,
  `generate_probes`, `seed_store`, `score_retrieval`, `snapshot`,
  `store_report`, `sample_bg_output`, `compare_bg_models`,
  `harvest_vocabulary`, `run_arms.sh`.

## WHEN DONE

Propagate per the roadmap's rules, update the docs the change invalidates in the
**same** session, then rewrite this file — carrying the NEXT above into
TOLD → DID — and commit it last. **Do not push while the freeze holds.**
