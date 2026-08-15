# Handoff — 2026-08-13, ~22:20 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Everything else here is a
pointer — if a doc already records it, it does not get repeated. Overwritten
every session, committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

---

## TOLD → DID

**Told:** confirm G47's zero-fragment hypothesis, fix the instruments, measure
the noise floor, then the pipeline fixes in leverage order.

**Did:** all of it, and **the hypothesis was wrong**. Confirming it exposed that
the *instrument* was the finding: correcting four scorer defects moved recall@10
from **0.250 to 0.508 without changing one line of ICE**. A legacy-mode control
reproduced 0.250 exactly (`recall@1` identical to 17 decimal places), so the
delta is attributable to the harness and nothing else. Then G42, G43, G44, G45,
bi-temporal edges and the seeder were fixed, each tested twice.

**⚑ NOTHING IS COMMITTED. THE PUSH FREEZE HOLDS. NOTHING HAS BEEN RE-SEEDED.**

## ⚠ THE STORE IS A 2-TURN FRAGMENT — THIS IS THE FIRST THING TO FIX

A re-seed was started and **stopped by the user, twice, correctly**: the rule is
*nothing runs until every fix is in*. The store now holds ~2 turns and is
useless. Both baselines are snapshotted and restorable:

* `experiments/curation_files/snapshots/gemma4-26b.sql` — pre-session
* `experiments/curation_files/snapshots/pre-g42g43g44.sql` — post-cleanup, verified

## WHAT SHIPPED (all twice-tested; 35/35 verified in code, 347 tests green)

| | was | now |
|---|---|---|
| **freeze leak** | `access_count` written every retrieval, read by nothing; 26/40 identical runs | gated by `retrieval_strengthen_writes`; **40/40** |
| **[G43](ROADMAP.md#g43)** | 51 property relations (30% of edges) never checked their OBJECT | must occur in source turn; 30 edges (0.72%) demoted |
| **[G42](ROADMAP.md#g42)** | 247/247 habits invented from ONE turn | session-scoped, ≥2 cited messages enforced, cross-session reinforcement only |
| **[G44](ROADMAP.md#g44)** | nodes named `8`,`3`,`i` typed *person* | refused at write; stub promotion (default OFF) |
| **[G45](ROADMAP.md#g45)** | 1,259 true relations destroyed per seed | open vocabulary + converse guard + 111-relation seed |
| **bi-temporal** | could not say when a fact was *learned* | `learned_at`/`unlearned_at`, migration `505f12031434` |
| **seeder** | 293 turns → 293 one-turn sessions | sittings → 6/9/4 sessions of 11–20 |
| **[G46](ROADMAP.md#g46)** | scorer measured itself | all four defects fixed |

## THE NUMBERS, AND WHAT EACH IS WORTH

| measured | value | trust |
|---|---|---|
| recall@10, production path | **0.508** | solid — legacy control reproduces 0.250 exactly |
| paired MDE (592 probes) | **0.037** (~22 probes) | solid; this is the bar for any tuning claim |
| fragments returned | median **14** (8–46) | solid; supersedes "2–5" |
| retrieval budget | **8,100–11,350** | solid; set by the growth ladder, not the model window |
| probe-set contamination | **380 of 592 (64%)** | solid, and it makes 0.508 soft |
| degradation chain | **fires** — 104/60 probes, 1,056 tok median saved | solid |
| relation threshold | 0.86 → **0.82** | 30 hand-built pairs — enough to leave a guess, not final |
| procedural threshold | 0.85, a true match scores **0.708** | ⚠ promotion still dead; needs post-seed data |

## ⚑ THE TWO THINGS THAT BLOCK REAL PROGRESS

1. **[G48](ROADMAP.md#g48) — the metric can only score ONE leg family.** Of 377
   hits, `bm25+vector` produced 376 and `vector` 1; **codex, procedural and
   summary scored zero, never** — they carry no `source_batch_id`. They still
   spend budget, so a leg-weight sweep would drive them to zero and report a
   win. **LEG-WEIGHT TUNING IS FROZEN.** Found by the user.
2. **The probe set is 64% contaminated.** The fixed ambiguity guard rejects
   **385 of 592** where the old one rejected 5.

Both are addressed by the **typed probe set** (below) — which is generated but
has **never been scored against a real store**.

## NEXT — in order, and NOTHING runs before the re-seed

1. ~~Merge the boost runs~~ **DONE.** `typed_probes.json` holds **420 probes**:
   episodic 232 · **codex 89** · procedural 40 · summary 40 · temporal 19.
   Codex went 12 → 89 once the prompt stopped demanding a verbatim quote from
   every host turn (that length was truncating the JSON mid-object, and the
   discarded generations were the probes). Each probe carries `_source_run`.
   ⚠ **Never scored against a real store** — the store is a 2-turn fragment.
2. **Re-seed both arms** — `qwen3:4b-instruct` then `gemma4:e4b`, whole 293
   turns each, snapshot per arm. Driver: the session scratchpad's
   `two_arm_seed.sh` (sets `CODEX_NODE_PROMOTION=true`). **~50 min per arm.**
   ⚑ **`gemma4:26b` is NOT the pick** — PROVENANCE's read ranked it third and
   named `qwen3:4b-instruct` the practical pick (tied on quality at 2.5 GB vs
   9.6 GB). `.env` still pins the 26B; that is why the first re-seed was stopped.
3. **Validate the pipeline fixes on the new store** — are habits evidenced by
   >1 turn, are invented property values gone, are junk nodes gone, did the
   1,259 destroyed relations survive.
4. **Score the typed probes** (`score_typed.py`) — five metrics, never averaged.
5. **Read the output** — `harvest_probe_context.py` dumps question + gold turn +
   every returned fragment, scoreless on purpose. Then an agent read, which is
   what overturned the model ranking last time.
6. **Re-run the model comparison** on the fixed pipeline, script *and* agent.

## WHAT WOULD OTHERWISE BE LOST

* **Everything measured is in [PROVENANCE.md](PROVENANCE.md)** under the
  2026-08-13 entry. New traps: **#24** (a guard that only fires in the harness),
  **#25** (`.env` keys undeclared in Settings take the whole app down),
  **#26** (embeddings cannot tell a converse from a synonym).
* **⚠ Three self-inflicted incidents, all caught, all instructive.** The test
  suite leaked 3 turns into the measured store (cleaned, verified back to
  293/3,671/4,170/247). `alembic --autogenerate` proposed dropping **every HNSW
  vector index**; the migration was rewritten by hand. Adding `PROBE_*` to
  `.env` broke **every** import of `src.api.config` while the script using them
  worked fine.
* **Every experiment artifact now carries `run_meta()`** — commit, dirty flag,
  resolved settings, corpus digests, redacted secrets. Use it; it is the reason
  the probe set can be dated and attributed later.
* **Probe generation uses a CLOUD model** (`deepseek-v4-flash`, OpenCode Go).
  Corpus excerpts leave the machine in those prompts. `PROBE_API_KEY` lives in
  gitignored `.env`; **the user should rotate it** — it was pasted in chat.
* **`codex_node_promotion` defaults OFF** and `procedural_similarity_threshold`
  was deliberately NOT retuned on one observation. Both are decisions, not
  oversights.

## OPEN, AND ONLY HERE

* `experiments/curation_files/` is gitignored and holds personal conversation
  text. **Never commit it.**
* New scripts: `generate_typed_probes.py`, `score_typed.py`,
  `harvest_probe_context.py`, `run_meta.py`, `seed_relation_vocab.py`.
* `data/relation_seed.json` (111 relations) IS tracked — relation words only, no
  conversation text.
* Scorer defect #4 is fixed in the generator but **the live 592-probe set was
  never regenerated**; it still carries the 64%.

## WHEN DONE

Propagate per the roadmap's rules, update the docs the change invalidates in the
**same** session, then rewrite this file — carrying the NEXT above into
TOLD → DID — and commit it last. **Do not push while the freeze holds.**
