# Handoff — 2026-08-15, ~07:45 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Overwritten every session,
committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

> **⚑ READ THIS WHOLE FILE BEFORE TOUCHING ANYTHING.** The last session was
> long, it changed seven subsystems, and it left the store deliberately empty.
> Three of the numbers you will find in older docs are **wrong and superseded**;
> they are named below. Starting work without reading this will reproduce
> conclusions that were already disproved.

---

## TOLD → DID

**Told:** confirm [G47](ROADMAP.md#g47)'s zero-fragment hypothesis, fix the Z1
instruments, measure the noise floor, then the pipeline fixes in leverage order.

**Did:** all of it — and **the hypothesis was wrong in a way that mattered more
than the hypothesis.** Confirming it exposed that the *instrument* was the
finding: fixing four scorer defects moved recall@10 from **0.250 to 0.508
without changing one line of ICE**. Then [G42](ROADMAP.md#g42),
[G43](ROADMAP.md#g43), [G44](ROADMAP.md#g44), [G45](ROADMAP.md#g45),
bi-temporal edges, and the seeder were fixed — each tested twice, once in
isolation and once through the path production actually takes.

## ⚑ START HERE — THE STATE OF THE WORLD

| | |
|---|---|
| **Store** | **~2 turns. Deliberately empty.** Unusable for any measurement. |
| **Git** | All work committed, **16 commits this session**. `main` is **82 commits ahead** of `origin/main` (`0c446d1`). Working tree clean. |
| **Push freeze** | **ON.** Nothing pushed. Do not push until the user lifts it. |
| **Alembic head** | `505f12031434` (bi-temporal codex edges). |
| **Tests** | 347/347 green (`tests/smoke`, `test_settings_freeze.py`, `test_dynamics_invariants.py`). |
| **Probes** | **420 typed probes** ready, never scored against a real store. |

**Both baselines are snapshotted and restorable** — the store was emptied on
purpose, not lost:

* `experiments/curation_files/snapshots/gemma4-26b.sql` — pre-session
* `experiments/curation_files/snapshots/pre-g42g43g44.sql` — post-cleanup, verified

## ⚠ THREE NUMBERS IN OLDER DOCS ARE WRONG

If you read these anywhere, they are superseded — the correction is in
[PROVENANCE.md](PROVENANCE.md) 2026-08-13 and in each roadmap entry:

1. **"recall@10 = 0.250"** → **0.508**. The scorer skipped the budget setter and
   hit a guard production cannot reach. A legacy-mode control reproduced 0.250
   *exactly* (recall@1 identical to 17 decimal places), which is what makes the
   delta attributable to the harness rather than to luck.
2. **"the system returns 2–5 memories"** → **median 14** (8–46). That was the
   orchestrator sitting at its `__init__` default of 5,000 tokens; real budgets
   are **8,100–11,350**, set by `context_growth_cap_ladder`, not the model window.
3. **"15 of 592 probes return zero fragments"** → **242 (41%)** in the legacy
   condition, **0** on the production path. 15 was the old scorer's
   `misses[:40]` truncation, not a count.

## WHAT SHIPPED (each twice-tested)

| item | was | now |
|---|---|---|
| freeze leak (G38) | `access_count` written every retrieval, read by nothing; 26/40 identical runs | gated by `retrieval_strengthen_writes`; **40/40** |
| **[G43](ROADMAP.md#g43)** | 51 property relations (30% of edges) never checked their OBJECT — `november --eye_color--> golden black` | value must occur in the source turn; 30 edges (0.72%) demoted |
| **[G42](ROADMAP.md#g42)** | 247/247 habits invented from ONE turn | session-scoped, ≥2 cited messages enforced in code, cross-session reinforcement only |
| **[G44](ROADMAP.md#g44)** | nodes named `8`, `3`, `i` typed **person** | refused at write (functional test, *not* a length rule); stub promotion, default OFF |
| **[G45](ROADMAP.md#g45)** | 1,259 true relations destroyed per seed (`i --didnt_get--> csi`) | open vocabulary, converse guard, 111-relation seed |
| bi-temporal | could not say when a fact was *learned* | `learned_at`/`unlearned_at`, migration `505f12031434` |
| seeder | 293 turns → **293 one-turn sessions** | sittings → 6/9/4 sessions of 11–20 turns |
| **[G46](ROADMAP.md#g46)** | scorer measured itself | all four defects fixed |
| **[G48](ROADMAP.md#g48)** | metric could only score ONE leg family | 420 typed probes + type-aware scorer |

## NEXT — IN THIS ORDER. NOTHING ELSE FIRST.

**1. Re-seed both arms.** ~50 min each.

```
sh <scratchpad>/two_arm_seed.sh      # or reproduce it: it sets CODEX_NODE_PROMOTION=true
```
Arm 1 `qwen3:4b-instruct`, arm 2 `gemma4:e4b`, whole 293 turns each, snapshot
per arm. The driver passes `--bg-model`, which **is** honoured all the way down
(verified: `seed_store.py:304` sets it, `:339` reads it into `seed_model`, `:440`
passes `model_used` to `evaluate_turn`, and `post_flight.py:114` forwards it to
both extractors).

> **⚑ `gemma4:26b` IS NOT THE PICK.** [PROVENANCE.md](PROVENANCE.md)'s A12 read
> ranked it **third** and named **`qwen3:4b-instruct` the practical pick** —
> tied with `gemma4:e4b` on summary quality at **2.5 GB against 9.6 GB**.
> **`.env` still pins the 26B**, which is why a re-seed was stopped twice. The
> pin does not affect the two-arm run, but it affects anything else that reads
> `background_model_name`.

**2. Validate the pipeline fixes on the new store.** This is what the re-seed is
*for* — not tuning:
* are procedural patterns evidenced by **more than one turn**?
* are invented property values gone (`november --eye_color--> …`)?
* are junk nodes gone (entities named `8`, `3`, `i`)?
* did the **1,259** destroyed relations survive?
* **count distinct relations** — over ~1,000 means canonicalisation is not
  binding ([G49](ROADMAP.md#g49)).

**3. Score the typed probes** — `scripts/z1/score_typed.py`. **Five metrics,
never averaged.**

**4. Read the actual output** — `scripts/z1/harvest_probe_context.py` dumps
question + gold turn + every returned fragment and computes **no score on
purpose**. Then an agent read. That pass is what overturned the model ranking
last cycle, against metrics that were green.

**5. Re-run the model comparison** on a pipeline that works — script *and* agent.

## ⚑ BEFORE YOU DEBUG ANYTHING — [TRAPS #27](TRAPS.md)

**TRAPS #27 is a checklist of nine things that LOOK broken and are not**, each
settled by a measurement this cycle: the procedural leg returning nothing, the
probe API's 403/1010, retrieval "non-determinism", ruff's `== None` warnings, an
empty model reply, `retrieval_max_per_conversation`, the stale `gemma4:26b` pin,
`data/`, and a coverage metric reading 1.000. **Read it before investigating any
of them** — every one presents as an obvious bug whose obvious fix is wrong.

## ⚑ THREE THINGS YOU WILL MISREAD — see [G49](ROADMAP.md#g49)

1. **Procedural probes will score ZERO, and it is NOT retrieval.**
   `_procedural_lookup` requires `is_active = true`; activation needs
   `reinforcement_count >= 3`; two extractions of the SAME habit **measure
   0.708** against a 0.85 threshold. Patterns are born at 1 and never activate.
   [G42](ROADMAP.md#g42)'s fabrication fix can be working perfectly and this
   still reads as a dead leg.
2. **`learned_at`/`unlearned_at` are written and read by nothing.** The
   `access_count` defect, reintroduced by the commit that fixed it.
3. **The relation count may explode** under the open vocabulary.

## ⚠ WHAT THE MEASUREMENTS STILL CANNOT DO

* **The old 592-probe set is 64% contaminated.** The fixed ambiguity guard
  rejects **385 of 592** where the old one rejected **5**. `0.508` inherits that.
* **After the re-seed, old and new numbers are NOT comparable** — different
  pipeline, different model, different probes. It is a new baseline, not a
  before/after. Only the **graph-shape query** (65% degree-1) is comparable.
* **Leg-weight tuning is FROZEN** ([G48](ROADMAP.md#g48)) until typed probes are
  scored. On the old metric, codex/procedural/summary scored **zero of 377
  hits**; a sweep would drive them to zero and report an improvement.
* **Tuning bar:** paired MDE **0.037** (~22 probes). Below that is noise.

## WHAT WOULD OTHERWISE BE LOST

* **Every measurement is in [PROVENANCE.md](PROVENANCE.md)** under 2026-08-13.
  New traps: **#24** (a guard that only fires in the harness makes production
  look broken), **#25** (`.env` keys undeclared in `Settings` take the whole app
  down), **#26** (embeddings cannot tell a converse from a synonym —
  `before`/`after` scored **0.8791**, above the merge threshold then in force).
* **⚠ FOUR SELF-INFLICTED INCIDENTS, all caught.** The test suite leaked 3 turns
  into the measured store (cleaned; store verified back to 293/3,671/4,170/247).
  `alembic --autogenerate` proposed dropping **every HNSW vector index** — the
  migration was rewritten by hand. Adding `PROBE_*` to `.env` broke **every**
  import of `src.api.config` while the script using them worked fine. And three
  separate harness bugs made a correct model look incapable (empty replies
  retried as failures, junk word-frequency anchors, truncated JSON discarded
  whole) — codex probes went **12 → 89** once the last was fixed.
* **Every experiment artifact now carries `run_meta()`** —
  `scripts/z1/run_meta.py`: commit, dirty flag, resolved settings, corpus
  digests, redacted secrets. **Use it in any new experiment script.**
* **Probe generation uses a CLOUD model** (`deepseek-v4-flash`, OpenCode Go,
  `PROBE_*` in gitignored `.env`). **Corpus excerpts leave the machine** in those
  prompts. A browser `User-Agent` is mandatory or Cloudflare answers 403/1010.
  **The user should rotate `PROBE_API_KEY`** — it was pasted in chat.
* **Two settings are deliberately NOT tuned:** `codex_node_promotion` defaults
  OFF (promotion merges identities), and `procedural_similarity_threshold` stays
  0.85 despite the 0.708 measurement (one observation is not a calibration).
  Both are decisions, not oversights.

## OPEN, AND ONLY HERE

* `experiments/curation_files/` is gitignored and holds personal conversation
  text. **Never commit it.** `data/labeled/` likewise (`.gitignore:11`) — and
  `data/` is 871 MB, so **never `git add data/` blind**; only
  `data/relation_seed.json` (111 relation words, no conversation text) is tracked.
* New scripts, all in `scripts/z1/`: `generate_typed_probes.py`, `score_typed.py`,
  `harvest_probe_context.py`, `run_meta.py`, `seed_relation_vocab.py`.
* **One stale line remains, and it is USER-GATED:** `CLAUDE.md:303` says *"The
  public release is gated on a good README"* — the repo has been public for a
  long time; there is no pending release. The user was shown a proposed fix and
  has not yet approved it.
* Commit style settled 2026-08-13: **subsystem-specific `area:` prefixes**
  (`retrieval:`, `codex:`, …), not generic buckets.
* **[TRAPS](TRAPS.md) is now 28 entries.** #27 is the "looks broken but is not"
  checklist; **#28–30 were mined out of five overwritten handoffs** at the end of
  this session — durable lessons that had only ever lived in a file that gets
  replaced every time. **#28 is the one to read first: nine consecutive roadmap
  entries were found wrong about their own subject, and overstating remaining
  work is the direction that wastes an entire session.**

## WHEN DONE

Propagate per the roadmap's rules, update the docs the change invalidates in the
**same** session, then rewrite this file — carrying the NEXT above into
TOLD → DID — and commit it last. **Do not push while the freeze holds.**
