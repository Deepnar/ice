# Handoff — 2026-08-11, ~23:00 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Everything else here is a
pointer — if a doc already records it, it does not get repeated. Overwritten
every session, committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

---

## TOLD → DID

**Told:** verify the derived ground-truth key, then build Z1's retrieval scorer.

**Did:** the key was verified twice by the user and **failed twice**, and both
failures were informative rather than fatal. The tuning instrument was rebuilt
around what they showed, and the corpus plan changed as a result. Also: a side
session's 30 commits were reviewed line by line, G6 was closed, and the
bookkeeping debt from that session was paid.

## ⚑ THE PUSH FREEZE IS ON

**All commits since `3fd49b5` are local and unpushed.** Do not push until the
user lifts it (Z1's experiment-phase rule, CLAUDE.md).

## WHERE THINGS STAND

→ [ROADMAP.md](ROADMAP.md)'s **CURRENT POSITION**. Facts it does not carry:

- **Alembic head is now `f2a7c9e04b31`** (G6's btree indexes) — it moved.
- **Store: 1 conversation, 0 turns** — production state, verified after the
  seeding smoke test cleaned up after itself.
- **Background model pinned** to `gemma4:26b-a4b-it-q4_K_M` in `.env`.
- **A12's shortlist is being PULLED overnight** (`granite4`, `ministral`,
  `gemma4:e4b`, `nemotron-mini`) — script and log in the session scratchpad;
  **check what actually resolved before planning around them**, the registry
  names are guesses at the paper names.

## THE TWO THINGS THE USER'S VERIFICATION FOUND

Both are recorded properly ([G40](ROADMAP.md#g40) owns the taxonomy, [Z2](ROADMAP.md#z2)
owns the protocol) — read those, not this. In one line each:

1. **A whole class of probe cannot score retrieval at all.** Questions asking
   for *synthesis* ("go through my entire story") have no single answer-bearing
   turn: **38 of 91** span ≥6 gold turns, max 17. Unreachable inside a token
   budget, and many different turn sets are equally correct. **This is a
   property of the task, not of the derivation** (user) — do not try to fix it.
2. **The key ran the hard direction.** Deriving turns *backwards* from a written
   answer mis-grounded ~20% of claims even after two rounds of fixes.

## THE PLAN CHANGED — THE PROBE SET IS NOW BUILT FORWARDS

**Old:** derive gold turns from the curated answers, tune on those 91.
**Measured:** only **29** are usable, giving SE ≈ 0.093 against a **+0.05**
keep-rule. Unresolvable — no care in the derivation fixes a sample-size wall.

**Now, and the user rejected the first alternative for good reason:**

- **NOT synthetic conversations.** Scripting facts and having a model write
  dialogue around them bets on realistic dialogue, on facts landing where the
  script says, and on focal points being hit — three unvalidated assumptions.
- **Instead: generate probes FROM real turns.** Pick a turn, write a question
  whose answer is in it. The gold turn is *chosen first*, so it is correct by
  construction — no shortlist, no confirmation, no circularity.
  `scripts/z1/generate_probes.py`. Smoked at 44% keep, **0 rejected for
  fabricated evidence**; questions come back in the speaker's own register.
- **The 91 derived probes are now the VALIDATION set** (29 usable), opened once
  at the end. They are the only *real questions a real person typed*, which is
  exactly what generated probes can never be. **They still need the user's
  check before that** — deferred deliberately, not forgotten.

## NEXT

1. **Full probe generation** — ~880 questions over 293 turns, ~385 expected to
   survive both gates. ~1 hour. `scripts/z1/generate_probes.py` (no flags).
2. **Seed the store** — `scripts/z1/seed_store.py`, this time **without**
   `--no-codex` so the codex leg is populated. Then `pg_dump` it: every tuning
   run restores that snapshot, which is what makes the loop deterministic
   despite the LLM, and disposes of [G38](ROADMAP.md#g38) without per-probe
   transaction surgery.
3. **Build the scorer** — the one piece not yet written. Coverage per probe
   (what fraction of its claims' turns came back), resolved settings dumped per
   run (the freeze test no longer catches `.env` drift).
4. **Noise floor, screening pass, then sweep.**
5. **The model arms.** User's decision: **all 7 of A12's shortlist**, with the
   bg-model comparison run *after or during* tuning. ⚠ Population cost is
   **unmeasured** — measure one arm before committing to seven.

## WHAT WOULD OTHERWISE BE LOST

- **The side session's 30 commits were reviewed and are CORRECT** — G34's
  inverted design, G41's pgvector rewrite (traced line by line; the `break`
  placement is right and `LIMIT 1 + len(seen_ids)` is provably sufficient),
  G35, G39, G27. Discrepancies found: it claimed 37 commits (actual 30) and
  used "rails" for a wider set than CLAUDE.md defines (347 vs `tests/smoke`'s
  183). One substantive nit: G34's "magnitude unchanged at 0.25" is true of the
  *setting* and false of the *effect*.
- **G41 had NO roadmap entry** — announced in a commit subject only. Fourth
  phantom-item after G27/G34/G35. Written now, with two caveats its session did
  not record.
- **⚠ I was wrong about one of those caveats and corrected it the same day:** I
  wrote "the pgvector query has no index" from reading the diff without checking
  the database. `idx_codex_entities_embedding` exists *and* is migration-covered
  by `b6e2f9a41c73`. Recorded in [ROADMAP_DONE](ROADMAP_DONE.md#g41) rather than
  deleted, because it is one more instance of the pattern that file tracks.
- **G6 was half-done and nobody knew** — the vector half has been
  migration-covered since G23/C17. Only the btree half was missing.

## OPEN, AND ONLY HERE

- **`derived_gt.json` currently holds all 91 probes** (claim-level, 80%
  grounded). `generated_probes.json` holds only the **8-probe smoke output** —
  it is overwritten by a full run, so do not mistake it for the corpus.
- **Timestamps are spread across a synthetic span when seeding.** If that is
  ever changed to a single "now", every recency knob will measure as *cosmetic*
  and the verdict will be an artifact of the fixture. The reason is in
  `seed_store.py`'s module docstring; keep it there.

## WHEN DONE

Propagate per the roadmap's rules, update the docs the change invalidates in
the **same** session, then rewrite this file — carrying the NEXT above into
TOLD → DID — and commit it last. **Do not push while the freeze holds.**
