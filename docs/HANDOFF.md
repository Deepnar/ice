# Session handoff

**Written 2026-08-10 17:38 IST.** Last commit before this one: `6c71033`
(2026-08-09 19:08 +0530).

> **What this file is.** The state of the work at the end of the last session,
> rewritten in place every session and committed as the session's **last**
> commit. It exists because a handoff carried only in chat is lost when a
> session ends badly — the same reason [TRAPS.md](TRAPS.md) was moved into the
> tree.
>
> **Three rules, and the first one matters most:**
> 1. **It is STATE, never a QUEUE.** [ROADMAP.md](ROADMAP.md) is the queue and
>    the only queue. A handoff that starts listing work becomes a second source
>    of truth, and the two drift. Point at roadmap items; never restate them.
> 2. **Carry the previous session's NEXT forward** into the section below, so
>    the incoming session can compare what was *supposed* to happen against
>    what the git log says happened. That comparison is the point of the file.
> 3. **Record only what would otherwise be lost.** If ROADMAP, TRAPS,
>    PROVENANCE, CLEANUP or ICE_Architecture already says it, link it.
>
> Overwritten in place — previous handoffs are in `git log -p docs/HANDOFF.md`.

---

## WHAT THE LAST SESSION WAS TOLD TO DO, AND WHAT HAPPENED

**Told:** `NEXT: G36, THEN G30, then G28.` G36 ahead of G30 because G30's job is
trustworthy retrieval tests and a leg that fails silently makes a passing test
meaningless. G28 stays inside Z1 and must not be pulled forward.

**Happened:** **G36 done, and G37 opened and closed on top of it** — seven
commits, `cc3dd16..6c71033`. The order held: G30 was not started, G28 was not
touched. The scope grew twice, both times with the user's explicit go-ahead
(fix the fail-open direction rather than only logging it; then close G37).

---

## POSITION

CLUSTERS ①, ② COMPLETE. **CLUSTER ③: G9 · G36 · G37 done — G30 is next.**

- **G36** (`fb07e7c` reporting · `b4a79a8` fail-closed · `4e3b43d` HyDE
  deletion · `c4e07a7` tests · `da6c30a` docs) — 31 `except` handlers in
  `orchestrator.py` on one event, `retrieval_leg_failed`, with the leg as a
  field; the two fail-open scope resolvers now fail closed.
- **G37** (`c7f37fd` · `fb0e81a` docs) — a scope that names conversations and
  resolves to no batches reads **nothing**, not everything.
- **`6c71033`** — `test_mcp_server` no longer leaks a global memory slot.

⚠ **ALEMBIC HEAD UNCHANGED — `d5c81a37e9b2`.** No migration this cycle.
Inventory **52** (48 mechanical top-level `- [ ]`).

## NEXT

**G30, then G28.** G30 is the retrieval/test-coverage item and it now inherits
two things from this session: `tests/test_retrieval_failopen.py` as the worked
example for its own headline complaint (it is the first suite that seeds codex
entities with **real** embeddings instead of the `[0.05]*1024` stub, which is
precisely the blind spot the entry describes), and the `retrieval_leg_failed`
event, which is what makes a green retrieval test mean something. **G28 still
runs INSIDE Z1's coverage matrix — do not pull it forward.**

---

## THE PATTERN, SEVENTH CONSECUTIVE ENTRY: G36 WAS WRONG ABOUT ITS OWN SUBJECT

Its count of 22 silent handlers was exact. Everything else was not: only **11**
of the 22 rolled back, **4 were on dead code**, and it missed the only dangerous
members of the set — **two handlers that fail OPEN**, which turned out to be the
actual bug. ⇒ **Verify an entry's claims before building from it.** Every entry
now records which way it was wrong.

## WHAT THIS SESSION DECIDED — do NOT re-derive

- **One event name for every leg failure**, including the nine handlers that
  already logged under their own names (user choice). Thirty-one names cannot
  answer "did any leg fail on this request".
- **Fix the fail-open direction, do not merely log it** (user). Both codex scope
  resolvers now fail closed.
- **A scope resolving to no batches reads nothing** (user): *"why would i need
  thing if i am starting anew, thats why the scoping for codex was built in the
  first place."*
- **Rollback is conditional on the error being the database's.** Mandatory after
  a DB error (Postgres refuses every later statement in an aborted transaction);
  gratuitous after a Python error, where it only expires the identity map of a
  session `main.py` keeps using.
- **HyDE deleted outright**, not left commented (user), with a note in its place.
- **Exp 1's HyDE row: record now, re-run at FINAL.** `experiments/` and the
  `.tex` untouched — the same disposition G19 took.

## ⚑ THE RULE THE NEXT SCOPE CHANGE DEPENDS ON

**Presence, not truthiness.** Failing closed whenever a batch set is empty would
have taken the codex leg out of **every `auto` request**, and nothing in the
suite would have noticed. The guard asks whether the scope *named* conversations
(`"conversation_ids" in scope`), not whether the result is non-empty. Three
CONTROL checks in `test_retrieval_failopen.py` pin it; they stay green when the
fix is reverted, which is how you know they are not mirroring it.

## OPENED / CORRECTED / CLOSED BY THIS SESSION

- **G37 opened and closed same day.** Full record in its entry.
- **G12 is down to two sites** — the third hardcoded `timeout=15.0` was inside
  the deleted `_hyde_rewrite`. **Retrieval now makes no LLM call at all.**
- **G19 has a 3-for-3 record.** `hyde` joins `recency_boost` and the
  `include_cross` signature break: three ablation flags that toggled nothing,
  each corrupting a published arm. **The fold is no longer upkeep, it is the fix
  for a defect class.** Its design is settled and written in the entry.
- **G20 item (2) closed** — the `_hyde_used` flag and its two emissions.
- **G30's stale codex-write-path bullet** marked closed (it was closed
  2026-08-03) so the next session does not re-plan it.
- **⚑ AN OLD QUESTION ANSWERED.** The user asked why Experiment 2 retrieved
  things from other conversations. At `v2-paper-eval`, `_traverse_graph` took
  **no scope parameter at all** — only the anchor entity was checked, then
  traversal ran three hops across the whole graph. Unscoped by construction.

## STILL OWED

- **G19's fold** (design settled, not built): delete the shadow subclass in
  favour of an `ablation` dict on the **orchestrator instance** via `_ablate(flag)`
  — *not* settings toggles, because settings are global and ablation state must
  stay per-instance. `ConfigurableOrchestrator` becomes a 3-line alias so the
  frozen runners keep constructing. Plus procedural extraction in the replay
  loop and the `simulation_runs` table.
- **Two duplicated constants** from G9: `context_budget_fallback` and
  `context_total_budget_fallback`, both 23,000 on different call paths. Merging
  them is a behaviour decision.

## TRAPS GAINED

**#15** — a deterministic id is not evidence of a test fixture. A production row
(`bookmarks.NOTES_CONVERSATION_ID`) was deleted as residue because its id was a
UUID5; restored with its original `created_at`, nothing lost. The check that
would have prevented it is `grep -rn uuid5 src/`, and it was run *after*.
**#16** — residue is not always inert. A `memory_slots` row leaked by
`test_mcp_server` was `is_active=True` + `scope_tier='global'`, and `main.py`
injects every such slot into the system prompt — so a test marker had been in
**every turn for twelve days**. The leak was a snapshot-and-restore with no
did-not-exist branch; `test_c10_c11` already had the right shape.

## METHOD THAT EARNED ITS KEEP

- **A `sys.settrace` probe keyed on `frame.f_code.co_filename`** counted every
  exception propagating through an `orchestrator.py` frame across 15 seeded
  suites: **zero**. That is what made "log every swallow" the right call rather
  than noise (TRAPS #13b). Re-run it after any change to the legs.
- **My own first reproduction was vacuous** — it asserted only "the out-of-scope
  entity is absent" and passed against a fixture returning nothing for *either*
  conversation, because the seeded entity embeddings were the constant stub. The
  positive control is what made it real. TRAPS #13, hit again by the person
  writing the fix for it.
- **Bisect before naming a suite** (TRAPS #6b) found the slot leaker in one
  command: clear the table, run one suite, count.

## VALIDATION, all green at close

smoke **123** · settings_freeze **147** · dynamics_invariants **17** ·
retrieval_failopen **27** (new) · classifier_v2 65 · timescope 61 · c10_c11 58 ·
documents 53 · coding_core 52 · maint_runtime 49 · services 48 · agent 43 ·
memory_decision 42 · session_scoping 40 · ingestion 36 · relation_gaps 33 ·
turn_density 32 · c4_c9 28 · longevity 27 · retrieval_coverage 25 ·
codex_write_path 23 · mcp_server 21 · context_budget 17 · reconcile_on_read 16 ·
density_c3 16 · document_chunking 15 · job_yield 14 · clustering_v5 13 ·
batch_summary_coverage 11 · c8_c15 8 · retrieval 3.

Every new behaviour check negative-controlled (revert ⇒ red). Ruff at parity
with base on every touched file. **Store: 1 row** — the deterministic
`ice://mcp-notes` conversation shell, which is production state, not residue.

## ENVIRONMENT — the non-obvious parts only

- ⚠ `== None` / `== False` in SQLAlchemy filters are **load-bearing** (IS NULL /
  IS false); ruff's E711/E712 fixes would break the query. `registry.py`'s
  import-after-logger (E402) is likewise deliberate.
- ⚠ `uv run` **re-syncs and removes** anything installed with `uv pip install` —
  use `.venv/bin/python` for scratch work needing extra deps.
- ⚠ Scratch scripts importing `src/` must run **from the repo root**; `nltk`
  cannot import from there, so run text analysis from `/tmp` and pass JSON.
- ⚠ **Clear `__pycache__` between edit-and-rerun cycles** — a same-second,
  same-size edit is served stale (TRAPS #13c).
  `find src tests -name __pycache__ -type d -exec rm -rf {} +`
- `OLLAMA_KEEP_ALIVE=-1` — nothing leaves VRAM; `ollama stop <model>` before
  GPU-heavy tests. Long runs need `setsid nohup … &`.
- `pgrep -f '[u]vicorn …' | xargs -r kill` — never plain `pkill -f`, which
  matches and kills the shell running it (TRAPS #9).
- End-to-end validation against the live stack **leaves residue** — clean it up
  in the same session, and read TRAPS #15 before deleting anything.
- No cloud keys.

## WHEN DONE

Propagate on completion per the roadmap's own rules, update the docs the change
invalidates in the **same** session, then **rewrite this file** — carrying the
NEXT above into the "what the last session was told to do" section — and commit
it last.
