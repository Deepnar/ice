# Session handoff

**Written 2026-08-10.** Covers two sittings: G36/G37 (2026-08-09) and the
documentation restructure + the public flip (2026-08-10).

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
>    the incoming session can compare what was *supposed* to happen against what
>    the git log says happened. That comparison is the point of the file.
> 3. **Record only what would otherwise be lost.** If ROADMAP, TRAPS,
>    PROVENANCE, CLEANUP or ICE_Architecture already says it, link it.
>
> Overwritten in place — earlier handoffs are in `git log -p docs/HANDOFF.md`.

---

## WHAT THE LAST SESSION WAS TOLD TO DO, AND WHAT HAPPENED

**Told:** `NEXT: G36, THEN G30, then G28.` G36 ahead of G30 because G30's job is
trustworthy retrieval tests, and a leg that fails silently makes a passing test
meaningless. G28 stays inside Z1 and must not be pulled forward.

**Happened:** **G36 done, G37 opened and closed on top of it, then a full
documentation restructure.** G30 was not started and G28 was not touched, so the
order held. Scope grew three times, each with an explicit go-ahead: fix the
fail-open direction rather than only logging it; close G37; restructure the docs.

---

## POSITION

**Cluster ③ of the route into Z1: [G9](ROADMAP.md#g9) · [G36](ROADMAP.md#g36) ·
[G37](ROADMAP.md#g37) done. [G30](ROADMAP.md#g30) is next.**

- ⚠ **ALEMBIC HEAD UNCHANGED — `d5c81a37e9b2`.** No migration this cycle.
- **51 open items**, all mechanically countable top-level `- [ ]` lines.
- **Store: 1 row** — the deterministic `ice://mcp-notes` conversation shell,
  which is production state, not residue.

### ⚑ THE REPO IS PUBLIC as of 2026-08-10

`github.com/Deepnar/ice`. **Every push is now a publication**, not a backup.
Pushing stays pre-authorized. Verified at the flip and recorded in
[CLEANUP.md](CLEANUP.md): `check_history_clean.sh --clone` clean against the
live remote, no personal or planning file tracked, no credential-shaped strings,
README links the canonical venue-agnostic paper, `v2-paper-eval` → `0521df9`.
⚠ **A history rewrite is no longer cheap** — the old objects stay fetchable by
SHA until GitHub GCs them (TRAPS #12 — a support ticket last time). Fix forward.

### Where things live now (the docs moved this session)

| File | Holds |
|---|---|
| [ROADMAP.md](ROADMAP.md) | **The queue only** — 51 open items plus a one-line stub per finished one. 65,475 → 32,202 words; preamble 540 → 266 lines. |
| [ROADMAP_DONE.md](ROADMAP_DONE.md) | The 70 finished entries **verbatim**. Every stub links here; the anchors stay in ROADMAP so `(#g36)` still resolves. |
| [outdated/roadmap_session_log.md](outdated/roadmap_session_log.md) | The 17 dated session notes + 4 stacked inventories that used to clog the preamble. **This file replaced them.** |
| [../CLAUDE.md](../CLAUDE.md) | 38% shorter, and it had been **wrong about the running system in eight places**. **Editing it now requires asking the user first, every time.** |

⚠ **Assume `./ice`, `./stop_ice` and `setup.sh` DO NOT WORK.** Unmaintained
scaffolding awaiting Track F's packaged app; conclude nothing from one failing.

## NEXT

**[G30](ROADMAP.md#g30), then G28.** G30 inherits two things from this session:
`tests/test_retrieval_failopen.py` as the worked example for its own headline
complaint — it is the first suite seeding codex entities with **real** embeddings
rather than the `[0.05]*1024` stub — and the `retrieval_leg_failed` event, which
is what makes a green retrieval test mean anything. **G28 runs INSIDE Z1's
coverage matrix — do not pull it forward.**

---

## ⚠ COMMIT FORMAT — READ THE LOG BEFORE WRITING ONE

This drifted badly across the 2026-08-09/10 commits: no area prefix, no roadmap
item id, no `Validated` line, prose bodies with rhetorical structure. **The rule
was already in CLAUDE.md and the log was simply never read.** It is now spelled
out there with a template, and `git log 6bac35d 1a30484 051cea9 5124efb` is the
reference. Granularity and format are **one** rule: a small commit still takes
the full shape.

⚠ The ~12 off-format commits from 2026-08-09/10 are **left as they are**. A
rewrite would invalidate ~20 hashes cited across five docs, and the repo is now
public. Not worth it — fix forward.

## THE PATTERN, SEVENTH CONSECUTIVE ENTRY: G36 WAS WRONG ABOUT ITS OWN SUBJECT

Its count of 22 silent handlers was exact. Nothing else was: only **11** of the
22 rolled back, **4 were on dead code**, and it missed the only dangerous members
— **two handlers that fail OPEN**, which were the actual bug. ⇒ **Verify an
entry's claims before building from it.** The roadmap now carries this as a rule
("check the ground before building on it").

## WHAT WAS DECIDED — do NOT re-derive

- **One event name for every leg failure**, including the nine that already
  logged under their own names. Thirty-one names cannot answer "did any leg fail
  on this request".
- **Fix the fail-open direction, do not merely log it.**
- **A scope resolving to no batches reads nothing**: *"why would i need thing if
  i am starting anew, thats why the scoping for codex was built in the first
  place."*
- **Rollback is conditional on the error being the database's.** Mandatory after
  a DB error (Postgres refuses every later statement in an aborted transaction);
  gratuitous after a Python error, where it only expires the identity map of a
  session `main.py` keeps using.
- **HyDE deleted outright**, not left commented, with a note in its place.
- **Exp 1's HyDE row: record now, re-run at FINAL.** `experiments/` and the
  `.tex` untouched — the same disposition G19 took.
- **Finished roadmap entries are never compressed**, only moved.

## ⚑ THE RULE THE NEXT SCOPE CHANGE DEPENDS ON

**Presence, not truthiness.** Failing closed whenever a batch set is empty would
have taken the codex leg out of **every `auto` request**, and nothing in the
suite would have noticed. The guard asks whether the scope *named* conversations
(`"conversation_ids" in scope`), not whether the result is non-empty. Three
CONTROL checks in `test_retrieval_failopen.py` pin it; they stay green when the
fix is reverted, which is how you know they are not mirroring it.

## OPENED / CORRECTED / CLOSED

- **[G37](ROADMAP.md#g37)** opened and closed same day — full record in its entry.
- **Three phantom items given entries: [G27](ROADMAP.md#g27),
  [G34](ROADMAP.md#g34), [G35](ROADMAP.md#g35)** — each counted in the inventory,
  none ever written. G27's description had been decision-complete in
  `specs/G_mechanical.md` the whole time. **G34 is the interesting one:** the
  relation detector puts 197/197 relations above its floor for the prompt `"ok"`,
  and raising the floor cannot fix it because absolute cosine is
  *anti-correlated* with relational content.
- **[Z2](ROADMAP.md#z2) had been handed five pieces of work its entry never
  named** — three from C16, one from A9b, one from the C13/C14 question. Now in
  the entry. ⇒ new roadmap rule: **work goes in an entry, never in preamble prose.**
- **Track G was in three places** (seven items filed under Track H); now G1–G37
  in order. **36 of 49 cross-references were dangling**; now 0.
- **A standing rule was lost and restored** (`d2710ec`) — the CLAUDE.md shrink
  compared a rule against its ROADMAP twin **by heading rather than by text**.
  That is why CLAUDE.md edits are now user-gated.
- **[G12](ROADMAP.md#g12) is down to two sites** — the third hardcoded
  `timeout=15.0` was inside the deleted `_hyde_rewrite`. **Retrieval now makes no
  LLM call at all.**
- **[G19](ROADMAP.md#g19) has a 3-for-3 record.** `hyde` joins `recency_boost`
  and the `include_cross` break: three ablation flags that toggled nothing, each
  corrupting a published arm. **The fold is the fix for a defect class now**; its
  design is settled in the entry.
- **G20 item (2) closed.** **G30's stale codex-write-path bullet** marked closed.
- **⚑ AN OLD QUESTION ANSWERED.** Why Experiment 2 retrieved things from other
  conversations: at `v2-paper-eval`, `_traverse_graph` took **no scope parameter
  at all** — only the anchor was checked, then three hops across the whole graph.

## TRAPS GAINED

**#15** — a deterministic id is not evidence of a test fixture. A production row
(`bookmarks.NOTES_CONVERSATION_ID`) was deleted as residue because its id was a
UUID5; restored with its original `created_at`, nothing lost. The check that
would have prevented it, `grep -rn uuid5 src/`, was run *after*.
**#16** — residue is not always inert. A `memory_slots` row leaked by
`test_mcp_server` was `is_active=True` + `scope_tier='global'`, and `main.py`
injects every such slot into the system prompt — **in every turn for twelve
days**. Cause: a snapshot-and-restore with no did-not-exist branch.

## METHOD THAT EARNED ITS KEEP

- **A `sys.settrace` probe keyed on `frame.f_code.co_filename`** counted every
  exception propagating through an `orchestrator.py` frame across 15 seeded
  suites: **zero**. That made "log every swallow" the right call rather than
  noise (TRAPS #13b). Re-run it after any change to the legs.
- **My own first reproduction was vacuous** — it asserted only "the out-of-scope
  entity is absent" and passed against a fixture returning nothing for *either*
  conversation, because the seeded embeddings were the constant stub. The
  positive control is what made it real.
- **Bisect before naming a suite** (TRAPS #6b) found the slot leaker in one
  command: clear the table, run one suite, count.
- **Word-accounting a doc move.** Every restructure step was checked as
  `before == after + archived ± header`, which is what proved nothing was lost.

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
with base on every touched file. Docs: 0 dangling anchors, 0 broken file links.

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

Propagate on completion per the roadmap's rules, update the docs the change
invalidates in the **same** session, then **rewrite this file** — carrying the
NEXT above into the "what the last session was told to do" section — and commit
it last, in the format CLAUDE.md pins.
