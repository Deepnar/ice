HANDOFF — 2026-08-10, 18:20 IST. Overwritten every session, committed last.
It is STATE, never a queue: docs/ROADMAP.md is the queue and the only queue.
Record only what the other docs would not already carry. Previous handoffs are
in `git log -p docs/HANDOFF.md`.

Read CLAUDE.md → docs/TRAPS.md → docs/ROADMAP.md's "HOW TO EXECUTE THIS
ROADMAP". Those carry the standing rules, now SIXTEEN failure modes, and the
queue. This note carries only what they don't.

LAST SESSION WAS TOLD: "NEXT: G36, THEN G30, then G28." G36 first because
G30's job is trustworthy retrieval tests and a leg that fails silently makes a
passing test meaningless; G28 stays inside Z1.
WHAT HAPPENED: G36 done, G37 opened and closed on top of it, then a docs
restructure. G30 not started, G28 not touched — the order held. Scope grew
three times, each with an explicit go-ahead.

POSITION. CLUSTERS ①, ② COMPLETE. CLUSTER ③: G9 · G36 · G37 DONE.
 ③ G36 (fb07e7c reporting · b4a79a8 fail-closed · 4e3b43d HyDE deletion ·
   c4e07a7 tests · da6c30a docs) · G37 (c7f37fd · fb0e81a docs) ·
   6c71033 test_mcp_server slot leak.
 Docs restructure: c57f522 · 942d5a9 · 2662498 · d2710ec · e475702 · 9e0afbb ·
   eb83a81 · ee5800d · f50a5bb · 25ecbf9.
 ⚠ ALEMBIC HEAD UNCHANGED — d5c81a37e9b2, no migration.
 Inventory 51, and for the first time the counted set equals the written set.
 Store 1 row — the deterministic ice://mcp-notes shell, production not residue.

NEXT: G30, THEN G28. G30 inherits two things from this session:
tests/test_retrieval_failopen.py, the first suite that seeds codex entities
with REAL embeddings instead of the [0.05]*1024 stub — which is the exact blind
spot G30's own entry names — and the retrieval_leg_failed event, which is what
makes a green retrieval test mean anything. G28 still runs INSIDE Z1's coverage
matrix — do not pull it forward.

⚠ ASK BEFORE TOUCHING src/. Held this session: every commit was named
file-by-file before any edit, and each time scope grew it went back to the user.
⚠ NEW: ASK BEFORE TOUCHING CLAUDE.md TOO — every time, even one word.
⚠ ASSUME ./ice, ./stop_ice AND setup.sh DO NOT WORK. Unmaintained scaffolding;
conclude nothing from one of them failing.

THE PATTERN, SEVENTH CONSECUTIVE ENTRY: G36 WAS WRONG ABOUT ITS OWN SUBJECT.
Its count of 22 silent handlers was exact. Nothing else was — only 11 of the 22
rolled back, 4 were on dead code, and it missed the only dangerous members of
the set: TWO HANDLERS THAT FAIL OPEN, which turned out to be the actual bug.
⇒ Verify an entry's claims before building from it. This is now a roadmap rule
("check the ground before building on it") — restored after the CLAUDE.md
shrink deleted it by comparing two bullets BY HEADING RATHER THAN BY TEXT.

WHAT THIS SESSION DECIDED (recorded in the entries — do NOT re-derive):
 • One event name for every leg failure — retrieval_leg_failed, leg as a FIELD —
   including the nine handlers that already logged under their own names.
   Thirty-one names cannot answer "did any leg fail on this request".
 • Fix the fail-open direction, not merely log it (user).
 • A scope resolving to no batches reads NOTHING (user): "why would i need thing
   if i am starting anew, thats why the scoping for codex was built".
 • Rollback conditional on the error being the DATABASE's. Mandatory after a DB
   error — Postgres refuses every later statement in an aborted transaction —
   and gratuitous after a Python error, where it only expires the identity map
   of a session main.py keeps using.
 • HyDE deleted outright, not left commented (user), with a note in its place.
 • Exp 1's HyDE row: record now, re-run at FINAL. experiments/ and the .tex
   UNTOUCHED — the same disposition G19 took.
 • Finished roadmap entries are never compressed, only moved.

⚑ THE RULE THE NEXT SCOPE CHANGE DEPENDS ON: PRESENCE, NOT TRUTHINESS.
Failing closed whenever a batch set is empty would have taken the codex leg out
of EVERY `auto` request, and nothing in the suite would have noticed. The guard
asks whether the scope NAMED conversations ("conversation_ids" in scope), not
whether the result is non-empty. Three CONTROL checks pin it; they stay green
when the fix is reverted, which is how you know they are not mirroring it.

⚑ THE REPO IS PUBLIC, and has been for a while — TRAPS #12's incident was
already a public clone. Every push is a publication. Verified 2026-08-10 and
recorded in CLEANUP.md: check_history_clean.sh --clone clean against the live
remote, no personal/planning file tracked, no credential-shaped strings, README
links the canonical venue-agnostic paper, v2-paper-eval → 0521df9. A history
rewrite is NOT cheap — old objects stay fetchable by SHA until GitHub GCs them.

⚠ THE DOCS MOVED. ROADMAP.md is the queue ONLY (51 open + a one-line stub per
finished item; 65,475 → 32,202 words, preamble 540 → 266 lines). The 70 finished
entries are VERBATIM in ROADMAP_DONE.md, reachable from every stub, and the
anchors stayed in ROADMAP so (#g36) still resolves. The 17 dated session notes
that used to clog the preamble are in outdated/roadmap_session_log.md — THIS
FILE replaced them. CLAUDE.md is 38% shorter and had been wrong about the
running system in eight places (DI3, Celery/Redis, 384 dims, 25 logits, the live
checkpoint, Sentinel models, the leg-weight location, redis in docker).

OPENED / CORRECTED BY THE SWEEP:
 • THREE PHANTOM ITEMS given entries — G27, G34, G35. Each was counted in the
   inventory and never written; a session told to do one would have found
   nothing. G27's description had been decision-complete in specs/G_mechanical.md
   the whole time. G34 is the one to read: the relation detector puts 197/197
   relations above its floor for the prompt "ok", and raising the floor CANNOT
   fix it — absolute cosine is anti-correlated with relational content.
 • Z2 HAD BEEN HANDED FIVE PIECES OF WORK its own entry never named (three from
   C16, one from A9b, one from the C13/C14 question), recorded in a preamble
   triage block. ⇒ new roadmap rule: work goes in an ENTRY, never in preamble
   prose.
 • TRACK G WAS IN THREE PLACES — G31/32/33 wedged between G4 and G5, and
   G28/29/30/34/35/36/37 filed under Track H, "Research follow-ups". Now G1–G37
   in order. 36 of 49 cross-references were dangling; now 0.
 • The Z1/Z2 definition block was nested INSIDE Z2's entry, unreachable from Z1;
   lifted to the SEMIFINAL head. Z1's entry described the stack as
   "postgres+redis, celery worker+beat" — C7 deleted both — and is corrected.
 • G12 down to TWO sites; the third hardcoded timeout=15.0 was inside the
   deleted _hyde_rewrite. RETRIEVAL NOW MAKES NO LLM CALL AT ALL.
 • G19 IS 3-FOR-3. `hyde` joins recency_boost and the include_cross break:
   three ablation flags that toggled nothing, each corrupting a published arm.
   ⇒ Exp 1's full_ice_no_hyde was the SAME CONFIG as full_ice. The fold is the
   fix for a defect class now, and its design is settled in the entry.
 • G20 item (2) closed. G30's stale codex-write-path bullet marked closed.
 • ⚑ AN OLD QUESTION ANSWERED — why Exp 2 retrieved things from other
   conversations. At v2-paper-eval, _traverse_graph took NO SCOPE PARAMETER AT
   ALL: only the anchor was checked, then three hops across the whole graph.

STILL OWED ON G19 (design settled, not built): delete the shadow subclass. Use
an `ablation` dict on the ORCHESTRATOR INSTANCE via a small `_ablate(flag)`
helper — NOT settings toggles, because settings are global and ablation state
must stay per-instance. ConfigurableOrchestrator becomes a 3-line alias so the
frozen runners keep constructing. Plus procedural extraction in the replay loop
and the simulation_runs table.

STILL TWO DUPLICATED CONSTANTS from G9: context_budget_fallback and
context_total_budget_fallback, both 23,000 on different call paths. Merging is a
behaviour decision.

TRAPS GAINED: **15** — a deterministic id is NOT evidence of a test fixture. A
production row (bookmarks.NOTES_CONVERSATION_ID) was deleted as residue because
its id was a UUID5; restored with its original created_at, nothing lost. The
check that would have prevented it — `grep -rn uuid5 src/` — was run AFTER.
**16** — residue is not always inert. A memory_slots row leaked by
test_mcp_server was is_active=True + scope_tier='global', and main.py injects
every such slot into the system prompt — so a test marker was in EVERY TURN FOR
TWELVE DAYS. Cause: a snapshot-and-restore with no did-not-exist branch.

METHOD THAT EARNED ITS KEEP — a sys.settrace probe keyed on
frame.f_code.co_filename counted every exception propagating through an
orchestrator.py frame across 15 seeded suites: ZERO. That is what made "log
every swallow" the right call rather than noise (TRAPS #13b); re-run it after
any change to the legs. Also: my own first reproduction was VACUOUS — it
asserted only "the out-of-scope entity is absent" and passed against a fixture
returning nothing for either conversation, because the seeded embeddings were
the constant stub; the positive control is what made it real. Bisect before
naming a suite (TRAPS #6b) found the slot leaker in one command. And every
restructure step was word-accounted as before == after + archived ± header,
which is what proved nothing was lost.

VALIDATION, all green at close. smoke 123 · settings_freeze 147 ·
dynamics_invariants 17 · retrieval_failopen 27 (new) · classifier_v2 65 ·
timescope 61 · c10_c11 58 · documents 53 · coding_core 52 · maint_runtime 49 ·
services 48 · agent 43 · memory_decision 42 · session_scoping 40 · ingestion 36 ·
relation_gaps 33 · turn_density 32 · c4_c9 28 · longevity 27 ·
retrieval_coverage 25 · codex_write_path 23 · mcp_server 21 · context_budget 17 ·
reconcile_on_read 16 · density_c3 16 · document_chunking 15 · job_yield 14 ·
clustering_v5 13 · batch_summary_coverage 11 · c8_c15 8 · retrieval 3.
Every new behaviour check negative-controlled (revert ⇒ red). Ruff at parity
with base on every touched file. Docs: 0 dangling anchors, 0 broken file links.

ENVIRONMENT, the non-obvious parts only.
 ⚠ `== None` / `== False` in SQLAlchemy filters are LOAD-BEARING (IS NULL /
   IS false); ruff's E711/E712 fixes would break the query. registry.py's
   import-after-logger (E402) is likewise deliberate.
 ⚠ `uv run` RE-SYNCS AND REMOVES anything installed with `uv pip install` —
   use `.venv/bin/python` for scratch work needing extra deps.
 ⚠ nltk CANNOT IMPORT from the repo root. Run analysis from /tmp, pass JSON.
 ⚠ Scratch scripts importing src/ must run FROM the repo root.
 ⚠ CLEAR __pycache__ between edit-and-rerun cycles — a same-second, same-size
   edit is served stale (TRAPS #13c):
   `find src tests -name __pycache__ -type d -exec rm -rf {} +`
 OLLAMA_KEEP_ALIVE=-1 — nothing leaves VRAM; `ollama stop <model>` before
   GPU-heavy tests. Long runs need `setsid nohup … &`.
 `pgrep -f '[u]vicorn …' | xargs -r kill` — never plain `pkill -f`, which
   matches and kills the shell running it (TRAPS #9).
 End-to-end validation against the live stack LEAVES RESIDUE — clean it up in
   the same session, and read TRAPS #15 before deleting anything.
 No cloud keys.

WHEN DONE: propagate-on-completion per the roadmap's rules, update the docs the
change invalidates in the SAME session, then rewrite this file — carrying the
NEXT above into the "LAST SESSION WAS TOLD" line — and commit it last.
