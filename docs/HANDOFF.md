# Handoff — ICE v3, 2026-10-17 (Claude Code)

**State, not a queue.** ROADMAP owns work; the v2 paper remains frozen.

## Told → did

**Previous handoff (2026-10-15 b) told:** the codex rework — remove and rewrite, not
patch (a relation is a term; facts carry validity, strength, confidence; the model sees
short close facts) — then Z2.

**This session did the codex rework end to end** ([CODEX_V4](specs/CODEX_V4.md), steps
1–10; ~50 local commits `9950854..HEAD`, none pushed — experiment-phase freeze). Z2 build
did not start, by that order.

- **Spec + qualification first:** blind grading (relationships gemma 14/16 vs NuExtract3
  9/16; sentences 9 right / 1 not self-contained / 2 wrong on the locked wording m5),
  §5.2 same/replaces/both-true 88/99 on 99 real pairs, name lookup 29/67. PROVENANCE
  2026-10-16.
- **Built:** one claim pass writes summaries and facts (`summary_assertion_method` now
  defaults to `v4`); fact writer, entity resolution, D6 reconciliation, facts leg, skip-if-
  source-in-prompt, D7 strength (support ⅔ + decayed usage ⅓), C10 deletion/forget over
  facts, maintenance merges that apply themselves with undo (D12).
- **Deleted the v2/v3 edge codex** (`def7309`): extractor, NuExtract3, claims, relation
  vocabulary, entity notes, compaction, 55 settings; 29 test files archived; ~40 suites
  ported. Migration `d5e8a1c3b7f2` **applied to ice_db** after a verified backup
  (`backups/ice_backup_20261017_070607_pre-codex-v4-migration.tar.gz`).
- **Bugs found while porting, fixed with tests that fail without the fix:** forget/
  deletion-expired facts leaking into history readers and the predecessor clause; as-of
  dropping facts stated before the window; one NER failure silencing the facts leg; the
  procedural leg's scope resolver deleted as "codex" (TRAPS 143–146).
- **Measured:** retrieval comparison on 67 dev questions — codex prompt tokens 54,767 →
  2,981, gold-turn coverage unchanged (101/108; all-gold 61 vs 62); one recorded turn
  through the replay path; re-embed proven on a migrated ice_db copy (5,227 rows).
  §5.3 strength bound: B = 0.30 (median adjacent rerank-logit gap) — inert on the dev
  corpus, 0/67 selections change at any B ≤ 2.0 (4 of 1,438 facts had a second source).

## Position

Codex rework done. **Next: G77, then build Z2** (spec unchanged). G77's five features
each have a decision brief in its ROADMAP entry — the next session starts by getting the
maintainer's yes/no per feature, then builds. Then Z2 in spec order (S0 → S12), with the
A6/A7 notes in ROADMAP CURRENT POSITION.

## Decided — do not re-derive

- Codex: facts, not edges; relation is a free term; gemma4:e4b writes and reconciles
  (a 27B cannot sit beside the background stack on 24 GB); no trigram tier; NuExtract3
  retired; review items for the codex retired (D12).
- No yield point inside the claim pass (abandoning mid-pass discards every piece
  written); the inert `timeline` leg weight is kept as a seam.
- Stays gated: `forget_request`, `finetune_proposal`.
- Migration applied to ice_db only after a verified `pg_dump` (maintainer, 2026-10-17);
  restore command in CLEANUP. CLAUDE.md code map updated (maintainer-approved).
- Z2 models: OpenCode free variants first (deepseek-v4.1-flash-free, mimo-v2.6-flash-
  free), Go only when the free quota is gone; Muse always on Go.

## The finding that outlives the items

A store with several reasons to end a row's validity needs every history reader to
say which reasons count — "expired" silently included forgotten and deleted content.
Each such leak was invisible to green suites until a ported test asked the question.

## Pending / known gaps

- D7's usage third is never exercised by read-only replays — Z2 measures it.
- Unframed pasted documents stay "asserted" (TRAPS 139) — Z2 D9 attribution axis.
- `tests/test_ingestion.py` "procedural rows appear" fails (pre-existing stub rot; the
  extractor needs a 3-turn session + cited-evidence JSON + semantic proof) — task chip.
- G78 re-embed batches by rows and OOMs on long turns (parked; not a Z2 risk).
- The old Z1 toolchain (seed_store, score_retrieval, …) still imports deleted code —
  left in place, runs at `8079eb2` (CLEANUP ledger).

## Validation

Quick group 530 passed after the last commit; full disposable rails 1,313 passed and every follow-up failure
ported and re-run green (chronological maintenance 15/15, C10/C11 55/55, services 54/54,
MCP 31/31, runtime 49/49, timescope 56/56, fail-open 27/27, coding core 54/54, facts
25/25). Alembic head `d5e8a1c3b7f2` on ice_db. Not pushed.
