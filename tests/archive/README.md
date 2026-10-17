# tests/archive/ — retired tests (frozen; do NOT run)

Dead/scratch tests moved here in the **2026-07-21 declutter** session, plus
`test_codex_2_0.py` (**2026-07-28** — see its row and the ⚠ note below). Kept (not
deleted) for the historical record — these are **broken or superseded** and are
**not part of any live suite**. Do not run or "fix" them; if you need what one
covered, the live replacement is named below. Recoverable from git history either
way (`git log --follow`).

The maintained suites are `tests/smoke/*` (pytest; the import-sweep walks `src/`
only, so nothing here was ever run by it) and the 22 spec-tied behavioral scripts
in `tests/` root.

| file | why retired | live replacement |
|---|---|---|
| `test_post_flight.py` | **broken import** — `is_lossless` was removed from `src/workers/post_flight.py` (C1); the import fails at module load | `test_turn_density.py`, `test_density_c3.py` |
| `test_full_pipeline_phase_9.py` | broken — reads missing `data/simulation_input.jsonl`, subprocesses missing `scripts/simulation/run_simulation.py`, needs Celery (removed C7); "already broken pre-C7" (CLEANUP ledger) | the C-track behavioral suite |
| `test_full_pipeline_phase_6.py` | early-phase eyeball test — needs Celery + old `ice_classifier_v2_final.pt` checkpoint; print-only, no assertions | `test_turn_density.py`; ~~`test_codex_2_0.py`~~ **also archived 2026-07-28 — no live replacement, flagged under G30** |
| `test_full_pipeline_phase_7.py` | early-phase — destructive `TRUNCATE` of shared tables every run, raw MiniLM-era `SentenceTransformer`; print-only | `test_retrieval.py` |
| `test_direct_codex.py` | scratch — hardcoded stale `batch_id`; no assertions | ~~`test_codex_2_0.py`~~ — **also archived 2026-07-28; no live replacement, flagged under G30** |
| `test_triplet.py` | scratch — manual raw-triplet print tool; no assertions | ~~`test_codex_2_0.py`~~ — **also archived 2026-07-28; no live replacement, flagged under G30** |
| `quick_probe_test.py` | scratch — old `v2_final` checkpoint, hardcoded Shinchan conv-id; no assertions | `test_retrieval.py` |
| `test_bg_json.py` | scratch — raw ping to stale port `:8001`; no assertions | — |
| `test_bg_non_thinking.py` | scratch — manual bg thinking-mode eyeball; no assertions | — |
| `test_judge.py` | scratch — 13-line raw `requests.post` to Ollama; no assertions, stale model tag | — |
| `test_codex_2_0.py` | **archived 2026-07-28; REPLACED 2026-08-03 by `tests/test_codex_write_path.py`** (20 checks, real model output, batch-namespaced rows and no TRUNCATE). Original reason for archiving: Three independent disqualifications at once: (1) **broken** — crashed at `orchestrator._load_ner_model()`, a method deleted when NER moved to `retrieval/ner_utils.py`; (2) **destructive** — opened with `TRUNCATE episodic_memory, conversations, codex_* … RESTART IDENTITY CASCADE` on the live DB, the same criterion that retired `test_full_pipeline_phase_7.py`; (3) two of its four sections were print-only or dead (extraction needs a live bg model; MERA was deleted by A4 and its section had degraded to an ImportError skip). | **`tests/test_codex_write_path.py`** — written once A9b/A12 had settled the behaviour it pins. |

---

## ⚠ Why `test_codex_2_0.py` was archived and NOT rewritten (2026-07-28)

It was rewritten once, in this session, and the rewrite was **reverted on the
maintainer's instruction** — correctly. The ground it covered (`handle_triplet`'s
write rules, and NER → entity-match → graph traversal) has **no other real
coverage anywhere**, but two scheduled items change exactly that ground:
**A9b** swaps the background NER for GLiNER behind the `extract_entities()`
seam, and **A9c** may retrain the pre-flight micro-NER. A test written now would
pin behavior both items exist to change.

**This is also why the 2026-07-21 sweep kept it, and why keeping it was wrong.**
That sweep named this file the *live replacement* for three of the tests above —
a promotion it could not carry, because it had already rotted. The sweep's method
checked whether a test's **imports** resolved; this one fails at **line 158**, at
runtime, so an import check could never see it. Re-check retired tests by
RUNNING them, not by importing them.

---

## `codex_v2_v3/` — the v2/v3 conversation graph's tests (archived 2026-10-16, CODEX_V4)

The CODEX_V4 rework deleted the edge codex: the NuExtract3 extractor
(`codex_extractor.py`, `nuextract_transport.py`, `extraction_result.py`), attributed
claims (`memory/claims.py`), relation/author support (`relation_support.py`,
`assertion_support.py`), entity notes, compaction, the relation vocabulary and
gap ledger, and the graph/claims/enumeration read legs. These suites tested only
that code; they cannot import on `main` after it. Last commit they run at:
`8079eb2` (or tag `v2-paper-eval` for the paper's v2 system).

| file(s) | covered | live replacement |
|---|---|---|
| `test_codex_write_path.py`, `test_g50_write_time_tier.py`, `test_extraction_completion.py`, `test_extraction_recovery.py`, `test_extraction_result.py`, `test_nuextract_transport.py` | NuExtract3 extraction, chunking, entity tiers on edges | `tests/test_codex_facts.py` (write, post-rules, entity tiers incl. `merge_key`), `tests/smoke/test_codex_v4_claims.py` (the claim pass) |
| `test_codex_claims.py`, `test_claim_evidence.py`, `test_author_assertion.py`, `test_author_assertion_writer.py`, `test_author_assertion_path.py`, `author_assertion_case.py` (helper), `test_codex_relation_support.py`, `test_source_relation_variants.py` | attributed claims, author/relation NLI | `tests/smoke/test_codex_v4_claims.py` (quote + NLI on every fact), `tests/smoke/test_source_support.py` |
| `test_conflict_evidence.py`, `test_ordered_reconciliation.py`, `test_ordered_reconciliation_path.py` | edge conflict/reconciliation | `tests/test_codex_facts.py::test_reconcile_*` (D6) |
| `test_relation_gaps.py`, `test_relation_fit.py`, `test_relation_killswitch.py` | the 197-word vocabulary, gaps ledger, relation fit/detection | none — a relation is a free term (D2) |
| `test_codex_fanout.py`, `test_v3_source_covered_graph.py`, `test_codex_source_scope.py`, `test_z1_quote_replay.py` | traversal, edge rendering, graph-quote skip, claim-leg scope | `tests/test_codex_facts_read.py` (scope through any source, skip-if-source-in-prompt, current + predecessor) |
| `test_codex_source_deletion.py` | C10 deletion over edges/claims | `tests/test_codex_facts.py::test_delete_conversation_*`, `test_forget_*` |
| `test_graph_retention.py` | edge strength retention | `tests/smoke/test_v3_lifecycle_plan.py` (fact usage decay), `tests/test_codex_facts_read.py::test_usage_*` |
| `test_entity_notes.py`, `test_entity_note_path.py` | reflection entity notes | none — removed for conversation entities (D11.1); the entity view lists facts |
| `test_compaction_checkpoint.py` | compaction snapshots | none — removed (D11.2) |
