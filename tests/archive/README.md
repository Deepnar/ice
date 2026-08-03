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
