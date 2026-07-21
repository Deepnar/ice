# tests/archive/ — retired tests (frozen; do NOT run)

Dead/scratch tests moved here in the **2026-07-21 declutter** session. Kept (not
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
| `test_full_pipeline_phase_6.py` | early-phase eyeball test — needs Celery + old `ice_classifier_v2_final.pt` checkpoint; print-only, no assertions | `test_turn_density.py`, `test_codex_2_0.py` |
| `test_full_pipeline_phase_7.py` | early-phase — destructive `TRUNCATE` of shared tables every run, raw MiniLM-era `SentenceTransformer`; print-only | `test_retrieval.py` |
| `test_direct_codex.py` | scratch — hardcoded stale `batch_id`; no assertions | `test_codex_2_0.py` |
| `test_triplet.py` | scratch — manual raw-triplet print tool; no assertions | `test_codex_2_0.py` |
| `quick_probe_test.py` | scratch — old `v2_final` checkpoint, hardcoded Shinchan conv-id; no assertions | `test_retrieval.py` |
| `test_bg_json.py` | scratch — raw ping to stale port `:8001`; no assertions | — |
| `test_bg_non_thinking.py` | scratch — manual bg thinking-mode eyeball; no assertions | — |
| `test_judge.py` | scratch — 13-line raw `requests.post` to Ollama; no assertions, stale model tag | — |
