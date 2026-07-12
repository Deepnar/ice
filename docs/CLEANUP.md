# Codebase cleanup — standing rules + ledger

Standing rule (user, 2026-07-10): **clean as we go, so it never becomes a
massive end-task.** Every implementation session leaves the files it touches
cleaner than it found them. Organizing, never destroying.

## The rules

1. **Touched-files only, never big-bang.** Cleanup applies to files a session
   already edits for its feature. No repo-wide reformat commits (they bury the
   real diff and pollute blame).
2. **Imports:** sorted + grouped (stdlib / third-party / local) and unused ones
   dropped — `uv run ruff check --fix <touched files>` + `ruff format` on those
   files. **No barrel re-exports in `__init__.py`** — they add import-time side
   effects, hide provenance, and pull heavy deps (torch) transitively; this
   codebase deliberately uses *lazy in-function imports to break circular
   dependencies* — keep those, and comment WHY on each.
3. **One-off scripts:** anything written for a single occasion goes to
   `scripts/oneoff/` (create per need) — moved, never deleted; fix any imports/
   paths the move breaks; new one-offs start there. `scripts/` keeps only
   living tools (training, database, ingestion).
4. **Frozen zones — do NOT reorganize:** the pre-FINAL `experiments/*` folders
   (mature/unmature/flaw_ablation + their scripts and results) are the paper
   era's historical record. FINAL builds fresh in `experiments/final/`. `data/`
   and `models/` artifacts likewise stay put.
5. **While in a file:** dead locals, commented-out corpses (unless a spec says
   keep, e.g. the relabeled `_hyde_rewrite`), and comments that lie about the
   code get fixed in place. Naming drift (same concept, two names) gets fixed
   only within the touched files.
6. **Log it.** Every move/rename or non-obvious cleanup gets one ledger line
   below — so old paths stay findable and nothing is ever "mysteriously gone."
7. **Never delete data, configs, or anything user-authored.** When in doubt:
   move to `scripts/oneoff/` or leave + note.

## Ledger

| date | what | from → to / change | why |
|---|---|---|---|
| 2026-07-10 | ledger created; ruff+pytest added as dev deps | — | phase-0 rails |
| 2026-07-11 | `scripts/replay_buffer.py` → `scripts/oneoff/replay_buffer.py` | moved + dead-since-C7 header note | the jsonl buffer it replays died with the Celery broker (C7 D8); nothing writes it anymore |
| 2026-07-11 | `src/workers/celery_app.py` deleted | replaced by `src/workers/runtime.py` (in-process maintenance runtime) + `src/api/core.py` | C7 D1 — celery+redis out of the stack |
| 2026-07-11 | legacy `.delay(...)` callers updated to direct calls | `tests/test_codex_extractor.py`, `tests/test_full_pipeline_phase_6.py`, `tests/test_full_pipeline_phase_9.py`, `scripts/insert_curated_and_fine_tune.py` | celery API gone; note: phase_9 was already broken pre-C7 (imports `is_lossless`, removed by C1) — import fixed, file otherwise untouched |
| 2026-07-11 | dead SGLang block + lying "port 8003" comment removed | `src/workers/bg_client_factory.py`, `src/workers/codex_extractor.py` | G2 (folded into C7) |
| 2026-07-12 | dormant recency tiebreakers removed from `_rows_to_fragments` | `src/retrieval/orchestrator.py` (~1826 age-hours bonus, ~1832 newer-count bonus) | dead since pre-experiment `616d770` (no leg SELECT carried `timestamp`); T1 re-added the column for date stamps — deleting beats silently awakening untuned scoring (T_temporal.md rev note 2) |
| 2026-07-12 | unused imports dropped (`os`, `openai.OpenAI`, `MemorySlot`; fn-local `BatchSummary`) | `src/retrieval/orchestrator.py` | boy-scout on touched file (Track T session) |
