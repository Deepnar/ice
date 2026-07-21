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
| 2026-07-12 | import blocks ruff-sorted (I001); unused `uuid` + `CodexEvent` imports dropped from reflection | `src/retrieval/orchestrator.py`, `src/retrieval/evolution.py`, `src/workers/reflection.py`, `src/workers/codex_inject_watcher.py` | boy-scout on touched files (T4 session); reflection's only `uuid`/`CodexEvent` uses died with the `context_appended` emit D13 replaced |
| 2026-07-12 | `ContextFragment.source_type` comment updated to list `"timeline"` | `src/retrieval/orchestrator.py` | comment lied by omission after T4 added the fragment type |
| 2026-07-12 | new one-off: `scripts/oneoff/paper_bootstrap_cis.py` | bootstrap 95% CIs for the paper's headline numbers, faithfully replicating compute_metrics_no_ice_dev.py's imputation + tournament counting | reads the frozen results read-only; point estimates verified to match every published table before trusting the intervals |
| 2026-07-12 | `docs/{ARCHITECTURE.md, ARCHITECTURE_V2.md, paper_rough_notes.md, related_work_notes.md}` → `docs/outdated/` (+ README there) | superseded/paper-era docs quarantined, never edited | user decision: keep "what the system was" separate from living docs; ARCHITECTURE_V2 is NOT the evaluated-system report — `docs/ICE_Architecture[real_v2].md` (user-added) is, and stays at docs/ root because the paper cites it |
| 2026-07-12 | git tag `v2-paper-eval` → e4019b6 ("ice v2 finished") | annotated tag marking the paper-evaluated snapshot | results gathered at 53c6a71 one minute earlier; e4019b6 completes the tree with the then-untracked migrations the evaluated system used |
| 2026-07-13 | added TMLR submission build: `experiments/paper/ICE_paper_tmlr.tex` (+ official `tmlr.sty`, `tmlr.bst`) | anonymized double-blind port of ICE_paper.tex onto the official TMLR style (github.com/JmlrOrg/tmlr-style-file); generic `ICE_paper.tex` kept for later arXiv | verified 30pp, no name/email/institution leak in text or PDF metadata; camera-ready = `[accepted]` option + uncomment author (see file header) |
| 2026-07-17 | `VALID_SLOTS` + slot helpers moved: `src/api/routers/memory_slots.py` → `src/services/slots.py` | routers now import from the service; single constant is C9's widening seam | E0 service extraction — one implementation, three adapters |
| 2026-07-17 | review-approve dispatch moved: `src/api/routers/user_control.py` → `src/services/review.py::approve` | + gained the D1/D2 `entity_merge`/`codex_reconciliation` arms; `merge_entities` stub born in `src/workers/codex_ops.py` | E0 D2(b) — the one intentional behavior change of the extraction |
| 2026-07-17 | routers rewritten as thin adapters; shared error translation added at `src/api/routers/adapter.py` | user_control.py 279→160 lines, memory_slots.py 172→63; byte-identical responses (parity 31/31, `tests/test_router_parity.py`; baseline lives in gitignored `logs/` — contains live-DB content, never commit) | E0 |
| 2026-07-17 | `errors.py` docstring reworded to avoid the literal grep-gate token; unused imports dropped from new services (`uuid`, `Optional`, `CodexEvent` in tests) | `src/services/*`, `tests/test_services.py` | boy-scout + keeps `grep fastapi\|HTTPException src/services src/mcp` empty |
| 2026-07-17 | pyproject gains `[build-system]` (hatchling) + `[project.scripts] ice-mcp` | project now installs editable into the venv (`ice==0.1.0`); `src.*` import paths unchanged | E7 — uv installs no entry points for build-system-less projects (spec rev 2) |
| 2026-07-17 | `src/workers/sentinel_monitor.py` deleted; `SentinelRule`/`SentinelEvent` models + tables dropped (migration `f7a3d9c21e46`, seed rules archived into the migration log first) | replaced by `src/workers/maintenance_agent.py` (D1); its two real checks live on as agent detectors 2 (pending-edge pileup) and 5 (stale pending_items slot) | D2 — audit verdict removal, not completion |
| 2026-07-17 | `scripts/seed_sentinel_rules.py` → `scripts/oneoff/seed_sentinel_rules.py` | dead once the table dropped (imports the removed model — historical record only) | D2; never-delete rule |
| 2026-07-17 | `review.py` no-op `sentinel_review` arm removed; stale "loud stub" docstrings fixed (`review.py`, `codex_ops.py`, `test_services.py`); `sentinel_events` dropped from phase_9's TRUNCATE list; runtime's sentinel comment updated | behavior identical (unmatched review types already just flip status; parity 31/31 re-verified) | D1/D2 boy-scout on touched files |
| 2026-07-18 | `tests/test_mcp_server.py` ice_where assertion updated (`"E1b" in engine` → `"codex" in engine`) | the E1b-gated honest-limit description died with the engine swap — the code graph is now the primary engine, codex name/alias the fallback | E-coding core; no moves/renames/deletes this session (new files only; ruff import-sort on the two files it flagged) |
| 2026-07-19 | latent PgVector bind bug fixed in place: `_procedural_lookup` + `_rag_lookup` embedding params gained `bindparams(type_=PgVector)` | `src/retrieval/orchestrator.py` | found during C9's widening — both legs raised `vector <=> double precision[]` on EVERY call and silently returned [] (exception → rollback → empty); the procedural intent gate had been hiding a 100%-dead leg. No moves/renames/deletes this session (new files only: migration, `src/workers/conversation_summary.py`, `tests/test_c4_c9.py`, `tests/test_reconcile_on_read.py`) |
| 2026-07-19 | `tests/test_services.py` review section updated to the C9-D7 contract (conversation-tier fixture slot instead of a marker-named direct-row write) | `tests/test_services.py` | the old test relied on approve's unvalidated direct write — exactly what D7 replaced; the new check also asserts `proposed_by` → `updated_by` |
| 2026-07-19 | `initialize_slots`' lying "unique constraint" docstring made true | `src/services/slots.py` + migration `a7c5e91d3f28` | memory_slots had NO name uniqueness (only the id pkey); C9's NULLS-NOT-DISTINCT index now provides what the comment always claimed |
| 2026-07-19 | `ReviewQueue.status` comment fixed: `# pending, approved, rejected` → the full pending/approved/rejected/resolved/stale vocabulary | `src/memory/models.py` | was already lying about `resolved` (live since D1/D2); C10 adds `stale` — the comment now matches every writer. No moves/renames/deletes this session (new files only: `src/services/conversations.py`, `src/api/chat_commands.py`, `tests/test_c10_c11.py`; ruff import-sort dropped one unused import from conversations.py) |
| 2026-07-19 | `tests/test_retrieval.py` rewritten house-style: its `TRUNCATE episodic_memory, conversations, codex_entities, codex_edges CASCADE` on every run deleted (the truncating-tests half of BRUTAL_ASSESSMENT's ⚡⚡ item) + its `all-MiniLM-L6-v2` embedder (the MiniLM-era relic the whole 384 shim existed for) replaced with the shared `get_embedder()` | `tests/test_retrieval.py` | G23/C17; now marked fixtures + cleanup-in-finally + 3 real checks |
| 2026-07-19 | five per-worker `SentenceTransformer(..., truncate_dim=384)` copies consolidated into `src/memory/embedder.py::get_embedder()` (classifier, codex_extractor, procedural_extractor, batch_summarizer, clustering; fine_tune/dataset.py keep their training-device instances but slice384) | one process = one embedder (G13 made true); smoke 34s→13s | C17/D6; stub vectors in 18 test files widened 384→1024 (`* 383`→`* 1023` etc.); `.gitignore` gains `backups/` + `exports/` (private archives). New files this session: `src/memory/{embedder,store_meta,portability,reembed,backup}.py`, `scripts/{ice_backup.sh,ice_export.py,ice_import.py,ice_reembed.py}`, migration `b6e2f9a41c73`, `tests/test_longevity.py`; no moves/renames/deletes |
| 2026-07-20 | no moves/renames/deletes — new files only: `src/ingestion/{__init__,importer,formats,raw_slicer}.py`, `src/services/ingestion.py`, `scripts/ice_replay_import.py`, migration `69873bf8e0c8`, `tests/test_ingestion.py`, `tests/fixtures/ingestion/*` | F10/F14 conversation import | `src/ingestion/__init__.py` is intentionally empty (no barrel re-exports, standing rule); ruff clean on all touched files; the real Claude/DeepSeek exports in `data/simulation/raw_chats/` were read for FORMAT reference + dry-run only, never imported or committed as fixtures (synthetic fixtures used instead) |
| 2026-07-20 | `scripts/oneoff/paper_bootstrap_cis.py` → `experiments/paper/exp2_bootstrap.py` (git mv; `Run:` docstring path updated) | paper reframe | co-locates the Exp-2 CI script beside `exp3_bootstrap.py` as the paper's self-contained CI-repro bundle; re-verified it reproduces the published Exp-2 CIs from the new path (sys.path `../..` still resolves to repo root) |
| 2026-07-21 | **pre-B1 cleanup — DATA (user-authorized exception to rules 4 + 7)**: DELETED `data/ner/embeddings_cache.pt` (4.4 GB, stale old-384 train cache) + empty `data/dataset/`; MOVED `data/ner/*.jsonl`, `data/unlabeled/`, `data/raw_logs/` → `data/archive/` (+ `data/archive/README.md`) | `data/` 4.8 GB → 411 MB | user explicitly approved deleting the regenerable/stale cache + archiving the rest before the B1 retrain (overrides the standing "data stays put / never delete data" rules for this session only). Corpus backed up FIRST → `backups/classifier_data_backup_20260721_185005.tar.gz` (9.8 MB, gitignored — the 25k corpus had no prior backup). `data/archive/` added to `.gitignore` |
| 2026-07-21 | **pre-B1 cleanup — CLASSIFIER SCRIPTS**: `scripts/classifier/{promt_extraction,promt_labeling}`, `scripts/training/`, and loose `scripts/{build_probe_input,probes_count,ltm_fix,insert_curated_and_fine_tune}.py` → `scripts/classifier/legacy/` (git mv) | four scattered locations → one home | pre-B1 consolidation; `scripts/classifier/pipeline/` created for the v2 rewrites; `scripts/classifier/README.md` maps old→new and preserves what each legacy piece did. Nothing in `src/`/`tests/` imports these (verified); `scripts/` root now holds only live ops tools (`ice_*`, `register_project`) |
| 2026-07-21 | **pre-B1 cleanup — NER SCRIPTS**: `scripts/ner/*.py` → `scripts/ner/legacy/` (git mv) | stale (built for old 384 encoder) | pre-B1; `scripts/ner/README.md` notes the A9/B1 1024 rework + the archived (`data/archive/ner/`) / deleted (4.4 GB cache) intermediates; legacy `data/ner/…` paths in these scripts no longer resolve (record, not runnable) |

## 2026-07-21 — whole-tree DECLUTTER session (dedicated; post-`b0d3e5f`, before B1 resume)

Second slice of the declutter effort (the first was `b0d3e5f`). Method: exhaustive
read-only inventory + grep of every candidate BEFORE moving; explicit maintainer
sign-off on all deletions; `git mv` for tracked (history = recovery), plain `mv`
for gitignored local data (reversible), every script's file-path references
rewritten + re-grep-verified. Tracked deletions recover from **`b0d3e5f`** (session
HEAD). Validation: `pytest tests/smoke` 82 passed; all `experiments/**/*.py` compile;
grep proves zero stale `results/<intermediate>` refs and no shallow `../..`/`.parent`
depth in moved one-offs.

### A. Repo-root strays

| date | what | from → to / change | why |
|---|---|---|---|
| 2026-07-21 | **DELETED** `celerybeat-schedule.db` | DELETED (tracked; recover git `b0d3e5f`) | stale Celery beat scheduler DB; Celery removed in C7 (`celery_app.py` gone) — never regenerates, 0 refs |
| 2026-07-21 | **DELETED** `main.py` (repo root) | DELETED (tracked; recover git `b0d3e5f`) | `uv init` hello-world stub; real entrypoint is `src.mcp.server:main` (pyproject `[project.scripts]`); hatchling packages only `src`, so not in build; 0 real refs |
| 2026-07-21 | **DELETED** `docs/SRC_STRUCTURE.md` | DELETED (tracked; recover git `b0d3e5f`) | generated by `raw_src.py`, 0 refs; **regenerates** via `scripts/oneoff/raw_src.py` |
| 2026-07-21 | root one-offs → `scripts/oneoff/` (git mv) | `T1.py`, `create_cluster_links.py`, `raw_src.py`, `raw_append.py`, `extract_raw.py`, `extract_raw_src.py`, `generate_ice_doc.py`, `split_ice_doc.py`, `modelfile_qwen3_4_32` | scratch/doc-generator/DDL one-offs + an Ollama Modelfile; all 0 refs; abs/CWD-relative paths unaffected by the move |
| 2026-07-21 | `scripts/data/*.py` → `scripts/oneoff/` (git mv + header note) | `extract_claude.py`, `extract_deepseek.py`, `extract_gpt.py`, `merge.py`; empty `scripts/data/` removed | one-off sim-corpus prep that built `data/simulation/simulation_full.jsonl` (frozen Exp-1/2/3 input); **overlaps/superseded by** `src/ingestion/formats.py` (F10) — recorded in each file's header; kept as the corpus reproduction path |
| 2026-07-21 | `data/simulation/separate/` (27 per-conv jsonls, gitignored) → `data/archive/simulation_separate/` (plain mv) | archived, not deleted (reversible) | dead derived intermediates — per-conversation slices of `simulation_full.jsonl` (conv-ids present in the master, e.g. one appears 25×); 0 code refs |

Loose `data/simulation/` exports (`chatgpt_1.json`, `chatgpt_2.json`, `gemini.html`,
`deepseek.json`) were reviewed and **left untouched** by maintainer decision (personal
data, possibly staged for F10 import testing).

### B. `experiments/` reorg (maintainer-directed; **nothing deleted** — frozen paper zone)

Target shape per experiment: **main pipeline scripts at top**, `oneoff/` = test/debug/fix
scripts, `intermediates/` = raw+progress files, `results/` = only final `paper_summary*.md`
+ final metrics `*.json`. `.tex` cites only numbers (no script/result paths), so
relocations don't touch the paper build. Gitignored data files moved with plain `mv`;
`.gitignore` consolidated old per-file ignores into `intermediates/`+`results/` patterns.

| date | what | from → to / change | why |
|---|---|---|---|
| 2026-07-21 | **mature/** one-offs → `mature/oneoff/` (git mv) | `clean_repetitions`, `collapse_repeats`, `debug_probe`, `dedup_answers`, `deep_clean`, `diagnose_codex`, `diagnostics`, `fix_flaw_progress`, `fix_shinchan_tokens`, `test_claw_machine_now`, `test_probes_manual`, `test_retrieval_fix`, `test_single_probe`, `test_thinking`, `test_vector_baseline` `.py` | declutter; each moved script's `__file__` depth (`.parent`→`.parent.parent`, `../..`→`../../..`) + `results/`→`intermediates/` paths rewritten |
| 2026-07-21 | **mature/** intermediates → `mature/intermediates/` (plain mv, gitignored) | `generated_probes.json`, `_completed.txt`, `_last_turn.txt`, `_corrected_gt_progress.txt`; from `results/`: `master_results.json`, `evaluation_raw.json`, `fragments.jsonl`, `manual_evaluation.json`, `corrected_ground_truths.json` | `results/` now holds only `metrics_complete_report*.json` + `*paper_summary*.md`; 23 mature scripts repointed (`mig_mature`) |
| 2026-07-21 | **unmature/** one-offs → `unmature/oneoff/` (git mv) | `append_claude_to_flaw`, `fix_exp1_tokens`, `fix_flaw_timestamps`, `invalid_ground_truth` `.py` | declutter |
| 2026-07-21 | **unmature/** intermediates → `unmature/intermediates/` (plain mv, gitignored) | from `results_phase2/`: `_completed.txt`, `evaluation_raw.json`, `ground_truth_progress.json`, `master_results.json`, `master_results_corrected.json`, `vector_contexts.json` | `results_phase2/` now holds only `metrics_complete_report.json` + `paper_summary.md`; 7 scripts repointed AND the pre-existing stale `experiments/results_phase2/` prefix corrected → `experiments/unmature/results_phase2/` |
| 2026-07-21 | **flaw_ablation/buildup/** split | intermediates → `buildup/intermediates/` (`_completed.txt`, `evaluation_raw.json`, `fragments.jsonl`, `ground_truths_for_review.jsonl`, `master_results.json`); results → `buildup/results/` (`metrics_report.json`, `paper_summary.md`) | same treatment; `buildup_*` scripts repointed |
| 2026-07-21 | **flaw_ablation/subtraction/** | empty `subtraction/results/` + `subtraction/intermediates/` created; scripts repointed | consistency (subtraction produced no data files yet) |
| 2026-07-21 | **paper/** → tex+pdf+sty+bst + `notes/` only | `exp2_bootstrap.py`, `manual_eval_table.py` → `mature/`; `exp3_bootstrap.py` → `flaw_ablation/buildup/`, `exp3_bootstrap_report.{json,md}` → `flaw_ablation/buildup/results/`; `FIDELITY_AUDIT.md`, `REVISION_PLAN_v2.md` → `paper/notes/`; `sync_tmlr.py` → `scripts/oneoff/` (its `HERE` repointed to `experiments/paper`) | maintainer: paper folder holds only the paper; bootstrap scripts+reports live with their experiment (exp2=mature, exp3=ablation). Gitignored LaTeX build artifacts (`*.aux/.log/.out/…`) left in place (regenerate via latexmk) |
| 2026-07-21 | `scripts/citation_check/` → `experiments/citation_check/` (git mv) | whole dir (`verify_citations.py`, `references.json`, `RELATED_WORK.md`, `report.md`, `VERIFICATION_MEMO.md`) | maintainer: citation tooling belongs with the paper/experiments. `--refs` default in `verify_citations.py` and the ref in `docs/PUBLISHING.md` updated `scripts/`→`experiments/` |
| 2026-07-21 | **cross-experiment path fix** | `buildup_runner.py` + `subtraction_runner.py`: `experiments/mature/generated_probes.json` → `.../intermediates/…`; `experiments/mature/results/corrected_ground_truths.json` → `.../intermediates/…` | both runners read mature's probe/GT files, which moved to `mature/intermediates/` this session |

### C. `tests/` — retired dead/scratch tests → `tests/archive/`

10 broken/superseded test scripts moved to `tests/archive/` (git mv → recoverable;
`tests/archive/README.md` records each file's reason + live replacement). Deep-checked
by evidence, not date: verified imports, referenced files, and spec/roadmap citations.
The 22 spec-tied behavioral tests in `tests/` root + `tests/smoke/*` were left untouched.
Smoke's import-sweep walks `src/` only, so none of these were ever in a live suite.

| date | what | from → to | why |
|---|---|---|---|
| 2026-07-21 | retired tests → `tests/archive/` (git mv) | `test_post_flight.py`, `test_full_pipeline_phase_{6,7,9}.py`, `test_direct_codex.py`, `test_triplet.py`, `quick_probe_test.py`, `test_bg_json.py`, `test_bg_non_thinking.py`, `test_judge.py` | broken (`is_lossless` gone C1; phase_9 reads missing `data/simulation_input.jsonl` + gone `scripts/simulation/run_simulation.py` + Celery) or scratch eyeball tools (old `v2_final` checkpoint, hardcoded ids/ports, no assertions), all superseded by the behavioral suite — see `tests/archive/README.md` |
