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
7. **⚑ DELETION SWEEP — the rule that stops zombie docs (standing rule,
   2026-08-01; moved here from CLAUDE.md 2026-08-09, because this is the file
   you open when you remove something).** Adding a section is the easy half.
   The half that keeps getting missed is **removing a thing**: a deleted
   component keeps living in the *overview prose*, the *diagrams*, and the
   *settings list* long after its own section says "DELETED". DI3 was deleted
   in D8 and its own §2.2 said so, while §1.1 still called the classifier a
   "two-stage pipeline (DI3 → 25-way MLP)", the component map still had a
   "DI3 + MLP" box, and §10 still documented seven `DI3_*` settings that no
   longer existed. So whenever a component is **deleted, replaced, renamed, or
   changes shape** (label counts, leg counts, dimensions, cadences),
   `grep -rin '<old name>' docs/ README.md CLAUDE.md` and fix **every** hit,
   classifying each as either a **live claim** (correct it) or **deliberate
   history** ("X was replaced by Y in D8" — keep it, it is the record).
   Check these five places specifically, because they are the ones that rot:
   **(1)** `ICE_Architecture.md` §1 System Overview prose · **(2)** its ASCII
   diagrams · **(3)** its §10 configuration/settings lists · **(4)**
   `README.md` · **(5)** **`CLAUDE.md`** — added 2026-08-09 after it was found
   describing DI3, Celery, Redis, a 384-dim encoder, a 25-logit head and the
   `ice_classifier_v3` checkpoint, all of which had been gone for weeks. It is
   the one doc loaded into *every* session, so it rots the most expensively
   and was the only one not on this list.
8. **Never delete data, configs, or anything user-authored.** When in doubt:
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

## B1 session — schema v2 classifier (2026-07-25)

No file moves or deletions. New modules, one frozen copy, and small in-place hygiene
on the files this session touched.

| date | what | from → to / change | why |
|---|---|---|---|
| 2026-07-25 | `data/labeled/label_schema.json` **v1 copy frozen** | copied → `data/labeled/label_schema_v1.json` | the v1 head layout (11/11/3, softmax ctx) must stay loadable: D5's non-regression gate has to RUN the old model to compare against it, and the live checkpoint is v1 until promotion. `schema.load_v1_schema()` reads it |
| 2026-07-25 | `label_schema.json` rewritten to schema v2 | flat label lists → `schema_version` + explicit `heads` with per-label `definition` strings | head widths/offsets become data (one loader, `src/classifier/schema.py`); the definitions also render the labeling rubric, so a label can't mean one thing to the labeler and another to the head |
| 2026-07-25 | **new** `src/classifier/{schema,templates,promotion}.py` | — | schema loader (stdlib-only, importable from scripts); the two encoder-input templates shared by training and inference (D3); one backup+atomic-swap shared by `workers/fine_tune.py` and `pipeline/promote.py` |
| 2026-07-25 | **new** `scripts/classifier/pipeline/*` | 8 stages + `common.py`, `rubric.py`, `serving.py`, `run_all.sh` | the B1 flow; `pipeline/README.md` carries the stage table and the traps |
| 2026-07-25 | `src/classifier/dataset.py` rewritten | v1 `{prompt, labels[]}` + `slice384` → v2 row shape, template-rendered, native 1024, cached embeddings | the v1 trainer is frozen under `legacy/` and is not run; caching exists because Z1-prep sweeps trunk width and re-encoding 25k rows per sweep point is wasted GPU time |
| 2026-07-25 | dead imports dropped (touched files only) | `ner_model.py`: unused `torch`; `di3_signals.py`: unused `typing.List` | boy-scout on files opened this session; `ruff check src/classifier/` clean |
| 2026-07-25 | `src/ingestion/formats.py::parse_jsonl` extended | + `{prompt, response, timestamp, conversation_id}` pair shape | three local exports (~5k real multi-turn rows) were silently parsing to zero conversations; a real F10 import would hit the same wall, so the shared adapter learns it rather than `extract.py` growing a private parser. `normalize_file` stays fail-loud; corpus building salvages malformed lines itself |
| 2026-07-25 | schema paths anchored to the repo root | `schema._resolve()`; `pipeline/common.py` chdirs to ROOT | stages run from their own directory, so `settings.label_schema_path` (repo-relative) resolved against the wrong place — and would have failed hours into a run when a late stage first read it |

| 2026-07-26 | **new** `scripts/oneoff/b1_authored/` | `batch01_needs_memory_cross.py`, `batch02_long_and_combinations.py`, `batch03_codebase_query.py`, `batch04_meta_and_gaps.py` | the Pile B authoring batches. Kept (not deleted) because they ARE the provenance of 289 hand-labeled training rows — the prompts and their labels only exist because these scripts wrote them, so the scripts are the record of what was authored and why |
| 2026-07-26 | **new** `scripts/classifier/pipeline/{authored,compare,build_eval_probes}.py` | — | Pile B loader+validator; two-pass agreement comparator (reuses the merge's own rule); the independent eval-probe builder |

## B1 run 2 — training, gating, and the label-quality diagnosis (2026-07-27)

No file moves or deletions. One real de-duplication, plus new eval assets.

| date | what | from → to / change | why |
|---|---|---|---|
| 2026-07-27 | **duplicated encoder consolidated** | `dataset.ICEClassifierDataset._encode` body → module-level `dataset.encode_rendered()`; `pipeline/evaluate.py::_encode` now calls it | `evaluate.py` had its own one-shot `model.encode(..., batch_size=256)` and **OOM'd on the 5,055-row test split**, while `dataset.py` already had the chunked, CPU-offloading, halving-backoff version. Two implementations of "encode a list of rendered rows", one of them wrong. Note the rule this preserves: batch size is the correct lever because it is semantically neutral — capping `max_seq_length` would fix the memory by making training truncate where inference does not, i.e. by reintroducing the exact train/inference mismatch B1 exists to remove |
| 2026-07-27 | **new** `scripts/classifier/pipeline/hard_probes.py` | — | 104 hand-authored adversarial probes. Deliberately a **script that emits JSONL** rather than a data file: `data/labeled/` is gitignored, so probes written as data would be untracked, and an eval set that is not version-controlled is not an eval set. Each probe carries the boundary it tests and what a failure would prove |
| 2026-07-27 | **new** `scripts/classifier/pipeline/{eval_probes,score_hard_probes,sweep_threshold}.py` | — | the two independent gates + the threshold fitter. `score_hard_probes.py` prints every miss with the model's probabilities, because a pass rate alone repeats the mistake the diagnosis exists to correct |
| 2026-07-27 | `compute_pos_weights` cap parameterised | default `20.0` → `3.0`, exposed as `train.py --pos-weight-cap` | the cap was a hardcoded literal doing calibration work; it is now swept and recorded in the checkpoint (see PROVENANCE) |
| 2026-07-27 | `evaluate.py` shared-subset width de-hardcoded | `[:11]` → `SHARED_INTENTS = len(load_v1_schema().labels(INTENT))` | the §3 grep-gate's last real hit; the v1 intent width is data, not a literal |
| 2026-07-27 | test width constants updated | `tests/test_classifier_v2.py` (28→27, intent 13→12), `tests/test_memory_decision.py` (`[0.0]*24` → `[0.0]*CTX0` read from the schema) | `Codebase_Query`'s drop moved the context head's offset 24→23. The memory-decision test now reads the offset from the schema instead of hardcoding it — a literal there silently mis-slices into the intent head rather than failing loudly |
| 2026-07-27 | `promotion.promote_checkpoint` anchors relative paths to the repo root | added `promotion._resolve()` (mirrors `schema._resolve`) | `promote.py` does not import `common` (which chdirs), so a repo-relative `settings.classifier_model_path` resolved against `scripts/classifier/pipeline/`. Promotion wrote the checkpoint to a fabricated `<cwd>/models/classifier/...`, found nothing to displace, and **skipped the backup while printing success**. Fixed in the shared module rather than the script because `workers/fine_tune.py` is the other caller and carries the same exposure |
| 2026-07-27 | live-checkpoint assertions made generation-agnostic | `tests/test_classifier_v2.py`, `tests/test_longevity.py` | both hardcoded `raw_probs == 25`, i.e. "the live path holds v1". True during B1's development, false after promotion, true again after a rollback — so both now assert the durable property (loads, serves at its own declared width, populates B2's seam). `test_longevity` keeps the slice384 bit-identity proof untouched: that is the real C17 claim and the micro-NER still depends on it |
| 2026-07-27 | stale pre-promotion claims corrected in place | `classifier.py` (class + `_tags_above` docstrings), `model.py` (module docstring incl. the `256→13` head width, `LegacyICEClassifierV1` docstring), `schema.py`, `ICE_Architecture.md` §2.1 | six places asserted "the live path serves a v1 checkpoint until promotion", which promotion made false. `LegacyICEClassifierV1`'s delete-me note was the actively dangerous one: its reason 2 expired, but reason 1 (D5's gate must be able to RUN the baseline) did not, and it is also what makes a rollback a file swap instead of a code change — so the note now says explicitly that the class is NOT yet deletable |
| 2026-07-27 | **new** `scripts/classifier/pipeline/tune_b2.py` | — | the B1→B2 recalibration sweep. Kept even though it recommended no change: the measurement is what settles the question, and it encodes two things a future hand-tune would get wrong — the recall-first objective (balanced accuracy trades away the catches a silent gate exists to make) and tie-handling that keeps the current value, so the three knobs that are inert on this data are not silently zeroed by argmax |
| 2026-07-27 | B1 spec closed out | `docs/specs/B1_classifier_retrain.md` rev `[2026-07-27e]` | the spec still described a plan that had happened, and carried two open deferrals (the 0.3 threshold, the pos-weight sweep) that were resolved during implementation. Rev note records both, plus the four measured refutations of the spec's own predictions and the label-ceiling finding that supersedes §5's F1 table as the quality instrument |
| 2026-07-27 | live checkpoint renamed to what it holds | `ice_classifier_v3_qwen_ft3.pt` (holding a v2 model) → **`ice_classifier_v4_schema2.pt`**; the displaced v1 file took back its own honest name `ice_classifier_v3_qwen_ft3.pt` instead of promotion's `_prev_<ts>` suffix | the old name asserted "v3, qwen, fine-tune 3" while holding a from-scratch schema-v2 retrain — invisible to code (which reads `schema_version` from the file) and misleading to a reader, who would reasonably conclude the classifier is a qwen fine-tune. Naming the v1 file honestly also makes the rollback self-documenting: `cp .../ice_classifier_v3_qwen_ft3.pt .../ice_classifier_v4_schema2.pt` |
| 2026-07-27 | `train.py --out` default moved off the live path | `ice_classifier_v4_schema2.pt` → `candidate.pt` | the rename made the training default and the live path the same file, so a no-argument `python train.py` would have overwritten the serving model with an untrained, ungated one. Promotion is the only writer of the live path |
| 2026-07-27 | sweep artifacts deleted | `models/classifier/sweep_cap{3,5,10,20}.pt` | 2.7 MB each, fully reproducible (`train.py --pos-weight-cap N`, seed 42, cached embeddings) and the numbers they produced are recorded in PROVENANCE. `models/` is gitignored, so these were local clutter only |
| 2026-08-17 | four G50/G51 measurement probes kept | → `scripts/oneoff/probe_merge_band_safety.py`, `probe_merge_difference_kinds.py`, `probe_deadend_audit.py`, `probe_containment_links.py` | they are the evidence base for both specs: cosine does not sort by merge safety, only 1.9% of the 2,247-pair backlog is deterministically mergeable, 75.1% of entities are dead ends, and 5,487 containment pairs are unconnected. Every number in G50/G51 is reproducible by re-running these against a snapshot |
| 2026-08-17 | two scratch probes kept, not deleted | → `scripts/oneoff/probe_procedural_pool.py`, `scripts/oneoff/probe_procedural_ranking.py` | they are the provenance of [TRAPS #37](TRAPS.md): the first feeds the retrieval path nonsense strings and shows the procedural leg returns one constant fragment; the second runs the leg's own SQL and shows the candidate pool is a single `is_active` row out of 61. The finding is only checkable by re-running them, so they are the evidence, not debris |
| 2026-08-01 | `docs/rough_post_paper_work.md` → `docs/outdated/` (git mv, 1,709 lines) | fully mined; no longer a work source | BRUTAL_ASSESSMENT.md §"Full sweep result" records that every item in it maps to a live roadmap entry, so it held no unclaimed work. Archived rather than deleted (outdated/ convention: never destroy). All four inbound references updated — CLAUDE.md and ROADMAP.md now state the roadmap **is** the queue rather than a distillation, BRUTAL_ASSESSMENT's two links repointed, and `docs/outdated/README.md` gained an entry explaining why it is inert |

## 2026-07-28 — D8 / A9a / E12 session

**Deleted (D8, commit `ba791db`).** Recoverable from git at `d981ca9`, the last
commit before the deletion; the measurement that justified it is frozen inside
`scripts/classifier/pipeline/eval_di3.py`, which carries a verbatim copy of the
signal functions and thresholds so the finding stays re-runnable.

| path | what it was |
|---|---|
| `src/classifier/di3.py` | the five-rule pre-classifier |
| `src/classifier/di3_signals.py` | the five density functions |
| `src/classifier/di3_config.py` | the seven `DI3_*` threshold reads |
| `src/classifier/di3_logger.py` | two structlog wrappers |

Also removed, all downstream of the same flag: `settings.ltm_bump_reference` and
the seven `DI3_*` settings (`src/api/config.py`); `reference_signal` on
`ClassificationResult`; the `reference_signal` arm of T2's joint gate plus its
kwarg and both call sites; `classify()`'s now-unreachable `conversation_history`
and `conversation_length` parameters (no caller ever passed either).

**Renamed, not deleted.** `schema._DI3_PRIORS` → `_LABEL_ONLY_PRIORS`. The table
was DI3-specific but the guard around it is not: `finalize_context_scalars` and
`orchestrator._head_confidences` both branch on all-zero `raw_probs`, and a
hand-built result reaching them without that branch would silently score
`p_ltm = 0` — "never retrieve" — which is the failure class this project refuses
to ship. Guard kept, DI3 vocabulary removed, both docstrings corrected.

**Consolidated (A9a, commit `c3a0f04`).** Six hand-rolled copies of the 384
narrowing → one `embedder.fit_width(vec, target_dim)`: `classifier._encode`,
`workers/fine_tune._encode`, and `pipeline/{evaluate,eval_probes,score_hard_probes,tune_b2}.py`.
Nothing deleted — A9a's "delete it, it's dead" premise was checked and is false
(it is the rollback path; see the roadmap entry's divergence note). `slice384`
and `test_longevity`'s bit-identity check untouched, as instructed.

**Added.** `scripts/classifier/pipeline/eval_di3.py` (D8's measurement + frozen
DI3) and `scripts/classifier/pipeline/audit_labels.py` (E12's consumer audit).
Both are stages, not one-offs — Z1's G28 sweep re-runs the second one.

**Stale things fixed in passing.** `ClassificationResult.raw_probs` said "28
under v2" (it is 27); `orchestrator._head_confidences` said the same; the smoke
suite's settings stub still carried `temporal_label_threshold=0.6` after B1
raised the live default to 0.85; `scripts/classifier/README.md` still listed
`di3*` as runtime code; `src/classifier/model.py` and `src/api/config.py`
described the slice384 call sites as dead. `tune_b2.py`'s `GRID` lost its
`ltm_bump_reference` row (the setting no longer exists) with a note recording
that the knob was inert in that sweep because nothing there ever set the flag.

**Not reformatted.** `ruff --select F` only, on touched files. The `I001`
import-order warnings in `scripts/classifier/pipeline/*` are **load-bearing and
must not be "fixed"**: importing `common` first is what runs the `sys.path.insert`
+ `chdir(ROOT)` that every subsequent `src.` import depends on. A note saying so
now sits in `eval_di3.py`.


## 2026-07-28b — G29's two bugs (scope leak + `decision_add`)

**Consolidated (commit `033b5b7`).** Three hand-rolled copies of the episodic
conversation filter → the shared `_conv_scope_filter`:
`orchestrator._cold_lookup`, `orchestrator._append_empty_window_note`'s
nearest-era probe, and `orchestrator._relevant_cluster_ids`. All three gained a
`scope` parameter; `configurable_orchestrator`'s `_relevant_cluster_ids`
override was widened to match (it shadows the base signature positionally, so
a new parameter there is not optional). Nothing deleted.

**Extracted.** `decision_extractor.run_decision_extraction`'s inline
dedupe/supersession block → `decision_extractor.reconcile_and_insert`, now the
single path that writes a `decisions` row. `services.projects.decision_add`
stopped calling `_insert` directly and calls it instead — the behavior its
docstring had been claiming since E8. Its `from src.api.config import settings`
moved with the block it serves.

**Behavior change worth knowing.** `decision_add` (and therefore the MCP
`decisions_add` action) no longer returns `{"status": "ok"}`; it returns E8's
own vocabulary — `recorded` / `duplicate` / `superseded` / `conflict_queued`.
No documented consumer depended on `"ok"` (grep: the MCP handler returns the
dict verbatim, no test asserted on it).

**Stale things fixed in passing.** `ICE_Architecture.md` §6.6 still described
`_relevant_cluster_ids` as scoring `sim + 0.3 × tag_overlap + 0.15 × name_sim`
and returning a flat top-10 — **C5 deleted the `name_sim` term** (it re-embedded
each cluster's name + description on the synchronous hot path, up to 30 forward
passes for a signal the centroid already carries) **and replaced the flat cut
with the adaptive 80 %-of-best band**. Section rewritten to the shipped code.

**Boy-scout.** Two unused imports dropped from
`src/retrieval/configurable_orchestrator.py` (`dataclasses.replace`,
`ClassificationResult`). `ruff --select F` clean on all five touched files. No
moves, no deletions, no reformatting.

---

## 2026-07-28c — C6: `custom_filter` dropped, the scope builder consolidated

**One column DELETED: `conversations.custom_filter`** (migration
`a1f6b8d94c22`, the new alembic head). Recovery: the column and its plumbing
are in every commit up to `c46e5df`; the migration's `downgrade()` restores the
column (empty — the data is not recoverable from the migration, but the live
store held no non-NULL values, and the *code* to read one never existed).

Why it went, recorded because "unused column" undersells it. `custom_filter`
was **v1's definition of `manual` scope**: the user would hand-write a SQL
`WHERE` fragment (`docs/outdated/ARCHITECTURE.md` §8.1 gives the example
`topic_tags @> ARRAY['Software_&_Tech'] AND timestamp > '2025-01-01'`) and the
orchestrator would append it to every episodic query, guarded by an allowlist
validator. The validator was never written and no reader was ever added, so the
value was set by `set_scope`, echoed by `get_scope`, plumbed through the REST
body and the MCP action, and read by nothing. C6 gave `manual` the *other*
meaning the user chose — tick the conversations you want — which the same mode
cannot also carry. `specs/G_mechanical.md`'s G20 sweep had already recorded the
DROP verdict; the user was asked anyway (standing rule: a measurement that
something is unused is evidence, not permission) and confirmed on 2026-07-28.

Removed with it: the `custom_filter` field on `ScopeUpdate`
(`api/routers/user_control.py`), the parameter on `scoping.set_scope`, the key
in `get_scope`'s response, and the passthrough in `chat_commands._cmd_scope`.

**Duplication collapsed: the retrieval-scope builder.** `api/main.py` and
`services/retrieval_svc.py` each built the scope dict from a conversation row.
The copies had already drifted — the service copy reproduced only the *project*
arm, so an MCP `ice_context` pull inside an incognito conversation missed the
`isolated`/`incognito` flags and ran the RAG and procedural legs against global
memory. Both now call `services/scoping.py::resolve_retrieval_scope`. Same
shape as the G29 clusters; found by looking for it rather than by a grep.

**Contract fixed, not just tidied.** Every id-set parameter on `set_scope` is
now `None` = leave unchanged, `[]` = clear. `cluster_ids` used to overwrite with
`[]` on `None`, which is the only reason `/scope` had to re-send the current
value on every call just to avoid destroying it (C10/C11 spec rev 13). That
passthrough is deleted; a bare `/scope` now changes the mode and nothing else.

**Not re-recorded on purpose.** `logs/router_parity_baseline.json` (untracked,
2026-07-17) now mismatches on 9 checks. Eight are stale — the baseline embeds
the pre-wipe live store's slot content that G23 destroyed on 2026-07-19 — and
the ninth (`scope_get.body`) is C6's intended change. Re-recording would erase
E0's byte-identical-extraction evidence, so the file is left alone and the
situation is written down here and in the C6 roadmap entry instead.

**Boy-scout.** `ruff --select F` clean on all eleven touched files. The `I001`
findings in `tests/test_session_scoping.py` are its house pattern (imports
inside the `try` block, next to the checks that use them) and were left. No
moves, no reformatting.

---

## C12a — documents (2026-07-28, commit `27f64eb`)

**Deleted: `src/workers/drop_zone.py`.** The v1 ingest path — a standalone
`watchdog.Observer` process with its own `main()` and `while True` loop.
Recoverable at `27f64eb^`. Three independent reasons, all of them measured
rather than assumed:

1. **Nothing started it.** `./ice` launches uvicorn; C7 deleted Celery and this
   module was never moved into `runtime.JOBS`, so `ingest_inbox/` had been inert
   for as long as C7 has been shipped.
2. **It was the last module excluded from the smoke import sweep** (G13: it
   instantiated a second `PyTorchClassifier` at import). `tests/smoke/
   test_imports.py`'s `EXCLUDED` set is now **empty** — every module under
   `src/` is swept, which is the first time that has been true.
3. Its output went to `rag_chunks`, which is gone.

Its replacement is `src/ingestion/documents/watch_folder.py`, an ordinary
`ingest_folder` cadence job (900 s) that calls the document service. **G13 is
closed by this deletion**, and G29's token-estimation cluster loses one of its
22 sites (`drop_zone.py:81`'s inline `len(split()) * 1.33`).

**Dropped: `rag_documents` + `rag_chunks` (migration `5fe5ad26480b`).** Their
only writer was the module above; their only reader was `_rag_lookup`, deleted
in the same commit. Recovery: the migration's `downgrade()` recreates both
tables exactly, and the live store was empty (see the roadmap's standing answer
on the empty DB), so nothing was lost. `rag_documents` is *succeeded* by
`documents` — a registry, not a content store.

**Deleted: `HybridRetrievalOrchestrator._rag_lookup` and every trace of the
`rag` leg** — the leg dict entry, the 1.0 base weight, the `ContextFragment`
source-type docstring, `ConfigurableOrchestrator._rag_lookup` and its ablation
flag, the wide net's `incognito` local (read only to gate this leg), and the
assembler-budget rule that preserved RAG fragments. Asked-and-recorded rather
than assumed: the leg was **user-confirmed for deletion** during the C12 design
session, on the evidence that it had no live writer, no scope filter, and a
five-English-noun gate.

**Two lint-adjacent fixes in passing.** `memory/reembed.py` lost the
`rag_chunks` rule and its `_RAG_NOT_NULL_RESTORE`; the `TableRule.post_sql`
seam that restore was the only user of is **kept**, with a comment saying so —
the next NOT NULL vector column will want it, and deleting a two-line
general mechanism to chase a zero-user count is the wrong trade.
`memory/portability.py` lost the matching NOT-NULL disarm step.

**Tests adapted, not deleted.** `tests/test_longevity.py` used
`RAGDocument`/`RAGChunk` as its export/import fixture and asserted the NOT NULL
re-arming. It now uses a **document + its conversation**, which tests a stronger
invariant: the portability walker builds its table list from
`Base.metadata.sorted_tables`, so a new table that round-trips proves it was
declared correctly. 26/26 still. `tests/test_turn_density.py`'s
`generate_summary` stub grew `**kw` for C12's `source_kind`/`source_title`.

**Boy-scout.** `ruff --select F,I001` clean on all touched files. The one
remaining `F401` in `src/model_registry/registry.py:5` is pre-existing and that
file was not touched this session, so it was left for G20/G29's pass. No
repo-wide reformat.

**Residue cleaned, and worth recording as a repeat of trap 6.** The first run of
`tests/test_documents.py` crashed in its own `finally` (it deleted
`codex_entities` before the `codex_edges` referencing them), leaving 1 turn, 4
conversations and 3 entities behind in a store that is supposed to be empty —
the exact failure the C6 session recorded. The suite's cleanup now deletes
edges and events before entities, and its final line prints **this run's**
remaining rows (must be 0) separately from a store-wide count, so a later
session is not misled into blaming this suite for another one's residue.

---

## `test_codex_2_0.py` archived (2026-07-28)

**Moved** `tests/test_codex_2_0.py` → `tests/archive/test_codex_2_0.py`
(`git mv`, recoverable). Three disqualifications at once, each one already
sufficient on its own by the 2026-07-21 sweep's own criteria:

1. **Broken** — crashed at `orchestrator._load_ner_model()`, deleted when NER
   moved to `src/retrieval/ner_utils.py`.
2. **Destructive** — `TRUNCATE episodic_memory, conversations, codex_entities,
   codex_edges, codex_events, codex_snapshots, idempotency_keys RESTART
   IDENTITY CASCADE` at the top of `main()`. That is the same criterion that
   retired `test_full_pipeline_phase_7.py`, and it ran **before** the crash —
   so running this file wiped the store and then failed.
3. **Half-dead** — extraction needs a live bg model (G30's standing gap), and
   the MERA section had degraded to an `ImportError` skip after A4 deleted it.

**It was rewritten house-style first, and the rewrite was reverted on the
user's instruction.** The reasoning is worth keeping: the ground it covered
(`handle_triplet`'s write rules; NER → match → graph) has no other coverage,
but **A9b** replaces the background NER behind the `extract_entities()` seam
and **A9c** may retrain the pre-flight micro-NER — so a test written today
would pin behavior both scheduled items exist to change. Flagged under G30 with
what the replacement owes, rather than written early.

**Two corrections this forced:**

- **`tests/archive/README.md` named this file the "live replacement" for three
  retired tests.** It could not carry that promotion — it was already broken
  when it was promoted. Those rows now say "none — flagged under G30". The
  sweep's method judged tests by whether their **imports** resolved, and this
  one fails at line 158 at *runtime*; the lesson (recorded in the archive
  README) is to re-check retired tests by **running** them.
- **The G23 entry and BRUTAL_ASSESSMENT's ⚡⚡ closure both said
  `test_retrieval.py` was "the one TRUNCATE-on-run test".** There were two.
  `test_retrieval.py`'s TRUNCATE is genuinely gone (only its docstring
  mentions the pre-G23 version); this file's was live until today. The backup
  half of that item stands; the "only one" claim is corrected in place.

**Not fixed here, recorded instead:** an unverified observation from the
aborted rewrite — after an A8 negation, an entity's `context_payload` still
advertised the retracted fact as a positive link, which would mean retrieval
keeps injecting it. The write side is correct and the renderer handles polarity,
so the suspicion is a one-step-behind regeneration. **Unconfirmed**; the
reproduction is written into G30's entry.

## C16 (2026-07-29) — consolidations, deletions, and one disarmed trap

**Consolidated into `src/memory/tokens.py` (G29's largest cluster, closed).**
Four incompatible token formulas across 21 sites became one real count:
`prompt_assembler._estimate_tokens`, `slots._estimate_tokens`,
`chunking.estimate_tokens`, 12 inline `int(len(x.split()) * 1.33)` copies in
`retrieval/orchestrator.py`, one in `services/retrieval_svc.py`, one in
`retrieval/evolution.py`, and the two `chars / 4.0` sites (`api/main.py`,
`workers/conversation_summary.py` — now `tokens.estimate_from_chars`, so the
4.0 lives in one place). The three inverse conversions (`SLOT_TOKEN_CAP / 1.33`
and friends) died with them; truncation is now a real token truncation.

**Deleted.**

- `api/main.py`'s post-assembly trimming loop (~22 lines). Hardcoded to 4096,
  measured `messages[0]` — the system message, which retrieval fragments never
  enter — so its condition could not change by popping fragments; built
  `reduced` as the COMPLEMENT of the survivors; and, because the condition was
  invariant, ran until both lists were empty and then restored every fragment.
  Zero behaviour, one `assemble_prompt` (and its DB query) per fragment on the
  latency path. Recoverable at `3d2ce7d^`.
- `api/prompt_assembler._trim_words` — its only caller went with the
  newest-first window rewrite. Recoverable at `c0326e5^`.

**Kept, not deleted.** `growth_cap` in `orchestrator.set_budget_from_turn_count`
stays even though residual coverage supersedes it, because removing it before
coverage is *measured* to bind would make ICE more expensive, not less (a short
conversation would jump from a 2,750-token ceiling to the full allowance). Z2
owns the deletion. Same reasoning kept `retrieval_leg_guarantee_enabled`
defaulting **on**: A10 designed that guarantee against measured leg
under-representation, so it is retired against a number and with the user.

**Trap 6, disarmed in `tests/test_documents.py`.** Its cleanup selected the
suite's rows via a hardcoded FILENAME allow-list, so every new fixture leaked
until someone noticed — and three document conversations had. It now snapshots
which non-chat conversations existed *before* the run and deletes the
difference. This trap had re-armed itself twice (C12a, then C12b); it cannot
now — **re-verified 2026-08-08**: a clean 53/53 run leaves 0 conversations and
`test_session_scoping` passes 40/40 immediately after.

> **⚠ A correction worth keeping, because the mistake is instructive.** On
> 2026-08-08 an orphan `kind='document'` conversation took
> `test_session_scoping` to 39/40, and this paragraph was briefly edited to say
> `test_documents` had leaked it. **It had not.** Bisecting one suite at a time
> against a cleaned store showed the leaker is **`tests/test_longevity.py`**,
> which creates `doc_conv` (line 241) for its portability round-trip and never
> deletes it — `doc_conv_id` appears only in the fixture and in one assertion.
> The lesson is the session's own TRAPS #13b in miniature: *the suite that
> looks guilty is the one that matches the symptom's vocabulary.* A document
> conversation appeared, so the document suite was blamed, on zero evidence.
> **Bisect before you name a cause** — it cost one command.

`tests/test_longevity.py` likewise gained a line that NAMES the mismatched
table when the export/import count check fails, instead of sending the reader
through a 28-table round trip for one row.

**Behaviour changes worth recording (no files moved).** Chunking now packs to a
real 550 tokens, so code chunks shrink and prose chunks grow slightly; nothing
needed re-chunking because the store was empty. The slot cap now admits 300
real tokens rather than ~550. `cold_storage` gained a vector column and a
`reembed.py` rule — G23's fail-loud guard refused to re-embed without it, which
is exactly how the omission announced itself.

## 2026-08-08 — G31 / G5 (cluster ①)

**Consolidated, not moved: five hand-rolled repo roots → one `src/paths.py`.**
`classifier/schema.py`, `classifier/promotion.py`, `memory/backup.py`,
`mcp/server.py` and `ingestion/documents/watch_folder.py` each derived the repo
root from `__file__` independently — four with `pathlib`, one with `os.path` —
and two carried a byte-equivalent `_resolve` helper. All five now import
`REPO_ROOT` / `resolve` from the shared module. **No behavior change from this
half**: these five already resolved correctly, which is exactly why the four
*broken* read sites (`.env`, the model registry, the micro-NER checkpoint, the
fine-tune paths) went unnoticed for so long. Recoverable at `c916812^`.

**Kept, not deleted.** `promotion.py::_resolve` survives as a one-line wrapper
rather than being replaced at its call sites, because its docstring is the
record of this bug class already having happened once — silently writing a
checkpoint to a fabricated `<cwd>/models/classifier/`, reporting
`backup → None`, and looking like success. The wrapper is cheap; the docstring
is not reproducible.

**Boy-scout on touched files only.** Dropped a now-unused `pathlib.Path` import
in `mcp/server.py` and in `classifier/promotion.py`; split `registry.py`'s
`import json, os, time, re` into four lines. ⚠ **`registry.py`'s
import-after-`logger` (E402) was deliberately left alone** — `bg_client_factory`
is imported below the logger for circular-import reasons, and reordering it is
not a formatting change. Same caution as `pipeline/*`'s load-bearing `I001`.

**Store residue removed (not a code change, recorded because it cost a debug
cycle).** Orphan `conversations` rows were deleted to restore the
documented-correct empty store: two `kind='chat'` shells from this session's own
live end-to-end G5 validation, plus `kind='document'` shells leaked by
**`tests/test_longevity.py`** (see the correction above — `test_documents` was
wrongly blamed first). All were verified empty (0 turns, 0 `documents`) before
removal. See TRAPS #6's third entry.

**`test_longevity`'s leak, fixed (`9aac4cd`).** Its `doc_conv` had no cleanup,
so every run left one orphan document conversation that broke
`test_session_scoping`'s exact-list assertion. The delete now runs after the
`documents` row (which FKs to it), and the suite gained the check that would
have caught it: a snapshot of conversation ids taken before the run, diffed
after cleanup, failing with the leaked ids named. Verified two-sided — removing
the delete makes the new check fail and print the id. 26 → **27 checks**, and
both orderings now pass with no manual cleanup between them
(longevity 27 → scoping 40/40; documents 53 → scoping 40/40).

## 2026-08-08b — cluster ② (G33, G11, G7) and the C10 gap sweep

**No files moved, renamed or deleted.** Recorded because three behaviour
changes and one new column need to stay findable.

**New column, new migration head.** `episodic_memory.batch_summary_id`
(migration `d5c81a37e9b2`, the new head) — the G11 coverage marker.
Hand-written, not autogenerated: `--autogenerate` proposes dropping every index
the ORM models do not declare, which is all eight HNSW vector indexes
(TRAPS #14). Verified 8 before and 8 after applying.

**Kept, not deleted — twice.**
`BatchSummary.start_turn_index` / `end_turn_index` are now provably useless as
coverage provenance (positions in a run's filtered, decay-ordered list, which
shifts between runs) and are read by nothing. They stay, annotated as
write-only, because a column drop is not G11's business and they are harmless;
the point of the annotation is that nobody starts trusting them.
The **property-value entity node** (G33) likewise stays rather than being
suppressed — removing it removes the edge, the `edge_expired` event, and any
T4 timeline for a property change.

**Test suites grew, none retired.** `test_codex_write_path` 20 → 23 (its G33
pin said "if this now fails, G33 was fixed — update this check", and it was),
`test_c10_c11` 53 → 58 (the relation-gap sweep, two-sided), and a new
`tests/test_batch_summary_coverage.py` (11). Every new fixture cleans up after
itself and verifies it did — the TRAPS #6 leak found earlier the same day is
why that is now stated rather than assumed.

**Ruff on the touched files:** unchanged counts against HEAD. The remaining
`E711`/`E712` in `codex_extractor.py` and `batch_summarizer.py` are
**load-bearing** — `== None` / `== False` inside a SQLAlchemy filter compiles to
`IS NULL` / `IS false`, while ruff's `is None` / `not x` would evaluate in
Python and silently break the query. Do not "fix" them.

---

## 2026-08-08 (later) — G9: constants → settings

**New module.** `src/retrieval/leg_weights.py` — the intent-dependent leg-weight
blend, lifted out of ~50 lines of literals rebuilt inside `retrieve()` on every
request. It owns the validation of the three weight settings; see its docstring
for why validation runs at read time rather than at config load.

**Deleted, with the reason each was safe.** All three were found by the sweep,
not looked for:
- `workers/decay.py::STRENGTHEN_AMOUNT` — declared and referenced **nowhere**,
  in this commit's tree and in the base commit's. The live strengthening site
  inlined the same `0.15` in `orchestrator._strengthen_retrieved`. One value
  now, wired to the site that uses it (`settings.decay_strengthen_amount`).
- `workers/codex_extractor.py::MAX_EXTRACTION_TOKENS` — its own comment said
  "retained for import compatibility"; nothing imported it, anywhere.
- `workers/fine_tune.py::MIN_ROWS_TO_PROMOTE` — the same promotion gate
  `settings.finetune_min_curated` already held, with a config comment asserting
  the two were "aligned" and nothing enforcing it. Collapsed into the setting.

Recovery for all three: `git show ae3abe8:<path>`.

**Kept deliberately, not swept.** `maintenance_agent.STALE_SLOT_NAME` (it names
the slot detector 5 watches — an identifier, not a knob), every controlled
vocabulary and prompt string, `embedder.SLICE_DIM` (a G23 bit-identity
contract), and the classifier's training hyperparameters in `classifier/model.py`
(B1 owns those, and they travel inside the checkpoint). The label SETS in the
orchestrator (`NARRATIVE_FACT_INTENTS`, `META_LEANING_INTENTS`) stayed for the
same reason: they name classifier labels, which is schema rather than tuning.

**Test suites grew, none retired.** New `tests/test_settings_freeze.py` (148),
`tests/smoke/test_leg_weights.py` (10) and `tests/smoke/test_ablation_flags.py`
(4). Six existing suites had reached for constants by name and now go through
settings: `test_batch_summary_coverage`, `test_maintenance_agent`,
`test_maintenance_runtime`, `test_ingestion`, `test_c8_c15`, `test_turn_density`.
⚠ **Two of those six (`test_c8_c15`, `test_turn_density`) were not in the suite
list run during the sweep** — they were found by grepping every deleted name
across the whole tree, which is now the closing check for a rename of this
shape. A suite that is not run cannot go red.

**Ruff on the touched files:** counts unchanged against the base commit
(orchestrator 9, clustering + maintenance_agent 5, everything else clean). One
new `E402` was introduced and fixed in the same session — the sweep put
`templates.py`'s settings import beside the constant it fed, mid-module. The
pre-existing `E711`/`E712` remain load-bearing SQLAlchemy comparisons; the note
above still applies.

---

## 2026-08-09 — G36: dead handlers, the HyDE rewriter, and three orphan row-sets

**Deleted from `src/retrieval/orchestrator.py`** (recovery commit: the parent of
`fb07e7c`).

- **Four unreachable `except` handlers.** Two in `_bm25_episodic` wrapped
  `tokens.append(w)` — appending a `str` to a `list`, which cannot raise — and
  the loop containing it. Two more were inside `_hyde_rewrite`, below.
- **`_hyde_rewrite`** and its commented-out call site, plus `_force_hyde`,
  `_hyde_used`, `_last_hyde_query`. **Not dead-by-neglect — dead since before
  `v2-paper-eval`**, and unreachable through the one path the docs claimed
  (`ConfigurableOrchestrator`'s `hyde` flag was never read). A comment stands at
  both deletion sites recording *why* real HyDE was rejected (roadmap P0.1:
  it would fabricate specifics about private history with a small model, and
  hallucinated specifics poison BM25) so the design is not rebuilt from git
  history. See PROVENANCE for what it does to Experiment 1's ablation table.
- **`self.bg_client` and the `get_bg_client` import.** `_hyde_rewrite` was its
  only consumer, so it had been constructing an OpenAI client per retrieving
  request for nothing. **Retrieval now calls no model at all**, which is
  checked at the seam in `test_retrieval_failopen` and `test_c10_c11`.
- **`hyde` from `configurable_orchestrator.py`** — the docstring entry and the
  special-case default in `_on()`. Nothing else referenced it.

**Deleted from `src/api/main.py`:** the `hyde_used` variable, its `getattr` read
off the orchestrator, and its two emissions — the `context_injection_complete`
log field and the SSE `retrieval` event field. Both were permanently `False`.
This closes item (2) of **G20**'s sweep verdict list.

**Adapted, not deleted:** `test_c10_c11`'s `BoobyTrappedClient`. It patched
`orchestrator.get_bg_client` to trip if `/search` ever asked for a completion;
with the import gone there was nothing to patch. The contract is now asserted
structurally (retrieval imports no client, holds no `bg_client`), which covers
every path rather than the one the test walks.

**Store rows removed** (backed up first to
`backups/orphan_rows_20260809_g36.json`, gitignored — full contents recoverable
from there):

| table | rows | what it was |
|---|--:|---|
| `memory_slots` | 1 | `test_services` marker (`svce0e62d24 …`) left 2026-07-28. `is_active=True`, `scope_tier='global'` — **injected into every system prompt for twelve days.** |
| `review_queue` | 4 | `decision_supersession` items from 2026-07-28 referencing six decision ids, none of which exist (`decisions` is empty). |
| `procedural_memory` | 1 | Newborn pattern from a live turn on 2026-08-08 whose source turn was later deleted outside the C10 cascade. Inert (`is_active=False`) but promotable by the next matching extraction. |

⚠ **One row was deleted in error and restored** with its original `created_at`:
the `conversations` shell `ab934e9c-…`, which is
`services/bookmarks.py::NOTES_CONVERSATION_ID` — production state, created
lazily by `ice_remember`, not test residue. Nothing was lost (empty shell,
deterministic id, get-or-create). The habit that would have prevented it is
TRAPS #15: grep `src/` for whatever generates an id before deleting a row you
did not create.

**Ruff on the touched files:** at parity with the base commit (orchestrator 9,
`main.py` and `configurable_orchestrator.py` clean). The pre-existing `E711` in
the orchestrator remains a load-bearing SQLAlchemy `== None`.

---

## 2026-08-10 — the documentation restructure

No code touched. Three files' worth of material moved, nothing deleted.

| From | To | Why |
|---|---|---|
| `CLAUDE.md` (17 rules/sections) | `docs/ROADMAP.md`, `docs/TRAPS.md`, `docs/CLEANUP.md` | It is loaded into every session; duplication is paid on every request. Six roadmap-working rules were near-verbatim copies of ROADMAP's own block, the provenance rationale a copy of PROVENANCE's header, the architecture section a restatement of ICE_Architecture. |
| `docs/ROADMAP.md` preamble (~7,100 words) | `docs/outdated/roadmap_session_log.md` | 17 dated "DOCS ARE IN SYNC"/"ADDENDUM" paragraphs + 4 stacked inventories. They lived in the roadmap because there was nowhere else to put a session note; `docs/HANDOFF.md` now owns that. |
| `docs/ROADMAP.md` — 70 finished entries (~26,000 words) | `docs/ROADMAP_DONE.md` | The queue was 59% finished items. Entries moved **verbatim, never compressed** — the completion notes are where look-ahead, propagation and validation are recorded, and the "seven consecutive entries were wrong about their own subject" pattern only exists because each one says which way it was wrong. |

**Net:** `ROADMAP.md` 65,475 → 34,607 words, and it is now the queue.

**Found by the move, and worth more than the tidying:**
- **Track G was in three places.** G1–G26 sat in Track G with G31/G32/G33
  wedged between G4 and G5; G28, G29, G30, G34, G35, G36 and G37 sat under
  **Track H — "Research follow-ups & open questions"** — behind a bare divider,
  an ops-and-bugs track filed as research. Every track is now in strict numeric
  order and Track G holds G1–G37 with no gaps.
- **Three items were counted but never written.** `G27` (announced 2026-07-25,
  description decision-complete in `specs/G_mechanical.md` the whole time),
  `G34` and `G35` (opened 2026-08-08 as two lines in a preamble note, with
  G34's measurements already in PROVENANCE). A session told to do any of them
  would have found nothing. All three now have entries; the open count is 51.
- **Z2 had been handed five pieces of work and its entry named none of them** —
  three from C16, one from A9b, one from the C13/C14 question — recorded in a
  preamble triage block the entry never pointed at. Carried into the entry.
- **The Z1/Z2 definition block was nested inside Z2's entry**, unreachable from
  Z1; lifted to the SEMIFINAL section head. **Z1's own entry described the stack
  as "postgres+redis, celery worker+beat"** — C7 deleted both — and is corrected.
- ⇒ **The rule this earned**, now in the roadmap's own rules block: *if it is
  work, it goes in an entry, never in preamble prose.*
- **`G34` and `G35` had no entries.** Both were counted in the PRE-FINAL list
  and G34 had a full measurement section in PROVENANCE, but neither was ever
  written as an item — they existed only as two summary lines in a preamble
  note from 2026-08-08. A session told to do G34 would have found nothing.
  Entries written from the recorded material; the inventory now agrees with
  what is actually written down (50, counted mechanically).
- **36 of 49 cross-references were dangling.** `(#a12)`, `(#b3)`, `(#g30)` and
  33 others pointed at anchors that did not exist — only 13 `<a id=>` tags had
  ever been added. Every item now carries one; all 49 resolve.
- **A rule was lost by the CLAUDE.md shrink and restored** (`d2710ec`): the
  "check the ground before building on it" clause, dropped because two bullets
  were compared by heading rather than by text. It is now a ROADMAP rule, and
  CLAUDE.md edits are user-gated because of it.

---

## 2026-08-10b — public-repo verification

Not a cleanup, recorded here because it changes what every later cleanup must
check. `github.com/Deepnar/ice` is **public, and has been for a while** — the
TRAPS #12 exposure incident closed 2026-08-04 was already a public clone.
CLAUDE.md's git section had gone on saying "going public" long after it had;
corrected 2026-08-10, and the state verified rather than assumed.

**Verified 2026-08-10** (re-run these after anything that touches history):

| check | result |
|---|---|
| `scripts/git/check_history_clean.sh --clone` | clean — against the live remote, i.e. what the public actually receives |
| personal / planning / career files tracked | none |
| credential-shaped strings in the tree | none |
| README's paper link | `experiments/paper/ICE_paper_v2.pdf` — the canonical venue-agnostic one, not a twin |
| `v2-paper-eval` | `0521df9`, post-rewrite |

⚠ **`docs/BRUTAL_ASSESSMENT.md` is tracked and therefore public.** That is
intended — it is a technical self-critique referenced from the roadmap, and it
was scanned for personal content at the flip and is clean. Do not confuse it
with the gitignored planning files.

⚠ **A rewrite is not cheap.** On a public remote the old objects stay
fetchable by SHA until GitHub garbage-collects them, which last time needed a
support ticket (TRAPS #12). Fix forward.

## 2026-08-11 — 1,596 orphaned job markers, FOUND AND NOT REMOVED

Counted while checking the blast radius of G29's idempotency-key change:

```
idempotency_keys   1596
episodic_memory       0
conversations         1     (the ice://mcp-notes shell — production state)
```

Every one of those markers says "job X already processed batch Y" about a turn
that no longer exists — residue from the 2026-08-09 sweeps (TRAPS #15/#16), which
removed turns without their markers.

**Not removed, deliberately.** The rule here is inventory-then-remove with a
backup, and the *reason* to remove is weaker than it looks: `batch_id` is a UUID,
so a re-created turn draws a new one and cannot collide with a dead marker. The
cost is 1,596 dead rows and a misleading count for anyone sizing the store.

⚠ **The reason to record it now is that it makes a store-emptiness check lie.**
"Is the store clean?" answered by row counts across all tables says no, while the
tables that matter say yes. A [Z1](ROADMAP.md#z1) run that asserts a clean start
should name the tables it means rather than counting everything.

If it is removed later: it is a plain `DELETE FROM idempotency_keys`, safe only
while `episodic_memory` is empty — after that, deleting a live marker re-runs the
job it belonged to. Check that first, back up the table, and log it here.

## 2026-08-11 — store restored after a diagnostic script seeded it

`tests/test_codex_extractor.py` is a **diagnostic printer, not an assertion
suite** — it seeds a graph and prints what the extractor did, and it has no
cleanup. Run three times while validating G29's JSON-salvage consolidation, it
left 22 `codex_entities`, 27 `codex_edges`, 77 `codex_events`, 7
`episodic_memory` and 15 `conversations`.

Removed, back to the documented-clean state: **1 conversation, everything else
0.** The kept row is `ab934e9c-3e3c-5ec2-a0f2-e98e8ee22e4a` — the deterministic
**uuid5** `ice://mcp-notes` shell created by `services/bookmarks.py`, which is
production state, not residue (TRAPS #15: grep `src/` for whatever generates an
id before deleting a row you did not create — that check is exactly why this row
survived a previous sweep's first pass).

Backed up before deletion (`pg_dump --data-only` of the five tables, 215 lines)
to the session scratchpad, per the inventory-then-remove rule.

⚠ **The suite is the thing to fix, not the rows.** Every other live-DB suite
here inserts and deletes its own fixtures; this one does not, so it will do this
again. It is also the reason a *different* suite failed earlier in the same
session — `test_retrieval_failopen` hit a unique-constraint violation on residue
from a crashed run (TRAPS #6, third time).

## 2026-08-13 — Z1 instrument and pipeline session

**Removed from the store (data, not code).** `tests/test_codex_extractor.py` was
run three times against the live measured store and leaked **3 episodic turns,
17 entities, 23 edges, 23 relation-gap rows** into the 293-turn Z1 corpus — the
failure TRAPS #6/#15/#16 describes, committed by the session doing the
measuring. Every row was backed up to JSON before deletion
(`leaked_rows_backup.json` in the session scratchpad) and the store was verified
back to its recorded baseline exactly: **293 / 3,671 / 4,170 / 247**. The
entities were unmistakably fixtures (`redis`, `celery`, `evaluate_turn`,
`pgvector extension`, `codex_edges`), none from the real conversations.

**Deleted from a generated migration.** `alembic revision --autogenerate` for the
bi-temporal columns proposed dropping **every HNSW vector index in the database**
(episodic, codex entities, chunks, clusters, decisions, batch summaries,
procedural) plus a dozen btree indexes including G6's and
`uq_memory_slots_name_tier_anchor`, and flipping six columns to nullable. All of
it was discarded; `505f12031434` adds two columns and one index. Those objects
are created by raw SQL in earlier migrations and are invisible to autogenerate —
see TRAPS #14, and the migration's own docstring records this so a future
regeneration does not re-propose it.

**Unused imports dropped** from files touched this session: `uuid` in
`scripts/z1/score_retrieval.py`, `turn_text` in `scripts/z1/seed_store.py`,
`pathlib.Path` in `scripts/z1/generate_probes.py`, `words` in
`generate_typed_probes.py`. Import order fixed in
`src/workers/procedural_extractor.py`.

**Deliberately NOT "fixed":** the `== None` / `== False` comparisons ruff flags
in `src/workers/codex_extractor.py` (E711/E712). They are **required** SQLAlchemy
filter idioms — `is None` does not generate SQL — and "correcting" them would
break the queries silently. Pre-existing E402s there are left alone too: the
late imports look deliberate (circular-import avoidance) and this session had no
evidence either way.

**New files, all under `scripts/z1/`:** `generate_typed_probes.py`,
`score_typed.py`, `harvest_probe_context.py`, `run_meta.py`,
`seed_relation_vocab.py`. `run_meta.py` is the one to reuse — every experiment
artifact from here carries its provenance block.

## 2026-08-23 — Z1 instrument scripts added (no moves, nothing deleted)

New under `scripts/z1/`, all tracked:

| file | why it exists |
|---|---|
| `production_parity.py` | the ONE reproduction of `main.py`'s pre-retrieval path; four harnesses had drifted copies |
| `check_reproducible.sh` | seeds the same turns twice and compares — a detector, deliberately containing no fix |
| `dump_graph_fingerprint.py` | compares two stores triplet-by-triplet and localises the FIRST divergent turn |
| `compare_judgements.py` | two judgement runs compared per TRIPLET rather than per rate |
| `reseed_postfix.sh` | one arm in arm-B config, varying only the code |
| `prompt_ab_noisefloor.sh` | four arms: each prompt twice, so effect and spread come from one run |
| `run_leg_ablations.sh` | the five per-leg ablations, carrying why its first attempt was stopped |
| `README.md` | the Z1 index — settled questions, open queue, the measurement floor, and §3b's falsification table |

Nothing was moved, renamed or deleted this session. Three score-run artifacts
were stamped `INVALIDATED` **inside the files** rather than removed, so a later
session cannot quote them without seeing why.

## 2026-08-23 (evening) — G59 comparison + the `wrong`-verdict review sheet

| file | why it exists |
|---|---|
| `scripts/oneoff/g59_compare.py` | pools the four `dir-*` judgement artifacts and computes the treatment-minus-control effect using `report_power`'s OWN interval formula, so the comparison uses the same method the arms were reported with. Written for the G59 null; kept because the next arm comparison needs exactly this. |
| `scripts/oneoff/dump_wrong_verdicts.py` | joins a judgement artifact's `wrong` verdicts back to their source turns. Needed because the artifact stores `edge_id` only — see the gap noted below. ⚑ the arm's store must be restored first; edge ids are per-store. |
| `scripts/oneoff/build_wrong_review.py` | renders both control arms' `wrong` verdicts as a human-readable review sheet with evidence windows, the judge's verbatim rubric, and a subject/object presence split. Output is `experiments/curation_files/wrong_review/` — **gitignored, carries raw corpus text.** |

⚠ **A broken presence test was caught here and is worth remembering.**
`build_wrong_review.py` first classified triplets with a plain substring match
plus a "last word of a multi-word term" fallback. That let `ending` match inside
*sending* and put **92%** of triplets in the both-terms-present bucket — a
presence test that says yes to everything, [TRAPS #37](TRAPS.md)'s exact shape.
Rewritten with word boundaries and no fallback: **77%**. The conclusion survived
all three strictness levels, and the sensitivity table is printed **in the
output file** rather than dropped once it came out clean.

⚠ **It hardcodes the per-arm turn counts** (124 each, read from the run logs).
That is not laziness: `judge_codex.py` writes `edge_id` into each verdict but
**not the source turn**, so a judgement artifact cannot be re-clustered by turn
after the fact. Fixing that is a one-line change to the artifact writer and is
noted in the Z1 README §4 rather than done mid-measurement.

Nothing was moved, renamed or deleted.

## 2026-08-25 — G63 extractor-decision scripts + the two new specs

All kept under `scripts/oneoff/`, none deleted.

| file | why it exists |
|---|---|
| `extractor_head_to_head.py` | ICE's production `extract_triplets()` vs NuExtract3 on the SAME random turns and the SAME NER list. ⚑ its first version was rigged — only ICE got the confirmed-entity block, then both were graded against it |
| `g63_guard_sweep.py` | the 8-condition guard sweep at 60 turns; checkpoints per condition so a crash late does not cost the early ones |
| `build_threeway_sheet.py` | N-arm blind labelling sheets from the sweep artifact (`--arms`, `--out`); arm hidden, paired by turn, key held separately |
| `build_extractor_blind_sheet.py` | the earlier 2-arm version, superseded by the above but kept |
| `calibrate_judge.py` | ⚑ **the judge gate** — scores any candidate against the maintainer's 15 blind labels. Caught `mimo-v2.5` at 27% (it missed 6 of 6 reversals) which would otherwise have been adopted on speed and price |
| `probe_direction_binary.py` | forced-choice direction probe; randomises which side the stored direction is on so a position bias cannot pass as understanding |
| `build_wrong_review.py`, `dump_wrong_verdicts.py` | the `wrong`-bucket review sheet and the judgement→source-turn join |
| `g59_compare.py` | pooled arm comparison using `report_power`'s own interval formula |

New specs (tracked): **`docs/specs/G63_extractor_decision.md`** (the decision
protocol and its checklist) and **`docs/specs/RESEED_PLAN.md`** (the clean break).

**Production files touched:** `src/api/config.py` (5 settings added or changed)
and `src/workers/codex_extractor.py` (template path, `</think>` strip, envelope
parse, adaptive chunk budget, and a **latent null-value parse bug** that had been
losing whole turns' extraction whenever any model emitted a null).

Nothing was moved, renamed or deleted.

## 2026-08-27 — background-model session

**Six new scripts in `scripts/z1/`** (none moved, renamed or deleted):

| script | purpose |
|---|---|
| `bg_model_bakeoff.py` | 11 candidate models × 9 background jobs, calling the PRODUCTION functions |
| `bakeoff_report.py` | merges every bake-off run; applies disqualifiers instead of a composite score |
| `judge_summaries.py` | summary faithfulness, with the planted-defect + verbatim-copy judge gate |
| `judge_bg_quality.py` | the quality half for batch summary / fold / cluster naming / procedural |
| `prompt_ab_fabrication.py` | prompt and must-term A/Bs; rewrites one substring in flight and aborts if it fails to differ |
| `probe_census.py`, `unify_probes.py`, `gt_feasibility.py` | probe inventory across 3 sources, the unified 618-probe set, and the gold-turn feasibility check |

**One doc written into the gitignored corpus dir:**
`experiments/curation_files/README.md` — what an `EC-*.json` checkpoint is, the
filename↔conversation-id mismatch, and where the full originals live
(`data/simulation/simulation_full.jsonl`).

**Store mutated, recoverably.** `dir-false-run1` no longer matches its `.counts`
file: `procedural_memory` 43→0, `batch_summaries` 2→0, `batch_summary_id`
cleared, procedural idempotency keys deleted — the bake-off's store-backed jobs
reset themselves so each model started from identical state. Episodic, codex and
chunks untouched. Rows backed up to `bakeoff/PRE_RESET_BACKUP.json`; full restore
is `snapshots/dir-false-run1.sql`. **Not restored on purpose** — G72 wipes this
store for the reseed.

## 2026-08-29 — paper-file NAMING RULE (nothing moved, nothing deleted)

The ACM TIST reject prompted the question "should the paper tex fork to v3?".
**Answer: no**, and the reasoning is recorded here because the trap it avoids is
permanent.

**⚑ PAPER DRAFT NUMBERS AND SYSTEM VERSION NUMBERS ARE DIFFERENT SCALES.**
CLAUDE.md's v1/v2/v3 are *system* versions. `ICE_paper_v2.*` means **paper draft
2** — the post-TMLR-desk-reject reframe. The two collide on one token and must
never be read across. The paper happens to report system v2 (tag
`v2-paper-eval`) because that is the frozen evaluated system; that is a
coincidence of history, not what the filename asserts.

**⛔ There is no `ICE_paper_v3` and there must never be one.** A future session
would read "paper v3" as "the paper about system v3" — false. The paper reports
system-v2 numbers and will keep reporting them as v3 work lands on `main`. The
two scales are most confusable exactly where they diverge most.

**⛔ `ICE_paper_v2.*` MUST NOT BE RENAMED.** `ICE_paper_v2.pdf` is a *published
URL*: `README.md:153` links it, and the same GitHub blob URL was sent to an
external researcher during arXiv endorsement correspondence. Renaming breaks a
live link someone outside this repo may still click. This outweighs the tidiness
win of a version-free name.

| file | status from 2026-08-29 |
|---|---|
| `ICE_paper_v2.tex` | **CANONICAL.** All substantive edits land here. |
| `ICE_paper_tist.tex` | **FROZEN RECORD** — exactly what TIST received. Header stamped in-file. Not edited again. |
| `ICE_paper_tmlr.tex`, `ICE_paper_v2_tmlr.tex` | frozen TMLR submissions |
| `ICE_paper.tex` | superseded original |

PUBLISHING.md §3's "substantive content changes in BOTH v2 and tist" rule applied
only while the TIST submission was live. It is now retired: tist is frozen.

**Naming rule for run artifacts that target the frozen system:** name them after
the **git tag**, not a bare version digit — e.g. `experiments/lme/v2-paper-eval/`.
`v2-paper-eval` is a unique string in this repo and cannot be misread as "paper
draft 2".

Nothing was moved, renamed or deleted. Two `.tex` headers were stamped in place.

## 2026-09-04 — rejected LME vLLM background diagnostic

The temporary vLLM launcher and its chat template were moved out of the active
LongMemEval harness after semantic parity failed:

| from | to | reason |
|---|---|---|
| `experiments/lme/run_bg_vllm.sh` | `scripts/oneoff/lme_vllm_background_rejected.sh` | the AWQ/vLLM model reversed the fixed extraction control; retained only to reproduce the rejected path |
| `experiments/lme/qwen3_bg_ollama_chat_template.jinja` | `scripts/oneoff/qwen3_bg_ollama_chat_template.jinja` | template used by that diagnostic; an Ollama-like form leaked reasoning into extraction |

The final matched cloud wrapper uses the exact Ollama
`qwen3:4b-instruct-bg`, kept resident. No production file was changed.


## 2026-09-12 — frozen-v2 paper release boundary

LongMemEval `runs/` is now ignored in full. The three previously tracked run
files are removed from the index only and remain on disk; no raw answers,
judgements or manifests are added by this release, and history is not rewritten.
Two aggregate mature-run sensitivity JSON reports are explicitly allowed through
the results ignore rule. Local agent instruction files remain local via
`.git/info/exclude`; their contents and all v3 work are unchanged. Paper scratch
work remains under the existing ignored `experiments/curation_files/` directory.
No source or canonical manuscript path was moved or renamed.

## 2026-09-12 — v3 extraction repair

Removed the order-sensitive triplet salvage regex and unsupported emotion/self-reference filters from `codex_extractor.py`; complete JSON parsing now lives in `workers/extraction_result.py`. This avoids silently discarding negation, valid emotion values and reflexive claims. Updated stale lossless-gate and bookmark-priority comments; sorted imports in touched files.

## 2026-09-12 — v3 shared turn readers

Moved duplicate turn-representation decisions into `memory/representation.py`.
Removed implicit 300-character fallback cuts and the literal reader threshold;
updated legacy NULL-coverage/abstract test expectations to the new contract.
Imports sorted in touched production files. No source files moved or deleted.
