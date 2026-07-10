# G23 + C17 — Export/backup/embedding-versioning + 384→1024 un-truncation

Assumes decided specs: `C7_scheduling.md` (no Celery; runner is a script/idle
job), `FINAL_experiments.md` (its pg_dump snapshot wrapper and this backup share
one module), `B1_classifier_retrain.md` (classifier consumes the 384-slice until
its retrain — the MRL staging), `T_temporal.md` (cold-resurrection re-embeds ride
the same embedder call). Grounded at commit `90710bd`: the seven live
`Vector(384)` columns — episodic_memory.embedding, episodic_chunks.embedding,
context_clusters.embedding (a MEAN centroid, not an encoding),
codex_entities.embedding, procedural_memory.embedding, rag_chunks.embedding
(NOT NULL), batch_summaries.embedding; `truncate_dim=384` set in classifier.py's
embedder; store_turn embeds `user_message` only; A4's re-normalization
workaround exists because truncated prefixes break unit norm.

## 1. Decisions

- **D1: store-level embedding identity, fail-loud.** New table
  `store_meta(key TEXT PK, value JSONB, updated_at)`; row `embedding =
  {model, dim, stamped_at}` plus per-table re-embed stamps. On startup, if
  settings' embedder/dim disagree with `store_meta` → vector legs REFUSE with an
  error naming the exact migration command. Silent wrong-dim cosine is the
  existential failure G23 exists to prevent.
- **D2: export = portable JSONL dump; import = staged restore.**
  `src/memory/portability.py` + `scripts/ice_export.py` / `ice_import.py`.
  Export: one JSONL per table (ALL memory stores: episodic incl. private,
  chunks, codex entities/edges/events/snapshots, procedural, slots, clusters +
  links, rag docs/chunks, batch + conversation summaries, cold_storage,
  review_queue, curated_labels, and projects/decisions/tasks once E1 exists) +
  `manifest.json` (alembic head, embedding meta, per-table counts, ICE version,
  date). **Vectors excluded by default** (derived + heavy; re-embed on import);
  `--with-vectors` for exact clones. Secrets never exported (no .env, no
  api-key settings). Import: manifest validation (alembic head must match or be
  upgradable) → id-preserving inserts (into an empty DB by default; `--merge`
  skips existing ids) → re-embed pass (D4) → `_regenerate_context_payload`
  sweep. This is state-copy; F10 replay remains the "relive it" alternative —
  both documented in the manifest header.
- **D3: backup = one wrapper, shared with FINAL.** `src/memory/backup.py`
  (pg_dump -Fc + `models/` + `model_registry.json` + config snapshot) with
  `scripts/ice_backup.sh` / restore instructions; FINAL's per-checkpoint
  snapshots call the same module.
- **D4: the re-embed runner is introspective and resumable.**
  `src/memory/reembed.py::run(db, embedder, tables="all", batch=256)`:
  discovers vector columns from `information_schema` (never a hardcoded list —
  a future vector column without a registered rule is a HARD ERROR naming this
  spec), maps each to its **source-text rule** (episodic → `user_message` part
  of raw_text, mirroring store_turn; chunks → chunk_text; entities → the
  extractor's embed text; procedural → pattern_description; rag_chunks →
  chunk_text; batch_summaries → summary_text; **cluster centroids are NOT
  re-encoded — recomputed as the mean of their members' new embeddings,
  after members finish**). Per-table progress stamps in `store_meta` →
  kill-safe resume; each writer's actual embed-text is verified against the
  writer code at implementation (grep list in §3).
- **D5 (C17): the migration.** One Alembic revision: drop vector indexes →
  `ALTER COLUMN ... TYPE vector(1024) USING NULL` per column (data is being
  re-encoded anyway; rag_chunks' NOT NULL is dropped and restored after) →
  recreate indexes at 1024 (HNSW fine ≤2000 dims — G6's indexes land through
  migrations, never the orphan SQL script). Then the D4 runner refills. Startup
  guard (D1) keeps the system honest between the two steps.
- **D6: MRL staging for the un-retrained consumers.** `truncate_dim=384` is
  removed; `encode()` returns native 1024 (unit-norm — delete A4's
  re-normalization for retrieval paths). Classifier + NER consume
  `slice384(vec)` (first 384 dims — mathematically identical to today's
  truncation) **with re-normalization kept for the sliced consumers only**,
  until B1 (classifier @1024) and A9-if-ungated (NER) retrain. One encode call
  serves both widths.
- **USER-REQUIRED (rule 11):** (a) run `ice_backup.sh` before the C17 migration
  (one command; done = archive file exists + logged size); (b) keep the machine
  on for the re-embed (estimate printed up-front: rows × ~20 ms CPU — the dev
  store is minutes, not hours; resumable regardless).
- **Empirical deferral:** none — retrieval-quality delta from 1024 is measured
  at Z1/FINAL as already planned (expectation stays "modest; chunking was the
  big win").

## 2. Files & integration points

`store_meta` + C17 migration (one revision) · `src/memory/{portability,backup,
reembed}.py` + `scripts/{ice_export.py,ice_import.py,ice_backup.sh}` ·
classifier.py / ner_utils.py (`slice384` + kept renorm) · orchestrator (delete
truncation-era renorm on full-width paths; embedding-dim literals audit:
`grep -rn "384" src/` — every hit becomes schema/meta-driven or a documented
slice) · startup guard in `create_core()` (C7's factory — one home) ·
`tests/test_longevity.py`.

## 3. Edge cases & failure modes

Kill mid-re-embed → resume from per-table stamp; legs already filter
`embedding IS NOT NULL`, so a half-filled table degrades recall, never crashes —
the startup guard message says re-embed is in progress. Import into a DB at a
different alembic head → refuse with the upgrade command. `--merge` id
collision → skip + count in report. Rows whose source text is empty (legacy
nulls) → embedding stays NULL, counted and reported. Embedder OOM/absent →
runner exits resumable with zero partial-row corruption (UPDATE per batch in a
transaction). Future embedder swap (not just widening) → same runner, D1 stamp
makes it mandatory; this is the "embedding versioning" promise generalized.

## 4. Validation checklist — `tests/test_longevity.py`

1) export→import round-trip on a seeded mini-store: per-table counts equal,
ids preserved, manifest validated, secrets absent from the archive (grep);
2) re-embed on the imported store: zero unexpectedly-NULL embeddings, all dims
1024, centroids equal the member mean (recomputed, not encoded); 3) retrieval
parity smoke: a fixture query's top-3 before (384, pre-migration snapshot) vs
after (1024) overlaps ≥2/3 and the known-correct row stays top-1; 4) startup
guard: doctored store_meta dim → vector legs refuse with the named command;
5) unregistered vector column (create a scratch one) → runner hard-errors;
6) resume: kill after table 2, rerun completes only tables 3+; 7) `slice384`
path: classifier outputs byte-identical pre/post migration on the same prompt
(MRL equivalence proof); 8) backup script produces a restorable pg_dump
(restore into scratch DB, count check).

## 5. Look-ahead constraints

B1/A9 retire `slice384` consumers (delete the slice, keep the helper for any
future MRL staging). F gets export/backup buttons (ledger) over these exact
scripts — no new logic. F10 import-UI cites D2's distinction (copy vs relive).
FINAL uses `backup.py` for snapshots (already assumed there). T3 resurrection
re-embeds at whatever `store_meta` says — no special-casing.

## 6. Traps

- Don't re-encode text the original writer didn't embed (e.g., full raw_text
  where store_turn embedded only the user half) — a silent relevance shift
  worse than the dim change itself; the source-text rule table is load-bearing.
- Don't re-encode centroids — they're means; encoding a cluster's *name* would
  quietly redefine every cluster.
- Don't keep `truncate_dim=384` "for safety" alongside slicing — two truncation
  mechanisms is how the next mismatch is born.
- Don't skip index drop/recreate — pgvector will happily seq-scan forever.
- Don't put vectors in the default export — portability means re-embeddable,
  not 500 MB JSON.
- Don't hardcode the seven tables anywhere — introspection + rules registry is
  the design (decisions/conversation_summaries arrive later and must be caught
  automatically).
