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
> **[cross-ref from the C10/C11 session, 2026-07-19 — for the implementing
> session's re-grounding]** Since this spec's grounding (`90710bd`), the vector
> column set grew to NINE (+`conversation_summaries.embedding` (C4) →
> summary_text; +`decisions.embedding` (E-core) → the decision text — D4's
> introspection will catch them, the source-rule registry needs their entries),
> and D2's export list must be re-derived from models.py (now also:
> `conversations` itself — episodic imports FK it — plus projects/project_state/
> decisions/tasks, three-tier slot anchors, `store_meta` once born). **Husk
> trap:** D1/D2 merge husks (`properties.merged_into`) and C10 deletion husks
> (`properties.deleted_reason`) carry `embedding=NULL` *deliberately* — they are
> expired-but-kept audit rows made unmatchable. The re-embed runner's
> codex_entities source rule MUST skip them (re-encoding a husk's renamed
> canonical text would silently resurrect it into vector matching).

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
  serves both widths. *(A9 revision 2026-07-11: A9's decided successor is a
  GLiNER-class model on the background NER tier, which doesn't consume the
  shared embedding at all — so if A9 lands before C17, only the pre-flight
  micro-NER remains a slice384 consumer; if after, nothing changes here. See
  the A9 roadmap entry for the two-tier split.)*
- **USER-REQUIRED (rule 11):** (a) run `ice_backup.sh` before the C17 migration
  (one command; done = archive file exists + logged size); (b) keep the machine
  on for the re-embed (estimate printed up-front: rows × ~20 ms CPU — the dev
  store is minutes, not hours; resumable regardless).
- **Empirical deferral:** none — retrieval-quality delta from 1024 is measured
  at Z1/FINAL as already planned (expectation stays "modest; chunking was the
  big win").

> **[rev 2026-07-19 — implementation-session re-grounding note (rule 12), recorded BEFORE coding.]**
> Grounded against HEAD `78537c9` (post C10/C11 `e3f24cb`; alembic head `a7c5e91d3f28`) + the live DB. Refinements:
> 1. **Nine vector columns confirmed** (the C10/C11 cross-ref was right): episodic_memory, episodic_chunks, context_clusters (centroid), codex_entities, procedural_memory, decisions, rag_chunks (NOT NULL), batch_summaries, conversation_summaries — all `Vector(384)`.
> 2. **Source-text rules verified per writer:** episodic → the `user_message` half of raw_text (store_turn format `User: …\n\nAssistant: …`; fall back to full raw_text when the format is absent — that fallback is exactly right for MCP notes, whose raw_text IS the embedded text). Resurrection (`_resurrect_cold_hits`) wrote `(summary_text or raw_text)[:2000]` — the runner still applies the episodic rule (D4's decision stands; resurrection rows carry standard-format raw_text). chunks → chunk_text; codex entities → `canonical_name` ONLY (not name+description); procedural → pattern_description (same in reflection's crystallization); rag_chunks → chunk_text; batch/conversation summaries → summary_text; decisions → the `decision` text; centroids → recompute via `clustering._recompute_centroid_from_members` (mean **then `_normalize`**) after members.
> 3. **Codex skip rule is husks + non-conversation sources:** merge husks (`properties.merged_into`), C10 deletion husks (`properties.deleted_reason`), AND `source != 'conversation'` — code-graph/static rows are written `embedding=None` by design (code_graph rev 9, project_facts; live DB: 1,289 static + 6 derived rows, zero embeddings).
> 4. **truncate_dim lived in SIX instantiation sites**, not one: classifier.py, codex_extractor, procedural_extractor, batch_summarizer, clustering (each its own SentenceTransformer copy — a pre-existing G13 violation), fine_tune (+dataset.py, training-side). Fix = new `src/memory/embedder.py::get_embedder()` process singleton (model/dim from new settings `embedding_model_name`/`embedding_dim`), all sites route through it; training-side feature gen uses `slice384` of native so features keep matching the un-retrained classifier.
> 5. **`slice384` = raw prefix, NO renorm — empirically bit-identical** to `truncate_dim=384` output (ST 5.5.1, maxdiff 0.0). Native 1024 output is ~unit-norm. The classifier/NER MLPs never re-normalized, so D6's "re-normalization kept for sliced consumers" resolves to: no renorm anywhere new; DELETE `_unit` at both orchestrator sites (gloss cache, prompt), and fix `_match_entities_by_similarity`'s "already normalised" comment (false at 384, true at 1024).
> 6. **No vector indexes exist in the live DB** — G6's `scripts/database/create_indexes.sql` was never applied. D5's "drop → recreate" becomes: defensive DROP IF EXISTS, then CREATE the canonical HNSW cosine set on all nine columns (`idx_<table>_embedding` naming), for the first time, at 1024.
> 7. **`scripts/ice_reembed.py` added** as the runner's user-facing entry (estimate print, `--tables`, resume); §2's script list named only export/import/backup.
> 8. **store_meta ships in the same single revision as C17** and is seeded at the post-migration reality: embedding stamp `{model, dim: 1024}` + per-table `reembed:<table>` = pending. Guard (one home: `create_core()`): settings↔stamp mismatch ⇒ raise naming the exact commands; stamps pending ⇒ boot + loud "re-embed in progress" warning; **no stamp row + any non-NULL embedding ⇒ refuse** (unknown provenance); no stamp + zero embeddings ⇒ bootstrap-stamp from settings (create_all test DBs).
> 9. **USER DECISION (2026-07-19): the live store is NOT re-embedded — archived then emptied.** Archive `backups/ice_backup_20260719_115426_paper-era-pre-1024-wipe.tar.gz` (17 MB; DB dump + models/ + .env) + the pre-existing Exp2 mature snapshot `~/ice_exp2_mature_snapshot.sql` (47 MB, 2026-06-29) cover the historical record; the new experiments regenerate everything else. The migration therefore runs on empty tables; the runner's machinery is validated on seeded fixtures (§4 checks 2/5/6), not the live store. The user also declined re-registering the ICE project post-wipe (the existing `projects` row was a prior session's E-core validation registration, not theirs).
> 10. **Export list re-derived from models.py (26 tables):** everything EXCEPT `maintenance_ledger` (machine-local scheduler state + the `runtime_lease` pseudo-row — importing a foreign lease is an arbitration hazard; documented in the manifest). `idempotency_keys`, `session_replays`, `session_summaries` ARE exported (faithful state-copy). `store_meta` is exported; import re-stamps its embedding keys after the import-side re-embed pass.
> 11. **Backup includes `.env`** — D2's no-secrets rule governs the *export* (portable, shareable); the *backup* is a local private archive whose restore needs the config. `backups/` is gitignored.
> 12. **Validation check 3 adapted to the wipe:** no live pre-migration state survives, so the "pre-migration snapshot" is an in-test reference — seeded fixtures ranked with a `truncate_dim=384` reference encoder vs the native-1024 path (tests are the one place truncate_dim legitimately survives, as the equivalence oracle for checks 3/7).

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
