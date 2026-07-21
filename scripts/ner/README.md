# scripts/ner/ — micro-NER training tooling

The pre-flight **entity recognizer** (`src/retrieval/ner_utils.py` loads `models/ner/ner_model.pt`
at runtime). This folder holds its *training* tooling only.

## Structure

```
legacy/   ← FROZEN. The v1 NER training chain (stale: built against the OLD 384-dim encoder).
            A9/B1 retrain NER at native 1024, which regenerates all of this.
```

`pipeline/` (the 1024 rework) is added when A9/B1 retrains NER — it inherits the shared
`src/memory/embedder.py::get_embedder()` (native 1024) instead of the old `slice384` shim.

## legacy/ — the v1 chain (run order, all stale for 1024)

`extract_turns.py` (turns from `data/labeled/labeled_prompts.jsonl`) → `label_entities.py`
(LLM entity labels) → `generate_bio.py` (BIO tags) → `train_ner.py` (the checkpoint). Plus
`integration_test.py` / `test_orchestrator_ner.py` (manual checks).

Their intermediates were archived in the 2026-07-21 cleanup:
`data/ner/{raw_turns,extracted_entities,training_data}.jsonl` → `data/archive/ner/`, and the
4.4 GB `data/ner/embeddings_cache.pt` (old-encoder cache) was **deleted** (regenerable). So the
legacy scripts' hardcoded `data/ner/…` paths no longer resolve — they are a record, not runnable
as-is. See `docs/CLEANUP.md`.
