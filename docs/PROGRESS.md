# ICE — Project Progress
> Last updated: 2026-06-06

---

## Current Phase & Step
Phase 3 — Step 3.1 – Start the database and Redis with Docker

---

## Last Completed
- `models/classifier/ice_classifier_v2_final.pt` — final trained PyTorch classifier weights (saved from the most recent training run) — does NOT include a TorchScript export or production inference API
- `scripts/training/train_classifier.py` — training driver that reproduces the classifier training and logs runs to `models/classifier/training_runs.jsonl` — does NOT handle distributed training or automated production export

---

## File Inventory
*(Append-only — never delete entries)*
- `scripts/classifier/promt_labeling/validate_promt.py` — validates labeled prompts by analyzing label frequencies, co-occurrences, and data completeness — complete
- `data/unlabled/personal_promts.jsonl` — extracted personal prompts from raw chat logs — raw, no labels
- `data/unlabled/wildchat_promts.jsonl` — 5k WildChat first-turn prompts — raw, no labels
- `data/unlabled/lmsys_promts.jsonl` — 5k LMSYS first-turn prompts — raw, no labels
- `data/unlabled/sharegpt_promts.jsonl` — 5k ShareGPT human turns — raw, no labels
- `data/unlabled/dataset_unlabeled.jsonl` — merged and deduplicated, ~19,710 rows — no labels
- `data/unlabled/dataset_cleaned_filtered.jsonl` — cleaned version after filtering edge cases — no labels
- `data/labled/labeled_prompts.jsonl` — 19,710 labeled rows with topic, intent, context_reliance fields — complete for Paper 1 training
- `data/labled/failed_prompts.jsonl` — rows that failed labeling — not used for training
- `scripts/classifier/promt_extraction/extract_promts.py` — Amnesia Method extraction from raw chat logs — complete
- `scripts/classifier/promt_extraction/wildchat_extractor.py` — WildChat source extractor — complete
- `scripts/classifier/promt_extraction/lmsys_extractor.py` — LMSYS source extractor — complete
- `scripts/classifier/promt_extraction/sharegpt_extractor.py` — ShareGPT source extractor — complete
- `scripts/classifier/promt_extraction/combine_dataset.py` — merges four sources into one JSONL — complete
- `scripts/classifier/promt_extraction/clean_dataset.py` — deduplication and edge case filtering — complete
- `scripts/classifier/promt_labeling/VLLM_label_dataset.py` — async parallel labeling via vLLM + instructor — complete, using 7B not 70B
- `scripts/classifier/promt_labeling/prune_failed_promts.py` — removes failed rows from output — complete
- `scripts/classifier/promt_labeling/compare_labeling.py` — compares labeling outputs for quality checks — complete
- `docs/BLUEPRINT.md` — step-by-step build guide — reference only, not a contract
- `docs/ARCHITECTURE.md` — full system design and invariants — authoritative
- `scripts/classifier/promt_labeling/generate_synthetic_data.py` — generates synthetic prompts based on provided labels — complete
- `scripts/classifier/promt_labeling/synth_promt_gen_number.csv` — defines the number of synthetic prompts to generate for each label combination — complete
- `scripts/classifier/promt_labeling/synth_promt_renumber.py` — renumbers synthetic prompt IDs in a JSONL file — complete
- `scripts/classifier/promt_labeling/validate_promt.py` — validates labeled prompts by analyzing label frequencies, co-occurrences, and data completeness — complete
- `data/curated_fixes.jsonl` — contains curated fixes for training — complete
- `scripts/training/build_training_data.py` — builds training data from labeled prompts — complete
- `scripts/training/fine_tune.py` — fine-tunes the classifier model on curated fixes — complete
- `scripts/training/test_classifier.py` — tests the classifier with various prompts — complete
- `scripts/training/train_classifier.py` — trains the classifier model — complete

- `models/classifier/ice_classifier_v2_final.pt` — final trained PyTorch classifier weights (saved artifact from latest run) — does NOT include TorchScript optimization or deployment wrapper
- `models/classifier/training_runs.jsonl` — JSONL log of training runs, seeds, and validation losses — does NOT provide full experiment metadata or remote tracking
- `scripts/training/train_classifier.py` — training script for the ICE classifier — does NOT handle multi‑GPU/distributed training or CI integration
- `scripts/training/fine_tune.py` — fine‑tuning script that applies curated fixes to a checkpoint and saves an updated state dict — does NOT implement large‑scale distributed fine‑tuning or third‑party experiment tracking
- `scripts/training/build_training_data.py` — converts labeled JSONL into vectorized training examples and writes the label schema — does NOT perform label cleaning or automated repairs
- `scripts/training/test_classifier.py` — smoke/integration tests that run the model on hard prompts and print predictions — does NOT run as a unit/CI test suite
- `src/classifier/model.py` — PyTorch MLP architecture (`ICEClassifier`) used for training and inference — does NOT include model export utilities (TorchScript/ONNX)
- `src/classifier/dataset.py` — dataset loader that precomputes `all‑MiniLM‑L6‑v2` embeddings for training_data.jsonl — does NOT persist embeddings to disk or support streaming very large datasets
- `src/classifier/classifier.py` — `PyTorchClassifier` inference wrapper and `ClassificationResult` dataclass — does NOT provide a network API (FastAPI) or batch inference endpoint
- `scripts/classifier/promt_labeling/synthetic_data.py` — async synthetic prompt generator (uses local model endpoint / OpenAI‑compatible client) — does NOT include robust retry/backoff or production authentication handling
- `scripts/classifier/promt_labeling/synth_promt_gen_number.csv` — CSV defining how many synthetic prompts to generate per label combination — does NOT validate or normalize counts
- `scripts/classifier/promt_labeling/synth_promt_renumber.py` — renumbers `synth_` IDs in a JSONL file — does NOT merge conflicting IDs safely (simple overwrite approach)
- `scripts/classifier/promt_labeling/validate_promt.py` — dataset validation and label frequency/co‑occurrence reporting — does NOT auto‑fix detected data issues
- `data/curated_fixes.jsonl` — curated fixes used for fine‑tuning (sampled lines shown in repo) — does NOT claim to be exhaustive; used as high‑signal correction set
- `data/labeled/labeled_prompts.jsonl` — labeled prompt dataset (first lines inspected) — does NOT include the full file here due to size; canonical labeled dataset lives in `data/labeled/`

---

## Deviations from Blueprint
*(Append-only)*
- [2025-XX-XX] Used uv instead of pip for all package management — Reason: project standard, cleaner dependency tracking
- [2025-XX-XX] Used vLLM (Qwen2.5-7B-AWQ) instead of Ollama (70B) for labeling — Reason: vLLM already set up, faster throughput, structured output via instructor
- [2025-XX-XX] Used instructor library for structured output instead of raw JSON parsing — Reason: more robust schema enforcement and retry logic
- [2025-XX-XX] Folder structure differs from BLUEPRINT.md — using scripts/classifier/ subdirectory instead of flat scripts/ — Reason: better organization
- [2026-05-06] Added `scripts/training/build_training_data.py`, `scripts/training/fine_tune.py`, `scripts/training/test_classifier.py`, and `scripts/training/train_classifier.py` — Reason: to fine-tune and test the classifier model
- [2025-XX-XX] Used uv instead of pip for all package management — Reason: project standard, cleaner dependency tracking
- [2025-XX-XX] Used vLLM (Qwen2.5-7B-AWQ) instead of Ollama (70B) for labeling — Reason: vLLM already set up, faster throughput, structured output via instructor
- [2025-XX-XX] Used instructor library for structured output instead of raw JSON parsing — Reason: more robust schema enforcement and retry logic
- [2025-XX-XX] Folder structure differs from BLUEPRINT.md — using scripts/classifier/ subdirectory instead of flat scripts/ — Reason: better organization

---

## Active Blockers
None.

---

## Next Step

PHASE 3 — The Database (Architecture‑Complete) from BLUEPRINT.md