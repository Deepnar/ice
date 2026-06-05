# ICE — Project Progress
> Last updated: [YYYY-MM-DD]

---

## Current Phase & Step
Phase 2B — Step 2B.4: Validating labeled prompts with `validate_promt.py`

*(Update this to whatever is actually true for your project right now)*

---

## Last Completed
- `data/unlabled/dataset_unlabeled.jsonl` — 19,710 merged and deduplicated prompts from four sources (personal, wildchat, lmsys, sharegpt) — does not include labels
- `scripts/classifier/promt_labeling/VLLM_label_dataset.py` — async parallel labeling pipeline using instructor + vLLM, with idempotency and failed-prompt isolation — does not cover fine-tuning loop or validation
- `scripts/classifier/promt_labeling/validate_promt.py` — validates labeled prompts by analyzing label frequencies, co-occurrences, and data completeness — complete

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

---

## Deviations from Blueprint
*(Append-only)*

- [2025-XX-XX] Used uv instead of pip for all package management — Reason: project standard, cleaner dependency tracking
- [2025-XX-XX] Used vLLM (Qwen2.5-7B-AWQ) instead of Ollama (70B) for labeling — Reason: vLLM already set up, faster throughput, structured output via instructor
- [2025-XX-XX] Used instructor library for structured output instead of raw JSON parsing — Reason: more robust schema enforcement and retry logic
- [2025-XX-XX] Folder structure differs from BLUEPRINT.md — using scripts/classifier/ subdirectory instead of flat scripts/ — Reason: better organization

---

## Active Blockers
None.

---

