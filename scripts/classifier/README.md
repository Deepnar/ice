# scripts/classifier/ — the classifier data + training pipeline

Single home for everything that builds the pre-flight **prompt classifier** (topic / intent /
context-reliance heads). Consolidated here in the **2026-07-21 pre-B1 cleanup** (see
`docs/CLEANUP.md`) from four scattered locations: `scripts/classifier/promt_*`,
`scripts/training/`, and loose `scripts/*.py` utilities.

> Runtime inference code is **not** here — it lives in `src/classifier/` (imported by the API):
> `model.py`, `classifier.py`, `schemas.py`, `dataset.py`, `di3*`, plus the B1 additions
> `schema.py` (schema loader) and `templates.py` (the two inference prompt templates).
> The label taxonomy is `data/labeled/label_schema.json`. The corpus is `data/labeled/`.

## Structure

```
pipeline/   ← ACTIVE. The B1 (schema v2) end-to-end pipeline. Built during B1.
legacy/     ← FROZEN. The v1 scripts that built the current 25k corpus + v3 checkpoint.
              Kept for provenance. Do NOT run as-is — many hardcode pre-cleanup paths
              (e.g. data/ner/ now lives at data/archive/ner/; data/datasets/ never existed).
```

## pipeline/ — the B1 v2 flow (run in this order)

| stage | script | does | rewrites which legacy script |
|---|---|---|---|
| 1 | `extract.py` | pull online sources (LMSYS/ShareGPT/WildChat, **more + diverse**) + parse the `data/simulation/` exports (via `src/ingestion/formats.py`) → prompt rows with context | `legacy/promt_extraction/*` |
| 2 | `stitch_icedev.py` | stitch the DeepSeek ice-dev chats into one chronological mega-conversation (shared asset with FINAL) | new |
| 3 | `synth.py` | schema-v2 synthetic generation (**local model**) to fill measured gaps | `legacy/promt_labeling/synthetic_data.py` |
| 4 | `label.py` | **two independent LOCAL labelers, different families** + agreement/tiebreak/human-queue | `legacy/promt_labeling/VLLM_label_dataset.py` + `compare_labeling.py` |
| 5 | `build.py` | render templates, ≥40% context-prefixed rows, ≥1k hard-negative context pairs, per-label floors, train/val/test split | `legacy/training/build_training_data.py` |
| 6 | `train.py` | trunk + 3-heads (1024-dim), per-head BCE | `legacy/training/train_classifier.py` |
| 7 | `evaluate.py` | per-head macro-F1 + the D5 non-regression gate vs the old model | `legacy/training/test_classifier.py` |
| 8 | `promote.py` | backup + atomic swap of `settings.classifier_model_path` | (was in `legacy/training/fine_tune.py`) |

`combos.csv` (synth generation grid) carries over from `legacy/promt_labeling/synth_promt_gen_number.csv`, updated for v2 labels.

## legacy/ — what each v1 piece was (knowledge kept, code frozen)

**promt_extraction/** — built the unlabeled corpus:
- `lmsys_extractor.py`, `sharegpt_extractor.py`, `wildchat_extractor.py` — pull each online dataset → per-source `*_promts.jsonl` (now in `data/archive/unlabeled/`).
- `extract_promts.py` — turned the user's raw chats (`data/archive/raw_logs/raw_chats.txt`) into `personal_promts.jsonl`.
- `combine_dataset.py` — dedup + merge all sources → `dataset_unlabeled.jsonl` (source-tagged).
- `clean_dataset.py` — text cleaning pass.

**promt_labeling/** — labeled it (single-labeler v1):
- `VLLM_label_dataset.py` — the strong labeler: source-aware LTM thresholds, 6 immunity traps, 6 signals A–F, reasoning-first. **Reused as the base for the v2 two-labeler `label.py`.**
- `synthetic_data.py` — the strong synthetic generator (casual-voice, per topic/intent/context). **Reused as the base for v2 `synth.py`.**
- `synth_promt_gen_number.csv` — the generation grid. `synth_promt_renumber.py` — id renumberer.
- `compare_labeling.py` — two-file agreement diff (stale paths → `data/datasets/` which never existed). Its diff logic informs the v2 two-labeler agreement step.
- `validate_promt.py`, `prune_failed_promts.py` — QC passes.
- `OLAMA_BAD_label_dataset.py` — a rejected Ollama labeler attempt (self-named "BAD"; stale paths). Kept only as a record of what didn't work.

**training/** — trained the v3 checkpoint:
- `build_training_data.py` → `data/labeled/training_data.jsonl`; `train_classifier.py` → the checkpoint; `fine_tune.py` → curated-fix fine-tunes + promotion; `test_classifier.py` → eval.

**loose utilities** (were at `scripts/` root):
- `build_probe_input.py`, `probes_count.py` — probe-set tooling; `ltm_fix.py` — LTM probe relabel; `insert_curated_and_fine_tune.py` — curated-label fine-tune driver.

## The 25k corpus (for reference)
25,354 labeled rows in `data/labeled/labeled_prompts.jsonl`: lmsys 5,255 + wildchat 5,257 +
sharegpt 5,159 (≈15.7k online breadth) + personal 7,711 + synthetic 1,972. Context-reliance v1:
Zero_Shot 18,844 / Long_Term_Memory 5,891 / Real_Time_Search 619. B1 re-labels this text under
schema v2 (labels not reused — see the B1 spec).
