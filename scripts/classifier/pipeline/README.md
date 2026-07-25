# scripts/classifier/pipeline/ — the B1 (schema v2) classifier pipeline

The active v2 rewrite of the classifier data+training flow, built during B1 (2026-07-25).
Old→new mapping lives in `../README.md`; the design decisions live in
[docs/specs/B1_classifier_retrain.md](../../../docs/specs/B1_classifier_retrain.md).

Every stage is a standalone script (house style), resumable by row id, and writes into
`data/labeled/v2/`.

## Order

```
extract  →  stitch_icedev  →  label  →  merge/tiebreak  →  synth*  →  build  →  train  →  evaluate  →  promote
   1             2             4          4                  3         5        6          7           8
```

\* **synth runs after the merge, not before** (numbering kept for the file names). A gap
measured from one labeler's pass is provisional in a known direction — agreement keeps the
*intersection*, so a single pass always overstates, and the tiebreak moves it again.
Generating against that number produces the wrong amount of the wrong thing. Waiting is
cheap: labelers resume by id, so re-labeling the newly generated rows touches only those.

| stage | what it does | output |
|---|---|---|
| `extract.py` | online pulls (bucket-weighted for diversity) + `data/simulation/` exports via F10's adapters + v1 corpus **text** reuse | `corpus_raw.jsonl` |
| `stitch_icedev.py` | the ICE-N DeepSeek chats → ONE chronological conversation (**shared asset with FINAL**) | `icedev_stitched{,_dialogue}.jsonl` |
| `synth.py` | casual-voice prompts for **measured** label gaps; rows carry a generation *hint*, never a label | `corpus_synth.jsonl` |
| `label.py` | two different-family local labelers → agreement → third-family tiebreak → human queue; T2 detector seeds `Temporal_Recall` | `labels_{a,b,tiebreak,final}.jsonl`, `review_queue.jsonl`, `audit_sample.jsonl` |
| `build.py` | join, ≥40% context-prefixed, hard-negative pairs, per-label floors, conversation-grouped split | `train/val/test.jsonl` |
| `train.py` | trunk + 3 heads at 1024-dim, per-head BCE with pos-weights, early stop | `models/classifier/ice_classifier_v4_schema2.pt` |
| `evaluate.py` | per-label F1 + **D5 non-regression gate** (both models, identical rows) | `*_eval.json` |
| `promote.py` | re-runs the gate, then backup + atomic swap of the live checkpoint | live path |

Support modules: `common.py` (paths, resumable JSONL, dedupe, diversity buckets),
`rubric.py` (the ~400-line labeling prompt, rendered from `label_schema.json`),
`serving.py` (local vLLM/SGLang lifecycle + per-model 24 GB profiles),
`compare.py` (agreement between any two passes — reuses the merge's own rule).

**Vet a labeler before you spend hours on it.** Run the candidate over a few hundred rows
another labeler already covered, then compare:

```bash
uv run python label.py --labeler A --model gpt-oss-20b --limit 200 --out ../../../data/labeled/v2/dryrun.jsonl
uv run python compare.py labels_b.jsonl dryrun.jsonl
```

~90% agreement means interchangeable; ~50% means one of them is reading the rubric
differently, and you want to know which *before* the five-hour pass, not after.

## Running it

```bash
# 1-2: corpus (minutes; --all = online 15k + personal exports + v1 text reuse)
uv run python extract.py --all
uv run python stitch_icedev.py

# 3-4: generation + labeling (GPU hours — one model at a time on 24 GB)
uv run python synth.py --per-label 300
uv run python label.py --labeler A          # then B, sequentially
uv run python label.py --labeler B
uv run python label.py --merge              # agreement + queues
uv run python label.py --labeler C --tiebreak-only
uv run python label.py --merge

# 5-8: dataset → model → gate → live
uv run python build.py
uv run python train.py
uv run python evaluate.py --candidate models/classifier/ice_classifier_v4_schema2.pt
uv run python promote.py --candidate models/classifier/ice_classifier_v4_schema2.pt --yes
```

Dry runs: `label.py --limit 200` (stratified sample), `build.py --dry-run`.

## Things that will bite you

**The GPU must be free.** Ollama pins the chat model in VRAM (`UNTIL: Forever` keeps it
resident); `ollama stop <model>` first or the server fails with "Free memory on device
cuda:0 … is less than desired GPU memory utilization". It reloads on the next chat request.

**Don't force `--quantization`.** Community requants of the same model disagree about their
own format — one ships AWQ, another compressed-tensors. Every checkpoint declares its
method in `config.json`; the profiles pass nothing and let the server read it.

**Not every requant loads.** `mattbucci/Qwen3.6-27B-AWQ` fails in vLLM 0.22.0 with "input
size is not aligned with the quantized weight shape" — its *vision tower* is misquantized.
That is a property of the repo, not of the model: another Qwen3.6-27B requant may be fine.
Check a model loads before planning a 6-hour run around it.

**SGLang is preferred but currently unusable here** — the pinned `0.3.6.post2` can't import
against the environment's Triton. `--backend vllm` is the default for that reason;
`--backend sglang` stays wired for when SGLang is upgraded.

**The two labelers must be different families.** Two Qwen variants agreeing measures a model
against itself, and the agreement rate — the number the whole methodology rests on — becomes
meaningless. Same rule for the tiebreak model.

## Settled labelers (vetted on real rows, 2026-07-25)

| slot | model | family | speed | note |
|---|---|---|---|---|
| A | `Qwen/Qwen3-14B-AWQ` | qwen | 1.77 rows/s (~6.2 h) | won the slot on speed |
| B | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | gemma | 2.34 rows/s (~4.7 h) | full pass DONE |
| C | `openai/gpt-oss-20b` | openai | 1.20 rows/s | tiebreak only; needs `reasoning_effort=low` |

Measured agreement against Gemma: the three memory signals **90.8-98.6%**, `High_Complexity`
83-84%, topic 78-80%, intent 64-66%. All-three-heads agreement is only 36-38%, which is why
`High_Complexity` is excluded from the agreement gate (`SOFT_CTX_LABELS`) and why the tiebreak
is sized as a local pass (~24k rows, ~4 h) rather than a handful of rows.
