# Provenance ledger

**What produced each artifact, recorded when it was produced.**

This file exists so that writing the paper is a matter of *reading* rather than
archaeology. Every entry below was captured at the time the run happened — model
revisions, library versions, row counts, decisions. Reconstructing any of it six
months later means guessing, and the parts that matter most (which exact weights,
which exact corpus) are the parts that become unrecoverable fastest.

**Standing rule (2026-07-26):** when a run produces an artifact that anything
downstream depends on — a corpus, a labeled set, a checkpoint, an experiment
result — add an entry here *in the same session*. It belongs in
`ICE_Architecture.md` eventually, but that describes the system as it *is*; this
records what was *done*.

**Pin model revisions, not just names.** A community quantization can be
re-uploaded, silently revised, or deleted. `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
is not a reproducible reference; that repo *at revision `0ef577a5…`* is. HF caches
the revision as the `snapshots/<sha>` directory name, so it costs nothing to record:

```bash
ls ~/.cache/huggingface/hub/models--<org>--<name>/snapshots
```

---

## Environment (as of 2026-07-26)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop, 24 GB (23.46 GiB usable), driver 610.43.03 |
| RAM | 62 GB |
| Python | 3.11.9 (`.python-version`), deps via `uv` |
| torch | 2.11.0+cu130 |
| transformers | 5.9.0 |
| vLLM | 0.22.0 — **the serving engine actually used** |
| SGLang | 0.3.6.post2 — installed but **unusable**: cannot import against this Triton (`default_cache_dir`) |
| Postgres | `pgvector/pgvector:pg16` via `docker/docker-compose.yml` |

---

## B1 — classifier retrain (schema v2), 2026-07-25/26

**Commits:** `1f834f1` core · `48e65cd` pipeline · `0e3c684` context-prefix fix ·
`04990b7` prompt cap + synth reorder · `b534863` gpt-oss profile + comparator ·
`7c7aeed` per-model request overrides · `6b275e8` labeler roster + High_Complexity
rule · `07e1f0c` hang fix · `5ed75ff` validation principle · `83f435a` two piles +
curation dedup. Migration `c4d7e91a2b58` (curated_labels schema v2).

### Embedder (unchanged from C17)

| | |
|---|---|
| model | `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| width | native **1024** (v2 classifier); `slice384` MRL prefix only for v1 checkpoints + micro-NER |

### Corpus (`data/labeled/v2/`)

**39,289 rows**, 34% context-prefixed in the live exchange format (asserted
byte-identical to `classifier._get_context_turns`: last 3 user→assistant
exchanges, 150-word cap each, 500-word total).

| source | rows | origin |
|---|---|---|
| personal | 11,009 | 6,320 v1-corpus text + 4,689 fresh from `data/simulation/` via F10 adapters |
| lmsys | 8,570 | `lmsys/chatbot_arena_conversations` |
| wildchat | 8,479 | `allenai/WildChat-1M` |
| sharegpt | 8,465 | `anon8231489123/ShareGPT_Vicuna_unfiltered` |
| synthetic | 1,468 | v1 synthetic rows, **text reused, labels discarded** |
| icedev | 1,298 | 6 stitched DeepSeek ICE-N chats, 3,473 turns, 2026-06-04 → 07-03 |

v1 labels were never mapped forward — the single-label 3-way context head is the
defect B1 removes, so every row was relabelled from scratch.

### Labelers

Three distinct lineages, run sequentially (24 GB holds one model at a time).
Served by vLLM 0.22.0, `--enable-prefix-caching`, context length 8192,
temperature 0.0, seed 42, JSON-schema constrained decoding via `response_format`.

| slot | model | revision | family | throughput | notes |
|---|---|---|---|---|---|
| A | `Qwen/Qwen3-14B-AWQ` | `31c69efc29464b6bb0aee1398b5a7b50a99340c3` | qwen | 1.77 rows/s | needed `enable_thinking: false` + `max_tokens 1400`; without it ~10% truncated JSON |
| B | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | `0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba` | gemma | 2.34 rows/s | cleanest run: 0.01% degenerate, 116 failures / 39,289 |
| C (tiebreak) | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | openai | 1.20 rows/s | MXFP4, MARLIN MoE kernel; needs the largest token budget of the three |

**Rejected candidates, with the reason** (so they are not re-picked):

| model | verdict |
|---|---|
| `mattbucci/Qwen3.6-27B-AWQ` | will not load — "input size is not aligned with the quantized weight shape" on `visual.blocks.*`; misquantized vision tower |
| `jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym` | loads and serves, emits **token soup** behind schema-valid JSON (18/18 rows). The reason `label.is_degenerate` + its abort gate exist |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 25.0 GB of weights > 23.5 GB VRAM |
| `hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4` | 24.7 GB — same |
| `cyankiwi/GLM-4.5-Air-AWQ-4bit` | 63.4 GB — not close |

**On community quantizations:** unavoidable at this size (24 GB cannot run these
models unquantized, and official 4-bit AWQ frequently does not exist — Google ships
gemma-4 as bf16 + QAT GGUF only). Mitigations actually in force: two of six
candidates were rejected on *measured* behaviour rather than reputation; labels come
from consensus across three lineages plus a human audit; and FINAL's judge and GT
generation are **cloud/full-precision**, so quantization never touches a scored
result. For FINAL's local conditions the rule is to serve the *same* model across
conditions, so quantization is a constant in the ICE-vs-baseline delta.

### Measured agreement (two independent pairs, same stratified sample)

| | vs Gemma (n=184, qwen3-14b) | vs Gemma (n=142, gpt-oss) |
|---|---|---|
| Needs_Memory | 90.8% | 94.4% |
| Temporal_Recall | 96.7% | 98.6% |
| Needs_Live_Info | 97.3% | 95.1% |
| High_Complexity | 84.2% | 83.1% |
| topic / intent | 77.7% / 66.3% | 80.3% / 64.1% |
| all three heads | 36.4% | 38.0% |

The memory signals — the ones that gate retrieval — agree 90%+. The all-three
figure is a property of the gate (it compounds topic/intent noise onto context),
which is why `High_Complexity` was removed from the agreement requirement.

### Run 1 merge (2026-07-26, before the tiebreak)

A 39,203 labeled / B 39,173 / overlap 39,154 → **19,710 settled (50.3%)**,
19,444 queued for tiebreak, human queue 0 (rows reach a human only after the
tiebreak fails). Per-head disagreement: intent 11,059 · topic 8,303 ·
context_reliance 5,512. T2's deterministic detector contributed 288
`Temporal_Recall` positives. Audit sample 983.

⚠ Label counts from this merge are **half-counts** over the settled subset —
`Codebase_Query` at 24 is not a drop signal until the tiebreak completes.

### Run 2 merge — after tiebreak, union rule, and Pile B (2026-07-26)

**38,669 settled of 39,154 (98.8%).** gpt-oss tiebreak resolved 19,214 rows; the
union rule auto-settled the fuzzy heads (intent 3,263 · topic 1,740); Pile B added
289 hand-authored rows. **Human review queue: 544, all of them context-reliance
disputes** — the fuzzy-head arguments never reach a person. T2's detector
contributed 612 `Temporal_Recall` positives.

**Every label clears the 300 floor. Nothing is dropped**, including the spec's
predicted casualty:

| context signal | settled |
|---|---|
| (derived Zero_Shot) | 28,242 |
| Needs_Memory | 7,997 |
| High_Complexity | 2,520 |
| Temporal_Recall | 2,297 |
| Needs_Live_Info | 1,262 |

`Codebase_Query` 316 · `Code_Change` 1,567 · every other intent ≥ 873.

**909 corpus rows (2.3%) never settled** — 230 short of the tiebreak queue, 67
never labeled by either A or B, the rest labeled by only one so there was nothing
to compare. Accepted as loss (user).

### Pile B — hand-authored rows (289)

Written because the corpus **could not contain** these classes, not because they
are rare. A label gated on a capability the collection environment lacked stays
rare however much data you gather:

| class | organic | why censored |
|---|---|---|
| Needs_Memory across conversations | **0** of 6,806 | assistant couldn't see other chats, so nobody phrased it that way |
| Codebase_Query | 65 | no repo access — "where is X in my project" was pointless |
| Memory + Live_Info | ~137 | no web search to make it worth asking |
| Meta_AI about ICE's own memory | ~0 | no system with memory to interrogate |

Labels ship WITH the prompt and never reach the labelers (`scripts/classifier/
pipeline/authored.py`); batch scripts under `scripts/oneoff/b1_authored/`.

### Independent evaluation set — 207 probes (NEVER trained on)

`data/labeled/v2/eval_probes_independent.jsonl`, built from the user's own
`data/labeled/probes_labeled_ltm.jsonl` (708 rows → 238 unique prompts; 42 dropped
as already present in the training corpus).

This exists because train/val/test all descend from the same two labelers and
therefore inherit their shared blind spots — a split cannot detect the bias of the
process that produced it. Pile B cannot serve as the exam either, since it is
trained on. These probes were written by the **user**, months earlier, for
Experiment-1 curation, and no labeler in this pipeline has touched them.

Asserts **one** label, `Needs_Memory`, which is true by construction (a curation
probe is asked to test recall). The v1 topic/intent labels came from a weak 7B
model and are carried as an unscored hint. 64 of the rows carry model reasoning
concluding `Zero_Shot` against a stored label of `Long_Term_Memory` — the user's
override, and correct.

### Training splits (2026-07-26)

**33,197 rows → train 23,756 / val 4,386 / test 5,055.** 38.8% context-prefixed
(v1: ~0%), 1,000 hard-negative context pairs, template render 100%.
Conversation-grouped split so turns of one conversation cannot straddle train and
test. Hand-authored rows are **exempt from the standalone down-sampling** — the
first build discarded 32 of them to hit the context ratio, which throws away the
only examples of the censored classes.
