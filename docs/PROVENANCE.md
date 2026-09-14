## 2026-09-14 — v3 adversarial NLI and source-sentence follow-through

`qualify_nli.py --attribution` compares the original multilingual candidate and
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` at
`b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7` on the same20 controls plus10 attribution
controls. Artifacts: `experiments/v3_repair/results/nli_multilingual_attribution.json`
and `nli_adversarial_attribution.json`. The second model accepts13/13 supported
and0/17 unsupported controls by argmax; it fixes the first model's three failures.
Complete-input float32 CUDA; short-pair inference+transfer1.397s, peak allocated
1.762GB. No source uploads. Training differences motivate selection, not a causal
claim from this comparison. The tiny non-English subset does not establish
multilingual reliability. Source/correctness labels are hand-authored synthetic
controls, not representative held-out ICE conversations.

`nli_production_path.json` records the same30 controls through
`src.memory.support.score_pairs` and `verify_support`:13 supported,17 not supported
at the explicit0.95 policy cutoff; an807-token input returns unknown before
inference, without truncation. The cutoff is not calibrated to a population.
Claim reading uses the shorter quotation only with supported current hashes;
otherwise it keeps the complete evidence paragraph. This establishes a consumer,
not a claim that NLI verifies world truth or solves every summary/conflict case.

A bounded live NuExtract3-Q8_0 template check added `source_sentence` to the three
existing slots. Its three exact source sentences preserved a PostgreSQL choice,
a hypothetical Redis cache and teaching direction; the Redis triple flattened
its condition. That motivates source-sentence rendering instead of trusting slot
order/modal information. This single synthetic response is not a new extractor
accuracy estimate. The subsequent controlled DB tests exercise the actual writer,
independent lexical/vector lookup, source roles/hashes, graph links and forgetting.

## 2026-09-13 — v3 NLI candidate diagnostic controls

`experiments/v3_repair/qualify_nli.py` ran the pinned multilingual mDeBERTa
candidate8adb042d on20 synthetic source/hypothesis pairs, float32 CUDA, complete
inputs without truncation. Artifact: `experiments/v3_repair/results/nli_qualification.json`.
Six supported controls were classified entailment; three of14 unsupported
controls were also classified entailment: hypothetical0.8634, quoted denial0.5722,
negated reporting0.8286. This disqualifies argmax entailment as an assertion
authority; no threshold was fitted. Source attribution/context preservation
remain mandatory. Inference plus GPU transfer1.285s, peak allocated1.136GB for
these short pairs, excluding model load and unrelated process VRAM. This is
candidate qualification, not ICE production semantic accuracy. No NLI gate activated.

# Provenance ledger

## 2026-09-12 — v3 local reranker qualification

One candidate: `Qwen/Qwen3-Reranker-0.6B` at HF revision
`e61197ed45024b0ed8a2d74b80b4d909f1255473`, downloaded into the local HF cache.
No corpus upload. Sentence Transformers 5.5.1, Transformers 5.9.0, Torch 2.11.0;
CUDA float16 on the 24 GB laptop GPU. Exact production instruction in
`src/retrieval/reranker.py`; no training or threshold sweep.

`uv run python tests/test_reranker_quality.py` scored 60 synthetic pairs:
five questions (port, discontinued tool, change date, deployment procedure,
authentication module), each in three forms and against four candidates.
**Correct item first 15/15; positive at zero 15/15; distractors below zero
41/45.** Four false admissions changed across phrasings: zero-floor rejection
is **unqualified and OFF by default**. Ordering is enabled; these controls do
not establish universal style invariance, multilingual performance, semantic
truth, multi-hop coverage or end-to-end answer improvement.

Both real `retrieve()` paths, with synthetic leg output and the actual scorer,
changed a fixed-budget selection from a distractor to the answering fact with
reranking on versus off (2/2). This tests selection wiring, **not database search
quality or a vector benchmark**. The final implementation requests only the
last-token logits, disables KV caching, and returns weights to CPU after scoring.
Final 60-pair call including lazy load: 2.876 s; process peak CUDA allocation
1.178 GiB. Not whole-proxy latency or peak under maximum-length inputs.

Artifact: `experiments/v3_repair/results/reranker_qualification.json`.
Earlier feasibility calls and the initial failed all-distractor rejection
assertion led to the ordering-only decision; no threshold was fitted to these
examples. Controlled-score smoke checks cover invalid results, repeated warning
fallback, representation-specific scoring, provenance and token packing.


## 2026-09-12 — frozen-v2 paper final analysis and NORA preparation

No new model run. Existing matched-cloud LongMemEval records reproduce ICE/vector
50.8/72.8% oracle and 43.0/69.5% full-S. Paired 20,000-resample intervals are
−22.0 [−26.6,−17.4] and −26.5 [−31.3,−21.8] points; common-question extra
phase degradation is +4.4 [−0.2,9.2]. Full-S abstention remains descriptive
(7/1 discordant; exact McNemar p=0.0703125). Aggregate outputs include paired
cells, four-outcome phase cells, provider-token and generation-time distributions.

LSREP repeats 219 probes into 1,211 observations at 52 checkpoints. Whole-probe
resampling yields ordinary-density Δ=+0.002 [−0.148,0.158]; without manual
replacements −0.020 [−0.169,0.136]. Merged vector-generalist scores include 130
failure assignments and two sibling substitutions. Complete-case all-data
Δ=+0.046 [−0.101,0.199] removes most density failures and does not replace
reliability. Paired ordinal ordinary counts are 216 ICE-higher, 215 vector-higher,
626 ties. These estimates condition on the same single-user histories.

Artifacts and reproducible entry points: `experiments/paper/ARTIFACTS.md`.
Canonical: `ICE_paper_v2.{tex,pdf}`; anonymous NORA research-track preparation:
`ICE_paper_arr.{tex,pdf}`. Paired/ordinal controls and PDF compilation validate
analysis and presentation, not universal system effectiveness or judge correctness.
Current v3 development state and the frozen architecture report are unchanged.


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

### Training run 1 (2026-07-26) — miscalibrated, superseded

Cap 20, threshold 0.3. Recall 0.87–0.97 with precision 0.04–0.81 on **all 28
labels** — a model that fires nearly everything. Kept in the record because the
shape of that failure is the diagnosis: at ~1% prevalence the neg/pos ratio
saturates a cap of 20, so a miss costs 20× a false alarm and "always yes" is the
cheapest policy each head can learn.

### Training run 2 (2026-07-27) — the shipped candidate

`models/classifier/ice_classifier_v4_schema2.pt`. Seed 42, identical splits and
cached embeddings, so every number below is reproducible by re-running `train.py`.

| | |
|---|---|
| architecture | trunk 512→256, heads 11 / **12** / 4 = **27 logits**, 663,324 params |
| pos-weight cap | **5** (module default 3) |
| tag_threshold | **0.65**, fitted on val and stamped into the checkpoint |
| early stop | epoch 55, best val loss 0.9078 |
| schema change | **`Codebase_Query` dropped** — intent 13 → 12 |

**pos-weight cap sweep** (identical splits/seed; mean macro-F1 on test at
per-label fitted thresholds): cap 20 → 0.585 · cap 10 → 0.595 · cap 5 → **0.602**
· cap 3 → 0.595. **The cap was not the root cause.** After fitting thresholds all
four are within 0.017; the cap mainly moves *where* the optimum sits (0.55–0.65 at
low caps vs 0.85–0.95 at cap 20). The threshold did the work: 0.526 at the
inherited 0.3 vs **0.610** fitted — a bigger gain than any architectural change in
B1. Per-label thresholds were measured and **rejected**: 0.610 vs 0.609 global,
and the head where they would matter (context_reliance) never passes through
`_tags_above`.

**Test scores, per-label fitted thresholds** — topic 0.654 macro / 0.798 weighted ·
intent 0.521 / 0.622 · context_reliance 0.654 / 0.708. Key labels: Needs_Memory
**0.794** · Needs_Live_Info 0.685 · High_Complexity 0.572 · Temporal_Recall 0.567 ·
Code_Change 0.481.

**Gate 1 — D5 non-regression** vs the live v1 checkpoint, each model at its own
threshold: topic +0.193, intent +0.227, context_reliance +0.169, overall
0.453 → 0.649. **PASS** (a floor, not proof — the baseline is graded on a v2 rubric).

**Gate 2 — 207 independent probes + a 3,774-row false-fire control.** Retrieval
fires on real memory prompts **0.705 → 0.831**; false fires on no-memory rows
**0.238 → 0.118**; separation +0.467 → **+0.713**.

**Gate 3 — 104 hand-authored adversarial probes** (`hard_probes.py`, authored this
session, never trained on). Full 3-head exact match 53/104. On the decision that
matters — does retrieval fire — **accuracy 84%, precision 0.83, recall 0.82**
against the live v1's 78% / 0.73 / 0.84: false alarms nearly halved (15 → 8) for
one extra miss. The context-twin check confirms B1's central claim works: the same
sentence scores p_mem 0.74 without context and 0.22 with it (also 0.88→0.75,
0.64→0.40 — correct direction every time), which v1's 3-way softmax was
structurally incapable of.

### The label ceiling — the run's most important measurement

Per-label inter-labeler agreement (A vs B, positive-class F1) placed beside the
trained model's per-label F1: **correlation 0.90, mean gap −0.01.**

| label | labelers agree | model scores |
|---|---|---|
| Codebase_Query | 0.10 | 0.10 |
| Open_Exploration | 0.26 | 0.37 |
| High_Complexity | 0.42 | 0.57 |
| Code_Change | 0.55 | 0.48 |
| Troubleshooting | 0.71 | 0.69 |
| Generation | 0.76 | 0.75 |
| Needs_Memory | 0.79 | 0.79 |

The model has extracted what its supervision contains. **No amount of further
training, tuning, or fine-tuning on this corpus can move these numbers** — only
supervision from outside these labelers can. Record this before anyone plans
another retrain.

Two supporting measurements. (a) **Intent disagreements are 90–100%
one-directional**, not mutual — A said `Factual_Retrieval` where B said
`Open_Exploration` 1,030 times vs 8 the other way; B uses `Open_Exploration` 4.1×
and `Ideation` 2.4× more than A. That is labeler calibration, *not* label overlap,
so merging confused labels would delete real distinctions to hide one model's
bias. (b) A **22-row random audit**, all heads judged by hand: context_reliance
~5% wrong, topic ~10–15%, intent ~25–30%, with over-tagging the dominant error and
almost all of it on rows where all three labelers split (8.4% of the corpus, left
on the union rule by user decision).

### Codebase_Query — dropped, and how to bring it back

219 training positives (above §4's <150 floor) but test F1 **0.10**, precision
0.16, recall 0.08. Not a scarcity failure: labeler A tagged 452 rows, B tagged
221, and they **overlapped on 33**. The head reproduced its supervision exactly.
The corpus is website chat with no repository access, so the class barely occurs.
The annotations remain in the data and `dataset.py` ignores schema-absent tags, so
re-adding the schema entry and retraining restores it **with no relabeling** —
worth doing only once E7's MCP surface produces real navigation traffic. Roadmap
**E12** owns the decision; `label_schema.json`'s `dropped_labels` block carries the
definition and rationale.

### B2's weights after the v2 swap — predicted mistuned, measured fine

**The prediction, and why it was reasonable.** B2 consumes the classifier as a
scalar so a retrain would not force a rewrite; that seam held. But type
compatibility is not distribution compatibility: v1's `p_ltm` was one class's share
of a 3-way softmax (compressed, roughly ±2 in logit space) while v2's is an
independent sigmoid that saturates (roughly ±4.6). Every additive bump was sized
against the old range. Supporting evidence: on the 104 adversarial probes the v2
*head* wins 84% vs 78% with half the false alarms, yet end-to-end through
`decide_memory_retrieval` the two **tie at 80%** (v2 TP42/FP14/TN41/FN7, v1
TP41/FP13/TN42/FN8).

**The measurement, which refuted it.** `scripts/classifier/pipeline/tune_b2.py`:
coordinate descent over seven knobs, two passes, scored on 256 positives / 655
negatives (207 user curation probes + 104 authored adversarial probes + 600
held-out rows with no `Needs_Memory` gold). Under the correct objective the shipped
defaults are already essentially optimal; the sole admissible change is
`ltm_bump_creative` 0.7 → 0.35, worth **+0.005 specificity with 0.000 change on
both probe families**. Noise. **Not applied — B2 ships unchanged.**

Shipped operating point on that set: **balanced accuracy 0.865, recall 0.922,
specificity 0.808** (TP236/FP126/TN529/FN20).

**The objective is the finding worth keeping.** Plain balanced accuracy *does* find
+0.0105 — by zeroing four bumps and trading recall for specificity (0.922 → 0.871),
and the user-probe family loses 0.058 doing it. For a silent gate that is the wrong
direction: a false negative means retrieval never ran and nothing says so, while a
false positive costs one round-trip the assembler's budget already bounds. So the
objective is *maximise specificity subject to recall ≥ the shipped baseline*, and
per-family accuracy is reported to catch precisely that overfit.

Two mechanical traps recorded because they would fool a hand-tune:
`ltm_bump_low_confidence`, `ltm_bump_reference` and `ltm_length_weight` are **inert
on this data** — every grid value scores identically, so an argmax zeroes them by
tie-breaking and it looks like a result; and `ltm_bump_reference` only fires through
DI3's anaphora path, which a direct-model harness never exercises. The script keeps
the current value on ties for this reason.

So the end-to-end tie is **not** B2 miscalibration. B2's bumps deliberately spend
specificity to buy recall, which is correct here, and that trade flattens the head's
precision gain when measured by a symmetric metric.

### Live configuration changed this session

`temporal_label_threshold` **0.6 → 0.85** (`src/api/config.py`). `Temporal_Recall`
fires as a shadow of `Needs_Memory` (79% co-occurrence in training positives; mean
p_temporal 0.87 on hand-authored memory prompts with no temporal content), and it
is OR'd with T2's deterministic detector, so a low threshold makes the parser
redundant and biases toward always-retrieve. Z1-prep owns the final value and must
sweep it against the **independent** probe sets, never the held-out split.

---

## D8 / A9a / E12 — the post-promotion audits, 2026-07-27/28

No model was trained and no artifact re-generated; these runs *measured* the
promoted checkpoint against the code around it. Recorded because three deletions
and one roadmap item (T5) rest on the numbers, and because two of the findings
contradict what the specs predicted.

**Inputs, common to all three.** Checkpoint
`models/classifier/ice_classifier_v4_schema2.pt` (schema_version 2,
template_version 2, input_dim 1024, tag_threshold 0.65). Rows: `test.jsonl`
5,055 + `val.jsonl` 4,386 = **9,441** held-out, read through
`ICEClassifierDataset` so the cached `.emb_*.pt` embeddings are reused — the
inference input is byte-identical to training. Device CPU; no GPU needed.
Probe sets: `hard_probes_authored.jsonl` (104) + `eval_probes_independent.jsonl`
(207). Reproduce: `scripts/classifier/pipeline/eval_di3.py --splits test val`
and `scripts/classifier/pipeline/audit_labels.py`.

### D8 — DI3 vs the v2 head, on the rows DI3 intercepts

Slices are first-match rule order, i.e. the population each path was responsible
for. `conversation_length` is 0 because no caller passes it.

| slice | rows | share | topic F1 DI3→model | intent F1 DI3→model | retrieval acc DI3→model |
|---|---|---|---|---|---|
| code | 675 | 7.1% | .878 → .928 | .191 → .515 | .692 → .803 |
| sentiment | 92 | 1.0% | .238 → .791 | .246 → .622 | .707 → .772 |
| meta | 149 | 1.6% | .248 → .790 | .156 → .594 | .745 → .805 |
| noise | 0 | 0.0% | — never fires — | | |
| reference | 1,694 | 17.9% | (emits no tags) | | .852 → .777 **with** the bump |
| passed to ML | 6,831 | 72.4% | | | |

Decisions taken: all five paths deleted; no inline noise guard (zero population);
`ltm_bump_reference` and `ClassificationResult.reference_signal` deleted with them.

End-to-end gate on the 311 independent probes — **the only instrument that can
see this change**, because `score_hard_probes.py` and `eval_probes.py` both load
the checkpoint directly and never call `classify()`:

| | accuracy | precision | recall | silent misses (hard / user) |
|---|---|---|---|---|
| pre-D8 | .884 | .951 | .906 | 8/49 · 16/207 |
| post-D8 | **.897** | .952 | **.922** | **7/49 · 13/207** |

T2 side-effect: non-current TimeScopes 465 → 416; all 49 removed were false
positives (long pasted documents, p_ltm 0.00–0.16). `REFERENTIAL_WORDS` as a
substitute measures worse (527).

### A9a — the rollback contract, verified before touching it

`ice_classifier_v3_qwen_ft3.pt` carries **no metadata dict** (bare state_dict);
`load_checkpoint` infers schema_version 1 / input_dim 384 / heads (11,11,3) from
the weight shapes. Loading it through the live `PyTorchClassifier` returns
`Long_Term_Memory p_ltm=0.987` — and does so *only* because of the narrowing
branch A9a proposed deleting. Six copies of that branch consolidated into
`embedder.fit_width`; post-change both generations serve (v1 0.987, v2 0.993),
an illegal 512 narrowing raises, and `eval_probes` reproduces B1's recorded gate
off the v1 arm exactly (retrieval .705 → .831, false-fire .238 → .118).

### E12 — do B1's labels reach a decision?

`Code_Change`: head fires on 543/9,441 (5.75%), precision 0.42 / recall 0.50
against gold; mean **1.79** intents when it fires (alone on 148 = 27.3%), so it
delivers ~56% of its profile weight and shifts the largest leg by **0.291**
(base weights 0.2–1.2). Companions: Troubleshooting ×269, Generation ×105.
Against the B1 label-ceiling table its 0.48 sits under a 0.55 labeler ceiling —
**5× more learnable than the dropped `Codebase_Query` (0.10)**. Kept.

`Temporal_Recall`: gold co-occurrence **P(Needs_Memory | Temporal) = 78.1%**
(557 temporal rows, 2,140 memory rows, 122 temporal-without-memory), and
**P(Temporal | Needs_Memory) = 20.3%**. Label fires ≥0.85 on 310 rows, T2
detector on 416, overlap 135; the 175 label-only rows carry mean p_ltm **0.931**
and 172 (98.3%) already retrieve, so disabling the whole arm moves **1 decision
in 9,441**. As a *filter* on T2's gate instead: precision 84% (416 rows) → 89%
(≥0.5, 259) → **93% (≥0.7, 187)** → 94% (≥0.85, 135). As an extra OR-arm: admits
2 of 116 refused rows, both wrong. 250 of 557 gold-temporal rows carry no
parseable date; 206 never reach the gate at all. → roadmap **T5**, post-Z1.

### The style-dependence measurement (feeds G28)

Per-source firing of T2's gate arms, the finding that reframed G28:

| source | n | has `?` | starts with an interrogative | interrogative in first 8 words | signal discarded by the first-word rule |
|---|---|---|---|---|---|
| **personal** | 2,859 | 40% | **2%** | 23% | **21%** |
| lmsys | 2,084 | 45% | 25% | 35% | 10% |
| wildchat | 2,060 | 31% | 14% | — | — |
| sharegpt | 2,042 | 30% | 13% | 22% | 10% |

The rule reads word position 0; people write `"so what about comparision to the
ground truth"`. Note the public corpora disagree with each other 2x as well (25% / 14% / 13%), so this
is instability across populations, not one unusual user.

**And the head is not automatically the fix.** Firing-rate spread across the same
four sources: `has "?"` 1.5x, `interrogative-1st` 10x, `p_temporal>=.85` 8.2x,
**`p_ltm>=.5` 16.2x** — the model's outputs swing more than the crudest
heuristic. This is confounded (personal rows genuinely need memory more than
one-shot public prompts) and **the confound is the finding**: firing-rate-by-source
cannot separate different people from different meanings. Standing consequence:
treat it as a smell detector, and use paraphrase invariance (same intent, N
surface forms, measure the decision-flip rate) as the acceptance test — for the
heuristic AND its replacement. Roadmap **G28** owns that probe set.


## C16 — context measurement runs (2026-07-29)

The ledger's trigger fires here: this session produced measurements that later
claims rest on. Machine: RTX 5090 Laptop (24 GB), CachyOS. Ollama 0.30.x.

### The ruler — `words × 1.33` vs the real tokenizer

Tokenizer: `Qwen/Qwen3-Embedding-0.6B` (the embedder's own, already loaded).
Ratios are estimate ÷ real, so <1.0 is an UNDERCOUNT:

| content | words×1.33 | chars/4 |
| --- | --- | --- |
| plain prose | 1.20× | 1.12× |
| fenced Python | 0.53× | 0.72× |
| ICE's `[date] User:/Assistant:` stamped turn format | 0.55× | 0.60× |

**Why this matters beyond C16:** Experiment 2 measured BOTH arms with
`words × 1.33`. The vector-RAG baseline injected raw prose (overcounted ~20%);
ICE injected stamped structured text (undercounted ~1.7×). The reported "~25%
fewer tokens" is therefore biased in ICE's favour by an unquantified amount.
Any re-measurement must use a real tokenizer for both arms.

### Exp 2 re-read (no new run — re-analysis of `experiments/mature/results/`)

*Not a new discovery: `docs/specs/FINAL_experiments.md`'s reviewer table already
lists the 4.26-vs-4.25 ROI and the 22,411-vs-21,025 tokens as criticisms 2 and 3.
What was NOT recorded anywhere is the measurement bias in the section above,
which changes how those numbers should be read.*

| slice | ICE tokens | baseline tokens | ICE score | baseline score |
| --- | --- | --- | --- | --- |
| all 1,211 probes | 22,099 | 29,550 | 4.27 | 3.87 |
| excluding `ice_dev` (1,057 probes) | 22,411 | 21,025 | 4.26 | 4.25 |

`ice_dev` is where the baseline fed 88k tokens, OOM'd on 145 probes and scored
1.23. The headline −25.2% is that one conversation; on everything else ICE
costs **+6.6% more for +0.01 score**. The baseline's std was 27,006 on a mean
of 29,550 — the mean summarised nothing. ICE's own std was 3,026 on 22,411,
i.e. it emitted ~22k regardless of the question. **Acceptance statistics from
here on: paired median and win-rate, never a mean of means.**

### Embedder device

Same model, same texts, `sentence-transformers` on this machine:

| | CPU | GPU |
| --- | --- | --- |
| one encode | 321 ms | 21 ms (11 ms warm) |
| batch of 100 | 22.5 s | 0.38 s |
| VRAM held | — | ~1.2 GB |

`device="cpu"` was hardcoded (uncommented) since G23/C17's embedder
consolidation `d4d0a79`. Every chat turn encodes the prompt on the pre-flight
path. Now `embedding_device` = auto | cuda | cpu, defaulting to GPU when one
exists (user decision 2026-07-29: GPU by default, never GPU-only).

**Cross-device vector difference:** cos(cpu, gpu) = 0.9998–0.9999, max
component |Δ| ≈ 2.2e-3, TF32 matmul off. **Decision impact, live v2 head
(`ice_classifier_v4_schema2.pt`), all 207 rows of
`data/labeled/v2/eval_probes_independent.jsonl`:**

- `context_reliance` (the memory gate): **207/207 identical**
- all labels identical: **199/207** — 8 borderline multi-label tags cross the
  0.65 threshold

So the gate does not move; leg weighting wobbles at the margin.

### Serving-window truth

`tinyllama:latest` — registry `8192`, GGUF `2048`, live runner (`/api/ps`)
`2048`. `derive_total_budget`'s minimum guardrail returned **4,000** for it,
i.e. twice the whole window before any output. Fixed: reality outranks the
floor, and `context_generation_reserve` (2,048) comes off the top.

### Silent truncation — confirmed, not inferred

Ollama's `/v1` shim **does** honour `stream_options: {include_usage: true}`.
Prediction vs the server's own `prompt_eval_count`, per request:

| prompt | predicted | actual | ratio |
| --- | --- | --- | --- |
| short | 283 | 331 | 0.855 |
| short | 341 | 408 | 0.836 |
| **oversized (~2,900 tokens)** | **2,909** | **2,047** | **1.42** |

The last row is the finding: the server received 2,047 tokens — exactly the
model's window — for a prompt of ~2,900. **The prompt was silently truncated,
and truncation keeps the newest messages, so the block most likely destroyed
is the retrieved-memory block.** The 0.83–0.86 ratios on the short prompts are
the embedder-vs-Llama vocabulary gap and are why
`token_count_safety_margin` was raised 1.10 → **1.20** (measured, not assumed).

---

## A9b / A12 / G4 — the background-pipeline evaluation (2026-08-03)

An evaluation session: no production code was written, and the ledger's trigger
fires because the A9b, A12 and G4 decisions now rest on the measurements below.
Machine: RTX 5090 Laptop (24,463 MiB), Ollama **0.30.7**, service env
`OLLAMA_KEEP_ALIVE=-1  OLLAMA_FLASH_ATTENTION=1  OLLAMA_KV_CACHE_TYPE=q4_0`.

### The turn set

58 turns from `data/simulation/simulation_full.jsonl` (gitignored, personal —
never committed, and no turn text appears in this file). Stratified over the
four Exp-2 conversations (Shinchan, Flaw, ICE-Dev, Masters) plus the rest of the
personal corpus, crossed with a length band (short <150 w / medium <600 w /
long), 4 per cell, turns over 2,500 words excluded. Seed **20260803**. Chosen
because these are the domains ICE actually serves and the ones Z2 reuses, so
results transfer.

### ⚠ The finding the rest of the session hangs off

**Every background LLM call returned nothing.** Reproduced on six real call
sites against the live default background model, each at its own shipped
`max_tokens`:

| call site | budget | as shipped | with `reasoning_effort="none"` |
| --- | --- | --- | --- |
| `codex_extractor.extract_triplets` | 500 | **0 triplets**, empty response | 6.5–7.5 triplets/turn |
| `post_flight._summary_llm_call` | 300 | **empty string** | median coverage 0.92 |
| `maintenance_agent` decider | 200 | **None** (`agent_llm_failed`) | `{"verdict": "merge"}` |
| `decision_extractor._extract_llm` | 300 | **None** | decision + rationale JSON |
| `clustering._generate_cluster_name` | 200 | **`"Unnamed Cluster"`** | `"Symbolism of Existential Flaws"` |
| `documents.detect_blob_kind` | 8 | `blob_kind_unparsed` → default | decides correctly |

*Not* individually tested: reflection ×5, batch_summarizer, conversation_summary,
procedural_extractor, the codex reconciler, raw_slicer, registry tagging. They
share the mechanism; that is an inference, not a measurement.

**Mechanism.** Every model in the live registry except `qwen3:4b-instruct` is a
reasoning model. Ollama spends the output budget inside a hidden reasoning
block, returns `finish_reason="length"` with `content=""`, and puts the thinking
in a separate `reasoning` field ICE never reads. Measured budget needed for ONE
summary: **gemma4:12b 810 tokens, gemma4:26b-a4b 919**. ICE allows 300.

**Thinking ON is not the alternative.** 26B, thinking on, budget raised 5× to
2,500: **0 triplets on 12/12 turns at 42.0 s each**, against 7.5 triplets at
6.6 s with thinking off. The reasoning block scales with input length.

**Two of the six degrade silently** — `"Unnamed Cluster"` and
`blob_kind_unparsed → document` are fallbacks that make a dead layer look like a
working system. That is why this survived undetected.

### What Ollama's OpenAI-compatible `/v1` endpoint accepts

⚠ **This table is what was TESTED, not the complete control surface.** It was
assembled by hitting the four things this session happened to need; the native
API exposes more, and the gap has not been enumerated. Treat it as evidence that
the shim drops parameters, not as the list of parameters it drops.

| parameter | honoured on `/v1`? | evidence |
| --- | --- | --- |
| `options` (`num_ctx`, …) | **NO** | ctx stayed 32,768; native `/api/chat` allocated 16,384 and 745 MiB less on a 4B |
| `keep_alive` | **NO** | expiry stayed at year 2318; native honoured 30 s exactly |
| `think` | **NO** | content still empty |
| `chat_template_kwargs` | **NO** | content still empty |
| ~~JSON-schema constrained decoding (`format`)~~ | ~~**NO** (native only)~~ | ~~documented upstream; OpenAI `response_format` is ignored by Ollama~~ |
| **⚠ THE ROW ABOVE IS WRONG — corrected by the G32 audit below (2026-08-03).** It was recorded from upstream documentation, not measured, and it is the one row in this table nobody tested. `response_format: {"type":"json_schema"}` **IS honoured** (8/8); only `{"type":"json_object"}` is ignored (0/8). | | see *G32 — the Ollama control-surface audit* |
| `reasoning_effort: "none"` | **YES** | `finish_reason=stop`, 35 output tokens, clean content |
| `stream_options.include_usage` | YES | C16 already relies on it |

### NER candidates

`numind/NuNER_Zero` @ **`c90187673f464518dca09f41689184ed6976242c`** — MIT,
448.9M params, backbone `microsoft/deberta-v3-large`, `max_len=384`
**GLiNER-words** (punctuation counts: 350 whitespace words measured 424 and was
silently truncated), `max_width=1`, so the card's `merge_entities` is mandatory.
`numind/NuNER_Zero-4k` @ `7a7cd8d65af2572c297054dbaf8f25c0d46da55d` (max_len
2048, longformer backbone) — exceeding its limit is a CUDA device-side assert,
not an exception. `urchade/gliner_large-v2.1` measured as a cross-check and lost
on every probe (`Redis/dataset`, `My character/character`, `competes/event`).

Latency, median ms, by device and input size:

| | micro-NER GPU | micro-NER CPU | NuNER GPU | NuNER CPU |
| --- | --- | --- | --- | --- |
| short prompt (pre-flight) | **13** | 272 | 16 | 446 |
| full turn (background) | 345 | **16,811** | **101** | 4,458 |

197 relation labels accepted in 0.088 s; 5 → 19 labels costs +7%. **Label count
is not a constraint**, which settles the open question in the A9b entry.

Style invariance (20 turns, meaning fixed, form varied; Jaccard of the
case-folded entity set against the original):

| variant | micro-NER | NuNER Zero |
| --- | --- | --- |
| lowercase | **0.000** (0/20 identical) | 0.732 |
| punctuation stripped | 1.000 | 0.855 |
| `"ok so like "` prefix | 1.000 | 1.000 |
| filler inserted | 1.000 | 0.932 |

### NER per consumer — the result that made A9b a division, not a swap

**Codex grounding** (`extract_triplets`, 10 turns, same model and prompt, only
the NER swapped): micro-NER **104** triplets total / median 9.5, winning on
**9 of 10** turns; NuNER 80 / median 7.0, winning on 0. The entity list is a
*permissive whitelist*, so a long noisy list gives the model more legal subjects
and it discards the junk itself, while a short clean list forbids real facts.

**Union arm, run 2026-08-03 after the `reasoning_effort` fix landed** (10 turns,
same model and prompt, three arms in one run):

| grounding | total triplets | median | grounded share | median s | paired vs micro |
| --- | --- | --- | --- | --- | --- |
| micro-NER | 166 | 17.5 | 0.952 | 6.97 | — |
| NuNER Zero | 84 | 7.5 | 0.738 | 5.54 | median −6.0, 2W/1T/7L |
| **union** | **170** | 15.0 | 0.953 | 7.94 | median **+1.0**, 5W/2T/3L |

So the union is a **tie-to-marginal-win over the micro-NER alone** (+4 triplets
over 10 turns), not the clear win the pre-flight numbers suggested — and it
costs +14% latency. NuNER alone remains clearly worst here, and its lower
grounded share (0.738 vs 0.952) is the mechanism made visible: the narrow
whitelist does not stop the model proposing facts, it just marks a quarter of
them low-confidence.

⚠ **Absolute counts vary run to run** — the micro-NER arm scored 104 in the
first run and 166 in this one on the same turns with the same model at
temperature 0. Only *within-run* comparisons are used anywhere in this entry.

**Clustering** (within- vs across-conversation entity overlap, five
conversations as weak labels):

| | mean within | mean across | ratio | entities in ≥3 conversations |
| --- | --- | --- | --- | --- |
| micro-NER | 0.0435 | 0.0221 | 1.97× | **28** (`and`, `but`, `for`, `not`, `now`, `let`, `like`, …) |
| NuNER | 0.0252 | 0.0009 | **28×** | **1** |
| union | 0.0395 | 0.0182 | 2.2× | 31 |

The micro-NER tags **function words** as entities; `_NER_STOP` does not contain
them and `_EDGE_TRIM` only trims them at span edges, not as standalone spans.

**Pre-flight graph matching** (entities from the user's prompt against real
Codex node names, built by running `extract_triplets` over the same
conversations):

| | node hits | prompts with ZERO hits (of 58) | median entities/prompt |
| --- | --- | --- | --- |
| micro-NER | 58 | **48** | **0** |
| NuNER | 43 | 33 | 3 |
| union | **94** | **30** | 3 |

### Background model candidates — generation

30 turns, `reasoning_effort="none"`, must-terms held fixed per turn, production
summariser path verbatim. Paired against the incumbent (median delta + win rate,
never a mean of means):

| model | size | median coverage | median Δ | W/T/L | win rate | speed |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma4:26b-a4b-it-q4_K_M` | 18.0 GB | 0.92 | — | — | — | 1.0× |
| `gemma4:12b` | 7.6 GB | 0.90 | −0.020 | 11/4/15 | 0.42 | 0.53× |
| `qwen2.5:7b` | 4.7 GB | 0.08 | −0.817 | 0/0/30 | 0.00 | — |
| `qwen3.5:4b` | 3.4 GB | 0.92 | 0.000 | 13/7/10 | 0.57 | 1.60× |
| `qwen3:4b-instruct` | 2.5 GB | **0.98** | +0.040 | 17/3/10 | **0.63** | 1.69× |

`qwen3:4b-instruct` emitted the required `Abstract:` line on only **60%** of
summaries — format compliance is a separate axis from coverage.

### Background model candidates — extraction

Same turns, ICE's own `extract_triplets`, `reasoning_effort="none"`:

| model | median triplets | mean | turns yielding zero |
| --- | --- | --- | --- |
| `gemma4:26b-a4b-it-q4_K_M` | **7.5** | 11.17 | **1 / 12** |
| `qwen3:4b-instruct` | 3.5 | 4.17 | 4 / 12 |
| `qwen3.5:4b` | 2.0 | 1.75 | 4 / 12 |

Only the 26B handled all four negation probes correctly. **This gap was measured
against an UNCONSTRAINED decoder** — see the A12 entry for why that makes the
number provisional.

### Rejected candidates, with the symptom (so nobody re-tests them)

* **`qwen2.5:7b`** — degenerate repetition on short *and* long inputs, with a
  recurring Spanish token (`pérdida`); `llama3:8b` and `tinyllama` are clean on
  the identical prompt, so it is a bad quant, not the q4_0 KV cache. It is
  currently `settings.default_fallback_model`.
* **`gemma4:12b`** — worse than a 4B on summaries at 3× the size and half the
  speed.
* **`numind/NuExtract3`** @ **`2e9fca82ee641e6bb6e1f5d905241e994be27a07`** —
  Apache-2.0, base `Qwen/Qwen3.5-4B`, 4.54B BF16, vLLM 0.22.0
  (`--max-num-seqs 16` required: the hybrid Mamba cache refuses the default
  256), **12,587 MiB resident**. Its template grammar *does* express ICE's
  triplet shape including a `"boolean"` negation field, and it was flawless on
  four short probes — it even split "no longer uses PostgreSQL, moved to SQLite"
  into a negated and a positive triplet, which the incumbent did not. **But with
  the 197-relation enum, 1 turn in 3 enters an infinite repetition loop** — the
  same triplet emitted forever, still looping at 6,000 output tokens. Enum size
  is the driver: 197 → 66.7% parse / 11.07 s; 51 (property leg) → 91.7% /
  4.89 s / 10.5 triplets; none → 100% / 0.58 s. Kept only as a property-leg
  candidate.
* **`knowledgator/gliner-multitask-large-v0.5`** — its "Open Information
  Extraction" mode is `labels=["match"]` plus a natural-language prompt, i.e.
  prompt-matched spans, **not** open triplets. This closes the one question the
  previous session left open; the earlier conclusion stands.

### Shortlist for the deferred model benchmark (selected by property, not size)

Chosen on the three properties the measurements showed actually decide the
outcome — non-reasoning, sound quantization, structured-output discipline — and
explicitly **not** on parameter count, since a 7B was the worst arm and a 12B
lost to a 4B:

* **IBM Granite 4.1 3B / 8B** (Apache-2.0) — non-reasoning by design, built for
  structured output, leads tool-calling benchmarks, notably token-efficient.
* **Ministral 3 8B** (Apache-2.0), **Gemma 4 E4B** — closest peers.
* **Nemotron-mini 4B** — reported to produce valid JSON where Llama 3.2 3B fails.
* Incumbents to beat: `qwen3.5:4b`, `qwen3:4b-instruct`.

**The benchmark is deliberately deferred until constrained decoding lands**,
because the extraction gap above was produced by an unconstrained decoder and a
schema constraint is expected to change the ranking.

### GPU / residency measurements

| | |
| --- | --- |
| chat model resident (`gemma4:26b-a4b-it-q4_K_M`) | 16,446 MiB, cold load **7.24 s** |
| `qwen3.5:4b` | 3,439 MiB, 3.27 s |
| `qwen3:4b-instruct` | 3,916 MiB, 1.62 s |
| 26B + a 4B co-resident | 19,661 MiB by Ollama's own accounting, **21,662 by nvidia-smi** (Ollama under-reports ~2 GB) |
| `num_ctx` effect on the 26B | 16,329 MiB at ctx 4,096 vs 16,446 at 32,768 — **120 MiB across an 8× range**, because the KV cache is q4_0 |
| explicit unload | `POST /api/generate {"model": X, "keep_alive": 0}` → `done_reason: "unload"`, gone from `/api/ps` |

A CUDA OOM occurred during the session with 22 GB held — a 26B nobody was using
plus a 4B in use — because `OLLAMA_KEEP_ALIVE=-1` means nothing is ever
released. One unload call freed 18 GB.

### Two silent-degradation traps confirmed, both CWD-relative model paths

| path | file | behaviour outside the repo root |
| --- | --- | --- |
| `models/ner/ner_model.pt` | `ner_utils.py` | silently falls back to a capitalized-word regex, **no log line** |
| `models/model_registry.json` | `registry.py` | `load_registry()` returns `{}` → `get_fallback_model()` → `settings.default_fallback_model` = `qwen2.5:7b`, the broken model |

Both fired accidentally during this session and both produced results that
looked like findings until the working directory was checked: one run's NER
entity count silently dropped 52.7 → 34.1, and one run's background model
silently became the broken quant.

### Tooling note

`gliner==0.2.28` + `onnxruntime` were installed into `.venv` with `uv pip
install` for this evaluation and are **not** in `pyproject.toml`/`uv.lock`; a
`uv sync` removes them. Scripts were run via `.venv/bin/python` directly,
because `uv run` re-syncs the environment on every invocation.

---

## G32 — the Ollama control-surface audit (2026-08-03)

The audit G32 makes its own first deliverable. No production code was written.
Same machine as the A9b/A12/G4 session above: RTX 5090 Laptop (24,463 MiB),
Ollama **0.30.7**, service env `OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q4_0`. Probe models `qwen3:4b-instruct-bg` and, for the
thinking arms, the resident `gemma4:26b-a4b-it-q4_K_M`. Full write-up in
[specs/G32_ollama_transport.md](specs/G32_ollama_transport.md) §0.

Every probe is two-sided against an **observable** effect (a number in
`/api/ps`, a response field, schema conformance). "It didn't error" is never
the test — the shim accepts unknown parameters silently.

### Correction to the table above

`response_format: {"type": "json_schema", …}` is **honoured** by Ollama's `/v1`
shim. Measured on the schema shape codex extraction needs — an array of
triplets whose `relation` is enum-constrained:

| arm | conformance (n=8) |
| --- | --- |
| native `format: <schema>` | **8/8** |
| `/v1 response_format: json_schema` | **8/8** |
| `/v1 response_format: json_object` | **0/8** |
| `/v1` no constraint (control) | **0/8** |

Conformance is checked two-sided: valid JSON **and** correct keys **and** every
`relation` inside the enum. A stronger arm confirms the constraint actually
binds rather than the model merely cooperating — with the enum narrowed to the
single value `RELATES_TO`, which no verb in the text implies, both the native
and `json_schema` arms emitted `RELATES_TO` five times out of five.

⇒ `json_object` — **the form `maintenance_agent.py:399` sends today** — is
indistinguishable from sending no constraint at all.

### Parameters still confirmed dropped by `/v1`

| control | native | `/v1` | observable |
| --- | --- | --- | --- |
| `options.num_ctx` | 8,192 allocated | 32,768 (the default) | `/api/ps` `context_length` |
| `options.top_k=1` @ temp 1.6 | 3 → **1** distinct | 6 → **6** distinct | output spread, n=6 |
| `keep_alive: 0` | **unloaded** | still resident | `/api/ps` membership |
| `think: false` | 0 ch thinking, real content | no such key | `message.thinking`, on the 26B |

Honoured on both: `temperature`, `seed` (1 distinct at temp 1.6, n=4), `stop`,
`max_tokens`. Native streaming carries `prompt_eval_count`/`eval_count`/
`done_reason` on the final chunk **unconditionally**; `/v1` needs
`stream_options.include_usage`.

### Truncation — the C16 finding explained, and one arm invalidated

⚠ **A first comparison here was invalid and is recorded because the shape
recurs**: the native arm sent `num_ctx=2048`, the `/v1` arm sent nothing — and
`/v1` *drops* `num_ctx`, so it ran at 32,768 and never truncated. The apparent
result ("native is loud, the shim is silent") was an artifact of two different
windows. Redone with the window pinned on the resident runner.

Ollama 0.30.7 does **not** truncate by default — it **reloads the runner at a
larger window**. A runner pinned at 2,048 received ~6,012 tokens and came back
resident at 32,768, answering normally, on both transports.

Silent truncation happens only when the model **cannot** grow. On `tinyllama`
(GGUF ceiling 2,048), a ~6,012-token prompt returned, on **both** transports:

    HTTP 200 · done_reason "stop" · prompt_eval_count 2047

~4,000 tokens discarded, no error, no flag, **natively too**. This reproduces
and explains C16's `predicted=2909 / actual=2047`: tinyllama was the model that
could not grow. **The native endpoint is not inherently louder about
truncation.** Loudness comes from the *combination* — send `options.num_ctx`
explicitly and the same request becomes a typed refusal:

    HTTP 400 exceed_context_size_error
    "request (6012 tokens) exceeds the available context size (2048 tokens)"
    n_prompt_tokens 6012 · n_ctx 2048

`n_prompt_tokens` and `n_ctx` are the two numbers ICE's budget arithmetic wants
to check itself against, and `num_ctx` is exactly what the shim drops.

### `/api/show` exposes a `capabilities` list

`gemma4:26b-a4b-it-q4_K_M` → `['completion','vision','tools','thinking']`;
`qwen3:4b-instruct-bg` → `['tools','thinking','completion']`; `gpt-oss:latest`
→ `['completion','tools','thinking']`.

⚠ **Template-derived, not behavioural.** `qwen3:4b-instruct-bg` advertises
`thinking` and emits **0 characters** of it under both `think:true` and
`think:false`. Usable as a negative filter (absent ⇒ unsupported), never as a
positive guarantee.

### Endpoints, and what ICE uses

All present on 0.30.7: `/api/chat`, `/api/generate`, `/api/embed`,
`/api/embeddings`, `/api/ps`, `/api/show`, `/api/tags`, `/api/pull`,
`/api/push`, `/api/create`, `/api/copy`, `/api/delete`, `/api/blobs/:digest`,
`/api/version`, `/api/signout`, plus `/v1/{chat/completions,completions,
models,embeddings,responses}`. ICE uses **four**: `/v1/chat/completions` (chat
path via raw httpx; background path via the OpenAI SDK), `/api/tags`
(`registry.populate_from_ollama`), and `/api/ps` + `/api/show`
(`runtime_probe`). `/api/embed` returned **501 "does not support embeddings"**
and is a non-lever regardless — ICE embeds locally and G23 pinned embedding
identity.

### Two live defects found while auditing, both transport-independent

1. `maintenance_agent.py:399` sends `response_format={"type":"json_object"}` —
   measured 0/8, i.e. no constraint at all.
2. `registry.py:162` hardcodes `model="Qwen/Qwen2.5-3B-Instruct-AWQ"` (the
   *dedicated*-mode default) while `get_bg_client()` in the default *shared*
   mode points at Ollama → verified `404 model 'Qwen/Qwen2.5-3B-Instruct-AWQ'
   not found`, swallowed by `except Exception: return {"topic_tags": [],
   "intent_tags": []}`. Background model auto-tagging has been dead in the
   default configuration, silently — a third instance of the CLAUDE.md
   silent-fallback rule.

### The relation-matching evidence run (2026-08-04)

Run to decide how G32/a1 should handle the relation vocabulary. **It overturned
the design the session had converged on**, so the numbers matter more than the
conclusion that preceded them.

**Setup.** 300 turns, **identical across both arms** (fixed seed 20260804):
150 personal (`data/simulation/simulation_full.jsonl`, gitignored) + 150 public
(50 each lmsys / wildchat / sharegpt from `data/labeled/v2/corpus_raw.jsonl`),
all ≥60 words. Model **`qwen3:4b-instruct`** — the 26B arms were dropped
mid-run because A12 already decided one small model, so characterising the 26B
would describe a model ICE will not ship. Harness is standalone (`harvest.py`),
**not** `extract_triplets`, because that function drops out-of-vocabulary
relations at `codex_extractor.py:514` — the dependent variable. Fidelity is
preserved where it counts: the prompt is **captured from the real
`extract_triplets` via a stub client**, asserted to contain **197/197**
relations, the category headers, and the *"if nothing fits, SKIP IT"* rule.
No filtering of any kind is applied afterwards.

**Arm C — JSON shape constrained, `relation` left free text.**

| | |
|---|---|
| triplets | 1,699 (689 distinct relations) |
| in-vocabulary | 547 — **32.2%** |
| **out-of-vocabulary** | **1,152 — 67.8%**, every one silently destroyed today |

Out-of-vocabulary rate by source: lmsys 62% · personal 66% · wildchat 70% ·
sharegpt 78%. The most frequent misses are not near-misses of vocabulary terms,
they are **ordinary English the 197-word list does not contain**: `is` (92×),
`has` (68×), `includes` (29×), then a long tail of free inventions
(`is_exam_of`, `exists_in`, `designed_as`, `has_task`).

**The deterministic ladder, measured on that real output.** Levels are
exact → lexical normalisation → rule-based (helper verbs, articles, preposition
synonyms) → alias dictionary → lemmatisation (nltk WordNet) → fuzzy
(rapidfuzz) → token containment → embedding.

| level | distinct resolved | occurrences resolved |
|---|---|---|
| L1 exact | 0/606 | 0.0% |
| L2 normalize | 2/606 | 0.2% |
| L3 rules | 5/606 | 0.7% |
| L6 lemmatize | 13/606 | 2.7% |
| L5 fuzzy WRatio ≥90 | 52/606 | 20.9% |
| L5 fuzzy ratio ≥92 | 4/606 | 0.5% |
| containment | 30/606 | 3.7% |
| L7 embedding ≥0.85 | 46/606 | 23.9% |
| **cascade L1▸L2▸L3▸L6▸L5(92)▸containment** | **38/606** | **5.7%** |

⚠ **A 31-probe hand-authored set scored this same cascade at 24/24 with zero
errors.** Real output scores it at **5.7%**. The probe set was written by the
same author as the ladder and tested the failures that author imagined
(space-vs-underscore, typos, helper verbs); the model's actual failure is
**inventing new concepts**, which no string method can map. Recorded as
[TRAPS](TRAPS.md) #13.

Hand-audit of the cascade's 29 distinct proposals: ~4 are wrong (≈14%), and
**two are direction inversions** — `is_used → uses` and `is created by →
created` reverse the subject and object. Both came from containment/lemmatise,
neither of which knows about direction.

**Arm D — same 300 turns, plus the 197-value enum at the decoder.**

| | |
|---|---|
| triplets | 1,673 (143 distinct) — essentially unchanged volume |
| in-vocabulary | **100%** by construction |

Paired against arm C on the **716** `(turn, subject, object)` triples both arms
produced identically: the relation agrees on 406 (**57%**), differs on 310, and
in **292** of those the shape arm's pick was out-of-vocabulary — i.e. those are
facts today's code destroys and the enum "rescues".

**Hand-audit of the top 45 rescues: roughly 8–10 are acceptable and ~35 are
wrong (≈75–80%).** The failure has a clear shape — a few relations act as
**attractors** that absorb anything inexpressible:

    is                    -> is_employed_by   (7x)
    is_defined_by         -> is_dating        (6x)
    is_family_member_of   -> is_dating        (3x)
    monitors              -> is_employed_by   (3x)
    is_sibling_of         -> is_separated_from(2x)
    has_environmental_impact -> is_founder_of (2x)

against genuine successes like `includes → contains` (10×), `feature →
features`, `worked_on → works_on`, `involved_in → participates_in`.

⇒ **The enum converts ~1,126 silently-dropped facts into confidently-wrong
ones.** A dropped fact costs recall; a wrong fact is retrieved and served as
truth. The enum is **not** adopted.

**Truncation — a separate live bug, confirmed.** At production's
`max_tokens=500`, `finish_reason == "length"` occurred on **2 of 12** turns and
the JSON was unparseable every time. **Constrained decoding does not prevent
this** — it guarantees a valid grammar *prefix*, not completion within budget.
`finish_reason` is present on every response and **no caller in `src/` reads
it**.

**What the run actually establishes.** The problem is not the transport, the
constraint, or the matching algorithm — it is that **the 197-relation
vocabulary does not fit real conversation**, and the three most-wanted
relations are `is`, `has` and `includes`. Neither dropping (32% kept) nor
forcing (100% kept, ~78% wrong) is acceptable, and string matching recovers
5.7%. The decision belongs to a vocabulary experiment with this data in hand —
scheduled at **Z2** — and the 606 ranked missing relations this run produced
are its input.

## G31 / G5 / G25 — cluster ①, the instrument-trust pass (2026-08-08)

**Why these are recorded.** None of them produced a corpus or a checkpoint, but
all three produced *measurements a later decision will rest on* — specifically,
two measurements that **contradict what the roadmap said was true**. A future
session scoping G25, or wondering whether a pre-2026-08-08 result is
trustworthy, needs these numbers rather than the entries' prose.

**Environment.** Same machine and stack as the 2026-08-03 sections. Postgres
`pgvector/pgvector:pg16` (docker), Ollama on :11434, live store **empty**
(0 conversations, 0 turns) before and after — the three residue rows created
during validation were removed, see CLEANUP.md.

### G31 — what the working directory silently changed

Probe: import `src.api.config` + `src.model_registry.registry` from two
directories and diff. Read-only, no DB, no model load.

| observable | from repo root | from `/tmp` (before fix) | from `/tmp` (after fix) |
|---|---|---|---|
| `.env` read | yes | **no** | yes |
| `confidence_fallback_threshold` | 0.5 | **0.75** | 0.5 |
| registry models | 6 | **0** | 6 |
| `get_fallback_model()` | `gemma4:26b-a4b-it-q4_K_M` | **`qwen2.5:7b`** | `gemma4:26b-a4b-it-q4_K_M` |
| micro-NER ckpt found | yes | **no** (→ regex) | yes |
| classifier ckpt found | yes | **no** (FileNotFoundError) | yes |
| label schema | yes | yes (already anchored) | yes |

**The `.env` row is the one that was not in the roadmap entry and matters most
for the paper**: `confidence_fallback_threshold` gates the orchestrator's
wide-net fallback, a *degraded single-leg retrieval mode*. Any result produced
by a script launched outside the repo root before 2026-08-08 ran with 0.75, not
the intended 0.5. The other three `.env` keys (`DATABASE_URL`,
`OLLAMA_BASE_URL`, `BACKGROUND_MODEL_MODE`, `CLASSIFIER_THRESHOLD`) happen to
equal their code defaults, so they are unaffected — by luck, not design.

⚠ **This retro-taints an unknown set of earlier measurements.** The 2026-08-03
session already recorded two runs where the NER and the background model
silently swapped; the `.env`/threshold arm was not known then and was not
checked for. Treat any pre-2026-08-08 number whose producing command's working
directory is not recorded as **suspect on this axis**.

### G5 — SSE damage rates on a healthy stream

Driven live through the proxy (`uvicorn src.api.main:app`) against Ollama
`qwen3:4b-instruct`, two single-turn conversations.

- Healthy 14-line stream, **first** taxonomy: `parsed=13, dropped=1`. The one
  "dropped" line was the **terminal usage chunk** (`"choices": []` → IndexError)
  — i.e. the new WARNING would have fired on **100% of turns**. Recorded because
  it is the measurement that changed the design.
- Same stream, **shipped** taxonomy: `dropped=0, salvaged=0, no_content=2`
  (usage chunk + `finish_reason` chunk). Zero damage warnings on two consecutive
  live turns; `raw_text` stored verbatim and correct both times.
- The splice defect was **not** exercised live (it needs a primary-model
  timeout); it is pinned by a two-sided unit assertion instead, which checks
  that the *old* flat join really did swallow the fallback's first line.

### G25 — what actually reaches `logs/`

Method: one real turn through the proxy against a live Ollama with an **empty
store**, then grep the resulting structlog output for the prompt and the
response text.

- **0** occurrences of the user's prompt text; **0** of the assistant's answer.
- **18** distinct structlog events emitted (`classified`, `memory_decision`,
  `context_ledger`, `prompt_measured`, `turn_stored`, `token_prediction_reconciled`,
  `context_window_truth`, `cluster_assignment_complete`, …) — all metadata.
- `logs/` is gitignored (`.gitignore:51`) and `git ls-files logs/` is **empty**:
  no log file has ever been committed, so this is not a public-repo exposure.
- Static sweep of every `log*.{info,warning,error,debug}` call in `src/` found
  the hot path logs counts by design (`query_words=len(...)`, `words=len(...)`,
  and `context_ledger` storing token counts, never text). Content-bearing lines
  that remain are derived and short: `pattern_text[:50]`, `cluster.name`,
  `canonical_name`, `title[:80]`, plus `error=str(exc)` as an indirect channel.

⚠ **Explicitly PARTIAL — do not cite this as a clean bill of health.** One turn,
empty store, so the background workers (which own every content-bearing line
listed above) were barely exercised. The number that matters — what a populated
store logs over a real session — is unmeasured. Re-run under Z2's conditions.

### G34 — the relation detector, measured (2026-08-08)

Read-only probe against the live vocabulary and the shared embedder
(`Qwen/Qwen3-Embedding-0.6B`, 1024 dims, GPU), run from the repo root. No DB
writes. Recorded because G34's design decision rests on these numbers and the
"raise the floor" fix looks obvious until you see them.

**Vocabulary:** 197 relations, of which **110 are single words** (those match on
any one-word hit in channel 1). `RELATION_SIM_FLOOR` 0.45, `RELATION_TOP_K` 5.

**Channel 2 — relations above the floor, by prompt:**

| prompt | above floor | top-1 |
|---|---|---|
| `"ok"` | 197 / 197 | `ally` 0.844 |
| `"hello"` | 195 / 197 | `ally` 0.736 |
| `"thanks, that helped"` | 189 / 197 | `complements` 0.625 |
| `"what does Kael own"` | 154 / 197 | `owned_by` 0.626 |
| `"what is 2 + 2"` | 143 / 197 | `complements` 0.600 |
| `"what is Kael using for the ritual"` | 123 / 197 | `wields` 0.586 |
| `"what does Kael use for the ritual"` | 109 / 197 | `wields` 0.585 |
| `"who is Rika married to"` | 66 / 197 | `married_to` 0.667 |
| `"who inspired Kael"` | 43 / 197 | `is_founder_of` 0.575 |
| `"write me a haiku about rain"` | **0 / 197** | `foreshadows` 0.409 |

The ordering is the finding: **absolute cosine is anti-correlated with
relational content** across these samples. Short contentless strings embed near
the centroid and are close to everything, so no absolute threshold can separate
`"ok"` (0.844) from `"who inspired Kael"` (0.575). The one clean negative is a
long, semantically specific non-relational prompt.

**Channel 1 — the stemmer's symmetry claim is false.** `_stem` only strips a
suffix when `len(w) > 4`, so short vocabulary words are never stemmed:

| vocabulary | prompt | → | → | meet? |
|---|---|---|---|---|
| `uses` | using | `uses` | `us` | NO |
| `uses` | use | `uses` | `use` | NO |
| `owns` | owning | `owns` | `own` | NO |
| `has` | have | `has` | `have` | NO |
| `inspired` | inspiring | `inspir` | `inspir` | yes |

`uses` and `owns` are both live vocabulary entries. Confirmed end-to-end:
*"what does Kael use for the ritual"* and *"what is Kael using for the ritual"*
produce **zero** channel-1 hits; *"what does Kael own"* finds `owned_by` and
misses `owns` — one concept, two entries, reachability decided by word length.

**Latency.** `_detect_relations` median **12.2 ms** (min 11.8, max 13.6, n=10)
on the synchronous pre-flight path. The cosine loop is pure Python: 197 × 1024
≈ 202k multiply-adds per turn. ⚠ Four other pure-Python dot-product loops exist
(`orchestrator.py:1193` entity resolution — **unbounded, loops every entity in
the graph**; `clustering.py:334`, `:348`, `:674` — background).
`retrieval/coverage.py` already uses `np.dot`.

---

## 2026-08-08 — Experiment 3's ablation ladder: two arms were the same configuration

Recorded here because it changes what a **published number means**, and the run
that produced it cannot be re-read to find this — the defect is in the harness,
not the data.

**What was measured.** `ConfigurableOrchestrator._apply_bonuses` implemented the
`recency_boost: False` arm by importing `BONUS_RECENT_TOP_10PCT` /
`BONUS_RECENT_TOP_30PCT` from `orchestrator.py` and rebinding them with
`global`. That writes the *subclass module's* copies; the scoring code is the
parent's `_apply_bonuses`, reading the *parent module's* copies. Verified
directly at the interpreter: after the ablation zeroes its own globals, the
subclass sees `0.0` and the parent still sees `1.0`.

**Consequence, in the buildup ladder** (`experiments/flaw_ablation/buildup/`):
the branch only runs when keyword is on and recency is off, which is exactly one
arm — `add_keyword_boost`. So that arm ran with recency bonuses **on**, making it
identical to `full_ice`. The recorded results agree: 27,769 vs 27,768 tokens and
15.1 vs 15.0 fragments, i.e. one configuration run twice.

⇒ **`add_keyword_boost`'s +0.12 step is keyword AND recency combined**, and
**`full_ice`'s −0.06 step is run-to-run noise, not the recency boost's
contribution.** Both lie inside CIs the paper already reports as spanning zero,
so no qualitative finding changes; the per-step attribution does.

A second defect in the same file meant the harness **could not run at all** on
current `main`: `_batch_summary_lookup` never gained the `include_cross`
parameter the parent took on with C6, and `retrieve()` passes it by keyword ⇒
`TypeError` at that leg.

**Disposition (user decision, 2026-08-08): record now, correct at FINAL.**
`experiments/` is a frozen record and was not touched; `ICE_paper_v2.tex` was not
edited. FINAL re-runs the ladder on the fixed flag and replaces the table. Both
defects are fixed on `main` (`73169a3`) and guarded by
`tests/smoke/test_ablation_flags.py`, whose signature check reproduces the
`include_cross` drift when it is re-introduced.

**No other artifact this session.** G9 produced no corpus, checkpoint or
experiment result — its equivalence runs (1,356 leg-weight combinations, 51,040
recent-fraction, 3,600 growth-cap, all zero divergence except 180 unreachable
`Null_Noise`-as-intent cases) are validation evidence, recorded in the roadmap
entry rather than here.

---

## 2026-08-09 — G36: a third ablation flag that toggled nothing, and Experiment 1's HyDE arm

**Not an artifact-producing session** — no corpus, checkpoint or experiment
result. Recorded here because it **falsifies a published ablation row**, which
is the same trigger the G19 entry above was written under.

**The finding.** `_hyde_rewrite` was unreachable by every path, and had been
since before the `v2-paper-eval` tag. Verified three ways: the call site is
commented out in the tagged file (`git show v2-paper-eval:src/retrieval/orchestrator.py`,
lines 313–322); nothing anywhere sets `_force_hyde`; and
`ConfigurableOrchestrator` never calls `_on("hyde")` and does not override
`retrieve()`, so its `hyde` flag reached no code.

**What that means for Experiment 1.** `phase2_run_evaluation_matrix.py`
(`RUN_HYDE_ABLATION = True`) built its `full_ice_no_hyde` arm by monkeypatching
`_hyde_rewrite` to return `None` — a method nothing invoked. **`full_ice_no_hyde`
and `full_ice` were therefore the same configuration.**
`experiments/unmature/generate_paper_summary.py:105` reports *"HyDE has minimal
impact (possibly because the background model is weak)"*; the parenthesised
cause is wrong. The true statement is that the arm never exercised HyDE. This
is the **third** instance of the shadow-subclass drift G19 warns about, after
`recency_boost` (Exp 3's `add_keyword_boost` ≡ `full_ice`) and the
`include_cross` signature break.

⚠ That runner cannot be executed today in any case — it imports
`src.workers.sentinel_monitor`, deleted by D1/D2.

**Disposition — identical to the G19 decision above: record now, correct at
FINAL.** `experiments/` was not touched and `ICE_paper_v2.tex` was not edited.
The method, the flag, the `_hyde_used` telemetry and `self.bg_client` are
deleted on `main` (`4e3b43d`), with an in-place comment at each deletion site
recording why real HyDE was rejected (roadmap P0.1) so it is not rebuilt from
the history.

**Also measured, and load-bearing for how the rest of G36 was judged.** A
`sys.settrace` probe counted every exception propagating through an
`orchestrator.py` frame while fifteen seeded suites ran (476 checks):
**zero swallows**, generator-teardown excluded. That is the baseline behind the
decision to log on *every* swallow rather than first-occurrence-only — a
warning that never fires in normal operation is signal, per TRAPS #13b. Re-run
the probe after any change to the legs; the method is a trace function keyed on
`frame.f_code.co_filename`.

**Store state at session end:** 1 row — `conversations`, the deterministic
`ice://mcp-notes` shell (`ab934e9c-…`), which is production state, not residue.
Three genuine orphan row-sets were removed; see CLEANUP.md and TRAPS #15/#16.

---

## 2026-08-11 — Z1 run configuration: two local overrides that no commit records

**Both live in `.env`, which is gitignored.** Recorded here because a run has to
be able to state what produced it, and neither of these is visible in the tree.

| Key | Value | Why |
|---|---|---|
| `BACKGROUND_MODEL_NAME` | `gemma4:26b-a4b-it-q4_K_M` | G27 option (b) — the pin |
| `CODEX_RELATION_DETECTION_ENABLED` | `false` | G34 — neutralises A4 for tuning |

### The pin changes nothing today, which is the point

Unset, `get_bg_model_name()` falls through to `get_fallback_model()`, which
returns **the first entry in the registry JSON carrying `confirmed: true`** —
insertion order, no scoring, no relation to the model serving the chat. Resolved
live before and after the pin:

```
get_bg_model_name() -> gemma4:26b-a4b-it-q4_K_M     (both)
```

So the pin is not a behaviour change; it removes a **silent** one. Regenerating
the registry, or confirming a new model that lands earlier in the file, would
have swapped the background model with nothing in any log or artifact to say so.
This is the model the 2026-08-10 retrieval ground-truth key was derived with, so
the key and everything scored against it now name the same thing.

Registry order at the time of the pin, first to last: `gemma4:26b-a4b-it-q4_K_M`,
`qwen3-coder:30b-a3b-q4_K_M`, `qwen3.6:27b`, `gemma4:12b`, `tinyllama:latest`,
`qwen2.5:7b` — **all six `confirmed: true`**, so the fallback is decided purely
by which one is written first.

### The relation override is the real off, not the boost

The plan was `CODEX_RELATION_OVERLAP_BOOST=0.0`. Measured and abandoned: that
setting is read at **one** site and leaves three effects of a detected relation
running — fact lines in the fragment text, fact edges into
`_reinforce_codex_edges` (which commits), and the `_codex_enumeration` gate. See
ROADMAP G34. `CODEX_RELATION_DETECTION_ENABLED=false` returns from
`_detect_relations` before any work, which is inert at all four sites.

⚠ **Both keys are run configuration, not decisions about the product.** The
declared defaults in `config.py` are unchanged (`background_model_name=None`,
`codex_relation_detection_enabled=True`), and the relation override carries a
revert marker in `.env`. `tests/test_settings_freeze.py` was changed the same
day to compare declarations rather than resolved values, so overrides like these
no longer turn that suite red — which means **this file is now the only record
that they are set.**

---

## 2026-08-11 — G34: three candidate detectors measured, and none of them works

Read-only, no DB writes, no LLM. `scripts/oneoff/g34_ablation.py`, against the
live 197-relation vocabulary and the shared `Qwen/Qwen3-Embedding-0.6B`. Run
before designing anything, because the obvious fixes look convincing until they
are scored.

**Candidates.** `base` = today's absolute cosine ≥ floor. `B` = centre prompt and
glosses on the vocabulary centroid first (embedding anisotropy is the suspected
cause: everything shares a large common component). `A` = z-score within the
prompt's *own* similarity distribution, so the question becomes "does this
relation stand out?" rather than "is it close?". `A+B` = both.

**Each design is swept over its own threshold grid.** Comparing them at one
number would be invalid — centring changes the scale of the similarities, and
judging `B` at `base`'s 0.45 silences it completely (0% firing) while making it
look perfectly style-invariant, which is the vacuous stability of a function
that always returns nothing.

**Best operating point per design** — `neg` = mean relations fired across the
five negative probes (want 0); `pos` = positives whose expected relation was
returned, of 5; `rate` = % of 120 real corpus turns firing anything.

| design | threshold | neg | pos | rate | verdict |
|---|---|---|---|---|---|
| `base` | 0.45 (live) | 4.00 | 4/5 | 82.5% | no operating point separates: `neg`→0 only at 0.85, where `pos`=0/5 |
| `B` | 0.30–0.40 | 0.00–0.20 | **1/5** | 2.5–18% | silences the noise *and* the signal; never exceeds 1/5 at any threshold |
| `A` | 2.5 | 1.00 | 3/5 | 83.3% | best of the four, and still not separation |
| `A` | 3.0 | 0.40 | 2/5 | 54.2% | |
| `A+B` | any | — | ≤2/5 | — | **strictly dominated by `A` alone** |

**Three findings, and the third is the one that matters.**

1. **Combining the candidates is worse than either alone.** `A+B` is dominated by
   `A` everywhere on the grid. The instinct to stack fixes is wrong here.
2. **`base`'s 4/5 is flattering and should not be read as recall.** It returns 5
   relations on essentially every prompt, so "the expected relation is among 5 of
   197" is nearly free. `A` at 2.5 returns fewer *and* still reaches 3/5.
3. **No design achieves separation.** On this evidence the embedding channel
   cannot answer "is this prompt relational?" at all. The candidates trade noise
   against recall along one curve; none breaks it.

**Style invariance (G28), same meaning varied form.** `base`, `A` and `A+B` all
**FLIP** — five distinct decisions across five phrasings of *"who inspired Kael"*.
`B` is stable only because it returns nothing. **No candidate passes.**

**⚠ A correction to the option set this run was built to test.** "Require a
resolved entity before trusting relations" was proposed as a separable fix. It is
**already the behaviour**: `_relation_facts` is reached only inside
`for anchor in matched:`, so relations are entity-gated on the fact path already
(the one exception is `_codex_enumeration`, which needs an explicit cue word).
The entity gate therefore cannot be the fix, and the real mechanism of harm is
narrower than "the detector fires on everything":

> Given a matched anchor, `_relation_facts` filters `CodexEdge.relation.in_(detected)`.
> With ~197 relations detected that filter is a **no-op**, so the leg returns the
> anchor's top-`codex_entity_edge_limit` edges by strength *regardless of the
> question*, and the `+0.25` overlap boost applies unconditionally. The
> "entity ∩ relation joint hit" that A4 documents as its precision anchor
> degenerates into "dump the anchor's edges".

So the requirement is not a better *score* — it is a relation set that is
**restrictive**, and the three candidates above fail to be restrictive without
also being empty. Design continues from here; nothing was changed in `src/`.

### G34 continued — inverting the question, and it works

Same script, same run. Design **D**: instead of asking "which of 197 abstract
glosses is this prompt near?", ask "given we are already rendering this anchor's
edges, **which of its own handful of relations** does the question point at?"

That is a different problem, and a much easier one. It is also what A4's docstring
has always claimed the leg does.

**Method.** The anchor's candidate relations are drawn at **random** from the
vocabulary with the true relation inserted — a hand-assembled neighbour set would
be a probe of the author's imagination (TRAPS #13) and would agree with whatever
design it was written for. 400 trials per row. Two baselines, because "better
than nothing" is not a result: `random` is chance, and `strength` is today's
behaviour (edges ordered by raw strength, i.e. the question ignored entirely).

**Top-1 accuracy of the expected relation within the anchor's set:**

| anchor edges | random | strength (today) | **D** | D + centred |
|---|---|---|---|---|
| 3 | 32.0% | 33.0% | **97.5%** | 95.5% |
| 5 | 23.0% | 19.0% | **94.2%** | 86.5% |
| 10 | 11.0% | 8.0% | **93.2%** | 83.0% |
| 20 | 3.0% | 4.8% | **83.5%** | 62.8% |

**Why this succeeds where the gate designs failed.** It never has to say "no".
The unanswerable question was *"is this prompt relational?"*; ordering an anchor's
existing edges is a strictly weaker claim that does not require one. `codex_max_fanout`
([G35](ROADMAP.md#g35), landed the same day) already bounds *how many* edges are
rendered — D decides *which*, and the two compose.

**Centring hurts here too** (83.0% vs 93.2% at 10 edges). That is the third
independent measurement in this run saying the same thing: stacking these
corrections makes things worse, not better.

**⚠ Caveats, both real.**
- **Style invariance is improved but NOT solved.** Across five phrasings of
  *"who inspired Kael"* the pick was `['co_authors', 'inspired_by', 'inspired_by',
  'inspired_by', 'knows']` — 3 distinct, with the typo variant landing on `knows`.
  Better than the gate designs (5 distinct) and not good enough to call invariant.
  This is a **single random candidate set** and therefore noisy; it must be
  re-measured over many seeds before any number is quoted.
- **Simulated anchors, not a real store.** The candidate sets are sampled, not
  read from `codex_edges`, because the store is empty. Re-run against Z1's
  populated graph before trusting the absolute numbers; the *ordering* of D
  against its two baselines is the durable part.

### G34 — the style-invariance gate, threshold agreed before measuring

The spec makes this **blocking**, and the acceptable rate was fixed with the user
**before** the run so it could not be chosen to fit the result: **≤ 10% flips.**

Meaning held fixed, form varied — ± question mark, "ok so"/"like" prefixes,
lowercase, typos, terse vs rambling, and a possessive rephrasing. Seven phrasings
per group, three groups, **60 random candidate sets each** (10-edge anchors,
relations drawn at random with the true one inserted). A "flip" is any phrasing
disagreeing with its group's majority pick.

| group | flip rate | top-1 correct |
|---|---|---|
| `inspired_by` | 8.3% | 91.7% |
| `married_to` | 0.0% | 100.0% |
| `owns` | 6.4% | 88.8% |
| **overall** | **4.9%** | **93.5%** |

**PASS** (4.9% against ≤10%).

⚠ **This supersedes the single-seed caveat recorded above**, which suggested ~3
distinct picks in 5 phrasings (~40%). That sample was one random candidate set and
was explicitly flagged as noise rather than a result — correctly, as it turns out.
The lesson is the cheaper half of TRAPS #13: a single fixture is not a measurement,
whichever direction it points.

**Still owed:** these anchors are simulated, because the store is empty. Re-run
against Z1's populated graph and record the numbers; the *ordering* against the
two baselines is the durable claim, the absolute rates are provisional.

## A12 — the eight-arm background-model comparison, 2026-08-12

**Setup.** Each candidate seeded the SAME 60 turns (20 from each of the three Z1
corpus conversations) through the real pipeline — `post_flight.evaluate_turn`
(density, grounded summary, chunking, codex, procedural) plus clustering and
batch summaries — so every background job was exercised, not the two with
text-level entry points. Store measured per arm by `scripts/z1/store_report.py`;
output sampled by `scripts/z1/sample_bg_output.py` and judged by a subagent
reading all five sections of all seven viable arms (nemotron-mini excluded on
measured grounds).

**Measured (60 turns each).**

| arm | entities | edges | patterns | summary coverage | junk % |
|---|---|---|---|---|---|
| gemma4:26b-a4b (incumbent) | 1142 | 1243 | 57 | 0.979 | 2 |
| gemma4:e4b | 761 | 554 | 49 | 0.976 | 8 |
| qwen3.5:4b | 233 | 190 | **0** | 0.970 | 1 |
| qwen3:4b-instruct | 353 | 281 | 57 | 0.956 | 3 |
| ministral-3:8b | 1611 | 1485 | 59 | 0.918 | 1 |
| granite4:micro | 252 | 176 | 25 | 0.913 | 6 |
| granite4:tiny-h | 250 | 230 | 50 | 0.895 | 7 |
| nemotron-mini:4b | 23 | 15 | 48 | 0.607 | 30 |

Every arm summarised exactly 50 of 60 turns — the density gate picks which turns
earn a summary, not the model, so coverage differences are about quality alone.

**⚑ THE EYEBALL OVERTURNED THE METRIC RANKING, and the reason is the metric's
stated blind spot.** `granite4:tiny-h` scored coverage **0.9375–1.0** — mid-pack
— while **12 of its 12 summaries were bare `Key terms: [list]` dumps with no
synthesis at all**. `summary_coverage` measures term PRESENCE, so a model can
score ~1.0 while producing no summary whatsoever. That caveat was written into
`store_report.py` before the run; this is it at the extreme.

**Ranking after reading (agent, all five sections):** gemma4:e4b > qwen3:4b-instruct
> gemma4:26b > ministral-3:8b > qwen3.5:4b > granite4:micro > granite4:tiny-h.

- **The 26B is NOT the winner** — "more generic-subject-heavy than gemma4_e4b for
  no quality gain, at nearly 2× the VRAM". ⇒ **A12's premise holds: the
  background pipeline does not need a 26B.**
- **Practical pick: `qwen3:4b-instruct`** — tied with e4b on summary quality at
  **2.5 GB against 9.6 GB**.
- `qwen3.5:4b` is bimodal: 5/12 summaries bare, **zero** procedural patterns, yet
  the best vocabulary-gap signal in the set.
- `granite4:tiny-h` fabricates *specific* unsupported detail (`kael --role-->
  fire mage`, a `goo blade`), contradicting its own output, with zero
  corroboration across 60 turns × 7 models.

**⚑ TWO DEFECTS ARE ICE'S, NOT THE MODELS' — present in ALL SEVEN arms.**

1. **Codex subjects are unusable.** Generic non-entities (`universe`, `story`,
   `music`, `post-study pathway`) and unresolved pronouns (`i --possesses-->
   diverse skills`) — a graph node named "i" cannot be looked up. Universal ⇒
   prompt/design, not model capability. Consistent with the store's shape: 65%
   of entities are degree-1 standalone triplets.
2. **Procedural patterns are fabricated by construction.** Every pattern from
   every model is a SINGLE-turn restatement wrapped in "the user consistently…",
   evidenced by exactly one occurrence. `extract_procedural` asks one turn to
   reveal a *recurring* habit, which it definitionally cannot. qwen3.5:4b's zero
   output is arguably the honest answer.

**Vocabulary harvest (feeds [Z2](ROADMAP.md#z2)).** 14,091 relations dropped
across the arms; **3,244 distinct candidates** after junk removal and
normalisation. Top gap: **`main --is--> X` dropped 875×, found independently by
TWO models** — "X is the primary/main choice" has no slot at all. Also
`global cs rank --has--> cmu` and `b1 german --has--> likely pr` at 238× each.
Candidates and their proposed opposites: `scripts/z1/harvest_vocabulary.py`.

**Real-world errors worth recording:** `granite4:micro` read a recurring **9.65
CGPA as "$9.65 per hour"**; `qwen3.5:4b` placed **CMU in Seattle** (Pittsburgh);
`gemma4:26b` attributed a CSI-Club fact to a turn that never mentions it.

**The output defects the read found, in full** (2026-08-12; the four not already
covered by [G42](ROADMAP.md#g42)–[G44](ROADMAP.md#g44) are recorded here because
they decide model selection and would otherwise survive only in a chat log):

1. **Bare key-terms degeneration** — the "summary" is a term list plus one
   generic sentence. `granite4:tiny-h` **12/12**, `qwen3.5:4b` **5/12**,
   `granite4:micro` 1/12. Coverage scores it 0.9375–1.0. (TRAPS #21.)
2. **Echoing the user's own ALL-CAPS emphasis as content.** `qwen3.5:4b` and
   `granite4:micro` did this on the *same* source turn — key terms came back as
   `SO, REGENRATE, THE, ENTIRE, CORNICLE…`, the user's typos preserved. Two
   models failing identically on one turn ⇒ an instruction/content boundary
   failure, not noise.
3. **Generation repetition loops.** `qwen3.5:4b`: *"across regions including
   India, Europe, SE Asia, Australia, Asia, India, Europe, SE Asia, USA…"*
4. **Role-label leakage / verbatim turn reproduction.** `granite4:micro`'s
   summaries open with literal `User:` / `Assistant:` and reproduce the source
   turn instead of paraphrasing it.
5. Unresolved pronoun subjects · 6. generic non-entity subjects → [G44](ROADMAP.md#g44).
7. Whole-clause-as-object contract violations → [G45](ROADMAP.md#g45).
8. Confident fabrication · 9. misattribution → [G43](ROADMAP.md#g43).
10. "Seen 1×" claimed as consistent → [G42](ROADMAP.md#g42).
11. **Real-world factual errors independent of the corpus** — `qwen3.5:4b` put
    **CMU in Seattle** (Pittsburgh); `granite4:micro` read a recurring **9.65
    CGPA as "$9.65 per hour"**.
12. **Raw formatting/encoding leakage** — `qwen3.5:4b` emitted unescaped UTF-8
    byte sequences (`<0xE2><0x80><0xAF>`); `ministral-3:8b` wrapped every cluster
    name in quadrupled markdown asterisks.

**⚑ PER-SECTION WINNERS DIVERGE, and that is the more useful finding than the
aggregate.** Best summariser `gemma4:e4b`; best vocabulary-gap signal
`qwen3.5:4b` — which is also the worst summariser and produced zero procedural
patterns. **No model is good at codex extraction and none produces a valid
procedural pattern**, which is why those two are [G43](ROADMAP.md#g43)/[G44](ROADMAP.md#g44)
and [G42](ROADMAP.md#g42) rather than model-selection criteria. If the pipeline
ever splits these into separate model calls, `qwen3.5:4b` is a poor default and
a genuinely useful second opinion for Z2's vocabulary repair.

---

## 2026-08-13 — the instrument was the finding: 0.250 → 0.508, and five pipeline fixes

**Everything below is on the restored 293-turn `gemma4:26b-a4b-it-q4_K_M` store
(3 conversations: `ecc64aab` 145, `355a5709` 87, `cca73c87` 61), alembic head
`505f12031434`, tree dirty throughout — no commit describes what ran.**

### 1. The retrieval numbers were about the harness

| | legacy scorer | production path |
|---|---|---|
| recall@1 | 0.076 | **0.135** |
| recall@10 | 0.250 | **0.508** |
| MRR | 0.108 | **0.209** |
| fragments returned (median) | 6 | **14** (8–46) |
| retrieval budget (median) | 5,000 | **10,700** (8,100–11,350) |
| zero-fragment probes | **242 / 592** | **0** |

**The control is what makes this attributable:** a legacy-mode run reproduced the
recorded numbers exactly — `recall@1 = 0.07601351351351351` to the last digit,
`recall@10 = 0.250`, MRR 0.1083 vs 0.1087 — so the delta comes from four
instrument fixes and nothing else. **ICE did not change.**

Four defects (G46): the scorer never called `set_budget_from_turn_count`; the
probe→row map came from a file the arm runs had overwritten; the classifier's
raw `Zero_Shot` reached a guard `main.py` makes unreachable; leg attribution was
first-leg-wins. Corrected attribution: of 377 hits, **`bm25+vector` produced 376
and `vector` 1** — see G48.

⚠ *"15 of 592 zero-fragment probes"* was itself an artifact: the old scorer wrote
`misses[:40]`, so 15 was the truncation. True legacy figure **242 (41%)**.

### 2. Noise floor

* **Determinism:** `access_count` is incremented on every retrieval and read by
  nothing; with it on, two identical runs agreed on **26/40** result sets, with
  it off **40/40**. Now gated by `retrieval_strengthen_writes`.
* Residual: pass 1 differs from passes 2–3 on 2/60 fragment counts (no rank
  changes) — process warm-up, cause not isolated, `_relation_gloss_cache` the
  prime suspect. Mitigation: one throwaway retrieval before scoring.
* **Paired MDE (592 probes): 0.037 absolute (~22 probes).** The instrument fix
  itself measured **+0.258 [+0.221, +0.296]**.
* Marginal 95% CI on recall@10: **[0.470, 0.549]**.

### 3. The probe set is 64% contaminated

The generator's ambiguity guard scored `question + answer` while retrieval sees
only the question. Replayed over the same 592 probes: the old rule rejects **5
(0.8%)**, the fixed rule **385 (65.0%)**. **380 probes (64%) are in the set only
because the guard scored words retrieval never sees**, and the 0.508 inherits it.

### 4. Pipeline defects, measured then fixed

* **G42:** all **247** stored patterns evidenced by exactly ONE turn; only 2 ever
  reached `reinforcement_count >= 3`. Session-scoped extraction now yields e.g.
  *"Restates or asks to restate the plan before proceeding"* citing messages
  [1,2,3,4,5]. ⚠ Two extractions of the SAME habit score **0.708** against a 0.85
  reinforcement threshold — promotion remains effectively dead; threshold now a
  swept setting, deliberately unchanged on one observation.
* **G43:** 51 property relations (30% of all edges) never had their OBJECT
  checked. `november --eye_color--> golden black`, `krishna --role--> god of
  love`. Verbatim source-text check marks **30 edges (0.72%)**.
* **G44:** 155/3,671 entities ≤3 chars, incl. `8`, `3`, `6`, `2`, `d` typed
  **person**. Refused at write; promotion folds zero-edge stubs only.
* **G45:** **1,259** relations destroyed in one seed (`i --didnt_get--> csi`,
  `my father --t_get--> first` from an apostrophe split). Vocabulary opened;
  111 multi-arm candidates seeded as `data/relation_seed.json`.
* **Degradation chain DOES fire:** 104 degradations / 60 probes, median **1,056**
  tokens saved, 113,921 total; 88.5% of returned fragments still raw and **1,462
  dropped while carrying an unused summary** — that is the headroom.
* **Corpus defect:** the seeder spread 293 turns evenly over 120 days, so
  `resolve_session_id` produced **293 sessions of one turn**. Sitting-shaped
  timestamps now give 6/9/4 sessions of 11–20 turns.

### 5. Relation-threshold calibration (30 pairs, live encoder)

Converses are inseparable by similarity: `before`/`after` **0.8791**,
`parent_of`/`child_of` 0.8569, `teaches`/`learns_from` 0.8159 — all above the
0.86 threshold then in force. With converses guarded deterministically the
highest unsafe pair falls to **0.787**, so the threshold moved **0.86 → 0.82**:
0 wrong merges either way, 11/15 true merges instead of 9/15. Classes still
overlap (one true synonym scores 0.645); the guards, not the threshold, do the
load-bearing work.

### 6. Typed probes

**420 after three runs** (`deepseek-v4-flash` via OpenCode Go): episodic 232 ·
**codex 89** · procedural 40 · summary 40 · temporal 19. 344 episodic candidates
rejected by the fixed ambiguity guard.

⚠ **The first run produced only 12 codex probes, and the cause was the harness
again.** The codex prompt demanded a verbatim quote from EACH of up to four host
turns; that text was the bulk of every reply and truncated the JSON mid-object,
whereupon the whole generation was discarded. Replacing the quotes with turn
numbers took codex from **12 → 89 on the same corpus and the same model** — the
material was never the limitation. (A partial-object salvage was added at the
same time and never fired: 0 events. Shortening the reply was the entire fix,
which is worth knowing before anyone attributes the gain to the salvage.) Every artifact from here carries a `run_meta` provenance block
(commit, dirty state, resolved settings, corpus digests, seeds).

## 2026-08-15 — the two-arm re-seed on the fixed pipeline

**What was run.** `scripts/z1/two_arm_seed.sh`, 293 turns per arm, whole corpus
(3 conversations: 87 / 61 / 145 turns), `CODEX_NODE_PROMOTION=true`, post-flight
chain complete (density, grounded summary, chunking, codex, procedural) plus
catch-up clustering and batch summaries.

| arm | model | wall clock | entities | edges | procedural | chunks | batch summaries |
|---|---|---|---|---|---|---|---|
| 1 | `qwen3:4b-instruct` (2.5 GB) | 85 min | 8,280 | 9,662 | 61 | 895 | 0 |
| 2 | `gemma4:e4b` (9.6 GB) | 108 min | 7,949 | 7,052 | 58 | 895 | 3 |

**293/293 post-flight per arm, 0 failures, 0 classify failures.** Snapshots:
`snapshots/fixed-qwen3-4b-instruct.sql` (132 MB), `fixed-gemma4-e4b.sql` (127 MB).
The store was TRUNCATED before arm 1 — the previous "empty" store still held 59
entities from the pre-fix pipeline (`hackathon team` typed *person*, whole
sentences as node names) that `seed_store.py --clean` preserves by design, plus
28 from test suites and 3,692 stale idempotency keys. Pre-truncate state saved as
`snapshots/pre-truncate-2026-08-15.sql`. **Anything measured on top of that
residue would have scored the OLD pipeline's junk against the new fixes.**

### What the fixes did (measured on arm 2 unless stated)

* **[G42](ROADMAP.md#g42) procedural — WORKS.** All 58 patterns cite **≥3 turns**
  (`source_batch_ids` 3–30), against the pre-fix **247/247 invented from ONE turn**.
* **[G43](ROADMAP.md#g43) property grounding — WORKS, no leak.** 463 property
  edges: **0 ungrounded values carry full confidence.** All 21 that fail
  `_value_in_source` sit at `codex_conf_rejected` 0.35; 331 grounded at 0.9; 111
  grounded but demoted (the subject, not the value, failed). ⚠ A raw SQL
  substring check reports 42 "ungrounded" — half of that is the check, not the
  store. Use `_normalize_term`, and cross it against `extraction_confidence`.
  The `november --eye_color--> golden black` class is **absent entirely**.
* **[G44](ROADMAP.md#g44) junk nodes — WORKS.** `codex_entity_name_unusable`
  fired **2,273 times** across the run; **0** nodes named `8`/`3`/`i`/`the`, 0
  non-alphabetic. ⚠ Residual: **5 single-LETTER nodes** (`d`, `z`, `e`, `f`, `g`,
  1–4 edges each) pass because the rule is functional, not a length rule — which
  is deliberate (`ai`, `ml`, `q4` must survive). 5 of 7,949 = 0.06%.
* **[G45](ROADMAP.md#g45)/[G49](ROADMAP.md#g49) relations — CANONICALISATION IS
  NOT BINDING.** **1,801 distinct relations** against G49's ~1,000 alarm line;
  **1,029 used exactly once (57%)**. Tense variants never meet: `has` 491 ·
  `have` 78 · `had` 60. `codex_relation_out_of_vocabulary` fired **once** — the
  vocabulary is open, and nothing is converging it.
* **Bi-temporal — written.** `learned_at` on all 7,052 edges, `unlearned_at` on
  1,350. Whether anything READS them is G49's second prediction, untested here.
* **[G49](ROADMAP.md#g49) prediction 1 CONFIRMED — procedural cannot activate.**
  0 of 58 active; 56 at `reinforcement_count` 1, 2 at 2, threshold 3; every
  pattern has `first_observed == last_observed`. **The procedural leg will score
  zero and it is NOT retrieval.**

### Node promotion does nothing on this corpus, and it nearly cost the run

`codex_node_promoted` fired **0 times in 586 turns**. `codex_node_promotion_skipped_in_flight`
fired **98 times** — every single candidate was an endpoint of the triplet being
written. Before the same-day fix each of those was a `ForeignKeyViolation` that
cost its turn the codex **and** procedural extraction (`evaluate_turn` re-raises),
while the run still completed and printed a total. Measured pre-fix: **1 turn in
6**; post-fix 0 in 6 with the guard firing 3 times, and 0 in 586 on the full run.
⇒ **G44's repair half is inert here.** Do not tune it; see [TRAPS #31](TRAPS.md).

### Also observed

`bg_model_output_truncated` **154 times** across the run (the salvage path from
2026-08-13 is what keeps those generations usable). Arm 1 produced **0** batch
summaries against arm 2's 3 — unexplained, and worth one query before any
summary-leg number is trusted.

## 2026-08-16 — the answer half, and a metric that could only score one leg

### Z2-mini: retrieve → assemble → ANSWER (the half no Z1 number measures)

`scripts/z1/answer_probes.py`, built this session. 150 typed probes stratified
30 per type (30 is the smallest count that resolves a per-type difference
against the ~22-probe paired MDE), the production chain end to end, then the
**real** `assemble_prompt`, then a generated answer. Answering model pinned to
`gemma4:26b-a4b-it-q4_K_M` for both arms — **neither arm under test**, so it
cannot favour its own summaries.
⚠ **CORRECTION 2026-08-22: this entry called it "A12's top-ranked model" and
that was wrong twice over.** A12 ranked it **THIRD** — see `:1485-1488` in this
same file, *"The 26B is NOT the winner"*, behind `gemma4:e4b` and
`qwen3:4b-instruct` — and A12 ranked models for **background extraction**, not
for answering, so its ranking does not transfer to this job at all. The
independence argument above is the real and sufficient justification; the
appeal to A12 was decoration, and it propagated into `MODELS.md` where it read
as a measured endorsement. Arm 1 147/150 answered,
arm 2 149/150.

### Judging: paired, blind, and the taxonomy is the instrument

`scripts/z1/judge_answers.py`. Absolute 1–5 scoring drifts between batches, so
the judge sees the question, the gold turn as SOURCE, and two answers as A/B,
with the slot randomised per probe and arm identity never disclosed. Verdict
plus a reason from a fixed enum, of which **`both_failed` is load-bearing** — a
tie because both answers are good and a tie because both are useless are
opposite findings, and collapsing them hides the most important result.

**Final run (reasoning on, no ceiling): 65 usable verdicts of 150; 85 lost to a
cloud usage limit mid-run.**

| type | qwen3:4b | gemma4:e4b | tie |
|---|---|---|---|
| codex_multihop | 4 | 3 | 6 |
| episodic_lookup | 3 | 2 | 8 |
| procedural | 4 | 2 | 7 |
| summary_synthesis | 3 | 2 | 8 |
| temporal | 4 | 1 | 8 |
| **total** | **18** | **10** | **37** |

Probe order is round-robin across types, so the 65 that landed are balanced
rather than 65 of one type — the reason a limit-killed run still produced a
readable result. Partial state is written after **every** probe for the same
reason.

### ⚠ THE JUDGE WAS WRONG ONCE, AND THE FIX FOR ONE BUG CAUSED IT

`deepseek-v4-flash` returns `reasoning_content`. At `max_tokens` 300 it spent the
whole budget inside the hidden block and returned empty content **only on long
inputs** — a toy prompt worked perfectly, which is exactly what makes it read as
a broken judge rather than an exhausted one (TRAPS #11). Setting
`reasoning_effort="none"` fixed that and **caused a worse failure**: the judge
stopped deliberating and over-called `both_failed` on **30 of 73**, several at
100% content overlap with the expected answer. Reasoning restored with the
ceiling removed, `both_failed` fell from **49% to 31%**. ⇒ **A comparison task
needs the thinking; pay for it in budget, never by switching it off.**

### ⚑ RECALL COULD ONLY EVER CREDIT THE EPISODIC LEG

The largest measurement finding of the session, and it changes every number in
the 2026-08-15 entry above. Fragments carried no link to the turns they were
derived from, so on one harvest:

| leg | fragments returned | ever credited as gold |
|---|---|---|
| episodic | 4,124 | 249 |
| codex | 474 | **0** |
| procedural | 400 | **0** |
| timeline | 207 | **0** |

Not because those fragments were wrong — because nothing could attribute them
to a turn. **Every recall number ICE has produced is an episodic-leg score
reported under the whole system's name.**

Fixed by carrying `origin_batch_ids` onto derived fragments. Measured on 25
typed probes with **ICE unchanged and only the metric altered**: codex 0 → 18
gold credits, procedural 0 → 4, probes with a gold hit **15/25 → 21/25 (60% →
84%)**.

⚠ **Two identifier spaces, and the first two attempts at this fix produced a
false finding.** A fragment's `source_batch_id` is the episodic **row id**;
`codex_edges.source_batch` and `procedural_memory.source_batch_ids` hold the
turn's **batch id**. 9,662 edges join on `batch_id` and **0** on row id, so the
comparison could never match — which read as *"the graph does not cover the gold
turns"*, a conclusion about to be written down. Carry both ids.

### Graph shape, re-measured from both snapshots (confirms 2026-08-15)

| arm | degree-1 | degree 5+ |
|---|---|---|
| `qwen3:4b-instruct` | 5,762/8,280 = **69.6%** | 8.5% |
| `gemma4:e4b` | 6,401/7,949 = **80.5%** | 4.4% |

**Dead ends are the LEAST duplicated part of the graph** (nearest-neighbour
cosine, 400 sampled per bucket): degree-1 mean **0.8194** with 12.2% above 0.90,
against degree-5+ mean **0.8908** with 47.6%. ⇒ fan-out is an **extraction**
property, not a resolution failure, and no merge policy moves it. Near-duplicates
concentrate in the hubs — exactly where merging is most dangerous, because a hub
carries facts to re-attribute.

**Nor is it one bad relation.** Relations touching a dead end are flat: `has`
6.1% of dead ends, `description` 3.0%, `contains` 2.1%, with the top 18 covering
~21% across a ~1,800-relation tail. There is no concentrated pattern to catch.

### Procedural activation was gated on the one signal the system cannot produce

61 patterns citing **514 turns** between them, **one active**. Patterns citing 20
turns sat at `reinforcement_count` 1, because activation asked whether a later
session re-emitted a *similar sentence* three times — and two write-ups of the
same habit measure **0.708** against a 0.85 bar. Activation now also fires on
cited turns (`procedural_min_cited_turns`, default 10). Projected on the existing
arm: **29 of 61 would activate**, which is where config's own C9 note anticipated
the calibration landing. ⚠ 10 is a defensible default, not a calibration.

### Relation canonicalisation converged to several attractors

`known_relations()` is read **once per extract_codex call**, and a turn yields
**24.5 distinct relations on average (164 at worst)** — so same-turn variants
were structurally blind to each other. That is why `have` merges into `has` on
demand at 0.9469 while the store still holds `has` 491 / `have` 78 / `had` 60.
Accepted relations are now fed back into the set the turn is matching against.

That fix exposed the threshold: at 0.82, feeding the set forward made things
**worse** — `has` merged into `contains` at 0.8608, and a wrong attractor then
captures everything after it. **Raised 0.82 → 0.90**: `have`→`has` 0.948 and
`had`→`has` 0.9255 still merge, `has`/`contains` (0.8848) no longer does,
`built`/`built_by` stays split. Cost, stated: `asks`/`asked` (0.8912) no longer
merge.

### An unknown relation defaulted to supersession and was destroying facts

`handle_triplet` superseded whenever `relation not in MULTI_VALUED_RELATIONS`,
so every relation the open vocabulary invented counted as single-valued and each
new object retired the previous one. Worked case: **eight genuine components of
one explanation, seven retired by the eighth** — silent, because the edges stay
in the table with `valid_until` set. Now supersedes only for relations *known*
single-valued. A missed supersession leaves a visible stale edge; a lost fact is
neither.

### Also measured, and NOT acted on

* **`PROPERTY_RELATIONS` needs no extension.** Only **50** edges use property
  relations against **9,602** open-vocabulary ones, and open-vocab facts render
  identically in the context payload. The list is a storage style, not a gate —
  so the claim that a closed property vocabulary was destroying facts (asserted
  repeatedly earlier in the session) is **withdrawn**.
* **Node promotion is inert:** 0 `entity_merged` events across the seeded arm.
* **Tier 0 auto-merge was structurally impossible** — it looked for a casefold
  collision on `canonical_name`, which is UNIQUE. Replaced with `merge_key()`
  (20 real groups). ⚠ Do **not** sort tokens: sorting adds 11 groups of which two
  are converses, the entity-side twin of TRAPS #26.
* **Entity consolidation is human-gated at ~5/run against 1,094 candidates** —
  correct for one user, impossible for a product. [G50](ROADMAP.md#g50).

---

## 2026-08-17 — the first full re-count under the fixed metric, and three of five types cannot see their subsystem

`scripts/z1/score_typed.py --tag g48b-full-recount`, 372 probes, ~9 min, local
and deterministic (no API, no Ollama). Store unchanged from 2026-08-16: arm 1
`fixed-qwen3-4b-instruct`, **293 turns / 8,280 entities / 9,662 edges / 61
procedural patterns / 0 batch summaries**. Artifact:
`experiments/curation_files/score_runs/20260817T110843_g48b-full-recount.json`
(probe file sha256 `dcfcd0d39e2d1c87`). Baseline for comparison is
`20260815T125535_post-reseed-qwen3-4b-instruct.json` — **same store, same code,
pre-[G48b](ROADMAP.md#g48) crediting rule**.

| type | n | 08-15 (old metric) | 08-17 (G48b metric) | comparable? |
|---|---|---|---|---|
| episodic_lookup | 232 | 0.591 | **0.767** | **yes** — identical probes |
| codex_multihop | 89 → 41 | 0.538 | 0.317 | **no** — different probe population |
| procedural | 40 | 1.000 | 1.000 | yes, and meaningless (below) |
| summary_synthesis | 40 | 0.324 | 0.324 | yes |
| temporal | 19 | 0.211 | 0.211 | yes |

### The one clean number, and what it is NOT

**episodic_lookup 0.591 → 0.767** is the isolated G48b effect: identical 232
probes against an identical store, with only the crediting rule changed so that
codex/procedural fragments derived from the gold turn can be attributed to it.
Gold rank found on **191/232**.

⚑ **This is the ruler being fixed, not ICE improving.** Nothing in `src/`
differs between the two runs. Quote it as "the first honest episodic-class
number on this store", never as a recall gain.

⚠ **The other three unchanged numbers moved by zero, not by a rounding
margin** — G48b's provenance fix bought them nothing, each for its own reason.

### Three of the five types cannot currently see their subsystem

* **`procedural` = 1.000 is a presence test over a pool of one.** 61 patterns,
  **60 `is_active=false`** (activation needs `reinforcement_count >= 3`; 59 sit
  at 1, one at 2, one at 4). The leg's entire candidate pool is the single
  `reinforcement_count=4` row at confidence 0.8, and it is returned to **every**
  query in the store. Verified directly against the leg's own SQL: the nonsense
  string `zzzq wumpus glorbnax fleeb` returns that same fragment at cosine
  0.4147, as do `how do I cook rice` and `explain gradient descent`. The metric
  asks *"did any procedural fragment come back at all"*, so it reads **1.000 for
  gibberish**. See TRAPS #37 — the same defect scored **0.000** on arm 2.
* **`codex_multihop` = 0.317 is mostly a damaged probe file.** Only **12 of 41**
  surviving codex probes still carry `anchor_entity`, the field the anchor half
  of the metric reads; for the other 29 it is absent, so `anchor_via_graph` is
  False **by construction** and the achievable ceiling is 0.5, not 1.0.

  | | n | anchor via graph | turn coverage | score | ceiling |
  |---|---|---|---|---|---|
  | anchored | 12 | **9/12 (75%)** | 0.500 | 0.625 | 1.000 |
  | unanchored | 29 | 0/29 *by construction* | 0.379 | 0.190 | 0.500 |
  | as reported | 41 | 9/41 | 0.415 | **0.317** | — |

  The 08-15 baseline had essentially all 89 probes anchored (`anchor_anywhere`
  **86/89**) at **61/89 = 68.5%** via graph. ⇒ **On probes the metric can score,
  the graph went 68.5% → 75%. It did not regress.** The 0.538 → 0.317 is the
  [TRAPS #35](TRAPS.md) probe destruction, and specifically a casualty not
  recorded at the time: the rebuild recovered `question` and `gold_turns` but
  **dropped `anchor_entity` for 29 of 41 records**.
* **`summary_synthesis` = 0.324 is pure turn-coverage.** `batch_summaries` holds
  **0 rows**, so `summary_frags` was 0 on all 40 probes and the metric's designed
  right answer — *a summary covering the span counts* — **cannot fire on this
  store**. The number is a coverage score wearing a synthesis score's name.

`temporal` = 0.211 found the gold rank on **4/19** with `outranked_by_newer` 0 —
timeline provenance is still unwired, as recorded 2026-08-16.

### The scorer cannot answer "which leg earned the credit"

`score_typed.py` records `{"rank": …}` and nothing else for `episodic_lookup`.
So the obvious follow-up to the 0.591 → 0.767 move — *which leg supplied the
newly-credited fragments* — **is not answerable from disk** and needs a re-run.
This is exactly the gap the 2026-08-16 entry closed in `answer_probes.py` and
`harvest_probe_context.py`; `score_typed.py` was not included in that fix.

---

## 2026-08-17 (later) — G50 shipped, the probe set rebuilt, and two subsystems that produce but never arrive

### Probes rebuilt — 372 → 444, and the anchor half is scoreable again

`generate_typed_probes.py --types codex,temporal --workers 2`, **local Ollama
`qwen3.8:27b`** (endpoint overridden from `.env`, so no paid quota was spent).
Backed up first as `typed_probes.pre-regen-20260817T190153.json`.

| | before | after |
|---|---|---|
| total | 372 | **444** |
| `codex_multihop` | 41 | **104** — past the 89 lost in the [TRAPS #35](TRAPS.md) destruction |
| …carrying `anchor_entity` | **12** | **75** |
| `temporal` | 19 | **28** (13 with `superseded_by`) |

⚠ **The 29 legacy anchorless probes survive** (the generator merges on question
text). `score_typed.py` now scores those on **turn coverage alone** and reports
the two populations separately, instead of counting a missing field as an anchor
failure — the thing that made a 68.5% → 75% improvement read as a regression.

### Typed score on the rebuilt set (arm 1, store unchanged apart from summaries)

| type | 08-17 early | **08-17 late** | n |
|---|---|---|---|
| `codex_multihop` | 0.317 | **0.532** | 41 → 104 |
| `episodic_lookup` | 0.767 | 0.772 | 232 |
| `procedural` | 1.000 | 1.000 | 40 |
| `summary_synthesis` | 0.324 | **0.322** | 40 |
| `temporal` | 0.211 | 0.214 | 19 → 28 |

**Anchor via graph is 53/75 = 70.7%** on properly anchored probes. Artifact:
`score_runs/20260817T151302_post-summaries-444.json`.

### ⚑ TWO SUBSYSTEMS PRODUCE OUTPUT THAT NEVER REACHES THE PROMPT

The session's structural finding, arrived at twice independently.

**1. Batch summaries.** `batch_summarize()` was never broken — `seed_store` calls
it, the call failed **once, transiently**, inside a `try/except` that prints a
warning, and `batch_summaries` stayed **0** from then on. Running it drained the
backlog: **89 turns summarised**, 3 left and correctly skipped below the 5-turn
floor, and **164 turns are `lossless_flag=True`** and are never batch-summarised
**by design** ("memory is earned").

Also fixed: `max_tokens` was hardcoded **500** and truncated **2 of 3** summaries
mid-sentence (one ended `"...**The plan is"`, another cut a list at item 15)
while `bg_model_output_truncated` logged it every time. Now
`batch_summary_max_tokens` = 1200; no truncation at the new ceiling.

⚑ **And `summary_synthesis` still scores 0.322.** Summaries exist for 30 of the
40 probes' conversations. `_batch_summary_lookup` called directly returns **2
fragments at cosine 0.52 / 0.48**, 322 and 467 tokens — **the leg works.** They
are eliminated downstream in RRF/bonuses/budget, where `leg_budget_share` shows
**episodic taking 95–99%** of every request. ⇒ Not a summariser bug, not a metric
bug — a **budget** one, and leg-weight tuning is frozen under [G48](ROADMAP.md#g48).

**2. The procedural leg**, same shape, recorded earlier the same day: one active
pattern returned to every query, scoring 1.000 on a presence test.

⇒ **"The subsystem produces output" and "the output reaches the prompt" are
different claims, and this repo has now confirmed the gap twice in one day.**
Check the second before crediting the first.

### What a seed run does and does NOT exercise

Checked before committing to a two-arm re-seed, because a fixture's gaps become
findings otherwise.

**Real production path:** `evaluate_turn` drives density, grounded summary,
chunking, codex and procedural extraction. Real `resolve_session_id` — not the
script's own labels. Real bg model via `get_bg_model_name()`. Turns land in
**sittings**, so session-keyed features are not inert (an even spread once
produced 293 sessions of one turn each).

**Thin or absent:** `context_clusters` **2** · `conversation_summaries` **0** ·
`session_summaries` **0** · `decisions` **0**. Decay, reflection and the
maintenance agent **do not run during a seed at all**.

⇒ **A seeded store is not a lived-in store.** Any claim about a subsystem only
those jobs populate is **unmeasured, not negative**.

### G50 shipped — and its spec's own numbers were off the production path

Implementation: `difference_kind()` (`e951a4d`), write-time `merge_key` tier +
migration `a1c4e7b90d22` (`77b129f`), typed rejection at detection (`89c1b57`).
On the live store with the cap lifted: **51 merge · 59 defer · 46 rejected**
(35 digits, 5 gender, 4 tense, 1 wordnum, **1 permutation catching a real
converse**). Second run writes **0** new memos.

⚠ **The spec's §1 split (2,247 → 42/583/1,622) was measured with a hand-written
query omitting two filters the real detector applies** — `source='conversation'`
and `entity_type` equality. Production sees **1,793** pairs. Spec corrected under
the divergence protocol (`500c899`). **Third instance this session of a harness
measuring something adjacent to the system and reporting it under the system's
name**; the other two were the probe sampling and the batch-summary eligibility
count.

### Also settled

* **The NER/grounding line closed with nothing shipped, correctly.** Four
  candidate fixes tested and rejected on evidence; the permissive-whitelist bet
  ("the model discards the junk itself") **confirmed on the current pipeline** —
  8 indefensible NER entries against 1,390 produced terms, exactly **1** reaching
  a triplet. A9b stands; **A9c (retrain MicroNER) stays parked**.

### The extraction shape-rule A/B — a clean negative, and the sampling that hid it

`scripts/z1/ab_entity_shape_rule.py`, paired, `qwen3:4b-instruct`, real
extraction path, per-turn checkpointing. Metric is **superset rate** — the share
of subjects/objects grounding only via `_ground_triplets`' superset arm, i.e.
the mechanism itself, deterministic and needing no fragment lexicon.

| run | sample | superset rate | triplets | verdict |
|---|---|---|---|---|
| v1 | 30 turns, **25 from one conversation** | 0.090 → 0.1008 | 12.5 → 12.53 | **invalid sampling** |
| v3 | 24 turns, **8/8/8 across conversations** | **0.0858 → 0.1166** | **58.4 → 49.1 (−16%)** | **rule fails** |

⇒ **No precision gain, real recall cost.** `codex_extraction_entity_shape_rule`
ships **False** and stays that way as the record of a tested negative.

⚑ **And the re-read overturned the item's premise.** Of 266 superset hits across
the three conversations, the overwhelming majority are **legitimate**:
`gemma-4-e4b`, `nvidia nemotron 3 nano`, `qwen-coder-32b-q4`, `separate /boot
partition`, `central board of secondary education`, `object oriented
programming`. Genuine fragments (`in room`, `first thing`, `became too good`) are
**≈0.4% of extracted terms** — a hand-read of a sample, **not a measurement**,
flagged as such because a hand-written lexicon tried the same job and reported
10.8% while passing `get validation` and `to stress or trauma` as real links.

⇒ **Fragments are UNIQUE; real entities REPEAT.** 0.4% of extracted terms but
24.6% of *distinct* store entities — 89 superset hits collapse to 39 distinct
terms. **Store counts and extraction rates have different denominators**, and
conflating them is what made a 0.4% tail look like an epidemic. ⇒ Fragments are
**not** why the graph is 75% dead ends; those are real one-mention entities
(`gemma 4 31b`, `boot efi`, `soft skills`), which makes [G51](ROADMAP.md#g51)
*more* valuable and demotes its `junk` verdict to a sub-case.

⚠ **Run-to-run variance is ±60% and it lied at n=2**: a 2-turn smoke reported
triplets −33%; at 30 turns the delta was 0.0. Size runs accordingly — the A9b
comparison cannot be re-decided on 10 turns.
* **CoE AI Gateway** available as a third serving path (see [MODELS.md](MODELS.md)).
  **Judging stays on `deepseek-v4-flash`** so verdicts remain comparable.

---

## 2026-08-20 — the two-arm NER re-seed (A9b), and the first measurement of whether the codex graph is TRUE

**⚑ EVERY NUMBER BELOW LIVES ONLY HERE.** `experiments/curation_files/` and
`logs/` are both gitignored (they carry the corpus), and `docs/SESSION.md` is
gitignored and emptied at session end. A result that is not copied into this
file is lost when the working tree is cleaned — see the standing rule added to
[TRAPS.md](TRAPS.md) the same day, after an arm's seed log was destroyed by a
reboot because it had been written to `/tmp`.

### The design — one variable

Both arms: 293 turns from the same three conversations, `qwen3:4b-instruct` as
the background model, the six older extraction fixes, G50's write-time
`merge_key` tier, `CODEX_NODE_PROMOTION=true` (matching the `arm1-post-g50`
snapshot they are compared against). **The only difference is which NER confirms
entities for codex grounding:**

| arm | snapshot | codex-grounding NER | settings |
|---|---|---|---|
| A | `ner-a-micro` | micro-NER | `codex_extraction_ner_tier=preflight` |
| B | `ner-b-nuner` | NuNER Zero | `codex_extraction_ner_tier=background`, `codex_grounding_ner_extra_types=concept,object` |

Clustering and key-term extraction use NuNER in **both** arms (unchanged since
A9b, 2026-08-03); the pre-flight request path keeps the micro-NER in **both**
(A9c, parked). ⇒ this is not "NuNER everywhere vs micro-NER everywhere": one of
four call sites differs. Config asserted from each running process's
`/proc/<pid>/environ`, not from the driver.

### Store outcome

| | arm A (micro-NER) | arm B (NuNER + concept/object) |
|---|---|---|
| turns / post-flight failures | 293 / 0 | 293 / 0 |
| codex_entities | 8,470 | **6,271** (0.74×) |
| codex_edges | 10,239 | **10,228** (−0.1%) |
| edges/entity | 1.21 | **1.63** |
| procedural | 64 | 65 |
| batch_summaries | 3 | 3 |

**Arm B holds an essentially identical graph over 26% fewer entities.** That is a
count, not a quality claim.

### Retrieval, typed score, all 444 probes — ARM B

| type | n | arm B | arm 1 (2026-08-17, OLD pipeline) |
|---|---|---|---|
| `codex_multihop` | 104 | 0.531 | 0.532 |
| `episodic_lookup` | 232 | 0.698 | 0.772 |
| `procedural` | 40 | 1.000 | 1.000 — **a tautology**, pool of one (TRAPS #37) |
| `summary_synthesis` | 40 | 0.303 | 0.322 |
| `temporal` | 28 | **0.500** | 0.214 |

Anchor via graph 56/75 (74.7%) against arm 1's 53/75 (70.7%). ⚠ arm 1 is the OLD
pipeline, so these differences conflate the six fixes, G50 and the NER. **Arm A
is what separates them and is not yet scored.**

### ⚑ CODEX QUALITY — the first time this store's triplets were checked for TRUTH

`scripts/z1/judge_codex.py`, **deepseek-v4-flash**, n=200, arm B. Each triplet
judged against its own source turn (`codex_edges.source_batch` →
`episodic_memory.batch_id`, verified 1:1, 293 distinct ids for 293 turns), blind
to the arm and to the stored confidence tier. Grouped by source turn so the judge
sees a turn's triplets together. Categorical taxonomy, not a 1–5 rating, for the
same drift reason `judge_answers.py` gives.

| label | n | share |
|---|---|---|
| correct | 40 | **20.0%** |
| **reversed** (subject/object swapped) | 50 | **25.0%** |
| wrong | 74 | 37.0% |
| malformed | 25 | 12.5% |
| vacuous | 8 | 4.0% |
| unjudgeable | 3 | 1.5% |

**One in five stored triplets is true.**

**A quarter are merely BACKWARDS** — right entities, right relation, inverted
direction (`india --lives_in--> maharashtra`, `student paper --is_used_for-->
turnitin`). Nothing guards this: the converse guard (`codex_extractor.py:463`)
covers only a curated `_ANTONYM_PAIRS` list. A direction check would move a
quarter of the graph from wrong to right without touching the NER, the relation
vocabulary or the model.

**And `extraction_confidence` is INVERTED against truth:**

| tier | n | correct | reversed | wrong |
|---|---|---|---|---|
| grounded 0.9 | 107 | **15.0%** | 24.3% | 42.1% |
| rejected 0.35 | 92 | **25.0%** | 26.1% | 31.5% |

The tier retrieval trusts more is the one less often true. This does **not** mean
grounding is broken — grounding predicts whether the subject string is present in
the source turn at all, and there it works (**7.3%** of grounded subjects absent
from their source turn vs **22.9%** of rejected ones). It means **span presence
is not truth**, while `extraction_confidence` is consumed downstream as if it
were. A3's confidence seeding, and any retrieval gate keyed on that column, are
ranking by the wrong quantity.

⚠ **None of this separates the arms.** It is a statement about the extractor.
Arm A's identical run (same script, same seed, same n) is pending its restore.

### Graph shape — arm B

6,271 entities, 9,411 live edges. **64.2% of entities carry exactly one edge** —
a single triplet and nothing more; 0.5% isolated, 4.9% hubs (degree ≥ 10), median
degree 1, max 333. Fan-out among entities with any outgoing edge: mean 2.24,
median 1. ⇒ [G51](ROADMAP.md#g51)'s premise confirmed on real data: the store is
mostly disconnected pairs, not a graph. Top hubs mix real entities (`deepesh`
179, `orien` 156, `krishna` 112) with generic ones the wide label list admitted
(`person` 333, `story` 257, `people` 121, `student` 101).

Relation vocabulary has collapsed toward vacuous predicates: `have` 816, `in`
788, `are` 631, `uses` 594 — ~30% of all edges in four near-meaningless
relations.

### Batch summariser — fixed the same day, and it is why both arms are scoreable

The fixed-50-turn batch sent **37,359 tokens at a 32,768 window** and returned
400; because one `try` wrapped the whole pass, that killed the two later batches
that would have fitted, and the arm ended with **zero** summaries — recorded the
previous session as a transient failure it was not. Now token-budgeted (26,200
content tokens) with per-batch isolation. Arm A drained to 3 summaries covering
89 turns; arm B produced 3 during its own seed. **`summary_synthesis` (40 probes)
is scoreable on both arms only because of this.**

### Known asymmetry against the "before" snapshot — state it when reporting

Every arm A/B `context_payload` opens `Properties: merge_key: <its own name>`,
because G50 writes `merge_key` into `properties` and the payload builder
serialises all properties. **176,391 of 498,654 payload characters (35%)**, ~9
tokens per injected entity, restating a name already in the `[Entity: X]`
header. Payloads carrying the line: `fixed-qwen3-4b-instruct` **0**,
`arm1-post-g50` **0**, `ner-a-micro` **8,435** — the G50 migration backfilled the
property without regenerating payloads. ⇒ **arm A vs arm B is clean** (both carry
it); **arm A vs `arm1-post-g50` is not** — the new pipeline spends tokens the old
one did not, from the same budget the evidence block competes for. Left in place
deliberately for this run (user, 2026-08-20).

### ⚑ ARM A vs ARM B — retrieval, and the result is the OPPOSITE of the structural read

Typed score, all 444 probes, same probe set, same scorer, one run each.

| type | n | arm A (micro-NER) | arm B (NuNER + concept/object) |
|---|---|---|---|
| `codex_multihop` | 104 | **0.559** | 0.531 |
| `episodic_lookup` | 232 | **0.707** | 0.698 |
| `procedural` | 40 | 1.000 | 1.000 — tautology, pool of one (TRAPS #37) |
| `summary_synthesis` | 40 | **0.324** | 0.303 |
| `temporal` | 28 | **0.643** | 0.500 |
| anchor via graph | 75 | 57/75 (76.0%) | 56/75 (74.7%) |

**Arm A wins every non-tautological type.** And yet arm B has the structurally
better graph on every measure:

| | arm A | arm B |
|---|---|---|
| entities | 8,470 | 6,271 |
| live edges | 9,755 | 9,411 |
| edges/entity | 1.15 | **1.63** |
| single-edge entities | 5,783 (**68.3%**) | 4,029 (**64.2%**) |
| isolated | 59 (0.7%) | 31 (0.5%) |
| hubs (degree ≥ 10) | 252 (3.0%) | 307 (**4.9%**) |
| mean / max degree | 2.30 / 204 | **3.00** / 333 |
| fan-out mean (entities with out-edges) | 1.99 | **2.24** |

⇒ **The denser, less fragmented graph retrieves WORSE.** This is the opposite of
the premise A9b's swap rests on, and it is the central result of the run.

⚠ **What this is not.** One run per arm, no variance estimate, and three of the
four gaps are 0.01–0.03 — only `temporal` (0.643 vs 0.500) is a wide gap. The
answer-level judging and the per-arm codex-truth judging are what decide whether
arm B's graph is at least *truer*; a cleaner graph that retrieves slightly worse
but answers better would be a trade rather than a loss.

### ⚑ CODEX TRUTH, BOTH ARMS — better extraction bought WELL-FORMEDNESS, not TRUTH

`judge_codex.py`, deepseek-v4-flash, n=200 per arm, same seed (`20260820`), same
taxonomy, blind to arm and to stored confidence.

| label | arm A (micro-NER) | arm B (NuNER + concept/object) |
|---|---|---|
| correct | 18.0% | **20.0%** |
| reversed | 16.5% | **25.0%** |
| wrong | 38.0% | 37.0% |
| **malformed** | **25.0%** | **12.5%** |
| vacuous | 2.5% | 4.0% |
| unjudgeable | 0.0% | 1.5% |

**NuNER HALVED malformed triplets (25.0% → 12.5%)** — cleaner spans doing
exactly what a better entity source should do. But the recovered mass did not
become *correct*; it became **reversed** (16.5% → 25.0%). Correctness moved
**2 points** and `wrong` did not move at all (38.0% → 37.0%).

⇒ **The best available extraction change moved correctness from 18% to 20%.**
Meanwhile the cleanup-addressable share — `reversed` + `malformed`, both fixable
by post-processing without touching extraction — is **41.5%** (arm A) and
**37.5%** (arm B). The lever is cleanup, not extraction.

**And the confidence inversion REPRODUCES INDEPENDENTLY IN BOTH ARMS:**

| tier | arm A correct | arm B correct |
|---|---|---|
| grounded 0.9 | 14.3% | 15.0% |
| rejected 0.35 | **22.1%** | **25.0%** |

Two stores, two different NERs, same inversion — so it is a property of the
system, not of either entity source. `extraction_confidence` measures whether
the NER confirmed the subject SPAN, and span presence is not truth; every
consumer that reads that column as a truth prior is ranking by the wrong
quantity (A3's seeding, and any retrieval gate keyed on it).

### ⚑ ARE SINGLE-EDGE ENTITIES REAL? — and this is NuNER's clearest win

64–68% of each arm's entities carry exactly ONE edge. The question was whether
those are genuine one-mention facts or [G51](ROADMAP.md#g51) missed links. **They
are mostly neither: they are nodes that should not exist.**

Structural tests could not separate junk from missed links — "one edge + many
mentions" turned out to measure JUNK (`will` 138 turns, `enough` 88, `between`
70, all with a single edge; a genuinely important entity that common would have
accumulated edges), and a multi-word cut failed too (`it is`, `the one`, `to
build`, `will be`). So the strings were judged directly, deepseek-v4-flash,
n=150 per arm, seeded random over single-edge entities:

| single-edge entities are… | arm A (micro-NER) | arm B (NuNER + concept/object) |
|---|---|---|
| a real entity | 41.3% | **59.3%** |
| a sentence FRAGMENT | 42.0% | **22.7%** |
| a common word | 16.7% | 18.0% |
| **NOT an entity** | **58.7%** | **40.7%** |

**An 18-point gap at n=150 is ~4.4 SE — the clearest, most significant
difference between the arms.** Fragments HALVED (42.0% → 22.7%), independently
corroborating the codex judge's malformed halving (25.0% → 12.5%). Two unrelated
instruments, same conclusion: **NuNER's gain is well-formedness, and it is
large.**

Absolute junk-node load: arm A ~**3,400** of 5,783 single-edge entities; arm B
~**1,640** of 4,029. **Arm B carries less than half the junk.**

⇒ **This revises G51's sizing the way G50's was revised.** Its "5,487 missing
edges" backlog is majority JUNK in arm A and minority junk in arm B — a linking
pass run over arm A would spend most of its effort connecting nodes that should
have been dropped. **Deduplicate/prune before linking, not after.**

### ⚑ ANSWER JUDGING, ARM A vs ARM B — 120 probes, blind, paired

`judge_answers.py`, deepseek-v4-flash, same 120 stratified probes answered by
`gemma4:26b-a4b-it-q4_K_M` on each arm's own store, arm identity hidden and A/B
slot randomised per probe.

| type | arm A (micro-NER) | arm B (NuNER) | TIE |
|---|---|---|---|
| `codex_multihop` | 2 | **4** | 18 |
| `episodic_lookup` | **9** | 3 | 12 |
| `procedural` | 1 | **3** | 20 |
| `summary_synthesis` | 1 | 1 | 22 |
| `temporal` | **10** | 2 | 12 |
| **TOTAL** | **23** | 13 | 84 |

Reasons: **both_failed 70**, more_grounded 18, equivalent 14,
contradicts_source 9, more_complete 7, no_memory_used 2.

**⚑ THE HEADLINE IS NOT THE ARM COMPARISON. `both_failed` is 70 of 120 (58%) —
neither arm answered nearly six probes in ten.** The NER question is being asked
inside a system that cannot answer most of what is put to it, which bounds how
much any entity-source decision can matter.

**⚑ AND THE TOTAL HIDES OPPOSITE EFFECTS.** Arm B wins `codex_multihop` (4–2) —
the only type that actually tests the graph — and `procedural` (3–1). Arm A's
lead comes entirely from `temporal` (10–2, the only per-type result clearing
noise at p≈0.02) and `episodic_lookup` (9–3). ⇒ **the cleaner graph helps GRAPH
questions; the larger, junkier entity set helps LEXICAL recall.** That is the
signature of A4 grounded query expansion (`orchestrator.py:558`), which appends
matched entity names and aliases to the BM25 search prompt: arm A has 8,470
entities to arm B's 6,271, so it expands more. **Reading the 23–13 total as
"micro-NER is better" would invert the finding on the type the swap was for.**

⚠ Overall 23 vs 13 across 36 decisive verdicts is binomial p≈0.07 — marginal.
Per type, only `temporal` is individually significant.

### ⛔ RETRACTION — the 58% `both_failed` is largely a JUDGE ARTIFACT (found by the user, same day)

The judge is shown a **median 12.7%** of the gold material it is asked to verify
answers against. Two truncations compose:
`answer_probes.py:226` keeps at most **3 gold turns**, each cut to **1,200
chars**; `judge_answers.py:107` then cuts the result to 4,000.

| | |
|---|---|
| gold content per probe | mean **38,638** chars |
| what the judge SEES | mean **2,606** chars |
| coverage | mean 19.7% · **median 12.7%** |
| probes where judge sees <50% of gold | **111 of 120** |

**46 of the 120 probes have 14 gold turns.** The judge sees three.

**⚑ AND COVERAGE PREDICTS THE TIE RATE MONOTONICALLY ACROSS ALL FIVE TYPES:**

| type | gold coverage | TIE rate |
|---|---|---|
| `summary_synthesis` | 6.3% | 92% (22/24) |
| `procedural` | 6.7% | 83% (20/24) |
| `codex_multihop` | 15.8% | 75% (18/24) |
| `temporal` | 32.2% | 50% (12/24) |
| `episodic_lookup` | 37.5% | 50% (12/24) |

⇒ **"Memory failed on 58% of probes" is NOT a finding.** A correct answer drawn
from gold turn 7 of 14 is ruled unsupported because the judge was shown turns
1–3. The arm-vs-arm verdicts are less affected — both arms were judged against
the same truncated source, so the comparison is internally fair — but the
`both_failed` RATE, and any absolute claim about memory quality built on it, are
withdrawn.

**Fix before any further answer judging**, or every subsequent run reproduces
the artifact: raise the per-turn cap and the turn count so the judge sees the
gold set, or select the gold turns that the probe's answer actually depends on
rather than the first three.

### ⛔ SECOND, INDEPENDENT GROUND-TRUTH DEFECT — 21.7% of probes are UNPASSABLE

Distinct from the truncation above. That was the judge being shown too little of
the gold; **this is whether the gold is RIGHT AT ALL.** `verify_gold.py`,
deepseek-v4-flash, n=60, gold turns untruncated, no system under test — just the
probe's stated answer against the turns it was labelled with.

| verdict | n | share |
|---|---|---|
| supported | 44 | 73.3% |
| partial | 3 | 5.0% |
| **unsupported** | **13** | **21.7%** |
| unanswerable | 0 | 0.0% |

| type | supported | unsupported |
|---|---|---|
| `episodic_lookup` | **100.0%** | 0.0% |
| `codex_multihop` | 58.3% | **29.2%** |
| `procedural` | 50.0% | **50.0%** |

*(n=60 covered `codex_multihop`, `episodic_lookup` and half of `procedural`;
`summary_synthesis` and `temporal` are UNMEASURED.)*

**A probe whose own gold cannot support its own answer scores every arm as a
failure no matter how good retrieval is.** And it corrupts both metrics at once:
`score_typed` computes recall/coverage against those turn ids, and
`judge_answers` shows those turns to the judge as SOURCE.

⚠ **The damage is concentrated in exactly the types that showed the highest tie
rates**, so the two GT defects and the `both_failed` rate are not independent
observations — they are substantially the same artifact seen three ways.
`procedural` is now known to be broken in TWO ways: half its gold is unsupported,
AND its metric is a presence check that any procedural fragment satisfies
(TRAPS #37), which is why it reports a constant 1.000.

⇒ **Fix the gold before any further arm comparison, ablation or codex-probe
work.** Repair `gold_turns` or drop the probe, then re-score. Until then, only
`episodic_lookup` (100% supported) rests on sound ground truth.

### ⚑ RE-JUDGED ON FULL GOLD — the arms are a DEAD HEAT, and the earlier per-type story is WITHDRAWN

Same 120 probes, same answers, same judge. Only the SOURCE changed: rebuilt from
the store untruncated (mean 2,612 → **38,649 chars, 14.8×**, 0 fallbacks).

| | truncated gold | **full gold** |
|---|---|---|
| arm A (micro-NER) wins | 23 | **31** |
| arm B (NuNER) wins | 13 | **30** |
| TIE | 84 | 59 |
| **both_failed** | **70 (58%)** | **37 (31%)** |

| type | truncated | full gold |
|---|---|---|
| `codex_multihop` | B 4–2 | **A 7–3** |
| `episodic_lookup` | A 9–3 | **7–7** |
| `procedural` | B 3–1 | A 6–5 |
| `summary_synthesis` | 1–1 | **B 8–5** |
| `temporal` | **A 10–2** | B 7–6 |

**1. `both_failed` nearly halved (58% → 31%).** The truncation inflated it by
~27 points. 31% is the credible figure and is still an overstatement while
21.7% of probes carry gold that cannot support their own answer.

**2. The arms are statistically indistinguishable on answer quality: 31 vs 30
across 61 decisive verdicts.** The earlier 23–13 was an artifact.

**3. ⛔ EVERY PER-TYPE SIGNAL FROM THE TRUNCATED RUN FLIPPED OR VANISHED**, and
the interpretation built on them is withdrawn. That interpretation was: arm B
wins the graph-testing type while arm A wins lexical types via A4 query
expansion (8,470 entities vs 6,271 feeding `orchestrator.py:558`). With full
source, `codex_multihop` reverses to A 7–3 and `temporal` evens out. **The
expansion hypothesis may still be true — it simply has no evidence behind it
now**, and testing it is what the planned codex ablation is for.

**What survives the correction:** the arms differ measurably in FORM — arm B
carries ~1,640 junk nodes against arm A's ~3,400, halves malformed triplets
(25.0% → 12.5%) and fragment entities (42.0% → 22.7%), and yields a denser graph
(edges/entity 1.63 vs 1.15) — while being **indistinguishable in answer
OUTCOME**. Nothing measured so far shows that better-formed memory produces
better answers.

⚠ **Method note.** Three successive readings of this run reversed as the
instrument was corrected. The rule that would have saved the intermediate
claims: *do not interpret a comparison until the metric has been checked against
its own ground truth.*

### ⚑ CORRECTION — the ground truth is SOUND; "21.7% unpassable" was over-escalated

`check_gold_consistency.py` (deterministic, no model, no cost) checks each
probe's `gold_turns` against its own `evidence` field — unused by any scorer
until now, and present on **400 of 444** probes.

| type | consistent | broken | unchecked (no evidence) |
|---|---|---|---|
| `codex_multihop` | 74 | 1 | 29 |
| `episodic_lookup` | 226 | 3 | 3 |
| `procedural` | **40** | **0** | 0 |
| `summary_synthesis` | 40 | 0 | 0 |
| `temporal` | 13 | 0 | 15 |

**4 broken of 400 checkable — 1%, not 21.7%.**

**Why the two measurements disagree, and which one governs.** `verify_gold.py`
asks whether the probe's **stated `answer`** is supported by the gold; 21.7%
were `unsupported`. But **`unanswerable` was 0.0%** — the judge never once said a
QUESTION could not be answered from its gold. And neither scorer reads
`expected_answer`: `score_typed` scores coverage against `gold_turns`, and
`judge_answers` shows those turns as SOURCE and compares two answers. ⇒ **the
21.7% is a defect in a field nothing consumes.** The probes are passable; their
`answer` label over-claims.

⇒ **The gold does NOT block the codex ablation or any further comparison.** The
one real ground-truth defect this session was the SOURCE TRUNCATION, which is
fixed and re-run.

**⚠ AND THE CHECK ITSELF NEARLY DESTROYED THE PROBE SET.** `evidence` numbering
is **type-dependent and undocumented**: `codex_multihop` (gold width 4) uses
ABSOLUTE turn numbers, while `procedural` (gold width 14, windows starting 15,
29, 43…) uses an index RELATIVE to the window — `[1, 3, 14]` against gold
`[15..28]` means turns 15, 17, 28. Read as absolute it looks like the gold omits
every evidence turn, and the first version of the check reported **34 of 40
`procedural` probes as broken and would have dropped them.** The checker now
tries absolute first and falls back to relative.

### ⚑ CODEX ABLATION (retrieval level) — the leg is NET-NEGATIVE outside its own metric

Arm B store, all 444 probes, `score_typed --ablate`, deterministic, no model.
Three conditions; the middle one exists because the codex graph reaches the
prompt by TWO paths — as fragments, and as A4 grounded query expansion appending
matched entity names to the BM25 search prompt (`orchestrator.py:558`).

| type | full | fragments dropped (expansion KEPT) | codex fully OFF | codex's net worth |
|---|---|---|---|---|
| `codex_multihop` | 0.531 | 0.406 | 0.296 | +0.235 — **circular, discount** |
| `episodic_lookup` | 0.698 | 0.664 | **0.664** | **+0.034** |
| `procedural` | 1.000 | 1.000 | 1.000 | 0 — tautology |
| `summary_synthesis` | 0.303 | 0.303 | **0.336** | **−0.033** |
| `temporal` | 0.500 | 0.536 | **0.536** | **−0.036** |
| anchor via graph | 56/75 | 30/75 | **0/75** | |

**1. ⚑ GROUNDED QUERY EXPANSION (A4) CONTRIBUTES NOTHING.** The gap between
`fragments` and `all` IS the expansion contribution. `episodic_lookup` is
**0.664 vs 0.664** and `temporal` **0.536 vs 0.536** — identical. Expansion
slightly HURTS `summary_synthesis` (0.303 → 0.336 when removed). ⇒ the
hypothesis that arm A's larger entity set won via better BM25 expansion is
**dead on evidence**, not merely unsupported.

**2. ⚑ THE CODEX LEG IS NET-NEGATIVE ON THE TYPES THAT CAN JUDGE IT FAIRLY.**
Against fully-off it gains **+0.034** on `episodic_lookup` and loses **0.036**
(`temporal`) and **0.033** (`summary_synthesis`) — netting to roughly zero,
**while consuming 0.5–32% of the prompt budget** (`leg_budget_share`). The
`codex_multihop` gain is discounted because that metric's anchor half is DEFINED
as "the anchor entity appeared in a codex or timeline fragment", so disabling
codex zeroes it by construction (0/75) — the metric cannot be used to justify
the leg it is built from.

**3. UNDOCUMENTED COUPLING: THE TIMELINE LEG DEPENDS ON THE CODEX GRAPH.**
`anchor via graph` runs 56 → 30 → **0**: dropping codex fragments leaves 30
anchors supplied by TIMELINE fragments, but disabling the codex leg kills those
too. Nothing in the architecture notes records that timeline cannot function
without codex.

⇒ **Consequence for the fix queue.** The direction check, junk filter and
confidence recalibration were ranked by how much of the GRAPH they repair. This
says repairing the graph buys little at the retrieval level, because the leg
carrying it is already worth about zero. **Confirm at the answer level (stage 2)
before acting** — retrieval score is not answer quality, and `codex_multihop`
answers were the one place arm B beat arm A on truncated gold.

### ⚑⚑ CODEX ABLATION, ANSWER LEVEL — THE LEG IS NET-HARMFUL ON ITS OWN PROBE TYPE

Arm B store, **all 104 `codex_multihop` probes** — the only type where codex can
earn credit as codex. Three answer sets generated on identical probes with
`gemma4:26b-a4b-it-q4_K_M`, differing only in codex; judged pairwise and blind
by deepseek-v4-flash against FULL untruncated gold.

| comparison | full system | ablated | TIE | decisive | one-tailed p |
|---|---|---|---|---|---|
| full vs **codex fragments dropped** (expansion kept) | 14 | **25** | 65 | 39 | ≈ 0.055 |
| full vs **codex leg disabled** (budget reclaimed) | 17 | **36** | 51 | 53 | **≈ 0.007** |

`both_failed` falls **42 → 29** between the two comparisons: with codex fully
off, the pair collectively answers MORE probes.

**⚑ THE PROGRESSION IS THE RESULT.** Dropping codex TEXT helps (suggestive,
p≈0.11 two-tailed). Disabling the LEG so other legs reclaim its budget helps
MORE and significantly (p≈0.013 two-tailed). Fragment counts confirm the
mechanism: 20.1/probe full → 16.9 with fragments dropped → **18.4** with the leg
off, i.e. other legs recover ~1.5 of the freed slots. ⇒ the claim is not merely
"codex fragments are noise" but "**the budget codex consumes is worth more to
other legs**".

**Paired set: 102 of 104.** Two probes excluded — both `ReadTimeout` at
`answer_probes.py:219`'s 180 s cap, both in the `fragments` condition, both from
the longest-turn conversation. Excluding them is safe in the direction that
matters: `fragments` carries FEWER fragments per probe (16.9 vs 20.1) and hence
shorter prompts, so the timeouts cannot have been caused by the condition
inflating the prompt. Empty-answer counts are balanced across conditions
(2/1/2).

**FOUR INDEPENDENT MEASUREMENTS NOW AGREE:**
| measurement | verdict |
|---|---|
| retrieval ablation (444 probes) | codex nets ~zero; −0.036 `temporal`, −0.033 `summary_synthesis` |
| **answer ablation (104 codex probes)** | **codex loses 17–36 on its own type** |
| codex truth judge (n=200/arm) | 20% of triplets correct, 25% reversed |
| A9b arm comparison (293 turns) | better extraction moved form, not outcome |

⇒ **CONSEQUENCE FOR THE FIX QUEUE, WHICH THIS INVERTS.** The queue (direction
check, junk filter, `extraction_confidence` recalibration, G51 linking, vacuous
relation pruning) was ranked by how much of the GRAPH each repairs — the right
ranking for a leg worth improving. The leg is currently net-harmful, so the
first question is not which repair to make but **whether to gate the leg while
its quality is this low**. ⚠ Gating it would also disable the TIMELINE leg,
which depends on the codex graph (anchor-via-graph 56 → 30 → 0).

⚠ **Scope of the claim.** One store (arm B), one probe type, one answering
model, single run. It does NOT show codex is worthless in principle — it shows
that a graph measured at 20% triplet correctness costs more budget than it
returns. The prediction that follows, and the way to falsify all of this:
**repair the graph first (direction check alone would move ~25% of edges) and
re-run this exact ablation. If codex still loses, the design is wrong; if it
wins, the quality bar is simply higher than the current extractor clears.**

---

## 2026-08-20 (close) — what the day actually established, and the question it opened

**Read this before re-reading the individual entries above.** Several were
written and later corrected as instruments were fixed; this is the surviving
account.

### What is now measured, and how confident to be

| claim | evidence | confidence |
|---|---|---|
| NuNER halves malformed triplets and fragment-entities | 25.0%→12.5% (n=200/arm) · 58.7%→40.7% non-entities (n=150/arm), both ~4.5 SE | **high** |
| NuNER does NOT improve triplet correctness | 18.0%→20.0%, inside 1 SE at n=200 | **high** (a real null) |
| NuNER does NOT improve answers | full-gold judging 31–30 across 61 decisive verdicts | **high** (a real null) |
| Only ~20% of stored triplets are TRUE | n=200/arm, judged against each triplet's own source turn | **high** |
| ~25% of triplets are merely REVERSED | same run; a distinct, mechanically repairable class | **high** |
| `extraction_confidence` is inverted against truth | grounded 14–15% correct vs rejected 22–25%, **independently in both arms** | **high** |
| A4 grounded query expansion contributes nothing | two ablations identical on two types (0.664/0.664, 0.536/0.536) | **high** |
| The codex leg is net-harmful on its own probe type | disabling it wins 36–17, 53 decisive, two-tailed p≈0.013 | **medium-high** — one store, one probe type, one model, single run |
| 64–68% of entities carry exactly one edge | direct count, both arms | **high** |
| ~41–59% of those single-edge entities are not entities at all | n=150/arm | **high** |

### What is NOT established, and must not be inferred from the above

- **That codex is worthless in principle.** What is shown is that a graph at 20%
  triplet correctness costs more budget than it returns. The falsification test
  is stated and cheap: **repair direction (~25% of edges) and re-run the same
  ablation.**
- **That the other legs work.** `procedural` scores a constant 1.000 because its
  metric asks only whether any procedural fragment came back (TRAPS #37) — its
  contribution has **never been measured**. `batch_summary` sits at 0.303.
  `timeline` was found to depend entirely on codex. **A per-leg ablation was
  started the same day to answer this.**
- **That decay, reflection or the maintenance agent do anything.** None of them
  run during a seed: `decay_score < 1.0` is **0 rows** in this store. Every
  number here describes a store that has never aged.
- **Anything about the write path.** The density evaluator, the earned-lossless
  decision (**164 of 293 turns lossless, 129 summarised**), procedural
  extraction, the B2 retrieve/don't-retrieve gate and the context ledger were
  **not under test today**. The day's findings concern one read-path leg.

### The instrument record, which bounds all of it

**Seven measurement defects in one day**, none of which raised an exception
(TRAPS #41). Three interpretations were written and withdrawn — the answer
comparison alone reversed twice before its judge was shown more than a median
12.7% of the gold. ⇒ **Treat any number here that has not been reproduced by a
second, independent instrument as provisional.** The four claims marked *high*
above each have two: a deterministic count and a model judgement, or two
independent arms agreeing.

---

## 2026-08-20 — PER-LEG ABLATION: what each retrieval leg is actually worth

Arm B store, 444 probes per condition, `score_typed --ablate-leg <name>`,
deterministic (classifier + embedder + micro-NER only — **no generative model**,
so no temperature and no run-to-run variance). Baseline is the full-system arm B
run; each row disables ONE leg.

| leg disabled | `episodic_lookup` | `codex_multihop` | `summary_synthesis` | `temporal` |
|---|---|---|---|---|
| — (baseline) | 0.698 | 0.531 | 0.303 | 0.500 |
| `procedural` | **0.737** (+0.039) | 0.543 (+0.012) | 0.296 | **0.214 (−0.286)** |
| `batch_summary` | 0.698 (**0**) | 0.531 (**0**) | 0.303 (**0**) | 0.536 (+0.036) |
| `vector` | 0.608 (**−0.090**) | 0.404 (**−0.127**) | **0.046 (−0.257)** | 0.571 (+0.071) |
| `rrf` | **0.478 (−0.220)** | **0.349 (−0.182)** | 0.138 (**−0.165**) | 0.464 (−0.036) |
| `mera` (graph enumeration) | 0.698 (0) | 0.530 (−0.001) | 0.303 (0) | 0.500 (0) |

*(`procedural` scores 1.000 in every row but its own — the metric asks only
whether any procedural fragment returned, TRAPS #37. It is excluded from
comparison throughout.)*

### ⚑ WHAT WORKS IS THE HYBRID-RAG CORE; THE MEMORY-STRUCTURE LEGS DO NOT PAY

**`rrf` is the largest contributor in the system** (−0.220 / −0.182 / −0.165) and
**`vector` is second** (−0.257 on `summary_synthesis`). Semantic + lexical
retrieval fused by reciprocal rank is doing the work. The three legs that make
ICE *not* a RAG system — codex, procedural, batch summaries — are the three that
currently contribute nothing or less than nothing.

**This must not be read as "the design is wrong."** Each has a named, mechanical
defect and none has been repaired:

1. **`batch_summary` HAS NEVER BEEN IN A PROMPT.** Zero appearances in **1,925**
   `leg_budget_share` lines across three independent runs. Called directly the
   leg WORKS — it returns a real fragment (`score=0.234`, 367 tokens, correctly
   typed). **The cause is structural, not a bug or a missing weight:** a leg
   absent from `retrieval_leg_base_weights` defaults to **1.0** in RRF
   (`orchestrator.py:2438`, joint-highest), but RRF scores
   `weight / (k + rank)` **summed over every leg that found the fragment** — so a
   turn found by BOTH `bm25` (0.8) and `vector` (1.0) accumulates ~1.8/(k+r)
   while a summary, which no other leg can produce, gets 1.0/(k+0) once.
   ⇒ **A leg with no cross-leg partner cannot win RRF, however relevant its
   content.** Codex, procedural and timeline share the disadvantage; batch
   summary is the extreme case, at exactly zero. **This is a sharper statement of
   [G48](ROADMAP.md#g48) than "the budget favours episodic": leg-weight tuning
   alone cannot fix it — a partnerless leg needs a reserved slot or a fusion rule
   that does not reward agreement.**
2. **`codex` is net-harmful at 20% triplet correctness** (25% merely reversed).
   The falsification test — repair direction, re-run the same ablation — is
   unrun.
3. **`procedural` is fabricated by construction** (A12): `extract_procedural`
   asks ONE turn to reveal a RECURRING habit. It also **hurts** episodic
   retrieval (+0.039 when removed).

### ⚠ TWO KINDS OF ZERO, AND THEY MEAN OPPOSITE THINGS

`batch_summary` and `mera` both score 0.000 delta. They are not the same result
and must never be tabulated as one:

- **`batch_summary` PRODUCES and always LOSES.** Called directly it returns a
  real fragment; it reaches the prompt 0 times in 1,925 requests. That is a
  **structural defect** — a measured failure with a known mechanism.
- **`mera` (graph enumeration) NEVER TRIGGERS.** It is cue-gated — it fires only
  on an explicit "list all the X" prompt carrying a grounded tag/relation
  signal, and the 444-probe set contains no such prompt. That is **UNTESTED ON
  THIS CORPUS**, not a null. Recording it as "contributes nothing" would be a
  false negative about a feature that was never asked to run.

⇒ To measure enumeration at all, the probe set needs enumeration-shaped
questions. None of the five typed classes generates them.

⚠ **`mera` is also a stale NAME.** MERA as a subsystem was deleted by A4; the
flag survives only because it was re-homed onto the enumeration path
(`configurable_orchestrator.py:99` → `enable_enumeration`). See TRAPS #42.

### Two effects with unresolved mechanisms — do not report either as established

- **`procedural` OFF collapses `temporal` 0.500 → 0.214**, the largest single
  effect measured. It may be capability, or it may be provenance credit: G48's
  rule counts a fragment if the gold turn is among the turns it was DERIVED
  from, and **65 procedural rows carry `source_batch_ids` spanning 250 of 293
  turns**, with 5 procedural fragments returned on every probe. **The decisive
  test is unrun:** record the `source_type` of the fragment that earned the hit.
- **`vector` OFF IMPROVES `temporal` (+0.071).** Coherent mechanism: `temporal`
  scores recall PLUS the superseded turn not outranking the current one, and
  semantic similarity cannot distinguish a stale value from a fresh one — an old
  "my CGPA is X" embeds almost identically to the new one. Plausible, not proven.

⚠ **Scope.** Retrieval level only, one store, no generative model. It says what
reaches the prompt, never whether answers improved — and those two disagreed
repeatedly this session. **And it does not touch the WRITE path at all**: the
earned-lossless density decision (164 of 293 turns lossless, 129 summarised), the
B2 retrieve/don't-retrieve gate, decay, and the context ledger were not under
test. Every claim here concerns read-path legs.

---

## ⛔⛔ 2026-08-21 — EVERY RETRIEVAL NUMBER FROM 2026-08-20 WAS MEASURED OFF THE PRODUCTION PATH

**The harnesses pass `scope=None`; production passes a scope carrying
`conversation_id`. That single difference changes which legs run.**

`orchestrator.retrieve` derives its conversation filter from the SCOPE only:

```python
conv_id = None
if scope and "conversation_id" in scope:
    conv_id = scope["conversation_id"]
```

The `conversation_id` PARAMETER that `retrieve()` also receives is never used for
this. `score_typed.py` and `answer_probes.py` both call with `scope=None`, so
`conv_id` is None, and `_batch_summary_lookup`'s own-conversation branch — gated
behind `if conv_id:` — never runs. `src/services/retrieval_svc.py:152` passes
`scope=scope` with `conversation_id` populated.

**Measured side by side, same probe, same store, same orchestrator:**

| call | fragments returned |
|---|---|
| `scope=None` (both harnesses) | `{episodic: 11, codex: 1, procedural: 5}` |
| `scope={"conversation_id": …}` (production) | `{episodic: 6, **batch_summary: 2**, procedural: 2}` |

### What this retracts

- **"`batch_summary` never reaches the prompt — 0 of 1,925 budget lines."**
  FALSE as a statement about ICE. The leg works; the harness disabled it. All
  1,925 observations come from `scope=None` runs.
- **"A partnerless leg cannot win RRF."** That mechanism was inferred to explain
  a zero that had a different cause. Withdrawn — it may still be true, but
  nothing here is evidence for it.
- **`summary_synthesis` = 0.303** was measured with the summary leg switched
  off. It is not a measurement of summarisation.
- **The per-leg ablation's `batch_summary` row (0.000 delta)** compared "leg off"
  against "leg already off". It measured nothing.
- **⚠ AND IT REACHES FURTHER THAN batch_summary.** The whole fragment mix
  differs — codex 1 → 0, procedural 5 → 2, episodic 11 → 6. **Every retrieval
  measurement of 2026-08-20 — the per-leg ablation, the codex ablation, and both
  arms' typed scores — was taken on a configuration production does not use.**

### What SURVIVES

Comparisons where both sides shared the same defect remain internally fair:
- **arm A vs arm B** (both `scope=None`) — the A9b verdict stands.
- **the codex ablation's three conditions** (all `scope=None`) — the ordering
  between conditions stands; the ABSOLUTE claim "codex is net-harmful in ICE"
  does not, because production runs codex under a scope that changes what it
  returns.
- Everything not routed through the orchestrator: the codex truth judging (20%
  correct, 25% reversed), the entity-reality split, graph shape, gold
  consistency. Those read the store directly.

### The rule this breaks, verbatim from CLAUDE.md

> *"Is everything there? Did the harness call what the real path calls? A scorer
> skipping the budget setter measures something — just not ICE."*

Eighth instrument defect of the cycle (TRAPS #41), and the most expensive: it
did not corrupt one number, it corrupted a class of them, and it survived a
whole day of cross-checking because every harness shared it — so the harnesses
agreed with each other.

**Before ANY retrieval number is trusted again: give the harnesses the
production scope and re-run.** Until then the 2026-08-20 retrieval tables are
provisional and marked so.

---

## 2026-08-22 — the extractor fixes are a NULL on truth, and the reversals are generated

**⚑ THE HEADLINE IS A NULL, AND IT IS THE MOST USEFUL RESULT OF THE DAY.**

Six write-path defects were found and fixed (G51, G45, A8, G50, G57 ×2), all
verified to change the *mechanisms* they targeted. A full 293-turn re-seed on
identical inputs — same model (`qwen3:4b-instruct`), same NER tier
(`background`), same promotion flag, **only the code differs** — then judged by
the same judge (`deepseek-v4-flash`, n=200, seed 20260820) as arm B.

**Store-level mechanism changes, arm B → `ner-b-postfix` (rates, not counts):**

| | arm B | postfix |
|---|---|---|
| edges expired | **7.99%** | **1.93%** |
| negations stored as `negated=True` | **0.11%** | **6.04%** |
| relation `in` (copula rewritten as containment) | **7.70%** (the #2 relation) | **0.013%** (1 edge) |
| `bg_model_output_truncated` | 138 | 73 |
| relations canonicalised | 4,012 | 1,323 |

**Truth-judge outcome — nothing moved.** n=190 judged of 200 (10 unreturned):

| | arm B | postfix | z | |
|---|---|---|---|---|
| correct | 20.0% | 15.8% | −1.09 | noise |
| reversed | 25.0% | 27.9% | +0.65 | noise |
| wrong | 37.0% | 42.6% | +1.14 | noise |
| malformed | 12.5% | 8.9% | −1.14 | noise |

**Not one difference clears significance at n≈200.** Every mechanism fix
worked and the graph is no more TRUE than before.

**⚑ WHAT THE NULL RULES OUT, WHICH IS THE POINT.** The merge guard blocked
**1,611 of 4,012** real canonicalisation merges — including `is` → `in` ×942,
`can` → `cannot` ×18, `is_the_same_as` → `is_not_the_same_as` ×14 — every one
of them a direction or polarity change. **The reversal rate did not move.** So
reversals are **not** manufactured downstream by canonicalisation; they are
generated that way. That is a hypothesis eliminated by measurement rather than
by argument, and it redirects the work.

⇒ The extraction prompt had **no direction instruction at all**: nothing named
which argument goes in `subject`, all three examples were active SVO, and
`ALLOWED_RELATIONS` mixes voices inside a single category (`created` active
beside `founded_by`; `manufactured_by` passive-only). G59 adds the rule; the
A/B is running.

**`extraction_confidence` is inverted, and worse than recorded:**

| | grounded (0.9) | rejected (0.35) | gap |
|---|---|---|---|
| arm B | 15.0% correct | 25.0% | 10.0 pts backwards |
| postfix | **2.5%** correct | **18.1%** | **15.6 pts backwards** |

Grounded triplets are correct **1 time in 40**. The retrieval trust floor
admits ~100% of that tier and excludes 94% of the more accurate one. This is
not a weak signal to tune — it is anti-correlated, and it must stop being used
as a truth prior regardless of what any model comparison finds.

**⚠ AN UNATTRIBUTED 24% VOLUME DROP — do not read the counts.** The re-seed
extracted **8,208** triplets against arm B's **10,806** on identical turns
(50.0 vs 65.9 per turn), giving −24% entities and −28% edges. The cause is
NOT established. My merge guard compounds within a turn — `_known_rels` grows
as relations are accepted, so blocking one merge changes every later decision,
and a static replay predicted 2,401 surviving canonicalisations where only
1,323 occurred. But blocking merges should mean *less* dedup and therefore
*more* triplets, which is the opposite sign. **Reported as unexplained.** The
rate table above is normalised and unaffected; the absolute counts are not.

⇒ This is why the noise floor is being measured before any model sweep: if two
identical runs can differ by 24%, **A12's eight-model ranking — one run per
model — was partly noise**, on top of being scored by a circular metric
([TRAPS #45](TRAPS.md)) and settled by a single subagent read
([TRAPS #31b](TRAPS.md)).

**Artifacts:** `experiments/curation_files/judgements/codex_quality_ner-b-postfix.json`,
snapshot `ner-b-postfix`, `logs/reseed_ner-b-postfix.log`,
`logs/judge_codex_postfix.log`.

**What this does NOT show:** anything about retrieval or answers. This is
store-level only — the judge reads each stored triplet against its own source
turn. No retrieval ran, no answering model was involved.


---

## 2026-08-23 — ⛔ THE "extraction_confidence IS INVERTED" FINDING IS WITHDRAWN

**It was a small-sample artifact, and it was recorded as settled in three
places.** The claim — grounded-0.9 triplets are LESS often true than
rejected-0.35 ones — came from flat judge sampling that drew only 40 grounded
triplets. One correct in forty read as 2.5% and looked dramatic.

Re-measured with turn-stratified sampling (`--per-turn 3`, 124 distinct turns
per arm instead of 76), on TWO independent seeds of the same configuration:

| tier | `dir-true-run1` | `dir-true-run2` |
|---|---|---|
| grounded (0.9) | **22.3%** (n=94) | **12.2%** (n=115) |
| ungrounded (0.7) | 21.4% (n=14) | 23.1% (n=13) |
| rejected (0.35) | **15.0%** (n=260) | **15.0%** (n=240) |

**The two runs disagree on the DIRECTION.** Run 1 makes grounded look better,
run 2 makes it look worse, and `rejected` is identical at 15.0% both times.

⇒ **`extraction_confidence` carries no reliable signal about correctness — in
either direction.** Not inverted. Not working. Uninformative. The practical
advice is unchanged (do not use it as a truth prior) but the reason is
different, and the difference matters to anyone deciding what to do with the
field.

⚠ **What produced the error.** A 40-triplet subgroup was quoted as a finding
without an interval. At that size the 95% interval on a rate near 15% is about
±11 points, which spans every number in the table above. This is the same
failure the session was convened to fix, committed by the session fixing it.

### And the RUN-TO-RUN variance question is answered

Same configuration, two independent seeds, full turn coverage:

| | correct | interval |
|---|---|---|
| `dir-true-run1` | **17.1%** | ±6.6 |
| `dir-true-run2` | **14.4%** | ±6.2 |

**2.7 points apart, well inside either interval.** So the aggregate rate is
stable run to run; the wobble that made 20.4% and 10.0% look like different
results was SAMPLING, not the model's cross-process nondeterminism.

⇒ **No heavy bootstrap is needed.** One run per arm is adequate **provided
every run reports its interval and covers turns rather than triplets.** That
keeps the model sweep affordable.

**Current honest figure for graph correctness: 14–17%, ±6 points.** Not 20%,
not 11%, not 10% — one number, measured properly, for the first time.

---

## 2026-08-23 — ICE cannot reproduce itself, and three of the four causes are fixed

**Nobody had ever run the same seed twice and compared.** Every arm comparison,
ablation and model ranking assumed it. `scripts/z1/check_reproducible.sh` now
answers it in ~15 minutes and names the first divergent turn.

**Baseline (no fixes):** two identical 20-turn/conversation seeds — same
corpus, model, NER tier, config — diverged on **31 of 48 turns**, 2,108 edges
against 1,740 (**−17%**), starting at **turn 1**. **Not noise, and not
near-misses: the two runs held different subjects making different claims**, and
one run's were visibly worse. The triplets carry personal specifics, so they
live at **`[PRIVATE:repro-divergence-examples]`** in `docs/PRIVATE_CONTEXT.md`
(gitignored) — read them before concluding the divergence is harmless.

**Three causes found and fixed** (`f76b65c`):

| # | cause | effect of fixing it |
|---|---|---|
| 1 | the FIRST extraction call after Ollama loads a model differs — **633 chars cold vs 650 warm**, identical input, temperature 0 | edge gap −17% → **−2.9%** |
| 2 | `known_relations()` had **no `ORDER BY`**; Postgres row order shifts as the table grows, and `canonical_relation` takes an argmax over that list where real candidates sit thousandths apart (`has`/`have` 0.9469) | turns 1–19 reproduced **exactly**; divergence moved to turn 20 |
| 3 | three unordered reads in `get_or_create_entity` — alias lookup, the **many-to-one-by-design** merge_key tier, and the promotion scan | (see the caveat below) |

⚠ **Fix 3 did not help and may have hurt** — after it, divergence returned to
turn 1 and 31 of 48 turns. Ordering by `canonical_name` changed *which* entity
wins a merge group, which cascades differently. Recorded as applied-but-
unvalidated: the ordering is still correct in principle (an unordered `.first()`
over a many-row match is a defect regardless), but it is not what fixes
reproducibility.

### ⛔ THE FOURTH CAUSE IS NOT FIXABLE AT THIS LAYER

Minimal repro, 3 turns, 90 seconds: **9 raw model responses per pass, 6
identical, 3 different**, first divergence on call 1 with the model already
warm.

```
pass X: [{"object":"detroit","relation":"manufactured_by","subject":"ford"}]
pass Y: [{"object":"detroit","relation":"lives_in","subject":"ford"}, …]
```

Within ONE process, 12 alternating warm calls were byte-identical — which is
why an earlier test wrongly concluded the model was deterministic. **Across
processes it is not.** Temperature 0 fixes sampling, not logits: batch
composition and kernel selection differ per process, and two near-tied relation
words then resolve differently.

⇒ **Exact reproducibility is unreachable. The strategy changes from removing
variance to measuring it.**

### But the aggregate RATE is stable, which is what actually matters

Two independent seeds of one configuration, judged with full turn coverage:

| | correct | 95% interval |
|---|---|---|
| `dir-true-run1` | **17.1%** | ±6.6 (124 turns) |
| `dir-true-run2` | **14.4%** | ±6.2 (124 turns) |

**2.7 points apart, inside both intervals.** The model's cross-process
nondeterminism changes WHICH triplets exist without moving the rate much.

⇒ **No heavy bootstrap is needed.** One run per arm suffices **provided it
reports its interval and samples TURNS rather than triplets**. That keeps the
model sweep affordable, which was the open question.

**Artifacts:** `logs/reproducibility_check.log`, `logs/repro_after_*.log`,
`experiments/curation_files/fingerprints/`, judgements `*-perturn`.

**What this does NOT show:** anything about retrieval or answers. Store-level
only.

---

## 2026-08-23 — ⛔ THE DIRECTION RULE (G59) IS A NULL. "+9.4 pts" IS WITHDRAWN

> ## ⚠⚠ SUSPENDED THE SAME EVENING — DO NOT CITE THE NULL BELOW YET
>
> **The judge that produced every number in this entry cannot count reversals,
> and `reversed` is the exact label the direction rule targets.** Found hours
> later when the maintainer read a `wrong` verdict and recognised it as an
> inversion (`villainess --contains--> file 1`; the turn says File 1 contains
> the villainess). Three defects in the instrument, all measured:
>
> 1. **The source was truncated at 6,000 chars** — 40% of turns with a `wrong`
>    verdict are longer, **30% of all source text was never shown**, and 21
>    triplets were marked "not supported by the source" while the entity being
>    asked about sat past the cut. The identical cap had already been removed
>    from `judge_answers.py` and was never carried across.
> 2. **Batched calls silently dropped verdicts** — "include every number exactly
>    once" failed 6 times per arm.
> 3. **Reversals were labelled `wrong`** despite the rubric explicitly ordering
>    the opposite.
>
> Re-judged with one triplet per call, full turn, `muse-spark-1.2-contributor`
> (reasoning): **`reversed` is 46.1%, not 32%.** Nearly half the graph has the
> arrow backwards — the largest defect ICE has, and every prior measurement
> understated it.
>
> ⇒ The null may well survive — both arms ate the same defects, so it is
> symmetric — but **it was measured by an instrument blind to its own subject**
> and is not quotable until all four `dir-*` arms are re-judged.
>
> ⚠ Also do not substitute the CoE 35B-A3B numbers (`correct` 32–35%). That
> model is **systematically blind to direction**: on 50 sampled triplets it
> answered `correct` while its own written justification described the reverse
> (`structure --built--> creator` → *"the creator built the structure"*). It is
> a 3B-active MoE with no reasoning and is not a usable judge. See the
> 2026-08-23 judge-instrument entry below.

**The last intervention that looked like it moved correctness does not.**

G59 added rule 8 to the codex extraction prompt — an explicit instruction about
which argument goes in `subject`, plus the `_by` passive mirror case and a
read-back check. It was built because ~25–28% of stored triplets are REVERSED
(`india --lives_in--> maharashtra`) and blocking 1,611 direction-changing
canonicalisation merges had left that rate flat, which said reversals are
GENERATED rather than merged in — and the prompt had no direction instruction
at all while all three of its examples were active SVO.

Four stores, seeded 2026-08-22 with identical code and config, varying **only**
the `codex_extraction_direction_rule` flag; two seeds per arm. Judged
`deepseek-v4-flash`, seed `20260820`, `--per-turn 3` (full turn coverage),
blind to arm and to confidence tier.

| arm | rule | triplets | turns | correct | reversed |
|---|---|---|---|---|---|
| `dir-true-run1` | ON | 368 | 124 | 17.1% ±6.6 | 32.9% ±8.3 |
| `dir-true-run2` | ON | 368 | 124 | 14.4% ±6.2 | 32.3% ±8.2 |
| **treatment pooled** | ON | 736 | 248 | **15.8% ±4.5** | **32.6% ±5.8** |
| `dir-false-run1` | OFF | 365 | 124 | 16.7% ±6.6 | 31.5% ±8.2 |
| `dir-false-run2` | OFF | 365 | 124 | 14.2% ±6.2 | 33.7% ±8.3 |
| **control pooled** | OFF | 730 | 248 | **15.5% ±4.5** | **32.6% ±5.8** |

**Effect on correctness: +0.28 points, 95% CI [−6.1, +6.7], z = 0.09.**
**Effect on `reversed` — the label the rule was built to move: +0.01 points,
z = 0.001.** The rule changed the reversal rate by essentially nothing.

Intervals use `report_power`'s own formula (`se = sqrt(p(1−p)/n_turns)`, each
turn one effective observation) so this is the method the arms were reported
with, not a second one. **Smallest detectable effect at this design: ±6.4 pts.**
A real effect smaller than that would not be visible here — but +9.4 was, and
it is not there.

### Where "+9.4 pts" came from — sampling, on an unchanged store

`dir-false-run1` judged two ways, same store, same judge, same seed:

| sampler | triplets | turns | correct |
|---|---|---|---|
| flat `--n 200` | 191 | ~76 | **11.0%** |
| `--per-turn 3` | 365 | **124** | **16.7%** |

**A 5.7-point swing with nothing changed but which turns were sampled.** That
gap is the whole of the claimed effect. ⇒ [TRAPS #46](TRAPS.md) again: a figure
from one sampler compared against a figure from another.

### What this means, and what it does not

Two independent interventions on the extractor prompt — the six write-path
mechanism fixes (2026-08-22) and now the direction rule — **each verifiably
changed the mechanism and neither moved truth.** That is no longer a
disappointment; it is a result about location. Prompt-level instruction is not
where the 14–17% ceiling lives.

⇒ **The Z1 §2 item-2 small-model sweep loses its premise.** It was queued on the
theory that A12's eight models all failed identically *because* they ran on the
broken prompt. The prompt is now fixed in two separate ways and correctness did
not move, so "re-run the sweep on the fixed prompt" no longer predicts a
different outcome. Re-scope it before spending ~2h + judging.

**What this does NOT show:** nothing about retrieval, answers, or the other
labels — `wrong` (37.8%/39.7% control) is the largest bucket and is untouched by
this test. It does not show the rule is *harmful*; it shows the effect is
indistinguishable from zero at ±6.4 pts. It does not license deleting the flag
— that is a user decision, not an implication of this measurement.

**Run losses:** four judge turns failed on provider 5xx (run1 turns 58, 108;
run2 turns 8, 20), costing 6 triplets per arm — 1.6% of turns, symmetric across
arms, immaterial to the rate.

**Artifacts:** `experiments/curation_files/judgements/codex_quality_dir-false-run{1,2}-perturn.json`,
`logs/judge_dir-false-run{1,2}-perturn.log`,
`scripts/oneoff/g59_compare.py`.

---

## 2026-08-24 — ⚑ NuExtract3 vs ICE's extractor: +60 POINTS, AND ZERO REVERSALS

**The first intervention in this project that has moved graph correctness.**

20 random `lossless` turns (600–6,000 chars, seed `ab-nuner-20260824`) drawn from
the `dir-false-run1` store. Both extractors ran on the **same turns** with the
**same NuNER-confirmed entity list**. ICE side calls `extract_triplets()` — the
real production function, not a reimplementation. Flags: direction rule ON,
entity-shape rule OFF, open vocabulary ON, `max_tokens` 1200, NER tier
`background`.

Truth labels are the **maintainer's own, written blind** — the extractor was
hidden, facts from both were shuffled and paired by turn. This is the only human
ground truth the project has, and it outperformed every model judge tested.

| label | ICE `qwen3:4b-instruct` | `NuExtract3-Q8_0` |
|---|---:|---:|
| **correct** | **10%** | **70%** |
| **reversed** | **40%** | **0%** |
| wrong | 40% | 10% |
| vacuous | 10% | — |
| malformed | — | 20% |

**Δ correct = +60 pts, 95% CI [+26, +94]** (n=10 per side).
**Δ reversed = −40 pts. NuExtract3 emitted no reversal at all.**

For scale, every prior attempt on this number: six write-path mechanism fixes
(null) · the direction rule G59 (null) · eight background models 4B→26B (all
failed identically) · a binary direction-verification pass (57% against a 50%
coin flip). This is the first non-null.

### Structural counts over the same 20 turns (no judge involved)

| | ICE | NuExtract3 |
|---|---:|---:|
| facts | 842 (42/turn) | 224 (11/turn) |
| grounded by NuNER | 14.6% | **27.2%** |
| unusable entity names | 130 (15%) | **8 (3.6%)** |
| distinct relations | 381 | **118** |
| singleton relations | **64.8%** | 71.2% |
| sec/turn | 17.5 | **10.3** |
| turns failed | 0 | 1 |

### What this does NOT show

- **n=10 per side.** The effect clears zero; the *magnitude* does not. Read it
  as "large", never as "+60 points".
- **Pre-write-path, both sides.** No canonicalisation, merge keys, dedup or
  conflict resolution ran. ICE emitted the identical wrong fact twice
  (`person --is_centerpiece--> building`), which the write path would have
  merged. ⇒ **do not cross-quote these against the store's ~20%**; the store
  benefits from cleanup these numbers do not.
- **Nothing about answers.** Whether truer facts change what a user reads is
  [G66](ROADMAP.md#g66), still unmeasured.
- **Nothing about NuExtract3 in production.** It has no grounding of its own, no
  canonicalisation, and it emitted `nobody`/`no one`/`no body` as three subjects
  on one turn — three unconnected nodes. Its 20% malformed rate is real.

### The immediate follow-up

`codex_extraction_entity_shape_rule` (prompt rule 7) targets NuExtract3's one
measured weakness verbatim — *"a subject or object must be a NOUN PHRASE NAMING
A THING, never a clause"* — against the clause-subject and clause-relation
cases in `PRIVATE_CONTEXT.md` `[PRIVATE:g63-malformed-examples]`. It is
**built, OFF by default, and has
never been measured.** Cheapest available next win.

**Artifacts:** `experiments/curation_files/extractor_ab/head_to_head.json`
(1,066 facts), `BLIND_SHEET.md`, `BLIND_KEY.md`,
`logs/head_to_head_nuner.log`, `scripts/oneoff/extractor_head_to_head.py`,
`scripts/oneoff/build_extractor_blind_sheet.py`.

### 2026-08-24 (same day, later) — IT IS THE MODEL. The format alone does not transfer.

The entry above changed **two variables at once** — model *and* extraction path —
and I presented it as a model result. It was not cleanly one. The maintainer
caught it. Two follow-up runs on the **same 20 turns** fill the matrix:

| model | path | result |
|---|---|---|
| `qwen3:4b-instruct` | ICE's production path (9 rules, chunking, entity list, 1200 tok) | 10% correct · 40% reversed · 842 facts |
| `qwen3:4b-instruct` | NuExtract3's template, no rules, no entity list, whole turn | ⛔ **unusable** — 14/20 parsed, **43 facts total**, subjects and relations are clauses |
| `NuExtract3-Q8_0` | ICE's production path | ⛔ garbage — reversals and nonsense (`affordable education --offers--> europe`; more in `PRIVATE_CONTEXT.md` `[PRIVATE:g63-malformed-examples]`) |
| `NuExtract3-Q8_0` | its own template + entity list | ✅ **70% correct · 0% reversed** |

**Each model is usable ONLY in its own format, and neither survives the other's.**

qwen given the bare template stops extracting and starts writing analysis:
`the user's deep-seated doubt about their own suffering --is rooted in--> a fear
of not being allowed to exist`. Clause subject, clause relation, unusable as a
node.

⇒ **The +60 points is attributable to the MODEL, not to removing ICE's
machinery.** The hypothesis that ICE's guards were the problem was tested
directly and is **false**: strip them from qwen and it gets dramatically worse.

⇒ **⚑ THE REFRAME THAT MATTERS: ICE's extraction machinery is a workaround for
the wrong model.** The nine-rule prompt, the examples, the entity whitelist —
all of it compensates for a generalist that was never built for extraction.
NuExtract3 does not need the *shaping* because it is trained for the task. The
*checking* machinery keeps its value regardless — grounding, the entity-name
gate and canonicalisation, because NuExtract3's one measured weakness (20%
malformed) is exactly what those catch.

**What this does NOT show:** NuExtract3 without the entity list was never run —
that cell of the matrix is empty by decision, not oversight. And nothing here
touches whether any of it changes an ANSWER ([G66](ROADMAP.md#g66)).

**Artifacts:** `experiments/curation_files/extractor_ab/qwen_template.json`,
`logs/qwen_template_test.log`.

### 2026-08-24 — ⛔ THE RELATION VOCABULARY IS WHAT DESTROYS TRUTH. Isolated.

Two blind rounds by the maintainer, same protocol (arm hidden, facts shuffled,
paired by turn, full source inline), drawn from the 60-turn G63 sweep.

**Round A — three-way, 30 facts:**

| arm | correct | reversed | wrong |
|---|---:|---:|---:|
| `ice_baseline` | 20% | 20% | 40% |
| **`nu_ref`** (bare template + entity list) | **50%** | **0%** | 20% |
| `nu_combined` (+ vocabulary + canon rule + chunking) | 20% | 20% | 60% |

**Round B — each guard ISOLATED, 29 facts:**

| arm | correct | reversed | vs `nu_ref` pooled (60%) |
|---|---:|---:|---|
| **vocabulary only** | **22%** | 11% | **−38 pts, 95% CI [−72, −3] — CLEARS ZERO** |
| canonicalisation rule only | **80%** | 0% | +20, CI [−13, +53] — no drop |
| chunking only | **80%** | 0% | +20, CI [−13, +53] — no drop |

⇒ **The relation vocabulary is the cause. Canonicalisation and chunking are
innocent.** `nu_combined` (20%) ≈ vocabulary alone (22%), because the vocabulary
dominates the damage.

**The mechanism, visible in the labels:** given a list, the model reaches for a
listed word when no listed word fits.

```
MIT/Stanford/CMU --works_at--> Google India        wrong
MIT/Stanford/CMU --works_at--> OpenAI              wrong
University of Melbourne --employs--> IBM           reversed
authors --cites--> authors                         vacuous
AI/ML paper --writes--> AI/ML paper                vacuous
```

The last two glue the SAME entity to both ends of a dictionary verb. This is
[G45](ROADMAP.md#g45)'s 67.8%-out-of-vocabulary finding showing up as *wrong
facts* rather than as dropped ones.

### ⚑ Pooled headline, two independent rounds

| arm | round 1 | round 2 | pooled (n=20/side) |
|---|---:|---:|---:|
| ICE `qwen3:4b` | 10% | 20% | **15%** [−1, 31] |
| **NuExtract3 bare** | 70% | 50% | **60%** [39, 81] |

**Δ +45 pts, 95% CI [+18, +72] — clears zero.**
**Reversed: 0 of 20 for NuExtract3, 6 of 20 for ICE.**

### ⚠ What this does NOT show

- **The winning combination was never run.** Best config is bare template +
  entity list + canonicalisation rule + chunking + 3000 tokens — **no
  vocabulary**. `nu_combined` included the vocabulary, so that config is
  untested *as a combination*; its parts each tested at 80% and 80%.
- **n is 9–10 per arm.** Intervals are ±25–27 pts. The vocabulary drop clears
  zero; the canon/chunk "improvements" do not and must not be quoted as gains.
- ⚠ **A verdict was retracted getting here.** The vocabulary was first called a
  WIN on its own target metric (in-vocab 50.9% → 75.2%). **It hit its target and
  made the output false.** Hitting the metric you aimed at is not improving the
  thing. A concentration test run to check the jamming hypothesis found nothing
  (19.5% vs 18.2%) — because it assumed jamming concentrates on ONE word, while
  the real pattern spreads across many dictionary words. Wrong test, not
  exonerating evidence.
- Nothing here touches ANSWERS ([G66](ROADMAP.md#g66)).

**Artifacts:** `experiments/curation_files/extractor_ab/` — `g63_sweep.json`,
`THREEWAY_SHEET.md`/`_KEY.md`, `ISOLATE_SHEET.md`/`_KEY.md`.

### 2026-08-24 — Relation-merge safety, measured on the REAL vocabulary

The 0.90 canonicalisation threshold was calibrated on **30 hand-built pairs**
([FEATURE_INVENTORY](FEATURE_INVENTORY.md), moved 0.82 → 0.90). Re-measured
against the **700 distinct relations the 60-turn G63 sweep actually produced**
(`ice_baseline` + `nu_ref` + `nu_chunked`), encoded with the live embedder.

| | |
|---|---|
| pairs at or above 0.90 | **76** |
| blocked by `_is_inverse_pair` | **62 (82%)** |
| allowed to merge | **14** |
| of those, unsafe | **3** |

⇒ **Measured wrong-merge rate: 3 of 76 above-threshold pairs (~4%).**

**The 11 safe merges are inflection/spelling variants** — `attracts`/`attract`
(0.9699), `has`/`have` (0.9523), `participates_in`/`participated_in` (0.9350),
`makes`/`made` (0.9098). Exactly what canonicalisation exists to collapse.

**The 3 unsafe ones, all at the bottom of the band:**

| sim | pair | why it is not a merge |
|---|---|---|
| 0.9118 | `poststudy_work_visa` / `duration_of_post_study_work_visa` | "has a visa" vs "how long it lasts" |
| 0.9053 | `prioritises` / `can_prioritize` | **modal** — does vs can |
| 0.9043 | `has_good_job_opportunities` / `has_higher_job_prospects` | absolute vs comparative |

⚠ **Raising the threshold does not fix it.** All three sit in 0.904–0.912, and
so do legitimate merges (`did`/`does` 0.9174, `makes`/`made` 0.9098,
`compare`/`compares` 0.9089). Moving to 0.92 would block 3 bad merges and 5
good ones.

⇒ **The gap is structural, not numeric — and it is the same shape as a rule that
already exists.** `_is_inverse_pair` handles `can`/`cannot` deterministically
*because similarity cannot see polarity* (measured there: 0.9134).
`prioritises`/`can_prioritize` is the same class — a **modal changing the
claim** — and is not covered. Quantifier/comparative pairs
(`good`/`higher`) are a second uncovered class.

**What this does NOT show:** only pairs above 0.90 were examined, so nothing is
said about relations that SHOULD merge and score below it (the false-negative
side). And `_is_inverse_pair`'s own comment stands — *"a hand list cannot
enumerate an open vocabulary"* — so the 82% block rate is a property of this
corpus's relations, not a guarantee.

**Artifact:** computed from `experiments/curation_files/extractor_ab/g63_sweep.json`.

---

## 2026-08-26 — ⚑ THE SUMMARY TRUST GATE SELECTS *FOR* FABRICATION, and the background-model bake-off

**Run:** `scripts/z1/bg_model_bakeoff.py` (11 candidate models × 9 background
jobs) then `scripts/z1/judge_summaries.py` (faithfulness).
**Corpus:** `data/simulation/simulation_full.jsonl`, 30 dense turns sampled
deterministically (seed 20260826), median 3,572 chars. Store-backed jobs ran
against `dir-false-run1`.
**Artifacts:** `experiments/curation_files/bakeoff/` — `bakeoff_*.json`,
`judge_calibration_20260826T091214Z.json`,
`summary_verdicts_20260826T093810Z.json`. ⚠ gitignored.

### The headline, and it is not the model choice

`summary_coverage` is the gate deciding whether a stored summary **replaces the
raw turn** in the assembled prompt (`inject_raw`, threshold 0.7). Graded by a
judge that passed its own controls first — planted fabrications caught **15/16**,
verbatim copies of the source called faithful **16/16**:

| verdict | n | mean coverage | clears the 0.7 gate |
|---|---|---|---|
| faithful | 29 | 0.803 | 23/29 = 79% |
| **fabricated** | 13 | **0.914** | **13/13 = 100%** |

**Every invented summary cleared the gate; a fifth of the honest ones did not.**
Of everything the gate admits in place of the raw turn, **36% contains
invention.** The direction holds *inside* every model tested — fabricated
summaries out-score faithful ones on coverage: `qwen3:4b-instruct` 0.901 vs
0.729 · `gemma4:e4b` 0.891 vs 0.791 · `ministral-3:8b` 0.933 vs 0.912 — and
across models the fabrication rate tracks coverage monotonically (0.859→43%,
0.814→29%, 0.768→21%).

**Mechanism:** `_summary_llm_call` instructs *"every one of these MUST appear
verbatim"*; a model that cannot ground a must-term invents context to carry it,
and coverage scores it for having done so. The instruction manufactures the
defect the metric rewards — the same prompt/metric coupling as
[TRAPS #45](TRAPS.md), one layer down.

### Model results — 5 of 11 disqualified on deterministic evidence

Disqualified: `nemotron-mini:4b` (coverage 0.160, 59% of slicer output invented,
fold collapse), `granite4:tiny-h` (0.174, no prose on 24/30),
`granite4:micro` and `mistral-nemo` (no `Abstract:` line on 29/30 and 26/30),
`lfm2.5:8b` (prose at **95% of source length** — copying, not summarising — and
reconciler 0/9).

Finalists, weighted by how often each job actually fires
(`settings.maintenance_intervals` + `runtime.py:JOBS`): the turn summary and
procedural extraction run on **every turn**; doc-kind and the raw slicer fire
only if the user ingests documents; the reconciler fired **0 of 106** in
production (G62); the maintenance agent has never run (G67).

| model | cov | no-Abstract | procedural | fabricated | VRAM beside NuExtract3 |
|---|---|---|---|---|---|
| `gemma4:e4b` | 0.768 | 1/30 | 0.750 | **21%** | 14.8 G |
| `qwen3:4b-instruct` | 0.814 | **0/30** | **0.750** | 29% | **7.7 G** |
| `ministral-3:8b` | **0.859** | 2/30 | 0.625 | **43%** | 11.2 G |

⚠ **`ministral-3:8b` had the best coverage and the worst faithfulness** — a
direct consequence of the defect above, and a warning against reading the
coverage column as quality.

### What this does NOT show

- **n=14 per model for faithfulness.** 21% vs 29% is **not resolvable**;
  separating them at 95% confidence needs ~400 per arm. Only the gate finding
  and the disqualifications are robust at this size.
- **One judge**, `deepseek-v4-flash`, which missed 1 of 16 planted fabrications
  ⇒ the true fabrication rate is **at least** these numbers, not at most.
- **Synthetic corruption is a floor.** A spliced sentence is more obvious than a
  subtly wrong paraphrase; the judge's real-world sensitivity is likely lower.
- **Nothing here tests whether removing the must-preserve instruction helps.**
  That is the experiment the mechanism implies and it has not been run.
- The three store-backed jobs ran on 180 turns of `dir-false-run1`; `procedural`
  and `batch_summary` required clearing their own markers first, because both
  are idempotent and had already run.

## 2026-09-14 — v3 source-rendered graph and cold restoration

Validation: `uv run python tests/support/disposable_database.py -m pytest
 tests/smoke tests/test_codex_claims.py tests/test_retrieval_write_boundary.py
 tests/test_graph_retention.py tests/test_source_lifecycle.py -q --tb=short`:
283 passed. `uv run python tests/support/disposable_database.py
 tests/test_timescope.py`:61 passed. Disposable databases removed; synthetic
fixtures only. These exercise source-attributed graph lines, filtered cached-note
replacement, warm/cold hash checks, privacy/scope/deletion, original/NULL embedding
restoration without an encoder, and unknown timestamp provenance.

Initial temporal run58/61: one obsolete cached-payload anchor assertion and two
real restore failures from inserting NULL into required timestamp provenance.
Updated anchor lookup to rendered entity identity; fixed provenance to unknown.
No answer-quality, multilingual generalization or vector-baseline claim.
