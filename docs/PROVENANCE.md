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
| JSON-schema constrained decoding (`format`) | **NO** (native only) | documented upstream; OpenAI `response_format` is ignored by Ollama |
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
**The union was not tested** — the open measurement.

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
