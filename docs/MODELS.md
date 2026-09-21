# MODELS — everything this machine can reach

**What this is.** One place listing every model ICE can call, by serving path,
with what it is used for. Written 2026-08-17 because the answer was scattered
across `.env`, `models/model_registry.json`, PROVENANCE tables and Ollama itself,
and three different sessions have re-derived it.

**What this is NOT.** Not the routing table — `models/model_registry.json` decides
which model serves which topic/intent tags at request time, and `src/model_registry/`
owns that. Not a record of runs — PROVENANCE owns what was actually done.

**⚑ Hardware ceiling: 24 GB (RTX 5090 *Laptop*), ~23.5 GB usable.** Anything
above that is out regardless of how good it is, and this has already rejected
real candidates (below). It is a laptop, so **heat is the binding constraint on
concurrency** — a 27B at `--workers 3` reached 105 °C ([TRAPS #36](TRAPS.md)).

---

## 1. Roles — which model does which job

| job | model | serving | notes |
|---|---|---|---|
| **background (9 jobs)** | ⚑ **`gemma4:e4b`** — DECIDED 2026-08-27 | Ollama | Measured over 11 candidates × 9 background jobs, then faithfulness-judged. **Wins or ties every quality-judged job**: turn summary **83% faithful vs `qwen3:4b-instruct`'s 54%** (n=70, the job that runs on EVERY turn) · fold 25% vs 0% (n=12) · cluster naming 28.6% vs 23.8% (n=21) · doc-kind 1.000 · reconciler 0.778 · slicer invention 0.000 · **0 truncations vs qwen's 17**. ⚑ **AND IT IS SMALLER IN VRAM — 3.4 G resident vs qwen's 4.1 G** (`ollama ps`); the 9.6 G in `ollama list` is DISK, and E4B is mix-n-match so per-layer embeddings need not be resident. ⚠ **The background prompts are now tuned FOR this model** ([BG_LAYER_FIXES.md §0](specs/BG_LAYER_FIXES.md)) — swapping it is a re-measure, not a config edit. ⚠ Several jobs it 'wins' are still BAD in absolute terms (fold 33% fabricated, naming 62% wrong): those are mechanism defects, not model defects. |
| **codex extraction** | ⚑ **`hf.co/numind/NuExtract3-GGUF:Q8_0`** | Ollama | [G63](ROADMAP.md#g63). Has its OWN setting (`codex_extraction_model`) because one background pin served ten jobs and only this one wants a specialist. ⚠ **NOT** the codex conflict reconciler, which reasons and stays on the general model. |
| **probe generation** | `qwen3.8:27b` | Ollama | user's preferred local model (2026-08-17); supersedes `qwen3.6:27b` |
| **probe answering** | `gemma4:26b-a4b-it-q4_K_M` | Ollama | Chosen because it is deliberately **neither arm under test**, so it cannot favour its own summaries. ⚠ **NOT "A12's top-ranked" — that label was wrong twice over** (corrected 2026-08-22). A12 ranked it **THIRD** (`PROVENANCE.md:1485-1488`: *"The 26B is NOT the winner"*, behind `gemma4:e4b` and `qwen3:4b-instruct`), **and** A12 ranked models for *background extraction*, which is not this job. The independence argument is the real and sufficient reason. |
| **judging** | `deepseek-v4-flash` | **cloud** | paired A/B judge. ⚠ reasoning model — `reasoning_effort="none"` made it confidently wrong ([TRAPS #34](TRAPS.md)) |
| **retrieval reranking (v3)** | `Qwen/Qwen3-Reranker-0.6B` @ `e61197ed45024b0ed8a2d74b80b4d909f1255473` | local cached CrossEncoder | ON by default for ordering; rejection floor unset. CUDA float16 when available, CPU between calls. Short synthetic controls peak 1.178 GiB; not maximum-input footprint. Provision the pinned Hugging Face snapshot before use; request path is local-files-only and warns/falls back to fusion if unavailable. Qualified only on the controls in PROVENANCE 2026-09-12. |
| **source-support verifier (v3)** | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` @ `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7` | local HF cache, float32; CPU between calls | Selected 2026-09-14 after13 supported/17 unsupported synthetic controls; same30 pass production scorer/decision. Claim compression consumer uses0.95 policy threshold, complete512-token maximum, unknown retains full evidence. Broader language/long-source reliability unqualified. Short-pair peak1.762GB, transfer+inference1.397s, not whole-stack footprint. |
| **source-support NLI candidate (v3)** | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` @ `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c` | local HF cache | Downloaded and bounded-tested 2026-09-13; diagnostic only, not activated. Argmax falsely entailed3/14 unsupported controls; no threshold fit. Multilingual entailment/neutral/contradiction candidate; source attribution remains a separate structural requirement. Model card warns about float16 support; use float32 qualification. |
| **embedding** | `Qwen/Qwen3-Embedding-0.6B` | in-process | native **1024-dim**, frozen. Resident on the same card as everything else |
| **NER (pre-flight + codex whitelist)** | **MicroNER — ours** | in-process | `models/ner/ner_model.pt`, 234 KB, over the `slice384` MRL prefix |
| **NER (background)** | `numind/NuNER_Zero` | in-process | GLiNER-family zero-shot, 448.9M, deberta-v3-large. Clustering + `turn_density` only |
| **classification** | `ice_classifier_v4_schema2.pt` | in-process | MLP head, 27 logits (11 topic + 12 intent + 4 context) |

---

## 2. Local — Ollama (28 models, 267.5 GB on disk)

All Q4_K_M unless noted. Sizes are on-disk, not VRAM.

⚑ **This table is INVENTORY, not evidence — holding a model is not having tested it.** For which models were actually run, for which job, and with what verdict, see **[§5](#5--what-has-actually-been-tested--by-job)**. Roughly half of what is listed here has never been through any ICE arm.

| model | size | params | quant |
|---|---|---|---|
| `granite4:small-h` | 19.5 G | 32.2B | Q4_K_M |
| `qwen3-coder:30b-a3b-q4_K_M` | 18.6 G | 30.5B | Q4_K_M |
| `gemma4:26b-a4b-it-q4_K_M` | 18.0 G | 25.8B | Q4_K_M |
| **`qwen3.8:27b`** | 17.7 G | 27.3B | Q4_K_M |
| `qwen-coder:latest` | 17.4 G | 27.8B | Q4_K_M |
| `qwen3.6:27b` | 17.4 G | 27.8B | Q4_K_M |
| `rpmax-22b-16k` / `Mistral-Small-22B-ArliAI-RPMax` | 15.7 G | 22.2B | Q5_K_M |
| `HammerAI/cydonia-v4.3` | 14.3 G | 23.6B | — |
| `gpt-oss:latest` | 13.8 G | 20.9B | **MXFP4** |
| `Cydonia-24B-v4.3-heretic-v3` | 12.9 G | 23.6B | — |
| `gemma4:e4b` | 9.6 G | 8.0B | Q4_K_M |
| `gemma4:12b` · `12b-64k` · `12b-128k` · `12b-256k` | 7.6 G ea | 11.9B | Q4_K_M |
| `mistral-nemo:latest` | 7.1 G | 12.2B | Q4_0 |
| `qwen3-vl:8b-thinking` | 6.1 G | 8.8B | Q4_K_M |
| `ministral-3:8b` | 6.0 G | 8.9B | Q4_K_M |
| `qwen2.5:7b` | 4.7 G | 7.6B | Q4_K_M |
| `llama3:8b` | 4.7 G | 8.0B | Q4_0 |
| `granite4:tiny-h` | 4.2 G | 6.9B | Q4_K_M |
| `qwen3.5:4b` | 3.4 G | 4.7B | Q4_K_M |
| `nemotron-mini:4b` | 2.7 G | 4.2B | Q4_K_M |
| **`qwen3:4b-instruct`** · `qwen3:4b-instruct-bg` | 2.5 G ea | 4.0B | Q4_K_M |
| `granite4:micro` | 2.1 G | 3.4B | Q4_K_M |
| `tinyllama:latest` | 0.6 G | 1B | Q4_0 |

**Routing registry** (`models/model_registry.json`) currently names only 6:
`gemma4:26b-a4b-it-q4_K_M`, `qwen3-coder:30b-a3b-q4_K_M`, `qwen3.6:27b`,
`gemma4:12b`, `tinyllama:latest`, `qwen2.5:7b` — all priority 5, 64k context
(tinyllama 8k). ⚠ It still names **`qwen3.6:27b`**, superseded by 3.8.

---

## 3. Cloud / remote

| model | endpoint | env prefix | used for |
|---|---|---|---|
| `deepseek-v4-flash` | `https://opencode.ai/zen/go/v1` | `PROBE_*` | probe generation and the paired judge |
| **`Qwen3.6-35B-A3B-NVFP4-Fast`** | `https://ai.tcetcercd.in/v1` | `COE_*` | **CoE AI Gateway** — TCET campus, free, unused so far |

### CoE AI Gateway (added 2026-08-17)

Campus-hosted at Thakur College. **NVIDIA DGX Spark, 119 GB unified memory**,
serving **Qwen3.6-35B-A3B** (35B MoE / 3B active) at NVFP4 4-bit with
multi-token prediction. Apache 2.0. OpenAI-compatible; verified live
2026-08-17. Reachable from off-campus.

* **`owned_by: "vllm"`** — the gateway is itself vLLM-served, and it is serving
  **exactly the model this machine had to reject**: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
  is 25.0 GB against 23.5 GB usable VRAM (§4). So it reaches a capability tier
  the local box cannot.
* **The `model` field is ignored** — single-model gateway, any string routes.
* **Thinking is OFF by default.** Opt in per request via
  `extra_body.chat_template_kwargs = {"enable_thinking": True, "reasoning_effort": "medium"}`
  (`low` | `medium` | `xhigh`). ⚠ Relevant to [TRAPS #34](TRAPS.md), where
  disabling reasoning on a reasoning model made a judge **confidently wrong** —
  if this is ever used as a judge, leave thinking ON and give it room.
* Defaults: `max_tokens` 2048, `temperature` 0.7. Vision supported; video not.
* Errors: 401 bad key · 400 malformed · 502 model server down · timeout = busy.

* **Context window: 64k** (as understood 2026-08-17; not independently verified
  against the server — measure before relying on a long-context run).

### Where to use it — settled 2026-08-17

**Use it wherever it helps.** The student guide's "~15 students, be considerate"
wording reads stricter than the service's actual operating position; see
`[PRIVATE:coe-gateway]` in `docs/PRIVATE_CONTEXT.md` (gitignored) for why. An
earlier draft of this section treated that wording as a hard constraint and
over-stated it.

| workload | verdict |
|---|---|
| **⚑ a model's judgement where spinning up a local model is the only alternative** | ✅ **the standout use.** No VRAM, no model swap, no thermal load on a laptop card, no 30 s cold start — a one-off classification or sanity check becomes an HTTP call |
| anything needing >27B, or vision | ✅ the local ceiling is 27B and the 35B AWQ does not fit in 23.5 GB |
| probe generation | ✅ viable; Ollama also does it free, so pick on quality |
| re-seed (hundreds–thousands of calls) | ⚠ prefer Ollama — not policy, just that a long local batch has no failure mode involving someone else's server |
| **the paired judge** | ❌ **stay on `deepseek-v4-flash`** — maintainer's call 2026-08-17. The paid quota has reset, and keeping the judge model fixed preserves comparability with the existing verdicts |

Key lives in `.env` (`COE_API_KEY`), never in a tracked file.

⚠ **Usage-limited.** A 150-probe judge run died at **65** on the limit
(2026-08-16). Budget judge runs deliberately; local scoring is free and should
absorb every question it can answer. Override `PROBE_API_BASE_URL` /
`PROBE_MODEL` to run generators against Ollama instead and spend nothing.

---

## 4. vLLM / AWQ — tested, with verdicts

⚠ **The handoff's claim that "no note exists in this repo about which models have
issues under vLLM" is WRONG** — this table is [PROVENANCE.md:84](PROVENANCE.md)
and predates it. Corrected 2026-08-17.

`background_model_mode=dedicated` starts vLLM on :8002 (manual, power-user path).

**Working:**

| model | revision | throughput | notes |
|---|---|---|---|
| `Qwen/Qwen3-14B-AWQ` | `31c69efc…` | 1.77 rows/s | needs `enable_thinking: false` + `max_tokens 1400`, else ~10% truncated JSON |
| `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | `0ef577a5…` | **2.34 rows/s** | cleanest: 0.01% degenerate, 116 failures / 39,289 |
| `openai/gpt-oss-20b` | `6cee5e81…` | 1.20 rows/s | MXFP4, MARLIN MoE kernel; needs the largest token budget of the three |

**Rejected — do not re-pick:**

| model | verdict |
|---|---|
| `mattbucci/Qwen3.6-27B-AWQ` | **will not load** — "input size is not aligned with the quantized weight shape" on `visual.blocks.*`; misquantized vision tower |
| `jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym` | loads and serves, emits **token soup behind schema-valid JSON** (18/18 rows). The reason `label.is_degenerate` and its abort gate exist |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 25.0 GB > 23.5 GB VRAM |
| `hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4` | 24.7 GB — same |
| `cyankiwi/GLM-4.5-Air-AWQ-4bit` | 63.4 GB — not close |

**Not yet obtained:** an AWQ/GPTQ build of **`qwen3:4b-instruct`**, wanted so the
re-seed's bg model can run under vLLM. A 4B is the best vLLM case available here
— small enough to batch heavily inside 24 GB. ⚠ Note the precedent above: a Qwen
27B AWQ was already found broken, so verify any Qwen AWQ actually loads before
planning a run around it.

⚠ **vLLM cannot serve the GGUF weights Ollama holds** (its GGUF support is
experimental and slow), so every vLLM model is a *separate download*, and it
pre-allocates ~90% of VRAM for KV cache by default — tighter than Ollama for the
same model. Expect to tune `gpu_memory_utilization`.

---

## 5. ⚑ WHAT HAS ACTUALLY BEEN TESTED — by job

**Why this section exists (added 2026-08-22).** §2 lists what the machine
*holds* and §4 lists vLLM verdicts, but nothing recorded **which models were
tried for which job, and what the result was** — so a session looking at a
28-model list could not tell a measured rejection from a model nobody had ever
loaded. Both directions cost: re-running a settled arm, and "discovering" a
candidate that was already ruled out. [TRAPS #42](TRAPS.md) is that failure.

**A model is only ever tested FOR A JOB.** ICE runs at least five distinct ones —
background extraction, corpus labelling, probe generation, probe answering,
judging — and a verdict does not transfer between them. A12 ranked models for
*background extraction*; that ranking was then cited in this file as an
endorsement of an *answering* model, which is how a wrong claim survived
(see §1).

### Background / codex extraction + summarisation — **A12, 2026-08-12**

Eight models, same 60 turns, full background pipeline (density, grounded
summary, chunking, codex, procedural, clustering, batch summaries).
Evidence: [PROVENANCE.md](PROVENANCE.md) A12 entry.

| model | entities | edges | patterns | summary coverage | junk % | verdict |
|---|---|---|---|---|---|---|
| `gemma4:e4b` | 761 | 554 | 49 | 0.976 | 8 | **ranked 1st** |
| `qwen3:4b-instruct` | 353 | 281 | 57 | 0.956 | 3 | **ranked 2nd — the practical pick**, tied on quality at 2.5 GB vs 9.6 |
| `gemma4:26b-a4b` (then incumbent) | 1142 | 1243 | 57 | 0.979 | 2 | **ranked 3rd** — "generic-subject-heavy for no quality gain at ~2× VRAM" |
| `ministral-3:8b` | 1611 | 1485 | 59 | 0.918 | 1 | 4th |
| `qwen3.5:4b` | 233 | 190 | **0** | 0.970 | 1 | 5th — **zero procedural patterns**; bimodal, but best vocabulary-gap signal |
| `granite4:micro` | 252 | 176 | 25 | 0.913 | 6 | 6th |
| `granite4:tiny-h` | 250 | 230 | 50 | 0.895 | 7 | 7th — **fabricates specific unsupported detail**; 12/12 summaries were bare term dumps |
| `nemotron-mini:4b` | 23 | 15 | 48 | 0.607 | 30 | **excluded on measured grounds** |

⚠ **HOW MUCH TO TRUST THIS RANKING — read before re-using it (2026-08-22).**
Four of its five columns are *volume* (entities, edges, patterns) or *format
compliance*, not correctness:

* `summary coverage` is the metric [TRAPS #45](TRAPS.md) shows is **circular** —
  the prompt orders the model to append the very terms the metric counts. Every
  arm scored 0.895–0.979, a spread of 0.08, which is what "did you follow the
  format" looks like rather than "is this any good".
* entity/edge/pattern counts say how much came out, never whether it is true.
* The ranking itself came from **one subagent reading samples** —
  [TRAPS #31b](TRAPS.md) records a subagent producing a confident wrong verdict
  on exactly this kind of read, which reversed its model preference until a
  single query overturned it.

**No model here was ever measured on whether its stored facts are TRUE.** That
judge (`scripts/z1/judge_codex.py`) was built 2026-08-20 and has only ever been
pointed at `qwen3:4b-instruct` — the 20% correct / 25% reversed figures are its
output, and there is no comparable number for any other model.

⇒ **What survives A12 unqualified is its other conclusion**: the two defects
were present in **all seven** viable arms, 4B through 26B, so they are
prompt/design and not capacity. "Use a bigger model" stays ruled out. **Which
small model is best is effectively unmeasured**, and a re-run should use the
truth judge as the primary metric, one variable at a time, with the NER tier
pinned (it is a second variable — see `codex_extraction_ner_tier`).

### Paper era (v1/v2) — judging and ground-truth correction, under **SGLang**

⚠ **A third serving stack this file did not mention.** Ollama and vLLM are in
§2/§4; the paper ran **SGLang on :8003** as well.

| model | stack | job | scale |
|---|---|---|---|
| `mattbucci/gemma-4-12B-AWQ` | SGLang, 150k ctx | **the paper's judge**, and ground-truth correction | 657 Exp-1 probes · **all 1,211 Exp-2 probes** · every Exp-3 ablation variant |
| `gemma4:26b-a4b-it-q4_K_M` | Ollama | the paper run's own model | 858 config references across `experiments/` |

The paper states its own caveat on this judge: it marked correctly-retrieved
information as hallucinated when the fact was absent from the *condensed
ground-truth summary* rather than from the conversation. Exp-2 hallucination
annotations were therefore hand-audited across all 1,211 probes; **Exp-3's were
not**, so Exp-3's absolute hallucination numbers carry automated-judge error and
only its relative differences hold.

⇒ These are v2 numbers and frozen. Listed here because **which models ran** is a
fact about this repo regardless of whether the numbers they produced are still
trusted.

### Corpus labelling (B1) — 2026-07-25, under vLLM

Three served and compared, five rejected. Full table with revisions,
throughput and failure modes: **§4 above**. Summary: `gemma-4-26B-A4B-it-AWQ`
won at 2.34 rows/s and 0.01% degenerate.

### Probe generation

| model | verdict |
|---|---|
| `qwen3.8:27b` | **current** (user preference, 2026-08-17) |
| `qwen3.6:27b` | superseded by 3.8; still named in `models/model_registry.json` |
| `deepseek-v4-flash` (cloud) | works; usage-limited, so local is preferred for volume |

### Probe answering

| model | verdict |
|---|---|
| `gemma4:26b-a4b-it-q4_K_M` | **current** — chosen because it is neither arm under test. ⚠ NOT on A12 grounds; see §1 |

### Judging

| model | verdict |
|---|---|
| `deepseek-v4-flash` (cloud) | **current and pinned** — keeping it fixed preserves comparability with existing verdicts. ⚠ reasoning model: leave thinking ON ([TRAPS #34](TRAPS.md)) |
| CoE gateway `Qwen3.6-35B-A3B` | ❌ rejected for the judge (maintainer, 2026-08-17), fine elsewhere |

### ⚑ ON DISK, NO RECORD FOUND OF A BACKGROUND-MEMORY TEST

⚠ **"NO RECORD FOUND", NOT "NEVER TESTED" — and the wording is the finding.**
This heading has now been wrong twice in one day: first "never tested for any
ICE job", then "never tested for a background-memory job". Each time the
maintainer said the models HAD been tested and the record was lost somewhere,
and each time they were right and I was not. Three confirmed so far:

| model | where the record actually was |
|---|---|
| `gemma4:12b` | the **paper's judge** (AWQ build, SGLang) — 1,211 Exp-2 probes |
| `tinyllama` | a **context-window/budget test** — `ROADMAP_DONE.md`: *"tinyllama serves 2,048 and was handed 4,000"* |
| `gpt-oss` | a **vLLM labelling arm** (§4), 1.20 rows/s |

⇒ **An absence of evidence in this repo is not evidence of absence.** A model
here means *I could not find a record*, which on a repo with a gitignored
corpus, gitignored planning files, `docs/outdated/`, and four years of the
maintainer's own memory is a weak claim. Treat every row as "check with the
maintainer before concluding it is untested."

⚠ **This section was originally built wrong (kept as the method note).** The maintainer said other disk models had been
tested and could not remember where; they were right, and the reason I missed it
is worth keeping: I built this section by grepping `docs/` and `logs/`, and the
paper-era model arms live in **`experiments/`**, which the same session had been
told to treat as out of scope for its *numbers*. Out of scope for a number is not
out of scope for a fact about what ran.

⇒ **Two searches, not one.** `grep -rhoE '"(model|model_name|answer_model)"\s*:\s*"[^"]+"' experiments/`
returns the config fields — that is what was actually run. A bare name grep does
not work on this repo: `gemma4:12b` has 3,639 mentions in `experiments/` and
almost all of them are the **corpus** — the maintainer's own conversations
discussing models — not configuration. Content and config look identical to grep.

Nothing below is a rejection; it is an absence of evidence **for background
extraction, summarisation or retrieval**. Several have been used for other jobs.

| model | size | note |
|---|---|---|
| **`gemma4:12b`** (+ `64k`/`128k`/`256k`) | 7.6 G | ⚠ **NOT untested — corrected 2026-08-22 within hours of first writing this row wrong.** A Gemma-4 12B *was* used heavily, as **the paper's judge**: `mattbucci/gemma-4-12B-AWQ` on **SGLang** at 150k context, judging 657 Exp-1 probes, all 1,211 Exp-2 probes and every Exp-3 ablation variant (`experiments/mature/correct_ground_truths.py:28`, and the paper's own limitations section). Different build (AWQ, not the Ollama GGUF) and a different job (judging, not extraction) — but "never tested for any ICE job" was **false**, and it was written here by grepping the docs rather than the experiments tree. The GGUF is still untested **for background extraction**, and it remains the obvious A12 gap: it sits between `e4b` (1st) and `26b` (3rd) and was not in that run |
| `granite4:small-h` | 19.5 G | the only Granite above `tiny-h`, and both smaller Granites ranked last |
| `qwen3-vl:8b-thinking` | 6.1 G | ⚠ reasoning model — [TRAPS #11](TRAPS.md): the whole budget can vanish into a hidden block |
| `mistral-nemo` | 7.1 G | |
| `qwen2.5:7b` | 4.7 G | **named in the routing registry** (`models/model_registry.json`), so it is a live routing target; no record of a background-job benchmark |
| `llama3:8b` | 4.7 G | |
| `gpt-oss:latest` | 13.8 G | ✅ **tested** — vLLM labelling arm, 1.20 rows/s (§4). No record under Ollama for extraction |
| `qwen3-coder:30b-a3b`, `qwen-coder` | 18.6 / 17.4 G | coding-scoped; untested for memory jobs |
| `rpmax-22b-16k`, `HammerAI/cydonia-v4.3`, `Cydonia-24B-v4.3-heretic-v3` | 12.9–15.7 G | roleplay/creative builds; untested |
| `tinyllama` | 0.6 G | ✅ **used** — the routing floor, and the model that proved silent prompt truncation (`predicted=2909 / actual=2047`). No record of a background-job benchmark |
| `granite4:small-h`, `qwen3.8:27b` for **extraction** | — | 3.8 is tested for probe *generation* only |

### Not obtained, worth obtaining

| candidate | why |
|---|---|
| **[NuExtract3](https://huggingface.co/numind/NuExtract3)** ([GGUF](https://huggingface.co/numind/NuExtract3-GGUF)) | ⚑ **A12 WAS SCOPED TO TEST THIS AND NEVER DID.** Its opening note (2026-07-29, `docs/outdated/roadmap_session_log.md`) defines the background pipeline as **two families**: *"extraction (**NuExtract-class specialist**; a 4B reportedly matches 27B generalists) and generation (no specialist exists; a model-size question, measurable with ICE's own `summary_coverage`)"*. A12 as run tested **only the generation family** — eight general models scored on `summary_coverage`, the metric [TRAPS #45](TRAPS.md) shows is circular. The specialist half was dropped and never re-opened. Same group as the `NuNER_Zero` model ICE already runs for grounding, and it targets the weakest measured subsystem in the system. The VRAM objection was already answered in that same note — *"a 3 GB specialist only fits beside an 18 GB chat model if ICE controls `keep_alive`"* — so it is a scheduling question, not a blocker. ⚠ exact footprint still unverified |
| `qwen3.5:9b` | the 3.5 family shipped 0.8/2/4/9B with 256K context; only the 4B is here, and it ranked 5th with **zero** procedural patterns — the 9B is the untested half of that family |
| `phi4-mini:3.8b` | dense-per-parameter, competitive at the `granite4:micro` tier which ranked 6th |
| AWQ/GPTQ build of `qwen3:4b-instruct` | already wanted in §4 so the bg model can run under vLLM |

### NuExtract3 — pulled and working (2026-08-24)

`hf.co/numind/NuExtract3-GGUF:Q8_0` — **5.2 GB on disk**, via Ollama. The
extraction specialist [MODELS.md §"Not obtained"](#not-obtained-worth-obtaining)
recorded as *"A12 was scoped to test this and never did"*. Same group as the
`NuNER_Zero` ICE already runs.

| | |
|---|---|
| quant pulled | `Q8_0` (4.48 GB download). `Q4_K_M` is **2.71 GB** — the production-footprint variant, not yet pulled |
| speed | **7.8 s/turn** on 1–3 KB turns, 11.7 facts/turn, 10/10 parsed |
| interface | ⚑ **TEMPLATE-FILLING, not instruction-following** |
| output | reasoning model — **split on `</think>`** |
| budget | `num_predict` **≥3000**; 1000 truncates mid-JSON |
| vision | `mmproj-NuExtract3-BF16.gguf` exists; not needed for text, not pulled |

```
<|input|>
### Template:
{"facts": [{"subject": "", "relation": "", "object": ""}]}
### Text:
<turn>
<|output|>
```

**Why it matters:** direction. See [ROADMAP G63](ROADMAP.md#g63) — on turns where
ICE stored a reversed fact, this got the direction right. ⚠ That sample was
selected for ICE's failures and is **not** a fair comparison. It has no
grounding, no canonicalisation, and its relation vocabulary still explodes.

### Judging — 2026-09-03 endpoint correction

`muse-spark-1.3-contributor` is reachable through OpenCode Go's
`/zen/go/v1/responses` endpoint: a minimal request returned HTTP 200 and exactly
`OK`. The same logical request returns HTTP 500 through `/chat/completions`.
Earlier tests established only that the Chat Completions and CLI paths were not
viable; they did not test Responses. The successful two-character answer used
404 output tokens, including 393 hidden reasoning tokens. It was subsequently
selected as the ICE-v2 LongMemEval judge after its role-specific calibration;
see the final section of this file.

⛔ **`mimo-v2.5` scored 27%** against the maintainer's blind labels and
**missed 6 of 6 reversals** — do not adopt it as a judge despite being fast and
free. ⚠ The gateway's `deepseek-v4-flash` scores **60%**, and is **not** the
same as the DeepSeek web product that scored 80%. **Calibrate with
`scripts/oneoff/calibrate_judge.py` before adopting any judge.** Full detail:
[ROADMAP G65](ROADMAP.md#g65).

### ⚑ Background / codex extraction — DECIDED 2026-08-24

| model | verdict |
|---|---|
| **`hf.co/numind/NuExtract3-GGUF:Q8_0`** | ⚑ **DECIDED — replaces `qwen3:4b-instruct`.** 60% vs 15% correct over two independent blind rounds (maintainer's labels), Δ +45 pts, 95% CI [+18, +72]; **reversed 0/20 vs 6/20** |
| `qwen3:4b-instruct` | superseded. Every published ICE graph number was produced by it |
| `gemma4:26b-a4b-it-q4_K_M` | ⚠ still the `.env` pin ([G58](ROADMAP.md#g58)) and never used for any measurement |

⚠ **NOT a config change — it needs `src/` work.** `--bg-model` cannot do it:
`extract_triplets()` builds ICE's instruction prompt and would hand it to a
template model, which measurably produces garbage. Needs a second extraction
path: template prompt, `</think>` stripping, higher token cap.

⚠ **The config is NOT settled** — one gate remains (spec G12). And the **relation
vocabulary must never be put in its prompt**: isolated at 22% correct against
the bare template's 60%.

⚠ **NuNER (`background` tier) beats micro-NER as its entity source** — junk names
8.7% vs 19.5% — but that was decided on SHAPE only and never judged.

### ⚑ ICE-v2 LongMemEval cloud roles — decided 2026-09-04

This decision is for the matched rerun of the paper's frozen **v2** system only;
it does not replace v3's production model choices or FINAL's provider rules.

| role | selected model/path | evidence |
|---|---|---|
| **answerer** | **`gpt-5.6-luna`**, OpenCode Go Responses API | 13/14 on a 14-question public oracle calibration, tied best; 3.2 s/answer and 84 output tokens mean; recovered a needle from 88,025 input tokens in 4.1 s |
| **judge** | **`muse-spark-1.3-contributor`**, OpenCode Go Responses API | strongest prior human-labelled judge calibration (73%); LongMemEval discrimination self-test 3/3; judged all 84 answerer-calibration outputs without a mute |
| **background** | **`qwen3:4b-instruct-bg`**, exact Ollama model, kept resident | both directed extraction controls correct; 63-character summary; real 6,731-character LME turn → 11 triplets in 3.3 s |

Answerer comparison under the same Muse judge: Luna, Omen Alpha, DeepSeek V4
Flash, Qwen3.8 Flash, and LongCat 2.0 each scored 13/14; MiMo V2.5 scored 12/14.
Luna was selected because it tied the best point quality and 3.2-second speed
while averaging 84 output tokens. Omen Alpha tied Luna's score/speed and passed
88K input, but averaged 324 output tokens and took 6.6 s on the long-context
control versus Luna's 4.1 s. Exact aggregate:
`experiments/lme/results/cloud_stack_calibration.md`.

⚑ Endpoint identity is part of model identity. Muse works at `/responses` and
500s at `/chat/completions`; Omen Alpha does the reverse. Luna works at
`/responses` only when the unsupported `temperature` field is omitted. The run
manifest records profile, endpoint, model, requested output cap, and whether
temperature is supported; a resume refuses artifacts from a different profile.
OpenCode's required `x-opencode-session` header is a deterministic UUID per
phase/question/condition (and per judgement), so retries reuse one conversation
identity without coupling independent benchmark cases.

Provider access is fail-fast. Authentication, quota, rate-limit, billing, and
credit-limit responses stop the LME runner/scorer immediately with completed
files intact. Rotate `PROBE_API_KEY` in the main repository `.env`, restart the
same command, and the atomic resume picks up only missing work; do not edit the
v2 worktree's database-only `.env`.

⛔ `Qwen/Qwen3-4B-AWQ` on vLLM is **not** the background path. It reversed the
fixed extraction control under non-thinking mode, while an Ollama-like template
leaked reasoning and exhausted useful extraction output. Cloud answering already
removes the local 26B model swap, so the exact 2.5 GB Ollama background can remain
resident without paying that correctness cost.

### v3 reconciliation consumer update — 2026-09-19

Existing background pin `gemma4:e4b` unchanged. Complete old/new role-attributed
source units and recorded timestamps now reach the reconciler; candidate triple
polarity is explicit. Bounded12-case qualification and12-case SQL/model writer
replay passed; no general accuracy claim. Maintenance uses the shared context
builder and source eligibility. See PROVENANCE and V3_REPAIR.

### v3 foreground/background role boundary — 2026-09-20

Turn summarization and procedural extraction now always resolve their model
through `get_bg_model_name()`. A foreground cloud model name in `model_used`
cannot override the background pin. No model promotion or cloud provider setup
accompanies this repair; the unpinned factory fallback remains explicitly warned.

### v3 long-source NLI candidate — 2026-09-20 (not deployed)

`tasksource/ModernBERT-base-nli`, pinned
`de4ab7e77845098b7fab7f6ab9d370ddff27b19c`, is being checked as a longer-input
source-support candidate. Its **actual config says2048 positions**, despite the
ModernBERT family's larger advertised window. Label order entailment/neutral/
contradiction verified from that revision. No production setting changed. At the unchanged0.95 cutoff, the same30
controls passed28: falsely admitted Hindi negation and withheld an informal
supported paraphrase. It does not qualify as a replacement. Model card includes document/dialogue/context NLI training:
https://huggingface.co/tasksource/ModernBERT-base-nli . Cached only;30-pair float32 run peaked at0.588GiB allocated and took0.87s
after load. Long-input qualification is not established. Artifact:
`experiments/v3_repair/results/nli_modernbert_candidate.json`.

Next long-input candidate (not deployed): `MoritzLaurer/bge-m3-zeroshot-v2.0`,
revision`9abf1c8aaeb82a2447809c20753ed0b106b76652`. Actual XLM-R config8194 positions
(8192 content positions); binary entailment/not_entailment, so it cannot provide
an independently calibrated contradiction probability. Multilingual model card:
https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0 . Cache/qualification only;
production DeBERTa unchanged. Do not map not_entailment to contradiction.

Qualification2026-09-21: BGE short controls29/30 (all17 unsupported rejected,
12/13 supported admitted); long complete706–4577-token controls13/24 (all12
unsupported rejected, only1/12 supported admitted). Peak long run2.317GiB. No
threshold fitting. Cached, **not promoted**: insufficient demonstrated compression
utility. Artifacts `nli_bge_candidate.json` and `nli_bge_long_candidate.json` under
`experiments/v3_repair/results/`. DeBERTa remains the deployed verifier.
