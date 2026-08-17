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
| **background / codex extraction** | `qwen3:4b-instruct` | Ollama | the re-seed's bg model. ⚠ `.env` pins `BACKGROUND_MODEL_NAME=gemma4:26b-a4b-it-q4_K_M`, which `two_arm_seed.sh` calls **STALE, not a decision** — arms pass `--bg-model` |
| **probe generation** | `qwen3.8:27b` | Ollama | user's preferred local model (2026-08-17); supersedes `qwen3.6:27b` |
| **probe answering** | `gemma4:26b-a4b-it-q4_K_M` | Ollama | A12's top-ranked; deliberately **neither arm under test**, so it cannot favour its own summaries |
| **judging** | `deepseek-v4-flash` | **cloud** | paired A/B judge. ⚠ reasoning model — `reasoning_effort="none"` made it confidently wrong ([TRAPS #34](TRAPS.md)) |
| **embedding** | `Qwen/Qwen3-Embedding-0.6B` | in-process | native **1024-dim**, frozen. Resident on the same card as everything else |
| **NER (pre-flight + codex whitelist)** | **MicroNER — ours** | in-process | `models/ner/ner_model.pt`, 234 KB, over the `slice384` MRL prefix |
| **NER (background)** | `numind/NuNER_Zero` | in-process | GLiNER-family zero-shot, 448.9M, deberta-v3-large. Clustering + `turn_density` only |
| **classification** | `ice_classifier_v4_schema2.pt` | in-process | MLP head, 27 logits (11 topic + 12 intent + 4 context) |

---

## 2. Local — Ollama (28 models, 267.5 GB on disk)

All Q4_K_M unless noted. Sizes are on-disk, not VRAM.

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

⚑ **FAIR USE IS A HARD CONSTRAINT, NOT A COURTESY — READ BEFORE PLANNING A RUN.**
The guide states the server is shared by **~15 students at a time** and
explicitly forbids *"bulk scraping, automated spam, or anything that violates
TCET policy"*, with per-key usage logged and keys revocable by the coordinator.

| ICE workload | calls | suitable? |
|---|---|---|
| re-seed (293 turns × chunks) | **hundreds–thousands** | ❌ **no** — this is exactly the bulk automation the policy forbids |
| probe generation | ~50–500 | ⚠ only with low concurrency, and Ollama already does it free |
| **paired judge** | ~150, one-off | ✅ **the best fit** — currently the only thing blocked on a paid quota |

⇒ **Default to Ollama for anything batch.** Reach for the gateway when the
local box genuinely cannot do the job — a 35B-class model, vision, or a judge
run that would otherwise wait on a cloud usage limit. Keep concurrency at 1–2.
**The key is a personal student credential; getting it revoked costs more than
the run saves.** Key lives in `.env` (`COE_API_KEY`), never in a tracked file.

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
