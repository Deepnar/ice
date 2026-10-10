# ICE v3 — background model identity consumers, 2026-10-10

**G27 — background identity/default**, for example a cloud answering choice
silently becoming the memory writer, owns this audit. Actual factory resolution
returns `gemma4:e4b`; the extraction setting is
`hf.co/numind/NuExtract3-GGUF:Q8_0`. Keep these explicit research pins. The wider
unpinned shared-mode fallback remains unresolved, with its existing warning on
every fallback call. No global default or model is changed.

## Code map, distinct from dynamic job coverage

| Core writer/job | Actual selection seam | Research identity |
|---|---|---|
| Turn summary/abstract and procedural extraction | `post_flight` / `procedural_extractor` call `get_bg_model_name`, rather than the foreground model. | E4B |
| Codex triplet extraction | `extract_triplets`: explicit override, otherwise extraction pin, then general fallback; native structured task. | NuExtract3; overrides require their own frozen identity |
| Author-assertion/cited-procedure review | `assertion_support` / `procedural_support` resolve the general pin for native calls. | E4B; NLI is separate |
| Cluster assignment/merge labels | `clustering`→`cluster_labels`→`conversation_summary._default_llm`; source entities use NuNER. | E4B labels; embedder/NuNER distinct |
| Conversation summary | `_default_llm`→background factory/client. | E4B plus separate NLI |
| Batch summary | `batch_summarizer` resolves general pin. | E4B plus separate NLI |
| Reflection/entity accounts | `reflection` resolves general pin; entity notes use `_default_llm`. | E4B plus source verification |
| Maintenance reconciliation | `make_llm_decider` resolves general pin, not extraction specialist. | E4B |
| Compaction and three decay jobs | Deterministic scans, expiry/archive/checkpoints. | No generative completion |

All ten periodic core jobs exist in the production `JOBS` registry. This map and
factory query do **not** prove every semantic branch made a model call. Prior
lifecycle controls exercised the schedule; inert branches remain disclosed.
Existing pure pin and actual-client controls verify selection seams, not
independent job quality or worst-case capacity. This audit calls no completion,
loads no model and makes no API request.

## Confirmed preparation defect and repair

The earlier offline preparation sealed code and a numeric/boolean knob inventory
but omitted string model pins. Actual reuse accepted independently changed E4B
and NuExtract settings with the same identity and eight unchanged products.
Those old products are preserved and never authorized replay.

`prepare_inputs.local_model_profile` now binds a safe allow-list: background
mode/pin, extraction pin, embedder, classifier checkpoint/schema paths,
background NER, reranker and NLI model/revision. Unset shared pins are unresolved.
Changed pins/revision refuse reuse **before** any product write; unchanged
preparation is byte-stable. No Settings dump or credentials enter the profile.
Declared identity does not certify downloaded weights, tokenizer revisions,
endpoint/decoding, serving capacity or semantic quality; freeze/runtime
qualification remains required.

Current private plan: `logs/v3-offline-input-plan-2026-10-10-r4`, still
`replay_ready=False` and `score_ready=False`. Actual changes to the background
pin, extraction pin and NLI revision each refuse reuse with all eight products
unchanged; the unchanged invocation preserves all eight too. Baseline/refusal,
registry and source-hash receipts are in
`logs/v3-taxonomy-consumer-audit-2026-10-10/`.

Decision: **retain and bind the explicit E4B/NuExtract research profile before
replay**. Do not substitute cloud routing or registry order for writer identity.
Wider unpinned/dedicated policy remains visible under the parent.
