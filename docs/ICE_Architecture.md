







**SYSTEMS ARCHITECTURE  ·  TECHNICAL REPORT**

**Architecture of ICE**

Infinite Context Engine — A Systems Description










Prepared: 2 July 2026

Derived from the ICE source tree  ·  Code is authoritative

# **Table of Contents**

Table of Contents	i

1. System Overview	1

1.1 Request lifecycle	1

1.2 High-level component map	2

2. Classification Engine	3

2.1 The PyTorch MLP classifier	3

2.2 DI3 — Dynamic Intent Inferencer (DELETED, D8)	4

2.3 Context-aware classification	5

2.4 The memory-retrieval decision (B2)	6

2.5 Training pipeline	6

2.6 The temporal joint gate & style-dependent rules	7

2.7 Experimental / partial features	7

3. Memory Architecture	8

3.1 Episodic memory	8

3.2 Codex (semantic) memory	9

3.3 Procedural memory	10

3.4 RAG memory	10

3.5 Memory Slots	10

3.6 Context Clusters	11

3.7 Decay and archival mechanics	12

3.8 Batch summaries	13

4. Codex Knowledge Graph (v2)	13

4.1 Entity and edge tables	13

4.2 Controlled relation vocabulary	14

4.3 Temporal versioning and automatic property updates	15

4.4 Extraction pipeline	15

4.5 MERA — Meta-Enumeration Retrieval Agent	16

4.6 Micro-NER model	17

4.7 Vector fuzzy matching for entity resolution	17

4.8 Retrieval and event-sourced compaction	17

5. Procedural Memory	18

5.1 Pattern extraction	18

5.2 Trigger-condition gating for retrieval	19

5.3 Decay and confidence promotion	19

5.4 Retrieval	19

6. Hybrid Retrieval Orchestrator	19

6.1 The retrieval legs	20

6.2 HyDE query rewriting	20

6.3 Dynamic leg weighting	21

6.4 Reciprocal Rank Fusion (RRF)	21

6.5 Post-fusion processing	22

6.6 Cluster-scoped retrieval	23

6.7 Dynamic token budget	23

6.8 Wide-net fallback	24

6.9 Feature Toggling for Ablation Studies	24

6.11 Time-scoped retrieval (Track T: T1–T3)	24

7. Prompt Assembly	24

7.1 Stable-prefix ordering	25

7.2 Per-component rendering	25

7.3 Emotional / creative bypass	25

7.4 Token budget enforcement during assembly	26

8. Background Maintenance Runtime	26

8.1 Runtime infrastructure: triggers, ledger, idle gating	26

8.2 Post-Flight Evaluator	27

8.3 Codex Extractor	28

8.4 Procedural Extractor	28

8.5 Decay Workers	28

8.6 Reflection Worker	28

8.7 Clustering Worker	29

8.8 Batch Summariser	29

8.9 Sentinel Monitor	29

8.10 Fine-Tune Worker	30

8.11 Compaction Worker	30

8.12 Drop Zone and Codex Inject Watcher	30

9. Model Registry and Mixture-of-Experts Routing	31

9.1 Dynamic registry	31

9.2 MoE selection	31

9.3 Session stickiness	31

10. Operational Infrastructure	32

10.1 FastAPI proxy	32

10.2 PostgreSQL + pgvector	33

10.3 In-process maintenance runtime (Celery + Redis removed, C7)	34

10.4 Idempotency architecture	34

10.5 GPU resource management	34

10.6 Configuration system	35

11. Service Layer & ICE-as-MCP (E0/E7)	36

11.1 The service layer (src/services/)	36

11.2 The ice-mcp server	36

11.3 Headless boot, the runtime lease, and standby	37

*This section describes the system as implemented in the source tree at the time of writing. Where the legacy design documents (\`architecture.md\`, \`architecture\_v2.md\`) conflict with the code, the code is treated as authoritative. Features that are experimental, gated off, or not yet wired into the live path are flagged inline.*


## **1. System Overview**

ICE (Infinite Context Engine) is an **OpenAI-compatible memory middleware** that sits between a conversational client and a pool of locally-served large language models (Ollama / SGLang). It is not a model itself: every request addressed to the synthetic model name "ice-proxy" is intercepted by a FastAPI proxy that classifies the turn, retrieves relevant context from four long-lived memory stores, assembles a KV-cache-friendly prompt, routes the request to a per-turn specialist model, streams the response back over Server-Sent Events (SSE), and then dispatches a fan-out of background workers that extract, decay, cluster, and crystallise the new turn into long-term memory. The net effect is that any downstream model — regardless of its native context window — operates against an effectively unbounded, personalised context that is selectively rebuilt on every turn.

ICE therefore occupies the layer conventionally called \*memory and retrieval orchestration\* in a conversational-AI stack: above the model-serving substrate (Ollama, SGLang, Hugging Face text-generation inference) and below the client. It exposes the standard /v1/chat/completions endpoint, so existing OpenAI-compatible clients require no modification beyond pointing at the proxy and (optionally) sending an X-ICE-Conversation-ID header.

### **1.1 Request lifecycle**

Each turn traverses a **pre-flight** (synchronous, in the request path) and a **post-flight** (asynchronous, after the stream closes) phase.

**Pre-flight.** (0) After conversation/scope resolution and *before* any classification, a message whose first line starts with `/` is handed to the deterministic chat-command parser (C11, §11.4) — a handled command streams its confirmation as a normal SSE completion and the rest of the lifecycle (classification, retrieval, the model, storage, post-flight) never runs. (i) The user message is classified by a two-stage pipeline — a rule-based \*Dynamic Intent Inferencer\* (DI3) followed, on miss, by a 25-way PyTorch MLP head — producing topic tags, intent tags, and a context\_reliance label. (ii) A single classifier-trusting decision (B2, §2.4) combines the reliance probability with a memory-pressure prior and soft bumps in log-odds space to decide whether long-term retrieval fires — it *prefers* memory but no longer *forces* it. (iii) A \*Hybrid Retrieval Orchestrator\* runs six retrieval legs (BM25, vector, Codex graph, procedural, RAG, batch summaries), fuses them with weighted Reciprocal Rank Fusion (RRF), and post-processes the fused list with keyword/recency/length bonuses, session diversification, deduplication, and a dynamic token budget. (iv) A \*Prompt Assembler\* concatenates the retrieved fragments with persistent memory slots, recent turns, and the live user message under a stable prefix that maximises KV-cache reuse. (v) A \*Mixture-of-Experts\* (MoE) router selects the best locally-served model for the assembled prompt, with per-conversation stickiness.

**Post-flight.** (i) A \*Post-Flight Evaluator\* runs lossless detection, document detection, and summary generation, writing the auxiliary columns of the new episodic\_memory row. (ii) In the same job it chains a \*Procedural Extractor\* unconditionally and, conditionally on the lossless flag, a \*Codex Extractor\* — direct calls, each idempotent (C7). (iii) The in-process maintenance runtime (§8) keeps the stores maintained over time — Decay, Clustering, Reflection, Batch Summariser, Compaction, Sentinel Monitor on ledger-driven cadences, plus the consent-gated Fine-Tune proposal.

### **1.2 High-level component map**

The system decomposes into the following components, each described in the corresponding section below:

                       ┌──────────────────────────────────────────────┐  
   Client ──HTTP/SSE──▶│                FastAPI Proxy                  │  
                       │  /v1/chat/completions  /memory-slots  /user-  │  
                       │  control  /model-registry                    │  
                       └───┬──────────────────────────────┬───────────┘  
                           │ pre-flight (sync)            │ post-flight (async)  
            ┌──────────────▼──────────────┐  ┌────────────▼─────────────────┐  
            │  Classification Engine       │  │  Maintenance Runtime (C7)     │  
            │  (DI3 + MLP + overrides)     │  │  (in-process asyncio + ledger)│  
            └──────────────┬──────────────┘  │  post\_flight · codex\_extractor │  
                           │                  │  procedural\_extractor · decay │  
            ┌──────────────▼──────────────┐  │  codex\_decay · procedural\_decay│  
            │  Hybrid Retrieval           │  │  reflection · clustering       │  
            │  Orchestrator (6 legs + RRF)│  │  batch\_summarizer · sentinel   │  
            └──────┬───────────────────────┘  │  fine\_tune · compaction        │  
                   │                          │  drop\_zone · codex\_inject      │  
            ┌──────▼──────────────┐           └────────────┬──────────────────┘  
            │  Prompt Assembler   │                        │  
            └──────┬──────────────┘                        │  
                   │                                         │  
            ┌──────▼─────────┐    ┌──────────────────────────▼──────────┐  
            │  MoE Router    │    │        PostgreSQL + pgvector          │  
            │  (registry)    │    │  episodic\_memory · codex\_entities     │  
            └──────┬─────────┘    │  codex\_edges · procedural\_memory     │  
                   │               │  rag\_chunks · context\_clusters       │  
            ┌──────▼─────────┐    │  memory\_slots · batch\_summaries      │  
            │  Ollama/SGLang │    │  review\_queue · sentinel\_rules       │  
            │  model pool    │    │  idempotency\_keys · cold\_storage     │  
            └────────────────┘    └───────────────────────────────────────┘


## **2. Classification Engine**

The classification engine produces, for each user turn, a set of **topic tags**, a set of **intent tags**, and — since B1 (2026-07-25) — **four independent context-reliance signals** from which the legacy single context-reliance string is *derived*. These outputs drive every downstream decision — which retrieval legs are weighted, how the token budget is split, whether the wide-net fallback fires, and which model the MoE router selects. The engine is deliberately a two-stage cascade: a cheap rule-based pre-classifier (DI3) resolves obvious cases in microseconds, and a small PyTorch MLP resolves everything else with sub-millisecond CPU inference.

**The head layout is data, not code.** data/labeled/label\_schema.json declares each head's labels, activation and decision rule; src/classifier/schema.py loads it and every consumer asks it for widths and offsets. Before B1 the layout lived as magic numbers scattered across classifier.py, orchestrator.py and fine\_tune.py (outputs\[:, 11:22\], probs\[22:\], torch.zeros(n, 25)), so adding one label meant hunting slices and hoping none were missed. A grep-gate (`\[:11\]|11:22|22:\]|== 25`) keeps src/ and scripts/ clean of them.

### **2.1 The PyTorch classifier (schema v2)**

The model is a small trunk with one head per label group (classifier/model.py), ~663k parameters:

```
Linear(1024→512) → GELU → Dropout(0.2) → Linear(512→256) → GELU → Dropout(0.2)
  ├── Linear(256→11)   topic
  ├── Linear(256→12)   intent
  └── Linear(256→ 4)   context_reliance
```

The input is the **native 1024-dim embedding** from the process-shared embedder (src/memory/embedder.py::get\_embedder()). C17 (2026-07-19) staged the classifier on the **slice384 MRL prefix** so the frozen v1 checkpoint kept working; B1 retrains at native width and retires that staging for this consumer (the micro-NER remains on slice384 until A9).

`forward()` returns the heads concatenated in schema order (27 logits), so every consumer slices with `schema.slice(head)`; `forward_heads()` returns them separately for per-head losses. All three heads are **sigmoid**; each head's loss is BCE-with-logits with per-label positive weights (neg/pos, **capped at 3**).

**The pos-weight cap is a calibration knob, and run 1 set it wrong.** At 20 (the original value) a ~1%-prevalence label saturates the cap, so the loss pays 20× more for a miss than a false alarm and every head learns to answer yes: run 1 produced recall 0.87–0.97 with precision 0.04–0.81 on all 28 labels. A 3/5/10/20 sweep over identical splits (2026-07-27) showed the cap matters far less than the *threshold* — at per-label fitted thresholds all four land within 0.585–0.602 mean macro-F1 — but the low caps put the optimum back in a sane band (0.55–0.65) instead of 0.85–0.95, which is the same finding read twice. The shipped model uses **cap 5**; the module default is 3.

- **topic** — 11 labels, unchanged: Software\_&\_Tech, STEM\_&\_Academics, Business\_&\_Finance, Creative\_&\_Media, Admin\_&\_Productivity, Lifestyle\_&\_Health, Social\_&\_Relationships, World\_&\_Current\_Events, Meta\_AI, Null\_Noise, General\_Reference\_&\_Trivia. A label is selected when prob > the checkpoint's **tag\_threshold**, with argmax fallback.

**The decision threshold travels inside the checkpoint.** A threshold is a property of trained weights, not of an installation: v1 was calibrated at 0.3 and the v2 head's own sweep puts its optimum at **0.65** (fitted on val, reported on test). One global setting cannot serve both, and both remain in play — v2 is live, and v1 is still loadable for the gate and for rollback. `sweep_threshold.py --write` therefore stamps `tag_threshold` into the checkpoint and `classifier._tags_above` reads it, falling back to `settings.classifier_threshold` for checkpoints predating the stamp. Promoting a model promotes its calibration with it — no .env edit to remember, and no way for the two to drift apart. This is not cosmetic: on the shipped model, scoring at the inherited 0.3 gives mean macro-F1 0.526 against **0.610** at the fitted threshold, so the wrong number costs more than every architecture change in B1 combined. Per-label thresholds were measured too and rejected — they score identically (0.610 vs 0.609), and the head where they would earn their complexity, context\_reliance, does not go through `_tags_above` at all (its consumers read the raw scalars and apply their own thresholds by design).
- **intent** — 12 labels: the original eleven (Factual\_Retrieval, Troubleshooting, Generation, Ideation, Analysis\_&\_Summarization, Strategic\_Planning, Decision\_Making, Emotional\_Processing, Utility\_Formatting, Casual\_Banter, Open\_Exploration) plus **Code\_Change** (the codebase should differ afterwards), appended **last** so v1 indices survive — which is what makes the D5 gate's shared-subset comparison possible. **Codebase\_Query was appended alongside it and then dropped before the shipped run** (see below).
- **context\_reliance** — 4 **independent** sigmoids: **Needs\_Memory**, **Temporal\_Recall**, **Needs\_Live\_Info**, **High\_Complexity**. Any combination is legal. Only the first two reach a live decision: `p_ltm` (= p\_mem) and `p_temporal` feed B2 (§2.4). **`p_complex` has no consumer at all** — it is computed, carried on the result, stored, and read by nothing; F11/B3 are its only prospective readers and neither exists. It is retained deliberately (user, 2026-07-27) so the signal keeps being recorded, **not** because it is trusted: measured against hand-authored probes the head behaves as a *verbosity* detector rather than a difficulty one (0.98 on "walk me through how a request flows…", 0.79 on the trivial "add a retry with exponential backoff", 0.20 on a genuinely hard two-domain prompt), and its inter-labeler agreement is the worst of the four at F1 0.42. Treat it as unvalidated.

**Why Codebase\_Query was dropped (2026-07-27).** It cleared §4's *count* floor (219 training positives, above the <150 drop rule) but failed the *learnability* bar: test F1 0.10 at fitted thresholds, precision 0.16, recall 0.08 — the head effectively never fired, and lowering the pos-weight cap did not rescue it. The cause is supervision, not architecture. The two labelers tagged 452 and 221 rows and **overlapped on 33** (agreement F1 0.10); the trained head scored 0.10, i.e. exactly its supervision's ceiling. The corpus could not contain the class: it is website-chat data where the assistant had no repository access, so "where is X in my project" was a pointless thing to ask. The label lives on in the data (rows keep their annotations; `dataset.py` ignores tags absent from the schema), so re-adding the schema entry and retraining restores it with **no relabeling** — but that is only worth doing once ICE-as-MCP (E7) produces real navigation traffic. Roadmap **E12** owns the decision.

**Why the context head changed shape.** v1 decoded this block with softmax over {Zero\_Shot, Long\_Term\_Memory, Real\_Time\_Search}, forcing every prompt to be exactly one. "What's the current price of the GPU I told you I was saving for" is both — it needs live data *and* memory — and the single-label head had to discard one. Real\_Time\_Search being orthogonal to memory is the original B1 observation; Temporal\_Recall and High\_Complexity join as signals the 3-way could never express.

**Zero\_Shot is no longer a label.** It is the derived state "all reliance signals low".

The output is wrapped in a ClassificationResult dataclass (topic\_tags, intent\_tags, context\_reliance, raw\_probs (schema-wide: 27 under v2, 25 under a v1 checkpoint), max\_confidence, prompt, plus the B2 scalars p\_ltm / p\_rts / ctx\_confidence / reference\_signal, B1's p\_temporal / p\_complex, and head\_confidences).

**Both checkpoint generations load.** A checkpoint records its own schema\_version, template\_version, input\_dim and label names; `load_checkpoint` dispatches on them, returning the v2 network or LegacyICEClassifierV1. This is not politeness — D5's non-regression gate has to *run* the old model to compare against it, and it is what makes a **rollback a file swap rather than a code change**. Since B1's promotion (2026-07-27) `settings.classifier_model_path` is `models/classifier/ice_classifier_v4_schema2.pt`; the displaced v1 model keeps its own name, `ice_classifier_v3_qwen_ft3.pt`, and is the rollback artifact. The live path was renamed in the same session — it had been the v3\_qwen\_ft3 filename holding a v2 model, which is harmless to code that reads `schema_version` and misleading to a human.

**The derivation layer (D6)** lives in schema.py as pure label logic (`finalize_context_scalars`), not on the classifier class, so it is testable without a loaded model:

```
zero  = 1 − max(p_mem, p_live)
context_reliance = argmax{Zero_Shot: zero, Long_Term_Memory: p_mem, Real_Time_Search: p_live}
p_ltm = p_mem ;  ctx_confidence = top1 − top2 of that derived three-way
```

Every pre-B1 consumer keeps reading `context_reliance` / `p_ltm` / `ctx_confidence` and never learns the head changed shape underneath it. B2's scalar seam was designed for exactly this swap (§2.4) and needed no change.

### **2.2 DI3 — Dynamic Intent Inferencer (DELETED, D8, 2026-07-27)**

For most of ICE's life a rule-based pre-classifier ran *before* the MLP and could answer without it. It computed five density signals from the raw prompt (code / sentiment / meta / noise / reference) and returned the first of five thresholded rules that fired. **D8 deleted it in full** — `di3.py`, `di3_signals.py`, `di3_config.py`, `di3_logger.py`, the seven `DI3_*` settings, `settings.ltm_bump_reference` and `ClassificationResult.reference_signal`.

The rule was promote → measure → delete: on the rows each path *actually intercepted* (first-match order, 9,441 held-out rows), the v2 head had to tie or win. It won every slice on every metric — code (675 rows) topic F1 .878 → .928 and intent .191 → .515; sentiment (92) .238 → .791 / .246 → .622; meta (149) .248 → .790 / .156 → .594 — and the retrieval decision improved on all three. Two paths deserve their own note. The **noise** rule fired on **zero** rows: it requires a pure-punctuation string of ≤3 distinct characters, so it never matched the keyboard-mash strings it explicitly listed (`asdf` scores 0.5, `zzzzzzzz` 0.5, against a 0.8 threshold) — the anticipated "keep a small noise guard" fallback had no population to justify it. The **reference** rule, which alone deferred to the MLP for tags and contributed only an anaphora flag, made B2's decision measurably *worse*: accuracy .852 → .777, buying six extra correct retrievals for sixty-five spurious ones. That knob had never actually been measured — `tune_b2.py` swept it but never set the flag it depended on, so it was one of that sweep's inert knobs.

Two findings outlived the deletion. `conversation_length` was passed by **no caller**, so three of the five rules' `Long_Term_Memory` branches were unreachable and the documented two-tier reference threshold never used its second tier — a documented behaviour that had never once executed. And DI3 reached into Track T: T2's joint gate consumed `reference_signal` as one of four arms, where it was admitting 49 false time-windows on long pasted documents whose *length* had accumulated enough instances of "the" to cross a density threshold (§2.6).

The measurement is preserved in `scripts/classifier/pipeline/eval_di3.py`, which carries a **frozen copy** of DI3's signal functions and thresholds so the finding stays re-runnable after the code it judged is gone — the same discipline `templates.py` applies to v1's prompt strings.

### **2.3 Context-aware classification**

When a conversation\_id is supplied, the MLP path queries the **last three \`episodic\_memory\` turns** for that conversation (\_get\_context\_turns, n=3, max\_total\_words=500), preferring summary\_text and falling back to the first 150 words of raw\_text with an ellipsis. The context is prepended to the prompt as natural-language text under a fixed template ("Conversation context (summarized):\\n\{context\}\\n\\n… User prompt: \{prompt\}") before embedding — there is no separate context vector or pooling. The same embedder is used with or without context.

**B1 (2026-07-25) made the templates shared and versioned.** They live in src/classifier/templates.py and are imported by *both* the inference path and the training pipeline. Before this, inference rendered the template inline while the trainer embedded bare prompt text — the model was trained on one distribution and served another, which was the single biggest known defect in the v1 classifier. Now the mismatch is impossible by construction: if training stops calling `templates.render`, nothing silently drifts because there is only one renderer. The `truncate_context` budget helper is shared the same way, so an offline context prefix can't be longer than the live one.

Templates are versioned alongside the schema — the v1 strings are frozen verbatim (D5's gate must render the *old* model's input the way that model actually saw it, or the comparison is rigged against the baseline), and v2 names the real v2 categories. A checkpoint records its `template_version`, so a loaded model always knows how its input must be rendered.

### **2.4 The memory-retrieval decision (B2)**

**Reworked 2026-07 (roadmap B2).** The old design forced Long\_Term\_Memory from four scattered places — `_apply_hard_overrides` (Creative ⇒ LTM; Software+referential ⇒ LTM), DI3's reference rule, an api/main.py bias (`turn_count>10 or max_confidence<0.95 ⇒ LTM`), and an orchestrator safety override (`Zero_Shot+conversation_id ⇒ LTM`). Together they made retrieval fire for almost every turn regardless of the classifier — "safe, not smart." All four are removed. In their place, `api/memory_decision.py::decide_memory_retrieval` makes **one** decision, in log-odds space, that *prefers* memory but never *forces* it:

```
logit(P_need_mem) = logit(P_ltm) + ltm_prior_bias
                    + ltm_length_weight · logit(P_len)      # memory pressure
                    + Σ bumps (creative / reference / referential / low-confidence)
retrieve  ⇔  P_need_mem > ltm_decision_threshold
```

- **`P_ltm`** is read directly from the context-reliance softmax mass (not the old `max(all 25 probs)`, which measured topic peakedness, not reliance), along with a `ctx_confidence` = top1−top2 margin. Both are populated by `classifier._finalize_confidence`; for DI3 fast-path results (no ML probs) a prior is derived from the label DI3 chose.
- **`P_len` (memory pressure)** is a *one-sided* logistic in how much conversation history sits **beyond the sliding window** (the recent-turn token budget, §6.3) — neutral while the window still covers the conversation, rising only as unseen history accumulates. This is the "sliding window + total turns + total context" signal: no `turn_count>10` cliff.
- **Bumps** are the old hard signals, demoted to additive nudges: Creative topic, DI3 anaphora (`reference_signal`), referential-word presence, and a low topic/intent-confidence safety net. **T2 adds `ltm_bump_timescope` (+3.0)** when a non-current TimeScope was detected (passed as a kwarg, not a ClassificationResult field — an explicit "what did I think in 2025" is definitionally a memory query, but it stays a log-odds term with breakdown telemetry, never an early-return override).
- **B1 D7 — the detector and the `Temporal_Recall` label are equivalent evidence for that bump: OR, never AND, never counted twice.** Either a fired detector or `p_temporal ≥ settings.temporal_label_threshold` (**0.85** since 2026-07-27, raised from 0.6) adds `ltm_bump_timescope` exactly once. The raise is measured, not cosmetic: the v2 `Temporal_Recall` head does not fire as an independent time signal but as a **shadow of `Needs_Memory`**, with which it co-occurs in 79% of its training positives — mean p_temporal 0.87 across hand-authored memory-needing prompts carrying no temporal content at all. Because the two evidences are OR'd, a low threshold makes the deterministic parser redundant and drags the decision toward always-retrieve, which is the failure B2 exists to prevent. They catch different things: the classifier catches "what was I leaning towards back then" (no parseable date, detector silent), the detector catches "in March 2026" on a prompt the head reads as ordinary. Note what the label explicitly does *not* do: **only the deterministic detector ever sets a time window.** A sigmoid inventing "two years ago" would be a hallucinated filter, so the label gates and boosts while the parser resolves.
- **⚠ E12 (2026-07-27) measured that temporal arm and it is inert.** Over 9,441 held-out rows the label fires without the detector on 175 and is right about them (85% genuinely need memory, against the detector's own 52%) — but those rows carry mean `p_ltm` **0.931**, so 172 of 175 already retrieve, and disabling the arm moves **one decision in 9,441**. The cause is structural rather than a threshold: measured on gold labels, **78.1%** of `Temporal_Recall` rows are also `Needs_Memory`, because a question about the past needs memory by definition. **A signal that is a subset of another cannot improve that signal's own decision**, so no value of `temporal_label_threshold` rescues this and Z1-prep should not sweep it expecting movement. The label is not wasted — only **20.3%** of memory queries are time-shaped, so it is the only "is this about the past" signal ICE has — but its earned consumers are both in Track T and both unwired: tightening §2.6's joint gate (precision 84% → 93%) and flattening the ranker's recency preference for the 250-of-557 time questions carrying no parseable date. Roadmap **T5**, scheduled post-Z1.
- **All weights are settings** (`ltm_decision_threshold`, `ltm_prior_bias`, `ltm_length_weight`, `ltm_pressure_midpoint_tokens`, `ltm_pressure_scale_tokens`, `ltm_bump_*`). This is deliberate: B2 sits on top of the *current* classifier, which roadmap B1 will retrain — so `P_ltm` is consumed as a scalar (surviving a softmax-3 → multi-label-sigmoid change) and the decision is re-tuned, not rewritten. The full `breakdown` dict is logged (`memory_decision` event) and is a candidate for the F5 SSE attribution layer.

When the decision is to retrieve, main.py sets `context_reliance = "Long_Term_Memory"` so downstream gates/storage/telemetry still key off the label. **Persistent memory is not gated by this decision:** memory slots and bookmarks are user-level standing context, so they (and prompt assembly generally) run on *every* turn — only the retrieval `fragments` are conditional. This matters because B2 genuinely skips retrieval on confident standalone turns, where the old design (which forced retrieval on nearly every turn) had incidentally always injected slots too.

### **2.5 Training pipeline (v2, B1 — `scripts/classifier/pipeline/`)**

Eight standalone stages, each resumable by row id, writing into data/labeled/v2/. Each rewrites its v1 predecessor, now frozen under scripts/classifier/legacy/ for provenance.

**Stage 1 — extract.** Three sources, deduplicated by normalised content hash: (i) the public datasets (LMSYS / WildChat / ShareGPT), pulled **larger and bucket-weighted** — a cheap keyword bucketer caps how much of the pull any one theme may occupy, because an unweighted pull is dominated by coding help and roleplay while thin topics stay thin; (ii) the user's own exports in data/simulation/, parsed through **F10's ingestion adapters** (src/ingestion/formats.py) rather than a bespoke parser; (iii) the v1 25k corpus — its **text is reused, its labels are discarded**. Unlike v1, which took only each conversation's first user turn, any turn *k* can be emitted with its prior turns attached.

**Stage 2 — stitch\_icedev.** ICE was designed across a series of separate DeepSeek chats (ICE-1, ICE-2, …), a new one each time the previous hit its context limit. They are one continuous project conversation that a UI boundary cut into pieces, and a decision made in ICE-2 is referenced in ICE-5 — exactly the beyond-the-window case ICE exists to serve. This stage stitches them chronologically into one conversation (3,473 turns, 2026-06-04 → 07-03) and writes it in the dialogue shape F10 can re-import. It is a **shared asset with FINAL**, where it serves as a real long-project memory test.

**Stage 3 — synth.** Casual-voice prompt generation for **measured** gaps: the stage reads actual per-label counts and generates only for labels under the floor. Rows carry `meta.target_label` as a generation *hint* and are then labeled like every other row — v1 stamped the intended label onto the row, which is self-certification, and it matters most for exactly the labels synth exists to seed.

**Stage 4 — label.** **Two labelers from different model families**, run sequentially (24 GB holds one at a time), each labeling from scratch, blind to each other and to the v1 labels. Independence is the whole quality signal: two variants of one family agreeing measures a model against itself. Agreement is kept; disagreement goes to a **third-family tiebreak**; a genuine three-way split goes to the **human review queue**, which concentrates the user's limited review time on exactly the rows where competent models disagree. Context reliance requires an *exact* set match to count as agreement (four binary signals, and they are what this retrain exists to fix); topic and intent require a non-empty intersection, since two competent labelers routinely pick 2-of-3 the same and demanding equality there would bury the user in reviews that don't change the model. The v1 rubric — source-aware evidence thresholds, six immunity traps, signals A–F, reasoning-before-labels — is reused nearly verbatim, but its decision *tree* is gone: v1 stopped at the first hit ("if real-time signals → Real\_Time\_Search. STOP."), while v2 asks four independent questions and answers all four. Label definitions are rendered **from label\_schema.json**, so a label cannot mean one thing to the labeler and another to the trained head. **Temporal\_Recall gets free weak supervision**: T2's deterministic detector (src/retrieval/timescope.py) runs over the corpus and every hit is a positive — the only label starting from zero positives that can be seeded without fabrication. Output is JSON-schema constrained at the server, which replaces v1's `instructor` retry-on-invalid-JSON loop.

**Stage 4b — reconciliation, and what reaches a human.** Agreement is decided per head, and the three heads are not equal. Context reliance demands an exact match on the three memory signals; `High_Complexity` is excluded (`SOFT_CTX_LABELS`) and settled by majority with a 1-1 split resolving to absent, because it is the weakest-agreeing label (83–84%) and its errors are **asymmetric in exactly one deployment**: local-only and cloud-only ignore the signal entirely, while in mixed local+cloud a false positive spends the user's credits and a false negative costs one turn's answer quality. Topic and intent are fuzzy multi-label sets, so a three-way split there **resolves by union rather than by a person** — measured, 4,657 of 5,201 queued rows were fuzzy-head-only disputes, i.e. 14 hours of reading to adjudicate tags that nudge leg weights, while the 544 context disputes decide whether memory is consulted at all. Union imports some false positives; that is the cheap direction. Net result over the real corpus: **98.8% settled, and the human queue is 544 rows, all context reliance**.

**Stage 4c — the two synthetic piles, which must not be conflated.** *Pile A* (`synth.py`) is bulk model-generated filler for measured gaps and earns its labels through the normal two-labeler path, because generation drifts (ask for `Codebase_Query`, receive a `Code_Change`). *Pile B* (`authored.py`) is hand-written **for** a specific label combination, ships its label WITH the prompt, and never reaches the labelers — sending it to them would let two local models overrule an authored ground truth, and for the labels Pile B exists to fix the labelers are precisely what gets them wrong.

Pile B exists because of **capability censoring**: a label gated on a capability the data-collection environment lacked stays rare no matter how much data is gathered, and scaling the corpus cannot fix it. The corpus is website chats, so: `Codebase_Query` had 65 rows (no repo access — "where is X in my project" was a pointless question); **`Needs_Memory` across conversations had ZERO of 6,806** (referring to another conversation is futile when the assistant cannot see it, so nobody phrases it that way — and that is the single case ICE exists to serve); `Memory + Live_Info` ~137 (no web search); `Meta_AI` about ICE's own memory none (no system worth interrogating). In the real run Pile A was never needed — Pile B closed the only remaining gap and `measure_gaps` returned empty.

**Stage 5 — build.** Joins labels to text (human decisions override models), down-samples *standalone* rows until context-prefixed rows reach **≥40%** (never the reverse — context rows are the scarce ones), synthesises **hard-negative context pairs**, reports per-label floors, and splits **grouped by conversation** so two turns of one conversation cannot straddle train and test.

The hard-negative pairs are the payoff of the context-aware exercise, and need no second labeling call: take a row the labelers saw *with* context and judged not to need memory — the context supplied the referent — whose text is referentially ambiguous alone ("so which of those should I pick?"); strip the context and the referent is definitionally gone, so the twin *does* need memory. Identical text, opposite answer, and the only difference is whether history was attached. Referential detection reuses `memory_decision.REFERENTIAL_WORDS` rather than a second private list.

**Stage 6 — train.** Trunk + three heads (§2.1) on the native 1024 embedding, per-head BCE with capped per-label pos-weights (`--pos-weight-cap`, shipped at 5), AdamW(lr 1e-3), early stopping on held-out loss. Every row is rendered through the shared templates (§2.3). Trunk width stays deferred to Z1-prep; the tag threshold is no longer deferred — it is fitted by **stage 6c** and stamped into the checkpoint (§2.1).

**Stage 6c — sweep\_threshold.** Fits the decision threshold on val, reports on test, and with `--write` stamps `tag_threshold` into the checkpoint. This exists because the inherited 0.3 was badly wrong for a pos-weighted model and no amount of retraining fixes a bad threshold: 0.526 → **0.610** mean macro-F1 from this step alone, larger than every architectural change in B1 put together.

**Stage 6b — the independent evaluation set (`build_eval_probes.py`).** `train/val/test` all descend from the same two labelers, so they inherit whatever those labelers are jointly wrong about; a split cannot detect the bias of the process that produced it, and Pile B cannot serve as the exam because it is trained on. `data/labeled/v2/eval_probes_independent.jsonl` holds **207 probes the user wrote months earlier** for Experiment-1 curation, untouched by any labeler in this pipeline (238 unique of 708 rows; 42 dropped as already present in the corpus). It asserts ONE label — `Needs_Memory`, true by construction, since a curation probe is asked precisely to test recall — and carries the old topic/intent as an unscored hint. The gate it provides is narrow and exactly the behaviour the memory system rests on: *on real memory-needing prompts the user actually typed, how often does the head fire `Needs_Memory`?*

**Stage 7 — evaluate.** Per-label and per-head macro-F1, plus **D5's non-regression gate**: both models scored on *identical* rows, each rendered as it was trained (the baseline through the frozen v1 template and slice384, the candidate through v2 at 1024), compared only on labels both can express — 11 topic, the first 11 intent, and the derived three-way. **Each model is scored at its own stamped `tag_threshold`**, since holding one number fixed across generations measures the calibration gap rather than the models. The roadmap's "we cannot know whether the retrain lands better or worse" made operational. Stated caveat: gold labels are v2 labels, so the baseline is graded on a rubric it never saw; the gate is a floor ("not worse at the old job"), not proof of improvement.

**Stage 7b — `eval_probes.py` and `score_hard_probes.py`, the independent gates.** The probe gate scores the 207 user-written probes for `Needs_Memory` firing, always beside a **false-fire control** drawn from held-out rows with no memory label — a recall-only set is trivially gamed by a head that fires on everything, which was run 1's exact pathology. `hard_probes.py` adds **104 hand-authored adversarial probes** (in the script, so they are version-controlled unlike `data/labeled/`), 52% positives and 48% must-stay-silent controls, each carrying the boundary it tests and what a failure would prove. `score_hard_probes.py` runs the real inference path and **prints every miss with the model's probabilities**, because a pass rate alone repeats the mistake the whole exercise exists to correct.

**What the v2 run actually established — the label ceiling (2026-07-27).** The most important result of B1 is not a score, it is a limit. Measuring per-label agreement *between the two labelers* and placing it beside the trained model's per-label F1 gives a correlation of **0.90 with a mean gap of −0.01**: on every label the model scores within a hair of what its own supervision agrees on (Needs_Memory 0.79 labelers / 0.79 model; Generation 0.76 / 0.75; Codebase_Query 0.10 / 0.10). **The model has extracted essentially everything its labels contain.** The operational consequences are concrete and should stop future sessions repeating the work: more epochs, a wider trunk, a different pos-weight cap, or a fine-tune on the same corpus cannot move these numbers, because none of them add information; the only lever that can is supervision that does not come from these labelers — hand-authored rows (Pile B's pattern) or real usage feedback. It also means **the held-out split has stopped being informative** and the independent probe sets (stage 6b, 7b) are the only instruments that can still detect a regression.

A second measured caution about the labels themselves: the intent head's disagreements are **90–100% one-directional** rather than mutual (labeler A said `Factual_Retrieval` where B said `Open_Exploration` 1,030 times, and the reverse 8 times; B uses `Open_Exploration` 4.1× and `Ideation` 2.4× more often than A overall). That is a *calibration* difference between two models, not evidence that the two labels overlap — so collapsing confused labels would destroy real distinctions to paper over one labeler's bias. Any future taxonomy rework must separate these two causes before merging anything.

**Stage 7c — `eval_di3.py` and `audit_labels.py`, the consumer audits.** Both exist because of a failure mode B1 kept surfacing: a signal can be trained, accurate, stored on every result, and still change nothing. `eval_di3.py` scored DI3 against the head on the rows DI3 intercepted (§2.2). `audit_labels.py` does the same for labels rather than components — it measures how much of an intent profile survives the orchestrator's `len(active_intents)` division, and whether a context signal moves the decision it is wired to. **Neither `evaluate.py` nor `eval_probes.py` can do this job:** both call `load_checkpoint` directly and never run `classify()`, so anything in the *pipeline* around the head is invisible to them. Changes to the pre-classification path are scored end-to-end on the independent probes instead.

**Stage 8 — promote.** Re-runs the gate (rather than trusting a possibly-stale report), then backup + atomic replace of settings.classifier\_model\_path via src/classifier/promotion.py — **one** implementation, shared with the B4 curated-label fine-tune worker.

### **2.5.1 Training pipeline (v1, historical)**

The v1 classifier was trained offline through a five-stage pipeline. Kept here because the current checkpoint was produced by it, and because the labeling rubric it developed is the basis of the v2 one.

**Stage 1 — Amnesia Method data harvesting.** Personal conversational archives are mined by scripts/classifier/promt\_extraction/extract\_promts.py, which uses qwen3-coder:30b-a3b-q4\_K\_M on Ollama to extract user-authored prompts from a raw chat corpus under a \*precision-over-recall\* contract: chunks of 12000 characters with 1000-character overlap, temperature=0.0, a confidence floor of 0.85, and 21 hard regex filters that reject AI openers ("sure,", "here is", "good catch", …), mid-sentence fragments, and structural markers. The same stage harvests 5 000 prompts each from three public datasets — lmsys/chatbot\_arena\_conversations, ShareGPT\_Vicuna\_unfiltered, and allenai/WildChat-1M — filtered to English first-turns. All sources are deduplicated by SHA-256 of the normalised text and shuffled under RANDOM\_SEED = 42 by combine\_dataset.py.

**Stage 2 — vLLM labelling.** scripts/classifier/promt\_labeling/VLLM\_label\_dataset.py labels the blended corpus with Qwen/Qwen2.5-7B-Instruct-AWQ served on vLLM (port 8001), temperature=0.0, seed=42, CONCURRENT\_REQUESTS=20, structured output enforced via instructor.Mode.JSON. The labelling prompt is a decision tree applied in strict order: (i) a \*source-aware calibration\* — personal sources use a \*low\* LTM threshold, the three public sources use an \*extremely high\* threshold that requires an explicit continuation phrase; (ii) six \*immunity traps\* that short-circuit to Zero\_Shot (pasted context, public entities, self-contained hypotheticals, role assignments, quoted pronouns, and a source-specific continuation-phrase allow-list for public datasets); (iii) Real\_Time\_Search signals (current price, live score, today's news, …); (iv) six Long\_Term\_Memory signals A–F covering demonstrative references, personal possessives, continuation language, named personal entities, implicit subjects, and questions about the user's own history; (v) default Zero\_Shot. The model is required to fill a reasoning field answering four questions (source, immunity, signals, decision) \*before\* emitting labels, which materially improves label consistency. Intent labels are capped at three.

**Stage 3 — Label vectorisation.** build\_training\_data.py converts each labelled record into a 25-dimensional multi-hot vector (11 topic + 11 intent + 3 one-hot context-reliance), skipping orphans where either topic or intent is empty.

**Stage 4 — From-scratch training.** train\_classifier.py trains all MLP parameters with Adam(lr=1e-3), batch\_size=32, epochs=30, a 10 % validation split, and early stopping (PATIENCE=5). The loss is BCEWithLogitsLoss(pos\_weight=topic\_pos\_weight) + BCEWithLogitsLoss(pos\_weight=intent\_pos\_weight) + CrossEntropyLoss(), where the pos\_weight for each of the 22 multi-label columns is num\_neg / num\_pos clamped at pos\_weight\_cap=15.0 (no weighting on the 3-dim context block). The best checkpoint by validation loss is saved as ice\_classifier\_v2.pt.

**Stage 5 — Iterative fine-tuning.** fine\_tune.py loads a checkpoint, **freezes \`fc1\`** so only the fc2 head is trainable, and trains for 10 epochs at lr=5e-5 with plain BCEWithLogitsLoss (no pos\_weight). Hand-curated corrections in data/curated\_fixes.jsonl are pre-encoded, repeated 50×, and interleaved into every training batch via itertools.cycle, weighted 10× in the loss — the human-in-the-loop correction mechanism. The output is ice\_classifier\_v3\_qwen\_ft.pt. The active inference path loads ice\_classifier\_v3\_qwen\_ft3.pt (set by settings.classifier\_model\_path).

### **2.6 The temporal joint gate, and what it revealed about style-dependent rules**

`retrieval/timescope.py::detect_timescope` decides whether a parsed time expression means "search that period". Detection is deliberately two-layered, and the layers are not equally sound.

**The parser is regex and that is correct.** `_scan_expressions` matches about twelve date shapes (ISO, `march 2025`, `5th of march`, quarters, `early/mid/late 2025`, seasons, anchored bare years, `two years ago`) and resolves each into a UTC window. This *converts* rather than *guesses*, and a model would be strictly worse: an invented window silently hides every memory outside it, which is why Track T's standing invariant is that **only the deterministic parser may ever set a window** — a sigmoid may gate and may rank, never resolve. Its ceiling is real but belongs elsewhere: 250 of 557 gold-temporal rows ("back when we started", "before the rewrite") match no shape it knows, and that half is T5(b)'s to handle by flattening recency, not the parser's to guess at.

**The gate above it infers intent from typography, and that is a defect.** A resolved expression flips the mode only if the prompt also looks like a recall question: `"?" in text`, else `first_word in _INTERROGATIVES` (a twelve-word list), else `p_ltm ≥ 0.5`. Two of those three arms read surface form. A fourth arm, DI3's reference density, was deleted in D8 after it was found admitting 49 false windows on long pasted documents.

**The measurement that makes this a standing lesson (2026-07-28).** The first-word interrogative test fires on 13–25% of the public corpora — and on **2%** of the user's own writing. The signal is not missing: an interrogative sits in the first eight words of **23%** of their rows. The rule reads *position 0*, and they write `"so what about comparision to the ground truth"`. It discards **21%** of their rows against 10% of the corpus's, i.e. it loses twice as much signal on the actual user as on the data ICE was evaluated against. A rule keyed on punctuation or word order is a bet on writing convention, and every corpus ICE was evaluated on is "people typing at a chatbot" — the bet is hidden, not absent. **The goal is invariance, not personalization:** the maintainer is an existence proof that conventions break, not a target population, and the rule is unstable across the public corpora too (25% / 14% / 13%). ⚠ **The trained head is not automatically the fix** — firing-rate spread across the four sources is 1.5x for `has "?"` against **16.2x for `p_ltm >= 0.5`**. That is confounded by real content differences, and the confound is the lesson: firing-rate-by-source cannot separate different people from different meanings, so it is a smell detector, not a test. The acceptance test holds meaning fixed and varies only form, measuring the **decision-flip rate** across stylistic variants of one prompt — and it applies to the replacement as much as to the rule. Roadmap **G28** owns the sweep and the style-variant probe set; the standing rule is in CLAUDE.md.

### **2.7 Experimental / partial features**

A ConfigurableOrchestrator subclass exists for ablation studies; it does not redefine the classification logic but exposes flags that toggle retrieval legs on and off (bm25, vector, codex, procedural, rag, cluster\_restrict, hyde, dynamic\_budget). The hyde flag is the only path that enables HyDE query rewriting — in the production orchestrator the HyDE call is commented out (see §6.2).


## **3. Memory Architecture**

ICE maintains **four distinct memory stores**, plus **Memory Slots** (persistent structured working memory), **Context Clusters** (conversation-scoped topical clusters), and a **cold-storage** archive. All stores live in a single PostgreSQL database with the pgvector extension; every vector column is **Vector(1024)** (C17, 2026-07-19 — native width, migration b6e2f9a41c73) because the same process-shared Qwen/Qwen3-Embedding-0.6B embedder (src/memory/embedder.py::get\_embedder() — one instance, no truncation) is used across classification, retrieval, clustering, and the workers; the micro-NER still consumes the slice384 MRL prefix of the same encode (A9-gated), and the classifier does too **only while a v1 checkpoint is live** — its v2 path reads the native 1024 and a loaded checkpoint declares which width it wants (B1, §2.1), so the slice retires for this consumer at promotion. store\_meta stamps the store's embedding identity — §10.7.

### **3.1 Episodic memory**

The episodic\_memory table is the system's primary store of conversational turns. Its schema (every column):

| **Column** | **Type** | **Notes** |
| - | - | - |
| **id** | UUID PK | uuid.uuid4 |
| **conversation\_id** | UUID FK → conversations.id |  |
| **cluster\_id** | UUID FK → context\_clusters.id (nullable) | legacy single-cluster pointer; the M2M link table is authoritative |
| **parent\_message\_id** | UUID self-FK | threading |
| **batch\_id** | UUID | groups a user turn + assistant reply; the join key used by Codex/Procedural workers |
| **session\_id** | UUID (nullable, indexed) | C6: one *sitting* — resolved at write time (src/memory/session.py): the previous turn's session is reused while the gap ≤ settings.session\_gap\_minutes (default 30), else a new one is minted (logged as `session_started`, the C7 trigger seam). NULL on pre-migration rows |
| **is\_private** | Boolean default false (indexed) | G16: incognito flag for none-scoped conversations — see §6.10 |
| **timestamp** | DateTime(tz) | utcnow |
| **topic\_tags / intent\_tags** | ARRAY(Text) | classifier output; Creative\_&\_Media triggers the decay floor |
| **context\_reliance** | Text | classifier label |
| **entropy\_score** | Float | C1: facts-per-token density (entity/figure/code/diversity blend) — written by the Post-Flight Evaluator (was NULL-forever until the 2026-07 rework) |
| **lossless\_flag** | Boolean (nullable) | NULL = not yet evaluated; True exempts from batch summarisation and gates Codex extraction |
| **raw\_text** | Text | full turn text |
| **summary\_text** | Text | grounded post-flight summary (stored ALONGSIDE raw — read time chooses, §6.1a) or compaction summary |
| **summary\_coverage** | Float (nullable) | C1: measured must-term retention of summary_text; below 0.7 the summary is never preferred or degraded to; NULL = no/legacy summary |
| **abstract\_text** | Text (nullable) | C3: one-line abstract (same LLM call as the summary) — last level of the raw→summary→abstract degradation hierarchy |
| **embedding** | Vector(1024) |  |
| **decay\_score** | Float default 1.0 | multiplied each decay cycle |
| **access\_count** | Integer default 0 | incremented on retrieval |
| **is\_archived / is\_bookmarked / decay\_immune / inject\_raw / is\_document** | Boolean | various flags |
| **idempotency\_key** | Text UNIQUE | API-layer deduplication |


**Population.** The FastAPI proxy inserts one row per user/assistant turn in store\_turn\_async (a BackgroundTasks callback that runs after the SSE stream closes). The Post-Flight Evaluator writes lossless\_flag, summary\_text, inject\_raw, and is\_document; the Batch Summariser writes summary\_text for decayed-but-not-yet-archived turns.

**Querying.** Two retrieval legs read this table (§6.1–6.2): the BM25 leg uses ts\_rank over to\_tsvector('english', coalesce(raw\_text,'')||' '||coalesce(summary\_text,'')) with a plainto\_tsquery built from the top-30 stop-word-filtered prompt tokens, filtered by decay\_score \> 0.2 AND is\_archived = false, LIMIT 100; the vector leg uses the decay- **and recency-**weighted cosine score (1 - (embedding \<=\> :prompt\_embedding)) \* COALESCE(decay\_score, 1.0) \* (1 + 0.25·exp(−age\_days/30)) under the same visibility invariant, LIMIT 100 — **C8 (2026-07):** the recency factor sits *inside* the SQL score (candidacy, not just post-fusion rank; a recent turn that misses the top-100 can't be rescued by bonuses), mirrors A11's codex formula, applies to the chunk leg and wide net too, and is skipped for Creative\_&\_Media (same recent-meta-is-noise rule as the bonuses). After fusion, retrieved turns are \*strengthened\*: access\_count += 1 and decay\_score = min(1.0, decay\_score + 0.15).**C2/C3 (2026-07): chunk-granular memory.** Chunking covers documents (C2) **and all long turns** (C3, > ~600 words — chapters, big technical answers): post-flight dispatches the chunker for both, and the 2h catch-up heals legacy rows (SQL length proxy). The shared chunker is **heading-aware** (C3: a markdown heading always starts a new chunk — section-level decomposition; no overlap is carried across a section boundary). In the vector leg, non-document parents stay in turn-level search and their chunks compete alongside with **dedupe** (a chunk drops out when its parent turn is already in the results); document parents are excluded entirely. **Documents are never injected whole anymore.** A doc row found by text search injects only its keyword-relevant chunks (≤2, opening chunk as fallback) from the **episodic\_chunks** table (chunked by workers/document\_chunker.py using the shared chunker src/memory/chunking.py — the A1 sentence/code-aware packer, extracted to serve extraction windows, document chunks, and future C3/C12 alike). The vector leg excludes is\_document rows entirely (a whole-doc embedding is semantic mush) and searches **chunk embeddings** instead (\_vector\_chunks: visibility — decay/archive/privacy/scope — enforced through the parent-turn join; ≤3 chunks per document; provenance points at the parent so strengthening lands on the turn). Legacy pre-C2 docs without chunks fall back to the normal 500-word cap until the runtime's 2h catch-up (run\_pending\_documents) heals them; new documents are chunked by a direct run\_chunk\_turn call inside the post-flight job (C7; private docs included — chunks inherit visibility via the join).

### **3.2 Codex (semantic) memory**

The Codex store is a versioned knowledge graph spread over four tables (full schema in §4): codex\_entities (canonical name, aliases, tags, properties JSONB, auto-regenerated context\_payload, embedding Vector(1024)), codex\_edges (typed relations with strength, confidence ∈ \{pending, active\}, valid\_from, valid\_until), codex\_events (append-only event log), and codex\_snapshots (compaction output). It is populated by the Codex Extractor (§4.3) and the manual Codex Inject Watcher (§8.9), enriched by the Reflection worker, and queried by the Codex graph-traversal retrieval leg (§4.8).

### **3.3 Procedural memory**

The procedural\_memory table stores recurring behavioural patterns (e.g. "the user always asks for tests after code generation"). Its schema:

| **Column** | **Type** | **Notes** |
| - | - | - |
| **id** | UUID PK |  |
| **pattern\_name / pattern\_description** | Text | name = first 80 chars of description |
| **topic\_tags** | ARRAY(Text) |  |
| **trigger\_conditions** | JSONB default \{\} | \{"topic\_tags":\[…\], "intent\_tags":\[…\]\} gating set |
| **reinforcement\_count** | Integer default 1 | incremented on each repeat |
| **confidence\_score** | Float | 0.3 for new patterns, promoted to 0.8 at reinforcement\_count ≥ 3 |
| **first\_observed / last\_observed** | DateTime(tz) |  |
| **is\_active** | Boolean | schema default True, but extractors create with False until promoted |
| **source\_batch\_ids** | ARRAY(UUID) | conversation-scoped retrieval key |
| **embedding** | Vector(1024) | of pattern\_description |


It is populated by the Procedural Extractor (§5.2) and the Reflection worker's \_crystallize\_patterns step, and queried by the procedural retrieval leg (§5.4).

### **3.4 RAG memory**

Two tables — rag\_documents (id, filename, file\_type, uploaded\_at, token\_count) and rag\_chunks (id, document\_id FK, chunk\_index, chunk\_text, embedding Vector(1024)) — hold externally-ingested documents. They are populated exclusively by the Drop Zone worker (§8.8), which chunks content into **512-word windows** and embeds each with the classifier's embedder. The RAG retrieval leg is the only consumer.

### **3.5 Memory Slots (three tiers since C9, 2026-07-19)**

The memory\_slots table is the system's \*persistent structured working memory\* — slots whose contents are prepended verbatim to every prompt. Schema: id UUID PK, slot\_name Text, content Text, token\_count Integer, version Integer (bumped on every update), last\_updated, updated\_by ∈ \{user, agent, reflection, chat\_command, mcp\_edit, system\}, is\_active Boolean, plus the C9 tier columns: **scope\_tier ∈ \{global, project, conversation\}** (default global), project\_id (FK, project tier only), conversation\_id (FK, conversation tier only). Uniqueness is a **NULLS NOT DISTINCT unique index** on (slot\_name, scope\_tier, project\_id, conversation\_id) — before C9 the table had *no* name uniqueness at all. Valid names live in ONE per-tier dict, `src/services/slots.py::VALID_SLOTS`:

global: persona · user\_preferences · tool\_guidelines · project\_context · guidance · pending\_items · session\_patterns  
project: project\_context · conventions · pending\_items · guidance  
conversation: conversation\_focus · pending\_items

The service validates names per tier and requires the tier's anchor (a project-tier write without a project\_id raises naming the missing attachment); **G14's 300-token cap** (`SLOT_TOKEN_CAP`) hard-truncates every write with a `truncated`/`warning` flag in the response. Slots are loaded on every chat request and rendered into the system message under === PERSISTENT CONTEXT === in tier order — global as `[NAME]`, then the attached project's as `[PROJECT · NAME]` (coding conversations inherit them via `scope["project_id"]` — the E1 payoff), then this conversation's as `[CONVERSATION · NAME]` (§7.2). Write paths: direct user update (REST, tier query params), `ice_slots`/`ice_remember` (MCP), batch initialisation (global tier only), **Reflection/agent-proposed but user-gated** updates (review\_queue; approval applies through the same service, recording the proposer as updated\_by — D7), and **Reflection auto-applied** updates to pending\_items only. The human-in-the-loop boundary stands: high-stakes steering slots require ratification, low-stakes bookkeeping is autonomous.

### **3.6 Context Clusters**

Context Clusters are conversation-scoped, automatically generated topical groupings of episodic turns. The context\_clusters table holds id, name (LLM-generated short name), description (structured as DOMAIN: / CONTENT\_TYPE: / RECURRING\_ENTITIES: / SETTING\_OR\_CONTEXT: lines), tags (union of member turns' topic\_tags), conversation\_id FK, created\_at / updated\_at, and embedding Vector(1024) — the **centroid** of member turn embeddings, renormalised to unit length after every member change. Membership is many-to-many through episodic\_cluster\_links(episodic\_id, cluster\_id) with a composite primary key, so a turn may belong to multiple clusters (soft multi-assignment).

The Clustering worker (workers/clustering.py::run\_cluster\_assignment, runtime-scheduled every 30 minutes + session-gap freshening, bounded to MAX\_TURNS\_PER\_RUN = 25 unassigned turns per invocation) assigns each unassigned turn to the best existing cluster — combined score embedding\_similarity + 0.08 per shared entity (capped at 0.30) + tag overlap — or to a new cluster when no candidate clears SIMILARITY\_THRESHOLD = 0.6. The cluster name/description are regenerated only every NAME\_REGEN\_INTERVAL = 5 members to avoid naming churn. A separate run\_cluster\_merge callable (runtime-scheduled every 3 h) merges clusters in the same conversation with centroid similarity above MERGE\_SIMILARITY\_THRESHOLD = 0.90 (and a raw-similarity floor of 0.82). Full v5 mechanics in §8.5.

### **3.7 Decay and archival mechanics**

Three independent decay workers, all runtime-scheduled every **5 400 s (1.5 h)** for CYCLES\_PER\_DAY = 16 cycles/day, implement access-weighted decay — each takes a `cycles` parameter so the runtime's ledger-derived catch-up compresses missed runs into one `rate ** cycles` pass (C7 D5).

**Episodic decay** (workers/decay.py) applies one of three per-cycle multipliers, gated by a 7-day recency cutoff (turns younger than seven days are immune):

- DECAY\_RATE\_UNACCESSED = 0.95^(1/16) ≈ 0.9968 (≈5 %/day) — access\_count = 0, non-creative.

- DECAY\_RATE\_ACCESSED = 0.98^(1/16) ≈ 0.9987 (≈2 %/day) — access\_count \> 0, non-creative.

- CREATIVE\_DECAY\_RATE = 0.99^(1/16) ≈ 0.9994 (≈1 %/day) — any turn tagged Creative\_&\_Media.

A **creative floor** clamps decay\_score to 0.3 for creative turns regardless of age or access, protecting long-form narrative from being summarised away. Turns whose decay\_score falls below ARCHIVE\_THRESHOLD = 0.1 are flipped to is\_archived = TRUE and drop out of *current-mode* retrieval (the legs filter is\_archived = false — but time-scoped queries see archived rows, §6.11/D10). **T3 (D11) fixed the archived-freeze bug:** the decay UPDATEs used to filter is\_archived = FALSE, freezing every archived row at ~0.1 forever — nothing could ever reach the cold threshold, so cold\_storage was empty and unreachable. Archived rows now keep decaying at their access-class rate, and a symmetric **un-archive clause** (run between decay and archive steps) restores any row whose score recovered above 0.1 — which is how a time-scoped retrieval hit (+0.15 write-on-read) or a resurrected row earns its way back. Turns that fall below COLD\_THRESHOLD = 0.05 are **moved** to the cold\_storage table (id preserved; archived\_at, raw\_text, summary\_text, topic\_tags, timestamp, and — since T3/D12 — conversation\_id, is\_private, batch\_id) and physically deleted from episodic\_memory — an idempotent INSERT … ON CONFLICT (id) DO NOTHING followed by DELETE. Cold storage is no longer retention-only: **time-scoped queries search it** (§6.11), and an injected cold hit is resurrected on probation (D-U1). Bookmarks set decay\_immune = TRUE and are exempt from all decay.

**Codex decay** (workers/codex\_decay.py) multiplies codex\_edges.strength by 0.99^(1/16) for **every live edge (valid\_until IS NULL) — pending included (A3)**; previously only active edges decayed, which let a retrieval-reinforced pending edge inflate without ever entering the decay cycle. Active edges are demoted to confidence = 'pending' once strength \< DEMOTION\_THRESHOLD = 0.3. Demotion does not set valid\_until (the edge remains \*true\*, just unreinforced). **A3 garbage collection:** pending edges that decay below EXPIRY\_THRESHOLD = 0.1 without ever being corroborated or retrieved are expired (valid\_until = NOW()), so uncorroborated low-trust residue (e.g. grounding-rejected triplets) leaves the live graph instead of accumulating. Re-asserting a demoted edge reinforces it (strength += 1.0) and re-promotes once strength ≥ 2.0; retrieval reinforcement can also promote (§4.8).

**Procedural decay** (workers/procedural\_decay.py) is boolean, not numeric: SET is\_active = FALSE WHERE is\_active = TRUE AND last\_observed \< now() - 180 days AND reinforcement\_count \< 3. A pattern that has been promoted (≥3 reinforcements) is permanently protected from time-based decay. Patterns are never hard-deleted.

### **3.8 Batch summaries**

The batch\_summaries table (id, conversation\_id FK, start\_turn\_index, end\_turn\_index, summary\_text, embedding Vector(1024), created\_at) holds compressed representations of decayed turns. The Batch Summariser (§8.7) groups turns with decay\_score \< 0.3 AND lossless\_flag = False AND is\_document = False by conversation, chunks them into 50-turn batches (skipping batches smaller than 5), and calls the background model under a strict preservation prompt (temperature=0.0, max\_tokens=500). The summary is the \*second life\* of a turn — it coexists with the still-live original until the original ages into cold storage, at which point the summary is the only live-retrievable trace under current mode (a time-scoped query can still reach the cold row itself, §6.11). Retrieval is conversation-scoped (top-3 by cosine), skipped under non-current TimeScope modes (D14).


## **4. Codex Knowledge Graph (v2)**

Codex v2 is a temporally-versioned, controlled-vocabulary knowledge graph. Its central design lever is a three-bucket relation vocabulary (property, single-valued, multi-valued) that forces the extractor to commit to a specific cardinality per fact, which in turn drives well-defined expiry, reinforcement, and retrieval semantics.

### **4.1 Entity and edge tables**

codex\_entities (PK id — a deterministic UUIDv5 in CODEX\_NAMESPACE for new entities) holds canonical\_name (unique), aliases ARRAY(Text), tags ARRAY(Text), **entity\_type Text (A7 — structural node type; ICE is general-purpose, so the vocabulary spans all domains: universal person/place/organization/event/concept/object, coding/research software/function/class/file/module/dataset, academic/business document/product, creative character/location/item/creature/faction; inferred from an entity's relations via \_infer\_entity\_type, or set deterministically by the code graph E1b)**, **description Text (A7 — the enriched "note body", written by the reflection enrichment worker)**, properties JSONB, an auto-regenerated context\_payload Text, embedding Vector(1024), and last\_updated. **A7 rich notes:** \_regenerate\_context\_payload assembles context\_payload as a bidirectional Obsidian-style "note" — the description (note body), then Properties, then **Links** (outgoing edges, strongest-first) *and* **Backlinks** (incoming edges — what points *to* this entity, previously invisible) — so an entity's full connection picture reaches retrieval in both directions. codex\_edges holds id, source\_id/target\_id FKs to entities, relation Text, strength Float (default 1.0; new non-property edges start at 1.0, property and replacement edges at 3.0), source\_batch UUID (the batch that asserted the edge), confidence Text ∈ \{pending, active\} (promoted to active at strength ≥ 2.0), **extraction\_confidence Float ∈ \[0,1\] (A3)** — how much the extraction itself is trusted: seeded by NER grounding at write time (0.9 grounded, 0.7 no-NER chunk, 0.35 grounding-rejected; legacy edges 1.0), raised to the max seen on corroborating re-extraction — valid\_from, and valid\_until — **\`NULL\` means currently true**; non-NULL means historically superseded. An edge's **effective trust** used by retrieval is strength × extraction\_confidence (strength carries usage dynamics; extraction\_confidence carries extraction trust). Two auxiliary tables complete the graph: codex\_events is an append-only event log (event\_type ∈ \{edge\_added, edge\_expired, edge\_strengthened, property\_updated, context\_appended\}, payload JSONB, batch\_source, compacted Boolean), and codex\_snapshots holds point-in-time full\_state JSONB produced by the compaction worker.

### **4.2 Controlled relation vocabulary**

The vocabulary is defined in workers/codex\_extractor.py as three disjoint dicts keyed by category, with an import-time assertion guarding against accidental overlap. Four design rules (quoted verbatim from the source comment) govern it: (i) one canonical relation per meaning (near-synonyms collapsed, look-alikes kept distinct); (ii) generic dumping-ground relations (is, has, applies\_to, connects\_to) removed entirely so the extractor is forced to be specific; (iii) relations grouped into labeled categories so the extraction prompt renders them grouped, not as a flat 150-item list; (iv) single-vs-multi-valued cardinality reassigned by real-world cardinality (e.g. works\_on, founded\_by, created moved to multi-valued).

- **\`PROPERTY\_RELATIONS\`** (11 categories, written to the properties JSONB): identity (name, alias, nickname, full\_name, title, description), demographics (age, gender, species, nationality, religion, birthday, blood\_type), appearance, professional (role, occupation, profession, affiliation, status), contact\_location, metadata\_generic (type, genre, format, license, version, language), task\_management, attribution (author), technical\_specs, narrative\_metadata, abstract\_metadata. A new property observation expires the prior active edge and overwrites the JSONB key.

- **\`MULTI\_VALUED\_RELATIONS\`** (16 categories): technical\_dependency (uses, imports, depends\_on, supports, integrates\_with, calls, returns, references, cites, extends, implements), technical\_distribution, structural\_containment, social\_relationship (friend, ally, enemy, colleague, knows), social\_action, organisational\_collab, support\_endorsement, activity\_participation, categorisation, works\_on\_projects, founding\_creation\_multi, data\_lineage (ICE-specific: derived\_from, trained\_on, configured\_with, evaluated\_on, benchmarks\_against), code\_structure, research\_relations, narrative\_structure, conceptual. Multiple active edges coexist; new edges never expire prior ones.

- **\`SINGLE\_VALUED\_RELATIONS\`** (11 categories): organisational\_position (part\_of, works\_at, reports\_to, managed\_by, …), executive\_role, succession, education, production\_singular, personal\_relationship (married\_to, parent\_of, child\_of, sibling\_of, …), biography\_location (lives\_in, born\_in, died\_in, …), deployment\_ownership, logical\_requirement, narrative\_singular, conceptual\_singular (instance\_of, subtype\_of). A new edge auto-expires the prior active edge of the same (source, relation).

### **4.3 Temporal versioning and automatic property updates**

handle\_triplet is the heart of the write path. Entity resolution (get\_or\_create\_entity) is two-stage: exact canonical\_name match, then alias match, else create with a deterministic UUIDv5 and a fresh embedding. The write branch then depends on the bucket:

- **Property.** Expire every active edge of (source, relation) by setting valid\_until = now, emit edge\_expired events, write a new edge at strength=3.0, confidence="active", set properties\[relation\] = object (with flag\_modified for in-place JSONB mutation), and regenerate context\_payload.

- **Non-property, same \`(source, target)\` pair, same relation (reinforcement).** strength += 1.0; if strength ≥ 2.0 and confidence == "pending", promote to active; emit edge\_strengthened.

- **Non-property, same pair, different relation.** If the \*old\* relation is not multi-valued, expire it (valid\_until = now, edge\_expired event); if it \*is\* multi-valued, leave it (this branch was the subject of an explicit bug fix — previously multi-valued edges were silently dropped when a different multi-valued relation was later asserted between the same pair). Always add the new edge at strength=3.0, confidence="active".

- **No existing edge between the pair.** If the relation is single-valued, expire any other active edge of (source, relation). Add the new edge at strength=1.0, confidence="pending" (or 3.0/active if it replaced a single-valued edge).

Every state change emits a CodexEvent; context\_payload is rebuilt from the property map plus the (up to) ten most recent active outgoing edges.

### **4.4 Extraction pipeline**

When the task is dispatched from the bookmark endpoint, it receives priority=True, which causes it to skip the GPU‑utilisation gate (is\_gpu\_busy()) and the shared‑mode user‑activity gate. This is the only code path that overrides the yield‑to‑user constraint (INV‑5), ensuring that a user‑bookmarked turn is immediately processed into the knowledge graph.

The Codex Extractor (workers/codex\_extractor.py::extract\_codex, plain callable since C7 — a direct call inside the post-flight job when lossless\_flag == True; user\_control's bookmark endpoint enqueues it standalone as the "codex\_extract" runtime job with priority=True) runs the following pipeline:

4. **Chunking (roadmap A1).** The turn is split into **sentence/code-line-aware chunks of ≈`CHUNK_TOKENS` = 550 tokens** with `OVERLAP_WORDS` = 50 words carried into each subsequent chunk. `_chunk_text` isolates fenced code blocks from prose (`_split_segments`), breaks each into atomic units that are never split across a chunk boundary — sentences for prose, non-blank lines for code (`_atomic_units`) — and greedy-packs those units up to the token budget; a single unit larger than the budget is hard word-split as a last resort. Code is counted at a heavier token density than prose (`_estimate_tokens(..., is_code=True)` takes the larger of a word- and char-based estimate). Short turns come back as a single chunk. Each chunk is extracted independently and triplets are concatenated then deduplicated. *(Rationale: a 3–4B extractor's attention dilutes past ~1k tokens; the previous ≈4,511-word windows caused mid-passage entity dropping and self-referential hallucinations like `fastapi uses fastapi`. The 550-token size is shared with the NER-grounding step so both run off one chunking pass. `MAX_EXTRACTION_TOKENS` = 6000 is retained only for import compatibility.)*

4b. **NER grounding — entity confirmation (roadmap A2).** Before the LLM call, each chunk is run through the shared CPU micro-NER (`extract_entities`, §4.6) to produce a confirmed entity list. When non-empty, that list is injected into the extraction prompt as a **CONFIRMED ENTITIES** block instructing the model to use only those as subjects/objects and introduce no new named entities. This splits the cognitive load — the LLM reasons about relations, not entity discovery.

5. **Background-model extraction.** The extractor calls the background model via `get_bg_client()` / `get_bg_model_name()` (mode-selected: dedicated vLLM :8002 Qwen2.5-3B, or shared Ollama; see §10.5). The prompt renders the relation vocabulary as grouped \# category: … headers, imposes six strict rules (use only individual relation words; canonicalise subjects/objects; property facts use the property relation as the relation; the relation must make logical sense; never output a category header as a relation; output only a JSON array), appends a code-specific sub-prompt when Software\_&\_Tech is in the topic tags, and appends the CONFIRMED ENTITIES block from 4b. Decoding: temperature=0.0, max\_tokens=500, timeout=30.0.

6. **Triplet validation.** (i) Markdown-fence strip; (ii) json.JSONDecoder().raw\_decode with a regex fallback r'\\\{\\s\*"subject"\\s\*:\\s\*"(\[^"\]+)"\\s\*,\\s\*"relation"\\s\*:\\s\*"(\[^"\]+)"\\s\*,\\s\*"object"\\s\*:\\s\*"(\[^"\]+)"\\s\*\\\}'; (iii) shape filter (all three keys present); (iv) vocabulary filter (relation in ALLOWED\_RELATIONS); (v) verb-phrase filter dropping triplets whose object is in \{"blush","laugh","cry","smile","angry","sad","happy","mad"\}; (vi) self-reference filter dropping triplets whose normalised subject equals its object (kills `fastapi uses fastapi`); (vii) **NER grounding → extraction confidence** (`_ground_triplets`, A2+A3) — each triplet's subject (and, for non-property relations, object) is checked against the confirmed entities by normalised-exact or token-subset comparison, and the result sets its confidence: CONF\_GROUNDED = 0.9 on pass, CONF\_REJECTED = 0.35 on fail (**stored anyway** — retrieval's trust gates keep it out of context until corroborated, and decay expires it if it never is), CONF\_UNGROUNDED = 0.7 when NER returned no entities for the chunk (nothing to ground against). Cross-chunk dedup keeps the highest confidence seen per (subject, relation, object). `handle_triplet` stores the value on new edges as extraction\_confidence and raises it to the max seen on corroborating re-extraction.

7. **Deduplication.** Keyed on the case-normalised (subject, relation, object) triple, so multi-chunk repeats across overlap windows collapse.

8. **Write + reconcile (A6).** Each surviving triplet is passed to handle\_triplet, which now runs a **bounded self-correction loop** before applying the fixed write rules (non-property relations only — property relations already supersede). `check_conflict` is a cheap deterministic pre-filter: it queries the graph only when the relation has a known **antonym** (friend↔enemy, married\_to↔is\_divorced\_from, endorses↔criticises, …) or when a **multi-valued** relation coincides with a **supersession cue** in the turn text ("migrated off", "no longer", "switched from", "replaced", …); the ~95% of triplets with neither take a dict-lookup fast path. On a hit, `reconcile_conflict` resolves it: **antonym reversals are deterministic** — the newly-asserted state expires its opposite for that pair, *no LLM* — while **ambiguous supersessions** ("migrated off X" vs "considered migrating") are the *only* case that consults a model: a bounded one-word reconciler (`make_llm_reconciler`, background model, max\_tokens=5) returns expire\_old / keep\_both / reject\_new; anything else, or no reconciler, falls to a `review_queue` row with the new edge kept (never auto-expire on a guess). This deliberately keeps a small model's write-authority over the source-of-truth minimal (the notes' full autonomous "Memory Maintenance Agent" is Track D). `handle_triplet(turn_text, reconciler)` and the standalone `check_conflict`/`reconcile_conflict` are callable units so the Track D agent can later drive the loop with its own reconciler. *(The review-queue fallback is not user-visible until the frontend renders it — F2.)*

9. **Idempotency.** idempotency\_key = sha256("codex:" + batch\_id); on a hit the callable returns immediately. The key insert shares the worker's transaction, so a crash before commit leaves no key (retry re-runs) and a crash after commit leaves the key (retry is a no-op) — turning the runtime's at-least-once retries into effectively-once side effects.

### **4.5 Relation-aware retrieval and enumeration (A4 — replaces MERA)**

*MERA (retrieval/mera.py) was removed in the post-paper A4 rework: it scored −0.21 in the buildup ablation (loose triggers, an LLM tag/relation mapping that often missed the DB's actual tags, and a flat 15-entity payload dump). Its capability — answering category/enumeration and relation-shaped queries from the graph — was re-homed into the primary Codex leg as follows.*

**Relation detection** (`_detect_relations`) maps the prompt onto the controlled relation vocabulary through two LLM-free channels: (1) a **lexical channel** — a relation's own content words appearing in the prompt after crude two-sided stemming ("who inspired X" → inspired\_by, "who is X married to" → married\_to); multi-word relations require all content words, single-word relations one; and (2) an **embedding channel** — true-cosine similarity between the prompt embedding and once-per-process cached gloss embeddings (both natively unit-norm since C17; A4's re-normalisation workaround is deleted) of all ~200 relations, top-k=5 above a 0.45 floor, for paraphrases ("who is X's wife"). Empirically the joint set is recall-oriented and *deliberately not trusted alone*: neutral prompts also score ~0.69 absolute against random glosses, so detected relations only ever act **jointly** — paired with matched entities (fact surfacing, §4.8) or with explicit enumeration cues (below). Precision comes from the join, never the detector.

**Enumeration** (`_codex_enumeration`) fires only for entity-less prompts when an explicit enumeration cue is present (list, all, who are, what are, every, each, which, name the, tell me about, enumerate) **and** a grounded signal exists: (a) a prompt token (singularised) matching an actual entity tag ("list all the characters" → tag character) — surfacing up to 8 tagged entities' payloads; and/or (b) a detected relation — surfacing up to 15 trust-gated edges as explicit `[Fact: a --rel--> b]` lines. Both channels honor project scope (§4.8) and the A3 trust floor. No LLM call, no separate module — score 1.0, no reinforcement (enumeration edges are not query anchors).

### **4.6 Micro-NER model**

The micro-NER model (classifier/ner\_model.py) is a BIO tagger — a 3-layer MLP 384 → 128 → 64 → 3 with two ReLU + Dropout(0.2) blocks — that classifies per-token embeddings (not token IDs) from the same shared embedder — since C17 the slice384 MRL prefix of its native output (bit-identical to the old truncate\_dim input) — into B-ENT, I-ENT, O. Each token string is embedded **in isolation** (context-free): the model learns which tokens tend to be entities, but cannot use surrounding words to disambiguate — a known limitation (good recall, noisy boundaries; roadmap A9 tracks a context-aware rework). Operating on embeddings lets it generalise across the embedding space (e.g. recognise a misspelling close to a known entity) at the cost of requiring the embedder at inference. A shared singleton in retrieval/ner\_utils.py::extract\_entities consolidates both the orchestrator's live-prompt NER and the clustering worker's full-chapter NER onto one loaded model; if the .pt file is missing it falls back to a regex \\b\[A-Z\]\[a-zA-Z\]\{2,\}\\b minus a stoplist (The, This, User, Assistant, Chapter, …). **Output post-processing (roadmap A2 NER cleanup):** decoded entity spans are snapped to whole-word boundaries (`_snap_to_words`, fixes subword truncation like `Pyd`→`Pydantic`), trimmed of leading/trailing function words (`_clean_entity` / `_EDGE_TRIM`, e.g. `on Pydantic`→`Pydantic`), filtered of pronoun/boilerplate junk (`_NER_STOP`), and deduplicated. Empirically ~95% entity recall on a mixed prose/technical probe set; residual verb-led bleed (`uses PostgreSQL`) and descriptor false-positives (`fire mage`) are absorbed by A2's token-subset grounding and await the context-aware rework. The training pipeline (scripts/ner/\*) is offline-only: extract\_turns combines simulation + labelled + synthetic prompts; label\_entities calls mattbucci/gemma-4-12B-AWQ at SGLang port 8003 to extract verbatim entity strings; generate\_bio aligns entities to token offsets (longest-first, no-overwrite) producing BIO labels; train\_ner trains with Adam(lr=1e-3), BATCH\_SIZE=16, EPOCHS=10, CrossEntropyLoss with inverse-frequency class weights clamped at 100.0, early stopping PATIENCE=3.

### **4.7 Vector fuzzy matching for entity resolution**

The Codex retrieval leg resolves prompt-extracted entities to stored CodexEntity rows in three stages (A4): (1) \_match\_entities\_by\_similarity(threshold=0.85) — embeds the prompt entity strings, linear-scans every entity with a non-null embedding, computes the dot product, and takes the best-scoring entity above 0.85; each prompt entity matches at most one stored entity and vice versa (seen\_ids); with an inline exact canonical\_name / aliases.any(norm) fallback per string. (2) The base-class \_match\_entities\_exact (exact/alias only, no vectors) — used when use\_fuzzy\_match=False (the ablation fuzzy\_match flag). (3) **Payload descriptor fallback** \_match\_entities\_by\_payload — when name-based matching finds nothing, content words (≥4 chars, minus a small stoplist) from the NER strings are ILIKE-searched inside context\_payload, ranked by hit count, top 2 accepted: "main fortress" resolves to The Obsidian Citadel because its payload mentions "fortress". This closes part of the semantic-vs-lexical gap without a schema change (payload *embeddings* remain future work).

### **4.8 Retrieval and event-sourced compaction**

The Codex retrieval leg (\_codex\_graph): (i) extracts prompt entities via the shared NER; (ii) resolves them by vector fuzzy matching; (iii) optionally restricts to a conversation scope (entities whose codex\_events.batch\_source appears in the conversation's batch set); (iv) **trust-gated BFS traversal (A3)** to max\_depth = 3 over *time-valid* edges — `_edge_valid_filters()` (T3/D4): valid\_until IS NULL under current mode, the bi-temporal `valid_at(T)` predicate (valid\_from ≤ T AND (valid\_until IS NULL OR valid\_until \> T), T = window end, legacy NULL valid\_from passes) under as\_of/range; evolution mode navigates the current graph (D5) — gated on **effective trust = strength × extraction\_confidence** (`_edge_trust`, whose A11 recency term re-anchors to |T − valid\_from| under a window): a matched entity's *direct* edge expands (and is collected as a query *anchor*) only when trust ≥ CODEX\_DIRECT\_TRUST\_FLOOR (0.5) — so grounding-rejected low-confidence edges sit in the graph but never reach context or gain strength until corroborated — and deeper hops require trust ≥ CODEX\_DEEP\_STRENGTH\_FLOOR (1.0), so weak/decayed/low-confidence edges no longer pull the whole 3-hop neighbourhood into context; appends "\[Entity: \{canonical\_name\}\]\\n\{context\_payload\}" per visited entity (payloads themselves list an entity's edges strongest-first); (v) returns a single ContextFragment of type codex whose score is **graded 1.0–1.5** from the mean effective trust of the matched entities' edges (replacing the old binary 1.5× active-edge boost). **Retrieval-reinforcement (A3):** `_reinforce_codex_edges` bumps the anchor edges' strength by CODEX\_REINFORCE\_INCREMENT (0.15, capped at 10.0) each time they're surfaced — the Codex analog of episodic access/decay strengthening (§6.5), balanced by codex\_decay (which decays **all** live edges, closing the loop) — and promotes a pending edge to active once strength ≥ CODEX\_PROMOTE\_STRENGTH (2.0) **and** extraction\_confidence ≥ CODEX\_PROMOTE\_MIN\_CONFIDENCE (0.5), so an edge can earn activation through repeated retrieval usefulness but a low-trust extraction cannot promote on popularity alone. **Multi-fragment representation (A10):** the leg now emits **one ContextFragment per anchor entity** (its payload + trust-gated neighborhood + its own relation facts), each scored by *that* anchor's direct-edge trust (1.0 + graded trust + relation-overlap boost), rather than concatenating the whole traversal into a single blob. Combined with the round-robin budget (§6.5), this fixes the structural under-representation that made codex only 3.3% of fragments — it could previously occupy at most one budget slot no matter how many relevant entities it found. Traversal shares one `visited` set across anchors, so an entity reachable from two anchors is rendered once. Enumeration (§4.5) likewise emits per-entity fragments plus one facts fragment. **Relation-aware fact surfacing (A4):** when \_detect\_relations (§4.5) finds relations relevant to the prompt, edges where a *matched entity* participates in a *detected relation* (either direction, trust-gated, strongest-first, up to 10) are appended as explicit `[Fact: a --rel--> b (since YYYY-MM)]` lines (T1-dated from valid\_from; negated edges render `NOT rel`), join the reinforcement anchors, and add a RELATION\_OVERLAP\_BOOST (+0.25) to the fragment score — the entity∩relation joint hit is the precision anchor, lifting the score ceiling to 1.75. **Project scope (A5):** under a project-scoped conversation the leg resolves scope once (\_codex\_scope\_sets → the conversation's batch set + the entity set touched by those batches) and honors it *throughout*: anchor entities are filtered (as before), traversal now also (a) drops edges whose source\_batch is outside the conversation, (b) never expands into entities outside the conversation's entity set, and (c) renders each visited entity's payload **on the fly from the conversation's own edges** instead of the stored global context\_payload — so a shared entity ("ice") no longer leaks facts from other conversations. Unscoped (auto) conversations keep global traversal and stored payloads; none-scoped conversations now pass the A5 `isolated` scope (empty set — matches nothing) per G16 incognito (§6.10). **Grounded query expansion (A4, the accepted replacement for HyDE):** the codex leg runs first in retrieve(); the canonical names + aliases of whatever entities it matched (≤8 terms) are appended to the BM25 search prompt, so lexical search finds turns that use the full or alternate name — nothing is generated, so nothing can be hallucinated; the vector leg keeps the original embedding. If the NER extracts no entities, the enumeration path (§4.5) answers category queries directly from the graph. *Note: pending edges are traversed here (subject to the trust gates) — retrieval never filtered on confidence='active' (only the old score boost did).* Full per-edge (rather than per-blob) scoring awaits the multi-fragment representation (roadmap A10).

The codex\_events append log is compacted by workers/compaction.py::compact\_entities (EVENT\_THRESHOLD = 100 uncompacted events per entity): for each entity crossing the threshold, the worker replays its uncompacted events in timestamp order, maintaining a set of "rel:target\_id" signatures (edge\_added adds, edge\_expired discards), writes a CodexSnapshot(full\_state = \{active\_edges, context\_payload, properties, aliases\}) with the last\_event\_id, and marks the consumed events compacted = True. This is textbook event sourcing: the live edge table is the current state, the event log is the audit trail, and the snapshot table is the compaction output that bounds replay cost. Since C7 the compaction worker is **runtime-scheduled every 24 h** (G10 settled) and remains lossless per the Track-T constraint — events are marked compacted, never deleted (the journal is user-facing history for T4 timelines).


## **5. Procedural Memory**

Procedural memory captures recurring behavioural patterns ("the user always X after Y") that, once crystallised, can be surfaced to gate or enrich future generation. Its lifecycle is detection → reinforcement → promotion → decay.

### **5.1 Pattern extraction**

The Procedural Extractor (workers/procedural\_extractor.py::extract\_procedural, plain callable since C7 — a direct call inside the post-flight job, **unconditionally** on every turn) calls the background model with a one-sentence pattern-detection prompt (temperature=0.0, max\_tokens=80, timeout=bg\_timeout(80)). If the model returns NONE, the callable exits. Otherwise it embeds the proposed pattern, queries procedural\_memory by cosine similarity LIMIT 1, and branches:

- **Match (\`sim \> 0.85\`).** Reinforce the existing pattern: reinforcement\_count += 1, last\_observed = now. If reinforcement\_count ≥ 3 AND confidence\_score \< 0.8, promote to confidence\_score = 0.8, is\_active = True.

- **No match.** Insert a new pattern with confidence\_score = 0.3, is\_active = False, reinforcement\_count = 1, source\_batch\_ids = \[batch\_id\].

The Reflection worker's \_crystallize\_patterns step runs the same workflow at session granularity, feeding on cross-turn patterns observed over the last 200 turns of each conversation.

### **5.2 Trigger-condition gating for retrieval**

Even active patterns are filtered at retrieval time by \_procedural\_trigger\_match: if the pattern's trigger\_conditions JSONB is non-empty, its topic\_tags and intent\_tags must each intersect the current classification's tags; an empty trigger\_conditions always passes. In the current implementation extractors always write trigger\_conditions = \{\}, so this gate is forward-looking infrastructure — since C9 (2026-07-19) patterns are gated by the confidence floor (§5.4) and the conversation-scope gate, plus this trigger gate when conditions exist; the old 3-intent whitelist is gone.

### **5.3 Decay and confidence promotion**

workers/procedural\_decay.py::decay\_procedural\_patterns (runtime-scheduled every 1.5 h; takes the uniform `cycles` param as a no-op — staleness is an absolute-age cutoff) runs a single boolean deactivation: SET is\_active = FALSE WHERE is\_active = TRUE AND last\_observed \< now() - 180 days AND reinforcement\_count \< 3. The two conditions are conjunctive — a pattern that has been promoted (≥3 reinforcements) is permanently immune to time-based decay. Confidence promotion is \*not\* in the decay worker; it happens in the extractors at the moment reinforcement\_count crosses 3. Patterns are never hard-deleted, only deactivated.

### **5.4 Retrieval (widened by C9, 2026-07-19)**

The procedural retrieval leg (\_procedural\_lookup) **always runs** — C9 deleted the old hard intent gate ({Strategic\_Planning, Generation, Open\_Exploration}), which is why nobody ever *felt* procedural memory. Precision now comes from three surviving signals: the vector cosine top-5 over procedural\_memory WHERE embedding IS NOT NULL AND is\_active = true **AND confidence\_score ≥ settings.procedural\_min\_conf** (0.3 — the floor that replaced the whitelist; Z1 re-tunes it), the conversation-scope gate (patterns whose source\_batch\_ids intersect the conversation's batch set; project-scoped patterns are invisible outside their project and batch-exempt inside it — E1), and the trigger-condition gate (§5.2). Returns up to five ContextFragments of type procedural, scored by raw cosine similarity. **Root-cause note (found while widening):** the leg had been *fully dead*, not just gated — its embedding bind lacked the PgVector type, so every execution raised `vector <=> double precision[]`, rolled back, and returned `[]`; the whitelist hid the crash. Fixed with the same `bindparams(type_=PgVector)` idiom the episodic legs use (the identical latent bug in \_rag\_lookup was fixed in passing). Exp2's "+0.02 contribution" number therefore measured a leg that never returned anything; FINAL re-measures the working leg.


## **6. Hybrid Retrieval Orchestrator**

The HybridRetrievalOrchestrator (retrieval/orchestrator.py) is the core of the pre-flight phase. It runs its retrieval legs (six always-on, the time-scoped cold leg, and the history-gated timeline leg, §6.11), fuses their outputs with weighted Reciprocal Rank Fusion, applies a battery of post-fusion transforms, and returns a token-budgeted list of ContextFragments to the Prompt Assembler. Since Track T (2026-07) every leg is additionally **TimeScope-aware**: a detected temporal query (§6.11) parameterizes the same leg SQL with a time window, archived-row visibility, and a movable recency origin — one query text per leg, never a timescoped twin.

The orchestrator's `retrieve()` is now only *called* when the single memory-retrieval decision (§2.4, B2) says so — main.py sets context\_reliance = Long\_Term\_Memory in that case. The old belt-and-suspenders overrides inside `retrieve()` (Zero\_Shot+conversation\_id ⇒ LTM, Creative ⇒ LTM), which silently re-forced retrieval regardless of that decision, have been removed; the Zero\_Shot / Real\_Time\_Search early-return guards remain purely defensive. So retrieval no longer fires for almost every turn — it fires when the classifier-trusting posterior crosses threshold, which for a genuinely self-contained prompt in a short conversation can be *not at all*.

### **6.1 The retrieval legs**

10. **BM25 (full-text)** — \_bm25\_episodic. Postgres ts\_rank over to\_tsvector('english', coalesce(raw\_text,'')||' '||coalesce(summary\_text,'')) against an OR-joined to\_tsquery of the top-30 stop-word-filtered prompt tokens. Hard filters decay\_score \> 0.2 AND is\_archived = false *under current mode* (under a TimeScope window a timestamp predicate applies and both filters relax — §6.11), optional conversation\_id and cluster\_ids scope. LIMIT 100, ordered by ts\_rank DESC.

11. **Vector** — \_vector\_episodic. pgvector cosine distance **with decay weighting**: (1 - (embedding \<=\> :prompt\_embedding)) \* COALESCE(decay\_score, 1.0), times the C8 in-score recency factor (origin-parameterized since T3 — §6.11). Same visibility invariant (current-mode filters relax under a window, like BM25), LIMIT 100. The decay multiplier is what distinguishes this leg from a pure semantic search: a high-similarity but decayed turn is down-ranked.

12. **Codex graph traversal** — \_codex\_graph. NER → three-stage entity resolution (vector fuzzy 0.85 / exact / payload-descriptor fallback) → trust-gated depth-3 BFS (deep hops require trust ≥ 1.0, direct ≥ 0.5) with A5 project-scope isolation, relation-aware fact surfacing with the entity∩relation overlap boost, retrieval-reinforcement of anchor edges, and grounded query expansion feeding the BM25 leg (§4.5, §4.8). Entity-less category queries answered by the enumeration path (re-homed MERA). Score graded 1.0–1.75.

13. **Procedural** — \_procedural\_lookup (§5.4). Always runs since C9: vector top-5 with the confidence floor, conversation-scope and trigger-condition gates (the intent whitelist is gone).

14. **RAG** — \_rag\_lookup. Triple-gated: requires context\_reliance == "Long\_Term\_Memory" AND intent\_tags ∩ \{Factual\_Retrieval, Analysis\_&\_Summarization\} ≠ ∅ AND the prompt contains one of \["document", "pdf", "reference", "manual", "guide"\]. On pass, SELECT chunk\_text, 1 - (embedding \<=\> :prompt\_embedding) AS score FROM rag\_chunks ORDER BY score DESC LIMIT 5. RAG is intentionally **global**, not conversation-scoped.

15. **Batch summaries** — \_batch\_summary\_lookup, two halves since C4 (2026-07-19). Half 1 (as built): conversation-scoped top-3 from batch\_summaries by cosine similarity, each prefixed \[summary, YYYY-MM-DD\] from created\_at (T1). Half 2: cross-conversation top-2 over **conversation\_summaries** embeddings (§8.13), prefixed \[conversation summary, YYYY-MM-DD\] — the active conversation's own row is excluded (the assembler injects it, §7.2 — never double-inject) and private conversations are excluded via the conversations join; the whole half is skipped under incognito (`include_cross=False` — a user-global read, G16). Both halves share source\_type batch\_summary (leg weights unchanged) and return \[\] under any non-current TimeScope mode (D14 — a summary is created long after the turns it compresses, so its content period is underivable; serving it under a window would mislead).

16. **Cold storage (T3 — time-scoped queries only)** — \_cold\_lookup. Fires only when the TimeScope carries a window (as\_of / range, or evolution *with* a window); never under current mode and never from the wide net. Windowed timestamp scan of cold\_storage with up to six ILIKE patterns built from prompt keywords + grounded expansion terms (a keyword-less "what was I thinking about in march" browse is allowed — the window itself bounds it), conversation/privacy scoping through the **shared `_conv_scope_filter`** like the live legs (G29, 2026-07-28 — it hand-rolled `if conv_id` until then, so a *project*-scoped query, whose ids live in `scope["conversation_ids"]` with conv\_id None, searched the entire global cold store), LIMIT timescope\_cold\_limit (5). Fragments are episodic-typed (they share the episodic budget lane), date-stamped, base score 0.6. A cold hit that survives the token budget is **resurrected on probation** (§6.11).

17. **Timeline (T4 — history-gated, any mode)** — emitted from inside \_codex\_graph, one per anchor entity that carries *real supersession history* (`history_exists`, §6.11), rendered by `retrieval/evolution.py::build_entity_timeline`. source\_type="timeline", scored 0.9× the anchor's own codex fragment, capped at timeline\_max\_fragments (2; 4 under evolution mode). retrieve() splits them out of the codex return into their own legs-dict entry, so RRF weights them independently (0.6) and the budget's round-robin gives them their own lane.

### **6.2 Query rewriting: grounded expansion in production, HyDE rejected**

A \_hyde\_rewrite method exists in the orchestrator but is **commented out** in the production retrieve() path (and note it is query *reformulation*, not actual HyDE — it never fabricates a hypothetical answer document). It is only reachable through the ConfigurableOrchestrator with the hyde=True ablation flag (default False). The post-paper review (roadmap P0.1) rejected shipping real HyDE: it would fabricate answers with the small background model over the user's *private* history, and hallucinated specifics corrupt the noise-sensitive BM25 leg. Production instead uses **grounded query expansion (A4)**: the BM25 search prompt is expanded with the canonical names + aliases of the entities the codex leg actually matched (§4.8) — expansion terms come from the graph, not a generator.

### **6.3 Dynamic leg weighting**

After the legs run, retrieve() blends a per-leg alpha map. Base weights are \{"bm25": 0.8, "vector": 1.0, "codex": 0.5, "procedural": 0.2, "rag": 1.0, "timeline": 0.6\}. Five **intent profiles** override the four non-RAG legs:

| **Profile (active intents)** | **vector** | **bm25** | **codex** | **procedural** |
| - | - | - | - | - |
| **Factual\_Retrieval, Utility\_Formatting** | 1.2 | 0.8 | 0.1 | 0.1 |
| **Troubleshooting, Strategic\_Planning** | 1.0 | 0.8 | 0.3 | 1.2 |
| **Generation, Ideation, Open\_Exploration** | 0.6 | 0.6 | 1.2 | 0.1 |
| **Emotional\_Processing, Analysis\_&\_Summarization, Decision\_Making** | 1.1 | 0.6 | 0.9 | 0.0 |
| **Casual\_Banter, Null\_Noise** | 0.5 | 0.2 | 0.0 | 0.0 |


For each active intent, profile\_weights\[leg\] / num\_active is added to blend\_weights\[leg\]; unknown intents fall back to base\_weights / num\_active. Two **cumulative topic overrides** then apply: Creative\_&\_Media ⇒ codex += 0.3; Software\_&\_Tech ⇒ procedural += 0.4. The **timeline weight is then pinned to its base 0.6 regardless of profile** (T4 — its firing condition, history\_exists on a matched anchor, is the gate; intent is not). Every weight is floored at 0.0. The ConfigurableOrchestrator does **not** redefine these tables — it only toggles legs on/off; the blend still runs unmodified, so an empty leg simply contributes nothing to fusion.

### **6.4 Reciprocal Rank Fusion (RRF)**

\_apply\_rrf(legs, alpha\_map, k=60) sorts each leg's fragments by native score, then accumulates per-fragment RRF scores keyed on the SHA-256 of frag.text:

*score\_RRF(f) = Σ\_ℓ ∈ legs α\_ℓ / (k + rank\_ℓ(f)),  k = 60*

where α\_ℓ is the blended weight (defaulting to 1.0 for unknown legs) and rank\_ℓ(f) starts at 1. Fragments are deduplicated \*during\* fusion — the first occurrence is registered and subsequent occurrences add to its RRF score. Output is sorted by RRF score descending.

### **6.5 Post-fusion processing**

After RRF, the pipeline runs five sequential transforms:

16. **\`\_apply\_bonuses\`** (additive, applied as a multiplier score \* (1 + bonus)):

- **Keyword boost** +1.0 if any prompt keyword (or its singular form via kw.rstrip('s')) appears in text.lower().

- **Length bonuses** (mutually exclusive): +1.5 if word\_count \> 800, +0.5 if \> 400, -0.7 if \< 80.

- **Recency boost** (episodic only, **skipped** for Creative\_&\_Media turns — recent meta turns are noise for creative work — and skipped under any non-current TimeScope mode, since it points at *now*): +1.0 if the fragment's source turn is in the top-10 % most-recent of its conversation (recency\_pct \< 0.10), +0.5 if top-30 %; requires total \> 20 turns.

- **Soft meta-discussion downweight** -0.45 (classifier-driven, not string matching): fires only when wants\_narrative\_fact (the intent set intersects NARRATIVE\_FACT\_INTENTS = \{Factual\_Retrieval, Decision\_Making\}) and the source turn's stored intent\_tags intersect META\_LEANING\_INTENTS = \{Analysis\_&\_Summarization\}.

- **Clamp** bonus ∈ \[-0.9, MAX\_TOTAL\_BONUS\_MULTIPLIER\] (effective multiplier range \[0.1×, 5.0×\]).

17. **Pre-RRF adjustments** in \_rows\_to\_fragments (episodic rows only, applied before RRF so they influence per-leg ranking): bookmark ×1.5, a dynamic word cap (default 500 words; 1500 if a prompt keyword matches; uncapped for is\_document), and the **T1 date stamp** — every fragment (and its degrade/abstract forms) is prefixed \[YYYY-MM-DD\] from the turn's timestamp. *(The time/turn-recency additive tiebreakers previously described here were dormant dead code — no leg SELECT carried `timestamp` since pre-experiment commit 616d770 — and were removed by T1 rather than silently awakened when the column returned for date-stamping.)*

18. **Sort** by score descending.

19. **\`\_session\_diversify\`** (max\_per\_conversation = 3): iterates sorted fragments; the active conversation is uncapped; foreign conversations are capped at 3 each; fragments with no conversation\_id (RAG, Codex, procedural) are always kept.

20. **\`\_deduplicate\`**: SHA-256 of text, first occurrence wins (defensive second pass after RRF).

21. **\`\_enforce\_token\_budget\`** (A10): two-phase packing against self.max\_retrieval\_tokens. Phase 1 — **leg-diversity guarantee**: the highest-scoring fragment of each leg (best\_per\_leg\[source\_type\]) is added first, sorted by score, while it fits. Phase 2 — **round-robin-with-slack across legs** (replaces the old flat greedy fill): each round every leg contributes its next-best fragment (highest-scoring leg first), so a fragment-rich leg (episodic emits dozens) no longer soaks the entire remainder while codex/procedural emit few — yet when other legs are sparse, exhausted legs drop out and their share flows to the rest, so the budget still fills fully. This works with the A10 codex change below (the codex leg now emits *multiple* fragments — one per anchor entity — so it has more than one fragment to contribute per round, fixing its structural under-representation).

22. **\`\_strengthen\_retrieved\`**: for each episodic fragment, access\_count += 1 and decay\_score = min(1.0, decay\_score + 0.15) — retrieval acts as a partial reversal of decay.

### **6.6 Cluster-scoped retrieval**

Before the legs run, \_relevant\_cluster\_ids(prompt\_embedding, classification, conversation\_id, top\_k=10, scope) identifies the relevant clusters: it pulls the top-30 clusters by centroid cosine similarity and re-scores each as **combined = sim + 0.3 × tag\_overlap** (the count of overlapping topic tags). If the best combined score is below 0.50 it returns \[\] and the orchestrator falls back to global search. Otherwise it returns an **adaptive band** — every cluster scoring within 80 % of the best, capped at top\_k — rather than a flat top-10: a single dominant cluster scopes tightly, several comparably-relevant clusters all stay in, and weak tails drop out instead of padding the scope to a fixed count. **C5 deleted the third term:** scoring used to add `0.15 × name_sim` against a *freshly embedded* name + description per row, costing up to 30 model forward passes on the synchronous hot path for a signal the centroid already encodes. **Scope (G29, 2026-07-28):** the cluster query resolves its conversation filter through the shared `_conv_scope_filter`, so a project-scoped request ranks only that project's clusters — it previously hand-rolled `if conversation_id`, which under project scope (conv\_id None, ids in `scope["conversation_ids"]`) emitted no filter and ranked every cluster in the store. The scoped cluster IDs are injected into scope\["cluster\_ids"\], and both episodic legs apply a filter that admits any turn linked to one of the scoped clusters **or** any turn with no cluster links yet (so unassigned turns stay retrievable). The ConfigurableOrchestrator can disable this with cluster\_restrict=False.

### **6.7 Dynamic token budget**

set\_budget\_from\_turn\_count(turn\_count, total\_tokens, classification, total\_budget) computes the split between the recent-turns window and the retrieval budget:

- **Model-aware total (C16, 2026-07):** the total is no longer a hardcoded 23k. main.py resolves the model **before** budgeting (§9.3) and derives `total_budget = clamp(context_window × context_input_fraction, [context_budget_min, context_budget_max])` via `derive_total_budget` (api/memory_decision.py), reading the window from the registry (`get_model_context_window`); an unknown model falls back to `context_budget_fallback` (23\_000). All four knobs are settings. So an 8k model assembles ~6k of context, a 32k model ~24k, and a 128k model is held at the max guardrail until C16's need-based filling makes larger budgets safe. `TOTAL_CONTEXT_BUDGET = 23_000` remains only as the class-level fallback for legacy/direct callers; OVERHEAD\_RESERVE = 1\_800 as before.

- The **recent-window fraction** starts from a length-based base (0.3 \<10 turns, 0.2 \<50/\<200, 0.15 \<500/else), is reduced by token-density (-0.15 if avg\_tokens\_per\_turn \> 3000, -0.10 if \> 1500, -0.05 if \> 800), shifted by intent (-0.10 for Factual\_Retrieval/Troubleshooting/Analysis\_&\_Summarization; +0.10 for Emotional\_Processing/Casual\_Banter), shifted by topic (+0.05 Creative\_&\_Media; -0.05 Software\_&\_Tech; +0.05 Social\_&\_Relationships/Lifestyle\_&\_Health), and clamped to \[0.05, 0.85\].

- recent\_budget = int(available \* fraction), raw\_retrieval = available - recent\_budget.

- A **growth cap** prevents long-tail conversations from over-allocating to retrieval: \<30 turns → 2\_000 + 150\*n; \<100 → 5\_000 + 100\*(n-30); \<500 → 10\_000 + 30\*(n-100); else raw\_retrieval. retrieval\_budget = min(raw\_retrieval, growth\_cap).

- Leftover budget (the gap between raw\_retrieval and growth\_cap) is **intentionally unused** — this is the lever that keeps ICE token-efficient relative to a vector-only baseline. The ConfigurableOrchestrator with dynamic\_budget=False reverts to fixed max\_retrieval\_tokens = 8000, recent\_token\_budget = 4000.

### **6.8 Wide-net fallback**

**C15 (2026-07):** retrieve() short-circuits to \_wide\_net\_fallback when **both per-head confidences are weak** — \_head\_confidences splits the topic and intent maxima out of raw\_probs (the old trigger used max over all 25 probs, which whichever peaked head dominated; DI3 fast-path results fall back to their explicitly-set max\_confidence) — a full vector scan fused with single-leg RRF (α = 1.0), post-processed identically, and truncated to a **dynamic ceiling of max(1 500, 0.3 × max\_retrieval\_tokens)** (was hardcoded 2 000) — still much tighter than the normal budget, to keep the response focused when the system is genuinely unsure what the user wants. **Scope honored (C6/G16, 2026-07):** the wide net widens *ranking*, not *visibility* — previously it ignored scope entirely (searched every conversation, called \_codex\_graph unscoped, always included RAG), which leaked project- and incognito-scoped memory exactly when the classifier was least sure. Now it applies the same rules as the normal legs: a scope conversation\_id restricts the scan, global scans exclude is\_private rows, \_codex\_graph receives the scope, and \_rag\_lookup is skipped under incognito. **TimeScope honored too (T3):** the wide net applies the same window / archived-visibility / decay-floor rules and re-anchored recency as the normal legs; it does *not* call the cold leg (widening ranking is its job, not reviving frozen memories).

### **6.10 Privacy & scope semantics (G16 incognito)**

`memory_scope_type` on the conversation now has three real behaviors (main.py scope resolution): **project** — episodic legs filtered to the conversation (+ optional cluster scope), codex per-conversation via A5; **auto** — global retrieval, but the episodic legs carry a `is_private = FALSE` visibility invariant so incognito turns are never surfaced; **none (incognito, "store private + read nothing")** — the turn is stored with `is_private = TRUE` and the scope carries `{conversation_id, isolated, incognito}`: episodic legs search only the conversation itself (an explicitly conv-scoped query is the *only* door to private rows), codex resolves the A5 `isolated` empty set, and the user-global legs (procedural, RAG) are skipped. On the write side, private turns keep their per-turn post-flight evaluation but **never feed the shared stores**: post\_flight skips codex/procedural extraction, and clustering, batch summarisation, and reflection all exclude `is_private` rows. Changing a conversation's scope through PUT /user-control/conversations/{id}/scope propagates the flag to its episodic rows in both directions (none→other clears it, other→none sets it).

### **6.9 Feature Toggling for Ablation Studies**

Every retrieval leg and post‑processing step can be independently enabled or disabled through a ConfigurableOrchestrator (a subclass of HybridRetrievalOrchestrator). An overrides dictionary—keyed by leg name ("bm25", "vector", "codex", "procedural", "batch\_summary", "rag", etc.) and post‑processing step ("rrf", "hyde", "cluster\_restrict", "session\_diversify", "dynamic\_budget", "keyword\_boost", "recency\_boost", "timescope")—controls whether each component participates in retrieval. The `timescope` flag (T2/T3) sets `timescope_allowed = False`, forcing every request to CURRENT mode — all temporal branches collapse to pre-Track-T behavior (the FINAL experiments' temporal ablation seam). Setting a key to False causes the corresponding leg to return an empty list or the corresponding transform to be skipped. This mechanism is used by the ablation experiments reported in the evaluation section, where features are added cumulatively from a bare vector‑only baseline to the full ICE stack, measuring the incremental contribution of each architectural addition.


### **6.11 Time-scoped retrieval & idea evolution (Track T: T1–T4, 2026-07)**

Storage always recorded time on three layers (episodic `timestamp`, bi-temporal codex edges, the event journal), but retrieval read none of them — every query answered "now." Track T makes the stores answerable *at* a time (T1–T3) and serves how ideas *changed* over time (T4). Spec: docs/specs/T_temporal.md. The journal already gives git's semantics; Track T built only the porcelain — valid\_at(T) filtering is `checkout` (T3), the timeline builder is `log` and entity\_diff is `diff` (T4).

- **T1 — date-grounding.** The system prompt opens with `Today's date: YYYY-MM-DD` plus an instruction explaining the fragment date stamps; every episodic/chunk/cold fragment is prefixed `[YYYY-MM-DD]` (degraded forms keep the stamp), codex fact lines render `(since YYYY-MM)` from valid\_from (month precision — day precision would imply false exactness; legacy NULL renders undated), batch summaries carry `[summary, date]`. Recent sliding-window turns stay deliberately undated (they are "now" by construction) and entity notes/previews stay undated (regenerated text owned by the extractor).

- **T2 — TimeScope detection** (`retrieval/timescope.py`, pure, \<1 ms, LLM-free). `detect_timescope` scans a code-stripped prompt against an expression grammar (absolute dates/months/quarters/halves/seasons/years, relatives like "two years ago"/"last summer", open ranges since/before/between, number-words) plus evolution cues ("how did X evolve", "originally", "over time"). **Joint gate** (A4's lesson): a resolvable expression alone never flips the mode — it must co-occur with a recall-shaped prompt (question mark / interrogative opener / DI3 reference\_signal / p\_ltm ≥ 0.5). Four modes: `current` (default; behavior identical to pre-T), `as_of` (point-in-time window), `range` (flat window), `evolution` (history request, window optional). Guards: future windows → current; vague pasts ("a while back") are never resolved into invented windows; windows are padded per granularity (settings `timescope_pad_*`) and clamped to now. The result travels as `scope["timescope"]` (D1 — zero signature churn); main.py logs `timescope_detected`; a non-current mode adds `ltm_bump_timescope` (+3.0 log-odds) to the B2 memory decision — a decisive bump, never a hard override. Kill switch: `timescope_enabled` (byte-identical rollback).

- **T3 — time-scoped reads.** Under `as_of`/`range`, every episodic leg gains `timestamp >= t0 AND timestamp < t1`; the `is_archived = false` filter and the 0.2 decay floor drop under any non-current mode (D10: "archived" means not-relevant-to-*now* — exactly what a time query is not asking). **Mode-aware recency (D9):** the C8 in-score factor becomes `EXP(-ABS(timestamp - center)/τ)` with a movable origin — current: center=now (numerically identical to the old formula for past rows); as\_of: center=window midpoint, boost applies even to creative (target-proximity, not freshness), τ widens with the window; range/evolution: flat. A11's codex edge-recency re-anchors to |T − valid\_from| the same way. **Codex valid\_at(T) (D4):** the four `valid_until IS NULL` read sites became `_edge_valid_filters()` — under a window, `valid_from <= T AND (valid_until IS NULL OR valid_until > T)` with T = window end ("state as of then" = everything established *by* then); evolution mode deliberately navigates the *current* graph (D5 — dead-edge walking makes incoherent neighborhoods; history is T4's timeline builder). Procedural patterns filter on observation-span overlap; post-fusion now-bonuses are gated off. **Cold second chance (D-U1, user decision):** cold hits that survive the budget are resurrected into episodic\_memory at their ORIGINAL timestamp with decay = `timescope_probation_score` (0.12, just above the 0.1 archive line) — unengaged, decay re-archives them within days; engaged, write-on-read strengthening saves them; never restored at full strength. Legacy cold rows (NULL conversation\_id, pre-migration) are cite-only. **Honest emptiness:** a windowed query with no episodic matches appends a `[Memory note]` naming the window (and the nearest eras from an unwindowed probe) — never silently widens. The probe drops the **window**, never the **scope**: it runs through `_conv_scope_filter` like every other episodic read (G29, 2026-07-28 — it hand-rolled `if conv_id` until then, so a project-scoped question was told about other projects' eras). **Decay fixes (D11/D12):** the archived-freeze bug is fixed (archived rows kept `is_archived = FALSE` filters on the decay UPDATEs, freezing them at ~0.1 forever — cold storage was unreachable); archived rows now keep decaying, a symmetric un-archive clause restores strengthen-driven recoveries, and the cold move carries conversation\_id/is\_private/batch\_id (migration e5b8c2d4a917, which also added the timestamp and (source,valid) indexes).

- **T4 — evolution surfacing** (`retrieval/evolution.py` — DB-reading, no LLM, free of orchestrator state so E0 wraps the functions as services and D1's agent can drive them; no REST endpoints by design). **The D6 discriminator** lives in this module and nowhere else: an expired edge counts as *history* only when its expiry is **event-backed** (a matching `edge_expired` journal event — A6 reconciliation, A8 negation, property update) and the reason is not `source_deleted` (C10: deletion is not evolution). An eventless expiry is codex\_decay forgetting — a faded idea is not a revised idea, so it never enters a timeline. Four functions: `history_exists(db, entity_id)` — the cheap EXISTS gate; `build_entity_timeline(db, entity, allowed_batch_ids, t0, t1, max_transitions)` — groups an entity's edges by (relation, direction) into supersession chains and renders one dated line per member at month precision (`2025-11 – 2026-02: saga --planned_as--> multiverse destruction  (superseded: antonym_superseded)` / `2026-02 – now: …`; negated edges render `NOT <relation>`; reason falls back to "updated"), returns None when no event-backed expired member exists, keeps the most recent max\_transitions lines ("(earlier history omitted)" when truncated), word-trims to timeline\_max\_tokens by dropping oldest lines; `entity_diff(db, entity, t0, t1)` — \{added, expired, retracted\} dicts with raw datetimes (the E0 service candidate and F3's timeline-overlay backend; exercised by tests until then); `log_description_update(db, entity, old, new, source)` — the **D13 never-overwrite journal**: every `entity.description` write site emits a `description_updated` CodexEvent with old/new snippets (reflection enrichment `source="reflection_enrichment"` — replacing its old opaque `context_appended` emit — and the inject watcher's update path `source="codex_inject"`; a future manual entity editor must use `source="manual_edit"`). **Wiring:** \_codex\_graph attaches a timeline fragment per history-carrying anchor **in any mode including current** (D-U2, user decision: evolution info is *provided, never forced* — the saga case needs no temporal wording; the model decides how much to narrate), scored 0.9× the anchor, capped (2 normal / 4 evolution), windowed by the active TimeScope; timelines fuse as their own leg (weight 0.6) with their own budget lane. Evolution mode additionally **era-stratifies the vector leg** (\_stratify\_by\_era): the candidate pool widens to LIMIT 300 (no window, flat recency), is sorted by timestamp, cut into evolution\_era\_buckets (4) equal-count buckets, and each bucket keeps its top evolution\_per\_era (3) by score — so the idea's early life is represented instead of whichever era dominates similarity (row-shape-agnostic: C4's era digests can join the same candidate list). Evolution traversal stays over the *current* graph (D5) — dead edges are the timeline builder's input, never navigation's. This also completes T1's dated-facts story: expired facts in timelines render with their supersession month, closing the "…, superseded YYYY-MM" requirement.

## **7. Prompt Assembly**

The Prompt Assembler (api/prompt\_assembler.py) concatenates the retrieved fragments with persistent memory slots, recent turns, and the live user message into a list of chat-completion messages. Its ordering is deliberately **stable-prefix** to maximise KV-cache reuse across consecutive turns.

### **7.1 Stable-prefix ordering**

assemble\_prompt returns messages in this exact order:

23. **System message** — a fixed ~180-word instruction block opening with `Today's date: YYYY-MM-DD` (T1 — the anchor that makes fragment date stamps resolvable), then role of history vs the live question, step-by-step reasoning, fact-change tracking, the fragment-dating explanation (\[YYYY-MM-DD\] prefixes, `(since YYYY-MM)` facts, \[Timeline: …\] history blocks — live since T4), and specificity — with an inline === PERSISTENT CONTEXT === block rendering the active slots in tier order (C9 D6: global \[SLOT\_NAME\], then the attached project's \[PROJECT · NAME\], then this conversation's \[CONVERSATION · NAME\]), then === PROJECT SESSION START === when the turn opens a coding sitting (E4), then === CONVERSATION SUMMARY === when the conversation has outgrown the sliding window (C4 D3a — the evolving summary from §8.13, stamped "(as of N turns ago)" when the burst is behind). (The date prefix changes at most once a day, so the stable-prefix KV-cache property survives.)

24. **Recent turns** — alternating user/assistant pairs from get\_recent\_turns(db, conv\_id, max\_tokens=max\_recent\_tokens, max\_count=10).

25. **Retrieved-context block** — a single user message with header === RETRIEVED CONTEXT === (optionally (clusters: Cluster A, Cluster B, …) when scope\["cluster\_ids"\] is populated), then "\\n\\n".join(f.text for f in retrieved\_fragments).

26. **Acknowledgment** — a single assistant message: "Understood — I have the background context. What would you like to know?" — a deliberate boundary marker so the model treats the \*final\* user message as the live question rather than another history turn.

27. **Live user question** — the actual prompt to answer.

Because the system message, slots, and most of the recent-turns prefix change slowly (only the recent window slides), most of the prefix K/V tensors are reusable across consecutive requests.

### **7.2 Per-component rendering**

- **System message + slots** — only slots with is\_active AND content are rendered; tier filtering happens here (project slots only when scope\["project\_id"\] matches, conversation slots only for this conversation — a foreign project's slots never render); no per-slot token cap inside assemble\_prompt (G14's 300-token cap is enforced at write time in the slots service, §3.5). The C4 summary block is computed by `conversation_summary_block(db, conv_id, turn_count, total_tokens, recent_window_tokens)` — injection only when `total_tokens > recent_window_tokens` (B2's window estimate, C16-aware), so a conversation the window still covers never pays the ~200 tokens.

- **Recent turns** — per-turn word caps are dynamic: 80 words when max\_tokens ≤ 1000, 150 when ≤ 3000, else min(500, max(100, max\_tokens // max(1, len(turns)) // 2)). Each turn is split into user/assistant parts by parsing the literal "User: " / "\\n\\nAssistant: " markers in raw\_text; parts exceeding the cap are trimmed with \_trim\_words (appends …). Greedy fill until tokens\_used + next\_pair\_tokens \> max\_tokens.

- **Retrieved-context block** — fragments are passed in already budgeted by the orchestrator's \_enforce\_token\_budget; the assembler just joins them with \\n\\n.

### **7.3 Emotional / creative bypass**

There is no separate emotional/creative branch inside assemble\_prompt. The bypass is **upstream**, in the classification and retrieval layers: creative turns force Long\_Term\_Memory (§2.4), skip the recency boost (§6.5), and add +0.3 to the Codex leg weight (§6.3); emotional turns shift the token-budget fraction toward the recent window (§6.7). The assembler renders whatever the orchestrator decided.

### **7.4 Token budget enforcement during assembly**

Assembly is a two-layer budget process. The orchestrator's \_enforce\_token\_budget packs fragments to max\_retrieval\_tokens (§6.5). Then api/main.py recomputes total\_words against int(0.9 \* 4096 / 1.33) ≈ 2 770 words and, if over budget, iteratively pops **procedural** fragments first, then **episodic** fragments, reassembling after each pop. RAG and Codex fragments are preserved — they are the hardest to recompute and the most likely to carry the answer.


## **8. Background Maintenance Runtime**

**Reworked 2026-07-11 (roadmap C7): Celery + Redis are gone.** ICE's long-term-memory consolidation runs **in-process** inside the core app: `src/workers/runtime.py` defines a `MaintenanceRuntime` (one asyncio tick task, ~60 s + 0–15 s jitter) started by `src/api/core.py::create_core()` from the FastAPI lifespan — the same HTTP-free factory E7's headless `ice-mcp` boot calls (§11.3; `create_core` is lease-checked both directions since E7). Postgres is the only external service left. The workers themselves are now **plain callables** (the `@app.task` wrappers and per-task `self.retry` scaffolding were deleted; names and signatures kept — D1's maintenance agent and the FINAL harness are built against them), executed via `asyncio.to_thread` under two lane semaphores: **gpu(1)** for LLM-calling jobs (bg work serializes against itself — the shared-mode contention fix) and **cpu(2)** for DB-only jobs.

### **8.1 Runtime infrastructure: triggers, ledger, idle gating**

**Three trigger classes replace beat:** (a) **event** — `runtime.enqueue(job_name, **kwargs)`: turn stored → the post-flight chain; (b) **overdue** — each tick, if the app is idle, the runtime compares the `maintenance_ledger` table against `settings.maintenance_intervals` and runs anything overdue, longest-overdue first (this is what keeps an always-open app maintained *and* what catches up after downtime); (c) **work-unit** — `notify_work_unit(kind, **ctx)` with `"session_gap"` wired (fired from `store_turn_async` when `resolve_session_id` opens a new sitting: enqueues per-conversation cluster freshening and an immediate overdue pass) and — since the E-coding core (2026-07-18) — `"commit"`: `create_core()` registers `reconciler.make_commit_handler` via `register_work_unit_handler` (the C7-reserved seam, zero runtime changes), which enqueues the `project_reconcile` cpu-lane job (§12.3). `"task_done"` remains reserved.

**Schedule state is the `maintenance_ledger` table** (job\_name PK, last\_started, last\_finished, last\_status, last\_error, runs) — it survives restarts, feeds the overdue computation, doubles as an optimistic **claim lease** (an `UPDATE … WHERE last_started < now − 2×interval` lock, so an accidental second app instance can't double-run decay), and is the surface Track D's maintenance agent runs from (§8.9). A job that crashed mid-run (started without finished) becomes overdue again after 2× its interval. Failures retry with exponential backoff (30 s / 120 s / 480 s), then `last_status='error'`; job errors never kill the tick. Every run emits `maintenance_job_started/finished` structlog events (the F5 telemetry source).

**Runtime lease & standby (E7, 2026-07-16).** A `runtime_lease` pseudo-row in the same ledger (same pattern as the `session_end_burst` stamp) marks which *process* owns maintenance: the owning runtime stamps it on start and every tick, and releases it (`last_started = NULL`) on clean stop. `create_core(start_runtime: bool | None = None)` is lease-checked by default: a fresh foreign lease (< 180 s ≈ 3 ticks) means another process — the app or an `ice-mcp` session — owns maintenance, so this core's runtime starts in **standby**: its event jobs (this process's own post-flight chains, bookmark extractions) still run, but periodic/overdue dispatch, the session-end burst, and lease stamping stay with the owner; when the owner exits and its lease goes stale, the standby **promotes itself** on the next tick (`maintenance_runtime_promoted`). Cross-process event duplication is already safe (idempotency keys + the per-job claim lease above); what the lease serializes is the *dispatcher* — at most one process runs the periodic maintenance schedule. `start_runtime=True` force-owns (recovery), `False` starts no runtime at all (tests, pure-read tools).

Cadences (`settings.maintenance_intervals`, seconds — the old beat schedule carried over, G9-aligned): cluster\_assignment 1 800 · cluster\_merge 10 800 · chunk\_pending\_documents 7 200 · decay\_episodic/codex/procedural 5 400 · reflection 7 200 · batch\_summarize 7 200 · **conversation\_summary 7 200 (C4, 2026-07-19 — the evolving whole-conversation summaries, §8.13; also a session-end-burst member; idempotent per covers\_through so cadence passes on quiet conversations are no-ops)** · **compaction 86 400 (newly scheduled — G10 settled)** · **maintenance\_agent 43 200 (D1, 2026-07-17 — replaced sentinel\_monitor's 1 800 entry; gpu lane, also rides the session-end burst)** · **project\_poll 600 (E3, 2026-07-18 — the reconcile-on-commit fallback: checks registered projects for hook marker files + HEAD drift; the hook is the design, this is the ≤10-min lag path, §12.3)**. **Fine-tuning has no cadence** — it is consent-gated (D6/H5, §8.8): enough curated labels + a session end → runs only when `settings.auto_finetune` (default False), else one pending review-queue proposal.

**Decay catch-up is closed-form:** the decay callables take `cycles: int = 1`, and the runtime computes `cycles = clamp(floor((now − last_finished)/interval), 1, 96)` from the job's own ledger row at dispatch — one UPDATE applies `rate ** cycles`, so a week of downtime collapses into a single statement and cycles can never double-count runs the tick already performed.

**Idle gating is in-process truth (no hardware polling in shared mode):** the proxy signals `note_user_activity()` on request receipt and `generation_started()/generation_finished()` around the stream. `is_idle()` = no generation in flight AND quiet past `user_active_threshold_seconds` (90) — gates overdue dispatch; queued **gpu-lane event jobs** drain after `idle_burst_seconds` (120) of quiet (cpu-lane events dispatch on the next tick regardless). Both knobs are Z1-tunable. A **session-end burst** fires when a tick finds the sitting over (idle > `session_gap_minutes`) and not yet reconciled: the heavy jobs (reflection, batch\_summarizer, the maintenance agent since D1, and the conversation-summary job since C4 — the quartet) run if stale for the sitting, plus the fine-tune consent check. `nvidia-smi` polling survives **only for dedicated mode** (threshold 70, result cached 10 s — §10.5); shared mode never polls hardware.

**Import replay is an event-only gpu-lane job (F10/F14, 2026-07-20, §13).** `import_replay` has no cadence: `services/ingestion.py` enqueues it, and each dispatch processes conversations for a ~10-min slice budget, then re-enqueues the next slice (`seq+1`) — so an hours-long import never starves live chat's post-flight behind it. Every resumption goes through the same `_gpu_ready(for_event=True)` gate, and mid-slice the engine pauses between turns while `generation_in_flight`. Resume after a kill rides the `import_conversations` hash ledger + per-turn idempotency keys, not the runtime.

### **8.2 Post-Flight Evaluator**

post\_flight.evaluate\_turn(batch\_id, prompt, response, conversation\_id, model\_used) is event-driven, enqueued as the "post\_flight" runtime job from store\_turn\_async after each turn commit (gpu lane; runtime backoff retries; idempotency key sha256(batch\_id) guarding the density stage). **Reworked 2026-07 (roadmap C1).** The old is\_lossless heuristic (code fence / \>500 words / ≥3 capitalized words — which fired on nearly everything) and the backwards branch that summarised the *densest* long turns into 2–3 sentences are replaced by the density pipeline in workers/turn\_density.py:

- **Key-term extraction** (extract\_key\_terms) — the turn's MUST-PRESERVE vocabulary: named entities via the shared MicroNER (reusing the worker's already-loaded embedder — no extra model copy), figures (numbers/dates/units), identifiers (snake\_case, CamelCase, dotted paths, acronyms).

- **Density** (compute\_entropy) — facts-per-token ∈ \[0,1\] from entity density, figure/identifier density, code presence, and lexical diversity — finally written to **entropy\_score** (NULL since v2). lossless\_flag = code ∨ creative/emotional ∨ entropy ≥ 0.35 (generous — it gates Codex extraction, which was historically starved) and still exempts from batch summarisation.

- **STORE BOTH, CHOOSE AT READ TIME (user design decision).** Every non-document turn \> 350 words gets a **grounded summary** stored *alongside* raw — the storage layer never permanently forces one representation. The summary prompt receives the must-preserve terms (ground-then-generate, as Codex A2 / clustering v5), asks for 4–6 sentences plus trailing `Key terms:` and `Abstract:` lines (C3: the one-line abstract rides in the same call — no extra inference — parsed into abstract\_text), and the result is **measured**: summary\_coverage = fraction of must-terms retained; one retry names any dropped terms; the score is stored in **summary\_coverage**. inject\_raw is demoted to a *default hint*: raw for documents/code/creative (continuity) and for short turns; for long dense/diffuse turns the coverage gate sets it (summary-by-default only when it provably kept the key terms). The actual per-query choice happens at retrieval (§6.1a). Emits `summary_quality` + `representation_decided` log events (F5 candidates).

- **Document detection** — raw\_words \> 2000 AND assistant\_count \< 3 ⇒ is\_document = True with raw injection — folded into the decision matrix, which also fixed the old **clobber bug** (the document branch's inject\_raw=True was previously overwritten by the general assignment two lines later, so documents silently *lost* raw injection).

After the density stage (C7): it calls extract\_codex(...) directly (only if lossless), extract\_procedural(...) directly (always), and — for is\_document turns and all long turns (> ~600 words, C3) — run\_chunk\_turn(db, turn) directly (runs for private turns too, since chunk visibility is enforced through the parent join); all in the same runtime job, each stage self-idempotent so a retry completes whatever a partial failure skipped. The codex/procedural calls are skipped entirely for private turns (§6.10). The old broker-down JSONL buffer is gone (C7 D8 — see §10.3).

**§6.1a Read-time representation choice (C1).** \_rows\_to\_fragments no longer follows inject\_raw blindly; \_choose\_representation picks per query, in order of authority: (1) *trust* — no summary, or summary\_coverage below 0.7, ⇒ raw (a summary that dropped must-terms is never used; NULL coverage = legacy summary, status-quo trust); (2) *keyword protection* — if the matched prompt keyword lives in raw but not in the summary ⇒ raw, and not degradable (degrading would remove the very term that made the fragment relevant); (3) *intent preference* — Factual\_Retrieval/Troubleshooting prefer raw (degradable), Analysis/Strategic\_Planning/Ideation/Open\_Exploration prefer the trusted summary; (4) otherwise the storage-side hint. Fragments carry **degrade\_text** (the trusted summary when raw was chosen) and **abstract\_text** (C3: the one-line abstract, generated in the *same* LLM call as the summary and stored in episodic\_memory.abstract\_text; attached under the same trust and keyword-protection rules, never *preferred*), and \_enforce\_token\_budget performs **degrade-before-drop** in both phases through the full hierarchy — raw → trusted summary → abstract — taking the first level that fits the remaining budget. Word-cap truncation is sentence-aware (C3, \_truncate\_at\_sentence: the cut lands on the last sentence boundary inside the cap when one exists past 60% of it).

### **8.3 Codex Extractor**

When the task is dispatched from the bookmark endpoint, it receives priority=True, which causes it to skip the GPU‑utilisation gate (is\_gpu\_busy()) and the shared‑mode user‑activity gate. This is the only code path that overrides the yield‑to‑user constraint (INV‑5), ensuring that a user‑bookmarked turn is immediately processed into the knowledge graph.

codex\_extractor.extract\_codex(batch\_id, model\_used, priority) (§4.4) — plain callable since C7, idempotency key sha256("codex:" + batch\_id), called directly from evaluate\_turn only when lossless\_flag == True (gating/retries live in the runtime). Reads EpisodicMemory by batch\_id, calls extract\_triplets, validates/deduplicates, and calls handle\_triplet per triplet.

### **8.4 Procedural Extractor**

procedural\_extractor.extract\_procedural(batch\_id, model\_used) (§5.1) — plain callable since C7, idempotency key sha256("procedural:" + batch\_id), called directly from evaluate\_turn unconditionally. Pattern detection → similarity matching → reinforcement or insertion.

### **8.5 Decay Workers**

Three independent schedulers (§3.7), all every 5 400 s, max\_retries=2, default\_retry\_delay=60: decay.apply\_decay (episodic, access-weighted with creative floor, archive at 0.1, cold-storage move at 0.05), codex\_decay.decay\_codex\_edges (edge strength decay, demotion at 0.3 — **conversation edges only since the E-coding core: all three UPDATEs carry `source = 'conversation'`, so derived static-analysis/fact edges are decay-exempt** — the code graph is regenerated from source, not earned through use; §12.2), procedural\_decay.decay\_procedural\_patterns (boolean deactivation after 180 days if \< 3 reinforcements).

### **8.6 Reflection Worker**

reflection.run\_reflection (runtime-scheduled every 2 h + the session-end burst) runs a five-prompt cascade over the 200 most recently active conversations (each with ≥10 turns, last 200 turns, oldest-first):

28. **\`\_synthesize\_session\`** (SUMMARY\_PROMPT) — emits \{topics\_covered, decisions\_made, unresolved\_items, entities\_updated, patterns\_observed\}, inserts a SessionSummary row, and appends unresolved items to the pending\_items memory slot directly (updated\_by = "reflection\_worker").

29. **\`\_crystallize\_patterns\`** (CRYSTALLIZATION\_PROMPT) — embeds each detected pattern, matches against procedural\_memory (sim \> 0.85 ⇒ reinforce; else insert at confidence=0.3, is\_active=False), promoting at reinforcement\_count ≥ 3.

30. **\`\_evolve\_memory\_slots\`** (SLOT\_EVOLUTION\_PROMPT) — proposes updates for project\_context, user\_preferences, guidance; inserts review\_queue rows with item\_type='memory\_slot\_update' and `proposed_by: "reflection"` (C9 D7 — approval applies through the slots service and records the proposer as the slot's updated\_by). **Does not write slots directly** — human approval required.

31. **\`\_detect\_motifs\`** (MOTIF\_PROMPT) — inserts review\_queue rows with item\_type='new\_cluster\_proposal'. There is no numeric motif threshold; motif identification is entirely model-driven, and cluster creation requires human approval.

32. **\`\_enrich\_codex\_entities\`** (ENRICHMENT\_PROMPT, global pass; A7.3) — fills each entity's **description** (the rich "note body"). It selects up to 25/run — entities with an *empty* description ranked by mention count first, then stale well-mentioned notes (>14 days) for refresh — summarises the originating episodic passages with the background model into a concise domain-general note, writes it to **description** (NOT context\_payload, which is auto-assembled and would be overwritten), then calls \_regenerate\_context\_payload so the note becomes description + properties + links + backlinks, and journals the overwrite via `log_description_update(..., source="reflection_enrichment")` — a `description_updated` CodexEvent with old/new snippets (T4/D13 never-overwrite; replaces the old opaque `context_appended` emit, which remains in old journals as readable history). Reflection **does not add edges** — it only enriches entity descriptions. This is the engine that turns codex nodes from label strings into real Obsidian-style notes.

### **8.7 Clustering Worker**

**Reworked 2026-07 (roadmap C5, "v5") — the structural fixes for Exp2's "2–3 mega-clusters + dozens of singletons," which were creation/assignment-side, not threshold-side.** The plain callables (`run_cluster_assignment(db, conversation_ids=None)` / `run_cluster_merge(db, conversation_ids=None)`) are driven directly by the maintenance runtime — cadence plus C7's session-gap freshening, which passes the new sitting's conversation id — and later by Track D's agent (the C5-era Celery wrappers were deleted with C7).

**Assignment** (run\_cluster\_assignment, every 30 min + session-gap, MAX\_TURNS\_PER\_RUN = 25, oldest-first — the queue previously had LIMIT with no ORDER BY): score = centroid dot-product + tag overlap (capped 0.10 — within one conversation tags are near-uniform, so uncapped it was a constant offset) + entity overlap (shared MicroNER, capped 0.30) + **session affinity** (+0.10 when the turn's C6 `session_id` appears among a cluster's members — one grouped query per conversation per run; replaces v4's temporal-predecessor bonus that cost two queries per turn×cluster). Threshold 0.6. **Assignment is exclusive** — single best cluster only: v4 linked a turn to *every* cluster above threshold, and each then recomputed its mean centroid including that turn, so clusters sharing members were dragged together every run until merge glued them (the mega-cluster feedback loop). **Creation is wait-for-a-friend:** a turn matching no cluster *waits*; each run, waiting turns are pairwise-scored (embedding + entity + same-session) and grouped by union-find — a cluster is created only from ≥2 mutually-similar turns (born with two turns of evidence: better centroid, better name), while a lone turn only becomes a singleton after SINGLETON\_AGE\_HOURS = 24 (one-off topics are real). Waiting is retrieval-safe: the cluster-scope filter passes link-less turns. v4's name-overlap bonus is removed (whole-turn-text ∩ description with stopwords sat at its cap against every cluster — a constant, not a signal). **Naming is NER-grounded:** name/description prompts receive the entities recurring across ≥2 member turns (A2's ground-then-generate pattern), regenerated every NAME\_REGEN\_INTERVAL = 5 members. Private (G16) turns are excluded.

**Merge** (run\_cluster\_merge, **scheduled every 3 h** — it previously never ran, half of bug G10): first a **singleton re-absorption pass** (a 1-member cluster whose turn scores ≥ the assignment threshold against a sibling is folded in and deleted — the repair path for singletons already accumulated), then conservative pairwise merging (raw-centroid floor 0.82, adjusted ≥ 0.90, entity bonus capped 0.10; merging is permanent, so it stays stricter than assignment). Centroids are renormalized after every recompute (an average of unit vectors is not unit length; unnormalized centroids deflate scores as clusters grow).

**Retrieval side** (\_relevant\_cluster\_ids, §6.6): the per-row name/description `embedder.encode` is gone (up to 30 model forward passes in the synchronous hot path for a signal redundant with the centroid), and the flat top-10 is replaced by an **adaptive band** — clusters within 80% of the best score (capped at top\_k), so one dominant cluster scopes tightly instead of padding the scope to a fixed count.

### **8.8 Batch Summariser**

batch\_summarizer.batch\_summarize (§3.8, runtime-scheduled every 2 h + the session-end burst) groups decayed-but-not-archived turns into 50-turn batches and writes compressed batch\_summaries rows. Distinct from §8.13's conversation summaries: batch rows are frozen *ranges* of decayed turns; the conversation summary is one *evolving* whole-conversation object.

### **8.9 Memory Maintenance Agent (D1/D2, 2026-07-17 — replaced the Sentinel Monitor)**

`maintenance_agent.run_maintenance_agent(db, llm_decider=default)` (gpu lane, overdue 12 h + the session-end burst) is the graph's self-maintenance pass — **a deterministic worklist + bounded LLM decisions, deliberately NOT a free-roaming tool loop** (a small model gets reliability from constrained choices; enums + tiers + caps are the design, not a v1 compromise). Cheap SQL **detectors** (a pluggable registry, `DETECTORS` — E1's coding-side decisions table reuses the pattern) produce typed work items; the background model decides per item from a **fixed enum** (one JSON-mode completion, temperature 0; unparseable or out-of-enum output ⇒ `unsure` ⇒ **no write**); execution goes through named callables (A6's `reconcile_conflict`, D5's `merge_entities`).

**Five detectors** (spec D3): (1) pending `codex_reconciliation` review items — re-decided WITH the source turns of *both* edges (the in-line reconciler only saw one turn; enum `expire_old`/`keep_both`/`reject_new`/`unsure`; settled items get `status="resolved"` — not "approved", the user didn't decide; still-unsure items get `item_content.agent_attempts += 1`, and ≥2 attempts ⇒ never retried); (2) **pending-edge pileup** — the sentinel's one real query ported (JOIN/HAVING shape; thresholds from its seed rule: >3 pending overlapping >2 active) → live pending edges duplicating a live active edge are expired, the active keeps the max extraction\_confidence; (3) **duplicate-entity candidates** — casefold name/alias intersection ∪ embedding cosine ≥ 0.90 (pgvector, same entity\_type, merged husks excluded), top 10/run: normalization-equal pairs auto-merge (Tier 0), cosine-only pairs need the LLM's one-word verdict and only `same` proposes (Tier 2); pairs the user rejected — and pairs already proposed/decided — are skipped forever via a sorted `pair_key` stamped in `item_content`; (4) **contradiction backlog** — a live positive + live negated edge for one (source, relation, target) (A8 residue), or two live antonym edges for one pair (A6 cross-batch misses) → deterministic newer-supersedes through `reconcile_conflict`'s antonym arm; (5) **stale `pending_items` slot** (the sentinel's absence rule; >14 d) → an LLM content suggestion filed as a `memory_slot_update` proposal with `proposed_by: "agent"` (C9 D7: the approve arm applies it through the slots service — tier-validated, G14-capped — and the proposer lands as updated\_by), blocked while one is pending or a rejection postdates the slot's last change; (6) **stale project work** (E3, 2026-07-18 — the first coding-side detector to join the registry): `tasks` rows pending/active and untouched >14 d → a deterministic Tier-2 `stale_work` proposal (no LLM — the payload, incl. branch + goal for drift visibility, IS the proposal), blocked while one is pending or a rejection postdates the task's last move. Detector 3 additionally filters `source = 'conversation'` since the E-core — merging regenerable derived entities is wrong, and the next re-parse would undo it.

**Three risk tiers** govern write authority: Tier 0 auto-applies deterministic reversible-ish ops (normalization-equal merges); Tier 1 auto-applies journaled graph ops (reconciliation re-decides, pileup dedupe, contradiction resolution); Tier 2 only ever *proposes* into the review queue — the queue stays the safety net until F2 surfaces it. **Caps per run** (D8, idempotent-incremental): ≤50 detector items scanned, ≤25 LLM decisions (counted at actual decider invocations), ≤10 Tier-0/1 applications, ≤5 Tier-2 proposals (a flooded queue is worse than no agent). **Auditability (D4/G17):** every run gets an `agent_run_id`; every graph write journals CodexEvents with `batch_source = agent_run_id` and `source: "maintenance_agent"` in payloads, and every action emits `agent_action` + a closing `agent_run_summary` structlog event (the F5 SSE-promotion candidates). With no bg model (`llm_decider=None`) the job degrades gracefully: Tier 0 and deterministic Tier 1 still run, LLM-gated decisions are skipped.

**`codex_ops.merge_entities(db, keep_id, absorb_id, agent_run_id=None)`** (D5, same module the future codex surgery lives in) is the merge both the Tier-0 path and the review-approve dispatch call: re-point *all* of absorb's edges to keep (a live duplicate of an existing live keep-edge is expired-journaled, the survivor keeping max strength/confidence and the earlier `valid_from`; edges between the pair are expired in place rather than becoming self-loops), union aliases/tags/properties (absorb's canonical becomes an alias of keep), keep the longer description (journaled via `log_description_update`), move absorb's `codex_events` to keep, regenerate keep's `context_payload`, then expire absorb as a **husk**: canonical renamed with a ` [merged:<id8>]` suffix (it must leave the resolution space — `get_or_create_entity` matches canonical before aliases), aliases emptied, `properties.merged_into/merged_at` stamped, row/embedding/description kept — **nothing is hard-deleted** (T-track history). Keep-ordering for detector-generated pairs: higher `codex_events` mention count, tie-break older first event.

The Sentinel itself is gone (D2/D7): `sentinel_monitor.py` deleted, `SentinelRule`/`SentinelEvent` models and tables dropped (migration `f7a3d9c21e46`, which archived the three seed rules into the migration log first), its runtime job and 30-min cadence removed. Of its rule engine only the two real checks survived — as detectors 2 and 5 above; the rest were stubs (audit verdict: removal, not completion).

### **8.10 Fine-Tune Worker**

fine\_tune.fine\_tune\_classifier (crontab Mon 04:00 UTC) periodically retrains the classifier head on the curated\_labels table (populated by the manual POST /user-control/batch/override-tags endpoint). **Reworked 2026-07 (roadmap B4 promotion / G1):** it now loads the **current live checkpoint** (settings.classifier\_model\_path) as the base — not a hardcoded stale one — encodes the curated prompts (embedder on CUDA), holds out a 20% split, trains all parameters (Adam(lr=1e-4), 10 epochs), and always saves a timestamped artifact. **Since B1 it is schema-driven and generation-agnostic:** it reads the head layout, template version and input width from the checkpoint it is fine-tuning, renders prompts through the shared templates (§2.3 — a fine-tune on bare prompt text would quietly reintroduce the very mismatch B1 removed), and applies each head's declared loss (sigmoid heads BCE with per-label pos-weights; a legacy softmax context head cross-entropy). Curated rows carry their own `schema_version`: a v1 row's single 3-way context string **cannot** be mapped into v2's four independent sigmoids — that collapse is the defect B1 removed — so when fine-tuning a v2 model those rows train topic and intent and are **masked out of the context head** rather than having a target invented for them. **Validated promotion:** if there are ≥20 curated rows and the candidate's held-out loss ≤ the live model's on the same split, it backs up the live checkpoint (`_prev_{ts}.pt`) and atomically replaces settings.classifier\_model\_path with the candidate — so the weekly run now actually changes the live model (a reload/restart applies it). A worse or too-small candidate is kept only as an artifact, never promoted. Remaining deferred: hot-reloading the running proxy without a restart, and the thumbs-up/down feedback-collection loop (F9).

### **8.11 Compaction Worker**

compaction.compact\_entities (§4.8, runtime-scheduled every 24 h since C7 — G10 settled) snapshots entities with ≥100 uncompacted events.

### **8.12 Drop Zone and Codex Inject Watcher**

Two standalone watchdog.Observer processes (outside the maintenance runtime) handle file-system ingestion:

- **Drop Zone** (workers/drop\_zone.py) watches ingest\_inbox/ for .txt/.jsonl/.md files, waits for file size to settle, creates a rag\_documents row, chunks into 512-word windows, embeds each via the classifier's embedder, writes rag\_chunks, and moves the file to processed/.

- **Codex Inject Watcher** (workers/codex\_inject\_watcher.py) watches codex\_inject/ for .yaml/.yml/.json files describing entities (canonical\_name, aliases, tags, properties, context\_payload, relations: \[\{target, relation\}\]). It resolves/creates entities with deterministic UUIDv5, and for each relation inserts a CodexEdge at strength=2.0, confidence="active" (manual injection = high confidence) plus a CodexEvent(event\_type="edge\_added", payload=\{"manual\_injection": True\}), with an edge-existence check serving as idempotency. When an inject file **updates an existing entity's description**, the overwrite is journaled via `log_description_update(..., source="codex_inject")` (T4/D13); creating a new entity with an initial description is not an overwrite and is not journaled.

### **8.13 Conversation Summary Worker (C4, 2026-07-19)**

`conversation_summary.run_conversation_summaries(db, llm=None, embedder=None, conversation_ids=None)` (gpu lane, 2 h cadence + the session-end burst — the quartet's fourth member) maintains **one evolving summary per conversation** in the `conversation_summaries` table (PK conversation\_id with ondelete CASCADE — C10's conversation deletion takes the summary with it; columns summary\_text, covers\_through, covers\_turns, embedding vector(1024), updated\_at). The contract is "the whole conversation so far, current" — never a range row like §8.8's.

Per pass: one aggregate query finds conversations with turns newer than their summary's `covers_through`; a conversation without a row earns one only once it **outgrows the sliding window** (the D3a condition — `total_tokens > estimate_recent_window_tokens(turn_count)` with the legacy default budget; the assembler re-checks with the model-derived budget at injection, so the mismatch can only cost an early/late row, never a wrong injection). New turns fold into the existing summary **incrementally** (never a whole-conversation resummarize): C1 representations (inject\_raw ⇒ raw, else the grounded summary, else a raw head; ≤400 words/turn) accumulate into ≤3.5k-word chunks, each folded by one bounded bg-model call — prompt = existing summary + the chunk, **grounded C1-style** (must-keep terms via `turn_density.extract_key_terms`/`must_terms` over the chunk; one retry naming dropped terms on a `summary_coverage` miss), output ≤250 words. Any failed call aborts that conversation with the old row untouched (the next burst retries — never half-advance). Success updates text + `covers_through` (last folded turn's timestamp) + `covers_turns` + a fresh embedding (the shared codex-extractor embedder, lazily imported — G13). Incognito conversations DO get summaries (their own context — the retrieval consumer's scope join is the privacy shield, §6.1 leg 15); `conversation_ids=` restricts a pass (how tests stay off real conversations). **Two consumers:** the assembler's `=== CONVERSATION SUMMARY ===` block for the active conversation (§7.1/§7.2) and the batch-summary leg's cross-conversation half (§6.1). T-track's era digests read these rows as-is (T4 look-ahead preserved).


## **9. Model Registry and Mixture-of-Experts Routing**

ICE does not train its own generation models; it routes each turn to the best locally-served model from a dynamically-populated registry.

### **9.1 Dynamic registry**

The registry is persisted at models/model\_registry.json (\{"models": \{name: entry\}, "updated\_at": …\}). populate\_from\_ollama() queries \{ollama\_base\_url\}/api/tags, and for each model not already in the registry: (i) fetches Hugging Face model-card tags via https://huggingface.co/api/models/\{id\} (with an \_ollama\_name\_to\_hf\_id best-effort mapping — qwen2.5 → Qwen/Qwen2.5-7B-Instruct, gemma4 → google/gemma-4-7b-it, etc.); (ii) if HF tags are non-empty, maps them through HF\_TOPIC\_MAP and HF\_INTENT\_MAP (e.g. code/coding/programming/python → Software\_&\_Tech + Generation; creative/roleplay/storytelling → Creative\_&\_Media + Generation; finance → Business\_&\_Finance + Analysis\_&\_Summarization) and marks confirmed = True; (iii) otherwise falls back to LLM tagging with Qwen/Qwen2.5-3B-Instruct-AWQ (temperature=0.0, max\_tokens=150) and marks confirmed = False. Each entry records topic\_tags, intent\_tags, priority (default 5), context\_window (default 8192), confirmed, base\_url, added\_at.

### **9.2 MoE selection**

find\_best\_model(topic\_tags, intent\_tags, required\_tokens=0) iterates the registry, skipping any entry with confirmed == False (LLM-tagged models never participate in routing) and any entry whose context\_window \< required\_tokens (context-window-aware routing). The score is:

*score = |topic\_tags ∩ entry.topic\_tags| + |intent\_tags ∩ entry.intent\_tags| + entry.priority*

Ties resolve to first-seen in JSON dict order. If no model qualifies, get\_fallback\_model() returns the first confirmed model, else settings.default\_fallback\_model ("qwen2.5:7b").

### **9.3 Session stickiness**

Stickiness is persisted on the conversation row since C7 (D9/G8): **conversations.sticky\_model + conversations.consecutive\_shifts** replace the old in-memory SESSION\_STATE dict (which reset on every restart). After each classification, topic and intent overlap with the previous turn is computed — the previous turn's tags are read from the conversation's latest episodic row (written post-stream, so a rapid-fire second message may compare against the turn before it; same fallback a fresh dict entry had); overlap ⇒ consecutive\_shifts = 0, no overlap ⇒ consecutive\_shifts += 1. The routing decision (only when the client requested model == "ice-proxy"): if a sticky model is set **and** consecutive\_shifts \< 3, keep it; else call find\_best\_model and reset consecutive\_shifts. The conversation key comes from the X-ICE-Conversation-ID header (or a fresh uuid.uuid4()). **Ordering reworked with C16 (2026-07):** selection now happens **before** budgeting and retrieval, so the context budget can derive from the chosen model's context window — the budget conforms to the model rather than the model being re-picked to fit the assembled prompt (the old post-assembly `required_tokens` re-selection is gone; it was one of two duplicated selection blocks, the first of which was dead code, and a third stray `ollama_url` assignment silently disabled registry `base_url` routing — all removed).


## **10. Operational Infrastructure**

### **10.1 FastAPI proxy**

api/main.py constructs FastAPI(title="ICE Proxy", description="Infinite Context Engine — OpenAI-compatible memory middleware", version="1.0.0") and includes the memory\_slots (/memory-slots) and user\_control (/user-control) routers. **Since E0 (2026-07-16) both routers are thin adapters**: parse → service → format, with domain errors translated to HTTP by `routers/adapter.py::service_errors()` (NotFoundError → 404, ValidationError → 400, ConflictError → 409) — every operation's logic lives in `src/services/` (§11.1), and the REST responses are byte-identical to the pre-extraction router (proved by the record-and-compare harness `tests/test_router_parity.py`). Endpoints:

| **Method** | **Path** | **Purpose** |
| - | - | - |
| **GET** | /health | liveness probe |
| **POST** | /v1/chat/completions | main OpenAI-compatible proxy; SSE streaming; enqueues the "post\_flight" runtime job after the stream (C7). Slash-command turns short-circuit here (C11, §11.4) |
| **GET/PUT** | /memory-slots/, /memory-slots/\{slot\_name\} | slot CRUD |
| **POST** | /user-control/initialize | bootstrap a conversation |
| **POST** | /user-control/turns/\{turn\_id\}/bookmark | bookmark a turn |
| **POST** | /user-control/batch/override-tags | bulk-correct labels → CuratedLabel |
| **PUT/GET** | /user-control/conversations/\{conv\_id\}/scope | set/read memory\_scope\_type and cluster\_ids |
| **DELETE** | /user-control/conversations/\{conv\_id\}\[?dry\_run=true\] | C10 deletion cascade; dry\_run returns the identical manifest without deleting (409 mid-generation) |
| **POST/PUT** | /user-control/clusters, /user-control/clusters/\{id\}/assign | explicit cluster creation/assignment |
| **GET/POST** | /user-control/review-queue, /…/\{item\_id\}/approve | human-in-the-loop queue |
| **POST/GET** | /user-control/import, /user-control/import\[/\{id\}\] | F10/F14 conversation import (replay): POST {source\_path, policy, dry\_run} starts a run (or returns the parse-only preview + estimate); GET reports progress (§13) |
| **GET/POST/PUT/DELETE** | /user-control/model-registry… | registry dump/refresh/edit/delete |


The SSE event types emitted during a chat completion are classified, retrieval, context\_ready, generating, and degraded (The five SSE event types are:

- classified – reports the topic tags, intent tags, context\_reliance label, and max\_confidence.

- retrieval – reports the active retrieval legs, whether HyDE was used, and the total tokens injected.

- context\_ready – reports the final fragment count and a breakdown by source (codex, episodic, procedural, rag).

- generating – reports the name of the model selected for generation.

- degraded – fires when the primary model times out; includes the reason and the fallback model name.

These events are consumed by the Textual TUI to render a live observability panel, and by any downstream monitoring system that listens on the SSE stream.

).

### **10.2 PostgreSQL + pgvector**

A single PostgreSQL instance with the pgvector extension is the unified store. api/db.py constructs create\_engine(settings.database\_url, pool\_size=50, max\_overflow=20, pool\_pre\_ping=True, pool\_recycle=3600) and a sessionmaker. Every vector column is Vector(1024) because the same embedder is used everywhere; every embedding column carries an **HNSW cosine index** (idx\_\<table\>\_embedding — first landed by migration b6e2f9a41c73; G6's orphan scripts/database/create\_indexes.sql predated it and was never applied to the live DB), and **store\_meta** stamps the store's embedding identity (§10.7). Filtered vector queries follow a uniform idiom — SELECT …, (1 - (embedding \<=\> :prompt\_embedding)) \[\* decay\_weight\] AS score FROM \<table\> WHERE embedding IS NOT NULL AND \<filters\> ORDER BY score DESC LIMIT :k — used by the episodic, procedural, RAG, batch-summary, and cluster-relevance queries. Worker-side variants cast the embedding explicitly (CAST(:emb AS vector)) because the embedding arrives as a Python list string.

**Schema management.** The schema is defined via SQLAlchemy ORM models in memory/models.py with **Alembic migrations** in alembic/versions/ (`uv run alembic upgrade head`; the DB URL is duplicated in alembic.ini). The pgvector extension is provisioned by the pgvector/pgvector:pg16 image. *(An earlier revision of this doc claimed no alembic existed — stale even then.)*

### **10.3 In-process maintenance runtime (Celery + Redis removed, C7)**

The former Celery-over-Redis stack (broker = backend = redis\_url, implicit default queue, the unused result backend flagged as G18) was deleted 2026-07-11 by roadmap C7 and replaced by the in-process `MaintenanceRuntime` described in §8/§8.1 — one asyncio tick inside the app process, ledger-backed schedule state, lane semaphores instead of queues. Redis's three remaining uses died with it: the broker (gone), the `ice:last_chat_completed` activity flag (now the runtime's in-process `last_activity`), and the `chat:completed` pub/sub (zero subscribers — deleted). docker-compose now runs **postgres only**. The `data/post_flight_buffer.jsonl` write-ahead fallback was deleted with the broker (its replayer never ran — an in-process enqueue's only failure mode is the app being down, in which case the turn wasn't stored either); idempotency keys remain the at-least-once guard.

### **10.4 Idempotency architecture**

Two complementary mechanisms turn the runtime's at-least-once execution (backoff retries, C7) into effectively-once side effects:

- **Worker-side \`idempotency\_keys\` table** (key TEXT PK, processed\_at TIMESTAMPTZ DEFAULT now()). Every GPU-touching worker opens its transaction with key = sha256(scope + ":" + batch\_id); if a row exists, the task returns immediately; otherwise the work runs and the key insert shares the worker's transaction. Scope strings: post-flight sha256(batch\_id) (implicit scope), codex sha256("codex:" + batch\_id), procedural sha256("procedural:" + batch\_id). A crash before commit leaves no key (retry re-runs); a crash after commit leaves the key (retry is a no-op).

- **API-layer deduplication** via the idempotency\_key column on EpisodicMemory itself (sha256(correlation\_id + ":" + user\_message)), keyed on a per-request correlation\_id UUID. Dedup at this layer is informational rather than enforced — there is no unique constraint in the model definition — but it gives the storage layer a defence against retried HTTP requests.

Every worker is a plain callable that re-raises after db.rollback() (C7 — the maintenance runtime owns retries: 30/120/480 s backoff, then error status), so retried executions re-enter from the top and re-check the idempotency key first. One C7 nuance in post-flight: its own key guards only the density/summary stage — the chained codex/procedural/chunk stages are *always* attempted on re-entry, each protected by its own idempotency, so a retry after a partial chain failure completes the missing stages instead of hitting an early return.

### **10.5 GPU resource management**

**Reworked 2026-07-11 (C7, shared-first).** workers/gpu\_check.py keeps only is\_gpu\_busy(), now **dedicated-mode-only** (threshold raised 20 → **70** — 20 stalled bg work whenever the desktop compositor blipped — result cached 10 s); in shared mode it always returns False and the maintenance runtime's in-process idle gating (§8.1) is the truth — the old is\_user\_active() Redis check is deleted. workers/bg\_client\_factory.py routes background-model calls: **shared mode (the default)** targets the main Ollama server, and get\_bg\_model\_name() resolves settings.background\_model\_name → when None (default), the registry's confirmed chat model — while the **per-turn pipeline uses the exact model that served the turn** (model\_used rides from the request through the enqueue into post-flight/codex/procedural). Pinning BACKGROUND\_MODEL\_NAME to a small always-available Ollama model (e.g. qwen3:4b-instruct) gives a "dedicated" bg model with no second server. True **dedicated mode** survives as a power-user escape hatch: a manually-started OpenAI-compatible server on :8002 (default model Qwen2.5-3B-Instruct-AWQ; ./ice no longer launches it). The dead commented SGLang experiment block was deleted (G2). Bg-call timeouts scale with requested output: bg\_timeout(max\_tokens) = 30 × clamp(max\_tokens/500, 1, 6) (G12); retries ride the runtime's backoff.

### **10.6 Configuration system**

api/config.py defines a Pydantic Settings(BaseSettings) with SettingsConfigDict(env\_file=".env", env\_file\_encoding="utf-8"). Env vars are auto-derived from uppercased field names (no explicit Field(env=…)). The fields, grouped by concern:

- **Database**: database\_url (default postgresql+psycopg://ice:ice\_local\_dev@localhost:5432/ice\_db). (redis\_url deleted with the broker, C7.)

- **Embedding identity (G23/C17)**: embedding\_model\_name (Qwen/Qwen3-Embedding-0.6B), embedding\_dim (1024) — the pair the startup guard compares against store\_meta (§10.7); changing either requires a migration plus scripts/ice\_reembed.py.

- **Maintenance runtime (C7)**: maintenance\_intervals (per-job cadences dict, §8.1), user\_active\_threshold\_seconds (90), idle\_burst\_seconds (120), auto\_finetune (False), finetune\_min\_curated (20), background\_model\_name (None ⇒ chat model), bg\_timeout\_base\_seconds (30).

- **Upstream LLM**: ollama\_base\_url (http://localhost:11434), default\_fallback\_model (qwen2.5:7b), background\_model\_mode (**shared** since C7 — dedicated is the manual power-user path).

- **Classifier**: classifier\_threshold (0.3), confidence\_fallback\_threshold (0.75), classifier\_model\_path (models/classifier/ice\_classifier\_v3\_qwen\_ft3.pt), label\_schema\_path (data/labeled/label\_schema.json).

- **DI3 density thresholds**: DI3\_ENABLED (True), DI3\_CODE\_DENSITY\_THRESHOLD (0.3), DI3\_SENTIMENT\_DENSITY\_THRESHOLD (0.4), DI3\_META\_DENSITY\_THRESHOLD (0.2), DI3\_NOISE\_DENSITY\_THRESHOLD (0.8), DI3\_REFERENCE\_DENSITY\_THRESHOLD (0.2), DI3\_LTM\_REFERENCE\_DENSITY\_THRESHOLD (0.1).

Worker tuning constants (GPU\_UTIL\_THRESHOLD, CYCLES\_PER\_DAY, all DECAY\_RATE\_\*, ARCHIVE\_THRESHOLD, COLD\_THRESHOLD, EVENT\_THRESHOLD, STALE\_DAYS, MIN\_REINFORCEMENT, the RRF k=60, all bonus constants, all retrieval thresholds, the token-budget constants) are **module-level Python constants**, not environment-configurable. The B2 memory-decision weights (ltm\_decision\_threshold, ltm\_prior\_bias, ltm\_length\_weight, ltm\_pressure\_midpoint\_tokens, ltm\_pressure\_scale\_tokens, ltm\_bump\_creative/reference/referential/low\_confidence) **are** in settings, added with that rework. The former classifier-path mismatches are **resolved (2026-07, G1)**: both fine\_tune.py and drop\_zone.py now read settings.classifier\_model\_path, and fine\_tune.py promotes a validated candidate onto that path — so the weekly fine-tune's input, output, and the active inference path all agree.

### **10.7 Data longevity: store\_meta, backup, export/import, re-embed (G23/C17, 2026-07-19)**

**Embedding identity, fail-loud.** The store\_meta table (key TEXT PK, value JSONB, updated\_at) stamps the store: key `embedding` holds \{model, dim, stamped\_at\}, keys `reembed:<table>` hold the re-embed runner's per-table progress. The guard has ONE home — `create_core()` calls `store_meta.check_embedding_stamp()` on every boot path (app lifespan and headless ice-mcp). A settings↔stamp mismatch **refuses to boot**, naming the exact commands (`uv run alembic upgrade head`, then `scripts/ice_reembed.py`); a matching stamp with pending re-embed tables boots with a loud `reembed_in_progress` warning (vector legs filter `embedding IS NOT NULL`, so a half-filled store degrades recall, never crashes); a missing stamp bootstraps from settings **only** when the store provably holds zero embeddings (fresh create\_all databases) and refuses unknown-provenance stores otherwise.

**Backup** (src/memory/backup.py + scripts/ice\_backup.sh): one archive = pg\_dump -Fc (host pg\_dump when installed, else the postgres container's) + models/ (incl. model\_registry.json) + the .env snapshot + backup\_info.json (alembic head, git commit, db size) + RESTORE.md with exact restore steps. Backups are local-private (`backups/` is gitignored — they contain the full store and config). FINAL's per-checkpoint snapshots call the same `snapshot_db()`.

**Export/import** (src/memory/portability.py + scripts/ice\_export.py / ice\_import.py): a **state-copy** — one JSONL per table (walked from Base.metadata.sorted\_tables, so new tables join automatically) + manifest.json (alembic head, embedding identity, per-table counts). Vectors are **excluded by default** (derived + heavy; re-encoded on import) — `--with-vectors` for exact clones; `maintenance_ledger` is excluded (machine-local scheduler state + the runtime-lease pseudo-row); secrets are never exported. Import is staged: manifest head must MATCH the target DB's (never skips schema versions) → id-preserving inserts into an empty store (`--merge` skips existing ids; self-referential FKs like episodic parent\_message\_id are stripped and backfilled) → re-embed pass (force — inherited stamps describe the source store) → codex context\_payload regeneration sweep. F10's replay import remains the "relive it" alternative; the manifest's `kind` field names which one an archive is.

**Re-embed runner** (src/memory/reembed.py + scripts/ice\_reembed.py): vector columns are **discovered from the catalog**, never hardcoded — a column without a registered source-text rule is a HARD ERROR naming the spec. Rules (verified against each writer): episodic → the user half of raw\_text (store\_turn embeds only user\_message; full raw\_text fallback covers MCP notes), chunks/rag → chunk\_text, codex entities → canonical\_name **skipping merge husks, C10 deletion husks, and non-conversation sources** (re-encoding a husk would resurrect a deleted entity into vector matching), procedural → pattern\_description, decisions → decision, batch/conversation summaries → summary\_text, cluster centroids → **never encoded**, recomputed as the normalized member mean after episodic finishes. Kill-safe: per-table stamps advance with each committed batch; rerun resumes; the row-count/time estimate prints up front; empty-source rows stay NULL and are counted.

**The C17 cutover (this store's history):** 2026-07-19 the live store was archived (backups/ice\_backup\_20260719\_115426\_paper-era-pre-1024-wipe.tar.gz, 17 MB; the Exp2 mature snapshot was already preserved separately at \~/ice\_exp2\_mature\_snapshot.sql) and then — by user decision — **emptied rather than re-embedded** (the FINAL experiments regenerate everything; the ~7,400-row re-embed would have been ~5 min but carried no value). Migration b6e2f9a41c73 then widened the empty schema, created the indexes, and seeded the stamps. Validation: tests/test\_longevity.py — 26 checks over the spec's 8 (round-trip, coverage+husk-skip+centroid, 384↔1024 parity, guard refusal, unregistered-column error, resume, MRL bit-identity, scratch-DB restore).

## **11. Service Layer & ICE-as-MCP (E0/E7)**

**Added 2026-07-16 (roadmap E0 + E7).** One operation catalog, many surfaces: every user-facing memory operation lives once in `src/services/`, and the REST routers, the `ice-mcp` server, C11's chat commands (adapter #3 since 2026-07-19, §11.4), and (later) F1's frontend are all thin adapters over the same functions.

### **11.1 The service layer (src/services/)**

Plain modules, one per domain — `bookmarks.py` (bookmark/list/latest-turn + `remember_note`, the MCP note store: a bookmarked, decay-immune, lossless turn in the deterministic `ice-mcp-notes` conversation, codex-extracted like any bookmark), `slots.py` (**`VALID_SLOTS` is the per-tier dict since C9, 2026-07-19** — global/project/conversation names in one place, per-tier validation naming the missing attachment, G14's 300-token cap on every write; CRUD + `append_to_slot`, all tier-parameterized), `scoping.py` (conversation scope incl. the G16 privacy re-sync, label overrides, and — since the E-core — the `project=` kwarg: slug/id attaches the conversation to a project, `""` detaches, incognito refuses attachment), `clusters.py`, `review.py`, `registry_svc.py` (file-backed registry under an `fcntl` sidecar lock — app UI and MCP can't corrupt concurrent edits), `graph.py`, `retrieval_svc.py`, `projects.py` (E1, 2026-07-18: register/bootstrap, tasks/decisions CRUD, commit notification, the session-start renderer — §12), `conversations.py` (C10, 2026-07-19: the deletion cascade + the /forget pair — §11.4 and below), with domain errors in `errors.py` (`NotFoundError`/`ValidationError`/`ConflictError`). Signature convention: `fn(db, ...) -> dict` (JSON-safe), services own their commits, **zero FastAPI imports** — grep-gated in `tests/test_services.py` (`fastapi|HTTPException` must not appear under `src/services` or `src/mcp`).

Two members deserve detail:

- **`graph.py` is the codex service surface** and enforces the A7/F3 rule in one place: `entity_view` (note + typed links/backlinks, resolution by UUID, casefolded canonical name, or alias), `entity_edit` — **where manual description editing is born** (T-spec rev 5): writes `description`, journals via T4's `log_description_update(source="mcp_edit" | "manual_edit")`, then regenerates `context_payload` via the extractor's `_regenerate_context_payload` — the payload is derived and never directly writable. `edges_list`, `entity_timeline`, `entity_diff` wrap `src/retrieval/evolution.py` as built (F3's graph view and D1's agent consume these same functions).

- **`retrieval_svc.context_for(db, task_text, scope, budget)`** wraps the chat pipeline's stages — classify → B2 memory-decision → orchestrate — over the **live** classifier/embedder reached through the core object (`ICECore.classifier`, a lazy property; one model load per process, G13) and returns **structured fragments** (text/source\_type/score/token\_count/provenance ids), never a rendered prompt — rendering belongs to each adapter. An explicit pull always orchestrates; the B2 decision is reported in the payload, not used to answer empty-handed. Since E11 (2026-07-19) it mirrors the chat path's D11 scope enrichment — a `conversation_id` whose conversation is project-attached resolves to `scope["project_id"]` + the project's conversation list, so `ice_context` under a coding conversation pulls coding scope — and a project scope freshens that project's working tree first (§12.3). Also here: `recent_turns` (G16-honoring), `conventions` (active procedural patterns), and `session_start_block` (slots + last session summary — E4 enriches this function in place).

- **`review.py::approve` applies, not just flips status** (D1/D2 D6): the dispatch table carries `memory_slot_update` (since C9 routed through the slots service — tier-validated, G14-capped, `proposed_by` recorded as the slot's `updated_by`) and `new_cluster_proposal` plus the two agent arms — `entity_merge` → `src/workers/codex_ops.py::merge_entities` (**the real D5 merge since D1, 2026-07-17** — approving a maintenance-agent proposal performs the full merge, §8.9) and `codex_reconciliation` → apply the supersession the in-line reconciler refused to guess at (`_expire_edge`, journaled, reason `supersession`) — and, since C10/C11 (2026-07-19), `forget_request` → `conversations.apply_forget` (delete the listed turns + their curated labels, expire the listed edges journaled with reason `user_forget`, tolerant of rows that vanished since the proposal). `reject()` exists for `ice_control` and the agent's "rejected pairs never re-proposed" query; the queue stays **agent-proposals-only with one deliberate exception** — chat/MCP writes apply immediately (D8), but `/forget` is fuzzy *and* destructive, so it queues (D1-tier consistency). Besides pending/approved/rejected, items the agent settles itself carry `status="resolved"`, and C10's deletion marks pending items that referenced the deleted conversation's content `status="stale"` (`codex_reconciliation` via the old edge's `source_batch`, `decision_supersession` via the decisions' `source_batch`, `forget_request` via its origin conversation or listed turns).

### **11.2 The ice-mcp server**

`src/mcp/server.py`, on the official `mcp` SDK (FastMCP), stdio transport by default (`--http` for streamable-http), installed as the `ice-mcp` entry point (pyproject `[project.scripts]`, which required adding the hatchling build-system — uv installs no entry points for build-system-less projects). Register once per harness, e.g. `claude mcp add ice -- uv run --directory <repo> ice-mcp`.

**Two-tier tool surface, bloat-controlled (10 tools).** Composite (the in-loop brain): `ice_context(task)` (context\_for; its description tells the model to call it *before* grepping or re-asking the user; since E8 it surfaces active `constraint` decisions FIRST whenever the task mentions their files), `ice_why(name)` (note + T4 timeline), `ice_recent(conversation_id|project)` (project = all of that project's conversations, E1), `ice_conventions`, `ice_where(symbol[, project])` (**engine swapped in place 2026-07-18 as promised**: the E1b code graph resolves definitions with file:line pointers, signature and docstring summary, falling back to codex name/alias resolution for non-code names; the optional `project` arg arrived with E11 — it names which project's working tree to freshen before the lookup, per the §12.3 D3 resolution order), `ice_remember(text, target)` (slot-append or bookmark-note). Micro (full user control, action-multiplexed with guided-self-correction errors on unknown actions): `ice_slots(list|get|set)` (since C9 with `tier`/`project`/`conversation_id` args — project slugs resolve in the adapter, tier names + the cap are documented in the tool description), `ice_graph(view|edit|edges|timeline|diff)`, `ice_control` — the E-core grew its enum from 7 to 18 actions and C10 to 20: the original `scope_get|scope_set` (scope\_set now takes `project`) `|review_list|review_approve|review_reject|registry_view|registry_edit` plus `project_register|project_list|project_status|project_goal|project_reconcile|arch_doc|decisions_list|decisions_add|task_add|task_list|task_status` (each a one-line hop into `projects.py`/`graph.py`; `project_register`'s description tells the model to ASK THE USER before installing the git hook) plus C10's `delete_conversation` (data.dry\_run previews; the description instructs the model to ALWAYS show the user the manifest and get an explicit go-ahead before the real run) and `forget_propose` (queues a `forget_request`; nothing deletes until `review_approve`), and F10/F14's `import_status` (`item_id` = a run id → that run; else the latest run + recent list; §13) — 21 actions — and `ice_bookmarks(list|add)`. Every handler is a short hop into a service; the one shared behavior is `_journal` — **every tool call logs an `mcp_tool_call` structlog event** (the E5 pull-discipline telemetry: if `ice_context` is consulted in <30 % of grepping sessions, the E5 dispatcher contingency activates — deliberately NOT pre-built). MCP writes carry `mcp_edit` provenance (slot `updated_by`, CodexEvent `source` — G17 vocabulary alongside `manual_edit` and D1's `maintenance_agent`). The `ice://session-start` resource renders the session\_start\_block as markdown, and — since E4 (2026-07-18) — appends every registered project's where-was-I block (state, diffstat since the last sitting, constraints, open tasks, fresh decisions; §12.4) under the slots+summary body — the promised in-place enrichment, URI stable. In stdio mode `main()` reroutes structlog to **stderr** — stdout belongs to the protocol framing.

### **11.3 Headless boot, the runtime lease, and standby**

`ice-mcp` startup (user decision: attach-if-running, else boot core, with linger): (1) try a DB connect — reachable ⇒ *attached*; unreachable ⇒ `docker compose up -d postgres` and poll up to 30 s (docker missing/down ⇒ exit nonzero with the exact command on stderr); (2) `create_core()` — lease-checked per §8.1: if the app's runtime holds a fresh `runtime_lease` the MCP core runs in **standby** (its own event jobs still run; maintenance dispatch stays with the app; self-promotion when the app exits), and symmetrically an app booting while an `ice-mcp` session owns the lease defers the same way; the classifier loads lazily on the first `ice_context` (one-time latency, documented in the tool description); (3) **linger** — on MCP shutdown the core stops (releasing the lease if owner) but docker stays up (`ice_core_linger` log); `./stop_ice`/the packaged app remain the explicit all-the-way-down paths.

**Validation (2026-07-16):** router parity 31/31 (`tests/test_router_parity.py`, record-and-compare); `tests/test_services.py` 48/48 (slot/bookmark/scope/cluster/review round-trips incl. both D1/D2 arms, registry-under-lock, graph edit journaling + payload regeneration, context\_for structured fragments, lease/standby/promotion boot logic, grep-gate); `tests/test_mcp_server.py` 21/21 (all 10 tools via the in-process FastMCP client, set/get round-trips, `mcp_edit` journaling, seeded-DB `ice_context`, resource render); plus a real `ice-mcp` stdio boot listing all 10 tools.

### **11.4 Conversation deletion & chat commands (C10/C11, 2026-07-19)**

**`services/conversations.py::delete_conversation(db, conv_id, dry_run=False)` is the ONE deletion path** — the REST DELETE (§10.1), `ice_control delete_conversation`, and `/delete-conversation` all call it. It is **manifest-first**: phase A is pure reads that build the per-store manifest *and* the exact work lists; phase B applies them in FK-safe order inside one transaction (one commit). `dry_run=True` runs phase A only, so the preview manifest is provably identical to what a real run deletes. Cascade semantics: the conversation's live conversation-source codex edges split into **corroborated** (an `edge_added` event whose `batch_source` lies *outside* the deleted conversation's batches — the fact stands on other conversations → kept) and **sole-support** (expired via `_expire_edge`, journaled `edge_expired` with reason `source_deleted` + source `user_deletion` — **excluded from T4 timelines**: deletion is not idea evolution; `evolution.py`'s `_expiry_events` already drops that reason). Edges with no journal events at all (pre-journal legacy) count as sole-support — unprovenanced facts from a deleted conversation violate the deletion's meaning. Entities left with zero live edges, `source='conversation'`, and no user-authored description (no `description_updated` event with source `mcp_edit`/`manual_edit`) are **husked** like D1/D2 merges (renamed ` [deleted:<id8>]`, aliases emptied, embedding NULLed, `context_payload="[deleted]"`, journaled `entity_expired`) — never hard-deleted; surviving touched entities get their `context_payload` regenerated. The sweep then removes turns (chunks CASCADE), cluster links (+ empty clusters deleted, born-here survivors detached), cold rows, batch/conversation/session summaries, session replays, conversation-tier slots, and curated labels; prunes procedural `source_batch_ids` (deactivating emptied patterns); expires live `decisions` whose `source_batch` came from the conversation (bi-temporal `valid_until`, the table's own model); and flips pending review items that referenced the conversation to `status="stale"`. Deletion is **refused mid-generation** (`runtime.generation_in_flight` — the live stream's post-flight would FK-fail into the deleted row). The manifest always restates the G25 caveat: **logs are not touched** — honesty over pretense; G25 owns log redaction.

**`src/api/chat_commands.py` is adapter #3** (C11): in `chat_completions`, after conversation/scope resolution (the parser needs the conv row) and *before* classification (a command must never burn pre-flight latency), a first line starting with `/` routes to `try_handle(db, runtime, conv, text, scope)`. Handled commands **short-circuit the LLM entirely** — the confirmation streams as a minimal OpenAI `chat.completion.chunk` SSE sequence (`model="ice-commands"`) any frontend renders, and the turn is *not* stored (no episodic row, no post-flight; the journal is the record). An unrecognized `/x` gets a help hint, **never silent fallthrough** — a typo'd command must not leak into chat as a prompt. Only the first line is parsed; remaining lines are payload. The v1 set, each handler a short hop into a service: `/remember <text> [in <slot>] [@project|@conversation]` (tiered `append_to_slot`; G14 truncation surfaced in the confirmation), `/slots [tier]`, `/bookmark` (last *stored* turn — the just-streamed reply isn't stored until post-flight), `/search <query>` (`retrieval_svc.context_for` with main.py's scope dict passed verbatim — chat-path parity incl. incognito rules and the E11 project freshen; results render deterministically, date-stamped and source-tagged, **zero LLM**), `/scope auto|project|none` (passes existing cluster/filter state through so a bare `/scope` never wipes assignments), `/forget <text>` (**fuzzy + destructive ⇒ queues** a `forget_request` proposal listing the top matching turns (embedding, visibility-guarded, PgVector bindparam) and live edges (entity-name containment); zero matches queues nothing), `/delete-conversation` (dry-run manifest, then `confirm` within a 10-minute process-local TTL; double-confirm is an idempotent refusal), `/help`. Every applied command journals — slot writes carry `updated_by='chat_command'`, proposals record their proposer, and one `chat_command` structlog event fires per handled command (the F5 telemetry stream beside `mcp_tool_call`; per-command counts feed the F discoverability design conversation).

**Validation (2026-07-19):** `tests/test_c10_c11.py` 56/56 — spec §4 checks 1–9: sole/corroborated/legacy edge fixture with explicit `edge_added` events, husk + hub survival, conversation B untouched, the full per-store sweep with manifest-count equality, dry-run identity, timeline/`history_exists` exclusion, mid-generation refusal, /remember tier + journaling, /slots render, /search dated-and-LLM-free (booby-trapped bg client), /forget queue → approve → apply, unknown-command hint, and the two-step confirm (incl. staling the deleted conversation's pending forget proposal). Regressions: services 48/48, mcp 21/21, c4\_c9 28/28, session\_scoping 13/13, smoke 72.


## **12. Coding ICE (E-coding core, 2026-07-18)**

**Added with roadmap E1+E1b+E9+E3+E4+E8 (commit `fe8df63`, migration `c4e9b7d15a20`; folds E2/E6).** ICE's second mode: memory for *projects* — a registered repo root whose code structure, facts, decisions, tasks, and conversations hang off one `projects` row. The governing distinction is **earned vs derived memory**: conversational codex entities remain earned (irreplaceable, decaying, corroborated); everything the coding core writes into the graph is **derived** — regenerable from source, decay-exempt, journal-free (no CodexEvents — a re-parse writing thousands of events would bury T-track's signal), bulk-rebuild safe, and invisible to conversational retrieval unless the query's project is attached. One codex, namespaced — the partition is columns + filters (`codex_entities.project_id`/`source`, `codex_edges.source`), never a second store.

### **12.1 Projects, tasks, decisions (E1)**

Tables: `projects` (name, slug, multi-root `roots TEXT[]`, `settings` JSONB — ignore globs, `hook_installed`, `unreachable` flag), `project_state` (goal, current\_branch, last\_task\_id, last\_session\_at, last\_reconciled\_commit), `tasks` (title/status pending|active|done|dropped, commit\_hashes, files\_changed; `updated_at` drives staleness), and **bi-temporal `decisions`** (decision/rationale/alternatives\_rejected/files\_affected, `decision_type ∈ decision|constraint|incident`, embedding vector(1024), valid\_from/valid\_until/`superseded_by` — supersession is first-class, no event journal needed; `source_batch` is G17 provenance). `daily_checklist` is a SQL **VIEW** over open tasks + a 14-day staleness flag, not a table. Conversations attach via `conversations.project_id` (`scoping.set_scope(project=…)`; incognito conversations refuse attachment). `src/services/projects.py` is the one operation surface (E0 pattern): registration/bootstrap, status, goal, tasks/decisions CRUD, commit notification, session-start rendering; adapters are `scripts/register_project.py` (CLI), `ice_control`'s `project_*`/`decisions_*`/`task_*` actions (MCP), and the thin `POST /user-control/projects/{ref}/commit` endpoint (the hook's target).

### **12.2 The code graph (E1b) + project facts (E9)**

`src/coding/code_graph.py::CodeExtractor` — Python-first stdlib-`ast` extraction behind a language seam (tree-sitter later, same iter/parse/sync surface). Entities: modules/classes/functions, canonical `"{project_slug}:{module}.{qualname}"` (casefolded; display name in properties), `entity_type` from A7's existing vocabulary (`module`/`class`/`function`), **pointers only** — file\_path, line\_start/end, signature, docstring **first line** — never source text. Code entities carry `source='static_analysis'`, deterministic uuid5 ids (canonical-stable across re-parses, so inbound cross-file edges survive incremental syncs), empty `aliases` (conversational `get_or_create_entity` can never re-attach a chat mention to a code row) and NULL embeddings (name/scope-resolved; the maintenance agent's cosine channel skips them via its source filter). Edges: `imports` (module→module, project-internal) and best-effort static `calls` (same-module names, imported names, `self.`-methods, dotted-module attributes — no type inference), **resolved 1.0 vs heuristic 0.6 encoded in `extraction_confidence`** (codex\_edges has no properties column; heuristic = less-trusted extraction is A3-consistent), all `confidence='active'` under the deterministic per-project batch `uuid5(NAMESPACE_URL, "ice:code-graph:<project_id>")`. Sync is incremental delete-and-recreate per changed file: surviving canonicals update in place, vanished symbols/files hard-delete with their edges (derived memory is regenerable, not history), outgoing static edges rebuild per file; parse errors keep the last good map and stamp `properties.parse_error` on the module unit; default ignore set + per-project `settings.ignore`; 20k-file bootstrap cap; the heuristic ratio is logged at bootstrap (empirical deferral: >60 % on ICE itself ⇒ tighten to imports-only).

`src/coding/project_facts.py` (E9; E6's OKF adaptation settled here): six deterministic parsers — dependencies+versions (pyproject/package.json, uv.lock resolution), DB schema (`__tablename__` ast scan + alembic head), run/test/build commands (Makefile/npm scripts/entry points), config surface (BaseSettings fields + `.env` **key names, never values**), infrastructure (compose services/images/ports), data shapes (`*schema*.json`) — each ONE `codex_entities` row: `entity_type='project_fact'`, `source='derived'`, structured properties + human description + file pointers, re-derived **hash-gated** (content\_hash over source files; the reconciler passes changed\_files so untouched parsers are skipped).

**Visibility (the D3 rules, wired through the orchestrator):** `retrieve()`/`_wide_net_fallback` set `self._scope_project_id` from `scope["project_id"]` (the `_active_timescope` pattern); `_entity_source_filters()` (SQL, applied in exact/similarity/payload matching and enumeration) + `_entity_visible()` (python, applied at traversal expansion) admit conversation entities always and derived entities only for the attached project; `_codex_scope_sets` grants project-scoped queries the **code-graph allowance** — the project's derived entity ids ∪ the deterministic static batch id — and never falls back to unscoped when a project is attached. Scoped rendering shows derived entities' full pointer payloads (they're built from the project itself — the leaks-other-conversations rationale doesn't apply).

### **12.3 Reconcile-on-commit (E3) + reconcile-on-read (E11)**

Commit is the semantic signal; polling is the fallback, never the design. Registration **offers** a post-commit hook (consent-gated, never overwrites a non-ICE hook): the hook curl-POSTs the commit to `/user-control/projects/{id}/commit` → `notify_work_unit("commit", …)` → the handler (registered in `create_core()`) enqueues the `project_reconcile` cpu-lane job; when the POST fails (app down) the hook writes `$GIT_DIR/ice_pending_commit`, which the `project_poll` job (600 s) consumes along with plain HEAD drift — declining the hook costs at most that poll lag. `reconciler.reconcile_project` per commit-range since `project_state.last_reconciled_commit`: (1) `git diff --name-only` → incremental E1b re-parse of changed `.py` files (unknown base — force-push/history rewrite — ⇒ full re-parse; branch switches are covered by the commit diff); (2) hash-gated E9 re-derive; (3) commit + file linking to the active task (status `active`, else most recent `pending`); (4) cue-gated decision extraction over the range's commit messages — enqueued to the **gpu-lane** `decision_extract` job (the reconcile job itself never calls the LLM); (5) state advance (head, branch, updated\_at). Unreachable roots flag `settings.unreachable` — tools report it rather than serving a stale graph as current. Stale-work detection is deliberately NOT here: the maintenance agent's `stale_work` detector (§8.9) re-derives it from the tasks table on the agent's cadence.

**Reconcile-on-read (E11, 2026-07-19, commit `548b433`) closes the gap both commit triggers share:** they anchor on HEAD moving, so large in-session work *without commits* never reconciled and the graph silently drifted from the working tree. `reconciler.freshen_working_tree(db, project)` is the pull-based fix — the graph is guaranteed correct at the moment it's queried (the only moment it matters), with zero idle cost (chosen over an always-on file watcher; the function is the seam if Z1's latency numbers ever justify a debounced `watchdog` push upgrade). Pipeline: `git status --porcelain` cheap gate → ignore-filtered dirty `.py` set (renames contribute both sides; deletions flow through `sync_files`' file-gone path) → throttle (skip when the `(path, mtime, size)` digest is unchanged since the last freshen, or inside `reconcile_on_read_min_interval_seconds` [2.0 s] — a burst of agent reads costs at most one `git status` per window) → the **existing** incremental `sync_files`. It **never advances `project_state.last_reconciled_commit`** — that stays a commit concept; the next real commit's `reconcile_project` re-runs over the same files idempotently. The gate lives in the **service layer only** (E0 rule — every adapter inherits it): `graph.where_symbol` (project resolved per D3: explicit `project` arg → the symbol's `slug:` prefix → the sole registered project → freshen nothing, bare cross-project lookups stay commit-fresh), `retrieval_svc.context_for` under a project scope, and `render_architecture_doc`. Episodic reads (`ice_recent`, `conventions`) never trigger it. Concurrency (D4): `sync_files` opens with `pg_advisory_xact_lock(hashtext(project_id))` — the choke point every writer path (read-freshen, commit reconcile, bootstrap `full_sync`) funnels through, so at most one reparse per project across the app + any `ice-mcp` process; the lock releases at the method's commit. Freshen failures roll back and log (`freshen_failed_read_stays_commit_fresh`) — a read is never broken by its freshness gate. Kill switch `reconcile_on_read` (default on) restores commit-fresh behavior. Validation: `tests/test_reconcile_on_read.py` 16/16.

### **12.4 Session-start block (E4) and the rationale layer (E8)**

`projects.session_start_data/render_session_start` build the where-was-I block — goal/branch/last task, `git diff --stat` since `last_session_at` (reflog `HEAD@{date}`, error-tolerant), open constraints, ≤3 pending/stale tasks, ≤3 freshest decisions. One renderer, two adapters: the chat assembler prepends it (`=== PROJECT SESSION START ===`) only when the turn **opens a sitting** (preflight gap check against the conversation's latest turn; `last_session_at` advances once per sitting), and the `ice://session-start` resource appends per-project blocks. Registration bootstrap = full E1b parse + E9 derive with a printed report; git-log replay is recorded but off by default (F10's importer honors it later).

`src/workers/decision_extractor.py` (E8) is **cue-gated — no cue, no LLM** (A6 stance): constraint cues ("don't touch"/"must stay" family) > incident arcs (error-word AND fix-word both present) > decision verbs ("decided to"/"instead of"/"switched to" …), code blocks stripped first. On a cue: one bounded JSON completion (temp 0, the agent's `make_llm_decider` contract) extracts `{decision, rationale, alternatives_rejected, files_affected, type}`; hallucination-guarded (empty decision ⇒ nothing stored). Dedupe/supersession against the project's active decisions by embedding similarity lives in **`decision_extractor.reconcile_and_insert`, the ONLY path that writes a `decisions` row**: ≥ `decision_duplicate_threshold` (0.95) + same type/files ⇒ silent skip; ≥ `decision_conflict_threshold` (0.85) + overlapping files ⇒ auto-supersede (valid\_until + superseded\_by); ≥ 0.85 with disjoint files ⇒ new row + a `decision_supersession` review proposal. It runs as a direct call in the post-flight gpu chain for project-attached turns and over commit messages at reconcile. **The manual path shares it (G29, 2026-07-28):** `projects.decision_add` — the MCP `decisions_add` action — used to call `_insert` directly while its docstring claimed otherwise, so every manual write added an unconditional duplicate *active* row; it now goes through the same function and returns the same vocabulary (`recorded` | `duplicate` | `superseded` | `conflict_queued`) rather than a bare `"ok"`, because a caller whose decision silently collided with an active one needs to be told. **The payoff paths:** `retrieval_svc.constraints_for_task` surfaces active `constraint` rows FIRST in `ice_context` whenever the task text mentions their files; `graph.py::render_architecture_doc` renders the architecture doc as a **view** — module tree with docstring one-liners + key decisions with rationale + constraints + project conventions (procedural rows) + the last 15 commits read live from git (never re-stored) — served via `ice_control action=arch_doc`; if a section looks wrong, the fix is in the underlying stores.

### **12.5 Coding-mode routing (E2/D11) and validation**

A project-attached conversation resolves its retrieval scope to the project's non-incognito conversations (`scope["conversation_ids"]` — the episodic legs' list-capable `_conv_scope_filter`, C6's seam shipped early, single-conversation behavior byte-identical) ∪ the code-graph allowance; B2 gains `coding_scope` → `ltm_bump_coding` (+0.7, breakdown-logged, bump-not-force). Procedural patterns extracted from project conversations carry `project_id` (conventions); project-scoped patterns are invisible outside their project and pass batch scoping inside it. Since C9 (2026-07-19) a project-attached conversation also **inherits the project's slot tier**: `[PROJECT · conventions/project_context/pending_items/guidance]` render in its PERSISTENT CONTEXT block automatically (§3.5/§7.2) — the standing-context counterpart of the retrieval allowance. Coding classifier labels wait for B1's single bundled retrain.

**Validation (2026-07-18):** `tests/test_coding_core.py` 46/46 — all spec §5 checks 1–12 over the committed `tests/fixtures/mini_repo/` fixture (copied to scratch, git-inited; live DB, self-cleaning, LLM stubbed, decision embedder monkeypatched to fixed vectors); regressions: maintenance agent 43/43, runtime 49/49, services 48/48, mcp 21/21, timescope 61/61, session-scoping 13/13, memory-decision 31/31, router parity 31/31, smoke 69; migration downgrade/upgrade round-trip.

## **13. Ingestion Engine (F10/F14 conversation import, 2026-07-20)**

`src/ingestion/` is **one replay engine, several front-ends**: it re-lives an exported chat history *through the full pipeline* (post-flight density/summary → codex → procedural → clustering → C4/batch summaries) so the result is mature memory, not a searchable archive. It is the **sibling of G23's `scripts/ice_import.py`** — that path is the id-preserving STATE-COPY (`kind: state-copy`); this one is REPLAY (`kind: replay` on the `ImportRun`). The store was empty at 1024 after the C17 wipe; F10 is its first real filler.

### **13.1 The engine (`importer.py::import_conversations`)**

`import_conversations(db, runtime, source: Iterable[NormalizedConversation], policy, *, run_id, classifier, embedder, llm, deadline, progress_cb) -> report`. For each conversation in turn: consecutive same-role messages merge, then messages pair user→assistant (a trailing user turn pairs with an empty assistant half; a leading orphan assistant turn is skipped + counted). Each pair is **stored with `store_turn_async` parity** — `raw_text = "User: … \n\nAssistant: …"`, embedding of the **user half only** via the ONE `src/memory/embedder.py::get_embedder()` (native 1024, never truncated), tags/`context_reliance` from the live classifier when supplied (else `Zero_Shot`), and the session resolved by `resolve_session_id` **from the ORIGINAL timestamp** (so historical sittings fall out of the gap logic for free) — then the real `post_flight.evaluate_turn` runs as a direct call (which itself chains chunking/codex/procedural/decision per C7). A finished conversation is clustered (`run_cluster_assignment(conversation_ids=[cid])`); the whole run ends with C4 conversation summaries over exactly the touched conversations + a batch-summary pass. Original timestamps are **sacred** (T-track): they are never rewritten to import-time.

**Idempotent + resumable (D4).** A conversation's normalized-content hash is its key: a fully-replayed conversation is recorded in `import_conversations` (hash PK) and skipped on re-run; a mid-conversation kill leaves no hash row, so it re-runs and its per-turn keys — `sha256("ice-import:{conversation_id}:{turn_index}:{pair_hash16}")`, with `conversation_id = uuid5(NS_ICE_IMPORT, "{provider}:{source_id}")` — dedupe the already-stored prefix. Deleting an imported conversation later (C10) leaves the hash tombstone: a re-import does not resurrect what the user forgot.

### **13.2 Format adapters (`formats.py`)**

Each returns a `NormalizedConversation {provider, source_id, title, turns:[{role,text,ts}], …}`, pure and golden-tested against committed fixtures: **ChatGPT** (`mapping` tree flattened along the `current_node` parent chain), **Claude** (`chat_messages` including abandoned edit-branches → the current path is the parent chain of the latest-created leaf; empty `text` falls back to joined `content[]` text blocks, thinking/tool blocks dropped), **DeepSeek** (mapping tree with **no `current_node`** — at a branch follow the child whose subtree has the latest `inserted_at`; dialogue lives in `fragments` REQUEST/RESPONSE, THINK/SEARCH/TOOL\_\* dropped), and **generic JSONL** (`{role, content, timestamp[, conversation, title]}` per line — the escape hatch). Other branches are counted (B6 territory), not imported. Garbage/missing timestamps get per-conversation monotonic synthesis from whatever anchors exist (flagged in the report). Unknown structure → error naming the supported shapes. *(FINAL's lme/synth replay adapters relocate here when FINAL is built — spec §5.)*

### **13.3 Decay policy at import (`compute_decay`)**

Per-turn score computed closed-form at insert, with per-day rates **derived from `decay.py`'s own constants** (`DECAY_RATE_UNACCESSED ** CYCLES_PER_DAY` ⇒ 0.95/day; creative 0.99/day, 0.3 floor). **`hybrid` is the default** (user decision): turns ≤30 days old get `preserve` semantics (score 1.0 + a 14-day self-expiring immunity window), older turns fast-forward with aging counted *from* the 30-day threshold — a smooth ramp, so a month-old chat arrives fresh, a year-old one arrives aged, a long-running one gets an aged head + fresh tail. The trio stays selectable: `preserve` (all fresh + 14-day immunity), `fast_forward` (all aged to real date), `fresh` (all 1.0, no immunity). Fast-forwarded non-creative scores floor at `COLD_THRESHOLD` (0.05) so the importer never deletes a row it just created — the next natural decay cycle takes truly-dead rows cold. The immunity window is the new **nullable `episodic_memory.decay_immune_until`** column; `decay.py`'s three decay UPDATEs skip rows whose window is still open (`decay_immune_until IS NULL OR < now`) — self-expiring, no sweeper. (`decay_immune` stays the permanent flag bookmarks own.) F14 raw-dump turns carry the new **`ts_provenance = 'synthetic_raw_import'`** column (vs `'original'`) so T-timelines caveat their synthesized dates.

### **13.4 Raw-log extraction v2 (`raw_slicer.py`, F14)**

For a roleless/timeless text dump: slice at word boundaries into ~2,000-word windows with 200-word overlaps (the shared C2 `chunk_text`); extract turns per slice with one bg-model call; **one open seam call per adjacent pair** reconciles the overlap (the model sees slice A's boundary turns + slice B's head + the raw overlap ONCE and returns the corrected boundary — replacing v1's cold-chunk speaker-guessing); dedup on a normalized-alphanumeric hash across the dump; synthetic timestamps END at the file mtime, spaced 1/min backwards. Output is a `NormalizedConversation` into the same engine. Empirical deferral: if boundary errors show in >10 % of spot-checked seams, raise the overlap to 400 words before touching the prompt.

### **13.5 Service, runtime job, adapters**

`src/services/ingestion.py` is the E0 service (`start_import`, `import_status`): it validates input, prints the D5 cost estimate (~6 s/turn), enforces **one import at a time** (a stale `running` `ImportRun` — heartbeat older than 10 min — is auto-aborted first), and either replays inline (no runtime — CLI/tests) or hands off to the runtime's **`import_replay` gpu-lane job** (§8.1) which slices ~10 min per dispatch and re-enqueues, yielding to live chat. Three adapters over the service (E0 parity): REST `POST /user-control/import` + `GET /user-control/import[/{id}]` (§10.1), `ice_control action="import_status"` (§11.2), and the foreground CLI `scripts/ice_replay_import.py` (boots `create_core(start_runtime=False)` for the store\_meta guard + classifier, replays inline). Schema: `import_runs` (lifecycle + counters + report JSONB) and `import_conversations` (hash ledger), migration `69873bf8e0c8`.

**Validation (2026-07-20):** `tests/test_ingestion.py` 36/36 — spec §4 checks 1–6 over committed synthetic fixtures per format (live DB, stub embedder + module-attr LLM stubs, snapshot-diff cleanup, never truncates): golden adapters (branch/thinking/empty-text handling), end-to-end replay (original timestamps, gap-derived sessions, codex/procedural/cluster rows, C4 summary), 100%-hash-skip re-run with zero duplicates, the decay trio + hybrid + the `decay_immune_until` window regression, the F14 slicer (word-boundary slices, stubbed seam, planted-duplicate dedup, synthetic-time provenance), and the service lifecycle. Real-export dry-runs confirmed the adapters on the user's own Claude (68 conversations/882 turns) and DeepSeek (9/1245) files without importing them. Regressions: longevity 26/26, c10\_c11 56/56, c4\_c9 28/28, services 48/48, mcp 21/21, timescope 61/61, smoke 82.
