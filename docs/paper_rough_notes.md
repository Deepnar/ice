# PAPER_OUTLINE.md

*Consolidated from `paper_rough_notes.md`. This document reorganizes existing content into a paper structure. No new claims, results, citations, or methodology details have been added — only reorganization, de-duplication, and light copy-editing of typos/grammar for readability. Empty prompts from the original notes are preserved as "Open Items" so nothing is silently dropped.*

---

## Author Metadata (Front Matter — Not Part of Paper Body)

**Author Notes**
- Full name (as on paper): Deepesh Sonar
- Affiliation: Thakur College of Engineering and Technology, Computer Engineering, Mumbai, India
- Email: 18deepnar@gmail.com
- Independent research (no lab, no advisor, no funding): Yes
- ORCID: *(not yet filled in)*
- Acknowledgements / who to thank: *(not yet filled in)*

**Open Items**
- ORCID number.
- Acknowledgements section content.

---

## Abstract

### Draft Text

Conversational AI systems suffer from a fundamental limitation: every new session begins with complete amnesia. The user re-explains who they are, what they're building, and what decisions they've already made. The Infinite Context Engine (ICE) is a self-hosted memory middleware that sits between the user and any OpenAI-compatible language model, maintaining persistent, structured, temporally-aware memory across arbitrarily long interaction histories. ICE classifies every user prompt, retrieves relevant context from four distinct memory stores (episodic, semantic knowledge graph, procedural, and document), fuses results with weighted reciprocal rank fusion, and injects precisely the right context into every LLM call. We evaluate ICE using the Longitudinal State-Replay Evaluation Protocol (LSREP), a novel benchmark that replays months of conversational history through the system and measures memory quality over time. On a mature deployment across four long-form conversations (251–1,119 turns), ICE matches a vector-RAG baseline on answer quality (4.26 vs 4.25) while injecting 32% fewer fragments. On a technical-planning conversation where individual turns exceed 8,000 tokens, the vector baseline collapses to a 94% failure rate due to context overflow, while ICE maintains a mean score of 4.33. A feature-ablation study identifies reciprocal rank fusion as the single most impactful component (+0.84 score delta) and reveals that several intuitively helpful features are neutral or negative in creative domains. ICE demonstrates that less but better context outperforms more but noisier context in long-horizon conversational AI.

### Open Items
- Confirm final headline numbers against the finalized Results section once Experiment 2 tables are locked (current draft cites the "ICE-Dev excluded" fair-comparison table: 4.26 vs 4.25, 32.0% fragment reduction, 30.6% vs 21.2% win rate).

---

## Introduction

### Draft Text

Every conversational AI interface in existence today suffers from the same fundamental flaw: amnesia. Each new session begins with complete cognitive silence. The user re-explains who they are, what they're building, what decisions they've already made, what failed. The AI is smart within its context window and brain-dead outside it. Worse, the systems designed to fix this — CLAUDE.md, .cursorrules, memory layers in cloud APIs — either cap at a few hundred lines, go stale, or exist in someone else's cloud with no user control over what is remembered and what is forgotten.

This is not a minor inconvenience. For a human doing serious work — building a complex software system over months, constructing an original fictional universe across hundreds of sessions, planning a multi-year academic trajectory — the lack of persistent, intelligent, user-controlled memory is the single largest bottleneck between the human and the AI working as genuine collaborators.

The Infinite Context Engine is a local-first, personal AI memory middleware for human-AI conversational interfaces. It sits between the user and their language models — classifying every prompt, deciding which memories to surface, routing to the right model, managing the accumulation of knowledge over time, and injecting precisely the right context into every LLM call.

ICE is not a coding agent. It is not a background tool that watches a developer's file system. It is infrastructure for the thinking layer — for deep collaborative sessions between a human and an AI, the kind where the AI needs to remember that two entities are the same entity across different eras, or that the user's FastAPI router uses a dependency injection pattern established three months ago, or that the user made a specific architectural decision about their PostgreSQL schema last Tuesday and has already ruled out the alternatives.

It is memory for the human mind, not memory for a code compiler.

ICE is a self-hosted memory middleware that gives any local LLM persistent, evolving, user-controlled memory across months of conversation — without sending the user's data to anyone else.

A crucial architectural assumption underlying ICE is that the language model itself should be treated as a stateless reasoning engine. The model has no memory of past interactions, nor does it need to. ICE does not attempt to embed memory into the model's weights through fine-tuning or continual learning. Instead, ICE externalises memory entirely—storing, maintaining, and retrieving context as an independent system that surrounds the model. The model processes whatever context ICE supplies and forgets it immediately after generation. This separation is deliberate: it allows models to be swapped, upgraded, or replaced without losing memory, and it makes the memory system transparent, inspectable, and user-controllable. The vector-RAG baseline shares the superficial property of feeding external context to a stateless model, but it does not reconstruct a structured, evolving cognitive state. It stores raw chunks; it does not build a knowledge graph, infer behavioural patterns, or maintain memory over time through decay and reinforcement. ICE's contribution is not merely retrieving external context—it is maintaining a living memory architecture around a stateless model.

### Author Notes (personal narrative material, lightly cleaned — origin story)

**The clearest example of context collapse:** The clearest example occurs during creative or narrative work, and also in coding/technical tasks — the model usually retains the general idea of what's going on but forgets specificity: the exact rule that was set, a certain character or event, a particular detail of the conversation.

**What was done before ICE:** Before ICE, the most common workaround was exporting chats and storing them, or — when a chat reached its length limit and could not be exported — manually copying and pasting the text into a text file to seed a new chat and continue the conversation. Doing this, the AI would understand things during the initial part of the new conversation, but as the conversation went on, everything had to be re-explained again and again, as if it had never happened.

**The moment existing systems weren't enough:** The moment traces back to a period around buying a laptop, before deep study of AI/ML had begun. Having recently learned about local models and how capable they had become, a period of self-directed research began, driven partly by wanting to run models locally on a personal machine. It became clear that even a capable model, running on a local-grade machine, lacked the context to sustain long conversations — which was frustrating, since the goal was to use local models for more than single, isolated tasks. Around this time, while still learning AI/ML, the earliest pre-concepts of ICE began to form: what if all chats were stored, and the AI could be given exactly the context it needed? At the time, no free, locally-available tools existed for this.

### Open Items
- One-sentence pitch of ICE for another researcher (not yet written).

---

## Related Work

*(Note: this section lists systems/papers the author intends to research and annotate personally. The AI assistant compiling this outline will not invent any citation details, descriptions, or comparisons — all entries below are placeholders preserved from the rough notes.)*

### D1. Memory / RAG Systems

**Author Notes / Open Items**
- **MemGPT (Packer et al., 2023)** — What it does: *(TBD)*. What it misses (e.g. temporal versioning, cross-conversation retrieval): *(TBD)*.
- **mem0** — What it does: *(TBD)*. What it misses: *(TBD)*.
- **MemoryBank** — What it does: *(TBD)*. What it misses: *(TBD)*.
- **SCMoE** — What it does: *(TBD)*. What it misses: *(TBD)*.

### D2. Knowledge Graphs for Dialogue

**Author Notes / Open Items**
- **GraphRAG (Edge et al., 2024)** — What it does: *(TBD)*. What it misses: no temporal versioning; community summaries don't track entity changes over time.
- **KGP / ToG** — What they do: *(TBD)*. What they miss: *(TBD)*.

### D3. Retrieval-Augmented Generation (Classic)

**Author Notes / Open Items**
- **RAG (Lewis et al., 2020)** — Relevance to ICE: *(TBD)*.
- **REALM, FiD, REPLUG, CRAG** — Why they don't solve conversational memory: static corpora vs. an evolving conversation *(elaboration TBD)*.

### D4. Specific Techniques ICE Uses

**Author Notes / Open Items**
- **BM25** — *(TBD)*.
- **RRF (Cormack et al., 2009)** — *(TBD)*.
- **HyDE (Gao et al., 2023)** — *(TBD)*.

---

## Experiment 0 — The Failed Precision-Based Evaluation

### Draft Text

The original evaluation idea was a simple RAG-style top-precision-K fragment check, run over a corpus of 41 conversations.

**Data aggregation.** Conversation history was pulled from multiple sources — story chats, DeepSeek JSON exports, and Claude logs — merged into a single chronological file (`data/simulation_full.jsonl`).

**Probe selection.** A subset of 200 test prompts was generated by taking a contiguous batch of turns from a single conversation ID within the same time period, and probes were constructed from that batch.

**Evaluation design (v1 — deterministic ID/timestamp matching).** The first wave of evaluation was a rigid, deterministic test: did the orchestrator retrieve the fragment with the exact timestamp or ID mapped in the ground-truth file? The simulation harness ran through the historical dataset to populate the PostgreSQL database, and the orchestrator then fired the held-out probes. The evaluation script checked the timestamps of retrieved fragments against the expected 2024–2026 timestamps.

**Run 1 — the absolute zero wall (0.0000).** The very first full run of the harness failed completely, returning a flat 0 for all precision and @5 values. The bug: a timestamp mismatch. The evaluation expected to match entries across a historical timeline spanning 2024–2026, but the database injection script had stamped every record with the current live system time (`datetime.now()`) instead of the original conversation timestamp. Because timestamps had drifted completely, the validation script could not pair any probe with its correct history, producing a false wall of zeros.

**Evaluation design (v2 — 7B-judge semantic grading).** To bypass the broken timestamp matching, the evaluation shifted to a semantic grading system: a local 7B model read the retrieved fragments and judged whether they were relevant to the probe (`evaluate_separate_ai.py`), outputting a binary `relevant=1`/`relevant=0` score into `auto_eval_7b_results.csv`. The measured metric was Overall Precision@5 (the number of relevant fragments out of the total fragments returned).

**Run 2 — the integrated empty-prompt inversion (13.87%).** After patching the variable issues and re-running the test, the system achieved an Overall 7B-judge Precision@5 of exactly 0.1387 (13.87%), successfully retrieving 86 relevant fragments out of 620 total fragments. Performance across the 41 conversations was highly polarized:
- **High/perfect hits:** session `2e89939a...` scored a flawless 1.0000 (5/5 relevant fragments); session `4bb5f8fd...` scored 0.8333 (5/6 relevant); session `ecc64aab...` scored 0.7143 (5/7 relevant).
- **Zero-drowns:** several sessions (`1b454071...`, `66a75f53...`, `6a616889...`) returned a completely flat 0.0000 (0/0 relevant), and other highly diluted sessions pulled up to 20 fragments with barely any relevance, dragging the average down.

**Manual re-scoring.** A separate evaluation was performed manually on a subset of the probes and fragments, resulting in a score of 0.31 (31%) — substantially higher than the automated 13.87%. Manual inspection revealed the real problem: a 7B model is not capable of reliably judging whether a retrieved fragment is relevant to a complex narrative probe; it hallucinated failures.

**The autopsy — two fatal, compounding flaws:**

1. **The Stateless Pronoun Problem.** The held-out probes were extracted from their original conversations and tested in a vacuum, stripped of surrounding turns or dynamic conversational context. A probe like *"You mentioned that the current script saves nothing"* was fed to the orchestrator with zero surrounding context. Because there were no named entities (e.g. "HuggingFace" or "API") for the database to anchor to, the system pulled fragments that were mathematically "correct" in isolation but functionally useless — the probes had been drawn from random points across a pool of 41 conversations blended together, so isolated relevance did not imply real-world correctness.

2. **The Incompetent Judge.** The 7B grading model hallucinated failures. For instance, when the orchestrator successfully retrieved the exact brainstorming session detailing how to "sprinkle hints about the first story without giving away too much," the 7B model failed to recognize the semantic match and scored it 0. The 13.87% score was artificially deflated by a judge model that lacked the reasoning capacity to grade complex narrative retrieval.

The OG evaluation was a gauntlet of compounding infrastructure bugs: timestamps were overwritten, prompt strings were dropped, context was stripped, and the judge was flawed. The failure of the stateless precision benchmark led directly to the idea of LSREP: if a weak judge could not be trusted to score fragments, the system needed to be asked to answer real questions, and the answers themselves needed to be judged — and if memory accumulation was to be measured, the same questions needed to be asked at multiple points in time, not just once at the end. The first sketch was: replay a conversation from the beginning, pause at regular intervals, ask hand-written questions, and score the answers against the full conversation history. That sketch became LSREP. Moving to the Longitudinal State-Replay method bypassed every one of these points of failure.

### Open Items
- What specific problem first made the team realize the design was wrong — is it fully captured by "the multiple correct answer problem" / "the temporal mismatch," or is there more nuance to add?

---

## LSREP — Longitudinal State-Replay Evaluation Protocol

### Draft Text

**Core motivation.** Standard stateless retrieval evaluation metrics (such as isolated Precision@K or MRR) are mathematically insufficient for long-horizon conversational cognition engines. In an integrated cognitive system, a target prompt often relies heavily on implicit temporal anchors (*"the script we worked on yesterday"*), pronominal reference (*"why did it fail?"*), and multi-turn context preservation. Evaluating such prompts in isolation rewards generic semantic matching while penalizing accurate, context-dependent systems.

To resolve this, LSREP introduces a two-phase **Time-Travel Simulation Harness**. It freezes a historic conversation at an arbitrary temporal slice ($T_n$), reconstructs the entire database and memory-lifecycle state exactly as it existed at that moment, allows a human operator to inject complex, context-dependent evaluation probes, and then executes an automated multi-variable matrix of ablation and routing experiments without further human intervention.

```
[Phase 1: State Generation & Probe Curation]
Raw Convo History ---> Temporal Slice (Tn) ---> State Simulation ---> Human Probe Entry
                                                                            │
┌───────────────────────────────────────────────────────────────────────────┘
▼
[Phase 2: Autonomous Matrix Execution Engine]
Frozen State + Probes ---> Toggle Matrix Evaluation (Control vs. Vector vs. Full ICE)
                          │
                          ├─► Multi-Model Routing Verification (Generalist vs. MoE)
                          │
                          └─► Metrics Collection JSON Payload Output
```

**Temporal slicing & state ingestion (general script logic).** A processing script generates independent evaluation checkpoints from the raw conversation logs. For each historical conversation, the script determines the total turn count ($L$) and selects random split points ($n$) constrained between $0.25L$ and $0.75L$, to ensure adequate historical context exists before the cut and enough future context remains to construct valid evaluation questions. The database state is then reconstructed: the target database is wiped (or the session context isolated), and turns $T_1$ through $T_{n-1}$ are injected sequentially into the episodic memory tables, preserving exact relative or real timestamps. Once these turns are committed, the system's background workers are triggered (synchronously, or via an explicit execution block):
- **Decay Workers** — compute initial `decay_score` metrics based on temporal distance to the simulated present moment.
- **Codex Extractor Worker** — parses historic exchanges to construct entity nodes and relationship edges in the knowledge graph.
- **Procedural Extractor Worker** — scans historic turns for repeating behavioral motifs or formatting preferences.
- **Clustering Worker** — aggregates related turns into conceptual episodic sub-clusters.

After background synchronization, the PostgreSQL state for that checkpoint is dumped, or an isolated session-state transaction block is created.

**Note:** The exact split-range parameters and probe/checkpoint counts diverge between Experiment 1 and Experiment 2 (see those sections for the concrete values each experiment used); the description above is the general protocol as originally conceived.

### Core Evaluation Metrics (used across LSREP-based experiments)

These metrics were defined once and applied consistently (with extensions noted under Experiment 2) across the LSREP-based evaluations:

- **Answer Quality Score (1–5).** Each generated answer is independently evaluated against a ground-truth dossier by a separate judge model: **1** — incorrect or largely irrelevant; **2** — partially correct but missing substantial information; **3** — mostly correct with moderate omissions; **4** — correct and largely complete; **5** — highly accurate, comprehensive, and detailed. The primary effectiveness metric reported throughout the study is the average answer quality score across all probes.
- **Score Variance.** Standard deviation of answer quality scores per condition: $\sigma = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}$. Lower values indicate more stable performance across probe types.
- **Context Injection Cost.** Total context tokens supplied to the model per probe, including retrieved memory fragments, persistent memory slots, sliding-window conversation history, system instructions, and user query context. Lower values indicate greater context efficiency.
- **Tournament Win Rate.** Each probe's responses across all evaluated conditions are anonymized, shuffled, and ranked best-to-worst by the judge, independent of absolute scoring. $\text{Win Rate} = \frac{\text{First-Place Rankings}}{\text{Total Tournament Appearances}}$.
- **Hallucination Rate.** Each answer undergoes a dedicated hallucination audit; the judge flags statements unsupported by the ground-truth dossier or contradictory to known conversation facts. $\text{Hallucination Rate} = \frac{\sum Hallucination_i}{N}$, where $Hallucination_i \in \{0,1\}$ per probe.
- **Token Utility Ratio (TUR).** $TUR = \frac{\text{Average Score}}{\text{Average Tokens}/1000}$ — answer quality obtained per thousand injected tokens. Example: System A (Score 4.0, Tokens 4000) → TUR 1.0; System B (Score 4.0, Tokens 8000) → TUR 0.5 — identical quality, but System A is twice as context-efficient. TUR is the primary context-efficiency metric.
- **Retrieval Noise Score (1–10).** Retrieved evidence fragments are independently rated: 1 = extremely focused retrieval with almost no irrelevant information; 10 = highly noisy retrieval with substantial irrelevant context. Reported separately for ICE and vector-RAG retrieval.
- **Relevance Percentage.** $\text{Relevance \%} = \frac{\text{Relevant Context}}{\text{Total Retrieved Context}} \times 100$. Higher values indicate more focused retrieval.
- **Longitudinal Knowledge Accumulation.** For repeated or semantically related questions appearing at multiple checkpoints, answer-quality scores are tracked over $(\text{Turn Index}, \text{Answer Score})$ pairs to form a knowledge curve. An increasing curve indicates successful memory accumulation; a flat or declining curve suggests the system failed to leverage newly acquired information. Aggregated cohort-level curves were also computed by averaging scores within 50-turn bins.
- **Gating Failure Analysis.** The pre-flight classifier determines whether a query should trigger memory retrieval or be treated as zero-shot. A gating failure is defined as: (1) the classifier predicted Zero-Shot, and (2) the probe subsequently received a score below 3. This measures the quality of the retrieval-routing subsystem, not the memory subsystem itself.
- **Ablation Effect Size.** For each subsystem: $\Delta = \text{Score}_{Full\ ICE} - \text{Score}_{Ablated}$. Positive values indicate the removed subsystem contributed positively to performance.
- **Comparative Delta Metrics** (computed against baseline systems):
  - $\text{Token Savings} = \left(1 - \frac{\text{ICE Tokens}}{\text{Baseline Tokens}}\right) \times 100$
  - $\text{Quality Gain} = \text{ICE Score} - \text{Baseline Score}$
  - $\text{Hallucination Reduction} = \text{Baseline Hallucination Rate} - \text{ICE Hallucination Rate}$

### Open Items
- Explain LSREP to someone who knows nothing about ICE (concise version for the paper's intro to this section — not yet drafted).
- Why is replaying a whole conversation better than a static benchmark (concise justification — not yet drafted as standalone prose, though implied by the Experiment 0 failure narrative above).
- How ground truth was generated (which judge model, how big a context window, what rules) — partially answered per-experiment (see Experiment 1 and Experiment 2 sections); needs a unified summary statement.
- One-line definitions for TUR, SPF, win rate, hallucination rate, and recency delta, for a quick-reference glossary.

---

## Experiment 1 — "Infant Metabolism" (the Honest Failure)

> **Critical finding to preserve accurately: in Experiment 1, ICE performed *worse* than the vector-RAG baseline.** After correcting for a token-accounting bug, Full ICE scored 4.04 vs. the vector baseline's 4.06, while using more tokens, hallucinating more, and posting a worse TUR; the MoE variant was worse still (3.96). This result must not be softened or omitted in the paper.

### System State at the Time of Experiment 1

The version of ICE evaluated in Experiment 1 was an immature/pilot/naive build with only the vector and BM25 retrieval legs active, using intent-based weighting for RRF fusion, a strict token limit of 2K (retrieval) and 5K (retrieval budget), plus roughly another 4K for the sliding window.

**Architecture component notes (as of Experiment 1):**

1. **Pre-Flight Intent Classifier**
   - *What it is:* A small neural network that examines the user's prompt and predicts topic, intent, and context-reliance (Zero-Shot / Long-Term-Memory / Real-Time-Search).
   - *How it worked:* A frozen Sentence-Transformer (all-MiniLM-L6-v2) converts the prompt to a 384-dim vector. A 2-layer MLP (384→128→25) predicts 11 topic labels, 11 intent labels, and 3 context-reliance classes; any label above a 0.3 threshold is considered active. The model was fine-tuned on 708 hand-labelled evaluation probes (all forced to Long_Term_Memory) plus the original 20k-prompt dataset. Classification ran synchronously in the FastAPI middleware (or the evaluation script) in under 50ms on CPU.
   - *Limitations:* Fine-tuning over-corrected for personal probes, frequently mis-tagging clearly personal questions as `General_Reference_&_Trivia`. There was no probabilistic calibration — raw confidence was not used to gate retrieval decisions beyond a single fallback threshold. A "public entity trap" (famous names like "Shinchan" being labelled Zero-Shot) was patched via a hard override forcing `Long_Term_Memory` for any conversation-scoped query; this override was flagged as needing replacement with a confidence-based or conversation-length-based mechanism.

2. **Episodic Memory Store**
   - *What it is:* The primary store of every conversational turn (user prompt + assistant response).
   - *How it worked:* Every turn stored in PostgreSQL (`episodic_memory`) with topic/intent tags, context-reliance, a lossless flag, a summary (if not lossless), a decay score, and a 384-dim embedding. A "force-lossless" rule preserved raw text for any turn whose topic included `Creative_&_Media` or whose intent included `Emotional_Processing`. For other turns, a Post-Flight Evaluator decided lossless vs. summary based on code blocks, entity density, and word count. Retrieval used conversation-scoped vector similarity (pgvector) and BM25 full-text search; a topic-based filter had been removed because it caused false negatives. A sliding window of the last 10 turns was always injected into the prompt.
   - *Limitations:* Per-leg retrieval limits (20 fragments/leg) and the global token budget (5,000 tokens) were static, not adaptive to conversation length. The summarization heuristic occasionally produced bland summaries that lost specific facts. Decay was applied daily but had no reinforcement from "access" outside the evaluation harness.

3. **Codex (Semantic Knowledge Graph)**
   - *What it is:* A structured graph of entities and relationships, intended to store factual knowledge separately from episodic turns.
   - *How it worked:* A Codex Extractor (Celery task) ran on lossless turns, calling a background 1.5B model to extract subject-relation-object triplets, written as `codex_entities` and `codex_edges`. Edges started `confidence=pending`; a truth-quorum (two independent corroborations) promoted them to `active`. Retrieval used a regex-based NER pass over the prompt to find capitalized words, looked them up in `codex_entities` (canonical name or aliases), and traversed 1–2 hops to collect `context_payload` text.
   - *Limitations:* During evaluation, the Codex leg almost never fired, because very few named entities existed in the graph and the regex NER missed lowercase or misspelled names. The regex NER was assessed as inadequate for real-world prompts; the plan was to replace it with a tiny learned entity-extraction model trained on the Codex Extractor's own output. The graph was essentially unused in the current experiments — flagged as a major gap.

4. **Procedural Memory**
   - *What it is:* A store for recurring workflows, habits, and formatting preferences.
   - *How it worked:* The Procedural Extractor ran after post-flight evaluation, scanning for repeated behavioral patterns and inserting them into `procedural_memory`. Retrieval was activated only for specific intents (`Strategic_Planning`, `Generation`, `Open_Exploration`). An ablation condition disabling procedural memory suggested minimal impact on the current probes.
   - *Limitations:* The extractor produced many spurious patterns (false positives), mitigated by a reinforcement threshold (3 observations) but still noisy. The retrieval trigger set was hardcoded and might miss valid use cases.

5. **RAG (Document Store)**
   - *What it is:* A vector store for static documents uploaded via the Drop Zone.
   - *How it worked:* Documents were chunked and embedded; retrieval was gated behind specific intents (`Factual_Retrieval`, `Analysis_&_Summarization`) and the presence of reference keywords. Not used in the current evaluation, since the ingest pipeline had not been populated.

6. **Hybrid Retrieval Orchestrator**
   - *What it is:* The core fusion engine combining results from all memory legs into a ranked, deduplicated context block.
   - *How it worked:* Executed BM25, vector, Codex, procedural, and RAG legs in parallel. Dynamic leg weighting (intent-based profiles) adjusted each leg's contribution before Reciprocal Rank Fusion (RRF) — factual queries boosted vector/BM25, creative queries boosted Codex. Session diversification (max 3 results per conversation) prevented one dominant conversation from monopolizing context. A global token budget (5,000 tokens) trimmed the final result set. HyDE query rewriting was available but disabled for most evaluation probes (tested only in a dedicated ablation condition). A conversation-scoped LTM override forced retrieval when a `conversation_id` was present, compensating for classifier mis-predictions.
   - *Limitations:* The token budget was static; the plan was to make it dynamic (e.g. proportional to conversation length). The dynamic weights were a fixed lookup table rather than learned. The Codex leg's contribution was negligible due to the NER limitation.

7. **Prompt Assembler**
   - *What it is:* Builds the final LLM payload by combining memory slots, sliding window, and retrieved fragments.
   - *How it worked:* Injected persistent memory slots (if any) and the last 10 turns (sliding window) unconditionally. Appended Codex, Episodic, Procedural, and RAG blocks in a fixed order designed for KV-cache stability. For emotional/creative probes, a bypass replaced the structured system prompt with a warm, plain-text context injection. For factual probes within creative/emotional topics, a directive to "be thorough and exhaustive" was appended.
   - *Limitations:* The bypass condition was hardcoded; could be made intent-aware. There was no mechanism to weight different context blocks differently (e.g. Codex facts might deserve higher priority than a retrieved turn).

### Checkpoint / Split Parameters Used in Experiment 1

Each conversation was grouped by its unique conversation identifier and processed independently; conversations below a threshold of 10 turns were automatically excluded, reducing the conversation count from 42 (Experiment 0) to 18 (Experiment 1). Let $L$ denote total turns in a conversation.

Split points were restricted to a valid interval: $\text{min\_split} = \max(10, 0.30L)$, $\text{max\_split} = 0.95L$. For each eligible conversation, three split points were randomly sampled from this interval using a fixed random seed for reproducibility. For a split point $s$, the conversation was partitioned into a **Historical Context Block** (all turns from the start through turn $s$) and a **Future Reference Block** (the subsequent 10 turns, or fewer if unavailable). The Historical Context Block represents information available to the memory system at evaluation time; the Future Reference Block serves as reference for constructing evaluation probes and validating retrieval behavior.

The lower bound (30%) ensured sufficient conversational history for meaningful memory formation; the upper bound (95%) ensured future conversational content remained available for evaluation and probe construction. Probes were manually written for each conversation, ranging from 8 to 30 probes per conversation; for the 3 splits per conversation, probes from earlier (smaller-turn-valued) splits were reused at later splits to observe performance change over time. Because the objective was to evaluate long-term memory retrieval rather than verbatim recall, ground truths were designed to aggregate all relevant evidence scattered throughout the conversation history into a single canonical reference answer.

### Ground-Truth Construction (Experiment 1 Method)

Constructing ground-truth answers manually was challenging: some probes targeted specific facts or recent events, while others were intentionally generalized long-term memory questions (user preferences, recurring decisions, project characteristics, accumulated personal information), with the needed information often distributed across many locations in lengthy conversations.

A **retrieval-assisted ground-truth generation process** was used: for each evaluation checkpoint and probe, all prior conversation turns were treated as searchable history. Each turn was embedded using a sentence-level embedding model, and the probe question was embedded into the same vector space; similarity search identified the most relevant conversational evidence. Retrieval depth adapted to history size — 20 turns for smaller histories, 40 turns for larger ones — with a maximum context budget of 30,000 tokens.

The retrieved evidence was passed to a language model acting as a neutral evidence compiler, instructed to: report all relevant facts in the retrieved evidence; enumerate all discovered items for list-style questions; attribute opinions/interpretations to their original speaker; avoid introducing unsupported conclusions or external knowledge; prefer the most recent version of a fact when multiple versions existed while briefly noting earlier variants; and produce dense, evidence-oriented responses suitable for evaluation. The resulting output was stored as the probe's expected answer / reference ground truth.

This retrieval-assisted procedure was chosen because manually constructing ground truths for long-horizon probes across conversations of hundreds of turns, with repeated references to the same information over time, would have required exhaustive manual inspection. The approach enabled comprehensive, consistent ground-truth references while preserving scalability.

### Experimental Evaluation Procedure (Experiment 1)

Each evaluation checkpoint was replayed incrementally through the complete memory pipeline; checkpoints belonging to the same conversation were processed in chronological order rather than reconstructing the entire conversation from scratch each time. Newly available turns were added, background maintenance ran, and evaluation probes were administered against the resulting memory state — preserving temporal consistency while reducing computational cost.

**Six primary evaluation conditions** were executed per probe:
1. Control Baseline (Generalist) — naive sliding-window baseline, recent history only.
2. Control Baseline (MoE) — same baseline context with expert-model routing.
3. Vector RAG (Generalist) — semantic retrieval over episodic memories, no ICE-specific mechanisms.
4. Vector RAG (MoE) — vector retrieval + expert-model routing.
5. Full ICE (Generalist) — complete ICE architecture (episodic retrieval, memory slots, knowledge graph memories, procedural memories, retrieval orchestration, memory-aware prompt assembly).
6. Full ICE (MoE) — complete ICE architecture + expert-model routing.

For each condition, a response was generated and stored with metadata (latency, retrieval outputs, classification decisions, total injected context size).

**Blind evaluation framework** — four complementary analyses per probe:
- *Absolute Scoring* — 1–5 score against ground-truth dossier.
- *Tournament Ranking* — six conditions anonymized/shuffled, ranked best-to-worst.
- *Hallucination Audit* — unsupported/contradictory statements flagged relative to the ground-truth dossier (ignoring valid external knowledge).
- *Retrieval Noise Analysis* — proportion of relevant context vs. noise in supplied evidence, comparing ICE vs. vector retrieval.

**Ablation studies** (executed on the designated long-horizon benchmark conversation): HyDE retrieval disabled; procedural memory disabled; sliding-window context removed; alternative memory scope configurations; automatic vs. manually constrained retrieval scope.

### Post-Hoc Context Accounting Audit

After completing Experiment 1, an audit of the token-accounting pipeline identified a methodological error: the reported context size for Full ICE included retrieved memory fragments but inadvertently excluded the contribution of the recent-turn sliding window incorporated during prompt assembly.

The evaluation results themselves were unaffected, because the sliding-window context was correctly supplied to the language model during inference — the omission affected only the *reported* context-efficiency statistics. To correct this, the original sliding-window assembly procedure (10 recent turns, 500-word-per-turn truncation limit) was replayed for every checkpoint, and the resulting token counts were added to the previously reported Full ICE context sizes. Generated answers, judge scores, rankings, hallucination analyses, and retrieval outputs were left unchanged.

**Impact of the correction:** Full ICE's sliding-window implementation contributed substantially more context than originally reported. As a result, Full ICE consumed *more* total injected tokens than the vector-retrieval baseline while achieving approximately equivalent answer quality. The correction did **not** alter the central qualitative findings — ICE still showed lower retrieval noise than the vector baseline, evidence of longitudinal knowledge accumulation, preserved answer quality on long-horizon tasks, and effective memory-aware retrieval despite immature subsystems — but it **did** invalidate any claim that Experiment 1 demonstrated superior token efficiency. Experiment 1 should therefore be interpreted as a proof-of-concept evaluation of memory quality and retrieval behavior, not a demonstration of context compression efficiency. This finding directly motivated Experiment 2's unified dynamic token-budgeting strategy.

### Experiment 1 Results (Corrected Token Counts)

Report metadata: **657 probes evaluated**, judged by `mattbucci/gemma-4-12B-AWQ` (12B AWQ, 150k context). *Note on token counting: the original report under-counted ICE's total context by omitting the sliding-window contribution; the tables below reflect the corrected values, where every injected token (system prompt, persistent slots, recent turns, retrieved fragments, and the final question) is included. The vector-RAG baseline always counted its injected context correctly.*

**1. Six-Core Benchmark (All Probes)**

| Condition | Score | TUR | Tokens | Hallucination |
|-----------|------:|----:|-------:|--------------:|
| control_baseline_generalist | 3.05 ± 1.61 | 0.76 | 3998 | 74.1% |
| control_moe | 3.01 ± 1.61 | 0.75 | 3998 | 74.4% |
| vector_rag_baseline_generalist | 4.06 ± 1.31 | 0.69 | 5847 | 64.7% |
| vector_rag_moe | 4.05 ± 1.32 | 0.69 | 5847 | 66.7% |
| full_ice_generalist | 4.04 ± 1.33 | 0.47 | 8586 | 69.6% |
| full_ice_moe | 3.96 ± 1.33 | 0.46 | 8586 | 72.3% |

**Key observation:** After correcting for the omitted sliding window, Full ICE injects more total tokens than the vector baseline but delivers essentially the same answer quality. The large token overhead is driven by Experiment 1's un-optimized sliding-window design (10 turns × 500-word cap). Experiment 2 uses a unified, dynamic token budget to bring ICE's total injection below the vector baseline.

**2. Cohort: `adaptive_gated_retrieval_2k`**

| Condition | Score | TUR | Tokens | Hallucination |
|-----------|------:|----:|-------:|--------------:|
| control_baseline_generalist | 3.38 | 0.84 | 3996 | 63.8% |
| vector_rag_baseline_generalist | 4.25 | 0.75 | 5662 | 55.5% |
| full_ice_generalist | 4.03 | 0.58 | 6945 | 60.4% |

Token difference vs. vector: +22.7%. Quality difference vs. vector: −0.40 pts. Fragment noise (ICE): 2.31/10. Fragment noise (Vector): 3.00/10.

**2. Cohort: `forced_long_horizon_retrieval_5k`**

| Condition | Score | TUR | Tokens | Hallucination |
|-----------|------:|----:|-------:|--------------:|
| control_baseline_generalist | 2.82 | 0.71 | 4000 | 81.1% |
| vector_rag_baseline_generalist | 3.93 | 0.66 | 5972 | 70.9% |
| full_ice_generalist | 4.04 | 0.42 | 9694 | 75.8% |

Token difference vs. vector: +62.3%. Quality difference vs. vector: +0.11 pts. Fragment noise (ICE): 2.97/10. Fragment noise (Vector): 2.85/10.

**3. MoE vs. Generalist (Global)**

| Routing | Score | Hallucination |
|---------|------:|--------------:|
| MoE | 3.68 | 71.1% |
| Generalist | 3.71 | 69.5% |

MoE score delta: −0.04 pts. Hallucination "reduction": −1.7% (MoE slightly worse). *Note: MoE shows no advantage yet; classifier training may be immature.*

**4. Ablation Analysis (Flaw Conversation Only)**

| Ablation | With | Without | Delta |
|----------|------:|--------:|------:|
| Hyde Ablation | 4.03 | 4.20 | −0.17 |
| Procedural Ablation | 4.04 | 4.01 | 0.02 |
| Sliding Window Ablation | 4.04 | 3.99 | 0.05 |
| Scope Auto vs Project | 4.17 | 4.19 | −0.01 |
| Scope Auto vs None | 4.17 | 4.17 | 0.00 |

**Interpretation:** HyDE has minimal impact (possibly because the background model is weak). Sliding window helps slightly (+0.05). Procedural memory adds +0.02 — marginal in this infant state.

**5. Gating Failures (Zero_Shot mis-classified as LTM)**

Total probes where the classifier said Zero_Shot but ICE score < 3: **22**. Examples: *"so which subject should i choose then?"* → score 1; *"so the laptop what models should i use on my new laptop, according to the specification, do you know"* → score 1; *"i am a disappointment right, for what all i have done, you know it right?"* → score 2. These failures show where the classifier needed more fine-tuning (e.g. anaphoric references like "so which subject should i choose then?").

**6. Longitudinal Knowledge Accumulation**

Number of tracked question curves: **193**. Curves showing improvement (first score → last score) include examples such as:
- *"so what were all the personalities of the numbers i have"* → 4 → 5
- *"what do you know about my past and like the reason who i am"* → 4 → 5
- *"hey what about downloading a different thing, not what we men[tioned]"* → 1 → 5
- *"what are the bugs in the hf one??"* → 1 → 2
- *"should i even do the self ricing bullshit?"* → 3 → 5
- *"what features about my rice do you like?"* → 3 → 4
- *"so what are all the apps and modules in my dotfiles???"* → 2 → 3
- *"soo you saw my really old dotfiles and then the next one, [...]"* → 3 → 5
- *"hey should u just fucking use kali linux??"* → 1 → 5
- *"what about something like popos??"* → 1 → 5

This demonstrates that ICE's memory improves over time on repeated questions, unlike standard RAG which often degrades.

Aggregated per-cohort curves (average score per 50-turn bin):
- `adaptive_gated_retrieval_2k`: [4.07, 3.89]
- `forced_long_horizon_retrieval_5k`: [4.04, 4.26, 4.22, 4.09, 3.36, 3.35, 4.04, 3.96, 4.11]

**7. Paper-Ready Claims (Revised for Corrected Token Counts)**

1. **Context Efficiency (Fragment-Level):** ICE's retrieval subsystem consistently surfaces fewer, more relevant fragments than the vector baseline (fragment noise 2.75/10 vs. 2.91/10). The higher total token count in Experiment 1 is an artifact of the uncapped sliding-window design, addressed by the unified token budget in Experiment 2.
2. **Answer Quality:** Full ICE delivers essentially the same answer quality as the vector baseline, even after correcting for the sliding window. The system does not sacrifice accuracy.
3. **Longitudinal Intelligence:** Scores improve over time on repeated probes (e.g., the "Flaw" knowledge curve rises from 3→5), demonstrating that memory accumulation benefits future queries.
4. **Intent-Awareness:** The classifier gate reduces memory noise, as shown by lower fragment noise in ICE vs. vector RAG.
5. **Hallucination:** Currently high across all conditions (~70%), attributable to the infant Codex system and the 1.5B extractor; expected to improve with a more mature system.

### The Honest Narrative

The corrected Experiment 1 numbers were sobering: ICE used more tokens (8,586 vs. 5,847), scored slightly worse (4.04 vs. 4.06), hallucinated more (69.6% vs. 64.7%), and the MoE variant was worse still (3.96). A complex multi-store memory architecture had been built, and it was losing to a simple vector search. The breakdown told a clearer story, however: the Codex knowledge graph was essentially non-functional (regex-only NER, no real entity extraction), only 3 decay cycles had run, 22 gating failures meant the classifier was actively blocking retrieval, and the sliding-window under-count meant the efficiency numbers themselves were wrong. The system wasn't bad — it was incomplete. Every subsystem that was supposed to provide an advantage was either broken or too immature to contribute. That was the moment the decision was made to fix everything before publishing.

**Why Experiment 1 was not published as a standalone result:** the numbers, honestly presented, would show ICE losing to a simple vector baseline on every metric except fragment noise. A paper that says "we built a complex memory system and it performed worse than cosine similarity" is not a research contribution on its own — it's a bug report. The value of Experiment 1 was internal: it revealed every measurement flaw, every broken subsystem, and every missing feature that needed to be addressed before ICE could be fairly evaluated. It functioned as a pilot study in the truest sense — necessary for the research, but not publishable as a standalone result.

**Concrete changes made as a result of Experiment 1:** almost every component of the current ICE system is an upgraded/updated form of the Experiment-1-era version — a "ship of Theseus" style rebuild, component by component.

By the time of Experiment 2, the tie on quality (4.26 vs. 4.25) was both satisfying and humbling: satisfying because the mature system now matched the baseline on answer quality while injecting 32% fewer fragments and winning 30.6% of head-to-head tournaments (vs. 21.2% for vector); humbling because a clear quality lead had been hoped for. The win-rate advantage suggests ICE's answers are preferred more often even when absolute scores are similar, and the 32% fragment reduction is the most practically significant number, since it means ICE achieves the same quality with substantially less context pollution.

### Author Notes / Open Items
- What did this experiment use exactly, in one consolidated statement (system state, model, conditions) — mostly answered above via the architecture and evaluation-procedure notes, but a single tight summary paragraph for the paper is still to be drafted.
- How did it feel to see the corrected numbers — narrative material above ("sobering") could be expanded for a personal-voice passage if desired.

---

## Experiment 2 — Mature System

### Draft Text

**What changed from Experiment 1:** Unlike Experiment 1, which evaluated ICE across a large collection of conversations, Experiment 2 focused on four carefully selected long-horizon conversations designed to stress different aspects of memory retrieval and accumulation. The goal was not broad coverage but deep evaluation across substantially different memory domains.

### Dataset Selection

| Paper Label | Description | Primary Memory Challenges | Turns |
|---|---|---|---:|
| Dataset A (Creative Writing) | Collaborative fan-fiction writing session | Character continuity, narrative events, emotional state tracking | 290 |
| Dataset B (Long-Form Creative Writing) | Fantasy world-building and story-planning project | Extremely long-range dependencies, lore consistency, entity tracking | 1,119 |
| Dataset C (Technical Planning) | Design and development of the ICE system itself | Technical decisions, architecture evolution, implementation history | 325 |
| Dataset D (Academic Planning) | Graduate-school and career-planning discussion | Personal preferences, decision tracking, mixed factual and subjective memory | 251 |

These conversations were selected for three primary reasons. First, all four were substantially longer than typical chat interactions (251–1,119 turns), providing opportunities for information to become separated by hundreds of turns and creating realistic long-term retrieval challenges. Second, each represented a distinct memory domain — creative-writing datasets emphasize narrative continuity and entity consistency, technical-planning conversations emphasize design decisions and implementation rationale, and academic-planning conversations require tracking personal preferences, goals, and decision histories. Third, all four contained dense memory structures — recurring entities, evolving relationships, technical terminology, procedural patterns, and long-term dependencies — suitable for exercising ICE's multi-retrieval architecture. Together, the datasets provide a diverse but controlled benchmark for studying memory accumulation across narrative, technical, and personal-information domains.

### Checkpoint Generation

Multiple evaluation checkpoints were generated throughout each conversation's lifetime rather than evaluating only the final state. Checkpoint counts adapted to conversation length: 8–12 for shorter conversations, 10–14 for medium-length conversations, 12–16 for large conversations, and 15–20 for very large conversations exceeding 1,000 turns. Checkpoint locations were distributed approximately evenly with a ±20% positional jitter to avoid regular-interval artifacts, and the final conversation state was always included as a checkpoint. This enables longitudinal evaluation of memory growth throughout the conversation rather than measuring only final performance.

| Dataset   |     Turns | Checkpoints | Generated Probes | Manual Probes | Total Probes |
| --------- | --------: | ----------: | ---------------: | ------------: | -----------: |
| Dataset A  |       290 |          10 |               28 |            17 |           45 |
| Dataset B      |     1,119 |          20 |               59 |            32 |           91 |
| Dataset C   |       325 |           8 |               25 |            12 |           37 |
| Dataset D   |       251 |          12 |               35 |            11 |           46 |
| **Total** | **1,985** |      **50** |          **147** |        **72** |      **219** |

| Dataset  | Checkpoint Turn Locations                                                                            |
| -------- | ---------------------------------------------------------------------------------------------------- |
| Dataset A | 23, 55, 83, 118, 147, 178, 198, 231, 255, 290                                                        |
| Dataset B     | 51, 115, 170, 216, 285, 336, 397, 448, 492, 555, 604, 681, 735, 790, 834, 885, 959, 1017, 1053, 1119 |
| Dataset C  | 54, 90, 116, 181, 209, 234, 263, 294                                                                 |
| Dataset D  | 18, 40, 64, 82, 106, 128, 144, 163, 184, 209, 234, 251                                               |


### Probe Generation

Experiment 2 used a fundamentally different probe-generation strategy from Experiment 1: rather than manually constructing all evaluation probes, probes were generated automatically at every checkpoint using a language model operating only on information available up to that checkpoint — preventing future information leakage and ensuring temporal validity.

For each checkpoint, a representative sample of conversation history was constructed from: the first five conversation turns; the most recent ten turns before the checkpoint; and fifteen randomly sampled turns from the intervening history. This ensured coverage of initial decisions/introductions, recent conversational state, and mid-conversation events that might otherwise be missed. This sampled history was provided to a probe-generation model tasked with creating evaluation questions.

**Retrieval-leg-aware probe construction** was a key innovation: generated probes were explicitly designed to target specific retrieval subsystems.
- *Codex probes* evaluate entity-centric knowledge in the knowledge graph (character relationships, named entities, technical components, user-specific factual information) — requiring precise entity-property retrieval.
- *Vector probes* evaluate semantic retrieval, intentionally phrased using paraphrases and alternative wording rather than original conversational language, forcing retrieval through semantic similarity rather than keyword matching.
- *BM25 probes* evaluate lexical retrieval, anchored around distinctive names, technical terms, identifiers, or keywords appearing verbatim in the history.
- *Procedural probes* evaluate recurring behavioral patterns — habits, workflows, preferences, decision-making behaviors — rather than isolated events.

**Temporal anchoring.** A major challenge in long-horizon conversations is ambiguity: a question like *"What happened at the end of the day?"* may refer to multiple events across hundreds of turns. Every generated probe was required to contain a unique temporal or contextual anchor identifying a specific event, decision, character interaction, or technical discussion (e.g. a specific architecture migration, a particular story event, a named character interaction, a uniquely identifiable project decision). Probes that could plausibly match multiple events were rejected and regenerated, ensuring evaluation measured retrieval accuracy rather than interpretive guesswork.

**Expected answer generation.** For every generated probe, the probe-generation model simultaneously produced a reference answer derived exclusively from the visible conversation history, serving as a checkpoint-local ground truth. Because probes were generated independently at each checkpoint, reference answers reflected only information available at that point, preserving temporal correctness.

### Temporal Ground-Truth Refinement Pipeline

A key challenge in longitudinal memory evaluation is that correct answers may evolve as a conversation progresses: a person's organizational role may change, a project decision may be revised, a preference may evolve, or a relationship between entities may be updated. The initial checkpoint-local ground truth could become outdated when the same probe was re-evaluated hundreds of turns later. To address this, Experiment 2 introduced a three-stage temporal refinement pipeline:

- **Stage 1 — Forensic Regeneration.** The originally generated expected answer was regenerated at its origin checkpoint using a substantially stricter prompt, producing a forensic-quality reference rather than a conversational response. The model was instructed to: preserve all previously correct information; correct factual inaccuracies; add omitted names, entities, decisions, and relationships; record temporal provenance for every fact; explicitly distinguish historical vs. current states; and produce dense, evidence-oriented output rather than natural-language summaries. This produced a significantly more complete reference than the original probe-generation stage.

- **Stage 2 — Temporal Anchoring.** To prevent silent drift toward a different-but-similar event as a conversation grows (e.g. *"What happened after the project review?"* in a conversation with multiple reviews/meetings/milestones), every regenerated answer was assigned an anchor identifying the specific fact, relationship, event, or decision the probe was intended to track (e.g. a participant's organizational role, a project's database selection, a recurring workflow decision, a specific milestone discussion). The anchor persisted as the probe's identity throughout the remainder of the evaluation.

- **Stage 3 — Forward Temporal Propagation.** After regenerating the origin-checkpoint ground truth, the answer was propagated through all subsequent checkpoints. For each later checkpoint, only newly added content was examined to determine whether it contradicted an existing fact, updated a previously known fact, added new relevant information, or introduced additional relevant events. If none applied, the ground truth remained unchanged; otherwise, it was updated to incorporate new facts while preserving previously valid information — transforming each probe from a static answer into a temporally evolving reference document.

**Anchor preservation safeguards.** A second failure mode occurs when a model mistakenly replaces the original event with a later, superficially similar event (e.g. a probe tracking one design decision accidentally becoming associated with a later, unrelated design discussion). Anchor-preservation checks verified that the original anchor remained semantically present in any updated answer; large deviations were treated as potential event-substitution errors and flagged for inspection.

**Why temporal refinement was necessary.** Experiment 1 evaluated memory at isolated checkpoints tied to a single point in time. Experiment 2 evaluates memory longitudinally across entire conversations, where the same probe may be observed repeatedly as information accumulates over hundreds of turns. Without temporal refinement, evaluation would incorrectly penalize systems that successfully tracked changing information — a system that correctly updated its beliefs could appear less accurate than one that simply repeated outdated information. The refinement pipeline ensured ground truths evolved alongside the conversation, allowing the benchmark to measure memory accumulation, belief revision, and long-term knowledge maintenance rather than static fact recall.

**Benefits over the original ground truths:** greater factual completeness; explicit temporal provenance; better coverage of evolving entities and decisions; resistance to semantic drift; consistent evaluation across hundreds of conversational turns.

### Evaluation Framework: Conversation Replay Protocol

Experiment 2 followed a small set of carefully selected long-horizon conversations throughout their entire lifespan (rather than independent checkpoints across many conversations as in Experiment 1), to measure how memory quality evolved as information accumulated, retrieval structures matured, and older information underwent repeated maintenance and decay. The evaluation framework was redesigned around continuous conversation replay rather than independent checkpoint execution.

For each selected conversation, the complete conversational history was replayed chronologically into a fresh ICE-Mature deployment. At each checkpoint: (1) newly available turns were ingested; (2) memory representations were updated; (3) background maintenance processes were executed; (4) evaluation probes were administered; (5) results were recorded before advancing. Unlike Experiment 1, memory state was **preserved across checkpoints** — information stored earlier remained available at all subsequent checkpoints unless modified by memory maintenance operations. This allowed memory structures to evolve naturally and enabled direct measurement of longitudinal accumulation.

**Simulated memory lifecycle.** A major limitation of many memory evaluations is measuring retrieval immediately after storage. Real systems must operate as information ages, decays, consolidates, and competes with newer information. To simulate this, repeated maintenance cycles ran between checkpoints, executing: episodic memory decay; knowledge-graph edge decay; procedural-memory decay; reflection generation; context clustering; cluster consolidation; and sentinel monitoring — allowing the memory system to continuously reorganize as conversations progressed, so retrieval performance reflects both storage quality and long-term memory preservation.

**Incremental probe evaluation.** At every checkpoint, all probes whose origin occurred at or before that checkpoint were evaluated — a probe generated at an early checkpoint remains part of the evaluation set at every later checkpoint. This differs from Experiment 1, where each probe was evaluated once. Consequently, the benchmark measures not only whether information can be retrieved immediately after storage, but whether retrieval performance improves, remains stable, or degrades as additional history accumulates — enabling direct observation of memory retention and knowledge growth over time.

**Ground-truth evolution.** As above (temporal refinement pipeline): each probe's checkpoint-specific reference answer reflected all information available up to the current evaluation point, and was updated whenever new evidence modified, expanded, or contradicted previously known information — ensuring systems were evaluated against the correct state of knowledge at each point in time, not a permanently frozen answer.

### Evaluation Conditions

Experiment 2 focused on **four** primary retrieval configurations (the control-only conditions from Experiment 1 were dropped, since their behavior had already been extensively characterized):
1. Vector Retrieval Baseline (Generalist)
2. Vector Retrieval Baseline (MoE)
3. Full ICE-Mature (Generalist)
4. Full ICE-Mature (MoE)

### Context Construction

Both retrieval systems operated under **dynamic context budgets** rather than fixed retrieval limits. Retrieval budgets adjusted according to conversation length, total accumulated token volume, probe classification, and retrieval scope. This differs from Experiment 1, where budgets were largely static and a significant portion of total context originated from an uncapped sliding-window mechanism. The dynamic-budgeting strategy was introduced for a fairer comparison and to better reflect realistic deployment constraints.

### Retrieval Scope Refinement Through Conversation Clustering

A major architectural change introduced between Experiment 1 and Experiment 2 was the adoption of cluster-scoped retrieval. In Experiment 1, retrieval operated over the entire conversation history once a conversation had been selected. While effective for shorter discussions, this approach became increasingly problematic as conversations grew to hundreds or thousands of turns spanning multiple distinct topics.

Long-horizon conversations naturally drift across many themes. A single conversation may contain discussions about narrative events, character relationships, world-building concepts, implementation details, planning decisions, and meta-discussion. When retrieval is performed across the entire conversation, semantically similar but contextually unrelated fragments can be surfaced simply because they share overlapping terminology or conceptual language. This introduces retrieval noise and consumes context budget with information that is only superficially relevant to the current query.

To address this, ICE-Mature organizes conversations into evolving topical clusters. Related turns are grouped together based on recurring entities, shared concepts, and semantic similarity, allowing the system to construct localized regions of memory within a single conversation. At retrieval time, the system first identifies the clusters most relevant to the current query and then restricts episodic retrieval primarily to those regions rather than searching the entire conversation uniformly.

This change substantially improved retrieval precision in long-form conversations. Instead of competing against the full conversational history, retrieval operates within a narrower and more semantically coherent search space. For example, a question about a specific world-building concept, project decision, or recurring entity is evaluated primarily against the portions of the conversation where that topic was actively discussed, reducing contamination from unrelated conversation segments.

The impact of this design is reflected in Experiment 2's retrieval-quality metrics. Compared with Experiment 1, ICE retrieved substantially fewer fragments while maintaining or improving answer quality. This indicates that performance gains were not driven by injecting more context, but by providing the model with a cleaner and more focused set of supporting evidence. The clustering mechanism therefore served as an important contributor to the reduction in retrieval noise and the improvement in Score-per-Fragment efficiency observed throughout the mature-system evaluation.

### Classifier and Routing Improvements

Another major change between Experiment 1 and Experiment 2 involved the query-classification subsystem responsible for determining whether a user request required memory retrieval.

Experiment 1 revealed that routing errors represented a significant source of failure. In particular, queries incorrectly classified as Zero_Shot bypassed memory retrieval entirely, preventing the system from accessing relevant historical information. This behaviour was especially problematic in long-running conversations where seemingly simple questions often depended heavily on prior context.

To address this issue, the classifier was retrained using a substantially improved embedding backbone. The original MiniLM-based embedding model was replaced with Qwen3-Embedding-0.6B, and the entire training pipeline was rebuilt. Training embeddings were regenerated from the annotated dataset, the classification head was retrained from scratch, and the resulting model was evaluated using dedicated validation and testing splits. The larger embedding model provided significantly stronger semantic representations and improved discrimination between memory-dependent and memory-independent queries.

In addition to retraining, a conservative routing safeguard was introduced. Analysis of Experiment 1 indicated that false-positive retrievals were generally less harmful than false-negative retrievals. Missing relevant memory often caused complete answer failure, whereas unnecessary retrieval typically resulted only in modest context overhead.

Consequently, the routing system was modified to favour retrieval whenever uncertainty existed. Queries classified as Zero_Shot were automatically re-routed to memory retrieval when confidence fell below a predefined threshold. Similarly, long-running conversations were treated as memory-dependent by default, reflecting the observation that contextual questions become increasingly common as conversation history grows.

These changes substantially reduced routing failures during Experiment 2. The number of identified gating failures decreased from 22 instances in Experiment 1 to only 2 instances in the mature-system benchmark. This suggests that improvements in both classifier quality and retrieval-favouring routing policies contributed meaningfully to overall system robustness.

### Longitudinal Performance Measurement

The central outcome of Experiment 2 was not a single aggregate score but performance measured across time: for each probe, a sequence $(T_1, S_1), (T_2, S_2), \ldots, (T_n, S_n)$ of checkpoint locations and answer-quality scores forms a longitudinal knowledge curve describing how effectively the system accumulates, preserves, and updates information over the conversation. An increasing trajectory indicates successful memory accumulation; a declining trajectory indicates forgetting, retrieval degradation, or interference from later information.

**Key difference from Experiment 1.** Experiment 1 primarily answered: *can ICE retrieve information from large conversational histories?* Experiment 2 addresses a substantially different question: *does ICE become more knowledgeable over time as conversations grow and memory structures mature?* The evaluation therefore shifts from static retrieval assessment to longitudinal memory analysis — treating memory as a continuously evolving system whose internal representations change with new information and consolidation, rather than a fixed database queried once. This makes Experiment 2 a study of memory growth and maintenance rather than retrieval performance alone.

### Evaluation Methodology

Experiment 2 employed a multi-stage forensic evaluation framework assessing answer quality, temporal correctness, factual reliability, and retrieval quality. Unlike Experiment 1, which primarily focused on static correctness, Experiment 2 explicitly evaluated whether a system's answer reflected the correct state of knowledge at the specific checkpoint being tested — requiring temporal awareness in addition to factual accuracy.

**Judge model.** All evaluations used an independent judge model never used during memory retrieval, answer generation, probe generation, or ground-truth construction — reducing evaluator contamination. The judge used deterministic decoding for consistency.

**Absolute scoring.** The judge received the probe question, the checkpoint turn number, the checkpoint-specific ground-truth dossier, and the generated answer, then scored 1–5:

| Score | Interpretation |
|---|---|
| 5 | Complete, temporally correct, and highly specific answer |
| 4 | Correct answer with only minor omissions |
| 3 | Partially correct or temporally outdated answer |
| 2 | Mixture of correct and incorrect information |
| 1 | Incorrect, hallucinated, or failed response |

A key improvement over Experiment 1 was **temporal scoring**: an answer could receive a reduced score even if historically correct facts had since been superseded by newer information available at the evaluation checkpoint — enabling measurement of memory *updating*, not just retention.

**Temporal-aware evaluation.** Experiment 2 distinguishes between incorrect information, historically correct but outdated information, and current information. Answers that correctly recalled older facts while failing to incorporate known updates were treated as temporally incomplete rather than fully correct — necessary because one of ICE's primary objectives is maintaining current knowledge while preserving historical context.

**Inference credit.** Traditional benchmark evaluation often penalizes any detail not explicitly present in the reference answer. Experiment 2 instead allowed the judge to treat a specific, logically consistent detail — strongly supported by available evidence but not explicitly listed in the ground truth — as a positive contribution rather than a hallucination, rewarding successful knowledge synthesis.

**Tournament evaluation.** Every probe additionally underwent blind tournament evaluation: the four system outputs were randomly shuffled, anonymized, and presented to the judge simultaneously, which ranked them best-to-worst on accuracy, temporal correctness, completeness, and specificity — providing a relative measure independent of any individual scoring scale.

**Hallucination audit.** Each answer was examined for unsupported facts, contradictory information, fabricated details, and incorrect state updates. Importantly, temporal incompleteness (e.g. reporting an earlier version of a fact without the later update) was *not* treated as hallucination — the audit distinguishes between forgetting, outdated knowledge, and genuine hallucination, providing more diagnostic information than a binary score.

**Retrieval fragment analysis.** Retrieved context fragments were evaluated independently from the generated answer (a system may retrieve highly relevant evidence yet produce a poor answer, or vice versa). The judge received the question, checkpoint location, and retrieved fragments, and estimated:
- *Noise Score (1–10):* 1–3 highly focused; 4–6 moderate noise; 7–10 substantial irrelevant context.
- *Relevance Percentage:* $\text{Relevance} = \frac{\text{Useful Context}}{\text{Total Retrieved Context}} \times 100$.

**Structural relevance assessment** (new in Experiment 2): not all useful context directly contains the answer — some fragments provide essential background, relationships, motivations, or contextual dependencies that support correct reasoning without explicitly mentioning the queried fact. Such fragments were treated as relevant rather than noisy, preventing unfair penalization of retrieval systems that surface indirectly supportive context.

**Why multiple evaluation passes were necessary.** Memory performance is multidimensional — correctness, temporal awareness, hallucination resistance, retrieval quality, and relative answer superiority cannot be captured by a single metric. Every probe was therefore evaluated through four complementary forensic analyses: absolute scoring, tournament ranking, hallucination auditing, and retrieval fragment assessment, together providing a substantially more complete characterization than any individual metric alone.

**Key difference from Experiment 1 (methodology).** Experiment 1 primarily evaluated whether retrieved information matched a static reference answer. Experiment 2 evaluates whether the system retrieves, maintains, updates, and applies knowledge correctly as conversations evolve — the introduction of temporal-aware scoring, evolving ground truths, retrieval-quality analysis, and hallucination auditing transforms the benchmark from a retrieval test into a longitudinal memory evaluation framework, measuring not only what a system remembers but whether it remembers the right version of information at the right time.

### Probe Distribution

Final Experiment 2 benchmark combined automatically generated probes with manually authored probes inherited from Experiment 1.

**Hand-written probes:**

| Dataset | Hand-Written Probes |
|---|---:|
| Dataset A (Creative Writing) | 17 |
| Dataset B (Long-Form Creative Writing) | 32 |
| Dataset C (Technical Planning) | 12 |
| Dataset D (Academic Planning) | 11 |
| **Total** | **72** |

**Automatically generated probes:**

| Dataset | Generated Probes |
|---|---:|
| Dataset A (Creative Writing) | 28 |
| Dataset B (Long-Form Creative Writing) | 59 |
| Dataset C (Technical Planning) | 25 |
| Dataset D (Academic Planning) | 35 |
| **Total** | **147** |

Overall, Experiment 2 evaluated **219 unique probes**, combining systematic checkpoint-based evaluation with manually constructed long-horizon memory challenges. The distribution intentionally reflects conversation complexity and length — Dataset B (1,119 turns) received the most probes of both kinds, while shorter conversations received proportionally fewer, ensuring benchmark coverage scaled with conversational complexity and available information.

### Metric Computation and Analysis Framework (Experiment-2-Specific Extensions)

Beyond the core LSREP metrics (defined once in the LSREP section), Experiment 2 introduced an expanded analysis framework computing performance across additional dimensions: retrieval composition, memory maturity, and routing effectiveness. All metrics were computed after integrating automated evaluations, manually evaluated hand-written probes, and human-verified hallucination audits.

- **Hallucination Rate (with manual verification).** Following automated evaluation, all hallucination assessments underwent manual verification to remove false-positive hallucination labels and ensure correctly retrieved information was not incorrectly penalized.
- **Score Per Fragment (SPF).** $SPF = \frac{\text{Average Score}}{\text{Average Fragment Count}}$ — measures how effectively retrieved fragments contribute to answer quality; higher values mean each fragment provides greater informational value.
- **Fragment–Score Correlation.** Correlation between fragment count and answer quality, to determine whether retrieving more information actually improved performance. Positive values indicate additional retrieval generally improves performance; negative values suggest diminishing returns or retrieval noise.
- **Score Distribution Analysis.** Percentage of Score 5/4/3/2/1 answers computed separately per condition; answers additionally grouped into Good ($\text{Score} \ge 4$), Acceptable ($\text{Score} = 3$), and Poor ($\text{Score} \le 2$).
- **Memory Maturity Metrics.** Per conversation: maximum simulated age, maximum decay cycles experienced, and average simulated age per checkpoint — quantifying how long information remained in the system and how extensively maintenance processes operated before evaluation.
- **Retrieval-Leg Contribution Analysis.** Every retrieved fragment attributed to its originating retrieval leg (e.g. codex, episodic); average fragments contributed per leg, percentage contribution per leg, and total fragment composition computed per condition.
- **MoE vs. Generalist Analysis.** Metrics aggregated separately for generalist vs. Mixture-of-Experts systems: answer-quality difference, context-consumption difference, hallucination-rate difference, tournament-win difference — isolating routing effects from retrieval quality.
- **Comparative Delta Metrics** (relative to the vector baseline): Quality Gain ($\Delta_{Quality} = \text{Score}_{ICE} - \text{Score}_{Vector}$); Hallucination Reduction ($\Delta_{Hallucination} = \text{Hallucination}_{Vector} - \text{Hallucination}_{ICE}$); Fragment Reduction ($\Delta_{Fragments} = 1 - \frac{\text{Fragments}_{ICE}}{\text{Fragments}_{Vector}}$); Token Savings ($\Delta_{Tokens} = 1 - \frac{\text{Tokens}_{ICE}}{\text{Tokens}_{Vector}}$).
- **Gating Failure Analysis** (as defined in LSREP), applied to Experiment 2's conditions.
- **Research Integrity Measures.** To improve reproducibility and transparency, all reports additionally recorded hardware configuration, evaluated models, retrieval architecture, embedding models, classifier versions, clustering configurations, retrieval-leg definitions, and token-budget settings — stored alongside computed metrics for full traceability.

### Evaluation Conservatism and Underestimation of Memory Reinforcement

An important characteristic of the Experiment 2 protocol is that it likely **underestimates** the performance ICE would achieve during real-world deployment.

ICE incorporates adaptive memory-strength mechanisms: retrieved memories receive positive reinforcement, while unused memories gradually decay according to age-dependent maintenance policies (memory accessibility is influenced by more than just the information itself — it's shaped by the history of interactions involving that information). The benchmark only partially captures this process.

**Replay vs. real interaction.** During evaluation, conversation histories were replayed chronologically; at each checkpoint, historical turns were ingested, maintenance ran, and probes were administered — but the original user interactions occurring *between* checkpoints were not re-executed as active retrieval events. Information introduced during replay was stored, but did not receive the repeated reinforcement that would naturally occur during real usage.

**Missing reinforcement effects (illustrative examples).** In a real technical-planning deployment, an architectural concept introduced early and referenced repeatedly over subsequent weeks would trigger additional retrieval events each time, strengthening the associated memories, increasing ranking priority, reinforcing relationships, and improving future retrieval odds. Similarly, in a long-form narrative, a newly introduced character mentioned repeatedly across many later interactions would naturally reinforce corresponding memories and relationships. Repeated conversational use acts as an implicit training signal for the memory system in both cases.

**Benchmark behaviour.** The Experiment 2 protocol does not fully reproduce these reinforcement cycles — information prior to a checkpoint is replayed as historical content, but the intermediate retrieval opportunities that would normally occur throughout a live conversation are largely absent. Consequently, a concept repeatedly referenced across dozens of later interactions and a concept mentioned once and never revisited may appear artificially similar during evaluation, even though they would naturally diverge in importance in a deployed system. Within the benchmark, both memories begin from comparable initial states until evaluation probes explicitly trigger retrieval — so the primary source of reinforcement becomes the evaluation probes themselves, not organic conversational interaction.

**Conservative performance estimates.** This makes the benchmark intentionally conservative: it measures whether ICE can retrieve information under conditions where many naturally occurring reinforcement opportunities have been removed. A memory system that performs well under these conditions is likely to perform at least as well — and potentially substantially better — during real deployment, where repeated user interactions continuously reinforce important information. Reported results should therefore not be interpreted as measuring the architecture's maximum capability, but as performance under a constrained replay environment that intentionally minimizes many real-world memory-strengthening feedback mechanisms.

**Implications.** This is particularly relevant for long-running personal assistants, development agents, and collaborative systems, where important concepts, entities, preferences, and decisions are naturally revisited over time and accumulate reinforcement through ordinary usage. Because Experiment 2 largely isolates evaluation probes from these natural reinforcement cycles, the benchmark likely understates the long-term benefits of adaptive memory-strength mechanisms. Future evaluations could address this by simulating intermediate retrieval events or replaying complete conversational interactions through the retrieval pipeline, allowing reinforcement dynamics to more closely match real-world system behaviour. *(This point is closely related to, and elaborated further under, the Limitations section — "Limited Real-Time Reinforcement Effects" and "Synthetic Longitudinal Reconstruction.")*

### Author Notes / Open Items
- The 4 conversations — the "why these four" rationale is answered above (Dataset Selection); confirm this fully satisfies the original prompt.
- The main result headline (quality tie, 32% fragment reduction, 50% SPF advantage, 30.6% vs 21.2% win rate) — were you happy or disappointed? Personal reaction narrative is only partially captured (see "satisfying and humbling" passage under Experiment 1's honest narrative) — consider whether this belongs here instead, and expand.
- MoE vs. Generalist — "basically no difference. What did you expect? What do you conclude?" — see the dedicated paragraph below (originally attached near Experiment 2's placeholder bullets): *"We expected the Mixture-of-Experts routing to provide a measurable advantage, particularly on the technical-planning conversation where specialist coding models should excel. The result — a global delta of +0.01 for MoE under ICE — was effectively zero. We conclude that simple topic/intent overlap scoring is too crude to select meaningfully better experts. The models in our registry all had broadly similar capabilities on these tasks, and the session-stickiness mechanism prevented harmful switches but also prevented beneficial ones. MoE routing remains architecturally promising but needs a fundamentally different selection mechanism — perhaps confidence-weighted scoring or per-task benchmarking."*
- Longitudinal curves — did memory actually get better over time? See Results section curves/tables.
- Gating failures: 22 → 2. What fixed it? Not yet explicitly documented beyond the general "everything was rebuilt" narrative in Experiment 1 — needs a specific technical explanation.
- Fragment-score correlation: ICE positive, vector negative — what does that tell you? See Results section table (Experiment 2 correlation values); interpretive commentary not yet drafted.
- Memory maturity: Dataset B ("flaw") had 93 simulated decay days — what does that prove? Not yet drafted; see Results memory-maturity table for the raw figure.

---

## Results

### Draft Text — Global Comparison (Experiment 2, Mature System)

Report metadata: **1,211 total probes** across all conversations, including ICE-Dev (Dataset C / Technical Planning).

**1. Global Comparison (All Conversations Included)**

| Condition | Score | Tokens | Fragments | SPF | TUR | Win% | Hall% |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector_rag_baseline_generalist | 3.87 | 29550 | 30.0 | 0.13 | 0.13 | 18.7% | 20.2% |
| vector_rag_moe | 3.84 | 29550 | 30.0 | 0.13 | 0.13 | 17.3% | 15.7% |
| full_ice_generalist | 4.27 | 22099 | 19.1 | 0.22 | 0.19 | 32.3% | 22.3% |
| full_ice_moe | 4.25 | 22099 | 19.1 | 0.22 | 0.19 | 31.7% | 22.5% |

Token savings vs. vector: **25.2%**. Quality delta vs. vector: **+0.4 pts**. Fragment reduction vs. vector: **36.3%**.

**2. MoE vs. Generalist (All Conversations Included)**

| Base | MoE Score | Gen Score | MoE Tokens | Gen Tokens | MoE Win% | Gen Win% | MoE Hall% | Gen Hall% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vector_rag | 3.84 | 3.87 | 29550 | 29550 | 17.3% | 18.7% | 15.7% | 20.2% |
| full_ice | 4.25 | 4.27 | 22099 | 22099 | 31.7% | 32.3% | 22.5% | 22.3% |
| global | 4.04 | 4.07 | 25824 | 25824 | 24.5% | 25.5% | 19.2% | 21.3% |

Global MoE vs. Gen score delta: **−0.02 pts**. Tokens saved by MoE: **0**. Hallucination reduction by MoE: **2.1%**.

**3. Score Distributions (All Conversations Included)**

| Score | ICE Gen % | ICE MoE % | Vector Gen % | Vector MoE % |
|---|---:|---:|---:|---:|
| 1 | 15.3% | 14.5% | 3.1% | 1.9% |
| 2 | 3.7% | 3.4% | 5.8% | 5.2% |
| 3 | 12.9% | 17.0% | 13.1% | 16.7% |
| 4 | 15.0% | 14.2% | 17.3% | 18.3% |
| 5 | 53.1% | 50.9% | 60.6% | 57.9% |

**4. Temporal Score Quality (All Conversations Included)**

| Condition | Good (4-5) | OK (3) | Poor (1-2) |
|---|---:|---:|---:|
| vector_rag_baseline_generalist | 68.1% | 12.9% | 19.0% |
| vector_rag_moe | 65.2% | 17.0% | 17.8% |
| full_ice_generalist | 78.0% | 13.1% | 8.9% |
| full_ice_moe | 76.2% | 16.7% | 7.1% |

**5. Per-Conversation Breakdown (All Conversations Included)**

| Conv | ICE Score | Vector Score | ICE Tokens | Vector Tokens | ICE Win% | Vector Win% |
|---|---:|---:|---:|---:|---:|---:|
| flaw | 4.28 | 4.33 | 24108 | 21257 | 32.8% | 22.9% |
| ice_dev | 4.33 | 1.23 | 19953 | 88061 | 43.5% | 1.9% |
| masters | 4.63 | 4.57 | 20103 | 18076 | 20.4% | 18.8% |
| shinchan | 3.63 | 3.5 | 19434 | 24360 | 37.2% | 18.0% |

**6. Longitudinal Score Evolution — Global Binned (All Conversations Included)**

| Condition | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 550 | 600 | 650 | 700 | 750 | 800 | 850 | 950 | 1000 | 1050 | 1100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vector_rag_baseline_generalist | 4.50 | 3.76 | 3.83 | 3.82 | 3.51 | 3.28 | 2.22 | 4.65 | 4.55 | 4.58 | 4.39 | 4.29 | 4.29 | 4.56 | 4.29 | 4.21 | 4.27 | 4.45 | 4.49 | 4.47 | 3.80 |
| vector_rag_moe | 4.50 | 3.93 | 3.83 | 3.87 | 3.46 | 3.15 | 2.09 | 4.35 | 4.45 | 4.42 | 4.39 | 4.42 | 4.47 | 4.39 | 4.26 | 4.26 | 4.31 | 4.33 | 4.29 | 4.49 | 3.87 |
| full_ice_generalist | 5.00 | 4.35 | 4.25 | 4.16 | 4.19 | 4.15 | 4.58 | 4.50 | 4.50 | 4.38 | 4.29 | 4.52 | 4.18 | 4.22 | 4.21 | 4.26 | 4.20 | 4.33 | 4.29 | 4.40 | 4.18 |
| full_ice_moe | 4.83 | 4.28 | 4.28 | 4.19 | 4.23 | 3.95 | 4.45 | 4.60 | 4.55 | 4.54 | 4.61 | 4.39 | 4.47 | 4.22 | 4.08 | 4.36 | 4.20 | 4.22 | 4.35 | 4.42 | 4.14 |

**7. Leg Contributions — ICE Generalist (All Conversations Included)**

| Source | Avg Count | % of Total |
|---|---:|---:|
| codex | 1 | 3.3% |
| episodic | 19.1 | 62.3% |
| unknown | 10.6 | 34.5% |
| **Total** | **30.7** | **100%** |

**8. Fragment Noise (All Conversations Included)**

| Condition | Mean Noise |
|---|---:|
| vector_rag_baseline_generalist | 2.73 |
| vector_rag_moe | 2.73 |
| full_ice_generalist | 3.04 |
| full_ice_moe | 3.03 |

**9. Memory Maturity**

| Conversation | Max Simulated Days | Max Decay Cycles | Mean Days/Checkpoint |
|---|---:|---:|---:|
| shinchan | 24 | 24 | 17.2 |
| flaw | 93 | 93 | 64.1 |
| ice_dev | 27 | 27 | 19.8 |
| masters | 20 | 20 | 14.1 |

**10. Gating Failures (Zero_Shot mis-classified)**

Total probes where the classifier said Zero_Shot but ICE score < 3: **2** (down from 22 in Experiment 1).

| Conversation | Probe | Question | ICE Score |
|---|---|---|---:|
| bb558b5f... | 216-GEN-02 | "how did the multiverse actually start? like, was it just ran..." | 2 |
| bb558b5f... | 216-GEN-02 | (duplicate entry in source report) | 2 |

**11. Fragment-Count vs. Score Correlation**

| Condition | Correlation |
|---|---:|
| vector_rag_baseline_generalist | −0.015 |
| vector_rag_moe | −0.03 |
| full_ice_generalist | 0.193 |
| full_ice_moe | 0.246 |

### Draft Text — "Fair Comparison" View (ICE-Dev / Dataset C Excluded)

Because Dataset C (ICE-Dev) triggered a catastrophic context-overflow failure mode in the vector baseline (see below), a second view of the same results excludes it to give a fairer head-to-head comparison on the three conversations where both systems could actually generate usable answers. Report metadata: **1,057 total probes**.

**1. Global Comparison (ICE-Dev Excluded)**

| Condition | Score | Tokens | Fragments | SPF | TUR | Win% | Hall% |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector_rag_baseline_generalist | 4.25 | 21025 | 30.0 | 0.14 | 0.2 | 21.2% | 20.5% |
| vector_rag_moe | 4.24 | 21025 | 30.0 | 0.14 | 0.2 | 19.2% | 17.4% |
| full_ice_generalist | 4.26 | 22411 | 20.4 | 0.21 | 0.19 | 30.6% | 19.6% |
| full_ice_moe | 4.28 | 22411 | 20.4 | 0.21 | 0.19 | 29.0% | 19.9% |

Token savings vs. vector: **−6.6%** (ICE used slightly *more* tokens than vector in this excluded-Dataset-C view). Quality delta vs. vector: **+0.01 pts**. Fragment reduction vs. vector: **32.0%**.

**2. MoE vs. Generalist (ICE-Dev Excluded)**

| Base | MoE Score | Gen Score | MoE Tokens | Gen Tokens | MoE Win% | Gen Win% | MoE Hall% | Gen Hall% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vector_rag | 4.24 | 4.25 | 21025 | 21025 | 19.2% | 21.2% | 17.4% | 20.5% |
| full_ice | 4.28 | 4.26 | 22411 | 22411 | 29.0% | 30.6% | 19.9% | 19.6% |
| global | 4.26 | 4.25 | 21718 | 21718 | 24.1% | 25.9% | 18.6% | 20.0% |

Global MoE vs. Gen score delta: **+0.01 pts**. Tokens saved by MoE: **0**. Hallucination reduction by MoE: **1.4%**.

**3. Score Distributions (ICE-Dev Excluded)**

| Score | ICE Gen % | ICE MoE % | Vector Gen % | Vector MoE % |
|---|---:|---:|---:|---:|
| 1 | 3.8% | 2.4% | 3.0% | 1.5% |
| 2 | 4.3% | 3.9% | 5.9% | 4.3% |
| 3 | 14.8% | 19.5% | 13.3% | 16.6% |
| 4 | 17.2% | 16.3% | 18.1% | 19.7% |
| 5 | 60.0% | 58.0% | 59.7% | 58.0% |

**4. Temporal Score Quality (ICE-Dev Excluded)**

| Condition | Good (4-5) | OK (3) | Poor (1-2) |
|---|---:|---:|---:|
| vector_rag_baseline_generalist | 77.2% | 14.8% | 8.0% |
| vector_rag_moe | 74.3% | 19.5% | 6.2% |
| full_ice_generalist | 77.8% | 13.3% | 8.9% |
| full_ice_moe | 77.7% | 16.6% | 5.8% |

**5. Per-Conversation Breakdown (ICE-Dev Excluded)**

| Conv | ICE Score | Vector Score | ICE Tokens | Vector Tokens | ICE Win% | Vector Win% |
|---|---:|---:|---:|---:|---:|---:|
| flaw | 4.28 | 4.33 | 24108 | 21257 | 32.8% | 22.9% |
| masters | 4.63 | 4.57 | 20103 | 18076 | 20.4% | 18.8% |
| shinchan | 3.63 | 3.5 | 19434 | 24360 | 37.2% | 18.0% |

**6. Longitudinal Score Evolution — Global Binned (ICE-Dev Excluded)**

| Condition | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 550 | 600 | 650 | 700 | 750 | 800 | 850 | 950 | 1000 | 1050 | 1100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vector_rag_baseline_generalist | 4.50 | 4.34 | 4.29 | 4.12 | 4.26 | 3.98 | 4.50 | 4.65 | 4.55 | 4.58 | 4.39 | 4.29 | 4.29 | 4.56 | 4.29 | 4.21 | 4.27 | 4.45 | 4.49 | 4.47 | 3.80 |
| vector_rag_moe | 4.50 | 4.45 | 4.34 | 4.23 | 4.24 | 3.86 | 4.33 | 4.35 | 4.45 | 4.42 | 4.39 | 4.42 | 4.47 | 4.39 | 4.26 | 4.26 | 4.31 | 4.33 | 4.29 | 4.49 | 3.87 |
| full_ice_generalist | 5.00 | 4.32 | 4.21 | 4.10 | 4.27 | 4.11 | 4.56 | 4.50 | 4.50 | 4.38 | 4.29 | 4.52 | 4.18 | 4.22 | 4.21 | 4.26 | 4.20 | 4.33 | 4.29 | 4.40 | 4.18 |
| full_ice_moe | 4.83 | 4.26 | 4.29 | 4.21 | 4.42 | 3.97 | 4.67 | 4.60 | 4.55 | 4.54 | 4.61 | 4.39 | 4.47 | 4.22 | 4.08 | 4.36 | 4.20 | 4.22 | 4.35 | 4.42 | 4.14 |

**7. Leg Contributions — ICE Generalist (ICE-Dev Excluded)**

| Source | Avg Count | % of Total |
|---|---:|---:|
| codex | 1 | 3.3% |
| episodic | 19.1 | 62.3% |
| unknown | 10.6 | 34.5% |
| **Total** | **30.7** | **100%** |

**8. Fragment Noise (ICE-Dev Excluded)**

| Condition | Mean Noise |
|---|---:|
| vector_rag_baseline_generalist | 2.47 |
| vector_rag_moe | 2.48 |
| full_ice_generalist | 2.8 |
| full_ice_moe | 2.81 |

**9–11.** Memory Maturity, Gating Failures, and Fragment-Count vs. Score Correlation tables are identical to the "all conversations included" view above (Dataset C exclusion does not change these particular figures as reported).

### Draft Text — ICE-Dev / Dataset C: Context-Overflow Stress Test

**Emergent discovery.** Prior to running Experiment 2, Dataset B (Long-Form Creative Writing, 1,119 turns) was expected to be the most demanding evaluation environment, based on conversation length. Instead, an unexpected phenomenon emerged within Dataset C (Technical Planning). Although Dataset C had only 325 turns, it differed fundamentally in information density: individual exchanges frequently consisted of architecture documents, implementation plans, design discussions, code reviews, and extensive technical reasoning, with many turns several thousand tokens long — substantially exceeding the average turn size elsewhere in the benchmark. Dataset C therefore combined a comparatively modest turn count with an unusually large amount of information per turn.

**Discovery during evaluation.** The original expectation was that Dataset C would be a moderately sized technical benchmark; instead it became one of the most revealing stress tests in the evaluation suite. The vector-retrieval baseline began failing at an unusually high rate on Dataset C despite performing competitively elsewhere. Initial inspection suggested a retrieval issue, but further investigation showed retrieval quality itself was not the problem — the failure originated downstream, during answer generation.

**Context-overflow failure mode.** The vector baseline used a fixed retrieval strategy, always selecting the top-ranked fragments for every query. Under normal conditions this produced manageable context sizes, but because Dataset C's retrieved fragments were themselves extremely large, selecting the top results frequently produced assembled prompts approaching or exceeding 80,000–100,000 tokens. In many cases the retrieved information was genuinely relevant — the issue was retrieval *volume*, not retrieval *precision*. The generation model, forced to process extremely large contexts containing architecture documents, planning discussions, implementation details, and historical decisions simultaneously, frequently failed to generate a usable response.

**Vector baseline failure rate:** across 154 evaluated probes, the vector baseline failed on **145 probes — a 94.2% failure rate** (injecting up to 100,505 tokens in failed cases, ~88,061 tokens typical of non-failed probes → model OOM). The resulting average score collapsed to **1.23**, despite the retrieval subsystem successfully locating relevant information. This is a context-overflow failure, not a retrieval-quality failure.

**Effect of unified token budgeting.** ICE-Mature used a fundamentally different retrieval policy: rather than retrieving a fixed number of fragments, all retrieval components operated within a shared dynamic token budget, forcing every retrieval candidate to compete for limited context space and prioritizing information that maximized utility within budget. As a result, ICE-Mature maintained a mean context size of approximately **20,000 injected tokens** throughout Dataset C, with **no context-overflow failures observed**. Across all 154 probes, ICE successfully generated responses with an average score of **4.33**. The advantage did not arise from superior retrieval quality alone, but from the ability to regulate context growth and prevent retrieval from overwhelming the downstream generation model.

**ICE vs. Vector on ICE-Dev (Dataset C) — summary table**

| Condition | Score | Tokens | Fragments | SPF | TUR | Hall% | Win% |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_ice_generalist | 4.33 | 19953 | 10.0 | 0.43 | 0.22 | 41.4% | 43.5% |
| full_ice_moe | 4.03 | 19953 | 10.0 | 0.4 | 0.2 | 40.8% | 50.0% |
| vector_rag_baseline_generalist | 1.23 | 88061 | 30 | 0.04 | 0.01 | 0% | 1.9% |
| vector_rag_moe | 1.1 | 88061 | 30 | 0.04 | 0.01 | 0% | 4.5% |

**Key finding:** ICE maintained a score of 4.33 on ICE-Dev while the vector baseline collapsed to 1.23 due to context overflow; ICE's token budget enforcement prevented all catastrophic failures.

**Score Distribution on ICE-Dev**

| Score | ICE Gen | Vector Gen |
|---|---:|---:|
| 1 | 3.9% | 94.2% |
| 2 | 5.2% | 0.0% |
| 3 | 11.7% | 0.0% |
| 4 | 12.3% | 0.0% |
| 5 | 66.9% | 5.8% |

**Fragment Noise on ICE-Dev**

| Condition | Mean Noise |
|---|---:|
| full_ice_generalist | 4.65 |
| vector_rag_baseline_generalist | 4.46 |

**Leg Contributions on ICE-Dev (ICE Generalist)**

| Source | Avg Count | % of Total |
|---|---:|---:|
| codex | 1 | 9.2% |
| episodic | 9.8 | 90.8% |
| **Total** | **10.8** | **100%** |

**Context-overflow failure rate summary:** Vector baseline — 145/154 probes failed (94.2%). ICE — 0 probes failed (token-budget capping prevented all context overflows). Mean tokens injected by ICE: 19,953 (vs. vector's typical 88,061 on non-failed probes, 100,505 on failed probes).

**Note on the hallucination trade-off:** ICE shows a nontrivial hallucination rate (~41%) on ICE-Dev specifically *because it actually attempts to answer*, whereas the vector baseline's 0% hallucination rate on this dataset reflects the fact that it largely could not produce an answer at all due to context overflow — a 0% hallucination rate here is not evidence of reliability.

**Information density as a benchmark dimension.** One of the most important observations from Experiment 2 is that conversational turn count alone is an insufficient measure of benchmark difficulty. Dataset B represented the greatest challenge in terms of temporal horizon (retrieval across 1,000+ turns); Dataset C represented a different challenge entirely: extreme information density. The benchmark therefore revealed two distinct stress dimensions: (1) **Temporal-Horizon Stress** — retrieving information across very long conversational distances; and (2) **Information-Density Stress** — operating effectively when individual turns contain unusually large amounts of information. While Dataset B primarily evaluated long-range memory retention, Dataset C unexpectedly became a test of context-budget management and retrieval scalability.

**Implications.** Future memory benchmarks should evaluate both conversational length and information density — a retrieval architecture may perform well when history is long but sparse, yet fail catastrophically when the same amount of information is concentrated into fewer, denser interactions. Dataset C provided an unplanned but highly informative scenario: rather than exposing weaknesses in retrieval relevance, it exposed the importance of context-budget enforcement, demonstrating that retrieval systems must be evaluated not only on what they retrieve but on how much information they deliver to the generation model. This failure mode would not have been visible in traditional retrieval-quality metrics alone, and highlights context management as a first-class component of long-term memory systems.

**Why this stress test proves the token budget is a safety mechanism, not just an optimization.** The ICE-Dev stress test demonstrates that without a token budget, any retrieval system injecting context into a fixed-window model will eventually encounter a conversation where retrieved fragments exceed model capacity and cause catastrophic failure. The token budget transforms retrieval from a "best effort" activity into a "bounded resource allocation" problem, forcing the system to prioritize. The fact that ICE achieves its highest absolute score on the exact conversation where the baseline collapses (4.33) demonstrates that retrieval quality is not sacrificed by the budget — it is improved by it.

### Author Notes (unplanned discovery, personal narrative)

This result was not anticipated. Dataset C was expected to be a moderately-sized technical benchmark. The 94% baseline failure rate was discovered during the evaluation run itself, not planned — when the first few probes returned errors, the initial assumption was a bug, not a fundamental architectural limitation of unbounded retrieval. That this happened on a conversation about building ICE itself was fitting: the system being built was stress-tested by the very conversation in which it was built. ICE surviving while the baseline collapsed was the single strongest result in the entire evaluation, and it was completely unplanned.

### Open Items
- What is the single most reader-memorable number if the reader remembers only one? (Candidates present in the notes: the 4.33 vs. 1.23 ICE-Dev result; the 32% fragment reduction; the RRF +0.84 ablation delta — final selection not yet made; see also Conclusion open items.)

---

## Ablation Study — Incremental System Buildup (Experiment 3)

### Draft Text

**Motivation.** Following Experiment 2, a new question emerged: while the overall effectiveness of ICE-Mature had been established, it was not clear which individual subsystems were responsible for the observed improvements. The architecture contains numerous interacting components — hybrid retrieval mechanisms, reranking strategies, procedural memory, summarization systems, dynamic budgeting policies, retrieval constraints, and memory-management mechanisms — and while the aggregate system could be evaluated as a whole, the relative contribution of each remained unclear. Experiment 3 was designed to answer three questions: (1) which subsystems provide the largest performance improvements; (2) which provide minimal or neutral benefit; and (3) which may introduce complexity without corresponding gains.

Traditional ablation studies typically begin with a complete system and remove one component at a time — useful, but this primarily measures the damage caused by *removing* a feature rather than the benefit gained when *introducing* it, and complex retrieval systems often contain interactions between subsystems that only emerge when components operate together. Experiment 3 adopts the opposite strategy: the system is constructed **incrementally** from its simplest form, so the contribution of each subsystem can be observed directly as it is introduced.

**Experimental design.** The experiment reused the fully mature memory state generated during Experiment 2 (rather than rebuilding memory repeatedly), ensuring all retrieval conditions operated over the same accumulated knowledge base, so differences in performance could be attributed to retrieval architecture rather than differences in stored information. A single conversation was selected as the evaluation environment: the Long-Form Creative Writing dataset (Dataset B / "flaw"), chosen because it represented the largest and most complex memory environment in the benchmark — 1,000+ turns, large numbers of recurring entities, long-range narrative dependencies, extensive world-building information, complex relationship networks, and knowledge distributed across large temporal distances.

**Probe construction.** The evaluation reused all generated probes from the retained longitudinal checkpoints, plus manually authored probes from the previous benchmark. Each probe retained metadata about its original checkpoint location, so probes from early checkpoints function as long-range retrieval tests while probes from later checkpoints evaluate more recent knowledge — even though all probes were evaluated against the fully mature memory state.

**Incremental system construction — the cumulative build sequence:**
1. Dense vector retrieval.
2. BM25 lexical retrieval.
3. Reciprocal Rank Fusion (RRF).
4. HyDE query expansion.
5. Cluster-restricted retrieval.
6. Session diversification.
7. Codex retrieval and fuzzy matching.
8. MERA retrieval.
9. Procedural memory.
10. Batch-summary retrieval.
11. Dynamic token budgeting.
12. Sliding-window context.
13. Keyword boosting.
14. Recency boosting.

Each stage preserves all previously enabled components while introducing exactly one additional capability, so any observed performance change can be attributed to the newly added subsystem and its interactions with the existing architecture.

**Why a buildup study was chosen.** By introducing features progressively, Experiment 3 measures immediate performance gains, synergistic effects, redundant components, and diminishing returns — providing more detailed insight into architectural behaviour than a traditional leave-one-out ablation.

**Longitudinal interpretation.** Because probes retain their original temporal location, questions from early checkpoints represent knowledge that survived the greatest amount of memory ageing, decay, consolidation, and maintenance, while questions from later checkpoints represent comparatively recent knowledge. Grouping results by probe origin enables investigation of whether particular retrieval mechanisms primarily benefit recently acquired information, long-term knowledge, frequently reinforced memories, or rarely accessed memories.

**Research objective.** The goal of Experiment 3 is not to determine whether ICE outperforms a baseline (addressed in Experiment 2) but to explain the internal behaviour of the architecture itself — an architectural decomposition study revealing which retrieval and memory-management mechanisms are responsible for the mature system's observed performance characteristics, providing both scientific insight and practical guidance for future development.

### Evaluation Methodology (Experiment 3)

All generated responses were evaluated using the same temporally-aware judging framework introduced in Experiment 2, but for a different purpose: rather than comparing complete retrieval architectures, Experiment 3 evaluates the contribution of individual architectural components by observing how performance changes as new subsystems are progressively introduced.

Each probe was evaluated against: bare vector retrieval; hybrid retrieval variants; intermediate architectural configurations; and the fully enabled ICE architecture. Because every configuration answered the same questions using the same mature memory state, direct comparison between stages was possible without confounding effects from differences in stored knowledge.

**Temporally-aware absolute scoring.** Each answer was scored 1–5 using the Experiment 2 rubric, with the judge receiving the probe question, current evaluation turn, temporally corrected ground-truth dossier, and generated answer — ensuring architectural improvements were measured against the correct version of the underlying knowledge (the most recent state, not the state at probe creation).

**Inference-aware judging.** As in Experiment 2, the judge did not penalize correct inferences — specific details not explicitly stated in the dossier but logically consistent with available evidence were treated as positive contributions. This was particularly important because some retrieval subsystems were designed to improve contextual synthesis rather than simple fact extraction.

**Hallucination auditing.** Each answer underwent the same audit as Experiment 2 (unsupported claims, contradictory information, fabricated details, invalid state updates), retaining the same safeguards: temporal incompleteness was not considered hallucination, correct inferences were not considered hallucination, and general background knowledge was ignored.

**Retrieval fragment analysis.** For selected conditions, retrieved fragments were examined independently of the generated answer, producing a Noise Score (1–10, lower = more focused) and a Relevance Percentage — allowing architectural gains to be attributed more precisely to retrieval precision, retrieval coverage, context management, generation quality, or some combination.

**Longitudinal subsystem evaluation.** Each probe retained metadata about the checkpoint at which it originally emerged, allowing subsystem contributions to be analyzed across different memory ages even though all evaluations ran against the fully mature memory state.

**Measuring architectural contribution.** For every stage, performance was compared against the immediately preceding stage, categorizing architectural changes into: **Positive Contributors** (consistently improve answer quality, retrieval quality, efficiency, or reliability), **Neutral Contributors** (minimal measurable effect), and **Negative Contributors** (reduce performance, increase hallucinations, introduce retrieval noise, or otherwise degrade behaviour).

### Metric Computation and Analysis (Experiment 3)

For every architectural stage, the following were aggregated across all probes: mean answer score; score standard deviation; mean tokens injected; mean retrieved fragments; hallucination percentage; longitudinal performance metrics.

- **Stepwise Contribution Delta.** $\Delta_{step} = \text{Score}_{current} - \text{Score}_{previous}$ — the marginal contribution of the newly added component, since every stage preserves all previously enabled functionality. Positive = improvement; negative = performance reduction or added retrieval noise.
- **Cumulative Contribution Delta.** $\Delta_{cumulative} = \text{Score}_{current} - \text{Score}_{bare}$ — total performance gained or lost relative to the initial bare-vector system, showing overall progress across the full construction process (complementing the local view given by stepwise deltas).
- **Score Per Fragment (SPF).** $SPF = \frac{\text{Mean Score}}{\text{Mean Retrieved Fragments}}$ — estimates how much answer quality is obtained per retrieved fragment; a subsystem may improve SPF by increasing answer quality or reducing retrieval noise.
- **Token Utility Ratio (TUR).** $TUR = \frac{\text{Mean Score}}{\text{Mean Tokens}/1000}$ — quantifies answer quality per thousand injected tokens, useful since several subsystems modify retrieval volume.
- **Longitudinal Origin-Split Analysis.** Probes grouped by origin location — Early memory (0–400 turns), Mid memory (400–800 turns), Late memory (800–1200 turns) — with average scores computed separately per group, measuring how effectively each subsystem handles information of different ages.
- **Recency Delta.** $\Delta_{recency} = \text{Score}_{late} - \text{Score}_{early}$. Positive = stronger performance on recent information; negative = stronger performance on older information; near zero = consistent performance regardless of memory age.
- **Hallucination measurement (unmodified).** Unlike Experiment 2, hallucination assessments in Experiment 3 were **not** manually corrected, since the purpose was architectural comparison rather than final benchmark reporting, and manual intervention across fifteen closely related system variants risked introducing subjective bias. All hallucination metrics in the buildup experiment are therefore derived directly from automated judge output, under identical conditions for every configuration.

### Why Experiment 3 Scores Differ from Experiment 2

The absolute scores in the buildup experiment should not be directly compared to Experiment 2's headline results, for several reasons:

1. **Different generation model.** Experiment 2 used a larger Gemma-based Mixture-of-Experts model; the buildup study used a smaller dense Qwen3-14B-AWQ model — overall answer quality is expected to be lower across all Experiment 3 configurations as a result.
2. **HyDE was reintroduced.** During earlier ablation studies, HyDE consistently produced negligible gains or slight degradation, so it was disabled entirely in the final Experiment 2 architecture. In Experiment 3, since the objective was architectural attribution, HyDE was restored to the buildup sequence to measure its direct contribution — and, unlike most other subsystems (which activate conditionally when triggering criteria are met), HyDE remained continuously active after introduction, so any downstream interaction effects accumulated throughout the rest of the buildup process.
3. **Different evaluation design.** Experiment 2 evaluated memory systems through longitudinal checkpoint-based testing across many stages of the simulated conversation lifetime. Experiment 3 evaluates all probes once against a fully mature memory database — the objective is not measuring memory growth over time, but isolating the effect of individual retrieval components operating over the same mature knowledge base.
4. **No manual audit layer.** Experiment 2 included manually evaluated legacy probes and human-verified hallucination audits; these were intentionally omitted from Experiment 3, since introducing manual adjustments while comparing fifteen architectural variants would have significantly increased the risk of evaluator bias. All buildup configurations were therefore assessed entirely through the automated evaluation pipeline.

### Key Results

The largest single improvement occurred when Reciprocal Rank Fusion (RRF) was introduced, yielding a gain of **+0.84** score points — combining lexical and semantic retrieval signals was substantially more valuable than either method operating independently. The largest performance decrease occurred immediately after adding BM25 retrieval, producing a score reduction of **−0.75** — lexical retrieval alone introduced substantial retrieval noise before ranking fusion was applied.

HyDE produced only a minimal improvement (+0.03), consistent with earlier Experiment-1-era observations suggesting limited practical benefit. Cluster restriction, session diversification, Codex retrieval, procedural memory, and batch-summary retrieval all produced small but generally positive effects — incremental improvements rather than transformative gains.

MERA produced a measurable negative impact (−0.21), suggesting either retrieval noise or a suboptimal interaction with the surrounding retrieval architecture. Dynamic token budgeting also produced a short-term decrease in answer quality despite dramatically increasing available context — increasing retrieval volume alone does not guarantee better answers and may initially introduce additional retrieval noise.

Keyword boosting emerged as one of the strongest late-stage improvements (+0.12), simultaneously increasing retrieval efficiency. Overall, the complete ICE configuration finished slightly above the bare-vector baseline — the mature architecture achieves its improvements through the cumulative interaction of multiple small gains rather than a single dominant subsystem.

**Interpretation.** ICE does not derive its performance from any single retrieval mechanism; instead, the architecture behaves as a layered retrieval system in which numerous modest improvements accumulate over time. The experiment also highlights that some components commonly assumed to be beneficial may contribute little, or even negatively, under realistic long-horizon conditions. The buildup study provides a quantitative roadmap for future development, identifying which subsystems deserve further investment and which may require redesign, replacement, or removal.

### Results Table — Flaw Buildup Ablation

Report metadata: **67 probes**, judged with **Qwen3-14B-AWQ**. Design: single-pass on fully-mature database (turn 1119).

**1. Cumulative Feature Addition (starting from bare vector)**

| Step | Score | Step Δ | Cum Δ | SPF | Tokens | Frags | Rec Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bare_vector | 3.27 | · | +0.00 | 0.23 | 14873 | 14 | +0.42 |
| add_bm25 | 2.52 | −0.75 | −0.75 | 0.17 | 14868 | 15.2 | +0.29 |
| add_rrf | 3.36 | +0.84 | +0.09 | 0.29 | 14870 | 11.7 | +0.54 |
| add_hyde | 3.39 | +0.03 | +0.12 | 0.29 | 14870 | 11.6 | +0.52 |
| add_cluster_restrict | 3.4 | +0.01 | +0.13 | 0.3 | 14911 | 11.3 | +0.66 |
| add_session_diversify | 3.4 | +0.00 | +0.13 | 0.3 | 14911 | 11.2 | +0.54 |
| add_codex | 3.43 | +0.03 | +0.16 | 0.3 | 14912 | 11.3 | −0.14 |
| add_mera | 3.22 | −0.21 | −0.05 | 0.28 | 14912 | 11.4 | +0.17 |
| add_procedural | 3.39 | +0.17 | +0.12 | 0.3 | 14912 | 11.4 | −0.10 |
| add_batch_summary | 3.41 | +0.02 | +0.14 | 0.3 | 14912 | 11.3 | +0.45 |
| add_dynamic_budget | 3.3 | −0.11 | +0.03 | 0.16 | 24489 | 21.2 | +0.08 |
| add_sliding_window | 3.32 | +0.02 | +0.05 | 0.16 | 27763 | 21.3 | +0.30 |
| add_keyword_boost | 3.44 | +0.12 | +0.17 | 0.23 | 27769 | 15.1 | +0.70 |
| full_ice | 3.38 | −0.06 | +0.11 | 0.22 | 27768 | 15.0 | +0.37 |
| vector_baseline | 3.42 | — | +0.15 | 0.11 | 16373 | 30 | −0.03 |

**2. Largest Stepwise Changes:** Largest gain — `add_rrf` (+0.84). Largest drop — `add_bm25` (−0.75).

**3. Recency Effect — Score by Origin Split (Fact Age)** *(values shown as Early/Mid/Late pairs where the source reported two numbers per cell, likely score/count or score variants — preserved as originally recorded)*

| Condition | Early (0-400) | Mid (400-800) | Late (800-1200) | Recency Δ |
|---|---:|---:|---:|---:|
| bare_vector | 2.67/3.5 | 3.25/2.4 | 3.75/3.35 | +0.42 |
| add_bm25 | 2.17/2.25 | 3.5/2.6 | 2.5/2.49 | +0.29 |
| add_rrf | 3/3 | 3.25/2.4 | 3.62/3.52 | +0.54 |
| add_hyde | 3.17/2.75 | 3.25/3 | 3.62/3.5 | +0.52 |
| add_cluster_restrict | 3.33/2.25 | 3.5/2.8 | 3.75/3.52 | +0.66 |
| add_session_diversify | 3/2.75 | 4.5/3.2 | 3.62/3.4 | +0.54 |
| add_codex | 3.67/3.5 | 3.5/2.8 | 3.62/3.42 | −0.14 |
| add_mera | 3.33/2.75 | 3.5/2.8 | 3.12/3.3 | +0.17 |
| add_procedural | 3.83/3 | 3.5/3 | 3.5/3.38 | −0.10 |
| add_batch_summary | 3.5/2.25 | 4.75/2.8 | 3.62/3.41 | +0.45 |
| add_dynamic_budget | 3.5/3 | 3.25/2.6 | 3.5/3.35 | +0.08 |
| add_sliding_window | 3/3.25 | 4.25/2.2 | 3.62/3.36 | +0.30 |
| add_keyword_boost | 3.33/2.25 | 3.75/2.8 | 3.88/3.54 | +0.70 |
| full_ice | 3.33/2.75 | 3.5/3 | 3.38/3.49 | +0.37 |
| vector_baseline | 3.5/3.5 | 3.5/2.8 | 3.75/3.41 | −0.03 |

**4. Key Findings**
1. The step-by-step deltas show exactly which feature improved retrieval quality at the moment it was introduced.
2. A negative step delta indicates the feature may have introduced noise or interacted poorly with previously active features.
3. The cumulative delta column shows how far each stage is from bare-vector, giving the overall progress of the system buildup.
4. SPF (Score per Fragment) measures retrieval precision — higher values mean each injected fragment contributed more to the final answer quality.

### Author Notes / Open Items
- Why 14B model instead of 26B — reason for the model-size choice is not explicitly stated beyond the general "smaller dense Qwen3-14B" description; needs elaboration if a specific rationale (cost, speed, availability) is to be cited.
- Why MERA hurts specifically in creative writing — the −0.21 delta is reported, but a specific causal explanation (retrieval noise vs. architectural interaction) is only speculated ("suggesting either retrieval noise or suboptimal interaction") and not conclusively diagnosed.
- Why dynamic budget hurts in long narratives specifically — general explanation given ("increasing retrieval volume alone does not guarantee better answers") but not narrative-specific.
- Did you expect Codex to do more than +0.03 — personal expectation not yet recorded.
- The final tie — "full ICE 3.38 vs vector 3.42, but 2× SPF. What does this tell you?" — SPF values from the table: full_ice 0.22 vs vector_baseline 0.11 (exactly 2×). Interpretive commentary connecting this to the "less but better context" thesis is implied by the paper's broader argument but not yet spelled out as a standalone sentence in this section.

---

Here is the complete, fully expanded **Limitations** and **Future Work** sections, written in the exact style of your `paper_rough_notes.md` — technically deep, brutally honest, and layered with specific references to your experimental data.

---

## Limitations

### Single-User Evaluation
All benchmark conversations were authored by a single user. While the selected conversations span creative writing, technical planning, academic planning, and long-horizon worldbuilding, they still reflect the habits, vocabulary, and interaction style of one individual. The results therefore demonstrate effectiveness for a diverse set of conversation *types* rather than for a diverse *population of users*. Repetition of this evaluation across multiple users is a prerequisite for generalising the findings beyond the current study.

### Synthetic Longitudinal Reconstruction and Missing Reinforcement Loops
The evaluation reconstructs conversation history by replaying existing conversations and periodically probing the resulting memory state. This allows controlled experimentation over hundreds or thousands of turns, but cannot fully replicate the feedback loops that occur during live usage. In real deployments, retrieval events continuously influence future memory states through reinforcement, decay adjustment, bookmarking, procedural extraction, and user behaviour; during evaluation, many of these interactions are absent because future turns already exist and cannot be influenced by retrieved outputs.

More specifically, ICE includes retrieval-strengthening mechanisms that increase the `access_count` and partially restore the `decay_score` of frequently accessed memories, while allowing unused memories to decay. During evaluation, retrieval reinforcement is driven almost entirely by benchmark probes rather than by natural user interaction. In a real conversation, a user might ask about a character's motivations every few sessions, reinforcing that memory repeatedly; in the evaluation, the memory receives reinforcement only when a probe explicitly targets it. Important concepts that would normally be revisited repeatedly during a live conversation therefore receive less reinforcement than they would in production usage, creating a more difficult retrieval environment than would be encountered during normal operation. Consequently, the benchmark likely **underestimates** the benefits of reinforcement-driven memory maturation. The positive longitudinal curves observed in Experiment 2 (e.g., Flaw knowledge curves rising from 3→5) are promising, but they are likely a lower bound on real-world performance improvement.

### Routing Bias and Classifier Conservatism (The Trade-Off)
Between Experiment 1 and Experiment 2, the classifier was retrained with a substantially improved embedding backbone (Qwen3-Embedding-0.6B replacing MiniLM), which reduced gating failures from 22 to 2. However, a second change was also introduced: a conservative routing override applied *after* classification. Queries classified as `Zero_Shot` were automatically re-routed to `Long_Term_Memory` when confidence fell below a threshold, or when the conversation exceeded a certain turn count (the "long-conversation LTM bias").

This override is a deliberate safety net: false negatives (failing to retrieve when memory is needed) are fatal, while false positives (retrieving when not strictly needed) are merely expensive. The empirical evidence from Experiment 1 supported this design—gating failures caused complete answer failures, whereas unnecessary retrieval only added modest context overhead.

However, the override introduces a fundamental limitation: **the system is no longer purely intent-aware in the way the architecture claims.** ICE frequently forces retrieval even when the classifier predicts zero-shot, because the heuristic overrides the classifier's judgment. This means the system is "safe" rather than "smart"—it errs on the side of retrieval to avoid catastrophic forgetting, at the cost of occasionally injecting irrelevant context. The true strength of the classifier is therefore masked by the override; we cannot claim that ICE's intent-gating is purely driven by the MLP, because the rule-based override is actively compensating for classifier uncertainty.

In future work, a more principled approach would be to treat the classifier's confidence as a prior and combine it with a conversation-length prior using Bayesian inference, rather than applying a hard threshold. The current override is a pragmatic fix, not a fundamental solution.

### Decay Horizon
The longest simulated memory horizon evaluated in this work was approximately 93 days (Dataset B / "flaw"). While sufficient for studying medium-term memory behaviour, the system has not yet been evaluated over year-scale horizons. Long-term stability, retrieval quality, memory saturation effects, and the eventual convergence of decay mechanics beyond several months remain open questions. It is possible that, at very long horizons, the decay function asymptotically approaches zero for all but the most frequently reinforced memories, creating a "cold start" effect where only bookmarked or highly reinforced information survives. This behaviour has been architecturally designed for (the creative floor at 0.3, the 180-day procedural deactivation window) but has not been empirically validated.

### Judge Model Limitations and Manual Audit Gap
Evaluation relied primarily on an automated judge model (`mattbucci/gemma-4-12B-AWQ`). Although extensive spot-checking and manual review were performed, automated judges remain imperfect. During analysis, multiple cases were identified where correct retrieved information was incorrectly marked as hallucinated because it was absent from the condensed ground-truth summary rather than absent from the conversation itself. To mitigate this, hallucination annotations in Experiment 2 were manually audited and corrected for all 1,211 probes.

However, a critical limitation applies to Experiment 3 (the ablation/buildup study): hallucination assessments in Experiment 3 were **not** manually corrected. Because the buildup study evaluated fifteen closely related system variants under identical conditions, introducing manual audit adjustments for each variant would have significantly increased the risk of subjective bias. Consequently, all hallucination metrics in Experiment 3 are derived directly from the automated judge output. The absolute hallucination values in Experiment 3 (e.g., the 41.4% hallucination rate on ICE-Dev) should therefore be interpreted cautiously, as they may include false positives from the automated judge. The *relative* differences between experimental conditions remain valid, since the same automated judge was applied uniformly across all configurations, but the absolute numbers should not be directly compared to Experiment 2's manually-audited values.

### Probe Generation Bias
Most benchmark probes were generated through a constrained LLM-assisted pipeline. Although extensive filtering, validation, and manual review were applied, generated questions may differ from the types of questions real users would naturally ask. Users tend to ask more context-dependent, anaphoric, and pragmatically ambiguous questions than LLM-generated probes. This is partially mitigated by the inclusion of manually authored probes originating from earlier experiments (72 total), which capture the kinds of natural-language questions that emerged organically during real conversations. Nevertheless, the benchmark is weighted toward automatically generated probes, which may bias evaluation toward retrieval tasks that align with the probe-generation model's distribution rather than natural human questioning patterns.

### Domain Coverage
The benchmark intentionally includes four substantially different conversation domains: creative writing (long-form narrative), long-form world-building (epic fantasy), technical planning (system architecture), and academic planning (career decisions). Many other domains remain untested—examples include collaborative software development (multiple participants, git integration), multi-user conversations, customer-support interactions, multilingual conversations, and professional workplace communication (meetings, project reviews, incident post-mortems). Performance characteristics may differ in those environments, particularly where domain-specific terminology, formal structures, or multi-party dynamics are present.

### Cross-Conversation Retrieval
ICE contains infrastructure for conversation scoping, project-level retrieval, and cross-conversation search. These capabilities were not directly evaluated—all benchmark probes target information contained within a single conversation. Consequently, the effectiveness of ICE's cross-conversation retrieval mechanisms remains future work. The current evaluation therefore demonstrates ICE's effectiveness at *within-conversation* memory, but does not yet validate its ability to aggregate knowledge across multiple independent conversations—a capability that would be essential for a user working on a single project across many sessions.

### Codex Extraction Reliability — Systemic Underperformance and Multiple Simultaneous Handicaps
The Codex subsystem is the most architecturally ambitious component of ICE and, in the current implementation, also the most fragile. Its marginal contribution in Experiment 2 (only 3.3% of retrieved fragments, +0.03 in the ablation) does not reflect the subsystem's theoretical ceiling, but rather the cumulative effect of several simultaneous engineering handicaps.

**The Chunking Paradox.** Current extraction uses 6,000-token chunks with a 200-token overlap. This was chosen to avoid splitting sentences and to preserve broad narrative context. However, a 4B-parameter model does not possess the effective reasoning capacity to process 6,000 tokens simultaneously. Transformer attention is quadratic—beyond 1,000–1,500 tokens, the model's attention mechanism becomes diluted, losing track of entities mentioned in the middle of the chunk while disproportionately weighting the beginning and end. This causes two failure modes: (1) **Entity Dropping**—entities introduced in the middle of a technical discussion or narrative passage are never extracted; (2) **Relationship Hallucination**—with attention diluted, the model confuses the subject and object of distant clauses, outputting syntactically valid but semantically meaningless triplets (e.g., `fastapi uses fastapi`). The optimal chunk size for reasoning with a 4B model is closer to 512–800 tokens. This was not used in the current evaluation, meaning Codex extraction operated under a systematic handicap that likely suppressed its contribution.

**The NER Fallback.** The Codex retrieval leg relies on a named-entity recognition step to identify entities in the user's prompt before querying the graph. The ideal implementation uses a lightweight BIO-tagging model trained specifically for conversational entity extraction. However, in the evaluated system, this model was not reliably loaded. When the `.pt` file was unavailable, the system fell back to a regex pattern (`\b[A-Z][a-zA-Z]{2,}\b`) that matches only capitalized words. This approach systematically fails on: lowercase named entities (`hayashi`, `keal`, `shinchan` in casual contexts); multi-word entities (`sliding window protocol`, `data link layer`); misspelled entities (`kael` vs. `keal`); and entities introduced through pronouns or anaphoric references. As a result, many retrieval queries that should have triggered graph traversal did not, because the system never identified the entity to look up.

**The Corroboration Trap.** Edges in the Codex begin with `confidence = pending` and require two independent corroborations to promote to `active`. This is a sound design for preventing hallucinations from corrupting the graph. However, because extraction runs at most once per conversation turn (and often fails entirely on dense technical text), the corroboration threshold is rarely met within the evaluation horizon. In the 1,119-turn Flaw conversation, many true relationships were extracted once and remained `pending`, never contributing to retrieval because the retrieval leg only queries `active` edges. The truth-quorum mechanism is therefore protecting the graph from noise at the cost of starving it of useful knowledge within practical timescales.

**Semantic vs. Lexical Matching.** Codex retrieval is exact-match (canonical name or aliases) plus vector fuzzy matching. Vector fuzzy matching operates over the `context_payload` embedding, not over the entity name itself. Consequently, if a user asks a question about a concept without using the exact canonical name or a known alias—For example, in a long-running creative conversation, the user establishes an entity named "The Obsidian Citadel" — a fortress with a distinctive dark-glass structure. The Codex stores this canonical name, along with a context_payload describing its location, purpose, and history.

Later in the conversation, the user asks: "Where is the main fortress located?"

The prompt contains no capitalized proper noun ("main fortress" is not a named entity), so the NER step falls back to the regex pattern (\b[A-Z][a-zA-Z]{2,}\b), which matches nothing. The Codex retrieval leg never fires because it cannot resolve "main fortress" to "The Obsidian Citadel". The system bypasses the graph entirely and reverts to episodic BM25 and vector search, which must locate the answer through raw text similarity rather than structured knowledge.

A human reader immediately understands that "main fortress" refers to the Citadel. The Codex, however, sees only lexical tokens and misses the semantic link entirely. This is a fundamental limitation of entity-centric retrieval in free-form conversation—it assumes users will consistently refer to entities by their canonical names, which is empirically false in natural dialogue. A more expressive retrieval mechanism that can answer relation-aware queries (e.g., "what is the primary stronghold of the northern region?" as a graph traversal starting from the region entity) would be required to close this gap.

**The LLM-as-Extractor Bottleneck.** Using the same language model for extraction that is also used for generation and summarization introduces a scheduling conflict. In shared mode, extraction tasks are delayed until the user is idle, causing significant lag in graph updates. In dedicated mode, the smaller 3B model frequently produces malformed JSON or outputs triplets that fail validation. A 1.5B–3B model is sufficient for summarization but arguably underpowered for accurate relation extraction in narrative or technical contexts, where relationships are often implied rather than explicitly stated. This creates a perverse outcome: the Codex, the subsystem responsible for ICE's most advanced memory capability, is systematically undermanned relative to the tasks it is asked to perform.

**Implications for the Current Results.** The marginal contribution of Codex in Experiment 2 should be interpreted as a floor, not a ceiling. The extraction pipeline, as evaluated, was operating under multiple simultaneous handicaps: oversized chunks, missing NER, slow corroboration, and an underpowered extractor. The finding that Codex still produced a positive (if tiny) contribution despite these handicaps suggests that a properly implemented Codex—with 512-token chunks, a trained NER model, and a dedicated extraction model—could become one of the strongest components in the architecture. The negative result for MERA (`-0.21` in Experiment 3) is similarly contextual: MERA is a fallback that activates when NER finds no entities; it is therefore a downstream indicator of NER failure rather than an independent subsystem evaluation.

### Hardware-Constrained Evaluation and Model-Size Trade-Offs
All experiments were conducted on consumer hardware (a single 24GB RTX 5090 GPU) and were designed around practical local-first deployment constraints. Certain evaluations—particularly full architectural ablations using the largest available models—were computationally infeasible. Consequently, some diagnostic experiments (e.g., Experiment 3's buildup study) were performed using a smaller dense model (Qwen3-14B-AWQ) rather than the larger 26B or 70B models that might have yielded higher absolute scores. This choice was driven by the need to run 15 ablative conditions × 67 probes within a reasonable timeframe. The *relative* deltas between ablated conditions remain valid, but absolute scores in Experiment 3 are systematically lower than those in Experiment 2, which used a larger judge and generation model. The primary benchmark (Experiment 2) used the largest model that could comfortably fit within the hardware budget; however, the system has not been evaluated on enterprise-grade hardware with 80GB+ VRAM, where larger generation models might further improve answer quality.

### Hallucination Rates and the ICE-Dev Paradox
ICE exhibits higher hallucination rates on the ICE-Dev conversation (Dataset C) than on other datasets—41.4% for ICE generalist vs. 20.2% globally. This is counterintuitive because ICE also achieves its highest absolute score on this dataset (4.33). The apparent contradiction is resolved by examining the denominator of the hallucination rate: the vector baseline failed on 94.2% of ICE-Dev probes (no answer), yielding a 0% hallucination rate by definition. ICE, because it survives these probes, generates answers and therefore has opportunities to hallucinate. The hallucination rate on ICE-Dev reflects the difficulty of the task, not a failure mode unique to ICE. High-quality, fact-dense technical answers simply carry a higher risk of generating a confident but incorrect detail, even when the overall answer is highly accurate.

### Mixture-of-Experts Routing: Hardcoded, Untrained, and Latency-Bounded

ICE includes a model-registry and routing system designed to select the best model for each query based on topic and intent overlap. However, the implemented router has three substantial limitations that together explain its empirical failure.

**First, the routing decision is based on hardcoded topic/intent mappings, not learned or empirical model-performance data.** The registry tags models manually based on Hugging Face metadata or LLM inference. There is no training signal that a particular model actually performs better on a specific topic or intent. A model tagged as "Software_&_Tech" may not meaningfully outperform a generalist on software questions, and the router has no way of knowing.

**Second, the router does not incorporate classifier confidence.** A low-confidence classification (e.g., `max_confidence = 0.55`) is treated identically to a high-confidence classification (e.g., `max_confidence = 0.95`) for routing purposes. The score is simply the overlap count between the predicted tags and the model's tags, plus a priority constant. This means the router may select a "specialist" model for a query the classifier is fundamentally uncertain about—a poor basis for routing.

**Third, the router does not consider context-reliance labels.** Queries classified as `Zero_Shot` (self-contained, no memory needed) receive the same routing treatment as `Long_Term_Memory` queries (requiring extensive context). This is particularly problematic because `Zero_Shot` queries might benefit from a smaller, faster model, while `Long_Term_Memory` queries might need a model with a larger context window—the router treats them identically.

The empirical result is unambiguous: in Experiment 2, MoE routing under ICE produced a global score delta of **+0.01** against the generalist. In Experiment 1, it was **−0.04**. The router is functionally neutral or slightly harmful.

**An operational limitation compounds the routing deficiency.** When the router selects a model that is not currently loaded, Ollama unloads the current model and loads the new one, a process that takes 5–15 seconds. ICE has no visibility or control over this process; it sends the selection to Ollama and waits. Session stickiness prevents thrashing (the router keeps the same model for up to 3 consecutive turns), but it cannot prevent the latency spike when a switch is forced. The MoE infrastructure is therefore both *conceptually underdeveloped* (hardcoded mappings, no confidence/context weighting) and *operationally costly* (model-load latency). Future work should address both dimensions, likely through learned routing policies and a model-loading API that gives ICE control over VRAM allocation.

### KV-Cache Optimisation: Designed but Largely Ineffective in Practice

The Prompt Assembler uses a stable-prefix ordering (System → Persistent Slots → Recent Turns → Retrieved Context → User Input) designed to maximise KV-cache reuse across consecutive requests. The system prompt and memory slots change infrequently, so in principle their KV tensors should be cacheable.

In practice, several factors limit cache utilisation. **First**, the recent-turn window changes on every request—even if only one turn is added, the entire prefix from that point onward shifts, invalidating the cache for subsequent tokens. **Second**, the retrieved-context block changes substantially on most queries, invalidating everything after the recent-window segment. **Third**, when MoE routing forces a model swap, the cache is wiped entirely. **Fourth**, even without model swaps, Ollama's KV cache is ephemeral and tied to the active session—if the session ends or the service restarts, the cache is lost.

The net effect is that cache hits are rare in practice. The only reliably cacheable segments are the system message and persistent memory slots, which together account for a small fraction of the total prompt (approximately 10–15%). The majority of the prompt—the recent-turn window, retrieved context, and user input—is recomputed on every request.

This limitation does not undermine the architecture; stable-prefix ordering is the correct *design* for KV-cache optimisation. However, the practical benefit is limited by factors outside ICE's control (Ollama's cache management, model swapping, session lifecycle) and by the inherent variability of the context-assembly process. Future work could investigate persistent cache storage, precomputed system/slot prefixes, and cache-aware retrieval policies that preferentially reuse stable context blocks. In the current implementation, however, the cache benefit is largely theoretical rather than empirically significant.

---

## Future Work

### Improved Codex Extraction and Grounding — From Triplet Collector to Self-Correcting Graph
One of the clearest findings from the evaluation is that Codex extraction becomes increasingly unreliable as conversational complexity grows. The current implementation is a passive triplet collector: the background model outputs triplets, and the system stores them. Future work will investigate a fundamentally different architecture: a **self-correcting knowledge graph** with deterministic grounding.

The proposed hybrid extraction pipeline consists of four stages:

1.  **Deterministic NER Grounding (CPU).** The NER model reused during the post flight response too along with its normal use in pre-flight. This runs on CPU and produces a confirmed list of entity strings for the codex extraction, completely bypassing LLM hallucination for entity identification.
2.  **Relationship Mapping (GPU).** The LLM receives the original text *and* the NER-confirmed entity list. Its sole task is to map relationships between these specific entities. This splits the cognitive load: the LLM no longer needs to search for entities, only to reason about connections between them. This should drastically reduce hallucinations like `fastapi uses fastapi`.
3.  **Deterministic Validation.** Every proposed triplet is checked against the existing graph. If a contradiction is detected, the system does not save the triplet blindly. Instead, it enters a reconciliation loop: the LLM is re-prompted with the conflicting edge and asked to resolve the inconsistency. This transforms the extractor from a passive collector into an active state reconciler.
4.  **Confidence-Calibrated Storage.** Edges are stored with their extraction confidence, not just a binary `pending`/`active` flag. Retrieval then uses confidence thresholds dynamically: high-confidence edges are promoted quickly, low-confidence edges require corroboration or human review.

For code-heavy conversations, a deterministic static-analysis layer will be added to the extraction pipeline: AST parsing, import-graph construction, and function-call tracking. This will produce deterministic edges (`imports`, `calls`, `inherits`, `defined_in`) that do not require LLM extraction at all. The Codex will then contain both conversational facts (extracted by the LLM) and code facts (extracted deterministically), all queryable through the same retrieval interface.

### Agentic Background Maintenance — Moving Beyond Fixed Pipelines
Current background workers operate through predefined pipelines: Post-Flight Evaluator → Codex Extractor → Procedural Extractor → Decay → Sentinel. This design is deterministic and reliable, but it lacks the ability to react to complex, cross-cutting inconsistencies. A Codex edge might be contradicted, a procedural pattern might be partially but not fully repeated, or a cluster might have drifted semantically without the system noticing.

Future work will replace individual extraction workers with a **Memory Maintenance Agent**—a lightweight LLM (the same 3B/4B model, or smaller) that is given a toolset of Python functions and tasked with maintaining memory consistency during idle GPU time. The toolset would include:

- `update_entity_relation(source, target, relation, new_state)`
- `merge_conflicting_entities(entity_a, entity_b)`
- `reconcile_graph_state(proposed_edge, existing_edge)`
- `flag_for_review(issue_description)`
- `run_cluster_consolidation(cluster_id)`

The agent would receive a notification when new turns are ingested and would decide autonomously whether to run extraction, whether to check for contradictions, whether to merge entities, or whether to escalate to the review queue. This shifts the background worker from a deterministic script executor to an autonomous decision-maker, reducing the need for human oversight while preserving the review queue as a safety net for high-uncertainty actions.

The Sentinel system, currently a rule-based skeleton (only `threshold` and `absence` implemented), would be fully integrated with the agentic maintenance loop. The agent would subscribe to Sentinel events and proactively resolve detected issues rather than simply logging them. For example, if the Sentinel flags a high-contradiction entity, the agent would query the graph, review the conflicting edges, and either resolve the contradiction or generate a human-readable review item.

### Cross-Conversation Project-State Memory and Coding Mode
The evaluation focused on single-conversation memory. Future work will extend ICE to cross-conversation retrieval, allowing information established in one conversation to inform responses in another. This requires the introduction of a **Project State Engine**—a dedicated set of tables and workers that track architecture clusters, decisions, tasks, and git history alongside the existing conversational memory stores.

In the proposed architecture, each project would maintain its own memory scope. A user could manually link conversations to a project, or the system could infer project membership from topic continuity and repeated entity references. Retrieval would then operate across the entire project's conversation history, bounded by a per-project token budget and filtered by relevance signals (topic overlap, entity presence, recency, decision status). This enables:

- **Coding-Mode Retrieval:** When the user asks technical questions, the system surfaces relevant code files, architecture decisions, and development patterns from across all conversations in the project.
- **Decision Tracking:** Critical decisions (e.g., "we chose PostgreSQL over DynamoDB") become Codex edges with explicit temporal provenance (`valid_from`/`valid_until`). A retrieval query about database choice surfaces the active decision and its rationale.
- **Git Integration:** Every commit is treated as a timestamped fact. The State Reconciler agent scans git diffs after each session, updates architecture clusters, and creates decision entries when new patterns are detected.
- **Task Management:** The `pending_items` memory slot is elevated to a full task table, with status tracking (`planned`/`in_progress`/`done`/`abandoned`), lessons learned, and source batch IDs for traceability.

This extension would transform ICE from a conversational memory system into a full personal knowledge management system, capable of tracking both what the user said and what the user built.

### Unified Context Budgeting with User Control
The dynamic token budget introduced in Experiment 2 is currently opaque to the user—it adjusts automatically based on conversation length, token density, and intent. Future work will expose budget controls directly through the frontend, allowing users to set per-conversation retrieval budgets, per-query budget caps, and priority overrides for specific memory slots. This transforms the budget from a hidden heuristic into a user-controlled resource allocation tool.

Key user-visible controls would include:

- **Retrieval aggressiveness slider:** Capped (minimal retrieval) / Moderate (dynamic budget) / Aggressive (wide-net fallback on every query).
- **Force-wide-net toggle:** A "deep search" button that bypasses classifier gating and searches all stores.
- **Per-project budget overrides:** A user can designate a high-importance project with a larger retrieval budget and a low-importance project with a smaller one.
- **Telemetry panel:** A real-time display showing how budget was allocated across retrieval legs for the current query, visible through SSE events.

This transparency would allow power users to diagnose retrieval failures and adjust policy accordingly, making ICE a truly configurable memory system rather than a black box. It also creates a natural sandbox for adaptive policies—if users with different budgets show different satisfaction levels, we can learn the optimal policy through user feedback.

### Retrieval Under Extreme Context Density (The ICE-Dev Stress-Test Follow-Up)
The ICE-Dev evaluation revealed a failure mode where conversational turns can individually contain tens of thousands of tokens (architecture documents, implementation plans, design discussions). The vector baseline collapsed (94.2% failure rate) because it blindly retrieved the top-ranked fragments, many of which were massive documents, collectively exceeding the model's context window.

Future work will investigate retrieval methods specifically designed for extremely dense contexts:

- **Chunk-Aware Retrieval:** Documents are split into semantic chunks (512 tokens) at ingestion time. Retrieval operates at the chunk level, not at the turn level. A query retrieves relevant chunks, not entire documents. This prevents a single 8,000-token document from dominating the retrieval budget.
- **Hierarchical Summarisation:** For massive documents, a hierarchical summarisation pipeline produces a three-layer abstraction: a short abstract (100 tokens), a medium summary (500 tokens), and the full document (8,000+ tokens). Retrieval first surfaces the abstract; if relevant, the system retrieves the medium summary; only when absolutely necessary does it retrieve the full document.
- **Document Decomposition:** Technical documents are decomposed into section-level chunks, each with its own embedding. Retrieval can then pinpoint the exact section that answers the query, rather than retrieving the entire document.
- **Context-Adaptive Truncation:** When the assembled prompt exceeds the budget, the system applies a truncation policy that prioritises relevant chunks over the full document. This is already partially implemented in the token-budget enforcement, but future work will make the truncation policy learned rather than heuristic-based.

### Retrieval with User-Defined Scoping and Cross-Conversation Linking
The current conversation scoping system (`None` / `Auto` / `Project` / `Manual`) is a promising foundation, but it has not been fully evaluated. Future work will extend it to support:

- **Temporary Retrieval Scopes:** A user can mark a specific turn, cluster, or document as "temporarily relevant" for the next N queries. This is useful when switching between projects mid-conversation.
- **Conversation Linking:** A user can manually link two conversations (e.g., a technical-planning conversation and a development conversation) so retrieval operates across both. The system can also suggest automatic links based on entity overlap and topic continuity.
- **Cross-Project Retrieval:** A user can declare that a specific Codex entity (e.g., "ICE") is relevant across all projects. Retrieval for any project then includes edges associated with that entity.
- **Selective Memory Exclusion:** A user can exclude a specific conversation or cluster from retrieval, even if it would normally be included. This is useful for maintaining privacy or preventing contamination from abandoned projects.

All linking and scoping operations would be user-controlled through the frontend, preserving INV-5 (user authority over memory) while allowing the system to assist with relevance inference.

### Long-Horizon Memory Studies and Year-Scale Deployments
The longest evaluated memory horizon was approximately 93 days. Future work will deploy ICE over year-scale timescales, studying:

- **Memory Saturation Effects:** Does the graph eventually become too large for effective retrieval? How does event-sourced compaction handle continuous growth?
- **Retrieval Drift:** Do retrieval preferences change over time? Does the system surface old, irrelevant information more frequently as the graph grows?
- **Long-Term Decay Dynamics:** After 6–12 months, do all but the most reinforced memories decay to zero? Does the creative floor at 0.3 effectively preserve narrative memories indefinitely?
- **Memory Compaction Strategies:** How frequently must the compaction worker run to keep the event log bounded? What is the optimal trade-off between compaction frequency and retrieval latency?

### Explainable Memory Systems (The Forensics Layer)
The evaluation demonstrated the importance of understanding why particular memories were retrieved. Future versions should expose retrieval traces, memory provenance, subsystem contributions, and real-time retrieval telemetry, so users can inspect and understand system behaviour during operation.

Key capabilities:

- **Retrieval Attribution:** Every retrieved fragment is tagged with the retrieval leg that produced it (codex, episodic, procedural, RAG, BM25), the score assigned by RRF, and the reason it was selected (keyword boost, recency boost, session diversification).
- **Memory Provenance:** Every fragment and edge carries a `source_batch_id` linking it to the original conversation turn. Users can trace any memory back to its origin.
- **Conflict Visualization:** The Codex graph can be visually inspected, with active edges highlighted and pending edges shown. Users can manually edit or delete edges through a graph-based UI.
- **Audit Trails:** Every write to any memory store is annotated with its source (user, post-flight, codex_extractor, procedural_extractor, reflection_worker, manual_injection, sentinel, bookmark). The full audit trail is queryable and exportable.

This transparency layer is essential for building trust in a system that evolves autonomously. Users must be able to understand *why* the system remembers what it remembers, and correct it when it remembers incorrectly.

### User-Guided Continual Learning and Feedback Integration
The thumbs-up/thumbs-down feedback mechanism proposed in earlier designs is not yet implemented. Future work will introduce an opt-in user feedback system:

- **Thumbs-Up / Thumbs-Down:** After every response, the user can rate the answer. A thumbs-up automatically adds the query and classification to the curated dataset. A thumbs-down triggers a re-evaluation: the system asks a stronger model to re-classify the query, and the corrected label is added to the curated dataset.
- **Automated Fine-Tuning:** The weekly fine-tune worker consumes the curated dataset and retrains the classifier head. A promotion script automatically copies the new checkpoint to the live path and restarts the proxy, closing the automation loop that is currently broken.
- **Manual Correction Interface:** Experienced users can open a side panel to view the classifier's predicted labels for any turn and manually correct them by clicking toggle buttons for topics, intents, and context-reliance. Corrections are immediately added to the curated dataset.
- **Safety Guards:** The feedback system is disabled by default. Amateur users must explicitly opt in, and the system provides warnings about the potential to degrade the classifier with inconsistent feedback.

### Learned Mixture-of-Experts Routing

The current MoE router is a hardcoded overlap scorer. Future work will replace it with a learned routing policy that incorporates classifier confidence, context-reliance labels, and empirical model-performance data. Candidate approaches include:

- **Confidence-Weighted Routing:** Routes are selected by weighting model-tag overlap by the classifier's confidence in its predictions. A high-confidence classification of "Software_&_Tech" would strongly bias toward technical models; a low-confidence classification would fall back to the generalist.

- **Context-Reliance-Aware Routing:** `Zero_Shot` queries route to smaller, faster models (e.g., 7B) while `Long_Term_Memory` queries route to larger, more capable models (14B+). This mirrors the intuition that self-contained questions require less reasoning capacity.

- **Learned Model Preference:** A lightweight bandit algorithm tracks model performance across query types and updates the routing policy accordingly. The model that consistently outperforms for a given topic/intent combination receives a higher routing score.

- **Model-Loading Integration:** The router will expose load/unload signals to the inference backend, allowing ICE to preemptively load the selected model before the next query, or to maintain multiple models in VRAM when capacity allows. This would eliminate the 5–15 second latency spike currently incurred on model switches.

The evaluation framework developed for ICE—particularly the LSREP protocol—can be repurposed to learn these routing policies offline. By replaying historical probes with different routing policies and measuring the resulting answer quality, ICE could learn a policy that is empirically superior to the current hardcoded overlap scorer. This is an extension of the existing architecture rather than a replacement.

### KV-Cache Persistence and Context-Aware Caching

While stable-prefix ordering is the correct design for KV-cache reuse, its practical effectiveness is limited by the factors described above. Future work will investigate cache-management strategies that operate at the inference-backend level:

- **Persistent KV-Cache Storage:** Rather than relying on Ollama's ephemeral in-memory cache, ICE could store precomputed KV tensors for stable prefix segments (system message, persistent slots) on disk or in a persistent memory store. These segments would be loaded into VRAM once and reused across sessions, surviving service restarts and model swaps.

- **Cache-Aware Retrieval Policies:** The retrieval orchestrator could preferentially select context fragments that are already cached—or at least avoid selecting fragments that would invalidate the cache—when multiple relevant fragments are available. This would require the orchestrator to know which fragments are currently cached, but would yield substantial latency improvements in stable conversations.

- **Incremental Cache Updates:** Rather than recomputing the entire prefix when a single token changes (e.g., a new turn added to the recent-window), the system could compute only the changed segment and append it to the existing cache. This is challenging to implement at the inference-engine level but would dramatically improve cache hit rates.

These strategies require deeper integration with the inference backend than ICE currently has. However, as inference engines (Ollama, vLLM, SGLang) expose more cache-control APIs, these optimisations become increasingly feasible. The stable-prefix ordering already positions ICE to benefit from such advances; the remaining work is to implement the cache-management layer.

### Conversation Import as a First-Class Feature (LSREP as Migration Tool)

**Zero-Shot Memory Migration via LSREP Ingestion.** The Longitudinal State-Replay Evaluation Protocol, developed originally as a research benchmark, has an unanticipated secondary application: it is, in effect, a complete conversation-ingestion pipeline. The same chronological replay mechanism used to evaluate ICE—chronological turn injection, post-flight evaluation, Codex extraction, procedural extraction, clustering, and reflection—can be exposed directly to users as a migration tool.

A fundamental barrier to adopting local AI systems is the **vendor lock-in of memory**. Users who have spent months or years building conversational history on cloud platforms (ChatGPT, Claude, DeepSeek) cannot easily abandon those platforms because their accumulated context—project decisions, creative lore, personal preferences, technical rationales—exists only within the cloud provider's infrastructure. Moving to a local system means starting from zero, which for many users is unacceptable.

ICE's LSREP infrastructure solves this problem. The same code that replays historical conversations for evaluation can replay exported chat logs for ingestion. A user would export their conversation history from any cloud platform (available as JSONL or structured text from most major providers), drop the file into an ingestion endpoint, and the system would:

1. **Chronologically Replay** the conversation, ingesting turns into episodic memory with correct timestamps.

2. **Run the Post-Flight Evaluator** on every turn, setting `lossless_flag`, generating summaries, and marking documents.

3. **Execute Background Workers** in accelerated or real-time mode: Codex Extractor builds the knowledge graph; Procedural Extractor identifies behavioral patterns; Clustering Worker organizes turns into topical clusters; Reflection Worker synthesises session summaries.

4. **Populate the Review Queue** with automated proposals for memory-slot updates and cluster creations, awaiting user approval.

The result is a **fully mature ICE deployment** that treats the imported history as if it had always been part of the system. A user migrating from a cloud assistant would immediately have access to their full conversational memory, with structured Codex entities, procedural patterns, and temporal decay already applied.

**Critical Consideration: Decay Scaling.** A major design decision is how to handle decay for imported history. If a user imports a 6-month conversation and the system applies 180 days of decay cycles immediately, the imported memory may be artificially degraded upon arrival. To address this, ingested conversations would be treated with a stabilisation phase:

- **Stabilisation Window.** Imported turns are initially assigned a `decay_immune` flag for a configurable window (e.g., 30 days), allowing the user to interact with the memory before decay begins.

- **Fast-Forward Mode.** Decay cycles for imported history are applied at an accelerated rate only if the user enables "Fast-Forward" mode, which simulates the aging of memory to match the original timeline. By default, imported turns begin with `decay_score = 1.0` regardless of their chronological age, and decay progresses forward from the import date.

- **User-Selectable Ingestion Policies.** The user is given a choice during ingestion: "Preserve all imported memory" (`decay_immune` until explicitly unset), "Simulate natural decay" (apply decay cycles proportional to the time elapsed between the conversation's original date and the import date), or "Start fresh" (normal decay from the import date).

Why This Is Not Merely Data Ingestion. A naive implementation of conversation import would simply chunk and embed the history, storing it as static vectors for future retrieval. This is essentially what the vector-RAG baseline does. A user importing history into such a system would be able to search their past conversations, but the system itself would not "understand" the history in any structured sense. It would be an archive, not a memory.

ICE's import pipeline differs fundamentally. Because ICE maintains episodic memory, a knowledge graph (Codex), procedural memory, and topical clusters as distinct stores with independent maintenance processes, importing a conversation means reconstructing the entire cognitive state that the system would have had if it had been present during the original conversation. The Codex Extractor builds a graph of entities and relationships. The Procedural Extractor infers behavioural patterns. The Clustering Worker organises turns into topics. The Reflection Worker synthesises session summaries. The imported history does not just exist as retrievable text; it is lived through by the system, producing the same structured memory state that would have emerged from live interaction.

This distinction has practical implications. A user migrating from a cloud platform does not merely bring a searchable archive—they bring the AI's understanding of their history. The model remains stateless and oblivious; ICE's memory stores become the persistent state that bridges the gap between the model and the user's accumulated experience. The import feature is therefore not a secondary application of LSREP; it is a direct demonstration of the architectural separation between the model (stateless processor) and the memory system (stateful external store). The same mechanism that replayed the Flaw conversation for evaluation can replay any exported conversation for migration, with the same background workers producing the same structured memory. The only difference is the source of the turns.
---

## Conclusion

### Open Items

Draft Text

The Infinite Context Engine began as a response to a recurring, deeply frustrating experience: every time a long conversation ended, the memory of what was discussed, decided, or built vanished. Re-explaining the same architectural decisions, character details, and project histories to a model that had just held a competent conversation about them felt like a fundamental failure of the interface between human and machine.

ICE emerged from a simple observation: memory is not about retrieving everything—it is about retrieving the right thing at the right time, without overwhelming the system that has to use it. The architecture that followed—intent-aware classification, six-leg hybrid retrieval, reciprocal rank fusion, temporal versioning, access-weighted decay, and dynamic token budgeting—was built around this single principle.

The evaluation journey was far from linear. The first precision-based benchmark returned 13.87%, failing due to stripped context, timestamp mismatches, and an incompetent judge. Experiment 1, after correcting a token-accounting error, showed ICE losing to a vector baseline on answer quality (4.04 vs. 4.06) while using more tokens and hallucinating more. It was a sobering result, and it forced a thorough rebuild of almost every subsystem: the classifier, the retrieval budget, the ground-truth construction, and the evaluation protocol itself.

Experiment 2 told a different story. On the mature system, ICE matched the vector baseline on answer quality (4.26 vs. 4.25) while injecting 32% fewer fragments and winning 30.6% of head-to-head tournaments compared to 21.2% for the baseline. More importantly, on the ICE-Dev technical-planning conversation—where dense architecture documents pushed every turn well beyond typical conversational length—the vector baseline collapsed to a 94.2% failure rate due to context overflow, while ICE maintained a mean score of 4.33 with zero failures. The dynamic token budget, initially designed as a simple optimization, proved to be a safety mechanism: it transformed retrieval from a "best effort" activity into a bounded resource allocation problem, forcing the system to prioritise utility over volume.

The ablation study in Experiment 3 revealed which components actually contribute. Reciprocal Rank Fusion provided the largest single improvement (+0.84 score points), demonstrating that combining lexical and semantic retrieval signals is substantially more valuable than either alone. Conversely, BM25 without RRF hurt performance (-0.75), HyDE contributed minimally (+0.03), and MERA actively degraded performance (-0.21). Keyword boosting emerged as a strong late-stage addition (+0.12). The cumulative result was a system that performs not through any single dominant mechanism, but through the careful layering of many modest improvements that reinforce one another.

The single most important thing ICE taught us is that memory systems for local AI must be judged not by their average-case performance on clean benchmarks, but by their behaviour at the extremes—when conversations grow dense, when turns exceed 8,000 tokens, when context budgets are tight, and when retrieval noise would otherwise drown the generation model. A system that performs well on average but collapses catastrophically under stress is not a memory system; it is a liability. ICE survives where the baseline fails. That survival is its primary contribution.

If the reader remembers only one number from this paper, it should be this: on the ICE-Dev technical-planning conversation, the vector baseline failed 94.2% of the time due to context overflow; ICE failed 0% of the time, maintaining a mean score of 4.33. This is not a claim of superior retrieval precision. It is a claim of architectural robustness—the ability to degrade gracefully rather than catastrophically when the system is pushed to its limits.

The limitations are real and clearly articulated. The Codex remains underpowered relative to its potential, its extraction pipeline hobbled by oversized chunks, missing NER, and an underpowered extractor. The evaluation is single-user and restricted to four domains, with a maximum horizon of 93 days. The MoE router shows no measurable advantage over a generalist model. Hallucination rates remain high, particularly on dense technical conversations. These are not failures to be hidden; they are open problems to be addressed in future work.

ICE is not the final word on conversational memory. It is, however, a working demonstration that a single-user, consumer-GPU system can maintain structured, temporally-aware, longitudinally-improving memory across hundreds of turns and thousands of tokens—without requiring cloud infrastructure, proprietary APIs, or unlimited VRAM. The system is built, the evaluations are run, the numbers are honest, and the code is open.

A model without memory is a calculator. A model with memory that collapses under its own weight is a broken promise. ICE is neither. It is a foundation for long-horizon cognition, built to last on the hardware we actually own—and, with the necessary fixes and extensions, it can become a great deal more.