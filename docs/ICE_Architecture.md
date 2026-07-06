







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

2.2 DI3 — Dynamic Intent Inferencer	4

2.3 Context-aware classification	5

2.4 Override rules	6

2.5 Training pipeline	6

2.6 Experimental / partial features	7

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

6.1 The six retrieval legs	20

6.2 HyDE query rewriting	20

6.3 Dynamic leg weighting	21

6.4 Reciprocal Rank Fusion (RRF)	21

6.5 Post-fusion processing	22

6.6 Cluster-scoped retrieval	23

6.7 Dynamic token budget	23

6.8 Wide-net fallback	24

6.9 Feature Toggling for Ablation Studies	24

7. Prompt Assembly	24

7.1 Stable-prefix ordering	25

7.2 Per-component rendering	25

7.3 Emotional / creative bypass	25

7.4 Token budget enforcement during assembly	26

8. Background Worker Cluster	26

8.1 Celery + Redis infrastructure and GPU gating	26

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

10.3 Celery over Redis	34

10.4 Idempotency architecture	34

10.5 GPU resource management	34

10.6 Configuration system	35

*This section describes the system as implemented in the source tree at the time of writing. Where the legacy design documents (\`architecture.md\`, \`architecture\_v2.md\`) conflict with the code, the code is treated as authoritative. Features that are experimental, gated off, or not yet wired into the live path are flagged inline.*


## **1. System Overview**

ICE (Infinite Context Engine) is an **OpenAI-compatible memory middleware** that sits between a conversational client and a pool of locally-served large language models (Ollama / SGLang). It is not a model itself: every request addressed to the synthetic model name "ice-proxy" is intercepted by a FastAPI proxy that classifies the turn, retrieves relevant context from four long-lived memory stores, assembles a KV-cache-friendly prompt, routes the request to a per-turn specialist model, streams the response back over Server-Sent Events (SSE), and then dispatches a fan-out of background workers that extract, decay, cluster, and crystallise the new turn into long-term memory. The net effect is that any downstream model — regardless of its native context window — operates against an effectively unbounded, personalised context that is selectively rebuilt on every turn.

ICE therefore occupies the layer conventionally called \*memory and retrieval orchestration\* in a conversational-AI stack: above the model-serving substrate (Ollama, SGLang, Hugging Face text-generation inference) and below the client. It exposes the standard /v1/chat/completions endpoint, so existing OpenAI-compatible clients require no modification beyond pointing at the proxy and (optionally) sending an X-ICE-Conversation-ID header.

### **1.1 Request lifecycle**

Each turn traverses a **pre-flight** (synchronous, in the request path) and a **post-flight** (asynchronous, after the stream closes) phase.

**Pre-flight.** (i) The user message is classified by a two-stage pipeline — a rule-based \*Dynamic Intent Inferencer\* (DI3) followed, on miss, by a 25-way PyTorch MLP head — producing topic tags, intent tags, and a context\_reliance label. (ii) Hard override rules coerce the label (e.g. Creative\_&\_Media ⇒ Long\_Term\_Memory). (iii) A \*Hybrid Retrieval Orchestrator\* runs six retrieval legs (BM25, vector, Codex graph, procedural, RAG, batch summaries), fuses them with weighted Reciprocal Rank Fusion (RRF), and post-processes the fused list with keyword/recency/length bonuses, session diversification, deduplication, and a dynamic token budget. (iv) A \*Prompt Assembler\* concatenates the retrieved fragments with persistent memory slots, recent turns, and the live user message under a stable prefix that maximises KV-cache reuse. (v) A \*Mixture-of-Experts\* (MoE) router selects the best locally-served model for the assembled prompt, with per-conversation stickiness.

**Post-flight.** (i) A \*Post-Flight Evaluator\* runs lossless detection, document detection, and summary generation, writing the auxiliary columns of the new episodic\_memory row. (ii) It unconditionally dispatches a \*Procedural Extractor\* and, conditionally on the lossless flag, a \*Codex Extractor\*; both are idempotent Celery tasks. (iii) A fan-out of scheduled background workers — Decay, Clustering, Reflection, Batch Summariser, Sentinel Monitor, and a weekly Fine-Tune worker — maintains the memory stores over time.

### **1.2 High-level component map**

The system decomposes into the following components, each described in the corresponding section below:

                       ┌──────────────────────────────────────────────┐  
   Client ──HTTP/SSE──▶│                FastAPI Proxy                  │  
                       │  /v1/chat/completions  /memory-slots  /user-  │  
                       │  control  /model-registry                    │  
                       └───┬──────────────────────────────┬───────────┘  
                           │ pre-flight (sync)            │ post-flight (async)  
            ┌──────────────▼──────────────┐  ┌────────────▼─────────────────┐  
            │  Classification Engine       │  │  Background Worker Cluster    │  
            │  (DI3 + MLP + overrides)     │  │  (Celery / Redis)             │  
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

The classification engine produces, for each user turn, a triple of outputs: a set of **topic tags**, a set of **intent tags**, and a single **context-reliance** label. These three outputs drive every downstream decision — which retrieval legs are weighted, how the token budget is split, whether the wide-net fallback fires, and which model the MoE router selects. The engine is deliberately a two-stage cascade: a cheap rule-based pre-classifier (DI3) resolves obvious cases in microseconds, and a small PyTorch MLP resolves everything else with sub-millisecond CPU inference.

### **2.1 The PyTorch MLP classifier**

The model is a deliberately tiny MLP (classifier/model.py):

class ICEClassifier(nn.Module):  
    def \_\_init\_\_(self):  
        super().\_\_init\_\_()  
        self.fc1 = nn.Linear(384, 128)  
        self.relu = nn.ReLU()  
        self.dropout = nn.Dropout(0.3)  
        self.fc2 = nn.Linear(128, 25)

The input is a single **384-dimensional embedding** produced by a frozen SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", truncate\_dim=384). The single hidden layer (384 → 128 with ReLU and Dropout(0.3)) feeds a **shared linear head** fc2: 128 → 25 whose 25 logits are sliced at inference time into three blocks rather than being produced by three separate heads:

- logits \[0:11\] — **topic** block, decoded with torch.sigmoid; a label is selected when prob \> classifier\_threshold (default 0.3), with argmax fallback if no label clears the threshold (multi-label).

- logits \[11:22\] — **intent** block, decoded identically (multi-label, max 3 retained downstream).

- logits \[22:25\] — **context-reliance** block, decoded with torch.softmax and argmax (single-label).

The eleven topic labels are Software\_&\_Tech, STEM\_&\_Academics, Business\_&\_Finance, Creative\_&\_Media, Admin\_&\_Productivity, Lifestyle\_&\_Health, Social\_&\_Relationships, World\_&\_Current\_Events, Meta\_AI, Null\_Noise, and General\_Reference\_&\_Trivia. The eleven intent labels are Factual\_Retrieval, Troubleshooting, Generation, Ideation, Analysis\_&\_Summarization, Strategic\_Planning, Decision\_Making, Emotional\_Processing, Utility\_Formatting, Casual\_Banter, and Open\_Exploration. The three context-reliance labels are Zero\_Shot, Long\_Term\_Memory, and Real\_Time\_Search. The output is wrapped in a ClassificationResult dataclass (topic\_tags, intent\_tags, context\_reliance, raw\_probs (length 25), max\_confidence, prompt).

### **2.2 DI3 — Dynamic Intent Inferencer**

DI3 (classifier/di3.py, gated by settings.DI3\_ENABLED) is a rule-based pre-classifier that catches obvious cases before the MLP runs. It computes five density signals ∈ \[0,1\] from the raw prompt (classifier/di3\_signals.py):

- **\`code\_density\`** — token-weighted sum over CODE\_FEATURES (e.g. \`\`\`  :0.4, def/class/function/import/==/!=/\{/\}:0.1 each, if/else/for/while/return/print\`:0.05), capped at 1.0.

- **\`sentiment\_density\`** — +0.1 per word in a 25-word sentiment lexicon, +0.2 for the patterns "i feel" / "i'm feeling", +0.15 per sentiment word when "i'm"/"im" is present, capped at 1.0.

- **\`meta\_density\`** — +0.1 per META\_KEYWORDS = \{"you","your","model"\}, +0.15 per META\_PHRASES = \{"prompt","prompting"\}, +0.2 per META\_PATTERNS ("how do i prompt", "what model", "which model", "how should i prompt").

- **\`noise\_density\`** — +0.2 if len \< 5, +0.6 if no alphabetic characters, +0.3 on a KEYBOARD\_MASH substring, +0.2 if the alphabet of the string has ≤3 distinct characters, capped at 1.0.

- **\`reference\_density\`** — per-word weights on anaphoric terms (this:0.15, that/these/those:0.10, it/the:0.05).

DI3 evaluates five rules **in order**, returning the first that fires (or None, in which case the MLP runs):

| **Rule** | **Threshold (default)** | **Output** | **Confidence** |
| - | - | - | - |
| **Noise** | noise\_density \> 0.8 | topic=\[Null\_Noise\], intent=\[Casual\_Banter\], ctx=Zero\_Shot | 0.95 |
| **Code** | code\_density \> 0.3 | topic=\[Software\_&\_Tech\], intent=\[Generation\], ctx=Long\_Term\_Memory if conversation\_length\>0 else Zero\_Shot | 0.90 |
| **Sentiment** | sentiment\_density \> 0.4 | topic=\[Lifestyle\_&\_Health, Social\_&\_Relationships\], intent=\[Emotional\_Processing\], ctx=Long\_Term\_Memory if length\>5 else Zero\_Shot | 0.85 |
| **Meta-AI** | meta\_density \> 0.2 | topic=\[Meta\_AI\], intent=\[Factual\_Retrieval\], ctx=Zero\_Shot | 0.90 |
| **Reference** | reference\_density \> 0.2 (or \> 0.1 when length \> 10) | topic=\[\], intent=\[\], ctx=Long\_Term\_Memory | 0.70 |


The reference rule is special: it returns empty topic\_tags and intent\_tags. The orchestrator in classifier.classify() detects this empty‑tag condition and, before applying hard overrides, runs the MLP to supply the topic and intent labels. The MLP’s context\_reliance output is then **overwritten** with Long\_Term\_Memory. This two‑stage decision—fast anaphora detection followed by precise topic/intent inference—gives the system a low‑latency path for recognising continuation prompts while still obtaining full label detail. The two‑tier threshold on the reference rule (0.2 for short conversations, 0.1 once the conversation exceeds ten turns) is the engine’s primary mechanism for implementing a *long‑conversation LTM bias*. The two-tier threshold on the reference rule (0.2 → 0.1 once a conversation exceeds ten turns) is the engine's primary mechanism for implementing a \*long-conversation LTM bias\*.

### **2.3 Context-aware classification**

When a conversation\_id is supplied, the MLP path queries the **last three \`episodic\_memory\` turns** for that conversation (\_get\_context\_turns, n=3, max\_total\_words=500), preferring summary\_text and falling back to the first 150 words of raw\_text with an ellipsis. The context is prepended to the prompt as natural-language text under a fixed template ("Conversation context (summarized):\\n\{context\}\\n\\n… User prompt: \{prompt\}") before embedding — there is no separate context vector or pooling. The same embedder is used with or without context.

### **2.4 Override rules**

After either DI3 or the MLP produces a result, \_apply\_hard\_overrides runs three rules **in evaluation order**:

1. **LTM immutability.** If context\_reliance == "Long\_Term\_Memory" is already set (by DI3 or by an API-level bias), return immediately — no override may downgrade an LTM decision.

2. **Creative → LTM.** If Creative\_&\_Media is in topic\_tags, force Long\_Term\_Memory. (Creative continuations almost always reference earlier narrative state.)

3. **Software & Tech + anaphora → LTM.** If Software\_&\_Tech is in topic\_tags and the lowercase prompt contains any of 23 referential words (my, our, mine, ours, we, us, this, that, these, those, the, it, they, them, their, previous, last, before, yesterday, earlier, again, still, same), force Long\_Term\_Memory. The match is a substring check, so common short words like the match liberally; the design tolerates the resulting false positives because the cost of a missed LTM fetch on a technical continuation is judged higher than the cost of an unnecessary one.

A second, API-level LTM bias lives outside the classifier in api/main.py: when a conversation has more than ten turns or the classifier's max\_confidence \< 0.95, a Zero\_Shot context-reliance label is upgraded to Long\_Term\_Memory before retrieval runs.

### **2.5 Training pipeline**

The classifier is trained offline through a five-stage pipeline.

**Stage 1 — Amnesia Method data harvesting.** Personal conversational archives are mined by scripts/classifier/promt\_extraction/extract\_promts.py, which uses qwen3-coder:30b-a3b-q4\_K\_M on Ollama to extract user-authored prompts from a raw chat corpus under a \*precision-over-recall\* contract: chunks of 12000 characters with 1000-character overlap, temperature=0.0, a confidence floor of 0.85, and 21 hard regex filters that reject AI openers ("sure,", "here is", "good catch", …), mid-sentence fragments, and structural markers. The same stage harvests 5 000 prompts each from three public datasets — lmsys/chatbot\_arena\_conversations, ShareGPT\_Vicuna\_unfiltered, and allenai/WildChat-1M — filtered to English first-turns. All sources are deduplicated by SHA-256 of the normalised text and shuffled under RANDOM\_SEED = 42 by combine\_dataset.py.

**Stage 2 — vLLM labelling.** scripts/classifier/promt\_labeling/VLLM\_label\_dataset.py labels the blended corpus with Qwen/Qwen2.5-7B-Instruct-AWQ served on vLLM (port 8001), temperature=0.0, seed=42, CONCURRENT\_REQUESTS=20, structured output enforced via instructor.Mode.JSON. The labelling prompt is a decision tree applied in strict order: (i) a \*source-aware calibration\* — personal sources use a \*low\* LTM threshold, the three public sources use an \*extremely high\* threshold that requires an explicit continuation phrase; (ii) six \*immunity traps\* that short-circuit to Zero\_Shot (pasted context, public entities, self-contained hypotheticals, role assignments, quoted pronouns, and a source-specific continuation-phrase allow-list for public datasets); (iii) Real\_Time\_Search signals (current price, live score, today's news, …); (iv) six Long\_Term\_Memory signals A–F covering demonstrative references, personal possessives, continuation language, named personal entities, implicit subjects, and questions about the user's own history; (v) default Zero\_Shot. The model is required to fill a reasoning field answering four questions (source, immunity, signals, decision) \*before\* emitting labels, which materially improves label consistency. Intent labels are capped at three.

**Stage 3 — Label vectorisation.** build\_training\_data.py converts each labelled record into a 25-dimensional multi-hot vector (11 topic + 11 intent + 3 one-hot context-reliance), skipping orphans where either topic or intent is empty.

**Stage 4 — From-scratch training.** train\_classifier.py trains all MLP parameters with Adam(lr=1e-3), batch\_size=32, epochs=30, a 10 % validation split, and early stopping (PATIENCE=5). The loss is BCEWithLogitsLoss(pos\_weight=topic\_pos\_weight) + BCEWithLogitsLoss(pos\_weight=intent\_pos\_weight) + CrossEntropyLoss(), where the pos\_weight for each of the 22 multi-label columns is num\_neg / num\_pos clamped at pos\_weight\_cap=15.0 (no weighting on the 3-dim context block). The best checkpoint by validation loss is saved as ice\_classifier\_v2.pt.

**Stage 5 — Iterative fine-tuning.** fine\_tune.py loads a checkpoint, **freezes \`fc1\`** so only the fc2 head is trainable, and trains for 10 epochs at lr=5e-5 with plain BCEWithLogitsLoss (no pos\_weight). Hand-curated corrections in data/curated\_fixes.jsonl are pre-encoded, repeated 50×, and interleaved into every training batch via itertools.cycle, weighted 10× in the loss — the human-in-the-loop correction mechanism. The output is ice\_classifier\_v3\_qwen\_ft.pt. The active inference path loads ice\_classifier\_v3\_qwen\_ft3.pt (set by settings.classifier\_model\_path).

### **2.6 Experimental / partial features**

A ConfigurableOrchestrator subclass exists for ablation studies; it does not redefine the classification logic but exposes flags that toggle retrieval legs on and off (bm25, vector, codex, procedural, rag, cluster\_restrict, hyde, dynamic\_budget). The hyde flag is the only path that enables HyDE query rewriting — in the production orchestrator the HyDE call is commented out (see §6.2).


## **3. Memory Architecture**

ICE maintains **four distinct memory stores**, plus **Memory Slots** (persistent structured working memory), **Context Clusters** (conversation-scoped topical clusters), and a **cold-storage** archive. All stores live in a single PostgreSQL database with the pgvector extension; every vector column is Vector(384) because the same Qwen/Qwen3-Embedding-0.6B embedder (with truncate\_dim=384) is used across classification, retrieval, clustering, and the workers.

### **3.1 Episodic memory**

The episodic\_memory table is the system's primary store of conversational turns. Its schema (every column):

| **Column** | **Type** | **Notes** |
| - | - | - |
| **id** | UUID PK | uuid.uuid4 |
| **conversation\_id** | UUID FK → conversations.id |  |
| **cluster\_id** | UUID FK → context\_clusters.id (nullable) | legacy single-cluster pointer; the M2M link table is authoritative |
| **parent\_message\_id** | UUID self-FK | threading |
| **batch\_id** | UUID | groups a user turn + assistant reply; the join key used by Codex/Procedural workers |
| **timestamp** | DateTime(tz) | utcnow |
| **topic\_tags / intent\_tags** | ARRAY(Text) | classifier output; Creative\_&\_Media triggers the decay floor |
| **context\_reliance** | Text | classifier label |
| **entropy\_score** | Float |  |
| **lossless\_flag** | Boolean (nullable) | NULL = not yet evaluated; True exempts from batch summarisation and gates Codex extraction |
| **raw\_text** | Text | full turn text |
| **summary\_text** | Text | post-flight or compaction summary |
| **embedding** | Vector(384) |  |
| **decay\_score** | Float default 1.0 | multiplied each decay cycle |
| **access\_count** | Integer default 0 | incremented on retrieval |
| **is\_archived / is\_bookmarked / decay\_immune / inject\_raw / is\_document** | Boolean | various flags |
| **idempotency\_key** | Text UNIQUE | API-layer deduplication |


**Population.** The FastAPI proxy inserts one row per user/assistant turn in store\_turn\_async (a BackgroundTasks callback that runs after the SSE stream closes). The Post-Flight Evaluator writes lossless\_flag, summary\_text, inject\_raw, and is\_document; the Batch Summariser writes summary\_text for decayed-but-not-yet-archived turns.

**Querying.** Two retrieval legs read this table (§6.1–6.2): the BM25 leg uses ts\_rank over to\_tsvector('english', coalesce(raw\_text,'')||' '||coalesce(summary\_text,'')) with a plainto\_tsquery built from the top-30 stop-word-filtered prompt tokens, filtered by decay\_score \> 0.2 AND is\_archived = false, LIMIT 100; the vector leg uses the decay-weighted cosine score (1 - (embedding \<=\> :prompt\_embedding)) \* COALESCE(decay\_score, 1.0) under the same visibility invariant, LIMIT 100. After fusion, retrieved turns are \*strengthened\*: access\_count += 1 and decay\_score = min(1.0, decay\_score + 0.15).Fragments originating from turns flagged as is\_document (long, low‑conversational‑density pastes detected by the Post‑Flight Evaluator, §8.2) bypass the 500‑word cap and are injected in full, subject only to the overall token budget.

### **3.2 Codex (semantic) memory**

The Codex store is a versioned knowledge graph spread over four tables (full schema in §4): codex\_entities (canonical name, aliases, tags, properties JSONB, auto-regenerated context\_payload, embedding Vector(384)), codex\_edges (typed relations with strength, confidence ∈ \{pending, active\}, valid\_from, valid\_until), codex\_events (append-only event log), and codex\_snapshots (compaction output). It is populated by the Codex Extractor (§4.3) and the manual Codex Inject Watcher (§8.9), enriched by the Reflection worker, and queried by the Codex graph-traversal retrieval leg (§4.8).

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
| **embedding** | Vector(384) | of pattern\_description |


It is populated by the Procedural Extractor (§5.2) and the Reflection worker's \_crystallize\_patterns step, and queried by the procedural retrieval leg (§5.4).

### **3.4 RAG memory**

Two tables — rag\_documents (id, filename, file\_type, uploaded\_at, token\_count) and rag\_chunks (id, document\_id FK, chunk\_index, chunk\_text, embedding Vector(384)) — hold externally-ingested documents. They are populated exclusively by the Drop Zone worker (§8.8), which chunks content into **512-word windows** and embeds each with the classifier's embedder. The RAG retrieval leg is the only consumer.

### **3.5 Memory Slots**

The memory\_slots table is the system's \*persistent structured working memory\* — slots whose contents are prepended verbatim to every prompt. Schema: id UUID PK, slot\_name Text, content Text, token\_count Integer, version Integer (bumped on every update), last\_updated, updated\_by ∈ \{user, system, reflection\_worker\}, is\_active Boolean. Only seven slot names are server-enforced as valid (api/routers/memory\_slots.py):

persona · user\_preferences · tool\_guidelines · project\_context ·  
guidance · pending\_items · session\_patterns

Slots are loaded on every chat request (db.query(MemorySlot).filter\_by(is\_active=True).all()) and rendered into the system message under an === PERSISTENT CONTEXT === block. They are written through four paths: direct user update (PUT /memory-slots/\{slot\_name\}), batch initialisation, **Reflection-proposed but user-gated** updates (proposals for project\_context, user\_preferences, guidance land in the review\_queue and require explicit approval), and **Reflection auto-applied** updates to pending\_items only (the one slot the worker may write without approval). This split implements a deliberate human-in-the-loop boundary: high-stakes steering slots require ratification, while low-stakes bookkeeping is autonomous.

### **3.6 Context Clusters**

Context Clusters are conversation-scoped, automatically generated topical groupings of episodic turns. The context\_clusters table holds id, name (LLM-generated short name), description (structured as DOMAIN: / CONTENT\_TYPE: / RECURRING\_ENTITIES: / SETTING\_OR\_CONTEXT: lines), tags (union of member turns' topic\_tags), conversation\_id FK, created\_at / updated\_at, and embedding Vector(384) — the **centroid** of member turn embeddings, renormalised to unit length after every member change. Membership is many-to-many through episodic\_cluster\_links(episodic\_id, cluster\_id) with a composite primary key, so a turn may belong to multiple clusters (soft multi-assignment).

The Clustering worker (workers/clustering.py::cluster\_turns, beat-scheduled every 30 minutes, bounded to MAX\_TURNS\_PER\_RUN = 25 unassigned turns per invocation) assigns each unassigned turn to the best existing cluster — combined score embedding\_similarity + 0.08 per shared entity (capped at 0.30) + tag overlap — or to a new cluster when no candidate clears SIMILARITY\_THRESHOLD = 0.6. The cluster name/description are regenerated only every NAME\_REGEN\_INTERVAL = 5 members to avoid naming churn. A separate merge\_similar\_clusters task (callable but **not** beat-scheduled) merges clusters in the same conversation with centroid similarity above MERGE\_SIMILARITY\_THRESHOLD = 0.90 (and a raw-similarity floor of 0.82).

### **3.7 Decay and archival mechanics**

Three independent decay workers, all beat-scheduled every **5 400 s (1.5 h)** for CYCLES\_PER\_DAY = 16 cycles/day, implement access-weighted decay.

**Episodic decay** (workers/decay.py) applies one of three per-cycle multipliers, gated by a 7-day recency cutoff (turns younger than seven days are immune):

- DECAY\_RATE\_UNACCESSED = 0.95^(1/16) ≈ 0.9968 (≈5 %/day) — access\_count = 0, non-creative.

- DECAY\_RATE\_ACCESSED = 0.98^(1/16) ≈ 0.9987 (≈2 %/day) — access\_count \> 0, non-creative.

- CREATIVE\_DECAY\_RATE = 0.99^(1/16) ≈ 0.9994 (≈1 %/day) — any turn tagged Creative\_&\_Media.

A **creative floor** clamps decay\_score to 0.3 for creative turns regardless of age or access, protecting long-form narrative from being summarised away. Turns whose decay\_score falls below ARCHIVE\_THRESHOLD = 0.1 are flipped to is\_archived = TRUE and immediately drop out of live retrieval (both legs filter on is\_archived = false). Turns that subsequently fall below COLD\_THRESHOLD = 0.05 are **moved** to the minimal cold\_storage table (id preserved, archived\_at, raw\_text, summary\_text, topic\_tags, timestamp) and physically deleted from episodic\_memory — an idempotent INSERT … ON CONFLICT (id) DO NOTHING followed by DELETE. Cold storage is a retention-only archive; no retrieval leg reads it. Bookmarks set decay\_immune = TRUE and are exempt from all decay.

**Codex decay** (workers/codex\_decay.py) multiplies codex\_edges.strength by 0.99^(1/16) for **every live edge (valid\_until IS NULL) — pending included (A3)**; previously only active edges decayed, which let a retrieval-reinforced pending edge inflate without ever entering the decay cycle. Active edges are demoted to confidence = 'pending' once strength \< DEMOTION\_THRESHOLD = 0.3. Demotion does not set valid\_until (the edge remains \*true\*, just unreinforced). **A3 garbage collection:** pending edges that decay below EXPIRY\_THRESHOLD = 0.1 without ever being corroborated or retrieved are expired (valid\_until = NOW()), so uncorroborated low-trust residue (e.g. grounding-rejected triplets) leaves the live graph instead of accumulating. Re-asserting a demoted edge reinforces it (strength += 1.0) and re-promotes once strength ≥ 2.0; retrieval reinforcement can also promote (§4.8).

**Procedural decay** (workers/procedural\_decay.py) is boolean, not numeric: SET is\_active = FALSE WHERE is\_active = TRUE AND last\_observed \< now() - 180 days AND reinforcement\_count \< 3. A pattern that has been promoted (≥3 reinforcements) is permanently protected from time-based decay. Patterns are never hard-deleted.

### **3.8 Batch summaries**

The batch\_summaries table (id, conversation\_id FK, start\_turn\_index, end\_turn\_index, summary\_text, embedding Vector(384), created\_at) holds compressed representations of decayed turns. The Batch Summariser (§8.7) groups turns with decay\_score \< 0.3 AND lossless\_flag = False AND is\_document = False by conversation, chunks them into 50-turn batches (skipping batches smaller than 5), and calls the background model under a strict preservation prompt (temperature=0.0, max\_tokens=500). The summary is the \*second life\* of a turn — it coexists with the still-live original until the original ages into cold storage, at which point the summary is the only live-retrievable trace. Retrieval is conversation-scoped (top-3 by cosine).


## **4. Codex Knowledge Graph (v2)**

Codex v2 is a temporally-versioned, controlled-vocabulary knowledge graph. Its central design lever is a three-bucket relation vocabulary (property, single-valued, multi-valued) that forces the extractor to commit to a specific cardinality per fact, which in turn drives well-defined expiry, reinforcement, and retrieval semantics.

### **4.1 Entity and edge tables**

codex\_entities (PK id — a deterministic UUIDv5 in CODEX\_NAMESPACE for new entities) holds canonical\_name (unique), aliases ARRAY(Text), tags ARRAY(Text), **entity\_type Text (A7 — structural node type; ICE is general-purpose, so the vocabulary spans all domains: universal person/place/organization/event/concept/object, coding/research software/function/class/file/module/dataset, academic/business document/product, creative character/location/item/creature/faction; inferred from an entity's relations via \_infer\_entity\_type, or set deterministically by the code graph E1b)**, **description Text (A7 — the enriched "note body", written by the reflection enrichment worker)**, properties JSONB, an auto-regenerated context\_payload Text, embedding Vector(384), and last\_updated. **A7 rich notes:** \_regenerate\_context\_payload assembles context\_payload as a bidirectional Obsidian-style "note" — the description (note body), then Properties, then **Links** (outgoing edges, strongest-first) *and* **Backlinks** (incoming edges — what points *to* this entity, previously invisible) — so an entity's full connection picture reaches retrieval in both directions. codex\_edges holds id, source\_id/target\_id FKs to entities, relation Text, strength Float (default 1.0; new non-property edges start at 1.0, property and replacement edges at 3.0), source\_batch UUID (the batch that asserted the edge), confidence Text ∈ \{pending, active\} (promoted to active at strength ≥ 2.0), **extraction\_confidence Float ∈ \[0,1\] (A3)** — how much the extraction itself is trusted: seeded by NER grounding at write time (0.9 grounded, 0.7 no-NER chunk, 0.35 grounding-rejected; legacy edges 1.0), raised to the max seen on corroborating re-extraction — valid\_from, and valid\_until — **\`NULL\` means currently true**; non-NULL means historically superseded. An edge's **effective trust** used by retrieval is strength × extraction\_confidence (strength carries usage dynamics; extraction\_confidence carries extraction trust). Two auxiliary tables complete the graph: codex\_events is an append-only event log (event\_type ∈ \{edge\_added, edge\_expired, edge\_strengthened, property\_updated, context\_appended\}, payload JSONB, batch\_source, compacted Boolean), and codex\_snapshots holds point-in-time full\_state JSONB produced by the compaction worker.

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

The Codex Extractor (workers/codex\_extractor.py::extract\_codex, Celery task, max\_retries=3, default\_retry\_delay=30, event-driven from the Post-Flight Evaluator only when lossless\_flag == True, with priority=True on bookmark) runs the following pipeline:

4. **Chunking (roadmap A1).** The turn is split into **sentence/code-line-aware chunks of ≈`CHUNK_TOKENS` = 550 tokens** with `OVERLAP_WORDS` = 50 words carried into each subsequent chunk. `_chunk_text` isolates fenced code blocks from prose (`_split_segments`), breaks each into atomic units that are never split across a chunk boundary — sentences for prose, non-blank lines for code (`_atomic_units`) — and greedy-packs those units up to the token budget; a single unit larger than the budget is hard word-split as a last resort. Code is counted at a heavier token density than prose (`_estimate_tokens(..., is_code=True)` takes the larger of a word- and char-based estimate). Short turns come back as a single chunk. Each chunk is extracted independently and triplets are concatenated then deduplicated. *(Rationale: a 3–4B extractor's attention dilutes past ~1k tokens; the previous ≈4,511-word windows caused mid-passage entity dropping and self-referential hallucinations like `fastapi uses fastapi`. The 550-token size is shared with the NER-grounding step so both run off one chunking pass. `MAX_EXTRACTION_TOKENS` = 6000 is retained only for import compatibility.)*

4b. **NER grounding — entity confirmation (roadmap A2).** Before the LLM call, each chunk is run through the shared CPU micro-NER (`extract_entities`, §4.6) to produce a confirmed entity list. When non-empty, that list is injected into the extraction prompt as a **CONFIRMED ENTITIES** block instructing the model to use only those as subjects/objects and introduce no new named entities. This splits the cognitive load — the LLM reasons about relations, not entity discovery.

5. **Background-model extraction.** The extractor calls the background model via `get_bg_client()` / `get_bg_model_name()` (mode-selected: dedicated vLLM :8002 Qwen2.5-3B, or shared Ollama; see §10.5). The prompt renders the relation vocabulary as grouped \# category: … headers, imposes six strict rules (use only individual relation words; canonicalise subjects/objects; property facts use the property relation as the relation; the relation must make logical sense; never output a category header as a relation; output only a JSON array), appends a code-specific sub-prompt when Software\_&\_Tech is in the topic tags, and appends the CONFIRMED ENTITIES block from 4b. Decoding: temperature=0.0, max\_tokens=500, timeout=30.0.

6. **Triplet validation.** (i) Markdown-fence strip; (ii) json.JSONDecoder().raw\_decode with a regex fallback r'\\\{\\s\*"subject"\\s\*:\\s\*"(\[^"\]+)"\\s\*,\\s\*"relation"\\s\*:\\s\*"(\[^"\]+)"\\s\*,\\s\*"object"\\s\*:\\s\*"(\[^"\]+)"\\s\*\\\}'; (iii) shape filter (all three keys present); (iv) vocabulary filter (relation in ALLOWED\_RELATIONS); (v) verb-phrase filter dropping triplets whose object is in \{"blush","laugh","cry","smile","angry","sad","happy","mad"\}; (vi) self-reference filter dropping triplets whose normalised subject equals its object (kills `fastapi uses fastapi`); (vii) **NER grounding → extraction confidence** (`_ground_triplets`, A2+A3) — each triplet's subject (and, for non-property relations, object) is checked against the confirmed entities by normalised-exact or token-subset comparison, and the result sets its confidence: CONF\_GROUNDED = 0.9 on pass, CONF\_REJECTED = 0.35 on fail (**stored anyway** — retrieval's trust gates keep it out of context until corroborated, and decay expires it if it never is), CONF\_UNGROUNDED = 0.7 when NER returned no entities for the chunk (nothing to ground against). Cross-chunk dedup keeps the highest confidence seen per (subject, relation, object). `handle_triplet` stores the value on new edges as extraction\_confidence and raises it to the max seen on corroborating re-extraction.

7. **Deduplication.** Keyed on the case-normalised (subject, relation, object) triple, so multi-chunk repeats across overlap windows collapse.

8. **Write + reconcile (A6).** Each surviving triplet is passed to handle\_triplet, which now runs a **bounded self-correction loop** before applying the fixed write rules (non-property relations only — property relations already supersede). `check_conflict` is a cheap deterministic pre-filter: it queries the graph only when the relation has a known **antonym** (friend↔enemy, married\_to↔is\_divorced\_from, endorses↔criticises, …) or when a **multi-valued** relation coincides with a **supersession cue** in the turn text ("migrated off", "no longer", "switched from", "replaced", …); the ~95% of triplets with neither take a dict-lookup fast path. On a hit, `reconcile_conflict` resolves it: **antonym reversals are deterministic** — the newly-asserted state expires its opposite for that pair, *no LLM* — while **ambiguous supersessions** ("migrated off X" vs "considered migrating") are the *only* case that consults a model: a bounded one-word reconciler (`make_llm_reconciler`, background model, max\_tokens=5) returns expire\_old / keep\_both / reject\_new; anything else, or no reconciler, falls to a `review_queue` row with the new edge kept (never auto-expire on a guess). This deliberately keeps a small model's write-authority over the source-of-truth minimal (the notes' full autonomous "Memory Maintenance Agent" is Track D). `handle_triplet(turn_text, reconciler)` and the standalone `check_conflict`/`reconcile_conflict` are callable units so the Track D agent can later drive the loop with its own reconciler. *(The review-queue fallback is not user-visible until the frontend renders it — F2.)*

9. **Idempotency.** idempotency\_key = sha256("codex:" + batch\_id); on a hit the task returns immediately. The key insert shares the worker's transaction, so a crash before commit leaves no key (retry re-runs) and a crash after commit leaves the key (retry is a no-op) — turning Celery's at-least-once delivery into effectively-once side effects.

### **4.5 Relation-aware retrieval and enumeration (A4 — replaces MERA)**

*MERA (retrieval/mera.py) was removed in the post-paper A4 rework: it scored −0.21 in the buildup ablation (loose triggers, an LLM tag/relation mapping that often missed the DB's actual tags, and a flat 15-entity payload dump). Its capability — answering category/enumeration and relation-shaped queries from the graph — was re-homed into the primary Codex leg as follows.*

**Relation detection** (`_detect_relations`) maps the prompt onto the controlled relation vocabulary through two LLM-free channels: (1) a **lexical channel** — a relation's own content words appearing in the prompt after crude two-sided stemming ("who inspired X" → inspired\_by, "who is X married to" → married\_to); multi-word relations require all content words, single-word relations one; and (2) an **embedding channel** — true-cosine similarity between the (re-normalised) prompt embedding and once-per-process cached gloss embeddings of all ~200 relations, top-k=5 above a 0.45 floor, for paraphrases ("who is X's wife"). Empirically the joint set is recall-oriented and *deliberately not trusted alone*: neutral prompts also score ~0.69 absolute against random glosses, so detected relations only ever act **jointly** — paired with matched entities (fact surfacing, §4.8) or with explicit enumeration cues (below). Precision comes from the join, never the detector.

**Enumeration** (`_codex_enumeration`) fires only for entity-less prompts when an explicit enumeration cue is present (list, all, who are, what are, every, each, which, name the, tell me about, enumerate) **and** a grounded signal exists: (a) a prompt token (singularised) matching an actual entity tag ("list all the characters" → tag character) — surfacing up to 8 tagged entities' payloads; and/or (b) a detected relation — surfacing up to 15 trust-gated edges as explicit `[Fact: a --rel--> b]` lines. Both channels honor project scope (§4.8) and the A3 trust floor. No LLM call, no separate module — score 1.0, no reinforcement (enumeration edges are not query anchors).

### **4.6 Micro-NER model**

The micro-NER model (classifier/ner\_model.py) is a BIO tagger — a 3-layer MLP 384 → 128 → 64 → 3 with two ReLU + Dropout(0.2) blocks — that classifies per-token embeddings (not token IDs) from the same Qwen/Qwen3-Embedding-0.6B embedder into B-ENT, I-ENT, O. Each token string is embedded **in isolation** (context-free): the model learns which tokens tend to be entities, but cannot use surrounding words to disambiguate — a known limitation (good recall, noisy boundaries; roadmap A9 tracks a context-aware rework). Operating on embeddings lets it generalise across the embedding space (e.g. recognise a misspelling close to a known entity) at the cost of requiring the embedder at inference. A shared singleton in retrieval/ner\_utils.py::extract\_entities consolidates both the orchestrator's live-prompt NER and the clustering worker's full-chapter NER onto one loaded model; if the .pt file is missing it falls back to a regex \\b\[A-Z\]\[a-zA-Z\]\{2,\}\\b minus a stoplist (The, This, User, Assistant, Chapter, …). **Output post-processing (roadmap A2 NER cleanup):** decoded entity spans are snapped to whole-word boundaries (`_snap_to_words`, fixes subword truncation like `Pyd`→`Pydantic`), trimmed of leading/trailing function words (`_clean_entity` / `_EDGE_TRIM`, e.g. `on Pydantic`→`Pydantic`), filtered of pronoun/boilerplate junk (`_NER_STOP`), and deduplicated. Empirically ~95% entity recall on a mixed prose/technical probe set; residual verb-led bleed (`uses PostgreSQL`) and descriptor false-positives (`fire mage`) are absorbed by A2's token-subset grounding and await the context-aware rework. The training pipeline (scripts/ner/\*) is offline-only: extract\_turns combines simulation + labelled + synthetic prompts; label\_entities calls mattbucci/gemma-4-12B-AWQ at SGLang port 8003 to extract verbatim entity strings; generate\_bio aligns entities to token offsets (longest-first, no-overwrite) producing BIO labels; train\_ner trains with Adam(lr=1e-3), BATCH\_SIZE=16, EPOCHS=10, CrossEntropyLoss with inverse-frequency class weights clamped at 100.0, early stopping PATIENCE=3.

### **4.7 Vector fuzzy matching for entity resolution**

The Codex retrieval leg resolves prompt-extracted entities to stored CodexEntity rows in three stages (A4): (1) \_match\_entities\_by\_similarity(threshold=0.85) — embeds the prompt entity strings, linear-scans every entity with a non-null embedding, computes the dot product, and takes the best-scoring entity above 0.85; each prompt entity matches at most one stored entity and vice versa (seen\_ids); with an inline exact canonical\_name / aliases.any(norm) fallback per string. (2) The base-class \_match\_entities\_exact (exact/alias only, no vectors) — used when use\_fuzzy\_match=False (the ablation fuzzy\_match flag). (3) **Payload descriptor fallback** \_match\_entities\_by\_payload — when name-based matching finds nothing, content words (≥4 chars, minus a small stoplist) from the NER strings are ILIKE-searched inside context\_payload, ranked by hit count, top 2 accepted: "main fortress" resolves to The Obsidian Citadel because its payload mentions "fortress". This closes part of the semantic-vs-lexical gap without a schema change (payload *embeddings* remain future work).

### **4.8 Retrieval and event-sourced compaction**

The Codex retrieval leg (\_codex\_graph): (i) extracts prompt entities via the shared NER; (ii) resolves them by vector fuzzy matching; (iii) optionally restricts to a conversation scope (entities whose codex\_events.batch\_source appears in the conversation's batch set); (iv) **trust-gated BFS traversal (A3)** to max\_depth = 3 over edges where valid\_until IS NULL, gated on **effective trust = strength × extraction\_confidence** (`_edge_trust`): a matched entity's *direct* edge expands (and is collected as a query *anchor*) only when trust ≥ CODEX\_DIRECT\_TRUST\_FLOOR (0.5) — so grounding-rejected low-confidence edges sit in the graph but never reach context or gain strength until corroborated — and deeper hops require trust ≥ CODEX\_DEEP\_STRENGTH\_FLOOR (1.0), so weak/decayed/low-confidence edges no longer pull the whole 3-hop neighbourhood into context; appends "\[Entity: \{canonical\_name\}\]\\n\{context\_payload\}" per visited entity (payloads themselves list an entity's edges strongest-first); (v) returns a single ContextFragment of type codex whose score is **graded 1.0–1.5** from the mean effective trust of the matched entities' edges (replacing the old binary 1.5× active-edge boost). **Retrieval-reinforcement (A3):** `_reinforce_codex_edges` bumps the anchor edges' strength by CODEX\_REINFORCE\_INCREMENT (0.15, capped at 10.0) each time they're surfaced — the Codex analog of episodic access/decay strengthening (§6.5), balanced by codex\_decay (which decays **all** live edges, closing the loop) — and promotes a pending edge to active once strength ≥ CODEX\_PROMOTE\_STRENGTH (2.0) **and** extraction\_confidence ≥ CODEX\_PROMOTE\_MIN\_CONFIDENCE (0.5), so an edge can earn activation through repeated retrieval usefulness but a low-trust extraction cannot promote on popularity alone. **Multi-fragment representation (A10):** the leg now emits **one ContextFragment per anchor entity** (its payload + trust-gated neighborhood + its own relation facts), each scored by *that* anchor's direct-edge trust (1.0 + graded trust + relation-overlap boost), rather than concatenating the whole traversal into a single blob. Combined with the round-robin budget (§6.5), this fixes the structural under-representation that made codex only 3.3% of fragments — it could previously occupy at most one budget slot no matter how many relevant entities it found. Traversal shares one `visited` set across anchors, so an entity reachable from two anchors is rendered once. Enumeration (§4.5) likewise emits per-entity fragments plus one facts fragment. **Relation-aware fact surfacing (A4):** when \_detect\_relations (§4.5) finds relations relevant to the prompt, edges where a *matched entity* participates in a *detected relation* (either direction, trust-gated, strongest-first, up to 10) are appended as explicit `[Fact: a --rel--> b]` lines, join the reinforcement anchors, and add a RELATION\_OVERLAP\_BOOST (+0.25) to the fragment score — the entity∩relation joint hit is the precision anchor, lifting the score ceiling to 1.75. **Project scope (A5):** under a project-scoped conversation the leg resolves scope once (\_codex\_scope\_sets → the conversation's batch set + the entity set touched by those batches) and honors it *throughout*: anchor entities are filtered (as before), traversal now also (a) drops edges whose source\_batch is outside the conversation, (b) never expands into entities outside the conversation's entity set, and (c) renders each visited entity's payload **on the fly from the conversation's own edges** instead of the stored global context\_payload — so a shared entity ("ice") no longer leaks facts from other conversations. Unscoped (auto/none) conversations keep global traversal and stored payloads. **Grounded query expansion (A4, the accepted replacement for HyDE):** the codex leg runs first in retrieve(); the canonical names + aliases of whatever entities it matched (≤8 terms) are appended to the BM25 search prompt, so lexical search finds turns that use the full or alternate name — nothing is generated, so nothing can be hallucinated; the vector leg keeps the original embedding. If the NER extracts no entities, the enumeration path (§4.5) answers category queries directly from the graph. *Note: pending edges are traversed here (subject to the trust gates) — retrieval never filtered on confidence='active' (only the old score boost did).* Full per-edge (rather than per-blob) scoring awaits the multi-fragment representation (roadmap A10).

The codex\_events append log is compacted by workers/compaction.py::compact\_entities (EVENT\_THRESHOLD = 100 uncompacted events per entity): for each entity crossing the threshold, the worker replays its uncompacted events in timestamp order, maintaining a set of "rel:target\_id" signatures (edge\_added adds, edge\_expired discards), writes a CodexSnapshot(full\_state = \{active\_edges, context\_payload, properties, aliases\}) with the last\_event\_id, and marks the consumed events compacted = True. This is textbook event sourcing: the live edge table is the current state, the event log is the audit trail, and the snapshot table is the compaction output that bounds replay cost. The compaction worker is **not** beat-scheduled — it must be invoked manually or via a sentinel rule with action\_type = "schedule\_worker".


## **5. Procedural Memory**

Procedural memory captures recurring behavioural patterns ("the user always X after Y") that, once crystallised, can be surfaced to gate or enrich future generation. Its lifecycle is detection → reinforcement → promotion → decay.

### **5.1 Pattern extraction**

The Procedural Extractor (workers/procedural\_extractor.py::extract\_procedural, Celery task, max\_retries=3, default\_retry\_delay=30, event-driven from the Post-Flight Evaluator **unconditionally** on every turn) calls the background model with a one-sentence pattern-detection prompt (temperature=0.0, max\_tokens=80, timeout=30.0). If the model returns NONE, the task exits. Otherwise it embeds the proposed pattern, queries procedural\_memory by cosine similarity LIMIT 1, and branches:

- **Match (\`sim \> 0.85\`).** Reinforce the existing pattern: reinforcement\_count += 1, last\_observed = now. If reinforcement\_count ≥ 3 AND confidence\_score \< 0.8, promote to confidence\_score = 0.8, is\_active = True.

- **No match.** Insert a new pattern with confidence\_score = 0.3, is\_active = False, reinforcement\_count = 1, source\_batch\_ids = \[batch\_id\].

The Reflection worker's \_crystallize\_patterns step runs the same workflow at session granularity, feeding on cross-turn patterns observed over the last 200 turns of each conversation.

### **5.2 Trigger-condition gating for retrieval**

Even active patterns are filtered at retrieval time by \_procedural\_trigger\_match: if the pattern's trigger\_conditions JSONB is non-empty, its topic\_tags and intent\_tags must each intersect the current classification's tags; an empty trigger\_conditions always passes. In the current implementation extractors always write trigger\_conditions = \{\}, so this gate is forward-looking infrastructure — patterns are currently gated only by the intent gate (§5.4) and the conversation-scope gate, not by trigger conditions.

### **5.3 Decay and confidence promotion**

workers/procedural\_decay.py::decay\_procedural\_patterns (beat-scheduled every 1.5 h) runs a single boolean deactivation: SET is\_active = FALSE WHERE is\_active = TRUE AND last\_observed \< now() - 180 days AND reinforcement\_count \< 3. The two conditions are conjunctive — a pattern that has been promoted (≥3 reinforcements) is permanently immune to time-based decay. Confidence promotion is \*not\* in the decay worker; it happens in the extractors at the moment reinforcement\_count crosses 3. Patterns are never hard-deleted, only deactivated.

### **5.4 Retrieval**

The procedural retrieval leg (\_procedural\_lookup) is the most heavily gated leg. A **hard intent gate** activates it only when classification.intent\_tags intersects \{"Strategic\_Planning", "Generation", "Open\_Exploration"\}; otherwise it returns \[\]. When active, it runs a vector cosine top-5 over procedural\_memory WHERE embedding IS NOT NULL AND is\_active = true, applies the conversation-scope gate (patterns whose source\_batch\_ids intersect the conversation's batch set) and the trigger-condition gate, and returns up to five ContextFragments of type procedural, scored by raw cosine similarity.


## **6. Hybrid Retrieval Orchestrator**

The HybridRetrievalOrchestrator (retrieval/orchestrator.py) is the core of the pre-flight phase. It runs six retrieval legs in parallel, fuses their outputs with weighted Reciprocal Rank Fusion, applies a battery of post-fusion transforms, and returns a token-budgeted list of ContextFragments to the Prompt Assembler.

Before any retrieval leg executes, the orchestrator receives a classification that may have been altered by two layers of LTM bias: (1) the classifier’s own \_apply\_hard\_overrides (§2.4) coerces Creative\_&\_Media and referential Software\_&\_Tech prompts to Long\_Term\_Memory, and (2) an API‑level bias in main.py upgrades Zero\_Shot to Long\_Term\_Memory whenever a conversation has more than ten turns or the classifier’s maximum confidence is below 0.95. Consequently, retrieval runs for almost every turn in a long conversation, regardless of the classifier’s initial belief.

### **6.1 The six retrieval legs**

10. **BM25 (full-text)** — \_bm25\_episodic. Postgres ts\_rank over to\_tsvector('english', coalesce(raw\_text,'')||' '||coalesce(summary\_text,'')) against an OR-joined to\_tsquery of the top-30 stop-word-filtered prompt tokens. Hard filters decay\_score \> 0.2 AND is\_archived = false, optional conversation\_id and cluster\_ids scope. LIMIT 100, ordered by ts\_rank DESC.

11. **Vector** — \_vector\_episodic. pgvector cosine distance **with decay weighting**: (1 - (embedding \<=\> :prompt\_embedding)) \* COALESCE(decay\_score, 1.0). Same visibility invariant, LIMIT 100. The decay multiplier is what distinguishes this leg from a pure semantic search: a high-similarity but decayed turn is down-ranked.

12. **Codex graph traversal** — \_codex\_graph. NER → three-stage entity resolution (vector fuzzy 0.85 / exact / payload-descriptor fallback) → trust-gated depth-3 BFS (deep hops require trust ≥ 1.0, direct ≥ 0.5) with A5 project-scope isolation, relation-aware fact surfacing with the entity∩relation overlap boost, retrieval-reinforcement of anchor edges, and grounded query expansion feeding the BM25 leg (§4.5, §4.8). Entity-less category queries answered by the enumeration path (re-homed MERA). Score graded 1.0–1.75.

13. **Procedural** — \_procedural\_lookup (§5.4). Hard intent gate, vector top-5, conversation-scope and trigger-condition gates.

14. **RAG** — \_rag\_lookup. Triple-gated: requires context\_reliance == "Long\_Term\_Memory" AND intent\_tags ∩ \{Factual\_Retrieval, Analysis\_&\_Summarization\} ≠ ∅ AND the prompt contains one of \["document", "pdf", "reference", "manual", "guide"\]. On pass, SELECT chunk\_text, 1 - (embedding \<=\> :prompt\_embedding) AS score FROM rag\_chunks ORDER BY score DESC LIMIT 5. RAG is intentionally **global**, not conversation-scoped.

15. **Batch summaries** — \_batch\_summary\_lookup. Conversation-scoped top-3 from batch\_summaries by cosine similarity.

### **6.2 Query rewriting: grounded expansion in production, HyDE rejected**

A \_hyde\_rewrite method exists in the orchestrator but is **commented out** in the production retrieve() path (and note it is query *reformulation*, not actual HyDE — it never fabricates a hypothetical answer document). It is only reachable through the ConfigurableOrchestrator with the hyde=True ablation flag (default False). The post-paper review (roadmap P0.1) rejected shipping real HyDE: it would fabricate answers with the small background model over the user's *private* history, and hallucinated specifics corrupt the noise-sensitive BM25 leg. Production instead uses **grounded query expansion (A4)**: the BM25 search prompt is expanded with the canonical names + aliases of the entities the codex leg actually matched (§4.8) — expansion terms come from the graph, not a generator.

### **6.3 Dynamic leg weighting**

After the legs run, retrieve() blends a per-leg alpha map. Base weights are \{"bm25": 0.8, "vector": 1.0, "codex": 0.5, "procedural": 0.2, "rag": 1.0\}. Five **intent profiles** override the four non-RAG legs:

| **Profile (active intents)** | **vector** | **bm25** | **codex** | **procedural** |
| - | - | - | - | - |
| **Factual\_Retrieval, Utility\_Formatting** | 1.2 | 0.8 | 0.1 | 0.1 |
| **Troubleshooting, Strategic\_Planning** | 1.0 | 0.8 | 0.3 | 1.2 |
| **Generation, Ideation, Open\_Exploration** | 0.6 | 0.6 | 1.2 | 0.1 |
| **Emotional\_Processing, Analysis\_&\_Summarization, Decision\_Making** | 1.1 | 0.6 | 0.9 | 0.0 |
| **Casual\_Banter, Null\_Noise** | 0.5 | 0.2 | 0.0 | 0.0 |


For each active intent, profile\_weights\[leg\] / num\_active is added to blend\_weights\[leg\]; unknown intents fall back to base\_weights / num\_active. Two **cumulative topic overrides** then apply: Creative\_&\_Media ⇒ codex += 0.3; Software\_&\_Tech ⇒ procedural += 0.4. Every weight is floored at 0.0. The ConfigurableOrchestrator does **not** redefine these tables — it only toggles legs on/off; the blend still runs unmodified, so an empty leg simply contributes nothing to fusion.

### **6.4 Reciprocal Rank Fusion (RRF)**

\_apply\_rrf(legs, alpha\_map, k=60) sorts each leg's fragments by native score, then accumulates per-fragment RRF scores keyed on the SHA-256 of frag.text:

*score\_RRF(f) = Σ\_ℓ ∈ legs α\_ℓ / (k + rank\_ℓ(f)),  k = 60*

where α\_ℓ is the blended weight (defaulting to 1.0 for unknown legs) and rank\_ℓ(f) starts at 1. Fragments are deduplicated \*during\* fusion — the first occurrence is registered and subsequent occurrences add to its RRF score. Output is sorted by RRF score descending.

### **6.5 Post-fusion processing**

After RRF, the pipeline runs five sequential transforms:

16. **\`\_apply\_bonuses\`** (additive, applied as a multiplier score \* (1 + bonus)):

- **Keyword boost** +1.0 if any prompt keyword (or its singular form via kw.rstrip('s')) appears in text.lower().

- **Length bonuses** (mutually exclusive): +1.5 if word\_count \> 800, +0.5 if \> 400, -0.7 if \< 80.

- **Recency boost** (episodic only, **skipped** for Creative\_&\_Media turns — recent meta turns are noise for creative work): +1.0 if the fragment's source turn is in the top-10 % most-recent of its conversation (recency\_pct \< 0.10), +0.5 if top-30 %; requires total \> 20 turns.

- **Soft meta-discussion downweight** -0.45 (classifier-driven, not string matching): fires only when wants\_narrative\_fact (the intent set intersects NARRATIVE\_FACT\_INTENTS = \{Factual\_Retrieval, Decision\_Making\}) and the source turn's stored intent\_tags intersect META\_LEANING\_INTENTS = \{Analysis\_&\_Summarization\}.

- **Clamp** bonus ∈ \[-0.9, MAX\_TOTAL\_BONUS\_MULTIPLIER\] (effective multiplier range \[0.1×, 5.0×\]).

17. **Pre-RRF bonuses** in \_rows\_to\_fragments (episodic rows only, applied before RRF so they influence per-leg ranking): bookmark ×1.5, time-recency additive +0.25·(1 - age\_hours/720h), turn-recency additive +0.1·(1 - newer\_count/turn\_count), and a dynamic word cap (default 500 words; 1500 if a prompt keyword matches; uncapped for is\_document).

18. **Sort** by score descending.

19. **\`\_session\_diversify\`** (max\_per\_conversation = 3): iterates sorted fragments; the active conversation is uncapped; foreign conversations are capped at 3 each; fragments with no conversation\_id (RAG, Codex, procedural) are always kept.

20. **\`\_deduplicate\`**: SHA-256 of text, first occurrence wins (defensive second pass after RRF).

21. **\`\_enforce\_token\_budget\`** (A10): two-phase packing against self.max\_retrieval\_tokens. Phase 1 — **leg-diversity guarantee**: the highest-scoring fragment of each leg (best\_per\_leg\[source\_type\]) is added first, sorted by score, while it fits. Phase 2 — **round-robin-with-slack across legs** (replaces the old flat greedy fill): each round every leg contributes its next-best fragment (highest-scoring leg first), so a fragment-rich leg (episodic emits dozens) no longer soaks the entire remainder while codex/procedural emit few — yet when other legs are sparse, exhausted legs drop out and their share flows to the rest, so the budget still fills fully. This works with the A10 codex change below (the codex leg now emits *multiple* fragments — one per anchor entity — so it has more than one fragment to contribute per round, fixing its structural under-representation).

22. **\`\_strengthen\_retrieved\`**: for each episodic fragment, access\_count += 1 and decay\_score = min(1.0, decay\_score + 0.15) — retrieval acts as a partial reversal of decay.

### **6.6 Cluster-scoped retrieval**

Before the legs run, \_relevant\_cluster\_ids(prompt\_embedding, classification, conversation\_id, top\_k=10) identifies the relevant clusters: it pulls the top-30 clusters by centroid cosine similarity, re-scores each as combined = sim + 0.3 \* tag\_overlap + 0.15 \* name\_sim (where tag\_overlap is the count of overlapping topic tags and name\_sim is the cosine between the prompt embedding and a freshly embedded name + " " + description), and returns the top-10 cluster IDs — unless the best combined score is below 0.50, in which case it returns \[\] and the orchestrator falls back to global search. The scoped cluster IDs are injected into scope\["cluster\_ids"\], and both episodic legs apply a filter that admits any turn linked to one of the scoped clusters **or** any turn with no cluster links yet (so unassigned turns stay retrievable). The ConfigurableOrchestrator can disable this with cluster\_restrict=False.

### **6.7 Dynamic token budget**

set\_budget\_from\_turn\_count(turn\_count, total\_tokens, classification) computes the split between the recent-turns window and the retrieval budget:

- TOTAL\_CONTEXT\_BUDGET = 23\_000, OVERHEAD\_RESERVE = 1\_800, so available = 21\_200.

- The **recent-window fraction** starts from a length-based base (0.3 \<10 turns, 0.2 \<50/\<200, 0.15 \<500/else), is reduced by token-density (-0.15 if avg\_tokens\_per\_turn \> 3000, -0.10 if \> 1500, -0.05 if \> 800), shifted by intent (-0.10 for Factual\_Retrieval/Troubleshooting/Analysis\_&\_Summarization; +0.10 for Emotional\_Processing/Casual\_Banter), shifted by topic (+0.05 Creative\_&\_Media; -0.05 Software\_&\_Tech; +0.05 Social\_&\_Relationships/Lifestyle\_&\_Health), and clamped to \[0.05, 0.85\].

- recent\_budget = int(available \* fraction), raw\_retrieval = available - recent\_budget.

- A **growth cap** prevents long-tail conversations from over-allocating to retrieval: \<30 turns → 2\_000 + 150\*n; \<100 → 5\_000 + 100\*(n-30); \<500 → 10\_000 + 30\*(n-100); else raw\_retrieval. retrieval\_budget = min(raw\_retrieval, growth\_cap).

- Leftover budget (the gap between raw\_retrieval and growth\_cap) is **intentionally unused** — this is the lever that keeps ICE token-efficient relative to a vector-only baseline. The ConfigurableOrchestrator with dynamic\_budget=False reverts to fixed max\_retrieval\_tokens = 8000, recent\_token\_budget = 4000.

### **6.8 Wide-net fallback**

When classification.max\_confidence \< settings.confidence\_fallback\_threshold (default 0.75), retrieve() short-circuits to \_wide\_net\_fallback: a full vector scan with **no conversation filter and no cluster filter** (only embedding IS NOT NULL AND is\_archived = false AND decay\_score \> 0.2, LIMIT 100), plus unscoped \_codex\_graph and \_rag\_lookup, fused with single-leg RRF (α = 1.0), post-processed identically, and truncated to a **hardcoded 2 000-token ceiling** — much tighter than the dynamic budget — to keep the response focused when the system is unsure what the user wants.

### **6.9 Feature Toggling for Ablation Studies**

Every retrieval leg and post‑processing step can be independently enabled or disabled through a ConfigurableOrchestrator (a subclass of HybridRetrievalOrchestrator). An overrides dictionary—keyed by leg name ("bm25", "vector", "codex", "procedural", "batch\_summary", "rag", etc.) and post‑processing step ("rrf", "hyde", "cluster\_restrict", "session\_diversify", "dynamic\_budget", "keyword\_boost", "recency\_boost")—controls whether each component participates in retrieval. Setting a key to False causes the corresponding leg to return an empty list or the corresponding transform to be skipped. This mechanism is used by the ablation experiments reported in the evaluation section, where features are added cumulatively from a bare vector‑only baseline to the full ICE stack, measuring the incremental contribution of each architectural addition.


## **7. Prompt Assembly**

The Prompt Assembler (api/prompt\_assembler.py) concatenates the retrieved fragments with persistent memory slots, recent turns, and the live user message into a list of chat-completion messages. Its ordering is deliberately **stable-prefix** to maximise KV-cache reuse across consecutive turns.

### **7.1 Stable-prefix ordering**

assemble\_prompt returns messages in this exact order:

23. **System message** — a fixed ~150-word instruction block (role of history vs the live question, step-by-step reasoning, fact-change tracking, specificity) with an inline === PERSISTENT CONTEXT === block rendering each active slot as \[SLOT\_NAME\]\\n\{content\}.

24. **Recent turns** — alternating user/assistant pairs from get\_recent\_turns(db, conv\_id, max\_tokens=max\_recent\_tokens, max\_count=10).

25. **Retrieved-context block** — a single user message with header === RETRIEVED CONTEXT === (optionally (clusters: Cluster A, Cluster B, …) when scope\["cluster\_ids"\] is populated), then "\\n\\n".join(f.text for f in retrieved\_fragments).

26. **Acknowledgment** — a single assistant message: "Understood — I have the background context. What would you like to know?" — a deliberate boundary marker so the model treats the \*final\* user message as the live question rather than another history turn.

27. **Live user question** — the actual prompt to answer.

Because the system message, slots, and most of the recent-turns prefix change slowly (only the recent window slides), most of the prefix K/V tensors are reusable across consecutive requests.

### **7.2 Per-component rendering**

- **System message + slots** — only slots with is\_active AND content are rendered; no per-slot token cap inside assemble\_prompt (the cap is enforced upstream by memory\_slots.py).

- **Recent turns** — per-turn word caps are dynamic: 80 words when max\_tokens ≤ 1000, 150 when ≤ 3000, else min(500, max(100, max\_tokens // max(1, len(turns)) // 2)). Each turn is split into user/assistant parts by parsing the literal "User: " / "\\n\\nAssistant: " markers in raw\_text; parts exceeding the cap are trimmed with \_trim\_words (appends …). Greedy fill until tokens\_used + next\_pair\_tokens \> max\_tokens.

- **Retrieved-context block** — fragments are passed in already budgeted by the orchestrator's \_enforce\_token\_budget; the assembler just joins them with \\n\\n.

### **7.3 Emotional / creative bypass**

There is no separate emotional/creative branch inside assemble\_prompt. The bypass is **upstream**, in the classification and retrieval layers: creative turns force Long\_Term\_Memory (§2.4), skip the recency boost (§6.5), and add +0.3 to the Codex leg weight (§6.3); emotional turns shift the token-budget fraction toward the recent window (§6.7). The assembler renders whatever the orchestrator decided.

### **7.4 Token budget enforcement during assembly**

Assembly is a two-layer budget process. The orchestrator's \_enforce\_token\_budget packs fragments to max\_retrieval\_tokens (§6.5). Then api/main.py recomputes total\_words against int(0.9 \* 4096 / 1.33) ≈ 2 770 words and, if over budget, iteratively pops **procedural** fragments first, then **episodic** fragments, reassembling after each pop. RAG and Codex fragments are preserved — they are the hardest to recompute and the most likely to carry the answer.


## **8. Background Worker Cluster**

ICE's long-term-memory consolidation runs out-of-band through a Celery worker fleet backed by Redis. All workers share a single implicit default queue (celery — there is no task\_routes), and all GPU-touching workers gate on is\_gpu\_busy() (and additionally is\_user\_active() in shared mode) at task entry, retrying with a fixed countdown when the GPU is saturated.

### **8.1 Celery + Redis infrastructure and GPU gating**

workers/celery\_app.py constructs Celery("ice\_workers", broker=settings.redis\_url, backend=settings.redis\_url) with JSON serializers and UTC timezone, including 13 worker modules. The **beat schedule** has eight entries:

| **Beat entry** | **Task** | **Schedule** |
| - | - | - |
| **cluster-turns** | clustering.cluster\_turns | 1 800 s (30 min) |
| **monitor-sentinels** | sentinel\_monitor.monitor\_sentinels | 1 800 s |
| **decay-episodic** | decay.apply\_decay | 5 400 s (1.5 h) |
| **decay-codex** | codex\_decay.decay\_codex\_edges | 5 400 s |
| **decay-procedural** | procedural\_decay.decay\_procedural\_patterns | 5 400 s |
| **reflection** | reflection.run\_reflection | 7 200 s (2 h) |
| **batch-summarize** | batch\_summarizer.batch\_summarize | 7 200 s |
| **fine-tune-weekly** | fine\_tune.fine\_tune\_classifier | crontab Mon 04:00 UTC |


Two tasks — compaction.compact\_entities and clustering.merge\_similar\_clusters — are **callable but not beat-scheduled**; they must be invoked manually or via a sentinel rule. The beat schedule is intentionally bimodal: lightweight bookkeeping every 30 min / 1.5 h, GPU-heavy reflection and summarisation every 2 h, fine-tuning weekly.

GPU gating (workers/gpu\_check.py) polls nvidia-smi --query-gpu=utilization.gpu (5-second timeout) and treats any single GPU above GPU\_UTIL\_THRESHOLD = 20 as busy. In shared mode, is\_gpu\_busy() always returns False and yielding is cooperative via is\_user\_active(), which checks the Redis key ice:last\_chat\_completed (set by the proxy after every turn commit) against a 10-second idle window. Retry countdowns are fixed (not exponential): 15 s post-flight, 30 s extractors, 60 s decay/sentinel/compaction/reflection, 120 s batch-summariser/cluster-merge.

### **8.2 Post-Flight Evaluator**

post\_flight.evaluate\_turn(batch\_id, prompt, response, conversation\_id, model\_used) is event-driven, enqueued from store\_turn\_async after each turn commit (max\_retries=5, default\_retry\_delay=15, idempotency key sha256(batch\_id)). It runs three analyses:

- **Lossless detection** (is\_lossless) — resolves the "Asymmetrical Value Problem" of which turns are worth preserving verbatim: returns True if the text contains a code fence, exceeds 500 words, or contains ≥3 proper nouns after stripping sentence-start capitals. A **force-lossless override** sets lossless=True, inject\_raw=True, summary=None when topic\_tags contains Creative\_&\_Media or intent\_tags contains Emotional\_Processing.

- **Document detection** — raw\_words \> 2000 AND assistant\_count \< 3 ⇒ is\_document = True, inject\_raw = True.

- **Summary generation** — background client, temperature=0.0, max\_tokens=200, timeout=30.0, under a preservation-prompt system message (named entities, numbers/lists/categories, code-snippet descriptions; no pleasantries, no speculation, no meta-commentary).

After commit, it dispatches extract\_codex.delay(...) (only if lossless) and extract\_procedural.delay(...) (always). If the Celery enqueue itself fails (broker down), store\_turn\_async appends a JSONL entry to data/post\_flight\_buffer.jsonl as a write-ahead log for later replay.

### **8.3 Codex Extractor**

When the task is dispatched from the bookmark endpoint, it receives priority=True, which causes it to skip the GPU‑utilisation gate (is\_gpu\_busy()) and the shared‑mode user‑activity gate. This is the only code path that overrides the yield‑to‑user constraint (INV‑5), ensuring that a user‑bookmarked turn is immediately processed into the knowledge graph.

codex\_extractor.extract\_codex(batch\_id, model\_used, priority) (§4.4) — Celery task, max\_retries=3, default\_retry\_delay=30, idempotency key sha256("codex:" + batch\_id), event-driven from evaluate\_turn only when lossless\_flag == True. GPU-gated (skipped when priority=True). Reads EpisodicMemory by batch\_id, calls extract\_triplets, validates/deduplicates, and calls handle\_triplet per triplet.

### **8.4 Procedural Extractor**

procedural\_extractor.extract\_procedural(batch\_id, model\_used) (§5.1) — Celery task, max\_retries=3, default\_retry\_delay=30, idempotency key sha256("procedural:" + batch\_id), event-driven from evaluate\_turn unconditionally. Pattern detection → similarity matching → reinforcement or insertion.

### **8.5 Decay Workers**

Three independent schedulers (§3.7), all every 5 400 s, max\_retries=2, default\_retry\_delay=60: decay.apply\_decay (episodic, access-weighted with creative floor, archive at 0.1, cold-storage move at 0.05), codex\_decay.decay\_codex\_edges (edge strength decay, demotion at 0.3), procedural\_decay.decay\_procedural\_patterns (boolean deactivation after 180 days if \< 3 reinforcements).

### **8.6 Reflection Worker**

reflection.run\_reflection (beat every 2 h, max\_retries=2, default\_retry\_delay=60) runs a five-prompt cascade over the 200 most recently active conversations (each with ≥10 turns, last 200 turns, oldest-first):

28. **\`\_synthesize\_session\`** (SUMMARY\_PROMPT) — emits \{topics\_covered, decisions\_made, unresolved\_items, entities\_updated, patterns\_observed\}, inserts a SessionSummary row, and appends unresolved items to the pending\_items memory slot directly (updated\_by = "reflection\_worker").

29. **\`\_crystallize\_patterns\`** (CRYSTALLIZATION\_PROMPT) — embeds each detected pattern, matches against procedural\_memory (sim \> 0.85 ⇒ reinforce; else insert at confidence=0.3, is\_active=False), promoting at reinforcement\_count ≥ 3.

30. **\`\_evolve\_memory\_slots\`** (SLOT\_EVOLUTION\_PROMPT) — proposes updates for project\_context, user\_preferences, guidance; inserts review\_queue rows with item\_type='memory\_slot\_update'. **Does not write slots directly** — human approval required.

31. **\`\_detect\_motifs\`** (MOTIF\_PROMPT) — inserts review\_queue rows with item\_type='new\_cluster\_proposal'. There is no numeric motif threshold; motif identification is entirely model-driven, and cluster creation requires human approval.

32. **\`\_enrich\_codex\_entities\`** (ENRICHMENT\_PROMPT, global pass after all conversations) — selects up to 10 CodexEntity rows with thin context\_payload, summarises the originating episodic passages, overwrites context\_payload, and emits a CodexEvent(event\_type="context\_appended"). Reflection **does not add edges** — it only enriches entity descriptions.

### **8.7 Clustering Worker**

clustering.cluster\_turns (§3.6, beat every 30 min, MAX\_TURNS\_PER\_RUN = 25) and clustering.merge\_similar\_clusters (callable, not beat-scheduled) maintain the Context Cluster graph.

### **8.8 Batch Summariser**

batch\_summarizer.batch\_summarize (§3.8, beat every 2 h) groups decayed-but-not-archived turns into 50-turn batches and writes compressed batch\_summaries rows.

### **8.9 Sentinel Monitor**

sentinel\_monitor.monitor\_sentinels (beat every 30 min) is a declarative rule engine. Rules live in the sentinel\_rules table (trigger\_type ∈ \{threshold, frequency, absence, contradiction, composite\}, trigger\_conditions JSONB, action\_type ∈ \{notify, schedule\_worker, create\_review\_item, log\_event, propose\_memory\_update\}, action\_payload JSONB, cooldown\_seconds, last\_fired\_at). Each cycle: skip if within cooldown, evaluate the rule, on True insert a SentinelEvent, update last\_fired\_at, and dispatch the action — log\_event (no-op beyond the event row), notify (logged), schedule\_worker (importlib import + .delay()), create\_review\_item (item\_type='sentinel\_review'), propose\_memory\_update (declared but not implemented). Implemented trigger\_type handlers cover threshold (pending-edges vs active-edges HAVING count), absence (stale pending\_items slot older than max\_age\_days), with frequency/contradiction/composite declared but not implemented.

### **8.10 Fine-Tune Worker**

fine\_tune.fine\_tune\_classifier (crontab Mon 04:00 UTC) periodically retrains the classifier head on the curated\_labels table (populated by the manual POST /user-control/batch/override-tags endpoint). It loads ice\_classifier\_v2\_final.pt, encodes all curated prompts with the embedder on CUDA, builds a (N, 25) label tensor, trains all MLP parameters (Adam(lr=1e-4), BCEWithLogitsLoss × 2 + CrossEntropyLoss, 10 epochs, single batch), and saves ice\_classifier\_finetuned\_\{timestamp\}.pt. **Versioning is filesystem-only** — there is no DB table tracking runs, and the active inference path (settings.classifier\_model\_path) is not updated; promoting a fine-tuned artifact requires manual file replacement.

### **8.11 Compaction Worker**

compaction.compact\_entities (§4.8, callable, not beat-scheduled, max\_retries=3, default\_retry\_delay=60) snapshots entities with ≥100 uncompacted events.

### **8.12 Drop Zone and Codex Inject Watcher**

Two standalone watchdog.Observer processes (not Celery tasks) handle file-system ingestion:

- **Drop Zone** (workers/drop\_zone.py) watches ingest\_inbox/ for .txt/.jsonl/.md files, waits for file size to settle, creates a rag\_documents row, chunks into 512-word windows, embeds each via the classifier's embedder, writes rag\_chunks, and moves the file to processed/.

- **Codex Inject Watcher** (workers/codex\_inject\_watcher.py) watches codex\_inject/ for .yaml/.yml/.json files describing entities (canonical\_name, aliases, tags, properties, context\_payload, relations: \[\{target, relation\}\]). It resolves/creates entities with deterministic UUIDv5, and for each relation inserts a CodexEdge at strength=2.0, confidence="active" (manual injection = high confidence) plus a CodexEvent(event\_type="edge\_added", payload=\{"manual\_injection": True\}), with an edge-existence check serving as idempotency.


## **9. Model Registry and Mixture-of-Experts Routing**

ICE does not train its own generation models; it routes each turn to the best locally-served model from a dynamically-populated registry.

### **9.1 Dynamic registry**

The registry is persisted at models/model\_registry.json (\{"models": \{name: entry\}, "updated\_at": …\}). populate\_from\_ollama() queries \{ollama\_base\_url\}/api/tags, and for each model not already in the registry: (i) fetches Hugging Face model-card tags via https://huggingface.co/api/models/\{id\} (with an \_ollama\_name\_to\_hf\_id best-effort mapping — qwen2.5 → Qwen/Qwen2.5-7B-Instruct, gemma4 → google/gemma-4-7b-it, etc.); (ii) if HF tags are non-empty, maps them through HF\_TOPIC\_MAP and HF\_INTENT\_MAP (e.g. code/coding/programming/python → Software\_&\_Tech + Generation; creative/roleplay/storytelling → Creative\_&\_Media + Generation; finance → Business\_&\_Finance + Analysis\_&\_Summarization) and marks confirmed = True; (iii) otherwise falls back to LLM tagging with Qwen/Qwen2.5-3B-Instruct-AWQ (temperature=0.0, max\_tokens=150) and marks confirmed = False. Each entry records topic\_tags, intent\_tags, priority (default 5), context\_window (default 8192), confirmed, base\_url, added\_at.

### **9.2 MoE selection**

find\_best\_model(topic\_tags, intent\_tags, required\_tokens=0) iterates the registry, skipping any entry with confirmed == False (LLM-tagged models never participate in routing) and any entry whose context\_window \< required\_tokens (context-window-aware routing). The score is:

*score = |topic\_tags ∩ entry.topic\_tags| + |intent\_tags ∩ entry.intent\_tags| + entry.priority*

Ties resolve to first-seen in JSON dict order. If no model qualifies, get\_fallback\_model() returns the first confirmed model, else settings.default\_fallback\_model ("qwen2.5:7b").

### **9.3 Session stickiness**

Stickiness is in-process (SESSION\_STATE dict keyed by conversation\_id, **not** persisted to Redis or the DB — it resets on API restart and is not shared across replicas). After each classification, topic and intent overlap with the previous turn is computed; overlap ⇒ consecutive\_shifts = 0, no overlap ⇒ consecutive\_shifts += 1. The routing decision (only when the client requested model == "ice-proxy"): if a sticky model is set **and** consecutive\_shifts \< 3, keep it; else call find\_best\_model and reset consecutive\_shifts. The conversation key comes from the X-ICE-Conversation-ID header (or a fresh uuid.uuid4()). required\_tokens = int((system\_words + user\_words) \* 1.33) is computed from the assembled messages and passed to find\_best\_model so the chosen model's context window can accommodate the prompt.


## **10. Operational Infrastructure**

### **10.1 FastAPI proxy**

api/main.py constructs FastAPI(title="ICE Proxy", description="Infinite Context Engine — OpenAI-compatible memory middleware", version="1.0.0") and includes the memory\_slots (/memory-slots) and user\_control (/user-control) routers. Endpoints:

| **Method** | **Path** | **Purpose** |
| - | - | - |
| **GET** | /health | liveness probe |
| **POST** | /v1/chat/completions | main OpenAI-compatible proxy; SSE streaming; enqueues evaluate\_turn.delay after the stream |
| **GET/PUT** | /memory-slots/, /memory-slots/\{slot\_name\} | slot CRUD |
| **POST** | /user-control/initialize | bootstrap a conversation |
| **POST** | /user-control/turns/\{turn\_id\}/bookmark | bookmark a turn |
| **POST** | /user-control/batch/override-tags | bulk-correct labels → CuratedLabel |
| **PUT/GET** | /user-control/conversations/\{conv\_id\}/scope | set/read memory\_scope\_type and cluster\_ids |
| **POST/PUT** | /user-control/clusters, /user-control/clusters/\{id\}/assign | explicit cluster creation/assignment |
| **GET/POST** | /user-control/review-queue, /…/\{item\_id\}/approve | human-in-the-loop queue |
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

A single PostgreSQL instance with the pgvector extension is the unified store. api/db.py constructs create\_engine(settings.database\_url, pool\_size=50, max\_overflow=20, pool\_pre\_ping=True, pool\_recycle=3600) and a sessionmaker. Every vector column is Vector(384) because the same embedder is used everywhere. Filtered vector queries follow a uniform idiom — SELECT …, (1 - (embedding \<=\> :prompt\_embedding)) \[\* decay\_weight\] AS score FROM \<table\> WHERE embedding IS NOT NULL AND \<filters\> ORDER BY score DESC LIMIT :k — used by the episodic, procedural, RAG, batch-summary, and cluster-relevance queries. Worker-side variants cast the embedding explicitly (CAST(:emb AS vector)) because the embedding arrives as a Python list string.

**Schema management.** The schema is defined purely via SQLAlchemy ORM models in memory/models.py; no alembic import, no migrations/ directory, and no CREATE EXTENSION vector DDL appears in the source. The pgvector extension and the underlying tables must be provisioned externally (manual CREATE EXTENSION vector; + Base.metadata.create\_all(engine) or an out-of-tree migration tool).

### **10.3 Celery over Redis**

broker = backend = settings.redis\_url (default redis://localhost:6379/0); JSON serializers; UTC timezone; a single implicit default queue (celery) with no task\_routes and no per-task queue=. The result backend is unused in the audited code paths — workers either return early or raise self.retry(...). The beat schedule is the eight-entry table in §8.1.

### **10.4 Idempotency architecture**

Two complementary mechanisms turn Celery's at-least-once delivery into effectively-once side effects:

- **Worker-side \`idempotency\_keys\` table** (key TEXT PK, processed\_at TIMESTAMPTZ DEFAULT now()). Every GPU-touching worker opens its transaction with key = sha256(scope + ":" + batch\_id); if a row exists, the task returns immediately; otherwise the work runs and the key insert shares the worker's transaction. Scope strings: post-flight sha256(batch\_id) (implicit scope), codex sha256("codex:" + batch\_id), procedural sha256("procedural:" + batch\_id). A crash before commit leaves no key (retry re-runs); a crash after commit leaves the key (retry is a no-op).

- **API-layer deduplication** via the idempotency\_key column on EpisodicMemory itself (sha256(correlation\_id + ":" + user\_message)), keyed on a per-request correlation\_id UUID. Dedup at this layer is informational rather than enforced — there is no unique constraint in the model definition — but it gives the storage layer a defence against retried HTTP requests.

Every worker is @app.task(bind=True, max\_retries=N, default\_retry\_delay=M) with self.retry(exc=exc) after db.rollback(), so retried executions re-enter from the top and re-check the idempotency key first.

### **10.5 GPU resource management**

workers/gpu\_check.py exposes is\_gpu\_busy() (polls nvidia-smi, threshold GPU\_UTIL\_THRESHOLD = 20, 5-second subprocess timeout, returns False on any subprocess error) and is\_user\_active(idle\_threshold\_seconds=10) (checks the Redis ice:last\_chat\_completed key). Polling is ad-hoc — every GPU-touching worker calls is\_gpu\_busy() at task entry; there is no background polling thread or caching. The background\_model\_mode setting ("dedicated" default, "shared" alternative) gates is\_gpu\_busy() behaviour: in shared mode it always returns False and yielding is cooperative via is\_user\_active(). The bg\_client\_factory can route background‑model calls through either a dedicated vLLM instance (port 8002, serving Qwen2.5‑3B‑Instruct‑AWQ) or through the same SGLang server that handles user‑facing generation (port 8001, serving Qwen3‑14B‑AWQ). In the latter configuration, an \_NonThinkingClient wrapper injects enable\_thinking: False into every request, ensuring that structured‑JSON extraction (clustering, triplet extraction, reflection) receives direct, non‑reasoning output. The active path is determined by the background\_model\_mode setting and, during the ablation experiments, uses the single‑server configuration to conserve GPU memory. The background\_model\_mode setting still gates is\_gpu\_busy() but no longer selects the inference server.

### **10.6 Configuration system**

api/config.py defines a Pydantic Settings(BaseSettings) with SettingsConfigDict(env\_file=".env", env\_file\_encoding="utf-8"). Env vars are auto-derived from uppercased field names (no explicit Field(env=…)). The fields, grouped by concern:

- **Database / Redis**: database\_url (default postgresql+psycopg://ice:ice\_local\_dev@localhost:5432/ice\_db), redis\_url (redis://localhost:6379/0).

- **Upstream LLM**: ollama\_base\_url (http://localhost:11434), default\_fallback\_model (qwen2.5:7b), background\_model\_mode (dedicated).

- **Classifier**: classifier\_threshold (0.3), confidence\_fallback\_threshold (0.75), classifier\_model\_path (models/classifier/ice\_classifier\_v3\_qwen\_ft3.pt), label\_schema\_path (data/labeled/label\_schema.json).

- **DI3 density thresholds**: DI3\_ENABLED (True), DI3\_CODE\_DENSITY\_THRESHOLD (0.3), DI3\_SENTIMENT\_DENSITY\_THRESHOLD (0.4), DI3\_META\_DENSITY\_THRESHOLD (0.2), DI3\_NOISE\_DENSITY\_THRESHOLD (0.8), DI3\_REFERENCE\_DENSITY\_THRESHOLD (0.2), DI3\_LTM\_REFERENCE\_DENSITY\_THRESHOLD (0.1).

Worker tuning constants (GPU\_UTIL\_THRESHOLD, CYCLES\_PER\_DAY, all DECAY\_RATE\_\*, ARCHIVE\_THRESHOLD, COLD\_THRESHOLD, EVENT\_THRESHOLD, STALE\_DAYS, MIN\_REINFORCEMENT, the RRF k=60, all bonus constants, all retrieval thresholds, the token-budget constants) are **module-level Python constants**, not environment-configurable. We note two mismatches worth flagging: settings.classifier\_model\_path points at ice\_classifier\_v3\_qwen\_ft3.pt, but fine\_tune.py loads from the hardcoded ice\_classifier\_v2\_final.pt and writes ice\_classifier\_finetuned\_\{timestamp\}.pt — neither the input nor the output of the weekly fine-tune matches the active inference path; and drop\_zone.py instantiates PyTorchClassifier(model\_path="models/classifier/ice\_classifier\_v2\_final.pt") — also hardcoded, also mismatched. Promoting a fine-tuned artifact to production therefore requires manual file replacement or a settings override.
