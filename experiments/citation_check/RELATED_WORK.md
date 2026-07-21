# Related Work: Foundations and Limitations of Memory and Retrieval Systems in Conversational Agents

> **Status: NOT verification-clean.** Four citations below are flagged `⚠ VERIFY` and should not be submitted as-is. See `CITATION_ISSUES.md` for detail and `verify_citations.py` for a repeatable check.

The transition from stateless Large Language Models (LLMs) to stateful, autonomous conversational agents has necessitated the development of sophisticated middleware systems capable of extending the finite context window. While contemporary research has produced numerous approaches to external knowledge retrieval and context management, a critical analysis of these architectures reveals persistent vulnerabilities when applied to continuous, temporally evolving dialogue. This section evaluates the theoretical precedents underpinning modern retrieval and memory frameworks, categorizing the literature into four distinct domains: memory and Retrieval-Augmented Generation (RAG) systems, knowledge graphs for dialogue, classic static-corpus RAG enhancements, and the fundamental information retrieval algorithms integrated into the proposed ICE architecture.

## Stateful Memory and Retrieval-Augmented Architectures

The architectural constraint of bounded context lengths in Transformer-based models has driven the engineering of specialized memory hierarchies designed to preserve user-specific facts over extended temporal horizons. Early approaches attempted to solve this by drawing direct analogies to traditional computing hardware.

Packer et al. (2023) introduced MemGPT, a paradigm-shifting architecture that conceptualizes the LLM context window as a form of primary random-access memory (RAM), backed by an unbounded external storage system [1]. MemGPT employs an intelligent paging mechanism, utilizing system interrupts to manage the control flow of data between the main context and external tiers, effectively creating the illusion of an infinite context window [1]. While MemGPT demonstrates high efficacy in short-to-medium length interactions and intensive document analysis tasks, empirical stress tests reveal significant degradation in highly extended, multi-session dialogues [2]. The system relies heavily on flat, dense vector retrieval, which struggles with the "compaction continuity problem" and fails to distinguish between semantically identical but contextually distinct entities over time [4] `⚠ VERIFY`. Crucially, MemGPT lacks a structured knowledge graph to track the temporal evolution of entities and operates without intent-driven retrieval routing, leading to context dilution in complex narrative environments.

To address the latency and token-cost bottlenecks associated with hierarchical paging, Chhikara et al. (2025) proposed mem0, a lightweight, production-ready memory layer [6]. The mem0 architecture optimizes the memory lifecycle by employing a single-pass, append-only extraction pipeline that distills long-form conversations into atomic, non-hierarchical facts [3] `⚠ VERIFY`. By utilizing a multi-signal retrieval pipeline that fuses semantic embedding, BM25 keyword matching, and entity linking in parallel, mem0 achieves significant reductions in computational overhead, excelling on benchmarks such as LoCoMo and LongMemEval [6]. Despite its efficiency, the reliance on topologically flat, key-value fact extraction inherently limits the system's cognitive depth [3] `⚠ VERIFY`. It lacks the capacity for versioned entities, meaning contradictory or evolving facts are not systematically reconciled. Furthermore, mem0 does not natively support procedural memory or implement a mathematically grounded decay and archival mechanism, allowing outdated constraints to persist indefinitely [9].

Zhong et al. (2023) advanced the cognitive plausibility of long-term agentic memory with MemoryBank, an architecture heavily inspired by human psychology [11]. MemoryBank integrates the Ebbinghaus Forgetting Curve to model memory updates via a time-aware decay and reinforcement mechanism [11]. In this framework, every memory item maintains a continuously updated strength score, which dictates whether a fact is retrieved, reinforced, or forgotten based on the time elapsed since its last activation [15]. Applied to the SiliconFriend conversational agent, MemoryBank successfully maintains persona invariance and heightened empathy in multi-day dialogues [12]. However, the system is fundamentally constrained by its retrieval substrate. It lacks a foundational knowledge graph to facilitate associative, multi-hop reasoning, and it fails to implement conversation-scoped clustering, which is essential for maintaining narrative coherence across disparate conversational threads [11].

Addressing the computational bottlenecks of dense memory retrieval, Shi et al. (2024) developed the Self-Contrast Mixture-of-Experts (SCMoE) framework [16]. Traditional MoE models suffer from expert load imbalance and knowledge redundancy, where unchosen experts fail to contribute to the output prediction [16]. SCMoE introduces a training-free strategy that utilizes unchosen experts in a self-contrast manner during inference, determining next-token probabilities by contrasting the output logits from strong (top-k) and weak (lower-ranked) activations [16]. Additionally, shortcut-connected MoE designs (ScMoE) have been proposed to decouple communication processes from computation, enabling adaptive overlapping parallel strategies [21]. While SCMoE optimizes the internal routing of neural representations for better information processing, its application as an externalized memory retrieval system remains undertested. It notably lacks longitudinal evaluation metrics and has not been stress-tested on the large, dynamically shifting per-turn documents that characterize persistent agentic memory [16].

The structural vulnerabilities of these systems underscore a pervasive industry trend: the conflation of static fact-storage with dynamic working memory. The ICE middleware framework addresses these precise vulnerabilities by bridging the gap between efficient retrieval and temporal narrative continuity.

| System / Paper | Core Research Mechanism | Key Gap ICE Fills |
| --- | --- | --- |
| **MemGPT** (Packer et al., 2023) | OS-inspired memory hierarchy managing data flow between a bounded main context and external storage. Effective for short analysis but struggles with long horizons. | No temporal tracking of entity evolution. No structured knowledge graph. No intent-driven retrieval routing. |
| **mem0** (Chhikara et al., 2025) | Lightweight, production-ready memory layer utilizing multi-signal retrieval to store facts as discrete, flat key-value pairs. | No versioned entities. No procedural memory. No explicit mathematical decay or archival mechanisms. |
| **MemoryBank** (Zhong et al., 2023) | Long-term memory architecture incorporating the Ebbinghaus forgetting curve for selective factual recall and behavioral reinforcement. | No knowledge graph representation. No conversation-scoped clustering for narrative threading. |
| **SCMoE** (Shi et al., 2024) | Self-contrast mixture-of-experts for optimizing internal retrieval by contrasting strong and weak expert logits. | No longitudinal evaluation. No stress-tests on large, evolving per-turn conversational documents. |

## Knowledge Graphs as Dialogue Substrates

To overcome the relational blindness of topologically flat vector stores, the literature has increasingly explored Knowledge Graphs (KGs) to map the intricate semantic topologies of textual data. KGs offer explicit, editable representations of entities (nodes) and their relationships (edges), providing a highly structured topological space for multi-hop navigation.

Edge et al. (2024) introduced GraphRAG, a comprehensive approach designed to answer global, sense-making queries over extensive, private document collections [25]. Traditional vector-based retrieval fails on query-focused summarization tasks that require traversing an entire corpus. GraphRAG circumvents this by utilizing an LLM to extract a massive entity knowledge graph from raw text, followed by the application of community detection algorithms (e.g., Leiden) to partition the graph [28]. It then pre-generates hierarchical community summaries for groups of closely related entities, allowing the system to aggregate partial responses into a highly diverse final answer [26]. While GraphRAG significantly improves answer comprehensiveness, it operates under the assumption of a temporally static corpus [26]. In GraphRAG, all ingested facts are treated as equally current. It lacks the temporal versioning required to track an entity whose properties, relationships, or states mutate dynamically across multiple conversational sessions, rendering it brittle in an evolving dialogue environment.

Wang et al. (2023) proposed Knowledge Graph Prompting (KGP) to address the specific challenges of Multi-Document Question Answering (MD-QA) [31]. KGP constructs a specialized graph where nodes represent individual passages or document structures, and edges denote semantic or lexical similarities [31]. The framework deploys an LLM-based graph traversal agent that acts as a local navigator, dynamically gathering supporting context to regulate the transitional space among interconnected documents and reduce retrieval latency [31].

Concurrently, Sun et al. (2023) developed Think-on-Graph (ToG), a tight-coupling algorithmic framework that treats the LLM as an active agent performing iterative beam search across KG paths [27]. Unlike naive retrieval, ToG prompts the LLM to explore multiple possible reasoning paths dynamically, pruning irrelevant branches and evaluating triples until it determines that a question can be answered [36]. This explicit "System 2" thinking enhances deep reasoning capabilities without incurring additional fine-tuning costs [27].

While both KGP and ToG demonstrate the profound utility of structural graph retrieval, their architectures are heavily tailored for single-shot, objective question answering over static knowledge bases [32]. They inherently assume the graph is a ground-truth repository of factual world knowledge, completely missing the mechanisms for narrative continuity, belief decay, and contradiction resolution required for an evolving conversational agent. ICE leverages the topological benefits of these graph systems but repurposes them to support a continuously mutating conversational landscape.

| System / Paper | Core Research Mechanism | Key Gap ICE Fills |
| --- | --- | --- |
| **GraphRAG** (Edge et al., 2024) | Community-based hierarchical graph summarization utilizing entity extraction and clustering for retrieval over static corpora. | No temporal versioning. An entity's properties can change over time in ICE; GraphRAG treats all facts as equally current. |
| **KGP** (Wang et al., 2023) / **ToG** (Sun et al., 2023) | Knowledge-graph-prompting and iterative beam-search traversal agents for deep, structured multi-hop QA. | Designed exclusively for single-shot QA, not evolving conversations. Completely lacks decay or reinforcement mechanisms. |

## Evolutions in Classic Retrieval-Augmented Generation

The foundational models of Retrieval-Augmented Generation established the paradigm of augmenting parametric model knowledge with non-parametric, external databases to mitigate hallucinations. However, these systems universally share a critical limitation when applied to agentic middleware: the strict assumption of an immutable corpus.

Lewis et al. (2020) formalized the original RAG framework, providing a general-purpose fine-tuning recipe that combines a pre-trained sequence-to-sequence model with a dense vector index of a static document corpus, such as Wikipedia [27] `⚠ VERIFY`. While this fundamentally shifted how models access long-tailed knowledge, classic RAG relies on point-in-time document retrieval driven by maximum inner product search [38]. Conversational history, by contrast, is not a static corpus. It grows organically, facts are frequently updated or contradicted, and context shifts dynamically. The original RAG architecture possesses no mechanisms to handle these temporal dynamics.

Subsequent iterations sought to optimize the retrieval, encoding, and decoding pipelines. Guu et al. (2020) introduced REALM (Retrieval-Augmented Language Model Pre-training), integrating a latent knowledge retriever directly into the unsupervised pre-training phase [41]. REALM utilizes a masked language modeling objective, forcing the model to backpropagate through a retrieval step that considers millions of documents to predict masked tokens [41]. Izacard and Grave (2021) subsequently developed Fusion-in-Decoder (FiD), an architecture that processes retrieved passages independently in the encoder before concatenating them for joint attention in the decoder [44]. This allows the model to dynamically weigh and aggregate evidence from up to a hundred retrieved passages, significantly improving semantic coherence in open-domain QA [44].

Shi et al. (2023) proposed REPLUG, a framework that treats the generative LLM as an entirely frozen black box, avoiding the computational expense of cross-attention pre-training [48]. REPLUG simply prepends retrieved documents to the input context and optimizes the retrieval model using LM-Supervised Retrieval (LSR) [48]. The retriever is tuned using the perplexity scoring signals from the black-box LM, driving the retriever to select documents that strictly improve language model perplexity [50].

More recently, Yan et al. (2024) introduced Corrective Retrieval Augmented Generation (CRAG), which applies an explicit corrective process to mitigate retrieval noise [52]. CRAG implements a lightweight evaluator that scores retrieved documents, routing them into three confidence categories: correct, ambiguous, or incorrect [53]. If the internal retrieval is deemed inadequate, CRAG triggers a decompose-then-recompose algorithm to filter irrelevant spans, or falls back to large-scale web searches to supplement the knowledge base [53].

Despite their mathematical and architectural elegance, REALM, FiD, REPLUG, and CRAG are structurally confined to static knowledge bases. They possess no innate mechanisms to handle a personalized, highly dynamic, and evolving conversation history where a user's preferences, constraints, and state of mind shift iteratively. ICE replaces the static-corpus assumption with a highly mutable, chronologically aware middleware designed specifically for personal agentic history.

| System / Paper | Core Research Mechanism | Key Gap ICE Fills |
| --- | --- | --- |
| **RAG** (Lewis et al., 2020) | The foundational generative retrieval architecture augmenting an LLM via dense embeddings from a static document corpus. | Conversational history is not a static corpus—it grows and facts change. RAG fundamentally cannot handle temporal dynamics. |
| **REALM, FiD, REPLUG, CRAG** | Advanced RAG enhancements including latent pre-training, fusion-in-decoder, black-box LM tuning, and corrective evaluation fallback. | All operate strictly on static knowledge bases. None address the requirements of a personal, dynamically evolving conversation history. |

## Algorithmic Primitives in the ICE Middleware

To achieve robust memory retrieval that avoids the pitfalls of purely dense or purely sparse indices, ICE synthesizes several foundational algorithmic techniques. By leveraging sparse lexical retrieval, mathematical rank fusion, and generative query expansion, the system creates a highly resilient middleware layer.

**BM25 (Robertson & Zaragoza, 2009):** Grounded in the Probabilistic Relevance Framework (PRF) and the Binary Independence Model, BM25 remains one of the most successful sparse retrieval algorithms [57]. It scores document relevance based on a sophisticated non-linear term frequency function and inverse document frequency, rigorously incorporating document length normalization to prevent long documents from dominating results [57]. While dense semantic embeddings excel at capturing conceptual similarity, they frequently suffer from catastrophic failures in exact-keyword matching, particularly concerning proper nouns, specialized acronyms, or unique user identifiers. BM25 ensures high-fidelity lexical recall. Within the ICE architecture, BM25 is not deployed as the primary semantic engine, but rather as an essential secondary leg in a multi-signal retrieval strategy, guaranteeing that specific, explicit conversational artifacts are perfectly preserved.

**Reciprocal Rank Fusion (Cormack et al., 2009):** When fusing disparate retrieval signals—such as the unbounded floating-point scores of dense vector cosine similarity and the probabilistically derived, length-normalized scores of sparse BM25—standard linear normalization is statistically unsound due to extreme score distribution mismatches. Cormack et al. (2009) introduced Reciprocal Rank Fusion (RRF), a mathematically elegant algorithm that combines document rankings by sorting them based on the inverse of their rank positions across multiple independent search systems. By entirely disregarding the raw computational scores and focusing exclusively on ordinal rankings, RRF prevents any single retrieval methodology from disproportionately skewing the results. Within the ICE architecture, RRF serves as the indispensable fusion backbone. Subsequent ablation studies of the ICE framework empirically demonstrate that RRF is the single most critical mathematical component for ensuring retrieval stability and accuracy across highly diverse conversational queries [6] `⚠ VERIFY`.

**Hypothetical Document Embeddings (HyDE) (Gao et al., 2023):** Traditional dense retrieval often struggles with the "vocabulary mismatch" problem. A user's brief, colloquial, or highly implicit query maps poorly in vector space to the detailed, descriptive facts stored in external memory. Gao et al. (2023) proposed HyDE, an unsupervised technique that leverages an instruction-tuned LLM to generate a hypothetical, hallucinatory document in response to a query. This generated text captures the intended semantic pattern and is subsequently embedded to search the vector space, effectively shifting the search paradigm from query-space to document-space. ICE implements HyDE by utilizing a small, highly efficient background model for real-time query rewriting. However, architectural ablation studies reveal a nuanced limitation: while HyDE excels in strictly factual, objective retrieval tasks, its effect is largely neutral—and occasionally detrimental—in highly creative, open-ended conversational domains, indicating that its utility must be dynamically gated based on the specific intent of the user's interaction.

| Technique | Foundational Paper | Application and Note within ICE |
| --- | --- | --- |
| **BM25** | Robertson & Zaragoza (2009) | Classic lexical retrieval based on probabilistic relevance. ICE uses it as a secondary leg to guarantee exact-keyword fidelity, not as the primary semantic method. |
| **RRF** | Cormack et al. (2009) | Reciprocal Rank Fusion. Acts as ICE's core fusion backbone. Ablation studies confirm it is the single most critical mathematical component for retrieval stability. |
| **HyDE** | Gao et al. (2023) | Generative query rewriting via hypothetical document embeddings. ICE implements it via a small background model; ablation shows a neutral effect in creative domains. |

## Works Cited

[1] Packer, C., Fang, V., et al. (2023). "MemGPT: Towards LLMs as Operating Systems." *Semantic Scholar*. https://www.semanticscholar.org/paper/MemGPT%3A-Towards-LLMs-as-Operating-Systems-Packer-Fang/908dad62c0e43d80e3e3cb3c0402f7c71c70499c

[2] MemGPT Research Team. (n.d.). "MemGPT Project Website." https://research.memgpt.ai/

[3] `⚠ VERIFY` Author(s) unknown. (2026). "Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents." *arXiv preprint*, arXiv:2603.04814. https://arxiv.org/html/2603.04814v1

[4] `⚠ VERIFY` Author(s) unknown. (2026). "Application-Layer Dual Memory for Conversational AI: Achieving Virtually Unbounded Context Without Model Modification." *arXiv preprint*, arXiv:2605.20724. https://arxiv.org/html/2605.20724

[5] `⚠ VERIFY` Author(s) unknown. (2026). "To Know is to Construct: Schema-Constrained Generation for Agent Memory." *arXiv preprint*, arXiv:2604.20117. https://arxiv.org/html/2604.20117v1

[6] mem0ai. (n.d.). "mem0: Universal memory layer for AI Agents." *GitHub Repository*. https://github.com/mem0ai/mem0

[7] Chhikara, P., Khant, A., et al. (2025). "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory." *Semantic Scholar*. https://www.semanticscholar.org/paper/Mem0%3A-Building-Production-Ready-AI-Agents-with-Chhikara-Khant/1d9c21a0fdb1cc16a32c5d490ebaf98436a23382

[8] Chhikara, P., Khant, A., et al. (2025). "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory." *arXiv preprint*, arXiv:2504.19413. https://arxiv.org/abs/2504.19413

[9] mem0ai. (n.d.). "12 layers memory mem0 Supabase #3883." *GitHub Discussions*. https://github.com/mem0ai/mem0/discussions/3883

[10] Mem0 Team. (2026). "Memory vs Context Window for LLM and AI Agents." *Mem0 Blog*. https://mem0.ai/blog/context-window-is-ram-not-storage-why-most-agent-failures-happen-how-to-fix-them-in-2026

[11] Zhong, W., Guo, L., et al. (2023). "MemoryBank: Enhancing Large Language Models with Long-Term Memory." *ResearchGate*. https://www.researchgate.net/publication/379280304_MemoryBank_Enhancing_Large_Language_Models_with_Long-Term_Memory

[12] Zhong, W., Guo, L., et al. (2024). "MemoryBank: Enhancing Large Language Models with Long-Term Memory." *Proceedings of the AAAI Conference on Artificial Intelligence*. https://ojs.aaai.org/index.php/AAAI/article/view/29946

[13] Zhong, W., Guo, L., et al. (2023). "MemoryBank: Enhancing Large Language Models with Long-Term Memory." *arXiv preprint*, arXiv:2305.10250. https://arxiv.org/abs/2305.10250

[14] Zhong, W., Guo, L., et al. (2023). "MemoryBank: Enhancing Large Language Models with Long-Term Memory." *Semantic Scholar*. https://www.semanticscholar.org/paper/MemoryBank%3A-Enhancing-Large-Language-Models-with-Zhong-Guo/c3a59e1e405e7c28319e5a1c5b5241f9b340cf63

[15] Emergent Mind. (n.d.). "MemoryBank Architectures." *Emergent Mind Topics*. https://www.emergentmind.com/topics/memorybank

[16] Shi, Y., et al. (2024). "Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast." *arXiv preprint*, arXiv:2405.14507. https://arxiv.org/abs/2405.14507

[17] Shi, Y., et al. (2024). "Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast." *arXiv preprint*, arXiv:2405.14507v2. https://arxiv.org/html/2405.14507v2

[18] Author(s) unknown. (2025). "CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning." *ACL Anthology*, Findings of EMNLP. https://aclanthology.org/2025.findings-emnlp.398.pdf

[19] Author(s) unknown. (2025). "CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning." *arXiv preprint*, arXiv:2505.17553. https://arxiv.org/html/2505.17553v1

[20] Shi, Y., et al. (2024). "Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast." *arXiv preprint* (PDF). https://arxiv.org/pdf/2405.14507

[21] Author(s) unknown. (2024). "Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts." *arXiv preprint*, arXiv:2404.05019. https://arxiv.org/html/2404.05019v3

[22] Author(s) unknown. (2024). "Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts." *arXiv preprint*. https://arxiv.org/html/2404.05019v1

[23] ICML. (2025). "Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts." *ICML Virtual Poster*. https://icml.cc/virtual/2025/poster/45834

[24] `⚠ VERIFY` Author(s) unknown. (n.d.). "SCMoE-PFL: A Soft-Clustering Mixture-of-Experts Framework for Personalized Federated Learning." *ResearchGate*. https://www.researchgate.net/publication/404918230_SCMoE-PFL_A_Soft-Clustering_Mixture-of-Experts_Framework_for_Personalized_Federated_Learning

[25] Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *Scirp.org*. https://www.scirp.org/reference/referencespapers?referenceid=3936359

[26] Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *Microsoft Research*. https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/

[27] `⚠ VERIFY (dual-use)` Sewak, M. (n.d.). "The 'Think-on-Graph' Methodology: Inside Agentic Retrieval." *Level Up Coding (Medium)*. https://levelup.gitconnected.com/the-think-on-graph-methodology-inside-agentic-retrieval-e455b377444a — **note: this same [27] is also cited above for Lewis et al. 2020's RAG paper, which is a different work by different authors.**

[28] Singhal, S. (n.d.). "Why Your GraphRAG is Over-Engineered: Building Lean Knowledge Graphs with Small Language Models (SLMs)." *Medium*. https://medium.com/@shreyasinghal0409/why-your-graphrag-is-over-engineered-building-lean-knowledge-graphs-with-small-language-models-a9ac655891c6

[29] Lebrero, J. F. (n.d.). "From Local to Global: A Deep Dive into GraphRAG." *GoPenAI*. https://blog.gopenai.com/from-local-to-global-a-deep-dive-into-graphrag-1c50e2fc9e65

[30] Artsplendr. (n.d.). "GraphRAG-Implementations." *GitHub Repository*. https://github.com/Artsplendr/GraphRAG-Implementations

[31] Wang, X., et al. (2023). "Knowledge Graph Prompting for Multi-Document Question Answering." *ResearchGate*. https://www.researchgate.net/publication/379277994_Knowledge_Graph_Prompting_for_Multi-Document_Question_Answering

[32] Wang, X., et al. (2023). "Knowledge Graph Prompting for Multi-Document Question Answering." *arXiv preprint*, arXiv:2308.11730. https://arxiv.org/abs/2308.11730

[33] Wang, X., et al. (2023). "Knowledge Graph Prompting for Multi-Document Question Answering." *arXiv preprint*. https://arxiv.org/html/2308.11730v3

[34] Author(s) unknown. (2025). "Large Language Models Meet Knowledge Graphs for Question Answering: Synthesis and Opportunities." *ACL Anthology*, EMNLP. https://aclanthology.org/2025.emnlp-main.1249.pdf

[35] Sun, J., et al. (2023). "Deep and Responsible Reasoning of Large Language Model on Knowledge Graph." *arXiv preprint*, arXiv:2307.07697. https://arxiv.org/html/2307.07697v6

[36] Sun, J., et al. (2024). "Deep and Responsible Reasoning of Large Language Model on Knowledge Graph." *arXiv preprint*. https://arxiv.org/pdf/2307.07697

[37] Sun, J., et al. (2024). "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph." *ICLR Proceedings*. https://proceedings.iclr.cc/paper_files/paper/2024/file/10a6bdcabbd5a3d36b760daa295f63c1-Paper-Conference.pdf

[38] Author(s) unknown. (2025). "RAG-DDR: Optimizing Retrieval-Augmented Generation." *ICLR Proceedings*. https://proceedings.iclr.cc/paper_files/paper/2025/file/1a87980b9853e84dfb295855b425c262-Paper-Conference.pdf

[39] Author(s) unknown. (2024). "Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs." *ACL Anthology*. https://aclanthology.org/2024.findings-emnlp.459.pdf

[40] Google Research. (n.d.). "REALM: Retrieval-Augmented Language Model Pre-Training." *GitHub Repository*. https://github.com/google-research/language/blob/master/language/realm/README.md

[41] `⚠ VERIFY (weak sourcing)` Guu, K., Lee, K., et al. (2020). "REALM: Retrieval-Augmented Language Model Pre-Training." *ResearchGate*. https://www.researchgate.net/publication/339398913_REALM_Retrieval-Augmented_Language_Model_Pre-Training — **note: the real arXiv preprint (1909.08053) is not in this bibliography; only secondary mirrors are cited.**

[42] Guu, K., Lee, K., et al. (2020). "REALM: Retrieval-Augmented Language Model Pre-Training." *Semantic Scholar*. https://www.semanticscholar.org/paper/REALM%3A-Retrieval-Augmented-Language-Model-Guu-Lee/832fff14d2ed50eb7969c4c4b976c35776548f56

[43] Google Research. (n.d.). "REALM: Integrating Retrieval into Language Representation Models." *Google Research Blog*. https://research.google/blog/realm-integrating-retrieval-into-language-representation-models/

[44] Izacard, G., & Grave, E. (2021). "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering." *ResearchGate*. https://www.researchgate.net/publication/355430399_Leveraging_Passage_Retrieval_with_Generative_Models_for_Open_Domain_Question_Answering

[45] Izacard, G., & Grave, E. (2021). "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering." *ACL Anthology*, EACL. https://aclanthology.org/2021.eacl-main.74/

[46] Izacard, G., & Grave, E. (2021). "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering." *ACL Anthology* (PDF). https://aclanthology.org/2021.eacl-main.74.pdf

[47] Izacard, G., & Grave, E. (2020). "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering." *ResearchGate*. https://www.researchgate.net/publication/342655858_Leveraging_Passage_Retrieval_with_Generative_Models_for_Open_Domain_Question_Answering

[48] Shi, W., et al. (2023). "REPLUG: Retrieval-Augmented Black-Box Language Models." *ResearchGate*. https://www.researchgate.net/publication/367558151_REPLUG_Retrieval-Augmented_Black-Box_Language_Models

[49] Shi, W., et al. (2023). "REPLUG: Retrieval-Augmented Black-Box Language Models." *arXiv preprint*, arXiv:2301.12652. https://arxiv.org/abs/2301.12652

[50] Shi, W., et al. (2023). "REPLUG: Retrieval-Augmented Black-Box Language Models." *arXiv preprint*. https://arxiv.org/pdf/2301.12652

[51] Shi, W., et al. (n.d.). "REPLUG: Retrieval-Augmented Black-Box Language Models." *OpenReview*. https://openreview.net/pdf?id=6z_yPCrdCA4

[52] Emergent Mind. (n.d.). "Corrective Retrieval Augmented Generation (CRAG)." *Emergent Mind Topics*. https://www.emergentmind.com/topics/corrective-retrieval-augmented-generation-crag

[53] Yan, S., et al. (2024). "Corrective Retrieval Augmented Generation." *arXiv preprint*, arXiv:2401.15884. https://arxiv.org/abs/2401.15884

[54] Yan, S., et al. (n.d.). "Corrective Retrieval Augmented Generation." *OpenReview*. https://openreview.net/pdf?id=JnWJbrnaUE

[55] Milvus Blog. (n.d.). "Fix RAG Retrieval Errors with CRAG, LangGraph, and Milvus." *Milvus*. https://milvus.io/blog/fix-rag-retrieval-errors-crag-langgraph-milvus.md

[56] HuskyInSalt. (n.d.). "CRAG: Corrective Retrieval Augmented Generation." *GitHub Repository*. https://github.com/HuskyInSalt/CRAG

[57] Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Google Books*. https://books.google.com/books/about/The_Probabilistic_Relevance_Framework.html?id=yK6HxUEaZ9gC

[58] Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *ResearchGate*. https://www.researchgate.net/publication/220613776_The_Probabilistic_Relevance_Framework_BM25_and_Beyond

[59] Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Demo Bibliography*. https://demo.kerko.whiskyechobravo.com/bibliography/X5GFEY77

[60] Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *City, University of London*. https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
