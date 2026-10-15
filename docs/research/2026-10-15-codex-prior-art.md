# Codex rework — prior art and design review (2026-10-15, v3)

Input for the codex rework session (ROADMAP CURRENT POSITION). Written from web
search plus reading the public source of Graphiti and Mem0 (shallow clones,
2026-10-15). Third-party claims are marked as claims; benchmark numbers in this
field are self-reported and not comparable across systems.

## 1. How the systems that "work" are built

**Graphiti / Zep** (Apache-2.0; paper arXiv 2501.13956). Per incoming message
("episode"):
1. extract entities (LLM), then resolve them against existing ones in stages —
   exact normalised name, fuzzy name match (MinHash/Jaccard over 3-gram shingles,
   threshold 0.9), semantic retrieval, and an LLM call only for the unclear rest;
2. extract facts **only between listed entities**; each fact has a relation name,
   a `fact` sentence **written by the model** (paraphrased but required to keep
   every concrete noun, number and descriptor), `valid_at` / `invalid_at`, and the
   source episode ids;
3. for each new fact, retrieve existing facts on the same entity pair plus
   semantically similar facts (fact-sentence embedding / BM25), and ask an LLM:
   which are **duplicates** (reinforce, add the episode) and which are
   **contradicted** ("cannot both be true"; explicit replacement only, never from
   recency alone). Contradicted facts get `invalid_at` / `expired_at` — kept, not
   deleted (bi-temporal);
4. retrieval is hybrid (BM25 + fact-embedding cosine + graph BFS, then rerankers)
   over **facts**; what the answering model sees is the fact sentences with their
   validity dates.
Backends: a graph database (Neo4j / FalkorDB / Kuzu). Several structured LLM
calls per message, designed around GPT-4-class models.

**Mem0** (Apache-2.0). Core memory is a **list of short fact strings** about the
user ("User is a software engineer"), embedded in a vector store. Per message:
one LLM call extracts candidate facts from the **user's** messages; for each, the
top-k similar existing memories are retrieved and one LLM call decides
**ADD / UPDATE / DELETE / NONE**. A graph store is an optional add-on, not the
core.

**LangGraph** is an agent-orchestration framework (stateful multi-step LLM
workflows), not a memory system. **LangMem** is LangChain's memory SDK on top of
it: semantic (profile/collection), episodic and procedural memory with an LLM
"memory manager" doing extraction and update — the Mem0 pattern. Neither fits
ICE as a dependency: ICE is a proxy, not a LangGraph agent.

**Benchmarks.** LoCoMo / LongMemEval / BEAM scores are vendor-reported with
different models, judges and setups and contradict each other (e.g. Zep vs Mem0
on LongMemEval). Some papers show structured memory beating plain RAG on temporal
and multi-session questions; plain retrieval is a strong baseline on lookup-heavy
sets — which matches ICE's own tuning (vector-only 63/67 vs full 61/67 on an easy
set). "Works so well" is partly marketing; the shared design is still sound.

## 2. Why ICE's codex underperforms (measured, PROVENANCE 2026-10-15)

| | Graphiti / Mem0 | ICE codex today |
|---|---|---|
| What the model sees | a short, self-contained **fact sentence** + validity | the **whole source message** (median 3,488 chars for a 52-char sentence), 34% duplicating a turn already in the prompt |
| Entity resolution | exact → fuzzy shingles → embeddings → LLM for ambiguous | lowercase canonicalisation; paraphrases become new nodes |
| Duplicate / change | LLM compares the new fact with **semantically similar** existing facts | needs the same subject + relation term; 2 reinforced / 1 expired of 1,356 |
| Admission | LLM extraction + dedupe; no NER gate | NER-token grounding hides 70–89% of edges below the read floor |
| Pronouns | resolved to named entities in the fact sentence | `I` / `you` facts dropped |
| Relation vocabulary | free-form name, optional typed list | 197-word list + canonicaliser (G45), now open but still central |
| Model | GPT-4-class cloud | local 4B (`gemma4:e4b`) / NuExtract3 |

The pattern: ICE stacked individually correct safety gates on a design centred on
the triple and the raw sentence; together they remove most facts and show the
rest as raw text. The others centre on a **verified natural-language fact** and
use structure only to find duplicates and contradictions.

## 3. Review of the maintainer's target design

Target: *a relation is a term; the triple carries validity (from / until),
strength and confidence; the attached source sentence(s) are what the answering
model sees — close facts, not walls of text.*

**Valid and matches the field:** facts as the unit; bi-temporal validity with
replaced facts kept, not deleted; provenance to the source; compact rendering.

**Change one thing — what the model sees.** A verbatim source sentence is often
not self-contained: development-store sentences look like `yeah I was the one who
did that` or `it has that too` (illustrative; real ones withheld). Shown
alone they need the surrounding turn — which is how the current code ended up
rendering whole messages. Every system above shows a **self-contained fact
sentence written by the model** instead. ICE can do that *and* keep its
guarantee: the SUMMARY_CLAIMS_V3 contract built on 2026-10-15 already produces
`{sources, quote, speaker, text}` claims where `text` is model-written and
self-contained, the `quote` is checked mechanically against the source and the
claim is NLI-verified against it. **A v3 claim is the fact sentence this design
needs**; the verbatim quote stays attached for provenance ("where did I say
that").

**De-emphasise the relation term.** Duplicate/contradiction candidates found by
entity + fact-sentence similarity (Graphiti) do not depend on a relation
vocabulary — the thing that failed in ICE three times (G45, G69, B6). Keep the
relation label as optional metadata for display/filtering, not as the key.

**Replace "strength" with source count.** Reinforcement never fired in practice
(2 of 1,356); a fact's support is the number of distinct source turns behind it,
which dedupe produces directly.

## 4. Implications for the rework (proposals, to decide in the spec)

- **One write pass for summaries and facts.** Turn summaries (v3) already emit
  verified atomic claims; make each claim a stored fact with entity links,
  instead of running a second, separate extractor. Summary text = the turn's
  kept claims.
- **Entities:** staged resolution as in Graphiti (exact → fuzzy shingles →
  embedding → LLM for ambiguous); `I/me/my` and `you` resolve by speaker role.
- **Dedupe / change:** per new fact, retrieve similar current facts (same
  entities, fact-embedding neighbours); one LLM call returns duplicate /
  contradicted / neither, with both quotes shown; ordering by conversation
  sequence; contradicted facts get `valid_until` and a link to the replacement.
- **Read:** retrieve facts (BM25 + vector + entity match); render
  `[fact · speaker · date · current|replaced]` + the fact sentence; history only
  for past-tense questions; skip facts whose turn is already in the prompt.
- **Storage:** stay on Postgres + pgvector. Graph traversal beyond one hop is not
  required by any measured question type yet; add it only if Z2 multi-hop
  questions show the need.
- **Dependencies:** port the ideas, not Graphiti/Mem0 as libraries — they bring a
  graph server or vector service, assume cloud-class models and drop ICE's
  quote/NLI verification. Their prompts (fact extraction rules, the duplicate/
  contradiction prompt and its examples) are a good starting point to qualify on
  `gemma4:e4b`.
- **Risk:** entity resolution and contradiction detection are where every system
  struggles; qualify both on real B/C/dev data (planted corrections + the real
  updates the Z2 ledger finds) before building the read side.

## Sources

- Graphiti source: https://github.com/getzep/graphiti (prompts/extract_edges.py,
  prompts/dedupe_edges.py, utils/maintenance/edge_operations.py,
  utils/maintenance/dedup_helpers.py; read 2026-10-15)
- Mem0 source: https://github.com/mem0ai/mem0 (mem0/configs/prompts.py; read 2026-10-15)
- Zep paper: https://arxiv.org/abs/2501.13956
- Benchmark claims and disagreements: https://mem0.ai/blog/ai-memory-benchmarks-in-2026 ,
  https://dev.to/everest_an/-i-benchmarked-ai-agent-memory-in-2026-and-the-numbers-tell-a-different-story-than-the-marketing-2ae4 ,
  https://vectorize.io/articles/mem0-vs-zep
- LangMem: https://blog.langchain.com/langmem-sdk-launch
- Structured memory vs RAG examples: https://arxiv.org/pdf/2511.01448 (LiCoMemory),
  https://arxiv.org/pdf/2601.14287 (Chain-of-Memory)
