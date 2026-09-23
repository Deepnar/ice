# ICE v3: fresh system review — 2026-09-12

This is a design review, not an experiment result or an implementation approval.
It covers the current v3 handoff, open roadmap inventory, selected architecture
and feature-inventory sections, and source paths behind the findings below.
No models, benchmark jobs, database mutations, or production edits were run.
Source inspection establishes mechanisms, not their frequency or user-visible
impact. Historical v2 paper and LongMemEval results are not v3 evidence.
The roadmap remains the queue; this document explains the recommendations.

## Judgment

ICE v3 has enough evidence to enter a focused implementation cycle. It does
not need another broad model competition or a fresh measurement for every
already-understood defect. It still needs bounded validation of replacements:
the project has repeatedly made the intended metric improve while actual
memory became worse.

The common design problem is that several distinct questions are represented
by loosely coupled proxies: was a name mentioned, is a claim supported, was it
asserted rather than hypothesised, is it still current, is it relevant now,
and did it help the answer? These require different evidence. A high embedding
similarity, repeated retrieval, or high term coverage cannot stand in for all
of them.

My recommendation is an evidence-backed memory system: retain source records
under the user's retention/deletion policy; derive searchable claims, summaries,
and graph links from them; preserve attribution and uncertainty; make every
transformation replaceable and traceable. Build on the existing episodic store,
source-batch references, event journal, services, and scope filters. A rewrite
of the storage stack is not required.

## What the current code makes concrete

| v3 finding | Source evidence | Implication |
|---|---|---|
| Extraction and general background generation are separate jobs with separate model selection. Template extraction, a 3,000-token output cap, and adaptive chunks capped at 4,096 are already configured in source. | `src/api/config.py`, `codex_extraction_model`, `codex_extraction_mode`, `codex_extraction_max_tokens`, `codex_extraction_chunk_max`; August 27 handoff | Do not reimplement the extractor migration or describe the old 550/1,200 limits as current. The larger-chunk question remains separate. |
| Graph extraction is gated twice by the lossless decision. | `src/workers/post_flight.py::evaluate_turn`; `src/workers/codex_extractor.py::extract_codex` | The already-decided inclusion of non-lossless turns requires both callers to agree. “I no longer use tool X” can be short and consequential. Importance and compressibility need separate decisions. |
| Summary eligibility is not controlled solely by the stored injection flag. | `src/retrieval/orchestrator.py::_choose_representation` | This reader independently uses literal coverage 0.7, treats missing coverage as trusted, and can select a summary by intent. Changing only the writer or flag does not establish a system-wide policy. |
| Recent history can degrade to summary or abstract without checking coverage in that branch. | `src/api/prompt_assembler.py::get_recent_turns` | A verified retrieval chooser would leave another route into the prompt. Abstracts need their own support check; a faithful summary does not automatically validate a separately generated abstract. |
| The rolling fold sees capped turn representations and its previous generated summary. | `src/workers/conversation_summary.py::_representation`, `_fold_prompt`, `run_conversation_summaries` | Two distinct risks: omitted source before generation and accumulated invention. Prompt softening alone addresses neither completely. |
| Codex edges preserve a source batch, polarity, and time fields, but have no first-class claim sentence, evidence span, speaker, or assertion-status fields. | `src/memory/models.py::CodexEdge`; extractor passes `turn.raw_text` | Attribution can be recovered from source text, but readers are not given a structured distinction between “user said X” and “assistant suggested X.” |
| Graph strength affects effective trust; reinforcement happens inside candidate generation. | `src/retrieval/orchestrator.py::_codex_graph`, `_reinforce_codex_edges`, `_edge_trust` | An edge can receive exposure credit before final fusion/budget selection. Frequency is not corroboration, and candidate generation is not successful use. This review has not measured its impact. |
| The converse/antonym conflict branch expires an old edge without the reconciler. | `src/workers/codex_extractor.py::check_conflict`, `reconcile_conflict` | Separate “relations must not canonicalise together” from “these assertions cannot coexist.” A person can buy and sell the same asset on different occasions. Historical non-firing does not validate the rule. |
| Explicit context pulls have their own preparation and serialization. | `src/services/retrieval_svc.py::context_for` | Rechecked 2026-09-23: returned dictionaries now include origin batch/edge IDs and producing leg. It still classifies without conversation context and budgets with zero turn/history counts. Forced retrieval is intentional; remaining preparation drift needs a scoped behavior check. |
| Manual cluster assignment wrote only the legacy single cluster column. **Repaired 2026-09-23:** it now adds authoritative membership links; selected-cluster vector retrieval includes the assigned turns and excludes a turn linked only elsewhere. | `src/services/clusters.py::assign_turns`; `tests/test_services.py` | This was a live mechanics mismatch. The passing scoped SQL control establishes link visibility, not an answer-quality gain. |

These are code-inspection findings. They do not establish new error rates.

## Natural Language Inference: useful, but at a specific boundary

NLI asks whether a hypothesis follows from a premise, contradicts it, or is
undetermined by it. In ICE v3, the useful pair is **source passage → proposed
claim**, not **old generated memory → new generated memory** as the only check.

Example: source “We considered SQLite, then chose PostgreSQL.” A candidate
“PostgreSQL was chosen” is supported; “SQLite was chosen” is not. A checker
must preserve the difference between considering, choosing, rejecting, and
merely mentioning. A reversed relation is often *unsupported*, not logically
contradicted: “Alex follows Sam” does not by itself rule out Sam following Alex.

The sentence unit matters. [SummaC](https://aclanthology.org/2022.tacl-1.10/)
identifies the mismatch between sentence-trained NLI and document-level
consistency checking, and uses segmentation and aggregation. That supports
checking individual claims against complete relevant evidence windows rather
than truncating a long conversation into one classifier input.
[AlignScore](https://aclanthology.org/2023.acl-long.634/) is another relevant
factual-consistency approach; neither paper establishes performance on ICE v3.

Use a local verifier asynchronously at the write boundary, with verdict,
evidence references, model/version, and an abstention outcome persisted.
Candidate discovery can use embeddings. A checker seeing insufficient evidence
must abstain; it must not label the claim false just because the necessary
sentence was outside its input window. Multi-turn claims require multiple
supporting passages. Negation, speakers, conditions, numbers, dates, and fictional
world/project scope must survive preparation.

**Attribution precedes verification.** “The assistant hypothesised that the
service needs a cache” supports a record of that hypothesis; it does not make
“the service needs a cache” established fact. NLI checks consistency with supplied
text, not truth in the world. A later assertion can also be a legitimate change,
not an error. Time and correction policy must decide how it affects history.

Do not use NLI alone to merge entities, canonicalise arbitrary relations, expire
old facts, or promote a hypothesis into a fact. Do not mistake the model's score
for a calibrated probability of truth. [HANS](https://aclanthology.org/P19-1334/)
shows NLI systems can exploit word-overlap and syntax heuristics: precisely the
kind of role-reversal mistake this project needs a verifier to catch.

Recommendation: implement one replaceable verification interface; qualify one
candidate on a small, existing, human-grounded error set before allowing it to
authorize summary substitution. No model tournament. Keep unsupported or
unverified derivatives distinguishable and retain their source; do not erase
source facts because a checker failed.

## Sentence-based Codex: yes, with evidence

**G64 — searchable fact sentences, e.g. “Project Atlas chose PostgreSQL” — is a
useful architectural step.** Preserve the original supporting span, a readable
claim sentence, source message/batch, scope, attribution, assertion status,
time, and existing triple links. Add a fact-search path using the sentence
embedding/text, with the same privacy and scope rules as existing retrieval.

A sentence generated from an already-reversed triple merely spells the mistake
out. Extract the sentence from source evidence and verify it independently of
the triple. Sentence and triple agreeing is not independent corroboration when
the same model generated both. Render the supported sentence with source and
status; use triples for navigation and explicit relational operations.

The current graph does have descriptor fallback and entity-less enumeration.
The gap is **general direct fact search**, not a total absence of fallback.
Do not automatically copy another memory framework's deduplication or expiry
policy. Keep the already-established conservative merge policy and open relation
vocabulary. A global dictionary that learns every proposed synonym can amplify
one mistaken merge; semantic equivalence requires evidence in context.

## The rest of ICE v3

| Area | Recommendation | Existing owner / worked example |
|---|---|---|
| Episodic storage and admission | Separate durable-source retention, extraction eligibility, and compact prompt representation. Respect deletion and incognito throughout derivatives. | **G57 — short turns excluded from Codex**, e.g. a one-sentence correction. |
| Summaries and compression | One eligibility policy for turn summaries, abstracts, recent history, rolling/batch summaries. Rebuild bounded summary sections from original evidence; retain source links. Check faithfulness, useful detail, and compression separately. | **G73 — repeated folds accumulate invention**, e.g. an invented decision survives the next fold; **G75 — term coverage is not a truth gate**, e.g. a false summary repeats all project names. |
| Retrieval and prompt assembly | Trace candidates → selected fragments → final prompt → answer evidence. Prefer compact supported facts with source access; do not infer relevance from leg presence or require every leg a quota. | **G70 — actual memory use**, e.g. a returned graph fragment gets dropped by final budgeting; **G66 — answer benefit**, e.g. graph on/off changes whether the answer recalls a decision. |
| Classifier and routing | Keep learned classification, but audit downstream heuristic overrides and unnecessary hard gates. A query rewrite may vary wording while preserving intent. No new routing learner without a demonstrated routing problem and training signal. | **G28 — style invariance**, e.g. punctuation changes a retrieval decision; **B3 — learned model routing**, e.g. choosing a specialist needs actual specialist-performance evidence. |
| Procedural memory | Distinguish explicit user preferences from inferred habits. Require independent supporting occurrences, relevant conditions, contradiction handling, and a direct correction path. Ten citations from one extraction are not ten independent confirmations. | **G49 — activation previously depended on unmatchable re-emissions**, e.g. paraphrases of one habit failed to reinforce; **G55 — presence-only metrics**, e.g. any returned habit scores success. |
| Time and reconciliation | Preserve current state alongside supported history. Default ambiguous relations to coexistence; distinguish inverse direction, contradiction, correction, and temporal change. | **G60 — unknown relation semantics**, e.g. several tools can coexist; **G62 — converse-triggered expiry**, e.g. buying and selling are not automatically mutually exclusive; **T5 — temporal label lacks its intended consumer**, e.g. “what changed since May?” |
| Clusters and scoping | Treat member selection, centroid retrieval, labels, and manual control as separate questions. Inspect actual members. A poor label is not proof that every member is misfiled. Scope correctness outranks label polish. | **G74 — invalid cluster-naming evaluation**, e.g. arbitrary consecutive turns were passed as a cluster; **G29 — duplicated policy drift**, e.g. legacy assignment differs from link-based readers. |
| Maintenance and decay | Treat archival, compaction and promotion as state transitions with replayable evidence. Test one bounded lifecycle, including deletion and restart. Avoid letting frequently retrieved errors immunize themselves. | **G67 — long-running maintenance outcome unconfirmed**, e.g. a memory ages, is retrieved, and is archived/resurrected. |
| Runtime and inference transport | Finish explicit context/residency controls, output-contract checks, structured failure versus legitimate empty extraction, and retry state. Keep selected extraction and general background models separate. | **G32 — native inference controls**, e.g. explicit context size; **G61 — silent extraction loss**, e.g. a failed generation is recorded as permanently processed. |
| Chat, MCP, imports, documents, coding | Share preparation/provenance where semantics agree; keep explicit pulls intentionally distinct. Exercise scope and deletion across adapters. Add project-document ingestion as the thin missing caller, not a second document system. | **G29 — adapter drift**, e.g. MCP loses source origins; **E10 — project docs integration**, e.g. README ingestion under project scope. |
| User control and frontend | Pull forward a minimal inspect/correct/reject/forget surface once the core representation is stable. It completes memory quality; graph visualization and broad settings UI can follow. | **F2 — review queue interface**, e.g. resolve a disputed fact; **F5 — telemetry interface**, e.g. see why it appeared; **F1 — packaged frontend foundation**, e.g. boot and connect reliably. |
| Operational hardening | Finish blocking-call/async hygiene, bounded queues, auditable actions and log privacy in touched paths. Keep retired launch scripts out of the work plan. Explicit context controls matter before caching. | **G24 — async hot-path work**, e.g. a synchronous call stalls concurrent requests; **G25 — log privacy**, e.g. source text should not leak into routine diagnostics; **G17 — action audit**, e.g. explain who expired an edge. |
| Classifier feedback | Keep promotion consent-gated, verify the live checkpoint actually changes and has a rollback, and collect corrected labels through the eventual UI. Do not schedule continual retraining just because the worker exists. | **B4 — feedback and promotion**, e.g. a corrected memory-intent label reaches training and a qualified checkpoint reaches inference; **H5 — training scheduling**, e.g. work waits while the user needs the GPU. |
| Performance and product expansion | Profile after the memory contract settles. Avoid invalidation-heavy caching, new routers, larger graph infrastructure, multi-user work, and full frontend breadth in this repair cycle. | **C13/C14 — retrieval/prefix caching**, e.g. stale cached fragments hide a just-written correction; **B6/F13 — conversation branches**, e.g. branch-specific history needs lineage semantics. |

Keep the useful infrastructure already built: raw episodic evidence, hybrid
candidate retrieval, user scoping, shared services, temporal/event records,
local execution and explicit user overrides. Their existence does not prove
quality, but their replacement is not the diagnosis either.

## A bounded implementation order

This is a **proposed priority change**, not a silent cancellation of earlier
user decisions. In particular, **G75 — summary substitution, e.g. a generated
summary replaces raw evidence — currently has an explicit post-reseed gate**.
Changing that sequence needs the maintainer's decision before production edits.

1. **Finish known correctness work and consolidate the execution contract.**
   Explicit generation success/failure and retry state; conservative conflict
   handling; the approved non-lossless extraction change; shared preparation,
   provenance and final-prompt tracing; correct the model/config documentation.
   Validate each mechanism in isolation and through its real caller. No broad
   quality campaign needed to establish that a failed call is not an empty fact set.
2. **Implement evidence-backed claims and consistent summary eligibility.**
   Define the claim/source/attribution contract first. Add sentence retrieval
   and source-backed rendering, then the bounded verifier if it passes its
   local gate. Replace recursive unsupported summary accumulation using original
   evidence. Land these as separable changes so a failed check identifies a cause.
3. **Use one properly versioned reseed and one compact answer review.**
   Avoid repeatedly rebuilding the same large store. Save writer/model/config
   versions, source mapping and stage outcomes. Reuse validated raw outputs to
   isolate downstream changes, and also check the live path before adoption.
4. **Only after the repaired core helps answers, finish maintenance and the
   minimal correction interface; then broaden tuning and product work.**

A small answer review should include real saved failures plus held-out ordinary
requests: exact recall, paraphrase, cross-turn synthesis, current vs old state,
negation, rejected plans, assistant hypotheses, procedures, long documents,
manual/incognito scope, and memory-irrelevant questions. Vary style while holding
meaning fixed. Compare matched recent-history-only, episodic retrieval, and full
memory contexts under the same final budget and answer model; inspect which
source actually supported the answer. Report latency and token cost as well.

Use roughly twenty representative cases to discover obvious regressions, not to
claim a small percentage improvement. If the effect is ambiguous, expand only
the affected comparison. A graph on/off null can mean redundancy, the wrong
queries, failed delivery, or insufficient resolution; it does not by itself
prove the graph has no possible value. Retrieval counts likewise do not prove
answer benefit.

Stop reopening settled model choices absent a changed requirement or reproduced
failure. Do not remeasure superseded historical findings just to refill a table.
Retire them explicitly if no current decision needs them. Keep a small reusable
source-backed acceptance set and one common judge input builder instead of
creating another slightly different harness per question.

## Documentation corrections and limits

The handoff correctly retracts cluster naming, but the experiment index still
advertised the withdrawn error rate. The reseed spec conflated general background
and extractor selection. The extractor decision entry still said its configuration
was undecided after its own spec recorded completion. The fact-sentence entry
claimed there was no entity-less fallback and cited a grounding example that the
project's own later investigation established was supported by the source.
Those stale claims should not drive new implementation.

This review records the missing claim-attribution/verification work in the
roadmap and attaches the other findings to their existing owners. No checkboxes
are earned by this review. No claim that every remaining implementation is
correct, or that NLI will fix all ICE v3 errors, is justified without execution.
