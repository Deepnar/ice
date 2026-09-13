# ICE v3 repair phase
Assumes decided specs: G63_extractor_decision.md (deployed extractor contract), BG_LAYER_FIXES.md (separate general background model)

## 1. Decisions

**Authorization, 2026-09-12:** implement the whole-system repair described in
[the review](../reviews/2026-09-12-v3-system-review.md), with focused behavioral
validation during implementation and broad Z-level comparisons afterwards.
The earlier per-change production approval and pre-reseed sequencing requirements
are superseded for this repair scope. Privacy, deletion, local-first behavior,
the frozen v2 record, publication freeze and AGENTS.md editing gate remain.
No additional G-number is needed to track this phase; existing items own fixes.

**Quality objective, clarified 2026-09-12:** correctness repairs are the
foundation, not completion. Optimize supported answer quality per prompt token
across exact recall, temporal change, cross-turn reasoning, procedures and scope.
Each memory representation must earn its token cost; equal leg shares are not an
objective. The end-stage evaluation must include matched-budget vector and
recent-history controls plus answer-quality-versus-token-cost curves. LME
saturation and universal wins are aspirations, never acceptance claims without
evidence. Do not tune on the final held-out benchmark.

First tranche: **G61 — extraction failures falsely completed**, e.g. a malformed
response becomes an empty list and permanently suppresses retry; and **G57 —
short-turn extraction**, e.g. a one-sentence correction must reach Codex.

A complete valid empty JSON list is success. Missing content, invalid envelopes,
malformed facts, non-boolean polarity and output truncated by the provider are
failures. Propagate errors to the existing bounded runtime retry queue. Do not
commit a partial graph or a completion key from a partial response. Remove the
unjustified emotion-object and self-reference filters: “Alex feels happy” and
“service monitors service” can be real claims. Existing source/name/polarity
checks still apply.

## 2. Algorithm and data model

Parse one complete JSON value (with optional fences and template thinking prefix).
Accept an array or an explicit `facts`/`triplets` envelope; require string, nonempty
subject/relation/object and a boolean `negated` when supplied. Unknown envelope,
scalar, trailing garbage or malformed element raises a typed extraction error
with reason and counts, not source text. Check finish reason before parsing.
Normalize/ground only fully parsed chunks. Commit graph changes and the completion
key atomically only after every chunk succeeds. Runtime retries a failed batch;
this deliberately favors atomicity over partial writes and avoids counting a
replayed partial edge as independent corroboration. No new retry loop or database
schema is needed for this first repair. Repeated successful chunks can cost
compute on retry; durable extraction artifacts belong with evidence versioning.

Run extraction on every existing non-private turn, regardless of lossless flag.
Guard privacy in the extractor itself as well as in post-flight. Missing turns
raise, enabling retry rather than false completion. Existing completed keys stay
valid; do not automatically reprocess the live store. The later versioned reseed
will exercise the expanded population.

## 3. Files and integration points

- `src/workers/extraction_result.py`: lightweight parse/error contract.
- `src/workers/codex_extractor.py`: strict completion, propagated errors, eligibility.
- `src/workers/post_flight.py`: all non-private turns reach extraction.
- Existing runtime retry/ledger behavior remains the consumer of failures.
- Focused parser tests and a live-DB writer test with controlled model responses.
- Roadmap, feature inventory and current architecture follow completed behavior.

## 4. Edge cases and failure modes

Empty JSON means no facts, not failure. Empty text is an output-contract failure.
An exception or cooperative user-activity yield must never become `[]`. A failed
later chunk cannot leave a completion key. Source data remains available on
failure. The standalone bookmark extractor must honor privacy too. No source
snippets in new warning/error events. Neither fixture cleanup nor migration work
may delete another task's rows.

## 5. Validation

- Valid empty arrays/envelopes, reordered keys, escaped strings and explicit
  negative facts survive; incomplete JSON, invalid fields, ambiguous envelopes,
  false-string polarity and provider truncation are rejected.
- Through `extract_codex`, a failed response leaves no completion key or graph
  writes; a subsequent healthy response completes; a legitimate empty result
  completes once. Test non-lossless and private turns explicitly.
- Through `evaluate_turn`, a short turn reaches extraction and a chained failure
  remains retryable despite completed representation evaluation.
- Smoke tests before each commit. Focused controls use real database transactions
  and controlled responses; they establish mechanics, not extractor truth rates.

## 6. Look-ahead

### Shared turn representation repair (2026-09-12)

Retrieval, the recent chat window and the explicit recent-turn service must use
one eligibility/selection function. A summary needs finite measured coverage in
the configured interval `[turn_summary_coverage_threshold, 1]`; NULL is unknown,
not a passing result. Coverage remains term retention, not semantic verification.
Do not silently cut raw evidence at 300 characters when no summary qualifies.
The caller's explicit token budget remains responsible for fitting raw text.
Protect **each** query term present in raw from disappearing during compression,
not merely any one matching term. Preserve current intent preferences after
eligibility is established.

Abstracts have no independent support score in the current schema. Until the
source-verification repair supplies one, allow an abstract as a budget alternative
only when it is a verbatim source span and preserves the matched query terms;
never inherit a different summary's coverage score. This verifies extractiveness,
not completeness or context-independent truth. Generated abstracts remain stored.
No trustworthy representation and no raw source means no injected text.
Recent-window degradation must consume only the returned eligible alternatives.

Validate null/invalid/threshold coverage, details after character 300, multiple
query terms, independent abstract eligibility, and actual retrieval, chat-window
and service readers. This tranche does not claim to repair rolling summaries or
replace term coverage with NLI; those remain in this phase below.

### Relevance reranking and token selection (2026-09-12)

Decision: qualify one local instruction-aware cross-encoder,
`Qwen/Qwen3-Reranker-0.6B`, on bounded synthetic controls before enabling it.
Compared with the English MS-MARCO MiniLM baseline, its multilingual/code
coverage and longer context better match ICE's input contract; compared with
BGE-v2-m3 it also accepts a task instruction. These are selection reasons, not
ICE benchmark wins. References: the official
[Qwen card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B),
[BGE card](https://huggingface.co/BAAI/bge-reranker-v2-m3),
[MiniLM card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
and [Graphiti search recipes](https://help.getzep.com/graphiti/working-with-data/searching).

Rank scoped candidates after fusion and before conversation caps/provenance
collapse can discard a better answer. Bound candidate count and model batches.
Score the **actual rendered text** and every eligible compressed alternative;
compression may not inherit the original text's relevance. Preserve all source
identity fields when selecting a representation. Then pack in relevance order,
without guaranteed source-type shares. Skip oversized candidates and continue
to smaller candidates, rather than stopping when one round cannot fit.

The model's yes/no logit difference is a relevance score, not a confidence that
the claim is true. A rejection floor must be explicit and qualified on both
answer-bearing and irrelevant controls; never apply an embedding-cosine floor to
this score. All-irrelevant candidates may produce empty context. No claims of
complete multi-hop reasoning from pairwise ranking. Diversity and redundancy
remain separate checks after ranking.

Load only local cached weights on the request path, with a pinned revision,
serialized model access and bounded input tokens; retain only the last-token
logits, disable generation KV caching, and return weights to CPU after scoring
so the main generation model does not lose permanent VRAM; do not silently truncate
unscored evidence. Model unavailable/invalid scores/over-budget inputs produce
a warning on every degraded call and retain the established fusion fallback.
No cloud calls or corpus uploads. Confirm memory footprint and latency before
default activation; add an ablation switch. Both normal and wide-net retrieval
must reach this stage. Controlled checks establish ordering, negative rejection,
invariance, actual budget decisions and fallback mechanics; final matched-budget
answer quality remains an end-stage evaluation.

Qualification decision: the first 15 synthetic queries (five intents, three
forms each) ranked the answering item first 15/15, but a zero-logit gate admitted
4/45 distractors and those admissions changed with phrasing. **Activate ordering
only after integration validation; keep the rejection floor unset by default.**
Do not tune a threshold to these controls. A configured experimental floor
remains available, with its rejection result explicitly unqualified. This is
relevance ordering, not a complete relevance/abstention solution.

### Lexical query preservation (2026-09-13)

Normalize the entire parameterized query with the same PostgreSQL `english`
text-search configuration used for stored text. OR together the resulting
quoted lexemes; do not strip digits/Unicode or discard terms after word 30.
Stopword-only input produces no lexical matches. Query punctuation is data,
never caller-supplied tsquery syntax. Keep existing scope, privacy, time, decay
and candidate limits; keep the existing warned AND fallback for database errors.
NER remains independent of lexical normalization. PostgreSQL `ts_rank` is the
current ranker, despite the historical BM25 name; do not claim true BM25 scoring.
Validate numeric, Unicode, punctuation, late-term and negative/scope cases
through the real database leg, including absence of fallback warnings.

### Timestamp presentation and provenance (2026-09-13)

Use existing episodic timestamp/provenance and Codex learned/valid times. Render
date and clock time with timezone and a current UTC datetime anchor. Recorded
time is not necessarily the event time described in the source. Label synthetic
import times; unknown timezone/provenance must not become UTC/original. Do not
invent event times or rewrite historical records in this presentation repair.

For explicit fact lines, resolve source-batch timestamps in batched queries after
scope filtering, separately from learned/recorded-validity time. Keep timestamps
on every eligible episodic alternative, chunks and recent history, and count
them in budgets. Summary prefixes name creation/update time, not event time.
Timeline timestamps describe recorded validity. Cold storage lacks timestamp
provenance today; expose that as unknown until the lifecycle repair preserves it.

Subsequent coherent repairs: conflict/time semantics and separation of evidence
from usage; shared read preparation, provenance and final-selection tracing;
source-backed sentence claims and qualified verification; consistent summaries;
procedural evidence, lifecycle/scoping/adapter controls; inference/runtime and
remaining operational defects; then the minimal user correction surface.

The future recorded-response replay must run pre-flight for every historical
prompt and post-flight on its recorded answer, preserving chronological state,
scoping and reinforcement. This is state reconstruction, not a measurement of
answer improvement. Broad LME/LSREP comparisons and multi-source evaluation-data
work remain the end-stage confirmation, as requested.

## 7. Traps

No nonempty-only completion gate: it retries honest empty results forever.
No salvage-regex success: it loses reordered/negative facts and marks truncated
output complete. No partial graph replay: it can manufacture reinforcement.
No lossless check hidden in a second caller. No broad exception swallowing a
cooperative yield. No claim that smoke tests measure semantic quality.
