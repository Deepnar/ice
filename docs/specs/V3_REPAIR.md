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

**v3 divergence and source-only decision, 2026-09-25:** a recorded-conversation
replay exposed a NuExtract template row with a nonempty subject/relation,
`object: null` and an exact `source_sentence`. The strict parser aborted the
whole turn; a repeat without the sentence field still produced a null object
in a later chunk. This is an attempted unary relation that cannot enter ICE's
three-slot graph, but its exact sentence is usable evidence. With attributed
sentence claims enabled, accept only this *typed source-only row* when its
source sentence occurs exactly in the current original chunk. Store that
sentence through the normal role-aware claim writer and omit the invalid
triple from graph canonicalization, reinforcement and NLI relation approval.
Count and warn on the fallback so a run cannot mistake it for graph coverage.
Any missing/blank/unsupported quote, null subject or relation, other invalid
field, or the same null-object row with sentence claims disabled still raises
and remains retryable. A source-only row is not an empty extraction or a
corroborating graph observation. This narrower output contract preserves the
original strict-failure rule for malformed graph assertions while allowing a
source sentence to survive when the specialist cannot express it as a triple.

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

### Authoritative source boundaries (2026-09-13)

Store compact source spans over immutable raw text: raw SHA256 plus ordered
role/start/end segments. API and replay writers know user/assistant boundaries;
document ingestion and explicit notes record document/user roles respectively.
Do not infer authorship from strings such as "Assistant:" inside user text.
Validate hash, ranges, nonoverlap and roles before using stored attribution;
legacy/malformed records return one unknown-speaker source unit. No backfill
from formatting conventions. The recent chat reader consumes authoritative spans
when rendering raw pairs; existing legacy rendering is presentation only and
must not seed verified claims.

Add nullable JSONB source_spans to episodic and cold storage; preserve it through
archive/restoration. Preserve timestamp provenance through that lifecycle too,
so imported times cannot silently become original after restoration. Migration
is additive and does not reinterpret existing records. Sentence-claim extraction
and verification consume these units next; this boundary alone does not qualify
summaries or extracted triples.

Validate adversarial role-like text, Unicode offsets, malformed/stale spans,
actual writer/read paths, archive/restoration and migration roundtrip in a
disposable database before upgrading the working store.

### Preserve retention without manufacturing support (2026-09-13)

User clarification: retain the original useful-memory survival loop, separately
from support/promotion. The A3 completion record confirms that strength originally
mixed usage, decay and corroboration. Removing only the read increment leaves
true quiet facts vulnerable to automatic expiry; that is incomplete repair.

Keep strength as bounded retention priority. Record usage_count/last_accessed_at
only for edge IDs attributable to rendered fact lines, after chat evidence
survives final ledger eviction or explicit context is returned. Name the stage
prompt_prepared/context_returned; neither means answer-used. Candidate lookup
never reinforces. Preserve origin_edge_ids through fragment replacement and
serialization. The write-off switch covers this signal too.

Decay retention priority to a nonzero floor, without confidence demotion or
valid_until/unlearned_at changes. Nonuse does not establish falsehood; quiet facts
remain queryable. No automatic reinterpretation of already expired legacy edges.
Explicit supersession/deletion remains separate. A bounded retention bonus ranks
quality-qualified edges; low source confidence cannot cross the quality floor
through repeated reads. Retention is not a calibrated probability.

Track distinct observed source batches separately. Repeated extraction of the
same batch cannot increase strength or promote; another batch may update source
observation count and extraction confidence. Read strength never triggers that
promotion. This prevents replay/read inflation, not same-speaker echo or model
truth errors; attributed sentence evidence will address those next.

Validate quiet-fact survival across accelerated decay, usefulness refresh without
confidence promotion, replay idempotence, exact served-edge IDs, final-eviction
exclusion and read-off. Preserve old published v2 behavior only in frozen docs.

### Reading is not corroboration (2026-09-13)

Remove graph strength/promotion writes from candidate retrieval. A fact does
not become better supported because its entity matched repeated questions,
and rejected candidates must never gain evidence strength. Delete the obsolete
read-promotion settings and update existing harness callers, without redesigning
experiments. Extraction corroboration remains a separate writer pending its
source-ledger repair. Existing candidate provenance is not precise final-prompt
or answer-use evidence; do not mislabel it.

The retrieval write switch must also cover cold resurrection. Deduplicate
selected episodic row IDs before recording access, so multiple chunks from one
turn count as one read; apply their access/decay changes in one transaction.
Keep this read-popularity effect separate from graph truth. Validate the actual
graph leg cannot strengthen/promote, multiple chunks count once, and write-off
prevents both access and cold resurrection. Final-prompt usage tracing follows
with evidence identities in the same repair phase.

### Relation-name conflict repair (2026-09-13)

Relation canonicalization's anti-merge map protects direction/polarity; it is
not evidence of contradiction. Separate that map from the small opposition
candidate map. Converses such as buys/sells and teaches/learns_from must coexist.
The ordinary writer must match the complete relation and polarity before
reinforcing; sharing two endpoints with a different relation does not establish
replacement or justify immediate activation.
Opposition candidates (for example friend/enemy) use the existing source-aware
reconciler or review path, never deterministic expiry. Background contradiction
candidates without original source evidence become review proposals, not expiry
instructions; deduplicate already reviewed as well as outstanding proposals by
the old/new edge pair. REST/MCP review approval must accept an explicit
`keep_edge_ids` subset (both/one/neither); missing choice leaves the item pending
and raises a validation error. Apply only selected expiries, journal them and
refresh both endpoint payloads. Never silently approve a no-op unknown action.

Reconciliation must parse an exact complete decision (not a substring), consume
the full supplied source within an explicit bound, and treat missing source,
truncation or invalid response as review. This repairs the vocabulary-authorized
expiry mechanism; attributed claims, source clocks and open-vocabulary conflict
resolution follow in the same phase. Do not relabel this as full conflict repair.
Validate coexisting converses, preserved canonical separation, no-source/invalid
reconciler behavior, and the actual background detector/apply path.

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

**Procedural evidence re-grounding (2026-09-24, shipped in v3):** `extract_procedural`
previously accepted any two citation integers, used all turns in the session as
support, and activated on their count; `_user_half` treated unknown-role raw
text as the user's words. The separate reflection writer had no cited evidence
at all. The repaired writer resolves only in-range citations against writer-attributed
user turns, persists just those cited batches, gates activation on the cited
count, and counts independent sessions by resolving cited batch IDs through warm
or cold originals. Project-scoped patterns match only their project. Unknown
speaker and unavailable support abstain. Reflection may report observed
patterns in its session summary, but does not write `procedural_memory` from
uncited snippets. Test invalid citations, two of ten cited
messages, same-session re-extraction and cross-project matching through the
actual database writer; do not infer semantic truth from a valid citation.

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

### Source sentences before graph assertions (2026-09-13)

Decision: add extractive sentence claims within Codex, independently searchable
without a recognized entity. NuExtract's template adds `source_sentence`; this
must be an exact span of the supplied source, not a verbalized triple. A bounded
live specialist check preserved a choice, a conditional and direction in three
source sentences, while the conditional's triple flattened its modality. Thus
render the attributed source sentence, never infer its truth from the triple.
The NLI candidate falsely entailed three of14 unsupported controls (conditional,
quoted denial, negated reporting); it does not authorize assertion/promotion.
No threshold is fitted to hide those failures. Keep it a diagnostic candidate.

Add `CodexClaim`: source batch, raw SHA256, exact start/end, source role,
source sentence, native1024 embedding, optional graph edge link, created time.
Unique source batch/hash/span prevents overlap/retry duplication. It is an
attributed source excerpt, not an independently verified world fact. No generated
sentence is accepted without an exact source match. Unknown legacy speaker stays
unknown. Extract each authoritative role unit independently; retain all units,
including assistant suggestions as assistant text. Ambiguous repeated matching
spans are retained independently. Expand an extracted span to its complete source
paragraph when surrounding text exists, so a chosen sentence does not hide a
nearby condition or correction. Store/search that context-preserving span.

Claim insertion and existing graph completion remain one transaction. Preserve
source sentences even when endpoint-name/canonicalization rejects their triples;
claim retrieval must not inherit NER's blind spots. Existing graph semantics are
unchanged in this tranche and remain a separate conflict repair. The specialist
setting wins over the foreground answer's `model_used`; explicit extraction-arm
overrides remain available only through `extract_triplets`.

Direct sentence lookup combines parameterized PostgreSQL lexical matching and
native embedding similarity before the existing global reranker and packing.
Return source_type codex, leg codex, with exact batch and optional edge lineage.
Respect explicit empty scope, batch/conversation/cluster inclusion, exclusions,
private/incognito restrictions and source availability. Join live source rows for
visibility; cold-source support must be explicit before claiming archive parity.
Date labels identify recorded source time, not event validity. These are historical
source excerpts; graph retirement does not change what a source said. Never
render orphan claims after source deletion. Graph navigation remains alongside
this direct lookup; replacing cached triple notes requires its own follow-up.

Tests: exact quotation and context boundaries; source roles and marker spoofing;
unknown/mismatched source hash; overlapping duplicate claims; rejected endpoints
still searchable; real SQL lexical/vector paths with no matched entity; privacy,
empty scope, exclusion, source deletion, budget packing and provenance. Additive
migration roundtrip before working-store application. No automatic reseed or
benchmark claim. Summary verification and autonomous semantic conflict resolution
remain in the same repair phase.

### Source-support verifier follow-through (2026-09-14)

The first failed NLI candidate does not close source verification. A second,
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` pinned
`b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`, passed13 supported and17 unsupported
controls, including correctly attributed suggestions and conditional claims.
Select it for the replaceable verifier implementation. Training includes
adversarial inference; that is a selection rationale, not a production guarantee.
The tiny Hindi/German subset does not qualify broad multilingual support.

Use local cached weights, float32 as tested, serialized inference and CPU offload
between calls. Each premise/hypothesis pair must fit512 model tokens in full;
never truncate a source or maximize over windows and call the whole source
verified. Oversized, missing, failed or nonfinite scores yield explicit unknown,
with a warning each degraded call. Return source/claim hashes and model revision
alongside all three scores. Candidate status: entailment>=0.95 is supported,
contradiction>=0.95 contradicted, otherwise unknown. These conservative policy
cutoffs are not calibrated probabilities or benchmark-tuned thresholds.

A source-supported claim remains attributed to its speaker and time; support
never establishes world truth, author independence, or a correction's effective
date. Contradicted/unknown does not delete source evidence. The downstream choice
is faithful source text instead of an unqualified assertion/summary, preserving
recall. Test both the scorer and each consuming decision; do not count a loaded
model with no consumer as a finished verification repair.

Verifier consumer refinement: each CodexClaim keeps both the exact selected
sentence and its complete containing paragraph. Verify paragraph -> sentence.
Only a supported, hash-current verdict permits the shorter sentence at read time;
unknown/contradicted uses the whole paragraph. This is faithful compression of
attributed evidence, never a license to flatten suggestions into user decisions.
The independent source SHA/offset check still precedes every read. Save both
texts and the verdict atomically; failure retains the full source representation.

Claim/edge linkage uses a many-to-many CodexClaimLink table: one quoted sentence
can support several graph candidates and one edge can have multiple attributed
source excerpts. Graph writes return their edge to the caller; link only the
matching source_sentence. Rejected graph candidates still keep their source
excerpts. Deleting an edge removes its navigation links, not source evidence.
Conversation deletion cascades claims; turn-forget deletes by original episodic
ID, which remains stable through archival. Claim search uses source exclusions;
entity deny sets derived from excluded batches must not hide unrelated claims.
Deletion of one of several graph sources also changes the edge: the shared
conversation/forget service confirms surviving non-private warm/cold originals
across the primary, observed and `edge_added`/`edge_strengthened` batches. It
rebases a deleted primary to a surviving batch so the corresponding claim link
continues to render (prefer a current linked quote, else the newest original);
it prunes deleted secondary observations, and expires an
edge whose last original is gone. A read or a second event in the same deleted
conversation does not count as independent support. The historical maximum
extraction confidence cannot yet be decomposed by source; G76's future
per-source trust record must preserve that information at write time.

Scoped graph reading must use the same source set. A conversational edge can
have an original primary batch in conversation A and a later independent
observation in conversation B. Under B-only scope, read B's current attributed
claim and credit B's batch; do not reject the edge because A was first. Warm
and cold originals participate in explicit allow and deny sets. An excluded or
private batch never supplies the rendered quote or origin credit, even when a
different observation keeps the edge visible. A derived code-graph edge keeps
its separate project-batch rule. A source-free/stale linked edge contributes no
unverified quote from a deleted source. Scope tests must cover traversal,
relation-fact and enumeration routes plus the production graph leg. Historical
time windows must not cite a later source observation for an older valid edge;
select a source recorded by the window end or abstain.

### Edge-proposition support before a new Codex assertion (2026-09-24)

The sentence-level verifier above checks a quote against its containing source
paragraph. That is faithful compression, not proof of the extractor's proposed
`subject --relation--> object`. The live writer currently calls `handle_triplet`
even when no quoted claim matched. G51's source-backed relation recovery needs
this missing boundary first.

For each proposed triple, match the exact `source_sentence` to an attributed
`CodexClaim` in the same role unit. Render a bounded proposition from the open
relation by replacing underscores with spaces: `Mira --lives_in--> Berlin`
becomes `Mira lives in Berlin.`; a negated edge becomes `It is false that Mira
lives in Berlin.` Preserve direction, polarity and exact entity names. Verify
the *complete containing paragraph* → this proposition with the configured
local NLI model, without truncation. No exact claim, unknown role, incomplete
paragraph, uncertain/contradicted NLI or absent relation text ⇒ do not call
`handle_triplet`; retain the quoted claim and raw turn for retrieval and later
reprocessing. A supported assistant assertion keeps its assistant attribution;
an assistant suggestion is not converted to a project decision merely because
the graph writer saw the sentence. NLI support is source consistency, not world
truth or independent corroboration.

`codex_sentence_claims=False` is an explicit opt-out from this source-claim
path. Preserve its previous graph-writing behavior and warn on every batch that
these legacy writes are unverified; never silently turn that setting into an
empty graph. The default remains `True`.

Persist the per-edge/per-claim verdict (including hashes/model revision) on the
existing `CodexClaimLink`, not on `CodexEdge` (one edge can have multiple source
claims) and not on `CodexClaim.verification` (one sentence can propose several
relations). Batch independent NLI pairs in one inference lease to avoid one
model transfer per triplet. An overlong pair returns explicit unknown; a
verifier outage aborts the transaction for retry instead of marking a partial
graph as extracted. Qualified unsupported
relations may be skipped while the source claim remains. All candidate writes
and the completion key stay in one transaction. Legacy edges without link
verdicts remain labeled unverified; do not delete or silently promote them.

Qualification: bounded ordinary, reversed, negated, conditional, assistant-
suggestion and wrong-relation pairs under the actual local scorer, followed by
the same cases through a disposable writer/reader path. The initial seven-pair
probe admitted clear positive relations (0.9992 and 0.9993 entailment),
rejected a reversal (0.0012) and a suggestion (0.0004); two additional negative
surface forms scored 0.9885–0.9887 entailment. These are feasibility controls,
not a precision/recall estimate. Include awkward open-relation phrasing and
record withheld true edges in a representative replay before claiming a graph
recall or answer-quality gain; raw source recall must stay available.

The next twelve-pair local probe covered `works_at`, `role`, passive `manufactured_by`,
`uses`, `didnt_get`, a correction and an assistant suggestion. The awkward true
`Kael role fire mage` scored 0.9831 entailment (above the unchanged 0.95
threshold). A real-model correction-to-added proposal scored 0.0007; a
suggestion-to-decision scored 0.0019. This is still a bounded control, not an
estimate of graph recall on organic conversations. The disposable writer/reader
control uses known scores to isolate the transaction and source-preservation
contract; model-score and writer controls are reported separately.

### Graph rendering consumes source claims (2026-09-14)

Preserve the rich-note purpose: nodes still present useful source information and
navigate in both directions. Replace cached relationship assertions in retrieval
with lines built from the actual filtered edge set. A linked, available source
claim renders its attributed sentence/full evidence selected by the verifier;
legacy relations are explicitly marked unverified. Do not delete rich notes:
legacy descriptions/manual payloads remain stored and can appear as labeled
unverified notes only when unscoped, without exclusions and in current mode.
Derived project/code pointer payloads keep their existing behavior.

Tag enumeration must use the same filtered renderer, not bypass it through a
cached payload. Negative relations remain non-navigable, but are eligible factual
answers; the relevance reranker decides their usefulness. Track exact rendered
edge identities separately from navigation candidates. Scope/time filtering must
precede source rendering, and missing/private/edited linked source must not fall
back to an unqualified triple. Qualified note-summary generation remains later
within this phase; this change does not claim all legacy descriptions verified.

Cold claim evidence uses the stable episodic ID to resolve warm or archived
source, with warm state taking precedence (never bypass a new privacy/edit flag
using an older cold copy). Independent claim search includes cold sources for
unscoped/conversation/time queries. Until cold cluster membership is preserved,
cluster inclusion/exclusion queries omit cold candidates explicitly rather than
pretending missing membership is a pass. This remaining archival metadata repair
stays in the lifecycle work; no cold-source text is deleted to implement it.

Cold resurrection must preserve the archived vector, including a missing vector,
without replacing it with an embedding of a truncated generated summary. Missing
legacy vectors remain lexical-searchable and emit a warning; restoration itself
must succeed. This avoids changing the represented evidence and adding foreground
encoder work. Missing timestamp provenance restores as `unknown`, never
`original`. The existing original timestamp and probation behavior stay intact.

### Uniform conflict evidence boundary (2026-09-14)

Property values, negative assertions and named single-valued relations must use
the same source reconciliation boundary as ordinary relations. A new edge alone
cannot expire another or gain extra confidence from having displaced it. Enumerate
all same-relation differing targets/polarities and known opposition candidates;
do not make candidate detection depend on an English correction phrase or select
an arbitrary first edge. Preserve unrecognized relation names.

Automatic reconciliation requires current linked old evidence and exact new
source-claim evidence from the same conversation and authoritative role, with
known original source timestamps ordered old <= new. Feed complete old and new
role units plus their exact claim sentences and recorded timestamps to the
bounded reconciler, never a triple plus only the new raw turn. Unknown authorship,
missing source, edited offsets, different conversations, synthetic/unknown dates,
older imports and ambiguous multiple evidence units retain claims for review.
Same-time source evidence may coexist; it cannot authorize chronological expiry.
Do not infer event dates from the model: expiry continues to mean the time ICE
recorded the decision, with source times supplied separately. General event-time
extraction and cross-conversation identity authority remain separate work.

Positive and negative edges use one identity/observation writer. Property JSON
remains a display projection of all live positive values (single string or list),
not a last-write-wins authority. Explicit source-backed correction can still
retire a previous edge; failed/unknown verification preserves both source claims.
The semantic reconciler's answer quality must be qualified separately from SQL
mechanical controls; a stubbed decision is never reported as model validation.

Maintenance must share the same source eligibility and full-context prompt, and
recheck evidence before applying a delayed result. Manual reconciliation uses an
explicit keep-edge subset, like contradiction review, rather than interpreting
Approve as permission to expire the older row automatically. Repeated source-batch
questions are deduplicated; a later candidate rejection rolls back earlier
expiries from that candidate's comparison set. Property display projections and
both endpoint notes refresh after explicit retirement too.

### Turn summary support at the substitution boundary (2026-09-19)

Persist independent support verdicts for the summary and abstract, with canonical
role-attributed source text/hash and each candidate hash. Coverage remains a
retention metric; it cannot authorize a representation. All shared readers require
current supported evidence before summary/abstract substitution. Never inherit
summary verification for an abstract or a changed source. Legacy verdicts are
unknown; preserve generated text as metadata and retrieve original evidence.

Use complete authoritative role units as a quoted source for NLI; unknown roles
cannot authorize compression. Run verification asynchronously in post-flight,
not during foreground reads. The existing pinned verifier and fixed threshold
apply without fitting to new controls. Overlength/error/contradiction stays raw;
no source prefix or independent chunk entailment is accepted as full-context
verification. This first turn consumer does not complete long-source compression
or rolling/batch fold verification: those remain active repairs with distinct
source manifests, not permission to credit coverage as faithfulness.

### Prompt block accounting and bookmark evidence (2026-09-19)

Bookmarks use the same supported representation selector as other turn readers;
no unchecked summary and no500-word source prefix. Preserve complete selected
representations and let the actual prompt budget drop whole optional blocks.

Assembler reports structured token costs while constructing each block, not by
parsing user-controlled headings afterward. Optional slots/session-start/bookmark/
conversation-summary costs must not be buried in the essential system block.
Include actual message envelopes and the retrieval acknowledgement. A shared
bounded assembly loop remeasures after each eviction and returns final content,
ledger and removed-block names. Do not send a known-overflow prompt; preserve
current question/essential instruction/project constraints and return an explicit
context-length error if even those cannot fit. Unknown provider window remains
explicitly unmeasured. Apply the safety margin consistently when planning drops.

Fetch active project constraints separately on every request, including within
an existing sitting; session-start presentation excludes its duplicate constraint
copy. Project/session-start service keeps its existing complete default for MCP.
Telemetry and graph usage reflect final survivors, including bookmark/slot counts.
Validate real assembler+budget consumer with static-before-evidence eviction,
constraint preservation, exact counted block totals, tiny windows and unknown
windows; verify bookmarked source late corrections through the SQL reader.

A window no larger than the generation reserve has zero prompt room. Do not
silently halve the reserve in accounting while sending the unchanged generation
request. Such requests must be refused until the selected capacity/reserve fits.

### Background model identity and complete generation (2026-09-19)

The foreground `model_used` field is provenance, never a background model
selection override. Turn summarization and procedural extraction select through
`get_bg_model_name`, just like other background jobs. This preserves local
background execution when the foreground reader is cloud-hosted. Existing
explicit background configuration/fallback behavior stays visible; this does not
configure a cloud provider or claim every unpinned registry entry is local.

Turn, conversation and batch summaries, plus procedural extraction, accept only
nonempty text from a completion with finish_reason=stop. Length/content-filter/
missing-choice/missing-finish results cannot become finished summaries or recorded
coverage. A shared parser raises an explicit retryable failure; job boundaries
log it and retain original source. Post-flight may keep raw on summary failure,
but must propagate JobYielded to the runtime rather than marking yielded work
complete. Timeout uses the actual configured summary output budget.

### Rolling-summary source freshness (2026-09-20)

Add a versioned source snapshot to ConversationSummary, binding its exact output
hash to ordered source row IDs, source fingerprints and recorded timestamps.
Fingerprints cover every representation input (raw, supplied role boundaries,
summary/inject_raw fields), batch identity, privacy and timestamp provenance.
Compute fingerprints in SQL so foreground validation does not transfer raw
conversation content merely to hash it. This is cache freshness, NOT an NLI
verdict, semantic coverage, or proof that every source word reached generation.

Both active-conversation and cross-conversation readers check the snapshot.
Legacy/missing/edited/deleted sources or changed summary output cannot be injected.
Newer appended turns may retain the explicitly dated previous snapshot; an older
or same-time import invalidates it. Maintenance compares source identities, not
only max(timestamp). Rebuild from surviving source representations after a source
change/backfill; increment only for strictly newer additions. A failed generation
keeps the old checkpoint unchanged, though readers may withhold it if stale.
Cold-source parity remains open: a missing warm source is unknown, never guessed.
No legacy snapshot backfill that falsely certifies a previous generation.

Validation discovered that ORM-created disposable stores omitted the existing
NULLS NOT DISTINCT slot identity index owned by the C4/C9 migration. Mirror that
exact index in ORM metadata: tests must exercise the production uniqueness
contract, not allow duplicate slots and then misdiagnose ambiguous service reads.

### Evaluation scope correction — user, 2026-09-20

After repairs and targeted integration/configuration checks, run ONLY LME oracle
and semi-LSREP. Full LME-S is outside this campaign. Semi-LSREP retains full ICE
pre/post-flight reconstruction using recorded historical responses, followed by
separate cloud answer probes; future responses cannot enter earlier state.
Optimize supported answers per prompt token; compare answer quality at matched
context budgets and context cost at matched quality. A smaller prompt with worse
answers is not success. These two evaluations cannot establish full-history
distractor robustness or multi-user generalization; do not add extra campaigns
without a new user instruction.

**v3 answerer update, 2026-09-25:** new cloud answer probes use `gpt-6-luna`
through OpenCode Go Responses after its model listing and a minimal completion
both succeeded. Keep prior `gpt-5.6-luna` artifacts labeled as such; this is a
forward model choice, not a retroactive change to a measured run. The general
background model remains local and separately pinned.

### Fold input/output preservation — 2026-09-20

Use the shared source-supported representation selector for every turn entering
conversation summarization, rather than unchecked summary_text or raw prefixes.
Keep the complete selected representation, with its source-recorded timestamp.
Remove conversation_summary_per_turn_words: a hidden prefix cannot become a
summary of the complete turn. Pack whole representations before crossing the
existing chunk word target; an oversized single representation remains whole.
Before the real background request, count all messages plus output reserve and
margin against the configured ceiling, additionally clamped by the observed
shared-Ollama window. Reject an oversized call explicitly without advancing its
checkpoint. Splitting/compressing such long units remains the long-source repair;
never substitute a prefix to make the request fit.

Do not cut an otherwise complete generated summary at a word boundary. The word
setting is a generation target; actual completion tokens and final prompt budget
remain bounds. Bump the snapshot policy version so previously prefix-derived
checkpoints are rebuilt. These changes remove deterministic input/output loss;
they do not establish faithfulness of recursively generated conversation folds.

### Independent source notes — 2026-09-20

Replace recursive fold generation with independently generated source-group notes.
The existing JSON source manifest stores each group's source IDs, exact rendered
note and support verdict; no new table is needed for this cache boundary. A new
tail generates new groups only. Changed/deleted/backfilled sources rebuild from
originals. Never put an earlier generated note in a later generation prompt.
Use complete original role-attributed source units, with recorded timestamps,
as both generation input and NLI premise. Unknown role metadata remains explicitly
unknown, never inferred from textual speaker markers. Remove the hard verbatim
term demand and its coverage retry: preserving a word is not preserving its claim.

Only a current positive NLI verdict allows a generated note to replace its source
group. Otherwise retain the complete original group, warn with status/reason,
and mark the part as source evidence. A complete provider failure still leaves
the previous checkpoint untouched. Compose parts verbatim in source order, with
explicit segment boundaries and an instruction that later evidence may change
earlier statements. Bind parts as well as output to snapshot policy version3;
earlier recursive checkpoints must regenerate. Reuse never treats the cached
text as a new source of truth. No sentence-prefix or output-word truncation.

This removes error propagation and unsupported substitution, not the long-source
compression problem: >512-token NLI pairs remain unknown, and lossless fallbacks
can make the composed context expensive. Query-selectable source segments and a
bounded overview remain part of G73, alongside actual model qualification; do not
mark the whole fold repair complete or claim reduced prompt cost from this step.
The existing final prompt budget continues to count/evict the complete block.
Because this composition can exceed the encoder's window, its overview vector
must not silently represent only a prefix. Split only the embedding input into
complete contiguous spans bounded by that encoder's tokenizer/window; pool their
length-weighted vectors and normalize. The supplied evidence itself is unchanged.
This pooling prevents prefix loss, not topic dilution; independently ranked
source segments remain the retrieval design work above.
Cross-conversation overview reads also exclude any conversation containing a
private source turn, even if the conversation itself is not incognito. Whole
overview scope cannot selectively redact one note without reconstructing it;
retain own-conversation access, and test a valid-manifest public conversation
with a private turn so missing provenance cannot mask a privacy failure.

### Summary retrieval scope and source credit — 2026-09-20

Independent notes make an existing retrieval bypass more visible: the summary
leg receives the active conversation ID but not the resolved retrieval scope.
Pass both identities separately, appending optional parameters to preserve callers.
Apply the same conversation, cluster inclusion and exclusion predicates used by
ordinary episodic retrieval to every covered source before summary ranking/limit.
A whole summary is eligible only when all covered sources are eligible; never
partially disclose an aggregate containing excluded evidence. Batch summaries use
their coverage FK; independent-note roots use manifest source IDs. Empty explicit
conversation sets match nothing. Keep own conversation identity for self-exclusion,
not as a substitute for the requested search scope.

Cross-note fragments carry conversation_id and exact covered batch IDs, so source
credit and diversification can see them. Add batch identity to source snapshots;
old snapshots naturally fail comparison and rebuild. This is scope/provenance,
not evidence that a summary answers well. Partial-scope queryable independent
segments remain the follow-on design, rather than hiding the eligibility rule.

### Background NER model-token boundary — 2026-09-25

**v3 divergence:** A9b's background NuNER path already divides input into
`background_ner_chunk_words=250` whitespace-word pieces, but NuNER's GLiNER
processor splits punctuation/code further and has `config.max_len=384` model
words. A real 20-pair replay emitted `Sentence of length 622 has been truncated
to 384`: the model silently never sees that piece's tail. The roadmap's old
"no length cap" candidate description cannot describe this installed runtime.

Retain the configured 250-word coarse chunks, then split each through the
loaded model's own `data_processor.words_splitter` into windows no longer than
`config.max_len`, with a small overlap across boundaries. Build each subpiece
from the splitter's original character offsets and deduplicate the resulting
entity names as the path already does. If the model cannot report its splitter
or positive limit, warn and use the micro-NER fallback; never call the
background model on a piece it will truncate. Do not alter the preflight
micro-NER, label set, confidence threshold or extraction contract. Validate
complete token coverage and model-bound lengths on long punctuation-heavy
text, a tail entity through the actual background caller, and the ordinary
short-piece path; rerun the source-to-prompt trace to check that the warning
ceases. This is input preservation, not a proven answer-quality gain.

### Complete candidates and budget-time alternatives — 2026-09-21

Code re-grounding found an existing all-turn chunk store, so do not add a second
raw-segment index. The vector leg currently discards chunks whenever it fetched
the parent, before knowing whether that parent fits; `_rows_to_fragments` also
clips ordinary parents to500/1500 words. Keep full eligible representations and
all retrieved chunk alternatives until final packing. Existing query-selected
document excerpts remain bounded excerpts; legacy documents without chunks keep
raw evidence rather than a fabricated prefix representation.

Apply per-source and per-conversation caps to actual budget survivors. Preserve
existing settings and ablation switches by using their shared decision helpers
at admission. A complete raw source excludes redundant excerpts of the same
parent; excerpts selected first exclude a later whole-source copy. Supported
summaries are not complete raw sources and cannot claim to cover every excerpt.
Degrading or reranking to another representation clears the whole-source marker.
Skip oversized candidates and continue to later choices even when an entire
round has no admission; queues still advance and terminate.

**v3 divergence and resolution, 2026-09-25:** the sentence-claim search path
added after this packing rule emits Codex excerpts with only a source *batch*
origin, while episodic fragments identify the source *row*. The existing
same-row collapse therefore cannot see their overlap. In a 20-pair development
replay, 236 of 560 selected claim excerpts repeated exact text already present
in a selected episodic fragment, spending 14,600 claim tokens. Carry the source
row ID and exact rendered source excerpt on claim fragments. At admission,
remove a claim only when a selected episodic representation from that same row
literally contains its exact excerpt; if the episodic representation arrives
later, reclaim the duplicated claim tokens only when the episodic candidate
fits. A summary or unrelated row never implies coverage. Preserve the claim
when its source fragment is too large or its excerpt is absent. This is an
exact, evidence-preserving budget repair, not a relevance or answer-quality
claim; re-run the same trace and then compare supported answers.

Reranker capacity is per complete pair: an oversized pair receives no score and
must not disable scoring of all smaller candidates. Keep unscored representations
as explicit lower-priority fallback candidates, never treat them as scored or
truncate their contents. An all-unscored call retains original order and reports
no successful ranking. Existing true scorer/model errors preserve the entire
candidate list and warn. This enables useful raw excerpts without pretending
long-source NLI has been solved. Query-selectable conversation-note overviews and
batch-summary faithfulness remain separate open work.

The cold reader has the same bypass (`unchecked summary or raw`, then300-word
prefix). Route it through the shared representation selector too. Current archive
rows lack support/coverage metadata, so complete raw evidence wins; metadata parity
remains archive work. Retire the now-unused sentence-prefix helper rather than
leaving an alternative that can silently reappear in another reader.

The wide-net fallback must query the existing chunk leg under the same resolved
scope/time filters too; a broader parent search alone cannot recover a late
excerpt when that parent exceeds the prompt budget.

### Query-selectable conversation notes — 2026-09-23

The original-source note parts and output-bound manifest remain the authority.
Materialize a derived vector index for each part, with its source turn and batch
identities, evidence mode, recorded range and ordinal. The writer updates this
index in the same transaction as the aggregate; an unchanged valid aggregate
can index its existing parts without calling the generator again. The index is
never a source of truth: a reader requires the aggregate snapshot to be current
and the indexed part to match the manifest exactly. Deleted/changed sources
invalidate the aggregate before any indexed note can be used. Incognito and
source-scope rules still apply before cross-conversation scoring.

The active conversation selects whole relevant parts against the current prompt
embedding inside a fixed, configurable token allowance. It may omit an
oversized complete-source fallback, but must never truncate that fallback or
pretend it was verified; raw retrieval/excerpts remain separately available.
Pass progressively smaller whole-note alternatives to the final prompt budget:
when the complete selected block would overflow, remove the least relevant note
and reassemble before considering eviction of the remaining note(s).
The compact block keeps evidence mode, source range and behind-turn stamp.
Cross-conversation retrieval searches indexed parts, not only a pooled root
embedding, and credits only the batches represented by each returned part.
Selection must not treat a good score or repeated read as corroboration. In the
absence of an index (e.g. existing rows before backfill), readers may use the
current full aggregate until the normal maintenance pass indexes it; report
that fallback, and never omit existing evidence silently. Validate real SQL,
scope/privacy, archive parity and final prompt packing. A short overview and
long-source semantic compression remain distinct work; do not manufacture them
from partial source text.

### Unsupported source-group granularity — 2026-09-25

**v3 divergence from the intended query-selectable read:** a 20-pair replay
created two rolling notes, each a complete 7–13-turn source fallback after
the 512-token NLI verifier returned unknown. Neither fit the 650-token note
allowance, so the real prompt read zero rolling-note evidence. The original
sources were safe, but grouping made the whole layer inert.

Keep generation and full-source verification at the existing group boundary.
If a group is unsupported/unknown, materialize one **complete, attributed,
timestamped original turn** per source part, in original order, instead of one
multi-turn source part. This is a derived packing boundary, not a claim that
each turn's generated summary is supported. Apply the same rule to rolling
and batch writers. Do not cut a single oversized turn; its original remains
available to episodic/chunk retrieval. Keep exact source IDs and batch credit
on every part. Rebuild existing multi-turn **rolling** fallback manifests from
the originals on the next maintenance pass; do not re-label cached generated
text as evidence. Existing batch rows still read as whole aggregates, so
batch-part selection and regeneration of older batch rows remain separate
open work. Verify actual writer, source snapshot, indexed-note selection,
and prompt inclusion on a long replay. This should raise usable evidence at
bounded context, but does not prove answer gain or solve long-turn compression.

### Batch-summary source contract — 2026-09-21

Batch summaries must use the same original-source independent notes as rolling
summaries, never an unchecked concatenation of generated derivatives. Add a
nullable JSON source_manifest to BatchSummary with an additive migration. Bind
exact ordered source identities/fingerprints, composed output, note parts and
current support policy. Readers require a complete matching manifest; legacy,
edited, partially deleted, reassigned or private sources invalidate the aggregate.
This does not delete the source turns. The worker unlinks stale aggregates before
rebuilding eligible original sources; abandoned generated caches may be removed.

Reuse whole-source grouping, source attribution, current NLI support and full
source fallback from conversation notes; a compact unsupported note cannot replace
its originals. Keep batch output ceiling configurable. Check the complete provider
request plus output reserve; never slice evidence to fit. Bind a snapshot before
generation and verify it again before committing coverage, so concurrent edits
cannot validate an older generated result against a newer fingerprint. Preserve
JobYielded and avoid logging provider payloads. Test writer and actual SQL reader,
including positive/negative support, stale sources/output/policy, legacy manifests,
provider completion failure and migration roundtrip, in disposable stores only.

### Query-selectable batch parts — 2026-09-27

**Reader divergence:** rolling notes have a part-level vector index, but own
batch lookup still ranks and returns the complete aggregate. A fallback that
contains five intact originals can exceed the available prompt budget even
when its one relevant original fits. Keep the source/support contract and
materialize the same derived index for batch parts, without adding a model.

Add `BatchNote`, keyed by batch-summary ID and ordinal, with the same text,
evidence mode, recorded range, source IDs, batch IDs and1024-vector fields as
ConversationNote. The parent FK cascades derived rows. An additive migration
creates the table and cosine HNSW index. Reuse the shared part-index builder;
the source manifest remains the authority, never the index. New writes index
parts atomically with the aggregate and coverage stamps. A valid existing
aggregate may backfill/rebuild its index during maintenance without generation.

Own batch lookup ranks indexed parts directly, with a candidate bound matching
rolling-note retrieval (max64 or16× configured result limit), then verifies
the complete parent snapshot and exact part/index correspondence. Return at
most `retrieval_batch_summary_limit` whole parts, crediting only each selected
part's source batches. Keep original mode/range labels and creation time.
Apply source-scope/batch/cluster constraints to every source of each indexed
part before ranking, while still checking the full parent's source freshness;
an unrelated excluded part must not hide a visible part. A full-aggregate
fallback remains readable only when ALL its sources are allowed. Privacy and
time rules still apply before ranking. Missing/mismatched
indexes warn and retain a current complete aggregate as a compatibility read
until maintenance repairs the index; never serve edited index text. An invalid
parent stays unreadable. Never slice an oversized original to fit.

Validate writer→SQL ranking→budget packing with a late relevant source in an
oversized aggregate, source-credit precision and a subset batch/cluster scope,
warm/cold sources, index backfill
without generation, stale parent/index handling and migration roundtrip. Use
disposable databases. This closes batch packing granularity, not long-source
semantic compression, factual answer quality or the final benchmark campaign.

### Complete original notes share one prompt slot with episodic originals — 2026-09-28

**Observed through the production SQL readers:** a current batch part in
`source` mode covering exactly one turn and that turn's complete episodic raw
fragment were both admitted by the final token budget. The five-turn disposable
control spent151 tokens for a112-token attributed note plus its39-token
original; both described the same source. This is prompt duplication, not
independent corroboration.

Carry the exact source row ID on an indexed batch or conversation note only
when its manifest part is `source` mode with exactly one source ID. A supported
generated note, multi-source part and aggregate fallback have no such marker.
At final budget packing, a complete episodic fragment and such a note for the
same row share one slot. Prefer the attributed note when it fits, replacing an
already admitted complete episodic fragment; otherwise retain the episodic
original. If the note is admitted first, skip the later episodic duplicate.
An exact sentence claim from that same row can also be omitted only when its
literal excerpt appears in the admitted complete note. This extends the
existing claim/episodic containment rule without inferring semantic equality.
Do not collapse an excerpt, a supported compression, two source rows, or a
note whose parent/index validity failed. Check both candidate orders through
the real SQL readers and budget method; quantify token savings, not answer gain.

### Active note and retrieved original overlap — 2026-09-28

The current conversation's independently indexed rolling note enters the
system block after retrieval has already budgeted batch parts. A disposable
five-turn control with source-mode fallbacks selected the same original in
both places. Do not suppress a batch part at retrieval time: the final prompt
may shrink or evict the active note, and that part must then return.

Carry the exact complete-original note body alongside the row ID on a
manifest-checked single-source `source` part. During each
`assemble_budgeted_prompt` iteration, omit a retrieved part only when its
entire source-note body is a literal substring of the currently selected
active-conversation block. Recompute from the original candidate list after
every summary-option shrink or eviction. Return the final visible fragment
list from the budget result and use it for exposure/answer-proof accounting;
filtered candidates must not receive credit. An active generated compression
that merely shares names does not suppress original evidence. No parsing or
semantic similarity decision is needed. Validate actual rolling/batch writers,
SQL readers, large-window omission and tight-window restoration in a disposable
store, then the chat preparation path's visibility contract.

The existing v3 development source-to-answer trace must request the same
active-note option/source-ID pairs and report the final visible fragments as
chat preparation. Otherwise its full/no-Codex/vector arms can credit a batch
part omitted from the actual prompt, and any paired answer comparison measures
a different packing policy. Check the trace helper against the same disposable
writer/reader fixture before interpreting new trace numbers; historical trace
artifacts retain their original instrument version.

The same verified single-source `source` note may also duplicate a complete
episodic original from its own row. The note's source ID and current manifest
establish complete-source identity even though its attributed role rendering
differs from the episodic fragment's rendering, so literal body containment is
not required for this one pair. Share their prompt slot only when the episodic
fragment is marked `covers_entire_source`, has the same source row and active
conversation, and that source note is in the currently selected option.
Generated/compressed notes, partial excerpts, and different rows remain
independent. Reconsider the original after every note shrink or eviction and
credit only the final visible representation. Validate actual rolling writer
plus selected complete-original fragment, tight-budget restoration and an
unrelated-source guard; record token savings without inferring answer gain.

The trace must also call `production_parity.prepare` for every replayed turn
and probe arm. Calling the B2 prior's raw `retrieve` directly bypasses the
enabled v3 source-aware skip/rescue decision and can answer from a prompt that
chat would never send. Apply arm-specific retrieval omissions through the
orchestrator factory used by shared preparation, preserving the same source
decision and final budget/exposure path. Report base and final decisions
separately; compare answer arms only when their actual selected prompts are
recorded. Do not relabel older raw-B2 traces as current-v3 evidence.

### Archive collision preserves latest source — 2026-09-21

A warm/cold duplicate ID is a retry/recovery state, not permission to discard the
current live source. Archive selection locks source rows. Cold insertion must
update every transferred evidence field on conflict before deleting the live row,
in the same transaction. Never keep an older cold copy while deleting a newer
correction or privacy change. Validate an actual decay cycle with an older cold
copy and a corrected private live source, including raw hash/roles, timestamps,
vector and source identity. Wider cold metadata and aggregate parity remain open.

### Cold representation and identity round trip —2026-09-21

Preserve nullable typed cold columns for summary_coverage,
representation_verification, abstract_text, lossless_flag, inject_raw, session_id,
intent_tags, context_reliance and idempotency_key. Archive them atomically with the
source; cold representation reads use the shared support selector. Restore these
values exactly when present. Legacy missing metadata falls back to complete raw,
unknown evaluation and existing explicitly labeled restoration defaults; never
invent a verifier verdict or an original session. Keep the existing read-write
switch and probation behavior. An idempotency collision must leave the cold source
intact rather than delete evidence. Additive nullable migration only; no inferred
backfill. Test actual decay, cold selection and restoration, including all metadata,
legacy fallback, write-off behavior and collision retention. At this checkpoint
cluster/chunk links and aggregate manifest parity were separate archive work,
repaired later. The parent pointer was subsequently audited as dormant:
chat/import ingestion does not set it and retrieval does not traverse it
(state-copy portability can round-trip externally populated values). A
cold-parent migration alone is not a branching repair.

Retrieval failure telemetry must not stringify database/provider exceptions:
SQLAlchemy exceptions can contain full raw source/verdict parameters. Keep the
leg, exception class and SQLSTATE (when available); preserve conditional rollback
and warning on every failure. Validate with an exception carrying planted private
source text, not only with a benign ValueError.

### Cold cluster visibility —2026-09-21

Preserve nullable `cluster_ids` (all link memberships) and `cluster_id` (original
primary pointer) on cold rows. New archives record [] for known-unlinked sources;
NULL denotes legacy unknown. Snapshot links before deleting them and the live row
in the same transaction. Restore only references to still-existing clusters;
never recreate a deleted cluster or infer a primary pointer from link ordering.
Cold queries enforce positive cluster scope, excluded clusters and explicit batch
allow-lists before ranking/limit. Known-unlinked rows keep the live-path allowance;
unknown legacy membership is withheld when any cluster constraint applies. Plain
unconstrained temporal retrieval remains available. Tests cover positive, negative,
unknown and empty-batch scopes plus actual archive/restore membership preservation.

### Foreground provider selection —2026-09-21 clarification

Cloud answering is an optional selectable backend alongside local answering, not
a cloud migration. Reuse provider/routing abstractions where they already exist;
keep memory, embedding, NLI, reranking and background-model selection independent.
Cloud judge selection is an evaluation setting, not an implicit production switch.
Verify provider endpoint family, streaming, context budget and error contracts
before declaring a configured backend usable. No provider account or credential
is configured by this decision. Final evaluation remains oracle + semi-LSREP only.

### Archived excerpt continuity —2026-09-21

Preserve the existing retrieval-grade chunks when moving a turn to cold storage.
Add cold_chunks with original chunk ID, cold parent FK (cascade delete), index,
complete text and unchanged embedding. Atomically replace the cold chunk snapshot
before deleting live source rows. Restore all chunks in the same transaction as
the parent; an unexpected identity collision aborts restoration and retains cold
evidence. No new chunk generation, re-embedding or inferred backfill.

Cold lookup keeps the scoped parent candidate and offers up to3 stored excerpts
per eligible parent. Rank by stored cosine where query embedding is available;
otherwise use full prompt keywords against complete chunks, deterministic index
for ties. Only parents passing existing time/privacy/cluster/batch filters can
supply excerpts. Final packing chooses parent versus excerpts under the existing
coverage contract. This improves packing within the bounded parent pool; it does
not claim independent global cold-chunk recall or normal-mode archive search.

### Summary sources across storage tiers —2026-09-22

Use one SQL source projection over warm and cold rows for summary fingerprints,
visibility and rolling-note rebuild input. A live duplicate ID takes precedence
over its cold copy; each source appears once. Preserve cold batch_summary_id with
a nullable FK/ON DELETE SET NULL; transfer and restore it only while its summary
exists. The shared projection retains identical fingerprint fields so archival
alone does not invalidate a supported snapshot. New/changed/deleted sources and
policy/output changes still do. Legacy missing metadata is not inferred.

Summary reader source membership, source batch credit, privacy and positive/
negative cluster scope must use the same projection before ranking/limit. Known
unlinked differs from unknown cold membership. Rolling notes scan and rebuild
from both tiers' original sources; unchanged archives do not trigger generation.
At this 2026-09-22 checkpoint, batch generation still operated on eligible warm turns,
while existing source-current batch caches survived storage moves. The separate
all-cold generation spec below now closes that gap. Rolling notes can rebuild from
those originals. Test actual archive + own/cross readers, no unnecessary rewrite,
cold-source edit/delete/private changes, cold cluster exclusion and source credit.

The shared cluster inclusion/exclusion builders gain an optional trusted membership
column for warm/cold projections. Both direct cold lookup and aggregate readers
use those builders, preserving the one-predicate contract instead of adding a
second independently maintained scope implementation.

### All-cold batch-summary generation —2026-09-23

Batch writer eligibility must span the same warm/cold source projection used
by manifest validation. Preserve age-or-decay semantics: cold rows gain nullable
original `decay_score` and `is_document` metadata through an additive migration
and exact archive transfer. Legacy NULL means unknown. A cold row can enter a
new batch only when known non-document, non-private, non-lossless and uncovered;
an age-qualified legacy row with unknown document status remains raw/searchable
but is not guessed safe for compression. Existing valid batch caches remain
readable; known document sources invalidate them. Restoration preserves known
document status and its existing probation decay policy.

Combine eligible warm and cold originals by conversation and source time,
deduplicate a transient warm/cold collision in favor of the live row, and keep
the five-turn floor and complete-token grouping. Bracket generation with
source snapshots. Before writing coverage, lock both tier sets and require
their IDs, eligibility and snapshot fingerprints still match. Stamp each
source's `batch_summary_id` in the same transaction as the aggregate. A
concurrent archive/edit/delete must retry rather than validate old text against
new identity. Stale-cache repair clears coverage in both tiers. Validate
all-cold and mixed batches, race/eligibility changes, cache invalidation,
archive/restore parity and migration roundtrip in disposable PostgreSQL.

### Explicit context-pull preparation parity —2026-09-23

A caller-supplied conversation ID is the *current conversation identity* for
classifier history, own-summary/recency ranking and B2 pressure. It is not a
SQL scope by itself: resolve the conversation row through the chat scope
resolver, so `auto` can search shared non-private memory while `none`, manual
and project retain their limits. If the ID is unknown, keep a closed filter
instead of silently widening. Preserve caller-supplied non-identity scope
keys. `/search` supplies the current conversation ID through this same path.

Chat and explicit pulls share the warm-turn count and approximate history-token
query, including the same zero-history result when no identity is supplied.
Explicit pulls use those values in B2 and dynamic retrieval budgeting but
continue to retrieve even when the reported B2 decision says no. They return
structured fragments rather than an assembled answer prompt. Validate an
`auto` cross-conversation hit, incognito isolation, unknown-ID closure,
classifier context forwarding, nonzero B2/budget history and `/search` behavior.

### Project constraint scope —2026-09-23

`Decision.project_id` is required. `constraints_for_task` must require a
resolved project ID before matching file paths and filter decisions to that
project in SQL. A projectless explicit pull returns no project constraints;
neither a basename match nor the existence of only one registered project
authorizes selecting one. Project-attached conversations continue to surface
their own constraints first. Validate two projects with the same
`files_affected` path, plus a projectless pull; keep the explicit tool
description honest about needing a project-attached conversation ID or an
explicit project selection. `ice_context(project=...)` resolves a complete
closed project scope across all retrieval legs: non-incognito chat
conversations attached to that project and documents enabled in those chats.
Reject simultaneous `project` and `conversation_id` and blank selectors; an
empty project scope stays empty. Test project selection through the MCP adapter,
including a different project's same-path constraint and document visibility.


### Semantic source-need gate qualification —2026-09-27

**Final implementation decision, user clarification 2026-09-27:** stop the
open-ended model/prompt search and deploy the best complete version already
measured, rather than return to the earlier baseline. Enable the existing
E4B question-first paired full-source/cited-context proof and bounded rescue
by default. The matched actual-preparation development panel was22/24 versus
the original17/24; the earlier apparent24/24 quote-only arm was invalidated by
false suppression and must not be revived. This supersedes the earlier
all-starter-cases promotion requirement for this bounded refinement, not the
need to report failures. Two style misses, long-source limits, contextual
utility and answer-quality/latency qualification remain open. Keep the prior
on missing/uncertain proof; admit only a proven fitting original. No new local
model: use the existing E4B and pinned NLI; request-time NLI stays on CPU.
Native calls retain keep_alive=0. Verify the CPU change with saved native
responses, then test default activation through the actual handler. Do not
repeat model training or another candidate hunt before moving on.

The v3 G28 repair concerns whether an answer needs an older source, not a
personal-reference lexicon. Keep topic/intent classification, source-support NLI,
retrieval and reranking independent. Task-specific Laya prototypes can supply
a candidate memory-need prior, but two supervised runs still made confident
supplied-answer mistakes. Neither is active in production. The final bounded
visible-source adjudicator reuses the existing pinned local general background
model; do not change cloud answering or train on final inputs. Earlier
qualification notes below retain their original run status; the final
implementation decision above owns current activation and next work.

A candidate must distinguish supplied evidence, general knowledge, missing
older personal evidence and unknown. A supplied-evidence verdict must return an
exact quote from the latest prompt or one of the actual recent messages, and
answer the requested attribution/value/time rather than a different current
value. Unknown, incomplete JSON, unsupported quote or model failure retains the
existing decision and reports why. An exact quote is a structural prerequisite,
not proof that it answers the question. Qualify against matched missing-source,
current-versus-old and generic controls before trusting the decision.

Before production integration, assemble the no-retrieval prompt with the actual
answerer's window allowance, standing slots, bookmarks, source notes, constraints
and final prompt-ceiling eviction. Adjudicate that exact prepared evidence and
the latest prompt. If older evidence is required, run the existing retrieval and
reassemble under the same ceiling. Do not substitute the classifier's shorter
prefix or the pre-eviction recent allowance for what the answerer will see.
Persistent blocks are not retrieval results and must remain in both arms. The eval-only starter now
uses authoritative separate sittings and checks the real recent reader excludes
the two old answers; previous five-turn results measured the classifier view
only. Model input must be complete within its verified bound; overlength is
unknown, never a silent source cut. No fabricated probability from an enum or
unqualified confidence should become a calibrated B2 prior.

Development supervision preserves exact named original source blocks, parent
dialogue IDs and recorded time. Source-literal answers alone do not validate
question qualifiers; teacher-approved dates, tense and preferences still need
inspection. Split full source conversations before use. Public-source tags do
not prove origin: reused-v1 rows without a traceable conversation must not be
credited as fresh human supervision. LoCoMo adaptations here are noncommercial
research development, not a LoCoMo benchmark or deployable general artifact.

Promotion requires corrected same-path invariance/semantic controls, independent
general and source-location negatives, full-call bounded cost, and source-backed
answer benefit without extra spurious prompt spending on a v3 replay. Inspect the
decisions that actually change. A failed model candidate leaves the defect open;
revise data, source handling or the gate design rather than mark it fixed. The
final broad tuning/combined Z1/Z2 and only LME oracle/semi-LSREP runs remain later.

**Conservative integration revision, 2026-09-27:** a source-plan autorater fixes
the starter and gives a supported invented-command answer, but a fresh public
qualification exposes missing-material errors: partial earlier equations/lists
are mistaken for the whole requested source. Do not give this judge authority
to replace all B2 decisions. Preserve every existing positive unless a supported
verbatim fact in the current user prompt itself supplies the request. A
general-knowledge verdict does not suppress a B2 positive; supplied recent
excerpts are not enough to prove an entire earlier model/list is present.
For this positive-prior exception, judge the current question as a standalone
source after the full baseline prompt has been assembled and fit-checked. This
asks the stronger question "does the current message alone supply its requested
fact?" and avoids irrelevant earlier instructions changing a simple supplied
value into a generic-method classification. The answerer still receives every
standing/recent block in its fitted baseline. A non-sourced verdict retains the
positive; this view is never used to decide that a negative needs no memory.
Negative rescue and post-search qualification always inspect the full fitted
prepared evidence, not the classifier prefix or a current-only substitute.

A missing-source verdict may initiate a provisional rescue of a B2 negative.
Fetch through the existing scope, budgets, fusion and reranker, assemble within
the same answer ceiling, and judge the actual candidate prompt. Admit a rescue
only if the judge identifies an exact supporting quote in a surviving retrieved
fragment. A related topic, quote from the original question/recent window, or
unknown/incomplete verdict cannot admit it. This checks evidence after search
rather than pretending the judge knows what is in a store it has never read.
Unknown/error preserves the B2 arm, with a warning and reason. Existing positives
keep their existing retrieval path and do not undergo this rescue rejection.

Provisional reads must defer episodic retention and cold restoration until final
admission/eviction. Candidate fetches are not exposures to the answerer. Graph
access already belongs after final assembly; do not promote any fact's truth
from a read. Preserve the ordinary orchestrator caller's existing behavior.
Qualification includes opposing unrelated source controls and actual rescued
answers, current-supplied token savings, prior-positive recall preservation,
and complete bounded input through the chosen real provider path. The full
G28 invariance sweep remains open; this narrower repair must not claim to solve
all classifier labels or all context-sufficiency decisions.

**Native-provider divergence, 2026-09-27:** the four-choice native judge
passes historical/general starter forms but calls explicitly supplied facts
`general_knowledge`, even in a current-only view. That overlapping category
cannot qualify suppression. Use a separate binary current-source proof task:
`visible_evidence` with an exact complete supporting quote, or `not_supplied`.
Public knowledge, new generation, partial sources and missing earlier values
are ordinary `not_supplied` outcomes; they retain the prior and do not emit an
outage warning. Malformed/incomplete responses still warn and preserve it.
Three unrelated source-location examples distinguish complete current facts,
missing old values and incomplete reports. The negative/rescue task retains
its four choices and full prepared evidence. Qualify the actual native caller,
not an earlier compatible-endpoint prototype; keep the switch OFF until then.

**Provisional qualification invalidated, 2026-09-27:** the initial50 current
source controls were insufficient. Before committing activation, stronger
same-speaker other-event controls exposed7/22 false suppressions, including a
question quoted as evidence. Keep both source gate and rescue OFF; current-only
23/24 and full-rescue24/24 are development arms, not repaired production.
Post-search quote-only qualification also admitted2/22 unrelated blocks. A quote
alone never authorizes admission or suppression. The repair continues below.

### Classifier context consumes source-supported representations —2026-09-27

Before this repair, `_get_context_turns` preferred `summary_text` directly,
bypassing the shared current support/coverage verdict used by answer retrieval.
Use `choose_representation` before rendering the prefix: a supported shorter
summary may remain preferred; unsupported, stale or unknown summaries fall
back to the original raw turn. Preserve the existing three-turn/global-word
budgets and raw-turn cap, shared template and trained checkpoint. No new
generation/verifier call in classification. Log context-read failure by error
class before retaining standalone classification, never SQL/source payload.
Validate the actual SQL reader and classify→render→encoder path with controlled
head/encoder and supported/unsupported/stale/unknown verdicts; then a real
loaded classifier input capture. This repairs input authority, not a claim of
improved label accuracy or final answer quality.


**Answer-claim development refinement, 2026-09-27:** test source support and
private-memory intent as separate targets. Source proof needs a minimal factual
answer claim retaining every requested qualifier, an exact source quote and
independent support from the COMPLETE attributed source; do not verify against
the question or a clipped excerpt. Avoid extraneous narrative/chronology that
can make a valid short answer unsupported. A relative date must still clear the
unchanged source verifier; unknown retains the prior, never lowers its threshold.
A vague recent report does not ground a precise onset month. Correct that gold
label explicitly and preserve the old artifact. A supplied fact can support a
generic question but does not prove that the task requires private memory.
Keep semantic development reports separate from independent qualification;
no new live Laya head or source-gate default yet.


**Question coverage is separate, 2026-09-27:** a supported claim can answer a
weaker question after dropping a requested date/event. Prototype a source-blind
coverage decision over the actual question and proposed claim; it assumes the
claim true and checks all requested subjects, events, relationships, time/status
and parts. It must not invent extra qualifiers or consult source/world facts.
Unknown/incomplete coverage cannot authorize suppression/rescue. NLI separately
checks source truth at its unchanged bound/threshold. Compare the pinned generic
Laya typed choices with the existing local model on actual candidate outputs and
matched qualifier controls. Measure final admission change, not merely emitted
labels. No trained-head, dependency or default activation before qualification.
Keep reasoning-mode source adjudication separate from global background
non-thinking generation policy. A broad QA question permitting another valid
sourced answer is ambiguous gold; preserve its artifact and exclude it from
clean opposing scoring rather than redefine truth around the benchmark answer.


**Question-first hypothesis refinement, 2026-09-27:** separate coverage scoring
still confuses recent music with future collaboration and can invent a required
person's name. Before reading any source, form a declarative frame from the
complete actual question with exactly one answer placeholder, preserving every
requested event/time/relationship/subject/status and part without adding facts.
Then read the COMPLETE attributed source to fill only that slot, with exact
source evidence. Never let source prose rewrite the frame into a weaker true
statement. Verify the filled hypothesis against the original source, under the
unchanged verifier bound/threshold. Unframeable generation/whole-material tasks,
missing answer/quote, changed frame, uncertainty or capacity failure retain B2.
This is a bounded factual-proof candidate; private-memory intent remains a
separate decision. Test opposing sources where dropping one qualifier would
make a true weak claim, then native final preparation before promotion. A source
block with an unresolved antecedent does not ground a more specific question
merely because the dataset answer is a literal span; mark uncertain rather than
rewrite source history or retrain on misqualified labels.


**Retain the answer category in the frame, 2026-09-27:** the first frame prototype
preserved date/event but replaced the whole interrogative noun phrase: "what
digestive issue" became "Sam experienced {{answer}} lately", admitting phone
frustration. Preserve the requested attribute/category OUTSIDE the placeholder
(e.g. "The digestive issue Sam experienced lately was {{answer}}"), with all
other requested qualifiers. Fill only the most-specific explicit value of that
category. Do not confuse a vague health scare with the named condition even
when an admission mask passes. Check returned value quality as well as admission;
no hardcoded medical/port vocabulary or threshold changes.


**Qualifier robustness within NLI, 2026-09-27:** typed relative-clause frames
can let a verifier overlook the date. The same unsupported onset proposition
scored.998 in "The hobby ... in October was photography" and.063 in the direct
"Dave picked up photography in October" formulation. Question-only generation
therefore proposes two equivalent hypotheses: an explicit typed-attribute form
and a direct event assertion containing the same answer category and ALL
requested qualifiers as asserted predicates. Freeze both before source reading,
fill one common source value, and require BOTH complete-source entailment scores
at the unchanged.95 threshold. Batch the two pairs; no extra generative call.
Validate semantic equivalence/qualifier retention as well as final source
admission and answer value. This is a bounded conservative candidate, not a
claim that two correlated NLI formulations are an independent truth oracle.
Unknown still preserves B2 and the original evidence; do not alter the global
summary/graph verifier solely from this QA-specific diagnostic.

**Disabled shared-path integration, 2026-09-27:** expose the qualified-development
mechanism in `src/api/source_proof.py`, retaining BOTH feature defaults OFF.
Current-message suppression uses the full latest request as both question and
original source. Post-search negative rescue may use only final, un-evicted,
complete ORIGINAL episodic turns (`covers_entire_source=True`); summaries,
excerpts, graph assertions and procedural prose cannot prove themselves. This
is a deliberately partial candidate scope, not full cross-leg rescue. Native
frame/fill calls own reasoning mode and 2048 output tokens, shared full-input
8192-token/32768-context bounds, exact source binding and paired unchanged NLI.
Verifier uncertainty/capacity failure returns unknown. The previous quote-only
current binary prompt and post-search quote check are superseded, not parallel
admission paths. Model/transport failure retains B2 and earns no new exposure.
Controlled tests must reach the real preparation and chat route; native tests
must include the joined latest request, not just detached question/source pairs.
No default activation from the 13 reused developmental controls alone.

**Current-message presupposition contamination, 2026-09-27:** the actual combined
latest request supplies the NLI premise with the QUESTION as well as evidence;
paired full-message entailment alone falsely establishes a date/body-part only
mentioned in that question. Require paired support from BOTH complete source
and an unchanged contiguous evidential unit selected from it. The unit retains
original speaker/date labels and necessary antecedents; a bare value span loses
attribution and rejects otherwise supported answers. It must establish the
whole filled proposition from assertions, never from a question's presupposed
conditions. Freeze grammatical frames with the unknown value only in its slot;
do not copy a supplied answer into a background condition or duplicate the
category after that slot. Null/unknown/capacity failure still preserves B2.
This citation-locality candidate must be tested for lost positives as well as
false admissions; the cached short-quote guard rejected both unsupported date/
body-part claims but also three true named-speaker claims, so short quotes are
NOT a qualified default. No punctuation/closed intent lexicon substitute.

**Preserve the citation's original leading context, 2026-09-27:** a correct
citation retained the named speaker but omitted the source's leading recorded
header, making a dated sharing claim unverifiable. Do not manufacture metadata
or relax NLI. The second premise is the unchanged original prefix through the
END of the exact citation's first occurrence (`source[:start+len(quote)]`),
retaining leading original speaker/date/antecedent context while excluding
later material, including the appended question. The FULL original is still
independently scored under its complete-input bound; this is a cited context
alternative, not clipping an overlength full input to pass NLI. If the citation
comes from the question itself this rule supplies no new authority; paired
support/negative qualification still apply. Reuse retained native outputs to
measure this isolated context change before any new generative probe. Check
both unsupported qualifiers and dated/speaker-attributed positives. The short
quote-only guard is superseded, not a third condition that would retain its
known false rejections. This remains a disabled candidate until independent
negative, same-path intent/style, latency and answer-benefit qualification.

**One frozen question per preparation, 2026-09-27:** negative-rescue candidates
share the SAME immutable `SourceQuestion` created once lazily after an eligible
original survives eviction. Reuse those two forms for each complete source;
never regenerate a subtly different question interpretation for each fragment.
This removes repeated question-model calls (two sources:3 rather than4 native
calls) without caching private requests globally or changing source/NLI policy.
Bind the object to its exact original question; a mismatch is unknown. Current
suppression still makes one frame and one fill call. Unframeable/failed framing
cannot authorize the factual rescue branch. This addresses avoidable cost and
source-dependent interpretation; it does not qualify task intent or claim that
a single factual slot can recover whole omitted material. Contextual-source
utility for those tasks remains separate open G28 repair work.

**Factual rescue packs its proven original, 2026-09-27:** one complete source
proving the requested factual answer cannot authorize injecting every other
candidate. Reassemble with only that original, recheck actual fit/eviction,
and admit/credit it only if it survives as the same complete source. Standing
slots/bookmarks/constraints remain. This changes factual negative rescue only;
normal prior-positive retrieval keeps its existing candidate/packing behaviour.
Test a rejected unrelated original followed by a supported original through
final preparation: one frozen question, two source fills, only the supported
source in the final prompt/exposure. This is a supported-context-cost repair,
not broader contextual-task utility or a default activation.
