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
