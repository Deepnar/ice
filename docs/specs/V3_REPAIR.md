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
