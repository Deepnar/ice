# ICE v3 — repairs after the first complete reseed, tuning, and Z2

Assumes decided specs: `V3_REPAIR.md` (existing whole-system repair authority
and source preservation), `RESEED_PLAN.md` (r4 identities and chronological
replay), `G63_extractor_decision.md` (NuExtract3 extraction),
`BG_LAYER_FIXES.md` (separate E4B background generation), `T_temporal.md`
(source time and lifecycle semantics). The dated decisions here supersede the
older execution order and experiment roster in `Z1_tuning_coverage.md` and
`FINAL_experiments.md` for this v3 phase; their historical measurements remain.

**Status: implementation underway, 2026-10-09.** P0 measurements and P1 bounded
verification are implemented with focused validation. Two confirmed P2 source
selection defects and P3 extraction/note mechanisms are repaired. Independent
quality review remains pending; procedural-output sizing and negative
receipts are implemented with focused validation. P4 author-assertion admission
and context preservation and source-predicate recovery are implemented;
calibrated confidence/identity utility and independent precision remain open.
P5 ordered-source authority and fair bounded maintenance scans are implemented
with focused controls; referenced-event versus source-record filtering is a
repaired exclusion mechanism with focused warm/cold/final-input controls, not
calendar precision or independent temporal answer accuracy. P6 source-semantic
procedures, source/observation clocks and retirement receipts have focused
qualification. P7 workplace breadth is repaired with opposing/style and actual
old-source rescue controls; broad gate/answer gains remain unqualified. P8's
blinded reference packet, independent absolute cache and preference-only runner
are implemented with controlled SDK/resume checks; human labels and rubric
qualification remain pending.
P9 completion diagnostics and the bounded deployed-component/resource profile
are implemented/measured; worst-case interactive/long-input capacity and throughput
remain unqualified. P10 entity accounts, cluster source/label contracts, graph
checkpoints and archival/probation recall have focused qualification. A bounded
shared-preparation/all-ten-periodic/cold-read pass preserves all42 originals;
many semantic branches are inert on that fixture. Broad maintenance utility and
end-answer/context gains remain unqualified. §2.1t records exact coverage.
ROADMAP is the queue. The numbered packages below decompose existing entries,
not a new G-track or claims of eleven newly discovered defects. The completed
r4 evidence is in [PROVENANCE](../PROVENANCE.md#2026-10-07--v3-completed-r4-development-campaign).
This is the next plan referenced by the retained, private `docs/SESSION.md`.

## 1. Decisions

### 1.1 Scope, sequence and completion

The maintainer requests evidence-led core memory repair, including bounded
design improvements where the likely cause is not yet settled. Execute:

1. Correct the measurement interpretation and freeze r4 as the development
   baseline; reuse its sources, outputs and 39 as-of stores.
2. Repair long-source compression, answer-bearing retrieval, failed writes,
   graph identity/update decisions, procedural usefulness and memory gates.
   Qualify the judge and review semantic outputs alongside these repairs.
3. Run the bounded hyperparameter protocol in §2.3, then freeze configuration,
   code, model/endpoint identities, labels and experiment manifests.
4. Run **Z2: two-history semi-LSREP with full memory lifecycle and
   additive/subtractive ablations, plus LongMemEval oracle**. “LME-S oracle”
   here means the oracle evidence-session condition, not a full S haystack run.
5. Diagnose final failures without silently tuning the same final results.
   A further repair creates a new revision; preserve and report the first run.

Quality means supported answers at a given total prompt budget, or fewer tokens
at preserved quality. More nodes, denser graphs, equal leg shares, fewer logged
errors, or 100% benchmark accuracy are not substitute objectives. Do not keep
changing a final benchmark until it becomes perfect. Repair an identified
mechanism now; do not wait for Z2 to rediscover r4's confirmed defects.

The initial planning/closeout pass changed no runtime. The subsequent repairs
are within the existing authorized v3 repair scope. No new model is required by
this plan. Keep the local/cloud switch; local E4B and NuExtract3 write memory,
cloud GPT-6-Luna answers, Muse Spark 1.3 Contributor Responses remains the
selected judge subject to qualification. No silent judge/provider substitution.
Keep all originals and source-linked sentences; a failed verifier is neither
deletion authority nor proof the source fact is false.

Product packaging, MCP/REST expansion, deletion edge cases, unrelated routing
features, broad model bake-offs and the frozen v2 paper are outside this repair
pass. Session instructions explicitly retain SESSION until the maintainer asks
to clear it. Publication authorization covers this closeout/merge/push only.

### 1.2 Evidence ledger and repair packages

**Confirmed** means the recorded output and traced contract identify the
mechanism. **Hypothesis** means a defect/weak outcome is observed but its remedy
needs a bounded comparison. **Unexercised** means r4 cannot establish behavior.
Each package ends with a decision and production-path validation; no endless
candidate search. Ground-check code and prior experiments before implementing.

| Package | Evidence and diagnosis | Decided next action; acceptance |
|---|---|---|
| P0 — trustworthy measurements | Confirmed: `report_v3_replay.py` checks whether `summary_support` is absent, not nested summary/abstract statuses. It reports zero unknowns despite 986 unknown summaries. A source-ID hit also survived after the required answer content disappeared. | Add status/reason/denominator counts and semantic evidence stages. Keep raw reports immutable; write corrected derived reports with input hashes. Test absent, empty, unknown, contradicted and supported independently. A deliberately missing answer span must fail semantic support while still passing source-ID recall. |
| P1 — useful long-source summaries and folds | Confirmed: all 986 nonempty summaries hit `SupportInputError`; the whole attributed turn plus summary must fit 512 NLI tokens. Almost every conversation/batch note therefore renders originals. This is input-contract failure, not 986 NLI contradiction decisions. | Implement source-linked units and bounded claim verification (§2.1), then reuse the same contract for turn summaries, abstracts and rolling/batch notes. Preserve complete-source coverage manifests, exact-value alternatives and later corrections. Verify actual substitutions and quality/token changes, not the mere existence of a summary. |
| P2 — answer-bearing source selection | Confirmed: original evidence can exist and its parent ID rank first while every supplied excerpt omits the required value. Other stable misses have no candidate or lose candidates before final assembly. Exact root differs by case. | Trace each stable miss to index/eligibility, candidate generation, rerank, chunk selection, representation, budget or answer use. Rank relevant child windows with original offsets; offer adjacent/parent evidence when a fragment lacks its referent or requested code block. Reuse the current embedder/reranker, deduplicate overlap, and repack under the same total budget. No gold text or expected answer may guide runtime selection. |
| P3 — failed or incomplete local writes | Confirmed: 58 terminal extraction-output failures, 3 terminal post-flight incomplete completions, 20 incomplete conversation folds; 14 admitted occurrences have a degraded gold source. Exact remaining parser/output causes require the saved response. | Bucket the saved receipts, reproduce each distinct contract failure, and repair template/chunk/output handling at its existing seam. Typed unsupported proposals remain quarantined; truncated structured output remains incomplete. Size requests using actual complete input plus output reserve. Validate repaired failed sources and matched successful controls, then replay affected chronological suffixes only in new development stores. |
| P4 — graph identity, relation precision and confidence | Observed: 2,409 components, 82.2% degree-one nodes, alias-like duplicates, 80% of edges at confidence 0.35 below the 0.5 direct-read floor. All edges have supported internal links; independent precision is unknown. Sparse topology alone is not a defect. | Review role/direction/polarity/modality and every confidence tier, including excluded facts. Repair source-backed alias resolution and unsafe relation canonicalization where review confirms errors. Preserve separate entities when identity is ambiguous. Treat confidence as extraction evidence, access as exposure. Decide threshold changes from calibrated source truth and answer effects; never connect nodes or lower trust just to improve counts. |
| P5 — temporal updates and maintenance reachability | Confirmed coverage restriction: 508 distinct reconciliation items touched across 1,000 inspections all have synthetic-time sources, so the original-timestamp guard prevents a decision call. The priority scanner spends its entire cap on that detector. This does not measure the model's reconciliation accuracy. | Record authority/eligibility/defer reasons per detector. Add verified within-conversation source order distinct from calendar authority; permit an explicitly supported later correction to compare ordered sources without inventing a date. Never infer cross-conversation order from import order. Prevent ineligible backlog from consuming the whole decision budget; fairly schedule eligible core detectors. Test genuine dates, ordinal-only corrections, coexistence, reported/hypothetical claims and historical reads. |
| P6 — procedural memory that contributes | Observed: 95 patterns, 2 active, zero selected procedural fragments on 117 answered occurrences. The activation/relevance/packing bottleneck is not yet isolated. | Trace cited distinct support → activation → eligible candidates → selected text → answer use on actual procedural questions. Repair matching/eligibility only where source-backed repeated behavior supports it; distinguish explicit instructions from inferred habits. Never reactivate the former reflection shortcut or count repeated reads as supporting turns. Keep inactive material stored; require relevant-answer gain or context savings to change activation policy. |
| P7 — gate, entity activation and style | Coverage gap: all 259 probes retrieve; a memory-question panel cannot measure unnecessary retrieval. Historical gating skipped some requests, but source-matched false-positive/false-negative labels are absent. | Reuse the existing source-need/style fixtures plus reviewed r4 misses. Hold meaning fixed across terse, rambling, lowercase, typo and punctuation variants; include current-supplied facts, banter and historical requests. Trace MicroNER/preflight activation and rescue to final evidence and answers. Keep the best complete existing source-proof mechanism unless a bounded replacement improves both misses and false suppression. |
| P8 — judge reliability and question intent | Confirmed: the same saved full answer receives different final absolute grades across contrasts on 36/111 fully graded cases. Some general questions are judged against an unstated historical-answer requirement. This is not proof that all disputed model grades are wrong. | Review disputed keys and saved answers; grade each unique answer independently once, cache by answer+question+sources+rubric+judge identity, then compare preferences separately. Qualify on correct/partial/incorrect/uncertain real examples and opposing controls (§2.2). Preserve old grades and label revisions; do not regenerate good answers just to repair the judge. |
| P9 — completion and resource costs | Confirmed: 12/468 answer records ended `incomplete`; receipts do not preserve a precise provider reason. Whole-stack latency/VRAM and real-time concurrency were not measured. Output-cap causality is a hypothesis. | Preserve completion details, reasoning/output usage and transport versus content failure. Reproduce a small failed-input set at the existing profile before changing token reserves. Profile actual local-job stages and resident/peak memory; use bounded batching and existing ownership/scheduling, not unrestricted local concurrency or more resident models. Retain per-request durable attempts and byte-stable completed artifacts. |
| P10 — semantic maintenance outputs and untested lifecycle | Observed: 21 clusters, 1,750 enriched entity descriptions, 109 session summaries, 8 batch summaries; these are output counts, not quality. No cold rows or cold retrieval occurred. | Review actual member/source-backed names, enriched descriptions and notes for attribution/completeness. Apply the P1 evidence contract to generated memory that reaches readers. Run a short controlled time-advance/restore path to exercise decay, compaction, warm→cold→read and all scheduled core jobs before Z2. Do not claim the synthetic dates establish natural calendar retention. |

P0–P10 are **11 execution packages**, not eleven roadmap checkmarks. Fixes may
share one mechanism (for example P1 supports P10). Complete the core repairs
before tuning; judge review can proceed alongside local repair. For a hypothesis,
use the retained failure and an opposing control, implement one reasoned
candidate, and allow at most one evidence-led revision before deciding keep,
replace, or retain the baseline with a precise limitation. A newly demonstrated
integrity defect overrides that search limit and must be repaired. Focused
experiments are allowed; repeated broad reseeds are not the default tool.

### 1.3 Removal and small decision-model policy

No component is authorized for deletion merely because r4 selected zero outputs.
Procedural memory may be starved; cold memory had no eligible population;
reconciliation was blocked before the model; summaries failed length admission.
First remove the cause, then measure utility. Defaults may be simplified only
after matched-budget omission preserves reviewed support/answers and reduces
cost across the relevant task slice. Retain the ablation switch and original
data. Structural deletion requires the existing explicit deletion decision and
consumer/doc sweep; this plan grants no blanket deletion permission.

Laya-like open decision models remain a bounded option for entity identity,
relation equivalence, reconciliation or candidate utility **after** eligible
source-backed tasks exist. Their prior source-need and generic-relation tests
do not qualify these jobs. Compare against deterministic resolution and the
existing E4B on the same opposing-source controls. A candidate must improve
the relevant decision/answer measure without new false merge/expiry controls,
and fit the measured whole-stack memory/latency budget. No extra permanently
resident model, hosted JEV dependency, or classifier/NLI replacement is implied.

## 2. Algorithm and data model

**Implementation grounding, 2026-10-08:** P0's saved trace has nested verdicts
and candidate texts. The corrected report schema counts missing candidates and
missing/invalid verdicts separately, binds trace/reporter hashes, and refuses to
overwrite an existing report. An identical current report can resume read-only
after full recomputation; old identities use a new derived path. Existing generated/ranked/budgeted receipts lack text;
their old answer-content coverage is unobserved. New child traces record text
at those stages. Exact-value survival is a separate diagnostic from semantic
support, which requires reviewed context/qualifiers rather than word presence.

### 2.1 Long-source support and evidence selection

The bounded implementation uses the existing NLI tokenizer to enumerate windows
within each authoritative role unit, retaining original character offsets and
overlap. Window text targets half the pair capacity minus 32 tokens, leaving the remainder for
the complete assertion and provenance; actual pair length is still checked by
the scorer. Generation receives numbered original windows and returns JSON
assertions with cited window IDs. The rendered summary is composed only from
those complete assertions, never from an independent uncited prose field.
Multiple cited windows are checked together as one premise or remain unknown
if the complete pair cannot fit. Every assertion must pass; no OR aggregation.
Summary and abstract proofs are independent. The verdict binds the complete
canonical source, candidate, tokenizer/model revision, window policy and every
cited premise/assertion. Whole-source legacy verification remains available for
short outputs lacking citation structure, with a visible warning for a missing
structured response. It cannot silently approve a long output by truncation.
An invalid optional abstract citation is independently warned and withheld;
it does not discard an otherwise valid cited summary. The model is asked for
the smallest sufficient citation set rather than all repeated background windows.
Citation metadata is excluded from assertion text, windows are not counted as
turns, and attribution does not invent an action or frequency from formatting.
The real rolling and batch clients put this instruction in the system role and
the complete original windows in the user role, matching the turn generator.
Capacity checks count those actual messages rather than an unsplit placeholder.
Adding the new support-method policy cannot invalidate an unchanged cache made
entirely of complete original-source parts. Its raw/source/visibility snapshot
and composition still must match; supported generated parts require the full
current policy. This preserves existing source evidence while changing derived
approval rules.

1. Enumerate **all** original role units with immutable source ID/hash and
   offsets. Build token-bounded windows with overlap and exact offsets using
   the deployed tokenizer, reserving space for hypothesis, role and provenance.
   Keep paragraph/context relationships; punctuation alone cannot decide truth.
2. Request source-linked summary assertions from the existing local background
   model. Every assertion includes its proposed source spans and qualifiers.
   Resolve spans to exact original text; unresolved spans are unknown. No
   generated assertion can become its own verification premise.
3. Check each complete assertion against its complete cited window(s), within
   the verifier's real bound. Retain speaker, event, time, modality and negation.
   If required context spans too many windows, keep the source alternative;
   do not truncate or OR together unrelated entailment scores. Check all
   assertions in a composed summary; unsupported additions fail that summary.
4. Store a versioned verification manifest: source hashes/offsets, assertion
   text/hash, verifier revision, method revision, actual token counts, status,
   reason, candidate hash and coverage over original units. Changes to policy
   invalidate old derived approval without deleting originals. Extend existing
   `representation_verification`/note manifests before adding duplicate stores.
5. **Faithfulness and completeness are separate.** An entailed short note may
   omit the needed answer. Keep original exact strings and query-selectable
   child windows; term coverage remains only a retention signal. Code/value,
   update and synthesis controls must verify required information survives
   final assembly. Do not turn term lists into a truth metric again.
6. Teach `choose_representation` and existing note readers to use only current
   manifests. Search/query-pack original excerpts when the compact view cannot
   cover the question. Count supported summary substitutions, source fallbacks,
   missing facts, prompt tokens and inference cost by input length.

This adapts the granularity insight in
[SummaC](https://aclanthology.org/2022.tacl-1.10/): sentence-sized NLI inputs
with document-level aggregation. That paper does not qualify ICE's model or
our authority/completeness checks. The previously tested ModernBERT and BGE
long-input candidates remain unpromoted; their failed controls are in MODELS.
Do not redo that model search before repairing how inputs are constructed.

P2 evidence receipts add exact rendered source spans, available answer-support
units for **evaluation only**, and distinct stages: indexed/eligible/generated/
ranked/budgeted/rendered/answer-supported. Evaluate required claims and literals
against reviewed alternatives and original context; substring checks suffice
only for exact-value questions. A semantic decision may be uncertain and must
not silently become recall=1. Gold annotations never enter query expansion,
candidate generation, reranking, or the answering prompt.

**P2 production-path finding, 2026-10-08:** recent history splits a turn's
allowance in half even when one speaker uses only a few tokens. This clips a
late user refusal despite the complete pair fitting the unchanged per-turn
allowance. Redistribute the unused role share to the other role before any
truncation; retain the same turn/window caps and explicit roles. If both roles
are oversized, the existing bounded split remains until source-window packing
is qualified. This repairs demonstrated answer-content loss without a larger
prompt budget or a new model.

**P2 document-selection finding, 2026-10-08:** the lexical SQL selects source
text/verification but omits `is_document`. Its converter consequently bypasses
the existing query-selected document chunks. The retained r4 exact-value source
has 2,020 words; its existing keyword chunk selection would include the required
value. Supply the actual flag to the converter, then check the original-source
reproduction and the final same-budget prompt. This activates the existing
mechanism; no new reranker or guessed document classification is added.

### 2.1a Local output failures: grounded repair, 2026-10-08

The saved receipts classify58 terminal extraction failures as43 provider output
length,9 invalid JSON,3 missing source sentence and3 invalid field failures.
All20 terminal conversation-fold failures and3 other terminal post-flight
completion failures also report length. Receipts do not contain model responses;
this prevents reconstructing the precise nullable/malformed fields after the fact.
Use a bounded retained-source reproduction with private response capture before
claiming the remaining parser mechanisms. Never publish those source/response
bytes or infer that every old length failure has the same generation behavior.

Extraction request sizing currently reserves the generalist prose prompt even
in specialist template mode, omits actual confirmed-entity overhead, and floors
usable content at550 tokens even when less space exists. Re-ground sizing on the
actual model-specific messages plus complete output reserve and the existing
count safety margin. Measure/log request capacity and returned finish/usage
metadata without source text. A request that cannot fit must not be sent or
silently truncated; split input at the existing source chunker if possible.

For a provider-marked output-length failure, discard the entire returned prefix
and retry that source chunk once as smaller source-aware chunks (at most the
existing550-token base, or half the failing input if smaller). Healthy complete
calls retain their adaptive input. Reuse the same model,3000-token output ceiling,
NER, template, quote resolution and relation post-filters. Recovered child calls
cannot themselves branch indefinitely: a second incomplete child remains a typed
failure. Every child must complete before the turn may write graph facts or its
completion key. Malformed complete JSON, polarity and missing quote failures
retain the strict contract; input subdivision is not permission to salvage them.
Yield checks remain between calls. Complete empty arrays remain valid success.

**Divergence found by the retained-source reproduction:** input subdivision
removed length truncation but a child returned stop with an unescaped JSON
control character. Inspection of the installed `/api/show` template and NuMind's
[NuExtract3 GGUF model card](https://huggingface.co/numind/NuExtract3-GGUF)
shows that the shipped legacy `<|input|>`/empty-string template is inserted into
the default document-content task, not the trained structured task. The current
NuExtract3 contract uses `string`/`verbatim-string` leaves plus structured-task
and template/instruction/document delimiters. Existing historical percentages
do not qualify this corrected input format. Record this as a model-contract
repair, not a new model promotion or permission to relax JSON parsing.

Use typed leaves: subject/relation/object=`string`, source_sentence=
`verbatim-string`. Put confirmed entities in the separate instruction block;
the source document contains only the original chunk. For shared Ollama, route
this explicitly selected NuExtract3 template request through native raw generate
using the verified text-only structured template (thinking disabled), avoiding
a second chat-template wrapper. For dedicated serving, send the official
`chat_template_kwargs`. Keep ordinary background/generic extraction transports
unchanged. Reuse the existing named-model lock/ownership and native timing helper;
set real context/output caps explicitly and preserve provider finish/usage.
All strict parsing, quote/polarity/NLI gates remain. Native request markers are
internal transport metadata and must not be sent to generic providers. No
fallback to the known-wrong legacy template. Verify rendered bytes against the
installed template and check actual retained failure plus matched controls.

Conversation-note JSON now needs citation/field overhead that the historical
hardcoded400-token cap did not reserve. Make its output limit an explicit
`conversation_summary_max_tokens=900`, matching the existing turn-summary cap;
the250-word content target remains and actual complete input+reserve must fit.
This is a bounded sizing choice, not a claim that900 prevents all truncation.

A returned provider-length note must not commit its generated prefix or force
the entire fold to be retried unchanged. Preserve the complete original source
group as a source-mode note with an explicit `incomplete_generation` receipt
(finish/usage/limit when known), warn every occurrence, and keep that reason
when original parts are made individually selectable. Source coverage may
advance because complete originals were retained, never because a generated
summary was approved. Unconfirmed provider calls, invalid capacity, transport
outages and cooperative yields still propagate. Record source-fallback and
incomplete-group counters separately. Check actual note writer/index/readers
on a retained failing source group and an opposing prefix-success control.
Do not mark whole P3 complete from the extraction seam alone.

### 2.1b Procedural completion sizing and negative receipts — 2026-10-08

The remaining three terminal post-flight completion failures occur in the
procedural caller: corresponding r4 logs report E4B output truncation at the
hardcoded160-token limit. Make this explicit as
`procedural_extraction_max_tokens=512`; keep the existing one-pattern/cited-user-
message contract and source/session/activation policies. Check actual complete
messages plus output reserve against the background serving capacity before a
call. Preserve finish/usage metadata through the shared typed completion error;
truncated patterns cannot stamp completion or reinforce memory. Qualify a
retained failed-session source block and an opposing complete control. A complete
provider response still requires valid independent source citations; completion
success alone is not semantic pattern qualification (P6 owns that).

Grounding also finds complete `NONE` returns without a session-bucket completion
key. This violates the existing stepwise-call design and repeatedly calls an
unchanged negative result. Only an exact complete `NONE` may stamp that negative
bucket result, with an observable no-pattern event. Repeated reads of the same
bucket make no call; a new bucket remains eligible. A malformed/unsupported/
truncated result is not empty success and must not acquire that key. Keep current
session-step grouping; do not count a negative receipt as supporting a pattern.

### 2.1c Graph assertion authority — grounded decision, 2026-10-08

The retained graph copy has no ambiguous stored aliases or normalization keys;
its complete row fingerprints match the producer's physical column projection.
Do not add another formatting resolver or connect components from topology.
Source review instead finds questions and proposed fictional scenarios written
as plain graph facts, including relationships between merely co-mentioned
references. Existing paragraph-to-proposition NLI does not reliably distinguish
an author's commitment. Speaker-framed NLI increased false admission; a flat
nonreasoning E4B decision also failed. Neither candidate is promoted.

Use the existing general background model in shared local mode for a bounded,
reasoning-enabled **author-assertion check**, in addition to existing NLI. A
six-case retained preview rejected three question/hypothesis errors but also
rejected a valid conjunct. The grounded clarification distinguishes proposition
support from completeness: each explicit conjunct can support its own relation,
and a fact inside a message containing a question remains an assertion. The same
candidate passed31 synthetic positive/opposing/style controls in one call; this
is mechanism qualification, not independent corpus precision. Check the retained
cases again through the implemented function and the actual writer before
claiming the mechanism repaired.

Current complete-original implementation passes30/31 controls without false
admission; one valid modal assertion remains unknown and keeps its source. The
actual E4B writer/readers admit3 asserted relations and withhold3 questions/
proposals while preserving6 source claims and repeat-call idempotency. Do not
repeat broad model searches merely to make that lone development control pass;
retain this conservative limitation for independent precision/answer review.

Pass complete authoritative role-unit text, original role and exact source
quote identity, with the verbalized relation and polarity. Batch at most12
NLI-supported proposals. Each model case contains its complete original role
unit and proposition; quotation hashes remain in receipts, not an isolated
quote presented as an authoritative alternative. Production-path validation
of the shared-source/isolated-quote envelope admitted a proposed declarative
sentence even after a prompt clarification. Re-ground on the complete-original
case format used by the successful standalone controls, and record its larger
input cost. Do not truncate
source units or use a question-word lexicon. Reserve4096 output tokens inside
32768 context, with an8192 estimated-input cap and conservative byte/capacity
guards. Shared native Ollama must report complete stop and actual input counts;
unknown completion, malformed/missing/duplicate IDs, overlength or provider
failure produces explicit unknown, never approval. Cooperative yields propagate.
Dedicated mode is explicitly unqualified/unknown for this native reasoning seam;
there is no silent provider or model substitution. No extra resident model.

Only NLI support **and** an affirmative author-assertion verdict whose governing
act is assertion or source-qualified report admit a new edge. A returned true
boolean cannot override a question/hypothetical classification: the retained
full-original test returned that internally inconsistent combination once.
Record both outputs and withhold the relation. Qualified proposed scenarios
remain searchable source claims, not established graph assertions. Preserve
asserted belief, modality, negation, past tense and reported
attribution when they are actually in the proposition. Questions can carry
explicit premises; proposed outcomes are not realized facts. Store versioned
source/candidate/quote/role/model/prompt/policy hashes and decisions in existing
claim/link JSON receipts, including withheld decisions. Confidence tiers and
identity resolution remain unchanged until separately calibrated.

A literal subclause is not safe compression merely because NLI accepts it.
Claim readers cannot shorten with the old substring proof. The author checker
uses the complete role unit, which can affirm a proposition in a different
paragraph from the selected quote; its global approval does not authorize
rendering only that quoted paragraph. Both direct claim search and linked graph
rendering therefore resolve the complete authoritative role unit from the
unchanged original, including positive and legacy receipts. Hash/offset or
unavailable-original cases retain the stored paragraph and warn when required.
Compact cited context needs a separately qualified proof before substitution. This
preserves cross-paragraph proposal framing without deleting a true source when
a classifier abstains. Old graph links are not relabeled or replayed, and
original r4 data is immutable. All new writer/settings identities require a
new development store. Observe calls, unknown rates and costs before tuning.

Validation: strict parser/capacity/yield controls; semantic positive, negative,
modal, reported, historical, mixed-question and conjunction fixtures; invariant
typo/punctuation variants; full writer with all NLI pairs deliberately positive
must still withhold unsupported author assertions and retain searchable complete
context. Check actual local model on retained cases and generic opposing
controls. Do not turn these development results into graph precision or answer
accuracy. P4's confidence/identity utility and broader source review remain open.

### 2.1d Source-verified relation canonicalization — decision, 2026-10-08

The retained source preview includes a past relation rewritten in present tense;
the new author gate correctly withholds that proposition. Existing code discards
the extractor's original relation before verification, so a safe reject also
loses a valid graph assertion. Preserve the mechanically normalized positive
relation immediately after existing negation separation and before vocabulary
matching. This is a source candidate, not automatic acceptance or a new synonym.

When canonical and source relations differ, enumerate both at the existing
paragraph NLI/complete-role author-check seams. Prefer canonical only when both
support gates admit its exact proposition. Otherwise admit the source relation
only if its own independent complete-source verdicts pass. Neither may pass by
borrowing the other's verdict. If both fail, retain the original claim without
an edge. No changed cosine/confidence threshold, inverse guard, world ontology
or extra model. Claims-off retains its explicitly unverified legacy behavior.

Keep distinct emitted relations and supporting quotations through within-turn
deduplication, even when vocabulary matching points them at one canonical word.
The writer chooses at most one relation per emitted proposal; same-source batch
observation still cannot count twice as corroboration. Store proposed/source/
selected relation and selection reason with the chosen link's exact proposition
proof. Counts distinguish attempted verification variants from admitted original
proposals. Existing author receipts count claim-proposition decisions, not unique
turns or model calls. Changed writer uses new child state; r4 remains untouched.

Actual native captures distinguish two residual caller failures: malformed
decision shape/IDs and rejecting a valid paraphrase solely for capitalization.
Use a native JSON schema with required typed decision fields, fixed admitted
ID enum and exact batch length; the strict parser still rejects duplicates and
truncated responses. Clarify semantic equivalence: ordinary name capitalization,
spacing and same-meaning verbs do not change support, while polarity, tense,
modality, direction and meaningful identifier case must remain. Record this as
author_assertion_v2, not a relaxed parser or vocabulary-based intent rule. Apply
style/opposing controls to this candidate too; schema completion is not truth.

Verify past→present and qualified→unqualified opposing cases, canonical-safe
synonyms, both-rejected proposals and same-batch idempotency. The actual writer
must recover a supported raw relation with injected positive NLI and real E4B,
and graph/claim readers must retain original evidence. Record source-qualified
graph recovery separately from independent precision or downstream answer gain.

### 2.1e Ordered-source reconciliation — grounded decision, 2026-10-08

Current `_reconciliation_evidence` requires original timestamps on every claim
and exactly one distinct old source unit. Synthetic but ordered histories cannot
reach the decider; a fact corroborated in two different turns also fails the
one-unit test. Keep the same-author/same-conversation boundary and full originals,
but select the latest comparable old observation rather than treating independent
corroboration as missing authority.

Use a hash-bound `source_spans.sequence` receipt, not synthesized calendar dates:
method=`ordered_conversation_v1`, sequence_sha256=SHA256 of the frozen ordered
source conversation, position=positive integer, raw_sha256=current complete turn.
The existing importer `_store_turn` accepts an optional explicitly supplied
receipt. The reseed supplies it from its verified canonical ordered conversation;
generic unknown-order imports do not infer authority from arrival/mtime/UUID.
Live source timestamps retain their existing authority. No new resident model,
schema table or conversion of an ordinal to a date. Existing raw/source offsets
and archive copies retain the JSON receipt; source edits invalidate it.

Compare original calendars when all relevant observations have verified original
dates. Otherwise compare only a shared verified sequence within one conversation
and role. Calendar/sequence disagreement is deferred. A later source must be
strictly later than the latest comparable old source. Missing, incomparable,
mixed-author or stale evidence preserves both assertions. Full source text still
decides explicit replacement versus coexistence/report/hypothesis; order alone
cannot expire an edge. Both inline and maintenance paths share this authority.

The existing validity interval describes when a fact is established in memory;
expiry remains a recorded operation time, not an inferred real-world event date.
Ordinal decisions journal sequence IDs/positions and ordering basis beside that
operation. Prompts/readers say calendar unknown and show conversation order.
Historical calendar precision is not established by this ordinal mechanism.
Source-record time versus referenced event time is a separate query-axis risk:
reproduce a later retrospective statement about an earlier event outside recent
history before changing time filters; do not claim their existing behavior is
correct from the code alone.

Maintenance scanning must reserve bounded opportunity for every core detector
rather than letting the first backlog fill the entire scan cap. Allocate a fair
per-detector quota from the existing total scan budget, preserving priority only
for remainder slots. Small budgets rotate the starting detector in runtime state;
no unbounded scan/refill, model-call-cap increase or silent detector failure.
Expose per-detector scanned/returned/deferred source-authority outcomes. Eligibility
and semantic decision remain separate measurements; a skipped candidate is not
a model rejection. Qualify blocked-backlog versus eligible later detector controls
through the actual maintenance job as well as detector isolation.

**2026-10-09 model-contract grounding:** the job's decision-counting wrapper
accepted only positional arguments, so `_ask_enum`'s `schema=` failed before
reaching the real JSON client and was retried without the schema. Forward the
schema through the wrapper to the real decider; adapt signature-limited test
callbacks without counting a rejected keyword as model inference. Verify the
actual request format through the job, not just the standalone JSON helper.
Require a complete provider finish before parsing a maintenance verdict; a
valid-looking JSON prefix at output length is not expiry authority.

Validation: original-only dates, ordinal-only explicit correction, same-source
and reverse order, incomparable namespace/conversation/role, source edits, two
old corroborating observations, coexistence and reported/hypothetical changes.
Exercise actual `_store_turn` receipt persistence, inline and periodic consumers,
event journaling and source retrieval. No working-store/r4 mutation; new child
state only. Primary reference: [Graphiti's temporal model](https://www.getzep.com/blog/beyond-static-knowledge-graphs/)
separates episode reference dates and optional fact validity. This motivates
separating source order/calendar authority; it does not qualify ICE's protocol.
[Mem0's current ADD-only integration discussion](https://mem0.ai/library/integrations/how-to-test-ai-agent-memory-with-mem0-a-practical-memory-simulation-guide)
also preserves history and interprets updates at read time; do not copy blanket
latest-wins into unrelated facts or unverified dates.

### 2.1f Query-date axis — grounded decision, 2026-10-09

The actual44-turn control in §2.1e confirms that parsed query dates are being
applied to a different field: source recording and memory-establishment times.
An event-year question loses a2020 statement about2010 while admitting a
non-answering2010 statement. The golden source is outside40 recent turns;
unwindowed lexical retrieval finds it. Do not mistake this for a ranking failure
or infer event dates from ingestion timestamps.

Decision: carry `axis=source_record|event|unknown` beside the resolved calendar
window. The existing deterministic expression resolver knows dates, not their
semantic role: its detected scopes therefore use `unknown`, without a new
intent lexicon or resident model. Existing explicitly constructed scopes and
legacy serialized scopes retain documented `source_record` behavior. A trusted
caller can explicitly choose an axis; the parser does not pretend to do so.

Only `source_record` authorizes hard source/knowledge-clock windows and date
recency. `event`/`unknown` preserve the requested dates as question conditions,
source time labels and telemetry, but search meaning across recording dates
under the same candidate and final prompt caps. Historical warm/cold evidence
remains reachable; unknown-axis cold search retains its bounded candidate cap.
Graph navigation remains on live edges, with original claims/history available
separately; unknown event time does not authorize traversing expired edges as
current facts. Timeline crop and procedural-observation filters must not borrow
an unresolved event date. Empty-window language is legal only for a real source
window. No event-date extraction, global newest-wins policy, new model call or
larger retrieval/prompt budget is introduced.

Qualify date-axis round-trip/legacy compatibility; unknown axis versus explicit
source_record on retrospective, source-history and wrong-year sources; form
variants; cold/source-claim/date-label preservation and final preparation.
Use opposing sources and check actual answer-bearing text, not just row presence.
This repairs evidence exclusion, not independent temporal answer accuracy.
Broader relevance/latency/corpus quality remains for the existing package gates.

### 2.1g Procedural candidate opportunity — grounded decision, 2026-10-09

Read-only r4 receipts refine P6's bottleneck:236 of259 contexts generate927
procedural candidates; only3 survive into pre-budget ranking on2 unlabeled
general probes, none on117 answered occurrences. Final95-pattern state has
14 citation-qualified rows,13 inactive with confidence.8/reinforcement1–2;
89 patterns have one source session. These counts cannot qualify semantic
habits or justify a lower activation threshold. Eligibility, lifecycle, candidate
access and answer usefulness remain distinct checks.

The reranker currently takes only the first64 RRF candidates. A66-fragment
actual pinned-model control ranks a valid procedural workflow above same-topic
noise when shown independently (6.5234 vs2.3906), yet that workflow never reaches
the model in the capped pool. This confirms a candidate-access failure, not
evidence to increase all leg weights, force prompt admission or add a model.

Decision: under the existing candidate cap, nominate the highest pre-rank
candidate of each produced source family, then fill remaining slots in original
fusion order. When the cap is smaller than the family count, original order
chooses the nominees. All nominated evidence still competes on actual relevance
and the same final prompt budget; no equal final shares or floor change. Preserve
strict complete-input scoring, alternate representation identity, model-outage
fallback and source provenance. Log available/scored-pool/omitted family counts
and the selection method. No extra setting, model or resident weights.

Qualify isolation plus normal production retrieval/budgeting with relevant and
irrelevant procedural candidates beyond the prefix cap, an actual pinned-model
comparison, same candidate/token limits and family counts. This does not resolve
semantic pattern support, cosine-only pattern merging, stale activation or
SQL eligibility after LIMIT; inspect those existing P6 seams next. Do not close
P6 or broad roadmap owners from this candidate opportunity mechanism alone.

### 2.1h Procedural semantic identity — grounded decision, 2026-10-09

An actual disposable-store/current-writer control with the real shared encoder
confirms that opposite workflow order is merged: tests-before-changes versus
changes-before-tests scores0.9861, passes the0.85 rule, increments the old habit
to two sessions and assigns opposite sources to it. The opposite description
is lost. This is semantic identity corruption, not a threshold-tuning question.
The existing NLI control supports an equivalent rewording in both directions,
withholds the opposite, and withholds a narrower Rust-only condition.

Decision: cosine generates a bounded five-candidate shortlist in the same
project scope; it cannot authorize merging. Exact identical descriptions are
equivalent by identity, with no model call. Different descriptions must pass
both complete-direction NLI checks under the current pinned verifier/policy.
Unknown, contradiction, bounds/provider errors or one-way implication keep
separate patterns. Choose the highest-ranked verified equivalent candidate;
an opposing nearest neighbor must not hide a valid next neighbor. No cosine
cutoff reduction, model addition or activation/confidence rule change.
An overlong candidate is refused before scoring so it cannot hide other valid
bounded candidates; no source/description truncation is authorized.

The bounded shortlist becomes `procedural_match_candidates` (default5, range
1–32), an explicit cost cap. The old similarity knob remains a documented
historical pin until its removal/tuning-ledger update; it no longer grants write
authority. Journal exact description/policy identities, direction statuses and
candidate outcomes in the retained procedural proof metadata/logs. Same-session
re-description still cannot reinforce or refresh observation time. Missing
prior source identities still cannot strengthen. Unchanged-bucket receipts
remain idempotent. Equivalence of descriptions does not prove their underlying
habit is true; source-semantic validation and explicit/inferred distinction
remain the next P6 work.

Qualify actual SQL/writer and new/session-repeat behavior on same/opposite/
narrower descriptions, valid non-nearest equivalence, pinned real NLI and
encoder, private source/hash receipts and both positive/negative controls.
Never borrow a candidate's support receipt for a different description.

Implemented and qualified on four synthetic actual-encoder/NLI/SQL writer
controls: paraphrase reinforces, opposite and narrower conditions remain
separate, and a verified second neighbor wins over the opposing first. Growing
or rereading the same sitting neither reinforces nor refreshes time. These are
mechanism checks with controlled extraction proposals, not independent habit
precision or corpus answer gains. Source-semantic admission remains next.

### 2.1i Procedural eligibility before the cap — grounded decision, 2026-10-09

Five actual SQL/retrieve/final-budget controls with candidate limit1 return
nothing despite a valid lower-ranked procedure. A higher-scoring foreign-project,
unallowed-source, excluded-source or topic/intent-mismatched pattern occupies
the SQL slot, then is removed in Python. This is a candidate-generation loss,
not a weak ranker or a reason to raise the cap.

Decision: apply existing project/source/deny/trigger eligibility in SQL before
similarity ordering and LIMIT. Keep the same candidate, trust, date and final
prompt bounds. Same-project conventions retain their batch-scope exemption;
all-excluded support still fails, partial surviving support remains eligible.
Empty sources keep their existing unscoped behavior. Empty/missing trigger
labels impose no restriction; reserved proof metadata is not a trigger label.
Malformed trigger-label shapes are ineligible rather than widening eligibility.
Retain Python admission checks as defense and unchanged fragment provenance.
Order equal similarity by ID for reproducibility. No extra model or setting.

Qualify both the direct SQL seam and normal retrieve/rerank/final-budget path,
including empty allow sets, project exemption, partial excluded support, topic/
intent controls and confidence/active/date guards. Do not credit merely generated
IDs or these synthetic fixtures as independent habit/answer-quality evidence.

Implemented: the five actual SQL/rerank/budget controls now supply the valid
procedure with exactly one scored candidate at limit1. Additional SQL controls
preserve empty allow sets, project exemption, partial source exclusions,
active/confidence and source-clock behavior. No candidate/prompt cap increases.

### 2.1j Procedural cited-message semantics — grounded decision, 2026-10-09

The old writer maps citation numbers, not semantic evidence. Prior direct NLI
gave a true imperative/request paraphrase only0.0027 entailment, so a plain
NLI threshold would discard legitimate requests. A bounded existing-E4B typed
source check passes20 authored positive/opposing/style controls: direct requests,
declared preferences, typos and qualified conditions survive; mere topics,
assistant quotes, imagined behavior, reversed order, another actor and invented
universal frequency do not. This is candidate qualification, not corpus precision.

Decision: review complete cited authoritative user messages against the exact
pattern before insertion/reinforcement, using existing E4B native typed reasoning.
Reuse its owned-client scheduling and residency; no new model or loose parser.
Per-source decisions distinguish explicit_instruction, observed_request,
self_report and unsupported. At least two distinct positively supported turns
are required. Only supported IDs contribute to citation counts and activation;
invalidly attached topic sources cannot turn two true citations into ten.
Explicit instructions render separately from observed behavior in the reader.

Fixed safety limits: batches12, complete estimated input8192 tokens/16384 bytes,
context32768, output2048. Split oversized groups, never truncate a source; a
single oversized original becomes unknown. Require actual complete native stop,
positive measured input count and confirmed input+reserve within capacity.
Malformed IDs/types, truncation, provider/bounds errors remain unknown; runtime
yield propagates. The exact source/pattern/model/prompt/policy hashes and per-
source statuses/kinds/usage accompany each admitted observation. Legacy rows
with no current source proof remain stored/unqualified and cannot reinforce a
new pattern. No automatic legacy migration, deletion or activation-rule change.

Persist proposal review separately from facts using existing StoreMeta keys
`procedural_review:<bucket job key>`, alongside normal completion keys. A complete
unsupported proposal records a negative completion without a habit; verifier
unknown records a diagnostic receipt and retains retry behavior, never false
completion. Admitted patterns carry source-proof observations in reserved
trigger metadata. Merged paraphrases retain their independent source proof plus
the mutual-entailment link to the stored description. Revalidate exact sources
through warm/cold storage before strengthening; reads are never evidence.
HTTP/connection failures preserve the existing recovery's transport type/status;
they must not be reclassified as responsive model/content uncertainty.

Qualification: source checks in isolation, then actual writer/SQL/reader with
real E4B and pinned equivalence scorer; two true citations plus eight irrelevant
cites must not activate, two repeated requests must survive, opposites/conditions
must remain distinct, and stale/legacy proofs must not manufacture corroboration.
Preview a fixed retained source sample without treating model labels as an
independent gold key. Source receipts, not turn length or well-formed JSON,
determine admission. End-answer/context gains remain for tuning/final evaluation.

Implemented and exercised through actual E4B source checks/SQL/writer/readers.
Ten cited messages with only two genuine requests produce two support IDs and
an inactive row; all-topic proposal produces no row and an idempotent negative
receipt. Model uncertainty has a diagnostic receipt and no completion key.
Four actual source/encoder/NLI/writer controls preserve paraphrases/opposites/
conditions/non-nearest matches; a third independent sitting activates the
source-qualified procedure and its typed wording reaches the real reader.
Replay observes current receipt counters, reporting latest outcome per unique
bucket rather than treating repeated receipt reads as corroboration. A fixed
retained preview selects two shortest complete cited originals for each of
four patterns: two have model support on both selected sources, two do not.
This selection does not grade every citation, establish the latter patterns
false, or provide independent corpus precision. No model/activation-rule change.

### 2.1k Procedural observation clocks and retirement — decision, 2026-10-09

The current reader has typed source labels but omits source dates and the
pattern's ICE observation interval. The writer sets `first_observed` and
`last_observed` from its operation clock; these are not when the user performed
the behavior. Retirement uses that same last-observation clock: active patterns
older than180 days retire only below three independent-sitting reinforcements.
Ten qualifying citations can activate a single-sitting pattern, but do not
grant the recurrence exemption. The r4 count of high-citation inactive rows
does not justify lowering activation or changing retirement.

Keep these thresholds and storage intact. Render compact source-record ranges
from authoritative aware original timestamps, mark other source calendars
unknown, and render verified ordinal ranges separately within each conversation.
Never compare ordinal positions across conversations or infer an event date.
Label first/last observation explicitly as ICE observation time. Presentation
uses only the source IDs eligible for that read; verification still checks the
complete supporting originals. Prefixes count toward the existing token budget.
Existing explicit source-record windows still select the stored observation
interval; that legacy knowledge-clock interpretation is not event-date recall.

Return a bounded receipt from the existing retirement worker with active rows
examined, stale rows protected by recurrence and actual rows retired, cutoff,
clock basis and policy. Replay's existing worker-result field carries it; the
derived report counts observed calls separately from older uninstrumented calls.
No source text, per-row identifiers or hard deletion is added. Validate the
boundary, same-sitting versus three-sitting exemption, idempotent second cycle,
stored-original preservation and retrieval under the unchanged budget, including
cold/synthetic-time originals. Synthetic controls establish these mechanisms,
not natural retention utility or end-answer gains; those remain for P10/Z2.

**Grounded integration failure:** the focused source-clock regression returned
127 tokens under a100-token cap:92 tokens of procedure plus an unbudgeted
post-selection temporal note. Fix that existing seam before proceeding. The
note states that no episodic records were selected, not that no memory exists.
Append only a whole note that fits remaining capacity; optional nearby-era
details may be omitted as a whole. Never evict useful admitted evidence for an
absence note or clip a source. Log note omission and its reason. Validate empty,
full and partial capacity through the real reader; keep source credit separate.

Implemented and qualified:48 focused controls,813 combined smoke/background
checks,66 standalone temporal regressions, plus actual SQL/pinned-reranker/final
budget controls. The production registry→retirement worker→replay/report path
reports one retirement then zero on the second cycle; all five patterns stay
stored. Cold/synthetic source clocks remain distinct, excluded source dates
are not presented, and source hashes/observation clocks are unchanged by reads.
P6's retirement disposition is keep the current rule pending answer/lifecycle
utility; do not lower activation or claim natural retention from these controls.

### 2.1l Factual question breadth — grounded decision, 2026-10-09

The captured actual request `Where did Petra work in2010?` became city-only
frames. Its true source names an employer and no city; fill returns null before
any NLI call. The source is present in the actual final prompt. This is invented
answer-category narrowing, not a retrieval miss or a measured NLI rejection.

Repair the existing question-only framing instructions: infer category from the
requested relationship, keep ambiguous breadth, and require an explicitly
requested subtype before narrowing. An employer/workplace/location may answer
an unspecified workplace request; an explicit city question still requires a
city. Do not add names, conditions, events, dates or frequency. Keep both frozen
typed/direct equivalent frames, complete-source fill, leading-context/full-source
NLI checks and current threshold/caps. No runtime keyword classifier, extra model,
new call stage or prior/default changes. The existing example-category list must
not license narrowing every `where` question to one listed example category.

Use the recorded failure plus same-meaning lowercase/rambling/typo variants.
Actual positive employer evidence must reach NLI/support; wrong event year,
hypothetical work, an undated past event and an unknown import clock cannot
supply the requested event year. A present-tense status reported at an
authoritative original timestamp is not a clean negative: its status may hold
then, without implying a job start date. Keep that case diagnostic rather than
mislabeling legitimate dated status as a false positive. Explicit city
positive/absent-city controls retain their narrower semantics. Check current-
supplied positive suppression and negative-prior rescue through shared final
preparation with the old source outside the40-turn recent window, complete
originals and no answer generation. Record frames, filled source inputs and
probability calls privately; never turn unknown into proof. This is bounded
local qualification, not population style invariance or improved cloud answers.

Implemented/qualified locally: four same-meaning workplace variants admit the
true employer; date/modality/explicit-city guards and current-source suppression
survive. The actual44-turn final preparation rescues one proven complete source
outside40 recent turns. The previous fixed13-case guard panel stays11/13 with
zero false admissions and its same two false negatives. Focused50 and combined
813 regressions pass. Keep this corrected best-current mechanism; broad entity
activation/style coverage, contextual-material rescue and end-answer utility
remain qualified-development limits, not new model-search work or parent closure.

### 2.1m Completion/resource receipt preservation — decision, 2026-10-09

The shared cloud adapter rejects non-stop/non-completed responses before
retaining response ID, usage and incomplete reason. The v3 answer/judge failure
therefore keeps only a coarse error class; r4 cannot prove output-cap causality.
Add a bounded text-free completion receipt to the existing exception/generation
and persist it in answer failures and raw judge errors, including attempts.
Allowlist status/model/ID, incomplete reason/error code and numerical standard
usage/reasoning/cache counters; never retain provider output/thinking/error
messages in this diagnostic. Missing fields remain unknown. Empty completed
outputs retain the same receipt. Do not change profiles, output caps, retry
counts, success admission or completed artifact/resume identities.

Native timings currently report load/prompt/evaluation/total independently.
Add nonnegative residual provider seconds only when all original durations are
finite and consistent; missing/inconsistent values remain unknown. Label the
residual unattributed, never claim it is thinking or a queue without evidence.
Validate protocol faults in isolation and through actual answering/judging
artifact writers with controlled SDK responses; no cloud run is required for
that instrumentation check. Whole-stack peaks/concurrency/latency qualification
remain P9 work, not established by this receipt repair.

**Integration grounding:** three full-suite failures correctly reject the new
runner hash in historical completed-answer transitions/input repair. Register
only the exact inspected metadata-only runner/adapter revision, with exact old
and new component hashes; retain finished-input/profile/sample/record checks.
Closed registered runner edges may compose. A changed registry binds its actual
current bytes, never a wildcard component match. This permits only byte-stable
read-only completed artifacts, including retained terminal faults, not new calls
or partial-stage write continuation. Unknown source/adapter/profile revisions
still fail. Prove the old real completed bundle matches without changing any
artifact, alongside the ordinary interrupted-current-run regression. New writer
behavior still requires new development state.

**Local qualification:** controlled SDK and actual artifact-writer/resume tests
preserve allowlisted incomplete details and usage, clear stale failure receipts
after success and reject unrelated compatibility changes. Combined disposable
smoke/background/cloud-provider rails852 passed42.11s; three additional actual
component-byte checks pass in the26-case completed-resume module. Four real r4
answer artifacts pass the closed read-only matcher with unchanged bytes. No
cloud call, old artifact rewrite or token-cap change was made. Original r4's
missing provider reasons remain unknown; telemetry cannot recover them.

### 2.1n Entity-note source contract — decision, 2026-10-09

Actual disposable SQL/worker baseline with a controlled provider writes a
length-marked false approval into an entity description. The original refusal
lies beyond600 characters and never reaches the old prompt. This establishes
two writer defects, not a population/model hallucination rate. Entity readers
already label legacy notes unverified; do not describe those as verified facts.

Replace enrichment's source prefixes with at most eight complete non-private
original turns from its event-linked warm/cold batches, deterministically ordered
by recording time/id for bounded selection only, not event-time authority.
Use the existing P1 original-role windows, cited-assertion parser and pinned NLI.
Generate an attributed source-backed account of the named entity with the shared
complete-input/capacity/finish guard and existing900-token note reserve. Unknown
roles/capacity/completion/parse/support retain the previous description and every
original. Neither a generated prefix nor a source-only fallback is a new note.

Keep existing enrichment scheduling/limits/retirement and no extra model.
Store current source/candidate/entity/policy support proof and separate latest
review diagnostics in reserved existing entity properties, excluded from factual
payload rendering. Success alone updates/journals the description; actual
attempt/supported/contradicted/unknown/failure counts replace candidate-count
claims. Reflection returns this receipt through its existing worker result.
The derived report separates observed, legacy-unobserved and invalid receipts;
it never infers model calls or independent precision from attempted entities.

At retrieval, re-resolve the same bounded current originals and require exact
source/candidate/entity/model/policy agreement before labeling the account
source-supported. Legacy/missing/stale proof remains explicitly unverified;
original graph/source evidence remains available. Warm-to-cold copying with
identical source identity/bytes does not invalidate support. A new/changed
selected source or description does. Support is a faithful account of supplied
sources, not independent world truth or all-fact completeness. Broader semantic
description/cluster quality and full lifecycle still belong to P10 qualification.

**Actual-path grounding:** first positive generation is unknown at citation
parsing; a diagnostic rerun admits a supported account and rejects the opposing
candidate with actual E4B/NLI. This is not corpus reliability. Add the existing
background JSON-schema constraint at the entity-note call, keeping complete
input/reserve validation and explicit finish checks. The shared note helper
accepts an optional schema; existing callers keep their previous behavior.
Account conservatively for schema bytes in the capacity check. Do not repair
format by accepting uncited prose or loosening semantic support. Retain complete
generated parse-failure text only in private, reserved review diagnostics, never
the description/context. Qualify this final candidate once on the actual path.
Final typed E4B/pinned NLI/SQL/reader positive and opposing controls pass,
25 focused source guards14.28s and13 registry/collector/report controls13.78s
pass; final combined rails882 passed42.78s. Old r4 has71 unobserved reflection
calls and unchanged trace bytes. These are mechanism controls, not independent
corpus precision, note utility or answer gains.

### 2.1o Undated cold recall — grounded decision, 2026-10-09

An isolated SQL/full-retrieval control stores one original fact only in cold
storage, with no graph, claim or procedure to rescue it. An ordinary memory
question returns no source; an explicit source-record window returns the source.
The original remains intact. Embedding, NER and reranking were controlled, so
this proves a candidate-eligibility restriction, not answer-quality accuracy.
The restriction implements the older T3 decision; this is a deliberate v3
design improvement under the existing repair authority, not a v2 regression.

Allow cold candidates for ordinary and undated-evolution memory searches under
the existing cold candidate cap. A real embedding or lexical query is required
for undated search; a query with neither cannot browse the whole archive.
Only an explicit source-clock window permits date-only browsing or filters by
recording time. Keep all existing conversation/project/privacy/exclusion/cluster/
batch rules, original representations, dates, global reranking and final budget.
Cold candidates share the episodic packing lane and get no guaranteed slots.
This follows the searchable slow-memory principle described in
[MemGPT](https://arxiv.org/abs/2310.08560), not a claim to reproduce its agent.

The low-confidence wide-net path must offer the same bounded cold leg. Direct
retrieval credits only selected survivors; production preparation still defers
exposure until final assembly. Candidate-only, evicted or write-off reads never
restore a row. Restoration retains the original identity/time/evidence and the
existing probation score, and does not strengthen graph corroboration.

Add an explicit `cold` configurable ablation flag. Turning off temporal parsing
only forces CURRENT; it no longer removes ordinary cold recall. Existing warm
vector and recent-only campaign controls explicitly disable the cold method and
remain narrow. Keep r4 and its resume identities immutable. This decision does
not change warm archived-row eligibility, decay thresholds or confidence floors;
test the actual archive/read/restore chain before claiming lifecycle quality.

### 2.1p Warm archive and probation reach — grounded decision, 2026-10-09

The actual decay → cold → final preparation → restoration control preserves
the original and restores it at 0.12, not archived. A second full ordinary query
then misses it because `_timescope_leg_filters` imposes a hard 0.2 floor. Moving
that same original to the warm archived state at 0.08 also removes it from an
ordinary query. This is a traced policy incompatibility: probation cannot earn
engagement through a reader that cannot see it. It is not a measured relevance
or confidence threshold and no graph/claim rescues this controlled source.

Supersede §2.1o's unchanged warm eligibility: BM25/vector/chunks/wide-net source
search may consider archived and probation warm originals under its existing
candidate caps. Remove the current-only archive exclusion and 0.2 hard floor;
retain the existing positive-score requirement and decay-dependent vector/chunk
ranking, source dates, query constraints, shared scopes/privacy, reranker and
prompt budget. Archival remains a storage/lifecycle state, not a source-truth
verdict. Current graph validity/confidence, procedural activation, retirement,
decay rates/thresholds, reinforcement amounts and source corroboration remain
unchanged. No forced slots or restoration at full strength.

Keep warm-vector controls physical-warm only (now including eligible warm
archived/probation rows); they still omit cold search. Check a repeated ordinary
question before/after restore, a warm archived positive and explicit source-date
exclusion, normal/wide paths, deferred/write-off reads and actual local ranking.
Expose the policy revision in the new experiment manifest. Existing r4 results
describe the old eligibility and remain immutable. Broad retention/answer and
context-efficiency effects still require the controlled lifecycle/Z2 study.

**Validation grounding:** the existing temporal fixture has several newer
same-conversation sources; diversification legitimately omits the low-ranked
archived source from its final wide-net list. Check actual SQL candidacy there,
and final admission in the isolated positive normal/wide controls. Do not remove
diversification or alter ranking to make an eligibility test pass. The actual
local positive admits the target plus two unrelated originals with the existing
unset rejection floor; this does not qualify precision or context efficiency.

**Implemented/locally qualified:** 47 focused SQL/final-preparation controls
pass, then two additional direct-exposure checks join the final 900 combined
smoke/background/cloud-provider rails (49.63s). The temporal suite 66 passes.
Actual shared local encoder, request NER function and pinned reranker recover
cold, warm archived and warm probation targets in normal/wide final prompts;
six dependent controls, no answer/precision/token-efficiency claim. Originals,
roles, dates, chunks and write-off/deferred exposure are preserved. Classifier/
source-need choices are controlled. Full scheduled lifecycle and resources remain.

### 2.1q Cluster source evidence and complete labels — decision, 2026-10-09

An actual disposable SQL merge control has two coherent two-member clusters,
raw cosine0.86 (above0.82 raw floor, below0.90 adjusted floor), and no shared
source entities. Planted generated `RECURRING_ENTITIES` supplies0.10 and merges
them. All four originals survive, but their organization changes on unsupported
generated metadata. A controlled length-marked name is also accepted. Short
originals and late source content are absent from naming input. These establish
mechanisms, not real-model hallucination rates or the retracted G74 aptness rate.

Keep creation/assignment/centroid/merge thresholds, exclusive membership,
session/tag bonuses and existing local model roles. Derive the entity-overlap
cast only from actual linked warm/cold originals, using existing NuNER and
literal source spans/case-whitespace normalization. Bounded deterministic ten-
member cohorts retain the prior size cap; source IDs deduplicate warm/cold copies
and repeated turns. Complete background NER already windows by its own token
contract: make clustering's optional character cap defaultNone, retain explicit
configured prefix mode with a warning, and never silently infer full coverage.
No persistent cache, new resident model or description-derived overlap evidence.
Normalize source-grounded entity surfaces identically on candidate and cohort
sides: a controlled assignment at cosine0.55 lost the0.08 entity bonus because
candidate `Mira` and source cast `mira` differed only in case. Preserve literal
mention grounding; capitalization must not change the similarity decision.

Generated names/descriptions are topical metadata, not factual memory or evidence
for assignment/merging. This follows the separation of cluster formation and
topic representations in [BERTopic's primary documentation](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html);
it does not adopt BERTopic or prove cluster coherence. Supply complete attributed
originals from the bounded cohort, including short turns and chapter/code text;
unknown legacy roles stay explicitly unknown. Replace word prefixes/chapter-line
stripping with complete-source input. Use the existing E4B complete-input/finish
guard and typed schemas, with160 name/500 description output reserves. No source
truncation to fit; capacity/finish/parse failures preserve existing labels, while
a new cluster retains the explicit unnamed fallback. Render recurring entities
from source-derived names, never a generated field. Emit bounded generation
status/counts through the worker/replay path. Broad name specificity/coherence
and downstream memory/latency effects remain unqualified; no G74 closure.

Qualify the false-overlap merge and its genuine-source opposing control through
the real SQL worker, before/after restoration and assignment refresh. Qualify
complete short/late/chapter inputs and truncated/schema/capacity controls at the
existing call seam. Then run one actual local NuNER/E4B coherent-member path,
preserving failures rather than searching endlessly for a favorable prompt.

**Cold membership grounding:** the actual warm/cold opposing controls expose
another storage-policy conflict: each two-member cluster has one warm link and
one known archived member. Warm-only counts call it a singleton, allowing the
looser assignment threshold to absorb it even though the pairwise merge bar is
not met. Use one logical warm+cold membership projection for counts, centroid,
sessions, source cast and label regeneration. Limit10 applies to label/cast
cohorts, not logical counts or centroid. Transfer known cold membership/primary
cluster IDs when a merge changes identity; preserve unknown membership as unknown.
Validate actual SQL/decay/merge/source reach and centroid parity without changing
any similarity/cadence/threshold. This does not authorize general deletion work.

**Implemented/validated disposition:** source-only cast, complete typed labels,
logical membership and worker/replay receipts are implemented. Focused 27 controls
pass in 13.94s, existing cluster behavioral 13 pass, and final disposable smoke/
background/cloud-provider rails 922 pass in 44.37s (280 existing warnings). These
overlap. Actual NuNER/E4B/SQL two-member creation accepts both labels in 9.358s
including first NuNER load. Final case-normalized path passes in 9.316s and a
third source at cosine0.55 joins through source overlap without label calls. Controlled vectors establish
mechanics, not grouping/coherence quality. Read-only r4 recount marks all 274
cluster calls unobserved for the new receipts and preserves trace SHA. No broad
model search, new weights, working-store mutation, cloud call or quality closure.
Full scheduled lifecycle/compaction and complete-stack resources remain work.

### 2.1r Complete graph checkpoints — grounded decision, 2026-10-09

An actual two-pass SQL control first snapshots two outgoing edges, then uses
the real expiration writer (whose event carries only `edge_id`) and adds a
third, negated edge. The next snapshot loses the unaffected second edge and
polarity. Current compaction starts from an empty set on each new event batch;
its relation/target replay cannot consume actual ID-only expirations. The journal
is retained, but that does not make the derived snapshot complete. No production
retrieval consumer currently reads CodexSnapshot; do not claim an answer gain.

Keep thresholds/cadence and the append-only journal. Snapshot the already
maintained entity/edge projection under one repeatable-read transaction rather
than inventing missing event payload or trusting incomplete legacy snapshots.
This follows the complete-state checkpoint role in [the primary event-sourcing
reference](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing);
ICE already has the projection, so no new event store/framework is adopted.
New JSON version2 explicitly records current-graph basis, all outgoing edge
records (identity, relation, target, polarity, source, trust and temporal/observation
metadata), unexpired edge IDs, entity note/properties/aliases and last consumed
event ID. These are recorded graph state, not newly verified facts. Preserve
old unversioned snapshots, all event bytes and all graph rows. Mark only the
observed pending events compacted atomically with the new checkpoint. Do not
build historical field-level replay or change source admission.

Validate real writer/registry/collector behavior across two rounds, polarity,
unaffected and expired edges, legacy snapshot preservation, no-op rerun, and
rollback on checkpoint failure. Record snapshot/consumed-event counts separately
from semantic correctness. Full lifecycle and resource work still precedes tuning.

**Implemented/validated disposition:** complete projection checkpoints and
atomic repeatable-read markers are implemented. Three actual SQL controls pass
in 13.44s; combined disposable rails 925 pass in 49.05s (283 existing warnings).
The real expiration writer, registry and collector participate. A concurrent
new event remains pending and commit failure preserves both old products.
Counts are lifecycle receipts; checkpoint retrieval contribution remains inert.


### 2.1s Reflection review-write schema parity — decision, 2026-10-09

The bounded 42-original real preparation/periodic control recovers the older
fact, then reflection fails inserting a review proposal with no ID. The
migrated production table has server UUID/status/time defaults; the ORM model
defines client defaults, and the disposable runner deliberately creates the
current ORM schema. This is a raw writer/schema mismatch, not evidence that the
production table is broken or a GPU outage. Keep experiment schema creation
unchanged: use the existing ReviewQueue ORM model for both reflection proposal
types so canonical ID/status/time defaults apply in either schema. Store the
existing payload as JSON, preserve pending human review, and add no automatic
slot/cluster update, proposal-quality claim or new model call.

Validate both producers through real reflection/SQL and the current isolated
schema; IDs, pending status and timestamps must be nonnull, payloads unchanged.
Then continue the bounded full periodic/cold-read control. This is a small
shared-writer correction within the core maintenance pass, not product expansion.

**Implemented/validated:** both producers use the canonical ORM writer. Two
actual SQL producer controls preserve their JSON payloads and pending status,
with nonnull IDs/times and no direct slot change. Final combined disposable
rails927/927 pass49.32s; these include the two controls. The bounded real E4B
reflection/periodic/cold-read pass subsequently completes. Production's migrated
defaults were checked read-only; neither the working store nor experiment schema
was altered. This qualifies write/schema parity, not proposal quality.

### 2.1t Bounded resources and scheduled cold-read path — 2026-10-09

**Measured decision:** retain the existing six substantial local components,
small shared-encoder heads, CPU-between-call policies, owned idle drain and
serial maintenance GPU lane. Add no resident model or unrestricted concurrency.
The completed component envelope runs12 stages44.928s, sampling15.45GiB global
GPU use. One serial/local overlap pair1.4581s/1.1094s is a diagnostic, not a
throughput study. Real source proof9.0197s and first NuNER load6.3693s expose
cost; do not label an unattributed provider residual as thinking or queue time.

The completed disposable integrated path runs54.365s, sampling16.44GiB global
GPU use versus the23.5GiB ceiling. Both E4B/NuExtract remain resident at32k
context; headCPU/encoderGPU and pinned reranker/NLI/NER identities are unchanged.
Minimum observed systemRAM available38.25GiB. Exact own CUDA allocator peaks and
Ollama/descendant RSS are retained separately; RSS includes mapped/shared pages
and cannot be summed as physical RAM. Global GPU sampling every0.2s is a lower
bound on the true peak, not worst-case/input-size or local-answerer capacity.

The fixture directly stores42 synthetic complete original-role turns with real
embeddings and controlled50-day clocks. One old source starts archived at0.001
strength. Actual classifier, B2, pressure/budget, request NER, reranking, source
proof and shared final preparation recover the same source before and after
real warm→cold decay; both final prompts fit1944 tokens. Read-strengthening is
disabled only in this diagnostic to preserve the controlled cold transition.
All42 original hashes survive. Ten actual registry/cadence jobs complete through
the synchronous replay boundary: reflection creates pending proposals, clustering
assigns25 rows/accepts2 labels, and episodic decay moves the gold source cold.
No graph/procedure population exists; reconciliation/compaction/graph/procedural
decay are inert, folds are below window, and enrichment has no entity candidate.
Their separate opposing/positive controls remain necessary. This is not HTTP,
streaming, original post-flight ingestion, async dispatch or a cloud answer test.

Preserve failed profiles: v1 scratch-name collision; v3 ambiguous generic query
with no declared history task; v4 raw ReviewQueue/default mismatch repaired in
§2.1s. V2 omitted llama-server descendants from its RSS collector, corrected in
v5; globalGPU/systemRAM measurements were unaffected. No failure is silently
reclassified as a successful complete pass. Private scripts, source hashes and
receipts are identified in [provenance](../PROVENANCE.md#2026-10-09--v3-bounded-resource-and-periodic-cold-read-profile).

These bounded lifecycle/resource checks fulfill the mechanism investigation,
not independent population quality or tuning/final throughput qualification.
Worst-case interactive/long-input resource checks belong to the actual frozen
tuning/runner configuration. No working-store or immutable r4 mutation occurred.

### 2.1u Source-proof quote copying invariance — decision, 2026-10-09

An opposing control supplies the same original proposition and model value,
changing only its copied evidence quote to lowercase/single whitespace. Exact
copy reaches all four proof pairs; changed copy is unknown before any NLI call.
The source-need parser likewise refuses visible evidence. This is a structural
copying restriction, not demonstrated semantic verifier failure or corpus rate.

Reuse the existing `original_source_quote` aligner already used by Codex.
Resolve only literal or unambiguous whitespace/case copies to original bytes;
retain punctuation/negation/identifier content and original spelling. Do not
normalize stored sources, model values or NLI premises/hypotheses. Ambiguous
distinct source spellings, paraphrases, invented text, malformed values and
incomplete responses stay unknown. Source-need requires one resolved original
spelling across its prepared messages. Copy alignment warns without private text.

Source proof keeps the complete original and original leading context, all four
typed/direct pairs, CPU verifier, threshold and bounds. Alignment is a source
anchor, never semantic support. The final preparation still admits only proven,
surviving originals; current-only evidence alone may suppress a positive prior.
Validate copying changes through actual proof/parser and final-preparation paths,
with ambiguous/invented/negated opposing controls, then actual pinned NLI on
original bytes. No new model, runtime intent lexicon or wider rescue policy.

**Implemented/validated:** actual parser/proof and SQL/final-rescue controls
pass70/70, including11 new copying/opposing cases. The real pinned CPU NLI
accepts3 original/case/space-copy controls and withholds explicit negation and
hypothetical controls (5/5); every premise retains original source bytes and
all four checks remain. Frames and provider fills are controlled in this NLI
check, so it does not measure E4B generation or population recall. Combined
disposable smoke/background/cloud-provider rails938/938 pass53.48s (327 existing
warnings, including new SQL-path repetitions). Checks overlap; no threshold or
scope change. The retained13-case panel's two frame/uncertainty misses are not
claimed repaired by this copying change.

### 2.1v Exact cited-span composition — repaired-development evidence,2026-10-10

The completed251-turn/67-probe development replay retains88 summary candidates,
38 supported and50 unknown under the existing verifier. Its summary/abstract
assertions include82 complete pairs rejected above512 tokens. Exact coalescing
of overlapping cited intervals makes25 fit:23 summary assertions and2 abstracts.
The actual pinned NLI supports24 of those25; one remains unknown. These are
dependent assertion checks, not independent precision or whole-summary/answer
improvement. Four opposing controls admit no unsupported assertion; a positive
control remains neutral/unknown, an explicit existing verifier limitation.

At `summary_support.verify_assertions`, compose one complete premise from the
union of every cited original interval. Resolve the quoted window text against
its role/hash/offset manifest. Merge only overlapping or adjacent intervals of
the same original unit with matching role, unit hash and source clock. Verify
overlap bytes exactly; keep different units/speakers and disjoint intervals
separate, in original window order. Retain all cited bytes, qualifiers and
corrections; do not select a favorable subset or invent an intervening gap.
Count and score that complete premise with the unchanged model/revision,
threshold0.95 and512-token limit. Larger unions remain unknown and retain
original evidence. This removes duplicated evidence overhead; genuinely broad
claims still need smaller source-backed generation, a separate qualification.

New receipts use `source_window_assertions_v2`, explicitly identify the exact
span-union composition policy and bind the merged intervals in assertion proofs.
Preserve exact supported legacy v1 approvals: they already scored all their
cited bytes using the same model/threshold/window policy. Check their complete
known legacy policy, proof hash, candidate/source identities and individual
verdicts; an unknown legacy receipt never becomes supported. Changed model,
threshold, source, candidate, window policy or unknown method still invalidates
approval. This compatibility decision avoids discarding previously verified
compression merely because redundant overlap is removed from new inputs.

The maintainer conditionally approves adoption if the evidence shows benefit.
Capacity/exact-source controls plus the bounded native24/25 support result meet
that mechanism condition; no answer-quality claim follows. Do not rewrite the
completed producer, original proofs or frozen tuning seals. Its proof-method-version1 measurements
remain version1; new producer/read-policy identities must explicitly bind this repair.
Validate the shared verifier and representation path, overlapping corrections,
role/gap/unit separation, malformed spans, complete-pair bounds and retained
legacy proofs. Then propagate actual tests and limits before local commit.

### 2.1w Exact graph arguments — completed-development source review,2026-10-10

Read-only review of four hashed IDs per observed confidence tier plus the
remaining negated edge finds one definite actor error among13 propositions.
Four draft/nominal-predicate cases remain uncertain, not automatically false.
All4060 source spans are current and all1392 links have internal support, so
neither source integrity nor NLI/author agreement establishes directional truth.
The high-tier failing edge makes a team apply for an organisation when the
original person applied for the team. E4B's saved reason describes the person
correctly yet its boolean approves the different candidate; NLI entailment is
0.989514. This grounds an argument-consistency repair, not a lower trust floor.

Give `AssertionInput` optional explicit subject/relation/object/polarity fields;
the graph writer supplies all four from each actual proposed relation variant.
Before inference, require complete typed arguments and exact equality between
their existing `relation_proposition` and the supplied proposition. Never infer
arguments by parsing spelling or punctuation. Legacy standalone inputs can
remain unstructured; the production writer always supplies the explicit tuple.
The native message carries both complete original/proposition and that tuple.

Keep the existing E4B, NLI, complete-input limits, output reserve, batches and
source-act policy. Require three typed booleans for subject-role, object-role
and qualifier/polarity/time agreement, alongside the existing support verdict.
Only an assertive/source-qualified report plus support and all three matches
admits the edge; any false match withholds, missing/malformed/uncertain output
stays unknown. Bind arguments in receipts under `author_assertion_arguments_v3`.
Proposals/hypotheses still require their qualified status. Established story
facts retain their preserved source-world scope; no real-world truth is inferred.
No lexicon, extra model, confidence change or old-edge relabeling.

The first actual native candidate gets40/42 expected decisions: no false
admissions, but it withholds an established-story positive and the previously
uncertain modal case. One clarification preserves established-story scope from
the complete source while still withholding proposed scenarios. Final native
42/42 expected decisions83.02s;31 reused older authored controls and11 focused
argument/source controls. This small dependent development set is not independent
corpus precision.15 candidate checks pass50.73s/21 existing warnings, including
two native SQL writer/readers:12 controlled proposals,6 admitted/6 withheld,
sources retained and repeat-call idempotency. Extraction/encoder/NLI are controlled;
E4B and transaction/readers are actual. Retained source reversal is checked
functionally, with synthetic actor/negation controls through the writer.

The maintainer-approved exact two-file patch is applied to shared
`src/memory/assertion_support.py` and the argument constructor in
`src/workers/codex_extractor.py`. Actual production E4B/SQL writer verification
admits three supported proposals and withholds three opposing actor/polarity
proposals, retains three source claims and their reader evidence, and makes no
new writer call on retry. Extraction/encoder/NLI remain controlled in this test.
Final disposable smoke/dynamics1121 pass376.21s/346 warnings; stores removed.
Two failures in the preceding regression were old fake canonicalization responses
missing the new required flags; the fixture was corrected, never the parser.
Frozen baseline/profile artifacts and all35 original table fingerprints remain
unchanged. Exact current-reader transitions are registered separately in§2.3m;
new writer outcomes require new affected-source development state. Tier
calibration and independent extractor/graph/answer utility remain open.

### 2.1x Matching extraction JSON contract — retained failures,2026-10-11

The completed development replay's five failures are reproduced through actual
unchanged extraction/NER/encoder/native transport on their exact original role
units. Five immediate nondegraded successor turns are controls:20 units/20 calls,
159.17s, five assistant-unit failures. Three emit bare tabs inside quoted JSON;
one has a delimiter error; the fifth omits source_sentence on
nine of15 otherwise parsed rows. All finish with normal stop, not length. Every
original table fingerprint remains unchanged. Old provider bodies were not
captured; this is current reproduction, not retroactive identification of those
bytes. Full private request/response/exception receipts precede parsing.

The previous exclusion of the generalist array constraint was correct: it does
not match NuExtract's facts-object envelope. Candidate decision: constrain the
native decoder to that exact existing envelope, preserving nullable subject/
relation/object leaves, optional boolean polarity and empty facts arrays. Require
the source_sentence field as a nonempty string when the existing template asks
for quotes; an unsupported/unavailable quote still cannot admit an edge. Native
source alignment, complete JSON parsing, source-only retention and NLI/author
admission remain unchanged. Do not accept partial JSON, replace original text,
infer missing facts or erase semantic failures to finish a run.

Use the existing codex_constrain_shape switch. With it disabled retain the
unconstrained native body; with it enabled add matching format to the raw native
request. Validate the narrow actual template before HTTP. Version the transport
policy; keep model/template/instructions/input/output/capacity/ownership otherwise
unchanged. Correct the extractor's comment to distinguish the mismatched
generalist array from the matching native envelope. No model or trust setting.

Before production approval: one bounded replay of the ten captured assistant
requests and the one captured empty-array control, changing only native format.
Require complete typed outputs on all five failed inputs, preserve the empty
control, inspect original-quote resolution and keep missing/unsupported evidence
in the denominator. Then verify isolated malformed/template/null/polarity/length
controls and the actual extraction path with candidate transport on retained
sources in a read-only store. Empty outputs or source arrival alone are not
semantic success. Independent extractor precision remains open. Native evidence
and the exact patch must be frozen before requesting the required src approval.
Current three-transition reader profiles and the completed writer remain sealed;
later adoption needs a new explicit registered identity, never old-hash updates.

Qualification completed before the maintainer's sleep pause:11/11 captured
requests return complete typed output114.34s. All five healthy controls retain
identical complete fact arrays, and the empty control remains empty. Eight of293
proposed quotes do not resolve to the original and remain unsupported; grammar
is not precision. Fourteen portable factory/template/null/length/polarity controls
pass0.30s. Two actual extraction/NER/encoder paths complete29.36s:38 relation
proposals plus44 source sentences, including six source-only nullable rows;
no graph write or independent semantic judgment. All35 original table identities
remain unchanged. Candidate/code/request/response/test/patch receipts are frozen
under private logs/v3-extraction-contract-2026-10-10-r2. The exact two-file patch
changes the native transport plus its extractor comment. The maintainer approved
this exact prepared repair on2026-10-11. Apply unchanged qualified source bytes,
promote the focused contract controls into the production suite, then qualify
the applied path and register a new exact reader profile. Preserve old producer
and profile artifacts. Do not rerun the completed producer or infer new writer
quality from its historical outputs. Applied-source64 focused checks pass16.22s.
Two actual extraction/NER/encoder units complete37.96s with40 relation proposals,
46 sentences and six source-only rows; all35 original tables are unchanged.
This is not graph admission or independent precision. Applied receipts are in
private logs/v3-extraction-applied-2026-10-11-r1. Final disposable smoke/
dynamics1135 pass354.82s/346 existing warnings; test stores removed.

### 2.2 Absolute scoring and qualification

Reuse all saved successful r4 answers. Build one blinded packet of the 36
cross-contrast disagreements plus 24 deterministically stratified agreement
controls (across grades, scopes, lengths and task types; backfill only if a
cell is empty). Show original question, complete dated sources and reviewed
requirements, but hide arm/model labels. Distinguish historical assistant
advice from world truth and requested current knowledge. Rewrite an ambiguous
question only in a new packet revision; its old answer cannot score the new
question without a new answer call.

**Packet grounding:** the three saved contrasts bind the same full-answer file.
Select the36 cross-contrast final-grade disagreements from the111 common fully
graded successful occurrences, then24 fixed-seed round-robin agreement controls
stratified by old grade, knowledge scope, task type and answer-length band.
Selection uses saved grades for coverage only; those grades/contrast names are
hidden from reviewers. Preserve complete source and the exact original question,
expected key and cutoff/provenance. The expected key is a reviewed reference,
not permission to demand a historical/private answer the question did not ask.
Unclear intent/key/source truth can be graded uncertain with a reason.

Freeze20 qualification rows before human labels, grouping repeated question
families together; use deterministic group selection that hits20 exactly or
fail with the observed constraint. The other40 are development. Store the
partition/mapping separately from the blinded packet, input hashes/policy/code
identity in a manifest, six readable ten-answer batches, and a separate mutable
label template. All artifacts are private under ignored logs. An identical
rerun verifies existing immutable packet/batch bytes and retains labels; changed
inputs/policy/code require a fresh output directory. No DB/model/API call is
needed for packet construction, and no existing campaign output is rewritten.
The r4 contrasts have `processing_complete=true` but `complete=false` because
terminal cloud/content faults are retained. Accept the terminal-processing flag
for cohort construction; only common successful answers with valid grades
enter the packet. An unfinished/operator-required artifact still fails. Do not
rewrite those flags or count exclusions as reviewed incorrect answers.

**USER-REQUIRED:** review the prepared 60-answer packet in batches, approximately
45–90 minutes total, with correct/partial/incorrect/uncertain and a brief source
reason. The agent prepares all evidence first and continues independent repairs
while labels are pending. Done means each selected row has a disposition;
uncertain rows remain uncertain rather than being forced into a binary label.

**Scheduling override, 2026-10-09:** review only batch1 now; park batches2–6
for late evaluation qualification. Continue independent core repairs/resource/
lifecycle and experiment preparation meanwhile. Ten labels are an initial
diagnostic reference, not a substitute for the reserved20 qualification rows
or permission to declare the judge qualified. Preserve all60 selected rows and
their fixed development/qualification partition; do not select a smaller easier
qualification cohort or tune on reserved labels. Final qualified accuracy claims
still require the stated qualification; earlier diagnostic work stays labeled.

Qualify the actual independent-answer rubric against these labels, including
planted role/date/negation errors and irrelevant same-speaker sources. Require
at least 90% agreement on definitive reviewed labels and no systematic
false-accept pattern on the opposing controls. Report the confusion matrix,
denominators and uncertainty; this is a development admission rule, not a
population guarantee. Before reading the labels, reserve 20 of the 60 cases by
stratified fixed-seed selection for qualification; the other 40 support rubric
development. Do not tune the rubric on the reserved subset. Failed qualification
permits one rubric repair and fresh controls, then a separately qualified second judge profile;
model choice never rewrites old verdicts.

Absolute grades are keyed by immutable input/output identities and are reused
across contrasts. Preferences still use both A/B orders with disagreement
reported. **Two-judge final protocol, reaffirmed 2026-10-09:** independently
grade every successful final answer with Muse Spark1.3 Contributor and a second
explicitly selected, separately qualified model family. This supersedes using
the second judge only for disputed cases/random audits. Freeze both identities
and rubrics before launch; each judge has its own immutable grade cache and
separate both-order preferences. Report each judge's scores, agreement matrix,
disagreements and unavailable/uncertain denominators separately. With two votes
there is no majority: do not silently average them into truth or select the
favorable verdict. Human review resolves remaining disagreements where available;
otherwise retain uncertainty. The existing v4.1 Flash profile is a possible
secondary, not an automatically adopted or qualified judge. Main Muse selection
is retained; qualification checks the current rubric/protocol against references,
not a new broad model search. Prior endpoint/synthetic and v2 calibration evidence
remains valid for its stated scope, but does not establish v3 source/intent grading.
The manual call/cost plan includes both judges for every final arm/occurrence.

### 2.2a Independent grading cache and preference-only path — 2026-10-09

The historical paired judge remains unchanged for r4 reuse. Add an explicit
private research runner `scripts/z1/score_frozen_answers.py` with separate
absolute and preference modes, both defaulting to read-only planning. Grade a
unique successful answer independently using exact question/reference/complete
source, knowledge scope/tasks, question time/provenance and occurrence/cutoff.
The rubric treats the key as a reference, preserves the question's actual intent,
and separates historical advice from world truth. Correct public knowledge need
not match unstated private history; missing evidence is not contradiction.
Return grade/reason/basis, never a comparator or arm label. Successful uncertain
is a completed disposition, distinct from an ungraded provider/schema failure.
Require explicit completed/stop status; a plausible grade without it stays
ungraded. This stricter new instrument does not rewrite historical r4 grades.

Content-address each task by complete evidence/answer, rubric/schema/decoding,
profile/model/endpoint/transport, scorer and call-helper identities. Arm name
and input-file location do not enter the absolute key. Different cutoff/source/
answer/rubric/provider invalidates reuse. A private per-task cache permits the
same absolute grade in multiple contrasts without regenerating answers. Exact
finished cache files are read-only. Independently hash-bind each selected input
collection in a private plan/report; adding a comparator makes a new plan but
may reuse identical tasks. Reject duplicate occurrences, unfinished/operator-
required inputs, incomplete sources and mismatched pair evidence before calls.
Terminal failed answers stay explicitly unavailable and consume no judge call.

Preference mode compares the first declared arm separately against each other
arm in both display orders. It requires the corresponding independent grade
receipts, presents no grades or arm labels to the model, and returns only
preference/reason/note. It cannot write absolute grades. Reverse disagreement
remains uncertain, and preferences/errors are reported separately from fixed
absolute grades. This does not prove a preferred reply is factually better.

Use existing Muse1.3 Responses profile/transport, no automatic fallback or
extra resident model. Reserve each of two maximum attempts atomically before
calling, retain incomplete/unconfirmed/error receipts, reuse successful orders
after interruption and never exceed the persisted attempt cap. Auth/quota/
unknown faults pause immediately; three consecutive exhausted recoverable tasks
pause. A lone exhausted task remains ungraded and later tasks can continue.
Use one cache-directory lock; identity/content validation failures never call
or overwrite. Completed reports are byte-stable on resume. No existing r4
answer/judge artifact is changed. This runner remains diagnostic/unqualified;
actual human qualification and final orchestration are later gates.

Validate cache keys and opposing rubric inputs in isolation, then actual SDK
MockTransport → runner → persisted answer/order receipts, interruption/resume,
failed answers, mismatch refusals and completed-byte reuse with zero network.
Read-only actual r4 planning counts distinct grades/contrasts without grading.
Fresh exhausted-task streaks pause a run; terminal cached failures are retained
without repeating calls or blocking unrelated work on an explicit later resume.
Preference plans/report bind independent grade receipt hashes and values, while
judge prompts receive none of those grades. Completed reports remain byte-stable.

**Local qualification:**19 controlled checks pass, including actual Responses
SDK/transport, unique-grade reuse after a comparator is added, first-order reuse
after interruption, two-call limits, quota pause, new-outage pause/resume,
unavailable answers, source mismatch, missing completion and cache corruption.
Combined disposable rails874 passed42.94s. Read-only real r4 planning sees
454 unique absolute tasks from456 successful answer occurrences;12 failed
occurrences remain unavailable. No grading API call or old artifact mutation.
These validate the instrument, not the proposed rubric's semantic agreement.

### 2.2b Reviewed task consumers and metric admission — grounded2026-10-10

The G40 taxonomy audit finds two current namespaces: generated `probe_type`
records catalog origin/target, while source-reviewed `task_types` records what
the question asks. Legacy retrieval scorers still use the former and their
source-ID/turn-span scores do not establish semantic support. Keep historical
artifacts, but do not use those mixed scores to select current tuning settings.
The independent grader already reads semantic labels in its blind input, yet
`collect`/`report` drop them and the development-repeat marker from its output.
That would make the new cached grades lose their valid task/age denominators.

Add one scorer-only metric contract consumed by offline development preparation
and independent grade reporting. Review labels must use the existing vocabulary
without duplicates; a generated type or target leg never fills a missing label.
Single-source rank tuning is eligible only for a reviewed episodic-lookup-only
private/assistant-history task with exactly one valid source turn, outside the
recent-window before controls. Other annotated source ranks/spans remain coverage
diagnostics. Multi-hop, synthesis, updates, temporal, procedures, negative and
abstention tasks never enter that single-source tuning cohort. Public/mixed
answers do not prove private-memory benefit. All answer/content scores still
require their separately qualified semantic instrument.

Store the contract on every prepared development key; report counts separately
by tuning/validation partition. Carry reviewed knowledge/task labels and repeat
phase through absolute/preference report records. Absolute reports show their
own arm's grades, including uncertain/pending/ungraded/unavailable denominators,
by overlapping reviewed task labels and exclusive knowledge scope. Put before
controls in a separate stratum; their source recency cannot inflate long-term
scores. Keep family counts conservative and explicitly non-independent. This
metadata is scorer-only and never enters runtime retrieval or a preference
prompt. Existing independent cache/attempt/completion rules remain unchanged.

Bind the new helper in preparation and grading identities. Modified instruments
produce a new revision and refuse old immutable plan reuse; no ignore-hash path
or mutation of old answers/reports. New preparation remains offline and not
replay/score ready. Test precise versus synthesis/temporal/negative/mixed controls,
conflicting labels, missing source identity and age strata in isolation, then
through actual preparation and SDK→cache→report/resume paths with controlled
responses. Independent corpus/source labels for final B+C still need review;
passing this contract does not close the whole G40 parent.

**Implemented qualification:**61 focused task/preparation/SDK controls pass;
actual four-arm planning keeps115 older/2 before per arm,454 pending unique grade
tasks and zero calls. Development metric contracts retain5 tuning/2 validation
rank-eligible occurrences, not73 recall questions. Final combined qualification
with §2.3c passes976 disposable smoke/background checks; semantic judge and final
source labels remain unqualified.

### 2.3 Hyperparameter tuning before Z2

**Inputs:** immutable r4 baseline, versioned repaired development stores and
reviewed semantic keys. All r4 questions/results are development material.
Partition new controls by conversation/source event/question family, never by
paraphrase row; reserve an untouched validation packet before looking at new
results. The latest fixture decision reserves B and C, leaving two development
histories (§2.3a); report fold heterogeneity and previous exposure, not a random
population or user-generalization interval. Final semi-LSREP histories/families
and oracle instances do not select hyperparameters.

**Knob inventory first:** enumerate current Settings and every actual consumer.
For each relevant knob record default/frozen value, eligible population, an
opposing-value decision delta, cost, and verdict: load-bearing, plateau,
unexercised, or dead consumer. A declared but unread knob is repaired before
sweeping. Historical duplicate constants/defaults are not assumed current.

**Stage order and bounded candidate sets:**

| Stage | Sweep | What stays fixed / keep rule |
|---|---|---|
| Budget and representation | Total prompt caps 4k, 8k, 16k, 32k, plus unchanged r4 policy as reference; then recent/evidence allocation at 0.75×, 1×, 1.25× current value within valid bounds | Same answering model, output reserve and current input question. Include all system/recent/memory tokens in cap. Refuse infeasible caps explicitly. Compare quality at equal cap and actual cost; preserve exact facts. |
| Candidate selection | Current warm-vector/BM25/claim/chunk limits at 0.5×, 1×, 2×, rounded/clamped to valid bounds; reranker shortlist 32, 64, 128 | Freeze selected representation contract; inspect missing answer spans and candidate redundancy. No writer replay for a read-only limit change. |
| Fusion and ranking | Current RRF k at 30, 60, 90; each evidenced leg weight at 0.5×, 1×, 1.5× current; reranker on/off as a distinct mechanism arm | One coordinate changes at a time; log real decision deltas, task outcomes and latency. Zero-weight is not automatically a complete ablation. |
| Trust and activation | Only source-reviewed graph tiers and procedural evidence thresholds; candidate range derived from labeled support/false-action cases, not arbitrary global lowering | Freeze independent source-truth constraints. Reads cannot create corroboration. If no eligible examples exist, mark unexercised and retain default. |
| Gates and temporal policy | Existing gate thresholds/bumps and temporal-label consumer on meaning-preserving memory/no-memory controls | Recall and false-suppression guardrails; temporal changes need original-date and ordinal-only strata. T5 temporal-label effects are tested only after a frozen baseline. |
| Lifecycle/resource | Solve declared half-life targets, then compare baseline and 0.5×/2× duration sensitivity on a time-advanced development history; batch sizes only within measured residency headroom | No future-time jumps merely to produce desired cold counts. Test reinforcement/decay interaction and real job ordering, not algebra alone. Local foreground residency is a separately labeled optional profile. |

The historical `procedural_similarity_threshold` is inert and excluded from
this tuning ledger. `procedural_match_candidates=5` is a bounded writer cost
cap; evaluate shortlist misses against reviewed equivalent/opposing pairs
before considering a different value. Writer-cap changes require new versioned
development state, never read-only rescoring of the old writer's output.

Run cheap exact/source-stage checks first. Spend new cloud answers only on the
baseline and at most two surviving candidates per stage. A candidate must pass
all source/authority controls and either improve reviewed supported-answer
count at equal budget or keep it equal while reducing actual median prompt
tokens by at least 10%. Any new failure on a previously correct required fact
is inspected before promotion; a net score alone cannot excuse false expiry
or source corruption. Ties keep the current setting. These are operational
development rules, not statistical proof of superiority.

Finish with up to 12 frozen-seed interaction checks around the three most
load-bearing selected knobs (valid ±20% settings), plus an all-originals
answer control on the targeted losses. Reuse cached answers only when complete
assembled input and provider identity match. Publish the knob verdict ledger,
quality/token curves, per-task losses, settings and a final freeze manifest.
No exhaustive combination search, no tuning against final scores.

### 2.3a Offline input and knob preparation — decision, 2026-10-09

Prepare a private immutable input manifest before any new replay/model call.
Both selected paper histories are reserved from new tuning (see §2.4): use the
other two r4 histories,352 original pairs, with the completed source/key reviews.
This supersedes the three-history tuning roster above for this replication.
Both folds remain development diagnostics; previous repair exposure is declared.
Bind original corpus/review/paper artifacts and production/helper code hashes.
Keep exact role text, recorded replies, constructed-clock provenance and original
line/turn identity. Preserve all originals and reviewed invalid/uncertain cases;
only fully reviewed valid through-cutoff keys enter a scoring plan.

Create a question-family tuning/validation partition before new outputs. Use
declared family IDs, otherwise conservatively group within a source conversation;
never count unreviewed paraphrases as independent families. Repeated occurrences
stay together. A two-history reserve is a development diagnostic, not new users
or an untouched external test. The planner may not import a final history into
the new development replay merely because the old r4 snapshots contain it.

Keep final paper probes as unqualified legacy references until exact source
turns, role, task/scope and corrections have been reviewed through every planned
cutoff. Bind each original historical occurrence/reference independently; never
borrow a future or nearest-checkpoint answer. Report historical source-text
matches/mismatches/missing coverage separately from selection identity. The old
curation can extend beyond the paper's replay endpoint; those rows never enter
the selected fixture. Do not silently repair either original artifact.

Draft an AST consumer inventory over current src for relevant numeric/boolean/
weight settings, capturing direct reads, import aliases, literal getattr and
unresolved dynamic reads. Preserve code default and locally resolved values in
the private plan; exclude credential/transport/product-only fields. A static
read is a candidate consumer, not a live decision or load-bearing verdict.
Mark all deltas/cost/verdicts unmeasured until the real tuning instrument changes
them through actual consumers. Missing static reads remain unresolved when a
dynamic path could consume the value; do not auto-delete a knob.

The offline command creates no DB, embedding, completion, grading or replay.
The downloaded oracle release has500 labeled questions:468 string answers and
32 integer answers. Preserve those exact official JSON values in scorer-only
keys; a string-only validator would reject valid quantitative questions. Validate
question/session identities and complete role-content arrays before calls. Bind
only the oracle file, never silently include the fullS file. Legacy semi-LSREP
keys remain unqualified; official oracle labels do not qualify the new adapter
or judge by themselves.
Atomic, hash-bound immutable artifacts resume byte-stably after interruption;
mutable source-review answers stay separate and are never overwritten on resume.
Input/code/policy drift rejects reuse, with no ignore-hash flag. The maintainer
still launches long/cloud runs manually; this preparation CLI is not their runner.

**Implemented preparation,2026-10-09:**
`experiments/v3_repair/prepare_inputs.py` seals the selected original pairs,
reviewed development keys, per-cutoff legacy references, oracle scorer keys and
static settings candidates in private `logs/` artifacts. Current inputs yield352
development pairs/73 admitted keys (67 tuning,6 validation) and1444 final pairs.
The remaining development dispositions are73 invalid and31 uncertain, retained
in the bound original review. The73 keys lack declared family IDs, so the
reservation conservatively groups the two complete histories; six validation
questions are not a new independent user sample. Final references contain792
legacy occurrences/128 families, all still unqualified for current scoring.

Historical archive coverage is1118/1119 matching B pairs with one disagreement
and81 rows after its paper endpoint; C matches248/325 pairs with77 not covered
by those curation files. Both selected originals and archives are preserved.
The manifest reports these limits instead of claiming proven full historical
byte identity. The173 static settings candidates all retain unmeasured live
verdicts. Two known inert declarations have no static reader; this does not
revive the retired procedural threshold or qualify the leg-guarantee switch.
Focused controls cover byte-stable resume, interrupted pre-seal recovery,
source-review drift, corruption/code drift, scorer separation and official
integer labels. This step does not create a repaired database, approve final
source labels, tune settings, schedule aging, or launch an experiment. See
[the offline preparation guide](../../experiments/v3_repair/README.md).

### 2.3b A reachable fusion-diversity control — grounded decision,2026-10-10

`retrieval_leg_guarantee_enabled=True` is a known unread declaration, not a
new roadmap issue. Actual opposing100-token budget calls select the same
episodic/Codex pair with the setting true and false. Merely guarding Phase1 is
insufficient: Phase2 starts by visiting the same best-per-family candidates.
An apparent wired switch could still have no decision effect.

Wire the existing setting to the complete **fusion-fallback family-diversity
strategy**, including best-per-family admission and round-robin-with-slack.
Default true preserves the current strategy. False uses global descending fused
score and the same admission/representation/source-redundancy/session/cap guards.
Successful reranking always retains the caller's relevance order, independent
of this flag; do not sort reranked candidates by their old fused scores.
Both main and wide-net retrieval reach this existing budget seam. Log the actual
selection mode (reranked relevance, fusion relevance, fusion family diversity)
with the existing token/leg-share receipt, without source text.

Verify a tight opposing budget changes selected families with the flag off,
while default-on output and successful-rerank order remain unchanged. Exercise
actual main/wide-net retrieve callers with controlled leg/scorer boundaries,
and preserve oversize-alternative, duplicate-source and admitted-conversation
caps in both modes. This establishes a usable control, not that disabling
diversity improves answers. A corpus with successful reranking throughout can
legitimately yield no flag effect; mark that fallback policy unexercised there.
No new model, default change, diversity retirement or old-r4 rewrite. New
production code identity requires a fresh preparation/development revision.

**Implemented2026-10-10:** ten new budget/main/wide controls are included in
55 passing focused controls; completed disposable smoke/background rails944/944
pass, plus18 controlled cloud-adapter checks. The actual opposing100-token call
now changes the off-mode selection while retaining on-mode output. Source-fetch/
scorer boundaries are controlled in caller tests; SQL/model semantics and answer
gain are not credited. Selection-mode receipts are observable. Remaining named
policy consumers and corpus effect qualification keep the parent audit open.

### 2.3c Offline declared model identities — grounded2026-10-10

Actual preparation of the same sealed eight products accepts an independently
changed `background_model_name` or `codex_extraction_model`: numeric/static
consumer inventory and code hashes do not bind these string settings. This is
an instrument identity defect before repaired-state replay, not evidence that
the pinned production jobs choose the wrong model. Resolve the actual shared
background factory and trace all ten periodic core jobs plus post-flight helpers.
General generation/reconciliation/author checks use E4B; only triplet extraction
uses NuExtract3. Decay/compaction have no generative model. Pinned NLI, embedding,
NuNER and reranker identities remain distinct from those two generators.

Include a safe allow-list of declared local model/role identifiers in the offline
preparation identity: background mode/pin, extraction pin, embedder, classifier
checkpoint/schema paths, background NER, reranker and NLI model/revision. Never
dump Settings or include credentials. Record whether the shared research writer
pins are resolved; unset pins can form an explicitly unready offline plan, but
cannot authorize replay. A pin/mode/revision change must refuse reuse before
rewriting any product. This binds declared identity only, not downloaded binary
bytes, effective context/residency or semantic quality; exact runtime artifacts
and endpoint/decoding still need freeze validation.

Keep current research E4B/NuExtract pins and defaults. Unpinned shared fallback
continues to warn and remains a separate G27 parent decision; cloud foreground
choice does not choose the memory writer. Check pure safe-profile construction,
then actual preparation changed-pin refusal/unchanged resume and the existing
actual-client background-pin controls. No new model, cloud call or old-writer
continuation follows from this audit.

**Implemented qualification:**actual earlier preparation accepts either changed
writer pin without changing its seal. Repaired E4B/extractor/NLI-revision changes
each refuse reuse with all8 products unchanged; identical resume preserves8 too.
68 focused controls and976 disposable smoke/background checks pass. This is
declared identity/consumer qualification, not per-job semantic coverage or a
weight/residency/final-launch certificate.

### 2.3d Fresh development replay adapter — grounded2026-10-10

Reuse `scripts/z1/seed_v3.run` and its production preparation, original writer,
ten-job serial schedule, read-only probes and durable row/trace checkpoints.
Its old CLI only knows the three original histories; injecting globals or
calling that CLI with a smaller input would keep the wrong expected-count and
completion identity. Add keyword-only expected counts, input files and a
selected-plan receipt to the existing engine. Defaults preserve the old plan;
the adapter supplies explicit counts, reviewed probes and source provenance.
Checkpoint identity binds that receipt as well as the existing code/settings/
writer manifests. Never grant continuation of the old r4 writer store.

`experiments/v3_repair/run_development.py` verifies the sealed preparation
bundle against current inputs/code/settings before provisioning. Select exactly
one complete partition history (tuning251 turns/67 keys; validation101/6), with
separate run directories and owned databases. Preserve reviewed source/cutoff/
task/answer keys; pass only each question into production preparation. Save
keys/metric contracts beside the trace as scorer data, never as runtime evidence.
Validation planning is allowed; its replay remains locked until the qualified
tuning/freeze protocol is implemented. Do not inspect its outcomes to choose
knobs. No final B/C or oracle rows enter development replay.

Default command is a no-call plan, writing only immutable private configuration.
`--run` provisions using the existing marker-checked campaign schema clone and
launches the selected replay child, with existing artifact-based progress.
Require the current declared shared E4B/NuExtract/background-NER research
profile; do not force a different profile silently. Bind adapter/helper/code
bytes, preparation identity, partition, exact products, settings hash and run
directory. Reject changed plans before DB/model calls. Repeated identical
commands resume the original trace/checkpoints; corruption or a trace without
a checkpoint refuses, preserving evidence. Retain explicit incomplete/degraded
outcomes and bounded worker policy; no outer blind retry loop. Ctrl+C preserves
the last committed checkpoint and unfinished tail for the existing recovery.

Isolation controls validate future/unknown/duplicate sources and immutable
plan/resume/refusal. A short disposable real-SQL path must use the shared replay
engine, original writer, read-only probes and actual snapshot/restore, with
provider/model seams controlled. This qualifies the adapter, not local model
semantics. The maintainer launches the real251-turn development replay manually;
neither planning nor controlled short runs are a current corpus-quality result.

`audit_development.py` consumes only a completed selected replay plus its
sealed reviewed keys. Verify expected counts, every question/cutoff/source ID
and the selected-plan receipt before scoring. Preserve actual task/scope/age
strata; top5/top10 distinct source rank enters only the admitted lookup cohort.
All other source coverage remains diagnostic. Export a separate private content
review packet binding this trace and actual source bodies; required-unit lists
start pending, never guessed from expected answers. The existing content auditor
evaluates explicit reviewed units/semantic receipts. Missing source IDs, task
labels, stage text or independent labels stay visible. A source hit cannot
become positive semantic support, and pending reviews cannot count as passes.
Atomic report/packet products resume unchanged; no model/API/DB calls. Retain
the original trace and do not manufacture a completed summary from partial data.

### 2.3e Bounded tuning-coordinate seal — grounded2026-10-10

Generate cheap read-side coordinates from resolved settings in
`experiments/v3_repair/tuning_plan.py`: baseline;4k/8k/16k/32k caps;0.75/1.25
base recent-fraction scaling;0.5/2 candidate limits for BM25/vector/chunks/claims;
32/64/128 reranker shortlist;30/60/90 RRF;0.5/1.5 per-leg base weights; reranker
on/off. Omit exact baseline duplicates. One policy coordinate changes per row;
the budget coordinate sets min/max/fallback consistently, while fraction scaling
changes the shared base ladder/default only, retaining density/label modifiers.
Validate the resolved Settings for every row. Field ranges/serving feasibility
still apply at execution; do not silently expand a small window to hit a cap.

Seal the candidate manifest, reviewed tuning keys and all173 consumer candidates
with live-effect/cost/verdict still unmeasured. Hash current preparation, helper/
code and resolved settings. No default changes, DB/model/API calls or writer
replay. Byte-stable identical planning resumes; changed settings/code/input
refuses before replacing a product. This is the coordinate/identity portion of
the tuning instrument, not its executor or a promoted configuration.

**Consumer correction before execution:** `context_budget_min/max` controls the
retrieval allocation, not the final prompt ledger. A4k allocation becomes1949
memory tokens on a short controlled question after existing output/question
reserves; the final ledger still has the routed serving window. Calling that
a hard4k total-input cap would be wrong. Budget coordinates therefore also
carry `input_prompt_cap`. `production_parity.build` records that explicit
research cap on its preamble, and `seed_v3.prepare_with_trace` applies the same
bounded window (cap + unchanged output reserve) to final assembly. Refuse
unknown serving windows or infeasible requested caps rather than invent capacity.
Keep safety margin and current production allocation/reserve policy intact;
report actual allocation and rendered/accounted prompt cost. Default callers
with no experimental cap retain their existing behavior. This fixes the new
instrument's proposed cap, not a demonstrated production-budget defect.

Trust/activation ranges await independent source labels; gate/style thresholds
await opposing source controls; lifecycle coordinates await the declared aging
targets/schedule. Keep these stages explicitly pending, not empty successful
sweeps. Temporal-label effects remain after the frozen baseline. Only baseline
and at most two surviving candidates per stage receive new cloud answers after
source guards pass, as §2.3 requires. Static readership and an enumerated grid
do not prove live deltas, semantic benefit or readiness to launch tuning.

### 2.3f Resumable local read-coordinate collector — grounded2026-10-10

`run_tuning_contexts.py` accepts a completed selected development bundle, the
sealed coordinate plan, one explicit coordinate ID and a new private output
directory. Default is no-call planning. Validate the complete producer trace,
reviewed keys, coordinate seal and current baseline policy before provisioning
or model load. Bind producer/trace/coordinate/helper/settings identity to each
immutable run; code or parameter drift refuses reuse. Validation remains locked.
This collector generates local retrieval inputs, not cloud answers or promotion.

On manual `--run`, take the development bundle's operator lock, require its
owned existing database marker and clone-schema identity, and run an isolated
child. Reuse the existing exact dump/table fingerprint restore of evaluation
states. Group unanswered probes by cutoff; restore that exact prefix once,
then observe each question under a read-only transaction with exposure off.
Load the shared encoder/classifier, run actual preamble and final traced
preparation with the one coordinate applied, and propagate its explicit hard
input cap. Do not change writer policy or re-ingest recorded turns.

Each successful local observation atomically saves a unit by immutable probe
ID/hash, including a digest of its entire saved body. Validate cached units
against run/input/code/content identity before reuse;
checkpoint/failed work never becomes a saved success. Rerun identical arguments
to complete missing units. Preserve baseline gold/source/task/cutoff identities
while recording new generated/ranked/budgeted/selected/final input. Recompute
source-ID coverage using the prefix's frozen source IDs; required-content and
answer grades remain unmeasured. Reuse exact producer source bodies as scorer
data, never append them or expected answers to the answering prompt.

Bind final and evaluation receipts to the producer plan, complete trace prefix,
database/tables, execution code, inputs and writer identities before model load.
Finally restore the producer's committed final snapshot and verify row identity,
including on interruption/failure and when every question is already cached.
If restoration fails, surface it and retain
the units/logs; never report success or silently declare the store pristine.
No asynchronous writer runs concurrently. Completed per-unit files and the final
context aggregate resume byte-stably; no blind outer retries. Short controlled
collector checks cover saved-unit refusal, grouping/resume/read-only policy,
cap propagation and final restoration on failure. Real local completion/quality
needs the maintainer's repaired development state. Cloud answer execution,
qualified source semantic review, independent grading and promotion remain
separate stages under §2.3.

**Controlled qualification,2026-10-10:** saved-body/source/code/settings drift,
default no-call planning and locked validation have opposing controls. A native
PostgreSQL snapshot/preparation control queries three observations over two
historical prefixes, interrupts after one durable unit, resumes without repeating
it, preserves scorer-only metadata, propagates the4k cap to all four arms and
restores every table fingerprint. Complete cache reuse loads no model and still
restores final state; simulated restoration failure cannot return success.
Encoder/classifier are controlled and the fixture writes are synthetic; this is
instrument qualification, not deployed-model/writer precision or answer gain.

### 2.3g Offline read-coordinate comparison — grounded2026-10-10

`compare_tuning_contexts.py` compares one completed baseline collection and one
completed coordinate on the same repaired producer. Validate producer trace,
all per-question content hashes, aggregate equality and common source/code/
settings/seal identity. Require identical occurrence/control rosters; changed
question, source, cutoff, task or expected-answer metadata refuses comparison.
No DB/model/API call and no tuning promotion occur here.

Report matched per-arm prompt/selected-token changes, rendered input equality,
gate/routed-model changes and selected-source additions/removals. Source-rank
changes enter precise lookup diagnostics only for the reviewed eligible cohort;
retain other task contracts separately. Dependence/family limits remain explicit.
These are observed local deltas, not inferred memory or answer gains.

An optional source-unit packet must bind the original producer trace, contain
unique known probe IDs, valid through-cutoff reasons and exact role/hash/offset
alternatives. Reuse the existing evidence-content auditor for each observed
stage. This shared packet declares source requirements only: reject content-bound
semantic receipts rather than silently carrying a baseline grade to a changed
candidate. Missing reviewed units/other task kinds remain unmeasured. Report
literal survival/omission separately from semantic support and answers; retaining
a number in an unrelated or negated statement cannot earn a positive grade.
Source-ID hits with missing literals remain visible instead of being pooled away.

Write a private immutable derived report with input/evaluator hashes; identical
reruns verify byte-stably, drift refuses overwrite. Opposing controls cover same
source ID/lost literal, retained-but-negated text, non-lookup rank exclusion,
metadata/body corruption, changed producer, pending labels and unknown occurrence
refusal. Baseline and at most two source-qualified survivors per stage still need
new Luna answers and independently qualified grading before any selection.

### 2.3h Bounded cloud-answer bridge — grounded2026-10-10

Prepare one baseline plus at most two explicit surviving read coordinates from
one cheap stage. Require complete, same-producer/source/code/settings/seal
collections and the original reviewed question/source identities (§2.3f–g).
This is a diagnostic development answer collection, not qualified tuning
selection, validation execution or scoring promotion. Source/content/writer
panels and independent judge qualification remain gates before a trusted winner.
Default seals a private call plan without credentials, DB, model or API calls.
Bind selected collection/arm/profile/decoding/code identity, display requested
questions and maximum bounded attempts, and preserve unchanged plans on resume.

On manual `--run`, project the already prepared four-arm question records into
the existing `answer_as_of` input schema. Preserve all original source/expected-
answer/task/age/repeat/health data as scorer metadata; only the stored final
messages reach the model. The legacy parser's `allow_partial` compatibility mode
cannot be mistaken for complete old-r4 or score-of-record qualification: the
new adapter has already validated the complete selected producer and records
that distinct policy explicitly. Reuse the existing actual TextGenerator SDK,
Luna profile/decoding, atomic answer cache, attempt-before-call receipts, two-
attempt isolated faults and persistent-outage pause. Do not copy its retry loop,
silently replace the provider or mutate writer-bound scripts.

Namespace cloud sessions by immutable coordinate/arm identity so the same
question across configurations cannot inherit another configuration's session.
Validate cached question/source/expected-answer/task metadata against the frozen
projection before reuse. Successful answers remain unchanged; exhausted faults
remain ungraded, not fabricated answers. Same manual command completes missing
answer units/arms and loads no SDK client when all work is cached. Profile/input/
source/implementation drift refuses reuse; originals and prior answers persist.
Manual output includes per-coordinate/arm receipts, original scorer evidence,
actual completion/usage/fault metadata and unqualified answer-quality flags.
Wire the private full-fault observer around the bridge's actual generator call,
inside the native recovery boundary. Bind its implementation in the answer plan.
Validate its journal before SDK construction even on complete-cache reuse.
Refresh credentials loaded later by the native runner at error-record time, so
unlabelled key values in exceptions are still redacted. Match
the native session to the exact frozen probe and durable pre-call reservation;
record source time, turn/cutoff, coordinate/arm, model/profile/window, input hash
and bounded attempt before re-raising unchanged. Unknown session/reservation
identity refuses the call. No outer retries, altered raw answers or paid test.
Final writer/judge/internal-catch hooks remain separate integration work.
Native answer-cache tags/bytes remain unchanged. Emit separately hashed immutable
grading inputs with distinct coordinate/arm names so the existing independent
grader can compare two full arms without duplicate-name rejection. Its absolute
answer cache continues to reuse the same answer/evidence task across contrasts;
this does not qualify its rubric or promote a score.

Controlled checks use the actual existing answer runner with a fake SDK client:
no-call planning, max-two survivor refusal, exact answering messages without
scorer text, separate session namespaces, interrupted and exhausted cloud-call
resume, cached metadata refusal and unchanged complete answers. Native cloud
output and independently qualified judgments remain separate later evidence;
the agent does not launch the new paid/local campaigns while implementing.

### 2.3i Reader frequency and full fault receipts — grounded2026-10-10

Maintainer requires stored-to-context counts and repeated per-item exposure for
semi-LSREP B/C, all ablation/control arms and oracle. Existing selected fragments
already come from `evidence_after_eviction`; aggregate by-leg counts are not
unique stored-artifact frequencies. Current fragment lineage can identify original
sources and credited graph edges, but not every procedural/note/chunk/slot/recent
artifact/version. Do not silently treat source IDs as those missing identities.

Add an offline private reader report over a complete audited repaired producer
and optionally a complete matched coordinate. For each question/arm, preserve
original-source inventory at that cutoff, leg/rank/budget/final counts, exact
selected-text visibility in final messages, coalesced original turn IDs, credited
edge IDs and source-note lineage. Deduplicate within an occurrence, count
frequency across occurrences, keep arms separate and include zero-read stored
originals. Fragment-content signatures describe rendered versions, not stable
database artifact IDs. Missing text/identity/eligibility is explicitly unobserved;
do not invent derived-artifact denominators or answer-use grades. Bind complete
producer/coordinate/instrument bytes and refuse changed immutable reports.

Final runner integration must additionally capture stable artifact/version IDs,
eligible inventory and final rendered blocks for recent history, slots, bookmarks,
conversation/batch/entity notes, procedures, claims and document/turn chunks.
Report lifetime writes separately from live inventory, eligibility and unique
rendered versions. Keep exact-answer support/use separate from exposure; repeats
are dependent measurements and different arms never add up to unique memories.
The oracle has evidence supplied externally; label that inventory supplied,
not a learned writer-store denominator. Capture remains an observer, never
changes decay strength/corroboration or feeds scorer keys to runtime.

Current worker faults contain only500 characters and no exception chain/stack.
Add a private fsynced append-only fault recorder with whole sanitized exception
chain/traceback, status/request/provider receipt and explicit context (run,
history, turn/cutoff, source/wall time, job/stage, attempt, model/profile/window,
artifact identity). Use private files and redact credentials in message/stack/
metadata; do not serialize arbitrary locals or requests/headers. Record before
an existing recovery policy catches the error, then re-raise the same exception.
No new retry policy, fabricated success or unbounded continuation. Record
system/DB/access faults too, but their original pause policy stays intact.
Append failures cannot pretend the error was safely logged. Tests exercise
actual WorkerRecovery with controlled transient and terminal errors, long
messages/chained exceptions, credential redaction and durable per-attempt IDs.

This helper alone cannot observe exceptions swallowed inside retrieval or
per-item maintenance. Final dispatcher/provider/structured-log seams must emit
full faults before those catches, with overlapping receipt IDs deduplicated.
Old truncated errors remain incomplete. Current producer-bound files stay frozen
while the maintainer manually runs development; helper qualification is not
final-run integration or a retrofit of old evidence.

### 2.3j Named tuning ledger and unattended policy — grounded2026-10-10

The30 sealed read coordinates are baseline plus29 alternatives across13 Settings
fields and a separate hard input cap. Publish exact fields, current resolved
values, actual candidate ranges, consumers and dependencies in the observation
audit; no generic claim that all173 static candidates are being swept. Export
the complete173-entry consumer ledger with every entry marked planned read-grid,
pending qualified stage, held or retired, its current/default value and static consumers. Static reachability is
not a live effect. Trust/entity/procedure activation, gate/style and lifecycle
parameters need opposing truth/false-expiry/aging panels before ranges are
grounded; retain current policy until then. Hard verification/source ordering/
privacy/completion guards and model identities are fixed, not score-maximizing
knobs. Maintenance caps/intervals not covered by the static prefix filter are
explicit fixed profile controls; absence from that inventory is not inertness.

Keep the ten core jobs automatic on their existing source-clock schedule in
full semi-LSREP; ablations disable the named mechanism and verify the resulting
effect. Keep existing qualified automatic changes and pending-review semantics.
Do not bulk-approve slots, motif proposals, fuzzy entity merges or ambiguous
conflicts to inflate activity. Report proposed/applied/withheld/pending/skipped/
errors separately, and qualify each automatic semantic branch before launch.
Fine-tuning remains disabled/frozen, project/import/document product jobs remain
outside these conversation experiments. Reflection's direct `pending_items`
append when that slot exists is distinct from its review-gated slot evolution;
its source/freshness/repetition behavior must be qualified before a seeded slot
profile can claim that branch. Scheduling a job or generating a proposal does
not demonstrate retrievable memory improvement. Oracle evidence-only protocol
does not invent conversational sessions or aging to credit maintenance.

**Implemented observer substeps,2026-10-10:** the
[observation and tuning audit](../reviews/2026-10-10-v3-observation-and-tuning-audit.md)
records exact sealed ranges and automatic/review gates. Offline reader frequency
and full fault observer have opposing and actual SQL/recovery controls; complete
final artifact/eligibility/internal-error wiring remains pending. The first immutable
ledger export records13 read-grid/25 conditional/134 held/1 retired fields and
sixteen fixed maintenance controls without any model/API call or tuning result.
The direct pending-items slot append is a named qualification branch, not
implicitly waived by the review policy. No whole roadmap owner closes.

### 2.3k Retention targets, exposure amounts and chronological gaps —2026-10-10

The maintainer explicitly includes decay amounts and limits in the remaining
pre-tuning work. Four consumer-bearing fields were mistakenly held by the first
ledger: `decay_strengthen_amount`, `codex_retention_increment`,
`codex_retention_cap`, `codex_retention_rank_weight`. Add them to the conditional
lifecycle stage in a new ledger revision, preserving the original export.
Exposure changes need independent chronological writer/read histories; graph
rank weight alone can be observed read-only but does not qualify the coupled
exposure/cap policy. Keep the read-grid seal and production defaults unchanged.
The revised export records13 read-grid/29 conditional/130 held/1 retired fields;
these are parameter dispositions, not the29 roadmap owners.

Grounding: episodic rates are DAILY multipliers, derived using each job's own
cadence. Eligibility starts strictly after seven days; a positive lifetime
access count permanently selects the slower accessed rate. Episodic exposure
adds the configured amount, capped at1.0, once per distinct selected source.
Graph exposure adds its separate increment up to the configured cap; normalized
strength/cap influences ranking but never confidence or factual corroboration.
Increasing that cap reduces the ranking bonus of unchanged existing strengths.
Consequently a larger limit is not automatically better memory. Keep source
truth and quiet-fact reachability as hard controls, not activity targets.

Both runtime/replay dispatch and native numeric workers clamp catch-up to96
cycles. At the90-minute cadence this covers six days. The replay then stamps
the successful job's last-run clock at the endpoint, so a longer gap's omitted
cycles cannot be recovered from that ledger. No inference of full calendar
aging from those calls is valid. A bulk call also uses endpoint eligibility:
it cannot equal chronological runs for a source becoming eligible mid-gap.
Actual native comparison additionally shows cold storage ends warm numeric
aging: chronological runs preserve the score/time of the first cold transition,
whereas endpoint batching overshoots that crossing. Even an always-eligible
source therefore need not have identical endpoint receipts. Preserve transitions,
not just an exponential final-score formula, when validating batch equivalence.

Decision: add an offline lifecycle planner, not a second decay implementation.
Read the actual registry/cadences/cap and safe resolved lifecycle fields. Report
daily/half-life/archive/cold horizons, exposure saturation and graph rank/cap
coupling, with explicit conditional assumptions. Emit ordered native cadence
events (`cycles=1`) and current-policy omitted-cycle counts for a declared clock
interval. Never execute jobs, restore a store, infer source dates or call a model
from this tool. Bind code/settings/clock interval and refuse changed-output reuse.
Final research orchestration must preserve this chronological schedule, complete
eligibility boundaries and all ten job semantics; optimizing batches requires
opposing native equivalence checks first. Do not raise the interactive runtime
cap or mutate the running producer to manufacture aging. Until final integration,
the planner is preparation and the existing replay's long-gap limitation remains.

Before selecting decay: preserve current rate baseline and derive0.5x/2x
half-life alternatives (`daily ** 2`, `sqrt(daily)`), not arbitrary percent drops.
After chronological source qualification, compare0/0.5x/1x/2x exposure increments
and graph caps at0.5x/1x/2x current, with cap/rank normalization reported together.
Zero increment still increments usage/access count and is not an exposure-off
ablation. Do not raise the episodic normalized ceiling1.0. Threshold/creative
floor/procedural age ranges still need source/false-retirement controls. Evaluate
quiet and revisited recall, source preservation, stale/corrected answer errors,
prompt cost and cold/probation behavior; select a changed value only under the
same support/answer/token promotion rules in section2.3. No default change from
algebra or synthetic accounting controls.

Validation: actual native workers under the existing isolated historical clock,
capped dispatch versus exact cadence events, old unaccessed/accessed/creative and
protected sources, a new source crossing eligibility during a gap, graph truth
remaining fixed, normalized cap ranking, exposure dedup/zero amount/master-off,
source retention after cold moves, schedule ordering/fractional remainder and
immutable output reuse. Disclose that these are mechanism controls, not empirical
selection of target half-lives or evidence of improved answers.

### 2.3l Chronological native maintenance gap adapter —2026-10-10

Implement the execution seam missing from section2.3k in new research code,
without modifying the active producer. This is an adapter for a future versioned
parent runner, not a replacement decay worker or completed final launch command.
Take a known per-job maintenance ledger, a declared until-clock/provenance and
explicit enabled core jobs. Merge each job's next exact cadence boundary into
one chronological stream, preserving partial intervals and different anchors.
Execute simultaneous jobs serially in a fixed order; use the actual ten-job
registry and its `needs_db`/`pass_cycles` signatures. Numeric workers get one
cycle. Do not feed exact boundaries into legacy `due_maintenance`: its strictly
positive-overdue gate would skip equality. This is an explicitly ideal cadence
research schedule, not measurement of production idle/lease/tick latency.

Require a parent-owned isolated database, native `Checkpoints` lease, matching
trace sink, complete durable state and exact adapter/settings/job/provenance
identity before any job. Known ledgers may differ by job; do not fabricate a
first successful run or order missing timestamps. Disabled jobs execute nothing,
retain their ledger and are reported separately. Refuse an unknown/duplicate
mechanism, invalid/reversed/future ledger, missing enabled cadence or oversized
interval before model/SQL job calls. Event limit is an explicit upfront refusal,
never truncated aging. Actual source-date authority remains the parent's job.

Use current `WorkerRecovery.call(..., attempts=1)` for each periodic invocation,
matching existing periodic policy, with `FaultRecorder` inside that boundary.
Keep structural/storage/authentication errors fatal, isolated responsive/transport
faults degraded and three successive transport failures a pause. Capture safe
turn/history/arm/job/clock metadata and full exception before native truncation.
Do not infer an actual model from a GPU lane: uncaught job-boundary identity is
`model=None` unless an available provider receipt resolves it. The parent owns
declared/downloaded model pins and actual request identity. Internal swallowed
errors remain a separate capture gap. Observed
no-op/withheld jobs are not quality gains. Reuse native before/after counts and
semantic changes; do not add a second interpretation of those fields.

Only a completely observed time group advances its successful/failed-attempt
ledgers and publishes a native store+trace+state checkpoint, outside the historical
clock. Persist the failure state before checking for outage. An interrupt or
observer/checkpoint failure propagates; the parent must restore its previous
durable checkpoint before any continuation. Lost suffixes remain archived by
native recovery, full fault journals stay append-only, and saved successful time
groups do not rerun. Identity includes complete settings hash, adapter/helpers,
native code hash/registry, enabled jobs and clock basis; changed code/policy cannot
reuse an older producer or bypass its continuation guards. The final parent still
needs its own source/turn/condition/freeze manifest and full writer/read/probe loop.

Validate actual isolated SQL with all ten native functions: positive episodic
cold transition, graph retention without truth changes and procedural absolute-
age retirement, plus explicit no-op/gated semantic jobs. Use the real native
snapshot/checkpoint restore after an interrupt mid-group and compare complete
tables/trace to the same uninterrupted path. Prove matched saved groups skip,
disabled mechanisms do not run, different anchors/remainders survive, full
recoverable receipts persist, and persistent/structural faults do not advance
uncommitted state. No model experiments or production-default changes are
necessary for these mechanism controls; empirical writer/retention quality and
the completed final runner remain separate prerequisites.

**Implemented boundary,2026-10-10:** native ten-job calls and real checkpoint/
restore controls pass, including uncaught job/observer/checkpoint/derived-pause
capture. The synthetic SQL fixture configures hourly cadences and one private
original to keep seven semantic jobs ineligible; only cold/graph/procedural
changes are positive effect controls. No model requests or semantic job-quality
claim. Full source/condition/reader/writer/probe parent integration remains.

### 2.3m Frozen-writer/current-reader adoption — completed-state guard,2026-10-10

The approved exact cited-span repair changes one producer-bound file after the
251-turn replay completed. The existing collector correctly refuses it. Do not
erase its175-file guard, update old hashes, reset/reseed that store or call old
proofs new-writer outcomes. Add an explicit optional research-reader profile.
Ordinary collection without this profile retains its exact old refusal.

`reader_adoption.py` recognizes only exact approved transitions (both old and
new SHA256): summary-support plus the author-argument checker and its graph
constructor, with recovered old bytes from the pre-repair commit and
the unchanged remainder of producer/coordinate code. New paths, missing files,
other edits or other registered-file bytes refuse before model load or restore. Bind the
original complete producer/config/trace, every final/evaluation checkpoint and
dump/manifest, coordinate seal, original settings/model identity and current
reader/instrument bytes in a separate immutable manifest. Reconstruct the old
native code digest from exact verified bytes; require it to match every recovery
identity. Never replace that original digest with the reader's current digest.

The collector accepts `--reader-adoption` only after recomputing the full profile
and checking every binding. Its output separately identifies the frozen writer
and current reader; it is not a writer replay or adoption of new proof outcomes.
Validate again before prefix restore, including complete-cache reuse. Existing
native all-table restore/read-only/missing-unit/final-restore behavior is retained.
The comparator also requires identical adoption profiles, so old/new readers may
not become a tuning coordinate comparison. A changed profile uses a new output
bundle. No source text enters the profile or answer request as an oracle.

Original supported v1 proofs remain valid under their exact unchanged contract;
unknowns are not upgraded. Generated-note composition policy may change fallback
behavior under the new reader; record that as a reader difference, never a new
summary-generation result. New-writer quality still requires a separately
identified affected-source development replay. Final/tuning readiness stays
false: source/role/tier/maintenance and judge qualification remain unfinished.

Validate registered versus unregistered bytes, missing/added paths, model/settings
drift, trace/checkpoint/dump drift, partition reservation, altered manifest and
cache identity, plus the actual isolated SQL prefix path with controlled models.
Interruption must keep completed units unchanged and restore the original final
tables. A planning/profile command must make no database/model calls.

Implemented:25 focused controls pass78.48s; final disposable smoke/dynamics1107
pass414.24s/327 existing warnings, including both ordinary and adopted native
SQL interruption/restore paths and distinct-reader comparison refusal. Actual
profile export/reuse is byte-stable; actual collector planning validates67
pending probes with no model/DB call. No coordinate collection or quality
promotion occurs. Original completed artifacts remain unchanged.

The initially exported summary-only profile is retained. The later approved
argument repair adds two exact byte transitions; new current-reader profiles
must bind all three. Runtime validation also repeats the resolved settings/model/
owned-database check before restore, rather than relying only on CLI setup.
Final focused67 controls pass91.34s/22 warnings, combining parser/argument/writer
and collector/comparison controls. Final post-application disposable regression
passes1121 smoke/dynamics checks376.21s/346 warnings; earlier1107 checks precede
the later production/registry revision. Current three-transition profile export
and identical reuse are byte-stable. Actual collector planning validates67
pending probes without model/DB calls; no coordinate collection or promotion.

**Approved extraction transition,2026-10-11:** section2.1x now changes the
native extraction protocol and its graph-constructor comment. Register those
exact approved bytes as a fourth transition and a revised constructor digest;
retain both earlier profiles unchanged and export a separate current profile.
The baseline native digest must still come from the original frozen bytes.
An old profile must refuse under these new bytes. New-writer outcomes remain
unobserved on the completed producer; this profile only qualifies later readers.
Applied-source64 controls pass16.22s and two native extraction units complete.
The new four-transition profile exports/reuses byte-and-mtime-stably; current
planning validates67 probes with no model/DB call. Ordinary and old-profile
commands retain their refusal. Private receipts:
logs/v3-reader-adoption-2026-10-11-r1. No collection or tuning promotion.

### 2.4 Z2 semi-LSREP: recorded replies, complete memory effects

**Fixture selection:** two separate complete long histories with ordered originals,
updates, recurring topics/procedures and interpretable sessions. Prefer a
different history from r4 to reduce development leakage. Inspect the existing
private mature/stitched and public multi-session assets before choosing; freeze
path/hash, source dates, role map, turn count, sessions, gaps, selected families
and source/answer reviews in a private manifest. Do not silently concatenate
different users into one person. If only a development history is usable,
label the experiment a development longitudinal replication, not held-out.
No corpus is claimed selected merely because a historic spec names it.

**Fixture decision,2026-10-09 (latest steering):** the maintainer selects both
original paper histories B and C, superseding the prior one-history roster and
brief C-only/B-only proposals. B has1,119 original pairs,91 legacy probe families,
20 observed historical cutoffs/638 dependent observations; C has325 original
pairs,37 families,10 observed cutoffs/154 observations. Run1444 original pairs
per condition across two isolated histories. B tests long-range entity/revision
memory; C tests dense long-input/compression behavior. Freeze the same complete
additive and independent leave-one-out roster for each; neither is merely a
small control. Report per-history trajectories/costs before any combined result.
They share one author, so two histories are not two independent users.

The stitched3,473-message project asset is a different corpus and is not selected.
Reuse original recorded pairs and legacy probe identities, then source-review
the improved repeated/correction/cold schedule before qualification. B already
supplied r4 repair evidence; C was not in r4 but is a previously used paper
fixture. Declare each lineage; do not call this a new untouched population test.
Reserve both from new tuning; freeze selected bytes and historical match coverage
in private manifests. Shared data does not license direct comparison to published
v2 scores when answerer/rubric/conditions differ. LongMemEval oracle remains the
separate public condition; no fullS haystack is added.

For **every recorded original prompt** in chronological order:

```text
restore/validate the condition's own durable prefix
set source clock; drain memory jobs due before this event
run the real shared preflight and final prompt assembly
apply the normal selected-evidence exposure/retention effects once
store the recorded original assistant reply (no regenerated replay reply)
run real post-flight, extraction, update and scheduled memory work
save the complete state, job cadence and append-only receipt
at scheduled checkpoints: observe and answer probes without memory writes
```

This measures the memory mechanism under **fixed recorded responses**. It does
not show what an alternate model would have said or how a user would then
respond. All conditions receive the same recorded turns. Only diagnostic probe
answers are generated by the cloud answerer; their questions/replies never enter
future state. Whole mechanisms disabled by a condition must also be absent
from historical preflight/exposure and writer/maintenance effects.

Use the actual runtime's ten periodic memory jobs and a declared synchronous
turn-boundary schedule, including time gaps, decay, compaction, clustering,
summary/reflection, maintenance and warm/cold transitions. Drain gap-due jobs
without ingesting future evidence. Preserve timestamps' origin: authentic
calendar gaps are one stratum; declared simulated aging is another. A model's
wall-clock generation time is not conversation time. Record job eligibility,
calls, changes, failures and source effects; no cold events means no claim about
cold quality, even when the function was called.

**Repeated probes:** freeze at least 20 distinct reviewed fact/task families if
the selected history supports them, covering exact recall, synthesis, updates,
procedures, temporal questions and abstention. Include each eligible family at
first answerability, first cutoff after the actual recent window, a later
section (target at least 100 intervening turns), and final checkpoint. Add a
pre/post-correction pair for update families and pre/post-cold pair only where
an authentic or explicitly simulated lifecycle transition exists. Review the
answer key through **each cutoff**, not only the first. Use the whole qualified
schedule, not a few favorable families; label unsupported category/age cells.

Measure first answerability, retained-correct fraction conditional on earlier
correctness, recovery, correct→incorrect and stale→updated transitions, evidence
survival and tokens/latency at each age. Keep original-state questions separate
from current-value questions. A family with five checkpoints is one dependent
family trajectory, not five independent users. A run with one conversation
supports a case study; block resampling by family is descriptive within that
conversation, not a population confidence claim.

### 2.5 Z2 baselines and additive/subtractive ablations

**Shared answer envelope:** same current question, source/query-time formatting,
recent window, system instruction, answering provider/decoding/output reserve
and total token cap for all memory conditions. Expected answers never enter
that envelope. Include recent-only to expose answerability without retrieval.

**Two vector controls are required:**

- **Warm vector:** the current ICE warm-vector leg with its shared gate,
  reranker and representation policy. Exclude graph/claim/procedure/summary
  legs, source notes, slots/bookmarks, timeline helpers and query expansion.
  Its same-store form measures direct read-path contribution; label that shared
  state inheritance. A lifecycle comparison uses its declared condition state.
- **Bare vector:** independent append-only index of every original role-tagged
  history chunk available at the cutoff, using the same embedding model,
  deterministic chunking/overlap, raw text and cosine top-k. No learned memory
  gate, reranker, BM25, graph/claims, summaries, procedures, importance/recency
  boosts, consolidation, decay, cold exclusion or memory-write strengthening.
  Deterministic overlap deduplication and budget-fitting are allowed and declared.
  Record parent/source spans. Tune its k/chunk-size on development data only;
  keep the frozen best baseline, not a deliberately weak choice.

Bare vector still needs embeddings, an index, source metadata and prompt
packing; “nothing” means none of ICE's additional memory mechanisms. Do not
make this control weak by giving it fewer original sources or a smaller cap.

Before launch, enumerate components from **actual reachable consumers**, with
switches for writers, readers, helpers, prompts, retention and scheduled jobs.
The initial mechanism groups (split only if effects are independently viable):

1. BM25/hybrid candidate search.
2. Reranker.
3. Attributed sentence-claim index.
4. Graph identity/relations/traversal and grounded query expansion (requires
   source claims; disabling graph may keep claims, disabling claims forces
   graph off unless a separately qualified source-only dependency exists).
5. Supported turn summaries/abstracts.
6. Rolling conversation and batch summary memory (requires supported-note
   contract; include source-mode notes as part of this mechanism).
7. Procedural extraction/activation/retrieval.
8. Learned/source-proof retrieval gating.
9. Cluster organization and scope/ranking effects.
10. Temporal query filtering/timeline interpretation (source dates remain in
    all conditions; disabling this is not permission for future evidence).
11. Decay/retention/cold lifecycle.
12. Semantic maintenance: reconciliation, enrichment/reflection and compaction
    effects not already owned by another group. Record its dependency overlap;
    split update versus reflection only when the switch truly isolates them.

**Additive:** start from the independently specified raw vector condition and
add groups in the above dependency-respecting order, one at a time; also report
the warm-vector and recent controls. Freeze the order before seeing Z2 answers.
Order-dependent gains are conditional contributions, not intrinsic values.
**Subtractive:** start from the same complete full-ICE configuration each time,
disable one group, rerun that condition, then restore full configuration before
removing another. Never cumulatively subtract. If a parent switch necessarily
disables dependents, name the compound removal and do not call it a singleton.

Run all requested additive and leave-one-out conditions on the frozen
semi-LSREP panel. Do not quietly select only cheap or winning arms. With K
independent groups, the naive layout has K+1 additive states and K full-minus
states plus controls; deduplicate identical endpoint configurations by complete
manifest hash. Produce the exact arm/call/cost count before operator launch.
Full-ICE and both vector baselines also run the oracle condition. Additional
oracle ablations are diagnostics only if explicitly included before its freeze;
the comprehensive mechanism ablations belong to semi-LSREP.

**State rules:** read-only omissions can reuse the identical frozen as-of store
only for labeled read-side contrasts. Writer, entity identity, summary,
procedural, gate-exposure, maintenance or decay changes require their own
chronological replay. Reuse source bytes, embeddings and response caches only
when all relevant prompt/model/config identities are identical; never reuse
full-system derived state as though an absent mechanism had not created it.
Separate immutable r4 originals from these new condition stores.

### 2.6 Z2 LongMemEval oracle

Pin the downloaded official oracle dataset file/revision and hash; inspect
all schema/count/type coverage before running. The
[official LongMemEval definition](https://github.com/xiaowu0162/LongMemEval)
uses evidence sessions only and contains 500 instances. Oracle session arrays
are not guaranteed chronologically sorted: sort linked ID/date/session tuples
stably, preserve every role turn, and verify question-time eligibility. Do not
substitute the full S haystack or the separately named LongMemEval-V2 dataset.

Per instance, isolate state and replay the supplied history through the repaired
memory path using recorded replies. Answer at `question_date` with full ICE,
warm vector and bare vector. Also send the complete supplied evidence sessions
directly to the **same answerer** as an all-evidence oracle control when they
fit its verified context window. Record any capacity failure instead of
truncating the reference. This separates memory loss from answerer/key limits.
Preserve upstream question types, source session/turn labels and abstentions;
do not infer new answers from answer-session IDs alone.

Target excellent oracle performance and inspect every miss, but do not require
manufactured 100% correctness to let the result exist. Freeze before the first
complete run. Any subsequent repairs evaluated on those failures are labeled
oracle development revisions; a future generalization claim needs untouched
data. Report the upstream-compatible score separately from richer source-based
partial/uncertain grades and completeness/errors.

## 3. Files and integration points

Use existing mechanisms; these are seams verified in current main, not promised
new independent implementations:

- P0: `scripts/z1/report_v3_replay.py::summarize`, `seed_v3.py` trace,
  `answer_as_of.py`, `judge_answers.py`; preserve existing artifact identities.
- P1/P10: `src/memory/support.py::score_pairs/verify_support/supported_current`,
  `representation.py::verify_representations/choose_representation`,
  `summary_snapshot.py`; `post_flight.py`, `conversation_summary.py`,
  `batch_summarizer.py`, `reflection.py` and their real reader consumers.
- P2/P4: `src/retrieval/orchestrator.py` existing chunk, claim, graph and budget
  paths; `reranker.py`; `src/memory/claims.py`, `relation_support.py`,
  `src/workers/codex_extractor.py`, `memory/assertion_support.py` and source-backed
  entity operations; author receipts on existing claim/link JSON, actual writer
  control in `tests/test_author_assertion_path.py`, and new seed/report counters.
- P3: `extraction_result.py`, `codex_extractor.py`, `post_flight.py`, existing
  bounded worker recovery; real rejected responses remain private fixtures.
- P5: `_reconciliation_evidence`, maintenance `_decide_reconciliation` and
  `detect_all`; extend source-order provenance through actual import/store/
  archive/claim consumers if needed, with a migration only for new fields.
- P6: `procedural_extractor.py`, `reflection.py`, orchestrator
  `_procedural_lookup`, existing support identities and temporal fields.
- P7/P9: `memory_decision.py`, `source_need.py`, `source_proof.py`, classifier/
  NER consumers, `bg_client_factory.py`, runtime scheduling; cloud completion
  receipts in `experiments/lme/cloud_provider.py` and existing z1 runners.
- Input preparation: `experiments/v3_repair/prepare_inputs.py::prepare` and
  `settings_inventory.py::build_inventory`, with
  `tests/smoke/test_v3_input_preparation.py`; existing r4 label validation is
  reused, historical v2 runners are not imported or executed.
- Tuning: extend `scripts/z1` current as-of/control/config seams and add a
  versioned private sweep manifest; document verdicts in `docs/tuning_report.md`
  when measurements exist, not invented results now.
- Z2: a new current-v3 coordinator under `experiments/v3_repair/` reuses
  proven checkpoint/cloud-recovery/clock modules; inspect existing
  `experiments/mature/` and `experiments/lme/` for reusable mechanisms. Keep
  frozen v2 experiment code unchanged. Document the actual CLI after it exists.

## 4. Failure modes and resumability

One operator entry point owns each campaign and can run all declared stages or
a named remaining stage. New Z2 runner design must retain r4's durable unit
semantics: full state+trace checkpoints, per-answer/per-order atomic writes,
reserved cloud attempts, model/endpoint/settings/source identity, active-run
locks, status and useful progress per condition. No manual sequence of loosely
coupled scripts with untracked dependencies. The runner is **not implemented
yet**; do not present a speculative command as runnable.

Ordinary isolated model failures retain originals, bounded attempts and explicit
degraded/ungraded records. Transport outages, auth/quota problems and identity
failures pause for the operator without deleting progress. Completed upstream
stages are byte-stable. Never append new model/settings results to a frozen r4
file or bypass changed-hash guards to save work. New repair children bind their
baseline snapshot and changed configuration explicitly.

Exercise Ctrl-C/restart during a local write, snapshot boundary, gap job,
cloud reservation, first judge order and report. Compare resumed versus
uninterrupted state fingerprints, scheduled-unit identities and costs. An
unconfirmed remote request may have been billed: do not promise exactly-once
billing. Progress separates planned, processed, successfully saved, degraded,
excluded and pending. Every final denominator reloads durable artifacts.

## 5. Validation and exit checklist

Each package gets a positive/opposing source control, one isolated check and
one actual production-path check. Use disposable DBs with current schema; keep
private counterexamples out of Git. Cheap semantic checks and saved-output
review precede costly cloud calls. No new test merely restates an implementation.

**Repair exit:** all P0–P10 have a written implemented/validated disposition or a
bounded empirical decision with the affected behavior explicitly retained and
its coverage limit named. Confirmed false evidence, false expiry, lost originals,
wrong chronological state or silent measurement errors cannot be waived as
“low benefit.” A hypothesis failing to improve can retain the measured best
baseline; do not claim that component has been fixed or qualified.

**Tuning exit:** live knob-consumer checks, matched-budget curves, source/answer
validation, resource profile and interaction checks complete; final manifests
frozen. The small source-support/negative/style panels remain mandatory even
if aggregate quality improves. Broad product coverage is not added to this gate.

**Z2 readiness:** each condition's real enabled/disabled paths differ as intended;
no unintended shared writer state, future evidence or observer side effect;
truth labels exist for every admitted claim; the exact arm/question/call table
and cost estimate are available; short full-cycle resume/control tests pass.

**USER-REQUIRED:** after implementation, inspect the short readiness summary,
provide the chosen cloud credentials locally if missing, and manually launch
the documented command in tmux. The agent does not start the long experiments.
Duration is estimated from the verified short path and frozen arm count, not
the historical four-night guess. Done means every planned unit has a durable
result/error/exclusion and final reports bind those identities. Review disputed
final semantic cases in prepared batches; no approval is needed for routine
independent repair work already authorized.

### 5.1 Repair disposition audit — implementation boundary, 2026-10-09

All11 packages now have a written mechanism or bounded keep decision. This is
the boundary for preparing a repaired development refresh and tuning instrument,
not a declaration that every memory feature improves answers or that29 roadmap
owners are closed. The retained baseline, failed candidates and original data
remain. New writer identities require new development state. The latest combined
938 rails and actual controls qualify named mechanisms, not independent accuracy.

| Package | Implemented/validated disposition | Explicit remaining qualification / decision |
|---|---|---|
| P0 — measurements, e.g.986 unknown summaries | Nested status/reason denominators and semantic evidence funnel repaired; old missing content remains unobserved, never inferred from source IDs. | New development traces and reviewed answer-support keys must qualify each observed stage before tuning scores are trusted. |
| P1 — long-source compression, e.g. a source exceeds512 NLI tokens | Role-bound windows/cited assertions/current manifests and complete-source fallbacks operate in writers/readers; actual substitution/opposing controls pass. | Independent completeness and quality/token curves remain tuning/evaluation measurements; unknown support retains originals. |
| P2 — evidence selection, e.g. the gold document loses its exact value | Child/query-selected original evidence, BM25 document flag and recent role-share redistribution repaired through actual final packing. | Current writer/read revisions need versioned development state to measure residual misses; no gold-guided runtime selection. |
| P3 — local writes, e.g. NuExtract receives a general chat template | Actual native structured task, complete input/reserves, bounded splitting and completion/negative receipts repaired; retained failed/success controls pass. | Broad failure rates and length coverage remain unqualified; old missing raw completions cannot be retrospectively diagnosed. |
| P4 — graph truth, e.g. a proposed scenario becomes an asserted relation | Source-author/NLI/canonical-predicate guards and complete attributed source rendering pass actual writer/opposing controls. Existing confidence/identity policy retained. | Independent role/direction/modality/tier precision and answer utility remain qualification gates. No lower trust floor, forced connectivity, new alias resolver or false promotion. |
| P5 — time/update authority, e.g. synthetic dates block real corrections | Hash-bound conversation order versus calendar authority, fair detector scans and event-versus-recording query axis repaired; real inline/periodic and warm/cold controls pass. | Historical answer/false-retirement precision remains measured on reviewed development/final sources; unknown cross-history order preserves both. |
| P6 — procedures, e.g. an opposite rule reinforces a habit | Source-cited semantic identity/admission, eligibility before cap, original/observation clocks and retirement receipts pass actual writer/reader controls. | Activation/cap defaults retained until reviewed relevance/context gain; no reading-as-corroboration or forced activation. |
| P7 — gates/style, e.g. a workplace request becomes city-only | Best existing factual proof retained with corrected category breadth, actual old-source rescue and original-copy alignment; opposing/context controls pass. | Two retained framing/uncertainty misses, contextual/multi-source and broad style/gate accuracy remain limits. No further model tournament or claim they are fixed. |
| P8 — judges, e.g. one answer changes grade by comparator | Unique independent-grade cache, both-order preference-only path and blinded packet implemented; actual controlled SDK/resume checks pass. Muse remains primary. | Human labels/rubric qualification are parked as directed; scores stay diagnostic. Final second family must be selected/qualified and report separately. |
| P9 — cost/completion, e.g. incomplete has no recorded reason | Text-free provider receipts, actual-duration scopes and bounded deployed-role resource profile implemented; sampled16.44GiB, no added model. | Keep serial maintenance/device/owned-drain policies. Worst-case frozen-run/long-input/interactive/thermal throughput still needs qualification before launch. |
| P10 — maintenance/lifecycle, e.g. a cold source is no longer found | Source-bound entity notes, original-member cluster decisions, logical warm/cold membership, full graph checkpoints and archive/probation access repaired. Actual ten-job/cold-read control preserves42 originals. | Inert branch counts are not semantic coverage. Blind cluster/note specificity, natural retention and downstream answers remain evaluation gates; checkpoint retrieval remains inert. |

No currently demonstrated integrity defect is waived by this audit. The named
false assertion/expiry, original-loss, chronology, quote-copy and measurement
mechanisms have repairs or explicit source-preserving abstention. This does not
assert the absence of unknown defects. The eleven rows are execution
dispositions, not eleven feature scores or roadmap checkmarks.

**Next order:** prepare new repaired development inputs with source/answer keys
and an untouched family-level validation partition → audit live knob consumers
and build the bounded resumable tuning instrument (§2.3). Prepare final Z2
orchestration alongside this work; complete its truth keys, additive/independent
leave-one-out switches, independent bare/warm-vector controls, dual-judge identity/
cost table and interruption/resume checks before freezing final manifests.
The maintainer launches long/cloud campaigns. The old r4 command is not the new
tuning/final launcher. Judge semantic qualification stays late as directed;
unqualified grades may diagnose but cannot select a score-of-record configuration.
No final campaign or hyperparameter result exists yet.

### 5.2 Remaining work by next evidence stage — rechecked2026-10-10

**The repairs are not the whole roadmap.** The formal count remains160 anchored
items:93 checked,67 open =29 dependency-sensitive owners +35 later items
+3 experiment gates. The eleven package dispositions do not close those29.
Offline B/C input preparation does not pass tuning or final readiness.

The following is an exclusive **next-work** partition of the existing29, not
29 new defects or a guarantee that every parent closes in its assigned stage.
Seven need prerequisite audits/instrument or scoring decisions, sixteen need
repaired development qualification and bounded tuning, four mainly need frozen
answer/contribution comparisons, and two are conditional/after-baseline work.
These shared observations overlap; a repaired mechanism can serve several owners.
All23 prerequisite/development owners need their relevant evidence or explicit
bounded disposition before final launch. The four final-comparison owners still
need working labels/instruments/conditions before launch; only their final
outcomes wait. Original broader/product scopes remain visible, never waived.

| Next work | Existing owner and concrete remaining evidence |
|---|---|
| Prerequisite (7) | **G40 — task labels:** [consumer audit/metric contract](../reviews/2026-10-10-v3-task-metric-consumers.md) is implemented (§2.2b), preserving task/age/error strata and separate rank admission. New final probe/source labels still need qualification. |
| Prerequisite | **G55 — truthful scoring:** the repaired source/hash/offset content funnel detects a present source ID with its required value missing. A verified disposable as-of reconstruction isolates64-candidate admission on one retained case; alternative representations were unrecorded. Broader required-unit/semantic review and actual reader qualification remain pending; no source-ID hit becomes answer accuracy. |
| Prerequisite | **G71 — old findings:** the16-entry [applicability register](../reviews/2026-10-10-v3-historical-finding-register.md) is audited; identity/counts re-recorded, invalid historical inferences withdrawn. Independent current role/tier truth and answer utility remain pending under the named development/final owners; no old extractor percentage becomes a v3 result. |
| Prerequisite | **G72 — current state:** the manually completed251-turn/67-probe tuning partition passes source/key/task validation and all35 live-table fingerprints match. [Current writer/reader audit](../reviews/2026-10-10-v3-repaired-development-audit.md) records5 degraded writers and unqualified semantic outputs. Preserve this completed baseline; new read/derived qualification must explicitly bind subsequent repairs, never bypass its producer identity or imply v1 receipts came from the version2 assertion-proof writer. |
| Prerequisite | **G29 — live consumers:** [all twelve named clusters reconciled](../reviews/2026-10-10-v3-policy-consumers.md), including opposing fusion/window controls; old bookmark/job-key bypass claims were stale. Actual full knob effects remain tuning work; keyword/style limitations and product-only scope remain visible. |
| Prerequisite | **G27 — background identity/default:** [core consumers/factory query](../reviews/2026-10-10-v3-background-model-consumers.md) audited; §2.3c repairs missing declared model pins in offline identity. Keep E4B/NuExtract research pins, qualify actual replay/freeze identities. Wider unpinned/routed fallback remains unresolved. |
| Prerequisite | **G65 — scoring reliability:** qualify independent absolute grading against reviewed references before trusting a tuning promotion. Human batches remain parked as directed; diagnostic work may continue without calling its grades qualified. Select/qualify the second final judge before freeze. |
| Development/tuning (16) | **A9 — entity activation:** trace meaning-preserving queries through actual entity search and source arrival. New NER architecture is conditional on measured bottlenecks, not mandatory here. |
| Development/tuning | **A12 — local model tradeoff:** qualify the deployed pair's job quality and frozen-run capacity. The old all-candidate comparison grid remains wider conditional scope; no new model tournament. |
| Development/tuning | **G4 — resource budget:** check the longest frozen input/real schedule and headroom;16.44GiB sampled on a small fixture is not a worst-case guarantee. Optional local answering remains a separate profile. |
| Development/tuning | **G15 — unnecessary retrieval:** include banter/current-supplied facts and historical questions; compare false suppression and actual prompt cost. |
| Development/tuning | **G28 — style invariance:** measure gate/source/answer changes for the same meaning. Two retained framing/uncertainty misses are explicit limitations, not fixed claims. |
| Development/tuning | **G49 — procedures:** inspect source-backed activation, relevance, time consumers and final evidence on actual procedure questions; no threshold lowering to force activity. |
| Development/tuning | **G57 — representation:** review new source-to-summary/claim outcomes and useful substitution at matched context budgets. |
| Development/tuning | **G63 — extractor precision:** independently review actual source role/direction/negation, with failures and withheld proposals kept in the denominator. |
| Development/tuning | **G68 — length bounds:** qualify source tails, complete request/output reserves and truncation rates across observed input lengths. |
| Development/tuning | **G69 — canonicalization:** review current modal/directional predicates and source recovery; a passing authored control is not corpus precision. |
| Development/tuning | **G73 — conversation folds:** judge support/completeness against actual originals, especially later corrections, and observe cost/substitution. |
| Development/tuning | **G74 — clusters:** review real members, source-grounded specificity and assignment separately; arbitrary consecutive turns remain an invalid instrument. |
| Development/tuning | **G60 — updates:** review actual coexistence/correction decisions and historical reads; measure false expiry before relaxing any source-authority guard. |
| Development/tuning | **G67 — maintenance:** review eligible/applied changes and entity notes under the real lifecycle schedule, beyond counting job completions. |
| Development/tuning | **G75 — compression trust:** measure actual substitution, required-detail omission and supported-answer/token effects before changing the keep/fallback decision. |
| Development/tuning | **G76 — attributed graph evidence:** review assertion/source/tier precision and relevant final context on repaired state; calibrated graph truth precedes trust-threshold tuning. |
| Frozen final comparison (4) | **G30 — full reader quality:** review complete source→stored representation→prompt→answer traces and disclose unexercised branches/product-only scope. |
| Frozen final comparison | **G70 — write/read/use:** report each representation's writes, candidates, selection, answer support and cost. Reads remain exposure, not new corroboration. |
| Frozen final comparison | **G66 — supported answers:** compare full versus warm/bare vector at matched budget with the frozen answerer and independent judges. |
| Frozen final comparison | **G48 — component value:** run the full additive and independent leave-one-out roster on both B and C; source presence alone cannot justify a component. |
| Conditional/after baseline (2) | **G51 — missing semantic links:** existing candidate remains NO-GO. Reopen only on safe opposing-source controls plus measured recovery/answer benefit; adding links is not a prerequisite to launching the current system. |
| Conditional/after baseline | **T5 — temporal-label ranking:** preserve a frozen baseline first, then separately compare the temporal-label gate/recency consumer; it may not invent date windows. This is not implicit in the existing temporal questions. |

**Execution order:** finish prerequisite consumer/label/old-finding audits and
prepare repaired development state → review current writer/maintenance outputs,
fix demonstrated failures and complete bounded live tuning/interaction checks →
freeze after qualified promotion/readiness → manually run B/C semi-LSREP and
oracle → inspect losses and close only earned original scopes. Source/correction/
repeat labels, lifecycle clock design, both vector controls, actual ablation
effects, dual-judge identities/costs and short per-unit recovery controls are
final readiness work, not work postponed until after the final run.

**Retention readiness clarification,2026-10-10:** section2.3k adds four omitted
exposure/cap fields and qualifies current catch-up limitations through native
SQL/dispatcher controls. The planner and native ten-job gap adapter in section2.3l
are implemented; full source-event/parent integration and revisited/quiet/corrected retention panels
still precede lifecycle parameter selection. Do not treat the current cap,
larger endpoint batches or a no-call schedule as that completed qualification.

After both final experiments, the35 later items remain outside this repair pass.
Conditional owners and broader A9/A12/G27/product acceptance can also remain
open. The number of evidence-led repairs after those results is unknown until
the results exist. No experiment completion automatically clears the roadmap.

**Count clarification, maintainer2026-10-10:** an interim7/7/4 operational
package count was a different grouping and incorrectly suggested18 roadmap
items. Withdraw it from progress counts. Use the existing exclusive29-owner
partition above:7 prerequisites,16 development/tuning,4 final outcomes,
2 conditional/after-baseline. Audits/implemented tools are completed substeps,
not whole-owner checkmarks. The251-turn/67-question command refreshes repaired
development memory and historical prompts; it does not run tuning or cloud
answering/judging. Relevant pre-tuning quality work remains pending on that
fresh state. B/C keys/schedule, full ablations/baselines, oracle adapter,
dual-judge qualification and orchestration/recovery/freeze are launch readiness,
not an additional seven roadmap items. Post-result repair count stays unknown.

## 6. Look-ahead and roadmap closure

Existing ownership, with concrete acceptance examples:

| Existing item | This plan supplies / what still earns completion |
|---|---|
| G40 — task labels, e.g. synthesis needs several facts | Current consumer audit and scorer-only rank/age/task contract implemented; only5/2 development occurrences rank-eligible. Final B+C task/required-fact/source labels still need qualification. |
| G55 — complete meaningful scoring, e.g. a gold-ID hit loses its value | P0/P2 full semantic funnel; the 1471-turn seed is complete, source IDs alone remain insufficient. |
| G57 — write representation, e.g. a long source never compresses | P1/P3 actual supported substitution and loss audit. |
| G63 — extractor truth, e.g. a linked relation reverses its source | P3/P4 independent source/role/direction review. |
| G68 — input/output ceilings, e.g. a long turn produces incomplete JSON | P1/P3 length-stratified coverage and failure-rate repair. |
| G69 — relation canonicalization, e.g. “can” becomes “does” | P4 opposing modal/directional controls and real relation review. |
| G71 — old findings, e.g. an old extractor percentage reused as v3 | All16 claims mapped in the dated register; current empirical gaps remain explicit under original owners. No borrowed scores or parent closure from an applicability audit. |
| G72 — fresh state and confidence tiers, e.g. low-tier facts excluded from direct reads | Fresh seed/snapshots done; P4 independent tier truth and P0 identity retention remain. |
| G73 — conversation folds, e.g. a later correction disappears | P1/P10 support, completeness and actual prompt-cost evaluation. |
| G74 — cluster names, e.g. a label unsupported by its actual members | P10 blind member-grounded specificity review. |
| G60 — supersession, e.g. two coexisting values wrongly treated as replacement | P5 eligible source-order/date controls and historical reads. |
| G67 — maintenance, e.g. every candidate is rejected before model inference | P5/P10 eligibility, fair scans, real decisions and semantic output review. |
| A9 — entity activation, e.g. a paraphrase fails to open graph search | P7 same-meaning preflight/source-arrival controls. |
| A12 — local model quality/cost, e.g. E4B and NuExtract residency | P3/P9 deployed-pair quality/latency/VRAM; old full model-comparison scope remains separate. |
| G4 — GPU budget, e.g. overlapping encoder and local writer peaks | P9 measured complete-stack profile, not sum of isolated model sizes. |
| G15 — noise gate, e.g. banter unnecessarily retrieves | P7 memory/no-memory opposing cases and token effects. |
| G27 — background defaults, e.g. unpinned fallback changes writer | Core consumers/factory audited; safe declared roles/pins/revision bound in preparation. Research pair retained; runtime/freeze identity and wider unpinned policy remain visible. |
| G28 — style invariance, e.g. punctuation changes source access | P7 source/gate/answer changes with intent held fixed. |
| G29 — duplicate consumers, e.g. tuning a setting leaves a literal active | Tune live consumers and audit the entry's named remaining copies; unrelated product work stays later. |
| G30 — reader quality coverage, e.g. smoke passes but a useful span is absent | P0/P2/P10 actual-model source-to-answer checks, with product-only readers deferred. |
| G49 — procedures, e.g. repeated supported instructions never activate | P6 and final procedural ablation; exercise learned/unlearned time consumers explicitly. |
| G51 — semantic links, e.g. related names incorrectly imply identity | P4 only source-proven links with answer benefit; no topology target. |
| G70 — write/read/use, e.g. exposure counted as corroboration | P0 per-representation semantic use, exposure and source support reported separately. |
| G75 — summary trust, e.g. invented or overlength summaries | P1 source-faithful, complete-enough compression at measured cost. |
| G65 — judge reliability, e.g. one answer changes grade by comparator | P8 independent absolute grades and human qualification; graph rubric separately qualified. |
| G66 — memory improves answers, e.g. full vs vector at the same budget | Tuning and both Z2 controls/answer comparisons; no source-presence surrogate. |
| G48 — leg utility, e.g. non-episodic evidence gets zero credit | P0 semantic credit plus full additive/subtractive Z2 conditions. |
| G76 — attributed claims, e.g. assistant speculation becomes user fact | P4 source linkage/role/precision and downstream answer review. |
| T5 — temporal-label effect, e.g. old-state ranking changes | Freeze baseline first, then the dedicated temporal-consumer comparison. |

The formal 29 dependency-sensitive items are not 29 mandatory reseeds and not
29 guaranteed closures. The first campaign completed their shared data
collection prerequisite and exposed repairs. ROADMAP checkmarks require each
original item's actual acceptance scope, not simply reaching this document's end.

## 7. Traps

- R4 is a development baseline, not a clean healthy-system estimate: local
  degraded work and ungraded cloud answers remain part of its result.
- A good source quote does not establish the proposed relation or cover every
  question qualifier. A good graph shape does not establish fact precision.
- Adding a decision model cannot fix a path that never admits a source pair.
- Full/no-Codex context comparison is not full graph deletion; warm vector is
  not independent bare vector; read-only probes are not ordinary remembered turns.
- Reusing 39 as-of snapshots saves read-side work. It cannot make a new writer
  or new decay history retroactively exist in an old store.
- One history, overlapping task labels, repeats and display orders do not
  multiply independent evidence. Correct-answer quality and provider failures
  use separate denominators; excluded/uncertain labels remain visible.
- Keep measured v3 results separate from the frozen v2 JMLR manuscript and
  from prior failed-instrument numbers. Repairing documentation changes no score.
