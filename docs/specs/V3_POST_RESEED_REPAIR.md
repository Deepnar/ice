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
4. Run **Z2: one long-conversation semi-LSREP with full memory lifecycle and
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

### 2.3 Hyperparameter tuning before Z2

**Inputs:** immutable r4 baseline, versioned repaired development stores and
reviewed semantic keys. All r4 questions/results are development material.
Partition new controls by conversation/source event/question family, never by
paraphrase row; reserve an untouched validation packet before looking at new
results. With only three histories, use leave-one-history-out diagnostics and
report heterogeneity; do not claim a random population or user-generalization
interval. Final semi-LSREP families/conversation and oracle instances do not
select hyperparameters.

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

### 2.4 Z2 semi-LSREP: recorded replies, complete memory effects

**Fixture selection:** one complete long conversation with ordered originals,
updates, recurring topics/procedures and interpretable sessions. Prefer a
different history from r4 to reduce development leakage. Inspect the existing
private mature/stitched and public multi-session assets before choosing; freeze
path/hash, source dates, role map, turn count, sessions, gaps, selected families
and source/answer reviews in a private manifest. Do not silently concatenate
different users into one person. If only a development history is usable,
label the experiment a development longitudinal replication, not held-out.
No corpus is claimed selected merely because a historic spec names it.

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

## 6. Look-ahead and roadmap closure

Existing ownership, with concrete acceptance examples:

| Existing item | This plan supplies / what still earns completion |
|---|---|
| G40 — task labels, e.g. synthesis needs several facts | P0/P8 reviewed task/required-fact labels; current 255 dispositions are complete, older taxonomy consumers still need audit. |
| G55 — complete meaningful scoring, e.g. a gold-ID hit loses its value | P0/P2 full semantic funnel; the 1471-turn seed is complete, source IDs alone remain insufficient. |
| G57 — write representation, e.g. a long source never compresses | P1/P3 actual supported substitution and loss audit. |
| G63 — extractor truth, e.g. a linked relation reverses its source | P3/P4 independent source/role/direction review. |
| G68 — input/output ceilings, e.g. a long turn produces incomplete JSON | P1/P3 length-stratified coverage and failure-rate repair. |
| G69 — relation canonicalization, e.g. “can” becomes “does” | P4 opposing modal/directional controls and real relation review. |
| G71 — old findings, e.g. an old extractor percentage reused as v3 | Map each remeasure claim to r4/new evidence or inapplicable; no borrowed scores. |
| G72 — fresh state and confidence tiers, e.g. low-tier facts excluded from direct reads | Fresh seed/snapshots done; P4 independent tier truth and P0 identity retention remain. |
| G73 — conversation folds, e.g. a later correction disappears | P1/P10 support, completeness and actual prompt-cost evaluation. |
| G74 — cluster names, e.g. a label unsupported by its actual members | P10 blind member-grounded specificity review. |
| G60 — supersession, e.g. two coexisting values wrongly treated as replacement | P5 eligible source-order/date controls and historical reads. |
| G67 — maintenance, e.g. every candidate is rejected before model inference | P5/P10 eligibility, fair scans, real decisions and semantic output review. |
| A9 — entity activation, e.g. a paraphrase fails to open graph search | P7 same-meaning preflight/source-arrival controls. |
| A12 — local model quality/cost, e.g. E4B and NuExtract residency | P3/P9 deployed-pair quality/latency/VRAM; old full model-comparison scope remains separate. |
| G4 — GPU budget, e.g. overlapping encoder and local writer peaks | P9 measured complete-stack profile, not sum of isolated model sizes. |
| G15 — noise gate, e.g. banter unnecessarily retrieves | P7 memory/no-memory opposing cases and token effects. |
| G27 — background defaults, e.g. unpinned fallback changes writer | P9 actual resolved identities and explicit default decision; no silent pin edit. |
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
