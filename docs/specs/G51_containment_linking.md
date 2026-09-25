# G51 — Source-backed containment recovery, not inferred graph edges
Assumes decided specs: G50_entity_consolidation.md, D1_D2_maintenance_agent.md, G45_open_relation_vocabulary.md, V3_REPAIR.md (G76 source claims and verification)

**Qualification status, 2026-09-25: NO-GO for v3 activation.** A bounded
detector/writer prototype passed six disposable SQL/model controls, but its
positive edge was already written by ordinary extraction. A genuinely missed
category relation was also missed by the specialist+NLI recovery path. A typed
category candidate rescued that positive and then wrote a false edge on a
matched unrelated source. The candidate and prototype were removed from
production before commit. The source-explicit Laya shadow selected `asserted`
for the unrelated, hypothetical and former-relation negatives. Keep original
sentence claims searchable; resume this item only with a source-explicit
positive/negative control set and a demonstrated retrieved-answer gain over
the existing claim/episodic readers. Historical name-overlap/degree counts
are not a reason to activate it.

## 1. Decisions

**v3 re-grounding, 2026-09-24. This document replaces the earlier G51 implementation plan.** That plan proposed writing a `CodexEdge(confidence="pending", extraction_confidence=0.6, source_batch=<agent run id>)` from name containment alone. In v3, pending edges are traversed whenever their trust clears the direct floor; `pending` is not a quarantine. The agent run ID is not a real episode and cannot identify an evidence passage, survive source deletion correctly, or receive fair origin-batch credit. A deterministic guard can reject a bad candidate; it cannot prove the remaining relationship. The old plan would therefore make plausible names into unsupported asserted facts.

The historical arm-1 count (8,280 nodes, 75.1% at degree 0–1, 5,487 unjoined containment pairs) described the old extractor. A read-only count on the current v3 working store found 4,380 conversation nodes, 6,603 live edges (5,314 pending), 3,037 nodes at degree 0–1, and 2,136 whole-token containment pairs, 42 with direct live edges. That store has zero `codex_claims` and zero writer-attributed episodic turns: it is older data, not evidence that the v3 sentence-claim writer is poor. These are structural counts, not retrieval or answer-quality scores.

**Decision:** containment is a *candidate generator*, not a relation or identity signal. `emotional validation` and `validation loss` both contain `validation` but need different treatment. G51 may recover an edge only from a complete, writer-attributed original source that supports the proposed directed relation. The edge uses the actual source batch and the existing Codex source-claim/write contract. An unsupported pair stays unlinked and its source is retained. No synthetic `source_batch`, no name-only `pending` edge, no automatic merge. A `same_entity` proposal goes to G50's existing identity route. Reflection, maintenance and retrieval do not silently turn a navigation hint into a fact.

**Further v3 re-grounding, 2026-09-24:** conversational entities are global:
`get_or_create_entity` leaves their `project_id` NULL, so a candidate detector
cannot group them by that column or infer project identity from a node. The
source turn's conversation supplies project and visibility identity. The old
scoped reader admitted only the primary source batch and resolved only warm
turns. G76 now admits independently observed warm/cold sources and renders and
credits the selected visible quote. Candidate discovery remains read-only and global;
one complete, visible source must support both endpoints and the relation.

Graph degree is a diagnostic, not the acceptance metric. The success question is whether source-supported answers improve per context token without extra false relations. A degree-1 node can be a perfectly valid isolated fact. [Graphiti](https://github.com/getzep/graphiti) attaches facts to episodes, and [Neo4j's graph builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html) extracts entities and relations from chunks with source links; neither motivates treating lexical containment itself as evidence.

## 2. Algorithm & data model

1. Generate whole-token containment candidates among live **global conversational** entities. Do not use `CodexEntity.project_id` to partition them; it is NULL for these nodes. Skip identical IDs, existing live direct edges, previously rejected pair revisions and G50 `difference_kind` rejections. Dedupe by ordered `(specific_id, general_id)`; prefer adjacent names in a nested chain but do not create transitive edges. Cap work by 5 head clusters / 50 pairs per pass; count both values explicitly. Candidate generation performs no write to `codex_edges`.
2. Resolve candidate evidence from original warm/cold source batches attached to either endpoint's existing edges or source claims. A candidate is processable only when **one** current, complete source unit has an authoritative role, belongs to an eligible conversation/project scope, and actually supports both endpoints. Never combine mentions from different projects or truncate a passage to fit a model. Unknown-role legacy sources remain available for raw retrieval but do not authorize a new relationship. Document visibility follows the existing enabled/shared-document policy.
   **v3 code re-grounding, 2026-09-25:** the current `CodexClaim` carries the exact attributed sentence, paragraph, warm/cold source identity and source hash. A short name occurs *inside* its longer containment name by construction; that single span cannot count as two endpoint mentions. Require a second, non-overlapping whole-name mention in one current original source sentence before spending a specialist call. Start the bounded discovery pass from current source claims; original turns without a claim remain retrievable and eligible for normal Codex extraction, but are not silently re-attributed by this detector. This is a recall limit to report, not evidence that those turns have no relation.
   This first automatic write path accepts **user-attributed** claims only; assistant text can still be read, but cannot settle a user's relation by itself. Store a `containment_considered` event with the real source batch and claim ID after a completed unsupported/no-proposal decision, keyed to the source and model revisions; unknown/verifier outages remain retryable. The switch defaults OFF until a source-to-answer comparison qualifies it.
3. Ask the existing NuExtract specialist for *proposals* constrained to the source excerpt and pair: a directed relation, polarity and exact quoted source sentence. The generator may propose zero or several relations. Require each quoted sentence to resolve to the original attributed source. Entity names are not synthetic evidence.
   **Rejected candidate after the opposing SQL/model control, 2026-09-25:** the pair-constrained specialist returned ordinary unrelated triples on two bounded cases despite endpoint instructions. On `I use emotional validation as my preferred kind of validation`, ordinary extraction wrote only `I --use--> emotional validation`; the pair specialist proposed vague `is`, whose NLI entailment 0.949 missed the 0.95 support floor. A hard-coded `is_a_kind_of` proposal scored 0.998 on that positive source, but an actual writer test then found NLI **0.991** on `I track emotional validation in the model, and validation matters to me` — a source that never asserts the category relation. The typed proposal wrote a false edge and was removed before enablement. The earlier negative `validation loss` and hypothetical controls were insufficient because the lexical pair itself changes NLI's prior. A future source-explicit judge must pass matched positive/negative pairs before this proposal is reconsidered. Do not lower the NLI threshold or claim graph improvement from the positive-only test.
4. On the first v3 repair path, pass every exact-source proposal through G76's source→proposition support check, including speaker, assertion/hypothesis, polarity, direction and temporal scope. Only a qualified supported relation enters the existing `handle_triplet`/`CodexClaimLink` transaction with the **real turn batch ID**. Canonicalize the proposed relation through the open-vocabulary ladder; do not hard-code `is_a` or create a separate edge writer. Source uncertainty, an unsupported proposal or verifier failure leaves both nodes and all original facts untouched. This is the single production path; it does not wait for an unrelated model search.
5. In the *same* v3 conversation trace, run one local typed-decision candidate in shadow over the excerpt, pair and up to four generated proposals, with fixed choices `proposal_1`…`proposal_4`, `same_entity`, `unrelated`, `junk`, `unknown`. A Laya/Decider-style model chooses or scores *existing options*; it cannot generate arbitrary relation text or an exact quote. Its choices do not authorize a graph write during G51. If it distinguishes supported from unsupported proposals without increasing missed true relations in that trace, carry it into Z1's bounded decision qualification for G50 identity, G60 relation supersession and temporal/profile updates; otherwise retain the existing source-backed path and record the failure. Laya's base checkpoint is below a majority baseline on its published zero-shot test; Decider 4B v2 is a local candidate around 8.4 GB, with 2B as a lighter fallback; SemIf's 4B typed logits are a no-fine-tuning baseline. None has ICE semantic qualification yet.
6. Record detector counters: candidates, source-resolvable, source-qualified, accepted edges, unsupported, unknown, rejected and actual extra context tokens. Keep rejection memos keyed by pair plus source revision, so genuinely new evidence can reopen a pair. No new durable navigation-edge table is justified until a read-path benefit is demonstrated.

G76's edge-level proposition verification and the source-level scoped reader are dependencies, not something this detector duplicates. The current `CodexClaim.verification` verifies an exact source sentence against its containing paragraph; it does **not** by itself prove the relation inferred from that sentence. G51 must not call that exact-quote check a relation verdict.

## 3. Files & integration points

| File | Change |
|---|---|
| `src/workers/maintenance_agent.py` | Register a bounded, read-only containment detector in `DETECTORS`; type only candidates with attributed original source. Reuse the existing job loop and warning conventions. |
| `src/memory/typed_decisions.py` | Shadow-only bounded choice/score/null adapter for one local candidate; preserve probabilities and abstention. The contract also fits edge reconciliation, entity-profile updates and temporal-status choices. Production G51 writes never depend on this adapter until a later Z1 qualification. |
| `src/workers/codex_extractor.py` and `src/memory/claims.py` | Reuse G76's source-qualified relation acceptance and existing edge/claim write transaction; do not fork an agent-specific edge writer. |
| `src/api/config.py` | One off-by-default containment-recovery switch and explicit cluster/pair caps; no `codex_conf_inferred` setting. |
| `docs/FEATURE_INVENTORY.md` / architecture | Show default OFF and distinguish candidate detection from source-qualified edge writes. |
| `tests/test_g51_containment.py` | Disposable SQL/writer and reader controls below. |

## 4. Edge cases & failure modes

- `validation loss` / `validation`, `a lack of validation` / `validation`, and `emotional validation` / `validation` must not receive the same unsupported default relation. A fragmentary or stopword-like node may be reported but not deleted by G51.
- Cross-project pairs and private or unshared-document sources cannot authorize another project's conversational graph edge. Existing source scoping and deletion must continue to work.
- Repeated name pairs, duplicate aliases, nested chains and concurrent runs must not produce duplicate edges or reinforcement from the same batch.
- If only an assistant hypothesis or unknown-speaker legacy text supports the pair, preserve it as source material but do not turn it into a user assertion. The source can be reprocessed if later attribution or correction becomes available.
- Missing, stale, overlong or hash-mismatched source spans, verifier failure and incomplete model output abstain without marking the pair permanently rejected. Never log raw source or provider exception parameters.
- A relation already asserted from another batch follows the existing source-qualified reinforcement and conflict logic; a repeated read is not corroboration.

## 5. Validation checklist

1. Read-only v3 candidate count and a small stratified sample, recording the store/extractor version. Do not compare the 2026-08-17 arm-1 percentage to the current store as an improvement.
2. Disposable DB: whole-token containment, nested-chain dedupe, project boundary and existing-edge skip. The detector itself writes zero `CodexEdge` rows.
3. Real writer path with G76 verifier: exact, attributed, supported directed source writes one edge with the source turn's batch and a linked claim; missing/assistant-hypothesis/negated/unknown source writes none and preserves originals. The shadow typed model cannot change these writes.
4. Same-source rerun is idempotent; a new independent source may corroborate; rejection memo reopens only on a source revision. Deletion of the sole source retires the edge through the existing cascade.
5. Run one local typed candidate in shadow on the same source-backed conversation decisions. Report wrong selections and missed supported proposals; do not enable a decision gate merely because it raises edge count. Broader qualification, only if promising here, belongs to Z1 rather than another G51-only campaign.
6. One complete recorded conversation through production retrieval, comparing matched answer evidence and context tokens with the switch off/on. If extra context displaces better evidence or the asserted-edge false-positive count is nonzero, leave the switch off and retain the candidates for diagnosis.

## 6. Look-ahead constraints

G50 owns identity decisions and `difference_kind`; G44 owns unusable entity repair. G76 owns source proposition support and G64 searchable sentence claims. G48 leg attribution requires a real turn batch on every new edge. G70 usage accounting must distinguish candidate discovery from final-prompt inclusion. Later LME oracle and semi-LSREP runs should use the same frozen answerer/context budget for the on/off comparison. There is no reason to implement `unmerge_entities` inside a feature that never merges.

## 7. Traps

- `pending` is eligible for graph traversal under the trust floor; it is not a safe container for guessed edges.
- A model's `link` verdict over names is still an inference over names. Even a perfect enum parse does not supply a supporting episode.
- More connected nodes, a higher degree score, or more fact lines can all be produced by false relations. Judge source support and answer evidence, including token displacement.
- The current store's zero attributed turns makes a write-side qualification vacuous; prepare the code against v3 writer fixtures, then judge it on a properly replayed v3 conversation. Do not backfill speaker roles from a raw delimiter guess.
