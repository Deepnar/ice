# G51 — Source-backed containment recovery, not inferred graph edges
Assumes decided specs: G50_entity_consolidation.md, D1_D2_maintenance_agent.md, G45_open_relation_vocabulary.md, V3_REPAIR.md (G76 source claims and verification)

## 1. Decisions

**v3 re-grounding, 2026-09-24. This document replaces the earlier G51 implementation plan.** That plan proposed writing a `CodexEdge(confidence="pending", extraction_confidence=0.6, source_batch=<agent run id>)` from name containment alone. In v3, pending edges are traversed whenever their trust clears the direct floor; `pending` is not a quarantine. The agent run ID is not a real episode and cannot identify an evidence passage, survive source deletion correctly, or receive fair origin-batch credit. A deterministic guard can reject a bad candidate; it cannot prove the remaining relationship. The old plan would therefore make plausible names into unsupported asserted facts.

The historical arm-1 count (8,280 nodes, 75.1% at degree 0–1, 5,487 unjoined containment pairs) described the old extractor. A read-only count on the current v3 working store found 4,380 conversation nodes, 6,603 live edges (5,314 pending), 3,037 nodes at degree 0–1, and 2,136 whole-token containment pairs, 42 with direct live edges. That store has zero `codex_claims` and zero writer-attributed episodic turns: it is older data, not evidence that the v3 sentence-claim writer is poor. These are structural counts, not retrieval or answer-quality scores.

**Decision:** containment is a *candidate generator*, not a relation or identity signal. `emotional validation` and `validation loss` both contain `validation` but need different treatment. G51 may recover an edge only from a complete, writer-attributed original source that supports the proposed directed relation. The edge uses the actual source batch and the existing Codex source-claim/write contract. An unsupported pair stays unlinked and its source is retained. No synthetic `source_batch`, no name-only `pending` edge, no automatic merge. A `same_entity` proposal goes to G50's existing identity route. Reflection, maintenance and retrieval do not silently turn a navigation hint into a fact.

Graph degree is a diagnostic, not the acceptance metric. The success question is whether source-supported answers improve per context token without extra false relations. A degree-1 node can be a perfectly valid isolated fact. [Graphiti](https://github.com/getzep/graphiti) attaches facts to episodes, and [Neo4j's graph builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html) extracts entities and relations from chunks with source links; neither motivates treating lexical containment itself as evidence.

## 2. Algorithm & data model

1. Generate whole-token containment candidates among live conversation entities with the same project identity (including two global/NULL nodes). Skip identical IDs, existing live direct edges, previously rejected pair revisions and G50 `difference_kind` rejections. Dedupe by ordered `(specific_id, general_id)`; prefer adjacent names in a nested chain but do not create transitive edges. Cap work by 5 head clusters / 50 pairs per pass; count both values explicitly. Candidate generation performs no write to `codex_edges`.
2. Resolve candidate evidence from the original warm/cold source batches attached to either endpoint's existing edges or source claims. A candidate is processable only when a current, complete source unit has an authoritative role and an excerpt that actually mentions the specific entity. Never truncate a passage to fit a model. Unknown-role legacy sources remain available for raw retrieval but do not authorize a new relationship. Under project identity, sources from another project do not qualify; document visibility follows the existing enabled/shared-document policy.
3. Ask a bounded typed decision model about a *source excerpt plus the candidate pair*, not the names alone. Allowed outcomes: `relation` (directed relation + exact supporting source sentence), `same_entity`, `unrelated`, `junk`, `unknown`. An unparseable/partial output is `unknown` with a warning, never `relation`. `junk` informs G44 entity-quality work without deleting nodes. `same_entity` delegates to G50 and never downgrades to a link.
4. Resolve the returned sentence as an exact source span and run G76's source→proposition support check, including speaker, assertion/hypothesis, polarity, direction and temporal scope. Only a qualified supported relation enters the existing `handle_triplet`/`CodexClaimLink` transaction with the **real turn batch ID**. Canonicalize the model's relation through the open-vocabulary ladder; do not hard-code `is_a` or create a separate edge writer. Source uncertainty, an unrelated verdict or model failure leaves both nodes and all original facts untouched.
5. Record detector counters: candidates, source-resolvable, source-qualified, accepted edges, unsupported, unknown, rejected and actual extra context tokens. Keep rejection memos keyed by pair plus source revision, so genuinely new evidence can reopen a pair. No new durable navigation-edge table is justified until a read-path benefit is demonstrated.

G76's edge-level proposition verification is a dependency, not something this detector duplicates. The current `CodexClaim.verification` verifies an exact source sentence against its containing paragraph; it does **not** by itself prove the relation inferred from that sentence. G51 must not call that exact-quote check a relation verdict.

## 3. Files & integration points

| File | Change |
|---|---|
| `src/workers/maintenance_agent.py` | Register a bounded, read-only containment detector in `DETECTORS`; type only candidates with attributed original source. Reuse the existing job loop and warning conventions. |
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
3. Real writer path with stubbed typed decision and G76 verifier: exact, attributed, supported directed source writes one edge with the source turn's batch and a linked claim; missing/assistant-hypothesis/negated/unknown source writes none and preserves originals.
4. Same-source rerun is idempotent; a new independent source may corroborate; rejection memo reopens only on a source revision. Deletion of the sole source retires the edge through the existing cascade.
5. Actual local model on a bounded gold set spanning kind/part, wrong sense, fragment, negation, reversed relation and style variants. Report unsupported-edge precision **and** missed supported relations before enabling the switch. Do not accept a mere rise in edges or degree.
6. One complete recorded conversation through production retrieval, comparing matched answer evidence and context tokens with the switch off/on. If extra context displaces better evidence or the asserted-edge false-positive count is nonzero, leave the switch off and retain the candidates for diagnosis.

## 6. Look-ahead constraints

G50 owns identity decisions and `difference_kind`; G44 owns unusable entity repair. G76 owns source proposition support and G64 searchable sentence claims. G48 leg attribution requires a real turn batch on every new edge. G70 usage accounting must distinguish candidate discovery from final-prompt inclusion. Later LME oracle and semi-LSREP runs should use the same frozen answerer/context budget for the on/off comparison. There is no reason to implement `unmerge_entities` inside a feature that never merges.

## 7. Traps

- `pending` is eligible for graph traversal under the trust floor; it is not a safe container for guessed edges.
- A model's `link` verdict over names is still an inference over names. Even a perfect enum parse does not supply a supporting episode.
- More connected nodes, a higher degree score, or more fact lines can all be produced by false relations. Judge source support and answer evidence, including token displacement.
- The current store's zero attributed turns makes a write-side qualification vacuous; prepare the code against v3 writer fixtures, then judge it on a properly replayed v3 conversation. Do not backfill speaker roles from a raw delimiter guess.
