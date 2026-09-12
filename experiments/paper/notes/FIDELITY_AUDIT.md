# Frozen ICE v2 fidelity audit — final paper account

Updated 2026-09-12 against `v2-paper-eval`, currently resolving to
`00d3d35eee99843fd12790d2bf704c177f3097d1`. Historical pre-rewrite hashes are
not current run identities. The frozen architecture report remains unchanged;
code at the tag and recorded evaluation outputs govern this audit. Nothing here
credits a v3 repair to v2.

## Execution is not utility

The six defined retrieval legs did not all contribute. Recorded source labels
are episodic 62.3%, unattributable 34.5%, and graph 3.3%; these are fragment
counts, not token or evidence shares. A whole graph traversal is concatenated
into one fragment, so its fraction establishes neither a lower bound on content
share nor graph quality. The trained NER checkpoint was loaded; the records do
not establish task-specific NER recall.

| Frozen-v2 component | Evidence and allowable interpretation |
|---|---|
| Vector retrieval | Typed pgvector bind; executed decay-weighted episodic retrieval. |
| Lexical retrieval | Executed PostgreSQL `ts_rank` search, historically called BM25; not the canonical BM25 formula. |
| RRF | Adding fusion after the lexical leg recovers +0.82 [0.39,1.24] score points in the 67-probe buildup. Scoped result, not a guarantee that adding legs is safe. |
| Graph | Produced output; one concatenated traversal fragment and extraction limitations prevent attributing answer-quality benefit. |
| Procedural | Untyped vector bind raises an operator error and falls back to empty output; defective. |
| Documents | Same bind defect plus no ingested documents; defective and unexercised. |
| Batch summaries / archival paths | No eligible content in the evaluated workload; no quality claim. |
| HyDE | Disabled in the mature run; active but constant across nominal ablation steps, so its toggle did not isolate it. |
| Budget | Executed selection; density contrast compares full ICE with unbudgeted vector. Its individual causal effect is not isolated. Paired buildup step −0.08 [−0.36,0.18]. |
| Clusters, session diversity, bonuses | Executed transforms; small or uncertain score contrasts are not formal equivalence. Foreign-session caps were rarely exercised in continuing-use LSREP. |
| Routing | Ordinary-density ICE MoE–generalist +0.03, vector −0.02; small observed effects. |
| Decay / reinforcement | Declared maintenance and probe accesses change state. This reconstructed trajectory is not an upper or lower bound on deployment. |
| Temporal intervals | Stored current/superseded graph status is inspectable; this does not prove historical-query or temporal-composition success. |

The normal retrieval legs execute sequentially. Budget diversity operates over
source types: lexical and vector records can both be episodic. Raw turns are
stored before the lossless decision; flags choose later representation. The
HTTP wrapper's secondary word check omits retrieved and recent messages and is
not a complete prompt-size guarantee. Evaluated adapters call selection and
assembly directly, without that faulty wrapper check.

## What the comparisons establish

The LSREP full-system comparison intentionally includes ICE's assembler, slots,
and recent window; vector uses top-30 retrieval plus its own simpler prompt.
This is a valid comparison of complete configurations. It does not isolate
retrieval from assembly or compare accuracy at equal context budgets.
Ordinary-density ICE selects 32% fewer fragments but uses 6.6% more estimated
prompt tokens. Its mean-score difference is +0.002 with a probe-cluster interval
[−0.148,0.158], not an equivalence finding or an efficiency victory.

All-data benefit is largely a density-reliability result against the unbudgeted
baseline. Removing failed-answer pairs changes the estimand and removes most of
that benefit. Manual-replacement and ordinal sensitivities are reported in the
paper and aggregate reports. No positive correlation proves a causal curation effect.

The matched public diagnostic decisively favours vector overall: oracle
50.8/72.8% and full-S 43.0/69.5% for ICE/vector. ICE's smaller input volume is a
quality–cost trade-off. Abstention is descriptive; multi-session and temporal
composition fail badly. The 22-point oracle gap precedes distractors.

## Reproduction

- `experiments/mature/exp2_bootstrap.py`: archived manual merge and missing-score policy.
- `experiments/mature/clustered_sensitivity.py`: whole-probe trajectory resampling.
- `experiments/mature/scoring_sensitivity.py`: manual-source, complete-case and ordinal checks.
- `experiments/flaw_ablation/buildup/exp3_bootstrap.py`: paired buildup contrasts, reused consistently across report views.
- `experiments/lme/analyze_matched.py`: paired arms, categories, phase transitions and cost distributions.

See [ARTIFACTS.md](../ARTIFACTS.md) for commands and private-data exclusions,
[CLAIM_AUDIT.md](../CLAIM_AUDIT.md) for citation and frozen-code checks, and the
canonical manuscript for the full architectural and experimental account.
