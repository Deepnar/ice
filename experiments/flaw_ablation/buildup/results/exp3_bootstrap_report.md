# Experiment 3 — Bootstrap CIs (paired, B=10,000, seed=42)

N = 67 probes. Deltas are paired within-probe; CI is 95% percentile.

## Step deltas (feature vs previous step)

| Step | Δ (point) | 95% CI | n | excl. 0 |
|------|----------:|:------:|--:|:-------:|
| add_bm25 | -0.74 | [-1.14, -0.36] | 66 | **yes** |
| add_rrf | +0.82 | [+0.39, +1.24] | 66 | **yes** |
| add_hyde | +0.03 | [-0.13, +0.21] | 67 | no |
| add_cluster_restrict | +0.01 | [-0.16, +0.21] | 67 | no |
| add_session_diversify | +0.00 | [-0.22, +0.22] | 67 | no |
| add_codex | +0.03 | [-0.21, +0.25] | 67 | no |
| add_mera | -0.21 | [-0.43, +0.01] | 67 | no |
| add_procedural | +0.16 | [-0.07, +0.40] | 67 | no |
| add_batch_summary | -0.02 | [-0.24, +0.23] | 66 | no |
| add_dynamic_budget | -0.08 | [-0.36, +0.18] | 66 | no |
| add_sliding_window | -0.02 | [-0.24, +0.23] | 66 | no |
| add_keyword_boost | +0.08 | [-0.22, +0.37] | 65 | no |
| full_ice | -0.03 | [-0.29, +0.23] | 65 | no |

## Cumulative deltas (vs bare_vector)

| Condition | Cum Δ | 95% CI | excl. 0 |
|-----------|------:|:------:|:-------:|
| add_bm25 | -0.74 | [-1.12, -0.36] | **yes** |
| add_rrf | +0.09 | [-0.18, +0.36] | no |
| add_hyde | +0.12 | [-0.15, +0.39] | no |
| add_cluster_restrict | +0.13 | [-0.13, +0.39] | no |
| add_session_diversify | +0.13 | [-0.16, +0.43] | no |
| add_codex | +0.16 | [-0.10, +0.43] | no |
| add_mera | -0.04 | [-0.37, +0.27] | no |
| add_procedural | +0.12 | [-0.15, +0.39] | no |
| add_batch_summary | +0.11 | [-0.21, +0.44] | no |
| add_dynamic_budget | +0.03 | [-0.25, +0.31] | no |
| add_sliding_window | +0.03 | [-0.26, +0.33] | no |
| add_keyword_boost | +0.17 | [-0.21, +0.53] | no |
| full_ice | +0.09 | [-0.24, +0.42] | no |

## Headline contrasts

| Contrast | Δ | 95% CI | n | excl. 0 |
|----------|--:|:------:|--:|:-------:|
| bm25_damage (add_bm25 - bare_vector) | -0.74 | [-1.12, -0.36] | 66 | **yes** |
| rrf_rescue (add_rrf - add_bm25) | +0.82 | [+0.39, +1.23] | 66 | **yes** |
| rrf_vs_bare (add_rrf - bare_vector) | +0.09 | [-0.18, +0.36] | 67 | no |
| full_ice_vs_bare (full_ice - bare_vector) | +0.09 | [-0.24, +0.42] | 66 | no |
| full_ice_vs_vecbaseline (full_ice - vector_baseline) | -0.06 | [-0.40, +0.27] | 63 | no |
