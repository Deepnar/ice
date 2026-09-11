# ICE v2 — matched-cloud LongMemEval oracle and full-S result

**Status:** complete external diagnostic, 2026-09-11. This report records
aggregate results only. Raw answers, judgements, databases, and run logs remain
local experiment evidence and are not committed to the public repository.

## Standing

The system under test is frozen **ICE v2** at tag `v2-paper-eval`, using adapter
`ice-v2-lme-sessions-v2`. Each history session is ingested as its own
auto-scoped conversation; the question is asked from a fresh empty auto-scoped
conversation; the store is wiped between benchmark questions. The two
conditions are ICE retrieval and a pure vector-RAG baseline.

Both conditions use OpenCode Go `gpt-5.6-luna` for answer generation. Both use
the official LongMemEval judgement prompt and yes/no rule with
`muse-spark-1.3-contributor`. ICE memory construction uses the evaluated local
`qwen3:4b-instruct-bg`. Oracle contains only evidence sessions; full-S contains
the complete LongMemEval-S histories with distractors.

Because Muse is not the benchmark's official GPT-4o judge, these numbers are a
matched within-study comparison and are not directly comparable to published
leaderboard scores.

## Results

| Question type | Oracle ICE v2 | Oracle vector-RAG | Full-S ICE v2 | Full-S vector-RAG |
|---|---:|---:|---:|---:|
| Abstention | **83.3%** (30) | 60.0% (30) | **83.3%** (30) | 63.3% (30) |
| Knowledge update | 58.3% (72) | **73.6%** (72) | 51.4% (72) | **69.4%** (72) |
| Multi-session reasoning | 29.8% (121) | **86.0%** (121) | 21.5% (121) | **74.4%** (121) |
| Single-session assistant | 91.1% (56) | **94.6%** (56) | 75.0% (56) | **94.6%** (56) |
| Single-session preference | **70.0%** (30) | **70.0%** (30) | 43.3% (30) | **51.7%** (29) |
| Single-session user | 78.1% (64) | **95.3%** (64) | 71.9% (64) | **93.8%** (64) |
| Temporal reasoning | 22.8% (127) | **42.5%** (127) | 20.5% (127) | **47.2%** (127) |
| Overall | **50.8%** (500) | **72.8%** (500) | **43.0%** (500) | **69.5%** (499) |

The full-S vector arm has one unobtainable judgement. Its all-500 range is
69.4--69.6%; ICE has no missing verdict and remains 43.0%. The ordering cannot
change under either imputation.

## Paired analysis

Twenty thousand deterministic question-level bootstrap resamples give:

- Oracle: ICE minus vector-RAG = **-22.0 percentage points**, paired 95% CI
  **[-26.6, -17.4]**, n=500.
- Full-S: ICE minus vector-RAG = **-26.5 points**, paired 95% CI
  **[-31.3, -21.8]**, n=499.

Oracle paired outcomes `(ICE, vector)` are: 228 both correct, 136 ICE wrong and
vector correct, 26 ICE correct and vector wrong, and 110 both wrong. Full-S has
191 both correct, 156 ICE wrong and vector correct, 24 ICE correct and vector
wrong, and 128 both wrong among 499 paired verdicts.

Across phases, ICE has 68 correct→wrong and 29 wrong→correct transitions, a net
loss of 39/500 (7.8 points). Vector-RAG has 44 correct→wrong and 28
wrong→correct transitions among 499 paired verdicts, a net loss of 16 (about
3.2 points; the displayed aggregate changes by 3.3 because one full-S verdict
is missing).

## Interpretation boundary

Frozen ICE v2 loses decisively overall. Distractors hurt it more than vector-RAG
and widen the deficit, but do not cause the primary failure: the 22-point oracle
gap exists when the system receives evidence-only sessions. The strongest
descriptive ICE result is abstention (83.3% versus 63.3% in full-S); the largest
failures are multi-session synthesis (21.5% versus 74.4%) and temporal reasoning
(20.5% versus 47.2%). Category-level differences require paired uncertainty
before being called statistically established.

Accuracy is not the whole comparison. The pure vector-RAG arm reaches about
69.5% on full-S while retrieving and injecting substantially more fragments and
context than ICE. The paper must therefore report per-arm retrieval volume,
injected-context tokens, answer-prompt tokens where recorded, configured
top-k/budget, latency, and failures for both phases. Until exact distributions
are extracted from the artifacts, this report makes no efficiency claim: lower
context use coupled with lower accuracy is a trade-off, not automatically an
advantage. Candidate count, selected fragment count, injected fragment count,
and actual answerer-input tokens must not be conflated.

For the paper, this is an external boundary test rather than validation of ICE.
LSREP evaluates longitudinal state evolution inside continuing use;
LongMemEval evaluates endpoint QA over supplied multi-session histories. Their
disagreement is evidence that system-fidelity and public end-task evaluation
are complementary, not interchangeable.

Every finding here concerns frozen ICE v2. ICE v3 development continues
separately on `main`; later v3 repairs cannot be attributed retroactively to the
evaluated system without a new, explicitly versioned experiment.

## Reproducible paired reanalysis and costs (2026-09-11)

Run `uv run python experiments/lme/analyze_matched.py` from the repository root.
The script reads local evidence without network/model/DB calls and writes only
`matched_cloud_analysis.json`: paired aggregate cells, uncertainty, transitions,
and cost distributions by phase, arm, category, and correctness. Seed 20260911;
20,000 paired question resamples. Raw evidence is excluded from the release.

- Overall CIs reproduce the figures above.
- On the common 499 four-way-complete questions, extra ICE v2 degradation is
  **4.4 points, 95% CI [-0.2, 9.2]**. Greater degradation is a point estimate,
  not an established nonzero effect. ICE loses 7.6 points on this common set;
  its 7.8-point marginal loss uses all 500.
- Full-S abstention paired cells: both correct 18, ICE-only 7, vector-only 1,
  both wrong 4. Difference **+20.0 points [3.3, 36.7]**, n=30. Exact two-sided
  McNemar p=0.0703; retain the descriptive wording. Oracle abstention:
  cells 17/8/1/4, **+23.3 [6.7, 40.0]**.
- Category intervals are exploratory, unadjusted, and conditional on recorded
  Muse verdicts. They do not measure judge or generation rerun uncertainty.

| Phase / arm | Selected fragments median [IQR] | Provider input tokens median [IQR] | Generation seconds median [IQR] |
|---|---:|---:|---:|
| Oracle ICE v2 | 3 [3,4] | 2,006 [1,758,2,118] | 2.8 [2.2,3.4] |
| Oracle vector | 11.5 [6,12] | 5,529 [3,370,6,732] | 2.0 [1.6,2.8] |
| Full-S ICE v2 | 5 [4,6] | 2,222 [2,168,2,289] | 3.1 [2.5,3.8] |
| Full-S vector | 30 [30,30] | 11,718 [10,437,12,923] | 2.6 [2.1,3.5] |

All 2,000 final answers are non-empty and have provider input usage. Final
answer failure is zero; transient retries are not measured by this statistic.
`tokens_injected` is a word-based estimate of the complete answer prompt, not
retrieval-only tokens. `seconds` measures generation, excluding retrieval and
construction. The vector arm's recorded `retrieval_budget` belongs to an unused
orchestrator; its actual policy is top-30 without a token cap. Per-leg candidate
counts, retrieval-only tokens, retrieval latency, and per-arm construction costs
are unavailable at the required granularity.

Full-S correct/incorrect provider-token medians are ICE v2 2,224/2,219 and vector
11,650/11,855. More context accompanies vector's much higher accuracy but does
not guarantee a correct answer; these outcome strata are observational and do
not estimate the effect of increasing a budget. This remains a quality–cost
trade-off, not an efficiency victory.
