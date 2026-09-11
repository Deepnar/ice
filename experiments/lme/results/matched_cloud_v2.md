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

For the paper, this is an external boundary test rather than validation of ICE.
LSREP evaluates longitudinal state evolution inside continuing use;
LongMemEval evaluates endpoint QA over supplied multi-session histories. Their
disagreement is evidence that system-fidelity and public end-task evaluation
are complementary, not interchangeable.
