# LongMemEval oracle — ICE v2, corrected adapter

**Status:** generation 500/500; judgement files 1,000/1,000; disk-grounded
report after the interrupted mute retry. **Do not substitute the archived
flattened-session run.**

## Evaluation identity

- System: ICE v2 at `v2-paper-eval`, plus documented run-enablement changes on
  `lme/v2-paper-eval` (reachable shared background route; CUDA execution for the
  same 384-dimensional embedder).
- Dataset: all 500 questions in LongMemEval's public evidence-only oracle file.
- Adapter: `ice-v2-lme-sessions-v2`. The database is wiped per question; each
  supplied history session is a separate auto-scoped conversation; the question
  is asked from a new empty auto-scoped conversation; retrieval is global over
  only that question's histories.
- Answerer: `gemma4:26b-a4b-it-q4_K_M`, shared by ICE and vector-RAG.
- Judge: LongMemEval prompts and yes/no rule, local Ollama `gemma4:12b`, not the
  official GPT-4o judge. Initial cap 1,024; mute retry cap 4,096.
- Missing-data rule: a mute judge response is excluded, never counted wrong;
  all-500 worst/best bounds assign every mute false/true.

## Result

| Question type | ICE v2 | Vector-RAG |
|---|---:|---:|
| Abstention | **96.6%** (29) | 93.3% (30) |
| Knowledge update | 69.2% (65) | **76.1%** (71) |
| Multi-session reasoning | 28.3% (120) | **86.6%** (119) |
| Single-session assistant | 89.3% (56) | **100.0%** (56) |
| Single-session preference | **89.7%** (29) | 82.8% (29) |
| Single-session user | 86.9% (61) | **100.0%** (62) |
| Temporal reasoning | 23.7% (118) | **52.1%** (117) |
| **Overall, spoken verdicts** | **55.2%** (264/478) | **80.2%** (388/484) |
| **All-500 bound** | **52.8–57.2%** | **77.6–80.8%** |

Judge mutes: ICE 22/500 (4.4%); vector-RAG 16/500 (3.2%). The ordering is
robust under every assignment of missing verdicts.

## Interpretation and stopped phase

The result is mixed rather than uniformly negative. ICE has higher point
estimates on abstention and preference, consistent with selective recall and
personal context. Those strengths do not compose: multi-session and temporal
reasoning account for the largest deficits, and vector-RAG wins decisively
overall.

LongMemEval-S was **not run**. The staged plan required the evidence-only oracle
to pass before the distractor-heavy phase. Once the corrected oracle rejected
aggregate fresh-session effectiveness, the full haystack could no longer serve
as positive external validation. The oracle is not a mathematical upper bound,
because retrieval can change non-monotonically when candidates are added; the
decision is a predeclared compute gate, not a claim that every individual S
answer must be worse.

## Adapter invalidation

The first adapter flattened all sessions into one conversation and passed a
UUID current id where fragments carried string ids. A real-path trace measured
45 BM25 and 100 vector candidates, 111 unique after RRF, then an erroneous
collapse to three at session diversification. Changing only the id type
preserved all 111; 31 fit the then-active 14,380-token budget. All answers and
ICE judgements from that adapter were invalidated and archived under
`runs/v2-paper-eval/<phase>/invalidated_adapter_v1/`.

The corrected score above is the only LME-v2 result admissible for the paper.
