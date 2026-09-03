# Paper — remaining submission pass

**Updated 2026-09-03 after the corrected ICE-v2 LongMemEval oracle run.** The
canonical manuscript is `ICE_paper_v2.tex`; `ICE_paper_tist.tex` remains the
frozen record of the rejected submission and must not be edited.

## Completed in the post-TIST repair

- The paper is framed protocol-first: LSREP is the first contribution and ICE
  v2 is the system under test.
- The abstract and introduction were compressed, the absolute forgetting claim
  was removed, ChatGPT-style cross-session memory is acknowledged, and the
  introduction now cites long-context and multi-session work.
- Related Work now follows a stability-of-knowledge argument rather than a list
  of systems. LoCoMo, LongMemEval, Zep, HippoRAG, Self-RAG, and LLM-judge work
  are cited.
- The private LSREP datasets now have turn, token, probe, checkpoint, and
  simulated-horizon counts, plus provenance and release-boundary text.
- Claims that replay, corpus difficulty, or an under-implemented graph produce
  a guaranteed lower bound were removed. Their direction is not identifiable.
- The complete 500-question LongMemEval evidence-only oracle is reported as a
  mixed external diagnostic, including missing-judgement bounds, adapter
  validation, and the reason the distractor-heavy phase stopped.

## LongMemEval result that must not drift

System: frozen ICE v2 at `v2-paper-eval`, adapter
`ice-v2-lme-sessions-v2`. Judge: official LongMemEval prompts and yes/no rule,
but local Ollama `gemma4:12b`, so the numbers are not comparable to the
GPT-4o-judged leaderboard.

| Question type | ICE v2 | Vector-RAG |
|---|---:|---:|
| Abstention | **96.6%** (29) | 93.3% (30) |
| Knowledge update | 69.2% (65) | **76.1%** (71) |
| Multi-session reasoning | 28.3% (120) | **86.6%** (119) |
| Single-session assistant | 89.3% (56) | **100.0%** (56) |
| Single-session preference | **89.7%** (29) | 82.8% (29) |
| Single-session user | 86.9% (61) | **100.0%** (62) |
| Temporal reasoning | 23.7% (118) | **52.1%** (117) |
| Overall, obtainable verdicts | **55.2%** (264/478) | **80.2%** (388/484) |
| All-500 bound | **52.8–57.2%** | **77.6–80.8%** |

This is mixed, not uniformly negative: ICE has higher point estimates on
abstention and preference. It nevertheless loses robustly overall, with the
largest failures on multi-session and temporal reasoning. Do not call the two
category point estimates statistically established wins.

LongMemEval-S was **not run**. The evidence-only oracle was the predeclared gate
for the expensive distractor phase. The oracle is not a mathematical upper
bound, because adding candidates can change retrieval non-monotonically; the
stopped phase is a compute decision, not a claim about every unrun answer.

The first flattened-session adapter and its scores are invalid. The admissible
run gives each supplied history session its own auto-scoped conversation and
asks from a fresh empty auto-scoped conversation, after a full store wipe per
question. Exact run identity and invalidation details live in
`experiments/lme/results/oracle_adapter_v2.md`.

## Still required before submission

1. **Choose the venue and enforce its format.** The current manuscript remains
   a long archival version. A conference submission will require a separate
   venue-formatted cut; do not mutilate the stable archival PDF before the page
   and appendix policy is known.
2. **Perform a claim-to-evidence pass.** Check every strong comparative claim,
   especially system-specific statements in Related Work. Prefer narrower
   wording where the cited paper does not run the claimed regime.
3. **Decide the public artifact package.** Publish the aggregate LongMemEval
   report and adapter/harness, but do not casually commit roughly 1,500 raw
   answer and judgement files. Preserve them locally as evidence until an
   artifact archive or release bundle is chosen.
4. **Re-check tables and appendix duplication after venue compression.** The
   main paper should carry the argument and headline numbers; reproducibility
   detail can move to an appendix or artifact report according to the venue.
5. **Repair the stale citation-checker note.** Its header still says network
   verification is unavailable even though arXiv/CrossRef checks succeeded in
   the TIST-repair session.

## Venue direction

TIST is final-reject/no-appeal. The existing journal order is IP&M, then
Information Retrieval Journal; conferences are now allowed too. Re-select only
after checking current calls, page limits, deadlines, and fit. The negative
oracle result makes this primarily a protocol, fidelity-audit, and honest
boundary-finding paper—not a state-of-the-art LongMemEval system paper.
