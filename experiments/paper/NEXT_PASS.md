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

**Subsequent decision, 2026-09-03:** the complete LongMemEval-S run is reopened.
Because it will use a cloud answerer/judge and a vLLM-served Qwen 4B background
model, the oracle must also be rerun under that identical stack. Do not combine
the current local oracle with the future cloud S result or rewrite the paper
before both matched phases finish. `docs/PUBLISHING.md` owns the run design.

The first flattened-session adapter and its scores are invalid. The admissible
run gives each supplied history session its own auto-scoped conversation and
asks from a fresh empty auto-scoped conversation, after a full store wipe per
question. Exact run identity and invalidation details live in
`experiments/lme/results/oracle_adapter_v2.md`.

## Still required before submission

1. **Build the ARR twin without changing the archive.** The current manuscript
   remains a 37-page archival version. The primary target is the 2026-10-12 ARR
   cycle, whose main-content limit is eight pages. Create a separate anonymous
   venue file; do not mutilate or rename the stable archival PDF.
2. **Make the body genuinely LSREP-first.** Reorder to Introduction → Related
   Work → LSREP → dataset/protocol instantiation → compact ICE-v2 case study →
   LSREP and LongMemEval results → fidelity audit/limitations/conclusion. The
   current title and front matter are LSREP-first, but Architecture still
   precedes the protocol and most result space is organised around ICE.
3. **Make LSREP concrete in the main text.** Add one compact protocol algorithm
   and one synthetic worked example: a fact at checkpoint T1, its revision at
   T2, and the corresponding evolving reference answer. No private raw example.
4. **Make the AE-visible evidence self-contained.** Move a compact four-dataset
   result table into the main paper. The full per-conversation table may remain
   in the appendix, but a reader must not need the appendix to see that all four
   datasets were evaluated.
5. **Compress ICE to a case-study description.** Keep roughly one main-text page
   covering only the lifecycle, stores, retrieval/fusion, and budget needed to
   interpret LSREP. Add a compact retrieval algorithm or parameter table; move
   schemas, workers, full configuration, pilot details, and full ablations after
   the references.
6. **Perform a claim-to-evidence pass.** Check every strong comparative claim,
   especially system-specific statements in Related Work. Prefer narrower
   wording where the cited paper does not run the claimed regime. Verify
   operational numbers such as latency and model recall too.
7. **Integrate the matched public benchmark only after it finishes.** Report the
   new oracle and S phases side by side, with exact answerer/judge/background
   identities, per-type scores, failure bounds, and deviations. Preserve the
   current local oracle as a separate historical diagnostic. Do not claim
   leaderboard comparability unless the official judge/protocol are used.
8. **Decide the public artifact package.** Publish the aggregate LongMemEval
   report and adapter/harness, but do not casually commit roughly 1,500 raw
   current answer and judgement files—or the larger matched run that replaces
   them. Preserve raw evidence locally until an artifact archive/release bundle
   is chosen.
9. **Re-check tables and appendix duplication after venue compression.** The
   main paper should carry the argument and headline numbers; reproducibility
   detail can move to an appendix or artifact report according to the venue.
10. **Repair the stale citation-checker note.** Its header still says network
   verification is unavailable even though arXiv/CrossRef checks succeeded in
   the TIST-repair session.

## Venue direction

TIST is final-reject/no-appeal. **Primary target: ARR 2026-10-12**, aiming to
commit suitable reviews to NAACL 2027 or COLING 2027. **Fallback: IP&M** if the
eight-page paper or matched LongMemEval run is not genuinely ready; do not rush
ICLR 2027's September deadline. The paper is a protocol, fidelity-audit, and
boundary-finding paper—not a state-of-the-art LongMemEval system paper.
