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

**Subsequent decision, updated 2026-09-04:** the complete LongMemEval-S run is
reopened. Because it uses a Luna cloud answerer and Muse cloud judge, the oracle
must also be rerun under that identical stack. Background construction remains
the exact evaluated Ollama `qwen3:4b-instruct-bg`; a vLLM AWQ substitute failed
the grounded direction control and was rejected. Do not combine
the current local oracle with the future cloud S result or rewrite the paper
before both matched phases finish. `docs/PUBLISHING.md` owns the run design.

The first flattened-session adapter and its scores are invalid. The admissible
run gives each supplied history session its own auto-scoped conversation and
asks from a fresh empty auto-scoped conversation, after a full store wipe per
question. Exact run identity and invalidation details live in
`experiments/lme/results/oracle_adapter_v2.md`.

## Framing decision: one evaluation-led systems paper

The paper must not become either an ICE architecture paper with LSREP attached,
or a supposed comparison between LSREP and LongMemEval. Keep one paper, with
three distinct roles:

| Element | Role in the paper | Question it answers |
|---|---|---|
| **LSREP** | Primary methodological contribution | How should an evolving deployed memory state be reconstructed and evaluated repeatedly over time? |
| **ICE v2** | Architectural contribution and audited system case study | What does a local-first, multi-store memory system do under that protocol, and which mechanisms actually carry its result? |
| **LongMemEval** | Complementary public transfer diagnostic | Do the conclusions transfer to supplied multi-session histories, fresh-session aggregation, and distractors? |

LSREP and LongMemEval therefore must not have their scores compared as if they
were two competing benchmarks over the same construct. Their corpora, query
placement, state lifecycle, answerers, and judging conditions differ. Compare
their *evaluation dimensions* and interpret disagreement as a scope boundary:
LSREP measures longitudinal state evolution inside continuing use;
LongMemEval measures endpoint question answering over externally supplied
histories. Use “complementary regimes,” “triangulation,” and “transfer
diagnostic,” not “LSREP outperforms/replaces LongMemEval.”

ICE must remain visible as more than a disposable system under test. Preserve
the lifecycle, typed stores, retrieval/fusion, dynamic budget, local-first
constraint, and component-fidelity audit in the main argument. At the same
time, claim only the mechanisms the audit shows were live: v2's weak or
unexercised graph, procedural, and cross-conversation paths are findings, not
architectural wins. A later repaired-v3 systems paper can test those mechanisms
as system contributions; this paper reports the frozen v2 honestly.

Organise results by research question rather than by artifact:

1. **RQ1 — longitudinal behaviour:** Under LSREP, does ICE preserve answer
   quality while controlling retrieved context as state accumulates and ages?
2. **RQ2 — mechanism fidelity:** Which ICE components were live, defective,
   unexercised, corrective, or neutral, and what actually carried the result?
3. **RQ3 — external transfer:** Under matched LongMemEval oracle and S runs,
   which within-conversation findings survive fresh-session aggregation and
   distractors?

Do not split the current work into an LSREP-only paper and an ICE-only paper.
LSREP currently has one private, single-user instantiation and needs the audited
system case study; ICE's strongest contribution is precisely that its apparent
success can be decomposed and bounded by LSREP plus the public diagnostic. The
combination is stronger than either half at present. Preserve the 37-page
canonical report as the full archival account, then express this same argument
in a separate venue-limited paper rather than deleting ICE detail from the
archive.

## Still required before submission

1. **Build the ARR twin without changing the archive.** The current manuscript
   remains a 37-page archival version. The primary target is the 2026-10-12 ARR
   cycle, whose main-content limit is eight pages. Create a separate anonymous
   venue file; do not mutilate or rename the stable archival PDF.
2. **Make the body genuinely evaluation-led.** Reorder to Introduction →
   Related Work → LSREP → compact ICE-v2 architecture and audit contract →
   dataset/protocol instantiation → results organised by RQ1/RQ2/RQ3 →
   limitations/conclusion. The current title and front matter are LSREP-first,
   but Architecture still precedes the protocol and most result space is
   organised around ICE. “Evaluation-led” keeps LSREP primary without erasing
   the system contribution.
3. **Make LSREP concrete in the main text.** Add one compact protocol algorithm
   and one synthetic worked example: a fact at checkpoint T1, its revision at
   T2, and the corresponding evolving reference answer. No private raw example.
4. **Make the AE-visible evidence self-contained.** Move a compact four-dataset
   result table into the main paper. The full per-conversation table may remain
   in the appendix, but a reader must not need the appendix to see that all four
   datasets were evaluated.
5. **Compress ICE without demoting it.** Keep one dense main-text section
   (roughly 1--1.5 pages in an eight-page venue version) covering the lifecycle,
   typed stores, retrieval/fusion, budget, and auditability needed to interpret
   the experiments. Retain one compact architecture figure plus a retrieval
   algorithm or parameter table; move schemas, worker implementation, full
   configuration, pilot details, and complete ablations after the references.
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

**Venue decision still open:** reconsidering a TMLR resubmission is an explicit
option, but its current resubmission policy, fit, and the value of returning
there versus ARR must be checked before choosing it. No TMLR decision has been
made yet.
