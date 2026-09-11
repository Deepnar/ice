# Paper — remaining submission pass

**Updated 2026-09-11 after the matched-cloud ICE-v2 LongMemEval runs.** The
canonical manuscript is `ICE_paper_v2.tex`; `ICE_paper_tist.tex` remains the
frozen record of the rejected submission and must not be edited.

## Version boundary — preserve the ongoing v3 programme

This manuscript reports **frozen ICE v2** at tag `v2-paper-eval`. Real ICE
product and research development continues independently on **v3**, which is
`main`. The paper-writing session must preserve that separation:

- Do not change `src/`, the v3 roadmap, current architecture documentation, or
  the project-level `docs/HANDOFF.md` as part of rewriting this paper.
- Do not use a v3 fix, feature, configuration, or expected behavior to explain
  what v2 did. Verify architectural statements against the frozen tag and
  `docs/ICE_Architecture[real_v2].md`.
- A weakness found in v2 may motivate future work, but remains a measured v2
  limitation unless a separately identified v3 experiment evaluates a repair.
- Keep v2 and v3 names on every number and behavioral claim. Storing the paper
  on `main` does not make `main` the evaluated system.
- Keep paper-only plans in this file or the gitignored live session file, not in
  the v3 roadmap or handoff unless they genuinely change the v3 programme.

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

The primary external diagnostic is now the completed, matched cloud run. System:
frozen ICE v2 at `v2-paper-eval`; adapter `ice-v2-lme-sessions-v2`; 500 questions
in each phase. Both ICE and vector-RAG answers use OpenCode Go
`gpt-5.6-luna`; both are judged with the official LongMemEval prompt and yes/no
rule through `muse-spark-1.3-contributor`. Background memory construction uses
the evaluated local `qwen3:4b-instruct-bg`. The non-official Muse judge makes
this a controlled within-study comparison, **not** a score directly comparable
to GPT-4o-judged leaderboard results.

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

Paired question-level inference is required in the paper, not only point
estimates. In the oracle, ICE trails vector-RAG by 22.0 percentage points with a
20,000-resample paired bootstrap 95% CI of [-26.6, -17.4] (n=500). In full-S,
ICE trails by 26.5 points with CI [-31.3, -21.8] over the 499 questions with both
verdicts. One full-S vector judgement is excluded; all-500 bounds are ICE
43.0--43.0% and vector-RAG 69.4--69.6%, so the ordering is robust.

Full-S distractors reduce ICE by 7.8 points: 68 oracle-correct answers become
wrong while 29 oracle-wrong answers become correct. Vector-RAG loses about 3.3
points (44 correct→wrong, 28 wrong→correct among 499 paired verdicts). The
distractor phase therefore widens ICE's deficit, but it is not the root cause:
ICE already trails by 22 points in the evidence-only oracle.

The result is negative overall but not featureless. ICE's clear descriptive
strength is abstention (83.3% versus 63.3% in full-S); its decisive failures are
multi-session synthesis (21.5% versus 74.4%) and temporal reasoning (20.5%
versus 47.2%). Do not call the abstention difference statistically established
until its paired uncertainty is computed. Do not frame LongMemEval as validating
ICE. Frame it as an external boundary test showing that LSREP's continuing-use
results do not imply strong fresh-session aggregation or temporal QA.

The old local-Gemma oracle is historical diagnostic evidence only. Never mix
its scores with the matched cloud table.

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
7. **Integrate the completed matched public benchmark.** Report oracle and S
   side by side with exact answerer/judge/background identities, paired
   bootstrap confidence intervals, the one-judgement failure bound, and
   oracle→S correctness transitions. Preserve the local-Gemma oracle only as a
   historical diagnostic. Do not claim leaderboard comparability.
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
11. **Keep the vector baseline visible and contextualise other memory systems
    honestly.** The pure vector-RAG arm is the matched, same-answerer baseline
    and must appear in the main results, not be hidden in an appendix. Add a
    retrieval-cost table beside accuracy: fragments retrieved and injected,
    retrieved-context tokens, total answer-prompt tokens where available,
    configured top-k/budget, latency, and failure rate for each arm and phase.
    The vector arm's roughly 69% full-S accuracy is meaningful, but it was
    achieved by admitting substantially more fragments and tokens than ICE;
    describe it as a strong high-context baseline, not as an efficiency-neutral
    comparator. Extract exact distributions from the artifacts before writing
    this claim: report median and spread, not only totals or configuration caps.
    Distinguish candidates retrieved, fragments surviving selection, fragments
    injected, and tokens actually presented to the answerer.

    Call ICE **token-efficient** only under a declared quality constraint or an
    explicit accuracy--token frontier. Defensible measures include tokens per
    correct answer, accuracy at matched token budgets, or Pareto dominance
    across budgets. “ICE used fewer tokens” alone establishes context economy,
    not efficiency, when accuracy is substantially lower. Without matched-budget
    reruns, report the observed quality--cost trade-off without declaring an
    efficiency winner.

    Add a
    compact table of published LongMemEval memory systems only as external
    context, clearly separating their official datasets, answerers, judges,
    prompts, and retrieval budgets. Published GPT-4o-judged scores are not
    head-to-head comparisons with the Muse-judged ICE run. A defensible direct
    comparison requires rerunning those systems under this matched stack or
    rejudging ICE and vector-RAG with the official judge.
12. **Compute uncertainty beyond the overall score.** Add paired bootstrap CIs
    for oracle, full-S, and the oracle→S degradation difference. For category
    claims—especially abstention—report paired counts and uncertainty or label
    them descriptive because n=30 is small. Bootstrap the question, preserving
    the ICE/vector pair; do not bootstrap arms independently.
13. **Analyse quality jointly with retrieval cost.** For both oracle and full-S,
    compare accuracy against fragment count, injected tokens, answer-prompt
    length, and latency. Stratify these quantities by question type and by
    correct/incorrect outcome so the paper can distinguish “ICE retrieved too
    little,” “vector succeeded by supplying much more context,” and “more
    context still failed.” Do not infer causality from correlation, and do not
    call ICE efficient merely because it injected less context while losing on
    accuracy. If answerer usage metadata is incomplete, state exactly which
    token quantity is measured and which is unavailable.

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
