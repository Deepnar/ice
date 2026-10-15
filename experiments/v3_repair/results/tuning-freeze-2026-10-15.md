# ICE v3 bounded read tuning — results and freeze (2026-10-15)

**Verdict: no read coordinate beats the current defaults; the configuration is
frozen unchanged.** Diagnostic development tuning, not a score of record.

## Setup

- Development history: 251 pairs, 67 reviewed questions (all older-source),
  completed repaired producer `logs/v3-repaired-dev-tuning-2026-10-10-r1`, reader
  profile r2 (`cce3f90b…`). Conversation fold withheld (frozen v1 summary proofs).
- Baseline settings digest `a03cae64…` (repository defaults at commit `ec909b2`).
- Answerer GPT 6 Luna (`opencode-luna6`, 1,500-token output reserve); judge Muse
  Spark 1.3 (`opencode-muse13`), independent absolute grades with the 67 editorial
  content contracts. Background writers gemma4:e4b and NuExtract3 (frozen state).
- Read passes: four-arm baseline; 30 sealed full-arm coordinates; 8 leg-profile
  supplement rows; repeated baseline (noise floor).

## Prompt-stage screen

- Budget caps change every prompt; 4k drops 56/101 gold sources. Budget is not
  swept: the derived budget (0.75 × model window, clamped 4k–40k) stays
  (maintainer).
- Base leg weights reach no decision (0/67 questions blend them; TRAPS 128).
- Fusion constant and candidate limits move gold sources by ±1–2 of 101.
- Reranker off loses 6 gold sources; reranker shortlist 32 loses 7.

## Answer stage (Muse, correct of 67)

| Arm | Correct | Net vs baseline (paired) | Median tokens |
|---|---:|---|---|
| Baseline full ICE | 61 (+4 partial, 1 incorrect, 1 ungraded) | — | 16,929 |
| Baseline re-asked (×2) | 58, 58 | −3 (+2/−5), −3 (+1/−4) | same |
| Recent share ×0.75 | 62 | 0 | −7% |
| Recent share ×1.25 | 58 | −3 | +9% |
| Vector candidates 200 | 61 | −1 | +1% |
| Claim candidates 128 | 61 | −1 | +1% |
| Rerank shortlist 128 | 62 | +1 | +1% |
| Leg profiles vector ×1.5 | 61 | 0 | +2% |
| Leg profiles codex ×1.5 | 61 | 0 | +2% |
| Control: no Codex | 63 | +2 | +2% |
| Control: ICE vector-only | 63 | +2 | +1% |
| Control: recent-only | 41 | −20 | −66% |

**Noise floor:** answering the identical baseline prompt again flips up to 7
questions, net ±3. Every candidate falls within it; none saves ≥10% tokens at
equal quality. Pre-registered rule ⇒ keep every current value.

## What this run can and cannot say

- The question set is near ceiling (61/67 = 91% for full ICE), so it has little
  power to separate read settings; most differences are answer/judge noise.
- Memory clearly matters (full ICE 61 vs recent-only 41), but on this history the
  graph and ICE's extra legs show no answer benefit over ICE vector-only (63) or
  no-Codex (63), consistent with the bare-vector span result (PROVENANCE
  2026-10-14). Development evidence only; Z2 (harder question types, B and C,
  bare vector answered) decides.
- Interaction checks around load-bearing knobs: skipped — no knob was
  load-bearing.
- Not tuned by decision: budget, lifecycle/decay, trust thresholds, gates/style.

## Freeze

Frozen for Z2: the repository settings at digest `a03cae64…` unchanged, plus the
Z2 answer reserve change to 4,000 tokens (Z2_EVALUATION D7). Muse is the judge.
