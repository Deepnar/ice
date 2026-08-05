# Fidelity Audit — what was actually alive at the paper-eval snapshot

> Purpose: establish the honest boundary of what the paper is *allowed to claim*, by
> classifying every mechanism ICE names as **working / degraded / dead / disabled /
> inactive** at the evaluated snapshot. Grounded in (a) the code at git tag
> `v2-paper-eval` = commit `0521df9` ("ice v2 finished", 2026-07-02) and (b)
> *(the tag pointed at `e4019b6` when this audit was written; the 2026-08-03
> history rewrite re-pointed it, and `e4019b6` no longer resolves. The tree is
> identical (`2990066b`), so every finding below still describes the same code —
> `e4019b6` in the table headers should be read as that snapshot.)* the frozen
> Experiment-2 result JSON (`experiments/mature/results/`, generated 2026-06-29).
> Written 2026-07-20. Companion to `REVISION_PLAN_v2.md` §11.

## 0. The one-line finding

The paper sells a **six-leg orchestrator**. At the snapshot, **three sources produced
every fragment** (episodic 62.3%, unknown/unattributable 34.5%, codex 3.3%) and **three
legs produced literally zero** (procedural, RAG, batch summaries). The metadata inside the
result file itself lists all six legs as active — that gap between *claimed* and *contributing*
is the whole reason for this audit. "Full ICE" was, in effect, **decay-weighted vector +
BM25 + RRF + token budget + post-fusion curation, with a 3.3% codex trickle.**

## 1. Retrieval legs

| Leg | Paper implies | State @ `e4019b6` | Evidence | Claimable? |
|---|---|---|---|---|
| **Vector (decay-wt)** | active, primary | **WORKING** | `_vector_episodic` has `bindparams(type_=PgVector)` (orchestrator.py:613); dominates leg contributions | Yes — real |
| **BM25** | active | **WORKING** | tsvector leg, no embedding bind needed; folded into "episodic" 62.3% | Yes — real |
| **Codex graph** | active, novel KG | **WORKING but under-contributing** | 3.3% / **1 fragment avg**. NOT an NER failure (see below). Real handicaps (ROADMAP l.18–19): 6k-token extraction chunks a 4B model can't reason over (entity-drop + relation hallucination *at graph-build time*), the whole traversal collapsed into **one concatenated fragment** (so 3.3% is a lower bound on content share, not a measure), leg under-representation, and an inert confidence/strength signal with no retrieval reinforcement | As "small content share, lower-bound, structurally under-weighted" — **not** as a validated KG, but **not** as broken entity detection either |
| **Procedural** | active leg | **DEAD** | no `type_=PgVector` bind at snapshot (orchestrator.py:761 → execute at :787) ⇒ `vector <=> double precision[]` ⇒ rollback ⇒ `[]`; intent gate hid it; **0.0 fragments** in Exp 2 | **No** — non-functional |
| **RAG** | active leg | **DEAD + inactive** | same missing bind (:824 → :841); *also* triple-gate rarely fired + no documents ingested; **0.0 fragments** | **No** — non-functional |
| **Batch summaries** | active leg, "second life of a turn" | **INACTIVE (no data)** | bind was *correct* (:864); but nothing decayed far enough to form batches, so `batch_summaries` was empty; **0.0 fragments**, absent from source keys | **No** — never exercised |

**Net:** 2 fully-working legs (vector, BM25) + 1 degraded (codex) + 3 that returned nothing.

## 2. Other named mechanisms

| Mechanism | State @ snapshot | Evidence | Claimable? |
|---|---|---|---|
| **RRF fusion** | **WORKING** — but fusing ~2–3 real sources, not 6 | ablation +0.82 [+0.39,+1.24] is over vector+BM25 only | Yes, **scoped**: "fusion of a lexical + a dense leg"; not "fuses six heterogeneous stores" |
| **Dynamic token budget** | **WORKING** | the ICE-Dev 94.2%-vs-4.33 survival is real and reproducible | Yes — real (the strongest mechanism claim) |
| **Session diversify / dedup / bonuses / strengthening** | **WORKING** | post-fusion transforms, no embedding bind | Yes |
| **Cluster-scoped retrieval** | **WORKING** | `_relevant_cluster_ids` bind correct (:81); ablation +cluster_restrict ≈ +0.01 | Yes, but effect ≈ 0 (a *real* neutral) |
| **HyDE** | **Exp 2: OFF · Exp 3: ON throughout (uncontrolled)** | Mature snapshot: call site commented (orchestrator.py:313–322). Jun-30 ablation build (`8306b1b`): call site LIVE, gated by `context_reliance == LTM` (**not** the flag), so real HyDE ran on every LTM probe in ALL buildup steps incl. `bare_vector`; the `+hyde` flag was a no-op | **No** — the `+hyde` step doesn't vary it; HyDE can't be isolated in either experiment |
| **MERA** | **FUNCTIONAL but rarely triggered** | fires *only* when NER returns no entities + a category trigger; since NER worked (~95% recall), that subset was small, so MERA seldom fired; ablation −0.21 [−0.43,+0.01] is **not significant** and rests on a small effective N | Report as "narrow trigger, underpowered, not significant" — a real-but-weak measurement, not a symptom of broken NER |
| **Micro-NER (mature / Exp 2 snapshot)** | **WORKING** (trained model, ~95% recall) | a trained NER model was introduced **for Exp 2**, replacing Exp 1's regex tagger; `ner_utils.py` loads `models/ner/ner_model.pt`, regex only if missing (ROADMAP l.18). **Exp 1's "regex-only NER" is CORRECT** (no trained model existed yet) — do not strip it. The error is only if the *mature* codex's 3.3% is blamed on NER | Yes for Exp 2 — mature codex weakness is chunk size / single-fragment representation, NOT entity detection |
| **MoE routing** | **WORKING but NEUTRAL** | hardcoded map; global Δ +0.01/−0.02; it *did run* | **Yes** — this is a genuine honest null (unlike HyDE) |
| **Classifier (Qwen3-Embedding-0.6B head)** | **WORKING** | gating failures 22→2 | Yes |
| **Decay mechanics** | **WORKING** | simulated decay days per conv (B=93, A=24, C=27, D=20) | Yes |
| **Memory slots / prompt assembly** | **WORKING** | — | Yes |

## 3. Reclassifying the Experiment-3 ablation steps

The ablation deltas split into **informative** (feature genuinely ran) and **uninformative**
(feature dead/disabled/degraded — the ~0 delta measures nothing about the mechanism):

- **Informative (feature genuinely ran):** `+bm25` (−0.74, harmful unfused), `+rrf` (+0.82,
  corrective), `+dynamic_budget` (−0.11, the fill-to-cap policy), `+keyword_boost` (+0.12),
  `+session_diversify`/`+cluster_restrict` (~0 — *real* neutrals). MoE-vs-generalist (~0) is a
  *real* null too.
- **Functional but underpowered (ran, but the ~0 is weak evidence):** `+codex` (worked, but
  extraction-handicapped by 6k chunks + collapsed to one fragment, so a ~0 step delta is expected
  regardless of KG quality) and `+mera` (fired only on the small NER-empty subset; −0.21 not
  significant). Report these as "inconclusive," not "neutral."
- **Non-functional (the ≈0 measures nothing):** `+procedural` (dead bind), `+batch_summary`
  (no data existed).
- **Constant confound (on across ALL steps, never isolated):** `+hyde` — in the ablation build
  HyDE ran on every LTM probe via `context_reliance`, not the flag, so it was active in
  `bare_vector` too; the `+hyde` step is a no-op. (The step deltas for BM25/RRF still hold —
  HyDE is held constant across them.)

⇒ The abstract's "several intuitive mechanisms … are neutral" is only defensible for **cluster
scoping, session-diversify, and MoE** (they ran and were ~0). HyDE/procedural/batch must be
recategorised as *non-functional*; codex/MERA as *inconclusive* — **none** as "tested and neutral."

## 4. Provenance of the confidence intervals (the "keep in mind" note)

Both experiments' bootstrap CIs are **recent post-hoc re-analyses** of the frozen per-probe JSON,
not part of the original 2026-06-29/30 evaluation runs — but **both are committed and reproducible**:
- **Exp 3** CIs: `experiments/paper/exp3_bootstrap.py` (2026-07-20).
- **Exp 2** CIs: `scripts/oneoff/paper_bootstrap_cis.py` — **verified 2026-07-20 to reproduce the
  paper's numbers exactly** (no-ICE-Dev paired +0.002 [−0.069, +0.073] = paper's +0.00 [−0.07,+0.07];
  win 30.6 [27.9,33.4] vs 21.2 [18.8,23.7]; all-data +0.396 [+0.308,+0.488]; stress +3.097
  [+2.851,+3.325]). It faithfully replicates the published imputation chain (failed answer → 1, else
  paired score, else round(probe avg), else 3) — which the naive drop-None version got ~0.4 off.

⇒ **Only remaining nit:** the two scripts live in different folders (`experiments/paper/` vs
`scripts/oneoff/`). Co-locate them for a self-contained repro bundle (move → `experiments/paper/
exp2_bootstrap.py`, log in CLEANUP.md). No integrity gap — provenance is intact.

## 5. What this does to the paper's claims

**Survives intact:** the efficiency/curation result (tie at fewer fragments; positive fragment–score
correlation), the budget-survival result, the fusion-rescues-unfused-BM25 result, the MoE-neutral
result, LSREP as a protocol. **These are enough for an honest paper.**

**Must shrink or move to disclosure:**
1. "Six-leg orchestrator" → "six-leg *design*; three legs contributed at the snapshot" (already
   drafted in the Limitations dead-leg subsection; extend it to name batch summaries + HyDE-disabled +
   the codex under-contribution — and **fix the paper's two false codex claims** flagged by ROADMAP
   l.18–19: NER was *not* regex-fallback (it loaded the model, ~95% recall), and `pending` edges *did*
   contribute to traversal. The honest codex story is "worked, but structurally under-weighted (one
   fragment, 6k chunks, no reinforcement)," not "broken entity detection.").
2. "Dissecting what matters" framing → cannot claim a *complete* dissection. Reframe toward
   **systems + LSREP + honest preliminary evaluation** (REVISION_PLAN §11.3 direction).
3. Ablation "nulls" → keep only cluster/MoE (and session-diversify) as measured neutrals; recategorise
   HyDE/MERA/procedural/batch/codex as non-functional-at-snapshot.
4. RRF claim → scope to "a lexical + a dense leg," not "six heterogeneous stores."

**Net:** the honest paper is *smaller in what it proved* but *cleaner and unattackable*. The elaborate
machinery was mostly inactive, and the system still matched a strong baseline at lower cost — which
makes the curation thesis stronger, not weaker, provided we never claim to have validated the parts
that weren't running.

## 6. Baseline vs ICE prompt assembly — a full-system-vs-RAG comparison (design choice, NOT a confound)

The harness (`experiments/mature/run_mature_experiment.py:578–623`) assembles the conditions
differently **by design**:
- **Baseline:** standard single-leg vector-RAG (top-30 similar turns + question).
- **ICE:** the full system — fused retrieval **plus** ICE's prompt assembler (memory slots, a
  recent-turns window, structured boundaries).

This is a legitimate full-system-vs-baseline comparison. The prompt assembler (including the recent
window) is **part of ICE's contribution and its curation**, not an unfair advantage — a memory system
is *supposed* to manage recent context. **Do NOT frame this as a limitation/confound** (corrected
2026-07-20, user).

The only real fix: the paper's "differ only in single- vs multi-leg retrieval" was imprecise → v2 §5.2
now describes it accurately as *full memory system vs standard vector-RAG*. Also:
- **"32% fewer fragments" is a curation WIN** (fewer *retrieved* fragments, higher SPF). The paper
  already states ICE uses ~6.6% **more** tokens, so there is **no** "fewer tokens" overclaim to fix.
- *Optional* future analysis (not a correction the paper owes): a matched-prompt ablation (baseline
  given the same recent window) would isolate retrieval from assembly — a nice-to-have for FINAL.
