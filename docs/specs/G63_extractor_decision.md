# G63 — Extractor decision protocol: NuExtract3 vs ICE's current extractor

**Purpose: settle the background extraction model ONCE, on evidence, and derive
the production config from the same runs rather than guessing it afterwards.**

Owner item: [ROADMAP G63](../ROADMAP.md#g63). Evidence: [PROVENANCE](../PROVENANCE.md) 2026-08-24.

**v3 repair update, 2026-09-12:** the deployed extractor/model decision stands.
Extraction eligibility and complete-response/error semantics now follow
[V3_REPAIR.md](V3_REPAIR.md); historical lossless-only and salvage behavior below
is superseded. The template default was already promoted on August 27.

---

## The rules this protocol runs under

1. **One variable per run.** The first 2026-08-24 comparison changed model *and*
   path together and had to be re-done. Do not repeat that.
2. **Same turns, same seed, every condition.** Random `lossless` turns,
   600–6,000 chars. Never a sample selected for one side's failures — that
   mistake was made once already and produced a non-result.
3. **Structural metrics need no judge and are computed for every condition.**
   Counting directly is the one class of measurement that has survived scrutiny
   in this project. Truth needs a human and is spent only where it decides.
4. **Both sides run through the SAME post-filters** (`_ground_triplets`,
   `is_unusable_entity_name`, `is_clausal_relation`, `canonical_relation`).
   Whatever ICE refuses from itself it refuses from NuExtract3.
5. **⚠ Count relations AFTER canonicalisation for both.** Early runs compared
   ICE's post-canonicalisation count against NuExtract3's raw one. Unfair;
   corrected here.
6. **Every row gets a WIN marked, or is struck as inconclusive.** No row is left
   as a vibe.

**Sample size: 60 turns** (up from 20). Store has 180 lossless. ICE ≈17.5 s/turn,
NuExtract3 ≈10.3 s/turn.

---

## PHASE 0 — Baseline

- [x] **B1. ICE production baseline at 60 turns.** ✅ **2,280 facts (38/turn),
      0 failures, 26.5 s/turn.** grounded 20.0% · junk names 9.3% · caps 1.0% ·
      in-vocab 74.0% · 1,143 distinct relations, 69.3% singletons.

## PHASE 1 — The guard matrix (structural only, no human time)

Each row is one NuExtract3 variant at 60 turns. Metrics: facts/turn ·
grounded % · unusable-name % · clausal % · distinct relations
**post-canonicalisation** · singleton % · sec/turn · parse failures.

| # | condition | status | winner |
|---|---|---|---|
| **G1** | template + entity list *(reference config)* | ✅ done @20 — 27.2% grounded, 3.6% junk, 11 facts/turn | — |
| **G2** | template, **no** entity list | ✅ done @20 — 6.6% grounded, 9.4% junk | **entity list WINS, keep it** |
| **G3** | ICE's nine-rule prompt on NuExtract3 | ✅ done — garbage (`affordable education --offers--> europe`) | **template WINS** |
| **G4** | NuExtract3's template on qwen3:4b | ✅ done — 14/20 parsed, 43 facts, clause-triplets | **NuExtract3 WINS on model** |
| **G5** | template + **relation vocabulary** as a preference | ✅ in-vocab **50.9 → 75.2%** (past ICE's 74.0). Cost: failures 3 → 17 | **vocabulary WINS on its own target** |
| **G6** | template + **canonicalisation rule only** | ✅ caps **54.0 → 30.0%**, articles 2.1 → 0.4%. Cost: failures 3 → 10 | **canon rule WINS on its own target** |
| **G7** | **chunked at 550** vs whole-turn | ✅ **failures 3 → 0**, facts 748 → 1,248. Cost: 13.2 → 36.6 s/turn | **chunking WINS — it fixes reliability** |
| **G8** | `max_tokens` 1200 vs 3000 | ✅ 1200 → **30 of 60 turns fail (50%)**, 231 facts | **3000 WINS decisively; 1200 is a bug** |
| **G9** | relations counted **after `canonical_relation`** | ✅ folded into every row above | — |
| **G10** | **NuNER vs micro-NER** feeding NuExtract3 | ✅ micro: more facts (838 vs 748) + better grounding (53.7 vs 38.2) but **junk names 8.7 → 19.5%**, caps 54 → 68.7% | **NuNER WINS on quality** ⚠ decided on SHAPE only, never judged |
| **G11** | **`nu_combined`** — every winning guard, no ICE prompt | ✅ **1,122 facts, 0 failures**, grounded **55.7%** (best), in-vocab **75.4%** (best), singletons **61.0%** (best), caps 29.5%. Cost: **43.0 s/turn** | **best SHAPE of any config** |

### ⚑ What Phase 1 established

**Every guard helps EXCEPT ICE's nine-rule prompt** — that is the only one that
measured a loss (it turns NuExtract3's output into garbage). Entity list,
vocabulary, canonicalisation rule, chunking and a real token budget each
delivered exactly the improvement they were added for, and **chunking absorbs
the reliability cost the prose additions impose** (that is why `nu_combined`
has zero failures despite carrying both).

⚠ **AND NONE OF IT MEASURES TRUTH.** `ice_baseline` tops this table on caps
(1.0%), in-vocab (74.0%) and volume — and measured **10% correct / 40%
reversed** against the maintainer's labels. Shape and truth are separate axes
and ICE is the proof. **No config may be chosen from this table alone.**

⚠ **A verdict was retracted here.** G5 and G6 were first called losses on parse-
failure rate before their targeted metrics were computed. Both hit their targets.
Measuring the generic number instead of the one the change was for is the error.

⚑ **G10's hypothesis (maintainer):** NuNER should pair better with NuExtract3 —
same group (`numind`), and the tagger's output is what the extractor is
conditioned on. Worth noting the config default is `preflight` (**micro-NER**)
while every measured arm was seeded `background` (**NuNER**) by env var, so the
"stay on NuNER" decision was never actually wired into the default. A9b left
this call site on the micro-NER deliberately, on 10 turns of evidence, for the
two-arm re-seed to settle. This is that settlement.

⚠ **G5 is the one with no good prior.** Closed vocabulary loses 67.8% of real
relations ([G45](../ROADMAP.md#g45)); open vocabulary gives 214 relations per
365 facts, 85% used once. ICE's rule 1 already tries the middle — *"prefer these
words, invent one if nothing fits"* — and that middle is what produced the 214.
So the middle is **tested and failed on qwen**; whether a specialist handles the
same nudge is genuinely unknown.

## ⚑ PHASE 2 RESULT — DONE 2026-08-24. The vocabulary is the culprit.

| arm | correct | reversed | verdict |
|---|---:|---:|---|
| `ice_baseline` | 20% | 20% | ⛔ loses |
| **`nu_ref`** bare template + entity list | **50–70%** (pooled **60%**) | **0%** | ✅ |
| `nu_combined` all three guards | 20% | 20% | ⛔ collapses to ICE |
| **vocabulary ONLY** | **22%** | 11% | ⛔ **−38 pts, CI [−72, −3]** |
| canonicalisation rule ONLY | **80%** | 0% | ✅ no drop |
| chunking ONLY | **80%** | 0% | ✅ no drop |

**Pooled, two rounds: ICE 15% vs NuExtract3 60%. Δ +45 pts, CI [+18, +72].
Reversed 6/20 vs 0/20.**

⇒ **WINNING CONFIG: NuExtract3 + entity list + canonicalisation rule + chunking
+ 3000 tokens, and NO relation vocabulary.**

⚠ **That exact combination has never been run** — `nu_combined` carried the
vocabulary. Its parts tested at 80% and 80% individually. **Run and judge it
before Phase 3.** (New row: **G12 `nu_novocab`**.)

⚠ **The vocabulary must not simply be deleted either** — without it relations
sprawl (424 distinct from 748 facts, 72% singletons). The fix is to apply the
vocabulary **after** extraction via `canonical_relation`, never in the prompt.
See the relation-growth note in [ROADMAP G45](../ROADMAP.md#g45).

## PHASE 2 — Truth (costs maintainer time, spend once)

- [ ] **T1. Blind label, finalist vs ICE, 40 facts** (20/side, up from 10).
      Same format as 2026-08-24: shuffled, extractor hidden, paired by turn,
      full source inline, key held separately.
      *Only the Phase-1 winning config is labelled — not every variant.*
- [ ] **T2. Score and record.** Δcorrect and Δreversed with intervals.
      **Decision gate:** NuExtract3 wins only if Δcorrect > 0 at 95% and
      reversed does not regress.

**Standing result to beat:** ICE 10% correct / 40% reversed vs NuExtract3 70% /
0% at n=10 per side, Δ +60 pts, CI [+26, +94].

## ⚑ THE MODEL IS DECIDED — 2026-08-24. The CONFIG is not.

**DECIDED: `NuExtract3-Q8_0` replaces `qwen3:4b-instruct` as the background
extractor.** Two independent blind rounds by the maintainer: **60% vs 15%
correct, Δ +45 pts, 95% CI [+18, +72]; reversed 0/20 vs 6/20.** This is the only
intervention in the project's history to move graph correctness.

**NOT DECIDED: the exact config.** One gate remains.

- [x] **G12 — DONE 2026-08-25. Config: `nu_ref`. Bare template + entity list + 3000 tokens. Nothing else.**

  | arm | correct | wrong | malformed | s/turn |
  |---|---:|---:|---:|---:|
  | **`nu_ref`** | **70%** (pooled **63%**, 19/30) | 30% | **0%** | **13.2** |
  | `nu_novocab` (+canon +chunk) | 40% | 30% | **30%** | 42.7 |

  **Δ −30 pts, 95% CI [−72, +12] — DOES NOT CLEAR ZERO.**
  ⚑ **This is a judgement call on thin evidence, not a measured win**, and must
  be cited that way. `nu_ref` is chosen because it is the **best-measured**
  config (only one with n=30), the simplest, and 3× faster — not because
  `nu_novocab` is proven worse.
  - **The malformed pattern is the one real signal:** 30% vs 0%, and the
    mechanism is visible: the canonicalisation rule's *"concise"* collapses
    specific referents into a generic `person`, where `nu_ref` kept the actual
    referent. Examples: `PRIVATE_CONTEXT.md` `[PRIVATE:g63-malformed-examples]`.
  - ⚠ **`nu_canonrule` alone scored 80% and `nu_chunked` alone 80%, yet together
    40%.** At n≈10 that is as likely noise as interaction. Unresolved.
  - **CHUNKING STAYS A SEPARATE DECISION.** Its benefit is mechanical, not
    statistical: `nu_ref` had 3 parse failures in 60 turns (5%), chunked configs
    had **zero**, because shorter inputs never hit the token ceiling. Make it
    **conditional on turn length** — ICE-Dev's massive turns need it.

### ⚑ Which existing codex fixes survive the model swap

| carries over (post-extraction) | does NOT apply (prompt-side) |
|---|---|
| grounding + confidence tiers — **works better**, 38–42% grounded vs ICE's 20% | [G59](../ROADMAP.md#g59) direction rule — moot, NuExtract3 gives 0 reversals |
| [G44](../ROADMAP.md#g44) entity-name gate — junk 8.7% vs 9.3% | rule 7 entity-shape |
| G43 property value must occur in source | rule 6 negation instruction — **but see below** |
| [G45](../ROADMAP.md#g45) canonicalisation + repair ladder | G32/a1 schema-constrained decoding |
| [G50](../ROADMAP.md#g50) merge_key · [G51](../ROADMAP.md#g51) · [G62](../ROADMAP.md#g62) · A6 reconciler · A7 payload | |

✅ **NEGATION SURVIVES, and improves.** The template has no `negated` field, but
NuExtract3 writes negation into the relation word and ICE's post-hoc
`_is_negated`/`_strip_negation` catch it — **7.5% of its relations detected as
negated against qwen's 1.0%**, with false friends (`unmatched`, `understood`)
correctly rejected.

⚠⚠ **UNTESTED AND IT BLOCKS THE RESEED: every measurement here is
PRE-WRITE-PATH.** NuExtract3's facts have never been through `handle_triplet` —
so their interaction with merge keys, supersession, the antonym branch and the
context payload is unknown. **Test end-to-end before P7.**

- [ ] ~~**G12. `nu_novocab` — the winning combination, run and judged.**~~
      Template + entity list + canonicalisation rule + chunking + 3000 tokens,
      **no relation vocabulary.** ⚑ **This exact combination has never been
      run.** Its parts scored 80% and 80% in isolation and the vocabulary
      scored 22%, so the prediction is ~80% — but a prediction is not a
      measurement, and `nu_combined` already demonstrated that guards interact.
      *(~45 min run + one 20-fact blind round.)*

⚑ **SEQUENCING DECISION (maintainer, 2026-08-25): measure G12 BEFORE writing any
production code.** Building the second extraction path now means building it for
an unvalidated config; if G12 surprises us the code is rebuilt. One run removes
that risk. *Phase 3 does not start until G12 is ticked.*

## PHASE 3 — Production changes, ONLY after G12

⚑ **Every item is a `src/` change and needs maintainer approval individually.**

- [x] **P1. Second extraction path** — `codex_extraction_mode` = `instruct` |
      `template`. Template sends the JSON schema to fill and nothing else,
      strips `</think>`, accepts a `{"facts": […]}` envelope, skips the
      schema constraint. **Default `instruct`**, so every published number
      still reproduces. *(2026-08-25)*
      - ⚑ **Fixed a LATENT bug found while wiring it:** the parse filter tested
        `all(k in item ...)` — key PRESENT, not value a string — so a `null`
        admitted a triplet that killed a `.strip()` 200 lines later and lost
        **the whole turn's extraction**. Invisible on the instruct path because
        the JSON schema guaranteed strings.
- [x] **P2. Token budget** — 1200 → **3000**. At 1200 template mode lost **30 of
      60 turns**; `bg_model_output_truncated content_chars=3648` also fired on
      **qwen**, so this was degrading the shipped path too. ⚠ Timeout couples:
      72s → 180s. ⚠ Reproduce old runs with `CODEX_EXTRACTION_MAX_TOKENS=1200`.
      *(2026-08-25)*
- [x] **P3. Adaptive chunking** — sized from `serving_window()` minus prompt and
      output budget, clamped by `codex_extraction_chunk_max` (**4096**), with a
      **loud warning** and a fall back to 550 when the probe fails. The
      `extraction_chunking` log now carries the budget and its source, not just
      the result. ☐ **Ceiling still open** — see [G68](../ROADMAP.md#g68) for the
      71,256-turn distribution and the deferred size sweep. *(2026-08-25)*
- [x] **P4. NER tier flipped to `background` (NuNER)** — the settlement A9b
      deferred: junk entity names **8.7% vs 19.5%** over 60 turns, and every
      measured arm already set it by env var. Post-filter stack kept unchanged
      (grounding, confidence tiers, name gate, clausal gate, canonicalisation,
      merge keys). ⚠ **Decided on SHAPE, never judged for truth.** *(2026-08-25)*
- [ ] ~~**P5. Malformed post-filter**~~ — **SKIPPED 2026-08-25 (maintainer).**
      The 20% malformed rate it targeted came from an early config; in the last
      judged round the shipped config (`nu_ref`) scored **0% malformed** and the
      30% belonged to `nu_novocab`, which was dropped. ⚠ `is_unusable_entity_name`
      once required a letter and destroyed **94 true numeric entities**
      (`3.80 --score--> maths`) — not worth tightening against a stale number.
      **Re-measure the malformed rate in the next judged round; act only if real.**
- [x] **P6. Docs in step with the code** — FEATURE_INVENTORY rows for the output
      budget, prompt shape, adaptive chunking, NER tier and the relation
      threshold; MODELS.md background-extractor decision; ROADMAP G63/G68.
      *(2026-08-25)*
- [ ] **P7. Full reseed** on the settled config → the store [G66](../ROADMAP.md#g66)
      needs to ask whether answers actually improve.

## What this protocol deliberately does NOT decide

- **Whether better facts produce better answers.** That is
  [G66](../ROADMAP.md#g66) and it is downstream of everything here. A clean win
  in Phase 2 still does not mean the product improves.
- **The `nobody`/`no one`/`no body` variant problem.** NuExtract3 mints
  morphological variants as separate nodes. [G50](../ROADMAP.md#g50) shipped a
  deterministic `merge_key` tier that will not catch these, and **already tested
  and rejected** the embed-cosine-LLM approach that would. Needs a new idea, out
  of scope here.
- **Anything about retrieval.** Extraction only.
