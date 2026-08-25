# Handoff — 2026-08-25, ~23:15 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Overwritten every session,
committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

> **⚑ START AT [`docs/specs/RESEED_PLAN.md`](specs/RESEED_PLAN.md).** The next
> action is the reseed. Everything before it is dead data.

---

## TOLD → DID

**Told:** the previous handoff said to start at `scripts/z1/README.md` §2 — judge
the two control arms, then work down a queue built on the theory that the
extraction *prompt* was what needed fixing.

**Did:** judged the controls (G59 is a null), then found the queue's premise was
wrong. **It was never the prompt. It was the model.** Replaced the background
extractor, and the graph went from **15% correct / 30% reversed** to **63% / 0
reversals** — the first intervention in this project's history to move graph
correctness. Shipped five production changes. Then, at the maintainer's
direction, **declared every prior store and most of the last two weeks' findings
dead** and wrote the plan to start over.

## 1. ⚑ WHAT IS TRUE NOW

| | |
|---|---|
| **Graph correctness** | **63%** (19/30, pooled over 3 blind rounds) with **NuExtract3** · **15%** (3/20) with the old `qwen3:4b-instruct`. Δ **+45 pts, 95% CI [+18, +72]** |
| **Reversals** | **0 of 20** (NuExtract3) vs **6 of 20** (qwen). The defect that dominated three sessions does not occur |
| **The judge** | ⛔ `muse-spark` (73% vs human) dead 3 days, server-side. ox-alpha 67% but 37 s/call · deepseek-v4-flash 60% · **mimo-v2.5 27%** |
| **Ground truth** | the maintainer's own blind labels — 65 facts across 3 rounds. Beat every model judge tested |
| **Store** | `dir-false-run1` restored, counts verified. ⛔ dead data — see §4 |
| **Git** | clean tree except this session's work, local only |
| **Tests** | smoke 183/183 · settings freeze 147/147 · dynamics invariants 17/17 |

## 2. THE FINDING, AND HOW IT SURVIVED SCRUTINY

**It is the MODEL, not ICE's machinery.** The full 2×2 was run because the
maintainer challenged the first framing:

| model | path | result |
|---|---|---|
| qwen3:4b | ICE's instruct prompt | 15% correct, 30% reversed |
| qwen3:4b | bare template | ⛔ clause-triplets, 30% unparseable, 43 facts/20 turns |
| NuExtract3 | ICE's instruct prompt | ⛔ garbage (`affordable education --offers--> europe`) |
| **NuExtract3** | **its own template** | ✅ **63% correct, 0 reversed** |

⇒ **"Remove ICE's guards" was tested and is FALSE** — qwen without them is far
worse. The guards were a workaround for a model never built for extraction.

⛔ **The relation vocabulary in the prompt is what destroys truth** — isolated at
**22% correct** against the bare template's 60%. Given a list, the model reaches
for a listed word when none fits: `MIT/Stanford/CMU --works_at--> Google India`,
`authors --cites--> authors`.

## 3. WHAT SHIPPED (`src/`, 2 files)

1. **`codex_extraction_mode`** = `instruct` | `template`. Template sends a JSON
   schema to fill and nothing else, strips `</think>`, accepts a `{"facts": […]}`
   envelope, skips the schema constraint. **Default `instruct`.**
2. **`codex_extraction_max_tokens` 1200 → 3000.** At 1200 template mode lost
   **30 of 60 turns**; truncation was logged on qwen too.
3. **Adaptive chunking** — sized from `serving_window()`, ceiling 4096, loud
   warning + fallback to 550 when the probe fails. Replaced a fixed 550 that
   split a 1,178-token turn into three.
4. **NER tier default → `background` (NuNER)** — junk names 8.7% vs 19.5%. The
   settlement A9b deferred; every measured arm already set it by env var.
5. ⚑ **A latent bug that had been silently losing whole turns' extraction** —
   the parse filter tested `all(k in item ...)` (key PRESENT, not value a
   STRING), so a `null` killed a `.strip()` 200 lines later. Invisible for
   months because the JSON schema guaranteed strings on the only path used.

**Verified end-to-end** on 5 turns through `extract_codex`: 94 edges, 76
entities, **zero expiries**, canonicalisation and merge-key both fired, entity
gate refused `you` 9 times. **The write path handles the new model.**

## 4. ⚑ DECISIONS MADE (do not re-litigate)

- **The background extractor is NuExtract3.** Settled on evidence, twice.
- **The relation vocabulary NEVER goes in the prompt.** Apply it after
  extraction via `canonical_relation` instead — zero prompt cost.
- **⛔ EVERY PRE-RESEED STORE IS DEAD DATA** *(maintainer)*: "lets just call ALL
  from before as we have no data, we are restarting ALL again." All Z1 arms were
  built by the 15%-correct configuration.
- **Re-measure the last two weeks**, not just the graph findings.
  [G71](ROADMAP.md#g71) is the register: 2026-08-11 → 08-24, 16 entries, **3
  survive · 3 already dead · 10 to re-measure**.
- **Reseed 1,000–1,500 DENSE turns**, not 180. ⚠ the median turn is 24 tokens;
  the 14,665 turns ≥100 tokens carry 91% of all content.
- ⚑ **STANDING RULE: pick a side on evidence, then DELETE the loser** — for what
  a feature does and for what we do. ⚠ **but commit AFTER measuring, never
  instead of**: [G51](ROADMAP.md#g51) picked the aggressive side without a
  measurement and silently destroyed **667 true facts**.
- **P5 (malformed post-filter) SKIPPED** — its 20% target came from a config we
  dropped; the shipped one scored 0% malformed. Re-measure before acting.
- **`CLAUDE.md` line 32 remains knowingly stale** ("dense enough") — unchanged
  from the previous handoff. Fix the behaviour first ([G57](ROADMAP.md#g57)).

## 5. NEW ITEMS OPENED

**G63** model decided · **G64** fact-as-sentence (the one Graphiti idea worth
taking) · **G65** judge instability + the calibration gate · **G66** do better
facts improve ANSWERS · **G67** maintenance agent / graph shape · **G68**
chunking + truncation *(shipped; ceiling sweep deferred)* · **G69** the relation
vocabulary growth loop is built and inert · **G70** read-side instrumentation
*(lands with Z2)* · **G71** the re-measurement register · **G72** the clean break.

## 6. WHAT IS STILL UNMEASURED

Whether better facts improve **answers** (G66 — the one that decides whether any
of this mattered) · the reject-but-keep tier, **79% of facts land at 0.35** and
`extraction_confidence` does not predict truth · the chunk ceiling above ~1,500
tokens · whether NuNER beats micro-NER for **truth** (decided on shape only) ·
the whole answer layer, decay, reflection, the maintenance agent, the B2 gate.

## 7. NEXT

**[`docs/specs/RESEED_PLAN.md`](specs/RESEED_PLAN.md).** Seed 1,000–1,500 dense
turns on the settled config → judge the new baseline → settle reject-but-keep
with a tier-stratified round → **G66** → work the G71 register.

⚠ **Open before step 1: which judge.** muse-spark is gone. For G66's coarser
question ("is answer A better than B") 60–67% may be enough; for triplet-level
truth it is not. Calibrate anything new with `scripts/oneoff/calibrate_judge.py`
before adopting it.

**⚑ Do not write this file, or close a session, without the user saying so.**
