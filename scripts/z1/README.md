# Z1 — the experiment index

**⚑ READ THIS BEFORE RUNNING ANY Z1 EXPERIMENT.** It exists for one reason: to
stop a session re-running something already settled, or re-deriving a number
already recorded. Both have happened repeatedly — [TRAPS #42](../../docs/TRAPS.md)
is the entry for it, and it names two cases from a single afternoon.

**This file is an INDEX, not a record.** One line per thing, with a pointer.
Nothing is explained here; everything lives in its normal home:

| what | where it lives |
|---|---|
| what was run, with what numbers, and what it does NOT show | [`docs/PROVENANCE.md`](../../docs/PROVENANCE.md) |
| the queue, and what is blocked on what | [`docs/ROADMAP.md`](../../docs/ROADMAP.md) |
| how a measurement lied, and how it was caught | [`docs/TRAPS.md`](../../docs/TRAPS.md) |
| which feature is on, off, or inert | [`docs/FEATURE_INVENTORY.md`](../../docs/FEATURE_INVENTORY.md) |
| which model did which job, and what was tested | [`docs/MODELS.md`](../../docs/MODELS.md) §5 |

⚠ **Z1 ONLY.** Z2 (the whole-system / answer-level test) gets its own index when
it starts. Do not mix them.

---

## 1. SETTLED — do not re-run these

| question | answer | where |
|---|---|---|
| Does a bigger background model fix extraction? | **No.** 8 models 4B→26B, defects in all 7 viable arms ⇒ prompt/design, not capacity | PROVENANCE `A12` (2026-08-12) |
| Does NuNER beat micro-NER for codex grounding? | **Form, not outcome.** malformed 25.0→12.5%, non-entity nodes 58.7→40.7%; correctness 18→20% (inside 1 SE) | PROVENANCE 2026-08-20 |
| Is `extraction_confidence` a usable truth signal? | **No — it is UNINFORMATIVE.** ⚠ The earlier "INVERTED" claim is **WITHDRAWN**: it rested on a 40-triplet subgroup quoted without an interval. Re-measured over 124 turns on two seeds, the runs disagree on the DIRECTION (grounded 22.3% vs 12.2%; rejected 15.0% both). Do not use it as a truth prior — because it says nothing, not because it is backwards | PROVENANCE 2026-08-23 |
| Do the six write-path mechanism fixes make the graph more TRUE? | **No — a null.** Every mechanism moved; correctness did not (z ≤ 1.14) | PROVENANCE 2026-08-22 |
| Are reversals created by relation canonicalisation? | **No.** Blocking 1,611 direction/polarity merges left the reversal rate flat ⇒ generated, not merged | PROVENANCE 2026-08-22 |
| Can a closed 197-word relation vocabulary work? | **No.** 67.8% of real relations are out-of-vocabulary; the repair ladder recovers 5.7% | PROVENANCE 2026-08-03/04 |
| Is the procedural leg's 1.000 a real score? | **No — a tautology.** It returns its limit (5 fragments) on every query, nonsense included | TRAPS #37 |

## 2. OPEN — the live queue, in order

**Next session starts at #1.** Gate 1 (can we measure?) is passed; do not spend
more time on instruments.

| # | do this | why it is next | cost |
|---|---|---|---|
| **1** | **Judge the two CONTROL arms at full coverage: `dir-false-run1` and `dir-false-run2`, `--per-turn 3`** | both treatment arms are already measured properly (`dir-true-run1` **17.1%**, `dir-true-run2` **14.4%**); only the controls are missing. `dir-false-run1` has a flat-sampler figure (11.0%) that is not comparable to them, and `dir-false-run2` has never been judged. ⚑ **DO THIS BEFORE TOUCHING THE EXTRACTOR AGAIN** — these arms were seeded with the 2026-08-22/23 code, and any further codex change makes them uninterpretable. | ~20 min |
| **2** | **Small-model sweep** on the FIXED prompt | A12 compared 8 models on the broken prompt, which likely explains why all 8 failed identically. Now affordable: one run per arm suffices (§3) | ~2 h + judging |
| **3** | **Let non-lossless turns feed the graph** ([G57](../../docs/ROADMAP.md#g57)) | maintainer decided YES. ⚠ sequence AFTER something moves correctness — at 14–17%, +44% input adds ~4 wrong facts per right one | small |
| 4 | Stop using `extraction_confidence` as a truth prior | it is uninformative, not inverted (§1). The retrieval trust floor keys on it | small |
| 5 | [G61](../../docs/ROADMAP.md#g61) silent extraction drops — four unlogged, plus a failed turn that commits its idempotency key and can never retry | latent but silent by construction; the salvage regex also cannot match a `negated` triplet, and those went 0.11% → ~6% | small |
| 6 | [G62](../../docs/ROADMAP.md#g62) `check_conflict`'s antonym branch expires edges deterministically, no LLM, no review | never fired (0 of 106 reconciles) — fix BEFORE widening `ANTONYM_OF` | small |
| 7 | [G60](../../docs/ROADMAP.md#g60) relation supersession semantics | 146 declared against ~2,026 in the store. Graph-inference is a measured dead end; use a cached one-shot model call via the maintenance agent | design |
| 8 | Re-run the five per-leg ablations | the 2026-08-20 set is dead (§3b); needs a clean same-commit baseline | ~1 h |
| 9 | NuNER vs micro on fixed code | maintainer: **stay on NuNER until the extraction side lands** | ~2 h |
| 10 | Z2 / answer layer | LAST. Needs the [G56](../../docs/ROADMAP.md#g56) judge fix, which shipped but is unexercised | expensive |

⚠ **Not on this list on purpose:** more instrument work, and chasing exact
reproducibility. The first is done; the second is unreachable ([TRAPS #47](../../docs/TRAPS.md)).

## 3. ⚑ THE MEASUREMENT FLOOR — read before quoting any number

- **Judge self-consistency: ±2.5 pts.** Same judge, same store, same seed, same
  200 triplets, run twice → 87.1% identical verdicts.
- **Sampling was the big term, and it is FIXED.** 200 triplets came from only
  ~76 turns, and triplets in a turn are correlated, so the independent unit is
  the TURN. `--per-turn 3` now covers **124 turns**, taking the interval from
  ±8 to **±6.2–6.6**.
- **Run-to-run variance in the RATE is small — measured 2026-08-23.** Two
  independent seeds of one configuration, full coverage: **17.1%** and
  **14.4%**, 2.7 points apart and inside both intervals. ⇒ **no heavy bootstrap
  is needed**; one run per arm suffices **provided it reports its interval**.
- ⚠ **The model is nondeterministic ACROSS PROCESSES** (temperature 0 fixes
  sampling, not logits). Two identical seeds share only 6 of 9 raw responses.
  That changes WHICH triplets exist; it does not move the aggregate rate much.
- ⇒ **Current honest figure: 14–17% correct, ±6 pts.** Every earlier number
  (20%, 15.8%, 11.0%, 20.4%, 10.0%) was one measurement under-sampled.

## 3b. ⛔ EVERY Z1 NUMBER, WITH ITS STATUS — the falsification table

**Read this before citing ANY figure from this project.** Nothing below is
deleted; each row says what the number is worth now. Added 2026-08-23 after a
session in which three separate "settled" findings were withdrawn — two of them
produced by the session doing the withdrawing.

| number | status | why |
|---|---|---|
| **recall@10 = 0.508** (2026-08-13) | ⛔ **DEAD, three ways** | episodic-only crediting (TRAPS #32) · 64% of that probe set is contaminated · the scorer classified without the conversation. Cite the *method* lesson, never the number |
| the 0.250 legacy control | ⛔ dead | same three defects; it is the other end of the same run |
| **20% triplet correctness** (2026-08-20) | ⚠ **superseded** | measured over 76 turns with no interval. Current figure is **14–17% ±6** over 124 turns |
| **25% reversed** (2026-08-20) | ⚠ superseded | same sampling; reversal now 21–37% depending on tier and run |
| **`extraction_confidence` INVERTED** | ⛔ **WITHDRAWN** | a 40-triplet subgroup. Two full-coverage runs disagree on the direction. The field is **uninformative**, not inverted |
| **`summary_synthesis` 0.303 / 0.322 / 0.324** | ⛔ dead | the metric was `max(coverage, presence)` — any summary fragment scored 1.0. Also measured with the summary leg switched off |
| **`procedural` 1.000** | ⛔ dead | presence test; the leg returns its limit on every query, nonsense included. Grounded score is **0.000** |
| **`episodic_lookup` 0.698 → 0.754** | ⛔ dead | 0.754 credited a 33-turn summary as "this turn came back". Corrected figure **0.599 ±0.032** |
| **`codex_multihop` 0.531** | ⚠ superseded | averaged two populations scored by different formulas. Split: anchored 0.46, unanchored 0.172. Corrected total **0.380** |
| **`temporal` 0.500 vs 0.643** | ⛔ never a result | n=28, ~1.9 SE apart. Was read as a difference; it is not one |
| **A4 grounded expansion "changes nothing"** | ⚠ **suspended** | a null from two conditions that were BOTH mis-called (65% of RRF weights wrong). Re-measure before deleting the feature |
| **`legoff-batch_summary` ablation** | ⛔ dead | ablated a leg that was already off. Four of five scores byte-identical to baseline |
| **the five per-leg ablation deltas** | ⛔ dead | same broken metric + no clean same-commit baseline |
| **codex ablation 25-14 / 36-17** | ⚠ ordering only | all three conditions shared the defect; the *rank order* survives, the absolutes do not |
| **answer verdicts 31-30, 58%/31% both_failed** | ⛔ dead | the judge truncated BOTH answers at 2,500 chars against a ~3,250 median |
| **A12's eight-model ranking** | ⚠ **weak** | scored on `summary_coverage`, which TRAPS #45 shows is circular; the ranking itself came from ONE subagent read. Its *conclusion* (defects in all 7 arms ⇒ prompt/design not capacity) stands |
| **direction rule +9.4 pts** | ⚠ **UNRESOLVED** | measured with the broken sampler; two same-config runs spanned 10.0–20.4%. Re-judge the existing arms with `--per-turn` |
| **the six write-path mechanism fixes** | ✅ **null, and trustworthy** | every mechanism verifiably moved; correctness did not (z ≤ 1.14) |
| **store-level counts** (entities, edges, degree-1 %, expiry rate, `in` = 1 edge) | ✅ **TRUSTED** | direct SQL counts, no sampling, no judge, no retrieval |
| **NuNER halves malformed / non-entity nodes** | ✅ trusted | two independent instruments agree, store-level |
| **run-to-run rate variance is small** | ✅ trusted | two seeds, full coverage, 17.1% vs 14.4% |

**The pattern worth carrying:** everything measured by **counting the store
directly** survived. Everything that went through **a sampler, a metric or a
judge** needed correcting. That is where to be suspicious first.

## 4. SCRIPTS — what each one is for

**Seeding / arms**
| script | purpose |
|---|---|
| `seed_store.py` | build a store from the corpus (`--clean`, `--limit`, `--bg-model`) |
| `ner_arm_seed.sh` | the A9b two-arm NER comparison (micro vs NuNER) |
| `two_arm_seed.sh` | earlier two-arm background-model seed |
| `reseed_postfix.sh` | one arm, arm-B config, varying ONLY the code |
| `prompt_ab_noisefloor.sh` | 4 arms: each prompt twice → effect + noise floor |
| `check_reproducible.sh` | **detector**: same seed twice, compare. No fix in it, deliberately |
| `snapshot.py` | `save` / `restore` / `list` / `counts` an arm |
| `drain_batch_summaries.py` | force + ASSERT batch summaries (a swallowed failure once cost 3 days) |

**Probes / ground truth**
| script | purpose |
|---|---|
| `generate_typed_probes.py` | build the typed probe set |
| `generate_probes.py` | the older flat probe set |
| `derive_retrieval_gt.py` | ground truth for retrieval |
| `verify_gold.py` / `check_gold_consistency.py` | validate the gold before trusting a score |

**Scoring / judging**
| script | purpose |
|---|---|
| `score_typed.py` | 5 per-type metrics. **Never averaged into one number** |
| `score_retrieval.py` | recall@k / MRR on the flat probe set |
| `answer_probes.py` | retrieve → assemble the REAL prompt → generate an answer |
| `judge_answers.py` | paired A/B verdicts on answers |
| `judge_codex.py` | **is the stored graph TRUE?** the only outcome metric here |
| `compare_judgements.py` | two judgement runs, per TRIPLET not per rate |
| `dump_graph_fingerprint.py` | compare two stores triplet-by-triplet; localises divergence |
| `harvest_probe_context.py` | reading instrument, computes no score |
| `run_leg_ablations.sh` | the five per-leg ablations |
| `store_report.py` | store statistics |
| `production_parity.py` | **the ONE reproduction of `main.py`'s pre-retrieval path** |
| `run_meta.py` | provenance block stamped into every artifact |

## 5. ARTIFACTS — where output lands

⚠ **All of `experiments/curation_files/` is GITIGNORED** (it carries the
corpus). A number that matters must be copied into PROVENANCE in the same
session — [TRAPS #40](../../docs/TRAPS.md).

| path | holds |
|---|---|
| `snapshots/<arm>.sql` + `.counts` | a full store, restorable |
| `score_runs/*.json` | typed / retrieval scores. ⚠ some are stamped `INVALIDATED` — read `meta` |
| `judgements/codex_quality_*.json` | per-triplet graph-truth verdicts |
| `probe_answers/*.json` | assembled prompts + generated answers |
| `fingerprints/*.json` | graph fingerprints for reproducibility checks |
| `typed_probes.json` | **the live probe set** (444). Others are older generations |
| `logs/` (repo root) | run logs. Gitignored but survive reboot — never write to `/tmp` |

## 6. ARMS — which store is which

| snapshot | NER | code | prompt | note |
|---|---|---|---|---|
| `ner-a-micro` | micro | pre-08-22 | old | A9b arm A |
| `ner-b-nuner` | NuNER | pre-08-22 | old | **A9b arm B — the baseline most numbers cite** |
| `ner-b-postfix` | NuNER | 08-22 fixes | old | the six write-path fixes, 293 turns |
| `dir-false-run1/2` | NuNER | 08-22 | old | direction-rule control, 180 turns |
| `dir-true-run1/2` | NuNER | 08-22 | **new** | direction rule ON |
| `fixed-*`, `gemma4-26b`, `pre-*`, `arm1-post-g50` | — | older | — | superseded; kept for history |

## 7. RULES THIS FILE EXISTS TO ENFORCE

1. **Check section 1 before proposing an experiment.** Grep PROVENANCE for the
   *subject*, not the item id you have in mind.
2. **Quote the interval, never the bare percentage.** §3 is why.
3. **One variable per arm.** Two changes and one number answers neither.
4. **Same judge, same seed, or the comparison is not a comparison.**
5. **A number that matters goes into PROVENANCE the same session.** The folder
   it was written to is gitignored.
6. **Update this file when an experiment lands.** One line. If it takes more
   than a line it belongs in PROVENANCE and this gets the pointer.
