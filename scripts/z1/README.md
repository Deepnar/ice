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

## 2. OPEN — the live queue

| question | blocked on | where |
|---|---|---|
| **Does the direction rule cut reversals?** +9.4 pts seen, but two same-config runs spanned 10.0–20.4% | judge sampling (below) | ROADMAP `G59` |
| **Can two identical seeds reproduce each other?** | running 2026-08-23 | `check_reproducible.sh` |
| Which small background model is best? | the two above | MODELS.md §5 |
| NuNER vs micro on FIXED code | maintainer: **stay on NuNER for now** | — |
| Retrieval numbers under the corrected metric | re-score pending | ROADMAP `G52` |
| Should the 44% of non-lossless turns feed the graph? | **maintainer decided YES** — needs implementing | ROADMAP `G57` |
| `extraction_confidence` must stop being a truth prior | not started | ROADMAP `G58`-adjacent, PROVENANCE 2026-08-22 |

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
