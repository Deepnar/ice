# Handoff — 2026-08-16, ~23:30 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Overwritten every session,
committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

> **⚑ ONE FINDING HERE INVALIDATES EVERY RETRIEVAL NUMBER THIS REPO HAS
> PUBLISHED**, including the 0.508 a whole cycle celebrated. It is the first
> section below. Read it before quoting any recall figure, and before designing
> anything that rests on one.

---

## TOLD → DID

**Told:** settle the background model, then the pre-re-seed fixes, then re-seed.

**Did:** built the two instruments that were missing (answer generation and a
paired judge), ran both arms through them, and **found that recall has never
measured four of the five legs.** The model question came out directional but
thin; the metric finding is the session's actual result. Six extraction fixes
landed. The re-seed did **not** run — deliberately, and the reason matters.

## ⚑ START HERE — THE STATE OF THE WORLD

| | |
|---|---|
| **Store** | **arm 1 (`fixed-qwen3-4b-instruct`) live** — 8,280 entities / 9,662 edges / 293 turns. Both arms snapshotted. |
| **Git** | Clean. Pushed through `07fc689`; later commits local. History was scrubbed and verified before the first push (see below). |
| **Tests** | 347/347 regressions · `test_codex_write_path.py` 32/32 · `test_maintenance_agent.py` 45/45 · `test_retrieval.py` 3/3 · `test_retrieval_failopen.py` 27/27. |
| **Cloud API** | **Usage limit reached.** Resets ~17:00 the following day. The judge died at 65 of 150 because of it. |
| **Probe set** | **372** (was 420) — see the loss below. |

## ⚑ 1. RECALL ONLY EVER SCORED THE EPISODIC LEG

Measured on one harvest:

| leg | fragments returned | ever credited as gold |
|---|---|---|
| episodic | 4,124 | 249 |
| codex | 474 | **0** |
| procedural | 400 | **0** |
| timeline | 207 | **0** |

Not because those fragments were wrong — because `ContextFragment` carried no
link to the turns it was derived from, so nothing could attribute them.
**Every recall number in this repo is an episodic-leg score reported under the
whole system's name.**

Fixed for codex and procedural via `origin_batch_ids`, in the two harnesses
**and in `score_typed.py`**, which carried the same blind logic in four places
and would otherwise have reproduced the old numbers. **Timeline is still
unwired.** Effect with ICE untouched and only the metric changed: codex 0 → 18
credits, procedural 0 → 4, probes with a gold hit **15/25 → 21/25 (60% → 84%)**
on a 25-probe sample. ⚠ **That is a sample, not a re-measurement** — the full
372-probe re-count is patched and ready but has NOT been run (~10 min, local,
deterministic, no API). It will be the first honest recall number this repo has.

⚠ **Two id spaces nearly produced a false finding.** `source_batch_id` is the
episodic **row id**; `codex_edges.source_batch` is the **batch id** — 9,662 edges
join on batch id, **0** on row id. The first two attempts compared them directly,
got zeros, and that read as *"the graph does not cover the gold turns."* That
sentence was one step from the roadmap. [G48b](ROADMAP.md#g48b), [TRAPS #33](TRAPS.md).

## 2. THE MODEL QUESTION — directional, thin, and qwen3 is the pick

**65 usable verdicts of 150; 85 lost to the API limit.** Probe order is
round-robin by type, so what landed is balanced rather than 65 of one type.

| type | qwen3:4b | gemma4:e4b | tie |
|---|---|---|---|
| codex_multihop | 4 | 3 | 6 |
| episodic_lookup | 3 | 2 | 8 |
| procedural | 4 | 2 | 7 |
| summary_synthesis | 3 | 2 | 8 |
| temporal | 4 | 1 | 8 |
| **total** | **18** | **10** | **37** |

**Pick `qwen3:4b-instruct`**, on three things that don't depend on the judge:
graph connectivity (**69.6% vs 80.5%** dead ends, re-measured from both
snapshots), size (**2.5 GB vs 9.6 GB**), and the pending fixes all landing on the
graph side where it already leads. The counterweight is real — gemma4 wins
episodic recall 0.664 vs 0.591 — but that number is now known to score one leg.
**Cheap to reverse:** re-seeding one arm is ~50 minutes.

## 3. WHAT SHIPPED — six extraction fixes, all awaiting a re-seed to be visible

| item | was | now |
|---|---|---|
| **[G44](ROADMAP.md#g44)** numbers | `8`, `19`, `2023`, `9.65` refused — **94 of 2,273** | numerals are entities; pronouns still refused |
| **[G44](ROADMAP.md#g44)** promotion | deleted the endpoint of the triplet writing it | `protect_ids` + SAVEPOINT; **98 saves** in 586 turns |
| **[G49](ROADMAP.md#g49)** clauses | whole sentences became relations | >5 words ⇒ demoted, never dropped |
| **[G45](ROADMAP.md#g45)** supersession | any unknown relation retired its predecessor — **7 of 8 components of one explanation** | supersedes only for *known* single-valued relations |
| **[G45](ROADMAP.md#g45)** attractors | `has` 491 / `have` 78 / `had` 60 coexisting | known-set fed forward within a turn; threshold **0.82 → 0.90** |
| **[G49](ROADMAP.md#g49)** activation | 61 patterns citing 514 turns, **1 active** | also activates on `procedural_min_cited_turns` (10) ⇒ **29 of 61** |

Plus **[G50](ROADMAP.md#g50)** `merge_key()` (Tier 0 was structurally dead —
`canonical_name` is UNIQUE), G16 privacy leak, G29 idempotency key, G46 leg
attribution completed, and per-leg reporting decoupled from the coverage flag.

## 4. WHY THE RE-SEED HAS NOT RUN, AND WHAT IT NEEDS FIRST

Every fix above is invisible until a re-seed, so they were **batched
deliberately** — one 50-minute run instead of six. It is blocked on:

0. **Run the full 372-probe re-count** — patched, ready, ~10 min, deterministic,
   no API. Cheapest real number available and it needs nothing else first.
1. **[G50](ROADMAP.md#g50) auto-apply policy** — consolidation is human-gated at
   ~5/run against **1,094 candidates**. The policy governs what the store
   *becomes*, so running it after the re-seed means seeding twice.
2. **Temporal + codex probe generation** — both launched, both ran for ~2 hours
   of GPU, **both lost everything**: the generator only wrote at the end. It now
   checkpoints every call to a `.raw-calls.jsonl` sidecar, so a re-run is
   salvageable. ⚠ **Killing the script does not stop Ollama** — queued requests
   outlive the client and the card ran to 105 °C until a restart. Use
   `ollama stop <model>` then `ollama ps`, and keep `--workers` low
   ([TRAPS #36](TRAPS.md)).
3. **Optional but cheap:** benchmark vLLM against the re-seed. One fixed model
   over 293 turns is where the hour goes and exactly vLLM's strength. **No note
   exists in this repo about which models have issues under vLLM** — searched
   tracked docs and the gitignored working files. Write it when it is tested.

## 5. ⚠ 48 PROBES WERE DESTROYED AND CANNOT BE RECOVERED

`generate_typed_probes.py --types temporal --limit 6` built 0 prompts and
**wrote the empty result over 420 typed probes.** `curation_files/` is
gitignored, so no history. 372 were rebuilt by merging an older 328-probe file
with records recovered from the answer-run JSONs. **Codex fell 89 → 41**, and
those 48 had cost a salvage fix to obtain in the first place.

Guards added and verified against the exact command: refuse to write zero over a
populated file, merge partial runs, keep the previous generation. **Run smoke
tests against a temp output path, never the live one.** [TRAPS #35](TRAPS.md).

## 6. THE JUDGE — and the fix that made it worse

Paired, blind, A/B slot randomised, reason from a fixed enum with **`both_failed`
load-bearing**. It writes partial state after **every** probe, which is the only
reason a limit-killed run left usable data.

⚠ **`reasoning_effort="none"` cured empty content and caused a worse failure.**
`deepseek-v4-flash` is a reasoning model; at `max_tokens` 300 it spent the budget
in the hidden block and returned empty content *only on long inputs*. Disabling
reasoning fixed that and made the judge **confidently wrong** — over-calling
`both_failed` on **30 of 73**, several at 100% content overlap with the expected
answer. Reasoning restored, ceiling removed: `both_failed` fell **49% → 31%**.
An entire 150-probe paid run was spent on verdicts that had to be discarded.
[TRAPS #34](TRAPS.md).

## 7. FAN-OUT IS AN EXTRACTION PROPERTY — five hypotheses died

Dead ends are the **least** duplicated part of the graph: degree-1
nearest-neighbour mean **0.8194** (12.2% above 0.90) against degree-5+ **0.8908**
(47.6%). Near-duplicates concentrate in the **hubs**, exactly where merging is
most dangerous because a hub carries facts to re-attribute.

Nor is it one bad relation: `has` touches 6.1% of dead ends, `description` 3.0%,
with the top 18 covering ~21% across a ~1,800-relation tail.

⇒ **No merge policy moves fan-out.** [G50](ROADMAP.md#g50) is a graph-*quality*
item; do not justify it by pointing at 80.5%.

## 8. CLAIMS WITHDRAWN THIS SESSION

* **"The closed `PROPERTY_RELATIONS` is the real defect behind `eye_color`"** —
  wrong, asserted repeatedly. 50 edges use property relations against 9,602
  open-vocabulary ones, and open-vocab facts render identically. It is a storage
  style, not a gate. **Do not extend that list** — adding corpus-specific words
  is the closed vocabulary G45 removed.
* **"59 of 73 failures were the answering model's fault"** — computed on the
  broken judge run. Dead.
* **"The vector leg is dead"** — no. `_apply_rrf` stamped only the first leg;
  `{'bm25': 4}` is really `{'bm25+vector': 4}`.

## 9. ⚑ EXPERIMENT ARTIFACTS MUST BE RE-SCORABLE WITHOUT A RE-RUN

The saved probe artifacts existed so scoring would not need retrieval re-run —
and they recorded only `leg / tokens / score / text`. **No fragment identity.**
So when the crediting rule changed today, nothing on disk could be re-scored and
a full GPU pass was the only route. Information that was never written down
cannot be recovered.

Both harnesses now persist `source_batch_id` and `origin_batch_ids` per
fragment, and `answer_probes` records real fragment entries rather than a bare
list of leg names. A metric change is now a JSON re-read.

**`run_meta()` was missing from both scripts written today** — CLAUDE.md makes it
a standing rule for every experiment artifact, and `score_typed.py` was the only
one that had it. Added to both.

⇒ **Before any experiment run, ask what a future metric change would need, and
record that — not just what today's metric reads.** The generator now checkpoints
every call for the same reason ([TRAPS #36](TRAPS.md)); two hours of GPU were
lost to a script that wrote only at the end.

## 10. HOUSEKEEPING

* **History was scrubbed before the first push.** Personal specifics were
  committed to tracked docs earlier in the session; six commits were rebuilt so
  neither blobs nor **commit messages** carry them — a cleanup message that
  describes what was removed is itself a signpost. CLAUDE.md now carries the
  structure and a **user-eyeball gate** before any commit with private detail.
* **CLAUDE.md** gained the v1/v2/v3 scheme — v2 and v3 numbers are not
  comparable and nothing in the output says which you hold.
* **[FEATURE_INVENTORY.md](FEATURE_INVENTORY.md)** has a 2026-08-16 corrections
  section; eight entries went stale in one day.
* `/tmp` is cleared aggressively here — three run logs vanished mid-session.
  **Write run output under `experiments/curation_files/`, not `/tmp`.**

## NEXT — IN THIS ORDER

0. **Run the full 372-probe re-count** — patched, ready, ~10 min, deterministic,
   no API. Cheapest real number available and it needs nothing else first.
1. **[G50](ROADMAP.md#g50) auto-apply policy** — needs a spec. Shape settled by
   [getzep/graphiti#1728](https://github.com/getzep/graphiti/issues/1728):
   **the model narrows, it never authorises.** ICE runs a 2.5 GB local judge, so
   this matters more here than there.
2. **Regenerate temporal + the 48 lost codex probes** (Ollama, `qwen3.6:27b`;
   the generator takes its endpoint from env, so no code change).
3. **Re-seed one arm with qwen3**, carrying all six fixes.
4. **Re-score and re-judge** once the API resets. ⚠ Recall will move on the
   METRIC FIX alone, so the delta is not a system improvement. The honest
   comparison after a one-arm re-seed is **new-pipeline qwen3 vs old-pipeline
   qwen3** on the same probes — both stores exist, arm 1's snapshot is the
   "before", and that answers "did the six fixes help?" rather than re-running
   a model bake-off.
5. **Wire timeline provenance** (207 fragments still uncredited).
6. **Finish the ledger** (only `evidence` and `recent_turns` are evictable).

**⚑ Do not write this file, or close a session, without the user saying so.**
