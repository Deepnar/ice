# The reseed — plan, and the clean break

**Written 2026-08-25. The dated execution claim below is historical.**

**v3 correction, 2026-09-29:** the agreed development corpus is 1,471 ordered
turns from three full conversations, but `seed_store.py` still imports the old
293-turn curated-checkpoint loader. It also writes rows directly before calling
post-flight, rather than running each historical prompt through v3 preflight
and then supplying its recorded answer. `snapshot.py` saves only 14 hardcoded
tables and omits current source-backed graph and note tables. Thus this plan is
**not executable as a complete v3 reseed yet**. The required repair and
verification order is in [the current harness audit](../reviews/2026-09-29-v3-reseed-harness-audit.md).
Local `gemma4:e4b` remains the general background pin, NuExtract3 the separate
Codex extractor, cloud `gpt-6-luna` the default *probe* answerer, and a cloud
answer judge needs current answer-pair calibration. The August tier/delete
decision and 1,000–1,500-turn sizing notes below are historical proposals,
not permission to delete v3 facts or call a partial seed complete.

**v3 execution contract, 2026-09-29:** replay the three full conversations
in source order with their original timestamps. Before each recorded reply is
stored, prepare that user's prompt through shared v3 preflight; then write the
existing reply with source-role spans and run the real post-flight chain. Log
candidate, selected and final-prompt evidence separately, including source
IDs, legs, token costs, gate decisions and failures. Periodic jobs need an
explicit recorded cadence and failure ledger, not a silent end-of-run catch-up.
The unified 618-probe catalog carries `split_turn` for the 444 typed probes,
although `typed_probes.json` does not. **The original typed cutoff equals the
latest gold turn on all 444 rows.** Those immediate prompts are diagnostics,
not a long-term-memory quality test: their gold can still be in the 40-turn
recent window. An optional delayed typed panel asks at original cutoff + the
live recent-window maximum; 378 fit the selected histories and 66 do not. The
default panel uses existing questions at later section checkpoints. Capture
each prompt before any later turn is inserted, then validate its expected
answer against source and intervening turns before cloud scoring. Thirty-two
non-typed curated rows have a split beyond the selected full conversation;
quarantine them until their cutoff is corrected. A final-store scorer is
end-of-history only, never a chronological score. Save a complete,
identity-checked snapshot including
claims, notes and all source links before tuning; keep answer generation and
cloud judgment separate from recorded-response reconstruction.

**Existing long-term checkpoint pool, recovered on 2026-09-29:** the unified
catalog also holds 93 mature and 81 curated questions for the selected three
histories. Their `split_turn` is a designed checkpoint, unlike the typed file's
latest-gold proxy. The older `derived_gt.json` supplies source-turn mappings for
a subset: 11 questions already have an in-corpus source at least 40 turns
before their checkpoint (six curated, five mature). They are scheduled at
their native cutoffs. The remaining
checkpoint questions lack reliable gold mapping or chronology and need source
labeling; do not silently convert an expected answer into a gold turn. The 11
derived mappings themselves still require source-support review.

**Source-first checkpoint expansion:** `generated_probes.json` has 592 older
questions with a known original gold turn and evidence quote. After remap to
the selected histories, exact quote verification, a corrected question-only
ambiguity screen, and placement at the first real checkpoint at least 40 turns
later, 113 are candidates. The ambiguity weights are fitted only on history
available at each checkpoint. Together with the 11 native mapped questions,
these 124 form the default existing-question panel. They span 11 checkpoint
times; the 1,119-turn conversation has only 14 candidates, all at its final
checkpoint, so this panel cannot establish retention across its earlier
sections. The native catalog spans 39 in-history checkpoint times; 131
additional questions need source-turn review, and 32 curated cutoffs exceed
the selected history. The private source-review packet ranks possible turns
for navigation, not as labels. Catalog IDs repeat across files; the replay
uses unique stable IDs and preserves each old ID as metadata.

**Implementation checkpoint (not a scored run):** `scripts/z1/seed_v3.py`
implements the sequential replay and captures full, vector-only and
recent-history prompt arms at each selected **long-term checkpoint** cutoff;
`report_v3_replay.py` reports
fragment-stage gold coverage, provenance and failure stages. The new snapshot
covers all ORM tables. A 3-turn disposable replay without periodic jobs passed;
one-turn disposable replay with all six due periodic jobs and matched arms also
passed. `build_longterm_label_review.py` now creates a private
gold-plus-intervening-turn packet balanced by conversation and checkpoint;
its current 30-candidate sample has no verdicts yet.
`build_checkpoint_source_review.py` prepares the 131 unlabeled native
questions for source-turn review. `answer_as_of.py` can plan cloud
Luna answers from frozen prompts and requires valid reviewed rows from the
matching packet before a full cloud campaign. No
full seed, new cloud answers, blind judge or quality claim has been made. The old
`seed_store.py`/final-store answer path cannot substitute for chronological
as-of scoring.

> **⚑ EVERYTHING BEFORE THIS RESEED IS DEAD DATA (maintainer, 2026-08-25).**
> *"lets just call ALL from before as we have no data, we are restarting ALL
> again."* Every store, every arm, every graph number in `PROVENANCE.md` before
> this date was produced by **`qwen3:4b-instruct` on the instruct prompt** — the
> configuration [G63](../ROADMAP.md#g63) measured at **15% correct with 30%
> reversed** and replaced. Nothing downstream depends on those arms any more.
> **Do not re-derive from them, do not compare against them, do not cite them
> except as method lessons.** [G71](../ROADMAP.md#g71) lists the specific
> findings that die with them.

---

## 1. The configuration being seeded

**v3 configuration correction, 2026-09-12:** template extraction is now the
source default. The August 27 handoff separates the general background model
from the extraction specialist; the reseed must preserve that separation.
The general background selection below is the decided run pin, not a claim
that `background_model_name` has a non-null source default.

| setting | value | why |
|---|---|---|
| `codex_extraction_mode` | **`template`** (source default) | specialist template path |
| general background model | **`gemma4:e4b`** (decided run pin) | summaries and other general background jobs; August 27 handoff |
| `codex_extraction_model` | **`hf.co/numind/NuExtract3-GGUF:Q8_0`** | dedicated extraction pin, e.g. template-filling facts rather than summarizing |
| `codex_extraction_ner_tier` | `background` (**NuNER**) | junk names 8.7% vs 19.5% |
| `codex_extraction_max_tokens` | `3000` | 1200 lost 30 of 60 turns |
| `codex_extraction_chunk_adaptive` | `True` | 550 split a 1,178-token turn into three |
| `codex_extraction_chunk_max` | `4096` | ⚠ ceiling unresolved — [G68](../ROADMAP.md#g68) |
| relation vocabulary in prompt | **NEVER** | 22% vs 60% correct |

**Verified end-to-end 2026-08-25** on 5 turns through `extract_codex`: 94 edges,
76 entities, **zero expiries**, canonicalisation fired (`has`→`have`,
`became`→`becomes`), merge key fired (`manhattan 5 lb book` →
`manhattan 5lb book`), entity gate refused `you` 9 times. The write path handles
this model.

## 2. Size — seed 1,000–1,500 DENSE turns, not 180

The corpus holds **71,256 turns**. The current arms hold 180–293.

⚠ **The median turn is 24 tokens.** Turn count is a misleading unit: the
**14,665 turns at or above 100 tokens carry 91% of all content**. Seed from
that population, not from a flat sample, or most of the run is spent extracting
from "ok" and "yes".

**Why bigger matters, and it is not vanity:** [G66](../ROADMAP.md#g66) asks
whether memory improves the ANSWER. At 180 turns a probe often has no answer in
the store at all — memory would look useless for reasons that have nothing to do
with its quality. The store has to be big enough that the right answer is
*there* before "did retrieval find it" is a fair question.

**Cost at ~13 s/turn:** 300 ≈ 1 h · 1,000 ≈ 3.5 h · 3,000 ≈ 11 h. Overnight is
affordable; start with 1,000 and extend if the first hour looks right.

## 3. ⚑ INSTRUMENT THE REJECT-BUT-KEEP QUESTION — it must be VISIBLE in the output

**The maintainer's standing irritation, and it deserves a real answer rather
than another year of "keep both".**

**Today:** grounded `0.9` · ungrounded `0.7` · rejected `0.35`, and **all three
are stored**. In the 2026-08-25 write-path test **79% landed at 0.35.**

**The fact that decides it:** `extraction_confidence` was measured
**uninformative about truth** — two full-coverage runs disagreed on which
direction it leaned. **If grounding does not predict truth, deleting rejected
facts removes true and false ones at the same rate** — that is not a filter,
it is shrinking the graph at random.

⚠ **But that was measured on qwen's output and is provisional ([G71](../ROADMAP.md#g71)).**

**So the reseed answers it:**

- [ ] Record the tier on every edge (already stored) **and surface the split in
      the run's own output** — grounded / ungrounded / rejected counts per turn
      and in total, printed, not buried in a table nobody queries.
- [ ] After the reseed, judge a **tier-stratified** sample: equal facts drawn
      from grounded and from rejected, blind, arm hidden.
- [ ] **Then pick a side and delete the loser:**
  - grounded measurably truer ⇒ grounding earns its place; **reject means
    delete**, and the 0.35 tier goes.
  - no difference ⇒ **the tier is theatre**; collapse to one confidence, and
    strip the retrieval trust floor that keys on it.

⚠ **The trust floor is why this is not cosmetic** — retrieval gates on
`extraction_confidence` today, so a meaningless tier is actively steering what
gets surfaced.

## 4. The standing rule this reseed adopts

> **⚑ PICK A SIDE ON EVIDENCE, THEN DELETE THE LOSER.**
> *(maintainer, 2026-08-25 — "lets start hacking and slashing away and picking
> sides, like how we have for nu thing")*

Applies twice over: to what a **feature** does — stop building one that caters
to every case — and to what **we** do — stop keeping both paths alive because
neither has been measured.

**[G63](../ROADMAP.md#g63) is the model for it.** Two models, two prompt shapes,
a blind round, a decision, and the loser is gone rather than parked behind a
flag.

⚠ **The one correction, and it is not a hedge — it is the failure this rule must
avoid.** The old open-relation default, later diagnosed under
[G51](../ROADMAP.md#g51), picked supersession without a measurement and silently
retired real multi-valued facts. The retained [PROVENANCE](../PROVENANCE.md)
documents eight components with seven retired; a separate “667 true facts”
number was repeated without a retained measurement and is withdrawn. ⇒ **the
rule is "commit after measuring", never "commit instead of measuring".** A side picked from a preference is that failure wearing this
rule's clothes.

## 5. Order of work

1. **Seed** 1,000–1,500 dense turns on the §1 config. Snapshot it.
2. **Judge** a blind sample — the new baseline, and the first number of the
   post-qwen era.
3. **Judge tier-stratified** (§3) and settle reject-but-keep.
4. **[G66](../ROADMAP.md#g66)** — codex leg ON vs OFF, answers judged blind.
   The reason the store had to be this size.
5. **[G71](../ROADMAP.md#g71) — re-measure the last two weeks.** Scope widened
   2026-08-25 by the maintainer: **not just the graph findings, everything
   accepted between 2026-08-11 and 2026-08-24.** G71 carries the register — 16
   PROVENANCE entries triaged: **3 survive, 3 already dead, 10 to re-measure or
   re-record.**
   - ⚠ **The retrieval half (08-13 → 08-20) cannot be redone before step 1.** A
     retrieval number measured over a 15%-correct graph is measuring the graph,
     not retrieval. That is why the reseed comes first.
   - **Drop rather than redo where the new extractor makes a finding moot** —
     e.g. the direction rule ([G59](../ROADMAP.md#g59)) targeted a defect
     NuExtract3 does not produce. Re-measuring it out of completeness is waste.

## 6. Open before step 1

- **The judge.** muse-spark (73%) has been down three days. ox-alpha is 67% but
  **37 s/call**; deepseek-v4-flash 60%. For G66's coarser question — "is answer A
  better than B" — 60–67% may be enough; for triplet-level truth it is not.
  [G65](../ROADMAP.md#g65).
- **The chunk ceiling** — 4096 leaves 34% of content split. Sweep deferred to
  the next test round ([G68](../ROADMAP.md#g68)).
