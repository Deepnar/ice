# Handoff — 2026-08-16, ~13:40 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Overwritten every session,
committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

> **⚑ READ THIS BEFORE TOUCHING ANYTHING.** Two items on this list are
> unfinished and one of them **blocks pushing**. Three roadmap entries were
> found wrong about their own subject, and four hypotheses about the graph were
> tested and disproved. Starting work without reading this will reproduce
> conclusions that were already killed.

---

## TOLD → DID

**Told:** re-seed both arms on the fixed pipeline, validate the pipeline fixes,
score the typed probes, read the output, re-run the model comparison.

**Did:** all of it — and **the re-seed nearly ran on a pipeline that would have
corrupted a third of it.** A smoke test before the long run caught a
`ForeignKeyViolation` in G44's node promotion; the guard it needed fired **98
times** across the real run. Then the measurements went four-for-four against my
own hypotheses, which is the story of the day: every theory about the sparse
graph was wrong, and the code that fixes it already existed.

## ⚑ START HERE — THE STATE OF THE WORLD

| | |
|---|---|
| **Store** | **arm 2 (`fixed-gemma4-e4b`) restored and live** — 7,949 entities / 7,052 edges / 293 turns. Both arms snapshotted. |
| **Git** | **97 commits ahead of `origin`. NOT PUSHED, and pushing is BLOCKED — see below.** Working tree clean. |
| **Tests** | 347/347 regressions · `test_codex_write_path.py` 32/32 · `test_maintenance_agent.py` 45/45. |
| **Alembic** | `505f12031434` (unchanged). |

### ✅ HISTORY WAS SCRUBBED BEFORE THE FIRST PUSH OF THIS CYCLE

Sensitive personal specifics were written into tracked docs and committed
earlier in the session. They were removed from the working tree, and then the
six affected commits were **rebuilt** (`git cherry-pick` + `--msg-filter`, scoped
to the unpushed range only so no published SHA changed) so that no blob and no
commit message in history carries them. Verified before pushing: 0 matches
across every blob in the range, 0 in every message.

**The structure that prevents a repeat is a standing rule in CLAUDE.md**
("PRIVATE DETAIL IN A TRACKED DOC"): the tracked doc carries the technical claim
and stands alone, specifics go in `docs/PRIVATE_CONTEXT.md` (gitignored, verified
with `git check-ignore` **before** the file is written), the tracked doc points
at a marker, and **the user eyeballs the exact lines before any commit.**
⚠ A cleanup commit message that *describes what was removed* is itself a
signpost — that is why the messages were rewritten too, not just the diffs.

## ⚠ FOUR HYPOTHESES ABOUT THE SPARSE GRAPH, ALL WRONG

The graph is **80.5% degree-1** (arm 2) / **69.6%** (arm 1) — worse than the
65% measured pre-fix. Tested and killed, in order:

1. **Entity duplication** — no. Only 10–14% of names are near-duplicates at
   cosine 0.90; merging every one leaves the problem.
2. **A missing resolver** — no. `maintenance_agent` has one, with an LLM judge
   and caps, better rails than the thing I was about to propose building.
3. **The job never ran** — it ran. Degree-1 moved **80.5% → 80.5%**.
4. **A starved detector** — a backlog of 69 `codex_reconciliation` items ate the
   `agent_max_scanned=50` budget, but draining it only revealed the real cause.

**THE ACTUAL CAUSE ([G50](ROADMAP.md#g50)):** Tier 2 merges are **proposed into
the review queue and never auto-applied**. 7 proposals, 0 applications, against
**1,094 candidate pairs** and a cap of 10/run — roughly 200 human review
sessions. Correct for one user; impossible for a product. And **Tier 0, the
auto-apply channel, was structurally dead**: it looked for a casefold collision
on `canonical_name`, which is UNIQUE.

## WHAT SHIPPED

| item | was | now |
|---|---|---|
| **[G44](ROADMAP.md#g44) promotion** | deleted the endpoint of the triplet writing it; cost that turn its codex **and** procedural extraction | `protect_ids` exemption + SAVEPOINT. **98 saves** across 586 turns |
| **[G44](ROADMAP.md#g44) numbers** | `8`, `19`, `2023`, `9.65`, `3/80` refused — **94 of 2,273** | numerals are entities; pronouns still refused |
| **[G49](ROADMAP.md#g49) clauses** | `taking_what_youve_built_and_what_youve_understood` was an edge | >5 words ⇒ demoted, never dropped |
| **[G50](ROADMAP.md#g50) Tier 0** | dead channel | `merge_key()` — 20 real groups, no model, no review |
| **G16 / G29 / G49** | enrichment read private turns; reflection hardcoded 0.85; codex hashed its own key | fixed, all three |
| **instruments** | harvest read a different probe set than the scorer, and crashed on typed probes | `--probes`, `gold_turns` lists, `probe_type` in every record |
| **[Z2](ROADMAP.md#z2)** | no way to test the half that matters | `answer_probes.py` — retrieve → assemble → **answer** |

## THE NUMBERS, AND WHAT THEY CAN'T SAY

Five metrics, never averaged. `n` matters more than the value:

```
                    n     arm2 gemma4:e4b   arm1 qwen3:4b
episodic_lookup    232        0.664            0.591      ← the only solid one
codex_multihop      89        0.535            0.538
summary_synthesis   40        0.361            0.324
procedural          40        0.000            1.000      ← ARTIFACT, see below
temporal            19        0.211            0.211      ← below the MDE
```

* **procedural is not a measurement.** Arm 1 had exactly ONE active pattern, so
  it returned on every probe and scored 1.000; arm 2 had none and scored 0.000.
  The metric cannot tell "found the right habit" from "one habit exists".
* **codex_multihop is not measuring the graph.** Codex gets **0.1–9% of the
  token budget**, usually one fragment; episodic takes 88–99%.
* **Model choice is genuinely split and unresolved.** gemma4:e4b wins episodic
  and pattern quality; qwen3 wins graph connectivity (69.6% vs 80.5% dead ends)
  at **2.5 GB against 9.6 GB**. `answer_probes.py` exists to settle it and
  **has never been run**.

## ⚑ THREE ROADMAP ENTRIES WERE WRONG ABOUT THEIR OWN SUBJECT

* **[G43](ROADMAP.md#g43)** — `november --eye_color--> golden black` was **not**
  an invented value. The grounding rule it shipped is measured working (0 of 463
  property edges ungrounded at full confidence) but was never the fix for that
  example. Real defect: `PROPERTY_RELATIONS` is closed on the **handling** side.
* **[G44](ROADMAP.md#g44)** — its name rule was destroying true facts.
* **[G50](ROADMAP.md#g50)** — replaces the four dead hypotheses above.

**And a subagent made the same class of error**, calling grounded output a
fabrication and letting that decide its model ranking. **[TRAPS #31b](TRAPS.md)**
is the rule: any claim that the model made something up gets a corpus query
before it is stated, and that binds subagents too.

## NEW: [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md)

**~433 features**, each with a verified `file:line`, its setting, that setting's
default, and whether it is **ON by default**. Plus **40 dead-or-inert** items.
Six that matter, three now fixed. Still open:

* **The context ledger's eviction is inverted** — only `evidence` and
  `recent_turns` are ever registered as evictable; slots, bookmarks and
  summaries are welded into one `system_prompt` block, which is `NEVER_EVICT`.
  **User decided: finish it** (label the blocks), do not delete it.
* **Per-leg attribution is read only inside the coverage path, which is off** —
  so a default run cannot say which leg did the work. **Decouple it**; it is
  three lines, and coverage (C16) is a stopping rule, unrelated to leg forcing.
* `decide_representation` returns `inject_raw: True` on every branch.

## NEXT — AND THE ORDER MATTERS

1. **Re-seed.** The numbers fix and the clause fix exist in code and in
   **neither snapshot** — every stored measurement predates them.
2. **Run `answer_probes.py` on both arms.** The only fair test of the background
   model, and the only one that sees whether the context was usable.
3. **[G50](ROADMAP.md#g50) needs a spec** — which merges may be auto-applied
   under what structural guard. Shape settled by
   [getzep/graphiti#1728](https://github.com/getzep/graphiti/issues/1728):
   **the model narrows, it never authorises.**
4. **Finish the ledger.** Attribution is done — `producing_legs` now rides on
   `leg_budget_share`, which fires on every retrieval.

## ⚑ THE UNLOCK NOBODY HAS BUILT

Opening `PROPERTY_RELATIONS` needs a way to tell "this object is a **value**"
from "this object is an **entity**". **Two structural predicates were tried and
both failed** — target degree and never-appears-as-subject both score `say`,
`suggests` and `states` exactly like `size` and `description`, because in an 80%
degree-1 graph nothing ever becomes a subject.

⇒ **The signal has to come from the text, not the graph: have the extractor
label the object type, and enforce it with constrained decoding** (Ollama
supports it). ⚠ Not free — one benchmark measured structured outputs *reducing*
extraction validity 51% → 37%, so it is a "test it on our corpus" change.

## WHERE ICE IS AHEAD OF GRAPHITI, MEASURED

Worth knowing, and it is publishable: their edge invalidation runs unscoped and
**41% of 3,950 facts carry an `invalid_at`**, 3 of 4 hand-audited being
collateral. ICE scopes contradiction to the entity pair, deterministically:
**1,253 of 1,350 retirements (93%) have a live successor**. Retrieval also
filters `valid_until IS NULL`, so retired facts are never served — their
separate open issue. The gap runs the other way on **entity resolution**:
`get_or_create_entity` is exact-match + alias only, while relations have
`canonical_relation`'s three-tier ladder. That asymmetry is unexamined, not
decided.

## OPEN, AND ONLY HERE

* **v1 / v2 / v3 is now in CLAUDE.md.** v2 numbers and v3 numbers are not
  comparable and nothing in the output says which you are holding.
* **`docs/PRIVATE_CONTEXT.md` exists and is gitignored.** It holds the case
  behind TRAPS #31b, and names a design gap worth tracked work once phrased
  corpus-agnostically: **ICE stores an assistant's hypothesis about the user as
  a fact about the user, at full confidence**, with nothing marking which is
  which.
* **Relation canonicalisation works but converges to several attractors** —
  `have→has` merges correctly on demand (0.9469) yet the store holds `has` 491,
  `have` 78, `had` 60. Likely cause, unconfirmed: the known-relation set is read
  **once per turn**, so same-turn variants cannot see each other.
* **Procedural can never activate** — 0 of 58; needs real pattern pairs to
  calibrate, not a guess.
* **97 retirements (7%)** have no live successor and only 3 negated edges exist
  to explain them.
* **MCP and REST have diverged** — neither is a superset. MCP is *not*
  read-mostly: it reaches codex extraction through bookmark/note/document
  ingest.
* One dubious merge proposal is sitting in the review queue:
  `devstral-small-2` ← `devstral-small`.
* `experiments/curation_files/` and `docs/PRIVATE_CONTEXT.md` are gitignored and
  hold personal material. **Never commit them.** `data/` is 871 MB — never
  `git add data/` blind.

## WHEN DONE

Propagate per the roadmap's rules, update the docs the change invalidates in the
**same** session, then rewrite this file — carrying the NEXT above into
TOLD → DID — and commit it last.

**⚑ Do not write this file, or close a session, without the user saying so.**
