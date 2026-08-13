# G45 — Open the relation vocabulary

**Status:** decided, ready to implement · written 2026-08-13
**Assumes decided specs:** `G_mechanical.md` (G32/G34/G35 context)
**Item:** [ROADMAP.md#g45](../ROADMAP.md#g45)

---

## 1. The problem, in one measurement

`normalize_relation()` maps a model-emitted relation onto a closed 197-word list
(`ALLOWED_RELATIONS`) and returns `None` for everything else. `extract_triplets`
then **discards** those triplets.

Measured on the 293-turn store, 2026-08-12:

* **1,259 relations destroyed** in a single seed, across 3 conversations.
* Real facts among them, verbatim from `codex_relation_gaps`:
  * `i --didnt_get--> csi` — true, and gone
  * `my father --t_get--> first` — an apostrophe split `didn't get` into `t get`
  * `binary --produce--> binary outcome`
  * `user --negated--> club` — the relation is the literal string `negated`
* Across the eight-arm run: **14,091 dropped, 3,244 distinct candidates**, with
  `main --is--> X` missing **875 times**, corroborated by two models independently.

Neither branch of the existing choice is acceptable: dropping keeps 32%, and
forcing the model to pick from the enum keeps 100% but ~78% of those are wrong.
**The closed vocabulary is itself the defect.**

## 2. What changes

**One function decides, and it stops saying "no".** `normalize_relation` becomes
a *canonicaliser* rather than a *gate*:

1. **Exact vocabulary hit** — unchanged, returns the canonical word. (Cheapest,
   and it keeps every existing relation stable.)
2. **Canonicalise against relations ALREADY IN THE GRAPH.** Embed the incoming
   relation, compare against the distinct relations already stored, and reuse the
   existing string above `codex_relation_canonical_threshold`. This is the
   write-time drift control Graphiti uses, and it is the ladder [G32](../ROADMAP.md#g32)
   already designed — moved from match-time rescue to write time.
3. **Otherwise, ACCEPT IT AS NEW.** A relation nobody has used before is a new
   relation, not an error.

**Nothing is torn out.** [G34](../ROADMAP.md#g34) (relation fit) needs *an
embedding per relation*, not a closed list — `_relation_gloss_cache` builds that
from `ALLOWED_RELATIONS` today and will build it from the union of
`ALLOWED_RELATIONS` and `SELECT DISTINCT relation FROM codex_edges`. G35 reads
relation fit, so it inherits the fix for free.

## 3. The inversion hazard, and why we do not fuzzy-match our way past it

The harvested candidates contain active/passive pairs that arose spontaneously:
`built`/`built_by`, `creates`/`created_by`, `makes`/`made_by`. Any similarity
measure will score these near-identical, and collapsing them **writes the fact
backwards** — `A built_by B` becoming `A built B` inverts subject and object.

**Rule: two relations may not be canonicalised together when one is the
passive/inverse form of the other.** Implemented as a cheap deterministic guard
(`_by` suffix, `is_`/`was_` prefix) checked BEFORE the similarity test, not as a
threshold to be tuned. A guard that is sometimes skipped is not a guard.

## 4. What it costs, stated plainly

* `codex_relation_gaps` loses its purpose — no drops means no ledger. The 3,244
  harvested candidates become a **one-time seed** for the canonicalisation table
  rather than a backlog.
* **Traversal gets noisier** until node promotion catches up: more distinct
  relation strings means a thinner spread per relation.
* The enum decision that [Z2](../ROADMAP.md#z2) owns **dissolves** — there is no
  enum to decide about.

## 5. Not in this item

* **Node promotion** — repairing generic nodes already stored. It is
  [G44](../ROADMAP.md#g44)'s second half and is specified there.
* **Bi-temporal edges** — tracking when ICE *learned* a fact, as opposed to when
  the fact was true. `codex_edges` today has `valid_from`/`valid_until` only
  (verified by query, 2026-08-13). Real, wanted, and a migration — separate item.

## 6. Validation

**Not "does it run".** The check is that the destroyed facts survive:

1. Re-run extraction over the same turns and assert `i --didnt_get--> csi` (and
   the other verbatim examples in §1) reach `codex_edges`.
2. Assert the **drop rate falls to ~0** and `codex_relation_gaps` stops growing.
3. Assert `built` and `built_by` remain **two distinct relations** — the
   inversion guard's regression test.
4. Assert [G34](../ROADMAP.md#g34) relation-fit still scores, with the vocabulary
   now sourced from the graph — i.e. the feature did not silently go inert.
5. Count distinct relations before/after. A number that explodes (say >1,000 on
   293 turns) means canonicalisation is not binding and the threshold is wrong.
