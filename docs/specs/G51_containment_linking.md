# G51 — The graph is disconnected because entities were never LINKED, not because duplicates were never merged
Assumes decided specs: G50_entity_consolidation.md, D1_D2_maintenance_agent.md, G45_open_relation_vocabulary.md

## 1. Decisions

**D0 — What this item exists to fix.** Measured on arm 1 (`fixed-qwen3-4b-instruct`,
8,280 entities / 9,662 live edges), 2026-08-17:

| | |
|---|---|
| degree 0 (no live edge at all) | **608** (7.3%) |
| degree 1 | **5,611** (67.8%) |
| **dead ends (0 or 1)** | **6,219 (75.1%)** |
| short entity names appearing inside a longer entity name | **5,616 pairs** |
| …of those, joined by a live edge | **129 (2.3%)** |
| …**not connected at all** | **5,487** |

The whole item in one cluster — `validation`, degree 8, and eleven satellites
**none of which are connected to it**: `emotional validation`, `external
validation`, `seeking validation`, `a lack of validation`, `false validation`,
`needs validation`, `get validation` (degree 0), `validation loss`, `whose
validation`, `only validation`, `more validation`.

⇒ **The extractor mints a new entity per surface phrase and never relates it to
its head noun.** That is what produces isolated triplets instead of a graph, and
no merge policy addresses it.

**D1 — Containment produces a LINK, not a merge.** This is the decision the
whole spec turns on, and the codebase already tried the other way:
`get_or_create_entity`'s promotion block (`codex_extractor.py:1012`) folds a
stored generic into an arriving specific — `plan` into `master plan`. It is
**off by default** (`codex_node_promotion`) and fired **0 times in 586 turns**.
Merging is the wrong operation because the pairs are usually not the same thing:
`validation loss` is an ML training metric, `emotional validation` is a
psychological concept, `a lack of validation` is the negation. A merge destroys
three distinctions to gain one node; a link keeps all three and connects them.

**D2 — The model classifies FOUR ways, and a structural guard authorises the
write.** This is where the model auto-merger the user asked for actually belongs
(2026-08-17). Over a cosine pair the model is asked "same or different?" about
`two sagas` / `four sagas` — a question the embedding already failed and the
strings do not help with. Over a containment pair it is asked a well-posed
question with visible structure:

| verdict | example (live, arm 1) | action |
|---|---|---|
| `merge` | `podar` / `rn podar school santacruz` | hand to G50's `merge_entities` |
| `link` | `emotional validation` / `validation` | write one edge, relation from the model |
| `unrelated` | `validation loss` / `validation` | memo, never re-ask |
| `junk` | `what the flaw` / `flaw` | memo + report to G44; **never linked** |

⚑ **`junk` IS THE VERDICT THIS DESIGN CANNOT SHIP WITHOUT, and it was missing
from the first draft.** A large share of satellites are not entities but
sentence fragments the extractor minted: `what the flaw`, `began flaw`, `after
flaw`, `when orien`, `named orien`, `to krishna`, `way krishna`, `everything
krishna`. Given only three options a model classifies every one of these as
`link` — they *are* about their head — and the pass would cement fragments into
the graph as structure. **Degree-1 share would fall sharply, the headline number
would look excellent, and the graph would be worse.** That is [TRAPS #37](../TRAPS.md)
in a new costume: an improvement a broken implementation produces just as well.

⚠ **Some HEADS are not entities either.** `first` (degree 29) carries 39
satellites — `first brick`, `first author`, `first message`, `first two volumes`
— which share an ordinal and nothing else. A head whose satellites have no
common referent must be rejected wholesale, not linked satellite by satellite.
The prompt therefore asks about a **cluster**, not a pair (D7), which is the
only framing in which this is visible.

⚠ **Do NOT trust a hand-written lexicon to pre-filter fragments.** One was tried
2026-08-17 (wh-words, function words, a verb list) and reported 10.8% fragments;
reading its own output shows it passing `get validation`, `needs validation` and
`to stress or trauma` as plausible links. **10.8% is a floor, not an estimate.**
A rule keyed on which words appear is exactly the style-dependent bet CLAUDE.md
forbids, and it fails the same way here. The true share comes from the
USER-REQUIRED hand-read sample in §5, and nowhere else.

⚑ **The model narrows; it never authorises** (getzep/graphiti [#1728](https://github.com/getzep/graphiti/issues/1728)).
Candidates are found deterministically by string containment — the model never
scans the store, it only answers about pairs handed to it — and every verdict
passes a structural guard (§2) before anything is written.

**D3 — A `merge` verdict is downgraded to `link` unless it ALSO passes G50's
deterministic test.** The model may propose a merge; it may not cause one on its
own word. If `difference_kind(a, b) != "merge"`, the pair is linked instead. So
`podar` / `rn podar school santacruz` becomes a link, not a merge, until an
alias or a `merge_key` match justifies more. **Losing a merge is recoverable;
losing a distinction is not.**

**D4 — The relation comes from the open vocabulary, not a fixed list.** G45
removed the closed relation list and CLAUDE.md forbids re-introducing one, so
the model returns a relation string and it goes through `canonical_relation`'s
existing three-tier ladder like any other. **Do not add an `is_a` constant.**
The common case will converge to `is_a` / `kind_of` on its own; if it does not,
that is data about the corpus, not a bug to hard-code away.

**D5 — `unmerge_entities()` ships BEFORE this item's writes are enabled.** G50
D5 defers it on the grounds that deterministic merges are re-derivable. That
argument does not extend here: a model-authorised link is not re-derivable from
the two names. Reversal is a prerequisite, per the user's inverse-ranking
principle (2026-08-17) — undo scales with how much judgement authorised the
write.

**D7 — Candidates are batched BY HEAD CLUSTER, not pair by pair.** Measured
funnel on arm 1: 8,280 entities → **5,616** raw containment pairs → 5,603 after
dropping stopword/relation-name heads (that filter removes only **13** — it is
not where the noise is) → **5,474** after dropping the 129 already joined →
**1,558 distinct heads**, median cluster size 2, max **47** (`flaw`), with
**412** satellites at degree 0 that would gain their first edge.

One call per head, all its satellites in the prompt. Three reasons, and the
second is the important one:

1. **Amortises the call** — `flaw`'s 47 pairs are one request, not 47.
2. **⚑ Gives the model contrastive context, which is what the hard cases need.**
   `validation loss` is only recognisable as a different *sense* of the word
   when it appears beside `emotional validation`, `external validation` and
   `seeking validation`. Asked in isolation it looks exactly like a link.
3. **Produces a connected neighbourhood per call** rather than edges scattered
   across the graph, so partial progress is still structurally useful.

**Cluster order: by `head_degree × satellite_count`, descending** — attach the
most orphans to the largest existing hubs first. Top of that ranking today:
`flaw` (degree 137, 47 satellites), `orien` (113, 22), `krishna` (84, 22),
`observer` (101, 18), `universe` (72, 21), `lethe` (71, 21). `agent_link_pairs_per_run`
becomes a **cluster** cap, default **5**.

**D6 — Degree-0 entities are NOT deleted by this item.** 608 of them contribute
nothing to graph traversal today, and deleting them is tempting. But this spec
is about to give many of them a first edge, which changes the population.
**Measure after the linking pass, decide then**, and ask the user — a designed
entity that turns out unused is evidence, not permission (CLAUDE.md).

## 2. Algorithm & data model

**Stage 1 — candidate generation (deterministic, no model, no embedding).**

```python
def containment_candidates(db, *, max_short_tokens=2, max_long_tokens=5):
    """Short entity name occurring as a whole-token span inside a longer one.

    Whole-token only: 'val' must not match 'validation'. Anchored at a token
    boundary on both sides, which is what `f" {short} " in f" {long} "` gives.
    """
```

Guards, each of which removes a measured false candidate:
* skip `short` shorter than 4 characters — `u.s`, `ai`, `ml` generate noise
* skip when `short` is a stopword-only name (`with`, `the`, `outer`) — these are
  G44 category errors and are **reported**, not linked (§4)
* skip pairs already joined by a live edge in either direction (129 today)
* skip pairs with a `merge_rejected` or `link_rejected` memo
* cap per run at `agent_link_pairs_per_run` (new setting, default **50**)

**Stage 2 — the model classifies.** One call per pair, batched 10 per prompt.

```
Two entities from a personal knowledge graph. One name contains the other.

  A: "emotional validation"      (appears in 3 facts)
  B: "validation"                (appears in 8 facts)

Answer with exactly one word:
  merge      — A and B refer to the SAME thing, one is an abbreviation or
               alternate spelling of the other
  link       — A and B are DIFFERENT things, and A is a kind/instance/part of B
  unrelated  — A and B are different things that merely share a word,
               including the same word used in two different senses

If B is a negation, opposite, or a different technical sense of A, answer
unrelated.
```

⚠ **The last line is load-bearing.** Without it `a lack of validation` and
`validation loss` classify as `link` — both are "about" validation. The failure
is silent and writes a false relation.

**Stage 3 — the structural guard (deterministic, authorises).** A verdict is
acted on only if **all** hold:

| guard | why |
|---|---|
| `difference_kind(a,b) != "reject:*"` | G50's typed rejections bind here too — a digit/gender/tense/permutation difference is never a link |
| both entities live (`merged_into` NULL) | concurrent runs |
| the proposed edge does not already exist in either direction | idempotency |
| the edge would not create a self-loop after any pending merge | `merge_entities` expires self-loops; do not create them |
| for `merge`: `difference_kind(a,b) == "merge"` | D3 — the model may not authorise a merge |
| for `link`: `short` is not stopword-only | G44 category errors are reported, not wired in |

**Stage 4 — write.** `link` ⇒ one `CodexEdge`, `source_id` = specific,
`target_id` = general, `relation` = model's word through `canonical_relation`,
`confidence` = `pending`, `extraction_confidence` = `settings.codex_conf_inferred`
(new, default **0.6**), `source_batch` = the agent run id.

⚑ **The edge is `pending`, not `active`, and that is deliberate.** It was
inferred from two names, not read from a turn. `_apply_pileup` already expires a
pending edge that duplicates a live active one, so a later real extraction of
the same fact supersedes the inference cleanly.

**Schema.** No new table. One new `CodexEvent` type, `link_rejected`, mirroring
G50's `merge_rejected`.

## 3. Files & integration points

| File | Change |
|---|---|
| `src/workers/maintenance_agent.py` | new detector `_detect_containment_links` registered in the `JOBS`-adjacent detector map beside `duplicate_entities`; new applier `_apply_link`; Tier 2 for `merge` verdicts, **Tier 1** (auto + journalled) for `link` verdicts that clear the guard |
| `src/workers/codex_ops.py` | new `unmerge_entities(db, absorbed_id)` driven by the `CodexEvent` journal (D5) |
| `src/workers/codex_extractor.py` | none. **Leave the promotion block alone** — it is off, it is G44's, and deleting it is user-gated |
| `src/api/config.py` | `agent_link_pairs_per_run: int = 50`, `codex_conf_inferred: float = 0.6`, `agent_containment_linking: bool = False` (ships OFF; see §5) |
| `docs/FEATURE_INVENTORY.md` | new row, `On by default? NO` |
| `tests/test_g51_containment.py` | new, see §5 |

⚠ **Register the detector in the existing map; do not add a second agent loop.**
The runtime's `JOBS` is the source of truth for what runs and how often
(CLAUDE.md code map).

## 4. Edge cases & failure modes

| case | handling |
|---|---|
| `short` is a stopword-only entity (`with`, `outer`, `the middle`) | never linked. Emit `codex_entity_category_error` at WARNING with the name, and count them — this is G44 evidence, and burying it in a skip would lose the signal |
| `short` is a relation name minted as an entity (`leans_toward`, `has parents`) | same handling — detect by `"_" in name` or a leading auxiliary verb, report, never link |
| one name contains the other twice (`validation validation`) | whole-token match is idempotent; dedupe candidates by `(short_id, long_id)` |
| chain: `trauma` ⊂ `developmental trauma` ⊂ `complex developmental trauma` | link each adjacent pair; do **not** transitively close. Two edges, not three — the graph derives the third |
| model returns something outside the enum | count as `unrelated`, log at WARNING with the raw string. ⚠ Check the **rate** before shipping (CLAUDE.md: a fallback firing on 100% of calls is an outage in costume) |
| model is unreachable | detector returns `[]` and logs `_leg_degraded`-style; **never** default to linking |
| the pair spans two conversations | allowed. Entities are global; this is not a retrieval-scope decision |
| linking makes a hub even bigger | accepted and expected — `validation` is *supposed* to be a hub. G50's degree ceiling guards **merges**, not links, because a link re-attributes nothing |

## 5. Validation checklist

`tests/test_g51_containment.py` — standalone, live DB, own rows, never truncates,
LLM stubbed.

1. `containment_candidates` finds `('trauma','developmental trauma')` and does **not** find `('val','validation')` (whole-token guard).
2. Does not propose a pair already joined by a live edge.
3. Stopword-only short name (`with`) is excluded and emits `codex_entity_category_error`.
4. Stubbed `link` verdict writes exactly one edge, `pending`, specific→general, and is idempotent across two runs.
5. Stubbed `merge` verdict on a pair where `difference_kind != "merge"` writes a **link**, not a merge (D3).
6. Stubbed `merge` verdict on a `merge_key`-equal pair calls `merge_entities`.
7. Stubbed `unrelated` writes a `link_rejected` memo and a second run skips the pair.
8. Guard rejects a `link` when `difference_kind` returns `reject:digits`.
9. `unmerge_entities` restores edge endpoints and clears `merged_into` after a `merge_entities` round-trip; assert edge count and endpoints match pre-merge.
10. Chain case: three nested names produce two edges, not three.
11. Regressions: `tests/test_maintenance_agent.py`, `tests/test_codex_write_path.py`, `uv run pytest tests/smoke -q`.

**USER-REQUIRED — dry run before the flag goes on (~20 min).** Ships behind
`agent_containment_linking=False`. Run the detector in report-only mode over the
arm-1 snapshot, then **read 40 sampled verdicts by hand** and confirm the
`unrelated` calls are genuinely unrelated. The `validation loss` / `emotional
validation` distinction is the one to check — it is the failure this design is
most exposed to. Done = 40 verdicts read and the flag flipped, or the prompt
revised and re-sampled.

⚑ **Do not measure this with a presence metric.** "Did the graph gain edges" is
answerable by a broken implementation writing garbage. The number that means
something is **degree-1 share before and after** (75.1% today) **plus** the
hand-read sample above. TRAPS #37 is the worked case of a presence test reading
1.000 over a pool of one.

## 6. Look-ahead constraints

* **[G50](G50_entity_consolidation.md)** owns `difference_kind` and `merge_key`;
  this spec imports both and must not fork them.
* **FINAL / Z1** — this changes edge counts substantially. A store seeded before
  it is not comparable on graph metrics; record which side a snapshot is on.
* **G48b leg attribution** — inferred edges carry the agent run id as
  `source_batch`, **not a turn batch id**, so a codex fragment built from one
  cannot be credited to a gold turn. ⚠ Recall may *drop* when this ships, purely
  because inferred edges displace attributable ones in the budget. Report it as
  such; it is not a regression.
* **H1 cross-conversation retrieval** — linking creates cross-conversation edges
  for the first time at volume. H1's measurement should postdate this.

## 7. Traps

* **⚑ "Merge them, it's simpler."** The obvious simpler version, already built
  (`codex_node_promotion`), already off, and it fired 0 times in 586 turns.
  `validation loss` / `emotional validation` / `a lack of validation` are three
  different things sharing a token.
* **⚑ "Add an `is_a` relation constant."** That is the closed vocabulary G45
  removed and CLAUDE.md's invariance rule forbids. Let the model's word go
  through `canonical_relation`.
* **"Transitively close the chains."** Produces `complex developmental trauma
  --is_a--> trauma` alongside the two real edges, inflating both edge count and
  fan-out while adding nothing traversal cannot derive.
* **"Link everything above the containment test — it's deterministic, so it's
  safe."** Deterministic *detection* is safe; the **classification** is not, and
  5,487 unreviewed inferred edges would be the largest single write this graph
  has ever taken. Hence the flag, the cap of 50/run, and the hand-read sample.
* **"Fan-out will finally drop."** It should — that is the point — but the
  roadmap's 80.5% figure came from cosine nearest-neighbours, an instrument
  shown on 2026-08-17 to be a poor proxy for identity. **Re-measure degree
  distribution directly; do not compare against the cosine-derived number.**
