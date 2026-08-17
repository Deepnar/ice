# G50 — Entity consolidation: deterministic merges, typed rejections
Assumes decided specs: D1_D2_maintenance_agent.md, G45_open_relation_vocabulary.md

> **Scope note.** This spec owns the **merge** half only. The half that actually
> connects the graph — 5,487 missing containment links — is [G51](G51_containment_linking.md),
> and it is the larger item. Read G50's roadmap entry with that split in mind:
> the entry was written believing consolidation would fix fan-out, and it does not.

## 1. Decisions

**D1 — Auto-apply stays deterministic. No LLM authorises a merge in this item.**
Measured on arm 1 (`fixed-qwen3-4b-instruct`), every entity pair above cosine
0.90 — **2,247 pairs**, bucketed by what actually differs between the two names:

| kind of difference | pairs | share | disposition |
|---|---|---|---|
| `merge_key` equal — pure formatting | **42** | 1.9% | **MERGE, auto** |
| digits differ (`8 gb`/`4 gb`) | 379 | 16.9% | **REJECT** |
| spelled-out number (`two sagas`/`four sagas`) | 37 | 1.6% | **REJECT** |
| gender word (`his father`/`her father`) | 40 | 1.8% | **REJECT** |
| tense word (`is rare`/`was rare`) | 121 | 5.4% | **REJECT** |
| same tokens reordered (`shiva without brahma`/`brahma without shiva`) | 6 | 0.3% | **REJECT** |
| strict token superset (`a 3d diagonal plane`/`3d diagonal plane`) | 535 | 23.8% | **DEFER → G51** |
| other token substitution (`villainess`/`villain`) | 1,087 | 48.4% | **DEFER → G51** |

⚑ **Cosine magnitude does not track merge safety, and the top of the band is
where it is worst.** At ≥0.98 the population is a coin flip between the safest
merges in the store and the most destructive: `qwen coder 32b-q4`/`qwen coder
32b q4` (0.9970, safe) sits beside `two sagas`/`four sagas` (0.9889),
`qwen3-32b`/`qwen3-30b` (0.9860), `8 gb`/`4 gb` (0.9852) and
`gemma-4-e4b`/`gemma-4-e4b-q4` (0.9851 — base versus quantised, different
artifacts). ⇒ **Do not band by cosine and auto-accept the model at the top.**
High cosine means *the embedding cannot separate these*, and a small judge shown
two near-identical strings is under the same pressure — the two instruments fail
together rather than checking each other.

**D2 — The 583 REJECT pairs are dropped at detection: never queued, never shown
to a model, never shown to the user.** They are 26% of the candidate set and
every one carries a token that changes the referent. This is the throughput fix
the roadmap entry was reaching for, and it costs no judgement at all.

**D3 — Resolution also moves to write time** (user decision, 2026-08-17).
`get_or_create_entity` gains a `merge_key` tier after the exact-name and alias
lookups, mirroring `canonical_relation`'s ladder. The asymmetry the roadmap
called "unexamined" is resolved in favour of symmetry. This is the main event,
not an optimisation: the 42 deterministic merges are the entire safe set, and
at write time they are never minted in the first place.

**D4 — A rejection is recorded, so it is not re-derived every run.** Detection
re-proposes the same pairs on every pass otherwise; at 2,247 pairs that is the
real cost, not the merging.

**D5 — `unmerge_entities()` is NOT a prerequisite for this item.** User's
inverse-ranking logic (2026-08-17): undo need tracks how much *judgement*
authorised the write. Deterministic merges are reproducible from the two names,
so a wrong one is re-derivable rather than lost. ⇒ Undo becomes a **prerequisite
of [G51](G51_containment_linking.md)**, where a model authorises writes, and it
stops blocking this spec.

**D6 — `agent_dup_cosine_threshold` is not tuned, and not removed.** Raising it
concentrates the reject buckets (they live at the top); lowering it buys volume
in the defer buckets. It stays at 0.90 as a **recall** knob for G51's candidate
generation, and stops being a safety knob. Both roles were previously conflated.

## 2. Algorithm & data model

No schema change. One new pure function and one new lookup tier.

```python
# src/workers/maintenance_agent.py  (beside merge_key)

_WORDNUM = {"one","two","three","four","five","six","seven","eight","nine",
            "ten","first","second","third","fourth","fifth","half","quarter"}
_GENDER  = {"his","her","he","she","him","hers","himself","herself",
            "mr","mrs","ms","sir","madam"}
_TENSE   = {"is","was","are","were","has","had","have","will","would",
            "did","does","do","being","been"}
_SPLIT   = re.compile(r"[\s\-_:;,()\[\]{}\"'/\\.]+")
_NUM     = re.compile(r"\d+")


def difference_kind(a: str, b: str) -> str:
    """'merge' | 'reject:<why>' | 'defer' — deterministic, no model, no vector.

    Ordered most-specific first: a pair differing by BOTH a digit and a tense
    word is a digit rejection, because the digit is the stronger signal.
    """
    if merge_key(a) == merge_key(b):
        return "merge"
    ta = {t for t in _SPLIT.split(a.lower()) if t}
    tb = {t for t in _SPLIT.split(b.lower()) if t}
    if sorted(_NUM.findall(a)) != sorted(_NUM.findall(b)):
        return "reject:digits"
    diff = ta ^ tb
    if diff & _WORDNUM:
        return "reject:wordnum"
    if diff & _GENDER:
        return "reject:gender"
    if diff & _TENSE:
        return "reject:tense"
    if ta == tb:                       # same tokens, different order
        return "reject:permutation"    # TRAPS #26 entity-side twin
    return "defer"
```

⚠ **`reject:permutation` is load-bearing and must stay a rejection even though
it looks like the safest case.** `shiva without brahma` / `brahma without shiva`
is live in the store at cosine 0.9639. Sorting tokens to find merges — the
obvious simplification — merges converses. See G50 roadmap entry and TRAPS #26.

**Write-time tier** in `get_or_create_entity` (`codex_extractor.py:974`), after
the alias lookup and **before** the promotion block:

```python
    # Tier 2 (G50): normalisation-equal match. canonical_name is UNIQUE and
    # aliases mirror it, so the two lookups above are exact-string only —
    # 'gemma-4-e4b q4' and 'gemma-4-e4b-q4' both miss and both get minted.
    key = merge_key(canonical)
    hit = db.query(CodexEntity).filter(
        CodexEntity.properties["merge_key"].astext == key,
        CodexEntity.properties["merged_into"].astext.is_(None),
    ).first()
    if hit:
        if canonical not in (hit.aliases or []):
            hit.aliases = [*(hit.aliases or []), canonical]
        logger.info("codex_entity_merge_key_hit",
                    name=canonical, resolved_to=hit.canonical_name)
        return hit
```

`properties["merge_key"]` is written on every entity create and backfilled by
the migration below; it is indexed:

```sql
-- alembic revision: backfill + index the merge key
UPDATE codex_entities
   SET properties = jsonb_set(coalesce(properties,'{}'::jsonb),
                              '{merge_key}', to_jsonb(<computed>))
 WHERE properties->>'merge_key' IS NULL;

CREATE INDEX ix_codex_entities_merge_key
    ON codex_entities ((properties->>'merge_key'))
 WHERE (properties->>'merged_into') IS NULL;
```

⚠ The backfill cannot be pure SQL — `merge_key()` is Python. Compute in a
batched loop inside the migration (`op.get_bind()`, 1,000 rows per chunk),
**not** a single UPDATE over 8,280 rows.

**Rejection memo** (D4): a `codex_relation_gaps`-style row is overkill; use
`CodexEvent` with `event_type="merge_rejected"` on the lower-id entity, payload
`{"other_id": …, "kind": "reject:digits"}`. Detection skips a pair that already
has one. Cheap, journalled, and already the table the merge path writes to.

## 3. Files & integration points

| File | Change |
|---|---|
| `src/workers/maintenance_agent.py` | add `difference_kind()` beside `merge_key()`; `_detect_duplicate_entities` calls it on every cosine-channel pair — `merge` ⇒ Tier 0, `reject:*` ⇒ write the memo and drop, `defer` ⇒ hand to G51's queue (until G51 ships, keep today's Tier 2 review behaviour) |
| `src/workers/codex_extractor.py` | `get_or_create_entity` gains the merge_key tier above; every create path stamps `properties["merge_key"]` |
| `alembic/versions/*` | backfill `properties.merge_key` + partial index |
| `src/api/config.py` | no new setting. `agent_dup_cosine_threshold` docstring updated to say it is a **recall** knob, not a safety knob (D6) |
| `tests/test_g50_consolidation.py` | new, see §5 |

⚠ **Extend `_detect_duplicate_entities`; do NOT add a parallel detector.** It
already resolves both channels, the review-queue exclusion and the
`merged_into` filters. A second path would re-acquire all three and is exactly
the shape G29's fourth leak site took.

## 4. Edge cases & failure modes

| case | handling |
|---|---|
| `merge_key` collides for three or more entities | merge pairwise in `_merge_order` order (most edges wins, ties by earliest `created_at`); the loser of round one is `merged_into` and filtered out of round two |
| write-time tier hits an entity the caller is holding | honour `protect_ids` exactly as promotion does — return the *held* entity, never fold it. This is the FK trap reproduced 2026-08-15 (1 turn in 6) |
| entity has no `merge_key` in properties (legacy row, migration mid-flight) | tier misses and the row is minted as today; the sweep catches it. **Never** compute merge_key in SQL as a fallback |
| both sides of a pair already `merged_into` | `merge_entities` skips loudly today (`keep_already_merged` / `absorb_already_merged`); unchanged |
| a rejection memo exists but the names have since changed | memo is keyed on the id pair, not the names. A rename is rare; re-proposing is harmless. Do not invalidate memos |
| `difference_kind` sees an empty name | impossible post-G44 (`codex_entity_name_unusable`), but return `"defer"` rather than raising — detection must never take down the agent run |
| concurrent agent run merges the same pair | `merge_entities` re-checks liveness inside its transaction; unchanged |

## 5. Validation checklist

`tests/test_g50_consolidation.py` — standalone, live DB, inserts and deletes its
own rows, **never truncates**.

1. `difference_kind` returns `merge` for `('qwen coder 32b-q4','qwen coder 32b q4')`, `('u.s','u.s.')`, `('digital note taking','digital note-taking')`.
2. Returns `reject:digits` for `('8 gb','4 gb')`, `('qwen3-32b','qwen3-30b')`, `('gemma-4-e4b','gemma-4-e4b-q4')`.
3. Returns `reject:wordnum` for `('two sagas','four sagas')`.
4. Returns `reject:gender` for `('his father','her father')`.
5. Returns `reject:tense` for `('is rare','was rare')`.
6. **Returns `reject:permutation` for `('shiva without brahma','brahma without shiva')`** — the converse guard.
7. Returns `defer` for `('villainess','villain')` and `('a 3d diagonal plane','3d diagonal plane')`.
8. Write-time: create `gemma-4-e4b q4`, then request `gemma-4-e4b-q4` ⇒ same entity id returned, second name present in `aliases`.
9. Write-time honours `protect_ids`: a held entity is never folded (assert no FK error and both ids survive).
10. Detection writes a `merge_rejected` event for a reject pair and **does not** enqueue it; a second run skips the pair without re-deciding.
11. Regression: `uv run python tests/test_codex_write_path.py` 32/32, `tests/test_maintenance_agent.py` 45/45, `uv run pytest tests/smoke -q`.

**Acceptance number:** on the arm-1 snapshot, a full detection pass proposes
**42 merges**, drops **583** with memos, and defers **1,622** — matching §1.
Any other split means `difference_kind` diverged from the measurement.

## 6. Look-ahead constraints

* **[G51](G51_containment_linking.md)** consumes the `defer` bucket as its
  candidate set. `difference_kind` must therefore return `defer` rather than
  silently dropping — a pair G50 discards is a link G51 never sees.
* **G44's promotion block stays where it is and stays off.** G51 decides its
  fate; this spec must not delete it (`ask before deleting planned work`).
* **FINAL / Z1** re-scores against a re-seeded store. The write-time tier
  changes what gets minted, so **any store seeded before this ships is not
  comparable** on entity counts. Record which side of it a snapshot is on.
* **G48b** attributes codex fragments by `origin_batch_ids`. `merge_entities`
  repoints edges and the batch ids travel with them, so merging does not break
  attribution — but a test asserting a specific fragment count across a merge
  will move. Assert on batch ids, not counts.

## 7. Traps

* **⚑ "Just raise the cosine threshold."** The measured reason it fails: the
  destructive pairs are at the TOP of the band, not the bottom. 0.98+ contains
  `two sagas`/`four sagas` and `8 gb`/`4 gb`. Raising the threshold concentrates
  the danger and discards the defer bucket G51 needs.
* **⚑ "Sort the tokens before comparing."** Adds 11 groups on the gemma arm, of
  which two are converses. `shiva without brahma` is live in the store.
* **"Send the top band to the LLM and auto-accept — they're vectorially
  identical so the model will be right."** Considered and rejected 2026-08-17.
  High cosine is precisely the condition under which the embedding *cannot*
  separate the pair; a small judge shown two near-identical strings fails the
  same way. The failures correlate, so this is one blind instrument checking
  another. The safe half of that band is already caught by `merge_key` with no
  model at all.
* **"Merge the containment pairs — `emotional validation` into `validation`."**
  This is G44's promotion idea and it is why promotion is off by default and
  fired 0 times in 586 turns. `validation loss` is an ML metric, `emotional
  validation` is a psychological concept, `a lack of validation` is the
  negation. Merging destroys three distinctions to gain one node. → G51 links
  them instead.
* **"The 1,094 (2,247 here) pair backlog proves review cannot scale."** It is a
  backlog of pending **rejections**, not pending merges — 98.1% of it should
  never merge. The cosine threshold manufactures the queue; it does not discover
  it. Do not size the solution to that number.
