# G34 — A relation detector that can say "no"
Assumes decided specs: none. Depends on **G35's fan-out cap** (landed `40c9cc5`), which bounds *how many* anchor edges are rendered; this spec decides *which*.

## 1. Decisions

**D1 — Invert the question. This is the whole spec.**
Today the leg asks *"which of the 197 vocabulary glosses is this prompt near?"*
and applies the answer to a matched anchor. It will ask *"which of **this
anchor's own** relations does the question point at?"* instead.

The reasoning is measured, not argued (PROVENANCE, 2026-08-11). Three candidate
fixes to the existing question were scored over their own threshold grids and
**none separated**: centring on the vocabulary centroid silences noise and signal
together (never above 1/5 positives at any threshold); z-scoring within the
prompt's own distribution is the best of them and still leaves 1.00 spurious
relations per negative probe at 3/5 recall; combining them is **strictly
dominated** by z-scoring alone. Absolute cosine is anti-correlated with
relational content — `"ok"` clears 197/197 at 0.844 while `"who inspired Kael"`
clears 43 at 0.575 — and no threshold fixes an inverted ordering.

Inverting works because **it never has to say "no"**. That was the unanswerable
part. Ranking an anchor's existing edges is a strictly weaker claim, and the
fan-out cap already decides how many survive.

| anchor edges | random | strength (today) | **inverted** |
|---|---|---|---|
| 5 | 23.0% | 19.0% | **94.2%** |
| 10 | 11.0% | 8.0% | **93.2%** |
| 20 | 3.0% | 4.8% | **83.5%** |

**D2 — Relation similarity becomes an ORDERING signal, never a gate.**
It must not decide whether the codex leg fires, whether a fragment is emitted, or
whether an edge is reinforced. It decides the order in which an anchor's edges
are considered, and nothing else. This is what makes D1 sound: every measured
failure came from using this signal to answer a yes/no question.

**D3 — Do NOT centre the embeddings.** Measured 83.0% vs 93.2% at a 10-edge
anchor. Three independent results in one run say stacking these corrections makes
things worse; take the simple one.

**D4 — Delete `codex_relation_overlap_boost`'s unconditional application.**
With D1, "the anchor has a relation matching the query" stops being a binary the
leg can assert, so a flat `+0.25` has nothing to key on. The per-anchor score
should instead reflect *how well the best-matching edge fits* — see §2. ⚠ This
setting has a `test_settings_freeze.py` row; changing its meaning rather than its
value needs that row retired deliberately, not silently.

**D5 — USER-REQUIRED before implementation.** Two calls are genuinely the user's:

- **(a) Retire channel 2's gate role entirely, or keep a reduced version?** D1
  removes the *need* for prompt→vocabulary matching, but `_codex_enumeration`
  still consumes `detected_relations` as half of its grounded gate (the
  entity-less "list all the characters" path). Recommended: **keep channel 2
  only for enumeration**, where an explicit cue word already carries the
  precision, and remove it from the anchor path. State this explicitly rather
  than leaving two callers of a function whose contract changed.
- **(b) What happens to channel 1 (lexical)?** Its `_stem` bug is real
  (`len(w) > 4`, so `uses`/`using` never meet) but the roadmap is explicit that
  a better stemmer is the same G28 bet at better odds. Under D1 channel 1 is no
  longer load-bearing for the anchor path. Recommended: **delete it from the
  anchor path** rather than fix it, and let §5's invariance test decide whether
  enumeration still needs it.

**D6 — Style invariance is NOT yet demonstrated, and this spec does not claim
it.** The inverted design improved to 3 distinct picks across 5 phrasings, on a
single random candidate set. That is one noisy sample, not a result. §5 makes a
multi-seed invariance measurement a **blocking** acceptance criterion.

## 2. Algorithm & data model

No schema change. No migration.

Replace the prompt→vocabulary comparison in the anchor path with a
prompt→anchor's-relations comparison, inside `_relation_facts`:

```
given: anchor, prompt_embedding, the anchor's valid edges E (already
       trust-gated and scope-filtered exactly as today)

1. R  = distinct relations present in E                # typically 3-20, not 197
2. if len(R) <= 1 or prompt_embedding is None:
       return today's strength ordering                # nothing to rank
3. G  = unit-norm gloss embedding per r in R           # cached; see below
4. s(r) = dot(prompt_embedding, G[r])                  # both already unit-norm
5. order E by (s(relation_of(e)), _edge_trust(e)) descending
6. take the first `codex_entity_edge_limit` of E       # unchanged
7. fit = s(best) - mean(s over R)                      # >= 0; 0 when flat
8. per-anchor score bonus = codex_relation_fit_weight * clamp(fit, 0, 1)
```

Step 7 is the replacement for D4's deleted flat boost: a contentless prompt
produces a flat `s` across `R`, so `fit ≈ 0` and the bonus vanishes **without a
threshold deciding it**. A pointed question produces a peak and earns the bonus
in proportion. This is the one place a "no" is expressed, and it is expressed as
a continuous zero rather than a gate.

`codex_relation_fit_weight` default **0.25**, matching the magnitude of the boost
it replaces so the change is not silently also a re-tuning. ⚠ Unmeasured, like
`codex_max_fanout` — a Z1 knob, and must be labelled as one wherever quoted.

**Gloss cache.** `_relation_gloss_cache` already embeds all 197 once per process
and is reused as-is; step 3 is a lookup, not an encode. The 12.2 ms pure-Python
cosine loop over 197 becomes a loop over `len(R)` (typically < 20) — G34's
latency half falls out of this for free. Use `np.dot` on a stacked array, per
`retrieval/coverage.py`'s existing shape.

## 3. Files & integration points

| File | Change |
|---|---|
| `src/retrieval/orchestrator.py` `_relation_facts` | The change lives here. It already receives the anchor and queries its edges; it gains `prompt_embedding` and returns edges ordered per §2 plus the `fit` scalar. **Extend it — do NOT add a parallel relation path.** |
| `src/retrieval/orchestrator.py` `_codex_lookup` | Stops passing `detected_relations` to `_relation_facts`; consumes `fit` in place of the flat `codex_relation_overlap_boost` at the per-anchor score. |
| `src/retrieval/orchestrator.py` `_detect_relations` | Per D5(a): keep only for `_codex_enumeration`, or delete with it. Its `codex_relation_detection_enabled` kill-switch (`960f3cc`) stays until this lands, then is removed with the code it guards. |
| `src/api/config.py` | Add `codex_relation_fit_weight`. Retire `codex_relation_overlap_boost` per D4, including its `FROZEN` row. |
| `tests/smoke/test_relation_killswitch.py` | Retire with the switch. |
| `.env` | Remove `CODEX_RELATION_DETECTION_ENABLED=false` — the tuning override exists only because the detector was broken. |

## 4. Edge cases & failure modes

| Case | Handling |
|---|---|
| Anchor has 0 or 1 distinct relations | Skip ranking, keep strength order, `fit = 0`. Ranking one item is theatre. |
| `prompt_embedding is None` (degraded encoder) | Same as above, and it must WARN — a silent fallback here is a different retrieval quality wearing the same name (CLAUDE.md, TRAPS #11). |
| Contentless prompt (`"ok"`) | `s` is flat over `R`, `fit ≈ 0`, no bonus. The failure mode this item exists to fix, handled by construction rather than by threshold. |
| A relation in `E` absent from `ALLOWED_RELATIONS` | Legacy/hand-written edges exist. Embed on demand and cache, or skip that edge for ranking — never crash the leg. |
| All of `E` filtered out by trust/scope | Unchanged: return `([], [])`, no fragment, no bonus. |
| Negated edges | Unchanged — already excluded from traversal; must stay excluded from ranking, or a "NOT uses" edge ranks first on a question about using. |
| Empty store | `R` is empty everywhere; the whole path is inert. This is the current dev state, so a passing test suite proves nothing without seeded edges — see §5. |

## 5. Validation checklist

- [ ] `tests/test_g34_relations.py`, standalone live-DB, inserts and **deletes its own rows** (never truncate — TRAPS #6/#15/#16). Seeds one anchor with ≥ 10 real edges across ≥ 6 distinct relations, with **real embeddings** — a constant-vector stub makes every similarity an exact tie and the suite passes vacuously (this is precisely how `test_retrieval_failopen`'s first draft failed; copy its seeding shape).
- [ ] **Two-sided:** the expected relation ranks first for a pointed question, **and** a different pointed question over the same anchor ranks a different one first. One-sided proves only that something sorted.
- [ ] **`fit ≈ 0` for a contentless prompt** over the same seeded anchor, and `fit > 0` for a pointed one. This is D4's replacement and needs its own assertion.
- [ ] **Ordering beats both baselines** on the seeded anchor: chance, and today's strength ordering. Re-run `scripts/oneoff/g34_ablation.py` against the populated store and record the numbers in PROVENANCE — the simulated-anchor figures in §1 are explicitly provisional.
- [ ] **⚑ BLOCKING — G28 style invariance over many seeds.** Hold meaning fixed, vary form (± question mark, "ok so"/"like" prefixes, lowercase, typos, terse vs rambling) across ≥ 50 random candidate sets, and report the **decision-flip rate**. The single-seed sample gave 3 distinct picks in 5 phrasings, which is not good enough to ship on. **Agree the acceptable flip rate with the user BEFORE measuring it**, or the threshold gets chosen to fit the result.
- [ ] Latency re-measured: the 197-gloss loop should be gone. Record median per prompt.
- [ ] Regressions: `test_retrieval`, `test_session_scoping`, `test_retrieval_failopen`, `test_timescope`, `test_codex_write_path`, `tests/smoke`.

## 6. Look-ahead constraints

- **G35's relation-awareness half depends on this** and becomes implementable the
  moment it lands: `_traverse_graph`'s frontier ranking can key on the same `s(r)`
  it already ranks by `_edge_trust`. Keep §2 step 4 callable from the traversal.
- **G35's per-leg budget-share check is still owed** and is the honest measure of
  whether better ordering helps: better-chosen edges that still consume the same
  share of the window have moved the problem, not solved it.
- **Z1** must not tune `codex_relation_fit_weight` against a store this feature
  has never run on. Land it, populate, then tune.

## 7. Traps

- **Do not reintroduce a threshold on `s`.** The measured lesson is that absolute
  similarity to a gloss carries no usable signal about relational intent. `s` is
  legitimate for *ordering within one anchor* and illegitimate for *deciding
  anything on its own*. If a future change compares `s` to a constant, this item
  has regressed.
- **Do not stack corrections.** Centring measured worse three separate ways in
  one run. The instinct that two fixes beat one is wrong here and was tested.
- **`_relation_facts` returning more edges is not the win.** It already returns
  the anchor's top edges; the change is *which*. A test asserting a bigger result
  set is measuring the wrong thing.
- **A passing suite on an empty store proves nothing** — `R` is empty, the path is
  inert, and every assertion is vacuous. Seed first.
- **The `+0.25` was applied to noise on every prompt for the whole of v2.** Any
  measurement taken before this lands, on a populated store, carries it. Say so
  where such numbers are quoted rather than silently comparing across the change.
