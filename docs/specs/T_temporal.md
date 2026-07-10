# Track T — Temporal retrieval & idea evolution (T1–T4)

Assumes decided specs: none (first S1 spec; grounded in source at commit `ef6f735`, 2026-07-10).

Scope: the whole track in one spec — the four items share one data model (`TimeScope`)
and one lifecycle, and splitting them would fragment a single design. Item boundaries
are marked so they can be implemented and validated as separate passes (T1 → T2+T3 →
T4 is the required order; T1 has no dependencies).

**Why this exists (one paragraph, so the executor knows what winning looks like):**
storage already records time on three layers (episodic `timestamp`, bi-temporal
`codex_edges.valid_from/valid_until`, the `codex_events` journal + `codex_snapshots`),
but retrieval reads none of them for time — all codex traversal filters
`valid_until IS NULL` (current-only), episodic legs have no timestamp predicate,
nothing parses "in early 2025", and the prompt shows the model neither today's date
nor any fragment date, while [prompt_assembler.py](../../src/api/prompt_assembler.py):143
*instructs* it to narrate how facts changed. After this track: prompts are date-grounded
(T1), temporal queries are detected and parsed (T2), retrieval can answer *at* a time
(T3), and idea evolution is served as context whenever history exists (T4).

---

## 1. Decisions

All settled. The only user-owned calls were asked and answered 2026-07-10:

- **D-U1 (user): cold storage = reach + resurrect-on-probation.** Time-scoped queries
  may search `cold_storage`; a cold memory that actually gets *injected* (survives the
  budget) is moved back into `episodic_memory` at a decay score just above the archive
  line (`timescope_probation_score = 0.12` vs `ARCHIVE_THRESHOLD = 0.1`) — it must
  re-earn its place: unengaged, normal decay re-archives it within days; engaged,
  write-on-read strengthening (+0.15/hit) saves it. Never restored at full strength.
- **D-U2 (user): evolution info is provided, never forced.** No answer-format
  mandate. The system attaches compact timeline context whenever retrieved anchors
  carry real supersession history — including for questions with *no* temporal wording
  (the saga case) — and the model decides how much to narrate (the existing
  assembler instruction already tells it to mention changes). Combination of
  "auto preamble" and "marker": the timeline fragment IS compact; the model expands
  it or ignores it.

Design decisions (with the why):

- **D1: `TimeScope` travels inside the existing `scope` dict** (`scope["timescope"]`),
  not as a new parameter. Every leg and the wide net already receive `scope`; zero
  signature churn, and `configurable_orchestrator`'s overrides (which mirror parent
  signatures — the G19 risk) keep working unmodified.
- **D2: detection is deterministic, LLM-free, and joint-gated.** Regex/lexicon time
  expressions + relative-date resolution. A window fires only when a *resolvable*
  expression co-occurs with a recall-shaped prompt (A4's joint-signal lesson: bare
  "yesterday" in a content clause must not flip modes — a false positive injects stale
  context into a normal answer, the mirror image of today's failure). Kill-switch
  setting `timescope_enabled` for instant rollback.
- **D3: four modes.** `current` (default, behavior identical to today),
  `as_of` (point-in-time → small window, recency re-anchored to the target),
  `range` (window, flat scoring inside), `evolution` (history request; may also carry
  a window). Evolution cue wins the mode when both match ("how did X change during
  2025" → evolution with t0/t1 set).
- **D4: `as_of` codex reads use `valid_at(T)` with T = window end** — "state as of
  then" means "everything established *by* then": `valid_from <= T AND (valid_until
  IS NULL OR valid_until > T)`. `current` keeps the exact existing predicate
  (`valid_until IS NULL` ≡ valid_at(now), since expiries are stamped at write time).
- **D5: evolution mode does NOT traverse dead edges.** Navigation stays over the
  current graph; history is served by the timeline builder (T4). Walking expired
  edges as if they were links produces incoherent neighborhoods.
- **D6: the decay/supersession discriminator is the event journal.** An expired edge
  WITH a matching `edge_expired` event = a semantic supersession (A6 reconciliation,
  A8 negation, property/conflict update) → belongs in timelines. An expired edge
  WITHOUT one = `codex_decay` forgetting/garbage-collection (codex_decay writes no
  events — verified) → excluded. A faded idea is not a revised idea. *(C10
  revision 2026-07-10: event-backed expiries with reason `source_deleted` are
  excluded too — deletion is not evolution.)*
- **D7: timelines are a first-class fragment type** (`source_type="timeline"`),
  hard-capped (`timeline_max_tokens=300`, ≤2 per query, ≤4 in evolution mode), riding
  A10's round-robin budget fairness like any other leg. A life story must not eat
  the window.
- **D8: memory decision gets a bump, not a force.** A non-current TimeScope adds
  `ltm_bump_timescope = +3.0` log-odds in `decide_memory_retrieval` — decisive in
  practice (an explicit "what did I think in 2025" is definitionally a memory query)
  but B2-idiomatic: no hard override returns, full breakdown telemetry.
- **D9: mode-aware recency = same exponential, movable origin.** `as_of` re-anchors
  C8's in-score boost to distance-from-window-center (and A11's edge recency to
  |T − valid_from|); `range`/`evolution` drop the factor (the user asked for a period,
  not proximity to its midpoint); `current` unchanged. One formula, parameterized —
  never fork the leg SQL into timescoped copies.
- **D10: archived rows become visible under non-current modes** (drop the
  `is_archived = false` filter and the wide net's `decay_score > 0.2` floor). Archive
  means "not relevant to *now*" — which is exactly what a time query is not asking.
- **D11: the archived-freeze bug is fixed here** (it blocks D-U1): all three decay
  UPDATEs in [decay.py](../../src/workers/decay.py) filter `is_archived = FALSE`, so an
  archived row's score freezes at ~0.1 forever and can never cross the 0.05 cold line —
  cold storage is effectively unreachable today. Fix: archived rows keep decaying, and
  a symmetric un-archive clause restores rows whose score recovers (strengthen-driven).
- **D12: cold_storage schema is extended** (`conversation_id`, `is_private`,
  `batch_id`) — today's cold move drops the conversation link and the privacy flag,
  which makes resurrection impossible and would leak incognito turns into time queries.
  Legacy cold rows (NULL `conversation_id`) are **cite-only**: searchable, never
  resurrected. Legacy rows cannot be private (cold rows predate G16/incognito —
  incognito didn't exist when they froze), so `is_private DEFAULT FALSE` is correct,
  not a guess.
- **D13: never-overwrite for note bodies.** Every `entity.description` write site
  emits a `description_updated` event (old/new, truncated) — the one mutable-in-place
  field joins the journal. Reflection's current `context_appended` emission is
  upgraded to this; old events remain as opaque history.
- **D14: RAG leg unchanged; batch-summary leg skipped under non-current modes.**
  Documents are reference material, not the user's thinking. Batch summaries carry
  `created_at` but summarize *decayed old* turns (created long after their content) —
  their content period is underivable without a turn-index→timestamp join; serving
  them under a window would mislead. v1 skips; revisit only if Z1 shows gaps.
- **D15: no branching, no content-addressing.** Linear supersession + coexisting
  non-conflicting edges + A8 negations cover parallel ideas; branching is B6-tier.
  The journal already gives git's semantics; we build only the porcelain
  (checkout=T3, log/diff=T4).
- **Empirical deferral (rule 2b, the only one):** detector precision in the wild.
  Measurement: the `timescope_detected` log line (fires with mode + matched text) over
  Z1's live usage. Decision rule: if >2% of *non-temporal* prompts fire a non-current
  mode (manual scan of a week's log), tighten the joint gate (require interrogative
  shape AND `p_ltm ≥ 0.5`, instead of OR) and/or blocklist the offending expression;
  if temporal prompts are *missed*, add expressions to the lexicon. No architecture
  change either way.

---

## 2. Algorithm & data model

### 2.1 `TimeScope` (new: `src/retrieval/timescope.py`)

```python
@dataclass(frozen=True)
class TimeScope:
    mode: str                       # "current" | "as_of" | "range" | "evolution"
    t0: Optional[datetime] = None   # UTC, inclusive
    t1: Optional[datetime] = None   # UTC, exclusive
    matched_text: str = ""          # expression that fired (telemetry)
    evidence: tuple = ()            # signals that passed the gate (telemetry; B1 seam)

CURRENT = TimeScope(mode="current")

def detect_timescope(prompt: str, *, now: Optional[datetime] = None,
                     intent_tags: Sequence[str] = (), p_ltm: float = 0.0,
                     reference_signal: bool = False) -> TimeScope

def to_scope_dict(ts) -> Optional[dict]      # {"mode","t0","t1"} iso strings; None for current
def from_scope(scope: Optional[dict]) -> TimeScope   # parse scope.get("timescope"), tolerant
```

Detection pipeline (pure, <1 ms):

1. **Strip** fenced code blocks and inline backticks (temporal words inside code are
   content, not intent).
2. **Scan** the expression grammar (case-insensitive), longest match wins:
   - absolute: `in? <month-name> <year>` · `in <year>` (1990–2099 guard) ·
     `q[1-4] <year>` · `(first|second) half of <year>` / `h[12] <year>` ·
     `(early|mid|late) <year|month-name [year]>` · `(spring|summer|autumn|fall|winter) [<year>]` ·
     explicit dates `YYYY-MM-DD` / `<month-name> <day>`
   - relative: `<N|number-word> (day|week|month|year)s? ago` · `last (week|month|year|
     <month-name>|summer…)` · `yesterday` · `the other day` (→ 1–7 days ago)
   - open ranges: `since <expr>` (t0=expr.start, t1=now, mode=range) ·
     `before <expr>` (t0=2000-01-01, t1=expr.start) · `between <expr> and <expr>`
   - month-name without year → most recent *past* occurrence (if this year's instance
     is in the future, use last year's).
   - number-words: one…twelve, "a"/"an" = 1, "couple" = 2, "few" = 3.
3. **Evolution cues** (independent scan): `how (did|has) .* (evolve|change|develop|
   progress)` · `(evolution|history|progression|timeline) of` · `over time` ·
   `originally|at first|initially|back when|at the (start|beginning)` ·
   `what did (I|we) (start|begin) with` · `compare .* (vs|with|to) .*` when both sides
   carry temporal expressions.
4. **Joint gate** — a non-current mode is returned only if:
   `resolvable_expression_or_evolution_cue AND recall_shaped`, where `recall_shaped` =
   prompt contains `?` OR starts with an interrogative/recall imperative
   (`what|when|how|show|list|tell|remind|which|where|did|was|were`) OR
   `reference_signal` OR `p_ltm ≥ 0.5`.
   Guards: a window whose t0 > now → `CURRENT` (future = scheduling, not memory);
   detector disabled (`settings.timescope_enabled = False`) → `CURRENT`.
5. **Mode assignment:** evolution cue present → `evolution` (window attached if an
   expression also resolved). Else `between/since/before`, halves, quarters, seasons,
   whole years → `range`; point-like expressions (N-ago, month+year, explicit date,
   yesterday) → `as_of`. Multiple disjoint expressions (e.g. "2024 vs 2025") → the
   enclosing range `[min t0, max t1]`, and the compare-cue makes it `evolution`.

**Window padding** (all in settings; the window is `[t0, t1)`):

| expression granularity | window |
|---|---|
| explicit date / "yesterday" | day ± `pad_min_days` (3) |
| week-ish relative (`N days/weeks ago`) | center ± max(3d, 0.4·Δ), cap 45d |
| month (+year), `last <month>` | month start−14d .. month end+14d |
| month/year relative (`N months/years ago`) | center ± max(14d, 0.15·Δ), cap 120d |
| quarter / half / season | span ± 21d |
| year | Jan 1−30d .. Dec 31+30d |

Sanity: after resolution, if t0 ≥ t1 → swap; clamp t1 to now.

### 2.2 Threading (T2)

- **main.py** (after the classify call — see G26 note in §3): 
  ```python
  tscope = detect_timescope(user_message,
                            intent_tags=result.intent_tags,
                            p_ltm=getattr(result, "p_ltm", 0.0),
                            reference_signal=getattr(result, "reference_signal", False))
  if tscope.mode != "current":
      log.info("timescope_detected", mode=tscope.mode, t0=..., t1=..., matched=tscope.matched_text)
  ```
  After the scope dict is built: `ts_dict = to_scope_dict(tscope)`, and if not None:
  `scope["timescope"] = ts_dict`.
- **memory_decision.py**: `decide_memory_retrieval(..., timescope_mode: Optional[str] = None)`;
  after the feature bumps: `if timescope_mode and timescope_mode != "current":
  lo += settings.ltm_bump_timescope`; add `"timescope": timescope_mode` to the breakdown.
  (Kwarg, NOT a ClassificationResult field — keeps the function pure and leaves the
  label question to B1.)
- **orchestrator**: at the top of `retrieve()` AND `_wide_net_fallback()`:
  `self._active_timescope = from_scope(scope)` (instance attr is safe — one orchestrator
  instance per request, main.py:372). Everything below reads `self._active_timescope`.
  Also gate: if `not settings.timescope_enabled or not getattr(self, "timescope_allowed", True)`
  → force `CURRENT` (the second flag is the ablation seam, §3.4).

### 2.3 Episodic legs (T3)

In `_bm25_episodic` (both main and fallback queries), `_vector_episodic`,
`_vector_chunks` (window applies to the **parent** turn's `e.timestamp`), and
`_wide_net_fallback` — build alongside the existing `privacy_filter`:

```python
ts = self._active_timescope
scoped = ts.mode in ("as_of", "range")
time_filter    = "AND timestamp >= :ts_t0 AND timestamp < :ts_t1" if scoped else ""
archived_filter= "" if ts.mode != "current" else "AND is_archived = false"
```
(chunk leg: `e.timestamp`, `e.is_archived`). The wide net additionally drops its
`AND decay_score > :min_decay` clause when `ts.mode != "current"`.

**Recency factor** (the C8 multiplier already in the SQL) becomes origin-parameterized.
Replace the fixed `(NOW() - timestamp)` with `(:ts_center - timestamp)` semantics:

```sql
* (1 + :recency_boost * EXP(-ABS(EXTRACT(EPOCH FROM (timestamp - :ts_center))) / 86400.0 / :recency_tau))
```
with params per mode:
- `current`: ts_center = NOW (pass `datetime.now(utc)`), boost = existing
  (`EPISODIC_RECENCY_BOOST`, 0 for creative), tau = 30 — *numerically identical to
  today for all past rows* (ABS is a no-op when center=now ≥ timestamp).
- `as_of`: ts_center = (t0+t1)/2, boost = `EPISODIC_RECENCY_BOOST` **including for
  creative** (this is target-proximity relevance, not freshness — the creative-skip
  rationale doesn't apply), tau = max(30, window_days/2).
- `range`/`evolution`: boost = 0 (flat).

**Post-fusion now-bonuses are gated off under non-current modes** (they all point at
now): in `_apply_bonuses`, skip the `_recency_bonus` add when
`self._active_timescope.mode != "current"`; in `_rows_to_fragments`, skip both
tiebreakers (the age-hours bonus at ~1826 and the newer-count bonus at ~1832) under
the same condition.

### 2.4 Codex (T3)

Central helper (used by `_traverse_graph` both directions, `_relation_facts`,
`_codex_enumeration` — the four `valid_until == None` sites):

```python
def _edge_valid_filters(self):
    ts = getattr(self, "_active_timescope", CURRENT)
    if ts.mode in ("as_of", "range") and ts.t1:
        T = ts.t1
        return [CodexEdge.valid_from <= T,
                or_(CodexEdge.valid_until == None, CodexEdge.valid_until > T)]
    return [CodexEdge.valid_until == None]
```

`_edge_trust` A11 re-anchor: when mode in ("as_of","range"),
`age_days = abs((ts.t1 - vf).total_seconds()) / 86400` (else unchanged). The
multiplier stays ≥ 1.0 — A11's "never penalize age" invariant survives re-anchoring.

**Entity-state-at-T** ("state of the project on date X") needs no snapshot replay in
v1: `valid_at(T)` edge filtering IS the state reconstruction for everything retrieval
renders (scoped renders build from edges; unscoped anchors render `context_payload`,
which is current-state — acceptable: the edges shown are era-correct, and the payload
is clearly the note body). `codex_snapshots` replay is deliberately NOT built until
a consumer needs field-level state (F3); note only.

**Cluster leg unchanged** (clusters restrict topic; the window restricts time — they
compose). **Procedural**: add overlap filter when scoped:
`first_observed <= :ts_t1 AND last_observed >= :ts_t0`. **RAG unchanged**;
**batch_summary returns [] under any non-current mode** (D14).

### 2.5 Cold-storage leg + resurrection (T3, D-U1)

New method `_cold_lookup(prompt_keywords)` — called from `retrieve()` (and NOT from
the wide net) only when `ts.mode in ("as_of","range","evolution") and ts.t0`:

```sql
SELECT id, conversation_id, raw_text, summary_text, topic_tags, timestamp, is_private
FROM cold_storage
WHERE timestamp >= :t0 AND timestamp < :t1
  AND (is_private = FALSE OR conversation_id = :conv_id)   -- :conv_id NULL-safe via COALESCE guard
  AND (:no_kw OR raw_text ILIKE ANY(:pats) OR summary_text ILIKE ANY(:pats))
ORDER BY timestamp DESC LIMIT :cold_limit
```
- `pats` = up to 6 `%kw%` patterns from prompt keywords + grounded expansion terms;
  `no_kw = TRUE` when there are none (a pure "what was I thinking about in march"
  browse — the window itself is the filter). Evolution without a window: skip cold
  (unbounded browse is noise).
- Fragments: `source_type="episodic"`, text = summary_text or raw_text (300-word cap,
  sentence-truncated), **date-stamped like all fragments (T1)**, base score 0.6
  (keyword/length bonuses still apply later), `source_batch_id=str(id)`,
  `conversation_id` carried. Record `self._cold_hits[str(id)] = row`.
- Incognito: the `(is_private = FALSE OR conversation_id = :conv)` clause mirrors the
  live legs' invariant; under incognito scope `conv_id` is the own conversation.

**Resurrection** — in `retrieve()` after `_enforce_token_budget` (only *injected*
memories get the second chance):

```python
for f in final:
    row = self._cold_hits.get(f.source_batch_id)
    if row is None: continue
    if row.conversation_id is None:          # legacy pre-migration row
        log cold_cited_only; continue
    INSERT INTO episodic_memory (id=row.id, conversation_id=row.conversation_id,
        batch_id=row.batch_id or uuid4(), timestamp=row.timestamp,   # ORIGINAL time — it's an old memory
        raw_text, summary_text, topic_tags, intent_tags=[],
        context_reliance='Long_Term_Memory', idempotency_key=f'cold-resurrect-{row.id}',
        decay_score=settings.timescope_probation_score, access_count=1,
        is_archived=FALSE, is_private=row.is_private, inject_raw=TRUE,
        embedding=encode((row.summary_text or row.raw_text)[:2000]))
    ON CONFLICT (id) DO NOTHING → if conflicted: keep cold row, log; else DELETE cold row
    log cold_resurrected
```
Long resurrected turns get re-chunked by C3's catch-up worker on its next pass
(length proxy — no work here). Clustering re-adopts them via the normal queue
(link-less turns pass the cluster filter — C5).

### 2.6 Honest emptiness (T3)

In `retrieve()`, when `ts.mode in ("as_of","range")` and the final list contains **no
episodic fragments** (codex/timeline may still exist): run the vector leg once
*without* the window (LIMIT 3), collect their dates, and append one small fragment:

```
[Memory note] No stored memories match this query between 2025-01-01 and 2025-06-30.
Closest matches outside that window are from 2024-11 and 2026-03.
```
(`source_type="episodic"`, score 5.0 so the budget keeps it, ~40 tokens; second
sentence only when the unwindowed probe found anything.) Never silently widen.

### 2.7 Timeline builder + diff (T4; new `src/retrieval/evolution.py`)

```python
def build_entity_timeline(db, entity, allowed_batch_ids=None,
                          t0=None, t1=None, max_transitions=8) -> Optional[str]
def entity_diff(db, entity, t0, t1) -> dict   # {"added": [...], "expired": [...], "retracted": [...]}
def history_exists(db, entity_id) -> bool     # cheap EXISTS gate (see below)
def log_description_update(db, entity, old, new, source) -> None   # D13 helper
```

- Candidate edges: `CodexEdge` where `source_id = e OR target_id = e`
  (+ `source_batch ∈ allowed_batch_ids` when scoped), `valid_from <= (t1 or now)`.
- **Inclusion rule (D6):** live edges always; expired edges only if an `edge_expired`
  event exists with `payload->>'edge_id' = str(edge.id)` (one query for all candidate
  ids: `SELECT payload->>'edge_id', timestamp, batch_source, payload->>'reason'
  FROM codex_events WHERE event_type='edge_expired' AND payload->>'edge_id' = ANY(:ids)`).
- Group by `(relation, direction)`; within a group sort by `valid_from` — consecutive
  members form a supersession chain. Render, oldest→newest, one line per member:

  ```
  [Timeline: saga 2 ending]
  2025-11 – 2026-02: saga 2 ending --planned_as--> multiverse destruction  (superseded: antonym_superseded)
  2026-02 – now:     saga 2 ending --planned_as--> pact mercy killing
  ```
  Dates at month precision (`%Y-%m`); negated edges render `NOT <relation>`; the
  expiry reason comes from the event payload (fallback "updated"). Keep the most
  *recent* `max_transitions` lines; if truncated, prepend "(earlier history omitted)".
  Word-trim to `timeline_max_tokens`.
- `history_exists(entity_id)`: `EXISTS(expired edge touching e whose id appears in an
  edge_expired event)` — same join, LIMIT 1.
- `entity_diff`: edges with `valid_from ∈ [t0,t1)` → added (negated → retracted);
  event-backed expiries with event `timestamp ∈ [t0,t1)` → expired. Returns dicts of
  `{relation, other, date, reason?}`. **Not wired to any endpoint here** — it's the
  E0 service candidate and F3's future backend; used by tests now.

**Wiring in `_codex_graph`** (after each anchor's fragment is appended):

```python
cap = settings.timeline_max_fragments_evolution if ts.mode == "evolution" \
      else settings.timeline_max_fragments
if len(timeline_frags) < cap and history_exists(self.db, anchor.id):
    tl = build_entity_timeline(self.db, anchor, allowed_batch_ids,
                               t0=ts.t0, t1=ts.t1, max_transitions=settings.timeline_max_transitions)
    if tl:
        timeline_frags.append(ContextFragment(text=tl, source_type="timeline",
                                              score=0.9 * anchor_fragment.score, ...))
```
Timeline fragments are returned alongside codex fragments; RRF/budget treat
`"timeline"` as its own leg (round-robin lane + leg-diversity guarantee come free
from `_enforce_token_budget` keying on `source_type`). Give it RRF weight via
`blend_weights["timeline"] = 0.6` added to `base_weights` (constant across profiles —
its firing condition is the gate, not the intent).

**Era-stratified episodic sampling (evolution mode only):** in `_vector_episodic`,
when `ts.mode == "evolution"`: fetch the normal candidate query WITHOUT window,
LIMIT 300, flat recency; python-side, sort rows by timestamp, split into
`evolution_era_buckets = 4` equal-count buckets, keep the top `evolution_per_era = 3`
by score from each, pass those ≤12 rows to `_rows_to_fragments`. (Python-side
bucketing — no fancy SQL, executor-proof. C4's future era digests plug in by adding
summary rows to the same candidate list.)

**D13 journaling:** every `entity.description = ...` write site (find them all:
`grep -rn "\.description\s*=" src/`; today: reflection enrichment ~line 328,
user_control's entity-edit endpoint, codex_inject_watcher if it writes descriptions)
calls `log_description_update(db, entity, old, new, source)` →
`CodexEvent(event_type="description_updated", payload={"old": old[:300],
"new": new[:300], "source": source}, batch_source=uuid4())`. Reflection's existing
`context_appended` emission is *replaced* by this (old events stay readable history).

### 2.8 Date-grounding (T1)

- `_rows_to_fragments`: after `_choose_representation`, when `row.timestamp` exists:
  `stamp = row.timestamp.strftime("[%Y-%m-%d] ")`; prefix `text`, and prefix
  `degrade_text`/`abstract_text` too when present (degradation must not lose the date).
- Codex fact lines (`_relation_facts` ~1092, `_codex_enumeration` ~1140):
  `[Fact: a --rel--> b (since 2026-03)]` from `edge.valid_from` (month precision;
  day precision implies false exactness). Skip when valid_from is NULL (legacy).
- Batch-summary fragments: prefix `[summary, %Y-%m-%d]` from `created_at`.
- Entity notes (`context_payload`) and depth-previews: **left undated** (regenerated
  text owned by the extractor; dating it is extractor surgery for marginal gain).
- Recent sliding-window turns: **undated** (they are "now" by construction; decided).
- `prompt_assembler.assemble_prompt` system message — prepend to the content:
  `f"Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. "` and extend
  the existing changed-over-time instruction with:
  *"Retrieved memory fragments are prefixed with the date they were written, like
  [2025-11-04]; facts may show (since YYYY-MM) or a [Timeline: …] history. Use these
  dates to order events and to tell earlier versions from the current one."*

### 2.9 Decay rework (T3, D11/D12) — `src/workers/decay.py`

1. Remove `AND is_archived = FALSE` from the three decay UPDATEs (archived rows keep
   decaying at their access-class rate).
2. Add, before the archive step: `UPDATE episodic_memory SET is_archived = FALSE
   WHERE is_archived = TRUE AND decay_score >= :archive_threshold` (strengthen-driven
   recovery — the probation reversal is automatic and symmetric).
3. Extend the cold INSERT to carry `conversation_id, is_private, batch_id`.
4. Task order: decay → un-archive → archive → cold-move (unchanged otherwise).

### 2.10 Migration (one Alembic revision)

```sql
ALTER TABLE cold_storage ADD COLUMN conversation_id UUID NULL;
ALTER TABLE cold_storage ADD COLUMN is_private BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE cold_storage ADD COLUMN batch_id UUID NULL;
CREATE INDEX ix_episodic_memory_timestamp ON episodic_memory (timestamp);
CREATE INDEX ix_cold_storage_timestamp ON cold_storage (timestamp);
CREATE INDEX ix_codex_edges_source_valid ON codex_edges (source_id, valid_until);
CREATE INDEX ix_codex_edges_target_valid ON codex_edges (target_id, valid_until);
```
(Timestamp indexes are what the new window predicates want — and a G6 down-payment.
Migration logs `SELECT count(*) FROM cold_storage`; expected ≈0 because of the freeze
bug — if >0, those rows are pre-incognito and legally `is_private=FALSE`.)

### 2.11 Settings (`src/api/config.py`, one block)

```python
# ── Track T: temporal retrieval & idea evolution ──
timescope_enabled: bool = True
ltm_bump_timescope: float = 3.0
timescope_pad_min_days: int = 3
timescope_pad_month_days: int = 14
timescope_pad_span_days: int = 21
timescope_pad_year_days: int = 30
timescope_rel_short_frac: float = 0.4    # ≤ week-granularity relatives
timescope_rel_short_cap_days: int = 45
timescope_rel_long_frac: float = 0.15    # month/year-granularity relatives
timescope_rel_long_cap_days: int = 120
timescope_probation_score: float = 0.12  # just above ARCHIVE_THRESHOLD (0.1)
timescope_cold_limit: int = 5
timeline_max_tokens: int = 300
timeline_max_transitions: int = 8
timeline_max_fragments: int = 2
timeline_max_fragments_evolution: int = 4
evolution_era_buckets: int = 4
evolution_per_era: int = 3
```

---

## 3. Files & integration points

| file | change |
|---|---|
| `src/retrieval/timescope.py` **(new)** | `TimeScope`, `detect_timescope`, window math, `to_scope_dict`/`from_scope`. Pure, no DB. |
| `src/retrieval/evolution.py` **(new)** | `build_entity_timeline`, `entity_diff`, `history_exists`, `log_description_update`, the expiry-event join. DB-reading, no LLM. |
| `src/retrieval/orchestrator.py` | `retrieve()`/`_wide_net_fallback()`: set `self._active_timescope`; legs: `time_filter`/`archived_filter`/decay-floor/params (§2.3); recency re-anchor in vector/chunk/wide-net SQL; `_edge_valid_filters()` helper replacing the four `valid_until == None` sites; `_edge_trust` re-anchor; `_apply_bonuses` + `_rows_to_fragments` now-bonus gating; T1 date stamps; `_cold_lookup` + `self._cold_hits` + resurrection pass; honest-emptiness note; timeline wiring in `_codex_graph`; evolution stratified sampling in `_vector_episodic`; `blend_weights["timeline"]`. **Extract, don't inline**: detector/timeline live in their own modules (the Track-C decomposition note — leave the God-class smaller than found). |
| `src/retrieval/configurable_orchestrator.py` | New ablation flag `timescope` → when off, subclass sets `self.timescope_allowed = False` in `__init__` (parent then forces CURRENT). Verify overridden legs still just call `super()` with the same signatures (they do — timescope travels inside `scope`). **G19: this is the lockstep edit; do it in the same commit.** |
| `src/api/main.py` | Call `detect_timescope` right after classification; log `timescope_detected`; put `to_scope_dict(tscope)` into `scope` after the scope block; pass `timescope_mode=tscope.mode` into `decide_memory_retrieval`. **⚠ G26 first:** as of `ef6f735`, main.py classifies at line ~231 using `conversation_id` before its first assignment (~273) — UnboundLocalError on every request. The G26 hotfix (move conversation resolution above classification) must land before or with this change; T2's detector call sits after classification either way. |
| `src/api/memory_decision.py` | `timescope_mode` kwarg + bump + breakdown key (§2.2). |
| `src/api/prompt_assembler.py` | Today's date + the fragment-dating instruction sentence (§2.8). |
| `src/api/config.py` | §2.11 block. |
| `src/workers/decay.py` | §2.9 (freeze fix, un-archive, cold columns). |
| `src/workers/reflection.py` | Enrichment writes: capture `old = entity.description` before overwrite; call `log_description_update(..., source="reflection_enrichment")` (replaces the `context_appended` emit). |
| `src/api/routers/user_control.py` | The entity-edit endpoint (writes `description`): same journaling, `source="manual_edit"`. Sweep any other `\.description =` writer found by grep. |
| `alembic/versions/<new>.py` | §2.10. |
| `tests/test_timescope.py` **(new)** | §5. |

---

## 4. Edge cases & failure modes

- **Timezones:** all columns are `DateTime(timezone=True)`; detector emits UTC-aware
  datetimes (`datetime.now(timezone.utc)` anchor). Never compare naive/aware.
- **Legacy NULLs:** rows with NULL `session_id` are irrelevant here; `timestamp` has a
  default and is effectively non-null — still guard `if row.timestamp` (already the
  pattern at ~1826). Codex edges with NULL `valid_from` (legacy): treated as
  `valid_from = -∞` (they pass `valid_from <= T` — implement as
  `or_(CodexEdge.valid_from == None, CodexEdge.valid_from <= T)`).
- **Future expressions** ("next week", "summer 2027"): t0 > now → CURRENT (never a
  memory window).
- **Month without year** when that month hasn't happened yet this year → previous year.
- **t0 ≥ t1 after padding** → swap; **t1 clamped to now**.
- **In-code temporal words:** stripped with fenced/inline code before scanning.
- **DI3 fast-path classifications** (all-zero raw_probs): `p_ltm`/`reference_signal`
  read via `getattr(..., default)` — detector still works from the prompt alone.
- **Incognito × timescope:** window filters compose with the existing
  conversation-only scoping; cold search's privacy clause mirrors the legs' invariant
  (own-conversation private rows only). Wide net under timescope keeps its scope rules
  (it already honors them post-G16) plus the new window/archived params.
- **Empty window:** honest-emptiness note (§2.6); codex/timeline fragments may still
  answer (the graph knows things episodic lost).
- **Cold row id collision on resurrection** (row somehow still live): `ON CONFLICT DO
  NOTHING`, keep the cold row, log `cold_resurrect_conflict`.
- **Legacy cold rows (NULL conversation_id):** cite-only — injected as context, never
  resurrected (no conversation to re-attach to). Logged.
- **Resurrected row lifecycle:** original `timestamp` preserved (it's an old memory);
  probation score 0.12; if untouched: ~2 days of cycles back under 0.1 → re-archived,
  then (freeze fix) onward to cold. If engaged: +0.15/hit outruns decay. Re-chunking
  (C3 catch-up) and clustering (C5 queue) re-adopt it automatically.
- **Archived row retrieved under timescope** gets `_strengthen_retrieved` +0.15 like
  any episodic hit → may cross 0.1 → next decay cycle un-archives it (D11's clause).
  That IS the archived-tier second chance — same mechanic, zero extra code.
- **Timeline for a heavily-churned entity:** most-recent `max_transitions` kept,
  "(earlier history omitted)" marker. Token-trimmed to `timeline_max_tokens`.
- **Evolution query with no matched entity:** no timeline (nothing to anchor);
  stratified episodic still serves the eras. Acceptable — codex-less evolution is
  exactly what era-stratification exists for.
- **Expired edge whose expiry event predates the event journal / is missing:**
  excluded from timelines (D6 discriminator errs toward silence, not noise).
- **`_deduplicate` / keyword scans:** date prefixes are per-turn-constant, so dedup
  (same-text) behavior is unchanged; keyword matching runs on lowercased text and
  `[2025-11-04]` never collides with real keywords.
- **Ablation flag off / `timescope_enabled=False`:** behavior must be byte-identical
  to today (CURRENT short-circuits every new branch; recency SQL with center=now and
  ABS is numerically identical for past rows). This is the regression guarantee.

---

## 5. Validation checklist — `tests/test_timescope.py`

Standalone live-DB script per house pattern: inserts its own rows (unique markers),
deletes them in `finally` (NEVER truncate), no LLM stubs needed (the track is fully
deterministic). Sections:

**Detector (pure):**
1. "what did I think about ice in march 2025?" → as_of, window ≈ [2025-02-15, 2025-04-15).
2. "two years ago" + question shape → as_of centered 2024-07-10, halfwidth ≈ 110d.
3. "my ideas from the first half of 2025" → range [~2024-12-11, ~2025-07-22).
4. "since january" → range [jan−pad, now).
5. "how did the saga 2 ending evolve?" → evolution (no window).
6. "how did my thinking change during 2025?" → evolution WITH window.
7. Joint gate: "I was in Paris last summer." (statement, no recall shape, p_ltm 0.1)
   → current. Same words + "what did I write about it?" → fires.
8. Future guard: "remind me next month" → current.
9. Code guard: "in 2024" inside a fenced block → current.
10. "in march" when March is future this year → last year's March.
11. `timescope_enabled=False` → always current.

**Legs & scoring (live DB):**
12. Three dated rows (2024-06, 2025-03, now) with near-identical embeddings + marker
    keyword: as_of(2025-03) retrieves ONLY the 2025-03 row (window predicate on
    vector + BM25 both).
13. An `is_archived=True` row inside the window is retrieved under as_of and NOT
    under current.
14. as_of recency re-anchor: of two in-window rows, the one nearer window-center
    outscores (same embedding); range mode: equal scores (flat).
15. `_rows_to_fragments` under current: text starts with `[YYYY-MM-DD]`; degrade_text
    carries the same stamp.
16. Codex valid_at: edge expired yesterday → absent under current, present under
    as_of(last week); edge born after the window → absent under as_of.
17. Fact line renders `(since YYYY-MM)`.
18. memory_decision: p_ltm=0.1 + timescope_mode="as_of" → retrieve=True; breakdown
    has "timescope".
19. Wide net honors window + archived rules (call `_wide_net_fallback` directly with
    a timescoped scope).

**Timeline / evolution:**
20. Fixture: entity with edge A (expired, WITH edge_expired event, reason
    "antonym_superseded") → successor edge B (live), plus edge C (expired, NO event —
    decay death). `build_entity_timeline` output contains A and B with date spans and
    the reason, does NOT contain C; `history_exists` True; for an entity with only C:
    False (no timeline fragment emitted).
21. Timeline fragment appears in `_codex_graph` output for the fixture anchor under a
    normal current-mode query (always-on enrichment), capped at
    `timeline_max_fragments`; budget keeps it (own round-robin lane).
22. Evolution mode: candidate rows spread across 4 eras → stratified output contains
    ≥1 fragment from each era (markers).
23. `entity_diff(t0,t1)` returns A in "expired" and B in "added" for the bracketing
    window.
24. `description_updated`: enrichment-style write via `log_description_update` →
    event row with old/new snippets.

**Cold & decay:**
25. Migration applied: cold_storage has the three new columns (+ count logged).
26. Cold row inside window + keyword → surfaced under as_of; NOT surfaced under
    current mode.
27. Budget-surviving cold hit with conversation_id → row back in `episodic_memory`
    (original id + original timestamp, decay 0.12, embedding NOT NULL, is_archived
    False), cold row deleted. NULL-conversation cold row → cite-only, still in cold.
28. Decay freeze fix: archived row's score decreases after `apply_decay`; archived
    row with score ≥ 0.1 (post-strengthen) is un-archived by the new clause.
29. Honest emptiness: as_of window with no matches → single `[Memory note]` fragment
    naming the window (and nearest-era dates when the unwindowed probe finds rows).
30. Regression: with no timescope in scope, a query's fragment set is identical to
    pre-change behavior modulo the date prefixes (compare against the same query with
    stamps stripped).

Also run the existing suites that touch shared code: `test_c8_c15.py` (recency SQL
params), `test_session_scoping.py` (leg SQL edits), C1/C2/C3 density suites
(`_rows_to_fragments` changed).

---

## 6. Look-ahead constraints

- **C6 (scoping rework):** timescope is an independent key in the same `scope` dict —
  cross-chat `conversation_ids` / `batch_ids` sets and the window compose as
  independent WHERE clauses. Nothing here assumes single-conversation scope.
- **C7 / D1:** no new workers, no beat entries. Resurrection is inline on the read
  path; decay edits stay inside the existing `apply_decay` callable. Timeline/diff/
  journaling are importable functions D1's agent can drive.
- **E0 (service layer):** `entity_diff`, `build_entity_timeline`, `history_exists`
  are the service candidates — keep them free of orchestrator state (db + ids in,
  data out), so E0 wraps them without surgery. Do not wire REST endpoints here.
- **F3 (graph view):** `entity_diff` is the timeline-overlay backend; month-precision
  render lives in the builder, raw datetimes in the diff dict (UI formats itself).
- **B1 (retrain):** the detector returns `evidence`; when B1 adds a temporal-recall
  label, it becomes one more evidence source feeding the same joint gate — the
  detector's window *resolution* stays deterministic regardless.
- **C17 (re-embed):** resurrection embeds with the current embedder; G23's re-embed
  runner must cover late-resurrected rows like any episodic row (it re-encodes the
  whole table — nothing special needed, just don't exempt them).
- **C4 (conversation summaries):** the evolution stratifier takes a candidate row
  list; era digests join as additional candidates later — keep the bucketing function
  row-shape-agnostic (needs only timestamp + score + the fragment fields).
- **FINAL (experiment redo):** the `timescope` ablation flag exists from day one;
  temporal probes (FINAL pt. 5) exercise as_of / range / evolution / which-is-current
  through the same public `retrieve()` path — no experiment-only seams.
- **G6:** the four new indexes land via Alembic (not the orphan SQL script).

## 7. Traps

- **Don't reintroduce a hard override in B2.** The timescope bump is large by design
  but must remain a log-odds term with breakdown telemetry — no early-return
  `retrieve=True`. (Tempting because "a time query obviously needs memory" — the
  posterior already encodes that at +3.0.)
- **Don't fork the leg SQL.** One query text per leg, param-driven filters — a
  `_vector_episodic_timescoped` twin is the obvious-simpler version and it rots
  instantly (G19; the ablation subclass would silently diverge).
- **Don't traverse expired edges in evolution mode.** History is the timeline
  builder's job; dead-edge navigation produces incoherent neighborhoods (D5).
- **Don't put decay-expired edges in timelines.** The eventless-expiry discriminator
  (D6) is the whole difference between "my idea changed" and "the system forgot" —
  collapsing them makes every timeline lie.
- **Don't stamp clock times, only dates** — and don't date the recent sliding window
  (it's "now"; dating it burns tokens and invites the model to over-narrate).
- **Don't resurrect on candidacy.** Only budget-survivors (actually-injected
  memories) get the second chance; resurrecting every cold *candidate* would bulk-
  revive garbage on one nostalgic query.
- **Don't restore resurrected rows at high decay or current timestamp.** Original
  timestamp + probation score are what make the mechanic honest (D-U1).
- **Don't let cold search run unbounded.** It requires a window; keywords optional
  only because the window bounds it. No window → no cold leg.
- **Don't resolve vague pasts** ("a while back", "long ago") into invented windows —
  they're evidence for the gate, never a window. An invented window silently hides
  the memories outside it.
- **Don't parse time with the bg LLM.** Latency on the synchronous path, and
  deterministic parsing is strictly more debuggable (A6 philosophy). The lexicon is
  the extension point.
- **Don't "fix" the ABS in the current-mode recency formula** — with center=now it is
  numerically identical for past rows; unifying the formula is the point (one code
  path, mode-parameterized).
- **Don't skip the G26 hotfix ordering** — T2's main.py edit sits after
  classification; if G26 isn't fixed first, you are editing a function that crashes
  on every request and validation will "fail" for an unrelated reason.
