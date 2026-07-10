# D1/D2 — Memory Maintenance Agent + Sentinel removal

Assumes decided specs: `C7_scheduling.md` (the agent is a runtime job — session-end
burst + overdue interval — using the ledger, lanes, and callables; NO Celery),
`T_temporal.md` (every graph mutation is journaled; description writes via
`log_description_update`). Grounded at commit `45ef591`: A6's callables
(`check_conflict` / `reconcile_conflict(db, conflict, subj, relation, obj, batch_id,
turn_text, reconciler)` / `make_llm_reconciler`, codex_extractor.py:727–840),
`review_queue` writers (reflection:247,360; sentinel:125; codex_extractor:802) and
reader (user_control:183–200), sentinel_monitor.py (one real query: pending-edge
pileup; one real absence check: stale `pending_items` slot; the rest stubs).

## 1. Decisions

- **D1: the "agent" is a deterministic worklist + bounded LLM decisions — not a
  free-roaming tool loop.** A6's philosophy scaled up: cheap SQL detectors produce
  typed work items; the LLM (bg model via C7's gpu lane, guided/JSON decoding)
  makes a **per-item decision from a fixed enum**; execution goes through named
  callables. A 3B/4B model gets reliability from constrained choices, not from
  open-ended planning. "Tool-using" = the decision dispatches tools; the loop
  itself is deterministic.
- **D2: three risk tiers govern write authority (minimal, per the roadmap):**
  - **Tier 0 — auto, no LLM:** normalization-equal duplicate merge ("FastAPI" vs
    "fastapi" after casefold/strip), re-queueing A7.3 enrichment for well-mentioned
    description-less entities, `run_cluster_merge`. Deterministic and reversible-ish.
  - **Tier 1 — LLM-decided, auto-applied, journaled:** re-running A6 reconciliation
    on its review-queue leftovers WITH more context (the source turns of both
    edges — the in-line reconciler only saw one turn); expiring the weaker of an
    exact-duplicate edge pair. Every action emits `CodexEvent`s (D4).
  - **Tier 2 — LLM-proposed, review-queued:** non-trivial entity merges, slot
    content rewrites, anything deleting user-visible knowledge. The queue stays the
    safety net; the agent may write ≤5 proposals per run (no queue flooding —
    today's queue already rots unwatched until F2).
- **D3: worklist detectors (all deterministic SQL, run first, capped):**
  1. `review_queue` rows `item_type='codex_reconciliation'`, status pending →
     Tier-1 re-reconcile with fuller context; if still ambiguous → leave, mark
     `item_content.agent_attempts += 1` (≥2 attempts → never retried again).
  2. **Pending-edge pileup** (the sentinel's one real query, ported verbatim):
     entities with >N pending edges overlapping active ones → Tier-1 dedupe/expire.
  3. **Duplicate-entity candidates:** pair generation = casefold/alias intersection
     ∪ embedding cosine ≥ 0.90 (both embeddings non-null) ∧ same `entity_type`;
     rank by cosine; top 10 pairs/run → Tier 0 if normalization-equal else Tier 2
     proposal with the LLM's one-word verdict (`same`/`different`/`unsure`;
     only `same` proposes).
  4. **Contradiction backlog:** live positive edge + live negated same-(src,rel,tgt)
     (A8 residue), or two live antonym edges for one pair (A6 cross-batch misses) →
     Tier 1 via `reconcile_conflict`.
  5. **Stale-slot check** (sentinel's absence rule, ported): `pending_items` slot
     untouched >14d → Tier-2 proposal "slot review" (content suggestion generated
     by the LLM from recent turns; user approves via queue/F2).
- **D4: full auditability.** Each run gets an `agent_run_id`; every action logs
  structlog `agent_action` + writes `CodexEvent`s through the existing helpers
  (merge emits `edge_expired`/`edge_added` chains + `description_updated` when note
  bodies combine; `batch_source = agent_run_id`). G17's source-annotation lands
  here for agent writes (`source: "maintenance_agent"` in event payloads).
- **D5: `merge_entities(db, keep_id, absorb_id, agent_run_id)` is a new callable**
  (new module `src/workers/codex_ops.py` — codex_extractor is big enough): re-point
  `absorb`'s edges to `keep` (dropping exact-duplicate edges, keeping max
  strength/confidence), union aliases (absorb's canonical_name becomes an alias),
  keep the longer description (journaled), move codex_events entity_id, regenerate
  `context_payload`, expire the absorbed entity (row kept with alias marker — no
  hard delete; T-track history must survive). Approval-applied for Tier 2.
- **D6: review-queue approval must APPLY, not just flip status.** The approve
  endpoint already applies `memory_slot_update` and `new_cluster_proposal`
  (user_control.py:198–213) — extend its dispatch table with the agent's types:
  `entity_merge` → `merge_entities`, `codex_reconciliation` → the chosen expire
  action. F2's panel then gets working buttons for free (E0 lifts this dispatch
  into the review service).
- **D7 (=roadmap D2): the Sentinel dies here.** Delete `sentinel_monitor.py`, the
  `SentinelRule`/`SentinelEvent` models + tables (alembic drop), its runtime
  interval entry (C7 ported it as-is until now), and stray imports. Its two real
  checks live on as detectors 2 and 5. No standalone completion, no parallel
  notification system (audit verdict).
- **D8: scheduling** (C7 vocabulary): job `maintenance_agent`, gpu lane,
  session-end burst + overdue 12h; caps per run: ≤50 detector items scanned,
  ≤25 LLM decisions, ≤5 Tier-2 proposals, ≤10 Tier-0/1 applications. Idempotent:
  detectors re-derive state; caps make runs incremental.
- **Empirical deferral (rule 2b):** the cosine threshold (0.90) and pileup N — Z1
  produces the candidate lists; rule: if >20% of `same` verdicts are wrong in the
  first manual review of proposals, raise to 0.94; if zero candidates ever appear,
  lower to 0.85. No design change.

## 2. Algorithm

```
run_maintenance_agent(db, llm_decider=None):        # llm_decider stubbed in tests
    run_id = uuid4()
    items = detect_all(db, caps)                    # D3, typed WorkItems
    for item in items[:25]:
        decision = decide(item, llm_decider)        # fixed enum per item type; guided JSON
        apply(decision, tier_rules, run_id)         # callables; Tier-2 → review_queue
    log agent_run_summary (counts per type/tier/outcome)
```
`decide` prompt per item type is a short template: the two edges/entities with
their source-turn texts (batch provenance → episodic lookup) + enum instruction;
one completion, temperature 0, `format=json`. Unparseable → treat as `unsure`
(never act on garbage).

## 3. Files & integration points

`src/workers/maintenance_agent.py` (new: detectors, decide, apply, run callable) ·
`src/workers/codex_ops.py` (new: `merge_entities`; future codex surgery lives here)
· `src/api/routers/user_control.py` (approve-dispatch, D6) · C7 runtime job table
(+`maintenance_agent`, −`sentinel_monitor`) · alembic (drop sentinel tables) ·
delete `src/workers/sentinel_monitor.py`, remove `Sentinel*` from models ·
`tests/test_maintenance_agent.py`.

## 4. Edge cases

- Merge cycle guard: a pair proposed and rejected by the user is stamped in
  `item_content`; detectors skip previously-rejected pairs (query review_queue
  rejected rows by pair key).
- Merge target ordering: keep = higher mention count, tie-break older `valid_from`
  of first event; never merge two entities that share zero edges AND zero alias
  overlap on embedding alone (cosine-only pairs need the LLM verdict, Tier 2).
- Concurrent chat extraction while agent runs: gpu-lane serialization (C7) prevents
  simultaneous LLM use; DB races are benign (idempotent detectors, journaled ops);
  merge re-checks both entities still live inside its transaction.
- Empty DB / no bg model: detectors return nothing / `llm_decider=None` skips
  Tier 1-2 decisions (Tier 0 still runs) — the job degrades gracefully.
- Legacy sentinel rows: the drop migration archives `sentinel_rules` contents into
  a log line first (they were seed rules; nothing user-authored).

## 5. Validation checklist — `tests/test_maintenance_agent.py`

Live-DB, stub decider, self-cleaning: 1) exact-dup entities → Tier-0 auto-merge
(edges re-pointed, aliases unioned, events journaled, absorbed entity expired);
2) near-dup pair → Tier-2 `entity_merge` proposal (stub says `same`), applied on
approve via the D6 dispatch → merged; rejected pair never re-proposed; 3) pileup
fixture → pending duplicates expired (Tier 1) with events; 4) positive+negated
contradiction → reconciled; 5) stale `pending_items` fixture → slot proposal
written, cap of 5 respected; 6) `agent_attempts` ≥2 skips; 7) caps respected
(seed 30 items → 25 decisions); 8) unparseable LLM output → `unsure`, no write;
9) sentinel tables/model/worker gone; runtime table has agent, not sentinel;
10) every applied action has matching CodexEvents with `agent_run_id`.

## 6. Look-ahead constraints

- **E0:** `run_maintenance_agent`, `merge_entities`, and the approve-dispatch are
  service candidates — keep them router-free and orchestrator-free.
- **F2:** proposals must render from `item_content` alone (self-describing JSON:
  pair names, evidence, suggested action) — the panel is a dumb renderer.
- **D3-roadmap (agentic telemetry):** `agent_action`/`agent_run_summary` structlog
  events are the SSE promotion candidates — name them now, promote in F5.
- **E1 decisions table:** reuses this exact pattern (reconcile via A6 machinery,
  agent maintenance) — keep detectors table-agnostic where cheap (item types are
  pluggable).
- **G17:** agent writes carry `source` in payloads — the audit-trail column can
  backfill from events.

## 7. Traps

- **Don't build a free tool-loop agent** ("let the model decide what to do") — a
  4B model with open tools on the user's memory graph is how memories get eaten;
  enums + tiers + caps are the design, not a v1 compromise.
- **Don't hard-delete anything** — absorbed entities are expired-with-alias, edges
  expire via the journal; T-track's history depends on it.
- **Don't let the agent write review-queue items unboundedly** — 5/run, and skip
  previously-rejected proposals; a flooded queue is worse than no agent.
- **Don't complete the Sentinel "while we're here"** — it's a removal (audit
  verdict: log/noop actions, `return False` stubs); resist the sunk-cost pull.
- **Don't run the agent per-turn** — it's a maintenance-window job (session-end /
  12h); per-turn graph surgery races live extraction for no benefit.
