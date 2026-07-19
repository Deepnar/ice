# C4 + C9 — Whole-conversation summaries · Procedural widening + three-tier slots

Assumes decided specs: `C7_scheduling.md` (session-end burst hosts the summary
job), `E_coding_core.md` (`projects` exists; `procedural_memory.project_id` added
there), `E0_E7_services_mcp.md` (slots service is the single write path; the
7-name constant lives in one place), `D1_D2_maintenance_agent.md` (slot proposals
via tiers), `T_temporal.md` (C4 summaries are future era-digest candidates — seam
only). Grounded at commit `b8c2122`: `_procedural_lookup` hard-gates on intents
`{Strategic_Planning, Generation, Open_Exploration}` (orchestrator.py:1398) and
then embedding-ranks with `_procedural_trigger_match`; `memory_slots.py` has the
fixed 7-slot list + versioned upserts; `prompt_assembler` injects all active
slots flat.

> **[rev 2026-07-19 — re-grounding at implementation time (`cd5c1b4` HEAD; the
> b8c2122 grounding predates Track T, E0/E7, D1/D2, the E-core AND E11). Ten
> refinements, none changing a decision:**
> 1. `VALID_SLOTS` + all slot helpers live in `src/services/slots.py` (E0);
>    `memory_slots.py` is a thin REST adapter. D5's dict lands in the service;
>    the router/MCP/`session_start_block` all inherit. Callers audited: only
>    `retrieval_svc.session_start_block`, `mcp/server.py` (`ice_slots`,
>    `ice_remember`), the router, and `test_services` — all keyword-safe.
> 2. `memory_slots` has **no unique constraint on slot_name today** (only the
>    id pkey; the `initialize_slots` docstring claims one — a lying comment).
>    The migration adds D5's composite uniqueness as a **`NULLS NOT DISTINCT`
>    unique index** (pg16) so `(name, 'global', NULL, NULL)` can't duplicate.
> 3. `_procedural_lookup` is at orchestrator.py:1668 (not 1398), takes `scope`,
>    and since E1 carries project logic (project-scoped patterns invisible
>    outside their project, batch-exempt inside) + T3's timescope span filter.
>    D4 extends the SQL with the confidence floor and deletes ONLY the
>    3-intent early-return; everything else stays.
> 4. **Extraction side of D4: nothing to kill.** Post-C7, `post_flight`
>    chains `extract_procedural` for every non-private turn — no intent
>    whitelist exists at dispatch anymore; the LLM's NONE output is the cue
>    filter. Audit result recorded, no change.
> 5. C4's job joins the session-end burst in `runtime._maybe_session_end_burst`
>    (the D1/D2 trio becomes a quartet) + the `JOBS` dict + a
>    `maintenance_intervals` entry (7200, aligned with `batch_summarize` —
>    burst-members are also cadence-run; the job is idempotent per
>    `covers_through`, so cadence passes on quiet conversations are no-ops).
> 6. D3a's injection condition compares main.py's existing `total_tokens`
>    against `estimate_recent_window_tokens(turn_count, total_budget)` (B2's
>    window estimate, C16-aware). The summary JOB gates row creation on the
>    same condition with the legacy default budget (the job doesn't know the
>    routed model; the assembler re-checks at injection, so a mismatch only
>    costs an early/late row, never a wrong injection).
> 7. The assembler grew `session_start_text` (E4) since grounding: system-msg
>    block order is PERSISTENT CONTEXT (tiered slots) → PROJECT SESSION START
>    → CONVERSATION SUMMARY (still before the retrieved-context message).
> 8. D3b: `_batch_summary_lookup`'s batch half stays conversation-scoped
>    (as built); the `conversation_summaries` half searches cross-conversation,
>    **excludes the active conversation** (its summary arrives via the
>    assembler — the double-inject trap) and excludes `memory_scope_type =
>    'none'` conversations via the join. Both halves keep the T3 non-current
>    skip and the `batch_summary` source_type (leg weights unchanged).
> 9. D7 rides what D1/D2 built: `review.py::approve`'s `memory_slot_update`
>    arm re-routes through the slots service (tier params + G14 cap enforced
>    there); proposal payloads (reflection + agent stale_slot) gain
>    `proposed_by`, and `updated_by` on application records the proposer
>    (`reflection`/`agent`), not `"user"`.
> 10. C4 summaries include a conversation's own private turns (incognito gets
>    its own context; the retrieval consumer's join is the shield), matching
>    the edge-case note; `covers_through`/`covers_turns` advance only on
>    successful LLM output — failure keeps the old row (retry next burst).
> **The C1 machinery reused by D2 is `turn_density.extract_key_terms` /
> `must_terms` / `summary_coverage` + the post_flight retry idiom; the job
> takes an injectable `llm=` + `embedder=` (house test pattern) and lazy-loads
> the shared embedder (no import-time model load).]**

## 1. Decisions

### C4 — conversation summary object
- **D1: one evolving summary per conversation**, new table:
  `conversation_summaries(conversation_id UUID PK FK, summary_text TEXT,
  covers_through timestamptz, covers_turns INT, embedding vector, updated_at)`.
  NOT another `batch_summaries` range row — the object's contract is "the whole
  conversation so far, current."
- **D2: maintained incrementally at session-end** (C7 burst job
  `conversation_summary`): for each conversation with turns newer than
  `covers_through` — prompt = existing summary + the new turns (C1
  representations) → revised ≤250-word summary, **grounded C1-style** (must-keep
  terms = top recurring NER entities of the conversation; one retry on coverage
  miss). Cheap: one bg-model call per active conversation per sitting.
- **D3: two consumers now.** (a) **Assembler**: for the active conversation, when
  `total turns > the sliding window's reach` (the B2 memory-pressure condition
  already computes this), inject the summary (~200 tokens) as a
  `=== CONVERSATION SUMMARY ===` block before retrieved context — the model
  answering "in isolation" finally has global conversation shape; prerequisite
  for ever shrinking the window. (b) **Retrieval**: `_batch_summary_lookup`
  extends to search `conversation_summaries` embeddings alongside batch
  summaries (cross-conversation overview hits). T4's era-stratified sampling may
  later consume these as digests — no work now (seam noted there).
- **Edge/validation hooks:** stale summary (covers_through < last turn) is still
  injected but stamped "(as of N turns ago)"; C10 deletes the row with its
  conversation; incognito conversations get summaries (their own context) but the
  retrieval consumer honors `is_private` via the conversation join.

### C9 — procedural widening + slot tiers
- **D4: kill the leg's hard intent gate.** `_procedural_lookup` always runs;
  precision comes from the three signals that already exist — embedding score
  (keep the LIMIT 5), `_procedural_trigger_match` (topic/intent/keyword
  conditions), and a new floor `confidence_score ≥ settings.procedural_min_conf`
  (default 0.3). The blend weights already down-weight the leg per intent
  profile; a hard gate on 3 intents is why nobody ever *felt* procedural memory.
  Extraction side: audit the dispatch gate at implementation (grep post-flight
  chain) and apply the same principle — extract on pattern cues, not intent
  whitelists; decay + reinforcement already handle junk.
- **D5: slots go three-tier** (global / project / conversation): columns
  `scope_tier TEXT DEFAULT 'global'`, `project_id UUID NULL`,
  `conversation_id UUID NULL`, unique on `(slot_name, scope_tier, project_id,
  conversation_id)`. Valid names per tier: global keeps today's 7; project =
  `{project_context, conventions, pending_items, guidance}`; conversation =
  `{conversation_focus, pending_items}`. The single `VALID_SLOTS` constant
  becomes a per-tier dict in the slots service (E0) — routers/MCP/C11 all
  inherit it.
- **D6: assembly order** — global slots, then attached-project slots, then
  conversation slots, under the existing PERSISTENT CONTEXT header with tier
  prefixes (`[PROJECT · CONVENTIONS]`). Coding-scoped conversations therefore
  inherit project slots automatically (the E1 payoff).
- **D7: agentic updates route through the existing safety model** — D1-agent
  proposals (Tier 2 → review queue) and C11 chat commands (precise writes apply +
  journal) both call the same slots service; `updated_by` records
  `user|agent|reflection|chat_command`. **G14 folds in here:** the service
  enforces a 300-token cap per slot (hard-truncate + warning in the response).
- **USER-REQUIRED:** none.
- **Empirical deferral:** `procedural_min_conf` and whether widened extraction
  floods — Z1 checks pattern count growth + injection rate; rule: >30
  active patterns with <5% injection rate → raise the floor to 0.5.

## 2. Files & integration points

Migration (conversation_summaries + slot columns/constraint) ·
`src/workers/conversation_summary.py` (new job; C7 table entry) ·
`prompt_assembler.py` (summary block + tiered slot rendering) ·
`_batch_summary_lookup` (union query) · `_procedural_lookup` (gate removal +
floor) · `services/slots.py` (tier dict, cap, tier params) · E7 `ice_slots`
(tier arg) · `tests/test_c4_c9.py`.

## 3. Edge cases

Conversation with 2 turns → no summary row until it crosses the window (D3a
condition); summary LLM failure → keep old row, log, retry next burst; slot tier
writes without project attached → ValidationError naming the missing attachment;
legacy rows default `scope_tier='global'` (backfill in migration); two projects
sharing a conversation is impossible by schema (one project_id).

## 4. Validation checklist — `tests/test_c4_c9.py`

1) summary job (stub LLM) creates + incrementally updates the row
(covers_through advances, old content carried into the prompt); 2) assembler
injects the block only past the window condition, with staleness stamp when
behind; 3) retrieval consumer finds another conversation's summary by embedding,
never a private one; 4) procedural leg returns patterns for a
previously-gated-out intent (Factual_Retrieval) when trigger+floor pass, and
still returns [] below the floor; 5) tiered slots: create/read per tier, unique
constraint enforced, assembler renders the three tiers in order, 300-token cap
truncates; 6) C10 hook: deleting a conversation removes its summary row
(asserted in C10's suite too).

## 5. Look-ahead constraints

T4 era digests read `conversation_summaries` shape as-is; F4/F surfaces expose
tiered slot editing (ledger already lists it); B2's window estimate is the
injection condition's source of truth — if C16's need-based filling changes it,
this condition follows automatically (one function, `estimate_recent_window_tokens`).

## 6. Traps

- Don't summarize per-turn (session-end only — C7's whole point).
- Don't let the summary replace retrieval ("it's already summarized") — it's
  orientation, fragments are evidence; keep it ~200 tokens.
- Don't widen procedural by deleting `_procedural_trigger_match` — the gate to
  remove is the intent whitelist, not the relevance machinery.
- Don't add a fourth slot tier "while we're here" (user/global/project/convo is
  C9's full scope; E1's per-project knowledge units cover the rest).
