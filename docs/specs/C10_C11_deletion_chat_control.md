# C10 + C11 — Conversation deletion cascade · Chat memory commands

Assumes decided specs: `E0_E7_services_mcp.md` (both are services; C11 is adapter
#3 — parser only), `T_temporal.md` (journaled expiries; timeline exclusion rule
amended here), `D1_D2_maintenance_agent.md` (fuzzy-destructive ops → review
queue, matching its tiers), `C4_C9_memory_quality.md` (conversation_summaries
cascade; tiered slot targets for commands), `C7_scheduling.md` (no Celery
anywhere). Grounded at commit `b8c2122`: episodic_chunks CASCADE from parent
(C2 migration), cold_storage now carries conversation_id (T3), corroboration
evidence = distinct `batch_source` count on an edge's `edge_added` events,
CuratedLabel carries batch_id, EpisodicClusterLink composite-PK link rows.

> **[rev 2026-07-19 — implementation-session re-grounding at `e02b21b` (rule 12);
> the `b8c2122` grounding predates E0/E7, D1/D2, the E-core, E11 AND C4/C9.
> Fifteen refinements, none reversing a decision:**
> 1. **Cascade members the grounding commit couldn't see:** `session_summaries`
>    (conversation FK, would block the row delete) and conversation-tier
>    `memory_slots` (C9's `conversation_id` FK) are deleted with the
>    conversation; live `decisions` rows with `source_batch ∈ batches` (E-core)
>    are **expired** (`valid_until = now` — the table's own bi-temporal model,
>    no event journal by design). `conversation_summaries` now CASCADEs at the
>    DB level (C4) — the manifest counts it and the service still deletes it
>    explicitly so the count is collected before the FK fires.
> 2. **Corroboration = external evidence, not raw event count.** "≥2 distinct
>    `batch_source`" read literally would keep an edge double-extracted inside
>    the *deleted* conversation. The decision's own rationale ("the fact stands
>    on other conversations") is the rule: keep iff an `edge_added` event exists
>    with `batch_source ∉ batches`; otherwise expire. Verified payload shape:
>    `edge_added` events carry `payload->>'edge_id'`, provenance in the event's
>    `batch_source` column.
> 3. **Entity "expiry" mechanics pinned** (CodexEntity has no liveness column —
>    D1/D2 rev 4): mirror the merge-husk pattern — canonical_name +
>    ` [deleted:<id8>]`, aliases emptied, `properties.deleted_reason=
>    'source_deleted'`, `context_payload = "[deleted]"`, embedding NULLed,
>    journaled `entity_expired` event. "User-authored description" = a
>    `description_updated` event with `payload->>'source'` ∈
>    {mcp_edit, manual_edit}.
> 4. **`_expire_edge` already takes `source=`** (D1/D2 rev 1); deletion expiries
>    pass `reason="source_deleted"`, `source="user_deletion"`, one fresh
>    deletion batch id per run. Surviving entities touched by expired edges get
>    `_regenerate_context_payload` (their notes must not list dead links).
> 5. **The T-amendment is already implemented** — evolution.py's
>    `_expiry_events`/`history_exists` drop `reason='source_deleted'` (shipped
>    with T4). C10 owes no evolution.py change; §4 check 4 validates it.
> 6. **Step-7 'stale' concretized:** only two pending item types can reference a
>    conversation's batches — `codex_reconciliation` (via `old_edge_id` →
>    edge.source_batch) and `decision_supersession` (via new_id/old_id →
>    decision.source_batch) — plus this spec's own `forget_request` (stale when
>    its originating conversation is the deleted one, or a listed turn id is;
>    approval of a survivor tolerates vanished rows either way). Status
>    vocabulary is now pending/approved/rejected/resolved(+stale); Text
>    column, **no migration**.
> 7. **Empty-cluster rule covers both membership mechanisms** — link rows AND
>    the legacy `episodic_memory.cluster_id` FK; surviving clusters born from
>    the conversation (`context_clusters.conversation_id` FK) get the anchor
>    NULLed (else the row delete FK-fails). Both counted in the manifest.
> 8. **Mid-generation refusal is in-process:** `runtime.generation_in_flight`
>    (new public property over C7's `_generation_inflight`). The MCP process
>    cannot see the app's stream — accepted single-user reality; the realistic
>    mid-stream deletion path (the app's own REST/frontend) is covered.
> 9. **`try_handle` gains `scope`:** `try_handle(db, runtime, conv, text,
>    scope)` — main.py has already built the scope dict (incognito/project
>    keys) at the insertion point (after the G26 conversation-resolution +
>    scope block, before `classifier.classify`); `/search` passes it to
>    `context_for` verbatim (chat-path parity incl. the E11 freshen; auto
>    scope stays {} = global, exactly like `ice_context` without a
>    conversation).
> 10. **Handled commands skip episodic storage AND post-flight entirely** — a
>    stored "/slots" turn would pollute retrieval; the `chat_command` journal
>    + `updated_by` tags are the record. The confirmation streams as OpenAI
>    `chat.completion.chunk` SSE (`model="ice-commands"`, one content chunk +
>    finish + `[DONE]`) so any OpenAI-compatible frontend renders it.
> 11. **`/delete-conversation` confirm state is process-local** (pending map,
>    10-min TTL; a restart clears it = safe refusal). "confirm" with nothing
>    pending → the idempotent hint; after deletion the entry is popped, so a
>    double confirm hits the same hint (and the service raises NotFound).
> 12. **`/forget` arm naming:** `item_type="forget_request"`; the review-approve
>    arm dispatches to the conversations service's `apply_forget` (turn deletes
>    + edge expiries `reason="user_forget"`, `source=` the proposer). A /forget
>    with zero matches queues nothing and says so. Matching: episodic top-5 by
>    embedding (visibility-guarded, **PgVector bindparam** — the C9 lesson) +
>    live conversation-source edges whose entity names (≥3 chars) appear in the
>    text.
> 13. **`/scope` passes through existing cluster_ids/custom_filter** —
>    `set_scope` overwrites them otherwise (a bare `/scope auto` must not wipe
>    cluster assignments); `project=None` leaves attachment untouched.
>    **⚠ SUPERSEDED by C6 (2026-07-28).** The passthrough is deleted because
>    the footgun it worked around is gone: every id-set parameter on
>    `set_scope` is now `None` = leave unchanged, `[]` = clear, so a bare
>    `/scope` cannot wipe anything. `custom_filter` no longer exists (dropped,
>    migration `a1f6b8d94c22`), and `/scope` now also accepts `manual`.
> 14. **`/bookmark` = `latest_turn` + `bookmark_turn`** (the stored-turn
>    reality: the just-streamed reply isn't stored until post-flight; a fresh
>    conversation gets a clear "nothing stored yet" instead of NotFound).
> 15. **REST adapters here are sync `def` + `service_errors()`** (house
>    pattern) — the DELETE endpoint follows; G24's to_thread applies only to
>    the async `chat_completions` caller, which wraps `try_handle` in
>    `asyncio.to_thread`.]**

## 1. Decisions

### C10 — deletion with correct cascade semantics
- **D1: one service, manifest-first.**
  `services/conversations.py::delete_conversation(db, conv_id, dry_run=False)`
  → manifest dict (per-store counts). `dry_run=True` powers the confirmation
  dialog (F) and the C11 `/delete-conversation` confirm step. One transaction.
- **D2: cascade order and rules:**
  1. Collect the conversation's `batch_ids`.
  2. **Codex:** for each live edge whose `source_batch ∈ batches`: corroborated
     (≥2 distinct `batch_source` values across its `edge_added` events) → keep
     (the fact stands on other conversations); sole-support → **expire** with an
     `edge_expired` event, reason `source_deleted` (journaled, auditable).
     Entities left with zero live edges, `source='conversation'`, and no
     user-authored description → expired the same way. **T-track amendment
     (recorded in T_temporal.md too): timeline rendering EXCLUDES reason
     `source_deleted`** — deletion is not idea evolution.
  3. **Episodic:** delete rows (chunks CASCADE); delete the conversation's
     `cold_storage` rows (T3 columns make this possible); delete cluster link
     rows; clusters left empty → deleted.
  4. **Procedural:** prune `source_batch_ids`; `is_active=False` when emptied.
  5. **Summaries/replays:** `batch_summaries`, `conversation_summaries` (C4),
     `session_replays` rows deleted.
  6. **Training hygiene:** `curated_labels` rows with `batch_id ∈ batches`
     deleted (B1's corpus builder never resurrects deleted content).
  7. **Review queue:** pending items referencing the conversation's batches →
     status `stale`.
  8. Conversation row last. Logs are NOT touched — G25 owns log redaction; the
     manifest says so explicitly (honesty over pretense).
- **USER-REQUIRED:** none at service level; the F-side confirm dialog is owed
  (ledger). Until F exists, deletion is reachable via REST + `ice_control`.

### C11 — chat commands (adapter #3)
- **D3: deterministic slash-prefix parser, pre-classification.** In
  `chat_completions`, a user message whose first line starts with `/` routes to
  `src/api/chat_commands.py::try_handle(db, runtime, conv, text)` → handled
  commands short-circuit the LLM entirely and stream a formatted confirmation as
  a normal SSE completion (any frontend renders it). Unrecognized `/x` → a help
  hint, NOT silent fallthrough (a typo'd command must never leak into chat as a
  prompt). Natural-language commands without `/` are explicitly v2 (false-
  positive risk; the MCP path already covers conversational control in
  harnesses).
- **D4: command set v1** (each handler ≤5 lines over E0 services):
  `/remember <text> [in <slot>] [@project|@conversation]` (append; default
  global `pending_items`; C9 tiers) · `/slots [tier]` · `/bookmark` (last
  assistant turn) · `/search <query>` — forced-retrieval via `retrieval_svc`,
  results rendered deterministically (dated per T1, source-tagged), zero LLM ·
  `/scope auto|project|none` · `/forget <text>` — **fuzzy + destructive ⇒
  review-queue proposal** (top matching turns/edges listed for approval; D1-tier
  consistency: precise writes apply, fuzzy destructive ops queue) ·
  `/delete-conversation` → dry-run manifest shown, requires
  `/delete-conversation confirm` within the same session · `/help`.
- **D5: every applied command journals** (`updated_by='chat_command'`, G17
  source tags; `mcp_tool_call`-style structlog event `chat_command` for F5).
- **Empirical deferral:** command discovery (do commands get used without UI
  affordances?) — F owns discoverability; telemetry counts per command inform
  the F design conversation.

## 2. Files & integration points

`services/conversations.py` (new) · `src/api/chat_commands.py` (new) ·
`main.py` (the `/`-check before classification — sits AFTER the G26 fix's
conversation resolution, it needs `conv`) · T_temporal.md §2.7 amendment line
(done with this spec) · REST adapter route `DELETE /user-control/conversations/
{id}` + `ice_control` actions `delete_conversation`, `forget_propose` ·
`tests/test_c10_c11.py`.

## 3. Edge cases

Sole-support check with legacy edges that predate the event journal (no
edge_added event rows): treat as sole-support only if `source_batch ∈ batches`
(their own provenance is the only evidence — expire; conservative is keeping
facts, but unprovenanced facts from a deleted conversation violate the
deletion's meaning; decision: expire, journaled). Conversation mid-stream
(active generation) → deletion refused with a clear error. `/search` in an
incognito conversation → scope rules apply unchanged (own-conversation only).
`/remember @project` with no attached project → error naming the fix. Command
text containing newlines → only the first line is parsed as the command, rest
is the payload. Double `confirm` → idempotent NotFound.

## 4. Validation checklist — `tests/test_c10_c11.py`

1) fixture conversation A asserts edge X (sole) and edge Y (also asserted by
conversation B): delete A → X expired with `source_deleted` event, Y alive;
entity with only X expired; B untouched; 2) episodic+chunks+cold+links+
summaries+curated rows gone; empty cluster deleted; manifest counts match;
3) dry_run deletes nothing, manifest identical; 4) T4 timeline for X's entity
does NOT show the deletion as a transition; 5) `/remember` writes the right
tier + journals; `/slots` renders; 6) `/search` returns dated fragments with no
LLM call (assert no bg/chat client invocation); 7) `/forget` creates a
review-queue proposal, applies on approve via D6's dispatch; 8) unknown command
→ help hint, never forwarded to the model; 9) `/delete-conversation` two-step
confirm works, single-step refused.

## 5. Look-ahead constraints

F2/F renders manifests and proposals as-is (self-describing JSON). B6's future
branch deletion reuses the same cascade per-branch (lineage = conversation set —
A5's batch-set primitive again). G25's log caveat is restated in the manifest
string so the user-facing claim never overpromises.

## 6. Traps

- Don't hard-delete codex edges on deletion — expire+journal keeps the graph's
  referential integrity (events reference edge ids) and the audit trail honest.
- Don't let `/forget` fuzzy-match-and-apply — it queues; the day a fuzzy match
  eats the wrong memory is the day trust dies.
- Don't parse commands after classification — they'd hit B2/retrieval and burn
  latency before the parser even looks.
- Don't render `/search` results through the model "to make them nicer" — the
  command's contract is showing what memory actually holds.
- Don't cascade into logs and claim full erasure — say exactly what was and
  wasn't removed (manifest), and point at G25.
