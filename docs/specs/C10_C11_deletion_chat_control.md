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
