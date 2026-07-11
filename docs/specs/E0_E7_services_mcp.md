# E0 + E7 — Service layer extraction + ICE-MCP server

Assumes decided specs: `C7_scheduling.md` (no Celery; `create_core()` boots DB +
maintenance runtime; ledger lease prevents double runtimes; `runtime.enqueue`
replaces `.delay`/`.apply_async`), `D1_D2_maintenance_agent.md` (review-dispatch
extension lives in the review service), `T_temporal.md` (`build_entity_timeline`,
`entity_diff`, `log_description_update` are the graph-history services). Grounded
at commit `79f9687`: [user_control.py](../../src/api/routers/user_control.py)
(bookmarks, label override, scoping incl. G16 privacy sync, clusters, review queue
with partial apply-dispatch, TUI helper, model-registry CRUD; note
`bookmark_turn` calls `extract_codex.apply_async` — killed by C7 2026-07-11:
it now enqueues the "codex_extract" job on the maintenance runtime),
[memory_slots.py](../../src/api/routers/memory_slots.py) (7 fixed slots, CRUD +
initialize), `model_registry/registry.py` (file-backed load/save/populate).

User decisions (2026-07-10): **headless boot = attach-if-running, else boot core,
with linger**; **write tools apply immediately + journal everything** (chat has
the same authority as the frontend edit button).

## 1. Decisions

- **D1: services are plain modules under `src/services/`, one per domain, zero
  FastAPI imports, zero HTTP concepts.** `bookmarks.py`, `slots.py`, `scoping.py`,
  `clusters.py`, `review.py`, `registry_svc.py`, `graph.py`, `retrieval_svc.py`.
  Signature convention: `fn(db, ...) -> dict | dataclass`; failures raise domain
  errors (`NotFoundError`, `ValidationError` in `services/errors.py`). Routers
  become thin adapters (parse → service → format; translate domain errors to
  HTTPException). E7's MCP tools and C11's chat commands are adapters two and
  three over the SAME functions — that is the whole point of E0.
- **D2: behavior-preserving extraction, with exactly two intentional changes:**
  (a) `bookmark_turn`'s `extract_codex.apply_async(...)` becomes
  `runtime.enqueue("codex_extract", batch_id=..., priority=True)` (C7 world);
  (b) the review-approve dispatch moves into `review.py::approve(db, item_id)`
  and gains D1/D2's `entity_merge` + `codex_reconciliation` arms. Everything else
  extracts verbatim — G16's privacy re-sync in `scoping.set_scope`, slot
  versioning, registry file I/O.
- **D3: `graph.py` is the codex service surface:** `entity_view(db, name_or_id)`
  (note + links + backlinks + type), `entity_edit(db, id, description=..., ...)`
  — **writes `description`, never `context_payload`, journals via T's
  `log_description_update(source="mcp_edit"| "manual_edit")`, then regenerates
  the payload** (the A7/F3 rule, now enforced in one place), `edges_list`,
  `entity_timeline` (T's builder), `entity_diff` (T's diff). F3 will consume
  these same functions.
- **D4: `retrieval_svc.py::context_for(db, task_text, scope=None, budget=None)`**
  wraps classify → memory-decision → orchestrate → assemble-fragments and returns
  structured fragments (text, source_type, score, provenance ids) — NOT a rendered
  prompt. It is `ice_context`'s engine, F1's preview-retrieval endpoint, and C11's
  "search specifically for Y" backend. It reuses the live classifier/embedder via
  the core object (no second model load — G13's lesson).
- **D5: ice-mcp is a thin additional entrypoint over the same install.**
  `[project.scripts] ice-mcp = "src.mcp.server:main"`, built on the official
  `mcp` Python SDK (FastMCP), stdio transport (harness default) + optional
  streamable-HTTP flag. No bespoke protocol code.
- **D6: boot story (user decision, refined into mechanics):** postgres IS the
  shared core state; the singleton to protect is the maintenance runtime + docker.
  `ice-mcp` startup: (1) try DB connect — reachable ⇒ *attached*; unreachable ⇒
  boot `docker compose up -d postgres` (the C7-era compose has no redis) and
  connect; (2) check C7's runtime **lease** — fresh ⇒ another process (the app)
  owns maintenance, do NOT start a second runtime; stale/absent ⇒ start the
  runtime in-process via `create_core()`; (3) **linger:** on MCP shutdown, docker
  is left up (postgres idles at ~zero cost; next session attaches instantly) and
  an `ice_core_linger` log line records it — `./stop_ice`/the packaged app remain
  the explicit "all the way down" paths. This satisfies attach-else-boot without
  inventing a daemon.
- **D7: two-tier tool surface, bloat-controlled: 6 composite + 4 multiplexed
  micro tools (10 total).** Composite (the in-loop brain): `ice_context(task)`,
  `ice_why(name)` (note + T-timeline), `ice_recent(conversation_id|project)`,
  `ice_conventions()` (procedural patterns), `ice_where(symbol)` (**E1b-gated:**
  until the code graph exists it resolves codex entities by name/alias and says
  so in its description), `ice_remember(text, slot|bookmark)` (the write-path
  composite). Micro (full user control, one action-multiplexed tool per domain):
  `ice_slots(action=list|get|set, ...)`, `ice_graph(action=view|edit|edges|
  timeline|diff, ...)`, `ice_control(action=scope_get|scope_set|review_list|
  review_approve|review_reject|registry_view|registry_edit, ...)`,
  `ice_bookmarks(action=list|add, ...)`. Every handler is a ≤5-line call into a
  D1 service. Session-start resource `ice://session-start` returns slots + last-
  session summary now; E4 enriches it into the full welcome-back block later.
- **D8: writes apply + journal (user decision).** No review-queue detour for chat
  writes; every write tool logs `mcp_tool_call` (F5/D3-telemetry candidate) and
  graph writes journal CodexEvents with source tags (G17 alignment). The review
  queue remains only for *agent* proposals (D1/D2 tiers).
- **D9: G24 rides along where touched:** service calls from async routers go
  through `asyncio.to_thread` at the adapter layer (one decided pattern), not
  per-call improvisation inside services.
- **USER-REQUIRED (rule 11):** after E7 lands, the user registers the server in
  their harness once — e.g. `claude mcp add ice -- uv run --directory
  /home/deepnar/Programs/ice ice-mcp` (Claude Code) or the Cursor MCP config
  equivalent. ~2 minutes. Done = the harness lists ICE's 10 tools and
  `ice_context("test")` returns fragments.
- **Empirical deferral (rule 2b):** pull-discipline (does the harness actually
  call `ice_context` before grepping?) — measure via `mcp_tool_call` telemetry
  over the first real coding sessions; if `ice_context` is consulted in <30% of
  sessions that grep the repo, activate the E5 contingency (thin dispatcher /
  skill rule), which remains NOT built until that evidence exists.

## 2. Structure

```
src/services/           errors.py bookmarks.py slots.py scoping.py clusters.py
                        review.py registry_svc.py graph.py retrieval_svc.py
src/mcp/server.py       FastMCP app: 10 tools + 1 resource; boot per D6; main()
src/api/routers/*.py    thin adapters over services (same URLs, same responses)
```
Extraction order (each router endpoint): move body → service fn; router keeps
pydantic parsing + error translation; behavioral parity test before/after (§5).

## 3. Files & integration points

New `src/services/*` and `src/mcp/server.py`; `pyproject.toml` (script entry +
`mcp` dependency); routers rewritten thin; `main.py` review/bookmark logic paths
unchanged (they already route through the routers); C7's `create_core()` gains a
`start_runtime: bool` parameter (lease-checked default) for D6 step 2;
`tests/test_services.py` + `tests/test_mcp_server.py`.

## 4. Edge cases & failure modes

- Docker missing/daemon down at ice-mcp boot → clear stderr message naming the
  exact command to run; MCP server exits nonzero (harness shows the error).
- Two harnesses spawn ice-mcp concurrently → both attach to postgres; lease
  ensures at most one runtime; tools are stateless per-call — safe.
- App starts while ice-mcp's runtime holds the lease → app's runtime start sees
  the fresh lease and defers (same check both directions; C7 lease is the single
  arbiter).
- `ice_context` with no classifier loaded (headless core skips model loads until
  first use) → lazy-load on first call; document the one-time latency in the tool
  description.
- Registry file concurrent writes (app UI + MCP) → `registry_svc` takes a file
  lock (`fcntl`) around load-modify-save; last-writer-wins inside the lock.
- Slot name outside the 7 → ValidationError listing valid names (C9's tier rework
  will widen this list — services read it from one constant, C9's seam).
- MCP tool called with an unknown action → error listing the action enum (guided
  self-correction for the calling model).

## 5. Validation checklist

`tests/test_services.py` (live-DB, self-cleaning): 1) each service round-trips
its domain (slot create/update/version bump; bookmark sets flags + enqueues via a
stub runtime; scope set syncs `is_private` both directions — the G16 invariant;
cluster create/assign; review approve applies slot content AND the D1/D2 arms via
stub `merge_entities`; registry edit persists under the lock); 2) domain errors
raised (no HTTPException anywhere under `src/services/`); 3) router parity: for
every endpoint, response JSON before-vs-after extraction is identical on the same
fixtures (record-and-compare harness); 4) `entity_edit` writes description,
journals `description_updated`, regenerates payload — and never touches
`context_payload` directly. `tests/test_mcp_server.py` (in-process FastMCP
client): 5) all 10 tools callable, schemas valid; 6) `ice_slots set/get`
round-trip; 7) `ice_graph edit` journals with `source="mcp_edit"`;
8) `ice_context` returns structured fragments on a seeded DB; 9) boot logic: with
a fresh lease fixture ice-mcp does NOT start a runtime, with a stale one it does;
10) session-start resource renders. Plus grep-gate: `grep -rn "fastapi\|HTTPException"
src/services src/mcp` → only the adapter files.

## 6. Look-ahead constraints

- **C11** implements chat commands as adapter #3 over these exact services — parser
  only, zero new operation logic (its roadmap entry already says so).
- **F1** builds against the services through the thin REST adapters — no bespoke
  endpoints; F2's panel = `review.py` renderings.
- **E1/E1b/E8:** `ice_where`/`ice_why` upgrade in place when the code graph and
  decisions table land — tool names and signatures are stable now, engines swap
  underneath; `ice_context` gains the decisions/constraints block (E8) without
  interface change. `decisions_*` actions are added to `ice_control` by E1, not
  pre-stubbed.
- **E4:** replaces the session-start resource's body; resource URI is stable.
- **G17:** the `source` tags written here (mcp_edit/manual_edit) are the audit
  trail's vocabulary — keep them consistent with D1's `maintenance_agent`.
- **G24:** the to-thread adapter pattern set here is the one E0's later router
  work follows (decide once).

## 7. Traps

- **Don't put logic in adapters** — the moment an MCP handler grows an `if`, the
  three surfaces start diverging; logic goes down into the service.
- **Don't spawn a second maintenance runtime** because "the MCP session should be
  self-sufficient" — the lease is the arbiter; double-running decay corrupts
  memory (C7 trap, restated because ice-mcp is where it WILL happen).
- **Don't pre-build the E5 dispatcher** — pull-discipline gets measured first
  (the deferral above); building it now is speculative harness engineering.
- **Don't expose 25 micro tools** — tool-selection quality degrades with count;
  the 4 multiplexed tools + action enums are the design, not a compression hack.
- **Don't let `entity_edit` accept `context_payload`** — it's derived; edits go
  to `description` and regeneration rebuilds the note (F3's graph view depends on
  this staying true).
- **Don't return rendered prompts from `retrieval_svc`** — structured fragments
  only; rendering belongs to each adapter (the assembler for chat, markdown for
  MCP, JSON for REST).
- **Don't hand-roll stdio framing or tool schemas** — the `mcp` SDK owns the
  protocol; ICE owns only handlers.
