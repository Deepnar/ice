# E1/E1b/E8/E9/E10 — Coding-ICE core (Project State Engine, code graph, rationale layer)

Assumes decided specs: `C7_scheduling.md` (`notify_work_unit("commit", …)` is the
trigger; reconcilers are runtime jobs), `E0_E7_services_mcp.md` (services are the
only interface layer; `ice_where`/`ice_why`/`ice_context` upgrade in place),
`D1_D2_maintenance_agent.md` (agent worklist item types are pluggable — stale-work
items join it), `T_temporal.md` (journaling discipline; timelines exclude derived
memory). Grounded at commit `94b84ca` in: `codex_entities`/`codex_edges` schema
(A7 typed rich notes), A5's batch-set scoping, `procedural_memory`, the Track-E
decisions 1–7 in the roadmap (which this spec turns into DDL and code).

> **[rev 2026-07-17 — implementing-session re-grounding at `14bd4dc` (rule 12).**
> The spec was grounded at `94b84ca`, which predates C7, Track T, E0/E7 AND
> D1/D2 — all seams verified against HEAD, thirteen refinements recorded
> before coding:]
> 1. **Agent-worklist seam concrete (D5 step 4):** "become D1-agent worklist
>    items" = a `stale_work` detector registered in
>    `maintenance_agent.DETECTORS` (the registry shipped with D1/D2). The
>    detector re-derives staleness from the `tasks` table itself (idempotent,
>    agent-cadence); the reconciler never writes worklist items. Items are
>    Tier-2 review-queue proposals (`item_type='stale_work'`, no LLM needed —
>    detection is deterministic), skipped while one is pending or rejected
>    since the task last moved (detector-5's blocker pattern).
> 2. **Commit seam concrete:** `runtime.register_work_unit_handler("commit",
>    …)` (reserved since C7), registered from `create_core()` (the composition
>    root both boot paths share). The handler enqueues a `project_reconcile`
>    runtime job (cpu lane — immediate dispatch, no idle wait). New JOBS
>    entries: `project_reconcile` (cpu), `project_poll` (cpu, interval 600s),
>    `decision_extract` (gpu). The reconcile job never calls the LLM: cue
>    checks are deterministic; commit-message extraction rides the separate
>    gpu-lane `decision_extract` event job. Post-flight's coding-turn
>    extraction is a direct call inside the post_flight gpu job (C7 chain
>    style).
> 3. **codex_edges has NO properties column** — the "tagged resolved vs
>    heuristic in edge properties" wording can't land as written. Encoding:
>    `extraction_confidence` 1.0 (resolved) vs 0.6 (heuristic) — A3-compatible
>    (a heuristic extraction IS less trusted); the bootstrap logs the ratio
>    for the empirical deferral. No new column.
> 4. **Hook transport:** the post-commit hook curl-POSTs to one new thin
>    endpoint `POST /user-control/projects/{id}/commit` (adapter over
>    `services/projects.notify_commit`), falling back to a marker file
>    `$GIT_DIR/ice_pending_commit` when the app is down; `project_poll` checks
>    marker files AND `git rev-parse HEAD` drift, so a missed POST degrades to
>    poll lag (≤10 min — same as declining the hook). An existing non-ICE
>    post-commit hook is never overwritten (registration reports it).
> 5. **Static-graph edges need a NOT NULL `source_batch`:** the deterministic
>    per-project batch id `uuid5(NAMESPACE_URL, "ice:code-graph:<project_id>")`
>    — which is also the "code-graph allowance" concretely: `_codex_scope_sets`
>    adds it to the allowed batch set for project-scoped queries.
> 6. **Source-visibility mechanics:** `_codex_scope_sets` keeps its
>    (entity_ids, batch_ids) contract. The D3 exclusion lands as (a)
>    `self._scope_project_id` set at `retrieve()`/`_wide_net_fallback` entry
>    (the `_active_timescope` pattern), (b) an `_entity_source_filters()`
>    clause in the entity-matching/enumeration queries + a python predicate at
>    traversal expansion, (c) project-scoped queries return union sets (never
>    `(None, None)`). `_render_codex_entity` renders the full payload for
>    non-conversation entities even under scope — the "leaks other convos"
>    rationale doesn't apply to entities derived from the project itself.
> 7. **Episodic project scope (D11):** main.py populates
>    `scope["conversation_ids"]` (the project's non-incognito conversations) +
>    `scope["project_id"]`; the episodic legs' conversation filter becomes
>    list-capable (`= ANY(:conv_ids)`) — the C6 seam, param-driven, no SQL
>    fork. Single-conversation behavior byte-identical.
> 8. **Session-start trigger is preflight:** the block prepends when the
>    conversation's latest turn is older than `session_gap_minutes` (or
>    absent); `project_state.last_session_at` advances only when itself stale
>    beyond the gap, so repeated renders inside one sitting don't move it.
> 9. **Schema details pinned:** `tasks.updated_at` drives staleness;
>    `decisions.embedding` written at insert via the shared embedder;
>    `daily_checklist` ships as a plain SQL VIEW over tasks (+staleness flag)
>    in the same migration. Code entities do NOT get embeddings (NULL —
>    name/scope-resolved; the agent's cosine channel excludes them via the
>    source filter anyway) and get `aliases = []` so conversational
>    `get_or_create_entity` can never re-attach to them.
> 10. **Attach path:** `scoping.set_scope` gains an optional `project` kwarg
>    (slug/id; empty string detaches); `ice_control scope_set` passes it
>    through; incognito ('none') conversations refuse attachment.
> 11. **Merged-entity husks + derived rows:** detector 3 already filters
>    `properties->>'merged_into' IS NULL`; it additionally gains
>    `source = 'conversation'` (merging regenerable derived entities is
>    wrong). Same filter on both the name and cosine channels.
> 12. **Syntax-error handling refined:** on a parse error the file's existing
>    entities are kept (last good map, stale-but-useful) and the module unit
>    is marked `properties.parse_error` — "skipped with a parse_error unit"
>    without destroying the map mid-edit.
> 13. **USER-REQUIRED surface:** registration = `scripts/register_project.py`
>    (CLI) or `ice_control action=project_register` (MCP); hook consent is the
>    `--hook/--no-hook` flag / `install_hook` param on either.

Covers the priority items E1, E1b, E8, E9, E10 **and folds E2/E3/E4/E6** (they are
inseparable halves of the same build: routing, reconcilers, session-start, and the
storage format). E5 remains resolved (no harness; contingency gated on E7's
pull-discipline measurement).

## 1. Decisions

- **D1 (E1): `projects` is first-class; three drafted tables are dropped.**
  Kept: `projects`, `project_state`, `decisions`, `tasks`. Dropped with reasons:
  `daily_checklist` (a view over tasks+staleness, per the roadmap's own amendment);
  `architecture_clusters` (A7.4's community layer over the code graph serves this
  when density justifies it — a parallel hand-rolled table would be torn out);
  `development_patterns` (decision 3 says behavioral = procedural extraction —
  coding conventions are `procedural_memory` rows scoped by a new
  `procedural_memory.project_id NULL` column, not a fourth pattern store; C9's
  slot tiers take the same shape).
- **D2 (E1b): Python-first deterministic extractor, language seam left open.**
  Stdlib `ast` for modules/classes/functions + import edges; call edges resolved
  best-effort statically (same-module names + imported names; no type inference)
  and tagged `resolved` vs `heuristic` in edge properties. Other languages arrive
  later behind the same `CodeExtractor` interface (tree-sitter candidate) — only
  Python ships now (ICE itself is the first project).
- **D3 (E1b): one codex, namespaced, derived-memory rules.** Code entities live in
  `codex_entities` with new columns `project_id UUID NULL` and
  `source TEXT DEFAULT 'conversation'` (`'static_analysis'`, `'derived'` for E9
  units). Canonical name = `"{project_slug}:{module}.{qualname}"` (collision-proof
  across projects; display name in properties). **Derived rules:** exempt from
  `codex_decay` (`WHERE source = 'conversation'` added there); excluded from
  conversational retrieval unless the conversation's project matches (scope filter
  extension in `_codex_scope_sets`: unscoped/auto queries add
  `source = 'conversation' OR project_id = :attached_project`); bulk-rebuild safe;
  **static-analysis mutations write NO CodexEvents** (regenerable churn would bury
  the journal; T-timelines therefore naturally exclude them). Pointers only:
  properties carry `file_path`, `line_start/end`, `signature`,
  `docstring_summary` — never source text (decision 2).
- **D4 (E1): decisions table carries its own bi-temporal history** (mirrors
  codex_edges: `valid_from`, `valid_until NULL`, `superseded_by UUID NULL`) — a
  new decision that conflicts supersedes explicitly; no event journal needed
  (supersession is first-class here). `decision_type ∈ {decision, constraint,
  incident}` — constraints ("do-not-touch") and incident/fix memory are rows, not
  separate systems (E8's two added types).
- **D5 (E3, folded): reconcile-on-commit with a consent-installed hook + polling
  fallback.** Project registration offers to install a `post-commit` hook that
  POSTs the new commit hash to the running core (or writes a marker file the
  runtime picks up); if declined, an overdue runtime job polls `git rev-parse
  HEAD` per registered project (10-min idle cadence). Reconciler steps, per
  commit-range since `project_state.last_reconciled_commit`: (1) `git diff
  --name-only` → E1b incremental re-parse of changed `.py` files (transactional
  delete-and-recreate of those files' derived entities/edges); (2) E9 re-derive
  for any changed manifest (hash-gated); (3) task linking — commit appended to the
  active task's `commit_hashes` (active = status `active`, else most recent
  pending); (4) stale-work detection — tasks untouched >14d and branch-vs-goal
  drift become D1-agent worklist items (`item_type='stale_work'`), not their own
  notifier; (5) `project_state.last_reconciled_commit` advances. Branch switches
  re-sync files that differ (E1b edge case b).
- **D6 (E4, folded): session-start block + cold-start bootstrap.** For
  coding-scoped conversations, the assembler prepends a compact block from:
  `project_state` (goal/branch/last task), `git diff --stat` since
  `last_session_at`, open constraints, ≤3 pending/stale tasks + ≤3 freshest
  decisions. Same renderer serves E7's `ice://session-start` resource.
  Registration bootstrap: full E1b parse + E9 derive + (once C12's backend
  exists) docs ingestion under project scope; **git-log replay is optional and
  off by default** (heavy; F10 fast-forward semantics), a registration flag.
- **D7 (E8): decision extraction is bounded and constraint-aware.** A runtime job
  (chained after post-flight for coding-scoped turns; also run over commit
  messages at reconcile): bounded LLM extraction (temperature 0, JSON) of
  `{decision, rationale, alternatives_rejected[], files_affected[], type}` — only
  when deterministic cues fire first (decision verbs / "instead of" / "don't
  touch" / an error→fixed arc for incidents), A6-style: no cue, no LLM call.
  Dedupe/supersession: embedding similarity ≥0.85 against active decisions on
  overlapping files → conflict path (supersede or `review_queue`, reusing D1's
  tiers). **`ice_context` surfaces `constraint` rows FIRST** whenever the task
  mentions their files (the do-not-touch payoff).
- **D8 (E8): architecture-doc-as-view is a service, not a file.**
  `services/graph.py::render_architecture_doc(db, project_id)` — markdown from:
  module tree w/ per-module one-liners (docstring summaries), key decisions with
  rationale (active, by recency), constraints, conventions (procedural rows),
  recent git timeline (last N commits, read live — decision 4: git IS the
  timeline, never re-stored). Consumed by E7 (`ice_control action=arch_doc`) and
  later F's "view architecture doc".
- **D9 (E9): project facts are derived codex entities, pointer-first (E6 settled
  here).** Parsers: deps+versions (pyproject/package.json + lockfile), DB schema
  (ORM models + alembic head), run/test/build commands (scripts, Makefile, CI
  yaml), config surface (settings class fields + `.env` KEY NAMES — **never
  values**), infrastructure (docker-compose services/ports/images), declared data
  shapes (e.g. `label_schema.json`). Each becomes one `codex_entities` row:
  `entity_type='project_fact'`, `source='derived'`, structured `properties` +
  human `description`, file pointer — re-derived by D5 step 2 on hash change.
  That IS the OKF adaptation (typed knowledge units in our tables, not markdown) —
  E6 needs no separate machinery.
- **D10 (E10): docs policy.** Project-internal docs → C12's pipeline under the
  project scope **when C12's backend lands** (ordering note: C12 spec precedes
  E10's ingestion in implementation order; nothing else here blocks on it).
  External library docs: never indexed; E9 knows the dependency list; lookups are
  the harness's tools; at most a pointer + project-relevant takeaway saved as a
  derived unit.
- **D11 (E2, folded): coding-mode routing is thin.** `chat_completions`: a
  conversation with `project_id` + coding scope resolves its scope to the
  project's conversations' batch-set (A5 primitive) ∪ its code-graph allowance
  (D3 filter); B2 gets `ltm_bump_coding` (+0.7 default — project queries almost
  always want context, pointers are cheap); coding intent labels wait for B1's
  single bundled retrain (already recorded there).
- **USER-REQUIRED (rule 11):** (a) project registration — point ICE at a repo
  root (one command/UI action; done = bootstrap report lists entities/units);
  (b) approve or decline the git-hook install per project (~seconds; declining
  costs up to 10 min of reconcile lag); (c) nothing else — extraction, graphs,
  and docs are autonomous.
- **Empirical deferrals (rule 2b):** call-edge precision (resolved vs heuristic
  ratio logged at bootstrap; if heuristic >60% on ICE itself, tighten to
  imports-only edges — recall matters less than trust); E8 cue recall (log
  `decision_cue_fired`; if real decisions visibly slip through in the first weeks
  of dogfooding, widen cues before touching the LLM prompt).

## 2. Data model (DDL sketch)

```sql
projects(id UUID PK, name TEXT UNIQUE, slug TEXT UNIQUE, roots TEXT[] NOT NULL,
         settings JSONB DEFAULT '{}', created_at timestamptz)
project_state(project_id UUID PK FK, goal TEXT, current_branch TEXT,
              last_task_id UUID NULL, last_session_at timestamptz NULL,
              last_reconciled_commit TEXT NULL, updated_at timestamptz)
decisions(id UUID PK, project_id UUID FK NOT NULL, decision TEXT NOT NULL,
          rationale TEXT, alternatives_rejected JSONB DEFAULT '[]',
          files_affected TEXT[] DEFAULT '{}',
          decision_type TEXT DEFAULT 'decision',   -- decision|constraint|incident
          source_batch UUID NULL, embedding vector(384) NULL,
          valid_from timestamptz, valid_until timestamptz NULL,
          superseded_by UUID NULL, created_at timestamptz)
tasks(id UUID PK, project_id UUID FK, title TEXT, status TEXT DEFAULT 'pending',
      commit_hashes TEXT[] DEFAULT '{}', files_changed TEXT[] DEFAULT '{}',
      created_at, updated_at)
ALTER TABLE conversations ADD COLUMN project_id UUID NULL REFERENCES projects(id);
ALTER TABLE codex_entities ADD COLUMN project_id UUID NULL, ADD COLUMN source TEXT DEFAULT 'conversation';
ALTER TABLE codex_edges    ADD COLUMN source TEXT DEFAULT 'conversation';
ALTER TABLE procedural_memory ADD COLUMN project_id UUID NULL;
```
Indexes: decisions(project_id, valid_until), codex_entities(project_id, source),
tasks(project_id, status). One Alembic revision.

New modules: `src/coding/code_graph.py` (CodeExtractor: full + incremental parse),
`src/coding/reconciler.py` (D5 steps, registered as the `commit` work-unit
handler), `src/coding/project_facts.py` (E9 parsers), `src/workers/
decision_extractor.py` (D7), `src/services/projects.py` (register/bootstrap/
state/tasks/decisions CRUD — E0 pattern; MCP `ice_control` gains
`project_*`/`decisions_*` actions; `ice_where` swaps to the code graph).

## 3. Files & integration points

Migration (above) · `_codex_scope_sets` + episodic leg scope resolution (D3/D11
filters — extend, don't fork; same invariant style as G16) · `codex_decay`
(conversation-only WHERE) · post-flight chain (coding-scope turns also enqueue
`decision_extractor`) · C7 runtime (register `commit` handler; polling job) ·
prompt_assembler (session-start block for coding scope) · E7 server (tool engine
swaps; new actions) · `tests/test_coding_core.py` + a committed
`tests/fixtures/mini_repo/` (tiny python package with imports, a class hierarchy,
a Makefile, pyproject, docker-compose — the parse fixture).

## 4. Edge cases & failure modes

Repo root missing/moved → project flagged `unreachable` at reconcile, tools say so
(never silently serve stale graph as current; as-of metadata per E7 staleness
honesty). Dirty worktree at parse → parse the working tree as-is (the map must
match what the agent sees — E1b decision), `last_reconciled_commit` only advances
on commit. Merge commits / rebases → diff against `last_reconciled_commit` still
lists net-changed files; force-push history loss → full re-parse fallback when the
old commit is unknown. Syntax-error files → skipped with a `parse_error` unit
(pointer + error), never crash the reconcile. Symlinks/vendored dirs → default
ignore set (`.git`, `.venv`, `node_modules`, `models/`, `data/`) + per-project
`settings.ignore`. Monorepo: multiple `roots` supported now; nested projects
explicitly NOT supported (one registration wins; documented). Huge repos: bootstrap
caps at 20k files with a warning (single-user laptop reality). Decision extractor
on non-coding chatter inside a coding conversation → cues gate it; incidents
require the error→fixed arc. `ice_where` on a symbol in two projects → results
carry project + path; conversation's attached project ranks first.

## 5. Validation checklist — `tests/test_coding_core.py` (live-DB, LLM stubbed)

1) bootstrap of `mini_repo`: expected entity/edge counts, qualified names, pointers
(file:line) correct, zero source bodies stored; 2) incremental: modify one file →
only its entities replaced (ids elsewhere stable); 3) decay run leaves derived
entities untouched; 4) conversational (no-project) retrieval never returns
`static_analysis` entities; project-scoped retrieval returns both; 5) `ice_where`
resolves a fixture symbol; 6) decisions: insert → conflicting insert supersedes
(valid_until + superseded_by set) → `ice_context` for an affected file surfaces the
constraint first; 7) incident row from a stubbed error→fixed arc; 8) E9: each
parser yields its unit from the fixture; editing pyproject re-derives only deps;
9) commit work-unit: fixture repo commit → reconciler advances
last_reconciled_commit, links the active task; 10) session-start block renders with
state+diffstat+constraints; 11) stale task → agent worklist item appears;
12) `render_architecture_doc` produces markdown with all four layers present.

## 6. Look-ahead constraints

A7.4 communities run over the code graph when built (D1 dropped the rival table for
this). C9's project slot tier hangs off `projects` + the procedural column added
here. C12's pipeline is the docs path (D10 ordering). B1 owns coding labels. F's
projects UI = `services/projects.py` renderings. FINAL/Z1: the coverage matrix
lists `src/coding/*` as new subsystems. G17: decision rows carry `source_batch`
provenance already — audit-trail-ready.

## 7. Traps

- **Don't store code bodies** — pointers survive edits; bodies rot instantly and
  bloat every prompt (decision 2; it WILL be tempting for docstrings — store only
  the first line).
- **Don't journal derived churn** — a re-parse writing thousands of CodexEvents
  destroys T-track's signal (D3); derived memory is regenerable, not history.
- **Don't build a second graph store** — one codex, namespaced; the partition is
  columns + filters, not tables (the "do we need 2 codexes" answer stays no).
- **Don't let the reconciler run on a timer when the hook works** — commit is the
  semantic signal (decision 5); polling is the fallback, not the design.
- **Don't extract decisions with an always-on LLM pass** — cue-gated or it burns
  the bg model on every coding turn and hallucinates rationale (A6 stance).
- **Don't hand-maintain the architecture doc** — it's a render; if a section needs
  editing, the fix is in the underlying stores (that's the persistence-tax point).
- **Don't index external library docs "just this once"** (E10) — the dependency
  list + harness tools are the design; a doc scraper is scope death.
