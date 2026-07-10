# C7 — Decay/worker scheduling rework + Celery-survival decision + shared-first

Assumes decided specs: `T_temporal.md` (decay.py already reworked there: freeze fix,
un-archive clause, cold columns — this spec builds on that decay.py, not today's),
`FINAL_experiments.md` (the experiment harness consumes worker *callables*, never
Celery tasks). Grounded at commit `b91232e`: `celery_app.py` (13 includes, 11 beat
entries), `gpu_check.py`, `memory/session.py`, `main.py::store_turn_async`
(dispatch + buffer fallback + redis publish), `bg_client_factory.py`, `./ice`.

## 1. Decisions

- **D1: Celery + Redis DO NOT survive. Replaced by an in-process maintenance
  runtime inside the core app process.** Rationale: single-user local product; a
  broker + beat + unused result backend (G18) + two extra OS processes exist to
  solve distribution problems ICE doesn't have, and they've been the biggest
  failure-class source (G21 boot crash from a bogus include; the never-replayed
  `post_flight_buffer.jsonl` black hole; silent always-False `is_user_active`).
  Both end-state destinations (packaged app, E7 headless MCP core) want one process
  that owns its own maintenance. Redis's three remaining uses are trivially
  in-process once workers are: broker (gone), `ice:last_chat_completed` flag (a
  variable), `chat:completed` pub/sub (zero subscribers — deleted, settles G20.6).
  Postgres is the only external service left; docker-compose drops redis.
- **D2: hand-rolled asyncio runtime, not APScheduler.** ~150 lines buys interval
  jobs with jitter, event jobs, single-flight locks, and GPU-lane serialization —
  exactly what's needed, no new dependency, no foreign thread model. Jobs are the
  existing sync callables, executed via `asyncio.to_thread` under two semaphores:
  `gpu_lane(1)` (LLM-calling jobs — also THE shared-mode contention fix: bg work
  serializes against itself) and `cpu_lane(2)` (DB-only jobs).
- **D3: schedule state lives in a `maintenance_ledger` table** (job_name PK,
  last_started, last_finished, last_status, last_error, runs). Survives restarts,
  feeds the catch-up runner ("what's overdue"), and is the surface D1's agent later
  reads. Beat's opaque schedule state dies.
- **D4: three trigger classes replace beat.** (a) **Event**: turn stored →
  post-flight chain; document stored → chunker. (b) **Overdue check**: a 60s tick
  compares ledger vs `settings.maintenance_intervals` — runs anything overdue *when
  idle* (this is what keeps an always-open app maintained). (c) **Work-unit**:
  `notify_work_unit(kind, **ctx)` with `kind="session_gap"` wired now (the
  `session_started` signal from `resolve_session_id`, main.py:131 — the seam exists)
  and `kind="commit"|"task_done"` reserved for E3/E4 (Track E decision 5: one
  concept, two signals). Session-gap catch-up + session-end burst are both this.
- **D5: decay catch-up is closed-form, not a loop.** Decay callables gain
  `cycles: int = 1`; missed cycles = `clamp(floor(gap/interval), 1, 96)`; the decay
  UPDATE applies `rate ** cycles` (the multipliers are exponential — one UPDATE
  covers any gap). Same param on codex/procedural decay.
- **D6: fine-tune is consent-gated, never unattended (settles H5).** No cron. When
  `curated_labels ≥ finetune_min_curated` AND a session ends AND
  `settings.auto_finetune` (default **False**) → run in the gpu lane; otherwise
  write a review-queue suggestion (F2 surfaces it). The weekly crontab dies.
- **D7: shared-first lands here (the C7/G2 bundle).**
  `background_model_mode="shared"` becomes the default; `background_model_name:
  Optional[str] = None` where None ⇒ **the chat model** (resolve via
  `get_fallback_model()` / the routed model) — the hardcoded `qwen3:4b-instruct-bg`
  and the dead commented SGLang block in `bg_client_factory.py` are deleted (G2).
  Idle gating in shared mode = in-process truth: `generation_inflight` flag (set
  around the Ollama stream) + `last_activity` recency (`user_active_threshold_
  seconds = 90`; today's 10s redis check is uselessly tight). `nvidia-smi` polling
  survives ONLY for dedicated mode: threshold 20 → **70**, result cached 10s (G4).
  Dedicated stays a power-user config behind the same `bg_client_factory` seam;
  `./ice` stops launching vLLM by default.
- **D8: the jsonl buffer fallback is deleted with the broker** (settles G20.7's
  half). In-process enqueue is a function call — its failure mode is the app being
  down, in which case the turn wasn't stored either. Idempotency keys (DB-level)
  remain the at-least-once guard.
- **D9: G8 folds in without Redis:** sticky-model state moves from the in-memory
  `SESSION_STATE` dict to two columns on `conversations` (`sticky_model TEXT NULL`,
  `consecutive_shifts INT DEFAULT 0`), read/written where the dict is today.
- **D10: core/app split for E7 (look-ahead, not full E7 work):** the runtime lives
  in `src/workers/runtime.py` with **zero FastAPI imports**, started from a
  `create_core()` factory that the FastAPI lifespan calls — the same factory E7's
  headless `ice-mcp` boot will call without HTTP. Do not wire it inside route code.
- **Empirical deferral (rule 2b):** `idle_burst_seconds` (default 120) and the
  overdue-tick idle requirement may feel laggy or too eager in real use — measure at
  Z1 (ledger timestamps vs chat log); adjust the two settings, no design change.

## 2. Algorithm & data model

### 2.1 Runtime (`src/workers/runtime.py`)

```python
class MaintenanceRuntime:
    def start(self, db_factory): ...          # spawns the tick task; called by create_core()
    async def stop(self): ...                 # drain: no new jobs, wait current, cancel tick
    # signals (replace redis + nvidia for shared mode)
    def note_user_activity(self): ...         # main.py: on request receipt
    def generation_started(self)/generation_finished(self): ...
    # triggers
    def enqueue(self, job_name: str, **kwargs): ...      # event jobs, FIFO queue
    def notify_work_unit(self, kind: str, **ctx): ...    # "session_gap" | "commit" | "task_done"
    # state
    def is_idle(self) -> bool:   # no generation in flight AND now-last_activity > user_active_threshold
```

Internals: one asyncio tick task (60s + 0–15s jitter). Each tick: (1) drain the
event queue; (2) if `is_idle()`, run overdue jobs (ledger + intervals), longest-
overdue first. Every job: `asyncio.to_thread(callable, **kwargs)` under its lane
semaphore + a per-job single-flight lock; ledger row updated started/finished/
status; failure → exponential backoff retry (3 attempts: 30s/120s/480s), then
status=error (tick retries next interval anyway). Job errors never propagate.

### 2.2 Job inventory (every current worker, its new trigger, its lane)

| callable (kept, `@app.task` wrapper deleted) | trigger | lane |
|---|---|---|
| `post_flight.evaluate_turn` → chains `codex_extractor.extract_codex` → `procedural_extractor` **as direct calls in one job** | event (turn stored) | gpu |
| `document_chunker.chunk_turn` / `run_pending_documents` | event + overdue 2h | cpu |
| `clustering.run_cluster_assignment` | overdue 30m + session-gap | cpu |
| `clustering.run_cluster_merge` | overdue 3h | cpu |
| `compaction.compact_entities` | **overdue 24h (newly scheduled — settles G10; lossless per Track-T constraint)** | cpu |
| `decay.apply_decay(cycles)` / `codex_decay` / `procedural_decay` | overdue 1.5h + session-gap catch-up | cpu |
| `reflection.run_reflection` (incl. A7.3 enrichment backlog) | session-end burst + overdue 2h | gpu |
| `batch_summarizer.batch_summarize` | session-end burst + overdue 2h | gpu |
| `sentinel_monitor.monitor_sentinels` | overdue 30m (ported as-is; deletion remains D2-with-D1) | cpu |
| `fine_tune.fine_tune_classifier` | consent-gated proposal (D6) | gpu |

`maintenance_intervals` becomes a settings dict carrying today's beat cadences as
defaults (G9-aligned).

### 2.3 Work-unit handling

`notify_work_unit("session_gap", conversation_id=..., gap_seconds=...)` — called
from `store_turn_async` where `session_started` is logged today (main.py:131; pass
`gap_seconds = now - last.timestamp` from `resolve_session_id`, which returns it —
extend its return to `(session_id, session_started, gap_seconds)`). Handler:
1. decay catch-up: `cycles = clamp(floor(gap/5400s), 1, 96)` → the three decay jobs
   with `cycles`;
2. per-conversation freshening: `run_cluster_assignment(db, [conversation_id])`;
3. anything overdue per ledger (the gap usually means the app was closed).

**Session-end burst:** when a tick finds `now - last_activity > session_gap_minutes`
AND the last burst ledger stamp predates last_activity → run the heavy pair
(reflection, batch_summarizer) + the fine-tune proposal check (D6). This is
"reconcile when the sitting ends" — the conversational twin of commit-triggered
reconciliation.

### 2.4 Dispatch replacement in `store_turn_async`

The `evaluate_turn.delay(...)` try/except + jsonl buffer (main.py:157–176) becomes
`runtime.enqueue("post_flight", batch_id=..., prompt=..., response=...,
conversation_id=..., model_used=...)`. The redis publish + last-chat SET block
(main.py:178–190) is deleted; `runtime.note_user_activity()` is called at request
receipt and `generation_started/finished` around the stream in `chat_completions`.

### 2.5 Migration & deletions (one commit, atomic)

Delete: `celery_app.py`; every `@app.task` decorator + `self.retry` scaffolding
(functions become plain callables — post_flight/codex_extractor keep their
signatures); `data/post_flight_buffer.jsonl` writing; redis publish; `is_user_active`
redis read (runtime flag instead); `settings.redis_url`; redis service from
`docker/docker-compose.yml`; celery + redis deps from `pyproject.toml`; the celery
line from `./ice` and the vLLM-bg default block (dedicated keeps a documented
manual path); celery pkill from `./stop_ice`. Alembic migration:
`maintenance_ledger` + the two `conversations` columns (D9).
Verify with: `grep -rn "celery\|redis\|\.delay(" src/ experiments/` → experiments
referencing `.delay` are updated to call the callables (FINAL's harness already
assumes callables).

## 3. Files & integration points

`src/workers/runtime.py` (new) · `src/api/main.py` (lifespan → `create_core()`;
dispatch swap; activity/generation signals; SESSION_STATE → conversations columns)
· `src/api/core.py` (new, ~30 lines: `create_core()` building db factory + runtime —
the E7 seam) · every `src/workers/*.py` (decorator strip; decay `cycles` param) ·
`src/workers/gpu_check.py` (dedicated-only nvidia path, threshold 70, 10s cache;
shared path reads runtime) · `bg_client_factory.py` (G2 cleanup, None ⇒ chat model)
· `src/api/config.py` (settings above) · `memory/session.py` (return gap_seconds) ·
`docker/docker-compose.yml`, `./ice`, `./stop_ice`, `setup.sh` (thin, per the
scaffolding rule) · alembic migration · `tests/test_maintenance_runtime.py`.

## 4. Edge cases & failure modes

- **App killed mid-job:** ledger shows started-without-finished → tick treats as
  overdue after `2× interval`; idempotency keys make re-runs safe.
- **Two app instances** (user double-starts): single-flight is per-process — add a
  ledger claim (`UPDATE ... WHERE last_started < now - lease` optimistic lock) so
  duplicate instances don't double-run decay.
- **`cycles` extremes:** cap 96; `rate**96` ≈ 0.73 for the 0.9968 unaccessed rate —
  safe; creative floor and thresholds unchanged (post-T decay semantics).
- **Event queue growth while busy:** unbounded deque + gpu-lane serialization is
  acceptable single-user; log queue depth > 20 as a warning (Z1 signal).
- **Job that itself calls the chat model while user starts typing:** in-flight LLM
  call is not preempted (Ollama serializes anyway); the *next* job yields because
  `is_idle()` is false. Accepted cost, noted in the shared-first decision.
- **Headless core (E7):** `create_core()` without the FastAPI app must start/stop
  the runtime cleanly — no import of `src.api.main` anywhere in `runtime.py`.
- **Dedicated mode:** everything identical except `gpu_lane` also consults the
  cached nvidia-smi gate before dispatch.

## 5. Validation checklist — `tests/test_maintenance_runtime.py`

Pure-logic (no GPU): 1) overdue math incl. jitter bounds; 2) cycles clamp math;
3) single-flight (two enqueues of one job → one run); 4) lane serialization (two
gpu jobs never overlap; cpu jobs pair); 5) retry/backoff then error status;
6) crash isolation (raising job doesn't kill the tick); 7) `notify_work_unit`
dispatch table; 8) fine-tune proposal gating (below threshold / auto off / auto on).
Live-DB: 9) turn stored → post-flight chain runs in-process (LLM stubbed) and
ledger rows appear; 10) session-gap event → decay ran with `cycles>1` and a marker
row's score == `rate**cycles`; 11) ledger lease blocks a second runtime instance;
12) importability sweep: `grep -rn "celery" src/` empty; all worker modules import
(the G22 check, now trivially green). Plus: `./ice` boots the stack without redis;
one real chat round-trip stores + extracts.

## 6. Look-ahead constraints

- **D1 (agent):** the runtime is the agent's substrate — `enqueue`/ledger/callables
  are its tools; keep every job a named callable with (db, **kwargs) signature.
- **E3/E4:** `notify_work_unit("commit", ...)` is reserved and routed through the
  same handler table — coding-side reconcilers register handlers, zero runtime
  changes.
- **E7:** `create_core()` is the headless boot's entrypoint; keep it HTTP-free.
- **G18/G3:** die with this spec (result backend gone; hot mode-switch parked).
- **FINAL:** replay/experiment scripts call callables only; the pg_dump snapshot
  wrapper (FINAL §D1) is unaffected by scheduling.
- **F5 telemetry:** ledger + job logs are the "background activity indicator" data
  source — emit structlog events (`maintenance_job_started/finished`) now.

## 7. Traps

- **Never run both schedulers "briefly".** Celery deletion and runtime activation
  are one commit; a transition period double-runs decay (double exponential decay =
  silent memory destruction).
- **Don't run job bodies on the event loop** — every callable goes through
  `to_thread`; one synchronous `db.query` on the loop stalls streaming (G24's
  lesson).
- **Don't keep the buffer file "just in case"** — an unreplayed safety net is a
  data black hole (G20.7); delete write AND file.
- **Don't auto-run fine-tune because the ledger says it's overdue** — it's
  consent-gated, not cadence-gated (D6/H5).
- **Don't move `is_user_active` semantics into nvidia-smi** — shared mode's truth
  is in-process (in-flight flag + activity), hardware polling is the dedicated-only
  path (G4).
- **Don't rename/reshape the `run_*` callables** — experiments, D1, and the FINAL
  harness are built against them.
- **Don't put the runtime in the FastAPI module** — the E7 headless boot is the
  whole reason for `create_core()`; importing main.py from runtime.py recreates the
  coupling this spec exists to remove.
