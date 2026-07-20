"""In-process maintenance runtime (C7) — replaces Celery + Redis.

One asyncio tick task owns all background maintenance inside the core app
process. Jobs are the existing worker callables (plain functions since C7
stripped the ``@app.task`` wrappers), executed via ``asyncio.to_thread`` under
two lane semaphores: ``gpu`` (LLM-calling jobs, width 1 — bg work serializes
against itself, the shared-mode contention fix) and ``cpu`` (DB-only jobs,
width 2). Schedule state lives in the ``maintenance_ledger`` table (D3), which
survives restarts, feeds the overdue catch-up, and is the surface Track D's
maintenance agent later reads.

Three trigger classes (D4):
  * event      — ``enqueue(job_name, **kwargs)``: turn stored → post-flight.
  * overdue    — a ~60s tick compares the ledger against
                 ``settings.maintenance_intervals`` and runs anything overdue
                 while the app is idle.
  * work-unit  — ``notify_work_unit(kind, **ctx)``: ``session_gap`` wired now;
                 ``commit``/``task_done`` reserved for E3/E4 (handlers register
                 via ``register_work_unit_handler`` — zero runtime changes).

Idle knobs (D7, Z1-tunable): ``user_active_threshold_seconds`` (90) gates
overdue dispatch via ``is_idle()``; ``idle_burst_seconds`` (120) gates when
queued gpu-lane *event* jobs start draining after the user goes quiet.
cpu-lane events dispatch on the next tick regardless — they don't contend.

Decay catch-up is closed-form (D5, rev 2026-07-11): for ``pass_cycles`` jobs
the runtime computes ``cycles = clamp(floor((now - last_finished)/interval),
1, 96)`` from the job's own ledger row at dispatch — one UPDATE with
``rate ** cycles`` covers any gap, and cycles can never double-count runs the
overdue tick already performed.

E7 constraint (D10): this module must stay importable without FastAPI — it
imports settings, gpu_check, and (lazily, at dispatch) the worker modules;
never ``src.api.main``.
"""

import asyncio
import heapq
import importlib
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.runtime")

CYCLES_CAP = 96


@dataclass(frozen=True)
class JobSpec:
    """One maintenance job: a dotted ``module:function`` path plus dispatch
    metadata. Callables keep their existing names/signatures (D1's agent and
    FINAL's harness are built against them)."""

    path: str                 # "src.workers.decay:apply_decay"
    lane: str                 # "gpu" | "cpu"
    needs_db: bool = False    # callable takes a db session as first arg
    pass_cycles: bool = False # decay family: runtime passes cycles= at dispatch


# Job inventory (C7 §2.2). Interval-triggered jobs take their cadence from
# settings.maintenance_intervals[job_name]; event-only jobs have no entry there.
# fine_tune is deliberately unscheduled — consent-gated via the session-end
# burst (D6/H5), never cadence-run.
JOBS: dict[str, JobSpec] = {
    "post_flight":             JobSpec("src.workers.post_flight:evaluate_turn", "gpu"),
    # Standalone codex extraction (user_control's bookmark endpoint) — the
    # normal path is the direct call inside post_flight.
    "codex_extract":           JobSpec("src.workers.codex_extractor:extract_codex", "gpu"),
    "chunk_pending_documents": JobSpec("src.workers.document_chunker:run_pending_documents", "cpu", needs_db=True),
    "cluster_assignment":      JobSpec("src.workers.clustering:run_cluster_assignment", "cpu", needs_db=True),
    "cluster_merge":           JobSpec("src.workers.clustering:run_cluster_merge", "cpu", needs_db=True),
    "compaction":              JobSpec("src.workers.compaction:compact_entities", "cpu"),
    "decay_episodic":          JobSpec("src.workers.decay:apply_decay", "cpu", pass_cycles=True),
    "decay_codex":             JobSpec("src.workers.codex_decay:decay_codex_edges", "cpu", pass_cycles=True),
    "decay_procedural":        JobSpec("src.workers.procedural_decay:decay_procedural_patterns", "cpu", pass_cycles=True),
    "reflection":              JobSpec("src.workers.reflection:run_reflection", "gpu"),
    "batch_summarize":         JobSpec("src.workers.batch_summarizer:batch_summarize", "gpu"),
    # C4: the evolving whole-conversation summary — burst member + cadence
    # (idempotent per covers_through; quiet conversations are no-ops).
    "conversation_summary":    JobSpec("src.workers.conversation_summary:run_conversation_summaries", "gpu", needs_db=True),
    # D1/D2: the maintenance agent replaced the sentinel monitor (its two real
    # checks live on as agent detectors 2 and 5).
    "maintenance_agent":       JobSpec("src.workers.maintenance_agent:run_maintenance_agent", "gpu", needs_db=True),
    "fine_tune":               JobSpec("src.workers.fine_tune:fine_tune_classifier", "gpu"),
    # E3/E8 (coding core): reconcile-on-commit (event, via the "commit"
    # work-unit handler), the polling fallback, and the cue-gated decision
    # extractor (the only LLM-touching stage — gpu lane; reconcile is pure
    # git+DB).
    "project_reconcile":       JobSpec("src.coding.reconciler:reconcile_project", "cpu", needs_db=True),
    "project_poll":            JobSpec("src.coding.reconciler:poll_projects", "cpu", needs_db=True),
    "decision_extract":        JobSpec("src.workers.decision_extractor:run_decision_extraction", "gpu", needs_db=True),
    # F10/F14: conversation import replay — a self-re-enqueueing sliced job
    # (event-only, no cadence). Each dispatch replays one ~10-min slice in the
    # gpu lane and re-enqueues the next, so an hours-long import never starves
    # live chat's post_flight; resume rides the import_conversations hash
    # ledger + per-turn idempotency keys.
    "import_replay":           JobSpec("src.ingestion.importer:run_import_replay", "gpu", needs_db=True),
}

# Ledger pseudo-job recording the last session-end burst (§2.3).
BURST_STAMP = "session_end_burst"

# E7 (D6): ledger pseudo-row marking which process owns maintenance. The
# owning runtime's tick stamps it; a fresh stamp makes any other core boot
# (app or ice-mcp — same check both directions) start its runtime in STANDBY:
# event jobs for that process's own work units still run (idempotency keys +
# per-job ledger claims make cross-process duplication safe), but periodic /
# overdue dispatch, the session-end burst, and lease stamping stay exclusive
# to the owner. A standby runtime promotes itself when the lease goes stale.
RUNTIME_LEASE = "runtime_lease"
RUNTIME_LEASE_TTL = 180.0   # ≈ 3 ticks


def runtime_lease_fresh(db_factory, ttl: float = RUNTIME_LEASE_TTL) -> bool:
    """Is another process's maintenance runtime alive (lease stamped within
    *ttl*)? Errors and absence both read as stale — the caller then owns
    maintenance, which is also the right call when the ledger is unreachable
    (the runtime can't run without the DB anyway)."""
    db = db_factory()
    try:
        row = db.execute(text(
            "SELECT last_started FROM maintenance_ledger "
            "WHERE job_name = :name"), {"name": RUNTIME_LEASE}).first()
        if row is None or row.last_started is None:
            return False
        age = (datetime.now(timezone.utc) - row.last_started).total_seconds()
        return age < ttl
    except Exception as exc:
        logger.warning("runtime_lease_check_failed", error=str(exc))
        return False
    finally:
        db.close()


# ── Pure helpers (unit-testable without a runtime) ──────────────────────────

def missed_cycles(elapsed_seconds: float, interval_seconds: float,
                  cap: int = CYCLES_CAP) -> int:
    """Closed-form decay catch-up (D5): how many whole intervals elapsed,
    clamped to [1, cap]. rate**96 ≈ 0.73 for the 0.9968 unaccessed rate."""
    if interval_seconds <= 0:
        return 1
    return max(1, min(cap, int(elapsed_seconds // interval_seconds)))


def tick_delay(base: float = 60.0, jitter: float = 15.0) -> float:
    """Tick sleep: base + uniform jitter (spec §2.1: 60s + 0–15s)."""
    return base + random.uniform(0.0, jitter)


def overdue_seconds(now: datetime, interval: float,
                    last_started: Optional[datetime],
                    last_finished: Optional[datetime]) -> float:
    """Seconds a periodic job is past due; <= 0 means not due.

    Never ran → due one interval's worth (runs on the first idle tick).
    Started without finishing (crashed instance) → due after 2× interval —
    the ledger-lease grace window.
    """
    if last_started is None and last_finished is None:
        return interval
    if last_finished is None or (last_started is not None
                                 and last_started >= last_finished):
        return (now - last_started).total_seconds() - 2 * interval
    return (now - last_finished).total_seconds() - interval


def _job_key(job_name: str, kwargs: dict) -> str:
    return f"{job_name}|{sorted(kwargs.items())!r}"


class MaintenanceRuntime:
    """The in-process scheduler. One instance per core process; a ledger
    claim (optimistic lease) keeps an accidental second instance from
    double-running periodic jobs (double exponential decay destroys memory)."""

    RETRY_DELAYS = (30.0, 120.0, 480.0)   # exponential backoff, then error
    QUEUE_DEPTH_WARN = 20                  # Z1 signal (spec §4)

    def __init__(self, jobs: Optional[dict] = None,
                 intervals: Optional[dict] = None):
        self._jobs = jobs if jobs is not None else JOBS
        self._intervals = (intervals if intervals is not None
                           else dict(settings.maintenance_intervals))
        self._db_factory: Optional[Callable] = None
        self.tick_base = 60.0
        self.tick_jitter = 15.0
        self.standby = False   # E7 D6: deferred to another process's runtime

        self._gpu_lane = asyncio.Semaphore(1)
        self._cpu_lane = asyncio.Semaphore(2)
        self._queue: deque = deque()          # (job_name, kwargs, attempt)
        self._retries: list = []              # heap of (monotonic_due, seq, name, kwargs, attempt)
        self._retry_seq = 0
        self._queued_keys: set = set()
        self._running_keys: set = set()
        self._lock = threading.Lock()         # enqueue may be called from job threads
        self._tasks: set = set()
        self._tick_task: Optional[asyncio.Task] = None
        self._stopping = False

        self._generation_inflight = 0
        self._last_activity: Optional[datetime] = None
        self._started_at: Optional[datetime] = None

        self._work_unit_handlers: dict[str, Callable] = {
            "session_gap": self._handle_session_gap,
            # "commit" / "task_done": reserved for E3/E4 — coding-side
            # reconcilers register handlers; the runtime never changes.
        }

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self, db_factory, standby: bool = False) -> None:
        """Spawn the tick task. Requires a running event loop (FastAPI
        lifespan, or E7's headless asyncio context). *standby* (E7 D6):
        another process's lease is fresh — run event jobs only, never stamp
        the lease, self-promote when the owner's lease goes stale."""
        global _active_runtime
        self._db_factory = db_factory
        self._started_at = datetime.now(timezone.utc)
        self._stopping = False
        self.standby = standby
        loop = asyncio.get_running_loop()
        self._tick_task = loop.create_task(
            self._tick_loop(), name="maintenance-tick")
        if not standby:
            # Claim the lease immediately (not on the first ~60s tick) so a
            # concurrently-booting second core sees it; off-loop per G24.
            stamp = loop.create_task(asyncio.to_thread(self._lease_stamp))
            self._tasks.add(stamp)
            stamp.add_done_callback(self._tasks.discard)
        _active_runtime = self
        logger.info("maintenance_runtime_started", standby=standby,
                    jobs=len(self._jobs), intervals=len(self._intervals))

    async def stop(self, drain_timeout: float = 60.0) -> None:
        """Drain: accept no new jobs, wait for current ones, cancel the tick."""
        global _active_runtime
        self._stopping = True
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except (asyncio.CancelledError, Exception):
                pass
            self._tick_task = None
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            done, still = await asyncio.wait(pending, timeout=drain_timeout)
            if still:
                logger.warning("maintenance_stop_jobs_unfinished", count=len(still))
        if not self.standby and self._db_factory is not None:
            # Release the lease so the next core boot owns maintenance
            # immediately instead of waiting out the TTL.
            await asyncio.to_thread(self._lease_release)
        if _active_runtime is self:
            _active_runtime = None
        logger.info("maintenance_runtime_stopped")

    # ── Signals (replace redis + nvidia-smi for shared mode, D7) ────────

    def note_user_activity(self) -> None:
        self._last_activity = datetime.now(timezone.utc)

    def generation_started(self) -> None:
        self._generation_inflight += 1
        self._last_activity = datetime.now(timezone.utc)

    def generation_finished(self) -> None:
        self._generation_inflight = max(0, self._generation_inflight - 1)
        self._last_activity = datetime.now(timezone.utc)

    @property
    def generation_in_flight(self) -> bool:
        """C10: destructive ops (conversation deletion) refuse to run while a
        chat stream is live — the stream's post-flight would FK-fail into a
        just-deleted conversation."""
        return self._generation_inflight > 0

    def _idle_seconds(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        anchor = self._last_activity or self._started_at or now
        return (now - anchor).total_seconds()

    def is_idle(self) -> bool:
        """No generation in flight AND the user has been quiet past the
        activity threshold — the gate for overdue dispatch."""
        return (self._generation_inflight == 0
                and self._idle_seconds() > settings.user_active_threshold_seconds)

    def _gpu_ready(self, for_event: bool) -> bool:
        if self._generation_inflight:
            return False
        threshold = (settings.idle_burst_seconds if for_event
                     else settings.user_active_threshold_seconds)
        if self._idle_seconds() <= threshold:
            return False
        # Dedicated mode additionally consults the cached nvidia-smi gate
        # (G4); in shared mode is_gpu_busy() is always False.
        if is_gpu_busy():
            return False
        return True

    # ── Triggers ─────────────────────────────────────────────────────────

    def enqueue(self, job_name: str, **kwargs) -> None:
        """Queue an event job. Identical (job, kwargs) already queued or
        running → dropped (single-flight; idempotency keys are the
        at-least-once guard underneath)."""
        if self._stopping:
            return
        if job_name not in self._jobs:
            logger.error("maintenance_unknown_job", job=job_name)
            return
        key = _job_key(job_name, kwargs)
        with self._lock:
            if key in self._queued_keys or key in self._running_keys:
                logger.debug("maintenance_duplicate_enqueue", job=job_name)
                return
            self._queued_keys.add(key)
            self._queue.append((job_name, kwargs, 0))
            depth = len(self._queue)
        if depth > self.QUEUE_DEPTH_WARN:
            logger.warning("maintenance_queue_depth", depth=depth)

    def register_work_unit_handler(self, kind: str, handler: Callable) -> None:
        self._work_unit_handlers[kind] = handler

    def notify_work_unit(self, kind: str, **ctx) -> None:
        handler = self._work_unit_handlers.get(kind)
        if handler is None:
            logger.warning("work_unit_unhandled", kind=kind)
            return
        try:
            handler(**ctx)
        except Exception as exc:
            logger.error("work_unit_handler_failed", kind=kind, error=str(exc))

    def _handle_session_gap(self, conversation_id: Optional[str] = None,
                            gap_seconds: Optional[float] = None) -> None:
        """New sitting (§2.3, rev): freshen the conversation's clusters and run
        an immediate overdue pass — decay cycles come from the ledger at
        dispatch, so catch-up needs no gap arithmetic here."""
        logger.info("work_unit_session_gap", conversation_id=conversation_id,
                    gap_seconds=gap_seconds)
        if conversation_id:
            self.enqueue("cluster_assignment", conversation_ids=[conversation_id])
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (direct call in tests) — next tick catches up
        task = loop.create_task(self._pump(force_overdue=True))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ── Tick / dispatch ──────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(tick_delay(self.tick_base, self.tick_jitter))
            try:
                if self.standby:
                    # Owner gone? (exited/crashed — its lease went stale)
                    if not await asyncio.to_thread(
                            runtime_lease_fresh, self._db_factory):
                        self.standby = False
                        logger.info("maintenance_runtime_promoted")
                if not self.standby:
                    await asyncio.to_thread(self._lease_stamp)
                await self._pump()
            except Exception as exc:
                # The tick must never die (job errors are contained in
                # _run_job; this catches pump-level bugs and DB outages).
                logger.error("maintenance_tick_failed", error=str(exc))

    async def _pump(self, force_overdue: bool = False) -> None:
        self._drain_due_retries()
        self._dispatch_events()
        if self.standby:
            return  # periodic dispatch + burst belong to the lease owner
        if force_overdue or self.is_idle():
            # Ledger reads happen in a thread (G24: no sync DB on the loop);
            # task spawning stays on the loop.
            for name in await asyncio.to_thread(self._overdue_names):
                self._spawn(name, {}, 0, source="overdue")
            await asyncio.to_thread(self._maybe_session_end_burst)

    def _drain_due_retries(self) -> None:
        now_mono = time.monotonic()
        with self._lock:
            while self._retries and self._retries[0][0] <= now_mono:
                _, _, name, kwargs, attempt = heapq.heappop(self._retries)
                self._queue.append((name, kwargs, attempt))

    def _dispatch_events(self) -> None:
        """Dispatch queued events into their lanes. gpu-lane events wait for
        the idle burst threshold; cpu-lane events go immediately."""
        kept = deque()
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        for name, kwargs, attempt in items:
            spec = self._jobs[name]
            if spec.lane == "gpu" and not self._gpu_ready(for_event=True):
                kept.append((name, kwargs, attempt))
                continue
            self._spawn(name, kwargs, attempt, source="event")
        if kept:
            with self._lock:
                self._queue.extendleft(reversed(kept))

    def _overdue_names(self) -> list:
        """Thread-side ledger read: due periodic jobs, longest-overdue first."""
        if self._db_factory is None:
            return []
        now = datetime.now(timezone.utc)
        rows = self._ledger_rows()
        due = []
        for name in self._jobs:
            interval = self._intervals.get(name)
            if not interval:
                continue  # event-only / consent-gated jobs have no cadence
            row = rows.get(name)
            od = overdue_seconds(
                now, interval,
                row["last_started"] if row else None,
                row["last_finished"] if row else None,
            )
            if od > 0:
                due.append((od, name))
        return [name for _, name in sorted(due, reverse=True)]

    def _spawn(self, name: str, kwargs: dict, attempt: int, source: str) -> None:
        """Loop-side only (callers: _dispatch_events, _pump)."""
        key = _job_key(name, kwargs)
        with self._lock:
            if key in self._running_keys:
                return
            self._running_keys.add(key)
            self._queued_keys.discard(key)
        task = asyncio.get_running_loop().create_task(
            self._run_job(name, kwargs, attempt, source))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_job(self, name: str, kwargs: dict, attempt: int,
                       source: str) -> None:
        spec = self._jobs[name]
        key = _job_key(name, kwargs)
        lane = self._gpu_lane if spec.lane == "gpu" else self._cpu_lane
        try:
            async with lane:
                interval = self._intervals.get(name) or 0
                call_kwargs = dict(kwargs)
                if spec.pass_cycles:
                    elapsed = await asyncio.to_thread(self._ledger_elapsed, name)
                    call_kwargs["cycles"] = missed_cycles(
                        elapsed if elapsed is not None else interval, interval)
                # Periodic runs claim the ledger row (lease = 2× interval) so a
                # duplicate app instance can't double-run decay; event runs
                # stamp unconditionally (idempotency keys guard re-runs).
                lease = 2 * interval if source == "overdue" else None
                claimed = await asyncio.to_thread(self._ledger_claim, name, lease)
                if not claimed:
                    logger.info("maintenance_job_lease_held", job=name)
                    return
                logger.info("maintenance_job_started", job=name, source=source,
                            attempt=attempt, **({"cycles": call_kwargs["cycles"]}
                                                if spec.pass_cycles else {}))
                t0 = time.monotonic()
                try:
                    await asyncio.to_thread(self._call, spec, call_kwargs)
                except Exception as exc:
                    await asyncio.to_thread(
                        self._ledger_finish, name, "error"
                        if attempt >= len(self.RETRY_DELAYS) else "retrying",
                        str(exc)[:500])
                    if attempt < len(self.RETRY_DELAYS):
                        delay = self.RETRY_DELAYS[attempt]
                        logger.warning("maintenance_job_retry", job=name,
                                       attempt=attempt + 1, delay=delay,
                                       error=str(exc))
                        with self._lock:
                            self._retry_seq += 1
                            heapq.heappush(self._retries,
                                           (time.monotonic() + delay,
                                            self._retry_seq, name, kwargs,
                                            attempt + 1))
                            self._queued_keys.add(key)
                    else:
                        logger.error("maintenance_job_failed", job=name,
                                     attempts=attempt + 1, error=str(exc))
                else:
                    await asyncio.to_thread(self._ledger_finish, name, "ok", None)
                    logger.info("maintenance_job_finished", job=name,
                                seconds=round(time.monotonic() - t0, 2))
        finally:
            with self._lock:
                self._running_keys.discard(key)

    def _call(self, spec: JobSpec, kwargs: dict) -> None:
        """Resolve the dotted path lazily (keeps runtime.py import-light) and
        invoke — with a fresh session when the callable takes one."""
        module_name, func_name = spec.path.split(":")
        fn = getattr(importlib.import_module(module_name), func_name)
        if spec.needs_db:
            db = self._db_factory()
            try:
                fn(db, **kwargs)
            finally:
                db.close()
        else:
            fn(**kwargs)

    # ── Session-end burst + consent-gated fine-tune (D6, §2.3) ───────────

    def _maybe_session_end_burst(self) -> None:
        """When the sitting has been over for session_gap_minutes and this
        sitting hasn't burst yet: run the heavy pair + the fine-tune check."""
        if self._db_factory is None or self._last_activity is None:
            return
        if self._generation_inflight:
            return
        if self._idle_seconds() < settings.session_gap_minutes * 60:
            return
        rows = self._ledger_rows()
        stamp = rows.get(BURST_STAMP)
        if stamp and stamp["last_started"] and \
                stamp["last_started"] > self._last_activity:
            return  # this sitting already reconciled
        self._ledger_claim(BURST_STAMP, None)
        self._ledger_finish(BURST_STAMP, "ok", None)
        logger.info("session_end_burst")
        # D8: the maintenance agent rides the burst with the heavy pair —
        # "reconcile when the sitting ends" — and C4's conversation summary
        # joins them (the quartet): the evolving summary is a session-end
        # artifact, never per-turn.
        for job in ("reflection", "batch_summarize", "maintenance_agent",
                    "conversation_summary"):
            row = rows.get(job)
            if row and row["last_finished"] and \
                    row["last_finished"] > self._last_activity:
                continue  # already fresh for this sitting
            self.enqueue(job)
        action = self._maybe_propose_finetune()
        if action == "run":
            self.enqueue("fine_tune")

    def _maybe_propose_finetune(self) -> Optional[str]:
        """D6: enough curated labels + session end → run only when
        settings.auto_finetune (default False); otherwise leave one pending
        review-queue proposal (F2 surfaces it). Never cadence-run."""
        db = self._db_factory()
        try:
            n = db.execute(text("SELECT count(*) FROM curated_labels")).scalar() or 0
            if n < settings.finetune_min_curated:
                return None
            if settings.auto_finetune:
                logger.info("finetune_autorun", curated=n)
                return "run"
            pending = db.execute(text(
                "SELECT 1 FROM review_queue "
                "WHERE item_type = 'finetune_proposal' AND status = 'pending' "
                "LIMIT 1")).first()
            if pending is None:
                db.execute(text(
                    "INSERT INTO review_queue (id, item_type, item_content, status) "
                    "VALUES (gen_random_uuid(), 'finetune_proposal', "
                    "CAST(:content AS jsonb), 'pending')"),
                    {"content": f'{{"curated_labels": {int(n)}, '
                                f'"job": "fine_tune"}}'})
                db.commit()
                logger.info("finetune_proposed", curated=n)
            return "proposed"
        except Exception as exc:
            db.rollback()
            logger.error("finetune_check_failed", error=str(exc))
            return None
        finally:
            db.close()

    # ── Runtime lease (E7 D6) ────────────────────────────────────────────

    def _lease_stamp(self) -> None:
        db = self._db_factory()
        try:
            db.execute(text(
                "INSERT INTO maintenance_ledger (job_name, last_started, last_status, runs) "
                "VALUES (:name, :now, 'running', 0) "
                "ON CONFLICT (job_name) DO UPDATE "
                "SET last_started = :now, last_status = 'running'"),
                {"name": RUNTIME_LEASE, "now": datetime.now(timezone.utc)})
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("runtime_lease_stamp_failed", error=str(exc))
        finally:
            db.close()

    def _lease_release(self) -> None:
        db = self._db_factory()
        try:
            db.execute(text(
                "UPDATE maintenance_ledger "
                "SET last_started = NULL, last_status = 'stopped' "
                "WHERE job_name = :name"), {"name": RUNTIME_LEASE})
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("runtime_lease_release_failed", error=str(exc))
        finally:
            db.close()

    # ── Ledger (D3) ──────────────────────────────────────────────────────

    def _ledger_rows(self) -> dict:
        db = self._db_factory()
        try:
            rows = db.execute(text(
                "SELECT job_name, last_started, last_finished, last_status "
                "FROM maintenance_ledger")).fetchall()
            return {r.job_name: {"last_started": r.last_started,
                                 "last_finished": r.last_finished,
                                 "last_status": r.last_status} for r in rows}
        finally:
            db.close()

    def _ledger_elapsed(self, job_name: str) -> Optional[float]:
        """Seconds since the job last finished; None if it never has."""
        db = self._db_factory()
        try:
            row = db.execute(text(
                "SELECT last_finished FROM maintenance_ledger "
                "WHERE job_name = :name"), {"name": job_name}).first()
            if row is None or row.last_finished is None:
                return None
            return (datetime.now(timezone.utc) - row.last_finished).total_seconds()
        finally:
            db.close()

    def _ledger_claim(self, job_name: str, lease_seconds: Optional[float]) -> bool:
        """Stamp the run start. With a lease: optimistic lock — the claim
        fails while another instance's run is inside its lease window."""
        now = datetime.now(timezone.utc)
        db = self._db_factory()
        try:
            db.execute(text(
                "INSERT INTO maintenance_ledger (job_name, runs) "
                "VALUES (:name, 0) ON CONFLICT (job_name) DO NOTHING"),
                {"name": job_name})
            if lease_seconds is None:
                result = db.execute(text(
                    "UPDATE maintenance_ledger "
                    "SET last_started = :now, last_status = 'running', "
                    "    runs = runs + 1 "
                    "WHERE job_name = :name"), {"name": job_name, "now": now})
            else:
                result = db.execute(text(
                    "UPDATE maintenance_ledger "
                    "SET last_started = :now, last_status = 'running', "
                    "    runs = runs + 1 "
                    "WHERE job_name = :name "
                    "  AND (last_status IS DISTINCT FROM 'running' "
                    "       OR last_started IS NULL "
                    "       OR last_started < :cutoff)"),
                    {"name": job_name, "now": now,
                     "cutoff": now - timedelta(seconds=lease_seconds)})
            db.commit()
            return result.rowcount > 0
        except Exception as exc:
            db.rollback()
            logger.error("ledger_claim_failed", job=job_name, error=str(exc))
            # Fail open for event jobs would double-run on a broken ledger;
            # fail closed instead — the tick retries next interval anyway.
            return False
        finally:
            db.close()

    def _ledger_finish(self, job_name: str, status: str,
                       error: Optional[str]) -> None:
        db = self._db_factory()
        try:
            db.execute(text(
                "UPDATE maintenance_ledger "
                "SET last_finished = :now, last_status = :status, "
                "    last_error = :error "
                "WHERE job_name = :name"),
                {"name": job_name, "now": datetime.now(timezone.utc),
                 "status": status, "error": error})
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error("ledger_finish_failed", job=job_name, error=str(exc))
        finally:
            db.close()


# Module-level handle so worker-side dispatchers can reach the process's
# runtime without importing the API app.
_active_runtime: Optional[MaintenanceRuntime] = None


def get_runtime() -> Optional[MaintenanceRuntime]:
    return _active_runtime
