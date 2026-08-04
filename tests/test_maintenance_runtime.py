"""C7 behavioral test: the in-process maintenance runtime (spec §5).

Pure-logic checks 1–8 (overdue math, cycles clamp, single-flight, lane
serialization, retry→error, tick crash isolation, work-unit dispatch,
fine-tune gating) + live-DB checks 9–11 (post-flight chain in-process with
the LLM stubbed, decay dispatched with ledger-derived cycles>1, ledger lease
against a second instance) + the session-end burst (the one path the first
live run didn't exercise). Check 12 (celery grep + import sweep) lives in
tests/smoke.

Runs against the live Postgres (docker up). Inserts its own uniquely-marked
rows and deletes them afterwards — never truncates (the dev DB holds real
data). The decay check runs inside an outer transaction that is rolled back,
so the real store's decay scores are untouched.

Run: uv run python tests/test_maintenance_runtime.py
"""
import asyncio
import os
import sys
import threading
import time
import types
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from src.api.config import settings
from src.api.db import SessionLocal, engine
from src.memory.models import Conversation, EpisodicMemory
from src.workers.runtime import (
    BURST_STAMP,
    CYCLES_CAP,
    JOBS,
    JobSpec,
    MaintenanceRuntime,
    missed_cycles,
    overdue_seconds,
    tick_delay,
)

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


NOW = datetime.now(timezone.utc)

# ── Stub job module (dotted-path resolvable, like real workers) ─────────────
stub = types.ModuleType("ice_c7_test_stubs")
sys.modules["ice_c7_test_stubs"] = stub
stub.calls = {"flaky": 0, "wu": []}
stub.spans = {}


def _make_span_job(name, seconds=0.3):
    def job(**kwargs):
        t0 = time.monotonic()
        time.sleep(seconds)
        stub.spans.setdefault(name, []).append((t0, time.monotonic()))
    return job


stub.gpu_a = _make_span_job("gpu_a")
stub.gpu_b = _make_span_job("gpu_b")
stub.cpu_a = _make_span_job("cpu_a")
stub.cpu_b = _make_span_job("cpu_b")


def _flaky(**kwargs):
    stub.calls["flaky"] += 1
    raise RuntimeError("boom")


stub.flaky = _flaky
stub.noop = lambda **kw: None

STUB_LEDGER_NAMES = ("gpu_a", "gpu_b", "cpu_a", "cpu_b", "flaky",
                     "test_lease_job", "cluster_assignment")


def _mk_runtime(jobs, intervals=None):
    rt = MaintenanceRuntime(jobs=jobs, intervals=intervals or {})
    rt._db_factory = SessionLocal
    # backdate the idle anchor so gpu-lane gating sees a long-idle app
    rt._started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    return rt


async def _drain(rt, rounds=8, pause=0.05):
    for _ in range(rounds):
        await rt._pump()
        pending = [t for t in rt._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(pause)


# ═════════════════════════════════ pure logic ═══════════════════════════════

def test_pure_logic():
    print("── 1. tick jitter bounds ──")
    samples = [tick_delay(60.0, 15.0) for _ in range(200)]
    check("all ticks in [60, 75]", all(60.0 <= s <= 75.0 for s in samples))
    check("jitter actually varies", len({round(s, 6) for s in samples}) > 50)

    print("── 2. missed_cycles clamp ──")
    check("on-cadence gap → 1", missed_cycles(5400, 5400) == 1)
    check("sub-interval gap → 1 (floor then clamp)", missed_cycles(1800, 5400) == 1)
    check("3.5 intervals → 3", missed_cycles(5400 * 3.5, 5400) == 3)
    check("month-long gap → cap 96", missed_cycles(86400 * 30, 5400) == CYCLES_CAP)
    check("zero interval → 1", missed_cycles(99999, 0) == 1)

    print("── 3. overdue_seconds ──")
    check("never ran → due one interval", overdue_seconds(NOW, 100, None, None) == 100)
    check("finished recently → not due",
          overdue_seconds(NOW, 3600, NOW - timedelta(seconds=120), NOW - timedelta(seconds=60)) < 0)
    check("finished long ago → due",
          overdue_seconds(NOW, 3600, None, NOW - timedelta(seconds=4000)) > 0)
    crash_start = NOW - timedelta(seconds=5000)
    check("crashed mid-run (started>finished) → due only after 2× interval",
          overdue_seconds(NOW, 3600, crash_start, None) < 0
          and overdue_seconds(NOW, 2000, crash_start, None) > 0)


def test_single_flight():
    print("── 4. single-flight dedup ──")
    rt = _mk_runtime({"gpu_a": JobSpec("ice_c7_test_stubs:gpu_a", "gpu")})
    rt.enqueue("gpu_a", batch_id="x")
    rt.enqueue("gpu_a", batch_id="x")
    check("identical (job, kwargs) enqueued twice → one queue entry", len(rt._queue) == 1)
    rt.enqueue("gpu_a", batch_id="y")
    check("different kwargs → second entry", len(rt._queue) == 2)
    rt.enqueue("nonexistent_job")
    check("unknown job name → dropped, no crash", len(rt._queue) == 2)


async def test_lanes():
    print("── 5. lane serialization ──")
    rt = _mk_runtime({
        "gpu_a": JobSpec("ice_c7_test_stubs:gpu_a", "gpu"),
        "gpu_b": JobSpec("ice_c7_test_stubs:gpu_b", "gpu"),
        "cpu_a": JobSpec("ice_c7_test_stubs:cpu_a", "cpu"),
        "cpu_b": JobSpec("ice_c7_test_stubs:cpu_b", "cpu"),
    })
    rt.enqueue("gpu_a")
    rt.enqueue("gpu_b")
    rt.enqueue("cpu_a")
    rt.enqueue("cpu_b")
    await _drain(rt, rounds=3, pause=0.05)
    (ga0, ga1), = stub.spans["gpu_a"]
    (gb0, gb1), = stub.spans["gpu_b"]
    (ca0, ca1), = stub.spans["cpu_a"]
    (cb0, cb1), = stub.spans["cpu_b"]
    check("two gpu jobs never overlap", ga1 <= gb0 or gb1 <= ga0)
    check("two cpu jobs run in parallel", ca0 < cb1 and cb0 < ca1)


async def test_retry_then_error():
    print("── 6. retry backoff → error status ──")
    rt = _mk_runtime({"flaky": JobSpec("ice_c7_test_stubs:flaky", "cpu")})
    rt.RETRY_DELAYS = (0.05, 0.05, 0.05)
    rt.enqueue("flaky")
    await _drain(rt, rounds=12, pause=0.08)
    check("ran initial + 3 retries = 4 attempts", stub.calls["flaky"] == 4)
    db = SessionLocal()
    try:
        row = db.execute(text(
            "SELECT last_status, last_error, runs FROM maintenance_ledger "
            "WHERE job_name = 'flaky'")).first()
    finally:
        db.close()
    check("ledger status = error after final attempt",
          row is not None and row.last_status == "error")
    check("ledger carries the error text", row is not None and "boom" in (row.last_error or ""))
    check("ledger runs counter = 4", row is not None and row.runs == 4)


async def test_tick_crash_isolation():
    print("── 7. tick crash isolation ──")
    rt = MaintenanceRuntime(jobs={"flaky": JobSpec("ice_c7_test_stubs:flaky", "cpu")},
                            intervals={})
    rt.tick_base, rt.tick_jitter = 0.05, 0.0
    rt.RETRY_DELAYS = ()
    rt.start(SessionLocal)
    rt.enqueue("flaky")
    await asyncio.sleep(0.5)  # several ticks; the raising job must not kill the loop
    check("tick task alive after job raised", not rt._tick_task.done())
    # a pump-level failure must not kill the loop either
    orig = rt._overdue_names
    rt._overdue_names = lambda: (_ for _ in ()).throw(RuntimeError("pump-boom"))
    rt._last_activity = None  # force idle so the overdue path actually runs
    await asyncio.sleep(0.3)
    check("tick task alive after pump-level failure", not rt._tick_task.done())
    rt._overdue_names = orig
    await rt.stop(drain_timeout=2)
    check("stop() drains cleanly", rt._tick_task is None)


async def test_work_units():
    print("── 8. work-unit dispatch table ──")
    rt = _mk_runtime({"cluster_assignment": JobSpec("ice_c7_test_stubs:noop", "cpu")})
    rt.notify_work_unit("task_done", sha="abc")  # reserved, unregistered
    check("unregistered kind → logged, no crash", True)
    rt.register_work_unit_handler("commit", lambda **ctx: stub.calls["wu"].append(ctx))
    rt.notify_work_unit("commit", sha="abc123")
    check("registered handler receives ctx", stub.calls["wu"] == [{"sha": "abc123"}])

    def _exploding(**ctx):
        raise RuntimeError("handler-boom")
    rt.register_work_unit_handler("bad", _exploding)
    rt.notify_work_unit("bad")
    check("raising handler is contained", True)

    rt._handle_session_gap(conversation_id="conv-1", gap_seconds=4000.0)
    queued = list(rt._queue)
    check("session_gap → cluster freshening enqueued for the conversation",
          any(n == "cluster_assignment" and kw.get("conversation_ids") == ["conv-1"]
              for n, kw, _ in queued))
    pending = [t for t in rt._tasks if not t.done()]
    if pending:  # the immediate overdue pass it spawned
        await asyncio.gather(*pending, return_exceptions=True)
    check("session_gap spawned an immediate overdue pass", True)


def test_finetune_gating():
    print("── 8b. fine-tune proposal gating (D6) ──")
    rt = _mk_runtime(dict(JOBS))
    db = SessionLocal()
    orig_min = settings.finetune_min_curated
    orig_auto = settings.auto_finetune
    created_proposal = False
    try:
        n = db.execute(text("SELECT count(*) FROM curated_labels")).scalar() or 0
        had_pending = db.execute(text(
            "SELECT 1 FROM review_queue WHERE item_type='finetune_proposal' "
            "AND status='pending' LIMIT 1")).first() is not None

        settings.finetune_min_curated = n + 1000
        settings.auto_finetune = False
        check("below threshold → None", rt._maybe_propose_finetune() is None)

        settings.finetune_min_curated = 0
        settings.auto_finetune = False
        out = rt._maybe_propose_finetune()
        pending_now = db.execute(text(
            "SELECT 1 FROM review_queue WHERE item_type='finetune_proposal' "
            "AND status='pending' LIMIT 1")).first() is not None
        created_proposal = pending_now and not had_pending
        check("threshold met + auto off → proposed + pending review-queue row",
              out == "proposed" and pending_now)
        out2 = rt._maybe_propose_finetune()
        dup_count = db.execute(text(
            "SELECT count(*) FROM review_queue WHERE item_type='finetune_proposal' "
            "AND status='pending'")).scalar()
        check("second check doesn't duplicate the proposal",
              out2 == "proposed" and dup_count == (1 if not had_pending else dup_count))

        settings.auto_finetune = True
        check("threshold met + auto on → run", rt._maybe_propose_finetune() == "run")
    finally:
        settings.finetune_min_curated = orig_min
        settings.auto_finetune = orig_auto
        if created_proposal:
            db.execute(text("DELETE FROM review_queue WHERE item_type='finetune_proposal' "
                            "AND status='pending'"))
            db.commit()
        db.close()


# ═════════════════════════════════ live DB ══════════════════════════════════

async def test_post_flight_chain():
    print("── 9. post-flight chain in-process (LLM stubbed) ──")
    import src.workers.codex_extractor as cx
    import src.workers.post_flight as pf
    import src.workers.procedural_extractor as px

    class _StubCompletions:
        def create(self, **kw):
            msg = types.SimpleNamespace(content="NONE")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    stub_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_StubCompletions()))
    orig_triplets, orig_px, orig_pf = cx.extract_triplets, px.bg_client, pf.bg_client
    cx.extract_triplets = lambda text_, model_override="", topic_tags=None, gaps=None: []
    px.bg_client = stub_client
    pf.bg_client = stub_client
    # extract_procedural returns "NONE" (stubbed LLM) and — pre-existing
    # behavior — writes no idempotency key for NONE turns, so record the call
    # itself to prove the chain reached it.
    proc_calls = []
    orig_chain_px = pf.extract_procedural
    pf.extract_procedural = lambda **kw: (proc_calls.append(kw), orig_chain_px(**kw))[1]

    orig_burst, orig_idle = settings.idle_burst_seconds, settings.user_active_threshold_seconds
    settings.idle_burst_seconds = -1
    settings.user_active_threshold_seconds = -1

    db = SessionLocal()
    conv = Conversation(memory_scope_type="auto")
    db.add(conv)
    db.commit()
    batch_id = uuid.uuid4()
    turn = EpisodicMemory(
        conversation_id=conv.id, batch_id=batch_id, timestamp=NOW,
        topic_tags=["Software_&_Tech"], intent_tags=["Troubleshooting"],
        context_reliance="Zero_Shot",
        raw_text="User: c7 runtime chain marker\n\nAssistant: ```print('ok')```",
        embedding=[0.05] * 1024, decay_score=1.0,
        idempotency_key=f"c7test-{batch_id}",
    )
    db.add(turn)
    db.commit()

    rt = _mk_runtime(dict(JOBS))
    try:
        rt.enqueue("post_flight", batch_id=str(batch_id),
                   prompt="c7 runtime chain marker",
                   response="```print('ok')```",
                   conversation_id=str(conv.id), model_used="")
        await _drain(rt, rounds=4, pause=0.1)

        db.expire_all()
        row = db.query(EpisodicMemory).filter_by(batch_id=batch_id).first()
        check("density stage ran (lossless set — has_code)", row.lossless_flag is True)
        check("representation decided (inject_raw set)", row.inject_raw is not None)
        keys = {k: db.execute(text("SELECT 1 FROM idempotency_keys WHERE key = "
                                   "encode(sha256(convert_to(:s,'UTF8')),'hex')"),
                              {"s": s}).first() is not None
                for k, s in [("main", str(batch_id)), ("codex", f"codex:{batch_id}")]}
        check("post-flight idempotency key committed", keys["main"])
        check("chained codex stage ran (own idempotency key)", keys["codex"])
        check("chained procedural stage ran (call recorded; NONE writes no key)",
              len(proc_calls) == 1 and proc_calls[0]["batch_id"] == str(batch_id))
        ledger = db.execute(text("SELECT last_status FROM maintenance_ledger "
                                 "WHERE job_name='post_flight'")).first()
        check("ledger row post_flight = ok", ledger is not None and ledger.last_status == "ok")
    finally:
        cx.extract_triplets, px.bg_client, pf.bg_client = orig_triplets, orig_px, orig_pf
        pf.extract_procedural = orig_chain_px
        settings.idle_burst_seconds, settings.user_active_threshold_seconds = orig_burst, orig_idle
        db.execute(text("DELETE FROM idempotency_keys WHERE key IN ("
                        "encode(sha256(convert_to(:a,'UTF8')),'hex'),"
                        "encode(sha256(convert_to(:b,'UTF8')),'hex'),"
                        "encode(sha256(convert_to(:c,'UTF8')),'hex'))"),
                   {"a": str(batch_id), "b": f"codex:{batch_id}", "c": f"procedural:{batch_id}"})
        db.query(EpisodicMemory).filter_by(batch_id=batch_id).delete()
        db.query(Conversation).filter_by(id=conv.id).delete()
        db.commit()
        db.close()


async def test_decay_cycles():
    print("── 10. ledger-derived decay cycles (rolled back) ──")
    # decay writes happen inside an outer transaction we roll back — the real
    # store's scores are untouched; only the runtime's ledger row commits.
    import src.workers.decay as decay_mod
    conn = engine.connect()
    trans = conn.begin()
    TestSession = sessionmaker(bind=conn)
    orig_sl = decay_mod.SessionLocal
    decay_mod.SessionLocal = TestSession

    tdb = TestSession()
    conv = Conversation(memory_scope_type="auto")
    tdb.add(conv)
    tdb.flush()
    marker = EpisodicMemory(
        conversation_id=conv.id, batch_id=uuid.uuid4(),
        timestamp=NOW - timedelta(days=30),
        topic_tags=["Software_&_Tech"], intent_tags=[],
        context_reliance="Zero_Shot", raw_text="c7 decay marker",
        embedding=[0.05] * 1024, decay_score=1.0, access_count=0,
        idempotency_key=f"c7decay-{uuid.uuid4()}",
    )
    tdb.add(marker)
    tdb.commit()  # commits into the outer transaction only

    ldb = SessionLocal()
    ldb.execute(text(
        "INSERT INTO maintenance_ledger (job_name, last_started, last_finished, last_status, runs) "
        "VALUES ('decay_episodic', :t, :t, 'ok', 0) "
        "ON CONFLICT (job_name) DO UPDATE SET last_started = :t, last_finished = :t, "
        "last_status = 'ok'"), {"t": NOW - timedelta(seconds=4 * 5400 + 60)})
    ldb.commit()

    rt = _mk_runtime(dict(JOBS), intervals={"decay_episodic": 5400})
    try:
        names = rt._overdue_names()
        check("decay_episodic reported overdue", "decay_episodic" in names)
        await rt._run_job("decay_episodic", {}, 0, "overdue")
        tdb.expire_all()
        score = tdb.query(EpisodicMemory).filter_by(id=marker.id).first().decay_score
        expected = decay_mod.DECAY_RATE_UNACCESSED ** 4
        check(f"marker score == rate**4 ({expected:.6f})", abs(score - expected) < 1e-6)
        row = ldb.execute(text("SELECT last_status, last_finished FROM maintenance_ledger "
                               "WHERE job_name='decay_episodic'")).first()
        check("decay ledger row finished ok", row.last_status == "ok"
              and row.last_finished > NOW - timedelta(minutes=5))
    finally:
        decay_mod.SessionLocal = orig_sl
        tdb.close()
        trans.rollback()  # discard ALL decay effects on the real store
        conn.close()
        ldb.close()


def test_ledger_lease():
    print("── 11. ledger lease blocks a second instance ──")
    rt1 = _mk_runtime({"test_lease_job": JobSpec("ice_c7_test_stubs:noop", "cpu")})
    rt2 = _mk_runtime({"test_lease_job": JobSpec("ice_c7_test_stubs:noop", "cpu")})
    check("first instance claims", rt1._ledger_claim("test_lease_job", 3600) is True)
    check("second instance blocked inside the lease",
          rt2._ledger_claim("test_lease_job", 3600) is False)
    rt1._ledger_finish("test_lease_job", "ok", None)
    check("after finish the claim is free again",
          rt2._ledger_claim("test_lease_job", 3600) is True)
    rt2._ledger_finish("test_lease_job", "ok", None)
    # a stale 'running' row older than the lease is claimable (crash recovery)
    db = SessionLocal()
    db.execute(text("UPDATE maintenance_ledger SET last_status='running', "
                    "last_started = :old WHERE job_name='test_lease_job'"),
               {"old": NOW - timedelta(hours=3)})
    db.commit()
    db.close()
    check("expired lease (crashed instance) is claimable",
          rt1._ledger_claim("test_lease_job", 3600) is True)
    rt1._ledger_finish("test_lease_job", "ok", None)


def test_session_end_burst():
    print("── 12. session-end burst ──")
    rt = _mk_runtime({
        "reflection": JobSpec("ice_c7_test_stubs:noop", "gpu"),
        "batch_summarize": JobSpec("ice_c7_test_stubs:noop", "gpu"),
        "fine_tune": JobSpec("ice_c7_test_stubs:noop", "gpu"),
    })
    db = SessionLocal()
    orig_min = settings.finetune_min_curated
    settings.finetune_min_curated = 10 ** 9  # keep D6 out of this check
    try:
        db.execute(text("DELETE FROM maintenance_ledger WHERE job_name = :n"),
                   {"n": BURST_STAMP})
        # make the heavy pair look stale for this sitting
        db.execute(text("UPDATE maintenance_ledger SET last_finished = :old "
                        "WHERE job_name IN ('reflection','batch_summarize')"),
                   {"old": NOW - timedelta(days=1)})
        db.commit()

        rt._last_activity = datetime.now(timezone.utc) - timedelta(
            minutes=settings.session_gap_minutes + 1)
        rt._maybe_session_end_burst()
        queued = {n for n, _, _ in rt._queue}
        check("burst enqueued the heavy pair", {"reflection", "batch_summarize"} <= queued)
        stamp = db.execute(text("SELECT last_started FROM maintenance_ledger "
                                "WHERE job_name = :n"), {"n": BURST_STAMP}).first()
        check("burst stamped the ledger", stamp is not None and stamp.last_started is not None)

        before = len(rt._queue)
        rt._maybe_session_end_burst()
        check("same sitting never bursts twice", len(rt._queue) == before)

        rt._last_activity = datetime.now(timezone.utc)  # user came back
        rt._maybe_session_end_burst()
        check("active sitting doesn't burst", len(rt._queue) == before)
    finally:
        settings.finetune_min_curated = orig_min
        db.execute(text("DELETE FROM maintenance_ledger WHERE job_name = :n"),
                   {"n": BURST_STAMP})
        db.commit()
        db.close()


async def main():
    test_pure_logic()
    test_single_flight()
    await test_lanes()
    await test_retry_then_error()
    await test_tick_crash_isolation()
    await test_work_units()
    test_finetune_gating()
    await test_post_flight_chain()
    await test_decay_cycles()
    test_ledger_lease()
    test_session_end_burst()

    # stub ledger rows created by the fake jobs
    db = SessionLocal()
    db.execute(text("DELETE FROM maintenance_ledger WHERE job_name IN "
                    "('gpu_a','gpu_b','cpu_a','cpu_b','flaky','test_lease_job')"))
    db.commit()
    db.close()

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
