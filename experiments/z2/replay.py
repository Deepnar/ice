"""Z2 S7: full-system semi-LSREP replay on the declared clock (spec Z2_EVALUATION D8).

Drives `scripts/z1/seed_v3.py` — every history turn runs the whole answering
pipeline to the final fragments with exposure and strengthening written, stops
before the answer model and stores the recorded reply — with three additions,
all from outside (no src change):

- the declared clock (`clock.py`, approved 2026-10-18) replaces B's/C's
  synthetic timestamps;
- `events.jsonl`: every retrieval strengthening (turn, episodic ids, amount),
  every fact-usage write (turn, fact ids), and every decay job's per-row change
  (episodic decay_score, codex usage, procedural confidence) — what D10's decay
  overlays replay from;
- a checkpoint store at every session end, taken when the user would come back:
  the overdue maintenance for the gap (decay included) has run at the next
  session's start time, the next turn not yet processed (`checkpoints/cp-<turn>.sql`);
  after the last session the return is one more seeded gap (clock.session_gap).

Questions are not asked here; read passes (S9) ask them on the saved stores.

    uv run --no-sync python experiments/z2/replay.py --history B \
        --out logs/z2-replay-<date>-r1 --database ice_z2_replay_b_r1 --create-db
"""

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-replay-1"
HISTORY_SLUG = {"B": "bb558b5f", "C": "a77c15cf"}
# C is not in seed_v3's corpus (only B and the development histories); the wrapper
# supplies it from the frozen input plan without editing the pinned seed_v3.
EXTRA_HISTORIES = {"C": ("logs/v3-offline-input-plan-2026-10-10-r4/final-originals-C.json", 325)}
DECAY_TABLES = {
    "decay_episodic": ("episodic_memory", "decay_score"),
    "decay_codex": ("codex_facts", "usage"),
    "decay_procedural": ("procedural_memory", "is_active"),   # stale patterns are switched off
}


class EventLog:
    def __init__(self, path):
        self.path = path
        self.handle = path.open("a")
        self.progress_path = path.with_name("progress.jsonl")
        self.turn = None
        self.source_time = None         # the replay clock the current maintenance runs at

    def progress(self, turn):
        with self.progress_path.open("a") as handle:
            handle.write(json.dumps(dict(turn=turn, wall=round(time.time(), 3))) + "\n")

    def trim_progress(self, completed):
        """The meter's per-turn file back to the restored checkpoint's turn count."""
        if self.progress_path.exists():
            rows = [json.loads(line) for line in self.progress_path.read_text().splitlines() if line.strip()]
            self.progress_path.write_text("".join(json.dumps(r) + "\n" for r in rows
                                                  if (r.get("turn") or 0) <= completed))

    def truncate(self, size):
        """Back to the byte size recorded with the restored checkpoint; the cut
        tail is kept beside it, never deleted."""
        if size is None:
            raise RuntimeError("restored checkpoint has no event-log position; cannot resume safely")
        self.handle.flush()
        if self.path.stat().st_size > size:
            with self.path.open("rb+") as handle:
                handle.seek(size)
                tail = handle.read()
                handle.truncate(size)
            with self.path.with_name(f"events-unfinished-{int(time.time())}.jsonl").open("wb") as out:
                out.write(tail)

    def write(self, **row):
        self.handle.write(json.dumps(dict(row, turn=self.turn, wall=round(time.time(), 3)), default=str) + "\n")
        self.handle.flush()


@contextmanager
def boundary_clock(stamp, engine):
    """Re-point the turn's historical clock at *stamp* for the boundary maintenance.

    seed_v3 runs maintenance inside the session-end turn's `historical_clock`, and
    that clock cannot nest (it only patches modules still bound to the real
    datetime). So every src module bound to a datetime class is re-patched to
    *stamp*, and a second SQL listener binds *stamp* whichever of the two rewrites
    NOW() first (SQLAlchemy runs retval listeners in registration order, `insert`
    notwithstanding). Restored on exit."""
    from sqlalchemy import event, text
    from sqlalchemy.sql.elements import TextClause
    from scripts.z1.historical_clock import SQL_NOW, source_datetime
    clock = source_datetime(stamp)
    modules = [m for n, m in list(sys.modules.items()) if n.startswith("src.")
               and isinstance(getattr(m, "datetime", None), type) and issubclass(m.datetime, datetime)]

    def bind(_connection, statement, multiparams, params, _options):
        if not isinstance(statement, TextClause):
            return statement, multiparams, params
        if SQL_NOW.search(statement.text):
            statement = text(SQL_NOW.sub("(:ice_replay_clock)", statement.text))
        if "(:ice_replay_clock)" in statement.text:      # ours, or the turn's rewrite: our stamp wins
            if multiparams:
                multiparams = [{**p, "ice_replay_clock": stamp} for p in multiparams]
            else:
                params = {**params, "ice_replay_clock": stamp}
        return statement, multiparams, params

    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(patch.object(module, "datetime", clock))
        event.listen(engine, "before_execute", bind, retval=True)
        stack.callback(event.remove, engine, "before_execute", bind)
        yield


def _column_state(db, table, column):
    from sqlalchemy import text
    return {str(r[0]): r[1] for r in db.execute(text(f"SELECT id, {column} FROM {table}"))}


def install(log, history, session_ends, next_start, snapdir, stack, turn_at, final=(None, None)):
    """Patches for the event log and the session-boundary checkpoints.
    *turn_at* maps a turn's replay timestamp to its number (filled by load_plan);
    *final* = (last turn, gap): the user's return after the last session, so the
    questions asked at the history's end have a checkpoint store too."""
    import importlib
    from scripts.z1 import seed_v3, snapshot
    from src.memory import usage
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    from src.workers.runtime import JOBS

    real_strengthen = HybridRetrievalOrchestrator._strengthen_retrieved

    def strengthen(self, fragments):
        ids = sorted({str(f.source_batch_id) for f in fragments
                      if f.source_type == "episodic" and f.source_batch_id})
        real_strengthen(self, fragments)
        if ids:
            from src.api.config import settings
            log.write(event="strengthen", episodic_ids=ids, amount=settings.decay_strengthen_amount,
                      enabled=settings.retrieval_strengthen_writes)

    real_usage = usage.record_graph_access

    def graph_access(db, fragments, *, stage):
        from sqlalchemy import text
        from src.api.config import settings
        facts = sorted({str(i) for f in fragments for i in (getattr(f, "origin_fact_ids", ()) or ())})
        # The writer increments only facts still valid; record exactly which (the overlay
        # replays usage from these, not from what reached the prompt).
        applied = sorted(str(r[0]) for r in db.execute(text(
            "SELECT id FROM codex_facts WHERE id = ANY(CAST(:ids AS uuid[])) AND valid_until IS NULL"),
            {"ids": facts})) if facts and settings.retrieval_strengthen_writes else []
        result = real_usage(db, fragments, stage=stage)
        if facts:
            log.write(event="fact_usage", fact_ids=facts, applied_ids=applied, stage=stage)
        return result

    real_resurrect = HybridRetrievalOrchestrator._resurrect_cold_hits

    def resurrect(self, final):
        from sqlalchemy import text
        from src.api.config import settings
        hits = sorted({str(f.source_batch_id) for f in final
                       if f.source_batch_id and (getattr(self, "_cold_hits", None) or {}).get(f.source_batch_id)})
        real_resurrect(self, final)
        if hits:
            back = sorted(str(r[0]) for r in self.db.execute(text(
                "SELECT id FROM episodic_memory WHERE id = ANY(CAST(:ids AS uuid[]))"), {"ids": hits}))
            if back:
                log.write(event="resurrect", episodic_ids=back, score=settings.timescope_probation_score)

    stack.enter_context(patch.object(HybridRetrievalOrchestrator, "_resurrect_cold_hits", resurrect))
    stack.enter_context(patch.object(HybridRetrievalOrchestrator, "_strengthen_retrieved", strengthen))
    stack.enter_context(patch.object(usage, "record_graph_access", graph_access))

    for job, (table, column) in DECAY_TABLES.items():
        module, function = JOBS[job].path.split(":")
        owner = importlib.import_module(module)
        real = getattr(owner, function)

        def decayed(*args, _real=real, _job=job, _table=table, _column=column, **kwargs):
            from src.api.db import SessionLocal
            with SessionLocal() as db:
                before = _column_state(db, _table, _column)
            result = _real(*args, **kwargs)
            with SessionLocal() as db:
                after = _column_state(db, _table, _column)
            changes = [(i, before.get(i), v) for i, v in after.items() if before.get(i) != v]
            moved = sorted(set(before) - set(after))        # e.g. episodic rows archived to cold
            log.write(event="decay", job=_job, table=_table, column=_column, rows=len(after),
                      changed=changes, moved_out=moved, kwargs=kwargs, source_time=log.source_time)
            return result

        stack.enter_context(patch.object(owner, function, decayed))

    real_maintenance = seed_v3.due_maintenance

    def maintenance(db, sink, source_time, conversation_id, last_run, recovery=None, context=None):
        turn = (context or {}).get("turn")
        log.turn, log.source_time = turn, source_time
        result = real_maintenance(db, sink, source_time, conversation_id, last_run, recovery, context)
        back = next_start.get(turn) or (source_time + final[1] if turn == final[0] else None)
        if turn in session_ends and back is not None:
            # Checkpoint = when the user comes back: run what is overdue at the next
            # session's start (decay over the gap) ON that start's clock — the 7-day
            # and staleness cutoffs read it — then save, before the next turn.
            log.source_time = back
            with boundary_clock(back, db.get_bind()):
                real_maintenance(db, sink, back, conversation_id, last_run, recovery,
                                 dict(context or {}, stage="session_boundary"))
            with patch.object(snapshot, "SNAPDIR", snapdir):
                if snapshot.save(f"cp-{turn}"):          # returns 1 on a failed pg_dump
                    raise RuntimeError(f"session-boundary checkpoint cp-{turn} could not be saved")
            log.write(event="checkpoint", store=f"cp-{turn}.sql", at=back)
        return result

    real_store = seed_v3._store_turn

    def store(*args, **kwargs):
        key = args[5] if len(args) > 5 else kwargs.get("idempotency_key")
        log.turn = int(str(key).rsplit("-", 1)[-1]) if key else log.turn
        return real_store(*args, **kwargs)

    real_clock = seed_v3.historical_clock

    @contextmanager
    def clock(stamp, engine):
        # Everything inside a turn's clock (preflight strengthening, fact usage,
        # post-flight, maintenance) is that turn's, at that turn's source time.
        log.turn, log.source_time = turn_at.get(stamp, log.turn), stamp
        with real_clock(stamp, engine) as stats:
            yield stats
        log.progress(log.turn)                      # the turn completed (the runner's meter)

    stack.enter_context(patch.object(seed_v3, "historical_clock", clock))

    # Resume safety: seed_v3 restores the store and truncates its trace to the last
    # durable checkpoint; the event log must go back to the same point, or a resumed
    # run would count the redone turns' strengthenings twice.
    from scripts.z1 import replay_checkpoint
    real_capture, real_recover = replay_checkpoint.Checkpoints.capture, replay_checkpoint.Checkpoints.recover

    def capture(self, sink, state, **kwargs):
        log.handle.flush()
        os.fsync(log.handle.fileno())
        return real_capture(self, sink, dict(state, z2_events_bytes=log.path.stat().st_size), **kwargs)

    def recover(self):
        state = real_recover(self)
        log.truncate(state.get("z2_events_bytes"))
        log.trim_progress(sum((state.get("completed") or {}).values()))
        log.write(event="resumed", completed=state.get("completed"))   # after the cut, so it survives
        return state

    stack.enter_context(patch.object(replay_checkpoint.Checkpoints, "capture", capture))
    stack.enter_context(patch.object(replay_checkpoint.Checkpoints, "recover", recover))
    stack.enter_context(patch.object(seed_v3, "due_maintenance", maintenance))
    stack.enter_context(patch.object(seed_v3, "_store_turn", store))


def extra_history(history):
    """Rows in seed_v3's conversation format from the frozen input plan."""
    from datetime import datetime
    path, expected = EXTRA_HISTORIES[history]
    rows = json.loads((ROOT / path).read_text())
    if len(rows) != expected or [r["turn_number"] for r in rows] != list(range(1, expected + 1)):
        raise ValueError(f"{history}: expected turns 1..{expected}")
    return [dict(turn_number=r["turn_number"],
                 timestamp=datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")),
                 ts_provenance=r["timestamp_provenance"], prompt=r["prompt"], response=r["response"])
            for r in rows]


def run_replay(args) -> int:
    from experiments.z2.clock import load_history, table
    from experiments.z2.profile_write_path import retime
    from scripts.z1 import seed_v3
    slug = HISTORY_SLUG[args.history]
    clock = table(args.history, load_history(args.history))
    sessions = clock["sessions"]
    session_ends = {s["last_turn"] for s in sessions}          # the last one too: questions end there
    from experiments.z2.clock import session_gap
    final = (sessions[-1]["last_turn"], session_gap(args.history, len(sessions) + 1))  # same seeded rule
    if args.limit and args.limit < final[0]:
        # A slice ends where it stops: its last turn gets the final return and
        # checkpoint exactly as the full history's does (so slices exercise it).
        session_ends.add(args.limit)
        final = (args.limit, final[1])
    next_start = {s["last_turn"]: datetime.fromisoformat(n["start"]) for s, n in zip(sessions, sessions[1:])}
    original_load = seed_v3.load_plan
    turn_at = {}
    base_expected = dict(seed_v3.EXPECTED)

    def load_plan(**kwargs):
        with patch.object(seed_v3, "EXPECTED", base_expected):          # its corpus check sees B/dev only
            conversations, probes, quarantined, ineligible = original_load(**kwargs)
        if args.history in EXTRA_HISTORIES:
            conversations[slug] = extra_history(args.history)
        retime(conversations, slug, args.history)
        turn_at.update({row["timestamp"]: row["turn_number"] for row in conversations[slug]})
        return conversations, probes, quarantined, ineligible

    argv = ["seed_v3.py", "--conversation", slug, "--probe-panel", "immediate", "--no-probe-controls",
            "--out", str(args.out / "seed.jsonl"), "--checkpoint-every", "10",
            "--worker-failure-policy", args.worker_failure_policy]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.resume:
        argv.append("--resume")
    snapdir = args.out / "checkpoints"
    snapdir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if args.history in EXTRA_HISTORIES:
            stack.enter_context(patch.dict(seed_v3.EXPECTED, {slug: EXTRA_HISTORIES[args.history][1]}))
            stack.enter_context(patch.dict(seed_v3.SOURCE_TIMESTAMP_PROVENANCE,
                                           {slug: "synthetic_raw_import"}))
        log = EventLog(args.out / "events.jsonl")
        log.write(event="run_start", policy=POLICY, clock_policy=clock["policy"], argv=argv,
                  resume=bool(args.resume))
        install(log, args.history, session_ends, next_start, snapdir, stack, turn_at, final)
        stack.enter_context(patch.object(seed_v3, "load_plan", load_plan))
        stack.enter_context(patch.object(sys, "argv", argv))
        return seed_v3.main()


def main() -> int:
    from experiments.z2.profile_write_path import create_database
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", choices=sorted(HISTORY_SLUG), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--limit", type=int, help="replay pairs 1..limit (a slice check)")
    parser.add_argument("--create-db", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-failure-policy", choices=("strict", "continue"), default="continue")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if os.environ.get("ICE_TEST_DATABASE") != args.database:
            raise RuntimeError("child started without its scratch database")
        return run_replay(args)
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("replay output is private: put it under logs/")
    if args.create_db == args.resume:
        raise ValueError("pass exactly one of --create-db (new run) or --resume")
    if args.create_db:
        if args.out.exists():
            raise RuntimeError(f"{args.out} exists; use --resume or a new run name")
        url = create_database(args.database)
    else:
        from sqlalchemy.engine import make_url
        from src.api.config import settings as live
        url = make_url(live.database_url).set(database=args.database).render_as_string(hide_password=False)
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DATABASE_URL=url, ICE_TEST_DATABASE=args.database)
    os.execve(sys.executable, [sys.executable, *[a for a in sys.argv if a != "--create-db"], "--child"], env)


if __name__ == "__main__":
    raise SystemExit(main())
