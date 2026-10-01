"""Replay maintenance shares the runtime's cadence, ordering and cycle cap."""
import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from scripts.z1 import seed_v3
from src.api.config import settings
from src.workers.runtime import JOBS


def test_periodic_schedule_uses_real_registry_and_missed_cycles(monkeypatch):
    calls = []
    functions = {}
    for name in seed_v3.MEMORY_JOBS:
        spec = JOBS[name]
        module, function = spec.path.split(":")
        def called(*args, _name=name, _needs_db=spec.needs_db, **kwargs):
            assert len(args) == int(_needs_db)
            calls.append((_name, kwargs))
        functions.setdefault(module, {})[function] = called
    monkeypatch.setattr(seed_v3, "importlib", SimpleNamespace(
        import_module=lambda module: SimpleNamespace(**functions[module])))
    monkeypatch.setattr(seed_v3, "memory_state", lambda _db: {"warm_turns": 2, "cold_turns": 1})
    monkeypatch.setattr(settings, "maintenance_intervals", dict.fromkeys(seed_v3.MEMORY_JOBS, 10))
    monkeypatch.setattr(settings, "runtime_cycles_cap", 3)
    db = SimpleNamespace(expire_all=lambda: None)
    sink = io.StringIO()
    last_run = {}
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    seed_v3.due_maintenance(db, sink, now, "example", last_run)
    assert [name for name, _ in calls] == sorted(seed_v3.MEMORY_JOBS, reverse=True)
    assert all(kwargs == ({"cycles": 1} if JOBS[name].pass_cycles else {})
               for name, kwargs in calls)
    calls.clear()
    seed_v3.due_maintenance(db, sink, now + timedelta(seconds=10), "example", last_run)
    assert not calls  # Production requires overdue > 0, not >= 0.
    seed_v3.due_maintenance(db, sink, now + timedelta(seconds=101), "example", last_run)
    assert len(calls) == 10
    assert all(kwargs == ({"cycles": 3} if JOBS[name].pass_cycles else {})
               for name, kwargs in calls)
    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(rows) == 20 and all(row["before"] == row["after"] for row in rows)
    assert all(row["event"] == "maintenance" for row in rows)
