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
    monkeypatch.setattr(seed_v3, "semantic_state", lambda _db, _job: {})
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


def test_semantic_changes_preserve_rewrites_deletions_and_composite_keys():
    before = {"conversation_notes": {
        '["conv",0]': {"text": "old answer", "source_ids": ["source-a"]},
        '["conv",1]': {"text": "removed", "source_ids": ["source-b"]},
        '["conv",2]': {"text": "unchanged", "source_ids": ["source-c"]},
    }}
    after = {"conversation_notes": {
        '["conv",0]': {"text": "corrected answer", "source_ids": ["source-d"]},
        '["conv",3]': {"text": "added", "source_ids": ["source-e"]},
        '["conv",2]': before["conversation_notes"]['["conv",2]'],
    }}
    assert len(before["conversation_notes"]) == len(after["conversation_notes"])
    changes = seed_v3.semantic_changes(before, after)
    assert [row["key"] for row in changes] == [["conv", 0], ["conv", 1], ["conv", 3]]
    assert changes[0]["before"]["source_ids"] == ["source-a"]
    assert changes[0]["after"]["text"] == "corrected answer"
    assert changes[1]["after"] is None and changes[2]["before"] is None
