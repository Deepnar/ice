"""A historical clock scoped to the isolated v3 replay, never the proxy.

Only ICE module clocks and explicit SQL NOW() calls are substituted. Request
timeouts, model inference timers and the host's clock remain unchanged.
"""
from __future__ import annotations

import importlib
import re
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

from sqlalchemy import event
from sqlalchemy.sql.elements import TextClause


# Load lazy memory-path consumers before patching their datetime bindings.
CLOCK_MODULES = (
    "src.memory.models", "src.memory.store_meta", "src.memory.usage",
    "src.api.prompt_assembler", "src.retrieval.orchestrator",
    "src.retrieval.timescope", "src.retrieval.evolution",
    "src.workers.post_flight", "src.workers.codex_extractor",
    "src.workers.codex_ops", "src.workers.procedural_extractor",
    "src.workers.clustering", "src.workers.conversation_summary",
    "src.workers.batch_summarizer", "src.workers.reflection",
    "src.workers.maintenance_agent", "src.workers.decay",
    "src.workers.codex_decay", "src.workers.procedural_decay",
    "src.workers.compaction",
)
SQL_NOW = re.compile(r"\bNOW\s*\(\s*\)", re.IGNORECASE)


def source_datetime(stamp: datetime):
    if stamp.tzinfo is None:
        raise ValueError("historical replay clock needs a timezone-aware timestamp")

    class ClockType(type):
        def __instancecheck__(cls, value):
            return isinstance(value, datetime)

    class SourceDateTime(datetime, metaclass=ClockType):
        @classmethod
        def now(cls, tz=None):
            return (stamp.astimezone(tz) if tz is not None
                    else stamp.astimezone().replace(tzinfo=None))

        @classmethod
        def utcnow(cls):
            return stamp.astimezone(timezone.utc).replace(tzinfo=None)

        @classmethod
        def today(cls):
            return cls.now()

    return SourceDateTime


@contextmanager
def historical_clock(stamp: datetime, engine):
    from scripts.z1.seed_v3 import isolated_database
    from src.api.config import settings
    from sqlalchemy.engine import make_url
    isolated_database()
    if engine.url.database != make_url(settings.database_url).database:
        raise ValueError("historical clock engine differs from the isolated replay database")
    clock = source_datetime(stamp)
    for name in CLOCK_MODULES:
        importlib.import_module(name)
    modules = [module for name, module in list(sys.modules.items())
               if name.startswith("src.") and getattr(module, "datetime", None) is datetime]
    stats = {"python_modules": sorted(module.__name__ for module in modules),
             "sql_now_statements": 0}

    def bind_clock(_connection, statement, multiparams, params, _options):
        if isinstance(statement, TextClause) and SQL_NOW.search(statement.text):
            from sqlalchemy import text
            stats["sql_now_statements"] += 1
            rewritten = text(SQL_NOW.sub("(:ice_replay_clock)", statement.text))
            if multiparams:
                multiparams = [{**parameters, "ice_replay_clock": stamp}
                               for parameters in multiparams]
            else:
                params = {**params, "ice_replay_clock": stamp}
            return rewritten, multiparams, params
        return statement, multiparams, params

    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(patch.object(module, "datetime", clock))
        event.listen(engine, "before_execute", bind_clock, retval=True)
        stack.callback(event.remove, engine, "before_execute", bind_clock)
        yield stats
