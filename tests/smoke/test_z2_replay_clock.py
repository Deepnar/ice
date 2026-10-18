"""Z2 S7 replay: the session-boundary maintenance runs on the NEXT session's clock,
not the session-end turn's (spec D8 checkpoints). Python clocks and SQL NOW() both."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy import create_engine, event, text

from experiments.z2.replay import boundary_clock
from scripts.z1.historical_clock import SQL_NOW, source_datetime
from src.workers import decay

TURN = datetime(2025, 1, 1, 20, 0, tzinfo=timezone.utc)
NEXT = TURN + timedelta(days=2, hours=5)


def turn_listener(stamp):
    """The shape of historical_clock's own NOW() rewrite, registered first."""
    def bind(_c, statement, multiparams, params, _o):
        if hasattr(statement, "text") and SQL_NOW.search(statement.text):
            return text(SQL_NOW.sub("(:ice_replay_clock)", statement.text)), multiparams, \
                {**params, "ice_replay_clock": stamp}
        return statement, multiparams, params
    return bind


def sql_now(engine):
    with engine.connect() as c:
        return str(c.execute(text("SELECT NOW()")).scalar())


def test_boundary_moves_python_and_sql_clocks_then_restores():
    engine = create_engine("sqlite://")
    outer = turn_listener(TURN)
    event.listen(engine, "before_execute", outer, retval=True)
    with patch.object(decay, "datetime", source_datetime(TURN)):
        assert decay.datetime.now(timezone.utc) == TURN and sql_now(engine).startswith("2025-01-01 20:00")
        with boundary_clock(NEXT, engine):
            assert decay.datetime.now(timezone.utc) == NEXT
            assert sql_now(engine).startswith("2025-01-04 01:00")
        assert decay.datetime.now(timezone.utc) == TURN and sql_now(engine).startswith("2025-01-01 20:00")
    event.remove(engine, "before_execute", outer)
