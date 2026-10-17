"""Run a standalone DB suite in a newly created, disposable PostgreSQL database.

Usage: uv run python tests/support/disposable_database.py tests/test_timescope.py
Creates current ORM tables plus pgvector; this checks behavior, not migrations.
Never substitutes the normal database when setup fails.
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.memory.models import Base


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Supply the standalone test script to run.")
    name = f"ice_test_{uuid.uuid4().hex}"
    source = make_url(settings.database_url)
    admin = create_engine(source.set(database="postgres"), isolation_level="AUTOCOMMIT")
    target = source.set(database=name)
    created = False
    engine = None
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        created = True
        engine = create_engine(target)
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(engine)
        env = dict(
            os.environ,
            DATABASE_URL=target.render_as_string(hide_password=False),
            ICE_TEST_DATABASE=name,
        )
        print("Running isolated suite in a newly created test database.", flush=True)
        result = subprocess.run([sys.executable, *sys.argv[1:]], env=env, check=False)
        return result.returncode
    finally:
        if engine is not None:
            engine.dispose()
        if created:
            with admin.connect() as conn:
                conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
            print("Disposable test database removed.", flush=True)
        admin.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
