"""Smoke controls, including explicitly isolated SQL/snapshot controls.

Every test is tagged automatically (no test file declares it):
  research — the module imports `experiments` or `scripts` research tooling
  db       — the module needs a disposable database (ICE_TEST_DATABASE)
  quick    — neither: ICE's own code, no database

Quick, per commit:      uv run --no-sync pytest tests/smoke -q -m quick
Research tooling:       uv run --no-sync pytest tests/smoke -q -m "research and not db"
Full suite (before long runs and pushes):
  uv run --no-sync python tests/support/disposable_database.py -m pytest tests/smoke -q
"""

import pathlib
import re
import sys

import pytest

# Repo root on sys.path (same convention as the standalone tests/ scripts).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

RESEARCH = re.compile(r"^\s*(from|import)\s+(experiments|scripts)\b", re.M)
DATABASE = re.compile(r"ICE_TEST_DATABASE|empty_replay_database|disposable_database")


def pytest_configure(config):
    for name, text in (("quick", "ICE code without a database"),
                       ("research", "research tooling under experiments/ or scripts/"),
                       ("db", "needs a disposable ICE_TEST_DATABASE")):
        config.addinivalue_line("markers", f"{name}: {text}")


def pytest_collection_modifyitems(items):
    groups = {}
    for item in items:
        path = pathlib.Path(str(item.fspath))
        if path not in groups:
            source = path.read_text()
            groups[path] = (bool(RESEARCH.search(source)), bool(DATABASE.search(source)))
        research, database = groups[path]
        database = database or "empty_replay_database" in getattr(item, "fixturenames", ())
        if research: item.add_marker("research")
        if database: item.add_marker("db")
        if not (research or database): item.add_marker("quick")


@pytest.fixture
def empty_replay_database(monkeypatch):
    """Replay needs a genuinely empty store, even after other smoke fixtures."""
    import os
    import uuid
    from types import SimpleNamespace as NS
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import sessionmaker
    from scripts.z1 import seed_v3, snapshot
    from src.api import db as database
    from src.api.config import settings
    from src.memory import embedder as embedding_module
    from src.memory.models import Base

    controlled = NS(encode=lambda *_a, **_k: [1.0] + [0.0] * (settings.embedding_dim - 1))
    monkeypatch.setattr(embedding_module, "get_embedder", lambda: controlled)
    from src.workers import post_flight
    source = make_url(settings.database_url)
    assert (os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
            and source.database == os.environ["ICE_TEST_DATABASE"])
    name = "ice_test_" + uuid.uuid4().hex
    admin = create_engine(source.set(database="postgres"), isolation_level="AUTOCOMMIT")
    engine = None
    created = False
    try:
        with admin.connect() as conn: conn.execute(text(f'CREATE DATABASE "{name}"'))
        created = True
        target = source.set(database=name)
        engine = create_engine(target)
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))   # CODEX_V4 name matching
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        for module in (database, seed_v3, post_flight):
            monkeypatch.setattr(module, "SessionLocal", sessions)
        monkeypatch.setattr(settings, "database_url", target.render_as_string(hide_password=False))
        monkeypatch.setattr(snapshot, "DB", name)
        monkeypatch.setenv("ICE_TEST_DATABASE", name)
        yield
    finally:
        if engine is not None: engine.dispose()
        if created:
            with admin.connect() as conn: conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
