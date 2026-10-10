"""Smoke controls, including explicitly isolated SQL/snapshot controls.

Full suite: uv run --no-sync python tests/support/disposable_database.py
            -m pytest tests/smoke -q
"""

import pathlib
import sys

import pytest

# Repo root on sys.path (same convention as the standalone tests/ scripts).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


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
        with engine.begin() as conn: conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
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
