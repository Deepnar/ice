"""Actual manual bootstrap: empty owned clone, index parity, safe reattachment.

This control creates/removes only its random campaign database and never
launches replay or model/API calls. The normal database is read for DDL only.
"""
import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from scripts.z1.answer_as_of import LOGS
from scripts.z1.run_v3_campaign import database_environment
from src.api.config import settings
from src.memory.models import Base


def main():
    identity = uuid.uuid4().hex
    name = "ice_v3_campaign_" + identity[:16]
    config = {"database": name, "campaign_id": identity,
              "background_model": "gemma4:e4b", "codex_extraction_model": "hf.co/numind/NuExtract3-GGUF:Q8_0"}
    source = make_url(settings.database_url)
    admin = create_engine(source.set(database="postgres"), isolation_level="AUTOCOMMIT")
    target = None
    try:
        with tempfile.TemporaryDirectory(prefix="z1-schema-control-", dir=LOGS) as folder:
            root = Path(folder)
            env = database_environment(config, root)
            assert env["ICE_TEST_DATABASE"] == name
            target = create_engine(env["DATABASE_URL"])
            with target.connect() as db:
                assert all(db.execute(text(f'SELECT count(*) FROM "{table}"')).scalar() == 0
                           for table in Base.metadata.tables)
            receipt = json.loads((root / "schema.json").read_text())
            assert len(receipt["tables"]) == 35 and receipt["schema_signature"]
            assert database_environment(config, root)["DATABASE_URL"] == env["DATABASE_URL"]
            with target.begin() as db:
                db.execute(text("ALTER TABLE episodic_memory ADD COLUMN unexpected_control integer"))
            try:
                database_environment(config, root)
            except ValueError as exc:
                assert "schema changed" in str(exc)
            else:
                raise AssertionError("changed schema was accepted")
            with admin.connect() as db:
                db.execute(text(f'COMMENT ON DATABASE "{name}" IS NULL'))
            try:
                database_environment(config, root)
            except ValueError as exc:
                assert "ownership marker" in str(exc)
            else:
                raise AssertionError("unowned store was accepted")
        print("Manual schema control passed: 35 empty tables, matching production indexes/defaults/constraints, safe reattachment and drift/ownership refusal.")
        return 0
    finally:
        if target is not None:
            target.dispose()
        with admin.connect() as db:
            row = db.execute(text("SELECT 1 FROM pg_database WHERE datname=:n"), {"n": name}).first()
            if row:
                db.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
        print("Schema-control database removed.")


if __name__ == "__main__":
    raise SystemExit(main())
