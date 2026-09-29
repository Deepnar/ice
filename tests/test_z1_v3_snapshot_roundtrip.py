"""Run through tests/support/disposable_database.py; never touch ice_db."""

import os
import tempfile
import uuid
from pathlib import Path

from scripts.z1 import snapshot
from src.api.db import SessionLocal
from src.memory.models import Conversation


def main():
    assert os.environ["ICE_TEST_DATABASE"] == snapshot.DB != "ice_db"
    with tempfile.TemporaryDirectory(prefix="ice-z1-snapshot-") as root:
        snapshot.SNAPDIR = Path(root)
        cid = uuid.uuid4()
        db = SessionLocal()
        try:
            db.add(Conversation(id=cid, memory_scope_type="auto"))
            db.commit()
            assert snapshot.save("control") == 0
            before = snapshot.fingerprints()
            row = db.get(Conversation, cid)
            row.memory_scope_type = "none"
            db.commit()
            assert snapshot.fingerprints() != before
            assert snapshot.restore("control") == 0
            db.expire_all()
            assert db.get(Conversation, cid).memory_scope_type == "auto"
            assert snapshot.fingerprints() == before

            # A changed dump must be rejected before touching the live rows.
            dump = snapshot.SNAPDIR / "control.sql"
            dump.write_bytes(dump.read_bytes() + b"\n-- damage\n")
            assert snapshot.restore("control") == 1
            db.expire_all()
            assert db.get(Conversation, cid).memory_scope_type == "auto"

            # A damaged/missing identity manifest must refuse before TRUNCATE.
            (snapshot.SNAPDIR / "control.manifest.json").unlink()
            assert snapshot.restore("control") == 1
            db.expire_all()
            assert db.get(Conversation, cid).memory_scope_type == "auto"
        finally:
            db.close()
    print("complete v3 snapshot/restore identity roundtrip passed")


if __name__ == "__main__":
    main()
