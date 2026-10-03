"""Actual two-turn replay recovery after a committed, unfinished second write.

Run in the disposable-database wrapper; private output stays under logs/.
This validates recovery mechanics, not memory quality or the full campaign.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from scripts.z1 import seed_v3
from src.api.db import SessionLocal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    argv = ["seed_v3.py", "--conversation", "355a5709", "--limit", "2",
            "--probe-panel", "immediate", "--checkpoint-every", "1", "--out", str(args.out)]
    original = seed_v3._store_turn
    stored = []

    def interrupt_after_write(*a, **kw):
        result = original(*a, **kw)
        stored.append(str(a[5]))
        if len(stored) == 2:
            raise RuntimeError("intentional interrupted second write")
        return result

    with patch.object(sys, "argv", argv), patch.object(seed_v3, "_store_turn", interrupt_after_write):
        try:
            seed_v3.main()
        except RuntimeError as exc:
            assert str(exc) == "intentional interrupted second write"
        else:
            raise AssertionError("interruption was not exercised")
    rows = [json.loads(line) for line in args.out.open()]
    first = next(r for r in rows if r["event"] == "written")
    assert rows[-1]["event"] == "failed" and rows[-1]["turn"] == 2
    with SessionLocal() as db:
        assert db.execute(text("SELECT count(*) FROM episodic_memory")).scalar() == 2
    resumed = []

    def count_write(*a, **kw):
        resumed.append(str(a[5]))
        return original(*a, **kw)

    with patch.object(sys, "argv", argv + ["--resume"]), patch.object(seed_v3, "_store_turn", count_write):
        assert seed_v3.main() == 0
    assert len(resumed) == 1 and resumed[0].endswith("-2")
    rows = [json.loads(line) for line in args.out.open()]
    written = [r for r in rows if r["event"] == "written"]
    assert [r["turn"] for r in written] == [1, 2]
    assert written[0] == first
    assert all({"summary_text", "abstract_text", "source_raw_sha256"} <= set(r)
               for r in written)
    maintenance = [r for r in rows if r["event"] == "maintenance"]
    assert maintenance and all({"semantic_observed_tables", "semantic_changes",
                                "observer_elapsed_ms", "job_elapsed_ms"} <= set(r)
                               for r in maintenance)
    assert all(r["semantic_observed_tables"] for r in maintenance)
    assert not any(r["event"].endswith("failed") for r in rows)
    assert sum(r["event"] == "resume" for r in rows) == 1
    assert rows[-1]["turns_by_conversation"] == {"355a5709": 2}
    with SessionLocal() as db:
        assert db.execute(text("SELECT count(*) FROM episodic_memory")).scalar() == 2
        assert str(db.execute(text("SELECT id FROM episodic_memory ORDER BY timestamp LIMIT 1")).scalar()) == first["episodic_id"]
    recovery = args.out.with_suffix(".recovery")
    assert list(recovery.glob("unfinished-*.jsonl"))
    assert len(list(recovery.glob("generation-*"))) == 2
    print("Actual replay recovery passed: first-turn identity preserved; unfinished second write restored/replayed once; failed tail retained privately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
