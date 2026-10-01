#!/usr/bin/env python3
"""Z1: complete v3 isolated-store snapshots for tuning and comparison.

Population is slow and model-dependent. A restored snapshot lets later scoring
use the exact same rows, including claims, notes, links and operational state.
The older 293-turn snapshots lack this complete identity contract and cannot
be restored with this version.

**It is also the answer to [G38](../../docs/ROADMAP.md).** Retrieval commits in
three places, so probe N mutates the store probe N+1 sees. `score_retrieval.py`
neutralises two of those with settings; restoring before each run removes the
question entirely, including anything a future change adds.

Only `alembic_version` is excluded. The table list comes from current ORM
metadata and includes operational state. Restore is limited to a dedicated
test database and verifies both the dump bytes and every restored row.

Run:
  uv run python scripts/z1/snapshot.py save   --arm gemma4-26b
  uv run python scripts/z1/snapshot.py list
  uv run python scripts/z1/snapshot.py restore --arm gemma4-26b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy.engine import make_url

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.z1.run_meta import run_meta
from scripts.z1.replay_validation import validate_complete_replay
from src.api.config import settings
from src.memory.models import Base

SNAPDIR = Path("experiments/curation_files/snapshots")
CONTAINER = "ice_postgres"
DB = make_url(settings.database_url).database
USER = "ice"

# All current ORM tables, including operational rows; alembic_version is not
# an ORM table and is deliberately outside a data-only snapshot.
TABLES = sorted(Base.metadata.tables)


def _isolated_database() -> None:
    """A snapshot restore may truncate tables; refuse the normal user store."""
    marker = os.environ.get("ICE_TEST_DATABASE")
    if not marker or marker != DB or DB == "ice_db":
        raise RuntimeError("Z1 snapshots require DATABASE_URL and ICE_TEST_DATABASE "
                           "to name the same dedicated non-ice_db database")


def _docker(cmd: list[str], **kw):
    return subprocess.run(["docker", "exec", CONTAINER] + cmd,
                          capture_output=True, text=True, **kw)


def counts() -> dict:
    sql = " UNION ALL ".join(
        f"SELECT '{t}', count(*) FROM {t}" for t in TABLES)
    r = _docker(["psql", "-U", USER, "-d", DB, "-tAc", sql])
    if r.returncode != 0:
        raise RuntimeError(f"snapshot counts failed: {r.stderr[:300]}")
    out = {}
    for line in r.stdout.strip().splitlines():
        if "|" in line:
            k, v = line.split("|", 1)
            out[k] = int(v)
    return out


def fingerprints() -> dict[str, str]:
    """Identity of *all* ORM rows, including claims, notes and source links."""
    out = {}
    for name in TABLES:
        table = Base.metadata.tables[name]
        order = ", ".join(f'"{c.name}"' for c in table.primary_key.columns)
        sql = f'SELECT row_to_json(t)::text FROM "{name}" t ORDER BY {order}'
        r = _docker(["psql", "-U", USER, "-d", DB, "-tA", "-c", sql])
        if r.returncode != 0:
            raise RuntimeError(f"snapshot fingerprint failed for {name}: {r.stderr[:300]}")
        out[name] = hashlib.sha256(r.stdout.encode("utf-8")).hexdigest()
    return out


def trace_identity(path: Path, live_counts: dict, live_fingerprints: dict) -> dict:
    """Bind a campaign snapshot to a complete, unchanged isolated replay."""
    logs = (Path(__file__).resolve().parents[2] / "logs").resolve()
    if not path.resolve().is_relative_to(logs):
        raise ValueError("private v3 replay traces must remain under logs/")
    from scripts.z1.seed_v3 import EXPECTED

    digest = hashlib.sha256()
    with path.open("rb") as source:
        def events():
            for line in source:
                digest.update(line)
                yield json.loads(line)
        last = validate_complete_replay(events(), EXPECTED)["complete"]
    if last.get("table_counts") != live_counts:
        raise ValueError("live store changed since the complete replay trace")
    if not last.get("table_sha256") or last["table_sha256"] != live_fingerprints:
        raise ValueError("live row identities changed since the complete replay trace")
    return {"sha256": digest.hexdigest(), "path": str(path.resolve()),
            "as_of_probes": last["as_of_probes"],
            "turns_by_conversation": last["turns_by_conversation"]}


def save(arm: str, trace: Path | None = None) -> int:
    _isolated_database()
    SNAPDIR.mkdir(parents=True, exist_ok=True)
    dest = SNAPDIR / f"{arm}.sql"
    c = counts()
    identity = fingerprints()
    seed_trace = trace_identity(trace, c, identity) if trace else None
    args = ["pg_dump", "-U", USER, "-d", DB, "--data-only", "--no-owner"]
    for t in TABLES:
        args += ["-t", t]
    r = _docker(args)
    if r.returncode != 0:
        print(f"pg_dump failed:\n{r.stderr[:800]}")
        return 1
    dest.write_text(r.stdout)
    if counts() != c or fingerprints() != identity:
        raise RuntimeError("isolated store changed while its snapshot was being saved")
    manifest = {"format": "ice-v3-z1-snapshot-2", "database": DB,
                "tables": TABLES, "counts": c, "sha256_by_table": identity,
                "dump_sha256": hashlib.sha256(r.stdout.encode()).hexdigest(),
                "complete_seed_trace": seed_trace,
                "meta": run_meta(script=__file__, args={"action": "save", "arm": arm})}
    (SNAPDIR / f"{arm}.manifest.json").write_text(json.dumps(manifest, indent=2))
    (SNAPDIR / f"{arm}.counts").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n" +
        "\n".join(f"{k}={v}" for k, v in sorted(c.items())))
    live = {k: v for k, v in c.items() if v}
    print(f"saved {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
    print(f"  {live}")
    return 0


def restore(arm: str) -> int:
    _isolated_database()
    src = SNAPDIR / f"{arm}.sql"
    if not src.exists():
        print(f"no snapshot at {src}")
        return 1
    manifest_path = SNAPDIR / f"{arm}.manifest.json"
    if not manifest_path.exists():
        print("old snapshot has no complete v3 manifest; refusing restore")
        return 1
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != "ice-v3-z1-snapshot-2"
            or manifest.get("database") != DB or manifest.get("tables") != TABLES):
        print("snapshot database/schema differs from the current isolated store")
        return 1
    dump = src.read_bytes()
    if hashlib.sha256(dump).hexdigest() != manifest.get("dump_sha256"):
        print("snapshot dump changed since manifest; refusing restore")
        return 1
    # One transaction: a failed COPY rolls back the truncate as well. No
    # CASCADE: an omitted dependent table is a manifest defect, not expendable.
    quoted_tables = ", ".join('"' + t + '"' for t in TABLES)
    truncate = f"TRUNCATE {quoted_tables} RESTART IDENTITY;"
    p = subprocess.run(["docker", "exec", "-i", CONTAINER,
                        "psql", "-U", USER, "-d", DB, "-v", "ON_ERROR_STOP=1"],
                       input=f"BEGIN;\n{truncate}\n{dump.decode()}\nCOMMIT;\n",
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"restore failed:\n{p.stderr[:800]}")
        return 1
    actual_counts = counts()
    actual_identity = fingerprints()
    bad = [t for t in TABLES if actual_counts.get(t) != manifest["counts"].get(t)
           or actual_identity.get(t) != manifest["sha256_by_table"].get(t)]
    if bad:
        print("restore identity mismatch in: " + ", ".join(bad))
        return 1
    live = {k: v for k, v in actual_counts.items() if v}
    print(f"restored {arm}\n  {live}")
    print("  every table count and row fingerprint matches")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["save", "restore", "list", "counts"])
    ap.add_argument("--arm", default=None)
    ap.add_argument("--trace", type=Path,
                    help="bind a campaign snapshot to a complete v3 trace under logs/")
    args = ap.parse_args()

    if args.action == "counts":
        print({k: v for k, v in counts().items() if v})
        return 0
    if args.action == "list":
        if not SNAPDIR.exists():
            print("no snapshots yet")
            return 0
        for f in sorted(SNAPDIR.glob("*.sql")):
            cf = f.with_suffix(".counts")
            head = cf.read_text().splitlines()[0] if cf.exists() else "?"
            print(f"  {f.stem:24} {f.stat().st_size/1e6:7.1f} MB   {head}")
        return 0
    if not args.arm:
        print("--arm is required for save/restore")
        return 1
    return save(args.arm, args.trace) if args.action == "save" else restore(args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
