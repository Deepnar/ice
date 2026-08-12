#!/usr/bin/env python3
"""Z1: snapshot and restore the seeded store, one snapshot per model arm.

**Why this exists.** Population is slow (~1.2 h) and non-deterministic — it runs
a background model over 293 turns. Tuning is the opposite: it must be fast and
byte-identical across runs, or a score difference cannot be attributed to the
setting that was changed. The resolution is to pay the LLM cost once per arm,
snapshot the result, and have every scoring run restore that snapshot.

**It is also the answer to [G38](../../docs/ROADMAP.md).** Retrieval commits in
three places, so probe N mutates the store probe N+1 sees. `score_retrieval.py`
neutralises two of those with settings; restoring before each run removes the
question entirely, including anything a future change adds.

**Why `pg_dump` of specific tables and not the whole database.** The store shares
its database with `alembic_version` and the operational tables. Restoring a full
dump would roll those back too, so a snapshot taken before a migration would
silently revert the schema underneath the code. The table list is derived from
the ORM metadata rather than typed out, for the same reason `seed_store.py`
discovers its dependents from the live schema: a hand-maintained list is a list
that goes stale the next time a table joins the pipeline.

Run:
  uv run python scripts/z1/snapshot.py save   --arm gemma4-26b
  uv run python scripts/z1/snapshot.py list
  uv run python scripts/z1/snapshot.py restore --arm gemma4-26b
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

SNAPDIR = Path("experiments/curation_files/snapshots")
CONTAINER = "ice_postgres"
DB, USER = "ice_db", "ice"

# Tables the store lives in. Everything the seeder or post-flight writes, plus
# the cluster link table; deliberately NOT alembic_version or the runtime's
# operational tables — see the module docstring.
TABLES = [
    "conversations", "episodic_memory", "episodic_chunks",
    "episodic_cluster_links", "context_clusters",
    "codex_entities", "codex_edges", "codex_events", "codex_relation_gaps",
    "procedural_memory", "batch_summaries", "conversation_summaries",
    "memory_slots", "cold_storage",
]


def _docker(cmd: list[str], **kw):
    return subprocess.run(["docker", "exec", CONTAINER] + cmd,
                          capture_output=True, text=True, **kw)


def counts() -> dict:
    sql = " UNION ALL ".join(
        f"SELECT '{t}', count(*) FROM {t}" for t in TABLES)
    r = _docker(["psql", "-U", USER, "-d", DB, "-tAc", sql])
    out = {}
    for line in r.stdout.strip().splitlines():
        if "|" in line:
            k, v = line.split("|", 1)
            out[k] = int(v)
    return out


def save(arm: str) -> int:
    SNAPDIR.mkdir(parents=True, exist_ok=True)
    dest = SNAPDIR / f"{arm}.sql"
    args = ["pg_dump", "-U", USER, "-d", DB, "--data-only", "--no-owner"]
    for t in TABLES:
        args += ["-t", t]
    r = _docker(args)
    if r.returncode != 0:
        print(f"pg_dump failed:\n{r.stderr[:800]}")
        return 1
    dest.write_text(r.stdout)
    c = counts()
    (SNAPDIR / f"{arm}.counts").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n" +
        "\n".join(f"{k}={v}" for k, v in sorted(c.items())))
    live = {k: v for k, v in c.items() if v}
    print(f"saved {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
    print(f"  {live}")
    return 0


def restore(arm: str) -> int:
    src = SNAPDIR / f"{arm}.sql"
    if not src.exists():
        print(f"no snapshot at {src}")
        return 1
    # Truncate ONLY the store tables, then reload. CASCADE is required because
    # the tables reference each other, and RESTART IDENTITY keeps nothing behind.
    tr = _docker(["psql", "-U", USER, "-d", DB, "-c",
                  f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE;"])
    if tr.returncode != 0:
        print(f"truncate failed:\n{tr.stderr[:600]}")
        return 1
    p = subprocess.run(["docker", "exec", "-i", CONTAINER,
                        "psql", "-U", USER, "-d", DB, "-v", "ON_ERROR_STOP=1"],
                       input=src.read_text(), capture_output=True, text=True)
    if p.returncode != 0:
        print(f"restore failed:\n{p.stderr[:800]}")
        return 1
    live = {k: v for k, v in counts().items() if v}
    print(f"restored {arm}\n  {live}")
    expect = SNAPDIR / f"{arm}.counts"
    if expect.exists():
        want = dict(l.split("=", 1) for l in expect.read_text().splitlines() if "=" in l)
        bad = [k for k, v in want.items() if str(counts().get(k, 0)) != v]
        print("  ⚠ mismatch vs saved counts: " + ", ".join(bad) if bad
              else "  counts match the snapshot exactly")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["save", "restore", "list", "counts"])
    ap.add_argument("--arm", default=None)
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
    return save(args.arm) if args.action == "save" else restore(args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
