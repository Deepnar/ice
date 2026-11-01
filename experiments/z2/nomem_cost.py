"""R10's other side (Z2 F4): how often the reranker rescue lets memory into a message that needs none.

R10 read 146 LongMemEval questions that all need memory. This reads 146 of the maintainer's own
messages labeled "needs no memory" (data/labeled/labeled_prompts.jsonl, source personal, Zero_Shot),
each in a fresh chat against a copy of B's store (`--source`, cloned once into `--database`; the
copy takes the fresh chats' rows, the original is never written). Messages stored word for word in
that store are left out — they would find themselves. The rescue reads with its bar off and records
its best score, so the report gives every bar from one read, beside R10's needed-memory questions.

    uv run --no-sync python experiments/z2/nomem_cost.py run --database ice_z2_nomem_b --out logs/z2-run-main/nomem-cost
    uv run --no-sync python experiments/z2/nomem_cost.py report --out logs/z2-run-main/nomem-cost
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LABELED = ROOT / "data/labeled/labeled_prompts.jsonl"
R10_UNITS = ROOT / "logs/z2-run-main/s12-r10/units/full_rescue_reranker"
BARS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
N, SEED = 146, 0                       # as many as R10 read


def candidates(path=LABELED):
    """The maintainer's messages labeled as needing no memory, in a fixed shuffled order."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        label = d.get("label") or {}
        if d.get("source") == "personal" and label.get("context_reliance") == "Zero_Shot" and d["prompt"].strip():
            rows.append(dict(id=d["id"], text=d["prompt"].strip()))
    rows.sort(key=lambda r: r["id"])
    random.Random(SEED).shuffle(rows)
    return rows


def stored_verbatim(db, text) -> bool:
    from sqlalchemy import text as sql
    return bool(db.execute(sql("SELECT 1 FROM episodic_memory WHERE strpos(raw_text, :t) > 0 LIMIT 1"),
                           dict(t=text[:200])).first())


def clone(source, database):
    """The copy the reads write their fresh chats into, made once from the source store."""
    from sqlalchemy import create_engine, text as sql
    from sqlalchemy.engine import make_url
    from src.api.config import settings
    for name in (source, database):
        if not re.fullmatch(r"ice_z2_[a-z0-9_]+", name):
            raise ValueError(f"not a Z2 database name: {name}")
    if source == database:
        raise ValueError("the reads write a fresh chat per message: read a copy, never the source")
    engine = create_engine(make_url(settings.database_url).set(database="postgres"), isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        if not conn.execute(sql("SELECT 1 FROM pg_database WHERE datname = :d"), dict(d=database)).first():
            conn.execute(sql(f'CREATE DATABASE "{database}" TEMPLATE "{source}"'))
    engine.dispose()


def unit_path(out, mid):
    return Path(out) / "units" / f"{mid}.json"


def run(out, limit=N):
    from experiments.z2 import oracle as O
    from experiments.z2.read_passes import Reader
    from src.api.db import SessionLocal
    from src.memory.models import Conversation, EpisodicMemory
    from sqlalchemy import func
    out = Path(out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    sample_path = out / "sample.json"
    with SessionLocal() as db:
        if sample_path.exists():
            sample = json.loads(sample_path.read_text())
        else:
            sample, left_out = [], 0
            for row in candidates():
                if len(sample) == N:
                    break
                if stored_verbatim(db, row["text"]):
                    left_out += 1
                    continue
                sample.append(row)
            sample_path.write_text(json.dumps(dict(seed=SEED, left_out_stored=left_out, messages=sample)) + "\n")
            sample = json.loads(sample_path.read_text())
        latest = db.query(func.max(EpisodicMemory.timestamp)).scalar()
    now = latest.replace(microsecond=0) + timedelta(hours=1)     # the user writes after the history
    reader = None
    failures = []
    for row in sample["messages"][:limit]:
        path = unit_path(out, row["id"])
        if path.exists():
            continue
        started = time.time()
        try:
            reader = reader or Reader()
            with SessionLocal() as db:
                chat = Conversation(id=uuid.uuid4(), memory_scope_type="auto", created_at=now)
                db.add(chat)
                db.commit()
                cid = str(chat.id)
            ask = dict(id=row["id"], type="no_memory", question=row["text"], answer=None, abstention=False)
            with SessionLocal() as db:
                stage = reader.read(db, cid, ask, "full", now, overrides=O.RESCUE_CONDITION[1])
        except Exception as exc:                     # this message only; the next run retries it
            failures.append(dict(id=row["id"], reason=f"{type(exc).__name__}: {exc}"[:500]))
            continue
        path.write_text(json.dumps(dict(id=row["id"], gate=stage.get("gate"),
                                        selected_tokens=stage.get("selected_tokens"),
                                        prompt_tokens=stage.get("prompt_tokens"),
                                        selected=len(stage.get("selected") or []),
                                        prompt_messages=stage.get("prompt_messages"),
                                        seconds=round(time.time() - started, 1)), default=str) + "\n")
        print(json.dumps(dict(message=row["id"], seconds=round(time.time() - started, 1),
                              b2_retrieve=stage["gate"].get("b2_retrieve"),
                              best=stage["gate"].get("rescue_best_score"))), flush=True)
    (out / "failures.json").write_text(json.dumps(failures, indent=1) + "\n")
    return failures


def progress(out, total=N):
    """(messages read, total) for the runner's meter, from files only."""
    return len(list((Path(out) / "units").glob("*.json"))), total


def report(out, r10_units=R10_UNITS):
    units = [json.loads(p.read_text()) for p in sorted((Path(out) / "units").glob("*.json"))]
    sample = json.loads((Path(out) / "sample.json").read_text())
    refused = [u for u in units if u["gate"].get("b2_retrieve") is False]
    gate_yes = len(units) - len(refused)
    scores = [u["gate"].get("rescue_best_score") for u in refused]
    nothing = sum(s is None for s in scores)
    needed = [json.loads(p.read_text())["stage"]["gate"].get("rescue_best_score") for p in sorted(Path(r10_units).glob("*.json"))]
    print(f"{len(units)} no-memory messages read (left out, stored word for word: {sample['left_out_stored']})")
    print(f"   the main gate already fetched memory on {gate_yes} (not the double-check's doing)")
    print(f"   the main gate said no on {len(refused)} — these are the ones the double-check sees")
    print(f"   nothing to score (the search found nothing) on {nothing}")
    print("at each bar, memory let in:")
    print(f"   {'bar':>5}   needed it (R10, of {len(needed)})   did not need it (of {len(refused)})")
    for bar in BARS:
        yes = sum(s is not None and s >= bar for s in needed)
        no = sum(s is not None and s >= bar for s in scores)
        print(f"   {bar:+5.0f}   {yes:4} ({100 * yes / max(len(needed), 1):3.0f}%)"
              f"                {no:4} ({100 * no / max(len(refused), 1):3.0f}%)")
    let_in = [u["selected_tokens"] for u in refused if u.get("selected_tokens")]
    if let_in:
        print(f"memory added when let in: median {statistics.median(let_in):.0f} tokens")
    secs = [u["seconds"] for u in refused if u.get("seconds")]
    if secs:
        print(f"seconds per message with the double-check (the whole read): median {statistics.median(secs):.1f}")
    return 0


def main() -> int:
    from scripts.z1.private_logs import under_logs
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--database", required=True)
    r.add_argument("--source", default="ice_z2_main_b")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int, default=N)
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not under_logs(args.out):
        raise ValueError("the messages are private: keep the output under logs/")
    if args.cmd == "report":
        return report(args.out)
    if os.environ.get("ICE_TEST_DATABASE") != args.database:
        from sqlalchemy.engine import make_url
        from src.api.config import settings as live
        clone(args.source, args.database)
        url = make_url(live.database_url).set(database=args.database).render_as_string(hide_password=False)
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  dict(os.environ, DATABASE_URL=url, ICE_TEST_DATABASE=args.database))
    failures = run(args.out, limit=args.limit)
    print(json.dumps(dict(failures=len(failures))))
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
