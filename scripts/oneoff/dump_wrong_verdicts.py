#!/usr/bin/env python3
"""Dump the `wrong` verdicts of one judged arm WITH their source turn.

Why this exists: `judge_codex.py` writes `edge_id` into each verdict but not the
source turn, so a judgement artifact cannot be reviewed by a human afterwards —
you cannot tell whether the judge was right without seeing what it judged
against. This restores that join.

⚑ The arm's store must be RESTORED before running this. Edge ids are per-store.

  uv run python scripts/z1/snapshot.py restore --arm dir-false-run1
  uv run python scripts/oneoff/dump_wrong_verdicts.py --arm dir-false-run1-perturn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text                        # noqa: E402

from src.api.db import SessionLocal                # noqa: E402

JUD = Path("experiments/curation_files/judgements")
OUT = Path("experiments/curation_files/wrong_review")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--all-labels", action="store_true",
                    help="keep every verdict, not just `wrong`. Needed to review "
                         "a TURN as a whole — you cannot tell a mislabelled "
                         "`reversed` from a `wrong` without seeing its siblings.")
    args = ap.parse_args()

    verdicts = json.load(open(JUD / f"codex_quality_{args.arm}.json"))["verdicts"]
    wrong = verdicts if args.all_labels else [v for v in verdicts if v["label"] == "wrong"]
    ids = [v["edge_id"] for v in wrong]

    db = SessionLocal()
    rows = db.execute(text("""
        select e.id::text as eid, m.id::text as turn_id, m.raw_text as source
        from codex_edges e
        join episodic_memory m on m.batch_id = e.source_batch
        where e.id::text = any(:ids)
    """), {"ids": ids}).fetchall()
    db.close()

    src = {r.eid: {"turn_id": r.turn_id, "source": r.source} for r in rows}
    missing = [i for i in ids if i not in src]
    for v in wrong:
        v.update(src.get(v["edge_id"], {"turn_id": None, "source": None}))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"{args.arm}.all.json" if args.all_labels else f"{args.arm}.wrong.json")
    p.write_text(json.dumps({
        "arm": args.arm,
        "n_wrong": len(wrong),
        "n_turns": len({v["turn_id"] for v in wrong if v["turn_id"]}),
        "n_unresolved": len(missing),
        "wrong": wrong,
    }, ensure_ascii=False, indent=1))
    print(f"{len(wrong)} wrong · {len({v['turn_id'] for v in wrong if v['turn_id']})} turns "
          f"· {len(missing)} unresolved → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
