"""Graph SHAPE per arm: how much is a single triplet, how much actually fans out.

The counts in the arm tables say how many entities and edges exist. They do not
say whether the result is a GRAPH or a pile of disconnected two-node facts —
which is [G51](../../docs/ROADMAP.md#g51)'s whole subject, and the thing a
higher edges/entity ratio is supposed to indicate.

Pure SQL, no model, no writes. Run against whichever arm is restored.
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sqlalchemy import text                       # noqa: E402
from src.api.db import SessionLocal               # noqa: E402

Q_DEGREE = """
with deg as (
  select e.id, e.canonical_name,
         (select count(*) from codex_edges x where x.source_id = e.id and x.valid_until is null) as outd,
         (select count(*) from codex_edges x where x.target_id = e.id and x.valid_until is null) as ind
  from codex_entities e
  where e.context_payload not like '[merged into %%' or e.context_payload is null
)
select * from deg
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="(live store)")
    args = ap.parse_args()
    db = SessionLocal()
    rows = db.execute(text(Q_DEGREE)).fetchall()
    n = len(rows)
    deg = [(r.outd + r.ind) for r in rows]
    outd = [r.outd for r in rows]

    iso = sum(1 for d in deg if d == 0)
    leaf = sum(1 for d in deg if d == 1)
    two = sum(1 for d in deg if d == 2)
    hub = sum(1 for d in deg if d >= 10)
    tot_edges = db.execute(text(
        "select count(*) from codex_edges where valid_until is null")).scalar()

    print(f"=== GRAPH SHAPE — {args.arm} ===")
    print(f"entities (non-husk) {n:,} · live edges {tot_edges:,} · "
          f"edges/entity {tot_edges/n:.2f}")
    print()
    print("DEGREE (in+out), the 'is it a graph or a pile of pairs' question:")
    for lab, v in (("isolated  (deg 0)", iso), ("SINGLE-EDGE (deg 1)", leaf),
                   ("deg 2", two), ("hub (deg >= 10)", hub)):
        print(f"  {lab:<22} {v:>7,}  ({100*v/n:5.1f}%)")
    print(f"  ⇒ entities that are a SINGLE TRIPLET and nothing more: "
          f"{leaf:,} ({100*leaf/n:.1f}%)")
    print()
    srt = sorted(deg, reverse=True)
    print(f"degree: mean {sum(deg)/n:.2f} · median {srt[n//2]} · "
          f"p90 {srt[int(n*0.10)]} · max {srt[0]}")
    fan = [o for o in outd if o > 0]
    if fan:
        fs = sorted(fan, reverse=True)
        print(f"FAN-OUT (out-degree, entities with >=1 outgoing): "
              f"n {len(fan):,} · mean {sum(fan)/len(fan):.2f} · "
              f"median {fs[len(fs)//2]} · max {fs[0]}")
    print()
    print("TOP 12 HUBS:")
    for r in sorted(rows, key=lambda r: -(r.outd + r.ind))[:12]:
        print(f"  {r.outd + r.ind:>4}  (out {r.outd:>3} / in {r.ind:>3})  {r.canonical_name[:58]}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
