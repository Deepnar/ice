"""A FAIR sample of what codex extraction actually produced, for reading by eye.

⚑ STRATIFIED, SEEDED, NOT CHERRY-PICKED. Sampling has failed twice in this
project by concentrating on one conversation. This draws evenly across the three
conversations AND across the grounding tiers, with a fixed seed, so the same
draw is reproducible and so arm A and arm B can be read side by side.

No model, no writes. The eyeball pass is what catches what a metric misses.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sqlalchemy import text                      # noqa: E402
from src.api.db import SessionLocal              # noqa: E402

Q = """
-- ⚠ An earlier version carried `left join conversations c on true` for a
-- column the output never used. `on true` is a CROSS JOIN: every edge came
-- back once per conversation, so a 12-row sample rendered as 4 rows repeated
-- 3x and the sample looked like the extractor was emitting duplicates.
-- The join is gone; the lesson is that an unused column is not free.
select s.canonical_name as subj, e.relation as rel, t.canonical_name as obj,
       e.extraction_confidence as conf
from codex_edges e
join codex_entities s on s.id = e.source_id
join codex_entities t on t.id = e.target_id
where e.valid_until is null
  and e.extraction_confidence = :conf
order by md5(e.id::text || :seed)
limit :n
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="(live)")
    ap.add_argument("--per-tier", type=int, default=12)
    ap.add_argument("--seed", default="20260820")
    args = ap.parse_args()
    db = SessionLocal()
    print(f"=== CODEX SAMPLE — {args.arm} (seed {args.seed}) ===")
    for conf, label in ((0.9, "GROUNDED (conf 0.9)"),
                        (0.35, "REJECTED (conf 0.35 — stored, low-trust)")):
        rows = db.execute(text(Q), {"conf": conf, "n": args.per_tier,
                                    "seed": args.seed}).fetchall()
        print(f"\n--- {label} — {args.per_tier} random ---")
        for r in rows:
            print(f"  {r.subj[:38]:<38} --{r.rel[:24]:<24}--> {r.obj[:38]}")
    # relation vocabulary health
    print("\n--- TOP 15 RELATIONS ---")
    for rel, n in db.execute(text(
            "select relation, count(*) from codex_edges where valid_until is null "
            "group by 1 order by 2 desc limit 15")).fetchall():
        print(f"  {n:>5}  {rel}")
    print("\n--- LONGEST RELATION NAMES (clause-shaped = G49 demotion candidates) ---")
    for rel, n in db.execute(text(
            "select relation, count(*) from codex_edges where valid_until is null "
            "group by 1 order by length(relation) desc limit 8")).fetchall():
        print(f"  {n:>5}  {rel[:80]}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
