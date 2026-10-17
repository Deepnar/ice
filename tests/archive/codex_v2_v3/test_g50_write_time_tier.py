#!/usr/bin/env python3
"""G50: the write-time merge_key tier in `get_or_create_entity`.

Live DB. Inserts its own rows and deletes them — NEVER truncates (TRAPS #6:
the dev store holds real data, and residue fails a DIFFERENT suite later).

Run: uv run python tests/test_g50_write_time_tier.py
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text  # noqa: E402

from src.api.db import SessionLocal  # noqa: E402
from src.memory.models import CodexEdge, CodexEntity  # noqa: E402
from src.workers.codex_extractor import get_or_create_entity  # noqa: E402
from src.workers.maintenance_agent import merge_key  # noqa: E402

MARK = f"g50tier{uuid.uuid4().hex[:8]}"
passed = failed = 0
created: list = []


def check(label: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def mk(db, name):
    e = get_or_create_entity(db, f"{name} {MARK}")
    created.append(e.id)
    return e


def main() -> int:
    db = SessionLocal()
    try:
        # 1 ── the tier resolves a normalisation-equal name to the same row
        a = mk(db, "gemma-4-e4b q4")
        db.flush()
        b = mk(db, "gemma-4-e4b-q4")
        check("normalisation-equal names resolve to ONE entity",
              a.id == b.id, f"{a.id} vs {b.id}")

        # 2 ── the surface form is recorded, so the exact tiers catch it next time
        check("the second surface form is stored as an alias",
              any("gemma-4-e4b-q4" in al for al in (b.aliases or [])),
              str(b.aliases))

        # 3 ── every new entity is stamped, or the tier is invisible to it
        check("a newly created entity carries properties.merge_key",
              (a.properties or {}).get("merge_key") == merge_key(a.canonical_name),
              str((a.properties or {}).get("merge_key")))

        # 4 ── ⚑ a reject-kind pair must NOT merge. Quantities differing by one
        # digit are the failure this whole item exists to avoid, and embeddings
        # score them 0.9388 — merge_key must keep them apart.
        g8 = mk(db, "8 gb")
        db.flush()
        g4 = mk(db, "4 gb")
        check("`8 gb` and `4 gb` stay SEPARATE entities",
              g8.id != g4.id, f"both resolved to {g8.id}")

        # 5 ── same for a converse: the entity-side twin of TRAPS #26
        c1 = mk(db, "shiva without brahma")
        db.flush()
        c2 = mk(db, "brahma without shiva")
        check("a converse pair stays SEPARATE (TRAPS #26 entity twin)",
              c1.id != c2.id, f"both resolved to {c1.id}")

        # 6 ── ⚑ protect_ids: an entity the caller is still holding is never
        # folded away. Resolving the object used to promote the SUBJECT away
        # mid-triplet, and the caller then wrote an edge at a deleted row —
        # 1 turn in 6 on a smoke seed (2026-08-15), costing that turn its codex
        # AND procedural extraction.
        held = mk(db, "plan alpha")
        db.flush()
        still = db.query(CodexEntity).get(held.id)
        other = get_or_create_entity(db, f"plan-alpha {MARK}", protect_ids={held.id})
        created.append(other.id)
        db.flush()
        check("a held entity survives resolution of a colliding name",
              db.query(CodexEntity).get(held.id) is not None,
              "the protected row was deleted")
        check("...and no edge could dangle (both ids resolvable)",
              still is not None and other is not None)

        # 7 ── the tier must not resurrect an entity already merged away
        m1 = mk(db, "merged away one")
        db.flush()
        m1.properties = {**(m1.properties or {}), "merged_into": str(uuid.uuid4())}
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(m1, "properties")
        db.flush()
        m2 = get_or_create_entity(db, f"merged-away-one {MARK}")
        created.append(m2.id)
        check("an entity marked merged_into is NOT matched by the tier",
              m2.id != m1.id, "the tier resolved to a merged-away row")

        db.commit()
    finally:
        # Clean up only what this run recorded, by id (TRAPS #6c: a cleanup
        # scoped to ids it never recorded cannot catch its own leaks).
        ids = [i for i in created if i]
        if ids:
            db.query(CodexEdge).filter(
                (CodexEdge.source_id.in_(ids)) | (CodexEdge.target_id.in_(ids))
            ).delete(synchronize_session=False)
            db.query(CodexEntity).filter(CodexEntity.id.in_(ids)).delete(
                synchronize_session=False)
            db.commit()
        left = db.execute(text(
            "SELECT count(*) FROM codex_entities WHERE canonical_name LIKE :m"),
            {"m": f"%{MARK}%"}).scalar()
        print(f"\n  cleanup: {len(ids)} ids removed, {left} rows remaining")
        db.close()

    print(f"\n{passed} passed, {failed} failed  (marker {MARK})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
