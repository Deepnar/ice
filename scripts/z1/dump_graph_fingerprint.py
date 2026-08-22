#!/usr/bin/env python3
"""Fingerprint the extracted graph so two runs can be compared exactly.

⚑ WHY A FINGERPRINT AND NOT A ROW COUNT. Two runs can produce the same number
of edges from different facts, and counts are what this project has been
comparing. `dir-true-run1` and `dir-true-run2` matched to a single edge and
`dir-false-run1`/`run2` differed by 1.1% — but a count cannot say whether the
1.1% is 78 extra triplets at the end or a different decision on turn 3 that
changed everything after it. **Those two need opposite fixes**, so the compare
below reports the FIRST turn that diverges, in seed order, not just a total.

Output goes to `experiments/curation_files/` (gitignored) because a fingerprint
contains the corpus's own subjects and objects.

    uv run python scripts/z1/dump_graph_fingerprint.py --label A
    uv run python scripts/z1/dump_graph_fingerprint.py --compare A B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text  # noqa: E402

from src.api.db import SessionLocal  # noqa: E402

OUT = Path("experiments/curation_files/fingerprints")


def fingerprint(db) -> dict:
    """Every stored triplet, grouped by the turn that produced it.

    Keyed by the turn's seed ordinal (from `idempotency_key`, `z1seed-<slug>-<n>`)
    rather than by batch_id, because batch ids are fresh uuids on every run and
    would make two runs incomparable by construction — the same two-id-space
    confusion that made codex fragments unscoreable for months (TRAPS #33).
    """
    rows = db.execute(text("""
        SELECT em.idempotency_key AS key,
               s.canonical_name  AS subj,
               e.relation        AS rel,
               t.canonical_name  AS obj,
               e.negated         AS neg
        FROM codex_edges e
        JOIN codex_entities s ON s.id = e.source_id
        JOIN codex_entities t ON t.id = e.target_id
        JOIN episodic_memory em ON em.batch_id = e.source_batch
        WHERE em.idempotency_key LIKE 'z1seed-%'
    """)).fetchall()
    by_turn: dict[str, list] = {}
    for r in rows:
        by_turn.setdefault(r.key, []).append(
            [r.subj, r.rel, r.obj, bool(r.neg)])
    for k in by_turn:
        by_turn[k].sort()
    ents = db.execute(text(
        "SELECT count(*), count(DISTINCT canonical_name) FROM codex_entities")).first()
    return {
        "turns": len(by_turn),
        "edges": sum(len(v) for v in by_turn.values()),
        "entities": ents[0],
        "distinct_entity_names": ents[1],
        "by_turn": by_turn,
    }


def _ordinal(key: str) -> tuple:
    try:
        _, slug, n = key.split("-", 2)
        return (slug, int(n))
    except Exception:                                        # noqa: BLE001
        return (key, 0)


def compare(a: dict, b: dict) -> int:
    print("\n" + "=" * 66)
    print("REPRODUCIBILITY — two identical runs")
    print("=" * 66)
    same_totals = True
    for k in ("turns", "edges", "entities", "distinct_entity_names"):
        av, bv = a[k], b[k]
        flag = "" if av == bv else "   <-- DIFFERS"
        if av != bv:
            same_totals = False
        print(f"  {k:22s} A={av:<8} B={bv:<8}{flag}")

    keys = sorted(set(a["by_turn"]) | set(b["by_turn"]), key=_ordinal)
    diverged = [k for k in keys
                if a["by_turn"].get(k) != b["by_turn"].get(k)]

    print(f"\n  turns compared          {len(keys)}")
    print(f"  turns that DIFFER       {len(diverged)}")

    if not diverged:
        print("\n  ⇒ IDENTICAL. Extraction reproduced exactly, turn for turn.")
        print("    Comparisons built on this configuration are readable.")
        return 0

    print(f"\n  ⇒ NOT REPRODUCIBLE. {len(diverged)} of {len(keys)} turns differ.")
    first = diverged[0]
    pos = keys.index(first)
    print(f"\n  FIRST divergence at turn {pos + 1} of {len(keys)}  ({first})")
    print("  — if this is turn 1, suspect process/model warm-up.")
    print("  — if it is late and everything after also differs, suspect a")
    print("    cascade: canonicalisation feeds accepted relations back into")
    print("    the vocabulary, so one changed decision alters every later one.")
    print(f"  — turns after the first divergence that also differ: "
          f"{sum(1 for k in keys[pos:] if k in set(diverged))} of {len(keys) - pos}")

    sa = {tuple(x) for x in a['by_turn'].get(first, [])}
    sb = {tuple(x) for x in b['by_turn'].get(first, [])}
    print(f"\n  at that turn:  A={len(sa)} triplets  B={len(sb)} triplets")
    for lbl, only in (("only in A", sa - sb), ("only in B", sb - sa)):
        for t in sorted(only)[:4]:
            print(f"    {lbl}: {t[0]} --{t[1]}--> {t[2]}{' [negated]' if t[3] else ''}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="dump the current graph under this label")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.label:
        db = SessionLocal()
        fp = fingerprint(db)
        db.close()
        p = OUT / f"{args.label}.json"
        p.write_text(json.dumps(fp))
        print(f"fingerprint[{args.label}]: {fp['turns']} turns, {fp['edges']} edges, "
              f"{fp['entities']} entities -> {p}")
        return 0

    if args.compare:
        a, b = (json.loads((OUT / f"{x}.json").read_text()) for x in args.compare)
        return compare(a, b)

    ap.error("need --label or --compare")


if __name__ == "__main__":
    raise SystemExit(main())
