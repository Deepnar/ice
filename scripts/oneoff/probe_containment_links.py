"""Of the containment pairs, which are MERGES and which are missing LINKS?

A short entity name appearing inside a longer one is either the same thing said
two ways (merge) or a hypernym of it (an edge that should exist and does not).
Both connect the graph; only one of them is a merge.
"""
import os, sys, re
sys.path.insert(0, os.path.abspath("."))
from collections import defaultdict
from sqlalchemy import text
from src.api.db import SessionLocal

db = SessionLocal()
rows = db.execute(text("""
    SELECT e.id, e.canonical_name AS n, count(g.id) AS d
    FROM codex_entities e
    LEFT JOIN codex_edges g ON (g.source_id = e.id OR g.target_id = e.id)
                            AND g.valid_until IS NULL
    WHERE (e.properties->>'merged_into') IS NULL
    GROUP BY e.id, e.canonical_name
""")).fetchall()
by_name = {r.n.lower().strip(): (str(r.id), r.d) for r in rows}
def ntok(s): return len([t for t in re.split(r"\s+", s.strip()) if t])

pairs = []
for short, (sid, sdeg) in by_name.items():
    if not (1 <= ntok(short) <= 2) or len(short) < 4:
        continue
    for long, (lid, ldeg) in by_name.items():
        if long == short or ntok(long) > 5:
            continue
        if long.startswith(short + " ") or long.endswith(" " + short):
            pairs.append((short, sdeg, long, ldeg))

print(f"\n{'='*80}")
print(f"CONTAINMENT PAIRS (short 1-2 words inside a <=5 word name): {len(pairs)}")
print(f"{'='*80}")
# Does an edge ALREADY join them? If not, the graph is missing the link.
linked = 0
for short, sdeg, long, ldeg in pairs:
    sid, lid = by_name[short][0], by_name[long][0]
    e = db.execute(text("""
        SELECT 1 FROM codex_edges
        WHERE valid_until IS NULL
          AND ((source_id = :a AND target_id = :b) OR (source_id = :b AND target_id = :a))
        LIMIT 1"""), {"a": sid, "b": lid}).fetchone()
    if e:
        linked += 1
print(f"  already joined by a live edge: {linked} ({linked/max(len(pairs),1)*100:.1f}%)")
print(f"  NOT joined — the graph does not connect them: {len(pairs)-linked}")
print(f"\n  sample (deg = live edges on each side):")
for short, sdeg, long, ldeg in pairs[:26]:
    print(f"    {short!r} (deg {sdeg})  ⊂  {long!r} (deg {ldeg})")
db.close()
