"""What ARE the dead ends? Junk, genuine singletons, or missed duplicates?

The roadmap concluded "fan-out is an extraction property, not a resolution
failure" from cosine nearest-neighbour distributions. Cosine has just been shown
to be a poor proxy for identity, so this re-asks the question with instruments
that do not use the embedding at all.
"""
import os, sys, re
sys.path.insert(0, os.path.abspath("."))
from collections import Counter, defaultdict
from sqlalchemy import text
from src.api.db import SessionLocal
from src.workers.maintenance_agent import merge_key

db = SessionLocal()

deg = {str(r.id): r.d for r in db.execute(text("""
    SELECT e.id, count(g.id) AS d
    FROM codex_entities e
    LEFT JOIN codex_edges g ON (g.source_id = e.id OR g.target_id = e.id)
                            AND g.valid_until IS NULL
    WHERE (e.properties->>'merged_into') IS NULL
    GROUP BY e.id
"""))}
names = {str(r.id): (r.canonical_name, r.entity_type)
         for r in db.execute(text(
             "SELECT id, canonical_name, entity_type FROM codex_entities "
             "WHERE (properties->>'merged_into') IS NULL"))}

dist = Counter(min(d, 6) for d in deg.values())
print(f"\n{'='*84}\nDEGREE DISTRIBUTION  (live edges only, {len(deg)} entities)\n{'='*84}")
for k in sorted(dist):
    lbl = f"{k}+" if k == 6 else str(k)
    print(f"  degree {lbl:<3} {dist[k]:>6}  ({dist[k]/len(deg)*100:5.1f}%)")

dead = [i for i, d in deg.items() if d <= 1]
print(f"\n  dead ends (degree 0 or 1): {len(dead)} ({len(dead)/len(deg)*100:.1f}%)")

def ntok(s): return len([t for t in re.split(r"\s+", s.strip()) if t])

print(f"\n{'='*84}\nWHAT DO DEAD-END NAMES LOOK LIKE?  (no embedding involved)\n{'='*84}")
buckets = Counter()
samples = defaultdict(list)
for i in dead:
    n, _t = names[i]
    w = ntok(n)
    b = ("A. 1 word" if w == 1 else "B. 2-3 words" if w <= 3
         else "C. 4-6 words (phrase)" if w <= 6 else "D. 7+ words (CLAUSE FRAGMENT)")
    buckets[b] += 1
    samples[b].append(n)
for b in sorted(buckets):
    print(f"\n  {b:<34} {buckets[b]:>6} ({buckets[b]/len(dead)*100:5.1f}%)")
    for s in samples[b][:5]:
        print(f"        {s[:82]!r}")

# Detection we do NOT currently have: exact merge_key collision at ANY cosine,
# and containment (abbreviation / expansion), neither of which uses embeddings.
by_key = defaultdict(list)
for i, (n, _t) in names.items():
    by_key[merge_key(n)].append(i)
key_groups = {k: v for k, v in by_key.items() if len(v) > 1}
dead_in_key = sum(1 for k, v in key_groups.items() for i in v if i in set(dead))
print(f"\n{'='*84}\nMISSED-DUPLICATE CHANNELS THAT DO NOT USE THE EMBEDDING\n{'='*84}")
print(f"  merge_key groups across the WHOLE store: {len(key_groups)} "
      f"({sum(len(v) for v in key_groups.values())} entities, "
      f"{dead_in_key} of them dead ends)")

lowered = defaultdict(list)
for i, (n, _t) in names.items():
    lowered[n.lower().strip()].append(i)
short = {n: i for n, i in lowered.items() if 1 <= ntok(n) <= 3}
contained = 0
contain_samples = []
for n, ids in short.items():
    if len(n) < 4:
        continue
    hits = [m for m in lowered
            if m != n and (f" {n} " in f" {m} " or m.startswith(n + " ") or m.endswith(" " + n))]
    if hits:
        contained += 1
        if len(contain_samples) < 8:
            contain_samples.append((n, hits[:2]))
print(f"  short names appearing INSIDE another entity name: {contained}")
for n, h in contain_samples:
    print(f"        {n!r}  ⊂  {[x[:56] for x in h]}")
db.close()
