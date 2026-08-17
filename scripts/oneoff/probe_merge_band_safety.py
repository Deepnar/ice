"""Does cosine magnitude track merge SAFETY? And are there cheaper signals?

The proposal: process high-cosine pairs first and auto-accept the LLM's verdict
there, because vectorially-identical pairs are the safe ones. This tests that
assumption on the live arm-1 store, and measures three signals the embedding
does not have.
"""
import os, sys, itertools, re
sys.path.insert(0, os.path.abspath("."))
from sqlalchemy import text
from src.api.db import SessionLocal

db = SessionLocal()

def cos(a, b):
    r = db.execute(text("""
        SELECT 1 - (a.embedding <=> b.embedding) AS c
        FROM codex_entities a, codex_entities b
        WHERE a.canonical_name = :a AND b.canonical_name = :b
    """), {"a": a, "b": b}).fetchone()
    return r.c if r else None

def neighbours(name):
    return {str(x[0]) for x in db.execute(text("""
        SELECT CASE WHEN e.source_id = me.id THEN e.target_id ELSE e.source_id END
        FROM codex_entities me JOIN codex_edges e
          ON e.source_id = me.id OR e.target_id = me.id
        WHERE me.canonical_name = :n
    """), {"n": name})}

def batches(name):
    return {str(x[0]) for x in db.execute(text("""
        SELECT DISTINCT e.source_batch
        FROM codex_entities me JOIN codex_edges e
          ON e.source_id = me.id OR e.target_id = me.id
        WHERE me.canonical_name = :n AND e.source_batch IS NOT NULL
    """), {"n": name})}

NUM = re.compile(r"\d+")
def digits(s): return sorted(NUM.findall(s))

# Find real candidate pairs above 0.90 on this store, high band first.
rows = db.execute(text("""
    SELECT a.canonical_name AS na, b.canonical_name AS nb,
           1 - (a.embedding <=> b.embedding) AS c
    FROM codex_entities a JOIN codex_entities b ON a.id < b.id
    WHERE a.embedding IS NOT NULL AND b.embedding IS NOT NULL
      AND (a.properties->>'merged_into') IS NULL
      AND (b.properties->>'merged_into') IS NULL
      AND 1 - (a.embedding <=> b.embedding) >= 0.95
    ORDER BY c DESC LIMIT 60
""")).fetchall()

print(f"\n{'='*94}")
print("TOP COSINE BAND (>= 0.95) ON ARM 1 — is the top of the band the SAFE end?")
print(f"{'='*94}")
print(f"{'cos':<8}{'digits differ':<15}{'co-occur':<10}{'nbr Jacc':<10}  pair")
n_digit_clash = n_cooccur = 0
for r in rows[:40]:
    da, db_ = digits(r.na), digits(r.nb)
    clash = "⚠ YES" if da != db_ else "no"
    if da != db_: n_digit_clash += 1
    ba, bb = batches(r.na), batches(r.nb)
    co = "⚠ YES" if (ba & bb) else "no"
    if ba & bb: n_cooccur += 1
    na_, nb_ = neighbours(r.na), neighbours(r.nb)
    j = len(na_ & nb_) / len(na_ | nb_) if (na_ | nb_) else 0.0
    print(f"{r.c:<8.4f}{clash:<15}{co:<10}{j:<10.3f}  {r.na[:34]!r} | {r.nb[:34]!r}")
print(f"\n  of the top 40: {n_digit_clash} have DIFFERENT digits, {n_cooccur} co-occur in a turn")
db.close()
