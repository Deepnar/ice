"""How much of the 1,094-pair backlog is actually mergeable?

Buckets every pair above cosine 0.90 by the KIND of difference between the two
names, which is what the top-40 sample says predicts safety — not the cosine.
"""
import os, sys, re
sys.path.insert(0, os.path.abspath("."))
from collections import Counter
from sqlalchemy import text
from src.api.db import SessionLocal
from src.workers.maintenance_agent import merge_key

db = SessionLocal()
rows = db.execute(text("""
    SELECT a.canonical_name AS na, b.canonical_name AS nb,
           1 - (a.embedding <=> b.embedding) AS c
    FROM codex_entities a JOIN codex_entities b ON a.id < b.id
    WHERE a.embedding IS NOT NULL AND b.embedding IS NOT NULL
      AND (a.properties->>'merged_into') IS NULL
      AND (b.properties->>'merged_into') IS NULL
      AND 1 - (a.embedding <=> b.embedding) >= 0.90
""")).fetchall()

NUM = re.compile(r"\d+")
WORDNUM = {"one","two","three","four","five","six","seven","eight","nine","ten",
           "first","second","third","fourth","fifth"}
GENDER = {"his","her","he","she","him","hers","himself","herself"}
TENSE = {"is","was","are","were","has","had","have","will","did","does"}

def toks(s): return set(re.split(r"[\s\-_:;,()\[\]{}\"'/\\.]+", s.lower())) - {""}

buckets = Counter()
examples = {}
for r in rows:
    ka, kb = merge_key(r.na), merge_key(r.nb)
    ta, tb = toks(r.na), toks(r.nb)
    diff = ta ^ tb
    if ka == kb:
        b = "1. merge_key EQUAL (deterministic, no judgement)"
    elif sorted(NUM.findall(r.na)) != sorted(NUM.findall(r.nb)):
        b = "2. DIGITS differ"
    elif diff & WORDNUM:
        b = "3. spelled-out NUMBER differs"
    elif diff & GENDER:
        b = "4. GENDER word differs"
    elif diff & TENSE:
        b = "5. TENSE word differs"
    elif ta == tb:
        b = "6. same tokens, different order (CONVERSE risk)"
    elif ta < tb or tb < ta:
        b = "7. one is a strict token SUPERSET (qualifier added)"
    else:
        b = "8. other token substitution"
    buckets[b] += 1
    examples.setdefault(b, []).append((r.c, r.na, r.nb))

print(f"\n{'='*88}")
print(f"ALL {len(rows)} PAIRS ABOVE COSINE 0.90, BUCKETED BY KIND OF DIFFERENCE")
print(f"{'='*88}")
for b in sorted(buckets):
    n = buckets[b]
    print(f"\n  {b:<52} {n:>5}  ({n/len(rows)*100:.1f}%)")
    for c, na, nb in sorted(examples[b], reverse=True)[:3]:
        print(f"        {c:.4f}  {na[:36]!r} | {nb[:36]!r}")
mergeable = buckets["1. merge_key EQUAL (deterministic, no judgement)"]
print(f"\n  ⇒ deterministically safe: {mergeable} of {len(rows)} pairs "
      f"({mergeable/len(rows)*100:.1f}%)")
print(f"  ⇒ everything else carries a token that CHANGES THE REFERENT: "
      f"{len(rows)-mergeable} ({(len(rows)-mergeable)/len(rows)*100:.1f}%)")
db.close()
