"""Is the procedural leg's RANKING constant, or is the budgeter keeping only 1?

Runs the leg's own SQL (no budget, no filters) for several very different
queries and prints the top-5 by cosine. If the order barely moves, the leg is
not discriminating; if it moves, the constant fragment is a budget artifact.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
from sqlalchemy import text, bindparam
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.embedder import get_embedder
from src.retrieval.orchestrator import PgVector

db = SessionLocal()
emb = get_embedder()

n_total = db.execute(text("select count(*) from procedural_memory")).scalar()
n_live = db.execute(text(
    "select count(*) from procedural_memory where embedding is not null "
    "and is_active = true and confidence_score >= :c"),
    {"c": settings.procedural_min_conf}).scalar()
print(f"\nprocedural_memory rows: {n_total} total | {n_live} pass "
      f"(is_active + embedding + confidence >= {settings.procedural_min_conf})")
print(f"retrieval_procedural_limit = {settings.retrieval_procedural_limit}\n")

Q = text("""
    SELECT id, pattern_description, 1 - (embedding <=> :e) as score
    FROM procedural_memory
    WHERE embedding IS NOT NULL AND is_active = true
      AND confidence_score >= :c
    ORDER BY score DESC LIMIT 5
""").bindparams(bindparam("e", type_=PgVector))

QUERIES = [
    "what did I decide about my college admissions",
    "zzzq wumpus glorbnax fleeb",
    "how do I cook rice",
    "what is my father's name",
    "explain gradient descent to me",
]
for q in QUERIES:
    e = emb.encode(q, convert_to_tensor=False).tolist()
    rows = db.execute(Q, {"e": e, "c": settings.procedural_min_conf}).fetchall()
    print(f"--- {q}")
    for i, r in enumerate(rows, 1):
        print(f"    {i}. {r.score:.4f}  {r.pattern_description[:88]}")
    print()

scores = db.execute(text(
    "select confidence_score, is_active, pattern_description "
    "from procedural_memory order by confidence_score desc limit 8")).fetchall()
print("top procedural rows by confidence:")
for s in scores:
    print(f"    conf={s.confidence_score} active={s.is_active} {s.pattern_description[:70]}")
db.close()
