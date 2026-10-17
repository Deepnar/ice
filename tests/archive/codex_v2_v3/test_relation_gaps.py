"""G32/a1 — constrained decoding, relation normalisation, and the gap ledger.

Run:  uv run python tests/test_relation_gaps.py     (needs postgres + Ollama)

Every assertion here is two-sided (TRAPS #5). The temptation in this area is
to check only that good relations survive; the defect that actually shipped
was that BAD ones vanished without trace, so each check also pins what must
NOT happen — a relation that should not map returns None, a gap row is written
for the dropped fact AND not written for the kept one.

Cleanup deletes exactly the rows this file created, found by id, never by a
hardcoded name list and never by TRUNCATE (TRAPS #6).
"""
import sys
import uuid

sys.path.insert(0, ".")

from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402
from src.memory.models import (  # noqa: E402
    CodexEdge,
    CodexEntity,
    CodexEvent,
    CodexRelationGap,
    Conversation,
    EpisodicMemory,
    IdempotencyKey,
)
from src.workers.bg_client_factory import json_schema  # noqa: E402
from src.workers.codex_extractor import (  # noqa: E402
    ALLOWED_RELATIONS,
    _relation_forms,
    extract_triplets,
    normalize_relation,
)

BG_MODEL = "qwen3:4b-instruct"
passed = failed = 0
made_convs: list = []


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")


# ── 1. normalisation: what maps, and what must NOT ──────────────────────
print("── relation normalisation ──")
for raw, want in [("works on", "works_on"), ("Works On", "works_on"),
                  ("works-on", "works_on"), ("is located in", "located_in"),
                  ("reports to", "reports_to"), ("depends upon", "depends_on"),
                  ("uses", "uses")]:
    check(f"{raw!r} maps to {want}", normalize_relation(raw) == want,
          f"got {normalize_relation(raw)!r}")

# The negative side is the load-bearing half: these have NO honest target, and
# inventing one is the ~78%-wrong failure the enum measurement rejected.
for raw in ("has", "is", "includes", "prefers running", "is_exam_of", ""):
    check(f"{raw!r} correctly refuses to map", normalize_relation(raw) is None,
          f"got {normalize_relation(raw)!r}")

# ── 2. the safety property normalisation depends on ─────────────────────
print("\n── vocabulary safety ──")
collisions = {}
for r in ALLOWED_RELATIONS:
    collisions.setdefault(list(_relation_forms(r))[-1], []).append(r)
dupes = {k: v for k, v in collisions.items() if len(v) > 1}
check("no two vocabulary entries collide after normalisation", not dupes,
      f"collisions: {dupes}")
check("every vocabulary entry maps to itself",
      all(normalize_relation(r) == r for r in ALLOWED_RELATIONS))

# ── 3. the schema constructor builds the form that BINDS ────────────────
print("\n── response_format construction ──")
rf = json_schema("t", {"type": "object"})
check("builds type=json_schema (the form that binds)",
      rf.get("type") == "json_schema", str(rf))
check("does NOT build type=json_object (measured 0/8, silently ignored)",
      rf.get("type") != "json_object")
check("carries the schema through", rf["json_schema"]["schema"] == {"type": "object"})

# ── 4. the enum capability is real, not decoration (Z2 must be able to flip it)
print("\n── enum capability (OFF by default; Z2 decides) ──")
check("shipped default is shape-only", settings.codex_constrain_shape is True
      and settings.codex_constrain_relation_enum is False,
      f"shape={settings.codex_constrain_shape} enum={settings.codex_constrain_relation_enum}")
_prev = settings.codex_constrain_relation_enum
try:
    settings.codex_constrain_relation_enum = True
    gaps_enum: list = []
    trips_enum = extract_triplets(
        "User: my laptop has an RTX 5090.\nAssistant: Noted.",
        BG_MODEL, gaps=gaps_enum)
    # With the decoder enum on, the model CANNOT emit an out-of-vocabulary
    # relation, so the gap sink must stay empty. That is the whole mechanism.
    check("enum ON ⇒ nothing can fall out of vocabulary", not gaps_enum,
          f"gaps: {[g.get('relation') for g in gaps_enum]}")
    check("enum ON ⇒ every relation is legal",
          all(t["relation"] in ALLOWED_RELATIONS for t in trips_enum))
finally:
    settings.codex_constrain_relation_enum = _prev

# ── 5. live extraction: the sink captures exactly what the filter drops ──
print("\n── live extraction + gap sink ──")
TEXT = ("User: my laptop has an RTX 5090 and includes 24GB of vram. "
        "I work on ICE and it uses postgres.\n"
        "Assistant: Noted — that is a capable machine for local models.")
gaps: list = []
trips = extract_triplets(TEXT, BG_MODEL, gaps=gaps)
check("kept relations are all in-vocabulary",
      all(t["relation"] in ALLOWED_RELATIONS for t in trips),
      str([t["relation"] for t in trips]))
check("gap relations are all OUT-of-vocabulary (the other side)",
      all(g["relation"] not in ALLOWED_RELATIONS for g in gaps),
      str([g["relation"] for g in gaps]))
check("a gap keeps subject AND object, so it can be replayed later",
      all(g.get("subject") and g.get("object") for g in gaps) if gaps else True)
check("extraction produced something at all", bool(trips) or bool(gaps))

# ── 6. persistence: gaps survive into the ledger via extract_codex ──────
print("\n── ledger persistence ──")
db = SessionLocal()
try:
    conv = Conversation(id=uuid.uuid4())
    db.add(conv)
    db.flush()
    made_convs.append(conv.id)
    batch = uuid.uuid4()
    turn = EpisodicMemory(
        batch_id=batch, conversation_id=conv.id,
        raw_text=TEXT, lossless_flag=True, topic_tags=[],
        intent_tags=[], context_reliance="Long_Term_Memory",
        idempotency_key=f"a1-relgap-test:{batch}",
    )
    db.add(turn)
    db.commit()

    before = db.query(CodexRelationGap).filter_by(batch_id=batch).count()
    check("no gap rows exist for this batch before extraction", before == 0)

    from src.workers.codex_extractor import extract_codex
    extract_codex(str(batch), BG_MODEL)

    rows = db.query(CodexRelationGap).filter_by(batch_id=batch).all()
    check("extract_codex wrote gap rows", bool(rows), f"{len(rows)} rows")
    if rows:
        r = rows[0]
        check("row keeps the model's own wording", bool(r.raw_relation))
        check("row keeps subject and object", bool(r.subject) and bool(r.object))
        check("row starts pending", r.status == "pending")
        check("stored relation is genuinely out-of-vocabulary",
              all(x.raw_relation not in ALLOWED_RELATIONS for x in rows),
              str([x.raw_relation for x in rows]))
        check("provenance links back to the conversation",
              all(x.conversation_id == conv.id for x in rows))
finally:
    db.rollback()   # a failed flush above must not block cleanup (TRAPS #6)
    # Delete exactly what this run created, children before parents. The codex
    # rows matter as much as the episodic ones: extract_codex creates entities
    # for every subject/object it sees, and leaving those behind is precisely
    # the orphan-residue that made an unrelated suite fail 39/40 (TRAPS #6).
    for cid in made_convs:
        turns = db.query(EpisodicMemory).filter_by(conversation_id=cid).all()
        for t in turns:
            db.query(CodexRelationGap).filter_by(batch_id=t.batch_id).delete()
            edges = db.query(CodexEdge).filter_by(source_batch=t.batch_id).all()
            ent_ids = {e.source_id for e in edges} | {e.target_id for e in edges}
            for e in edges:
                db.delete(e)
            db.query(CodexEvent).filter_by(batch_source=t.batch_id).delete()
            db.query(IdempotencyKey).filter(
                IdempotencyKey.key.like("%")).filter(
                IdempotencyKey.key == __import__("hashlib").sha256(
                    f"codex:{t.batch_id}".encode()).hexdigest()).delete()
            db.flush()
            for eid in ent_ids:
                still = db.query(CodexEdge).filter(
                    (CodexEdge.source_id == eid) | (CodexEdge.target_id == eid)).count()
                if not still:
                    db.query(CodexEvent).filter_by(entity_id=eid).delete()
                    db.query(CodexEntity).filter_by(id=eid).delete()
            db.delete(t)
        db.flush()          # turns must be gone before the FK parent
        db.query(Conversation).filter_by(id=cid).delete()
    db.commit()
    left = db.query(Conversation).filter(Conversation.id.in_(made_convs)).count() \
        if made_convs else 0
    check("cleanup left no rows behind", left == 0, f"{left} remaining")
    db.close()

print(f"\n  {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
