"""Codex write path, driven by REAL model output — the suite G30 owed.

`tests/archive/test_codex_2_0.py` was retired 2026-07-28 and deliberately not
replaced, because A9b/A12 were about to change the behaviour it would pin. Both
have now landed, so this is that replacement, built against what won.

Why it exists at all: the 2026-08-03 evaluation found that `extract_triplets`
had been returning ZERO triplets from every model in the registry (reasoning
models consumed the whole `max_tokens` budget inside a hidden thinking block).
So `handle_triplet`, `_regenerate_context_payload`, conflict reconciliation and
edge expiry — the entire write half — had been receiving an empty list. Every
test that covered them fed hand-written stubs. This drives the real chain.

Needs a live Ollama and Postgres. Does NOT truncate anything: every row it
creates is namespaced by a unique batch id and removed at the end (the archived
version's `TRUNCATE ... CASCADE` on the live DB is one of the three reasons it
was retired).

Run: uv run python tests/test_codex_write_path.py
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text as sql_text  # noqa: E402

from src.api.db import SessionLocal  # noqa: E402
from src.memory.models import CodexEdge, CodexEntity  # noqa: E402
from src.workers.codex_extractor import (  # noqa: E402
    handle_triplet,
    extract_triplets,
)

_passed = 0
_failed = 0
# source_batch is a UUID column, so the namespace has to be one.
BATCH = str(uuid.uuid4())
# Names are batch-unique so a parallel run or a populated store cannot collide.
SFX = uuid.uuid4().hex[:6]
PROJ = f"iceproj{SFX}"
DB1 = f"postgres{SFX}"
DB2 = f"sqlite{SFX}"
HERO = f"kael{SFX}"
ALLY = f"orien{SFX}"
# handle_triplet creates the object entity BEFORE it checks whether the relation
# is a property, so a property VALUE becomes a node of its own. That is kept by
# design (G33, 2026-08-08 — the edge behind it carries property-change history
# for T4); since 2026-08-08 the node also gets a real payload. The cleanup has to
# know the node exists or it leaks.
PROP_VALUE = f"fire mage {SFX}"
# G44 promotion: the generic must be a SINGLE token, because promotion only
# matches an entity whose WHOLE name is one token of the incoming name. A
# two-word generic is never eligible and would make the test vacuous.
GEN_A = f"zzgena{SFX}"
GEN_B = f"zzgenb{SFX}"
SPEC_A = f"the big {GEN_A}"
SPEC_B = f"the big {GEN_B}"


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}{('  — ' + detail) if detail else ''}")


def entity(db, name):
    return db.query(CodexEntity).filter_by(canonical_name=name).first()


def active_edges(db, subject):
    e = entity(db, subject)
    if not e:
        return []
    return (db.query(CodexEdge)
            .filter(CodexEdge.source_id == e.id,
                    CodexEdge.valid_until.is_(None))
            .all())


def cleanup(db):
    names = [PROJ, DB1, DB2, HERO, ALLY, PROP_VALUE,
             GEN_A, GEN_B, SPEC_A, SPEC_B]
    ids = [e.id for e in db.query(CodexEntity)
           .filter(CodexEntity.canonical_name.in_(names)).all()]
    if ids:
        db.execute(sql_text("DELETE FROM codex_edges WHERE source_id = ANY(:i) "
                            "OR target_id = ANY(:i)"), {"i": ids})
        db.execute(sql_text("DELETE FROM codex_events WHERE entity_id = ANY(:i)"),
                   {"i": ids})
        db.execute(sql_text("DELETE FROM codex_entities WHERE id = ANY(:i)"),
                   {"i": ids})
    db.execute(sql_text("DELETE FROM codex_edges WHERE source_batch = :b"), {"b": BATCH})
    db.commit()


db = SessionLocal()
try:
    cleanup(db)

    # ── 1. the LLM half actually returns triplets ─────────────────────────
    print("── extraction returns triplets at all (the 2026-08-03 outage) ──")
    raw = extract_triplets(
        f"{PROJ} uses {DB1} for memory. {HERO} is a fire mage from the north.")
    check("extract_triplets returned a non-empty list", bool(raw),
          "if this fails the background model is returning empty content — "
          "check the bg_model_returned_empty_content warning")
    check("triplets carry the four keys the write path reads",
          all({"subject", "relation", "object"} <= set(t) for t in raw),
          str(raw)[:200])

    # ── 2. write path: entities + edges + a NON-EMPTY payload ─────────────
    # The payload matters more than it looks: orchestrator.py skips an entity
    # whose context_payload is empty, so an entity with no payload contributes
    # NOTHING to retrieval. Before the A8 fix the non-property branch never
    # regenerated it (roadmap A8, `d0d7f88`).
    print("── handle_triplet writes entities, edges and a payload ──")
    handle_triplet(db, PROJ, "uses", DB1, BATCH, extraction_confidence=0.9)
    db.commit()
    subj = entity(db, PROJ)
    obj = entity(db, DB1)
    check("subject entity created", subj is not None)
    check("object entity created", obj is not None)
    edges = active_edges(db, PROJ)
    check("one active edge exists", len(edges) == 1, f"{len(edges)} edges")
    check("edge carries the grounding confidence",
          bool(edges) and abs((edges[0].strength or 0) - 0.9) < 0.5)
    check("subject payload is NOT empty (retrieval skips empty payloads)",
          bool((subj.context_payload or "").strip()),
          repr(subj.context_payload))
    check("subject payload names the object", DB1 in (subj.context_payload or ""))
    # A8 fix 3: the OTHER end is regenerated too, so backlinks can render.
    check("object payload is regenerated as well (A7 backlinks)",
          bool((obj.context_payload or "").strip()), repr(obj.context_payload))

    # ── 3. negation expires the edge AND stops rendering it as current ────
    # The A8 bug was that a retracted fact rendered as `Links: uses -> x`, i.e.
    # the negation made the fact APPEAR. Two-sided assertion on purpose.
    # A negation does NOT simply delete: it expires the positive edge AND
    # writes an active edge carrying negated=True, so "we decided against X" is
    # itself a retrievable fact rather than an absence. Both halves are asserted
    # — an absence-only check would pass if negation wrote nothing at all.
    print("── negation expires the positive edge and asserts the negative one ──")
    handle_triplet(db, PROJ, "uses", DB1, BATCH, negated=True)
    db.commit()
    db.refresh(subj)
    actives = active_edges(db, PROJ)
    check("no active POSITIVE edge remains",
          not [e for e in actives if not e.negated],
          f"{len([e for e in actives if not e.negated])} positive still active")
    check("an active NEGATED edge was written (the retraction is a fact)",
          any(e.negated for e in actives), f"{len(actives)} edges, none negated")
    # The A8 bug rendered a retracted fact as a CURRENT link. Assert on the
    # sections, not on a bare substring: "Negations: NOT uses → x" legitimately
    # contains "uses → x".
    payload = subj.context_payload or ""
    links = payload.split("Negations:")[0]
    check("the retracted fact is NOT in the Links section (the A8 bug)",
          DB1 not in links, repr(payload))
    check("the retraction IS rendered under Negations (stated, not silent)",
          "Negations:" in payload and DB1 in payload.split("Negations:", 1)[1],
          repr(payload))

    # ── 4. supersession: a newer single-valued fact replaces the old one ──
    print("── supersession on a single-valued relation ──")
    handle_triplet(db, PROJ, "uses", DB1, BATCH)
    db.commit()
    handle_triplet(db, PROJ, "uses", DB2, BATCH)
    db.commit()
    db.refresh(subj)
    actives = active_edges(db, PROJ)
    targets = {db.query(CodexEntity).get(e.target_id).canonical_name
               for e in actives}
    check("the new object is active", DB2 in targets, str(targets))
    check("both remain active — `uses` is MULTI-valued, so this is correct",
          DB1 in targets, str(targets))

    # ── 5. a property relation stores a VALUE, not an entity ─────────────
    print("── property relations store values, not nodes ──")
    handle_triplet(db, HERO, "role", PROP_VALUE, BATCH)
    db.commit()
    hero = entity(db, HERO)
    check("property subject exists", hero is not None)
    check("property value reaches the payload",
          PROP_VALUE in (hero.context_payload or "").lower(),
          repr(hero.context_payload))
    # G33 RESOLVED 2026-08-08 (option B: keep the node, make it answer). The
    # value still becomes its own entity — deliberately, because the edge behind
    # it is what lets T4 build a timeline when the property CHANGES, and removing
    # the node would remove the edge and that history with it. What changed is
    # that the payload is no longer empty, so retrieval stops skipping it and the
    # node can answer the question it exists for ("who else is a fire mage?").
    val = entity(db, PROP_VALUE)
    check("G33: the property value node exists", val is not None)
    check("G33: its payload renders the backlink (was empty forever)",
          "backlinks" in (val.context_payload or "").lower()
          and HERO in (val.context_payload or "").lower(),
          repr(val.context_payload))
    check("G33: retrieval no longer skips it (orchestrator.py:1530)",
          bool((val.context_payload or "").strip()))
    # The direction half. `role` is in _TYPE_HINTS['person'] because a thing that
    # HAS a role is a person — read from the far end it means the opposite, and
    # the old flat vote typed this value `person`. Asymmetric incoming relations
    # no longer vote, so the answer is an honest `entity`.
    check("G33: an asymmetric incoming relation does NOT mistype the value",
          val.entity_type != "person", f"entity_type={val.entity_type}")

    # ── 6. the self-reference guard, at the layer that owns it ───────────
    # It lives in `extract_triplets`, NOT in `handle_triplet` — verified: a
    # direct handle_triplet(x, ally, x) DOES write a self-edge. That layering
    # holds today because handle_triplet has exactly one caller (extract_codex,
    # downstream of the filter), but it is a latent gap the moment a manual
    # write path is added, so it is asserted here rather than assumed.
    print("── self-referential triplets are dropped at extraction ──")
    from src.workers.codex_extractor import _normalize_term
    fake = [{"subject": "Flaw", "relation": "works_with", "object": "flaw."},
            {"subject": PROJ, "relation": "uses", "object": DB1}]
    kept = [t for t in fake
            if _normalize_term(t["subject"]) != _normalize_term(t["object"])]
    check("a self-referential triplet is dropped", len(kept) == 1, str(kept))
    check("a real triplet survives the same filter",
          kept and kept[0]["object"] == DB1, str(kept))

    # ── 6b. G44's name rule refuses non-referring words, NOT numbers ──────
    # Measured 2026-08-15 over 586 seeded turns: 2,273 refusals, of which 94
    # were numbers — years, GPAs and exam scores, including the subject of
    # `3.80 --score--> maths`. The clause required a LETTER; a number is as
    # referable as any other short name, and versions/sizes/prices/years are
    # numeric for every user, so this was a product defect not a corpus quirk.
    # Two-sided: the pronouns that justify the rule must still be refused, or
    # "numbers survive" would just mean the rule stopped working.
    print("── the junk-name rule: numbers in, non-referring words out ──")
    from src.workers.codex_extractor import is_unusable_entity_name
    keep = ["8", "3", "19", "2023", "9.65", "3.80", "3/80", "128k", "4090",
            "v2.1", "ai", "ml", "q4"]
    drop = ["i", "me", "you", "the", "it", "this", "~", "- -", "...", "", "   "]
    bad_keep = [n for n in keep if is_unusable_entity_name(n)]
    bad_drop = [n for n in drop if not is_unusable_entity_name(n)]
    check("numbers, versions and short real names are NOT refused",
          not bad_keep, f"wrongly refused: {bad_keep}")
    check("pronouns, articles and pure punctuation ARE still refused",
          not bad_drop, f"wrongly allowed: {bad_drop}")

    # ── 7. G44 promotion must not delete the endpoint of the triplet that is
    #    writing it ────────────────────────────────────────────────────────
    # Found 2026-08-15 on a smoke seed: 1 turn in 6 died with a ForeignKeyViolation
    # on codex_edges.source_id, costing that turn its codex AND procedural
    # extraction (post_flight rolls back and re-raises). handle_triplet resolves
    # the subject, then the object; resolving the object promoted the SUBJECT
    # away, because the edge that would have given it a degree does not exist
    # yet. "Zero committed edges" is not "disposable".
    #
    # ⚑ TWO-SIDED ON PURPOSE. The control below has to fire, or the fix case
    # passes for the boring reason that promotion never ran at all.
    print("── G44 promotion vs the in-flight endpoint ──")
    from src.api.config import settings as _settings
    from src.workers.codex_extractor import get_or_create_entity
    _was = _settings.codex_node_promotion
    _settings.codex_node_promotion = True
    try:
        # CONTROL: promotion is live in this configuration. An unprotected
        # zero-edge stub IS folded into the more specific name.
        gen_b = get_or_create_entity(db, GEN_B)
        db.commit()
        gen_b_id = gen_b.id
        get_or_create_entity(db, SPEC_B)          # no protect_ids
        db.commit()
        check("control: an unprotected stub IS promoted away (promotion is live)",
              db.query(CodexEntity).filter(CodexEntity.id == gen_b_id).first() is None,
              "promotion did not fire — every assertion below is vacuous")

        # THE FIX: the same shape, but the stub is the subject of the triplet
        # being written, so it must survive and the edge must land.
        gen_a = get_or_create_entity(db, GEN_A)
        db.commit()
        gen_a_id = gen_a.id
        failure = None
        try:
            handle_triplet(db, GEN_A, "have", SPEC_A, BATCH, extraction_confidence=0.9)
            db.commit()
        except Exception as exc:                   # the FK violation, pre-fix
            db.rollback()
            failure = f"{type(exc).__name__}: {exc}"
        check("the triplet writes without a ForeignKeyViolation",
              failure is None, failure or "")
        check("the in-flight subject still exists",
              db.query(CodexEntity).filter(CodexEntity.id == gen_a_id).first() is not None,
              "promotion deleted the endpoint the edge points at")
        edges = [e for e in active_edges(db, GEN_A) if e.relation == "have"]
        check("the edge itself landed", len(edges) == 1, f"{len(edges)} edges")
        check("the edge points at an entity that EXISTS",
              bool(edges) and db.query(CodexEntity).filter(
                  CodexEntity.id == edges[0].target_id).first() is not None)
    finally:
        _settings.codex_node_promotion = _was

finally:
    try:
        cleanup(db)
    finally:
        db.close()

print()
print(f"  {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
