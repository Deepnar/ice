"""G36: a retrieval leg must announce its own failure, and a scope that cannot
be resolved must fail CLOSED.

Two properties, both of which used to be false.

1. **Observability.** Twenty-two `except` handlers on the retrieval path
   caught, rolled back and returned `[]` in silence, so a leg broken by a
   schema change or a bad bind was indistinguishable from an empty store.
   `_procedural_lookup` sat dead exactly that way from the psycopg3 move
   until C9 removed the intent gate hiding it.

2. **Direction.** `_codex_scope_sets` returned `(None, None)` on failure, and
   `None` there means UNSCOPED, not "no scope" — so failing to resolve the
   caller's scope handed the codex leg the whole graph. `_resolve_exclusion_sets`
   was the mirror: it cleared the deny sets, and the graph then served exactly
   the conversations the user had excluded. Both widen visibility on failure,
   which is the direction G16 forbids.

**Every scope check here is two-sided and carries a positive control.** The
first draft of this reproduction asserted only "the out-of-scope entity is
absent" — and passed against a fixture that returned nothing at all, for
either conversation, because the seeded entity embeddings were the constant
stub vector that ~12 suites use and similarity matching therefore never fired.
That is TRAPS #13 and the vacuous probe that opened G36 itself. The control
line ("the IN-scope entity IS present") is what makes the absence mean
something, so do not delete it.

Runs against the live Postgres and the real embedder (docker up). Inserts its
own uniquely-marked rows and deletes them afterwards — never truncates.

Run: uv run python tests/test_retrieval_failopen.py
"""
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import structlog
from sqlalchemy.exc import OperationalError

from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.embedder import get_embedder
from src.memory.models import (
    CodexEdge, CodexEntity, CodexEvent, Conversation, EpisodicMemory,
)
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


# Invented names so nothing in a real store can collide. Capitalised in the
# prompt because the micro-NER scores 0.000 Jaccard on a pure lowercase change
# (measured 2026-08-03) — a lowercase prompt extracts no entities at all and
# the whole fixture would go quiet for the wrong reason.
NAME_A = "quokkathon"
NAME_B = "zebracorn"
PROMPT = f"Where does {NAME_A.capitalize()} live and where does {NAME_B.capitalize()} live?"
STUB_EMB = [0.05] * 1024

db = SessionLocal()
embedder = get_embedder()
now = datetime.now(timezone.utc)
made = {"convs": [], "turns": [], "ents": [], "edges": [], "events": []}


def enc(text):
    v = embedder.encode(text, convert_to_tensor=False, show_progress_bar=False)
    return v.tolist() if hasattr(v, "tolist") else list(v)


def seed():
    conv_a = Conversation(memory_scope_type="auto")
    conv_b = Conversation(memory_scope_type="auto")
    db.add_all([conv_a, conv_b])
    db.commit()
    made["convs"] = [conv_a, conv_b]
    for conv, subj, obj in ((conv_a, NAME_A, "alphaville"),
                            (conv_b, NAME_B, "betaville")):
        batch = uuid.uuid4()
        turn = EpisodicMemory(
            conversation_id=conv.id, batch_id=batch, timestamp=now,
            topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
            context_reliance="Long_Term_Memory",
            raw_text=f"{subj} lives in {obj}", embedding=STUB_EMB,
            decay_score=1.0, idempotency_key=f"g36-{uuid.uuid4()}")
        db.add(turn)
        # Real embeddings, not the constant stub — see the module docstring.
        src = CodexEntity(canonical_name=subj, entity_type="person",
                          context_payload=f"# {subj}\n\nLinks: lives_in -> {obj}",
                          embedding=enc(subj.capitalize()))
        tgt = CodexEntity(canonical_name=obj, entity_type="place",
                          context_payload=f"# {obj}", embedding=enc(obj.capitalize()))
        db.add_all([src, tgt])
        db.commit()
        edge = CodexEdge(source_id=src.id, target_id=tgt.id, relation="lives_in",
                         source_batch=batch, confidence="active", strength=1.0,
                         valid_from=now)
        event = CodexEvent(entity_id=src.id, event_type="edge_added",
                           batch_source=batch, timestamp=now)
        db.add_all([edge, event])
        db.commit()
        made["turns"].append(turn)
        made["ents"] += [src, tgt]
        made["edges"].append(edge)
        made["events"].append(event)


def cleanup():
    for row in made["events"] + made["edges"]:
        db.delete(row)
    db.commit()
    for row in made["ents"] + made["turns"]:
        db.delete(row)
    db.commit()
    for row in made["convs"]:
        db.delete(row)
    db.commit()


def codex_text(scope):
    """The codex leg's output under *scope*, with the per-request resolvers run
    in the same order retrieve() runs them."""
    orch = HybridRetrievalOrchestrator(db, embedder)
    orch._active_timescope = orch._resolve_timescope(scope)
    orch._scope_project_id = orch._resolve_project_scope(scope)
    orch._resolve_exclusion_sets(scope)
    cls = ClassificationResult(
        prompt=PROMPT, topic_tags=["Software_&_Tech"],
        intent_tags=["Factual_Retrieval"], context_reliance="Long_Term_Memory",
        raw_probs=[0.0] * 27, max_confidence=0.9)
    return "\n".join(f.text for f in
                     orch._codex_graph(cls, scope, prompt_embedding=STUB_EMB))


def legs_reported(captured):
    return [e.get("leg") for e in captured
            if e.get("event") == "retrieval_leg_failed"]


class _CountingDB:
    """Just enough Session to observe whether _leg_degraded rolled back."""

    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


try:
    seed()
    conv_a, conv_b = made["convs"]
    good_scope = {"conversation_ids": [str(conv_a.id)]}
    # An id Postgres cannot cast to uuid makes the scope query raise where a
    # schema change or a bind mismatch would. The TRIGGER is incidental; what
    # is under test is what the handler does with any failure.
    broken_scope = {"conversation_ids": [str(conv_a.id), "not-a-uuid"]}

    print("── the fixture actually retrieves something (control) ──")
    unscoped = codex_text({})
    check("unscoped: A's entity is retrieved", NAME_A in unscoped)
    check("unscoped: B's entity is retrieved", NAME_B in unscoped)

    print("── scope resolution: working, then broken ──")
    scoped = codex_text(good_scope)
    check("scoped to A: A IS present (the control)", NAME_A in scoped)
    check("scoped to A: B is absent", NAME_B not in scoped)

    with structlog.testing.capture_logs() as cap:
        broken = codex_text(broken_scope)
    check("broken scope: B does NOT leak in", NAME_B not in broken)
    check("broken scope: A is dropped too (closed, not widened)",
          NAME_A not in broken)
    check("broken scope: reported as scope.codex_sets",
          "scope.codex_sets" in legs_reported(cap))

    print("── exclusion: working, then broken ──")
    excl_scope = {"exclude_conversation_ids": [str(conv_b.id)]}
    excluded = codex_text(excl_scope)
    check("exclusion: A IS still present (the control)", NAME_A in excluded)
    check("exclusion: B is absent", NAME_B not in excluded)

    broken_excl = {"exclude_conversation_ids": [str(conv_b.id), "not-a-uuid"]}
    with structlog.testing.capture_logs() as cap:
        broken = codex_text(broken_excl)
    check("broken exclusion: B does NOT come back", NAME_B not in broken)
    check("broken exclusion: reported as scope.exclusions",
          "scope.exclusions" in legs_reported(cap))

    print("── a broken leg names itself ──")
    orch = HybridRetrievalOrchestrator(db, embedder)
    orch._active_timescope = orch._resolve_timescope(None)
    with structlog.testing.capture_logs() as cap:
        # The historical case: a bind the leg's SQL cannot use. It returned []
        # on every call for months and said nothing.
        frags = orch._procedural_lookup("not-an-embedding", ClassificationResult(
            prompt=PROMPT, topic_tags=[], intent_tags=[],
            context_reliance="Long_Term_Memory", raw_probs=[0.0] * 27,
            max_confidence=0.9), None)
    check("broken procedural leg returns nothing", frags == [])
    check("...and reports leg=procedural", "procedural" in legs_reported(cap))

    print("── the rollback is conditional on the error being the DB's ──")
    fake = _CountingDB()
    orch = HybridRetrievalOrchestrator(fake, None)
    with structlog.testing.capture_logs() as cap:
        orch._leg_degraded("probe.db", OperationalError("stmt", {}, Exception()))
    check("a DB error rolls back", fake.rollbacks == 1)
    check("...and is reported", legs_reported(cap) == ["probe.db"])

    with structlog.testing.capture_logs() as cap:
        orch._leg_degraded("probe.python", ValueError("not the database"))
    check("a Python error does NOT roll back (still 1)", fake.rollbacks == 1)
    check("...but is reported all the same",
          legs_reported(cap) == ["probe.python"])

    print("── retrieval calls no LLM (the seam HyDE left behind) ──")
    import src.retrieval.orchestrator as orch_mod
    check("orchestrator imports no background client",
          not hasattr(orch_mod, "get_bg_client"))
    check("...and holds none", not hasattr(
        HybridRetrievalOrchestrator(db, None), "bg_client"))

finally:
    cleanup()
    print()
    print(f"{_passed} passed, {_failed} failed")
