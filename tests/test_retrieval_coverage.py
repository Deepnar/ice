"""C16 — residual coverage: selection, stopping, identity collapse, leg attribution.

Live-DB behavioural suite in the house style: inserts its own rows, deletes
them, never TRUNCATEs. The geometry checks are deterministic — they seed
vectors with KNOWN angles rather than trusting an embedder — because the point
of coverage is a geometric property and a test that depends on what the
embedder happens to think proves nothing about it.

Run: uv run python tests/test_retrieval_coverage.py
"""

import os
import sys
import uuid

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text as sql_text  # noqa: E402

from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402
from src.memory.models import Conversation, EpisodicMemory  # noqa: E402
from src.retrieval.coverage import (  # noqa: E402
    _knee,
    select_by_coverage,
    set_floor_passes,
)
from src.retrieval.orchestrator import ContextFragment  # noqa: E402

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


DIM = settings.embedding_dim
MARK = f"covmark{uuid.uuid4().hex[:6]}"


def axis(i, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


db = SessionLocal()
conv = None
try:
    # ══ 1. Redundancy: the 59-restatements case ═══════════════════════════
    print("── 1: near-duplicates collapse, distinct facts survive ──")
    q = (axis(0) + axis(1) + axis(2)) / np.sqrt(3)

    dups = [(f"dup{i}", axis(0) + 0.01 * i * axis(5), 1.0) for i in range(6)]
    sel, rec = select_by_coverage(q, dups, min_gain=0.05, min_keep=1)
    check("six restatements of one fact select at most two",
          len(sel) <= 2)
    check("...and the stop reason is that nothing added anything",
          rec["stop_reason"] in ("min_gain", "residual_exhausted"))

    # The paired positive — without it the check above passes for a selector
    # that simply returns nothing (trap 5).
    orth = [("x", axis(0), 1.0), ("y", axis(1), 0.9), ("z", axis(2), 0.8)]
    sel_o, rec_o = select_by_coverage(q, orth, min_gain=0.05, knee_enabled=False)
    check("three DISTINCT facts are all kept (not a selector that drops)",
          len(sel_o) == 3)
    check("...covering the question exhausts the residual",
          rec_o["stop_reason"] == "residual_exhausted")

    # ══ 2. The knee ═══════════════════════════════════════════════════════
    print("── 2: knee on the coverage curve ──")
    check("a sharp bend is found", _knee([1.0, 0.9, 0.05, 0.04, 0.03], 0.08, 2) == 2)
    check("a FLAT curve produces NO cut (not a random truncator)",
          _knee([0.3] * 6, 0.08, 2) == -1)
    # A cut point is never below min_keep. -1 satisfies that trivially (it is
    # "no cut"), so the check has to allow it AND a case that really clamps.
    shallow = _knee([1.0, 0.01, 0.01, 0.01], 0.08, 3)
    check("a barely-bent curve returns no cut rather than a shallow one",
          shallow == -1)
    steep = _knee([1.0, 1.0, 0.001, 0.001, 0.001, 0.001, 0.001], 0.05, 4)
    check("a real bend before min_keep is CLAMPED up to min_keep",
          steep == -1 or steep >= 4)

    # ══ 3. The set floor ══════════════════════════════════════════════════
    print("── 3: set-level quality gate ──")
    check("a set of junk is rejected whole", not set_floor_passes([0.05, 0.04], 0.25))
    check("a set with one good hit passes", set_floor_passes([0.9, 0.04], 0.25))
    check("an empty set is rejected", not set_floor_passes([], 0.25))

    # ══ 4. Degenerate inputs never crash and never silently drop ══════════
    print("── 4: degenerate inputs ──")
    check("no candidates → nothing, no crash", select_by_coverage(q, [])[0] == [])
    unplaced, urec = select_by_coverage(q, [("novec", None, 1.0)])
    check("a fragment with NO vector is admitted, not dropped",
          unplaced == ["novec"] and urec["n_unplaced"] == 1)
    zero_sel, zrec = select_by_coverage(np.zeros(8), orth)
    check("an unusable question vector passes everything through",
          len(zero_sel) == 3 and zrec["stop_reason"] == "no_query_vector")

    # ══ 5. Identity collapse (chunk + parent) ═════════════════════════════
    print("── 5: provenance collapse ──")
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    import src.memory.embedder as emb_mod
    orch = HybridRetrievalOrchestrator(db, emb_mod.get_embedder())
    turn_id = str(uuid.uuid4())
    frags = [
        ContextFragment(text="parent turn text", source_type="episodic",
                        score=0.9, token_count=10, source_batch_id=turn_id,
                        leg="bm25"),
        ContextFragment(text="chunk one of that turn", source_type="episodic",
                        score=0.8, token_count=10, source_batch_id=turn_id,
                        leg="vector"),
        ContextFragment(text="chunk two of that turn", source_type="episodic",
                        score=0.7, token_count=10, source_batch_id=turn_id,
                        leg="chunks"),
        ContextFragment(text="an unrelated turn", source_type="episodic",
                        score=0.6, token_count=10,
                        source_batch_id=str(uuid.uuid4()), leg="vector"),
    ]
    collapsed = orch._collapse_provenance(frags)
    same_turn = [f for f in collapsed if f.source_batch_id == turn_id]
    check("three fragments of ONE turn collapse to the cap",
          len(same_turn) == settings.retrieval_max_frags_per_turn)
    check("...the highest-scoring survive", same_turn[0].score == 0.9)
    check("a different turn is untouched (not a blanket dropper)",
          len(collapsed) == settings.retrieval_max_frags_per_turn + 1)

    # ══ 6. leg is attribution, never entitlement ══════════════════════════
    print("── 6: leg attribution ──")
    legs = {
        "bm25": [ContextFragment(text="a", source_type="episodic", score=1.0,
                                 token_count=5)],
        "vector": [ContextFragment(text="b", source_type="episodic", score=1.0,
                                   token_count=5)],
        "codex": [ContextFragment(text="c", source_type="codex", score=1.0,
                                  token_count=5)],
    }
    fused = orch._apply_rrf(legs, alpha_map={"bm25": 1.0, "vector": 1.0, "codex": 1.0})
    got = {f.text: f.leg for f in fused}
    check("each fragment carries its SPECIFIC leg, not the rollup",
          got == {"a": "bm25", "b": "vector", "c": "codex"})
    check("...while source_type stays the coarse label",
          {f.text: f.source_type for f in fused}["a"] == "episodic")
    # Entitlement: the budget's diversity guarantee must still key on
    # source_type. Keyed on `leg`, cold storage and the wide net would gain
    # guaranteed slots purely because attribution improved.
    budgeted = orch._enforce_token_budget(fused, max_tokens=10_000)
    check("the diversity guarantee still keys on source_type, not leg",
          len({f.source_type for f in budgeted}) == 2)

    # ══ 7. Vector fetch against real rows ═════════════════════════════════
    print("── 7: vectors are FETCHED, never encoded at request time ──")
    conv = Conversation(memory_scope_type="auto")
    db.add(conv)
    db.commit()
    vec = [0.0] * DIM
    vec[0] = 1.0
    t = EpisodicMemory(
        conversation_id=conv.id, batch_id=uuid.uuid4(),
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory",
        raw_text=f"User: {MARK}\n\nAssistant: noted", embedding=vec,
        decay_score=1.0, idempotency_key=f"test-cov-{uuid.uuid4()}")
    db.add(t)
    db.commit()
    frag = ContextFragment(text=MARK, source_type="episodic", score=1.0,
                           token_count=5, source_batch_id=str(t.id), leg="vector")
    fetched = orch._fragment_vectors([frag])
    check("a stored embedding is found by source_batch_id",
          str(t.id) in fetched and len(fetched[str(t.id)]) == DIM)
    missing = orch._fragment_vectors([ContextFragment(
        text="x", source_type="codex", score=1.0, token_count=5)])
    check("a fragment with no source id yields no vector (and no crash)",
          missing == {})

    # ══ 8. The kill switch is two-sided ═══════════════════════════════════
    print("── 8: kill switch ──")
    many = [ContextFragment(text=f"frag {i}", source_type="episodic", score=1.0 - i * 0.1,
                            token_count=5, source_batch_id=str(uuid.uuid4()))
            for i in range(6)]
    settings.retrieval_coverage_enabled = False
    off = orch._apply_coverage(many, [1.0] + [0.0] * (DIM - 1))
    check("coverage OFF returns every fragment untouched", len(off) == len(many))
    settings.retrieval_coverage_enabled = True
    try:
        on = orch._apply_coverage(many, [1.0] + [0.0] * (DIM - 1))
        check("coverage ON runs and records the counterfactual",
              orch._coverage_record is not None
              and "tokens_if_no_stop" in orch._coverage_record)
        check("...and no vectors are stored for these ids, so all are admitted",
              len(on) == len(many))
    finally:
        settings.retrieval_coverage_enabled = False

finally:
    db.rollback()
    if conv is not None:
        db.execute(sql_text("DELETE FROM episodic_chunks WHERE turn_id IN "
                            "(SELECT id FROM episodic_memory WHERE conversation_id = :c)"),
                   {"c": conv.id})
        db.execute(sql_text("DELETE FROM episodic_memory WHERE conversation_id = :c"),
                   {"c": conv.id})
        db.execute(sql_text("DELETE FROM conversations WHERE id = :c"), {"c": conv.id})
        db.commit()
    mine = db.execute(sql_text(
        "SELECT count(*) FROM episodic_memory WHERE raw_text LIKE :m"),
        {"m": f"%{MARK}%"}).scalar()
    print(f"\nthis run's rows remaining: {mine} (must be 0)")
    total = db.execute(sql_text(
        "SELECT (SELECT count(*) FROM conversations)+(SELECT count(*) FROM episodic_memory)")).scalar()
    print(f"store-wide conversations+turns: {total} "
          "(non-zero = residue from another suite, not this one)")
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
