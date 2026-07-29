"""C16 — the recent window: sitting-bounded, newest-first, filtered.

Live-DB behavioural suite, house style: own rows, deleted at the end, never
TRUNCATE. Every scope-ish check is two-sided — the thing that should be there
IS, and the thing that should not be there is NOT — because a window function
that returns nothing passes every "X is absent" check for free.

Run: uv run python tests/test_context_budget.py
"""

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text as sql_text  # noqa: E402

from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402
from src.api.memory_decision import derive_total_budget  # noqa: E402
from src.api.prompt_assembler import get_recent_turns  # noqa: E402
from src.memory.models import Conversation, EpisodicMemory  # noqa: E402

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


MARK = f"budmark{uuid.uuid4().hex[:6]}"
DIM = settings.embedding_dim
NOW = datetime.now(timezone.utc)

db = SessionLocal()
conv = doc_conv = None
made = []


def mk(conv_id, minutes_ago, session_id, body, *, private=False,
       is_document=False, promoted=None, assistant="ok"):
    t = EpisodicMemory(
        conversation_id=conv_id, batch_id=uuid.uuid4(), session_id=session_id,
        is_private=private, is_document=is_document,
        promoted_document_id=promoted,
        timestamp=NOW - timedelta(minutes=minutes_ago),
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory",
        raw_text=f"User: {body}\n\nAssistant: {assistant}",
        embedding=[0.0] * DIM, decay_score=1.0, inject_raw=True,
        idempotency_key=f"test-bud-{uuid.uuid4()}")
    db.add(t)
    db.commit()
    made.append(t.id)
    return t


def texts(msgs):
    return " ".join(m["content"] for m in msgs)


try:
    conv = Conversation(memory_scope_type="auto")
    db.add(conv)
    db.commit()

    sess_now, sess_old = uuid.uuid4(), uuid.uuid4()
    # Realistically sized turns. With three-word turns every budget holds
    # everything and the "what falls off" checks below pass vacuously.
    PAD = " context " * 40
    # Previous sitting: three turns, oldest first.
    mk(conv.id, 300, sess_old, f"{MARK} OLDEST of the previous sitting{PAD}")
    mk(conv.id, 290, sess_old, f"{MARK} MIDDLE of the previous sitting{PAD}")
    mk(conv.id, 280, sess_old, f"{MARK} LAST of the previous sitting{PAD}")
    # Current sitting: three turns.
    mk(conv.id, 20, sess_now, f"{MARK} first of this sitting{PAD}")
    mk(conv.id, 10, sess_now, f"{MARK} second of this sitting{PAD}")
    mk(conv.id, 1, sess_now, f"{MARK} NEWEST turn of all{PAD}")

    # ══ 1. The sitting is the boundary ════════════════════════════════════
    print("── 1: bounded by the sitting, with one bridge turn ──")
    msgs = get_recent_turns(db, conv.id, max_tokens=8000)
    body = texts(msgs)
    check("this sitting's turns are all present",
          all(s in body for s in ("first of this sitting",
                                  "second of this sitting",
                                  "NEWEST turn of all")))
    check("the previous sitting's LAST turn bridges across",
          "LAST of the previous sitting" in body)
    check("...but the rest of the previous sitting does NOT (two-sided)",
          "OLDEST of the previous sitting" not in body
          and "MIDDLE of the previous sitting" not in body)

    settings.recent_window_scope = "count"
    try:
        allm = texts(get_recent_turns(db, conv.id, max_tokens=8000))
        check("scope='count' is a real kill switch — the old sitting returns",
              "OLDEST of the previous sitting" in allm)
    finally:
        settings.recent_window_scope = "session"

    # ══ 2. Under pressure the OLDEST fall off, not the newest ═════════════
    # This is the bug that mattered most: selection used to run oldest-first
    # and `break` on the first overflow, so one fat old turn cost you every
    # turn after it — including the one you had just sent.
    print("── 2: a tight budget drops the OLDEST, never the newest ──")
    tight = get_recent_turns(db, conv.id, max_tokens=40)
    tbody = texts(tight)
    check("the NEWEST turn survives a budget that cannot hold them all",
          "NEWEST turn of all" in tbody)
    check("...and older turns are what were dropped (two-sided)",
          "LAST of the previous sitting" not in tbody)
    check("chronological order is preserved for the model",
          tbody.index("second of this sitting") < tbody.index("NEWEST turn of all")
          if "second of this sitting" in tbody else True)

    # ══ 3. The filters every retrieval leg already had ════════════════════
    print("── 3: private / document / promoted turns stay out ──")
    mk(conv.id, 5, sess_now, f"{MARK} PRIVATE turn", private=True)
    mk(conv.id, 4, sess_now, f"{MARK} DOCUMENTY turn", is_document=True)
    after = texts(get_recent_turns(db, conv.id, max_tokens=8000))
    check("a private turn is excluded", "PRIVATE turn" not in after)
    check("an is_document turn is excluded", "DOCUMENTY turn" not in after)
    check("...while ordinary turns of the same sitting are still there",
          "NEWEST turn of all" in after)

    doc_conv = Conversation(memory_scope_type="auto", kind="document")
    db.add(doc_conv)
    db.commit()
    mk(doc_conv.id, 3, uuid.uuid4(), f"{MARK} SECTION of a document")
    check("a document conversation's own window is empty (kind filter)",
          get_recent_turns(db, doc_conv.id, max_tokens=8000) == [])

    # ══ 4. One turn may not eat the window ════════════════════════════════
    print("── 4: per-turn share ──")
    mk(conv.id, 2, sess_now, f"{MARK} " + ("verbose " * 4000))
    capped = get_recent_turns(db, conv.id, max_tokens=2000)
    longest = max((len(m["content"]) for m in capped), default=0)
    check("no single turn consumes the whole window",
          longest < 2000 * 4 * settings.recent_window_max_turn_frac + 400)
    check("...and the window still returns something", len(capped) > 0)

    # ══ 5. Budget arithmetic ══════════════════════════════════════════════
    print("── 5: the budget fits inside the window ──")
    check("a small model's budget leaves room to answer",
          derive_total_budget(8192, settings) + settings.context_generation_reserve
          <= 8192)
    check("a tiny model is not handed more than its whole window",
          derive_total_budget(2048, settings) < 2048)
    check("a large model is still capped by the max guardrail",
          derive_total_budget(200_000, settings) == settings.context_budget_max)

    # ══ 6. Empty conversation ═════════════════════════════════════════════
    print("── 6: empty conversation ──")
    empty = Conversation(memory_scope_type="auto")
    db.add(empty)
    db.commit()
    check("no turns → no window, no crash",
          get_recent_turns(db, empty.id, max_tokens=4000) == [])
    db.execute(sql_text("DELETE FROM conversations WHERE id = :c"), {"c": empty.id})
    db.commit()

finally:
    db.rollback()
    conv_ids = [c.id for c in (conv, doc_conv) if c is not None]
    if conv_ids:
        db.execute(sql_text(
            "DELETE FROM episodic_cluster_links WHERE episodic_id IN "
            "(SELECT id FROM episodic_memory WHERE conversation_id = ANY(:c))"),
            {"c": conv_ids})
        db.execute(sql_text(
            "DELETE FROM episodic_chunks WHERE turn_id IN "
            "(SELECT id FROM episodic_memory WHERE conversation_id = ANY(:c))"),
            {"c": conv_ids})
        db.execute(sql_text(
            "DELETE FROM episodic_memory WHERE conversation_id = ANY(:c)"),
            {"c": conv_ids})
        db.execute(sql_text("DELETE FROM conversations WHERE id = ANY(:c)"),
                   {"c": conv_ids})
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
