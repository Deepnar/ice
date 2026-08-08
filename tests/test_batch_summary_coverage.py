"""G11: batch-summary coverage — the marker, the widening, and the FK contract.

Three properties, and the middle one is the bug this item existed for:

  1. WIDENING — an old turn compresses even when it has NOT decayed. Only
     `decay_score < 0.3` qualified before, so a frequently-accessed turn in a
     long conversation never compressed. (The spec's stated validation:
     "seeded old undecayed turn gets a batch summary".)
  2. IDEMPOTENCE — a second pass must produce NO new summary. Before the
     `batch_summary_id` marker nothing excluded already-summarised turns, while
     the docstring claimed otherwise, so every 2-hour pass re-ran the LLM and
     appended another summary row for the retrieval leg to inject twice.
  3. FK CONTRACT — deleting a summary must set its turns' marker back to NULL
     (ON DELETE SET NULL), making them eligible again rather than orphaning the
     reference or blocking C10's cascade.

House pattern: live Postgres, uniquely-marked fixtures, stub LLM (no model
load), cleanup in `finally` — NEVER truncates.

Run: uv run python tests/test_batch_summary_coverage.py
"""
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text as sql_text

from src.api.db import SessionLocal
from src.memory.models import BatchSummary, Conversation, EpisodicMemory

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))


HEX = uuid.uuid4().hex[:6].translate(str.maketrans("0123456789", "abcdefghij"))
MARK = "bscovmark" + HEX
NOW = datetime.now(timezone.utc)
EMB = [0.02] * 1024


class StubCompletion:
    """One canned summary; counts how many times the LLM was actually asked."""
    calls = 0

    class _Msg:
        content = "A stub batch summary of the seeded turns."

    class _Choice:
        message = None

    def create(self, **kw):
        StubCompletion.calls += 1
        c = StubCompletion._Choice()
        c.message = StubCompletion._Msg()
        return type("R", (), {"choices": [c]})()


class StubChat:
    completions = StubCompletion()


class StubClient:
    chat = StubChat()


class StubEmbedder:
    def encode(self, text, convert_to_tensor=False, **kw):
        import numpy as np
        return np.array(EMB, dtype="float32")


db = SessionLocal()
conv_id = None
old_conv_id = None
turn_ids: list = []

from src.api.config import settings
import src.workers.batch_summarizer as bs_mod

_orig_client, _orig_embedder = bs_mod.bg_client, bs_mod.embedder
bs_mod.bg_client = StubClient()
bs_mod.embedder = StubEmbedder()
bs_mod.get_bg_model_name = lambda: "stub-model"

try:
    # ═══ fixtures ═══════════════════════════════════════════════════════
    # Conversation A: six OLD turns that have NOT decayed (decay_score 1.0).
    # Under the old decay-only predicate none of these would ever compress.
    conv = Conversation(memory_scope_type="auto")
    db.add(conv)
    db.commit()
    conv_id = conv.id

    old_ts = NOW - timedelta(days=settings.batch_summary_age_days + 5)
    for i in range(6):
        t = EpisodicMemory(
            conversation_id=conv_id, batch_id=uuid.uuid4(),
            timestamp=old_ts + timedelta(minutes=i),
            context_reliance="Zero_Shot",
            raw_text=f"{MARK} old undecayed turn {i}",
            decay_score=1.0,               # explicitly NOT decayed
            lossless_flag=False, is_document=False, is_private=False,
            idempotency_key=f"{MARK}-old-{i}",
        )
        db.add(t)
        db.flush()
        turn_ids.append(t.id)

    # Conversation B: six RECENT undecayed turns — the negative side. The
    # widening must not turn into "summarise everything".
    conv_recent = Conversation(memory_scope_type="auto")
    db.add(conv_recent)
    db.commit()
    old_conv_id = conv_recent.id
    for i in range(6):
        t = EpisodicMemory(
            conversation_id=old_conv_id, batch_id=uuid.uuid4(),
            timestamp=NOW - timedelta(hours=i + 1),
            context_reliance="Zero_Shot",
            raw_text=f"{MARK} recent turn {i}",
            decay_score=1.0,
            lossless_flag=False, is_document=False, is_private=False,
            idempotency_key=f"{MARK}-new-{i}",
        )
        db.add(t)
        db.flush()
        turn_ids.append(t.id)
    db.commit()

    def summaries_for(cid):
        return db.query(BatchSummary).filter_by(conversation_id=cid).count()

    def marked(cid):
        return db.query(EpisodicMemory).filter(
            EpisodicMemory.conversation_id == cid,
            EpisodicMemory.batch_summary_id.isnot(None)).count()

    # ═══ 1) the widening ════════════════════════════════════════════════
    print("\n── G11: an OLD but UNDECAYED turn compresses ──")
    check("precondition: turns are old and undecayed",
          db.query(EpisodicMemory).filter(
              EpisodicMemory.conversation_id == conv_id,
              EpisodicMemory.decay_score >= 0.3).count() == 6)

    StubCompletion.calls = 0
    bs_mod.batch_summarize()
    db.expire_all()

    check("old undecayed conversation got a batch summary",
          summaries_for(conv_id) == 1, f"got {summaries_for(conv_id)}")
    check("its turns are stamped with the covering summary",
          marked(conv_id) == 6, f"marked {marked(conv_id)}")
    check("the LLM was called exactly once", StubCompletion.calls == 1,
          f"calls={StubCompletion.calls}")

    # Negative side — recency must still be excluded, or "widening" would just
    # mean "summarise everything the moment it is written".
    check("a RECENT undecayed conversation is NOT summarised",
          summaries_for(old_conv_id) == 0, f"got {summaries_for(old_conv_id)}")
    check("...and its turns carry no marker",
          marked(old_conv_id) == 0)

    # ═══ 2) idempotence — the actual bug ════════════════════════════════
    print("\n── G11: a second pass must do NOTHING ──")
    StubCompletion.calls = 0
    bs_mod.batch_summarize()
    db.expire_all()

    check("second pass created NO new summary",
          summaries_for(conv_id) == 1, f"got {summaries_for(conv_id)}")
    check("second pass called the LLM ZERO times",
          StubCompletion.calls == 0, f"calls={StubCompletion.calls}")

    # ═══ 3) the FK contract ═════════════════════════════════════════════
    print("\n── G11: deleting a summary frees its turns (ON DELETE SET NULL) ──")
    summary_id = db.query(BatchSummary).filter_by(
        conversation_id=conv_id).first().id
    db.query(BatchSummary).filter_by(id=summary_id).delete(
        synchronize_session=False)
    db.commit()
    db.expire_all()

    check("the summary row is gone",
          db.query(BatchSummary).filter_by(id=summary_id).count() == 0)
    check("its turns' markers were SET NULL, not orphaned",
          marked(conv_id) == 0, f"still marked: {marked(conv_id)}")
    # And that means they are eligible again — the property that makes C10's
    # cascade safe rather than merely non-crashing.
    StubCompletion.calls = 0
    bs_mod.batch_summarize()
    db.expire_all()
    check("freed turns are re-summarised on the next pass",
          summaries_for(conv_id) == 1 and StubCompletion.calls == 1,
          f"summaries={summaries_for(conv_id)} calls={StubCompletion.calls}")

finally:
    bs_mod.bg_client, bs_mod.embedder = _orig_client, _orig_embedder
    try:
        for cid in (conv_id, old_conv_id):
            if cid is None:
                continue
            db.query(EpisodicMemory).filter_by(conversation_id=cid).update(
                {"batch_summary_id": None}, synchronize_session=False)
            db.query(BatchSummary).filter_by(conversation_id=cid).delete(
                synchronize_session=False)
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(
                synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(
                synchronize_session=False)
        db.commit()
        # Trap 6: verify, do not assume. Scoped to this run's marker.
        left = db.execute(sql_text(
            "SELECT count(*) FROM episodic_memory WHERE raw_text LIKE :m"),
            {"m": f"%{MARK}%"}).scalar()
        print(f"\ncleanup: {left} fixture turns remaining (want 0)")
    except Exception as exc:
        print(f"cleanup warning: {exc}")
    finally:
        db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
