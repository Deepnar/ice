"""G77 decision 1 — reflection writes only the conversation's own slots.

Fails on the pre-G77 code, which queued GLOBAL slot proposals and motif
clusters as `pending`. The NLI verifier and the background model are stubbed
(isolation); the production path is the real-model probe in PROVENANCE.
"""

from types import SimpleNamespace as NS

import pytest

from src.memory.models import MemorySlot, ReviewQueue
from src.services import review
from src.services.errors import ValidationError
from src.workers import reflection
# needs ICE_TEST_DATABASE (via the shared db fixture)
from test_g77_self_applied import MARK, _conv_slot, _conversation, db  # noqa: F401


def _stub_model(monkeypatch, payload, open_check=None):
    """Answers the slot call with *payload* and the open-item check with
    *open_check* (default: every item pending, none closed)."""
    import json
    seen = {}

    def create(**kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        if name == "open_check":
            n = kwargs["response_format"]["json_schema"]["schema"]["properties"]["notes"]["maxItems"]
            seen["open_check"] = kwargs["messages"][-1]["content"]
            if isinstance(open_check, Exception):
                raise open_check
            body = {"notes": open_check or [{"pending": True, "closed_later": False}] * n}
        else:
            seen["schema"] = kwargs["response_format"]["json_schema"]
            seen["prompt"] = kwargs["messages"][-1]["content"]
            body = payload
        return NS(choices=[NS(finish_reason="stop", message=NS(content=json.dumps(body)))])

    monkeypatch.setattr(reflection, "bg_client", NS(chat=NS(completions=NS(create=create))))
    return seen


def test_reflection_writes_only_this_conversations_slots_and_undo_restores(db, monkeypatch):
    global_before = {(s.slot_name, s.content, s.version)
                     for s in db.query(MemorySlot).filter_by(scope_tier="global")}
    conv, turns = _conversation(db, [
        (f"I am migrating the {MARK} billing service to Postgres.", "Sounds good."),
        ("I still need to decide on the index for invoices.", "A btree on customer_id works."),
    ])
    payload = {
        "focus": [{"sources": ["w0"], "quote": "migrating the", "speaker": "user",
                   "text": f"The user is migrating the {MARK} billing service to Postgres."}],
        "open": [{"sources": ["w2"], "quote": "decide on the index for invoices", "speaker": "user",
                  "text": f"The user still needs to decide on the {MARK} invoice index."},
                 {"sources": ["w2"], "quote": "words that are not there", "speaker": "user",
                  "text": "The user wants a cache."},
                 {"sources": ["w0"], "quote": "migrating the", "speaker": "user",
                  "text": "UNSUPPORTED The user finished the migration."}],
    }
    seen = _stub_model(monkeypatch, payload)
    monkeypatch.setattr(reflection, "_verify", lambda premise, claim: {
        "status": "unknown" if claim.startswith("UNSUPPORTED") else "supported",
        "scores": {"entailment": 0.9}, "reason": "stub"})

    out = reflection._refresh_conversation_slots(db, conv.id, turns)

    # The model saw full turns as windows, not 200-character cuts.
    assert "decide on the index for invoices" in seen["prompt"]
    assert set(seen["schema"]["schema"]["properties"]) == {"focus", "open"}
    focus = _conv_slot(db, conv.id, "conversation_focus")
    pending = _conv_slot(db, conv.id, "pending_items")
    assert focus.content == payload["focus"][0]["text"] and focus.updated_by == "reflection"
    assert pending.content == payload["open"][0]["text"]
    assert sorted(d["reason"] for d in out["dropped"]) == ["quote_not_in_source", "unknown"]
    # No global slot was touched, and nothing waits on a person.
    assert {(s.slot_name, s.content, s.version)
            for s in db.query(MemorySlot).filter_by(scope_tier="global")} == global_before
    rows = db.query(ReviewQueue).filter(
        ReviewQueue.item_content["conversation_id"].astext == str(conv.id)).all()
    assert {r.status for r in rows} == {"applied"} and len(rows) == 2
    assert not db.query(ReviewQueue).filter_by(item_type="new_cluster_proposal",
                                                status="pending").filter(
        ReviewQueue.item_content["cluster_name"].astext.like(f"%{MARK}%")).count()

    # The newest turn is already reflected on: no second model call.
    monkeypatch.setattr(reflection, "bg_client", None)
    assert reflection._refresh_conversation_slots(db, conv.id, turns)["reason"] == "no_new_turns"

    # One call undoes the write; a second undo is refused.
    focus_row = next(r for r in rows if r.item_content["slot_name"] == "conversation_focus")
    assert review.undo(db, str(focus_row.id))["status"] == "undone"
    db.expire_all()
    assert _conv_slot(db, conv.id, "conversation_focus").is_active is False
    with pytest.raises(ValidationError):
        review.undo(db, str(focus_row.id))
    assert not hasattr(reflection, "_detect_motifs")


def _two_open_items(db, monkeypatch, open_check):
    conv, turns = _conversation(db, [
        (f"Should the {MARK} cache be Redis or memcached?", "Want me to compare them?"),
        ("Not now.", "Fine. In short, both work."),
    ])
    payload = {"focus": [], "open": [
        {"sources": ["w0"], "quote": "Redis or memcached", "speaker": "user",
         "text": f"The user has not yet chosen between Redis and memcached for the {MARK} cache."},
        {"sources": ["w3"], "quote": "both work", "speaker": "assistant",
         "text": "The assistant concluded that both work."}]}
    seen = _stub_model(monkeypatch, payload, open_check)
    monkeypatch.setattr(reflection, "_verify", lambda premise, claim: {
        "status": "supported", "scores": {"entailment": 0.9}, "reason": "stub"})
    out = reflection._refresh_conversation_slots(db, conv.id, turns)
    return conv, out, seen


def test_open_check_drops_items_that_are_not_pending_or_were_closed(db, monkeypatch):
    conv, out, seen = _two_open_items(db, monkeypatch, [
        {"pending": True, "closed_later": False}, {"pending": False, "closed_later": False}])
    assert "1. The user has not yet chosen" in seen["open_check"]
    assert _conv_slot(db, conv.id, "pending_items").content.startswith("The user has not yet chosen")
    assert [d["reason"] for d in out["dropped"]] == ["not_pending_or_closed"]


def test_failed_open_check_writes_no_open_item(db, monkeypatch):
    conv, out, _ = _two_open_items(db, monkeypatch, RuntimeError("model down"))
    assert _conv_slot(db, conv.id, "pending_items") is None
    assert len(out["dropped"]) == 2
