"""Reflection proposals use canonical ORM defaults and remain pending review."""

import os
from types import SimpleNamespace as NS

import pytest

from src.api.db import SessionLocal
from src.memory.models import MemorySlot, ReviewQueue
from src.workers import reflection


@pytest.mark.parametrize("kind", ["slot", "motif"])
def test_actual_reflection_review_write_in_orm_created_schema(monkeypatch, kind):
    assert os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
    value = ('{"guidance":"Review the evidence."}' if kind == "slot"
             else '["Cobalt observations"]')

    def create(**kwargs):
        assert kwargs["response_format"]["json_schema"]
        return NS(choices=[NS(finish_reason="stop", message=NS(content=value))])

    monkeypatch.setattr(reflection, "bg_client", NS(chat=NS(completions=NS(create=create))))
    with SessionLocal() as db:
        before_slots = db.query(MemorySlot).count()
        before_ids = {r.id for r in db.query(ReviewQueue)}
        producer = (reflection._evolve_memory_slots if kind == "slot"
                    else reflection._detect_motifs)
        producer(db, [NS(raw_text="Synthetic observation.")])
        db.flush()
        row = db.query(ReviewQueue).filter(~ReviewQueue.id.in_(before_ids)).one()
        rid = row.id
        try:
            db.commit()
            db.expire_all()
            row = db.get(ReviewQueue, rid)
            assert row.id is not None and row.status == "pending" and row.created_at is not None
            expected = ({"slot_name": "guidance", "proposed_content": "Review the evidence.",
                         "proposed_by": "reflection"} if kind == "slot"
                        else {"cluster_name": "Cobalt observations"})
            assert row.item_content == expected
            assert row.item_type == ("memory_slot_update" if kind == "slot"
                                     else "new_cluster_proposal")
            assert db.query(MemorySlot).count() == before_slots
        finally:
            db.rollback()
            db.query(ReviewQueue).filter_by(id=rid).delete(synchronize_session=False)
            db.commit()
