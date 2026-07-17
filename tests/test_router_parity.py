"""E0 router-parity harness (record-and-compare, spec §5 check 3).

Records the exact HTTP responses of every memory_slots + user_control endpoint
on seeded fixtures, then compares a later run against that recording — the
gate proving the E0 service extraction left the REST surface byte-identical
(after normalizing run-varying UUIDs/timestamps).

    uv run python tests/test_router_parity.py --record    # before extraction
    uv run python tests/test_router_parity.py --compare   # after extraction

Runs against the live Postgres. Inserts its own uniquely-marked rows and
deletes them afterwards — never truncates (the dev DB holds real data). The
model-registry endpoints are pointed at a temp registry file. The two live
memory slots it touches (via PUT/initialize) are snapshot-and-restored, so a
run leaves the DB exactly as found and the harness is idempotent. The baseline
JSON contains live-DB content (slots, review queue), so it lives in the
gitignored logs/ — never commit it.

No LLM, no classifier, no runtime: the harness mounts the two routers on a
bare FastAPI app (no lifespan), so the bookmark endpoint exercises its
runtime-absent branch in both modes.
"""
import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.model_registry.registry as reg_mod
from src.api.db import SessionLocal
from src.memory.models import (
    Conversation,
    ContextCluster,
    CuratedLabel,
    EpisodicMemory,
    MemorySlot,
    ReviewQueue,
)

BASELINE = os.path.join(os.path.dirname(__file__), "..", "logs",
                        "router_parity_baseline.json")
MARK = "parityE0"  # FIXED marker — both runs must see identical state

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[0-9:.+\-Z]*")


def normalize(body: str, step: str = "") -> str:
    body = UUID_RE.sub("<UUID>", body)
    body = ISO_RE.sub("<TS>", body)
    if step == "review_list":
        # the endpoint has no ORDER BY — Postgres row order is unspecified and
        # varies between runs; compare as a sorted set of items instead
        items = json.loads(body)
        body = json.dumps(sorted(items, key=lambda i: json.dumps(i, sort_keys=True)))
    return body


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    # Registry endpoints work on a temp file, not the real registry.
    tmp_registry = os.path.join("/tmp", f"ice_{MARK}_registry.json")
    with open(tmp_registry, "w") as f:
        json.dump({"models": {"parity-model:1b": {
            "topic_tags": ["Software_&_Tech"], "intent_tags": ["Generation"],
            "priority": 5, "context_window": 8192, "confirmed": True,
            "base_url": None, "added_at": 0,
        }}, "updated_at": 0}, f)
    reg_mod.REGISTRY_PATH = tmp_registry

    # Routers imported AFTER the registry patch (they bind functions, the
    # functions read REGISTRY_PATH at call time — patch works either way).
    from src.api.routers import memory_slots, user_control

    app = FastAPI()
    app.include_router(memory_slots.router)
    app.include_router(user_control.router)
    client = TestClient(app)

    db = SessionLocal()
    created_slot_names: list[str] = []
    slot_snapshot = None
    recorded = []

    def call(step, method, url, **kw):
        resp = client.request(method, url, **kw)
        recorded.append({
            "step": step, "method": method, "url": url,
            "status": resp.status_code,
            "content_type": resp.headers.get("content-type", ""),
            "body": normalize(resp.text, step),
        })
        return resp

    try:
        # ── Fixtures ────────────────────────────────────────────────────
        conv = Conversation(memory_scope_type="auto")
        conv_empty = Conversation(memory_scope_type="auto")
        db.add_all([conv, conv_empty])
        db.commit()
        batch_id = uuid.uuid4()
        turns = []
        for i in range(2):
            t = EpisodicMemory(
                conversation_id=conv.id, batch_id=batch_id,
                context_reliance="Zero_Shot",
                raw_text=f"{MARK} turn {i} raw text",
                timestamp=datetime(2026, 7, 1, 12, i, tzinfo=timezone.utc),
                idempotency_key=f"{MARK}-{uuid.uuid4()}",
            )
            db.add(t)
            turns.append(t)
        slot_row = MemorySlot(slot_name=f"slot_{MARK}", content="old",
                              token_count=1, version=1,
                              last_updated=datetime.now(timezone.utc),
                              updated_by="user", is_active=False)
        rq_slot = ReviewQueue(item_type="memory_slot_update", item_content={
            "slot_name": f"slot_{MARK}", "proposed_content": f"{MARK} proposed"})
        rq_cluster = ReviewQueue(item_type="new_cluster_proposal",
                                 item_content={"cluster_name": f"cluster_{MARK}"})
        rq_sentinel = ReviewQueue(item_type="sentinel_review",
                                  item_content={"note": MARK})
        db.add_all([slot_row, rq_slot, rq_cluster, rq_sentinel])
        db.commit()
        turn_ids = [str(t.id) for t in turns]
        conv_id, conv_empty_id = str(conv.id), str(conv_empty.id)
        rq_ids = [str(rq_slot.id), str(rq_cluster.id), str(rq_sentinel.id)]

        # snapshot the one live slot the PUT touches
        live = db.query(MemorySlot).filter_by(slot_name="session_patterns").first()
        if live:
            slot_snapshot = {c: getattr(live, c) for c in
                             ("content", "token_count", "version",
                              "last_updated", "updated_by", "is_active")}

        # ── memory_slots ────────────────────────────────────────────────
        r = call("init", "POST", "/memory-slots/initialize")
        created_slot_names = r.json().get("created", [])
        call("slots_list", "GET", "/memory-slots/")
        call("slot_get", "GET", "/memory-slots/persona")
        call("slot_get_bad_name", "GET", "/memory-slots/not_a_slot")
        call("slot_put", "PUT", "/memory-slots/session_patterns",
             json={"content": f"{MARK} parity content"})
        call("slot_put_bad_name", "PUT", "/memory-slots/not_a_slot",
             json={"content": "x"})

        # ── user_control ────────────────────────────────────────────────
        call("bookmark", "POST", f"/user-control/turns/{turn_ids[0]}/bookmark")
        call("bookmark_404", "POST",
             f"/user-control/turns/{uuid.uuid4()}/bookmark")
        call("bookmarks_list", "GET",
             f"/user-control/bookmarks?conversation_id={conv_id}")
        call("override_tags", "POST", "/user-control/batch/override-tags",
             json={"batch_id": str(batch_id), "topic_labels": ["Software_&_Tech"],
                   "intent_labels": ["Generation"],
                   "context_reliance": "Zero_Shot"})
        call("scope_put_project", "PUT", f"/user-control/conversations/{conv_id}/scope",
             json={"memory_scope_type": "project", "cluster_ids": []})
        call("scope_get", "GET", f"/user-control/conversations/{conv_id}/scope")
        call("scope_put_none", "PUT", f"/user-control/conversations/{conv_id}/scope",
             json={"memory_scope_type": "none"})
        call("scope_put_auto", "PUT", f"/user-control/conversations/{conv_id}/scope",
             json={"memory_scope_type": "auto"})
        call("scope_404", "GET", f"/user-control/conversations/{uuid.uuid4()}/scope")
        r = call("cluster_create", "POST", "/user-control/clusters",
                 json={"name": f"created_{MARK}", "description": "parity"})
        cluster_id = r.json()["id"]
        call("cluster_assign", "PUT", f"/user-control/clusters/{cluster_id}/assign",
             json={"turn_ids": turn_ids})
        call("cluster_assign_404", "PUT",
             f"/user-control/clusters/{uuid.uuid4()}/assign",
             json={"turn_ids": turn_ids})
        call("review_list", "GET", "/user-control/review-queue?status=pending")
        call("review_approve_slot", "POST",
             f"/user-control/review-queue/{rq_ids[0]}/approve")
        call("review_approve_cluster", "POST",
             f"/user-control/review-queue/{rq_ids[1]}/approve")
        call("review_approve_sentinel", "POST",
             f"/user-control/review-queue/{rq_ids[2]}/approve")
        call("review_approve_404", "POST",
             f"/user-control/review-queue/{uuid.uuid4()}/approve")
        call("latest_turn", "GET",
             f"/user-control/conversations/{conv_id}/latest-turn")
        call("latest_turn_404", "GET",
             f"/user-control/conversations/{conv_empty_id}/latest-turn")

        # ── model registry (temp file) ──────────────────────────────────
        call("registry_get", "GET", "/user-control/model-registry")
        call("registry_put", "PUT", "/user-control/model-registry/parity-model:1b",
             json={"topic_tags": ["Creative_&_Media"], "confirmed": True})
        call("registry_put_404", "PUT", "/user-control/model-registry/nope",
             json={"topic_tags": []})
        call("registry_get_after_put", "GET", "/user-control/model-registry")
        call("registry_delete", "DELETE",
             "/user-control/model-registry/parity-model:1b")
        call("registry_delete_404", "DELETE", "/user-control/model-registry/nope")

    finally:
        db.rollback()
        # Marker-keyed cleanup — restore the DB exactly as found.
        db.query(ReviewQueue).filter(
            ReviewQueue.item_content["slot_name"].astext == f"slot_{MARK}").delete(
            synchronize_session=False)
        db.query(ReviewQueue).filter(
            ReviewQueue.item_content["cluster_name"].astext == f"cluster_{MARK}").delete(
            synchronize_session=False)
        db.query(ReviewQueue).filter(
            ReviewQueue.item_content["note"].astext == MARK).delete(
            synchronize_session=False)
        db.query(CuratedLabel).filter_by(batch_id=batch_id).delete(
            synchronize_session=False)
        # turns before clusters (assign points cluster_id at the created one)
        db.query(EpisodicMemory).filter(
            EpisodicMemory.raw_text.like(f"{MARK}%")).delete(
            synchronize_session=False)
        db.query(ContextCluster).filter(
            ContextCluster.name.in_([f"created_{MARK}", f"cluster_{MARK}"])).delete(
            synchronize_session=False)
        db.query(Conversation).filter(
            Conversation.id.in_([uuid.UUID(conv_id), uuid.UUID(conv_empty_id)])).delete(
            synchronize_session=False)
        db.query(MemorySlot).filter_by(slot_name=f"slot_{MARK}").delete(
            synchronize_session=False)
        for name in created_slot_names:
            db.query(MemorySlot).filter_by(slot_name=name).delete(
                synchronize_session=False)
        if slot_snapshot and "session_patterns" not in created_slot_names:
            live = db.query(MemorySlot).filter_by(
                slot_name="session_patterns").first()
            if live:
                for k, v in slot_snapshot.items():
                    setattr(live, k, v)
        db.commit()
        db.close()
        os.unlink(tmp_registry)

    if args.record:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w") as f:
            json.dump(recorded, f, indent=1)
        print(f"Recorded {len(recorded)} responses -> {BASELINE}")
        return 0

    with open(BASELINE) as f:
        baseline = json.load(f)
    failures = 0
    base_by_step = {b["step"]: b for b in baseline}
    for cur in recorded:
        base = base_by_step.get(cur["step"])
        if base is None:
            print(f"  FAIL  {cur['step']}: not in baseline")
            failures += 1
            continue
        for field in ("status", "content_type", "body"):
            if base[field] != cur[field]:
                print(f"  FAIL  {cur['step']}.{field}:\n"
                      f"    baseline: {base[field]!r}\n"
                      f"    current : {cur[field]!r}")
                failures += 1
                break
        else:
            print(f"  PASS  {cur['step']}")
    missing = set(base_by_step) - {c["step"] for c in recorded}
    for step in sorted(missing):
        print(f"  FAIL  {step}: recorded in baseline but not exercised now")
        failures += 1
    print(f"\n{len(recorded) - failures}/{len(recorded)} parity checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
