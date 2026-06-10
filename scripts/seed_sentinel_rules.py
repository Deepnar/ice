#!/usr/bin/env python3
"""Insert a few default sentinel rules into the database."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.memory.models import SentinelRule

rules = [
    {
        "name": "Stale pending items",
        "description": "If pending_items slot has content older than 14 days and no recent retrieval, notify.",
        "is_active": True,
        "trigger_type": "absence",
        "trigger_conditions": '{"table": "memory_slots", "field": "content", "key": "pending_items", "max_age_days": 14}',
        "action_type": "notify",
        "action_payload": '{"message": "Pending items may be stale – review them."}',
        "cooldown_seconds": 86400
    },
    {
        "name": "High contradiction entity",
        "description": "If a Codex entity has >3 pending edges and >2 active edges overlapping, create review item.",
        "is_active": True,
        "trigger_type": "threshold",
        "trigger_conditions": '{"entity": true, "min_pending_edges": 3, "min_active_overlap": 2}',
        "action_type": "create_review_item",
        "action_payload": '{"item_type": "codex_contradiction"}',
        "cooldown_seconds": 43200
    },
    {
        "name": "Retrieval health degradation",
        "description": "If 5 consecutive Long_Term_Memory turns return zero results, schedule clustering.",
        "is_active": True,
        "trigger_type": "threshold",
        "trigger_conditions": '{"consecutive_zero_retrieval": 5}',
        "action_type": "schedule_worker",
        "action_payload": '{"worker": "src.workers.clustering.cluster_turns"}',
        "cooldown_seconds": 3600
    }
]

db = SessionLocal()
for r in rules:
    existing = db.query(SentinelRule).filter_by(name=r["name"]).first()
    if not existing:
        db.add(SentinelRule(**r))
db.commit()
db.close()
print("Default sentinel rules inserted.")