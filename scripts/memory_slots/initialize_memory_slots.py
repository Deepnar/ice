#!/usr/bin/env python3
"""Initialise the seven default memory slots (if they don't already exist)."""

import sys, os
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.memory.models import MemorySlot

VALID_SLOTS = [
    "persona",
    "user_preferences",
    "tool_guidelines",
    "project_context",
    "guidance",
    "pending_items",
    "session_patterns",
]

db = SessionLocal()
created = []
for name in VALID_SLOTS:
    existing = db.query(MemorySlot).filter_by(slot_name=name).first()
    if not existing:
        slot = MemorySlot(
            slot_name=name,
            content="",
            token_count=0,
            version=1,
            last_updated=datetime.now(timezone.utc),
            updated_by="system",
            is_active=True,
        )
        db.add(slot)
        created.append(name)
        print(f"  Created slot '{name}'")
    else:
        print(f"  Slot '{name}' already exists – skipping")

db.commit()
db.close()
print(f"\nDone. Created {len(created)} new slots.")