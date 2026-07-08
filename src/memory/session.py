"""Session identity (C6): a conversation is a series of *sittings*.

A new turn belongs to the previous turn's session while the silence between
them is at most ``settings.session_gap_minutes``; a longer gap opens a new
session. The session_id is the primitive that session-aware clustering (C5),
the session-gap maintenance trigger (C7), and the cross-chat scoping UI (C6-F)
hang off — kept in its own light module so tests and workers can import it
without dragging in the API app (and its model loads).
"""

import uuid
from datetime import timedelta


def resolve_session_id(db, conversation_id, now, gap_minutes: int):
    """Return ``(session_id, session_started)`` for a turn written at *now*.

    Reuses the conversation's last session when the gap since its latest turn
    is within *gap_minutes*; otherwise mints a new session. Rows that predate
    the C6 migration have NULL session_id — a new session starts after them
    rather than guessing a backfill.
    """
    from src.memory.models import EpisodicMemory

    last = (
        db.query(EpisodicMemory.timestamp, EpisodicMemory.session_id)
        .filter(EpisodicMemory.conversation_id == conversation_id)
        .order_by(EpisodicMemory.timestamp.desc())
        .first()
    )
    if (
        last is not None
        and last.session_id is not None
        and last.timestamp is not None
        and (now - last.timestamp) <= timedelta(minutes=gap_minutes)
    ):
        return last.session_id, False
    return uuid.uuid4(), True
