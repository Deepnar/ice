"""Shared warm-turn history pressure for chat and explicit context pulls."""

import uuid

from sqlalchemy import func

from src.memory.models import EpisodicMemory
from src.memory.tokens import estimate_from_chars


def conversation_pressure(db, conversation_id) -> tuple[int, float]:
    """Return turn count and approximate tokens for the current conversation.

    B2 uses the approximation only to estimate history beyond the recent
    window; actual prompt spending uses the shared tokenizer separately.
    """
    if conversation_id is None:
        return 0, 0.0
    conv_uuid = uuid.UUID(str(conversation_id))
    count, chars = db.query(
        func.count(EpisodicMemory.id),
        func.coalesce(func.sum(func.length(EpisodicMemory.raw_text)), 0),
    ).filter(EpisodicMemory.conversation_id == conv_uuid).one()
    return int(count), estimate_from_chars(int(chars))
