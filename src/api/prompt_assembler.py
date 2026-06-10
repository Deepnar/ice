"""Context Structural Assembly Plane – builds the final prompt payload,
   now including the sliding window of recent turns."""

from typing import List, Optional
from sqlalchemy.orm import Session
from src.retrieval.orchestrator import ContextFragment
from src.memory.models import MemorySlot, EpisodicMemory

SYSTEM_RULES = (
    "You are an AI assistant with access to a personal memory system (ICE).\n"
    "The following context has been automatically retrieved from past conversations and knowledge.\n"
    "Use it to answer the user's question accurately. If the context is irrelevant, ignore it."
)


def get_recent_turns(db_session: Session, conversation_id: str, n: int = 10) -> List[str]:
    """Return the text of the last N turns from the current conversation."""
    turns = db_session.query(EpisodicMemory).filter_by(
        conversation_id=conversation_id
    ).order_by(EpisodicMemory.timestamp.desc()).limit(n).all()
    turns.reverse()  # chronological order
    fragments = []
    for t in turns:
        if t.inject_raw and t.raw_text:
            text = t.raw_text
        elif t.summary_text:
            text = t.summary_text
        else:
            text = (t.raw_text or "")[:300]
        words = text.split()
        if len(words) > 500:
            text = " ".join(words[:500]) + "…"
        fragments.append(text)
    return fragments


def assemble_prompt(
    memory_slots: List[MemorySlot],
    retrieved_fragments: List[ContextFragment],
    user_message: str,
    db_session: Optional[Session] = None,
    conversation_id: Optional[str] = None,
) -> List[dict]:
    """Assemble the final prompt in stable‑prefix order."""
    system_content = SYSTEM_RULES

    # 0. Recent context (sliding window)
    if db_session and conversation_id:
        recent_texts = get_recent_turns(db_session, conversation_id, n=10)
        if recent_texts:
            system_content += "\n\n=== RECENT CONTEXT ===\n" + "\n\n".join(recent_texts)

    # 1. Persistent Memory Slots
    if memory_slots:
        slot_lines = []
        for slot in memory_slots:
            if slot.is_active and slot.content:
                slot_lines.append(f"[{slot.slot_name.upper()}]\n{slot.content.strip()}")
        if slot_lines:
            system_content += "\n\n=== PERSISTENT CORE PREFERENCES ===\n" + "\n\n".join(slot_lines)

    # 2. Codex (absolute facts)
    codex_frags = [f for f in retrieved_fragments if f.source_type == "codex"]
    if codex_frags:
        codex_text = "\n\n".join(f.text.strip() for f in codex_frags)
        system_content += f"\n\n=== CODEX KNOWLEDGE GRAPH ASSERTIONS ===\n{codex_text}"

    # 3. Episodic context
    episodic_frags = [f for f in retrieved_fragments if f.source_type == "episodic"]
    if episodic_frags:
        episodic_text = "\n\n".join(f.text.strip() for f in episodic_frags)
        system_content += f"\n\n=== RETRIEVED EPISODIC INTERACTIONS ===\n{episodic_text}"

    # 4. Procedural patterns
    procedural_frags = [f for f in retrieved_fragments if f.source_type == "procedural"]
    if procedural_frags:
        proc_text = "\n\n".join(f.text.strip() for f in procedural_frags)
        system_content += f"\n\n=== PROCEDURAL EXECUTION PATTERNS ===\n{proc_text}"

    # 5. RAG chunks
    rag_frags = [f for f in retrieved_fragments if f.source_type == "rag"]
    if rag_frags:
        rag_text = "\n\n".join(f.text.strip() for f in rag_frags)
        system_content += f"\n\n=== REFERENCE MATERIAL ===\n{rag_text}"

    return [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_message},
    ]