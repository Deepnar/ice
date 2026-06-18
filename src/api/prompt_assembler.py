"""Context Structural Assembly Plane – builds the final prompt payload,
   now including sliding window and bookmarked memories."""

from typing import List, Optional
from sqlalchemy.orm import Session
from src.retrieval.orchestrator import ContextFragment
from src.memory.models import MemorySlot, EpisodicMemory



BASE_SYSTEM_RULES = (
    "You are an AI assistant with access to a personal memory system (ICE).\n"
    "Below is context retrieved from past conversations.\n\n"
    "RULES:\n"
    "1. Answer based ONLY on the context provided below.\n"
    "2. If the context does NOT contain the answer, say so naturally.\n"
    "3. Do NOT add details not present in the context.\n"
    "4. Write naturally. Don't over-explain and don't add citations.\n"
)



INTENT_INSTRUCTIONS = {
    "Factual_Retrieval": (
        "Be precise. List facts, names, and numbers exactly as they appear in the context."
    ),
    "Troubleshooting": (
        "Describe the problem and solution exactly as they appear in the context."
    ),
    "Generation": (
        "Generate based ONLY on what's in the context. Follow examples and templates exactly."
    ),
    "Emotional_Processing": (
        "Respond with empathy, but base your response ONLY on the context. "
        "If the context doesn't mention a specific memory, don't invent it."
    ),
}


def get_intent_instruction(intent_tags: List[str]) -> str:
    """Return the appropriate instruction based on the first matching intent."""
    for intent in intent_tags:
        if intent in INTENT_INSTRUCTIONS:
            return INTENT_INSTRUCTIONS[intent]
    return ""


def get_recent_turns(db_session: Session, conversation_id: str, n: int = 10) -> List[str]:
    """Return the text of the last N turns from the current conversation."""
    turns = db_session.query(EpisodicMemory).filter_by(
        conversation_id=conversation_id
    ).order_by(EpisodicMemory.timestamp.desc()).limit(n).all()
    turns.reverse()
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
    bookmarked_texts: Optional[List[str]] = None,
    classification=None,
) -> List[dict]:
    """Assemble the final prompt with safety-first instructions."""


    if classification and (
        "Emotional_Processing" in classification.intent_tags
        or "Social_&_Relationships" in classification.topic_tags
        or "Creative_&_Media" in classification.topic_tags
    ):
        context_texts = []
        if bookmarked_texts:
            context_texts.append("=== BOOKMARKED MEMORIES ===\n" + "\n\n".join(bookmarked_texts))

        for f in retrieved_fragments:
            context_texts.append(f.text)

        plain_context = "\n\n".join(context_texts) if context_texts else "No relevant context retrieved."

        intent_instruction = get_intent_instruction(classification.intent_tags)

        system_message = (
            "You are a personal AI assistant with access to past conversations.\n\n"
            "RULES:\n"
            "1. Answer based ONLY on the context below.\n"
            "2. If the context does NOT contain the answer, say so naturally.\n"
            "3. Do NOT invent memories or details.\n"
            "4. Write naturally and conversationally.\n"
        )

        if intent_instruction:
            system_message += f"\nINTENT: {intent_instruction}\n"

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{plain_context}\n\nQuestion: {user_message}"}
        ]



    system_content = BASE_SYSTEM_RULES

    if classification:
        intent_instruction = get_intent_instruction(classification.intent_tags)
        if intent_instruction:
            system_content += f"\n\nINTENT: {intent_instruction}\n"

    # 0. Bookmarked memories
    if bookmarked_texts:
        system_content += "\n\n=== BOOKMARKED MEMORIES ===\n" + "\n\n".join(bookmarked_texts)

    # 1. Persistent Memory Slots
    if memory_slots:
        slot_lines = []
        for slot in memory_slots:
            if slot.is_active and slot.content:
                slot_lines.append(f"[{slot.slot_name.upper()}]\n{slot.content.strip()}")
        if slot_lines:
            system_content += "\n\n=== PERSISTENT CORE PREFERENCES ===\n" + "\n\n".join(slot_lines)

    # 2. Recent context (sliding window)
    if db_session and conversation_id:
        recent_texts = get_recent_turns(db_session, conversation_id, n=10)
        if recent_texts:
            system_content += "\n\n=== RECENT CONTEXT ===\n" + "\n\n".join(recent_texts)

    # 3. Codex (absolute facts)
    codex_frags = [f for f in retrieved_fragments if f.source_type == "codex"]
    if codex_frags:
        codex_text = "\n\n".join(f.text.strip() for f in codex_frags)
        system_content += f"\n\n=== CODEX KNOWLEDGE ===\n{codex_text}"

    # 4. Episodic context
    episodic_frags = [f for f in retrieved_fragments if f.source_type == "episodic"]
    if episodic_frags:
        episodic_text = "\n\n".join(f.text.strip() for f in episodic_frags)
        system_content += f"\n\n=== PAST INTERACTIONS ===\n{episodic_text}"

    # 5. Procedural patterns
    procedural_frags = [f for f in retrieved_fragments if f.source_type == "procedural"]
    if procedural_frags:
        proc_text = "\n\n".join(f.text.strip() for f in procedural_frags)
        system_content += f"\n\n=== PATTERNS ===\n{proc_text}"

    # 6. RAG chunks
    rag_frags = [f for f in retrieved_fragments if f.source_type == "rag"]
    if rag_frags:
        rag_text = "\n\n".join(f.text.strip() for f in rag_frags)
        system_content += f"\n\n=== REFERENCE MATERIAL ===\n{rag_text}"

    system_content += (
        "\n\nREMINDER: Answer naturally and conversationally. "
        "If you don't know something, say so honestly. "
        "Don't add details not in the context."
    )

    return [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_message},
    ]