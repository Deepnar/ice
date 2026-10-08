"""A short reply cannot strand half the allowance while a late source fact is cut."""

from types import SimpleNamespace as NS

from src.api import prompt_assembler as assembler
from src.memory.source import chat_provenance
from src.memory.tokens import count


def test_short_assistant_share_preserves_complete_long_user_under_same_cap(monkeypatch):
    user = "Background discussion. " * 350 + "Only if approved: use Redis. Approval was refused."
    assistant = "Keep Redis disabled."
    row = NS(raw_text=f"User: {user}\n\nAssistant: {assistant}",
             source_spans=chat_provenance(user, assistant), summary_text=None,
             abstract_text=None, inject_raw=True, timestamp=None, ts_provenance="unknown")
    monkeypatch.setattr(assembler, "_recent_turn_rows", lambda *a: [row])
    messages = assembler.get_recent_turns(None, "test", max_tokens=4000)
    assert user in messages[0]["content"] and assistant == messages[1]["content"]
    assert "Approval was refused" in messages[0]["content"]
    assert sum(count(m["content"]) for m in messages) <= 1400


def test_short_user_share_preserves_complete_long_assistant_under_same_cap(monkeypatch):
    user = "Explain the policy."
    assistant = "Background discussion. " * 350 + "Final answer: do not deploy Redis."
    row = NS(raw_text=f"User: {user}\n\nAssistant: {assistant}",
             source_spans=chat_provenance(user, assistant), summary_text=None,
             abstract_text=None, inject_raw=True, timestamp=None, ts_provenance="unknown")
    monkeypatch.setattr(assembler, "_recent_turn_rows", lambda *a: [row])
    messages = assembler.get_recent_turns(None, "test", max_tokens=4000)
    assert assistant == messages[1]["content"]
    assert sum(count(m["content"]) for m in messages) <= 1400
