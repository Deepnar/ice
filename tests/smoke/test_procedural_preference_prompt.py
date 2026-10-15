"""Audit A10 (2026-10-15): procedural memory holds answer preferences, scoped by topic.

Measured: the earlier prompt stored "the user seeks validation" paraphrases (8
of them filled procedure notes in 37/67 prompts) and personality notes; and no
pattern ever carried trigger topic tags, so every pattern matched every prompt.

Run:  uv run --no-sync pytest tests/smoke/test_procedural_preference_prompt.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.workers.procedural_extractor import cited_topics, pattern_prompt  # noqa: E402


def test_prompt_targets_answer_preferences_without_copyable_examples():
    p = pattern_prompt("1. list them\n2. give me a final list", 2)
    assert "answer preference" in p and "NONE" in p and "EVIDENCE" in p
    assert "Examples:" not in p, "example sentences were copied verbatim 8/25 in qualification"


def test_cited_topics_are_only_the_cited_turns_tags():
    turns = [NS(batch_id="a", topic_tags=["Software_&_Tech"]),
             NS(batch_id="b", topic_tags=["Career_&_Education", "Software_&_Tech"]),
             NS(batch_id="c", topic_tags=["Creative_&_Media"])]
    assert cited_topics(turns, {"a", "b"}) == ["Career_&_Education", "Software_&_Tech"]
    assert cited_topics(turns, set()) == []
