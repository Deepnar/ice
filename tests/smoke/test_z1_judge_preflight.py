"""Paired answer judgements require complete, matched source and models."""

import json
import sys

import pytest

from scripts.z1 import judge_answers


def _record(conversation="first", *, model="gpt-6-luna", source="full source"):
    return {
        "conversation": conversation, "gold_turns": [1], "question": "Same wording?",
        "gold_source_complete": True, "gold_turn_text": source,
        "answer_model": model, "answer_profile": "opencode-luna6",
        "answer": "supported answer", "error": None,
    }


def _run(tmp_path, monkeypatch, left, right):
    paths = []
    for name, records in (("a", left), ("b", right)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"tag": name, "records": records}))
        paths.append(path)
    monkeypatch.setattr(judge_answers, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["judge_answers.py", "--a", str(paths[0]),
                                      "--b", str(paths[1])])
    return judge_answers.main()


def test_same_question_in_two_conversations_stays_two_pairs(tmp_path, monkeypatch):
    seen = []
    def judge(_question, source, _a, _b):
        seen.append(source)
        return {"verdict": "TIE", "reason": "equivalent"}
    monkeypatch.setattr(judge_answers, "judge_one", judge)
    left = [_record("first", source="source one"),
            _record("second", source="source two")]
    assert _run(tmp_path, monkeypatch, left, list(reversed(left))) == 0
    assert seen == ["source one", "source two"]


def test_mismatched_model_or_incomplete_source_blocks_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_answers, "judge_one",
                        lambda *_args: pytest.fail("judge called before preflight"))
    with pytest.raises(ValueError, match="different answering models"):
        _run(tmp_path, monkeypatch, [_record()], [_record(model="other")])
    with pytest.raises(RuntimeError, match="Complete gold source unavailable"):
        old = _record()
        old.pop("gold_source_complete")
        old["gold_turn_text"] = "prefix only"
        monkeypatch.setattr(judge_answers, "_GOLD_IDX", {})
        _run(tmp_path, monkeypatch, [old], [old])
