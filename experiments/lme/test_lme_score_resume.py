from __future__ import annotations

import json
import sys
from pathlib import Path


LME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LME_DIR))

import score


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_retry_mutes_requeues_only_selected_mute_arm(tmp_path):
    records = [
        {"question_id": "q1", "answers": {"full_ice": {}, "vector_rag": {}}},
        {"question_id": "q2", "answers": {"full_ice": {}, "vector_rag": {}}},
    ]
    _write(tmp_path / "q1__full_ice.json", {"condition": "full_ice",
                                             "spoke": False, "judge_raw": ""})
    _write(tmp_path / "q1__vector_rag.json", {"condition": "vector_rag",
                                               "spoke": True, "judge_raw": "yes"})
    _write(tmp_path / "q2__full_ice.json", {"condition": "full_ice",
                                             "spoke": True, "judge_raw": "no"})

    judged, pending = score.partition_judgements(
        records, tmp_path, retry_mutes=True, retry_condition="full_ice"
    )

    assert {(record["question_id"], condition) for record, condition in pending} == {
        ("q1", "full_ice")
    }
    assert {(entry["condition"], entry["judge_raw"]) for entry in judged} == {
        ("vector_rag", "yes"), ("full_ice", "no")
    }


def test_normal_resume_requeues_only_missing_files(tmp_path):
    records = [
        {"question_id": "q1", "answers": {"full_ice": {}, "vector_rag": {}}},
    ]
    _write(tmp_path / "q1__full_ice.json", {"condition": "full_ice",
                                             "spoke": False, "judge_raw": ""})

    judged, pending = score.partition_judgements(records, tmp_path)

    assert len(judged) == 1  # a mute is existing missing data, not missing work
    assert [(record["question_id"], condition)
            for record, condition in pending] == [("q1", "vector_rag")]
