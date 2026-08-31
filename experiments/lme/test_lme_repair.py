from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path


LME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LME_DIR))

import lme_run
import repair_invalid_adapter


def test_invalid_adapter_repair_is_reversible_and_preserves_vector(tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(repair_invalid_adapter, "DEFAULT_OUT", tmp_path)
    phase_dir = tmp_path / "oracle"
    answers_dir = phase_dir / "answers"
    judgements_dir = phase_dir / "judgements"
    answers_dir.mkdir(parents=True)
    judgements_dir.mkdir()

    original = {
        "question_id": "q1",
        "conversation_id": str(uuid.uuid4()),
        "status": "complete",
        "answers": {
            "full_ice": {"answer": "invalid", "fragments": 3},
            "vector_rag": {"answer": "valid", "fragments": 30},
        },
    }
    answer_path = answers_dir / "q1.json"
    answer_path.write_text(json.dumps(original))
    (judgements_dir / "q1__full_ice.json").write_text("{}")
    (judgements_dir / "q1__vector_rag.json").write_text("{}")
    (phase_dir / "MANIFEST.json").write_text("{}")

    dry = repair_invalid_adapter.repair_phase("oracle", apply=False)
    assert dry["answers_to_repair"] == 1
    assert json.loads(answer_path.read_text()) == original

    applied = repair_invalid_adapter.repair_phase("oracle", apply=True)
    repaired = json.loads(answer_path.read_text())
    archived = json.loads(
        (phase_dir / "invalidated_adapter_v1" / "answers" / "q1.json").read_text()
    )

    assert applied["vector_answers_preserved"] == 1
    assert archived == original
    assert repaired["adapter_version"] == lme_run.ADAPTER_VERSION
    assert repaired["status"] == "adapter_v2_pending"
    assert set(repaired["answers"]) == {"vector_rag"}
    assert repaired["answers"]["vector_rag"]["answer"] == "valid"
    assert not (judgements_dir / "q1__full_ice.json").exists()
    assert (judgements_dir / "q1__vector_rag.json").exists()
    assert (phase_dir / "invalidated_adapter_v1" / "judgements" /
            "q1__full_ice.json").exists()
    assert not (phase_dir / "MANIFEST.json").exists()
    assert (phase_dir / "invalidated_adapter_v1" / "MANIFEST.json").exists()
