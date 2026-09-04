from __future__ import annotations

import sys
from pathlib import Path


LME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LME_DIR))

import calibrate_cloud_models as calibration
from cloud_provider import get_profile


def test_calibration_artifact_identity_is_strict():
    profile = get_profile("opencode-omen-alpha")
    good = {
        "status": "complete",
        "answer": "OK",
        "provider": profile.metadata(),
    }
    assert calibration.answer_matches_profile(good, profile)

    wrong_model = {
        **good,
        "provider": {**profile.metadata(), "model": "another-model"},
    }
    assert not calibration.answer_matches_profile(wrong_model, profile)

    mute = {**good, "status": "mute", "answer": ""}
    assert not calibration.answer_matches_profile(mute, profile)


def test_calibration_judge_identity_checks_endpoint_and_profile():
    profile = get_profile("opencode-muse13")
    record = {
        "status": "complete",
        "judge": profile.metadata(),
        "judge_raw": "yes",
    }
    assert calibration.judgement_matches_profile(record, profile)

    wrong_endpoint = {
        **record,
        "judge": {**profile.metadata(), "endpoint": "chat_completions"},
    }
    assert not calibration.judgement_matches_profile(wrong_endpoint, profile)
