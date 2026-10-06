"""Reuse pinned provider transport without altering the frozen seed's settings."""
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from scripts.z1 import judge_answers as judge, run_v3_campaign as campaign


@pytest.mark.parametrize("profile", ["opencode-muse13", "opencode-deepseek-v41-flash"])
@pytest.mark.parametrize("completion", ["complete", "incomplete", "outage", "quota"])
def test_explicit_judge_profile_has_correct_endpoint_and_bounded_failure(monkeypatch, profile, completion):
    monkeypatch.setenv("ICE_JUDGE_PROFILE", profile)
    monkeypatch.setenv("PROBE_API_KEY", "synthetic-key")
    monkeypatch.setenv("PROBE_API_BASE_URL", "https://test.invalid/v1")
    monkeypatch.setenv("PROBE_MODEL", "legacy-model-must-not-be-used")
    monkeypatch.setattr(judge, "load_selected_env", lambda: None)
    calls = []
    grade = {"verdict": "A", "reason": "contradicts_source", "A_grade": "correct", "B_grade": "incorrect"}
    def response(request):
        calls.append(request)
        if completion in {"outage", "quota"}:
            return httpx.Response(500 if completion == "outage" else 429,
                                  json={"error": {"type": "quota" if completion == "quota" else "server", "message": "synthetic failure"}})
        if profile == "opencode-muse13":
            payload = {"id": "fixture-response", "object": "response",
                       "status": "completed" if completion == "complete" else "incomplete",
                       "output": [{"type": "message", "role": "assistant", "content": [
                           {"type": "output_text", "text": json.dumps(grade)}]}]}
        else:
            payload = {"id": "fixture-response", "object": "chat.completion", "model": "fixture",
                       "choices": [{"index": 0, "finish_reason": "stop" if completion == "complete" else "length",
                                    "message": {"role": "assistant", "content": json.dumps(grade)}}]}
        return httpx.Response(200, json=payload)
    original = openai.OpenAI
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: original(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(response))))
    verdict = judge.judge_one("Which port?", "User chose port 7813. END_SOURCE", "7813", "9123",
                              expected_answer="7813", session_id="synthetic-both-orders")
    assert len(calls) == 1, "hidden SDK retries bypassed the persisted attempt policy"
    request = calls[0]
    assert request.headers["User-Agent"] == judge.UA
    assert request.headers["x-opencode-session"] == "synthetic-both-orders"
    payload = json.loads(request.content)
    responses = profile == "opencode-muse13"
    assert request.url.path.endswith("/responses" if responses else "/chat/completions")
    assert payload["model"] == ("muse-spark-1.3-contributor" if responses else "deepseek-v4.1-flash")
    messages = payload["input" if responses else "messages"]
    assert messages[0]["content"] == judge.SYSTEM_V3
    assert "END_SOURCE" in messages[1]["content"]
    if completion == "complete":
        assert verdict["A_grade"] == "correct" and verdict["provider"]["response_id"] == "fixture-response"
    else:
        assert verdict["verdict"] == "ERROR"
        assert verdict.get("operator_required", False) is (completion == "quota")
        assert verdict["reason"] == {"incomplete": "incomplete_completion", "outage": "api", "quota": "api_http_429"}[completion]


def test_judge_identity_pins_model_endpoint_and_profile(monkeypatch):
    monkeypatch.setenv("PROBE_API_BASE_URL", "https://test.invalid/v1")
    monkeypatch.setattr(judge, "load_selected_env", lambda: None)
    monkeypatch.setenv("ICE_JUDGE_PROFILE", "opencode-muse13")
    muse = judge.judge_provider_identity()
    monkeypatch.setenv("ICE_JUDGE_PROFILE", "opencode-deepseek-v41-flash")
    deepseek = judge.judge_provider_identity()
    assert muse != deepseek
    assert muse["judge_provider"]["endpoint"] == "responses"
    assert deepseek["judge_provider"]["endpoint"] == "chat_completions"
    assert muse["judge_transport_sha256"] == deepseek["judge_transport_sha256"]
    assert campaign.judge_selection({"judge_profile": "opencode-muse13"})["model"] == "muse-spark-1.3-contributor"


def test_manual_profile_reaches_only_judge_children(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "status", lambda *_: {"reviewed_label_candidates_available": True})
    monkeypatch.setattr(campaign, "validate_labels", lambda *_: 1)
    monkeypatch.setattr(campaign, "progress_totals", lambda *_: (1, 1))
    monkeypatch.setattr(campaign, "database_environment", lambda *_: {"ICE_JUDGE_PROFILE": "unwanted-parent-profile"})
    monkeypatch.setattr(campaign, "load_selected_env", lambda: None)
    monkeypatch.setattr(campaign, "campaign_report", lambda *_: {})
    monkeypatch.setattr(campaign, "PROFILES", {name: SimpleNamespace(
        resolved_base_url=lambda: "https://test.invalid", resolved_api_key=lambda: "fixture")
        for name in ("opencode-luna6", "opencode-muse13")})
    monkeypatch.setenv("PROBE_API_KEY", "fixture")
    monkeypatch.setenv("PROBE_API_BASE_URL", "https://test.invalid")
    seen = []
    def child(command, **kwargs):
        name = command[1].rsplit("/", 1)[-1]
        seen.append(name)
        assert kwargs["env"].get("ICE_JUDGE_PROFILE") == ("opencode-muse13" if name == "judge_answers.py" else None)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess, "run", child)
    config = {"answer_profile": "opencode-luna6", "judge_profile": "opencode-muse13",
              "checkpoint_every": 1, "probe_panel": "existing", "snapshot_arm": "fixture"}
    assert campaign.execute(tmp_path, config, "all") == 0
    assert seen.count("judge_answers.py") == 3 and seen.count("answer_as_of.py") == 4
