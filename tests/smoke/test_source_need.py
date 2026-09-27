"""Source decisions cannot silently discard prior recall or invent citations."""
import json
from types import SimpleNamespace

import pytest

from src.api import source_need as sn
from src.api.config import settings


def encoded(decision, quote=None):
    return json.dumps(dict(required_information="The chosen port.",
                           decision=decision, evidence_quote=quote))


def test_visible_quote_must_be_exact_and_response_complete():
    messages = [{"role": "user", "content": "The indexer port is 7813."}]
    good = sn.parse_verdict(encoded("visible_evidence", "port is 7813"), "stop", messages)
    assert good == sn.SourceNeed("visible_evidence", "port is 7813")
    assert sn.parse_verdict(encoded("visible_evidence", "port is 7619"), "stop", messages).decision == "unknown"
    assert sn.parse_verdict(encoded("older_memory"), "length", messages).reason == "incomplete_response"


@pytest.mark.parametrize("content", ["", "[]", "null", '{"decision":"older_memory"}',
    encoded("general_knowledge", "invented quote")])
def test_unqualified_output_is_explicit_unknown(content):
    assert sn.parse_verdict(content, "stop", [{"content": "Input"}]).decision == "unknown"


@pytest.mark.parametrize("decision", ["general_knowledge", "older_memory", "unknown"])
def test_non_sourced_verdict_preserves_positive_prior(decision):
    assert sn.source_action(True, sn.SourceNeed(decision), "Continue our equations.") == "keep"


def test_recent_excerpt_cannot_suppress_positive_prior():
    verdict = sn.SourceNeed("visible_evidence", "Previous equations: x = y.")
    assert sn.source_action(True, verdict, "Continue our equations.") == "keep"
    prompt = "Our port is 7813. Which port?"
    assert sn.source_action(True, sn.SourceNeed("visible_evidence", "Our port is 7813."), prompt) == "skip"


def test_current_proof_missing_is_normal_and_cannot_suppress_prior(caplog):
    messages = [{"role": "user", "content": "Which port did we choose before?"}]
    verdict = sn.parse_verdict(encoded("not_supplied"), "stop", messages,
                               current_only=True)
    assert verdict == sn.SourceNeed("not_supplied")
    assert sn.source_action(True, verdict, messages[0]["content"]) == "keep"
    assert "memory_source_need_unknown" not in caplog.text
    assert sn.parse_verdict(encoded("general_knowledge"), "stop", messages,
                            current_only=True).reason == "invalid_shape"


def test_rescue_requires_quote_in_admitted_retrieval_not_question():
    verdict = sn.SourceNeed("visible_evidence", "port is 7813")
    assert not sn.rescue_has_source(verdict, [SimpleNamespace(text="Port used previously.")])
    assert not sn.rescue_has_source(verdict, [])
    assert sn.rescue_has_source(verdict, [SimpleNamespace(text="User: Our port is 7813.")])
    assert sn.source_action(False, sn.SourceNeed("older_memory"), "Which port did we choose?") == "rescue"


def test_native_call_owns_bounds_and_reads_complete_prepared_messages(monkeypatch):
    calls = []
    messages = [{"role": "system", "content": "Standing preference: blue."},
                {"role": "user", "content": "Our port is 7813. Which port?"}]
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(
            done=True, done_reason="stop", prompt_eval_count=321,
            message=dict(content=encoded("visible_evidence", "Our port is 7813."))))
    monkeypatch.setattr(sn.httpx, "post", post)
    result = sn.judge_source_need(messages)
    assert result.decision == "visible_evidence"
    url, args = calls[0]
    assert url.endswith("/api/chat")
    body = args["json"]
    supplied = json.loads(body["messages"][1]["content"])
    assert supplied["visible_context_messages"] == messages[:-1]
    assert supplied["latest_user_prompt"] == messages[-1]["content"]
    assert body["options"]["num_ctx"] == settings.memory_source_gate_context_tokens
    assert body["keep_alive"] == 0 and body["think"] is False


def test_complete_overlength_input_never_calls_provider_or_truncates(monkeypatch):
    monkeypatch.setattr(settings, "memory_source_gate_input_tokens", 1)
    monkeypatch.setattr(sn.httpx, "post", lambda *_a, **_k: pytest.fail("Provider should not be called"))
    messages = [{"role": "user", "content": "The final source must stay intact."}]
    assert sn.judge_source_need(messages).reason == "complete_input_exceeds_gate_bound"
    assert messages[-1]["content"].endswith("intact.")


def test_positive_proof_uses_binary_task_and_only_current_message(monkeypatch):
    calls = []
    def post(_url, **kwargs):
        calls.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(
            done=True, done_reason="stop", prompt_eval_count=321,
            message=dict(content=encoded("not_supplied"))))
    monkeypatch.setattr(sn.httpx, "post", post)
    assert sn.judge_source_need([dict(role="user", content="What was our choice?")],
                                current_only=True).decision == "not_supplied"
    assert calls[0]["format"]["properties"]["decision"]["enum"] == ["visible_evidence", "not_supplied"]
    assert calls[0]["messages"][0]["content"] == sn.CURRENT_INSTRUCTIONS
    assert sn.judge_source_need([dict(role="system", content="Prior source."),
        dict(role="user", content="Question")], current_only=True).reason == "invalid_current_source_messages"
    assert len(calls) == 1
