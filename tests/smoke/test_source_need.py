"""Source decisions cannot silently discard prior recall or invent citations."""
import json
from types import SimpleNamespace

import pytest

from src.api import source_need as sn
from src.api import source_proof as sp
from src.api.config import settings


def encoded(decision, quote=None):
    return json.dumps(dict(required_information="The chosen port.",
                           decision=decision, evidence_quote=quote))


def test_visible_quote_must_resolve_to_source_and_response_complete():
    messages = [{"role": "user", "content": "The indexer port is 7813."}]
    good = sn.parse_verdict(encoded("visible_evidence", "port is 7813"), "stop", messages)
    assert good == sn.SourceNeed("visible_evidence", "port is 7813")
    assert sn.parse_verdict(encoded("visible_evidence", "port is 7619"), "stop", messages).decision == "unknown"
    assert sn.parse_verdict(encoded("older_memory"), "length", messages).reason == "incomplete_response"


@pytest.mark.parametrize("content", ["", "[]", "null", '{"decision":"older_memory"}',
    encoded("general_knowledge", "invented quote")])
def test_unqualified_output_is_explicit_unknown(content):
    assert sn.parse_verdict(content, "stop", [{"content": "Input"}]).decision == "unknown"


@pytest.mark.parametrize("decision", ["visible_evidence", "general_knowledge", "unknown"])
def test_negative_prior_stays_negative_without_a_private_source_need(decision):
    assert sn.source_action(sn.SourceNeed(decision)) == "keep"


def test_negative_search_still_requires_private_source_decision():
    assert sn.source_action(sn.SourceNeed("older_memory")) == "rescue"


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
    assert body["keep_alive"] == -1 and body["think"] is False


def test_complete_overlength_input_never_calls_provider_or_truncates(monkeypatch):
    monkeypatch.setattr(settings, "memory_source_gate_input_tokens", 1)
    monkeypatch.setattr(sn.httpx, "post", lambda *_a, **_k: pytest.fail("Provider should not be called"))
    messages = [{"role": "user", "content": "The final source must stay intact."}]
    assert sn.judge_source_need(messages).reason == "complete_input_exceeds_gate_bound"
    assert messages[-1]["content"].endswith("intact.")


def test_rescue_question_frame_reads_only_the_current_message(monkeypatch):
    calls = []
    def post(_url, **kwargs):
        calls.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(
            done=True, done_reason="stop", prompt_eval_count=321,
            message=dict(content=json.dumps(dict(frame=None, assertion_frame=None,
                                                 reason="unframeable")))))
    monkeypatch.setattr(sn.httpx, "post", post)
    assert sp.freeze_source_question("What was our choice?").status == "not_supplied"
    assert calls[0]["format"] == sp.FRAME_SCHEMA
    assert json.loads(calls[0]["messages"][1]["content"]) == {
        "latest_user_request": "What was our choice?"}
    assert calls[0]["think"] is False
    assert calls[0]["options"]["num_predict"] == settings.memory_source_proof_output_tokens
    assert len(calls) == 1
