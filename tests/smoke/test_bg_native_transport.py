"""G32(a)/G4: background calls over Ollama's own /api/chat (2026-10-15).

Measured the same day on Ollama 0.32.13: the /v1 door drops `num_ctx`, and both
doors cut an oversized input silently (HTTP 200, "stop") — the model then
invented the dropped fact. These tests pin the translation, the context sizing,
and that a cut input FAILS instead of returning a confident answer.

Run:  uv run --no-sync pytest tests/smoke/test_bg_native_transport.py -q
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.workers.bg_client_factory as factory  # noqa: E402
from src.api.config import settings  # noqa: E402
from src.workers import bg_native_transport as native  # noqa: E402
from src.workers.completion_text import IncompleteCompletion, complete_text  # noqa: E402


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.text = data, status, str(data)

    def json(self):
        return self._data


def _ok(prompt_eval_count=1000, content='{"a": 1}', reason="stop"):
    return {"done": True, "done_reason": reason, "prompt_eval_count": prompt_eval_count,
            "eval_count": 7, "message": {"role": "assistant", "content": content}}


@pytest.fixture
def wire(monkeypatch):
    sent, warnings = [], []
    reply = {"data": _ok()}
    monkeypatch.setattr(native, "estimate_input_tokens", lambda m: 1000)
    monkeypatch.setattr(native, "model_maximum", lambda m: 131072)
    monkeypatch.setattr(native.httpx, "post",
                        lambda url, json, timeout: sent.append((url, json, timeout)) or _Resp(reply["data"]))
    monkeypatch.setattr(native.logger, "warning", lambda ev, **kw: warnings.append((ev, kw)))
    monkeypatch.setattr(settings, "background_model_mode", "shared")
    monkeypatch.setattr(settings, "background_transport", "auto")
    return NS(sent=sent, warnings=warnings, reply=reply)


def _call(**over):
    kwargs = dict(model="gemma4:e4b", messages=[{"role": "user", "content": "x"}],
                  temperature=0.0, max_tokens=900, timeout=54.0,
                  extra_body={"reasoning_effort": "none"})
    kwargs.update(over)
    return native.chat_completion(**kwargs)


def test_context_size_stays_on_the_floor_for_ordinary_inputs_and_doubles_for_long_ones():
    floor = settings.bg_num_ctx_floor
    assert native.fit_num_ctx(5_000, 131072) == floor
    assert native.fit_num_ctx(floor + 1, 131072) == floor * 2
    assert native.fit_num_ctx(70_000, 131072) == 131072
    assert native.fit_num_ctx(70_000, 65536) == 65536, "never above the model maximum"


def test_request_is_translated_to_native_controls(wire):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    result = _call(response_format=factory.json_schema("t", schema))
    url, body, timeout = wire.sent[0]
    assert url.endswith("/api/chat")
    assert body["options"] == {"num_ctx": settings.bg_num_ctx_floor, "temperature": 0.0,
                               "num_predict": 900}
    assert body["keep_alive"] == settings.bg_keep_alive
    assert body["think"] is False, "reasoning_effort none must become think false"
    assert body["format"] == schema
    assert timeout > 54.0, "the timeout gains an input-size term"
    assert complete_text(result) == '{"a": 1}'
    assert result.usage.prompt_tokens == 1000
    assert wire.warnings == []


def test_a_long_input_gets_a_larger_context(wire, monkeypatch):
    monkeypatch.setattr(native, "estimate_input_tokens", lambda m: 60_000)
    wire.reply["data"] = _ok(prompt_eval_count=58_000)
    _call()
    assert wire.sent[0][1]["options"]["num_ctx"] == 131072


def test_a_server_side_cut_fails_instead_of_answering(wire):
    """Two-sided: 1000 sent, 950 read (tokenizer disagreement) passes; 400 read fails."""
    wire.reply["data"] = _ok(prompt_eval_count=950)
    _call()
    wire.reply["data"] = _ok(prompt_eval_count=400)
    with pytest.raises(IncompleteCompletion) as caught:
        _call()
    assert caught.value.finish_reason == "input_truncated"
    assert any(ev == "bg_input_truncated" for ev, _ in wire.warnings)


def test_an_input_too_big_for_the_model_is_refused_before_sending(wire, monkeypatch):
    monkeypatch.setattr(native, "estimate_input_tokens", lambda m: 200_000)
    with pytest.raises(IncompleteCompletion) as caught:
        _call()
    assert caught.value.finish_reason == "input_over_model_window"
    assert wire.sent == [], "nothing may be sent once it cannot fit"


def test_an_untranslated_parameter_is_reported_not_dropped(wire):
    _call(top_p=0.5, extra_body={"reasoning_effort": "none", "chat_template_kwargs": {}})
    reported = [kw["params"] for ev, kw in wire.warnings if ev == "bg_native_params_not_translated"]
    assert ["top_p"] in reported and ["chat_template_kwargs"] in reported


def test_truncated_output_still_reads_as_incomplete(wire):
    wire.reply["data"] = _ok(reason="length")
    with pytest.raises(IncompleteCompletion):
        complete_text(_call())


def test_a_schema_refusal_keeps_the_unconstrained_retry_path(wire, monkeypatch):
    calls = []

    def post(url, json, timeout):
        calls.append(json)
        if "format" in json:
            return _Resp({"error": "invalid json schema format"}, status=400)
        return _Resp(_ok())
    monkeypatch.setattr(native.httpx, "post", post)
    monkeypatch.setattr(factory.logger, "warning", lambda ev, **kw: wire.warnings.append((ev, kw)))
    client = factory._BgClient(NS(chat=NS(completions=NS(create=None))), owned=True)
    result = client.chat.completions.create(
        model="gemma4:e4b", messages=[{"role": "user", "content": "x"}], max_tokens=50,
        temperature=0.0, timeout=30.0, response_format=factory.json_schema("t", {"type": "object"}))
    assert complete_text(result) == '{"a": 1}'
    assert "format" in calls[0] and "format" not in calls[1]
    assert any(ev == "bg_schema_unsupported_retrying_unconstrained" for ev, _ in wire.warnings)


@pytest.mark.parametrize("mode,transport,expect_native", [
    ("shared", "auto", True), ("shared", "ollama", True),
    ("shared", "openai", False), ("dedicated", "auto", False)])
def test_which_door_is_used(wire, monkeypatch, mode, transport, expect_native):
    monkeypatch.setattr(settings, "background_model_mode", mode)
    monkeypatch.setattr(settings, "background_transport", transport)
    inner_calls = []

    def inner_create(**kw):
        inner_calls.append(kw)
        return NS(usage=NS(prompt_tokens=990),
                  choices=[NS(message=NS(content="ok", model_extra={}), finish_reason="stop")])
    completions = factory._OwnedCompletions(NS(create=inner_create))
    completions.create(model="gemma4:e4b", messages=[{"role": "user", "content": "x"}],
                       max_tokens=10, temperature=0.0, timeout=30.0)
    assert bool(wire.sent) is expect_native
    assert bool(inner_calls) is not expect_native


def test_the_openai_door_still_catches_a_cut_input(wire, monkeypatch):
    monkeypatch.setattr(settings, "background_transport", "openai")
    completions = factory._OwnedCompletions(NS(create=lambda **kw: NS(
        usage=NS(prompt_tokens=300),
        choices=[NS(message=NS(content="ok", model_extra={}), finish_reason="stop")])))
    with pytest.raises(IncompleteCompletion):
        completions.create(model="gemma4:e4b", messages=[{"role": "user", "content": "x"}],
                           max_tokens=10, temperature=0.0, timeout=30.0)


def test_a_native_call_is_owned_for_idle_release(wire, monkeypatch):
    monkeypatch.setattr(factory, "_owned_models", set())
    monkeypatch.setattr(factory, "_model_locks", {})
    factory._OwnedCompletions(NS(create=None)).create(
        model="gemma4:e4b", messages=[{"role": "user", "content": "x"}], max_tokens=10,
        temperature=0.0, timeout=30.0)
    assert wire.sent and factory._owned_models == {"gemma4:e4b"}
