"""v3 raw structured-template dispatch, exact source separation and completion."""

import json
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.workers.bg_client_factory import _BgClient
from src.workers.extraction_result import (
    ExtractionOutputError,
    original_source_quote,
    parse_extraction_response,
)
from src.workers.nuextract_transport import PROTOCOL, render_prompt


class Response:
    def __init__(self, data):
        self.data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self.data


def request(template=None):
    if template is None:
        template = json.dumps(dict(facts=[dict(subject='string', relation='string',
            object='string', source_sentence='verbatim-string')]))
    return dict(model='local-specialist', messages=[{'role': 'user', 'content': 'Aster uses Redis.'}],
                temperature=0., max_tokens=3000, timeout=60,
                extra_body={'ice_template_protocol': PROTOCOL, 'ice_context_window': 8192,
                            'chat_template_kwargs': {'template': template,
                                                     'instructions': 'Preserve facts.', 'enable_thinking': False}})


def test_owned_factory_routes_template_to_native_with_caps_and_source(monkeypatch):
    from src.workers import nuextract_transport as transport
    native, chat = [], []
    def post(url, **kw):
        native.append((url, kw))
        return Response(dict(done=True, done_reason='stop', response='{"facts": []}',
                             prompt_eval_count=99, eval_count=7))
    monkeypatch.setattr(transport.httpx, 'post', post)
    inner = NS(chat=NS(completions=NS(create=lambda **kw: chat.append(kw))))
    completion = _BgClient(inner, owned=True).chat.completions.create(**request())
    assert chat == [] and len(native) == 1
    url, kwargs = native[0]
    payload = kwargs['json']
    assert url == settings.ollama_base_url.rstrip('/') + '/api/generate'
    assert payload['raw'] is True and payload['stream'] is False and payload['think'] is False
    assert payload['options'] == dict(temperature=0., num_ctx=8192, num_predict=3000)
    assert '【task】structured' in payload['prompt']
    assert '【document_start】\nAster uses Redis.\n【document_end】' in payload['prompt']
    assert '【instructions_start】Preserve facts.【instructions_end】' in payload['prompt']
    assert '<|input|>' not in payload['prompt']
    assert completion.choices[0].finish_reason == 'stop'
    assert completion.usage.total_tokens == 106


def test_ordinary_owned_chat_uses_the_selected_door_not_the_extraction_protocol(monkeypatch):
    """Since 2026-10-15 ordinary background chat goes over native /api/chat by
    default; the NuExtract template protocol stays its own path either way."""
    from src.workers import bg_native_transport, nuextract_transport
    from src.api.config import settings
    monkeypatch.setattr(nuextract_transport, 'native_completion',
                        lambda **kw: pytest.fail('ordinary chat must not use the extraction protocol'))
    calls, native = [], []
    inner = NS(chat=NS(completions=NS(create=lambda **kw: calls.append(kw) or NS(choices=[]))))
    monkeypatch.setattr(bg_native_transport, 'chat_completion',
                        lambda **kw: native.append(kw) or NS(choices=[]))
    monkeypatch.setattr(settings, 'background_transport', 'auto')
    _BgClient(inner, owned=True).chat.completions.create(model='general-background', messages=[])
    assert len(native) == 1 and not calls
    assert native[0]['extra_body']['reasoning_effort'] == 'none'
    monkeypatch.setattr(settings, 'background_transport', 'openai')
    _BgClient(inner, owned=True).chat.completions.create(model='general-background', messages=[])
    assert len(calls) == 1 and calls[0]['extra_body']['reasoning_effort'] == 'none'


@pytest.mark.parametrize('data', [dict(done=False, done_reason='stop', response='{"facts": []}'),
                                    dict(done=True, response='{"facts": []}'),
                                    dict(done=True, done_reason='load', response='{"facts": []}')])
def test_missing_completion_authority_is_a_typed_failure(monkeypatch, data):
    from src.workers import nuextract_transport as transport
    monkeypatch.setattr(transport.httpx, 'post', lambda *_a, **_kw: Response(data))
    with pytest.raises(ExtractionOutputError, match='incomplete'):
        transport.native_completion(**request())


def test_provider_length_is_preserved_even_with_valid_empty_json(monkeypatch):
    from src.workers import nuextract_transport as transport
    monkeypatch.setattr(transport.httpx, 'post', lambda *_a, **_kw: Response(
        dict(done=True, done_reason='length', response='{"facts": []}')))
    completion = transport.native_completion(**request())
    assert completion.choices[0].finish_reason == 'length'
    assert completion.usage is None


def test_renderer_does_not_put_instruction_or_schema_inside_the_source_document():
    prompt = render_prompt('  Aster uses Redis.  ', '{}', 'Preserve facts.')
    document = prompt.split('【document_start】\n')[1].split('【document_end】')[0]
    assert document == 'Aster uses Redis.\n'
    assert prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')


def native_factory(monkeypatch, calls, content='{"facts": []}', finish='stop'):
    from src.workers import nuextract_transport as transport

    def post(url, **kwargs):
        calls.append(kwargs['json'])
        return Response(dict(done=True, done_reason=finish, response=content,
                             prompt_eval_count=99, eval_count=7))

    monkeypatch.setattr(transport.httpx, 'post', post)
    inner = NS(chat=NS(completions=NS(create=lambda **_: pytest.fail(
        'specialist called ordinary chat'))))
    return _BgClient(inner, owned=True)


@pytest.mark.parametrize('enabled', [False, True])
def test_native_shape_switch_preserves_envelope_and_other_fields(monkeypatch, enabled):
    monkeypatch.setattr(settings, 'codex_constrain_shape', enabled)
    calls = []
    result = native_factory(monkeypatch, calls).chat.completions.create(**request())
    assert result.choices[0].finish_reason == 'stop' and result.usage.total_tokens == 106
    body = calls[0]
    assert body['raw'] is True and body['think'] is False and body['keep_alive'] == -1
    assert body['options'] == dict(temperature=0., num_ctx=8192, num_predict=3000)
    assert ('format' in body) == enabled
    if enabled:
        schema = body['format']
        item = schema['properties']['facts']['items']
        assert schema['type'] == 'object' and schema['required'] == ['facts']
        assert set(item['required']) == {'subject', 'relation', 'object', 'source_sentence'}
        assert item['properties']['subject']['type'] == ['string', 'null']
        assert item['properties']['source_sentence'] == dict(type='string', minLength=1)
        assert item['properties']['negated']['type'] == 'boolean'
        assert 'negated' not in item['required']
        assert schema['additionalProperties'] is False and item['additionalProperties'] is False


@pytest.mark.parametrize('template', ['not-json', '[]', '{}', '{"facts": []}',
    '{"facts": ["string"]}',
    '{"facts": [{"subject": "integer", "relation": "string", "object": "string"}]}',
    '{"facts": [{"subject": "string", "relation": "string", "object": "string", "extra": "string"}]}'])
def test_wrong_template_refuses_before_native_provider(monkeypatch, template):
    monkeypatch.setattr(settings, 'codex_constrain_shape', True)
    calls = []
    client = native_factory(monkeypatch, calls)
    with pytest.raises(ExtractionOutputError, match='invalid facts template'):
        client.chat.completions.create(**request(template))
    assert calls == []


def test_quotes_off_keeps_matching_three_field_template(monkeypatch):
    monkeypatch.setattr(settings, 'codex_constrain_shape', True)
    calls = []
    client = native_factory(monkeypatch, calls)
    client.chat.completions.create(**request(json.dumps(dict(facts=[dict(
        subject='string', relation='string', object='string')]))))
    item = calls[0]['format']['properties']['facts']['items']
    assert set(item['required']) == {'subject', 'relation', 'object'}
    assert 'source_sentence' not in item['properties']


def test_escaped_source_and_nullable_tuple_keep_exact_evidence(monkeypatch):
    monkeypatch.setattr(settings, 'codex_constrain_shape', True)
    source = 'Aster uses "Redis".\tThe exact mode is unknown.'
    content = json.dumps(dict(facts=[
        dict(subject='Aster', relation='uses', object='Redis', source_sentence='Aster uses "Redis".'),
        dict(subject=None, relation='has_mode', object=None, source_sentence='The exact mode is unknown.'),
    ]))
    client = native_factory(monkeypatch, [], content)
    value = client.chat.completions.create(**request()).choices[0]
    facts = parse_extraction_response(value.message.content, value.finish_reason,
                                     template_mode=True, allow_source_only=True)
    assert len(facts) == 2 and facts[1]['_source_only'] is True
    assert all(original_source_quote(source, fact['source_sentence']) == fact['source_sentence']
               for fact in facts)


@pytest.mark.parametrize('content,finish', [('{"facts": []}', 'length'),
    ('{"facts": [', 'stop'),
    ('{"facts": [{"subject":"Aster","relation":"uses","object":"Redis","negated":1}]}', 'stop')])
def test_grammar_keeps_strict_completion_and_polarity_checks(monkeypatch, content, finish):
    monkeypatch.setattr(settings, 'codex_constrain_shape', True)
    choice = native_factory(monkeypatch, [], content, finish).chat.completions.create(**request()).choices[0]
    with pytest.raises(ExtractionOutputError):
        parse_extraction_response(choice.message.content, choice.finish_reason,
                                  template_mode=True, allow_source_only=True)
