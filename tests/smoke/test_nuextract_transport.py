"""v3 raw structured-template dispatch, exact source separation and completion."""

from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.workers.bg_client_factory import _BgClient
from src.workers.extraction_result import ExtractionOutputError
from src.workers.nuextract_transport import PROTOCOL, render_prompt


class Response:
    def __init__(self, data):
        self.data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self.data


def request():
    return dict(model='local-specialist', messages=[{'role': 'user', 'content': 'Aster uses Redis.'}],
                temperature=0., max_tokens=3000, timeout=60,
                extra_body={'ice_template_protocol': PROTOCOL, 'ice_context_window': 8192,
                            'chat_template_kwargs': {'template': '{"facts": ["string"]}',
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


def test_ordinary_owned_chat_still_uses_existing_client(monkeypatch):
    calls = []
    inner = NS(chat=NS(completions=NS(create=lambda **kw: calls.append(kw) or NS(choices=[]))))
    _BgClient(inner, owned=True).chat.completions.create(model='general-background', messages=[])
    assert len(calls) == 1
    assert calls[0]['extra_body']['reasoning_effort'] == 'none'


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
