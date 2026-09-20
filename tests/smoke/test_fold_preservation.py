"""v3 fold input/output must preserve evidence before bounded generation."""
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.memory.source import single_provenance
from src.memory.support import verify_support
from src.workers import conversation_summary as worker
from src.workers.completion_text import IncompleteCompletion


def turn(raw, **kwargs):
    defaults = dict(raw_text=raw, inject_raw=False, summary_text=None,
                    summary_coverage=None, timestamp=datetime.now(timezone.utc),
                    ts_provenance='original', source_spans=single_provenance(raw, 'user'))
    defaults.update(kwargs)
    return NS(**defaults)


def test_late_correction_survives_and_unqualified_summary_is_not_used():
    raw = 'Discussing alternatives. ' * 800 + 'Final decision: do not deploy Redis.'
    row = turn(raw, summary_text='Deploy Redis.', summary_coverage=1.0)
    rendered = worker._original_source(row)[0]
    assert raw in rendered and 'Final decision: do not deploy Redis.' in rendered
    assert 'Deploy Redis.' not in rendered


def test_source_chunks_pack_before_boundary_without_cutting_units(monkeypatch):
    monkeypatch.setattr(settings, 'conversation_summary_chunk_words', 20)
    rows = [turn('First statement. ' * 8), turn('Second statement. ' * 8),
            turn('A long indivisible statement. ' * 40 + 'Late correction.')]
    chunks = [text for _, text, _ in worker._original_groups(rows)]
    assert len(chunks) == 3
    assert all(row.raw_text in chunks[i] for i, row in enumerate(rows))


def test_complete_generated_tail_is_not_cut_to_word_target():
    generated = 'Background detail. ' * 200 + 'The final correction is decisive.'
    supported = lambda p, h: verify_support(p, h, scorer=lambda pairs:
        [dict(entailment=.99, neutral=.005, contradiction=.005)])
    result = worker._source_note([NS(id='one', timestamp=None, ts_provenance='unknown')], 'Source.', True,
                                lambda *a, **k: generated, supported)
    assert result['text'] == generated and result['mode'] == 'supported'


@pytest.mark.parametrize('known_roles, score', [(True, .01), (False, .99)])
def test_unsupported_or_unattributed_note_keeps_complete_original(known_roles, score):
    source = 'A conditional proposal. ' * 500 + 'Final correction: do not deploy.'
    verifier = lambda p, h: verify_support(p, h, scorer=lambda pairs:
        [dict(entailment=score, neutral=1 - score, contradiction=0)])
    result = worker._source_note([NS(id='one', timestamp=None, ts_provenance='unknown')], source, known_roles,
                                lambda *a, **k: 'Deploy now.', verifier)
    assert result['text'] == source and result['mode'] == 'source'
    assert result['source_ids'] == ['one']


def test_generation_uses_original_even_when_turn_has_qualified_summary():
    row = turn('Atlas keeps Redis disabled. Redis stays disabled.',
               summary_text='Atlas keeps Redis disabled.', summary_coverage=1.0)
    source, known = worker._original_source(row)
    assert known and 'Redis stays disabled.' in source
    assert 'The user said:' in source
    prompt = worker._note_prompt(source)
    assert source in prompt and 'EXISTING SUMMARY' not in prompt
    assert 'MUST-PRESERVE' not in prompt


def test_oversized_real_request_never_reaches_provider(monkeypatch):
    from src.workers import bg_client_factory
    from src.model_registry import registry, runtime_probe
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'controlled-model')
    monkeypatch.setattr(registry, 'get_model_context_window', lambda *a: 512)
    monkeypatch.setattr(runtime_probe, 'serving_window', lambda *a: 512)
    monkeypatch.setattr(settings, 'background_model_mode', 'shared')
    def unexpected_client():
        raise AssertionError('Over-budget input must not be submitted')
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', unexpected_client)
    with pytest.raises(IncompleteCompletion, match='capacity'):
        worker._default_llm('A complete long source. ' * 1000)


def test_complete_in_budget_request_preserves_source_and_model_output(monkeypatch):
    from src.workers import bg_client_factory
    from src.model_registry import registry, runtime_probe
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(finish_reason='stop', message=NS(content='Complete summary.'))])
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda:
                        NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'local-background')
    monkeypatch.setattr(registry, 'get_model_context_window', lambda *a: 32768)
    monkeypatch.setattr(runtime_probe, 'serving_window', lambda *a: 32768)
    prompt = 'The source says to keep the original database schema.'
    assert worker._default_llm(prompt, max_tokens=321) == 'Complete summary.'
    assert calls[0]['messages'][-1]['content'] == prompt
    assert calls[0]['model'] == 'local-background' and calls[0]['max_tokens'] == 321


def test_overview_embedding_includes_tail_without_encoder_truncation():
    import numpy as np
    calls = []
    def encode(text, **kwargs):
        calls.append(text)
        assert len(text) + 2 <= 16
        return np.array([1., float('TAIL' in text)])
    encoder = NS(max_seq_length=16,
                 tokenizer=lambda text, **kw: {'input_ids': list(range(len(text) + 2))},
                 encode=encode)
    source = 'x' * 28 + 'TAIL'
    result = worker._summary_embedding(source, encoder)
    assert ''.join(calls) == source
    assert result[1] > 0 and np.isclose(np.linalg.norm(result), 1)
