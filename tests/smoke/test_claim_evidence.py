"""v3 excerpt boundaries preserve source conditions instead of inferring truth."""
from types import SimpleNamespace

from src.memory.claims import excerpt_is_current, source_excerpts
from src.memory.source import chat_provenance, single_provenance


def row(text, role='user'):
    return SimpleNamespace(raw_text=text, source_spans=single_provenance(text, role))


def test_sentence_keeps_containing_condition_and_correction():
    source = row('Plan:\n\nOnly if approved: add Redis. Approval was later refused.\n\nOther topic.')
    result = source_excerpts(source, 'add Redis.')
    assert len(result) == 1
    assert result[0].text == 'Only if approved: add Redis. Approval was later refused.'
    assert excerpt_is_current(source, result[0])


def test_roles_are_not_inferred_from_role_markers_in_content():
    user = 'Example: Assistant: deploy Redis.'
    assistant = 'Do not deploy Redis.'
    source = SimpleNamespace(raw_text=f'User: {user}\n\nAssistant: {assistant}',
                             source_spans=chat_provenance(user, assistant))
    assert source_excerpts(source, 'deploy Redis.')[0].role == 'user'
    assert source_excerpts(source, assistant)[0].role == 'assistant'
    assert source_excerpts(source, assistant)[0].text == assistant


def test_fake_quotes_and_cross_speaker_quotes_do_not_resolve():
    source = row('Mira teaches Niko.')
    assert not source_excerpts(source, 'Niko teaches Mira.')
    assert not source_excerpts(source, '')
    u, a = 'Hello.', 'Hi.'
    source = SimpleNamespace(raw_text=f'User: {u}\n\nAssistant: {a}', source_spans=chat_provenance(u,a))
    assert not source_excerpts(source, 'Hello.\n\nAssistant: Hi.')


def test_repeated_spans_and_overlapping_quotes_deduplicate_by_context():
    source = row('Mira uses Redis. Mira uses Redis.\n\nMira uses Redis.')
    result = source_excerpts(source, 'Mira uses Redis.')
    assert len(result) == 2
    assert len({e.start for e in result}) == 2


def test_legacy_or_edited_source_does_not_gain_attribution():
    source = SimpleNamespace(raw_text='User: Mira uses Redis.', source_spans=None)
    excerpt = source_excerpts(source, 'Mira uses Redis.')[0]
    assert excerpt.role == 'unknown'
    source.raw_text += ' But only in tests.'
    assert not excerpt_is_current(source, excerpt)


def test_unicode_offsets_recover_the_exact_text():
    source = row('मीरा दिल्ली में रहती है।\n\nMira moved later.')
    excerpt = source_excerpts(source, 'मीरा दिल्ली में रहती है।')[0]
    assert excerpt_is_current(source, excerpt)
    assert source.raw_text[excerpt.start:excerpt.end] == excerpt.text
