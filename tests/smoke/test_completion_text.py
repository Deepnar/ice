from types import SimpleNamespace as NS

import pytest

from src.workers.completion_text import IncompleteCompletion, complete_text


def completion(content='A complete statement.', reason='stop'):
    return NS(choices=[NS(message=NS(content=content), finish_reason=reason)])


@pytest.mark.parametrize('reason', ['length', 'content_filter', 'tool_calls', None])
def test_even_plausible_prefixes_are_not_completed_output(reason):
    with pytest.raises(IncompleteCompletion):
        complete_text(completion(reason=reason))


@pytest.mark.parametrize('content', ['', '  ', None, []])
def test_empty_or_nontext_results_are_explicit_failures(content):
    with pytest.raises(IncompleteCompletion):
        complete_text(completion(content=content))


def test_missing_choice_is_explicit_failure():
    with pytest.raises(IncompleteCompletion):
        complete_text(NS(choices=[]))


def test_complete_output_is_preserved_without_prefix_or_word_cut():
    content = 'Grounded sentence. ' * 400 + 'The final correction matters.'
    assert complete_text(completion(content=content)) == content
