"""Complete text output for background writers; a valid prefix is not a result."""


class IncompleteCompletion(ValueError):
    """Retryable provider output failure, with no source content in the error."""


def complete_text(completion):
    choices = getattr(completion, 'choices', None)
    if not choices:
        raise IncompleteCompletion('completion has no choices')
    choice = choices[0]
    reason = getattr(choice, 'finish_reason', None)
    if reason != 'stop':
        raise IncompleteCompletion(f'completion is not complete: {reason!r}')
    content = getattr(getattr(choice, 'message', None), 'content', None)
    if not isinstance(content, str) or not content.strip():
        raise IncompleteCompletion('completion has no text')
    return content.strip()
