"""G5: the SSE parser must not lose content silently.

The bug had two halves and the tests below are split the same way.

  1. THE SPLICE. A turn can involve two upstream streams — the primary, and
     the fallback model after a timeout. They shared one flat chunk list, so
     the primary's truncated tail was concatenated onto the fallback's first
     line, producing one corrupt line that the parser dropped without a word.
     Content the user had already watched appear on screen never reached
     `raw_text`.

  2. THE SILENCE. Every unparseable line hit a bare `except: continue`. A
     short `raw_text` is indistinguishable from a short answer.

Pure string handling — no DB, no GPU, no network.
"""

import json

from src.api.main import _parse_sse_text, _salvage_content


def _line(text: str) -> str:
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": text}}]})


def _join(segments: list[list[str]]) -> str:
    """Exactly what store_turn_async does: "" within a stream, "\\n" between."""
    return "\n".join("".join(seg) for seg in segments)


# ── 1. the splice ───────────────────────────────────────────────────────────

def test_truncated_primary_cannot_corrupt_the_fallback_stream():
    """The regression that motivated the fix."""
    # The primary emitted two full lines and then died mid-third.
    primary = [_line("Hello") + "\n" + _line(" world") + "\n",
               'data: {"choices":[{"delta":{"content":"par']
    fallback = [_line("Fresh") + "\n" + _line(" answer") + "\n",
                "data: [DONE]\n"]

    text, stats = _parse_sse_text(_join([primary, fallback]))

    # The fallback's first line survives — that is the whole point.
    assert "Fresh answer" in text
    # And the primary's completed content is still there.
    assert "Hello world" in text
    assert stats["dropped"] <= 1, stats

    # Two-sided: the OLD behavior (one flat list, "" join) really did corrupt
    # it, so this test would have failed before the fix rather than passing
    # vacuously.
    old_style = "".join(primary + fallback)
    old_text, _ = _parse_sse_text(old_style)
    assert "Fresh" not in old_text, (
        "the flat join should swallow the fallback's first line; if it does "
        "not, this test is no longer pinning the bug it was written for")


def test_single_stream_socket_splits_still_repair():
    """Joining WITHIN a segment must stay "" — chunk boundaries are arbitrary."""
    whole = _line("alpha") + "\n" + _line("beta") + "\n"
    mid = len(whole) // 2
    segments = [[whole[:mid], whole[mid:]]]

    text, stats = _parse_sse_text(_join(segments))
    assert text == "alphabeta"
    assert stats["dropped"] == 0


# ── 2. the silence ──────────────────────────────────────────────────────────

def test_unparseable_lines_are_counted_not_swallowed():
    stream = "\n".join([
        _line("good"),
        'data: {"choices":[{"delta":{"conte',   # cut inside the KEY
        _line("also good"),
        "data: [DONE]",
    ])
    text, stats = _parse_sse_text(stream)
    assert text == "goodalso good"
    assert stats["dropped"] == 1, stats
    assert stats["parsed"] == 2, stats


# Verbatim from a live qwen3:4b turn through the proxy, 2026-08-08. The first
# hand-written version of this suite had no usage chunk in it, so it never saw
# that the parser called one "damage" on every healthy stream — the probe set
# tested the failures its author imagined (TRAPS #13), and the live run found
# the real one in a minute.
_REAL_TAIL = (
    'data: {"id":"chatcmpl-221","object":"chat.completion.chunk",'
    '"created":1786185358,"model":"qwen3:4b-instruct",'
    '"system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":'
    '{"role":"assistant","content":""},"finish_reason":"stop"}]}\n'
    'data: {"id":"chatcmpl-221","object":"chat.completion.chunk",'
    '"created":1786185358,"model":"qwen3:4b-instruct",'
    '"system_fingerprint":"fp_ollama","choices":[],'
    '"usage":{"prompt_tokens":288,"completion_tokens":13,"total_tokens":301}}\n'
    "data: [DONE]"
)


def test_a_real_healthy_stream_reports_zero_damage():
    """The usage chunk and the finish_reason chunk are normal, not damage."""
    stream = _line("The three primary colors") + "\n" + _REAL_TAIL

    text, stats = _parse_sse_text(stream)
    assert text == "The three primary colors"
    assert stats["dropped"] == 0, stats
    assert stats["salvaged"] == 0, stats
    # Both content-free chunks land in their own bucket rather than looking
    # like loss.
    assert stats["no_content"] == 2, stats


def test_content_free_chunks_are_not_damage():
    """Role-only openers and empty choices lists must not trip the warning."""
    stream = "\n".join([
        'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}',
        _line("text"),
        'data: {"choices":[],"usage":{"prompt_tokens":1}}',
    ])
    _, stats = _parse_sse_text(stream)
    assert stats["dropped"] == 0 and stats["salvaged"] == 0, stats
    assert stats["no_content"] == 2, stats


def test_unclosed_object_with_intact_string_is_salvaged():
    """The realistic truncation: braces missing, content string complete."""
    stream = "\n".join([
        _line("start "),
        'data: {"choices":[{"delta":{"content":"recovered"',   # no closing }}]}
        "data: [DONE]",
    ])
    text, stats = _parse_sse_text(stream)
    assert text == "start recovered"
    assert stats["salvaged"] == 1, stats
    assert stats["dropped"] == 0, stats


def test_salvage_refuses_a_cut_string_rather_than_inventing_one():
    """A string cut mid-value is NOT recoverable; it must count as dropped."""
    assert _salvage_content('{"choices":[{"delta":{"content":"half') is None
    assert _salvage_content('{"choices":[{"delta":{}}]}') is None
    assert _salvage_content("not json at all") is None


def test_salvage_preserves_escapes():
    line = '{"choices":[{"delta":{"content":"a\\"quote\\" and \\n newline"'
    assert _salvage_content(line) == 'a"quote" and \n newline'


def test_clean_stream_reports_no_damage():
    """The negative side: a healthy stream must not trip the WARNING."""
    stream = "\n".join([_line("a"), _line("b"), "data: [DONE]"])
    text, stats = _parse_sse_text(stream)
    assert text == "ab"
    assert stats == {"lines": 2, "parsed": 2, "no_content": 0,
                     "salvaged": 0, "dropped": 0}


def test_empty_stream_is_not_damage():
    text, stats = _parse_sse_text("")
    assert text == ""
    assert stats["dropped"] == 0 and stats["salvaged"] == 0
