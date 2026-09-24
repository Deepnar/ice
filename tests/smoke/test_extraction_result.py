"""Extraction output must distinguish an honest empty answer from data loss."""

import json

import pytest

from src.workers.extraction_result import (
    ExtractionOutputError,
    parse_extraction_response,
)


@pytest.mark.parametrize("content", ["[]", '{"facts": []}', '{"triplets": []}'])
def test_empty_results_are_complete(content):
    assert parse_extraction_response(content, "stop") == []


def test_negative_fact_survives_key_order_and_escaped_strings():
    facts = [
        {"object": 'tool "A"', "negated": True, "relation": "uses", "subject": "atlas"}
    ]
    content = (
        "<think>deliberation</think>```json\n" + json.dumps({"facts": facts}) + "\n```"
    )
    assert parse_extraction_response(content, "stop", template_mode=True) == facts


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "  ",
        "null",
        "true",
        "{}",
        '{"facts": null}',
        '{"facts": [], "triplets": []}',
        '["atlas", "uses", "postgres"]',
        '[{"subject": null, "relation": "uses", "object": "postgres"}]',
        '[{"subject": "atlas", "relation": "uses", "object": " "}]',
        '[{"subject": "atlas", "relation": "uses", "object": "postgres", "negated": "false"}]',
        '[{"subject": "atlas", "relation": "uses", "object": "postgres"},',
        "[] trailing content",
        "```json\n[]",
        "```json\n[]\n``` trailing content",
    ],
)
def test_incomplete_or_invalid_output_is_not_empty_success(content):
    with pytest.raises(ExtractionOutputError):
        parse_extraction_response(content, "stop")


@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls"])
def test_incomplete_completion_cannot_commit_even_valid_json(reason):
    with pytest.raises(ExtractionOutputError, match="incomplete completion"):
        parse_extraction_response("[]", reason)


def test_errors_do_not_echo_source_content():
    with pytest.raises(ExtractionOutputError) as error:
        parse_extraction_response("SECRET_SOURCE_TEXT")
    assert "SECRET_SOURCE_TEXT" not in str(error.value)
