"""ICE v3 extraction output contract: empty success is distinct from failure."""

import json

from src.workers.llm_json import strip_fences


class ExtractionOutputError(ValueError):
    """Incomplete or invalid extraction; callers must not mark work complete."""


def parse_extraction_response(content, finish_reason=None, *, template_mode=False):
    """Return complete facts, preserving polarity and arbitrary JSON key order.

    Errors carry structure/reasons only, never source text or model responses.
    A partially recoverable response is still incomplete: committing its prefix
    would make omissions permanent or count replay as fresh corroboration.
    """
    if finish_reason not in (None, "stop"):
        raise ExtractionOutputError(f"incomplete completion: {finish_reason}")
    if not isinstance(content, str) or not content.strip():
        raise ExtractionOutputError("missing extraction content")
    if template_mode and "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    content = content.strip()
    if content.startswith("```") and (
        not content.endswith("```") or content.count("```") != 2
    ):
        raise ExtractionOutputError("incomplete or ambiguous extraction fence")
    try:
        facts = json.loads(strip_fences(content))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ExtractionOutputError("invalid or incomplete extraction JSON") from exc
    if isinstance(facts, dict):
        envelopes = [key for key in ("facts", "triplets") if key in facts]
        if len(envelopes) != 1:
            raise ExtractionOutputError("expected one facts or triplets envelope")
        facts = facts[envelopes[0]]
    if not isinstance(facts, list):
        raise ExtractionOutputError("extraction must be a fact array")
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict) or not all(
            isinstance(fact.get(key), str) and fact[key].strip()
            for key in ("subject", "relation", "object")
        ):
            raise ExtractionOutputError(f"invalid fact fields at index {index}")
        if "negated" in fact and not isinstance(fact["negated"], bool):
            raise ExtractionOutputError(f"non-boolean polarity at index {index}")
    return facts
