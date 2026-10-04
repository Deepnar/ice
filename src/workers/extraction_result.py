"""ICE v3 extraction output contract: empty success is distinct from failure."""

import json
import re

from src.workers.llm_json import strip_fences


class ExtractionOutputError(ValueError):
    """Incomplete or invalid extraction; callers must not mark work complete."""


def original_source_quote(source: str, quote: str, *, allow_case_copy=False) -> str | None:
    """Resolve copying changes to original bytes, never accept a paraphrase.

    Whitespace-only matching is the default. Source selection may explicitly
    permit case copies; ambiguous originals still stay unresolved. The returned
    value always has the original spelling, including case and whitespace.
    """
    if not isinstance(quote, str) or not quote.strip():
        return None
    quote = quote.strip()
    if quote in source:
        return quote
    def resolve(case_copy):
        parts, positions = [], []
        for match in re.finditer(r"\s+|\S", source):
            part = " " if match.group().isspace() else match.group()
            part = part.casefold() if case_copy else part
            parts.append(part)
            positions.extend([(match.start(), match.end())] * len(part))
        normalized = "".join(parts)
        wanted = re.sub(r"\s+", " ", quote)
        wanted = wanted.casefold() if case_copy else wanted
        matches, offset = set(), 0
        while (start := normalized.find(wanted, offset)) >= 0:
            end = start + len(wanted)
            original = source[positions[start][0]:positions[end - 1][1]]
            spelling = re.sub(r"\s+", " ", original)
            # A match must consume complete source characters, including a
            # Unicode case-fold expansion (never match half of ß → ss).
            if (spelling.casefold() if case_copy else spelling) == wanted:
                matches.add(original)
            offset = start + 1
        return next(iter(matches)) if len(matches) == 1 else None

    original = resolve(False)
    return original if original is not None or not allow_case_copy else resolve(True)


def parse_extraction_response(content, finish_reason=None, *, template_mode=False,
                              allow_source_only=False):
    """Return complete facts, preserving polarity and arbitrary JSON key order.

    Errors carry structure/reasons only, never source text or model responses.
    A partially recoverable response is still incomplete: committing its prefix
    would make omissions permanent or count replay as fresh corroboration. The
    explicit source-only form retains an exact quote without asserting a triple.
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
        if not isinstance(fact, dict):
            raise ExtractionOutputError(f"invalid fact fields at index {index}")
        # NuExtract uses null for unavailable fields. An exact sentence remains
        # useful evidence even without a complete subject/predicate/object.
        # The caller must check the quote against its original source chunk;
        # this row cannot become a graph assertion.
        fields = ("subject", "relation", "object")
        source_only = (template_mode and allow_source_only
                       and all(key in fact for key in fields)
                       and any(fact[key] is None for key in fields)
                       and all(fact[key] is None or
                               (isinstance(fact[key], str) and fact[key].strip())
                               for key in fields)
                       and isinstance(fact.get("source_sentence"), str)
                       and bool(fact["source_sentence"].strip()))
        if source_only:
            fact["_source_only"] = True
        elif not all(isinstance(fact.get(key), str) and fact[key].strip()
                     for key in ("subject", "relation", "object")):
            raise ExtractionOutputError(f"invalid fact fields at index {index}")
        if "negated" in fact and not isinstance(fact["negated"], bool):
            raise ExtractionOutputError(f"non-boolean polarity at index {index}")
    return facts
