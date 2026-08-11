"""G29 — the one place a model's reply is turned into JSON.

Five call sites each had their own answer, and they disagreed about the only
thing that matters, which is what happens when parsing fails:

  * ``model_registry/registry.py`` — bare ``json.loads``. Any fenced reply threw,
    and the caller returned empty tags, which is **indistinguishable from a model
    that genuinely has no tags**. That is how a 404 stayed invisible.
  * ``codex_extractor`` / ``maintenance_agent`` — byte-identical fence-strips,
    written twice.
  * ``reflection`` — the only one that got it right: extract, and if that fails,
    **say so at WARNING with the head of the text**. Its reasoning is the module
    docstring's now.
  * ``ingestion/raw_slicer`` — a third regex variant for arrays.

The rule this module encodes: **a salvage that silently returns nothing is worse
than a crash**, because an empty result reads as "the model had nothing to say"
and a whole background layer can look healthy while doing nothing (CLAUDE.md's
silent-fallback rule; TRAPS #11). Every path here is loud.

Not in scope: reading JSON *files* (``portability``, ``formats``,
``project_facts``, ``classifier/dataset``). Those parse data we wrote ourselves,
where a failure is corruption rather than an unreliable narrator, and they should
keep raising.
"""

import json
import re

import structlog

logger = structlog.get_logger("ice.workers.llm_json")


def strip_fences(raw: str) -> str:
    """Remove a markdown code fence if the model wrapped its reply in one.

    Byte-identical to the copies in codex_extractor and maintenance_agent, which
    is why it lives here now.
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw[3:]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _parse(raw: str, pattern: str, kind, *, where: str):
    text = strip_fences(raw)
    if not text:
        logger.warning("llm_json_empty", where=where, kind=kind.__name__)
        return None
    try:
        return _validated(json.loads(text), kind, where, text)
    except Exception:
        pass                      # fall through to extraction, then complain once
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return _validated(json.loads(match.group(0)), kind, where, text)
        except Exception as exc:
            logger.warning("llm_json_unparseable", where=where, kind=kind.__name__,
                           error=str(exc)[:120], head=text[:120])
            return None
    # Every call site is schema-constrained, so getting here means the constraint
    # did not hold; truncation (finish_reason="length") is the known cause.
    logger.warning("llm_json_absent", where=where, kind=kind.__name__,
                   chars=len(text), head=text[:120])
    return None


def _validated(value, kind, where, text):
    if isinstance(value, kind):
        return value
    logger.warning("llm_json_wrong_type", where=where, expected=kind.__name__,
                   got=type(value).__name__, head=text[:120])
    return None


def parse_object(raw: str, *, where: str = "?"):
    """A JSON object from model output, or None — never a silent ``{}``."""
    return _parse(raw, r"\{.*\}", dict, where=where)


def parse_array(raw: str, *, where: str = "?"):
    """A JSON array from model output, or None — never a silent ``[]``."""
    return _parse(raw, r"\[.*\]", list, where=where)
