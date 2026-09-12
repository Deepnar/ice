"""Shared eligibility for stored turn representations, before caller budgeting.

Coverage checks term retention, not semantic truth. Abstracts cannot borrow the
summary's score: without independent verification only source spans qualify.
"""

import math

from src.api.config import settings

INTENTS_PREFER_RAW = {"Factual_Retrieval", "Troubleshooting"}
INTENTS_PREFER_SUMMARY = {
    "Analysis_&_Summarization",
    "Strategic_Planning",
    "Ideation",
    "Open_Exploration",
}


def choose_representation(row, classification=None, prompt_keywords=()):
    """Return (preferred text, shorter summary, source-extractive abstract)."""
    raw = getattr(row, "raw_text", None) or ""
    summary = getattr(row, "summary_text", None) or ""
    coverage = getattr(row, "summary_coverage", None)
    trusted = (
        isinstance(coverage, (int, float))
        and not isinstance(coverage, bool)
        and math.isfinite(coverage)
        and settings.turn_summary_coverage_threshold <= coverage <= 1.0
    )
    raw_lower = raw.lower()
    # Retain the existing singular fallback, but protect every matched term.
    matched = [
        (kw.lower(), kw.lower().rstrip("s") or kw.lower())
        for kw in prompt_keywords
        if kw
    ]
    matched = [forms for forms in matched if any(f in raw_lower for f in forms)]

    def preserves_terms(candidate):
        low = candidate.lower()
        return all(any(f in low for f in forms) for forms in matched)

    abstract = getattr(row, "abstract_text", None) or None
    if not (abstract and raw and abstract in raw and preserves_terms(abstract)):
        abstract = None
    if not (summary and trusted and preserves_terms(summary)):
        return raw or None, None, abstract
    if not raw:
        return summary, None, None
    intents = set(getattr(classification, "intent_tags", None) or [])
    if intents & INTENTS_PREFER_RAW:
        return raw, summary, abstract
    if intents & INTENTS_PREFER_SUMMARY:
        return summary, None, abstract
    if getattr(row, "inject_raw", True):
        return raw, summary, abstract
    return summary, None, abstract
