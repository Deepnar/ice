"""Shared eligibility for stored turn representations, before caller budgeting.

Coverage checks term retention, not semantic truth. Summaries and abstracts need
independent current source-support verdicts before replacing original evidence.
"""

import math

from src.api.config import settings
from src.memory.support import supported_current

INTENTS_PREFER_RAW = {"Factual_Retrieval", "Troubleshooting"}
INTENTS_PREFER_SUMMARY = {
    "Analysis_&_Summarization",
    "Strategic_Planning",
    "Ideation",
    "Open_Exploration",
}


def choose_representation(row, classification=None, prompt_keywords=()):
    """Return (preferred text, shorter summary, independently supported abstract)."""
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

    source = representation_source(row)
    records = getattr(row, "representation_verification", None)
    records = records if isinstance(records, dict) else {}
    abstract = getattr(row, "abstract_text", None) or None
    if not (abstract and supported_current(records.get("abstract"), source, abstract)
            and preserves_terms(abstract)):
        abstract = None
    if not (summary and trusted and preserves_terms(summary)
            and supported_current(records.get("summary"), source, summary)):
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


def representation_source(row):
    """Canonical attributed input for verification; never infer legacy roles."""
    import json
    from src.memory.source import source_units

    units = source_units(row)
    if not units or any(u.role == "unknown" for u in units):
        return None
    return "\n".join(f"The {u.role} said: {json.dumps(u.text, ensure_ascii=False)}"
                     for u in units if u.text.strip()) or None


def verify_representations(row, summary, abstract, *, verifier=None, proof=None):
    """Background-only verification; readers reuse immutable verdict identities."""
    from dataclasses import asdict
    import structlog
    from src.memory.support import verify_support

    source = representation_source(row)
    if source is None:
        if summary or abstract:
            structlog.get_logger("ice.memory.representation").warning(
                "representation_support_unknown", reason="source_roles_unknown")
        return {}
    result = {}
    proof_current = True
    if proof is not None:
        from src.memory.summary_support import original_units, windows_current
        proof_current = windows_current(original_units([row]), proof.get("windows", []))
    for kind, candidate in (("summary", summary), ("abstract", abstract)):
        if not candidate:
            continue
        if proof is not None:
            from src.memory.summary_support import verify_assertions
            if proof_current:
                dropped = [d for d in proof.get("dropped") or [] if d.get("kind") == kind]
                result[kind] = verify_assertions(source, candidate, proof[kind],
                                                 proof["windows"], verifier=verifier,
                                                 dropped=dropped)
            else:
                from src.memory.source import digest
                structlog.get_logger("ice.memory.representation").warning(
                    "representation_support_unknown", reason="source_window_identity_changed", kind=kind)
                result[kind] = {"status": "unknown", "scores": {}, "reason": "source_window_identity_changed",
                    "source_sha256": digest(source), "claim_sha256": digest(candidate),
                    "verifier": f"{settings.source_support_model}@{settings.source_support_revision}"}
        else:
            result[kind] = asdict((verifier or verify_support)(source, candidate))
    return result
