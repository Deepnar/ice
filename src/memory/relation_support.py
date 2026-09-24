"""Check a proposed Codex relation against its attributed original passage."""

import re

import structlog

from src.api.config import settings
from src.memory import support
from src.memory.source import digest


logger = structlog.get_logger("ice.memory.relation_support")


def relation_proposition(subject: str, relation: str, object_name: str,
                         *, negated: bool = False) -> str:
    """Verbalize the open relation without guessing an ontology or tense."""
    if any(not isinstance(value, str) or not value.strip()
           for value in (subject, relation, object_name)):
        return ""
    predicate = re.sub(r"\s+", " ", relation.replace("_", " ")).strip()
    statement = f"{subject.strip()} {predicate} {object_name.strip()}."
    return f"It is false that {statement}" if negated else statement


def verify_relation_pairs(pairs: list[tuple[str, str]], *, scorer=None):
    """Score a turn's proposals in one lease; only oversize pairs abstain.

    A provider/model outage propagates so the graph and completion key roll
    back together. A single passage beyond the model's complete-input bound is
    an unknown verdict; splitting isolates it without starving short pairs.
    """
    if not pairs:
        return []
    score = scorer or support.score_pairs
    try:
        scored = score(pairs)
    except support.SupportInputError:
        if len(pairs) > 1:
            middle = len(pairs) // 2
            return (verify_relation_pairs(pairs[:middle], scorer=score)
                    + verify_relation_pairs(pairs[middle:], scorer=score))
        source, claim = pairs[0]
        logger.warning("relation_support_input_unknown", source_chars=len(source),
                       claim_chars=len(claim))
        return [support.SupportVerdict(
            "unknown", {}, "SupportInputError", digest(source), digest(claim),
            f"{settings.source_support_model}@{settings.source_support_revision}")]
    if len(scored) != len(pairs):
        raise ValueError("relation verifier returned the wrong number of scores")
    verdicts = [support.verify_support(source, claim,
        scorer=lambda _pairs, result=result: [result])
        for (source, claim), result in zip(pairs, scored)]
    if any(v.reason != "model_score" for v in verdicts):
        raise ValueError("relation verifier returned invalid scores")
    return verdicts
