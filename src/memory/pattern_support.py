"""Description equivalence is independent of habit-source truth."""

from dataclasses import asdict

import structlog

from src.api.config import settings
from src.memory.source import digest
from src.memory.support import score_pairs, support_tokenizer, verify_support

METHOD = 'mutual_entailment_v1'
logger = structlog.get_logger('ice.memory.pattern_support')


def equivalence_receipts(proposed, existing, *, scorer=None):
    """Batch both complete directions; unknown never grants merge authority."""
    receipts, pairs, positions = [], [], []
    for index, candidate in enumerate(existing):
        record = dict(method=METHOD, proposed_sha256=digest(proposed),
                      existing_sha256=digest(candidate),
                      verifier=f'{settings.source_support_model}@{settings.source_support_revision}',
                      policy_threshold=settings.source_support_threshold,
                      status='unknown', directions=[])
        receipts.append(record)
        if proposed == candidate:
            record.update(status='equivalent', reason='exact_description_identity')
            continue
        if scorer is None:
            # A bad candidate cannot poison the other bounded comparisons.
            try:
                tokenizer = support_tokenizer()
                if any(len(tokenizer(p, h, truncation=False)['input_ids']) >
                       settings.source_support_max_tokens
                       for p, h in ((proposed, candidate), (candidate, proposed))):
                    record['reason'] = 'SupportInputError'
                    logger.warning('procedural_equivalence_unknown',
                                   reason=record['reason'], candidate=index)
                    continue
            except Exception as exc:
                from src.workers.runtime import JobYielded
                if isinstance(exc, JobYielded):
                    raise
                record['reason'] = type(exc).__name__
                logger.warning('procedural_equivalence_unknown',
                               reason=record['reason'], candidate=index)
                continue
        pairs.extend([(proposed, candidate), (candidate, proposed)])
        positions.append(index)
    try:
        scores = (scorer or (lambda values: score_pairs(values, device='cpu')))(pairs) if pairs else []
        if len(scores) != len(pairs):
            raise ValueError('incomplete equivalence scores')
    except Exception as exc:
        from src.workers.runtime import JobYielded
        if isinstance(exc, JobYielded):
            raise
        logger.warning(
            'procedural_equivalence_unknown', reason=type(exc).__name__, candidates=len(positions))
        for index in positions:
            receipts[index]['reason'] = type(exc).__name__
        return receipts
    for offset, index in enumerate(positions):
        directions = [asdict(verify_support(premise, claim, scorer=lambda _p, sc=score: [sc]))
            for (premise, claim), score in zip(pairs[offset * 2:offset * 2 + 2],
                                             scores[offset * 2:offset * 2 + 2])]
        statuses = [v['status'] for v in directions]
        receipts[index].update(directions=directions, reason='bidirectional_model_score',
            status='equivalent' if statuses == ['supported', 'supported']
            else 'different' if 'contradicted' in statuses else 'unknown')
    return receipts
