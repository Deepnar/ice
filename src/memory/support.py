"""Local source consistency with explicit uncertainty and complete-input bounds."""

import math
import threading
from dataclasses import dataclass

import structlog

from src.api.config import settings
from src.memory.source import digest

logger = structlog.get_logger('ice.memory.support')
_lock = threading.Lock()
_loaded = None
_loaded_key = None


class SupportInputError(ValueError):
    pass


@dataclass(frozen=True)
class SupportVerdict:
    status: str
    scores: dict
    reason: str
    source_sha256: str
    claim_sha256: str
    verifier: str


def score_pairs(pairs, *, model_name=None, revision=None, device=None):
    """Score complete pairs; errors propagate to the uncertainty boundary."""
    global _loaded, _loaded_key
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.memory.embedder import resolve_device

    model_name = model_name or settings.source_support_model
    revision = revision or settings.source_support_revision
    device = device or settings.source_support_device
    if not pairs:
        return []
    if any(not isinstance(p, str) or not p.strip()
           or not isinstance(h, str) or not h.strip() for p, h in pairs):
        raise SupportInputError('missing source or claim')
    with _lock:
        key = (model_name, revision)
        if _loaded_key != key:
            _loaded = None
            _loaded_key = key
        if _loaded is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision,
                local_files_only=True, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, revision=revision, local_files_only=True,
                trust_remote_code=False, dtype=torch.float32).eval()
            labels = {int(i): label.lower() for i, label in model.config.id2label.items()}
            if set(labels.values()) != {'entailment', 'neutral', 'contradiction'}:
                raise ValueError('unrecognized NLI label mapping')
            _loaded = (tokenizer, model, labels)
        tokenizer, model, labels = _loaded
        # Validate all inputs before moving weights or returning any scores.
        encoded = [tokenizer(p, h, return_tensors='pt', truncation=False)
                   for p, h in pairs]
        if any(e['input_ids'].shape[-1] > min(settings.source_support_max_tokens, model.config.max_position_embeddings)
               for e in encoded):
            raise SupportInputError('complete source/claim exceeds NLI token bound')
        target = resolve_device(device)
        try:
            model.to(target)
            result = []
            for inputs in encoded:
                with torch.inference_mode():
                    probabilities = model(**{k: v.to(target) for k, v in inputs.items()}).logits.softmax(-1)[0].cpu().tolist()
                scores = {labels[i]: float(p) for i, p in enumerate(probabilities)}
                if not all(math.isfinite(v) for v in scores.values()):
                    raise ValueError('nonfinite NLI output')
                result.append(scores)
            return result
        finally:
            model.cpu()
            if target.startswith('cuda'):
                torch.cuda.empty_cache()


def verify_support(source, claim, *, scorer=None):
    """Unknown/contradicted evidence selects the source, never source deletion."""
    source, claim = source or '', claim or ''
    verifier = f'{settings.source_support_model}@{settings.source_support_revision}'
    try:
        scores = (scorer or score_pairs)([(source, claim)])[0]
        if (set(scores) != {'entailment', 'neutral', 'contradiction'}
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values())
                or not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-4)):
            raise ValueError('invalid NLI probabilities')
        status = ('supported' if scores['entailment'] >= settings.source_support_threshold else
                  'contradicted' if scores['contradiction'] >= settings.source_support_threshold else 'unknown')
        reason = 'model_score'
    except Exception as exc:
        # A cooperative runtime yield is not a failed model verdict.
        from src.workers.runtime import JobYielded
        if isinstance(exc, JobYielded):
            raise
        scores, status, reason = {}, 'unknown', type(exc).__name__
        logger.warning('source_support_unknown', reason=reason,
                       source_chars=len(source), claim_chars=len(claim))
    return SupportVerdict(status, scores, reason, digest(source), digest(claim), verifier)


def supported_current(record, source, claim):
    """A persisted verdict authorizes only this exact source/candidate/model."""
    if not isinstance(record, dict) or not source or not claim:
        return False
    score = (record.get('scores') or {}).get('entailment') if isinstance(record.get('scores'), dict) else None
    return (record.get('status') == 'supported'
            and record.get('source_sha256') == digest(source)
            and record.get('claim_sha256') == digest(claim)
            and record.get('verifier') == f'{settings.source_support_model}@{settings.source_support_revision}'
            and isinstance(score, (int, float)) and not isinstance(score, bool)
            and math.isfinite(score) and settings.source_support_threshold <= score <= 1.0)
