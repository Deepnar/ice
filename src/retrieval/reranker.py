"""Local query/evidence relevance; neither entailment nor truth verification."""

import math
import threading
import time
from dataclasses import replace

import structlog

from src.api.config import settings
from src.memory.tokens import count as count_tokens

logger = structlog.get_logger("ice.retrieval.reranker")
INSTRUCTION = (
    "Given a question or task, retrieve memories that provide evidence useful "
    "for answering it. Related topic alone is insufficient. Historical or "
    "negative facts can be relevant."
)
_lock = threading.Lock()
_model = None
_model_key = None
_retry_after = 0.0


class RerankerInputTooLong(ValueError):
    """The complete templated input cannot be scored inside the work bound."""


def score_pairs(pairs):
    """Score complete inputs using cached weights; serialize GPU model access."""
    global _model, _model_key, _retry_after
    import torch
    from sentence_transformers import CrossEncoder

    from src.memory.embedder import resolve_device

    key = (
        settings.retrieval_rerank_model,
        settings.retrieval_rerank_revision,
        settings.retrieval_rerank_device,
    )
    with _lock:
        if _model_key != key:
            _model = None
            _retry_after = 0.0
            _model_key = key
        device = resolve_device(settings.retrieval_rerank_device)
        if _model is None:
            if time.monotonic() < _retry_after:
                raise RuntimeError("reranker load retry cooling down")
            try:
                _model = CrossEncoder(
                    settings.retrieval_rerank_model,
                    revision=settings.retrieval_rerank_revision,
                    local_files_only=True,
                    trust_remote_code=False,
                    device="cpu",
                    model_kwargs={
                        "dtype": torch.float16
                        if device.startswith("cuda")
                        else torch.float32
                    },
                    prompts={"ice": INSTRUCTION},
                    default_prompt_name="ice",
                )
                logger.info(
                    "reranker_loaded", model=key[0], revision=key[1], device=device
                )
            except Exception:
                _retry_after = time.monotonic() + 60
                raise

        try:
            scores = []
            batch_size = settings.retrieval_rerank_batch_size
            processing = {"text": {"truncation": False}}
            for offset in range(0, len(pairs), batch_size):
                batch = pairs[offset : offset + batch_size]
                # The exact inference template must fit: tokenizing raw strings
                # alone misses the instruction/chat wrapper. Never score a prefix
                # and credit the unseen rest with its relevance.
                features = _model.preprocess(
                    batch, prompt=INSTRUCTION, processing_kwargs=processing
                )
                if (
                    features["input_ids"].shape[-1]
                    > settings.retrieval_rerank_max_tokens
                ):
                    raise RerankerInputTooLong(
                        "reranker input exceeds configured token limit"
                    )
                scores.extend(
                    _model.predict(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        activation_fn=torch.nn.Identity(),
                        processing_kwargs=processing,
                        device=device,
                        logits_to_keep=1,
                        use_cache=False,
                    ).tolist()
                )
            return scores
        finally:
            if device.startswith("cuda"):
                # Retrieval precedes generation. Retain the cached weights in
                # RAM, but return GPU residency to the answering model.
                _model.to("cpu")
                torch.cuda.empty_cache()


def rerank(query, fragments, scorer=None):
    """Return (ranked fragments, succeeded); failures retain the entire input.

    Each eligible representation competes using its own score. An alternative
    can replace the primary text, but only shorter scored alternatives remain
    available for downstream budget degradation. Provenance is preserved.
    """
    if not settings.retrieval_rerank_enabled or not fragments or not query:
        return fragments, False
    start = time.monotonic()
    try:
        candidates = fragments[: settings.retrieval_rerank_candidates]
        variants = [
            list(
                dict.fromkeys(t for t in (f.text, f.degrade_text, f.abstract_text) if t)
            )
            for f in candidates
        ]
        pairs = [(query, t) for texts in variants for t in texts]
        scores = list((scorer or score_pairs)(pairs))
        if len(scores) != len(pairs) or any(
            not math.isfinite(float(s)) for s in scores
        ):
            raise ValueError("reranker returned invalid scores")
        output, offset = [], 0
        for fragment, texts in zip(candidates, variants):
            ranked = sorted(
                zip(texts, scores[offset : offset + len(texts)]),
                key=lambda p: (-p[1], count_tokens(p[0])),
            )
            offset += len(texts)
            floor = settings.retrieval_rerank_min_score
            ranked = [(t, float(s)) for t, s in ranked if floor is None or s >= floor]
            if not ranked:
                continue
            primary, score = ranked[0]
            tokens = count_tokens(primary)
            alternatives = [t for t, s in ranked[1:] if count_tokens(t) < tokens]
            output.append(
                replace(
                    fragment,
                    text=primary,
                    score=score,
                    token_count=tokens,
                    degrade_text=alternatives[0] if alternatives else None,
                    abstract_text=alternatives[1] if len(alternatives) > 1 else None,
                )
            )
        output.sort(key=lambda f: f.score, reverse=True)
        logger.info(
            "retrieval_reranked",
            candidates=len(candidates),
            omitted_by_cap=max(0, len(fragments) - len(candidates)),
            pairs=len(pairs),
            accepted=len(output),
            elapsed_ms=round((time.monotonic() - start) * 1000, 1),
        )
        return output, True
    except Exception as err:
        # No query, memory text, or provider error payload in logs.
        logger.warning(
            "retrieval_rerank_degraded",
            reason=type(err).__name__,
            candidates=len(fragments),
        )
        return fragments, False
