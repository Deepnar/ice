"""Local query/evidence relevance; neither entailment nor truth verification."""

import math
import threading
import time
from collections import Counter
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
            scores = [None] * len(pairs)
            batch_size = settings.retrieval_rerank_batch_size
            processing = {"text": {"truncation": False}}

            def fits(pair):
                # The exact inference template must fit: tokenizing raw strings
                # alone misses the instruction/chat wrapper.
                features = _model.preprocess(
                    [pair], prompt=INSTRUCTION, processing_kwargs=processing
                )
                return features["input_ids"].shape[-1] <= settings.retrieval_rerank_max_tokens

            # Audit A4 (2026-10-15): an over-bound passage used to stay
            # unscored and was appended after every scored one, so the
            # longest (often densest) memories always ranked last. It is now
            # scored by its best window: every window is read, none is
            # credited for text the model did not see.
            units, owners = [], []
            for index, pair in enumerate(pairs):
                windows = [pair] if fits(pair) else _windows(pair, fits)
                if not windows:
                    logger.warning("retrieval_rerank_pair_unscored",
                                   reason="no_window_fits_token_bound")
                    continue
                if len(windows) > 1:
                    logger.info("retrieval_rerank_pair_windowed", windows=len(windows))
                units.extend(windows)
                owners.extend([index] * len(windows))
            eligible = list(range(len(units)))
            for offset in range(0, len(eligible), batch_size):
                indices = eligible[offset:offset + batch_size]
                batch = [units[index] for index in indices]
                values = _model.predict(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        activation_fn=torch.nn.Identity(),
                        processing_kwargs=processing,
                        device=device,
                        logits_to_keep=1,
                        use_cache=False,
                    ).tolist()
                for index, value in zip(indices, values, strict=True):
                    owner = owners[index]
                    scores[owner] = value if scores[owner] is None else max(scores[owner], value)
            return scores
        finally:
            if device.startswith("cuda"):
                # Retrieval precedes generation. Retain the cached weights in
                # RAM, but return GPU residency to the answering model.
                _model.to("cpu")
                torch.cuda.empty_cache()


def candidate_pool(fragments, cap):
    """Let each source family compete under the existing overall input cap."""
    if cap < 1:
        raise ValueError('reranker candidate cap must be positive')
    if len(fragments) <= cap:
        return fragments
    selected, families = set(), set()
    for index, fragment in enumerate(fragments):
        if fragment.source_type not in families:
            selected.add(index)
            families.add(fragment.source_type)
            if len(selected) == cap:
                break
    for index in range(len(fragments)):
        if len(selected) >= cap:
            break
        selected.add(index)
    return [fragments[index] for index in sorted(selected)]


def _windows(pair, fits):
    """Overlapping passage windows that each fit with the query and template.

    Character windows halve until every window fits; a quarter overlaps the
    next, so evidence on a boundary is read whole by one of them.
    """
    query, passage = pair
    size = max(1, len(passage) // 2)
    while size >= 64:
        step = max(1, size - size // 4)
        starts = range(0, max(1, len(passage) - size + step), step)
        windows = [(query, passage[a:a + size]) for a in starts if passage[a:a + size].strip()]
        if windows and all(fits(w) for w in windows):
            return windows
        size //= 2
    return []


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
        candidates = candidate_pool(fragments, settings.retrieval_rerank_candidates)
        # Z2 F17: "best_score" lets a shorter form out-score the text choose_representation picked —
        # on B the policy then had no visible effect (~50% summaries whatever it preferred).
        # "policy" scores only the picked text; the shorter forms stay for budget degradation.
        policy = settings.retrieval_rerank_representation == "policy"
        variants = [
            [f.text] if policy else list(
                dict.fromkeys(t for t in (f.text, f.degrade_text, f.abstract_text) if t)
            )
            for f in candidates
        ]
        pairs = [(query, t) for texts in variants for t in texts]
        scores = list((scorer or score_pairs)(pairs))
        if len(scores) != len(pairs) or any(
            s is not None and not math.isfinite(float(s)) for s in scores
        ):
            raise ValueError("reranker returned invalid scores")
        if all(s is None for s in scores):
            return fragments, False
        output, unscored, offset = [], [], 0
        for fragment, texts in zip(candidates, variants):
            values = scores[offset:offset + len(texts)]
            for text, value in zip(texts, values):
                if value is None:
                    unscored.append(replace(fragment, text=text,
                        token_count=count_tokens(text), degrade_text=None, abstract_text=None,
                        covers_entire_source=fragment.covers_entire_source and text == fragment.text))
            ranked = sorted(
                ((t, s) for t, s in zip(texts, values) if s is not None),
                key=lambda p: (-p[1], count_tokens(p[0])),
            )
            offset += len(texts)
            floor = settings.retrieval_rerank_min_score
            ranked = [(t, float(s)) for t, s in ranked if floor is None or s >= floor]
            if not ranked:
                continue
            primary, score = ranked[0]
            margin = settings.retrieval_rerank_original_margin
            if margin and primary != fragment.text:
                # Z2 F14: a summary/abstract stands in for the turn's own text only when it
                # scores clearly better; it was faithful but complete on ~63% of B/C turns.
                own = next((s for t, s in ranked if t == fragment.text), None)
                if own is not None and score - own <= margin:
                    primary, score = fragment.text, own
            tokens = count_tokens(primary)
            alternatives = ([t for t in (fragment.degrade_text, fragment.abstract_text) if t] if policy
                            else [t for t, s in ranked if t != primary and count_tokens(t) < tokens])
            output.append(
                replace(
                    fragment,
                    text=primary,
                    score=score,
                    token_count=tokens,
                    degrade_text=alternatives[0] if alternatives else None,
                    abstract_text=alternatives[1] if len(alternatives) > 1 else None,
                    covers_entire_source=fragment.covers_entire_source and primary == fragment.text,
                )
            )
        output.sort(key=lambda f: f.score, reverse=True)
        output.extend(unscored)
        kept = []
        if settings.retrieval_rerank_keep_unranked and settings.retrieval_rerank_min_score is None:
            # Z2 F2: candidates past the cap stay, behind every reranked one and in fusion
            # order; their fusion scores are on another scale, so they sit below the lowest.
            # Under a relevance floor they were never shown to pass it, so they are not kept.
            pooled = {id(f) for f in candidates}
            floor = min((f.score for f in output), default=0.0) - 1.0
            kept = [replace(f, score=floor - i * 1e-6)
                    for i, f in enumerate(f for f in fragments if id(f) not in pooled)]
            output.extend(kept)
        logger.info(
            "retrieval_reranked",
            candidates=len(candidates),
            omitted_by_cap=max(0, len(fragments) - len(candidates) - len(kept)),
            kept_unranked=len(kept),
            pool_method='family_representative_v1',
            available_families=dict(Counter(f.source_type for f in fragments)),
            pool_families=dict(Counter(f.source_type for f in candidates)),
            pairs=len(pairs),
            scored=len(output) - len(unscored),
            unscored_fallbacks=len(unscored),
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
