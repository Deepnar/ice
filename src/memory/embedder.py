"""G23 D6: the process's ONE embedding model.

Before G23, five modules each loaded their own SentenceTransformer copy
with ``truncate_dim=384`` (classifier, codex_extractor, procedural_extractor,
batch_summarizer, clustering) — a standing G13 violation and ~5× the RAM.
Every writer and retrieval path now shares this singleton. ``encode()``
returns the model's native width (settings.embedding_dim; unit-norm — the
Normalize module runs last, which is why A4's re-normalization workaround
could be deleted from the retrieval paths).

The 384-input consumers take ``slice384()`` of the same vector: the raw
first-384 prefix is empirically bit-identical to the old ``truncate_dim=384``
output (sentence-transformers truncates AFTER Normalize and does NOT
re-normalize; verified against ST 5.5.1, maxdiff 0.0), so their checkpoints
keep working unchanged.

**Who still needs it, after B1 (2026-07-27).** Two consumers, for two
different reasons:

* the **micro-NER**, which is simply still a 384-input model — A9c may retrain
  it at native width one day, and until then this is a live path;
* the **v1 classifier checkpoint**, which is no longer served but IS the
  rollback artifact and D5's non-regression baseline. A9a proposed deleting
  the classifier's narrowing on the grounds that it was dead code; it is not
  — it is what makes a rollback a file swap instead of a code change, exactly
  like ``LegacyICEClassifierV1``. So it stays, and ``fit_width`` below is the
  one place it lives instead of the six copies it had grown into.
"""

import structlog

logger = structlog.get_logger("ice.memory.embedder")

_embedder = None

# The MRL prefix width the pre-B1 classifier / micro-NER checkpoints were
# trained on (== the old truncate_dim).
SLICE_DIM = 384


def resolve_device(preference: str = "auto") -> str:
    """Turn the `embedding_device` setting into a real torch device.

    "auto" (default) = the GPU if there is one, else CPU. Explicit "cuda" or
    "cpu" is honoured as written — with "cuda" falling back loudly rather than
    crashing a boot on a machine that has no GPU, because a memory system that
    refuses to start is worse than a slow one.
    """
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if pref == "cuda":
        if not has_cuda:
            logger.warning("embedding_device_unavailable", requested="cuda",
                           using="cpu")
            return "cpu"
        return "cuda"
    return "cuda" if has_cuda else "cpu"


def get_embedder():
    """Lazy process singleton. Torch/sentence-transformers imports are
    deferred so light callers (scripts, migrations, smoke) don't pay the
    model load until someone actually encodes.

    C16: the device is a SETTING, defaulting to the GPU when one exists. It was
    hardcoded to "cpu" — carried over, uncommented, when G23/C17 consolidated
    five separate embedder copies into this one — and measured on this repo's
    own model that cost **321 ms per encode against 21 ms on GPU**, 22.5 s
    against 0.38 s for a batch of 100. Every chat turn encodes the user's
    prompt on the latency-sensitive pre-flight path, so ~300 ms was being added
    to every single request while the card sat idle. The embedder holds ~1.2 GB
    of VRAM; `embedding_device="cpu"` gives it back on a machine where the
    generation model needs every byte.
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        from src.api.config import settings
        device = resolve_device(getattr(settings, "embedding_device", "auto"))
        try:
            _embedder = SentenceTransformer(settings.embedding_model_name,
                                            device=device)
        except Exception as exc:
            # The GPU can be full — the generation model shares it, and on this
            # machine Ollama alone can hold 17+ GB of a 24 GB card. A memory
            # system that refuses to start because the card is busy is worse
            # than a slow one, so fall back and say so at WARNING. Set
            # embedding_device="cpu" to stop the retry entirely.
            if device == "cpu":
                raise
            logger.warning("embedder_gpu_load_failed", error=str(exc),
                           falling_back_to="cpu")
            device = "cpu"
            _embedder = SentenceTransformer(settings.embedding_model_name,
                                            device=device)
        logger.info("embedder_loaded", model=settings.embedding_model_name,
                    device=device)
    return _embedder


def slice384(vec):
    """First 384 dims of a native embedding (raw prefix, NO re-norm — see
    module docstring). Works on 1-D and batched 2-D tensors/ndarrays."""
    return vec[..., :SLICE_DIM]


def fit_width(vec, target_dim: int):
    """Narrow a native encode to the width a loaded checkpoint expects.

    A9a (2026-07-27). Six call sites had grown their own copy of the same three
    lines — ``classifier._encode``, ``workers/fine_tune._encode`` and the four
    B1 pipeline stages (``evaluate``, ``eval_probes``, ``score_hard_probes``,
    ``tune_b2``) — each re-deciding what a 384-wide head means. That is the
    duplication G23 already removed once for ``truncate_dim``, growing back one
    ``if`` at a time, and it is the kind that rots unevenly: a fix applied to
    five of six copies is indistinguishable from a fix applied to all of them
    until a rollback exercises the sixth.

    The **only** legal narrowing is the MRL prefix the pre-B1 checkpoints were
    trained on. Anything else raises rather than silently feeding a head an
    input it was not trained on — a wrong-width tensor that happens to be
    accepted produces plausible garbage, which is the worst failure available.
    """
    have = vec.shape[-1]
    if target_dim == have:
        return vec
    if target_dim == SLICE_DIM:
        return slice384(vec)
    raise ValueError(
        f"cannot fit a {have}-dim embedding to a {target_dim}-dim head: the "
        f"only supported narrowing is the {SLICE_DIM}-dim MRL prefix")
