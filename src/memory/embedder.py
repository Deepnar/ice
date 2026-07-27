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

_embedder = None

# The MRL prefix width the pre-B1 classifier / micro-NER checkpoints were
# trained on (== the old truncate_dim).
SLICE_DIM = 384


def get_embedder():
    """Lazy process singleton. Torch/sentence-transformers imports are
    deferred so light callers (scripts, migrations, smoke) don't pay the
    model load until someone actually encodes."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        from src.api.config import settings
        _embedder = SentenceTransformer(settings.embedding_model_name,
                                        device="cpu")
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
