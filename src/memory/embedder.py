"""G23 D6: the process's ONE embedding model.

Before G23, five modules each loaded their own SentenceTransformer copy
with ``truncate_dim=384`` (classifier, codex_extractor, procedural_extractor,
batch_summarizer, clustering) — a standing G13 violation and ~5× the RAM.
Every writer and retrieval path now shares this singleton. ``encode()``
returns the model's native width (settings.embedding_dim; unit-norm — the
Normalize module runs last, which is why A4's re-normalization workaround
could be deleted from the retrieval paths).

The un-retrained 384-input consumers (classifier MLP, micro-NER) take
``slice384()`` of the same vector: the raw first-384 prefix is empirically
bit-identical to the old ``truncate_dim=384`` output (sentence-transformers
truncates AFTER Normalize and does NOT re-normalize; verified against ST
5.5.1, maxdiff 0.0), so their checkpoints keep working unchanged until
B1/A9 retrain at native width — at which point slice384's call sites are
deleted (the helper stays for any future MRL staging).
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
