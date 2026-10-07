# Frozen ICE v2 paper models

These release packages belong to the LSREP / **ICE v2** paper, evaluated at
`v2-paper-eval` (`0521df9171b4a7d69f82d12d70497138c77b2678`). They are separate
from current ICE v3 development.

- [Classifier](https://huggingface.co/Deepnar/ice-v2-classifier)
- [MicroNER](https://huggingface.co/Deepnar/ice-v2-microner)
- [ICE / LSREP Reproducibility collection](https://huggingface.co/collections/Deepnar/ice-lsrep-reproducibility-6ac6210257541dd6f8594e14)

`RELEASE_MANIFEST.json` records original paths, checkpoint hashes, byte counts,
base revision and frozen-source hashes. `PUBLIC_VERIFICATION.json` records
the immutable upload commits and unauthenticated public hash checks.
The classifier's historical `v3_qwen_ft3` filename is a training generation;
it is the **system-v2** checkpoint selected by the frozen configuration.

The Git tag stores code and checkpoint paths, not weight blobs. These are
the locally retained originals; the release checksum is recorded now and is
not represented as a checksum archived at evaluation time. The pinned Qwen
snapshot is independently documented in `experiments/lme/README.md`.

The original `.pt` files are on Hugging Face, not duplicated in this Git
directory. Model cards, configurations, inference helpers and verbatim
frozen source are retained here for review. No `src/` code is changed.
The copied historical modules are audit artifacts; do not update them to
follow ICE v3. The new helpers expose only the historical learned paths.

Validation used the pinned base on CPU with authored public examples:
both checkpoint dimensions/parameter counts; exact raw-probability parity
against the frozen classifier with and without context; and exact span
parity against frozen MicroNER with full text, a character cap and empty text.
This establishes loading/preprocessing equivalence, not new quality scores.

Both ICE heads/code and the separately fetched Qwen base use Apache-2.0.
The Hub packages include the repository's `LICENSE` and `NOTICE`; base
weights, training corpora, private text, caches and credentials are excluded.
