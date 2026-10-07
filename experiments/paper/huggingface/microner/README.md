---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3-Embedding-0.6B
library_name: pytorch
tags:
- conversational-memory
- named-entity-recognition
- lsrep
- ice-v2
- reproducibility
- arxiv:2609.16730
---

# ICE v2 / LSREP MicroNER

This releases the retained original MicroNER checkpoint loaded by frozen
**ICE v2**, evaluated in the [LSREP paper](https://arxiv.org/abs/2609.16730).
It is **not a release of current ICE v3 research models**.

- Repository: [Deepnar/ice](https://github.com/Deepnar/ice).
- Evaluated tag: [`v2-paper-eval`](https://github.com/Deepnar/ice/tree/v2-paper-eval).
- Evaluated commit: `0521df9171b4a7d69f82d12d70497138c77b2678`.
- Original path: `models/ner/ner_model.pt`.
- Original file size: 234,093 bytes; bare PyTorch state dict.
- SHA-256: `23e596654065bde16f822db4b6dacf1830f67ae3613234bb7d808f5ea214c1b2`.

The `.pt` file is copied byte for byte, without retraining or re-export. The
frozen Git tag records the loader and checkpoint path, but does not contain the
checkpoint blob or a historical checksum. The release hashes the retained original.

## Architecture, labels and input

`Linear(384,128) -> ReLU -> Dropout(0.2) -> Linear(128,64) -> ReLU
-> Dropout(0.2) -> Linear(64,3)`; 57,731 trainable parameters.
The exact frozen `ner_model.py` is included. Input shape is `(batch,tokens,384)`;
output is `(batch,tokens,3)`. Disable dropout with `eval()`. The label order is:

```text
0 = B-ENT   beginning of a generic entity
1 = I-ENT   continuation of a generic entity
2 = O       outside
```

These are generic entity spans, not typed person/organization/location classes.
Each token is decoded by argmax. There is no CRF, contextual token encoder,
softmax threshold, or typed-entity classification head in this artifact.

## Required preprocessing and span reconstruction

Use the tokenizer from `Qwen/Qwen3-Embedding-0.6B`, revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
Tokenize the supplied text with `return_offsets_mapping=True` and
`add_special_tokens=False`, then call `convert_ids_to_tokens()`.
Pass those **raw tokenizer token strings**, including any subword markers,
individually to `SentenceTransformer.encode()` as a list of separate inputs.
Do not replace them with decoded whole words or transformer hidden states.

The frozen base has native width 1024; set `truncate_dim=384`. Preserve the
snapshot's built-in pooling/normalization. Do not request additional
`normalize_embeddings=True`, renormalize the 384-coordinate prefix, or use
current ICE v3's native-width encoder path.

Use tokenizer character offsets to reconstruct spans. An `I-ENT` without an
open `B-ENT` is ignored. Whitespace-adjacent predicted spans are joined, matching
the frozen loader. Retrieval defaults to full text (`max_chars=None`); callers
may explicitly truncate text before tokenization, as clustering does. The
original `frozen_ner_utils.py` is supplied verbatim for audit; the included
`microner_inference.py` implements its trained-model path without its regex fallback.

## Minimal inference

The frozen tag uses Python 3.11, `torch==2.11.0`,
`sentence-transformers==5.5.1`, and `transformers==5.9.0`.

```python
import hashlib
import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

folder = Path(snapshot_download("Deepnar/ice-v2-microner"))
sys.path.insert(0, str(folder))
from ner_model import MicroNER
from microner_inference import extract_entities

checkpoint = folder / "ner_model.pt"
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
    "23e596654065bde16f822db4b6dacf1830f67ae3613234bb7d808f5ea214c1b2"
)
head = MicroNER()
head.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
head.eval()
base = "Qwen/Qwen3-Embedding-0.6B"
revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
embedder = SentenceTransformer(base, revision=revision, device="cpu", truncate_dim=384)
print(extract_entities(head, tokenizer, embedder, "Alice works at Example Labs in Berlin."))
```

For durable reuse, pin `snapshot_download(..., revision=<release commit>)`
to the upload commit in the GitHub release verification report. This custom
head is not a Transformers token-classification pipeline. Neither a successful
load nor a nonempty span is a validation of entity correctness or graph utility.

## Limitations and license

The paper establishes that the trained ICE v2 checkpoint was present and loaded;
it does not establish task-specific NER recall or graph accuracy. No new quality
measurement is claimed here. Independent token-string embeddings, generic BIO
labels, subword segmentation and whitespace joining limit span accuracy.
Private corpora, probes and training data are excluded; the release supports
checkpoint/inference reuse, not independent regeneration of private training.

The ICE head and accompanying ICE code use the repository's Apache-2.0 license;
`LICENSE` and `NOTICE` are included. The separately fetched Qwen base model is
Apache-2.0 according to its pinned upstream model card. No upstream base weights,
private text, caches, secrets, unrelated checkpoints or ICE v3 artifacts are uploaded.
