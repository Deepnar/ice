---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3-Embedding-0.6B
library_name: pytorch
tags:
- conversational-memory
- lsrep
- ice-v2
- reproducibility
- arxiv:2609.16730
---

# ICE v2 / LSREP experiment classifier

This releases the retained original classifier checkpoint selected by frozen
**ICE v2**, the system evaluated in the [LSREP paper](https://arxiv.org/abs/2609.16730).
It is **not the current ICE v3 classifier**. The `v3` in the historical
checkpoint filename denotes a classifier training generation, not ICE's system version.

- Repository: [Deepnar/ice](https://github.com/Deepnar/ice).
- Evaluated source tag: [`v2-paper-eval`](https://github.com/Deepnar/ice/tree/v2-paper-eval).
- Evaluated commit: `0521df9171b4a7d69f82d12d70497138c77b2678`.
- Original path: `models/classifier/ice_classifier_v3_qwen_ft3.pt`.
- Original file size: 212,827 bytes; bare PyTorch state dict.
- SHA-256: `25c758b6a7e5cf449f3e4c8bb250db759d37cb4f0ab7dd8e0c1acd8afbf05831`.

The checkpoint is copied byte for byte, without retraining or re-export.
The frozen Git tag records code and the selected path, but does not contain
checkpoint blobs or a historical checkpoint checksum. This release records the
checksum of the retained original, rather than claiming a checksum existed at evaluation time.

## Architecture and label order

`Linear(384,128) -> ReLU -> Dropout(0.3) -> Linear(128,25)`;
52,505 trainable parameters. Call `eval()` to disable dropout.
The frozen `model.py` is included verbatim. Output coordinates, in order:

```text
0:11 topics:
Software_&_Tech, STEM_&_Academics, Business_&_Finance,
Creative_&_Media, Admin_&_Productivity, Lifestyle_&_Health,
Social_&_Relationships, World_&_Current_Events, Meta_AI,
Null_Noise, General_Reference_&_Trivia

11:22 intents:
Factual_Retrieval, Troubleshooting, Generation, Ideation,
Analysis_&_Summarization, Strategic_Planning, Decision_Making,
Emotional_Processing, Utility_Formatting, Casual_Banter, Open_Exploration

22:25 context:
Zero_Shot, Long_Term_Memory, Real_Time_Search
```

Topic and intent use sigmoid, a strict `> 0.3` threshold, and an argmax
fallback when a block has no selected label. Context uses softmax and argmax,
not independent sigmoids. `max_confidence` is the largest of the 25 decoded
probabilities. Exact order and settings are also in `config.json`.

## Embeddings and preprocessing

Use frozen `Qwen/Qwen3-Embedding-0.6B`, revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, with
`SentenceTransformer(..., device="cpu", truncate_dim=384)`.
The upstream native width is 1024; ICE v2 takes its first 384 coordinates.
Use the snapshot's built-in pooling and normalization. Do **not** add
`normalize_embeddings=True` to `encode()`, renormalize the truncated prefix,
add a query instruction, or use the current ICE v3 native-width embedding path.

Embed the exact instructional prefix produced by `build_input()` in the included
`classifier_inference.py`, not the bare user prompt. With context, ICE v2
selects the last three episodic rows in timestamp order, prefers each summary,
otherwise uses raw text capped at 150 whitespace words, and caps the combined
context at 500 words. The frozen `frozen_classifier.py` preserves the exact
context selection and truncation behavior, including ellipses and the
context-specific prefix. An empty context uses the no-context prefix.

## Minimal learned-head inference

The frozen tag's dependency versions are `torch==2.11.0`,
`sentence-transformers==5.5.1`, and `transformers==5.9.0`.
Use Python 3.11 and `huggingface_hub` to fetch the release:

```python
import hashlib
import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

folder = Path(snapshot_download("Deepnar/ice-v2-classifier"))
sys.path.insert(0, str(folder))
from model import ICEClassifier
from classifier_inference import predict_head

checkpoint = folder / "ice_classifier_v3_qwen_ft3.pt"
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
    "25c758b6a7e5cf449f3e4c8bb250db759d37cb4f0ab7dd8e0c1acd8afbf05831"
)
head = ICEClassifier()
head.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
head.eval()
embedder = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    device="cpu", truncate_dim=384,
)
print(predict_head(head, embedder, "Explain how a database index works."))
```

For archival reuse, pin `snapshot_download(..., revision=<release commit>)`
to the upload commit recorded in the GitHub release verification report.
This is a custom PyTorch head, not a Transformers `AutoModel` or hosted pipeline.
The example returns **learned-head predictions**. The complete ICE v2 classifier
also runs the DI3 pre-classifier, hard overrides, and API-level memory policy;
reproduce those through the frozen Git repository. Neither this helper nor the
weights alone reproduce full-system routing or the paper's end-to-end scores.

## Limitations and license

No new classifier accuracy claim is made by this release. The paper's fidelity
audit and negative results remain applicable. Context, preprocessing, rule
overrides and workload affect behavior. Training data and private conversational
corpora are not released, so this is checkpoint/inference reproducibility,
not a claim that private-data training can be independently regenerated.

The ICE head and accompanying ICE code are released under the repository's
Apache-2.0 license; `LICENSE` and `NOTICE` are included. The separately fetched
Qwen base model is also Apache-2.0 according to its pinned upstream model card.
No Qwen base weights, private corpora, caches, secrets, or ICE v3 artifacts are included.
