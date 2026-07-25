"""Training dataset for the prompt classifier (schema v2).

Reads the JSONL that ``scripts/classifier/pipeline/build.py`` writes:

    {"text": ..., "context_text": ... | null,
     "topic": [...], "intent": [...], "context_reliance": [...],
     "source": ..., "labeler_agreement": ...}

and turns it into ``(embedding, multi-hot target)`` pairs.

Two things here are load-bearing rather than incidental:

* **Rows are rendered through ``templates.render``**, exactly as the live
  classifier renders an incoming prompt. Embedding bare prompt text — what the
  v1 trainer did — is the train/inference mismatch B1 exists to fix.
* **Encoding is at the embedder's native width** (1024). The v1 heads consumed a
  384-dim MRL prefix; nothing about the classifier ever required that, it was an
  artefact of the old default.

Embeddings are cached next to the JSONL, keyed by row count + template version,
because Z1-prep sweeps trunk width and thresholds — re-encoding 25k rows for each
sweep point is minutes of GPU time spent recomputing an identical tensor.
"""

import hashlib
import json
import os

import torch
from torch.utils.data import Dataset

from . import templates
from .schema import CONTEXT_RELIANCE, INTENT, TOPIC, load_schema

# The build stage emits head names as keys; this maps them for older row shapes
# that used the flat v1 field names.
_LEGACY_KEYS = {TOPIC: "topic_labels", INTENT: "intent_labels",
                CONTEXT_RELIANCE: "context_reliance_labels"}


def _row_labels(row, head):
    value = row.get(head.name, row.get(_LEGACY_KEYS.get(head.name, ""), []))
    if isinstance(value, str):
        return [value]
    return list(value or [])


class ICEClassifierDataset(Dataset):
    def __init__(self, training_data_path, schema=None, device="cpu",
                 batch_size: int = 256, cache: bool = True,
                 show_progress: bool = True):
        self.schema = schema or load_schema()
        self.data = []
        with open(training_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        self.labels = torch.zeros((len(self.data), self.schema.total_width),
                                  dtype=torch.float32)
        for i, row in enumerate(self.data):
            for head in self.schema.heads:
                for tag in _row_labels(row, head):
                    if head.has(tag):
                        self.labels[i, head.index(tag)] = 1.0

        rendered = [
            templates.render(row.get("text", row.get("prompt", "")),
                             row.get("context_text"),
                             version=self.schema.template_version)
            for row in self.data
        ]

        cache_path = self._cache_path(training_data_path, rendered) if cache else None
        if cache_path and os.path.exists(cache_path):
            self.embeddings = torch.load(cache_path, map_location="cpu")
        else:
            self.embeddings = self._encode(rendered, device, batch_size, show_progress)
            if cache_path:
                torch.save(self.embeddings, cache_path)

    def _cache_path(self, data_path, rendered):
        digest = hashlib.sha256()
        digest.update(str(len(rendered)).encode())
        digest.update(str(self.schema.template_version).encode())
        # First and last rendered rows pin the content without hashing 25k strings.
        if rendered:
            digest.update(rendered[0].encode())
            digest.update(rendered[-1].encode())
        return f"{data_path}.emb_{digest.hexdigest()[:12]}.pt"

    def _encode(self, rendered, device, batch_size, show_progress):
        from sentence_transformers import SentenceTransformer

        from src.api.config import settings
        model = SentenceTransformer(settings.embedding_model_name, device=device)
        out = model.encode(rendered, convert_to_tensor=True, batch_size=batch_size,
                           show_progress_bar=show_progress)
        return out.detach().cpu().float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
