# src/classifier/model.py
"""B1 D2: the classifier network — shared trunk, one head per label group.

v1 was ``Linear(384→128) → ReLU → Dropout(0.3) → Linear(128→25)``: a single flat
output whose three label groups were carved out by hardcoded slices, fed by the
384-dim MRL prefix of the embedding. v2 is

    Linear(1024→512) → GELU → Dropout(0.2) → Linear(512→256) → GELU → Dropout(0.2)
      ├── Linear(256→11)   topic
      ├── Linear(256→12)   intent
      └── Linear(256→ 4)   context_reliance

~700k params — still minutes to train on the laptop. Three changes carry weight:

* **Native 1024 input** (C17). The classifier's input never depended on the DB's
  stored vector width, so this is just "stop throwing away 640 dims".
* **A shared trunk with separate heads** instead of one flat layer. Topic and
  intent genuinely share structure; context-reliance does not, and forcing all
  three through one output matrix made the ctx head the loser of every gradient
  argument.
* **All-sigmoid.** The v1 context head was a 3-way softmax, i.e. it *had* to
  claim a prompt was exactly one of Zero_Shot / LTM / Real_Time_Search. B1's
  whole point is that those aren't mutually exclusive.

(The intent head is 12, not the 13 originally specced: ``Codebase_Query`` was
appended and then dropped before the shipped run — its labelers overlapped on 33
of 640 rows and the head scored exactly that supervision's F1. See
``label_schema.json``'s ``dropped_labels``.)

Both generations stay loadable: ``load_checkpoint`` dispatches on a
``schema_version`` recorded inside the checkpoint, so D5's non-regression gate can
score the old model and a rollback is a file swap. The live path holds v2 since
B1's promotion (2026-07-27).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .schema import LabelSchema, load_schema, load_v1_schema

CHECKPOINT_FORMAT = "ice-classifier"
DEFAULT_TRUNK: Tuple[int, ...] = (512, 256)
DEFAULT_DROPOUT = 0.2


class ICEClassifier(nn.Module):
    """Schema-driven trunk + per-head classifier (schema v2)."""

    def __init__(self, schema: Optional[LabelSchema] = None,
                 input_dim: Optional[int] = None,
                 trunk_dims: Sequence[int] = DEFAULT_TRUNK,
                 dropout: float = DEFAULT_DROPOUT,
                 head_widths: Optional[Dict[str, int]] = None):
        super().__init__()
        self.schema = schema if schema is not None else load_schema()
        self.input_dim = int(input_dim or self.schema.input_dim)
        self.trunk_dims = tuple(trunk_dims)
        self.dropout_p = float(dropout)
        self.template_version = self.schema.template_version

        layers = []
        prev = self.input_dim
        for width in self.trunk_dims:
            layers += [nn.Linear(prev, width), nn.GELU(), nn.Dropout(dropout)]
            prev = width
        self.trunk = nn.Sequential(*layers)

        widths = head_widths or {h.name: h.width for h in self.schema.heads}
        # ModuleDict keys must be valid identifiers; head names already are.
        self.heads = nn.ModuleDict({name: nn.Linear(prev, widths[name])
                                    for name in self.schema.head_names})

    # ── forward ──────────────────────────────────────────────────────────
    def forward_heads(self, x) -> Dict[str, torch.Tensor]:
        """Per-head logits. Training reads this (each head gets its own loss)."""
        z = self.trunk(x)
        return {name: head(z) for name, head in self.heads.items()}

    def forward(self, x) -> torch.Tensor:
        """Logits concatenated in schema head order — (B, schema.total_width).

        Kept flat so every consumer slices with ``schema.slice(head)`` and the
        shape story is identical for v1 and v2 checkpoints.
        """
        out = self.forward_heads(x)
        return torch.cat([out[name] for name in self.schema.head_names], dim=-1)

    # ── persistence ──────────────────────────────────────────────────────
    def checkpoint_dict(self, notes: str = "", **extra) -> dict:
        """The self-describing payload written to disk.

        Carries the label names, not just the widths: a checkpoint stays
        interpretable even if ``label_schema.json`` moves on, and
        ``verify_against`` can catch a schema that changed under a trained model.
        """
        payload = {
            "format": CHECKPOINT_FORMAT,
            "schema_version": self.schema.version,
            "template_version": self.template_version,
            "input_dim": self.input_dim,
            "trunk_dims": list(self.trunk_dims),
            "dropout": self.dropout_p,
            "head_labels": {h.name: list(h.labels) for h in self.schema.heads},
            "state_dict": self.state_dict(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        payload.update(extra)
        return payload

    def verify_against(self, schema: LabelSchema) -> None:
        """Raise if *schema* disagrees with the head layout this model was built
        with — a loud failure beats silently predicting label 7 as label 8."""
        for head in schema.heads:
            # nn.ModuleDict has no .get() — membership then index.
            if head.name not in self.heads:
                raise ValueError(f"checkpoint has no head {head.name!r}")
            mine = self.heads[head.name]
            if mine.out_features != head.width:
                raise ValueError(
                    f"head {head.name!r}: checkpoint width {mine.out_features} "
                    f"!= schema width {head.width} ({schema.path})")

    @classmethod
    def load(cls, path: str, map_location="cpu") -> nn.Module:
        """Load any generation of checkpoint. See ``load_checkpoint``."""
        model, _ = load_checkpoint(path, map_location=map_location)
        return model


class LegacyICEClassifierV1(nn.Module):
    """The pre-B1 network, kept **only** so old checkpoints stay loadable.

    Reason 2 below expired when B1 promoted (2026-07-27); reason 1 did not.
      1. D5's non-regression gate scores the old model on the same held-out rows
         as the new one — you cannot gate against a model you cannot run, and
         every future candidate is still gated against the pre-B1 baseline.
      2. ~~``settings.classifier_model_path`` points at a v1 checkpoint~~ — it
         points at ``ice_classifier_v4_schema2.pt`` now. The v1 model keeps its
         own honest name, ``ice_classifier_v3_qwen_ft3.pt``, and is the rollback
         artifact — a rollback must not require restoring this class.

    So this is NOT deletable yet: it is what makes both the gate and the rollback
    work. ``embedder.slice384``'s **classifier** call sites, however, are now dead
    for the live path (A9) — but keep ``slice384`` itself, the micro-NER uses it.
    """

    def __init__(self, input_dim: int = 384, hidden: int = 128, out_dim: int = 25,
                 dropout: float = 0.3):
        super().__init__()
        self.schema = load_v1_schema()
        self.input_dim = input_dim
        self.template_version = 1
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

    def forward_heads(self, x) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        return {h.name: out[..., h.slice] for h in self.schema.heads}


def _looks_like_v1(state: dict) -> bool:
    return "fc1.weight" in state and "fc2.weight" in state


def load_checkpoint(path: str, map_location="cpu",
                    schema: Optional[LabelSchema] = None) -> Tuple[nn.Module, dict]:
    """Load a checkpoint of either generation.

    Returns ``(model, meta)``. *meta* always carries ``schema_version``,
    ``template_version`` and ``input_dim`` so callers (classifier.py) know how to
    render and encode input for THIS model without guessing.
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)

    # v1: torch.save(model.state_dict()) — a bare tensor dict, no metadata.
    if not isinstance(blob, dict) or _looks_like_v1(blob):
        model = LegacyICEClassifierV1()
        model.load_state_dict(blob)
        model.eval()
        return model, {"schema_version": 1, "template_version": 1,
                       "input_dim": model.input_dim, "path": path,
                       "legacy": True}

    version = int(blob.get("schema_version", 2))
    if version == 1:
        model = LegacyICEClassifierV1()
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model, {**{k: v for k, v in blob.items() if k != "state_dict"},
                       "path": path, "legacy": True}

    sch = schema if schema is not None else load_schema()
    head_widths = {name: len(labels)
                   for name, labels in blob.get("head_labels", {}).items()} or None
    model = ICEClassifier(
        schema=sch,
        input_dim=blob.get("input_dim"),
        trunk_dims=blob.get("trunk_dims", DEFAULT_TRUNK),
        dropout=blob.get("dropout", DEFAULT_DROPOUT),
        head_widths=head_widths,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    model.template_version = int(blob.get("template_version", sch.template_version))
    meta = {k: v for k, v in blob.items() if k != "state_dict"}
    meta.update({"path": path, "legacy": False})
    return model, meta


def head_losses(logits_by_head: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                schema: LabelSchema,
                pos_weights: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
    """Per-head loss, honouring each head's declared activation.

    Sigmoid heads get BCE-with-logits (optionally per-label ``pos_weight``, which
    is what keeps a 600-positive label from being ignored next to a 19,000-positive
    one). A softmax head — only v1's context head — gets cross-entropy, so the
    same helper can fine-tune a legacy checkpoint.

    *targets* is the flat multi-hot matrix (B, schema.total_width).
    """
    out: Dict[str, torch.Tensor] = {}
    for head in schema.heads:
        logits = logits_by_head[head.name]
        target = targets[:, head.slice]
        if head.activation == "softmax":
            out[head.name] = nn.functional.cross_entropy(logits, target.argmax(dim=1))
        else:
            pw = (pos_weights or {}).get(head.name)
            out[head.name] = nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pw)
    return out


DEFAULT_POS_WEIGHT_CAP = 3.0


def compute_pos_weights(targets: torch.Tensor, schema: LabelSchema,
                        cap: float = DEFAULT_POS_WEIGHT_CAP) -> Dict[str, torch.Tensor]:
    """Per-label ``neg/pos`` ratios, capped.

    Uncapped, a label with 12 positives in 25k rows gets weight ~2000 and the
    head trains on nothing else. The cap keeps rare labels *heard* without
    letting them shout — labels too rare even for this get dropped from the
    schema instead (the <150-positive rule, §4).

    **The cap is the calibration knob, and 20 was too high.** Run 1 of the B1
    retrain used it and every one of the 28 labels came out with recall
    0.87–0.97 and precision 0.04–0.81: at ~1% prevalence the ratio saturates the
    cap, so the loss pays 20× more for a miss than for a false alarm and the
    cheapest policy each head can learn is "say yes". The default below is the
    winner of a 3/5/10/20 sweep — see the sweep note recorded with run 2.
    """
    weights: Dict[str, torch.Tensor] = {}
    for head in schema.heads:
        if head.activation == "softmax":
            continue
        block = targets[:, head.slice]
        pos = block.sum(dim=0)
        neg = block.shape[0] - pos
        w = torch.where(pos > 0, neg / pos.clamp(min=1.0), torch.ones_like(pos))
        weights[head.name] = w.clamp(max=cap)
    return weights


__all__ = [
    "ICEClassifier", "LegacyICEClassifierV1", "load_checkpoint",
    "head_losses", "compute_pos_weights", "CHECKPOINT_FORMAT",
]
