"""B1 D1: the label schema, loaded once and read by everyone.

Before B1 the head layout lived as magic numbers scattered through the tree —
``outputs[:, 11:22]``, ``probs[22:]``, ``torch.zeros(n, 25)`` — so adding a
single label meant hunting slices across ``classifier.py``, ``orchestrator.py``
and ``fine_tune.py`` and hoping none were missed. This module makes
``data/labeled/label_schema.json`` the one place head widths and offsets are
declared; every consumer asks the schema.

Deliberately dependency-light (stdlib only, no torch / no settings import at
module scope) so pipeline scripts, the trainer and the API all share it without
dragging the runtime in.

Two schema generations coexist on purpose:

* **v2** (``label_schema.json``) — three all-sigmoid heads, 11 topic / 13 intent
  / 4 context-reliance = 28 logits.
* **v1** (``label_schema_v1.json``, frozen) — the pre-B1 11/11/3 layout with a
  softmax context head. Kept loadable because D5's non-regression gate scores
  the OLD checkpoint on the same rows as the new one, and because the live
  checkpoint is still v1 until promotion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

# Head names — the keys every consumer uses. Constants, not string literals
# sprinkled at call sites.
TOPIC = "topic"
INTENT = "intent"
CONTEXT_RELIANCE = "context_reliance"

# v2 context-reliance label names. Named because the derivation layer (D6) and
# memory_decision (D7) reason about these specific four.
NEEDS_MEMORY = "Needs_Memory"
TEMPORAL_RECALL = "Temporal_Recall"
NEEDS_LIVE_INFO = "Needs_Live_Info"
HIGH_COMPLEXITY = "High_Complexity"

# The derived v1 three-way, still the public `context_reliance` string.
ZERO_SHOT = "Zero_Shot"
LONG_TERM_MEMORY = "Long_Term_Memory"
REAL_TIME_SEARCH = "Real_Time_Search"

_DEFAULT_V2 = "data/labeled/label_schema.json"
_DEFAULT_V1 = "data/labeled/label_schema_v1.json"


@dataclass(frozen=True)
class Head:
    """One output head: its labels and where they sit in the flat logit vector."""

    name: str
    labels: Tuple[str, ...]
    definitions: Tuple[str, ...]
    activation: str          # "sigmoid" | "softmax"
    decision: str            # "multi_label" | "independent" | "single_label"
    offset: int              # start index in the concatenated output
    width: int

    @property
    def slice(self) -> slice:
        """Where this head lives in a flat 28-wide (v2) / 25-wide (v1) vector."""
        return slice(self.offset, self.offset + self.width)

    def index(self, label: str) -> int:
        """Absolute index of *label* in the flat vector (not head-relative)."""
        return self.offset + self.labels.index(label)

    def local_index(self, label: str) -> int:
        return self.labels.index(label)

    def has(self, label: str) -> bool:
        return label in self.labels

    def definition(self, label: str) -> str:
        return self.definitions[self.labels.index(label)]


@dataclass(frozen=True)
class LabelSchema:
    """The full head layout for one schema generation."""

    version: int
    template_version: int
    input_dim: int
    heads: Tuple[Head, ...]
    path: str
    derived_ctx_labels: Tuple[str, ...]

    # ── lookup ────────────────────────────────────────────────────────────
    def head(self, name: str) -> Head:
        for h in self.heads:
            if h.name == name:
                return h
        raise KeyError(f"{self.path}: no head named {name!r} "
                       f"(have {[h.name for h in self.heads]})")

    def slice(self, name: str) -> slice:
        return self.head(name).slice

    def labels(self, name: str) -> Tuple[str, ...]:
        return self.head(name).labels

    @property
    def head_names(self) -> Tuple[str, ...]:
        return tuple(h.name for h in self.heads)

    @property
    def total_width(self) -> int:
        return sum(h.width for h in self.heads)

    @property
    def head_widths(self) -> Tuple[int, ...]:
        return tuple(h.width for h in self.heads)

    def split(self, vector):
        """Split a flat sequence/tensor into {head_name: sub-sequence}."""
        return {h.name: vector[h.slice] for h in self.heads}

    def empty_probs(self) -> list:
        """The all-zero convention for rows that never ran the model (DI3
        fast-path). Width follows the schema — see orchestrator._head_confidences,
        which treats all-zero as 'no ML probabilities here'."""
        return [0.0] * self.total_width


def _parse_v2(raw: dict, path: str) -> LabelSchema:
    heads = []
    offset = 0
    for h in raw["heads"]:
        labels = tuple(lbl["name"] for lbl in h["labels"])
        defs = tuple(lbl.get("definition", "") for lbl in h["labels"])
        heads.append(Head(
            name=h["name"],
            labels=labels,
            definitions=defs,
            activation=h.get("activation", "sigmoid"),
            decision=h.get("decision", "multi_label"),
            offset=offset,
            width=len(labels),
        ))
        offset += len(labels)
    derived = tuple(raw.get("derived", {})
                       .get("context_reliance_v1", {})
                       .get("labels", (ZERO_SHOT, LONG_TERM_MEMORY, REAL_TIME_SEARCH)))
    return LabelSchema(
        version=int(raw["schema_version"]),
        template_version=int(raw.get("template_version", raw["schema_version"])),
        input_dim=int(raw.get("input_dim", 1024)),
        heads=tuple(heads),
        path=path,
        derived_ctx_labels=derived,
    )


def _parse_v1(raw: dict, path: str) -> LabelSchema:
    """The frozen pre-B1 layout: flat label lists, softmax context head."""
    spec = (
        (TOPIC, raw["topic_labels"], "sigmoid", "multi_label"),
        (INTENT, raw["intent_labels"], "sigmoid", "multi_label"),
        (CONTEXT_RELIANCE, raw["context_reliance_labels"], "softmax", "single_label"),
    )
    heads = []
    offset = 0
    for name, labels, activation, decision in spec:
        labels = tuple(labels)
        heads.append(Head(name=name, labels=labels, definitions=("",) * len(labels),
                          activation=activation, decision=decision,
                          offset=offset, width=len(labels)))
        offset += len(labels)
    return LabelSchema(
        version=1,
        template_version=1,
        input_dim=384,          # the slice384 MRL prefix the v1 heads were trained on
        heads=tuple(heads),
        path=path,
        derived_ctx_labels=tuple(raw["context_reliance_labels"]),
    )


@lru_cache(maxsize=8)
def _load(path_str: str) -> LabelSchema:
    raw = json.loads(Path(path_str).read_text())
    if "schema_version" in raw:
        return _parse_v2(raw, path_str)
    if "topic_labels" in raw:
        return _parse_v1(raw, path_str)
    raise ValueError(f"{path_str}: unrecognised label schema "
                     f"(no schema_version, no topic_labels)")


def load_schema(path: Optional[str] = None) -> LabelSchema:
    """Load (and cache) a label schema. Defaults to settings.label_schema_path.

    The settings import is deferred so scripts that only want the taxonomy
    don't pay for the runtime config.
    """
    if path is None:
        try:
            from src.api.config import settings
            path = settings.label_schema_path
        except Exception:
            path = _DEFAULT_V2
    return _load(str(path))


def load_v1_schema(path: Optional[str] = None) -> LabelSchema:
    """The frozen v1 schema — D5's gate and the legacy checkpoint path."""
    return _load(str(path or _DEFAULT_V1))


@lru_cache(maxsize=8)
def _by_width() -> Dict[int, LabelSchema]:
    out: Dict[int, LabelSchema] = {}
    for loader in (load_schema, load_v1_schema):
        try:
            s = loader()
        except Exception:
            continue
        out.setdefault(s.total_width, s)
    return out


def derive_context_reliance(ctx_probs: Dict[str, float]) -> Tuple[str, float, float, float]:
    """D6: the v1 three-way string, derived from v2's independent sigmoids.

    A derivation layer, NOT a parallel path — every pre-B1 consumer keeps reading
    ``context_reliance`` / ``p_ltm`` / ``ctx_confidence`` and never learns the head
    changed shape underneath it.

    Returns ``(context_reliance, p_ltm, p_rts, ctx_confidence)``.
    """
    p_mem = float(ctx_probs.get(NEEDS_MEMORY, 0.0))
    p_live = float(ctx_probs.get(NEEDS_LIVE_INFO, 0.0))
    zero = 1.0 - max(p_mem, p_live)
    three = {ZERO_SHOT: zero, LONG_TERM_MEMORY: p_mem, REAL_TIME_SEARCH: p_live}
    ordered = sorted(three.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[0][0], p_mem, p_live, float(ordered[0][1] - ordered[1][1])


# DI3 fast-path priors: DI3 picks a three-way label without producing
# probabilities, so B2's combination still needs a scalar to work with.
_DI3_PRIORS = {
    LONG_TERM_MEMORY: (0.85, 0.05),
    ZERO_SHOT: (0.12, 0.05),
    REAL_TIME_SEARCH: (0.15, 0.70),
}


def finalize_context_scalars(result, schema: LabelSchema) -> None:
    """Populate the context-reliance scalars every downstream consumer reads.

    Pure logic over ``result.raw_probs`` + *schema* — no model, no torch — so the
    smoke suite can exercise the B2 seam and both derivation generations without
    loading a checkpoint.

    B2's seam (``p_ltm`` / ``p_rts`` / ``ctx_confidence``) predates the v2 head and
    is unchanged on purpose: ``memory_decision`` combines scalars, so the
    softmax→sigmoid swap changes only how they are computed. v2 additionally sets
    ``p_temporal`` / ``p_complex`` and makes ``context_reliance`` a *derived*
    string rather than an argmax over real labels.
    """
    probs = result.raw_probs or []
    ctx = probs[schema.slice(CONTEXT_RELIANCE)] if len(probs) >= schema.total_width else []

    if not ctx or not any(v > 0.0 for v in ctx):
        # DI3 fast-path: raw_probs all zero → prior from the label DI3 chose.
        result.p_ltm, result.p_rts = _DI3_PRIORS.get(result.context_reliance, (0.30, 0.05))
        result.ctx_confidence = float(result.max_confidence)
        return

    by_label = dict(zip(schema.labels(CONTEXT_RELIANCE), (float(v) for v in ctx)))

    if schema.version == 1:
        # Legacy softmax head: the three probabilities ARE the three-way.
        result.context_reliance = max(by_label, key=by_label.get)
        result.p_ltm = by_label.get(LONG_TERM_MEMORY, 0.0)
        result.p_rts = by_label.get(REAL_TIME_SEARCH, 0.0)
        ordered = sorted(by_label.values(), reverse=True)
        result.ctx_confidence = float(ordered[0] - ordered[1])
        return

    label, p_ltm, p_rts, confidence = derive_context_reliance(by_label)
    result.context_reliance = label
    result.p_ltm = p_ltm
    result.p_rts = p_rts
    result.ctx_confidence = confidence
    result.p_temporal = by_label.get(TEMPORAL_RECALL, 0.0)
    result.p_complex = by_label.get(HIGH_COMPLEXITY, 0.0)


def resolve_by_width(width: int) -> Optional[LabelSchema]:
    """Find the schema whose flat vector is *width* wide.

    For consumers that receive a bare `raw_probs` list with no model attached
    (orchestrator._head_confidences on a hand-built ClassificationResult). Kept
    explicit rather than inlining `probs[:11]` — that inlining is exactly what
    B1 removes.
    """
    return _by_width().get(width)
