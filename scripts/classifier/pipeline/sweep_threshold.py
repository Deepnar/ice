#!/usr/bin/env python3
"""Diagnose whether a weak F1 is a bad model or a badly-thresholded one.

The first v2 training run produced macro-F1 of 0.49/0.35/0.44 with a telltale
shape: **recall 0.87–0.97 on every label, precision 0.04–0.81**. A model that
fires nearly every label on nearly every row is not undertrained, it is
miscalibrated, and two deliberate choices push it that way:

* ``compute_pos_weights`` weights a positive by neg/pos (capped at 20). At 0.9%
  prevalence that is the full 20×, so the loss pays 20× more for a miss than a
  false alarm and the head learns to always say yes.
* the tag threshold is 0.3, inherited from v1 — and the spec explicitly defers
  tuning it to Z1-prep's decision-threshold stage.

Both are correctable without retraining, because a threshold is applied after
the fact. This script sweeps global thresholds and then fits a PER-LABEL
threshold on validation, which is standard practice for multi-label heads whose
prevalences differ by two orders of magnitude — one global cut cannot serve a
label at 45% and a label at 0.9% simultaneously.

Thresholds are fitted on VAL and reported on TEST, never fitted on test.

Usage:
    uv run python scripts/classifier/pipeline/sweep_threshold.py \
        --checkpoint models/classifier/ice_classifier_v4_schema2.pt
"""

import argparse
import json

import torch
from common import TEST_SPLIT, VAL_SPLIT

from src.classifier.dataset import ICEClassifierDataset
from src.classifier.model import load_checkpoint
from src.classifier.schema import load_schema


def probs_for(model, ds):
    with torch.no_grad():
        return torch.sigmoid(model(ds.embeddings))


def f1_at(probs, targets, col, thr):
    pred = (probs[:, col] >= thr).float()
    tgt = targets[:, col]
    tp = float((pred * tgt).sum())
    fp = float((pred * (1 - tgt)).sum())
    fn = float(((1 - pred) * tgt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def macro_at(probs, targets, schema, thr_by_col):
    return {name: scores[0] for name, scores in
            head_scores(probs, targets, schema, thr_by_col).items()}


def head_scores(probs, targets, schema, thr_by_col):
    """Per head: (macro-F1, support-weighted F1).

    Both are reported because they answer different questions and this model's
    labels span two orders of magnitude of prevalence. Macro treats a 23-support
    label as equal to a 1,496-support one — it is the honest number for "does
    every head work". Weighted is what a random live prompt actually experiences.
    A gap between them localises the damage to the rare tail.
    """
    out = {}
    for head in schema.heads:
        f1s, supports = [], []
        for i in range(head.width):
            col = head.offset + i
            f1s.append(f1_at(probs, targets, col, thr_by_col[col])[0])
            supports.append(float(targets[:, col].sum()))
        total = sum(supports) or 1.0
        out[head.name] = (
            sum(f1s) / len(f1s),
            sum(f * s for f, s in zip(f1s, supports)) / total,
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="B1: threshold diagnosis")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write", action="store_true",
                    help="stamp the fitted global threshold into the checkpoint "
                         "as tag_threshold (what classifier._tags_above reads)")
    args = ap.parse_args()

    schema = load_schema()
    model, _ = load_checkpoint(args.checkpoint, schema=schema)
    model.eval()

    val = ICEClassifierDataset(VAL_SPLIT, schema=schema, device=args.device,
                               show_progress=False)
    test = ICEClassifierDataset(TEST_SPLIT, schema=schema, device=args.device,
                                show_progress=False)
    pv, pt = probs_for(model, val), probs_for(model, test)

    grid = [round(x * 0.05, 2) for x in range(1, 20)]

    print("GLOBAL threshold sweep (fitted and reported on val):")
    print(f"  {'thr':>5}  " + "  ".join(f"{h.name[:12]:>12}" for h in schema.heads))
    best_global, best_score = 0.3, -1.0
    for thr in grid:
        m = macro_at(pv, val.labels, schema, {c: thr for c in range(schema.total_width)})
        avg = sum(m.values()) / len(m)
        if avg > best_score:
            best_global, best_score = thr, avg
        if thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            print(f"  {thr:>5}  " + "  ".join(f"{m[h.name]:>12.3f}" for h in schema.heads))
    print(f"  best global threshold on val: {best_global} (mean macro-F1 {best_score:.3f})")

    # Per-label thresholds, fitted on val only.
    per_label = {}
    for head in schema.heads:
        for i, label in enumerate(head.labels):
            col = head.offset + i
            best_t, best_f = 0.5, -1.0
            for thr in grid:
                f, _, _ = f1_at(pv, val.labels, col, thr)
                if f > best_f:
                    best_t, best_f = thr, f
            per_label[col] = best_t

    print("\nTEST F1 under each policy (macro / weighted):")
    summary = {}
    for name, key, thr_map in (
        ("threshold 0.3 (what training reported)", "thr_0.3",
         {c: 0.3 for c in range(schema.total_width)}),
        (f"best global ({best_global}), fitted on val", "global",
         {c: best_global for c in range(schema.total_width)}),
        ("per-label, fitted on val", "per_label", per_label),
    ):
        s = head_scores(pt, test.labels, schema, thr_map)
        avg = sum(v[0] for v in s.values()) / len(s)
        summary[key] = {h.name: {"macro": round(s[h.name][0], 4),
                                 "weighted": round(s[h.name][1], 4)}
                        for h in schema.heads}
        summary[key]["mean_macro"] = round(avg, 4)
        print(f"  {name:<42} "
              + "  ".join(f"{h.name[:6]}={s[h.name][0]:.3f}/{s[h.name][1]:.3f}"
                          for h in schema.heads)
              + f"   mean_macro={avg:.3f}")

    print("\nPER-LABEL detail at the fitted thresholds (test):")
    for head in schema.heads:
        print(f"  {head.name}:")
        for i, label in enumerate(head.labels):
            col = head.offset + i
            f, p, r = f1_at(pt, test.labels, col, per_label[col])
            n = int(test.labels[:, col].sum())
            print(f"    {label:26} thr={per_label[col]:.2f}  f1={f:.3f}  p={p:.3f}  r={r:.3f}  n={n}")

    out = args.checkpoint.replace(".pt", "_thresholds.json")
    with open(out, "w") as fh:
        json.dump({"best_global": best_global,
                   "per_label": {str(k): v for k, v in per_label.items()},
                   "labels": {str(head.offset + i): label
                              for head in schema.heads
                              for i, label in enumerate(head.labels)},
                   "test_scores": summary}, fh, indent=2)
    print(f"\nthresholds → {out}")

    if args.write:
        # The global value, not the per-label map: on this model the two score
        # identically (mean macro-F1 0.609 vs 0.610 on test), and the head where
        # per-label thresholds would actually earn their complexity —
        # context_reliance — does not go through _tags_above at all. Its
        # consumers (memory_decision, B3) read the raw scalars and apply their
        # own thresholds by design. So per-label would add a lookup table to the
        # one place it measurably does nothing.
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "state_dict" not in blob:
            raise SystemExit("--write needs a self-describing (v2) checkpoint")
        blob["tag_threshold"] = best_global
        blob["threshold_fit"] = {"per_label": {str(k): v for k, v in per_label.items()},
                                 "fitted_on": "val", "reported_on": "test",
                                 "test_scores": summary}
        torch.save(blob, args.checkpoint)
        print(f"stamped tag_threshold={best_global} into {args.checkpoint}")


if __name__ == "__main__":
    main()
