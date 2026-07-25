#!/usr/bin/env python3
"""Stage 6 — train the schema-v2 classifier.

Rewrites ``legacy/training/train_classifier.py``: schema-driven head widths,
native-1024 input, per-head BCE with per-label positive weights, early stopping
on held-out loss.

Nothing here is clever, deliberately — the spec calls training "standard-practice
mechanical" (AdamW, lr 1e-3, early stop) and defers trunk width and the sigmoid
tag threshold to Z1-prep's sweep. The parts that decide whether this model is any
good happened upstream, in the labels.

Usage:
    uv run python scripts/classifier/pipeline/train.py
    uv run python scripts/classifier/pipeline/train.py --epochs 60 --device cuda
"""

import argparse
import json
import os
from datetime import datetime, timezone

import torch
from common import TRAIN_SPLIT, VAL_SPLIT, ensure_data_dir

from src.classifier.dataset import ICEClassifierDataset
from src.classifier.model import ICEClassifier, compute_pos_weights, head_losses
from src.classifier.schema import load_schema

SEED = 42
DEFAULT_OUT = "models/classifier/ice_classifier_v4_schema2.pt"


def per_label_f1(probs, targets, schema, threshold: float):
    """F1 per label plus per-head macro-F1, computed straight from the tensors.

    Written out rather than pulled from sklearn so the number is inspectable: a
    label with zero predicted positives scores 0.0 here, which is exactly what a
    dead head should look like in the report.
    """
    preds = (probs >= threshold).float()
    report, macro = {}, {}
    for head in schema.heads:
        f1s = []
        for i, label in enumerate(head.labels):
            col = head.offset + i
            tp = float((preds[:, col] * targets[:, col]).sum())
            fp = float((preds[:, col] * (1 - targets[:, col])).sum())
            fn = float(((1 - preds[:, col]) * targets[:, col]).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            report[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                             "f1": round(f1, 4), "support": int(targets[:, col].sum())}
            f1s.append(f1)
        macro[head.name] = round(sum(f1s) / max(1, len(f1s)), 4)
    return report, macro


def evaluate(model, embeddings, targets, schema, threshold):
    model.eval()
    with torch.no_grad():
        logits = model(embeddings)
        probs = torch.sigmoid(logits)
    return per_label_f1(probs, targets, schema, threshold)


def main():
    ap = argparse.ArgumentParser(description="B1 stage 6: train the v2 classifier")
    ap.add_argument("--train", default=TRAIN_SPLIT)
    ap.add_argument("--val", default=VAL_SPLIT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trunk", type=int, nargs="*", default=None,
                    help="trunk widths (default 512 256; Z1-prep sweeps this)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    schema = load_schema()
    ensure_data_dir()

    print(f"[train] encoding splits (device={args.device})")
    train_ds = ICEClassifierDataset(args.train, schema=schema, device=args.device)
    val_ds = ICEClassifierDataset(args.val, schema=schema, device=args.device)
    print(f"[train] train {len(train_ds)} rows, val {len(val_ds)} rows, "
          f"input dim {train_ds.embeddings.shape[1]}")

    x_train, y_train = train_ds.embeddings, train_ds.labels
    x_val, y_val = val_ds.embeddings, val_ds.labels

    pos_weights = compute_pos_weights(y_train, schema)
    model = ICEClassifier(schema=schema, input_dim=x_train.shape[1],
                          trunk_dims=args.trunk or (512, 256))
    print(f"[train] {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss, best_state, since_best = float("inf"), None, 0
    n = x_train.shape[0]

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            optimizer.zero_grad()
            losses = head_losses(model.forward_heads(x_train[idx]), y_train[idx],
                                 schema, pos_weights)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)

        model.eval()
        with torch.no_grad():
            val_losses = head_losses(model.forward_heads(x_val), y_val, schema,
                                     pos_weights)
            val_loss = float(sum(val_losses.values()))

        if val_loss < best_loss - 1e-4:
            best_loss, since_best = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1

        if epoch % 5 == 0 or since_best == 0:
            heads = " ".join(f"{k}={float(v):.3f}" for k, v in val_losses.items())
            print(f"  epoch {epoch:3d}  train {total / n:.4f}  val {val_loss:.4f}  "
                  f"[{heads}]{'  *' if since_best == 0 else ''}")
        if since_best >= args.patience:
            print(f"[train] early stop at epoch {epoch} (best val {best_loss:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    from src.api.config import settings
    report, macro = evaluate(model, x_val, y_val, schema, settings.classifier_threshold)
    print(f"\n[train] held-out macro-F1 by head: {macro}")
    print(f"[train] per-label F1 (threshold {settings.classifier_threshold}):")
    for head in schema.heads:
        print(f"  {head.name}:")
        for label in head.labels:
            row = report[label]
            print(f"    {label:26} f1={row['f1']:.3f}  p={row['precision']:.3f}  "
                  f"r={row['recall']:.3f}  n={row['support']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.checkpoint_dict(
        notes="B1 schema-v2 retrain",
        train_rows=len(train_ds), val_rows=len(val_ds),
        val_macro_f1=macro, best_val_loss=best_loss,
        trained_at=datetime.now(timezone.utc).isoformat()), args.out)
    print(f"\n[train] checkpoint → {args.out}")

    with open(args.out.replace(".pt", "_val_report.json"), "w") as fh:
        json.dump({"macro_f1": macro, "per_label": report,
                   "best_val_loss": best_loss}, fh, indent=2)


if __name__ == "__main__":
    main()
