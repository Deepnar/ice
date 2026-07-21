import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import the *correct* dataset and model (not redefining them!)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
from classifier.dataset import ICEClassifierDataset
from classifier.model import ICEClassifier

# ──────────────────────────────── CLI arguments ────────────────────────────────
parser = argparse.ArgumentParser(description="Train the ICE classifier.")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--model_path", type=str, default="models/classifier/ice_classifier_v2.pt")
parser.add_argument("--log_path", type=str, default="models/classifier/training_runs.jsonl")
parser.add_argument("--pos_weight_cap", type=float, default=15.0)
args = parser.parse_args()

# ──────────────────────────────── Reproducibility ────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ──────────────────────────────── Device ────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────── Load data ─────────────────────────────────────
full_dataset = ICEClassifierDataset("data/labeled/training_data.jsonl")
total = len(full_dataset)
val_size = int(total * args.val_split)
train_size = total - val_size
train_set, val_set = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(args.seed)
)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

# ─────────────────── Compute pos_weight from the training set ───────────────────
# Gather all training labels to count positives
train_labels = full_dataset.labels[train_set.indices]   # shape (train_size, 25)
topic_gt = train_labels[:, :11]      # 0–10
intent_gt = train_labels[:, 11:22]   # 11–21
# context is softmax, no weighting needed there

def compute_pos_weight(label_matrix, cap):
    """Return a tensor of weights for each class (shape C)."""
    num_pos = label_matrix.sum(dim=0)          # count of 1s per class
    num_neg = label_matrix.size(0) - num_pos   # count of 0s per class
    # avoid division by zero
    pos_weight = torch.where(num_pos > 0, num_neg / num_pos, torch.ones_like(num_pos))
    # cap
    pos_weight = torch.clamp(pos_weight, max=cap)
    return pos_weight

# Instead of moving to device here, keep on CPU
topic_pos_weight = compute_pos_weight(topic_gt, args.pos_weight_cap).to(device)
intent_pos_weight = compute_pos_weight(intent_gt, args.pos_weight_cap).to(device)



print("Topic pos_weight:", topic_pos_weight)
print("Intent pos_weight:", intent_pos_weight)

# ──────────────────────────────── Model & losses ────────────────────────────────
model = ICEClassifier().to(device)
# Then define losses without moving:
bce_topic = nn.BCEWithLogitsLoss(pos_weight=topic_pos_weight)
bce_intent = nn.BCEWithLogitsLoss(pos_weight=intent_pos_weight)
ce_context = nn.CrossEntropyLoss()   # no weights needed for softmax head

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ──────────────────────────────── Training loop ─────────────────────────────────
best_val_loss = float("inf")
patience_counter = 0
PATIENCE = 5

for epoch in range(1, args.epochs + 1):
    # ---- Train ----
    model.train()
    train_loss = 0.0
    for emb, labels in train_loader:
        emb, labels = emb.to(device), labels.to(device)

        outputs = model(emb)
        topic_out = outputs[:, :11]
        intent_out = outputs[:, 11:22]
        ctx_out = outputs[:, 22:]

        topic_gt = labels[:, :11]
        intent_gt = labels[:, 11:22]
        ctx_gt = labels[:, 22:].argmax(dim=1)   # index 0/1/2

        loss = (bce_topic(topic_out, topic_gt) +
                bce_intent(intent_out, intent_gt) +
                ce_context(ctx_out, ctx_gt))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for emb, labels in val_loader:
            emb, labels = emb.to(device), labels.to(device)
            outputs = model(emb)
            topic_out = outputs[:, :11]
            intent_out = outputs[:, 11:22]
            ctx_out = outputs[:, 22:]

            topic_gt = labels[:, :11]
            intent_gt = labels[:, 11:22]
            ctx_gt = labels[:, 22:].argmax(dim=1)

            loss = (bce_topic(topic_out, topic_gt) +
                    bce_intent(intent_out, intent_gt) +
                    ce_context(ctx_out, ctx_gt))
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ---- Early stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ──────────────────────────────── Log the run ───────────────────────────────────
log_entry = {
    "seed": args.seed,
    "epochs_completed": epoch,
    "final_val_loss": best_val_loss,
    "timestamp": datetime.now().isoformat(),
    "model_path": args.model_path,
    "args": vars(args)
}
os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
with open(args.log_path, "a") as f:
    f.write(json.dumps(log_entry) + "\n")

print(f"Training complete. Best model saved to {args.model_path}")