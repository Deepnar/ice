#!/usr/bin/env python3
"""Train the micro‑NER MLP with caching, class weights, and per‑class metrics."""
import json
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Ensure the project root is on sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

DATA_PATH = "data/ner/training_data.jsonl"
MODEL_OUT = "models/ner/ner_model.pt"
CACHE_PATH = "data/ner/embeddings_cache.pt"
EMBEDDER_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
VAL_SPLIT = 0.1
PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NERDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.samples = list(zip(embeddings, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    max_len = max(len(item[0]) for item in batch)
    emb_dim = batch[0][0].shape[-1]
    padded_emb = torch.zeros(len(batch), max_len, emb_dim)
    padded_labels = torch.full((len(batch), max_len), 2, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (emb, labels) in enumerate(batch):
        L = len(emb)
        padded_emb[i, :L] = emb
        padded_labels[i, :L] = labels
        mask[i, :L] = True
    return padded_emb, padded_labels, mask

def compute_class_weights(labels_list):
    """Compute inverse frequency weights for [B-ENT, I-ENT, O]."""
    all_labels = torch.cat([l for l in labels_list])
    counts = torch.bincount(all_labels, minlength=3).float()
    total = counts.sum()
    # Weight = total / (n_classes * count), cap at 100
    weights = total / (3 * counts)
    weights = torch.clamp(weights, max=100.0)
    return weights

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    loss_fn = nn.CrossEntropyLoss()  # no class weights during eval (optional)
    with torch.no_grad():
        for emb, labels, mask in tqdm(loader, desc="Evaluating", leave=False):
            emb = emb.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            logits = model(emb)
            active_logits = logits[mask]
            active_labels = labels[mask]
            loss = loss_fn(active_logits, active_labels)
            total_loss += loss.item()
            preds = torch.argmax(active_logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    per_class_acc = {}
    for cls_idx, name in [(0, "B-ENT"), (1, "I-ENT"), (2, "O")]:
        mask_cls = all_labels == cls_idx
        if mask_cls.sum() > 0:
            per_class_acc[name] = (all_preds[mask_cls] == cls_idx).mean()
        else:
            per_class_acc[name] = 0.0
    return total_loss / len(loader), accuracy, per_class_acc

def build_cache(embedder, limit=None):
    all_tokens = []
    all_labels = []
    all_lengths = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if limit:
            lines = lines[:limit]
        for line in tqdm(lines, desc="Parsing data"):
            obj = json.loads(line.strip())
            tokens = obj["tokens"]
            labels = obj["labels"]
            if not tokens:
                continue
            all_tokens.extend(tokens)
            all_labels.extend(labels)
            all_lengths.append(len(tokens))

    print(f"Encoding {len(all_tokens)} tokens...")
    all_embeddings = embedder.encode(all_tokens, convert_to_tensor=True, show_progress_bar=True)

    embeddings_list = []
    labels_list = []
    start = 0
    for length in tqdm(all_lengths, desc="Splitting"):
        emb = all_embeddings[start:start+length]
        labs = all_labels[start:start+length]
        label_ids = torch.tensor([0 if l == "B-ENT" else 1 if l == "I-ENT" else 2 for l in labs])
        embeddings_list.append(emb)
        labels_list.append(label_ids)
        start += length

    cache = {"embeddings": embeddings_list, "labels": labels_list}
    torch.save(cache, CACHE_PATH)
    print(f"Cache saved to {CACHE_PATH}")
    return embeddings_list, labels_list

def load_cache(embedder, limit=None):
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached embeddings from {CACHE_PATH}...")
        cache = torch.load(CACHE_PATH, map_location="cpu")
        embeddings = cache["embeddings"]
        labels = cache["labels"]
        if limit:
            embeddings = embeddings[:limit]
            labels = labels[:limit]
        return embeddings, labels
    else:
        print("Cache not found. Building...")
        return build_cache(embedder, limit)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    try:
        embedder = SentenceTransformer(EMBEDDER_NAME, device=DEVICE, truncate_dim=384)
    except TypeError:
        embedder = SentenceTransformer(EMBEDDER_NAME, device=DEVICE)

    if args.rebuild and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print("Cache removed. Will rebuild.")

    embeddings_list, labels_list = load_cache(embedder, args.limit)
    print(f"Loaded {len(embeddings_list)} samples from cache")

    if len(embeddings_list) == 0:
        print("No samples found.")
        return

    class_weights = compute_class_weights(labels_list)
    print(f"Class weights (B-ENT, I-ENT, O): {class_weights.tolist()}")

    dataset = NERDataset(embeddings_list, labels_list)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Training: {train_size}, Validation: {val_size}")

    from src.classifier.ner_model import MicroNER
    model = MicroNER().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for emb, labels, mask in tqdm(train_loader, desc=f"Epoch {epoch} (train)"):
            emb = emb.to(DEVICE)
            labels = labels.to(DEVICE)
            mask = mask.to(DEVICE)
            logits = model(emb)
            active_logits = logits[mask]
            active_labels = labels[mask]
            loss = loss_fn(active_logits, active_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        val_loss, val_acc, per_class = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        print(f"  Per-class acc: B-ENT={per_class['B-ENT']:.3f}, I-ENT={per_class['I-ENT']:.3f}, O={per_class['O']:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  Saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()