import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from itertools import cycle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from classifier.dataset import ICEClassifierDataset
from classifier.model import ICEClassifier

# ────────────── CLI arguments ──────────────
parser = argparse.ArgumentParser(description="Fine‑tune ICE classifier on curated fixes.")
parser.add_argument("--curated-path", type=str, default="data/curated_fixes.jsonl")
parser.add_argument("--checkpoint", type=str, default="models/classifier/ice_classifier_v2.pt")
parser.add_argument("--output", type=str, default="models/classifier/ice_classifier_v2_ft.pt")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--curated-repeat", type=int, default=50,
                    help="How many times to repeat each curated example in a batch")
parser.add_argument("--curated-weight", type=float, default=10.0,
                    help="Multiplier for curated loss term")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────── Load full training data ──────────────
full_dataset = ICEClassifierDataset("data/labeled/training_data.jsonl")
# We'll use all data for fine‑tuning (no validation split needed)
train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

# ────────────── Load curated fixes ──────────────
curated_embeddings = []
curated_labels = []

# We need a sentence transformer for encoding the curated prompts (on CPU)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

with open(args.curated_path, "r") as f:
    for line in f:
        item = json.loads(line)
        prompt = item["prompt"]
        label_dict = item["label"]

        # Convert labels to 25-dim vector using the same label order as training
        TOPIC_LABELS = [
            "Software_&_Tech", "STEM_&_Academics", "Business_&_Finance",
            "Creative_&_Media", "Admin_&_Productivity", "Lifestyle_&_Health",
            "Social_&_Relationships", "World_&_Current_Events", "Meta_AI",
            "Null_Noise", "General_Reference_&_Trivia"
        ]

        INTENT_LABELS = [
            "Factual_Retrieval", "Troubleshooting", "Generation", "Ideation",
            "Analysis_&_Summarization", "Strategic_Planning", "Decision_Making",
            "Emotional_Processing", "Utility_Formatting", "Casual_Banter",
            "Open_Exploration"
        ]
        CONTEXT_RELIANCE_LABELS = ["Zero_Shot", "Long_Term_Memory", "Real_Time_Search"]

        # Build vector
        topic_vec = [1.0 if lbl in label_dict["topic_labels"] else 0.0 for lbl in TOPIC_LABELS]
        intent_vec = [1.0 if lbl in label_dict["intent_labels"] else 0.0 for lbl in INTENT_LABELS]
        ctx_vec = [1.0 if lbl == label_dict["context_reliance"] else 0.0 for lbl in CONTEXT_RELIANCE_LABELS]
        label_vector = topic_vec + intent_vec + ctx_vec

        # Encode prompt
        embedding = embedder.encode(prompt, convert_to_tensor=True)  # shape (384,)
        curated_embeddings.append(embedding)
        curated_labels.append(torch.tensor(label_vector, dtype=torch.float32))

curated_embeddings = torch.stack(curated_embeddings)
curated_labels = torch.stack(curated_labels)

# Repeat the curated examples C times
curated_embeddings = curated_embeddings.repeat(args.curated_repeat, 1)
curated_labels = curated_labels.repeat(args.curated_repeat, 1)

curated_dataset = TensorDataset(curated_embeddings, curated_labels)
curated_loader = DataLoader(curated_dataset, batch_size=32, shuffle=True)

# ────────────── Load checkpoint and freeze layers ──────────────
model = ICEClassifier().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))

# Freeze everything except fc2
for param in model.parameters():
    param.requires_grad = False
for param in model.fc2.parameters():
    param.requires_grad = True

print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}")

# ────────────── Loss and optimizer ──────────────
# No pos_weight here – curated examples already give enough signal
bce_topic = nn.BCEWithLogitsLoss()
bce_intent = nn.BCEWithLogitsLoss()
ce_context = nn.CrossEntropyLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

# ────────────── Fine‑tuning loop ──────────────
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    batches = 0

    # Interleave curated batches with full data (simple approach: process both loaders each epoch)
    curated_cycle = cycle(curated_loader)
    for emb_main, labels_main in train_loader:
        emb_cur, labels_cur = next(curated_cycle)        # Main batch
        emb_main, labels_main = emb_main.to(device), labels_main.to(device)
        outputs = model(emb_main)
        loss_main = (bce_topic(outputs[:, :11], labels_main[:, :11]) +
                     bce_intent(outputs[:, 11:22], labels_main[:, 11:22]) +
                     ce_context(outputs[:, 22:], labels_main[:, 22:].argmax(dim=1)))

        # Curated batch
        emb_cur, labels_cur = emb_cur.to(device), labels_cur.to(device)
        outputs_cur = model(emb_cur)
        loss_cur = (bce_topic(outputs_cur[:, :11], labels_cur[:, :11]) +
                    bce_intent(outputs_cur[:, 11:22], labels_cur[:, 11:22]) +
                    ce_context(outputs_cur[:, 22:], labels_cur[:, 22:].argmax(dim=1)))

        curated_weight = args.curated_weight
        loss = loss_main + curated_weight * loss_cur
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1
        if batches % 20 == 0:
            print(f"  Batch {batches}: main_loss={loss_main.item():.4f}, curated_loss={loss_cur.item():.4f}, total={loss.item():.4f}")

    avg_loss = total_loss / batches if batches > 0 else 0
    print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {avg_loss:.4f}")
    

# ────────────── Save fine‑tuned model ──────────────
os.makedirs(os.path.dirname(args.output), exist_ok=True)
torch.save(model.state_dict(), args.output)
print(f"Fine‑tuned model saved to {args.output}")