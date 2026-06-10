"""Fine‑tuning worker: retrains the MLP head on user‑curated labels."""

import torch
import numpy as np
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

from src.workers.celery_app import app
from src.api.db import SessionLocal
from src.memory.models import CuratedLabel
from src.classifier.model import ICEClassifier

# ------------------------------------------------------------------
# Constants – these MUST match the ones used during initial training
# ------------------------------------------------------------------
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

MODEL_PATH = "models/classifier/ice_classifier_v2_final.pt"
SCHEMA_PATH = "data/labeled/label_schema.json"

# ------------------------------------------------------------------
@app.task(bind=True, max_retries=1)
def fine_tune_classifier(self):
    """Load curated labels, encode, train head, save new checkpoint."""

    db = SessionLocal()
    try:
        rows = db.query(CuratedLabel).all()
        if not rows:
            return "No curated labels found – skipping fine‑tuning."

        # 1. Encode prompts with frozen SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        prompts = [row.prompt for row in rows]
        embeddings = embedder.encode(prompts, convert_to_tensor=True, show_progress_bar=False)

        # 2. Build label tensors from the curated labels
        labels = torch.zeros((len(rows), 25), dtype=torch.float32)
        for i, row in enumerate(rows):
            # Topic labels (positions 0–10)
            for tag in row.corrected_topic_labels:
                if tag in TOPIC_LABELS:
                    labels[i, TOPIC_LABELS.index(tag)] = 1.0
            # Intent labels (positions 11–21)
            for tag in row.corrected_intent_labels:
                if tag in INTENT_LABELS:
                    labels[i, 11 + INTENT_LABELS.index(tag)] = 1.0
            # Context reliance (positions 22–24, one‑hot)
            if row.corrected_context_reliance in CONTEXT_RELIANCE_LABELS:
                labels[i, 22 + CONTEXT_RELIANCE_LABELS.index(row.corrected_context_reliance)] = 1.0

        # 3. Load the trained model and freeze the encoder (not used in this script, but
        #    the model only contains the head – the SentenceTransformer is separate).
        model = ICEClassifier()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.train()

        # 4. Simple training loop (few epochs, tiny dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn_topic = torch.nn.BCEWithLogitsLoss()
        loss_fn_intent = torch.nn.BCEWithLogitsLoss()
        loss_fn_ctx = torch.nn.CrossEntropyLoss()

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(embeddings)

            topic_out = outputs[:, :11]
            intent_out = outputs[:, 11:22]
            ctx_out = outputs[:, 22:]

            topic_gt = labels[:, :11]
            intent_gt = labels[:, 11:22]
            ctx_gt = labels[:, 22:].argmax(dim=1)

            loss = (
                loss_fn_topic(topic_out, topic_gt) +
                loss_fn_intent(intent_out, intent_gt) +
                loss_fn_ctx(ctx_out, ctx_gt)
            )
            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                print(f"  Epoch {epoch}, loss = {loss.item():.4f}")

        # 5. Save updated checkpoint
        new_path = f"models/classifier/ice_classifier_finetuned_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.state_dict(), new_path)
        return f"Fine‑tuned model saved to {new_path}"

    except Exception as exc:
        db.rollback()
        raise self.retry(exc=exc)
    finally:
        db.close()