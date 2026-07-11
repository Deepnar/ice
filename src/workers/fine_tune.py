"""Fine‑tuning worker: retrains the MLP head on user‑curated labels.

B4 / G1 (2026-07): closed the previously-inert loop.
  * **Single source of truth** — trains from and promotes to
    ``settings.classifier_model_path`` (was hardcoded ``v2_final.pt`` while the
    live path was ``v3_qwen_ft3.pt``, so the weekly run changed nothing).
  * **Validated promotion** — the candidate is scored on a held-out split and
    only replaces the live checkpoint if it beats the current one on that split
    (with a safe timestamped backup + atomic replace). A bad/tiny curated set
    can no longer silently ship a worse classifier.

Still deferred (see ROADMAP B4 / F9): the *feedback-collection* half
(thumbs-up/down → CuratedLabel) is frontend, and hot-reloading the running
proxy without a restart is a later slice — promotion writes the live path, so a
reload/restart picks it up.
"""

import os
import shutil
import numpy as np
import torch
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CuratedLabel
from src.classifier.model import ICEClassifier

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

# Promotion guardrails.
MIN_ROWS_TO_PROMOTE = 20     # below this, train an artifact but never auto-promote
VAL_FRACTION = 0.2           # held-out split for the promotion decision


def _build_labels(rows) -> torch.Tensor:
    labels = torch.zeros((len(rows), 25), dtype=torch.float32)
    for i, row in enumerate(rows):
        for tag in row.corrected_topic_labels:
            if tag in TOPIC_LABELS:
                labels[i, TOPIC_LABELS.index(tag)] = 1.0
        for tag in row.corrected_intent_labels:
            if tag in INTENT_LABELS:
                labels[i, 11 + INTENT_LABELS.index(tag)] = 1.0
        if row.corrected_context_reliance in CONTEXT_RELIANCE_LABELS:
            labels[i, 22 + CONTEXT_RELIANCE_LABELS.index(row.corrected_context_reliance)] = 1.0
    return labels


def _combined_loss(outputs, labels) -> torch.Tensor:
    loss_fn_topic = torch.nn.BCEWithLogitsLoss()
    loss_fn_intent = torch.nn.BCEWithLogitsLoss()
    loss_fn_ctx = torch.nn.CrossEntropyLoss()
    return (
        loss_fn_topic(outputs[:, :11], labels[:, :11]) +
        loss_fn_intent(outputs[:, 11:22], labels[:, 11:22]) +
        loss_fn_ctx(outputs[:, 22:], labels[:, 22:].argmax(dim=1))
    )


def _val_loss(state_dict, emb_val, lbl_val) -> float:
    model = ICEClassifier()
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        return float(_combined_loss(model(emb_val), lbl_val).item())


def _promote(candidate_state, live_path: str) -> str:
    """Back up the current live checkpoint, then atomically replace it with the
    validated candidate so the next classifier load uses the better model."""
    os.makedirs(os.path.dirname(live_path), exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if os.path.exists(live_path):
        backup = live_path.replace(".pt", f"_prev_{ts}.pt")
        shutil.copy2(live_path, backup)
    tmp = live_path + ".tmp"
    torch.save(candidate_state, tmp)
    os.replace(tmp, live_path)  # atomic on the same filesystem
    return live_path


def fine_tune_classifier():
    """Consent-gated (C7 D6/H5): the maintenance runtime enqueues this only
    when settings.auto_finetune is on — otherwise it lands as a review-queue
    proposal. Never cadence-run; the weekly crontab died with beat."""
    db = SessionLocal()
    try:
        rows = db.query(CuratedLabel).all()
        if not rows:
            return "No curated labels found – skipping fine‑tuning."

        live_path = settings.classifier_model_path
        if not os.path.exists(live_path):
            return f"Live classifier checkpoint missing at {live_path} – aborting."

        # 1. Encode with the frozen encoder (same as the live classifier).
        embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", truncate_dim=384)
        prompts = [row.prompt for row in rows]
        embeddings = embedder.encode(prompts, convert_to_tensor=True, show_progress_bar=False)
        embeddings = embeddings.clone().detach().float().requires_grad_(False)

        labels = _build_labels(rows)

        # 2. Held-out split for the promotion decision (deterministic shuffle).
        n = len(rows)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        n_val = int(n * VAL_FRACTION)
        val_idx = torch.tensor(perm[:n_val], dtype=torch.long)
        train_idx = torch.tensor(perm[n_val:], dtype=torch.long)
        can_promote = n >= MIN_ROWS_TO_PROMOTE and n_val >= 1 and len(train_idx) >= 1

        emb_train = embeddings[train_idx] if can_promote else embeddings
        lbl_train = labels[train_idx] if can_promote else labels

        # 3. Fine-tune from the CURRENT LIVE checkpoint (not a stale base).
        base_state = torch.load(live_path, map_location="cpu")
        model = ICEClassifier()
        model.load_state_dict(base_state)
        for param in model.parameters():
            param.requires_grad = True
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with torch.enable_grad():
            for epoch in range(10):
                optimizer.zero_grad()
                loss = _combined_loss(model(emb_train), lbl_train)
                loss.backward()
                optimizer.step()
                if epoch % 2 == 0:
                    print(f"  Epoch {epoch}, loss = {loss.item():.4f}")

        model.eval()
        candidate_state = model.state_dict()

        # 4. Always keep the candidate as a timestamped artifact.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifact = f"models/classifier/ice_classifier_finetuned_{ts}.pt"
        torch.save(candidate_state, artifact)

        # 5. Validated promotion: only replace the live model if it's better.
        if not can_promote:
            return (
                f"Fine‑tuned artifact saved to {artifact}; NOT promoted "
                f"(need ≥{MIN_ROWS_TO_PROMOTE} curated rows for a held-out check, have {n})."
            )

        emb_val, lbl_val = embeddings[val_idx], labels[val_idx]
        cand_loss = _val_loss(candidate_state, emb_val, lbl_val)
        live_loss = _val_loss(base_state, emb_val, lbl_val)

        if cand_loss <= live_loss:
            promoted_to = _promote(candidate_state, live_path)
            return (
                f"Promoted fine‑tuned classifier to {promoted_to} "
                f"(val loss {cand_loss:.4f} ≤ live {live_loss:.4f}). Reload/restart to apply. "
                f"Artifact: {artifact}"
            )
        return (
            f"Fine‑tuned artifact saved to {artifact}; NOT promoted "
            f"(val loss {cand_loss:.4f} > live {live_loss:.4f} — live model kept)."
        )

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
