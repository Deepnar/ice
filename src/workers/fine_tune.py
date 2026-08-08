"""Fine‑tuning worker: retrains the MLP head on user‑curated labels.

B4 / G1 (2026-07): closed the previously-inert loop.
  * **Single source of truth** — trains from and promotes to
    ``settings.classifier_model_path`` (was hardcoded ``v2_final.pt`` while the
    live path was ``v3_qwen_ft3.pt``, so the weekly run changed nothing).
  * **Validated promotion** — the candidate is scored on a held-out split and
    only replaces the live checkpoint if it beats the current one on that split
    (with a safe timestamped backup + atomic replace). A bad/tiny curated set
    can no longer silently ship a worse classifier.

B1 (2026-07-25): made schema-driven and generation-agnostic. It reads the head
layout from the checkpoint it is fine-tuning, renders prompts through the SAME
templates the live classifier serves with (D3 — a fine-tune on bare prompt text
would quietly reintroduce the train/inference mismatch B1 exists to fix), and
shares the backup+swap with the pipeline's promote stage
(``src/classifier/promotion.py``).

Curated rows carry their own ``schema_version``. Rows labelled under v1 have a
single 3-way context string, which **cannot** be mapped into v2's four
independent sigmoids (that single-label collapse IS the defect B1 removed), so
when fine-tuning a v2 model those rows train the topic/intent heads and are
masked out of the context head rather than fabricating a target.

Still deferred (see ROADMAP B4 / F9): the *feedback-collection* half
(thumbs-up/down → CuratedLabel) is frontend, and hot-reloading the running
proxy without a restart is a later slice — promotion writes the live path, so a
reload/restart picks it up.
"""

import os
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.api.config import settings
from src.api.db import SessionLocal
from src.classifier import templates
from src.classifier.model import (compute_pos_weights, head_losses,
                                  load_checkpoint)
from src.classifier.promotion import promote_checkpoint
from src.classifier.schema import CONTEXT_RELIANCE, INTENT, TOPIC
from src.memory.embedder import fit_width
from src.memory.models import CuratedLabel
from src.paths import resolve

# Promotion guardrails.
MIN_ROWS_TO_PROMOTE = 20     # below this, train an artifact but never auto-promote
VAL_FRACTION = 0.2           # held-out split for the promotion decision


def _build_labels(rows, schema):
    """Multi-hot targets in *schema*'s layout + a per-row context-head mask.

    The mask is 0 for curated rows whose context labels don't exist in this
    schema generation (see module docstring) — those rows still teach topic and
    intent.
    """
    labels = torch.zeros((len(rows), schema.total_width), dtype=torch.float32)
    ctx_mask = torch.zeros(len(rows), dtype=torch.bool)
    topic_head = schema.head(TOPIC)
    intent_head = schema.head(INTENT)
    ctx_head = schema.head(CONTEXT_RELIANCE)

    for i, row in enumerate(rows):
        for tag in (row.corrected_topic_labels or []):
            if topic_head.has(tag):
                labels[i, topic_head.index(tag)] = 1.0
        for tag in (row.corrected_intent_labels or []):
            if intent_head.has(tag):
                labels[i, intent_head.index(tag)] = 1.0

        multi = list(getattr(row, "corrected_context_labels", None) or [])
        single = row.corrected_context_reliance
        row_version = int(getattr(row, "schema_version", 1) or 1)
        applied = False
        for tag in multi:
            if ctx_head.has(tag):
                labels[i, ctx_head.index(tag)] = 1.0
                applied = True
        # A v2 row with an empty label set is meaningful, not missing: "none of
        # the four" is the derived Zero_Shot state, a real training signal.
        if not applied and row_version >= 2 and schema.version >= 2:
            applied = True
        elif not applied and single and ctx_head.has(single):
            labels[i, ctx_head.index(single)] = 1.0
            applied = True
        ctx_mask[i] = applied

    return labels, ctx_mask


def _encode(prompts, meta, device="cuda"):
    """Render through the live templates, then encode at the head's width."""
    embedder = SentenceTransformer(settings.embedding_model_name, device=device)
    rendered = [templates.render(p, None, version=int(meta.get("template_version", 1)))
                for p in prompts]
    emb = embedder.encode(rendered, convert_to_tensor=True, show_progress_bar=False)
    emb = fit_width(emb, int(meta.get("input_dim", emb.shape[-1])))
    return emb.clone().detach().float().requires_grad_(False)


def _single(schema, head_name):
    """A one-head view of *schema* so heads can be scored individually.

    Head offsets are absolute, so the view still indexes the full-width target
    matrix correctly.
    """
    return replace(schema, heads=(schema.head(head_name),))


def _loss(model, emb, labels, ctx_mask, schema, pos_weights):
    """Summed per-head loss, with the context head restricted to masked rows."""
    logits = model.forward_heads(emb)
    total = torch.zeros((), dtype=torch.float32)
    for head in schema.heads:
        if head.name == CONTEXT_RELIANCE:
            continue
        total = total + head_losses({head.name: logits[head.name]}, labels,
                                    _single(schema, head.name),
                                    pos_weights)[head.name]
    if bool(ctx_mask.any()):
        total = total + head_losses(
            {CONTEXT_RELIANCE: logits[CONTEXT_RELIANCE][ctx_mask]},
            labels[ctx_mask], _single(schema, CONTEXT_RELIANCE), pos_weights,
        )[CONTEXT_RELIANCE]
    return total


def _val_loss(model_or_path, emb_val, lbl_val, mask_val, schema, pos_weights) -> float:
    if isinstance(model_or_path, str):
        model, _ = load_checkpoint(model_or_path)
    else:
        model = model_or_path
    model.eval()
    with torch.no_grad():
        return float(_loss(model, emb_val, lbl_val, mask_val, schema, pos_weights).item())


def fine_tune_classifier():
    """Consent-gated (C7 D6/H5): the maintenance runtime enqueues this only
    when settings.auto_finetune is on — otherwise it lands as a review-queue
    proposal. Never cadence-run; the weekly crontab died with beat."""
    db = SessionLocal()
    try:
        rows = db.query(CuratedLabel).all()
        if not rows:
            return "No curated labels found – skipping fine‑tuning."

        # G31: anchored to the install, not the CWD. Unanchored, this guard
        # aborted the whole fine-tune with "checkpoint missing" whenever the
        # process had been started from anywhere but the repo root — naming a
        # file that was sitting right there. `promote_checkpoint` resolves the
        # same path internally, so the two disagreed.
        live_path = resolve(settings.classifier_model_path)
        if not os.path.exists(live_path):
            return f"Live classifier checkpoint missing at {live_path} – aborting."

        # 1. Load the live model. Its checkpoint declares the schema, the
        #    template generation and the input width to train against.
        model, meta = load_checkpoint(live_path)
        schema = model.schema

        embeddings = _encode([row.prompt for row in rows], meta)
        labels, ctx_mask = _build_labels(rows, schema)
        pos_weights = compute_pos_weights(labels, schema)

        # 2. Held-out split for the promotion decision (deterministic shuffle).
        n = len(rows)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        n_val = int(n * VAL_FRACTION)
        val_idx = torch.tensor(perm[:n_val], dtype=torch.long)
        train_idx = torch.tensor(perm[n_val:], dtype=torch.long)
        can_promote = n >= MIN_ROWS_TO_PROMOTE and n_val >= 1 and len(train_idx) >= 1

        if can_promote:
            emb_train = embeddings[train_idx]
            lbl_train = labels[train_idx]
            mask_train = ctx_mask[train_idx]
        else:
            emb_train, lbl_train, mask_train = embeddings, labels, ctx_mask

        # 3. Fine-tune from the CURRENT LIVE checkpoint (not a stale base).
        #    The live loss is measured BEFORE training mutates these weights.
        emb_val = embeddings[val_idx] if can_promote else None
        lbl_val = labels[val_idx] if can_promote else None
        mask_val = ctx_mask[val_idx] if can_promote else None
        live_loss = (_val_loss(model, emb_val, lbl_val, mask_val, schema, pos_weights)
                     if can_promote else None)

        for param in model.parameters():
            param.requires_grad = True
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with torch.enable_grad():
            for epoch in range(10):
                optimizer.zero_grad()
                loss = _loss(model, emb_train, lbl_train, mask_train, schema, pos_weights)
                loss.backward()
                optimizer.step()
                if epoch % 2 == 0:
                    print(f"  Epoch {epoch}, loss = {loss.item():.4f}")

        model.eval()
        candidate = (model.checkpoint_dict(notes="B4 curated-label fine-tune")
                     if hasattr(model, "checkpoint_dict") else model.state_dict())

        # 4. Always keep the candidate as a timestamped artifact.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # G31: anchored — otherwise the artifact was written into a fabricated
        # `<cwd>/models/classifier/`, which is the exact failure promotion.py's
        # docstring records having already happened once (it reported
        # "backup → None", skipped the backup, and looked like success).
        artifact = resolve(f"models/classifier/ice_classifier_finetuned_{ts}.pt")
        torch.save(candidate, artifact)

        # 5. Validated promotion: only replace the live model if it's better.
        if not can_promote:
            return (
                f"Fine‑tuned artifact saved to {artifact}; NOT promoted "
                f"(need ≥{MIN_ROWS_TO_PROMOTE} curated rows for a held-out check, have {n})."
            )

        cand_loss = _val_loss(model, emb_val, lbl_val, mask_val, schema, pos_weights)

        if cand_loss <= live_loss:
            promoted = promote_checkpoint(candidate, live_path)
            return (
                f"Promoted fine‑tuned classifier to {promoted['live_path']} "
                f"(val loss {cand_loss:.4f} ≤ live {live_loss:.4f}; backup "
                f"{promoted['backup']}). Reload/restart to apply. Artifact: {artifact}"
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
