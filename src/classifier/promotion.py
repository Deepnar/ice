"""Checkpoint promotion — ONE implementation, two callers.

Swapping the live classifier is the same operation whether it comes from B4's
curated-label fine-tune (``src/workers/fine_tune.py``, small nudges, runs on a
schedule) or from B1's full retrain (``scripts/classifier/pipeline/promote.py``,
a whole new schema generation, runs by hand). Two copies of "back up, then
atomically replace ``settings.classifier_model_path``" is exactly the kind of
duplication that drifts until one of them forgets the backup.

Contract:
  * the outgoing checkpoint is copied to ``<name>_prev_<utc-ts>.pt`` first — no
    promotion is unrecoverable;
  * the new file is written to a temp path and ``os.replace``d, so a reader that
    opens the live path mid-promotion gets either the old or the new model, never
    a half-written one (same filesystem ⇒ atomic).

Neither caller decides *whether* to promote here — the gates (B4's held-out loss
comparison, B1's D5 non-regression check) stay with the caller that has the
evaluation data.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import torch

# src/classifier/promotion.py → repo root
_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> str:
    """Anchor a relative checkpoint path to the repo root.

    ``settings.classifier_model_path`` is repo-relative, but callers do not all
    run from the repo root: the pipeline stages run from their own directory
    (they import ``common``, which chdirs — ``promote.py`` did not), and a worker
    inherits whatever cwd the process was started with.

    Without this, promotion resolved the live path against the caller's cwd and
    **silently did the wrong thing twice over**: it wrote the new checkpoint to a
    fabricated `<cwd>/models/classifier/...` and, finding nothing at that path to
    displace, reported `backup → None` and skipped the backup. The live model was
    never replaced, the gate had already printed PASS, and every log line looked
    like success. Mirrors ``schema._resolve`` — same convention, same reason.
    """
    p = Path(path)
    return str(p if p.is_absolute() else _ROOT / p)


def promote_checkpoint(candidate: Union[dict, str], live_path: str,
                       backup_dir: Optional[str] = None) -> dict:
    """Install *candidate* as the live classifier.

    *candidate* is either an in-memory checkpoint payload (what a trainer holds)
    or a path to a saved ``.pt`` artifact (what the pipeline holds).

    Returns ``{"live_path", "backup", "promoted_at"}`` — the backup path is None
    only when there was no live checkpoint to displace.
    """
    live_path = _resolve(live_path)
    if isinstance(candidate, str):
        candidate = _resolve(candidate)
    os.makedirs(os.path.dirname(live_path) or ".", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    backup = None
    if os.path.exists(live_path):
        name = os.path.basename(live_path).replace(".pt", f"_prev_{ts}.pt")
        backup = os.path.join(backup_dir or os.path.dirname(live_path), name)
        os.makedirs(os.path.dirname(backup) or ".", exist_ok=True)
        shutil.copy2(live_path, backup)

    tmp = f"{live_path}.tmp"
    if isinstance(candidate, str):
        shutil.copy2(candidate, tmp)
    else:
        torch.save(candidate, tmp)
    os.replace(tmp, live_path)  # atomic on the same filesystem

    return {"live_path": live_path, "backup": backup,
            "promoted_at": datetime.now(timezone.utc).isoformat()}
