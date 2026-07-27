#!/usr/bin/env python3
"""Stage 8 — promote the candidate to the live classifier.

Was buried inside ``legacy/training/fine_tune.py``. The mechanism now lives in
``src/classifier/promotion.py`` and is shared with the B4 curated-label
fine-tune worker, so "back up, then atomically replace
``settings.classifier_model_path``" has exactly one implementation.

This stage is deliberately thin and deliberately gated:

* it **re-runs D5's gate** rather than trusting a report file — a stale
  ``_eval.json`` next to a rebuilt checkpoint is the obvious way to promote a
  model that never passed;
* it requires **explicit confirmation** (``--yes``). Promotion is USER-REQUIRED
  (D-USER d): it changes what every subsequent turn of the running system does.

The live path keeps its filename; only its contents change. The proxy picks the
new head up on reload/restart (hot-reload without a restart is still B4's open
half).

Usage:
    uv run python scripts/classifier/pipeline/promote.py \
        --candidate models/classifier/ice_classifier_v4_schema2.pt --yes
"""

import argparse
import os
import subprocess
import sys

import torch

from src.api.config import settings
from src.classifier.model import load_checkpoint
from src.classifier.promotion import promote_checkpoint


def run_gate(candidate: str, baseline: str) -> bool:
    """Re-run stage 7 and read its exit code (0 = PASS)."""
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, os.path.join(here, "evaluate.py"),
         "--candidate", candidate, "--baseline", baseline],
        cwd=here)
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="B1 stage 8: promote the new classifier")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--live", default=settings.classifier_model_path)
    ap.add_argument("--yes", action="store_true", help="confirm the swap")
    ap.add_argument("--skip-gate", action="store_true",
                    help="promote without re-running D5 (records the override)")
    args = ap.parse_args()

    if not os.path.exists(args.candidate):
        raise SystemExit(f"candidate not found: {args.candidate}")

    model, meta = load_checkpoint(args.candidate)
    print(f"[promote] candidate: {args.candidate}")
    print(f"[promote]   schema_version={meta.get('schema_version')} "
          f"template_version={meta.get('template_version')} "
          f"input_dim={meta.get('input_dim')}")
    print(f"[promote]   trained_at={meta.get('trained_at')} "
          f"val_macro_f1={meta.get('val_macro_f1')}")
    print(f"[promote] live path: {args.live}")

    if args.skip_gate:
        print("[promote] ⚠ D5 gate SKIPPED by flag — recorded in the checkpoint notes")
    elif not run_gate(args.candidate, args.live):
        raise SystemExit("[promote] D5 gate FAILED — not promoting. "
                         "Diagnose the data before touching the live model.")

    if not args.yes:
        print("\n[promote] dry run — pass --yes to actually swap the live checkpoint.")
        return

    payload = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and args.skip_gate:
        payload["notes"] = f"{payload.get('notes', '')} [promoted with --skip-gate]"
    result = promote_checkpoint(payload, args.live)

    print(f"[promote] promoted → {result['live_path']}")
    print(f"[promote] previous checkpoint backed up → {result['backup']}")
    print("[promote] restart/reload the proxy to serve the new head.")
    print(f"[promote] rollback: cp {result['backup']} {result['live_path']} "
          "(a checkpoint declares its own schema_version, so a rollback is a file "
          "swap — no code change).")


if __name__ == "__main__":
    main()
