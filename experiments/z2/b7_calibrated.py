"""B7 classifier variants, judged the way the live classifier was chosen (B1, PROVENANCE 2026-07-27).

The first B7 run (logs/fix-pass/b7_variants.py) trained at train.py's default pos-weight cap 3 and
compared untuned held-out F1 — but the live checkpoint was trained at cap 5 and its tag threshold
fitted on val (0.65, stamped in the checkpoint); B1 measured the threshold, not the training
detail, as what moves F1 (0.526 untuned -> 0.610 fitted). So "retrained beats live" there was a
threshold artifact. This run: each variant trained at cap 5, its global threshold fitted on VAL by
sweep_threshold.py and stamped, test macro-F1 at that threshold; the live checkpoint scored the
same way as the reference. Style flips on Z2's 35 base/messy pairs at each model's own threshold.
Keep a variant only if it beats live on test macro-F1 and does not flip more; 35 pairs resolve only
large differences (about +-15 points).

    uv run --no-sync python experiments/z2/b7_calibrated.py        # resumable: trained variants are reused
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LIVE = "models/classifier/ice_classifier_v4_schema2.pt"
OUT = "models/classifier/b7-cap5"
VARIANTS = {"retrained": [], "standardize": ["--input-norm", "standardize"],
            "layernorm": ["--input-norm", "layernorm"], "kaiming": ["--init", "kaiming"]}
PY = [sys.executable]


def style_pairs():
    pairs = []
    for h in "BC":
        fams = {f["id"]: f for f in json.loads((ROOT / f"logs/z2-questions-final/{h}/families.json").read_text())["kept"]}
        pairs += [(fams[f["base_family"]]["question"], f["question"]) for f in fams.values()
                  if f["type"] == "messy_variant" and f.get("base_family") in fams]
    return pairs


def calibrated(path, write):
    """Fit (and optionally stamp) the global threshold on val; return its test scores."""
    thr = path.replace(".pt", "_thresholds.json")
    if not os.path.exists(thr) or write:
        subprocess.run(PY + ["scripts/classifier/pipeline/sweep_threshold.py", "--checkpoint", path,
                             *(["--write"] if write else [])], check=True, cwd=ROOT)
    return json.loads((ROOT / thr).read_text())


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer
    from src.api.config import settings
    from src.classifier.model import load_checkpoint
    os.makedirs(ROOT / OUT, exist_ok=True)
    pairs = style_pairs()
    enc = SentenceTransformer(settings.embedding_model_name, device="cuda")
    A = enc.encode([a for a, _ in pairs], convert_to_tensor=True).float().cpu()
    B = enc.encode([b for _, b in pairs], convert_to_tensor=True).float().cpu()
    rows = {}
    models = {"live": LIVE, **{name: f"{OUT}/{name}.pt" for name in VARIANTS}}
    for name, path in models.items():
        if name != "live" and not (ROOT / path).exists():
            subprocess.run(PY + ["scripts/classifier/pipeline/train.py", "--out", path, "--pos-weight-cap", "5",
                                 *VARIANTS[name]], check=True, cwd=ROOT)
        scores = calibrated(path, write=name != "live")                     # never re-stamp the live model
        model, _ = load_checkpoint(str(ROOT / path))
        model.eval()
        schema, thr = model.schema, float(getattr(model, "tag_threshold", None) or scores["best_global"])
        with torch.no_grad():
            pa, pb = torch.sigmoid(model(A)), torch.sigmoid(model(B))
        ctx, intent = schema.slice("context_reliance"), schema.slice("intent")
        mem = schema.heads[[h.name for h in schema.heads].index("context_reliance")].labels.index("Needs_Memory")
        rows[name] = dict(test=scores["test_scores"]["global"], threshold=scores["best_global"],
                          memory_flip=((pa[:, ctx][:, mem] >= thr) != (pb[:, ctx][:, mem] >= thr)).float().mean().item(),
                          intent_flip=((pa[:, intent] >= thr) != (pb[:, intent] >= thr)).any(dim=1).float().mean().item())
    print(f"{len(pairs)} style pairs; test macro-F1 at each model's val-fitted global threshold")
    for name, r in rows.items():
        t = r["test"]
        print(f"{name:12} thr {r['threshold']:.2f}  topic {t['topic']['macro']:.3f}  intent {t['intent']['macro']:.3f}  "
              f"context {t['context_reliance']['macro']:.3f}  mean {t['mean_macro']:.3f}  "
              f"memory flip {r['memory_flip']:.2f}  intent-set flip {r['intent_flip']:.2f}")
    (ROOT / OUT / "report.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
