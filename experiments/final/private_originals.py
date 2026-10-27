"""FINAL private histories as Z2-format originals (FINAL_experiments.md §2.1).

From the v1 corpus (data/simulation/simulation_full.jsonl, gitignored) through
experiments/v3_repair/prepare_inputs.py's loader, so rows carry the same fields and hashes as
Z2's B and C. Output (private, gitignored): logs/final-private-inputs/final-originals-<KEY>.json
and a manifest with the corpus fingerprint.

    uv run --no-sync python experiments/final/private_originals.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data/simulation/simulation_full.jsonl"
OUT = ROOT / "logs/final-private-inputs"
# key -> (conversation id prefix, pairs). LAP: the laptop-decision chat (the maintainer approved
# it 2026-10-26 in place of a DeepSeek thread that turned out to be an excluded conversation).
# D (ecc64aab, 251 pairs) was dropped from FINAL 2026-10-28 (the v3 hyper-parameter tuning history).
HISTORIES = {"A": ("633e26f8", 290), "SW": ("bd94e64b", 128), "LAP": ("942699a4", 101)}


def main() -> int:
    from experiments.v3_repair.prepare_inputs import fingerprint, load_corpus
    groups = load_corpus(CORPUS, {prefix: n for prefix, n in HISTORIES.values()})
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(corpus=fingerprint(CORPUS), histories={})
    for key, (prefix, n) in HISTORIES.items():
        rows = groups[prefix]
        path = OUT / f"final-originals-{key}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        manifest["histories"][key] = dict(prefix=prefix, pairs=len(rows), file=path.name,
                                          first=rows[0]["timestamp"], last=rows[-1]["timestamp"])
        print(f"{key}: {len(rows)} pairs, {rows[0]['timestamp'][:10]}..{rows[-1]['timestamp'][:10]}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
