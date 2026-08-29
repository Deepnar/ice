#!/usr/bin/env python3
"""Fetch the LongMemEval corpus at a PINNED revision and record a manifest.

Why pinned: PROVENANCE.md's standing rule and FINAL_experiments.md rev 3c --
"pin revisions, not names". A HF dataset repo can be re-uploaded or silently
revised, so `xiaowu0162/longmemeval` is not a reproducible reference; that repo
*at revision 2ec2a557...* is.

Downloads to experiments/lme/data/ (gitignored -- 278 MB + 15 MB).

    uv run python experiments/lme/fetch_dataset.py
    uv run python experiments/lme/fetch_dataset.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "xiaowu0162/longmemeval"
# Pinned 2026-08-29. Re-pin deliberately, never silently -- a changed corpus
# invalidates every number measured against the old one.
REVISION = "2ec2a557f339b6c0369619b1ed5793734cc87533"

# longmemeval_m (2.7 GB) is deliberately NOT fetched: FINAL_experiments.md 2.8
# specifies the S variant, and M's 500-sessions-per-instance haystack is far
# beyond what this hardware can replay through a full post-flight pipeline.
FILES = {
    "longmemeval_s": "the run corpus -- ~500 questions, ~40 sessions each",
    "longmemeval_oracle": "CONTROL. Evidence sessions only. If ICE cannot answer "
    "from these, the adapter is broken, not the memory.",
}

BASE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(name: str, dest: Path) -> None:
    url = BASE.format(repo=REPO, rev=REVISION, name=name)
    print(f"  fetching {name} ...", flush=True)
    tmp = dest.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="hash what is on disk and check it against the manifest")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "MANIFEST.json"

    entries = {}
    for name, why in FILES.items():
        dest = DATA / name
        if not dest.exists():
            if args.verify_only:
                print(f"  MISSING {name}")
                return 1
            _download(name, dest)
        digest = _sha256(dest)
        entries[name] = {
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "purpose": why,
        }
        print(f"  {name}: {dest.stat().st_size / 1e6:.1f} MB  sha256={digest[:16]}...")

    if args.verify_only and manifest_path.exists():
        old = json.loads(manifest_path.read_text())["files"]
        for name, ent in entries.items():
            if old.get(name, {}).get("sha256") != ent["sha256"]:
                print(f"  ⛔ {name} CHANGED since the manifest was written.")
                return 1
        print("  all files match the manifest")
        return 0

    manifest_path.write_text(json.dumps({
        "repo": REPO,
        "revision": REVISION,
        "license": "MIT",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }, indent=2) + "\n")
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
