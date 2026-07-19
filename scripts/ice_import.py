"""Staged memory import (G23 D2) — restore an ice_export directory.

Default restores into an EMPTY store (id-preserving); --merge skips rows
whose ids already exist. The target DB's alembic head must match the
manifest's. Ends with the re-embed pass (unless vectors were carried and
match settings) and the codex context_payload sweep.

Usage:
    uv run python scripts/ice_import.py --in exports/ice_export_20260719_120000
    uv run python scripts/ice_import.py --in <dir> --merge
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal  # noqa: E402
from src.memory.portability import import_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", required=True,
                        help="export directory (holds manifest.json)")
    parser.add_argument("--merge", action="store_true",
                        help="skip existing ids instead of requiring an empty store")
    parser.add_argument("--skip-reembed", action="store_true",
                        help="leave embeddings NULL (run scripts/ice_reembed.py later)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        report = import_store(db, args.in_dir, merge=args.merge,
                              skip_reembed=args.skip_reembed)
    finally:
        db.close()

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
