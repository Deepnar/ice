"""Portable memory export (G23 D2) — one JSONL per table + manifest.json.

State-copy: ids preserved, vectors EXCLUDED by default (re-embedded on
import; --with-vectors for exact clones). Secrets are never exported.
Done = the printed directory holds manifest.json + per-table JSONL files.

Usage:
    uv run python scripts/ice_export.py
    uv run python scripts/ice_export.py --out exports/my_export --with-vectors
"""
import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal  # noqa: E402
from src.memory.portability import export_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="output directory (default: exports/ice_export_<ts>)")
    parser.add_argument("--with-vectors", action="store_true",
                        help="include embeddings (exact clone; heavy)")
    args = parser.parse_args()

    out = args.out or ("exports/ice_export_"
                       + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    db = SessionLocal()
    try:
        manifest = export_store(db, out, with_vectors=args.with_vectors)
    finally:
        db.close()

    total = sum(manifest["tables"].values())
    print(f"Exported {total} rows across {len(manifest['tables'])} tables "
          f"to {out}/")
    print(f"alembic head: {manifest['alembic_head']}  |  embedding: "
          f"{manifest['embedding']}")
    print("Import with: uv run python scripts/ice_import.py --in " + out)


if __name__ == "__main__":
    main()
