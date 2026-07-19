"""The re-embed runner's CLI (G23 D4) — re-encode every vector column.

Prints the row-count estimate up front; resumable (kill-safe per-table
stamps in store_meta) — keep the machine on, rerun to resume. Run after
any embedder change (the startup guard names this command) or a
vectorless import.

Usage:
    uv run python scripts/ice_reembed.py
    uv run python scripts/ice_reembed.py --tables episodic_memory,codex_entities
    uv run python scripts/ice_reembed.py --force     # restart, ignore stamps
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal  # noqa: E402
from src.memory.reembed import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default="all",
                        help="'all' (default) or a comma-separated subset")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--force", action="store_true",
                        help="restart tables from scratch, ignoring 'done' stamps")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        run(db, tables=args.tables, batch_size=args.batch, force=args.force)
    finally:
        db.close()


if __name__ == "__main__":
    main()
