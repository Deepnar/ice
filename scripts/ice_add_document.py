#!/usr/bin/env python3
"""C12 CLI: ingest a document into ICE memory.

    uv run python scripts/ice_add_document.py <file> --dry-run
    uv run python scripts/ice_add_document.py <file> --conversation <uuid>
    uv run python scripts/ice_add_document.py <file> --kind transcript

Sibling of `scripts/ice_replay_import.py` (conversation exports). The document
becomes its own conversation and lives through the real pipeline — codex,
clustering, density, summaries — and is readable ONLY in the conversations
where it is enabled. Without --conversation it lands in the library, enabled
nowhere, for you to switch on where you want it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal                     # noqa: E402
from src.services import documents as documents_svc     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest a document into ICE")
    ap.add_argument("path")
    ap.add_argument("--conversation", default=None,
                    help="enable the document in this conversation")
    ap.add_argument("--project", default=None, help="project id")
    ap.add_argument("--kind", default=None, choices=("document", "transcript"),
                    help="override the document/transcript detection")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + estimate only; writes nothing")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        result = documents_svc.add_document(
            db, conversation_id=args.conversation, path=args.path,
            doc_kind=args.kind, project_id=args.project, dry_run=args.dry_run)
    except Exception as exc:
        print(f"failed: {exc}")
        return 1
    finally:
        db.close()

    if args.dry_run:
        print(f"{result['filename']} — {result['kind']}, "
              f"{result['n_sections']} sections, "
              f"~{result['estimate_human']} of background extraction")
        print(result.get("note", ""))
        return 0
    print(f"document {result['id']} — {result['status']}, "
          f"{result['n_sections']} sections")
    if result.get("deduplicated"):
        print("(identical bytes were already ingested — enabled it here instead)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
