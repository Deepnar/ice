"""F10/F14 conversation import — the REPLAY path (re-live an exported chat
history through the full ICE pipeline: post-flight → codex → procedural →
clustering → summaries, producing mature memory rather than an archive).

Sibling of scripts/ice_import.py (G23's id-preserving STATE-COPY). This one
runs history through the pipeline instead of copying rows.

Supported inputs (auto-detected): ChatGPT / Claude / DeepSeek exports
(conversations.json), generic JSONL ({role, content, timestamp} per line),
and raw .txt dumps (F14 slicer). Decay policy: hybrid (default) / preserve /
fast_forward / fresh.

Runs INLINE and synchronously in the foreground (the natural "machine on for
the import" UX) — the REST endpoint (POST /user-control/import) is the path
that yields to live chat via the runtime's gpu-lane slices. The real run needs
Ollama up (background-model extraction per turn).

Usage:
    uv run python scripts/ice_replay_import.py --in data/simulation/raw_chats/claude.json
    uv run python scripts/ice_replay_import.py --in dump.txt --policy fast_forward
    uv run python scripts/ice_replay_import.py --in export.json --dry-run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.core import create_core          # noqa: E402
from src.api.db import SessionLocal            # noqa: E402
from src.ingestion import importer             # noqa: E402
from src.services import ingestion as ingestion_svc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="ICE conversation import (replay)")
    ap.add_argument("--in", dest="source", required=True,
                    help="path to an export file (json/jsonl) or raw .txt dump")
    ap.add_argument("--policy", default="hybrid", choices=importer.POLICIES,
                    help="decay policy (default: hybrid)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + count + estimate only; import nothing")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        print(f"error: file not found: {args.source}")
        return 1

    # create_core runs the store_meta embedding-identity guard and gives the
    # shared classifier; no runtime (this CLI replays inline in the foreground).
    core = create_core(start_runtime=False)
    db = SessionLocal()
    try:
        result = ingestion_svc.start_import(
            db, args.source, args.policy, dry_run=args.dry_run,
            runtime=None, classifier=core.classifier)
    finally:
        db.close()

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for k in ("source_format", "policy", "total_conversations",
                  "total_turns", "estimate_human"):
            print(f"  {k}: {result.get(k)}")
        print(f"  note: {result.get('note')}")
        return 0

    print("\n=== IMPORT COMPLETE ===")
    for k in ("status", "done_conversations", "skipped_conversations",
              "done_turns", "failed_turns"):
        print(f"  {k}: {result.get(k)}")
    if result.get("report"):
        r = result["report"]
        print(f"  branch messages skipped (other branches): "
              f"{r.get('branch_messages_skipped')}")
        print(f"  conversations with synthesized timestamps: "
              f"{r.get('ts_synthesized_conversations')}")
    print("\nImported conversations arrived auto-scoped and non-private — "
          "re-scope any of them from the review/scope tools if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
