"""Register a repo with ICE's coding core (E1 — USER-REQUIRED step).

Points ICE at a repo root, bootstraps the code graph (E1b) + project facts
(E9), and optionally installs the consent-gated post-commit hook (E3).
Done = the printed report lists entities/edges/facts.

Usage:
    uv run python scripts/register_project.py --name ICE --root /home/deepnar/Programs/ice --hook
    uv run python scripts/register_project.py --name myproj --root ~/code/myproj --no-hook
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal  # noqa: E402
from src.services.projects import register_project  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="project display name")
    parser.add_argument("--root", required=True, help="repo root directory")
    parser.add_argument("--slug", default=None, help="short slug (default: from name)")
    hook = parser.add_mutually_exclusive_group(required=True)
    hook.add_argument("--hook", action="store_true",
                      help="install the post-commit hook (instant reconcile)")
    hook.add_argument("--no-hook", action="store_true",
                      help="decline the hook (10-min poll fallback)")
    parser.add_argument("--replay-git-log", action="store_true",
                        help="record the git-log replay request (honored when "
                             "F10's importer lands; off by default — heavy)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        report = register_project(
            db, args.name, args.root, slug=args.slug,
            install_git_hook=args.hook,
            replay_git_log=args.replay_git_log,
        )
    finally:
        db.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
