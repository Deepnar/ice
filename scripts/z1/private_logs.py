"""Private outputs stay under the checkout's logs/ (gitignored).

The path is checked as given — absolute, ".." collapsed, symlinks NOT followed — so a worktree's
logs/<run> that links into the main checkout's logs (fix-pass's logs/z2-run-main, 2026-11-01) is
accepted, while a path that leaves logs/ by ".." is refused. Following the link first refused every
question of a run whose output sat in a linked folder.
"""

import os
from pathlib import Path

LOGS = Path(__file__).resolve().parents[2] / "logs"


def under_logs(path) -> bool:
    return Path(os.path.abspath(path)).is_relative_to(LOGS)
