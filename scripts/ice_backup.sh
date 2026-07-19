#!/usr/bin/env bash
# G23 D3: one-command full backup of the ICE store (DB dump + models/ +
# config snapshot) into backups/ice_backup_<timestamp>.tar.gz.
# Done = the printed archive path exists and the size is logged.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m src.memory.backup "$@"
