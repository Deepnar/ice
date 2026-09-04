#!/usr/bin/env bash
# Score one completed matched cloud LME phase with Muse Spark through Responses.
set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${ICE_V2_WORKTREE:-/home/deepnar/Programs/ice-worktrees/v2-paper-eval}"
RUN_ROOT="${LME_CLOUD_RUN_ROOT:-$HARNESS/runs/v2-paper-eval-cloud-v1}"
PHASE="${1:-}"

if [[ "$PHASE" != "oracle" && "$PHASE" != "full" ]]; then
  echo "usage: $0 {oracle|full} [score options]" >&2
  exit 2
fi
shift

cd "$WORKTREE"
exec uv run --no-sync python "$HARNESS/score.py" \
  --phase "$PHASE" \
  --out "$RUN_ROOT" \
  --judge-profile opencode-muse13 \
  --max-tokens 4096 \
  --workers 3 \
  "$@"
