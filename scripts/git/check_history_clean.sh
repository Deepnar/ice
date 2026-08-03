#!/usr/bin/env bash
# Refuse to let personal data reach the public remote.
#
# WHY THIS EXISTS (2026-08-03). The repo was made public after a git-filter-repo
# run purged ~5,700 personal prompts. The purge rewrote `main` — and nobody
# checked the TAGS. `v2-paper-eval` still pointed into the pre-filter chain, so
# for two days a plain `git clone` of the public repo handed you a 20 MB file of
# personal prompts. The verification at the time was `git log` on main, which
# cannot see that; the only check that would have caught it is the one below,
# run against EVERY ref.
#
# Usage:
#   scripts/git/check_history_clean.sh              # every local ref
#   scripts/git/check_history_clean.sh <ref|sha>... # specific ones
#   scripts/git/check_history_clean.sh --clone      # the paranoid check: clone
#                                                   # the public remote and scan
#                                                   # what the world actually gets
set -uo pipefail

# Paths that must never appear in ANY commit of ANY pushed ref. Matched against
# full history, not the current tree — the whole point is that a file deleted in
# a later commit is still in the earlier one.
SENSITIVE_PATHS=(
  "data/labeled/labeled_prompts.jsonl"
  "data/labeled/failed_prompts.jsonl"
  "data/curated_fixes.jsonl"
  "data/simulation_input.jsonl"
  "docs/PUBLISHING.md"
  "PERSONAL_ROADMAP.md"
  "REVISION_PLAN_v2.md"
  ".env"
)
# Anything under these ever, by glob — catches a renamed or newly-added sibling
# that nobody thought to list above.
SENSITIVE_GLOBS=(
  "data/simulation/**"
  "data/labeled/**"
  "data/archive/**"
)

fail=0

scan_refs() {
  local refs=("$@")
  for path in "${SENSITIVE_PATHS[@]}"; do
    hits=$(git log --format=%H --full-history "${refs[@]}" -- "$path" 2>/dev/null | wc -l)
    if [ "$hits" -gt 0 ]; then
      echo "  ✖ $path — in $hits commit(s)"
      git log --oneline --full-history "${refs[@]}" -- "$path" 2>/dev/null | head -3 | sed 's/^/      /'
      fail=1
    fi
  done
  for glob in "${SENSITIVE_GLOBS[@]}"; do
    hits=$(git log --format=%H --full-history "${refs[@]}" -- "$glob" 2>/dev/null | wc -l)
    if [ "$hits" -gt 0 ]; then
      echo "  ✖ $glob — in $hits commit(s)"
      fail=1
    fi
  done
}

if [ "${1:-}" = "--clone" ]; then
  remote=$(git remote get-url origin)
  tmp=$(mktemp -d)
  echo "cloning $remote to check what the public actually receives…"
  git clone -q "$remote" "$tmp/repo" || { echo "clone failed"; exit 2; }
  cd "$tmp/repo" || exit 2
  scan_refs --all
  cd - >/dev/null || true
  rm -rf "$tmp"
else
  if [ "$#" -gt 0 ]; then
    scan_refs "$@"
  else
    # --all is the load-bearing part: it covers tags, not just branches.
    scan_refs --all
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "REFUSING: personal data is reachable from the refs above."
  echo "A file removed in a later commit is still in the earlier one — rewriting"
  echo "history is the only fix, and it must rewrite TAGS as well as branches."
  exit 1
fi
echo "history clean — no sensitive path reachable from the refs checked"
exit 0
