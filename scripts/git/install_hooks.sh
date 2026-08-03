#!/usr/bin/env bash
# Install the repo's git hooks. Idempotent; run once per clone.
#
# Hooks live outside version control, so a check that only exists in .git/hooks
# is a check the next clone does not have. This installer is the tracked half.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
hook="$root/.git/hooks/pre-push"

cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Refuse to push a ref whose history contains personal data. Checks EVERY ref
# being pushed, tags included — a tag pointing into pre-rewrite history is
# exactly how ~5,700 personal prompts reached the public repo in Aug 2026.
set -uo pipefail
root=$(git rev-parse --show-toplevel)
checker="$root/scripts/git/check_history_clean.sh"
[ -x "$checker" ] || exit 0            # checker removed ⇒ do not block the user

refs=()
while read -r _local_ref local_sha _remote_ref _remote_sha; do
  # all-zero local sha = a deletion; nothing to inspect
  case "$local_sha" in *[!0]*) refs+=("$local_sha") ;; esac
done

[ ${#refs[@]} -eq 0 ] && exit 0

if ! "$checker" "${refs[@]}"; then
  echo
  echo "pre-push blocked. Override only if you are certain:  git push --no-verify"
  exit 1
fi
exit 0
HOOK

chmod +x "$hook"
chmod +x "$root/scripts/git/check_history_clean.sh"
echo "installed: $hook"
