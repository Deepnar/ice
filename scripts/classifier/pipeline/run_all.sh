#!/usr/bin/env bash
# B1 — the whole labeling → training run, in the only order 24 GB allows.
#
# Each labeler is a full pass over the corpus and they cannot overlap: one model
# occupies the card at a time. At ~2.3 rows/s over ~38k rows that is roughly
# 4–5 hours per labeler, so this is an overnight job. Everything is resumable by
# row id — if it dies at row 30,000, re-running skips what is already on disk.
#
#   ./run_all.sh              full run
#   ./run_all.sh --resume     same thing (resume is always on; kept for clarity)
#
# Logs land in logs/b1_<stage>.log. Progress: tail -f logs/b1_label_a.log
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

step() {
  local name="$1"; shift
  local log="$LOGS/b1_${name}.log"
  echo "=== $(date '+%H:%M:%S')  $name  ==="
  if ! uv run python "$@" >> "$log" 2>&1; then
    echo "!!! $name FAILED — see $log"
    tail -20 "$log"
    exit 1
  fi
  echo "    done ($(date '+%H:%M:%S')) → $log"
}

# The GPU must be free: Ollama pins the chat model resident (UNTIL: Forever) and
# the server will refuse to start behind it. It reloads on the next chat request.
if command -v ollama >/dev/null 2>&1; then
  ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
    [ -n "$m" ] && echo "unloading ollama model $m" && ollama stop "$m"
  done
fi

step label_a  label.py --labeler A     # qwen3-14b  (~6.2 h)
step label_b  label.py --labeler B     # gemma-4-26b-a4b (~4.7 h)
step merge_1  label.py --merge
step label_c  label.py --labeler C --tiebreak-only   # gpt-oss-20b (~4 h)
step merge_2  label.py --merge

# Synth runs HERE, not earlier. A gap measured from one labeler's pass is a
# guess: agreement keeps the intersection, so a single pass OVERSTATES every
# count, and the tiebreak can move them again. Generating against a provisional
# number produces the wrong amount of the wrong thing.
#
# The cost of waiting is small — both labelers resume by id, so the two re-label
# steps below only touch the newly generated rows (minutes, not another pass).
# If `gaps: {}` prints, real data already cleared every floor and the three
# steps are no-ops.
step synth      synth.py --per-label 300
step relabel_a  label.py --labeler A
step relabel_b  label.py --labeler B
step merge_3    label.py --merge

step build    build.py
step train    train.py
step evaluate evaluate.py --candidate "$ROOT/models/classifier/ice_classifier_v4_schema2.pt"

echo
echo "=== $(date '+%H:%M:%S')  RUN COMPLETE ==="
echo "Human steps remain (they are the point, not an afterthought):"
echo "  1. review data/labeled/v2/review_queue.jsonl   (rows all three models split on)"
echo "  2. audit  data/labeled/v2/audit_sample.jsonl   (5% of AGREED rows — shared bias"
echo "     is invisible to the disagreement queue by construction)"
echo "  3. re-run: build.py && train.py && evaluate.py   to fold those decisions in"
echo "  4. promote.py --candidate ... --yes             swaps the live classifier"
