#!/usr/bin/env bash
# Follow the progress of a training: tail -f of its train.log.
#
# Usage: helpers/train_progress.sh [<run folder|train.log>]
#   helpers/train_progress.sh                                   (the latest outputs/supervised/*/train.log)
#   helpers/train_progress.sh outputs/supervised/2026-10-21_0839
#
# The latest run is the train.log written most recently, so it is the one still training.
# Ctrl+C stops only the tail, not the training.
#
# -h, --help: this help.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

input=""

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ -z "$input" ]; then
    input="$1"
  else
    die "extra argument: $1"
  fi
  shift
done

if [ -z "$input" ]; then
  log="$(ls -t "$ROOT"/outputs/supervised/*/train.log 2>/dev/null | head -1 || true)"
  [ -n "$log" ] || die "no outputs/supervised/*/train.log: no training has ever started"
elif [ -d "$input" ]; then
  log="$input/train.log"
else
  log="$input"
fi
[ -f "$log" ] || die "$log not found"

run_dir="$(dirname "$log")"
echo "Run: $run_dir"
if [ -f "$run_dir/supercombo.pt" ]; then
  echo "Finished: the model is in $run_dir/supercombo.pt"
else
  echo "Running (supercombo.pt comes at the end)."
fi
echo

tail -n 20 -f "$log"
