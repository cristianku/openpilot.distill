#!/usr/bin/env bash
# avg miss of one model on the held-out segments, the ones training never sees
# (python -m evaluation.holdout).
#
# Usage: helpers/eval_holdout.sh --model=<model> [--segments=heldout|all|<seg>,<seg>] [--rate=5|20]
#          [--device=cuda|cpu] [--force]
#   helpers/eval_holdout.sh --model=stock
#   helpers/eval_holdout.sh --model=run:2026-10-21_1544
#
# Models: stock (the original CD210 of the comma), teacher, logged (the plan that really ran in
#   the car, from the rlog), run:<folder in outputs/supervised>, hf:<name on Hugging Face>.
# --segments: heldout (default) = the ones supervised/dataset.py keeps out of the training.
# --rate: frames evaluated per second, 5 (default) or 20.
# --force: recomputes even when outputs/eval/ already has the result for the same model.
# Output: outputs/eval/<model>.json; at the end it prints OUTPUT_JSON=...
# With --device=cuda (default) it takes the GPU lock shared with training and speedlimits.
#
# -h, --help: this help.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/helpers/common.sh"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

device="cuda"
has_model=0
expect_device=0

# The options go to evaluation.holdout as they are (it validates them); here only what decides
# the GPU lock and the obvious mistakes.
for argument in "$@"; do
  if [ "$expect_device" = "1" ]; then
    device="$argument"
    expect_device=0
  elif [ "$argument" = "-h" ] || [ "$argument" = "--help" ]; then
    usage
  elif [ "$argument" = "--device" ]; then
    expect_device=1
  elif [[ "$argument" == --device=* ]]; then
    device="${argument#--device=}"
  elif [[ "$argument" == --model=* ]] || [ "$argument" = "--model" ]; then
    has_model=1
  fi
done

[ "$has_model" = "1" ] || die "missing --model=<model> (e.g. stock or run:2026-10-21_1544)"
if [ "$device" != "cuda" ] && [ "$device" != "cpu" ]; then
  die "--device needs cuda or cpu, not '$device'"
fi

distill_load_env
if [ "$device" = "cuda" ]; then
  distill_gpu_lock_or_die
fi

export PYTHONUNBUFFERED=1
cd "$ROOT"
distill_uv_run python -m evaluation.holdout "$@"
