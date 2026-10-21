#!/usr/bin/env bash
# avg miss of one model on the held-out segments, the ones training never sees
# (python -m evaluation.holdout).
#
# Usage: helpers/eval_holdout.sh --model=<modello> [--segments=heldout|all|<seg>,<seg>] [--rate=5|20]
#          [--device=cuda|cpu] [--force]
#   helpers/eval_holdout.sh --model=stock
#   helpers/eval_holdout.sh --model=run:2026-10-21_1544
#
# Modelli: stock (il CD210 originale del comma), teacher, logged (il piano che girava davvero in
#   auto, dal rlog), run:<cartella in outputs/supervised>, hf:<nome su Hugging Face>.
# --segments: heldout (default) = quelli che supervised/dataset.py tiene fuori dal training.
# --rate: frame valutati al secondo, 5 (default) o 20.
# --force: ricalcola anche se outputs/eval/ ha gia' il risultato per lo stesso modello.
# Output: outputs/eval/<modello>.json; alla fine stampa OUTPUT_JSON=...
# Con --device=cuda (default) prende il lock della GPU condiviso con training e speedlimits.
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

[ "$has_model" = "1" ] || die "manca --model=<modello> (es. stock o run:2026-10-21_1544)"
if [ "$device" != "cuda" ] && [ "$device" != "cpu" ]; then
  die "--device vuole cuda o cpu, non '$device'"
fi

distill_load_env
if [ "$device" = "cuda" ]; then
  distill_gpu_lock_or_die
fi

export PYTHONUNBUFFERED=1
cd "$ROOT"
distill_uv_run python -m evaluation.holdout "$@"
