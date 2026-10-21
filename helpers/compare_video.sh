#!/usr/bin/env bash
# Comparison video of two models on one recorded segment, like the Route42 posts
# (python -m evaluation.compare): the reference on top (white line), the student below (green
# line), each with its "avg miss" on the stretch shown.
#
# Usage: helpers/compare_video.sh --segment=<segmento> --student=<modello> [--reference=<modello>]
#          [--start=<s>] [--duration=<s>] [--speed=0.25|0.5|1] [--inset|--no-inset] [--show-gt]
#          [--device=cuda|cpu]
#   helpers/compare_video.sh --segment=00000126f84ad127f010 --student=run:2026-10-21_1544
#   helpers/compare_video.sh --segment=00000126f84ad127f010 --reference=teacher \
#     --student=hf:2026_10_21_1544 --start=20 --duration=15 --speed=0.25 --show-gt
#
# Modelli: stock (il CD210 originale del comma, default del riferimento), teacher,
#   logged (il piano che girava davvero in auto, dal rlog), run:<cartella in outputs/supervised>,
#   hf:<nome su Hugging Face>.
# --start: secondi dall'inizio del segmento (minimo 5.2: prima i modelli si riscaldano), default 10.
# --duration: secondi di video (default 15). --speed: 0.5 = rallentato 2 volte (default).
# --show-gt: anche il percorso reale (giallo). --no-inset: senza la vista dall'alto.
# Output: outputs/compare/<id>.mp4 e <id>.json; alla fine stampa OUTPUT_MP4=... e OUTPUT_JSON=...
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
has_segment=0
has_student=0
expect_device=0

# The options go to evaluation.compare as they are (it validates them); here only what decides
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
  elif [[ "$argument" == --segment=* ]] || [ "$argument" = "--segment" ]; then
    has_segment=1
  elif [[ "$argument" == --student=* ]] || [ "$argument" = "--student" ]; then
    has_student=1
  fi
done

[ "$has_segment" = "1" ] || die "manca --segment=<segmento> (una cartella di data/)"
[ "$has_student" = "1" ] || die "manca --student=<modello> (es. run:2026-10-21_1544)"
if [ "$device" != "cuda" ] && [ "$device" != "cpu" ]; then
  die "--device vuole cuda o cpu, non '$device'"
fi

distill_load_env
if [ "$device" = "cuda" ]; then
  distill_gpu_lock_or_die
fi

export PYTHONUNBUFFERED=1
cd "$ROOT"
distill_uv_run python -m evaluation.compare "$@"
