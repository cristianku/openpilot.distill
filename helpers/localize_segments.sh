#!/usr/bin/env bash
# Run only the localizer on segments already in data/ (writes localizer.safetensors
# and frame_info.safetensors next to rlog.zst).
#
# Usage: helpers/localize_segments.sh <segment> [...]
#   helpers/localize_segments.sh 0000011c8152ba83cf01                       (nome cartella in data/)
#   helpers/localize_segments.sh 6616faac453a3064/0000011c--8152ba83cf/1    (formato connect)
#   helpers/localize_segments.sh 0000011c--8152ba83cf 1 2 3
#   helpers/localize_segments.sh 0000011c--8152ba83cf 2-9
#   helpers/localize_segments.sh --missing    (tutte le cartelle in data/ senza localizer.safetensors)
#
# Segments recorded today fail on the GNSS download (ionosphere maps come out ~1 day later):
# rerun them tomorrow.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"

die() { echo "ERROR: $*" >&2; exit 1; }

names=""
route=""
missing=0
while [ $# -gt 0 ]; do
  if [ "$1" = "--missing" ]; then
    missing=1
  elif [[ "$1" =~ ^([0-9a-f]+/)?([0-9a-f]+--[0-9a-f]+)/([0-9]+)$ ]]; then
    # connect format: <dongle_id>/<route>/<segment>
    names="$names $(echo "${BASH_REMATCH[2]}" | tr -d '-')$(printf %02d "${BASH_REMATCH[3]}")"
  elif [[ "$1" =~ ^[0-9a-f]+--[0-9a-f]+$ ]]; then
    route="$(echo "$1" | tr -d '-')"
  elif [ -n "$route" ] && [[ "$1" =~ ^[0-9]+-[0-9]+$ ]]; then
    first="${1%-*}"
    last="${1#*-}"
    for ((s = first; s <= last; s++)); do
      names="$names $route$(printf %02d "$s")"
    done
  elif [ -n "$route" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    names="$names $route$(printf %02d "$1")"
  elif [[ "$1" =~ ^[0-9a-f]+$ ]]; then
    names="$names $1"
  else
    die "segmento non valido: $1"
  fi
  shift
done

if [ "$missing" -eq 1 ]; then
  for dir in "$DATA_DIR"/*/; do
    [ -f "$dir/rlog.zst" ] || continue
    if [ ! -f "$dir/localizer.safetensors" ]; then
      names="$names $(basename "$dir")"
    fi
  done
fi

[ -n "$names" ] || die "nessun segmento da localizzare"

failed=""
for name in $names; do
  echo "== $name"
  if [ ! -f "$DATA_DIR/$name/rlog.zst" ]; then
    echo "manca $DATA_DIR/$name/rlog.zst"
    failed="$failed $name"
    continue
  fi
  if ! (cd "$ROOT" && uv run python -m localizer.run --segment "$name"); then
    failed="$failed $name"
  fi
done

if [ -n "$failed" ]; then
  echo "Falliti:$failed"
  exit 1
fi
echo "Fatto."
