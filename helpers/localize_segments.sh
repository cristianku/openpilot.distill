#!/usr/bin/env bash
# Run only the localizer on segments already in data/ (writes localizer.safetensors
# and frame_info.safetensors next to rlog.zst).
#
# Usage: helpers/localize_segments.sh [<segment> ...]
#   helpers/localize_segments.sh                                            (tutti quelli da fare)
#   helpers/localize_segments.sh --status                                   (solo stato, non lancia niente)
#   helpers/localize_segments.sh 0000011c8152ba83cf01                       (nome cartella in data/)
#   helpers/localize_segments.sh 6616faac453a3064/0000011c--8152ba83cf/1    (formato connect)
#   helpers/localize_segments.sh 0000011c--8152ba83cf 1 2 3
#   helpers/localize_segments.sh 0000011c--8152ba83cf 2-9
#
# State of a segment folder:
#   done     localizer.safetensors and frame_info.safetensors both present (written only on success)
#   failed   localizer.failed present (date + last lines of the error); removed on the next success
#   todo     neither
# Every run also leaves the full output in localizer.log.
#
# Segments recorded today fail on the GNSS download (ionosphere maps come out ~1 day later):
# they stay "failed" and are retried by the next run without arguments.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"

die() { echo "ERROR: $*" >&2; exit 1; }

segment_state() {
  local dir="$DATA_DIR/$1"
  if [ -f "$dir/localizer.safetensors" ] && [ -f "$dir/frame_info.safetensors" ]; then
    echo "done"
  elif [ -f "$dir/localizer.failed" ]; then
    echo "failed"
  else
    echo "todo"
  fi
}

all_segments() {
  for dir in "$DATA_DIR"/*/; do
    [ -f "$dir/rlog.zst" ] || continue
    basename "$dir"
  done
}

names=""
route=""
status_only=0
while [ $# -gt 0 ]; do
  if [ "$1" = "--status" ]; then
    status_only=1
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

if [ "$status_only" -eq 1 ]; then
  for name in $(all_segments); do
    state="$(segment_state "$name")"
    if [ "$state" = "failed" ]; then
      echo "$state  $name  ($(head -1 "$DATA_DIR/$name/localizer.failed"))"
    else
      echo "$state  $name"
    fi
  done
  exit 0
fi

# No segment given: every folder in data/ that is not done yet (todo or failed).
if [ -z "$names" ]; then
  for name in $(all_segments); do
    if [ "$(segment_state "$name")" != "done" ]; then
      names="$names $name"
    fi
  done
  [ -n "$names" ] || { echo "Niente da fare: tutti i segmenti sono gia' localizzati."; exit 0; }
fi

ok=""
failed=""
for name in $names; do
  dir="$DATA_DIR/$name"
  echo "== $name"
  if [ ! -f "$dir/rlog.zst" ]; then
    echo "manca $dir/rlog.zst"
    failed="$failed $name"
    continue
  fi
  if (cd "$ROOT" && uv run python -m localizer.run --segment "$name") 2>&1 | tee "$dir/localizer.log"; then
    rm -f "$dir/localizer.failed"
    ok="$ok $name"
  else
    {
      echo "$(date '+%Y-%m-%d %H:%M') $(tail -1 "$dir/localizer.log")"
      tail -20 "$dir/localizer.log"
    } > "$dir/localizer.failed"
    failed="$failed $name"
  fi
done

echo "OK:${ok:- nessuno}"
if [ -n "$failed" ]; then
  echo "Falliti:$failed"
  exit 1
fi
