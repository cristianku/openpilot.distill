#!/usr/bin/env bash
# Run only the localizer on segments already in data/ (writes localizer.safetensors
# and frame_info.safetensors next to rlog.zst).
#
# Usage: helpers/localize_segments.sh [<segment> ...]
#   helpers/localize_segments.sh                                            (da fare + da raffinare)
#   helpers/localize_segments.sh --status                                   (solo stato, non lancia il localizer)
#   helpers/localize_segments.sh 0000011c8152ba83cf01                       (nome cartella in data/)
#   helpers/localize_segments.sh 6616faac453a3064/0000011c--8152ba83cf/1    (formato connect)
#   helpers/localize_segments.sh 0000011c--8152ba83cf 1 2 3
#   helpers/localize_segments.sh 0000011c--8152ba83cf 2-9
#
# State of a segment folder:
#   done     localizer.safetensors and frame_info.safetensors both present (written only on success)
#   refine   done, but a better ionosphere map is out now than the one used (see below)
#   failed   localizer.failed present (date + last lines of the error); removed on the next success
#   todo     neither
# Every run also leaves the full output in localizer.log.
#
# Ionosphere map: the localizer writes the one it used in localizer.ionex ("2026/293 P1D").
# P1D (prediction) is the only one for a drive of today, RAP comes out about one day later,
# FIN about two weeks later (details in localizer/ionex.py). Every run without arguments
# redoes the segments for which a better map came out: P1D -> RAP -> FIN, one step at a time.
# Segments done before localizer.ionex existed have no record and are left alone.
#
# Unusable segments: when the localizer fails because of the data itself (car never above
# 18 km/h, no GPS fix, fewer than four raw GPS satellites in every epoch, no calibration yet,
# incomplete segment: exit code 3, see localizer/errors.py)
# the folder is moved to data_scartati/, so the training never sees it. The reason stays in
# its localizer.failed. Network or GNSS map problems stay in data/ as "failed" and are retried.
# To take a segment back: mv data_scartati/<name> data/

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"
DISCARD_DIR="$ROOT/data_scartati"
# Exit code of localizer.run for a segment unusable because of its data (localizer/errors.py).
UNUSABLE_SEGMENT_EXIT_CODE=3

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

# Fills $ionex_lines with "<folder> <map used> <better map out now, or ->" for every
# done segment with a localizer.ionex (one call to localizer/ionex.py for all of them).
read_ionex_status() {
  dirs=""
  for name in $(all_segments); do
    if [ "$(segment_state "$name")" = "done" ] && [ -f "$DATA_DIR/$name/localizer.ionex" ]; then
      dirs="$dirs $DATA_DIR/$name"
    fi
  done
  ionex_lines=""
  if [ -n "$dirs" ]; then
    ionex_lines="$(cd "$ROOT" && uv run python -m localizer.ionex $dirs)"
  fi
}

# Map used and better map out for one segment, from $ionex_lines: sets $ionex_used, $ionex_better.
ionex_of() {
  ionex_used="?"
  ionex_better="-"
  while read -r dir used better; do
    if [ "${dir##*/}" = "$1" ]; then
      ionex_used="$used"
      ionex_better="$better"
    fi
  done <<< "$ionex_lines"
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
  read_ionex_status
  for name in $(all_segments); do
    state="$(segment_state "$name")"
    if [ "$state" = "failed" ]; then
      echo "$state  $name  ($(head -1 "$DATA_DIR/$name/localizer.failed"))"
    elif [ "$state" = "done" ]; then
      ionex_of "$name"
      if [ "$ionex_better" != "-" ]; then
        echo "refine  $name  (mappa $ionex_used, uscita $ionex_better)"
      else
        echo "done  $name  (mappa $ionex_used)"
      fi
    else
      echo "$state  $name"
    fi
  done
  for dir in "$DISCARD_DIR"/*/; do
    if [ ! -d "$dir" ]; then
      continue
    fi
    echo "scartato  $(basename "$dir")  ($(head -1 "$dir/localizer.failed" 2>/dev/null))"
  done
  exit 0
fi

# No segment given: every folder in data/ that is not done yet (todo or failed),
# plus the done ones for which a better ionosphere map came out (refine).
if [ -z "$names" ]; then
  read_ionex_status
  for name in $(all_segments); do
    if [ "$(segment_state "$name")" != "done" ]; then
      names="$names $name"
    else
      ionex_of "$name"
      if [ "$ionex_better" != "-" ]; then
        echo "$name: rifaccio con la mappa $ionex_better (prima $ionex_used)"
        names="$names $name"
      fi
    fi
  done
  [ -n "$names" ] || { echo "Niente da fare: tutti i segmenti sono gia' localizzati con la mappa migliore uscita."; exit 0; }
fi

ok=""
failed=""
discarded=""
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
    status=$?
    {
      echo "$(date '+%Y-%m-%d %H:%M') $(tail -1 "$dir/localizer.log")"
      tail -20 "$dir/localizer.log"
    } > "$dir/localizer.failed"

    # Unusable because of its data: out of data/. A segment that already has outputs
    # (failed refine) is never moved: its old outputs are still good.
    if [ "$status" -eq "$UNUSABLE_SEGMENT_EXIT_CODE" ] && [ ! -f "$dir/localizer.safetensors" ]; then
      mkdir -p "$DISCARD_DIR"
      if [ -e "$DISCARD_DIR/$name" ]; then
        # Already discarded once (downloaded again): keep both, never overwrite.
        mv "$dir" "$DISCARD_DIR/$name.$(date '+%Y%m%d%H%M%S')"
      else
        mv "$dir" "$DISCARD_DIR/$name"
      fi
      echo "$name: dati inutilizzabili, spostato in data_scartati/"
      discarded="$discarded $name"
    else
      failed="$failed $name"
    fi
  fi
done

echo "OK:${ok:- nessuno}"
if [ -n "$discarded" ]; then
  echo "Scartati (in data_scartati/):$discarded"
fi
if [ -n "$failed" ]; then
  echo "Falliti:$failed"
  exit 1
fi
