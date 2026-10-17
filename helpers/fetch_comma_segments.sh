#!/usr/bin/env bash
# Copy selected segments of a route straight from the comma device into data/,
# renamed to the hex folder name the dataset expects, then run the localizer on each.
#
# Usage: helpers/fetch_comma_segments.sh [--host=<ip|nome>] <route> --segments=<lista> [--no-localizer]
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --segments=1,2,3,4
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --segments=1,2,5-9
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf/1   (formato connect)
#   helpers/fetch_comma_segments.sh --host 192.168.88.29 6616faac453a3064/0000011c--8152ba83cf/1
#   helpers/fetch_comma_segments.sh 0000011c--8152ba83cf 0 1 2
#   helpers/fetch_comma_segments.sh 0000011c--8152ba83cf 2-9
#
# Folder name: route without dashes + segment number on two digits
#   0000011c--8152ba83cf--3  ->  data/0000011c8152ba83cf03
# (supervised/dataset.py does int(segment, 16): dashes would crash it.)
#
# The comma only accepts the keys registered on GitHub: connect to this machine
# with agent forwarding (ssh -A) so the Mac's key is used.
#
# --host: IP or ssh name of the comma; "comma@" is added when no user is given.
# Env: COMMA_HOST (default comma@192.168.88.29), overridden by --host

set -euo pipefail

COMMA_HOST="${COMMA_HOST:-comma@192.168.88.29}"
REMOTE_DIR="/data/media/0/realdata"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"
FILES="rlog.zst fcamera.hevc ecamera.hevc"

die() { echo "ERROR: $*" >&2; exit 1; }

route=""
segments=""
run_localizer=1

# Adds one segment item to $segments: a number (3) or a range (2-9).
add_segment() {
  if [[ "$1" =~ ^[0-9]+-[0-9]+$ ]]; then
    first="${1%-*}"
    last="${1#*-}"
    for ((s = first; s <= last; s++)); do
      segments="$segments $s"
    done
  elif [[ "$1" =~ ^[0-9]+$ ]]; then
    segments="$segments $1"
  else
    die "segmento non valido: $1"
  fi
}

# Adds a comma separated list (1,2,5-9) to $segments.
add_segment_list() {
  [ -n "$1" ] || die "--segments vuole una lista (es. --segments=1,2,3)"
  IFS=',' read -r -a items <<< "$1"
  for item in "${items[@]}"; do
    add_segment "$item"
  done
}

while [ $# -gt 0 ]; do
  if [ "$1" = "--no-localizer" ]; then
    run_localizer=0
  elif [ "$1" = "--host" ]; then
    shift
    [ -n "${1:-}" ] || die "--host vuole un IP o un nome"
    COMMA_HOST="$1"
  elif [[ "$1" == --host=* ]]; then
    COMMA_HOST="${1#--host=}"
    [ -n "$COMMA_HOST" ] || die "--host vuole un IP o un nome"
  elif [ "$1" = "--segments" ]; then
    shift
    add_segment_list "${1:-}"
  elif [[ "$1" == --segments=* ]]; then
    add_segment_list "${1#--segments=}"
  elif [ -z "$route" ] && [[ "$1" =~ ^([0-9a-f]+/)?([0-9a-f]+--[0-9a-f]+)/([0-9]+)$ ]]; then
    # connect format with segment: <dongle_id>/<route>/<segment>
    route="${BASH_REMATCH[2]}"
    segments="$segments ${BASH_REMATCH[3]}"
  elif [ -z "$route" ] && [[ "$1" =~ ^([0-9a-f]+/)?([0-9a-f]+--[0-9a-f]+)$ ]]; then
    # route alone, with or without the dongle id: <dongle_id>/<route>
    route="${BASH_REMATCH[2]}"
  elif [ -z "$route" ]; then
    die "route non valida: $1 (es. 6616faac453a3064/0000011c--8152ba83cf)"
  else
    add_segment "$1"
  fi
  shift
done

# Without an explicit user the comma login user is "comma".
if [[ "$COMMA_HOST" != *@* ]]; then
  COMMA_HOST="comma@$COMMA_HOST"
fi

[ -n "$route" ] || die "manca la route (es. 0000011c--8152ba83cf)"
[ -n "$segments" ] || die "mancano i segmenti (es. --segments=1,2,3)"

ssh -o ConnectTimeout=8 "$COMMA_HOST" true \
  || die "comma '$COMMA_HOST' non raggiungibile via ssh (hai fatto ssh -A?)"

route_hex="$(echo "$route" | tr -d '-')"
for s in $segments; do
  src="$REMOTE_DIR/$route--$s"
  name="$route_hex$(printf %02d "$s")"
  dst="$DATA_DIR/$name"
  echo "== $route--$s -> data/$name"

  for f in $FILES; do
    if ! ssh "$COMMA_HOST" "test -f $src/$f"; then
      die "$src/$f non esiste sul comma"
    fi
  done

  mkdir -p "$dst"
  for f in $FILES; do
    rsync -avP "$COMMA_HOST:$src/$f" "$dst/"
  done

  if [ "$run_localizer" -eq 1 ]; then
    (cd "$ROOT" && uv run python -m localizer.run --segment "$name")
  fi
done

echo "Fatto."
