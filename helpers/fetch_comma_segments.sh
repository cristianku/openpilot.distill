#!/usr/bin/env bash
# Copy selected segments of a route straight from the comma device into data/,
# renamed to the hex folder name the dataset expects, then run the localizer on each.
#
# Usage: helpers/fetch_comma_segments.sh [--host=<ip|nome>] <route> --segments=<lista> [--no-localizer]
#        helpers/fetch_comma_segments.sh [--host=<ip|nome>] <route> --skip=<lista> [--no-localizer]
#        helpers/fetch_comma_segments.sh [--host=<ip|nome>] <route> --strip-ends [--no-localizer]
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --skip=0,1,2,last
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --skip 0,1,14
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --strip-ends
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
# --skip: every segment of the route found on the comma, except the listed ones.
#         The list takes numbers, ranges (2-5), "last" (the highest segment, the arrival)
#         and "last-N" (N segments before the last one).
# --strip-ends: same as --skip=0,1,last-1,last (first two and last two). No segment list needed.
# -h, --help: this help (also shown when called without arguments).
# --host: IP or ssh name of the comma; "comma@" is added when no user is given.
# Env: COMMA_HOST (default "comma", the Host in ~/.ssh/config), overridden by --host

set -euo pipefail

COMMA_HOST="${COMMA_HOST:-comma}"
REMOTE_DIR="/data/media/0/realdata"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"
FILES="rlog.zst fcamera.hevc ecamera.hevc"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

route=""
segments=""
run_localizer=1
skip=""
skip_mode=0

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

# Adds a comma separated skip list (0,1,14 or 0,1,2,last or 2-5) to $skip.
# "last" stays a word here: the highest segment is known only after asking the comma.
add_skip_list() {
  [ -n "$1" ] || die "--skip vuole una lista (es. --skip=0,1,last)"
  IFS=',' read -r -a items <<< "$1"
  for item in "${items[@]}"; do
    if [ "$item" = "last" ]; then
      skip="$skip last"
    elif [[ "$item" =~ ^last-[0-9]+$ ]]; then
      skip="$skip $item"
    elif [[ "$item" =~ ^[0-9]+-[0-9]+$ ]]; then
      skip_first="${item%-*}"
      skip_last="${item#*-}"
      for ((s = skip_first; s <= skip_last; s++)); do
        skip="$skip $s"
      done
    elif [[ "$item" =~ ^[0-9]+$ ]]; then
      skip="$skip $item"
    else
      die "segmento da saltare non valido: $item"
    fi
  done
  skip_mode=1
}

# --host without an explicit user: the comma login user is "comma".
# The default "comma" is left alone, user and IP come from ~/.ssh/config.
add_comma_user() {
  if [[ "$COMMA_HOST" != *@* ]]; then
    COMMA_HOST="comma@$COMMA_HOST"
  fi
}

if [ $# -eq 0 ]; then
  usage
fi

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--no-localizer" ]; then
    run_localizer=0
  elif [ "$1" = "--strip-ends" ]; then
    add_skip_list "0,1,last-1,last"
  elif [ "$1" = "--skip" ]; then
    shift
    add_skip_list "${1:-}"
  elif [[ "$1" == --skip=* ]]; then
    add_skip_list "${1#--skip=}"
  elif [ "$1" = "--host" ]; then
    shift
    [ -n "${1:-}" ] || die "--host vuole un IP o un nome"
    COMMA_HOST="$1"
    add_comma_user
  elif [[ "$1" == --host=* ]]; then
    COMMA_HOST="${1#--host=}"
    [ -n "$COMMA_HOST" ] || die "--host vuole un IP o un nome"
    add_comma_user
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

[ -n "$route" ] || die "manca la route (es. 0000011c--8152ba83cf)"
if [ "$skip_mode" -eq 1 ]; then
  [ -z "$segments" ] || die "--skip/--strip-ends non va insieme a una lista di segmenti"
else
  [ -n "$segments" ] || die "mancano i segmenti (es. --segments=1,2,3, --skip=0,last oppure --strip-ends)"
fi

ssh -o ConnectTimeout=8 "$COMMA_HOST" true \
  || die "comma '$COMMA_HOST' non raggiungibile via ssh (hai fatto ssh -A?)"

# --skip / --strip-ends: read the segments on the comma, then drop the skipped ones.
if [ "$skip_mode" -eq 1 ]; then
  found="$(ssh "$COMMA_HOST" "ls -d $REMOTE_DIR/$route--* 2>/dev/null || true" | sed 's/.*--//' | sort -n)"
  [ -n "$found" ] || die "nessun segmento di $route sul comma"

  last_segment=""
  for s in $found; do
    last_segment="$s"
  done

  # "last" becomes the highest segment found on the comma, "last-N" that minus N.
  skip_numbers=""
  for k in $skip; do
    if [ "$k" = "last" ]; then
      skip_numbers="$skip_numbers $last_segment"
    elif [[ "$k" =~ ^last-[0-9]+$ ]]; then
      skip_numbers="$skip_numbers $((last_segment - ${k#last-}))"
    else
      skip_numbers="$skip_numbers $k"
    fi
  done

  for s in $found; do
    skipped=0
    for k in $skip_numbers; do
      if [ "$s" -eq "$k" ]; then
        skipped=1
      fi
    done
    if [ "$skipped" -eq 0 ]; then
      segments="$segments $s"
    fi
  done

  [ -n "$segments" ] || die "saltando$skip_numbers non resta nessun segmento"
  echo "Sul comma: $(echo $found). Salto:$skip_numbers. Scarico:$segments"
fi

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
