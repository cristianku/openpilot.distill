#!/usr/bin/env bash
# Copy selected segments of a route straight from the comma device into data/,
# renamed to the hex folder name the dataset expects; with --localizer it also runs the localizer on each.
#
# Usage: helpers/fetch_comma_segments.sh [--host=<ip|nome>] [--since=<AAAA-MM-GG>] [--localizer]
#        helpers/fetch_comma_segments.sh [--host=<ip|nome>] <route> --segments=<lista> [--localizer]
#        helpers/fetch_comma_segments.sh [--host=<ip|nome>] <route> --skip=<lista> [--localizer]
#        helpers/fetch_comma_segments.sh [--host=<ip|nome>] <route> [--strip-ends] [--localizer]
#   helpers/fetch_comma_segments.sh                                         (automatico, vedi sotto)
#   helpers/fetch_comma_segments.sh --since=2026-10-25
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf   (= --strip-ends)
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
# Without a route (automatic): every route on the comma with a segment written on or
# after --since (default 2026-10-20, date of the segment folder on the comma) is checked
# for dashcam mode (carParams.dashcamOnly in the qlog of its first segment, via
# helpers/dashcam_mode.py); the dashcam ones are downloaded with --strip-ends.
# Already downloaded files are skipped by rsync, so it can be run again after every drive.
#
# --skip: every segment of the route found on the comma, except the listed ones.
#         The list takes numbers, ranges (2-5), "last" (the highest segment, the arrival)
#         and "last-N" (N segments before the last one).
# --strip-ends: same as --skip=0,1,last-1,last (first two and last two). No segment list needed.
#               It is the default when neither segments nor --skip are given.
# --localizer: also run the localizer on each segment (off by default: it needs the
#              GNSS ionex maps of the drive day, published on gnss-data about a day later).
# --since: automatic mode only, first day to look at (AAAA-MM-GG).
# -h, --help: this help.
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
run_localizer=0
skip=""
skip_mode=0
since="2026-10-20"

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

# Fills $segments with the segments of $route found on the comma, minus the ones in $skip.
# Returns 1 (with a message) when the route is not there or nothing is left.
select_segments() {
  found="$(ssh "$COMMA_HOST" "ls -d $REMOTE_DIR/$route--* 2>/dev/null || true" | sed 's/.*--//' | sort -n)"
  if [ -z "$found" ]; then
    echo "nessun segmento di $route sul comma" >&2
    return 1
  fi

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

  if [ -z "$segments" ]; then
    echo "$route: saltando$skip_numbers non resta nessun segmento" >&2
    return 1
  fi
  echo "$route sul comma: $(echo $found). Salto:$skip_numbers. Scarico:$segments"
}

# Copies $segments of $route into data/, then runs the localizer when asked.
download_segments() {
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
}

# Automatic mode: finds the routes since $since recorded in dashcam mode and puts
# them in $dashcam_routes.
find_dashcam_routes() {
  # One line per segment folder: "2026-10-20 14:03:22.123456789 +0000 00000123--8fe1b32468--5"
  listing="$(ssh "$COMMA_HOST" "cd $REMOTE_DIR && stat -c '%y %n' *--*--* 2>/dev/null || true")"

  # Routes with at least one segment written on or after $since.
  recent_routes=""
  while read -r day hour zone dir; do
    if [ -z "$dir" ]; then
      continue
    fi
    if [[ "$day" < "$since" ]]; then
      continue
    fi
    r="${dir%--*}"
    if [[ " $recent_routes " != *" $r "* ]]; then
      recent_routes="$recent_routes $r"
    fi
  done <<< "$listing"

  dashcam_routes=""
  if [ -z "$recent_routes" ]; then
    echo "Nessuna route sul comma dal $since."
    return 0
  fi
  echo "Route sul comma dal $since:$recent_routes"

  qlog_dir="$(mktemp -d)"
  trap 'rm -rf "$qlog_dir"' EXIT

  # The qlog of the first segment still on the comma, one per route.
  qlogs=""
  for r in $recent_routes; do
    first_segment=""
    while read -r day hour zone dir; do
      if [ "${dir%--*}" = "$r" ]; then
        s="${dir##*--}"
        if [ -z "$first_segment" ] || [ "$s" -lt "$first_segment" ]; then
          first_segment="$s"
        fi
      fi
    done <<< "$listing"

    if rsync -a "$COMMA_HOST:$REMOTE_DIR/$r--$first_segment/qlog.zst" "$qlog_dir/$r.qlog.zst"; then
      qlogs="$qlogs $qlog_dir/$r.qlog.zst"
    else
      echo "$r: qlog del segmento $first_segment non copiato, salto la route" >&2
    fi
  done

  if [ -z "$qlogs" ]; then
    return 0
  fi

  # Lines "<qlog_dir>/<route>.qlog.zst dashcam|normal|unknown".
  states="$(cd "$ROOT" && uv run python -m helpers.dashcam_mode $qlogs)"
  while read -r qlog state; do
    r="$(basename "$qlog" .qlog.zst)"
    echo "  $r: $state"
    if [ "$state" = "dashcam" ]; then
      dashcam_routes="$dashcam_routes $r"
    fi
  done <<< "$states"
}

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--since" ]; then
    shift
    since="${1:-}"
    [[ "$since" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--since vuole una data AAAA-MM-GG (es. --since=2026-10-20)"
  elif [[ "$1" == --since=* ]]; then
    since="${1#--since=}"
    [[ "$since" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--since vuole una data AAAA-MM-GG (es. --since=2026-10-20)"
  elif [ "$1" = "--localizer" ]; then
    run_localizer=1
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

auto_mode=0
if [ -z "$route" ]; then
  if [ -n "$segments" ] || [ "$skip_mode" -eq 1 ]; then
    die "manca la route (es. 0000011c--8152ba83cf)"
  fi
  auto_mode=1
elif [ "$skip_mode" -eq 1 ]; then
  [ -z "$segments" ] || die "--skip/--strip-ends non va insieme a una lista di segmenti"
elif [ -z "$segments" ]; then
  # No segments and no --skip: the default is --strip-ends.
  add_skip_list "0,1,last-1,last"
fi

ssh -o ConnectTimeout=8 "$COMMA_HOST" true \
  || die "comma '$COMMA_HOST' non raggiungibile via ssh (hai fatto ssh -A?)"

if [ "$auto_mode" -eq 1 ]; then
  find_dashcam_routes
  if [ -z "$dashcam_routes" ]; then
    echo "Nessuna route in dashcam mode da scaricare."
    exit 0
  fi
  echo "In dashcam mode:$dashcam_routes"

  for r in $dashcam_routes; do
    route="$r"
    segments=""
    skip=""
    add_skip_list "0,1,last-1,last"
    if select_segments; then
      download_segments
    fi
  done
else
  # --skip / --strip-ends: read the segments on the comma, then drop the skipped ones.
  if [ "$skip_mode" -eq 1 ]; then
    select_segments || exit 1
  fi
  download_segments
fi

echo "Fatto."
