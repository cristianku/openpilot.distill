#!/usr/bin/env bash
# Copy selected segments of a route straight from the comma device into the shared folder of the
# comma videos, and link them into data/ under the hex folder name the dataset expects;
# with --localizer it also runs the localizer on each.
#
# Usage: helpers/fetch_comma_segments.sh [--host=<ip|name>] [--since=<YYYY-MM-DD>] [--localizer]
#        helpers/fetch_comma_segments.sh [--host=<ip|name>] <route> --segments=<list> [--localizer]
#        helpers/fetch_comma_segments.sh [--host=<ip|name>] <route> --skip=<list> [--localizer]
#        helpers/fetch_comma_segments.sh [--host=<ip|name>] <route> [--strip-ends] [--localizer]
#   helpers/fetch_comma_segments.sh                                         (automatic, see below)
#   helpers/fetch_comma_segments.sh --since=2026-10-25
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf   (= --strip-ends)
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --skip=0,1,2,last
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --skip 0,1,14
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --strip-ends
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --segments=1,2,3,4
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf --segments=1,2,5-9
#   helpers/fetch_comma_segments.sh 6616faac453a3064/0000011c--8152ba83cf/1   (connect format)
#   helpers/fetch_comma_segments.sh --host 192.168.88.29 6616faac453a3064/0000011c--8152ba83cf/1
#   helpers/fetch_comma_segments.sh 0000011c--8152ba83cf 0 1 2
#   helpers/fetch_comma_segments.sh 0000011c--8152ba83cf 2-9
#
# Where the files go: rlog.zst, fcamera.hevc and ecamera.hevc are downloaded once into the
# shared folder (COMMA_VIDEOS_DIR, default comma-videos next to this repo, e.g. /root/comma-videos),
# one folder per segment as on the comma; sunnypilot.speedlimits uses the same files.
# data/<folder>/ only has absolute links to them, plus what the localizer writes (never shared).
#   <COMMA_VIDEOS_DIR>/0000011c--8152ba83cf--3/fcamera.hevc        (the file)
#   data/0000011c8152ba83cf03/fcamera.hevc -> the file above       (the link)
# A file still downloaded straight into data/ (old layout) is moved into the shared folder and
# replaced by the link; if the shared folder already has a different copy it stops, nothing deleted.
# Both segment folders must be real folders: when data/<folder> or the shared one is a link
# (the two would be the same files) it stops too, nothing deleted.
#
# Folder name in data/: route without dashes + segment number on two digits
#   0000011c--8152ba83cf--3  ->  data/0000011c8152ba83cf03
# (supervised/dataset.py does int(segment, 16): dashes would crash it.)
#
# The comma only accepts the keys registered on GitHub: this machine needs one of them (the
# training server has its own), or connect to it with agent forwarding (ssh -A) from the Mac.
# ssh never asks anything (BatchMode): a missing key or an unknown host key is an error.
#
# Without a route (automatic): every route on the comma with a segment written on or
# after --since (default 2026-10-20) is checked for dashcam mode. Before fetching,
# exclude whole trips lasting less than 300 seconds (measured from the very first
# to the very last qlog, INCLUDING the first/last two segments). Then discard
# first two + last two segments and any remaining segment whose rlog contains
# carState.gearShifter=reverse. Missing/unreadable logs are skipped, never assumed safe.
# Explicit route/segment selection remains manual and does not apply these automatic filters.
# The same script lives in sunnypilot.speedlimits, but the two copies are no longer identical:
# this one loads helpers/common.sh (.env, uv extra, ssh options) and has the strip-ends fix.
# In speedlimits its data links use raw/routes/<route>/<segment>/ rather than Distill's hex
# segment folders, while both share COMMA_VIDEOS_DIR.
# Requires a Python log reader: Distill uses uv run python (helpers/common.sh distill_uv_run);
# speedlimits can use its .venv with [log-reader], NNSLR_OPENPILOT_ROOT and NNSLR_OPENDBC_ROOT.
# COMMA_FETCH_PYTHON can override the Python interpreter (speedlimits).
# Already downloaded files are skipped by rsync, so it can be run after every drive.
#
# --skip: every segment of the route found on the comma, except the listed ones.
#         The list takes numbers, ranges (2-5), "last" (the highest segment, the arrival),
#         "last-N" (N segments before the last one) and first, second, second_last
#         (the lowest, the second lowest and the second highest segment found).
# --strip-ends: skip first two and last two AVAILABLE segment folders (even with gaps). No list needed.
#               It is the default when neither segments nor --skip are given.
# --localizer: also run the localizer on each segment (off by default: helpers/localize_segments.sh
#              does it later and redoes it when a better ionosphere map comes out).
# --since: automatic mode only, first day to look at (YYYY-MM-DD).
# -h, --help: this help.
# --host: IP or ssh name of the comma; "comma@" is added when no user is given.
# Env: COMMA_HOST (default "comma", the Host in ~/.ssh/config), overridden by --host
#      COMMA_VIDEOS_DIR (default comma-videos next to this repo), shared folder of the comma files
#      Both can also be set in the repo .env (see helpers/common.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/helpers/common.sh"
distill_load_env

COMMA_HOST="${COMMA_HOST:-comma}"
REMOTE_DIR="/data/media/0/realdata"
# Distill's localizer wants flat hex folders; speedlimits' reader wants raw/routes.
if [ -d "$ROOT/src/nnslr_tools" ]; then
  DATA_LAYOUT="speedlimits"
  DATA_DIR="${NNSLR_DATA_ROOT:-$ROOT/data}/raw/routes"
else
  DATA_LAYOUT="distill"
  DATA_DIR="$ROOT/data"
fi
VIDEOS_DIR="${COMMA_VIDEOS_DIR:-$(dirname "$ROOT")/comma-videos}"
FILES="rlog.zst fcamera.hevc ecamera.hevc"

if [ -n "${COMMA_FETCH_PYTHON:-}" ]; then
  FETCH_PYTHON=("$COMMA_FETCH_PYTHON")
elif [ "$DATA_LAYOUT" = "distill" ]; then
  # A shell function works as the first word of the array: "${FETCH_PYTHON[@]}" calls it.
  FETCH_PYTHON=(distill_uv_run python)
elif [ -x "$ROOT/.venv/bin/python" ]; then
  FETCH_PYTHON=("$ROOT/.venv/bin/python")
else
  FETCH_PYTHON=(python3)
fi

inspect_log() {
  (cd "$ROOT" && "${FETCH_PYTHON[@]}" "$ROOT/helpers/comma_fetch_log.py" "$@")
}

die() { echo "ERROR: $*" >&2; exit 1; }

# Size in bytes of a file.
file_size() {
  wc -c < "$1" | tr -d ' '
}

# Old layout: $1 (in data/) still a regular file downloaded there. It goes to $2 (shared
# folder): moved when the shared folder does not have it yet, removed when the shared copy
# is identical. A shared copy that differs is left to the user: stop, nothing deleted.
# $1 and $2 being the same file (one reaches the other through a link) also stops: cmp
# would call them identical and the rm would delete the only copy.
move_old_file() {
  local old="$1"
  local shared="$2"
  if [ -L "$old" ] || [ ! -e "$old" ]; then
    return 0
  fi
  if [ ! -f "$old" ]; then
    die "$old is neither a file nor a link: check it by hand"
  fi
  if [ ! -e "$shared" ]; then
    mv "$old" "$shared"
    echo "  $(basename "$old"): moved from data/ into the shared folder"
  elif [ "$old" -ef "$shared" ]; then
    die "$old and $shared are the same file (one reaches the other through a link): nothing deleted, check it by hand"
  elif cmp -s "$old" "$shared"; then
    rm "$old"
    echo "  $(basename "$old"): identical to the one in the shared folder, removed from data/"
  else
    die "$old ($(file_size "$old") bytes) and $shared ($(file_size "$shared") bytes) differ: nothing deleted, check it by hand"
  fi
}

# Makes $2 (in data/) an absolute link to $1 (in the shared folder).
link_file() {
  local target="$1"
  local link="$2"
  if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    return 0
  fi
  if [ -L "$link" ]; then
    rm "$link"
  fi
  ln -s "$target" "$link"
}

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
    die "invalid segment: $1"
  fi
}

# Adds a comma separated list (1,2,5-9) to $segments.
add_segment_list() {
  [ -n "$1" ] || die "--segments needs a list (e.g. --segments=1,2,3)"
  IFS=',' read -r -a items <<< "$1"
  for item in "${items[@]}"; do
    add_segment "$item"
  done
}

# Adds a comma separated skip list (0,1,14 or 0,1,2,last or 2-5) to $skip.
# first, second, second_last, last and last-N stay words here: the segments they mean are
# known only after asking the comma (select_segments turns them into numbers).
add_skip_list() {
  [ -n "$1" ] || die "--skip needs a list (e.g. --skip=0,1,last)"
  IFS=',' read -r -a items <<< "$1"
  for item in "${items[@]}"; do
    if [ "$item" = "first" ] || [ "$item" = "second" ] || [ "$item" = "second_last" ] || [ "$item" = "last" ]; then
      skip="$skip $item"
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
      die "invalid segment to skip: $item"
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
  found="$(ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" "ls -d $REMOTE_DIR/$route--* 2>/dev/null || true" | sed 's/.*--//' | sort -n)"
  if [ -z "$found" ]; then
    echo "no segments of $route on the comma" >&2
    return 1
  fi

  first_segment=""
  second_segment=""
  second_last_segment=""
  last_segment=""
  for s in $found; do
    if [ -z "$first_segment" ]; then
      first_segment="$s"
    elif [ -z "$second_segment" ]; then
      second_segment="$s"
    fi
    second_last_segment="$last_segment"
    last_segment="$s"
  done

  # "last" becomes the highest segment found on the comma, "last-N" that minus N.
  skip_numbers=""
  for k in $skip; do
    if [ "$k" = "first" ]; then
      skip_numbers="$skip_numbers $first_segment"
    elif [ "$k" = "second" ]; then
      [ -z "$second_segment" ] || skip_numbers="$skip_numbers $second_segment"
    elif [ "$k" = "second_last" ]; then
      [ -z "$second_last_segment" ] || skip_numbers="$skip_numbers $second_last_segment"
    elif [ "$k" = "last" ]; then
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
    echo "$route: skipping$skip_numbers leaves no segments" >&2
    return 1
  fi
  echo "$route on the comma: $(echo $found). Skipping:$skip_numbers. Downloading:$segments"
}

# Copies $segments of $route into the shared folder, links them into data/, then runs the
# localizer when asked.
download_segments() {
  route_hex="$(echo "$route" | tr -d '-')"
  for s in $segments; do
    src="$REMOTE_DIR/$route--$s"
    shared="$VIDEOS_DIR/$route--$s"
    name="$route_hex$(printf %02d "$s")"
    if [ "$DATA_LAYOUT" = "speedlimits" ]; then
      dst="$DATA_DIR/$route/$s"
    else
      dst="$DATA_DIR/$name"
    fi
    # Discarded by localize_segments.sh (Distill only): not again.
    if [ "$DATA_LAYOUT" = "distill" ] && [ -d "$ROOT/data_scartati/$name" ]; then
      echo "== $route--$s: discarded by the localizer (data_scartati/$name), skipping"
      continue
    fi
    echo "== $route--$s -> $shared (link in $dst)"

    for f in $FILES; do
      if ! ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" "test -f $src/$f"; then
        die "$src/$f does not exist on the comma"
      fi
    done

    # A segment folder that is a link would make data/$name/<file> and the shared file the
    # same file (the localizer outputs would also end up in the shared folder): stop.
    if [ -L "$dst" ]; then
      die "$dst is a link to a folder: data/ needs a real folder with the links to the files inside. Nothing deleted, check it by hand"
    fi
    if [ -L "$shared" ]; then
      die "$shared is a link to a folder: the shared folder needs a real folder. Nothing deleted, check it by hand"
    fi

    mkdir -p "$shared" "$dst"
    for f in $FILES; do
      move_old_file "$dst/$f" "$shared/$f"
      rsync -avP -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$src/$f" "$shared/"
      link_file "$shared/$f" "$dst/$f"
    done

    if [ "$run_localizer" -eq 1 ]; then
      (cd "$ROOT" && distill_uv_run python -m localizer.run --segment "$name")
    fi
  done
}

# Automatic mode: finds the routes since $since recorded in dashcam mode and puts
# them in $dashcam_routes.
find_dashcam_routes() {
  # One line per segment folder: "2026-10-20 14:03:22.123456789 +0000 00000123--8fe1b32468--5"
  listing="$(ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" "cd $REMOTE_DIR && stat -c '%y %n' *--*--* 2>/dev/null || true")"

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
    echo "No routes on the comma since $since."
    return 0
  fi
  echo "Routes on the comma since $since:$recent_routes"

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

    if rsync -a -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$REMOTE_DIR/$r--$first_segment/qlog.zst" "$qlog_dir/$r.qlog.zst"; then
      qlogs="$qlogs $qlog_dir/$r.qlog.zst"
    else
      echo "$r: qlog of segment $first_segment not copied, skipping the route" >&2
    fi
  done

  if [ -z "$qlogs" ]; then
    return 0
  fi

  for r in $recent_routes; do
    if [ ! -f "$qlog_dir/$r.qlog.zst" ]; then
      continue
    fi
    state="$(inspect_log dashcam "$qlog_dir/$r.qlog.zst")" || state="unknown"
    echo "  $r: $state"
    if [ "$state" = "dashcam" ]; then
      dashcam_routes="$dashcam_routes $r"
    fi
  done
}

# Called in automatic mode after select_segments: $found contains ALL indices
# including the four that are removed from the training set.
route_is_long_enough() {
  local first_segment="" last_segment="" s seconds temp
  for s in $found; do
    [ -n "$first_segment" ] || first_segment="$s"
    last_segment="$s"
  done
  [ -n "$first_segment" ] || return 1

  temp="$(mktemp -d)"
  # Both endpoints are needed to measure true duration; number of folders alone
  # would wrongly accept a five-segment route ending before 5 minutes.
  if ! rsync -a -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$REMOTE_DIR/$route--$first_segment/qlog.zst" "$temp/first.zst" \
       || ! rsync -a -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$REMOTE_DIR/$route--$last_segment/qlog.zst" "$temp/last.zst"; then
    echo "$route: first/last qlog not available, trip excluded" >&2
    rm -rf "$temp"
    return 1
  fi
  seconds="$(inspect_log duration "$temp/first.zst" "$temp/last.zst")" || seconds="unknown"
  rm -rf "$temp"
  if [[ "$seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] && [ "${seconds%%.*}" -ge 300 ]; then
    echo "$route: full duration ${seconds}s (>= 300s), ok"
    return 0
  fi
  echo "$route: duration ${seconds}s (< 300s or unknown), trip excluded"
  return 1
}

# Scan full rlogs, not just qlogs: a single reverse gear event rejects the
# entire segment. We never include an unreadable segment in automatic mode.
filter_reverse_segments() {
  local s state remaining="" shared logfile
  for s in $segments; do
    shared="$VIDEOS_DIR/$route--$s"
    logfile="$shared/rlog.zst"
    if [ -L "$shared" ] || [ -L "$logfile" ]; then
      echo "$route--$s: unsafe link in the shared folder, segment excluded" >&2
      continue
    fi
    # Cache the full log once in COMMA_VIDEOS_DIR: repeated automatic fetches
    # do not transfer the same (potentially large) rlog on every run.
    if [ ! -s "$logfile" ]; then
      mkdir -p "$shared"
      if ! rsync -a -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$REMOTE_DIR/$route--$s/rlog.zst" "$shared/"; then
        echo "$route--$s: rlog not available, segment excluded" >&2
        continue
      fi
    fi
    state="$(inspect_log reverse "$logfile")" || state="unknown"
    case "$state" in
      forward) remaining="$remaining $s" ;;
      reverse) echo "$route--$s: reverse gear detected, segment excluded" ;;
      *) echo "$route--$s: reverse gear cannot be checked, segment excluded" >&2 ;;
    esac
  done
  segments="$remaining"
  [ -n "$segments" ]
}

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--since" ]; then
    shift
    since="${1:-}"
    [[ "$since" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--since needs a date YYYY-MM-DD (e.g. --since=2026-10-20)"
  elif [[ "$1" == --since=* ]]; then
    since="${1#--since=}"
    [[ "$since" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--since needs a date YYYY-MM-DD (e.g. --since=2026-10-20)"
  elif [ "$1" = "--localizer" ]; then
    run_localizer=1
  elif [ "$1" = "--no-localizer" ]; then
    run_localizer=0
  elif [ "$1" = "--strip-ends" ]; then
    add_skip_list "first,second,second_last,last"
  elif [ "$1" = "--skip" ]; then
    shift
    add_skip_list "${1:-}"
  elif [[ "$1" == --skip=* ]]; then
    add_skip_list "${1#--skip=}"
  elif [ "$1" = "--host" ]; then
    shift
    [ -n "${1:-}" ] || die "--host needs an IP or a name"
    COMMA_HOST="$1"
    add_comma_user
  elif [[ "$1" == --host=* ]]; then
    COMMA_HOST="${1#--host=}"
    [ -n "$COMMA_HOST" ] || die "--host needs an IP or a name"
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
    die "invalid route: $1 (e.g. 6616faac453a3064/0000011c--8152ba83cf)"
  else
    add_segment "$1"
  fi
  shift
done

auto_mode=0
if [ -z "$route" ]; then
  if [ -n "$segments" ] || [ "$skip_mode" -eq 1 ]; then
    die "missing route (e.g. 0000011c--8152ba83cf)"
  fi
  auto_mode=1
elif [ "$skip_mode" -eq 1 ]; then
  [ -z "$segments" ] || die "--skip/--strip-ends cannot be combined with a list of segments"
elif [ -z "$segments" ]; then
  # No segments and no --skip: the default is --strip-ends.
  add_skip_list "first,second,second_last,last"
fi

if [ "$DATA_LAYOUT" = "speedlimits" ] && [ "$run_localizer" -eq 1 ]; then
  die "--localizer is only available in openpilot.distill"
fi

# Absolute path: Distill links must keep working when a folder moves to data_scartati/.
mkdir -p "$VIDEOS_DIR" || die "cannot create the shared folder $VIDEOS_DIR (COMMA_VIDEOS_DIR)"
VIDEOS_DIR="$(cd "$VIDEOS_DIR" && pwd)"

ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" true \
  || die "comma '$COMMA_HOST' not reachable over ssh (on and online? ssh key registered? from the Mac use ssh -A)"

if [ "$auto_mode" -eq 1 ]; then
  find_dashcam_routes
  if [ -z "$dashcam_routes" ]; then
    echo "No routes in dashcam mode to download."
    exit 0
  fi
  echo "In dashcam mode:$dashcam_routes"

  for r in $dashcam_routes; do
    route="$r"
    segments=""
    skip=""
    add_skip_list "first,second,second_last,last"
    if select_segments && route_is_long_enough && filter_reverse_segments; then
      download_segments
    else
      echo "$route: no eligible segments to download"
    fi
  done
else
  # --skip / --strip-ends: read the segments on the comma, then drop the skipped ones.
  if [ "$skip_mode" -eq 1 ]; then
    select_segments || exit 1
  fi
  download_segments
fi

echo "Done."
